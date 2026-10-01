#!/usr/bin/env python3
"""
Fed-path analogue analysis (v2)

Finds the historical windows whose policy/macro path most resembles today's,
then lines up what rates, macro and assets did around those windows.

Matching   : weighted distance over the cumulative-change paths of several
             variables (Fed funds, 10y, CPI y/y, unemployment by default),
             each scaled by its own historical volatility so weights are
             comparable, plus optional penalties on the *level* of Fed and CPI
             at the window end.
Policy rate: daily effective Fed funds (DFF). History = monthly average
             (== FEDFUNDS); the current partial month = latest daily print, so
             a mid-month FOMC move is fully reflected.
Lookback   : "auto" anchors the window at the Fed's last cycle peak.

Edit the CONFIG block and run the file. Requires: pandas numpy matplotlib
yfinance, and fredapi (with FRED_API_KEY set) or pandas_datareader.
"""

# ================================ CONFIG ==================================== #
START = "1954-01-01"          # history start for the data pull
ASOF = None                   # e.g. "2007-06-30" to treat a past date as "today"; None = latest

LOOKBACK = "auto"             # int months, or "auto" = months since the last Fed peak + LOOKBACK_PAD
LOOKBACK_PAD = 3              # months added before the peak when LOOKBACK == "auto"
LOOKBACK_BOUNDS = (18, 48)    # clip for the auto lookback
PEAK_SEARCH_MONTHS = 60       # look for the Fed peak within this many months
HORIZON = 24                  # months forward to examine after t=0
TOP = 8                       # number of analogues
MIN_SEP = None                # min months between analogue ends; None = lookback // 2
REQUIRE_FULL_HORIZON = True   # drop candidates whose forward window is incomplete

# Direction filter: candidate's Fed move over the last RECENT_DIR_MONTHS must have the
# same sign as today's (hike/cut/flat, with RECENT_DIR_MIN_BP as the flat band). Keeps
# windows where the Fed was doing the opposite out of the set. Set to 0 to disable.
RECENT_DIR_MONTHS = 3
RECENT_DIR_MIN_BP = 15
# Extra distance term w * (1 - corr(fed path, today's fed path)); penalises sign-flipped shapes.
SHAPE_CORR_WEIGHT = 0.5

# Variables in the distance metric and their weights (0 or absent = not used).
# fed/y10/y30 in %, cpi/pce = y/y inflation in %, unrate in %, others = % return.
# cape = Shiller CAPE (valuation), oil_mom = 12m % change in WTI (inflation impulse).
MATCH_WEIGHTS = {"fed": 1.0, "y10": 0.5, "cpi": 0.5, "unrate": 0.5, "cape": 0.25, "oil_mom": 0.5}
# Penalty per 1 std-dev gap in the *level* at t=0 (regime match, not shape).
LEVEL_WEIGHTS = {"fed": 0.5, "cpi": 0.5, "cape": 0.25}

# Shiller CAPE source. The script scrapes the current ie_data.xls link from
# shillerdata.com (the URL carries a changing version hash), falls back to the
# old Yale URL, then to CAPE_FILE (a local ie_data.xls, or a CSV with columns
# date,cape). Needs `pip install xlrd` for the .xls. Set CAPE_FILE = None to
# rely on download only; set MATCH_WEIGHTS["cape"] = 0 to drop CAPE entirely.
CAPE_FILE = None
COMPOSITE_STAT = "median"     # "median" or "mean" for composite paths in plots/console

# Data-gap fillers
GOLD_LBMA = True              # gold from LBMA daily PM fix (USD, from 1968) instead of Yahoo GC=F (2000+)
Y30_PROXY_DGS20 = True        # fill 30y before 1977 with the 20y constant-maturity yield (DGS20), flagged in output
OIL_MOM_FLOOR_YEAR = 1983     # WTI was administratively priced before this; oil_mom set NaN earlier (matcher ignores it there)

FWD = 12                      # horizon (months) for the regime table / scatter / OLS
DEAD_BAND_BP = 25             # "flat" if |change| below this over FWD months
OUTDIR = "fed_analogue_output"
DEMO = False                  # True = synthetic data, no internet (smoke test only)
# =========================================================================== #

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --------------------------------------------------------------------------- #
# Variable catalogue
# --------------------------------------------------------------------------- #
# kind: "level" -> aligned path is change in percentage points (x100 = bp for rates)
#       "price" -> aligned path is cumulative % return
VARS = {
    "fed":    {"label": "Fed funds",        "kind": "level", "unit": "bp"},
    "y10":    {"label": "10y yield",        "kind": "level", "unit": "bp"},
    "y30":    {"label": "30y yield",        "kind": "level", "unit": "bp"},
    "cpi":    {"label": "CPI y/y",          "kind": "level", "unit": "pp"},
    "pce":    {"label": "Core PCE y/y",     "kind": "level", "unit": "pp"},
    "unrate": {"label": "Unemployment",     "kind": "level", "unit": "pp"},
    "bei":    {"label": "10y breakeven",    "kind": "level", "unit": "bp"},
    "cape":   {"label": "Shiller CAPE",     "kind": "level", "unit": "x"},
    "oil_mom":{"label": "WTI 12m momentum", "kind": "level", "unit": "pp"},
    "spx":    {"label": "S&P 500",          "kind": "price", "unit": "%"},
    "ndx":    {"label": "Nasdaq Comp.",     "kind": "price", "unit": "%"},
    "oil":    {"label": "WTI crude",        "kind": "price", "unit": "%"},
    "gold":   {"label": "Gold",             "kind": "price", "unit": "%"},
    "dxy":    {"label": "Dollar index",     "kind": "price", "unit": "%"},
}
FRED_CODES = {           # FRED code -> panel column
    "DFF": "fed", "DGS10": "y10", "DGS30": "y30",
    "CPIAUCSL": "cpi_idx", "PCEPILFE": "pce_idx", "UNRATE": "unrate",
    "T10YIE": "bei", "WTISPLC": "oil",             # WTI spot, monthly, from 1946
}
YAHOO_TICKERS = {        # Yahoo ticker -> panel column
    "^GSPC": "spx", "^IXIC": "ndx", "GC=F": "gold", "DX-Y.NYB": "dxy",
}
RATE_PANEL = ["fed", "y10", "y30", "cpi", "pce", "unrate", "bei", "cape", "oil_mom"]
ASSET_PANEL = ["spx", "ndx", "oil", "gold", "dxy"]
CHECKPOINTS = (1, 3, 6, 12, 18, 24, 36)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def month_end_rule():
    try:
        pd.Series(dtype=float, index=pd.DatetimeIndex([])).resample("ME")
        return "ME"
    except (ValueError, TypeError):
        return "M"


def load_fred(start):
    key = os.environ.get("FRED_API_KEY")
    out = {}
    if key:
        from fredapi import Fred
        fred = Fred(api_key=key)
        fetch = lambda code: fred.get_series(code, observation_start=start)
    else:
        import pandas_datareader.data as pdr
        fetch = lambda code: pdr.DataReader(code, "fred", start)[code]
    for code, name in FRED_CODES.items():
        try:
            out[name] = fetch(code)
        except Exception as e:  # noqa: BLE001
            print(f"  WARN: {code} failed ({e}); column '{name}' will be empty")
            out[name] = pd.Series(dtype=float)
    df = pd.DataFrame(out)
    df.index = pd.to_datetime(df.index)
    return df


def load_fred_series(code, start):
    key = os.environ.get("FRED_API_KEY")
    if key:
        from fredapi import Fred
        return Fred(api_key=key).get_series(code, observation_start=start)
    import pandas_datareader.data as pdr
    return pdr.DataReader(code, "fred", start)[code]


def load_yahoo(start):
    import yfinance as yf
    out = {}
    for tkr, name in YAHOO_TICKERS.items():
        try:
            px = yf.download(tkr, start=start, auto_adjust=False, progress=False)
            close = px["Close"] if "Close" in px else px["Adj Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close.index = pd.to_datetime(close.index).tz_localize(None)
            out[name] = close
        except Exception as e:  # noqa: BLE001
            print(f"  WARN: {tkr} failed ({e}); column '{name}' will be empty")
            out[name] = pd.Series(dtype=float)
    return pd.DataFrame(out)


def _parse_ie_data(path_or_buf):
    """Parse Shiller's ie_data.xls -> monthly CAPE series (month-end index).

    The sheet has a multi-row header; the word "CAPE" appears both over the P/E10
    column and inside "Excess / CAPE / Yield", so the column is chosen by checking
    that the values beneath look like a P/E (median between 5 and 60), not a yield.
    """
    raw = pd.read_excel(path_or_buf, sheet_name="Data", header=None)
    candidates = []
    for i in range(min(15, len(raw))):
        for j, x in enumerate(raw.iloc[i].tolist()):
            if str(x).strip().upper() in ("CAPE", "P/E10", "PE10"):
                vals = pd.to_numeric(raw.iloc[i + 1: i + 400, j], errors="coerce").dropna()
                if len(vals) > 50 and 5 < vals.median() < 60:
                    candidates.append((i, j))
    if not candidates:
        raise ValueError("could not locate the CAPE column in ie_data.xls")
    hdr_row, cape_col = candidates[0]
    df = raw.iloc[hdr_row + 1:, [0, cape_col]].copy()
    df.columns = ["ym", "cape"]
    df["ym"] = pd.to_numeric(df["ym"], errors="coerce")
    df = df.dropna(subset=["ym"])
    df = df[(df["ym"] > 1800) & (df["ym"] < 2200)]
    ystr = df["ym"].map(lambda v: f"{v:.2f}")          # 2026.10 is stored as 2026.1
    years = ystr.str[:4].astype(int); months = ystr.str[-2:].astype(int).clip(1, 12)
    df.index = pd.to_datetime({"year": years, "month": months, "day": 1}) + pd.offsets.MonthEnd(0)
    cape = pd.to_numeric(df["cape"], errors="coerce").rename("cape").dropna()
    return cape[~cape.index.duplicated(keep="last")].sort_index()


def _fetch(url, timeout=60):
    import urllib.request
    ua = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
    return urllib.request.urlopen(urllib.request.Request(url, headers=ua), timeout=timeout).read()


def _cape_from_shillerdata():
    import html as htmllib, io, re, urllib.parse
    page = _fetch("https://shillerdata.com/").decode("utf-8", "ignore")
    page = htmllib.unescape(page)
    page = urllib.parse.unquote(page).replace("\\/", "/").replace("\\u002F", "/").replace("\\u002f", "/")
    # any URL-ish token containing ie_data (scheme optional, protocol-relative allowed)
    hits = re.findall(r"(?:https?:)?//[^\s\"'<>()\[\]{}]*?ie_data[^\s\"'<>()\[\]{}]*", page, flags=re.I)
    if not hits:
        ctx = [page[max(0, m.start() - 160): m.end() + 60].replace("\n", " ")
               for m in re.finditer("ie_data", page, flags=re.I)]
        raise ValueError("no ie_data link; context: " + " || ".join(ctx[:3]))
    url = hits[0]
    if url.startswith("//"):
        url = "https:" + url
    return _parse_ie_data(io.BytesIO(_fetch(url))), url


def _cape_from_multpl():
    """multpl.com republishes Shiller's monthly CAPE as an HTML table, updated daily."""
    import io
    html = _fetch("https://www.multpl.com/shiller-pe/table/by-month").decode("utf-8", "ignore")
    tables = pd.read_html(io.StringIO(html))
    t = next(t for t in tables if t.shape[1] >= 2 and "Date" in str(t.columns[0]))
    t = t.iloc[:, :2]; t.columns = ["date", "cape"]
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t["cape"] = pd.to_numeric(t["cape"].astype(str).str.replace(r"[^0-9.]", "", regex=True), errors="coerce")
    t = t.dropna()
    cape = t.set_index("date")["cape"].sort_index()
    cape.index = cape.index + pd.offsets.MonthEnd(0)
    return cape[~cape.index.duplicated(keep="last")].rename("cape")


def load_cape():
    import io
    sources = [("shillerdata.com", lambda: _cape_from_shillerdata()[0]),
               ("multpl.com", _cape_from_multpl),
               ("yale (stale copy)", lambda: _parse_ie_data(io.BytesIO(_fetch("http://www.econ.yale.edu/~shiller/data/ie_data.xls"))))]
    if CAPE_FILE and os.path.exists(CAPE_FILE):
        def _local():
            if CAPE_FILE.lower().endswith((".xls", ".xlsx")):
                return _parse_ie_data(CAPE_FILE)
            c = pd.read_csv(CAPE_FILE, parse_dates=["date"]).set_index("date")["cape"]
            return c.resample(month_end_rule()).last().rename("cape")
        sources.insert(0, ("CAPE_FILE", _local))
    best = None
    for name, fn in sources:
        try:
            cape = fn()
            age = (pd.Timestamp.today() - cape.index[-1]).days // 30
            print(f"  cape    source: {name}; last obs {cape.index[-1]:%Y-%m} = {cape.iloc[-1]:.1f}x"
                  + ("" if age <= 3 else f"  ** {age} months old **"))
            if best is None or cape.index[-1] > best.index[-1]:
                best = cape
            if age <= 3:
                return cape
        except Exception as e:  # noqa: BLE001
            print(f"  WARN: CAPE from {name} failed ({str(e)[:400]})")
    if best is not None:
        print("  WARN: only stale CAPE found — today's CAPE will be missing; set CAPE_FILE to a fresh ie_data.xls")
        return best
    print("  WARN: no CAPE data at all; set CAPE_FILE or MATCH_WEIGHTS['cape'] = 0")
    return pd.Series(dtype=float, name="cape")


def load_gold_lbma():
    """LBMA gold PM price in USD, daily from 1968 (public JSON)."""
    import json
    raw = json.loads(_fetch("https://prices.lbma.org.uk/json/gold_pm.json"))
    df = pd.DataFrame(raw)
    df["d"] = pd.to_datetime(df["d"])
    usd = df["v"].map(lambda v: v[0] if isinstance(v, list) and v else np.nan)
    g = pd.Series(usd.values, index=df["d"], name="gold").astype(float)
    return g[g > 0].sort_index()


def build_panel(start, asof=None):
    rule = month_end_rule()
    fred = load_fred(start)
    yah = load_yahoo(start)
    if GOLD_LBMA:
        try:
            lbma = load_gold_lbma()
            yf_gold = yah["gold"].dropna()
            tail = yf_gold[yf_gold.index > lbma.index[-1]]          # fill any gap after LBMA's last print
            yah["gold"] = pd.concat([lbma, tail]).sort_index()
            print(f"  gold    source: LBMA PM fix from {lbma.index[0]:%Y-%m}" + (f", Yahoo tail {len(tail)}d" if len(tail) else ""))
        except Exception as e:  # noqa: BLE001
            print(f"  WARN: LBMA gold failed ({e}); using Yahoo GC=F")
    if Y30_PROXY_DGS20:
        try:
            y20 = load_fred_series("DGS20", start)
            missing = fred["y30"].isna() & (fred.index < "1977-02-15")
            fred.loc[missing, "y30"] = y20.reindex(fred.index)[missing]
            print(f"  y30     pre-1977 filled with DGS20 (20y) proxy: {int(missing.sum())} days")
        except Exception as e:  # noqa: BLE001
            print(f"  WARN: DGS20 proxy failed ({e})")
    daily = pd.concat([fred, yah], axis=1)

    m = daily.resample(rule).last()
    cape = load_cape()
    if len(cape):
        m["cape"] = cape.reindex(m.index).ffill(limit=2)
    else:
        m["cape"] = np.nan
    # Policy rate: monthly average of DFF for history, latest print for the partial month.
    fed_mean = daily["fed"].resample(rule).mean()
    m["fed"] = fed_mean
    last_obs = daily["fed"].dropna().index[-1]
    if not last_obs.is_month_end:
        m.loc[m.index[-1], "fed"] = daily["fed"].dropna().iloc[-1]
        print(f"Note: final month is partial (DFF through {last_obs:%Y-%m-%d}); Fed value = latest print.")
    # Inflation: y/y from index levels; monthly series are released with a lag, so ffill.
    m["cpi"] = m["cpi_idx"].pct_change(12) * 100
    m["pce"] = m["pce_idx"].pct_change(12) * 100
    for c in ["cpi", "pce", "unrate", "oil"]:
        m[c] = m[c].ffill(limit=2)
    m = m.drop(columns=["cpi_idx", "pce_idx"])
    m["oil_mom"] = m["oil"].pct_change(12) * 100
    m.loc[m.index.year < OIL_MOM_FLOOR_YEAR, "oil_mom"] = np.nan   # administered prices: momentum meaningless
    m = m.dropna(subset=["fed", "spx"])
    if asof:
        m = m.loc[:asof]
    for c in m.columns:
        s = m[c].dropna()
        if len(s):
            print(f"  {c:<7} {s.index[0]:%Y-%m} -> {s.index[-1]:%Y-%m}  latest {s.iloc[-1]:,.2f}")
        else:
            print(f"  {c:<7} (no data)")
    return m


def build_demo_panel(months=800, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1960-01-31", periods=months, freq=month_end_rule())
    fed = np.zeros(months); level, drift = 4.0, 0.0
    for i in range(months):
        if rng.random() < 0.04:
            drift = rng.choice([-0.35, -0.15, 0.0, 0.2, 0.4])
        level = max(0.05, level + drift + rng.normal(0, 0.08)) ; level -= 0.02 * (level - 4.0)
        fed[i] = level
    y10 = np.clip(fed + 1 + np.cumsum(rng.normal(0, 0.12, months)) * 0.15, 0.3, None)
    y30 = y10 + 0.5 + rng.normal(0, 0.1, months); y30[500:560] = np.nan
    cpi = np.clip(3 + 0.5 * (fed - 4) + np.cumsum(rng.normal(0, 0.2, months)) * 0.1, -1, None)
    un = np.clip(6 - 0.3 * (fed - 4) + np.cumsum(rng.normal(0, 0.1, months)) * 0.2, 2.5, 12)
    def walk(mu, sig, start=100):
        return start * np.cumprod(1 + rng.normal(mu, sig, months))
    df = pd.DataFrame({"fed": fed, "y10": y10, "y30": y30, "cpi": cpi, "pce": cpi - 0.3, "unrate": un,
                       "bei": y10 - 2, "spx": walk(0.006, 0.04), "ndx": walk(0.008, 0.06),
                       "oil": walk(0.003, 0.08), "gold": walk(0.004, 0.045), "dxy": walk(0.0, 0.02)}, index=idx)
    df["cape"] = np.clip(20 + np.cumsum(rng.normal(0, 0.6, months)) * 0.3, 8, 45)
    df["oil_mom"] = df["oil"].pct_change(12) * 100
    df.loc[df.index < "2003-01-01", "bei"] = np.nan
    df.loc[df.index < "2000-01-01", "gold"] = np.nan
    return df


# --------------------------------------------------------------------------- #
# Lookback
# --------------------------------------------------------------------------- #
def resolve_lookback(panel):
    if LOOKBACK != "auto":
        return int(LOOKBACK), None
    fed = panel["fed"].iloc[-PEAK_SEARCH_MONTHS:]
    peak_date = fed[fed >= fed.max() - 0.03].index[-1]   # last month at the cycle peak
    months_since = len(fed) - 1 - fed.index.get_loc(peak_date)
    lb = int(np.clip(months_since + LOOKBACK_PAD, *LOOKBACK_BOUNDS))
    return lb, peak_date


# --------------------------------------------------------------------------- #
# Analogue search
# --------------------------------------------------------------------------- #
def path_of(series, end_pos, lookback):
    seg = series.iloc[end_pos - lookback: end_pos + 1].to_numpy(dtype=float)
    if VARS_KIND(series.name) == "price":
        return (seg / seg[0] - 1) * 100
    return seg - seg[0]


def VARS_KIND(name):
    return VARS.get(name, {}).get("kind", "level")


def find_analogues(panel, lookback, horizon):
    n = len(panel); today = n - 1
    match_vars = [v for v, w in MATCH_WEIGHTS.items() if w > 0 and v in panel]
    level_vars = [v for v, w in LEVEL_WEIGHTS.items() if w > 0 and v in panel]
    # A variable with no value for today (or gaps in today's lookback) cannot be matched on.
    for v in list(set(match_vars) | set(level_vars)):
        seg = panel[v].iloc[today - lookback: today + 1]
        if seg.isna().any():
            print(f"  WARN: '{v}' missing for today's window ({int(seg.isna().sum())} NaN) -> dropped from matching")
            match_vars = [m for m in match_vars if m != v]; level_vars = [m for m in level_vars if m != v]

    # Scale: std of `lookback`-month changes (or % returns) for each variable, full history.
    scale = {}
    for v in set(match_vars) | set(level_vars):
        s = panel[v]
        chg = (s.pct_change(lookback) * 100) if VARS_KIND(v) == "price" else s.diff(lookback)
        scale[v] = float(chg.std()) or 1.0
    lvl_scale = {v: float(panel[v].std()) or 1.0 for v in level_vars}

    today_paths = {v: path_of(panel[v], today, lookback) for v in match_vars}
    today_lvl = {v: panel[v].iloc[today] for v in level_vars}

    def recent_dir(pos):
        chg = (panel["fed"].iloc[pos] - panel["fed"].iloc[pos - RECENT_DIR_MONTHS]) * 100
        return 0 if abs(chg) < RECENT_DIR_MIN_BP else int(np.sign(chg))
    today_dir = recent_dir(today) if RECENT_DIR_MONTHS else None

    last_ok = today - horizon if REQUIRE_FULL_HORIZON else today - 1
    rows = []
    for pos in range(lookback, last_ok + 1):
        if today - pos < lookback:
            continue
        if today_dir is not None and recent_dir(pos) != today_dir:
            continue
        rec = {"pos": pos, "end": panel.index[pos]}
        dist = 0.0; ok = True
        for v in match_vars:
            p = path_of(panel[v], pos, lookback)
            if np.isnan(p).any():
                if v == "oil_mom":          # pre-1983 administered oil: neutral contribution, flagged
                    rec[f"d_{v}"] = np.nan; continue
                ok = False; break
            rmse = np.sqrt(np.mean((p - today_paths[v]) ** 2)) / scale[v]
            rec[f"d_{v}"] = MATCH_WEIGHTS[v] * rmse
            dist += rec[f"d_{v}"]
        # neutral fill for skipped optional terms = median contribution of that term across candidates (applied below)
            if v == "fed":
                corr = np.corrcoef(p, today_paths[v])[0, 1] if p.std() > 0 else 0.0
                rec["fed_shape_corr"] = corr
                rec["d_shape"] = SHAPE_CORR_WEIGHT * (1 - corr)
                dist += rec["d_shape"]
        if not ok:
            continue
        for v in level_vars:
            lv = panel[v].iloc[pos]
            if np.isnan(lv):
                ok = False; break
            rec[f"lvl_{v}"] = LEVEL_WEIGHTS[v] * abs(lv - today_lvl[v]) / lvl_scale[v]
            dist += rec[f"lvl_{v}"]
        if not ok:
            continue
        rec["distance"] = dist
        rows.append(rec)
    if not rows:
        sys.exit("No candidate windows: matching variables have too little joint history. Reduce MATCH_WEIGHTS/LEVEL_WEIGHTS.")
    cands = pd.DataFrame(rows)
    if "d_oil_mom" in cands and cands["d_oil_mom"].isna().any():
        fill = cands["d_oil_mom"].median()
        miss = cands["d_oil_mom"].isna()
        cands.loc[miss, "distance"] += fill
        cands.loc[miss, "d_oil_mom"] = fill
        cands["oil_mom_na"] = miss
    cands = cands.sort_values("distance")

    min_sep = MIN_SEP or max(6, lookback // 2)
    chosen = []
    for _, r in cands.iterrows():
        if all(abs(r["pos"] - c["pos"]) >= min_sep for c in chosen):
            chosen.append(r)
        if len(chosen) >= TOP:
            break
    return pd.DataFrame(chosen).reset_index(drop=True), len(cands)


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
def aligned(panel, end_pos, lookback, horizon):
    lo, hi = end_pos - lookback, min(end_pos + horizon, len(panel) - 1)
    seg = panel.iloc[lo: hi + 1]; t0 = panel.iloc[end_pos]
    out = pd.DataFrame(index=np.arange(lo - end_pos, hi - end_pos + 1))
    for v, meta in VARS.items():
        if v not in panel:
            continue
        if meta["kind"] == "price":
            out[v] = (seg[v] / t0[v] - 1).to_numpy() * 100
        else:
            mult = 100 if meta["unit"] == "bp" else 1
            out[v] = (seg[v] - t0[v]).to_numpy() * mult
    return out


def wide(paths, var):
    return pd.DataFrame({k: p[var] for k, p in paths.items() if var in p})


# --------------------------------------------------------------------------- #
# Conditional performance
# --------------------------------------------------------------------------- #
def regime_frame(panel, fwd, asset):
    d = pd.DataFrame({
        "fed_chg": panel["fed"].shift(-fwd) - panel["fed"],
        "y10_chg": panel["y10"].shift(-fwd) - panel["y10"],
        "ret": (panel[asset].shift(-fwd) / panel[asset] - 1) * 100,
    }).dropna()
    thr = DEAD_BAND_BP / 100
    d["fed_dir"] = np.select([d.fed_chg > thr, d.fed_chg < -thr], ["hike", "cut"], "flat")
    d["y10_dir"] = np.select([d.y10_chg > thr, d.y10_chg < -thr], ["up", "down"], "flat")
    return d


def regime_table(d):
    g = d.groupby(["fed_dir", "y10_dir"])["ret"]
    return pd.DataFrame({"n": g.size(), "mean_%": g.mean(), "median_%": g.median(),
                         "hit_%": g.apply(lambda s: (s > 0).mean() * 100),
                         "min_%": g.min(), "max_%": g.max()}).round(1)


def ols(d):
    X = np.column_stack([np.ones(len(d)), d["fed_chg"], d["y10_chg"]])
    b, *_ = np.linalg.lstsq(X, d["ret"].to_numpy(), rcond=None)
    r2 = 1 - (d["ret"].to_numpy() - X @ b).var() / d["ret"].var()
    return b, r2, len(d)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _overlay(ax, paths, today, var, title):
    w = wide(paths, var)
    if w.empty or w.isna().all().all():
        ax.set_title(f"{title} (no data)"); return
    w = w.dropna(axis=1, how="all")
    title = f"{title}  [n={w.shape[1]}]" if w.shape[1] < len(paths) else title
    for c in w.columns:
        ax.plot(w.index, w[c], lw=1, alpha=0.55, label=c)
    mean = w.median(axis=1) if COMPOSITE_STAT == "median" else w.mean(axis=1)
    ax.plot(mean.index, mean, color="red", lw=2.2, ls="--", label=f"Analogue {COMPOSITE_STAT}")
    ax.fill_between(w.index, w.min(axis=1), w.max(axis=1), color="red", alpha=0.08)
    if var in today:
        ax.plot(today.index, today[var], color="black", lw=3, label="Today")
    ax.axvline(0, color="grey", ls=":"); ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(title)


def plot_grid(paths, today, vars_, fname, suptitle, outdir):
    n = len(vars_); cols = 3; rows_ = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(6 * cols, 4.2 * rows_), sharex=True)
    for ax, v in zip(axes.ravel(), vars_):
        _overlay(ax, paths, today, v, f"{VARS[v]['label']} ({VARS[v]['unit']} vs t=0)")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Months from t=0")
    axes.ravel()[0].legend(fontsize=7, ncol=2)
    fig.suptitle(suptitle); fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=140); plt.close(fig)


def plot_scatter(d_all, d_ana, fwd, asset, outdir):
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {"hike": "tab:red", "cut": "tab:green", "flat": "tab:grey"}
    for k, sub in d_all.groupby("fed_dir"):
        ax.scatter(sub["y10_chg"] * 100, sub["ret"], s=12, alpha=0.35, color=colors[k], label=f"Fed {k} (all months)")
    if len(d_ana):
        ax.scatter(d_ana["y10_chg"] * 100, d_ana["ret"], s=140, marker="*", color="black",
                   edgecolor="yellow", zorder=5, label="Analogue t=0")
        for idx, r in d_ana.iterrows():
            ax.annotate(pd.Timestamp(idx).strftime("%Y-%m"), (r["y10_chg"] * 100, r["ret"]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="grey", lw=0.8); ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel(f"Change in 10y yield over next {fwd}m (bp)")
    ax.set_ylabel(f"{VARS[asset]['label']} return over next {fwd}m (%)")
    ax.set_title(f"{VARS[asset]['label']} vs yield and Fed moves"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f"fig4_scatter_{asset}.png"), dpi=140); plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    pd.set_option("display.width", 250, "display.max_columns", 60)

    if DEMO:
        panel = build_demo_panel(); print("** DEMO MODE: synthetic data **")
    else:
        try:
            panel = build_panel(START, ASOF)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"Data load failed: {type(e).__name__}: {e}") from e
    panel.to_csv(os.path.join(OUTDIR, "data_monthly.csv"))

    lookback, peak = resolve_lookback(panel)
    t0 = panel.index[-1]
    print(f"\nPanel {panel.index[0]:%Y-%m} -> {t0:%Y-%m} ({len(panel)}m). Lookback {lookback}m"
          + (f" (auto: Fed peak {peak:%Y-%m} at {panel.loc[peak, 'fed']:.2f}%)" if peak is not None else "")
          + f", horizon {HORIZON}m.")
    last = panel.iloc[-1]; prev = panel.iloc[-1 - lookback]
    print(f"Today: Fed {last.fed:.2f}%  10y {last.y10:.2f}%  30y {last.y30:.2f}%  CPI {last.cpi:.1f}%  "
          f"core PCE {last.pce:.1f}%  U {last.unrate:.1f}%  CAPE {last.cape:.1f}x  WTI 12m {last.oil_mom:+.0f}%")
    print(f"Over lookback: Fed {100*(last.fed-prev.fed):+.0f}bp  10y {100*(last.y10-prev.y10):+.0f}bp  "
          f"CPI {last.cpi-prev.cpi:+.1f}pp  U {last.unrate-prev.unrate:+.1f}pp  "
          f"S&P {100*(last.spx/prev.spx-1):+.0f}%  Nasdaq {100*(last.ndx/prev.ndx-1):+.0f}%  "
          f"WTI {100*(last.oil/prev.oil-1):+.0f}%")

    if RECENT_DIR_MONTHS:
        rc = (last.fed - panel["fed"].iloc[-1 - RECENT_DIR_MONTHS]) * 100
        print(f"Recent Fed move ({RECENT_DIR_MONTHS}m): {rc:+.0f}bp -> candidates restricted to same direction")
    analogues, n_cands = find_analogues(panel, lookback, HORIZON)
    paths = {pd.Timestamp(r["end"]).strftime("%Y-%m"): aligned(panel, int(r["pos"]), lookback, HORIZON)
             for _, r in analogues.iterrows()}
    today = aligned(panel, len(panel) - 1, lookback, HORIZON)

    # --- analogue table -------------------------------------------------- #
    dcols = [c for c in analogues.columns if c.startswith(("d_", "lvl_"))]
    rows = []
    for _, r in analogues.iterrows():
        pos = int(r["pos"]); lab = pd.Timestamp(r["end"]).strftime("%Y-%m"); p = paths[lab]
        row = {"t0": lab, "distance": round(r["distance"], 2), "fed_corr": round(r.get("fed_shape_corr", np.nan), 2)}
        if bool(r.get("oil_mom_na", False)):
            row["note"] = "oil_mom n/a (pre-1983)"
        row.update({c: round(r[c], 2) for c in dcols})
        row.update({f"{v}_t0": round(panel[v].iloc[pos], 2) for v in ["fed", "y10", "y30", "cpi", "unrate", "cape", "oil_mom"]})
        for h in CHECKPOINTS:
            if h <= HORIZON and h in p.index:
                for v in ["fed", "y10", "spx", "ndx", "oil", "gold", "dxy"]:
                    if v in p:
                        row[f"{v}_{h}m"] = round(p.loc[h, v], 1)
        rows.append(row)
    ana_tbl = pd.DataFrame(rows); ana_tbl.to_csv(os.path.join(OUTDIR, "analogues.csv"), index=False)

    # --- aligned paths & composite --------------------------------------- #
    comp = {}
    for v in VARS:
        if v not in panel:
            continue
        w = wide(paths, v); w.to_csv(os.path.join(OUTDIR, f"paths_{v}.csv"))
        comp[f"{v}_mean"] = w.mean(axis=1); comp[f"{v}_median"] = w.median(axis=1)
        comp[f"{v}_min"] = w.min(axis=1); comp[f"{v}_max"] = w.max(axis=1)
        comp[f"{v}_today"] = today[v]
    comp = pd.DataFrame(comp).round(1); comp.index.name = "months_from_t0"
    comp.to_csv(os.path.join(OUTDIR, "composite_summary.csv"))

    # --- regimes / OLS -------------------------------------------------- #
    ana_dates = panel.index[analogues["pos"].astype(int)]
    regimes = {}
    for asset in ["spx", "ndx"]:
        d_all = regime_frame(panel, FWD, asset)
        d_ana = d_all.loc[d_all.index.intersection(ana_dates)]
        regimes[asset] = (d_all, d_ana)
        regime_table(d_all).to_csv(os.path.join(OUTDIR, f"regime_{asset}.csv"))
        plot_scatter(d_all, d_ana, FWD, asset, OUTDIR)

    # --- figures --------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(11, 6))
    _overlay(ax, paths, today, "fed", f"Fed funds path: {lookback}m matched, {HORIZON}m forward (bp vs t=0)")
    ax.set_xlabel("Months from t=0"); ax.legend(fontsize=8, ncol=2); fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig1_fed_paths.png"), dpi=140); plt.close(fig)
    plot_grid(paths, today, RATE_PANEL, "fig2_policy_macro.png", f"Policy & macro around analogues (t=0 = {t0:%Y-%m})", OUTDIR)
    plot_grid(paths, today, ASSET_PANEL, "fig3_assets.png", f"Assets around analogues (t=0 = {t0:%Y-%m})", OUTDIR)

    # --- console --------------------------------------------------------- #
    print(f"\n=== Closest analogues ({n_cands} candidate windows; match on {list(MATCH_WEIGHTS)}, level on {list(LEVEL_WEIGHTS)}) ===")
    base_cols = ["t0", "distance", "fed_corr"] + dcols + [c for c in ana_tbl.columns if c.endswith("_t0")]
    print(ana_tbl[base_cols].to_string(index=False))
    print("\n=== Forward checkpoints per analogue ===")
    for v in ["spx", "ndx", "fed", "y10", "oil", "gold", "dxy"]:
        cols = [c for c in ana_tbl.columns if c.startswith(f"{v}_") and c.endswith("m") and c[-2].isdigit()]
        cols = [c for c in cols if c.split("_")[1][:-1].isdigit()]
        if cols:
            print(f"-- {VARS[v]['label']} ({VARS[v]['unit']})")
            print(ana_tbl[["t0"] + cols].to_string(index=False))
    hs = [h for h in CHECKPOINTS if h <= HORIZON]
    print(f"\n=== Composite forward path (analogue {COMPOSITE_STAT}) ===")
    cols = [f"{v}_{COMPOSITE_STAT}" for v in ["fed", "y10", "y30", "cpi", "unrate", "spx", "ndx", "oil", "gold", "dxy"] if f"{v}_{COMPOSITE_STAT}" in comp]
    print(comp.loc[hs, cols].to_string())
    # Composite split by what the Fed did next (net change over FWD months at t0)
    split_vars = [v for v in ["fed", "y10", "spx", "ndx", "oil", "gold", "dxy"] if v in panel]
    fwd_dir = {}
    for lab, p in paths.items():
        if FWD in p.index:
            c = p.loc[FWD, "fed"]
            fwd_dir[lab] = "hiked" if c > DEAD_BAND_BP else ("cut" if c < -DEAD_BAND_BP else "held")
    print(f"\n=== Composite split by Fed direction over the next {FWD}m ===")
    for grp in ["hiked", "held", "cut"]:
        labs = [l for l, g in fwd_dir.items() if g == grp]
        if not labs:
            continue
        sub = {l: paths[l] for l in labs}
        agg = (lambda w: w.median(axis=1)) if COMPOSITE_STAT == "median" else (lambda w: w.mean(axis=1))
        tbl = pd.DataFrame({VARS[v]["label"]: agg(wide(sub, v)) for v in split_vars}).loc[hs].round(1)
        print(f"-- Fed {grp} ({len(labs)}): {', '.join(labs)}")
        print(tbl.to_string())
    for asset, (d_all, d_ana) in regimes.items():
        b, r2, n = ols(d_all)
        print(f"\n=== {VARS[asset]['label']} {FWD}m fwd return by Fed / 10y direction (full sample, {DEAD_BAND_BP}bp dead-band) ===")
        print(regime_table(d_all).to_string())
        if len(d_ana):
            print(f"-- analogue t0 points only:"); print(regime_table(d_ana).to_string())
        print(f"OLS: {b[0]:+.1f}%  {b[1]:+.1f}%/100bp Fed  {b[2]:+.1f}%/100bp 10y   R² {r2:.2f}  n={n}")
    print(f"\nFiles written to {os.path.abspath(OUTDIR)}")


if __name__ == "__main__":
    main()
