"""
treasury_equity_fed_analysis.py  (v3.1 - trading-day calendar fix)
--------------------------------
Analyze correlations between US Treasury yield changes (2y/10y/30y) and S&P 500
returns across multiple horizons, plus a Fed-cycle similarity ("analogue") engine.

REQUIREMENTS:
  pip install -U pandas numpy matplotlib scipy statsmodels fredapi yfinance curl_cffi pandas_datareader
  (yfinance changes often; the TypeError "'NoneType' object is not subscriptable"
   almost always means an outdated yfinance or Yahoo throttling -> upgrade + retry.)
FRED API KEY (free): https://fred.stlouisfed.org/docs/api/api_key.html
  export FRED_API_KEY="your_key_here"

EQUITY DATA SOURCE ORDER (first success wins):
  1. Yahoo Finance (^GSPC) via yfinance  - retried with exponential backoff,
     alternating yf.download() and Ticker().history()
  2. Local cache  data/spx.csv           - written after any successful download
  3. Stooq (^SPX) via pandas_datareader  - free, daily history back to 1789 index reconstruction / 1950s
  4. FRED SP500                          - LAST resort: only ~10 years, kills pre-2016 analogues
  You can also force a file:  python treasury_equity_fed_analysis.py --csv my_spx.csv
  (CSV needs a date column + a 'Close' or 'Adj Close' column.)

CORRELATION LOOKBACKS:
  Correlations are reported for trailing 1y/3y/5y/10y/20y/all windows plus an
  exponentially-weighted version (3y half-life). 'n_indep' = n / horizon is the
  approximate number of NON-overlapping observations; cells with n_indep < 5 are
  greyed out in the heatmaps and should not be trusted.

FED-PATH COMPARISON:
  (a) shape-matching of the trailing 36-month fed funds path against every historical
      36-month window (RMSE + dynamic time warping), (b) detection of every
      "re-hike after easing" pivot since 1954, ranked by macro similarity to today.

CAVEATS:
  * Correlation is not causation. Yields & stocks share drivers (growth, inflation, Fed).
  * Overlapping windows inflate significance -> use non-overlapping / Newey-West / bootstrap.
  * Small sample of Fed cycles; analogue distances are indicative, not predictive.
  * 30y CMT (DGS30) discontinued 2002-02-18 to 2006-02-09 -> gap handled as NaN.
  * z-scoring on full history introduces mild look-ahead; use trailing-only for strict tests.
"""

import os
import sys
import time
import random
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# 0. CONFIG
# ----------------------------------------------------------------------
FRED_SERIES = {
    "DGS2": "y2", "DGS10": "y10", "DGS30": "y30",
    "DFEDTARU": "ffr_upper", "DFEDTARL": "ffr_lower",
    "DFEDTAR": "ffr_single", "FEDFUNDS": "ffr_eff",
    "CPIAUCSL": "cpi", "UNRATE": "unrate",
}
HORIZONS = {"1m": 21, "3m": 63, "6m": 126, "12m": 252, "18m": 378, "24m": 504}
LOOKBACKS = {"1y": 252, "3y": 756, "5y": 1260, "10y": 2520, "20y": 5040, "all": None}
EW_HALFLIFE = 756          # trading days (~3y) for the exponentially-weighted correlation
MIN_INDEP = 5              # min approx independent obs (n/horizon) to trust a correlation
PATH_WINDOW = 36           # months of fed-funds path used for shape matching
START = "1954-07-01"     # FEDFUNDS starts 1954-07; DGS10 1962, DGS2 1976, DGS30 1977
OUTDIR = "output"
CACHE_DIR = "data"
SPX_CACHE = os.path.join(CACHE_DIR, "spx.csv")
YF_MAX_ATTEMPTS = 6        # total Yahoo attempts
YF_BASE_DELAY = 2.0        # seconds; doubles each attempt (+ jitter), capped at 60s
MIN_ROWS_OK = 2000         # < this many rows -> treat as failed/partial download
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# 1. DATA DOWNLOAD
# ----------------------------------------------------------------------
def get_fred():
    """Download FRED series via fredapi (preferred) or pandas_datareader (fallback)."""
    key = os.environ.get("FRED_API_KEY")
    frames = {}
    try:
        from fredapi import Fred
        fred = Fred(api_key=key)
        for code, name in FRED_SERIES.items():
            for attempt in range(3):
                try:
                    frames[name] = fred.get_series(code, observation_start=START)
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f"  ! FRED {code} failed after 3 tries: {e}")
                    else:
                        time.sleep(2 ** attempt)
        df = pd.DataFrame(frames)
    except Exception as e:
        print(f"fredapi unavailable ({e}); trying pandas_datareader...")
        import pandas_datareader.data as web
        df = web.DataReader(list(FRED_SERIES), "fred", START)
        df.columns = [FRED_SERIES[c] for c in df.columns]
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _clean_close(obj):
    """Normalize whatever yfinance/pandas returns into a clean float Series named 'spx'."""
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        # yfinance >=0.2.40 returns MultiIndex columns like ('Close','^GSPC')
        if isinstance(obj.columns, pd.MultiIndex):
            lvl0 = obj.columns.get_level_values(0)
            col = "Adj Close" if "Adj Close" in lvl0 else "Close"
            obj = obj.xs(col, axis=1, level=0)
        else:
            col = "Adj Close" if "Adj Close" in obj.columns else "Close"
            obj = obj[col]
        if isinstance(obj, pd.DataFrame):
            obj = obj.iloc[:, 0]
    s = pd.to_numeric(obj, errors="coerce").dropna()
    s.index = pd.to_datetime(s.index)
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.name = "spx"
    return s


def _download_yahoo(ticker="^GSPC", start=START,
                    max_attempts=YF_MAX_ATTEMPTS, base_delay=YF_BASE_DELAY):
    """Yahoo Finance with retries. Alternates between yf.download() and
    Ticker().history() because they hit different code paths in yfinance
    (one often works when the other is throttled). Exponential backoff + jitter."""
    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance not installed (pip install -U yfinance curl_cffi).")
        return None

    last_err = None
    for attempt in range(1, max_attempts + 1):
        method = "download" if attempt % 2 == 1 else "history"
        try:
            if method == "download":
                raw = yf.download(ticker, start=start, progress=False,
                                  auto_adjust=False, threads=False, timeout=30)
            else:
                raw = yf.Ticker(ticker).history(start=start, auto_adjust=False, timeout=30)
            s = _clean_close(raw)
            if s is not None and len(s) >= MIN_ROWS_OK:
                print(f"  Yahoo OK via {method} (attempt {attempt}): "
                      f"{len(s)} rows, {s.index[0].date()} -> {s.index[-1].date()}")
                return s
            last_err = f"empty/partial result ({0 if s is None else len(s)} rows)"
        except Exception as e:  # TypeError NoneType, JSONDecodeError, HTTP 429, timeouts...
            last_err = f"{type(e).__name__}: {e}"
        if attempt < max_attempts:
            delay = min(60, base_delay * (2 ** (attempt - 1))) + random.uniform(0, 1.5)
            print(f"  Yahoo attempt {attempt}/{max_attempts} failed ({last_err}); "
                  f"retrying in {delay:.1f}s...")
            time.sleep(delay)
    print(f"  Yahoo gave up after {max_attempts} attempts: {last_err}")
    return None


def _load_csv(path):
    """Load S&P 500 closes from a CSV (date column + Close/Adj Close)."""
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if c.lower() in ("date", "datetime", "index")), df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col)
    for c in ("Adj Close", "Close", "close", "spx", "value"):
        if c in df.columns:
            return _clean_close(df[c])
    return _clean_close(df.iloc[:, 0])


def _download_stooq():
    """Stooq via pandas_datareader (^SPX). Long history, no key needed."""
    try:
        import pandas_datareader.data as web
        raw = web.DataReader("^SPX", "stooq", start=START)   # returns newest-first
        s = _clean_close(raw)
        if s is not None and len(s) >= MIN_ROWS_OK:
            print(f"  Stooq OK: {len(s)} rows, {s.index[0].date()} -> {s.index[-1].date()}")
            return s
    except Exception as e:
        print(f"  Stooq failed: {type(e).__name__}: {e}")
    return None


def _download_fred_spx():
    """FRED SP500 - ONLY ~10 years of history. Last resort."""
    try:
        from fredapi import Fred
        s = _clean_close(Fred(api_key=os.environ.get("FRED_API_KEY")).get_series("SP500"))
        print(f"  FRED SP500 OK but SHORT history: {s.index[0].date()} -> {s.index[-1].date()}\n"
              "  !! Pre-2016 canonical episodes will show 'no equity data'. Provide --csv or fix Yahoo.")
        return s
    except Exception as e:
        print(f"  FRED SP500 failed: {e}")
        return None


def get_equity(csv_path=None):
    """Return daily S&P 500 close Series with the fallback chain described in the header."""
    if csv_path:
        print(f"Loading S&P 500 from {csv_path} ...")
        return _load_csv(csv_path)

    print("Downloading S&P 500 from Yahoo Finance ...")
    s = _download_yahoo()
    if s is not None:
        s.to_csv(SPX_CACHE, header=True)
        return s

    if os.path.exists(SPX_CACHE):
        print(f"Using cached {SPX_CACHE} ...")
        s = _load_csv(SPX_CACHE)
        if s is not None and len(s) >= MIN_ROWS_OK:
            print(f"  cache covers {s.index[0].date()} -> {s.index[-1].date()} (may be stale)")
            return s

    print("Trying Stooq ...")
    s = _download_stooq()
    if s is not None:
        s.to_csv(SPX_CACHE, header=True)
        return s

    print("Trying FRED SP500 (short history) ...")
    s = _download_fred_spx()
    if s is not None:
        return s
    raise RuntimeError("No equity data source succeeded. Pass --csv <file> with S&P 500 closes.")


# ----------------------------------------------------------------------
# 2. BUILD DAILY PANEL
# ----------------------------------------------------------------------
def build_panel(csv_path=None):
    fred = get_fred()
    spx = get_equity(csv_path)
    df = fred.join(spx, how="outer").sort_index()
    # Unified fed funds target: upper bound since 2008-12-16, single target before,
    # effective rate as last resort (pre-1994 era before explicit targets).
    df["fed_target"] = df["ffr_upper"].combine_first(df.get("ffr_single"))
    df["fed_target"] = df["fed_target"].combine_first(df["ffr_eff"]).ffill()
    # Monthly series are dated the 1st of the month (may be a weekend) -> carry forward
    df[["cpi", "unrate", "ffr_eff"]] = df[["cpi", "unrate", "ffr_eff"]].ffill(limit=40)
    for c in ["y2", "y10", "y30", "ffr_upper", "ffr_lower", "ffr_single"]:
        if c in df:
            df[c] = df[c].ffill(limit=5)
    # IMPORTANT: DFEDTAR/DFEDTARU are dated on EVERY calendar day (incl. weekends), so the
    # joined index is a 7-day calendar from 1982 on. Restrict to equity TRADING days so that
    # "252 rows" really means ~12 months everywhere, not ~8.3 months post-1982.
    df = df.reindex(spx.dropna().index)
    df["spx"] = df["spx"].ffill(limit=5)
    df["s2s10"] = (df["y10"] - df["y2"]) * 100.0
    df["s10s30"] = (df["y30"] - df["y10"]) * 100.0   # NaN 2002-02..2006-02 (DGS30 gap)
    eq = df["spx"].dropna()
    print(f"Equity coverage in panel: {eq.index[0].date()} -> {eq.index[-1].date()} ({len(eq)} days)")
    return df


# ----------------------------------------------------------------------
# 3. TRANSFORMS
# ----------------------------------------------------------------------
def add_changes_returns(df):
    out = df.copy()
    for hname, h in HORIZONS.items():
        for y in ["y2", "y10", "y30", "s2s10", "s10s30"]:
            if y in out:
                mult = 1.0 if y in ("s2s10", "s10s30") else 100.0
                out[f"d_{y}_{hname}"] = (out[y] - out[y].shift(h)) * mult
        out[f"ret_trail_{hname}"] = out["spx"] / out["spx"].shift(h) - 1.0
        out[f"ret_fwd_{hname}"] = out["spx"].shift(-h) / out["spx"] - 1.0
    return out


# ----------------------------------------------------------------------
# 4. CORRELATIONS
# ----------------------------------------------------------------------
def weighted_corr(x, y, w):
    """Weighted Pearson correlation."""
    w = np.asarray(w, float); w = w / w.sum()
    x = np.asarray(x, float); y = np.asarray(y, float)
    mx, my = np.sum(w * x), np.sum(w * y)
    cov = np.sum(w * (x - mx) * (y - my))
    vx, vy = np.sum(w * (x - mx) ** 2), np.sum(w * (y - my) ** 2)
    return cov / np.sqrt(vx * vy) if vx > 0 and vy > 0 else np.nan


def correlation_tables(df):
    """Pearson/Spearman for each (lookback x horizon x series x mode).
    mode 'contemp' : yield change over window h vs S&P return over the SAME window
    mode 'predict' : yield change over trailing h vs S&P return over the NEXT h
    Lookbacks slice the last N trading days ending today; 'ew_hl3y' uses all data
    with exponentially decaying weights (half-life EW_HALFLIFE days)."""
    from scipy.stats import pearsonr, spearmanr
    rows = []
    series_list = ["y2", "y10", "y30", "s2s10", "s10s30"]
    T = len(df)
    age = np.arange(T)[::-1]                       # 0 = most recent row
    ew_w_full = 0.5 ** (age / EW_HALFLIFE)
    for hname, h in HORIZONS.items():
        for y in series_list:
            dcol = f"d_{y}_{hname}"
            if dcol not in df:
                continue
            for mode, rcol in (("contemp", f"ret_trail_{hname}"), ("predict", f"ret_fwd_{hname}")):
                pair = df[[dcol, rcol]]
                # fixed lookbacks
                for lb_name, lb in LOOKBACKS.items():
                    sub = pair if lb is None else pair.iloc[-lb:]
                    sub = sub.dropna()
                    n = len(sub)
                    if n < 30:
                        continue
                    pr, _ = pearsonr(sub[dcol], sub[rcol])
                    sr, _ = spearmanr(sub[dcol], sub[rcol])
                    rows.append([mode, lb_name, hname, y, n, n / h, pr, sr,
                                 sub.index[0].date(), sub.index[-1].date()])
                # exponentially weighted (all data)
                mask = pair.notna().all(axis=1).values
                if mask.sum() >= 30:
                    pr_ew = weighted_corr(pair[dcol].values[mask], pair[rcol].values[mask], ew_w_full[mask])
                    n_eff = ew_w_full[mask].sum() ** 2 / (ew_w_full[mask] ** 2).sum()  # Kish effective n
                    rows.append([mode, "ew_hl3y", hname, y, int(mask.sum()), n_eff / h, pr_ew, np.nan,
                                 pair.index[mask][0].date(), pair.index[mask][-1].date()])
    res = pd.DataFrame(rows, columns=["mode", "lookback", "horizon", "series", "n", "n_indep",
                                      "pearson", "spearman", "from", "to"])
    res["reliable"] = res["n_indep"] >= MIN_INDEP
    res.to_csv(f"{OUTDIR}/correlations.csv", index=False)
    return res


def print_corr_summary(res, series="y10", mode="contemp"):
    """Compact table: rows = lookback, cols = horizon, values = Pearson r (* = unreliable)."""
    sub = res[(res["series"] == series) & (res["mode"] == mode)]
    order = list(LOOKBACKS) + ["ew_hl3y"]
    pv = sub.pivot(index="lookback", columns="horizon", values="pearson").reindex(order)
    rel = sub.pivot(index="lookback", columns="horizon", values="reliable").reindex(order)
    pv = pv[[h for h in HORIZONS if h in pv.columns]]
    out = pv.copy().astype(object)
    for i in pv.index:
        for c in pv.columns:
            v = pv.loc[i, c]
            out.loc[i, c] = "" if pd.isna(v) else f"{v:+.2f}" + ("" if rel.loc[i, c] else "*")
    print(f"\n{series} yield change vs S&P return ({mode}); Pearson r, * = n_indep < {MIN_INDEP}")
    print(out.to_string())


def newey_west_pvalue(x, y, lags):
    try:
        import statsmodels.api as sm
    except ImportError:
        print("  (statsmodels not installed -> Newey-West skipped; pip install statsmodels)")
        return np.nan, np.nan, np.nan
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 30:
        return np.nan, np.nan, np.nan
    X = sm.add_constant(d.iloc[:, 0])
    model = sm.OLS(d.iloc[:, 1], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return model.params.iloc[1], model.tvalues.iloc[1], model.pvalues.iloc[1]


def block_bootstrap_corr(x, y, block=63, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    d = pd.concat([x, y], axis=1).dropna().values
    n = len(d)
    if n < block * 3:
        return (np.nan, np.nan)
    nblocks = int(np.ceil(n / block))
    corrs = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, nblocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        samp = d[idx]
        if samp[:, 0].std() > 0 and samp[:, 1].std() > 0:
            corrs.append(np.corrcoef(samp[:, 0], samp[:, 1])[0, 1])
    return (float(np.percentile(corrs, 2.5)), float(np.percentile(corrs, 97.5)))


def nonoverlapping_corr(df, dcol, retcol, step):
    sub = df[[dcol, retcol]].dropna().iloc[::step]
    if len(sub) < 10:
        return np.nan, len(sub)
    return sub[dcol].corr(sub[retcol]), len(sub)


# ----------------------------------------------------------------------
# 5. ROLLING STOCK-BOND CORRELATION
# ----------------------------------------------------------------------
def rolling_stock_bond_corr(df, window=756):
    d = pd.DataFrame(index=df.index)
    d["spx_ret"] = df["spx"].pct_change()
    d["bond_ret_proxy"] = -df["y10"].diff()
    roll = d["spx_ret"].rolling(window).corr(d["bond_ret_proxy"])
    plt.figure(figsize=(12, 5))
    roll.plot()
    plt.axhline(0, color="k", lw=0.8)
    plt.title("Rolling 3y Stock-Bond (price) Correlation\n"
              "Positive in inflationary eras (1970s-90s, 2022+); negative ~2000-2021")
    plt.ylabel("correlation")
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/rolling_stock_bond_corr.png", dpi=130)
    plt.close()
    return roll


# ----------------------------------------------------------------------
# 6. FED-CYCLE ANALOGUE ENGINE
# ----------------------------------------------------------------------
def build_feature_panel(df):
    m = df.resample("ME").last()
    m["cpi_yoy"] = m["cpi"].pct_change(12) * 100.0
    feat = pd.DataFrame(index=m.index)
    feat["fed_level"] = m["fed_target"]
    feat["fed_chg_12m"] = m["fed_target"].diff(12)
    feat["s2s10"] = m["s2s10"]
    feat["s2s10_chg_6m"] = m["s2s10"].diff(6)
    feat["y10_chg_12m"] = m["y10"].diff(12) * 100.0
    feat["cpi_yoy"] = m["cpi_yoy"]
    feat["unrate"] = m["unrate"]
    feat["spx_ret_12m"] = m["spx"].pct_change(12) * 100.0
    feat["uninverted"] = (m["s2s10"] > 0).astype(float)
    chg = m["fed_target"].diff()
    since_hike, since_cut, cum_ease = [], [], []
    h = c = 0
    ease = 0.0
    for v in chg:
        if pd.isna(v):
            since_hike.append(np.nan); since_cut.append(np.nan); cum_ease.append(np.nan); continue
        if v > 0:
            h = 0; ease = 0.0
        else:
            h += 1
        if v < 0:
            c = 0; ease += -v
        else:
            c += 1
        since_hike.append(h); since_cut.append(c); cum_ease.append(ease)
    feat["months_since_hike"] = since_hike
    feat["months_since_cut"] = since_cut
    feat["cum_easing"] = cum_ease
    return feat.dropna(how="all")


def _forward_returns(spx, dt):
    bi = spx.index.searchsorted(dt)
    if bi >= len(spx) or dt < spx.index[0]:
        return None
    base = spx.iloc[bi]
    return {hname: (spx.iloc[bi + h] / base - 1) * 100 if bi + h < len(spx) else np.nan
            for hname, h in HORIZONS.items()}


def analogue_ranking(feat, df, top_n=15, exclude_recent_months=18):
    """z-score features, Euclidean & Mahalanobis distance to the latest month.
    exclude_recent_months drops the trailing months so the 'analogues' aren't
    just last quarter (they dominated your first run)."""
    from numpy.linalg import pinv
    cols = ["fed_level", "fed_chg_12m", "s2s10", "s2s10_chg_6m", "y10_chg_12m",
            "cpi_yoy", "unrate", "spx_ret_12m", "months_since_hike", "cum_easing"]
    X = feat[cols].dropna()
    z = (X - X.mean()) / X.std()
    current = z.iloc[-1]
    eucl = np.sqrt(((z - current) ** 2).sum(axis=1))
    cov_inv = pinv(np.cov(z.values.T))
    diff = z.values - current.values
    maha = np.sqrt(np.einsum("ij,jk,ik->i", diff, cov_inv, diff))
    rank = pd.DataFrame({"euclid": eucl, "mahal": maha}, index=z.index)
    cutoff = z.index[-1] - pd.DateOffset(months=exclude_recent_months)
    rank = rank[rank.index < cutoff]
    top = rank.sort_values("mahal").head(top_n)

    spx = df["spx"].dropna()
    recs = []
    for dt in top.index:
        fr = _forward_returns(spx, dt)
        if fr is None:
            continue
        recs.append({"date": dt.date(), "mahal": round(top.loc[dt, "mahal"], 2), **fr})
    fwd = pd.DataFrame(recs)
    summ = {}
    for hname in HORIZONS:
        if hname in fwd:
            s = fwd[hname].dropna()
            if len(s):
                summ[hname] = {"mean": s.mean(), "median": s.median(), "min": s.min(),
                               "max": s.max(), "hit_rate": (s > 0).mean() * 100, "n": len(s)}
    summary = pd.DataFrame(summ).T
    top.to_csv(f"{OUTDIR}/analogue_ranking.csv")
    fwd.to_csv(f"{OUTDIR}/analogue_forward_returns.csv", index=False)
    summary.to_csv(f"{OUTDIR}/analogue_forward_summary.csv")
    return top, fwd, summary


CANONICAL = {
    "1995-07 soft-landing cut": "1995-07-06",
    "1998-09 insurance cut":    "1998-09-29",
    "2001-01 recession cut":    "2001-01-03",
    "2007-09 crisis cut":       "2007-09-18",
    "2019-07 mid-cycle cut":    "2019-07-31",
    "2024-09 cutting cycle":    "2024-09-18",
    "1967-11 stop-go hike":     "1967-11-20",
    "1973-01 stop-go hike":     "1973-01-15",
    "1977-01 stop-go hike":     "1977-01-03",
    "1980-08 Volcker re-hike":  "1980-08-01",
}


def canonical_forward(df):
    spx = df["spx"].dropna()
    recs = []
    for label, ds in CANONICAL.items():
        fr = _forward_returns(spx, pd.Timestamp(ds))
        if fr is None:
            recs.append({"episode": label, "note": f"no equity data (series starts {spx.index[0].date()})"})
        else:
            recs.append({"episode": label, **{k: round(v, 1) for k, v in fr.items()}})
    out = pd.DataFrame(recs)
    out.to_csv(f"{OUTDIR}/canonical_episodes.csv", index=False)
    return out


# ----------------------------------------------------------------------
# 6b. FED-PATH COMPARISON: which historical period is most like today's rate agenda?
# ----------------------------------------------------------------------
def _dtw(a, b):
    """Dynamic time warping distance between two 1-D arrays (plain numpy)."""
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf); D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, m]


def _monthly_macro(df):
    m = df.resample("ME").last()
    m["fed"] = m["fed_target"].ffill()
    m["cpi_yoy"] = m["cpi"].pct_change(12) * 100.0
    m["spx_ret_12m"] = m["spx"].pct_change(12) * 100.0
    return m


def fed_path_shape_match(df, window=PATH_WINDOW, top_n=5, min_gap=24, exclude_recent=12):
    """Compare the trailing `window`-month fed funds path with every historical window.
    Paths are expressed relative to their END level (shape), then scored on RMSE, DTW and
    a level-difference penalty. Greedy selection keeps matches >= min_gap months apart."""
    m = _monthly_macro(df)
    fed = m["fed"].dropna()
    vals, idx = fed.values, fed.index
    cur = vals[-window:]; cur_shape = cur - cur[-1]
    recs = []
    for end in range(window - 1, len(vals) - exclude_recent):
        seg = vals[end - window + 1:end + 1]
        shape = seg - seg[-1]
        recs.append({"date": idx[end],
                     "rmse": np.sqrt(np.mean((shape - cur_shape) ** 2)),
                     "dtw": _dtw(shape, cur_shape) / window,
                     "level_diff": abs(seg[-1] - cur[-1]),
                     "fed_level": seg[-1]})
    res = pd.DataFrame(recs).set_index("date")
    z = lambda s: (s - s.mean()) / s.std()
    res["score"] = z(res["rmse"]) + z(res["dtw"]) + 0.5 * z(res["level_diff"])
    ranked = res.sort_values("score")
    picks = []
    for dt in ranked.index:
        if all(abs((dt - p).days) > min_gap * 30 for p in picks):
            picks.append(dt)
        if len(picks) >= top_n:
            break
    top = ranked.loc[picks]
    spx = df["spx"].dropna()
    fwd_rows = []
    for dt in picks:
        fr = _forward_returns(spx, dt)
        row = {"date": dt.date(), "score": round(top.loc[dt, "score"], 2),
               "fed_level": round(top.loc[dt, "fed_level"], 2),
               "cpi_yoy": round(m.loc[dt, "cpi_yoy"], 1) if dt in m.index else np.nan,
               "unrate": m.loc[dt, "unrate"] if dt in m.index else np.nan}
        if fr:
            row.update({k: round(v, 1) for k, v in fr.items()})
        fwd_rows.append(row)
    fwd = pd.DataFrame(fwd_rows)
    res.to_csv(f"{OUTDIR}/fed_path_scores_all.csv")
    fwd.to_csv(f"{OUTDIR}/fed_path_top_matches.csv", index=False)
    _plot_path_matches(fed, spx, picks, window)
    return fwd, res


def _plot_path_matches(fed, spx, picks, window, fwd_months=24):
    """Left: fed funds path shapes (aligned at month 0 = today / match date).
    Right: S&P 500 indexed to 100 at match date for the following fwd_months."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    x = np.arange(-window + 1, 1)
    cur = fed.values[-window:]
    ax1.plot(x, cur - cur[-1], color="black", lw=2.5, label=f"current (to {fed.index[-1].date()})")
    spx_m = spx.resample("ME").last()
    for dt in picks:
        i = fed.index.get_loc(dt)
        seg = fed.values[i - window + 1:i + 1]
        ax1.plot(x, seg - seg[-1], lw=1.2, alpha=.8, label=str(dt.date()))
        j = spx_m.index.searchsorted(dt)
        path = spx_m.iloc[j:j + fwd_months + 1]
        if len(path) > 1:
            ax2.plot(range(len(path)), path.values / path.values[0] * 100, lw=1.2, label=str(dt.date()))
    j = spx_m.index.searchsorted(fed.index[-1])
    ax1.axhline(0, color="grey", lw=.6); ax1.set_xlabel("months (0 = match date)")
    ax1.set_ylabel("fed funds minus end level (pp)"); ax1.set_title("Fed funds path shape: current vs best historical matches")
    ax1.legend(fontsize=8)
    ax2.axhline(100, color="grey", lw=.6); ax2.set_xlabel("months after match date")
    ax2.set_ylabel("S&P 500 (match date = 100)"); ax2.set_title("S&P 500 after the matched dates")
    ax2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/fed_path_matches.png", dpi=130); plt.close(fig)


def rehike_after_easing_episodes(df, min_ease=0.50, lookback=36, quiet=6, hike_thr=0.20, dedupe=12):
    """Find every month since 1954 where the fed funds rate rose >= hike_thr after
    (a) having fallen >= min_ease from its max over the prior `lookback` months and
    (b) no rise >= hike_thr in the prior `quiet` months  -> a 'stop-go' pivot back to hiking.
    Episodes are ranked by z-scored distance to today on: cpi_yoy, unrate, cumulative easing
    before the pivot, fed level, trailing 12m S&P return. Forward S&P returns reported."""
    m = _monthly_macro(df)
    fed = m["fed"]
    d = fed.diff()
    events = []
    for i in range(lookback, len(fed)):
        if pd.isna(d.iloc[i]) or d.iloc[i] < hike_thr:
            continue
        prior = fed.iloc[i - lookback:i]
        ease = prior.max() - fed.iloc[i - 1]
        if ease >= min_ease and (d.iloc[i - quiet:i].fillna(0) < hike_thr).all():
            if events and (fed.index[i] - events[-1]["date"]).days < dedupe * 30:
                continue
            events.append({"date": fed.index[i], "fed_level": fed.iloc[i], "easing_before": ease,
                           "cpi_yoy": m["cpi_yoy"].iloc[i], "unrate": m["unrate"].iloc[i],
                           "spx_ret_12m": m["spx_ret_12m"].iloc[i]})
    ev = pd.DataFrame(events).set_index("date")
    if ev.empty:
        print("  no re-hike-after-easing episodes detected"); return ev
    # current state (last month) on the same features
    cur_i = len(fed) - 1
    cur_prior = fed.iloc[max(0, cur_i - lookback):cur_i]
    cur = pd.Series({"fed_level": fed.iloc[-1], "easing_before": cur_prior.max() - fed.iloc[cur_i - 1],
                     "cpi_yoy": m["cpi_yoy"].iloc[-1], "unrate": m["unrate"].iloc[-1],
                     "spx_ret_12m": m["spx_ret_12m"].iloc[-1]})
    feats = ["fed_level", "easing_before", "cpi_yoy", "unrate", "spx_ret_12m"]
    hist = ev[feats].dropna()
    z = (hist - hist.mean()) / hist.std()
    zc = (cur[feats] - hist.mean()) / hist.std()
    ev.loc[z.index, "similarity_dist"] = np.sqrt(((z - zc) ** 2).sum(axis=1))
    ev = ev[ev.index < fed.index[-1] - pd.DateOffset(months=3)]      # exclude the current pivot itself
    spx = df["spx"].dropna()
    for dt in ev.index:
        fr = _forward_returns(spx, dt)
        if fr:
            for k, v in fr.items():
                ev.loc[dt, f"fwd_{k}"] = v
    ev = ev.sort_values("similarity_dist")
    ev.index = ev.index.date
    ev.round(2).to_csv(f"{OUTDIR}/rehike_after_easing_episodes.csv")
    print("\nCurrent state used for ranking:\n" + cur.round(2).to_string())
    return ev


# ----------------------------------------------------------------------
# 7. PLOTS
# ----------------------------------------------------------------------
def plot_corr_heatmap(res, mode="contemp"):
    """Grid of heatmaps, one per lookback window; unreliable cells (n_indep < MIN_INDEP) greyed."""
    lbs = list(LOOKBACKS) + ["ew_hl3y"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.ravel()
    for k, lb in enumerate(lbs):
        ax = axes[k]
        sub = res[(res["mode"] == mode) & (res["lookback"] == lb)]
        if sub.empty:
            ax.set_visible(False); continue
        pv = sub.pivot(index="series", columns="horizon", values="pearson")
        rel = sub.pivot(index="series", columns="horizon", values="reliable")
        cols = [h for h in HORIZONS if h in pv.columns]
        pv, rel = pv[cols], rel[cols]
        ax.imshow(pv.values.astype(float), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols)
        ax.set_yticks(range(len(pv.index))); ax.set_yticklabels(pv.index)
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                v = pv.values[i, j]
                if np.isnan(v):
                    continue
                ok = bool(rel.values[i, j]) if not pd.isna(rel.values[i, j]) else False
                if not ok:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="lightgrey", alpha=.85))
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="black" if ok else "dimgrey")
        rng = f"{sub['from'].min()} to {sub['to'].max()}" if lb != "ew_hl3y" else "all data, 3y half-life"
        ax.set_title(f"lookback={lb}\n{rng}", fontsize=9)
    axes[-1].set_visible(False)
    fig.suptitle(f"Yield-change vs S&P return Pearson r ({mode}); grey = too few independent obs", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/heatmap_{mode}.png", dpi=130)
    plt.close(fig)


def plot_fed_path(df):
    plt.figure(figsize=(13, 5))
    df["fed_target"].plot(label="Fed target (upper/single)")
    df["y2"].plot(label="2y yield", alpha=0.7)
    df["y10"].plot(label="10y yield", alpha=0.7)
    df["y30"].plot(label="30y yield", alpha=0.7)
    for ds in CANONICAL.values():
        plt.axvline(pd.Timestamp(ds), color="grey", ls="--", lw=0.6)
    plt.ylabel("%"); plt.legend(loc="upper right"); plt.title("Fed funds target & 2y/10y/30y Treasury yields (dashed = canonical episodes)")
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/fed_path.png", dpi=130); plt.close()


def plot_forward_dist(fwd):
    cols = [h for h in HORIZONS if h in fwd.columns]
    data = [fwd[c].dropna().values for c in cols]
    plt.figure(figsize=(10, 5))
    try:
        plt.boxplot(data, tick_labels=cols, showmeans=True)      # matplotlib >= 3.9
    except TypeError:
        plt.boxplot(data, labels=cols, showmeans=True)           # older matplotlib
    plt.axhline(0, color="k", lw=0.8)
    plt.ylabel("Forward S&P return (%)")
    plt.title("Forward S&P returns after top analogue dates")
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/forward_distribution.png", dpi=130); plt.close()


# ----------------------------------------------------------------------
# 8. MAIN
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="local CSV with S&P 500 closes (skips all downloads)")
    ap.add_argument("--exclude-recent", type=int, default=18,
                    help="months before today to exclude from analogue search (default 6)")
    args = ap.parse_args()

    df = build_panel(args.csv)
    print("Computing changes & returns...")
    df = add_changes_returns(df)

    print("Correlation tables...")
    res = correlation_tables(df)
    for s in ["y2", "y10", "y30"]:
        print_corr_summary(res, s, "contemp")
    print_corr_summary(res, "y10", "predict")
    print(f"(full table incl. Spearman & all series -> {OUTDIR}/correlations.csv)")

    ex = newey_west_pvalue(df["d_y10_12m"], df["ret_fwd_12m"], lags=252)
    print(f"\n[Newey-West] 12m 10y-change -> fwd 12m return: slope={ex[0]:.4f}, "
          f"t={ex[1]:.2f}, p={ex[2]:.3f} (HAC, 252 lags)")
    ci = block_bootstrap_corr(df["d_y10_12m"], df["ret_trail_12m"])
    print(f"[Block bootstrap 95% CI] contemp 12m corr: ({ci[0]:.3f}, {ci[1]:.3f})")
    no_c, no_n = nonoverlapping_corr(df, "d_y10_12m", "ret_trail_12m", 252)
    print(f"[Non-overlapping] 12m contemp corr={no_c:.3f} (n={no_n})")

    print("\nRolling stock-bond correlation...")
    rolling_stock_bond_corr(df)

    print("Building feature panel & analogue ranking...")
    feat = build_feature_panel(df)
    top, fwd, summary = analogue_ranking(feat, df, exclude_recent_months=args.exclude_recent)
    print(f"\nTop analogue months (Mahalanobis; excluding last {args.exclude_recent} months):")
    print(top.to_string())
    print("\nForward S&P return summary after analogues (%):")
    print(summary.round(1).to_string())

    print("\nCanonical episode forward returns (%):")
    print(canonical_forward(df).to_string(index=False))

    print(f"\nFed-path shape matching (trailing {PATH_WINDOW}m fed funds path vs history):")
    path_top, _ = fed_path_shape_match(df)
    print(path_top.to_string(index=False))

    print("\n'Re-hike after easing' pivots since 1954, ranked by macro similarity to today")
    print("(fwd_* = S&P 500 % return after the pivot month):")
    ev = rehike_after_easing_episodes(df)
    if not ev.empty:
        show = [c for c in ev.columns if c in ("fed_level", "easing_before", "cpi_yoy", "unrate",
                                                "spx_ret_12m", "similarity_dist") or c.startswith("fwd_")]
        print(ev[show].round(1).to_string())

    print("\nPlots...")
    plot_corr_heatmap(res, "contemp")
    plot_corr_heatmap(res, "predict")
    plot_fed_path(df)
    plot_forward_dist(fwd)
    print(f"\nDone. Outputs in ./{OUTDIR}/")


if __name__ == "__main__":
    main()
