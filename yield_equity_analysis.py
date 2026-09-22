"""
yield_equity_analysis.py
------------------------
How, and how much, do US Treasury yields affect S&P 500 performance and valuation?
Goes beyond correlation with mechanism, decomposition, identification and cross-section:

  1. Multi-horizon correlations over 1y/3y/5y/10y/20y/all lookbacks + exponentially weighted
  2. Rolling "rate beta": % S&P move per +100bp in 10y (nominal, real, breakeven)
  3. Decomposition regressions: returns on d(real yield) + d(breakeven); on expected-path vs
     term-premium component; each split by inflation regime (CPI above / below threshold)
  4. Shock classification (Cieslak-Pang style): growth-type vs inflation/policy-type days,
     rolling share -> explains the sign flips in the stock-bond correlation
  5. Event study: S&P reaction to yield surprises on FOMC days vs ordinary days
  6. Large-yield-move event windows: average S&P path after top-decile yield jumps/drops
  7. Valuation channel: earnings yield vs real yield, implied ERP, P/E sensitivity to real
     yields, and a scenario table (what does +50/+100bp do to P/E at constant ERP?)
  8. Cross-section: rate betas by sector ETF and growth-vs-value (long vs short duration)

REQUIREMENTS
  pip install -U pandas numpy matplotlib scipy statsmodels fredapi yfinance curl_cffi xlrd pandas_datareader requests
  export FRED_API_KEY=...   (free key: https://fred.stlouisfed.org/docs/api/api_key.html)

NETWORK / SSL (corporate proxies)
  If you see "CertificateVerifyError", "curl: (60) SSL certificate", or every ticker reported
  as "possibly delisted; no price data found", your network re-signs HTTPS traffic. Options:
    --ca-bundle /path/to/corp-ca.pem   (preferred: export the proxy's root cert from your browser
                                        or ask IT; also sets SSL_CERT_FILE/REQUESTS_CA_BUNDLE)
    --insecure                          (disables certificate verification for Yahoo/Stooq only;
                                        market data is public, but only use this on a network you trust)
  Fallback order per ticker: Yahoo (retries, alternating API paths) -> local cache -> Stooq.

OPTIONAL INPUTS
  --fomc-dates fomc.csv     one date per line (YYYY-MM-DD) of FOMC decision days. Without it
                            the event study uses days the target CHANGED (biased: action days only).
  --shiller-csv shiller.csv Shiller-style monthly data (Date, P, E, CAPE) if the xls download fails.
  --spx-csv spx.csv         local S&P 500 closes (skips Yahoo).
  --cpi-threshold 3.0       inflation regime split (CPI YoY %).

DATA NOTES / LIMITS
  * DFII10 (10y TIPS real yield) and T10YIE (breakeven) start 2003; THREEFYTP10 (Kim-Wright
    term premium) starts 1990. Modules that need them run on the available sample.
  * Pre-2003 "real yield" proxy = nominal 10y minus trailing 12m CPI inflation (crude).
  * Yield changes are in basis points; returns in %. Betas are read as "% per +100bp".
  * All regressions on overlapping windows use HAC (Newey-West) standard errors when
    statsmodels is present; treat long-horizon p-values with suspicion regardless.
"""

import os, sys, time, random, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# ============================================================ CONFIG
FRED_SERIES = {
    "DGS2": "y2", "DGS10": "y10", "DGS30": "y30",
    "DFII10": "r10",            # 10y TIPS real yield (2003-)
    "T10YIE": "be10",           # 10y breakeven inflation (2003-)
    "THREEFYTP10": "tp10",      # Kim-Wright 10y term premium (1990-)
    "DFEDTARU": "ffr_upper", "DFEDTAR": "ffr_single", "FEDFUNDS": "ffr_eff",
    "CPIAUCSL": "cpi",
}
HORIZONS = {"1m": 21, "3m": 63, "6m": 126, "12m": 252, "18m": 378, "24m": 504}
LOOKBACKS = {"1y": 252, "3y": 756, "5y": 1260, "10y": 2520, "20y": 5040, "all": None}
EW_HALFLIFE = 756
MIN_INDEP = 5
START = "1954-07-01"
OUTDIR = "output_yield_equity"
CACHE_DIR = "data"
SECTOR_ETFS = {"XLK": "Tech", "XLF": "Financials", "XLU": "Utilities", "XLRE": "Real Estate",
               "XLE": "Energy", "XLV": "Health Care", "XLP": "Staples", "XLY": "Discretionary",
               "XLI": "Industrials", "XLB": "Materials", "XLC": "Comm Services"}
STYLE_ETFS = {"IWF": "Russell 1000 Growth", "IWD": "Russell 1000 Value"}
SHILLER_URL = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
os.makedirs(OUTDIR, exist_ok=True); os.makedirs(CACHE_DIR, exist_ok=True)


# ============================================================ HELPERS
def ols(y, X, hac_lags=0):
    """OLS with optional Newey-West SEs. X: DataFrame/2D (constant added). Returns DataFrame
    coef/t/p indexed by variable, plus n and r2. Falls back to classical SEs if no statsmodels."""
    d = pd.concat([pd.Series(y, name="_y"), pd.DataFrame(X)], axis=1).dropna()
    if len(d) < 30:
        return None
    yv = d["_y"].values; Xm = d.drop(columns="_y")
    try:
        import statsmodels.api as sm
        Xc = sm.add_constant(Xm)
        fit = (sm.OLS(yv, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
               if hac_lags > 0 else sm.OLS(yv, Xc).fit())
        out = pd.DataFrame({"coef": fit.params, "t": fit.tvalues, "p": fit.pvalues})
        out.attrs.update(n=int(fit.nobs), r2=fit.rsquared)
        return out
    except ImportError:
        Xc = np.column_stack([np.ones(len(d)), Xm.values])
        beta, res, *_ = np.linalg.lstsq(Xc, yv, rcond=None)
        resid = yv - Xc @ beta; s2 = resid @ resid / (len(yv) - Xc.shape[1])
        se = np.sqrt(np.diag(s2 * np.linalg.pinv(Xc.T @ Xc)))
        t = beta / se
        from scipy.stats import t as tdist
        p = 2 * (1 - tdist.cdf(np.abs(t), len(yv) - Xc.shape[1]))
        out = pd.DataFrame({"coef": beta, "t": t, "p": p}, index=["const"] + list(Xm.columns))
        out.attrs.update(n=len(yv), r2=1 - (resid @ resid) / ((yv - yv.mean()) @ (yv - yv.mean())))
        return out


def fmt_reg(res, label, scale_note="% per +100bp", mult=100):
    """mult=100 converts a per-bp slope to per-100bp; use mult=1 for regressions in % units."""
    if res is None:
        print(f"  {label}: insufficient data"); return
    print(f"  {label}  (n={res.attrs['n']}, R2={res.attrs['r2']:.3f})")
    for v, row in res.drop(index="const").iterrows():
        print(f"    {v:<14} {row['coef']*mult:+7.2f} {scale_note}   t={row['t']:+.2f}  p={row['p']:.3f}")


def weighted_corr(x, y, w):
    w = np.asarray(w, float); w = w / w.sum(); x = np.asarray(x, float); y = np.asarray(y, float)
    mx, my = np.sum(w * x), np.sum(w * y)
    cov = np.sum(w * (x - mx) * (y - my)); vx = np.sum(w * (x - mx) ** 2); vy = np.sum(w * (y - my) ** 2)
    return cov / np.sqrt(vx * vy) if vx > 0 and vy > 0 else np.nan


# ============================================================ DATA
def get_fred():
    key = os.environ.get("FRED_API_KEY")
    frames = {}
    try:
        from fredapi import Fred
        fred = Fred(api_key=key)
        for code, name in FRED_SERIES.items():
            for attempt in range(3):
                try:
                    frames[name] = fred.get_series(code, observation_start=START); break
                except Exception as e:
                    if attempt == 2: print(f"  ! FRED {code} failed: {e}")
                    else: time.sleep(2 ** attempt)
        df = pd.DataFrame(frames)
    except ImportError:
        import pandas_datareader.data as web
        df = web.DataReader(list(FRED_SERIES), "fred", START); df.columns = [FRED_SERIES[c] for c in df.columns]
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _clean_close(obj, name):
    if obj is None: return None
    if isinstance(obj, pd.DataFrame):
        if isinstance(obj.columns, pd.MultiIndex):
            lvl0 = obj.columns.get_level_values(0)
            obj = obj.xs("Adj Close" if "Adj Close" in lvl0 else "Close", axis=1, level=0)
        else:
            obj = obj["Adj Close" if "Adj Close" in obj.columns else "Close"]
        if isinstance(obj, pd.DataFrame): obj = obj.iloc[:, 0]
    s = pd.to_numeric(obj, errors="coerce").dropna()
    s.index = pd.to_datetime(s.index)
    if getattr(s.index, "tz", None) is not None: s.index = s.index.tz_localize(None)
    s = s[~s.index.duplicated(keep="last")].sort_index(); s.name = name
    return s


NET = {"insecure": False, "ca_bundle": None, "ssl_problem_seen": False}
STOOQ_SYMBOL = {"^GSPC": "^SPX"}          # other tickers: <TICKER>.US


def _install_yf_log_watch():
    """Capture yfinance's logged warnings so SSL failures can be diagnosed (they are logged, not raised)."""
    import logging
    class _H(logging.Handler):
        def emit(self, rec):
            msg = rec.getMessage()
            if any(k in msg for k in ("Certificate", "certificate", "SSL", "curl: (60)")):
                NET["ssl_problem_seen"] = True
    lg = logging.getLogger("yfinance")
    if not any(isinstance(h, _H) for h in lg.handlers): lg.addHandler(_H())


def _verify_arg():
    if NET["ca_bundle"]: return NET["ca_bundle"]
    return not NET["insecure"]


def _yf_session():
    """curl_cffi session (what recent yfinance requires) honouring --ca-bundle / --insecure."""
    try:
        from curl_cffi import requests as cr
        return cr.Session(impersonate="chrome", verify=_verify_arg())
    except Exception:
        return None


def _requests_session():
    import requests
    ses = requests.Session(); ses.verify = _verify_arg()
    ses.headers.update({"User-Agent": "Mozilla/5.0"}); return ses


def _ssl_hint():
    if NET["ssl_problem_seen"] and not (NET["insecure"] or NET["ca_bundle"]) and not NET.get("hinted"):
        NET["hinted"] = True
        print("  !! SSL certificate verification is failing (proxy/self-signed chain). Re-run with "
              "--ca-bundle <corp-ca.pem> (preferred) or --insecure.")


def _stooq_close(ticker, start, name):
    sym = STOOQ_SYMBOL.get(ticker, f"{ticker}.US")
    try:
        import pandas_datareader.data as web
        raw = web.DataReader(sym, "stooq", start=start, session=_requests_session())
        s = _clean_close(raw, name)
        if s is not None and len(s) > 0:
            print(f"  Stooq OK for {ticker} ({sym}): {len(s)} rows {s.index[0].date()} -> {s.index[-1].date()}"); return s
    except Exception as e:
        print(f"  Stooq {sym} failed: {type(e).__name__}: {e}")
    return None


def yahoo_close(ticker, start=START, min_rows=100, max_attempts=6, base_delay=2.0):
    """Daily close with layered fallbacks:
       1) Yahoo via yfinance, alternating Ticker().history() and download() (history() first: it
          tolerates a failed cookie/crumb fetch, download() does not), exponential backoff,
          custom session honouring --ca-bundle/--insecure
       2) local CSV cache from a previous successful run
       3) Stooq via pandas_datareader (^SPX for the index, <TICKER>.US for ETFs)
    Yahoo's 'possibly delisted; no price data found' usually means a blocked/failed request, not a delisting.
    A logged 'Cookie/crumb fetch failed' warning is NOT fatal on its own; only a raised certificate
    error on both API paths aborts the Yahoo stage for this ticker."""
    cache = os.path.join(CACHE_DIR, f"{ticker.replace('^', '')}.csv")
    try:
        import yfinance as yf; _install_yf_log_watch()
    except ImportError:
        yf = None; print("  yfinance not installed")
    if yf is not None:
        ses = _yf_session(); last = None; raised_ssl = 0
        for attempt in range(1, max_attempts + 1):
            method = "history" if attempt % 2 else "download"
            try:
                if method == "history":
                    raw = yf.Ticker(ticker, session=ses).history(start=start, auto_adjust=False, timeout=30)
                else:
                    raw = yf.download(ticker, start=start, progress=False, auto_adjust=False, threads=False,
                                      timeout=30, session=ses)
                s = _clean_close(raw, ticker)
                if s is not None and len(s) >= min_rows:
                    if attempt > 1: print(f"  Yahoo OK for {ticker} via {method} (attempt {attempt})")
                    s.to_csv(cache, header=True); return s
                last = f"{0 if s is None else len(s)} rows via {method}"
            except TypeError:
                ses = None; last = "yfinance too old for session kwarg -> pip install -U yfinance curl_cffi"
            except Exception as e:
                last = f"{type(e).__name__}: {e}"
                if any(k in last for k in ("certificate", "SSL", "curl: (60)")):
                    NET["ssl_problem_seen"] = True; raised_ssl += 1
            # abandon only if BOTH paths have raised certificate errors and nothing can change that
            if raised_ssl >= 2 and not (NET["insecure"] or NET["ca_bundle"]):
                break
            if attempt < max_attempts:
                time.sleep(min(60, base_delay * 2 ** (attempt - 1)) + random.uniform(0, 1.5))
        print(f"  ! Yahoo {ticker} failed ({last})"); _ssl_hint()
    if os.path.exists(cache):
        s = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]; s.name = ticker
        if len(s) >= min_rows:
            print(f"  using cached {cache} ({s.index[0].date()} -> {s.index[-1].date()}, may be stale)"); return s
    s = _stooq_close(ticker, start, ticker)
    if s is not None and len(s) >= min_rows:
        s.to_csv(cache, header=True); return s
    return None


def get_spx(csv_path=None):
    if csv_path:
        raw = pd.read_csv(csv_path); dc = raw.columns[0]
        raw[dc] = pd.to_datetime(raw[dc]); raw = raw.set_index(dc)
        col = next((c for c in ("Adj Close", "Close", "close", "spx") if c in raw.columns), raw.columns[0])
        return _clean_close(raw[col], "spx")
    s = yahoo_close("^GSPC", min_rows=2000)
    if s is None:
        raise RuntimeError("No S&P 500 data. Provide --spx-csv.")
    s.name = "spx"; return s


def get_shiller(csv_path=None):
    """Monthly Shiller data -> DataFrame[P, E, CAPE] at month end. Returns None (and says why) on failure."""
    try:
        if csv_path:
            d = pd.read_csv(csv_path)
        else:
            try:
                import xlrd  # noqa: F401
            except ImportError:
                print("  ! Shiller .xls needs xlrd:  pip install xlrd   (or pass --shiller-csv). Valuation module skipped.")
                return None
            import io
            content = None
            for attempt in range(3):
                try:
                    r = _requests_session().get(SHILLER_URL, timeout=60); r.raise_for_status(); content = r.content; break
                except Exception as e:
                    if attempt == 2: raise
                    time.sleep(3 * (attempt + 1))
            raw = pd.read_excel(io.BytesIO(content), sheet_name="Data", header=None, skiprows=8, engine="xlrd")
            d = raw.iloc[:, [0, 1, 3, 12]].copy(); d.columns = ["Date", "P", "E", "CAPE"]
            d = d[pd.to_numeric(d["Date"], errors="coerce").notna()]
            yr = np.floor(d["Date"].astype(float)).astype(int)
            mo = np.round((d["Date"].astype(float) - yr) * 100).astype(int).clip(1, 12)
            d["Date"] = pd.to_datetime(dict(year=yr, month=mo, day=1))
        d["Date"] = pd.to_datetime(d["Date"]); d = d.set_index("Date").sort_index()
        for c in ("P", "E", "CAPE"): d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=["P"]); d.index = d.index + pd.offsets.MonthEnd(0)
        d[["P", "E", "CAPE"]].to_csv(os.path.join(CACHE_DIR, "shiller_cache.csv"))
        print(f"  Shiller data OK: {d.index[0].date()} -> {d.index[-1].date()}")
        if (pd.Timestamp.today() - d.index[-1]).days > 120:
            print(f"  !! Shiller series ends {d.index[-1].date()} - the Yale .xls is no longer maintained. Valuation "
                  "figures below are as of that date; pass --shiller-csv with a current CAPE/earnings series for today's view.")
        return d[["P", "E", "CAPE"]]
    except Exception as e:
        cache = os.path.join(CACHE_DIR, "shiller_cache.csv")
        if os.path.exists(cache):
            print(f"  Shiller download failed ({type(e).__name__}); using cached {cache}")
            return pd.read_csv(cache, index_col=0, parse_dates=True)
        print(f"  ! Shiller data unavailable ({type(e).__name__}: {e}); valuation module skipped. "
              "Tip: download ie_data.xls manually, save Date,P,E,CAPE as CSV and pass --shiller-csv")
        return None


def build_panel(spx_csv=None):
    fred = get_fred(); spx = get_spx(spx_csv)
    df = fred.join(spx, how="outer").sort_index()
    df["fed_target"] = df["ffr_upper"].combine_first(df.get("ffr_single")).combine_first(df["ffr_eff"]).ffill()
    df[["cpi", "ffr_eff"]] = df[["cpi", "ffr_eff"]].ffill(limit=40)
    for c in ["y2", "y10", "y30", "r10", "be10", "tp10"]:
        if c in df: df[c] = df[c].ffill(limit=5)
    df = df.reindex(spx.index)                      # trading-day calendar (DFEDTAR is 7-day)
    df["spx"] = df["spx"].ffill(limit=5)
    df["cpi_yoy"] = df["cpi"].pct_change(252) * 100      # ~12m trailing inflation (daily-index approx)
    df["ret1d"] = df["spx"].pct_change() * 100
    for c in ["y2", "y10", "y30", "r10", "be10", "tp10"]:
        if c in df: df[f"d{c}"] = df[c].diff() * 100          # daily change, bp
    df["y10_path"] = df["y10"] - df["tp10"]                    # expected-rate component (bp change below)
    df["dy10_path"] = df["y10_path"].diff() * 100
    # pre-2003 crude real-yield proxy for long-history valuation work
    df["r10_proxy"] = df["r10"].combine_first(df["y10"] - df["cpi_yoy"])
    df["s2s10"] = (df["y10"] - df["y2"]) * 100
    eq = df["spx"].dropna(); print(f"Panel: {eq.index[0].date()} -> {eq.index[-1].date()} ({len(eq)} trading days)")
    return df


def add_changes_returns(df):
    out = df.copy()
    for hname, h in HORIZONS.items():
        for y in ["y2", "y10", "y30", "r10", "be10", "s2s10"]:
            mult = 1.0 if y == "s2s10" else 100.0
            out[f"d_{y}_{hname}"] = (out[y] - out[y].shift(h)) * mult
        out[f"ret_trail_{hname}"] = (out["spx"] / out["spx"].shift(h) - 1) * 100
        out[f"ret_fwd_{hname}"] = (out["spx"].shift(-h) / out["spx"] - 1) * 100
    return out


# ============================================================ 1. CORRELATIONS
def correlation_tables(df):
    from scipy.stats import pearsonr, spearmanr
    rows = []; T = len(df); ew = 0.5 ** (np.arange(T)[::-1] / EW_HALFLIFE)
    for hname, h in HORIZONS.items():
        for y in ["y2", "y10", "y30", "r10", "be10", "s2s10"]:
            dcol = f"d_{y}_{hname}"
            for mode, rcol in (("contemp", f"ret_trail_{hname}"), ("predict", f"ret_fwd_{hname}")):
                pair = df[[dcol, rcol]]
                for lb_name, lb in LOOKBACKS.items():
                    sub = (pair if lb is None else pair.iloc[-lb:]).dropna()
                    if len(sub) < 30: continue
                    rows.append([mode, lb_name, hname, y, len(sub), len(sub) / h,
                                 pearsonr(sub[dcol], sub[rcol])[0], spearmanr(sub[dcol], sub[rcol])[0],
                                 sub.index[0].date(), sub.index[-1].date()])
                m = pair.notna().all(axis=1).values
                if m.sum() >= 30:
                    n_eff = ew[m].sum() ** 2 / (ew[m] ** 2).sum()
                    rows.append([mode, "ew_hl3y", hname, y, int(m.sum()), n_eff / h,
                                 weighted_corr(pair[dcol].values[m], pair[rcol].values[m], ew[m]), np.nan,
                                 pair.index[m][0].date(), pair.index[m][-1].date()])
    res = pd.DataFrame(rows, columns=["mode", "lookback", "horizon", "series", "n", "n_indep",
                                      "pearson", "spearman", "from", "to"])
    res["reliable"] = res["n_indep"] >= MIN_INDEP
    res.to_csv(f"{OUTDIR}/correlations.csv", index=False); return res


def print_corr_summary(res, series, mode="contemp"):
    sub = res[(res["series"] == series) & (res["mode"] == mode)]
    if sub.empty: return
    order = list(LOOKBACKS) + ["ew_hl3y"]
    pv = sub.pivot(index="lookback", columns="horizon", values="pearson").reindex(order)
    rel = sub.pivot(index="lookback", columns="horizon", values="reliable").reindex(order)
    cols = [h for h in HORIZONS if h in pv.columns]; out = pv[cols].astype(object).copy()
    for i in out.index:
        for c in cols:
            v = pv.loc[i, c]; out.loc[i, c] = "" if pd.isna(v) else f"{v:+.2f}" + ("" if rel.loc[i, c] else "*")
    print(f"\n{series} change vs S&P return ({mode}); Pearson r, * = n_indep < {MIN_INDEP}"); print(out.to_string())


def plot_corr_heatmap(res, mode="contemp"):
    lbs = list(LOOKBACKS) + ["ew_hl3y"]; fig, axes = plt.subplots(2, 4, figsize=(18, 8)); axes = axes.ravel()
    for k, lb in enumerate(lbs):
        ax = axes[k]; sub = res[(res["mode"] == mode) & (res["lookback"] == lb)]
        if sub.empty: ax.set_visible(False); continue
        pv = sub.pivot(index="series", columns="horizon", values="pearson")
        rel = sub.pivot(index="series", columns="horizon", values="reliable")
        cols = [h for h in HORIZONS if h in pv.columns]; pv, rel = pv[cols], rel[cols]
        ax.imshow(pv.values.astype(float), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols); ax.set_yticks(range(len(pv.index))); ax.set_yticklabels(pv.index)
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                v = pv.values[i, j]
                if np.isnan(v): continue
                ok = bool(rel.values[i, j]) if not pd.isna(rel.values[i, j]) else False
                if not ok: ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="lightgrey", alpha=.85))
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="black" if ok else "dimgrey")
        ax.set_title(f"lookback={lb}\n{sub['from'].min()} to {sub['to'].max()}", fontsize=9)
    axes[-1].set_visible(False)
    fig.suptitle(f"Yield-change vs S&P return Pearson r ({mode}); grey = too few independent obs")
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/heatmap_{mode}.png", dpi=130); plt.close(fig)


def rolling_stock_bond_corr(df, window=756):
    roll = df["ret1d"].rolling(window).corr(-df["dy10"])
    plt.figure(figsize=(12, 4.5)); roll.plot(); plt.axhline(0, color="k", lw=.8)
    plt.title("Rolling 3y correlation: daily S&P returns vs 10y bond-price proxy (-d yield)\n"
              ">0: bonds & stocks move together (inflation/policy regime); <0: bonds hedge stocks (growth regime)")
    plt.tight_layout(); plt.savefig(f"{OUTDIR}/rolling_stock_bond_corr.png", dpi=130); plt.close(); return roll


# ============================================================ 2. ROLLING RATE BETA
def rolling_rate_beta(df, window=252):
    """Rolling OLS of daily S&P return (%) on daily yield change (bp). Beta*100 = % per +100bp."""
    out = pd.DataFrame(index=df.index)
    for c, lab in (("dy10", "10y nominal"), ("dr10", "10y real"), ("dbe10", "10y breakeven"), ("dy2", "2y nominal")):
        cov = df["ret1d"].rolling(window).cov(df[c]); var = df[c].rolling(window).var()
        out[lab] = cov / var * 100
    fig, ax = plt.subplots(figsize=(12, 5))
    out.dropna(how="all").plot(ax=ax, lw=1.1); ax.axhline(0, color="k", lw=.8)
    ax.set_ylabel("% S&P move per +100bp (rolling 1y beta)"); ax.set_title("S&P 500 rate sensitivity over time")
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/rolling_rate_beta.png", dpi=130); plt.close(fig)
    out.to_csv(f"{OUTDIR}/rolling_rate_beta.csv")
    print("\nCurrent 1y rolling rate betas (% S&P per +100bp, daily data):")
    print(out.dropna(how="all").iloc[-1].round(2).to_string())
    print("Full-sample and recent betas with HAC t-stats:")
    for c, lab in (("dy10", "10y nominal"), ("dr10", "10y real"), ("dbe10", "10y breakeven")):
        for lb_name, lb in (("all", None), ("5y", 1260), ("1y", 252)):
            d = df if lb is None else df.iloc[-lb:]
            fmt_reg(ols(d["ret1d"], d[[c]], hac_lags=5), f"{lab:<14} [{lb_name}]")
    return out


# ============================================================ 3. DECOMPOSITION REGRESSIONS
def decomposition_regressions(df, cpi_threshold=3.0):
    """S&P return on components of the 10y move: real vs breakeven; expected-path vs term premium.
    Daily, weekly and monthly frequencies; split by inflation regime."""
    print("\n=== Decomposition: which part of a yield move hurts equities? ===")
    print("Reading: coef = % S&P move for +100bp in that component, holding the other fixed.")
    for freq, lab, lags in (("D", "daily", 5), ("W-FRI", "weekly", 4), ("ME", "monthly", 3)):
        g = df[["spx", "r10", "be10", "y10_path", "tp10", "cpi_yoy"]].resample(freq).last()
        g["ret"] = g["spx"].pct_change() * 100
        for c in ("r10", "be10", "y10_path", "tp10"): g[f"d{c}"] = g[c].diff() * 100
        print(f"\n-- {lab} --")
        fmt_reg(ols(g["ret"], g[["dr10", "dbe10"]], lags), "real + breakeven        [all]")
        for name, mask in (("low-infl", g["cpi_yoy"] < cpi_threshold), ("high-infl", g["cpi_yoy"] >= cpi_threshold)):
            fmt_reg(ols(g.loc[mask, "ret"], g.loc[mask, ["dr10", "dbe10"]], lags), f"real + breakeven        [CPI {name} {cpi_threshold}%]")
        fmt_reg(ols(g["ret"], g[["dy10_path", "dtp10"]], lags), "exp-path + term-premium [all]")
        for name, mask in (("low-infl", g["cpi_yoy"] < cpi_threshold), ("high-infl", g["cpi_yoy"] >= cpi_threshold)):
            fmt_reg(ols(g.loc[mask, "ret"], g.loc[mask, ["dy10_path", "dtp10"]], lags), f"exp-path + term-premium [CPI {name}]")


# ============================================================ 4. SHOCK CLASSIFICATION
def shock_classification(df, window=252):
    """Cieslak-Pang style sign classification of daily co-moves:
       yields up & stocks up   -> growth news          (positive yield/stock corr)
       yields down & stocks dn -> growth news (negative)
       yields up & stocks down -> hawkish/inflation     (negative yield/stock corr)
       yields down & stocks up -> dovish/disinflation
    Rolling share of 'monetary/inflation-type' days explains the sign of the correlation."""
    d = df[["ret1d", "dy10"]].dropna()
    d = d[(d["ret1d"].abs() >= 0.1) & (d["dy10"].abs() >= 1)]   # both must move; unchanged yields would be misclassified
    same = np.sign(d["ret1d"]) == np.sign(d["dy10"])
    share_growth = same.astype(float).rolling(window).mean()
    yr = pd.DataFrame({"growth_type_share": same.groupby(d.index.year).mean(),
                       "n_days": same.groupby(d.index.year).size()})
    fig, ax = plt.subplots(figsize=(12, 4.5))
    share_growth.plot(ax=ax, lw=1.2); ax.axhline(.5, color="k", lw=.8)
    ax.set_ylabel("share of days"); ax.set_title("Rolling 1y share of 'growth-type' days (yields and stocks move together)\n"
                                                 "above 0.5 => bonds hedge equities; below 0.5 => policy/inflation dominates")
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/shock_type_share.png", dpi=130); plt.close(fig)
    yr.to_csv(f"{OUTDIR}/shock_type_by_year.csv")
    print("\nShare of growth-type days (yields & stocks same direction), last 12 years:")
    print(yr.tail(12).round(2).to_string())
    return share_growth


# ============================================================ 5. FOMC EVENT STUDY
def load_event_dates(path, df):
    if path and os.path.exists(path):
        dts = pd.to_datetime(pd.read_csv(path, header=None).iloc[:, 0], errors="coerce").dropna()
        idx = df.index; dts = pd.DatetimeIndex([idx[idx.searchsorted(d)] for d in dts if d <= idx[-1]])
        print(f"  FOMC dates from {path}: {len(dts)} decision days (incl. holds)"); return dts, False
    chg = df["fed_target"].diff().abs() > 0.05
    dts = df.index[chg & (df.index >= "1990-01-01")]
    print(f"  No --fomc-dates given: using {len(dts)} target-CHANGE days since 1990 (biased: action days only, "
          "intermeeting moves included)")
    return dts, True


def fomc_event_study(df, fomc_path=None):
    """Regress S&P day return on the day's 2y yield change (proxy for policy surprise) on FOMC days
    vs non-FOMC days. Classic result (Bernanke-Kuttner 2005): an unexpected 25bp cut ~ +1% S&P."""
    print("\n=== Event study: S&P reaction to 2y-yield 'surprise' on FOMC days ===")
    dts, biased = load_event_dates(fomc_path, df)
    d = df[["ret1d", "dy2", "dy10", "cpi_yoy"]].dropna(); d = d[d.index >= "1990-01-01"]
    d["fomc"] = d.index.isin(dts)
    for lab, sub in (("FOMC days", d[d["fomc"]]), ("non-FOMC days", d[~d["fomc"]])):
        fmt_reg(ols(sub["ret1d"], sub[["dy2"]], 0), f"{lab:<14} ret ~ d(2y)", "% per +100bp in 2y")
        fmt_reg(ols(sub["ret1d"], sub[["dy10"]], 0), f"{lab:<14} ret ~ d(10y)", "% per +100bp in 10y")
    ev = d[d["fomc"]].copy(); ev["hawkish"] = ev["dy2"] > 0
    print("\n  FOMC-day averages by direction of 2y move:")
    print(ev.groupby("hawkish")[["dy2", "dy10", "ret1d"]].agg(["mean", "count"]).round(2).to_string())
    # 5-day and 21-day follow-through after hawkish vs dovish FOMC days
    fwd5 = (df["spx"].shift(-5) / df["spx"] - 1) * 100; fwd21 = (df["spx"].shift(-21) / df["spx"] - 1) * 100
    ev["fwd_5d"] = fwd5.reindex(ev.index); ev["fwd_21d"] = fwd21.reindex(ev.index)
    print("\n  Follow-through after FOMC day (mean %):")
    print(ev.groupby("hawkish")[["ret1d", "fwd_5d", "fwd_21d"]].mean().round(2).to_string())
    ev.sort_values("dy2").to_csv(f"{OUTDIR}/fomc_event_days.csv")
    fig, ax = plt.subplots(figsize=(7, 6)); ax.scatter(ev["dy2"], ev["ret1d"], s=14, alpha=.6)
    ax.axhline(0, color="grey", lw=.6); ax.axvline(0, color="grey", lw=.6)
    ax.set_xlabel("2y yield change on FOMC day (bp)"); ax.set_ylabel("S&P return on FOMC day (%)")
    ax.set_title("FOMC-day surprise vs equity reaction" + (" (target-change days only)" if biased else ""))
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/fomc_scatter.png", dpi=130); plt.close(fig)
    return ev


# ============================================================ 6. LARGE YIELD MOVE WINDOWS
def big_move_windows(df, q=0.95, pre=5, post=21, cpi_threshold=3.0, since="1990-01-01"):
    """Average cumulative S&P path around top-(1-q) daily 10y jumps and drops, split by inflation regime."""
    d = df[df.index >= since]; thr = d["dy10"].abs().quantile(q)
    px = np.log(df["spx"]); paths = {}
    for lab, mask in (("10y jump", d["dy10"] > thr), ("10y drop", d["dy10"] < -thr)):
        for reg, rmask in (("all", pd.Series(True, d.index)), ("CPI<thr", d["cpi_yoy"] < cpi_threshold), ("CPI>=thr", d["cpi_yoy"] >= cpi_threshold)):
            days = d.index[mask & rmask]; rows = []
            for t in days:
                i = df.index.get_loc(t)
                if i - pre < 0 or i + post >= len(df): continue
                seg = px.iloc[i - pre:i + post + 1].values; rows.append((seg - seg[pre - 1]) * 100)
            if rows: paths[f"{lab} [{reg}] n={len(rows)}"] = np.mean(rows, axis=0)
    fig, ax = plt.subplots(figsize=(11, 5)); x = np.arange(-pre, post + 1)
    for k, v in paths.items():
        ax.plot(x, v, lw=1.3 if "[all]" in k else .9, ls="-" if "jump" in k else "--", label=k)
    ax.axvline(0, color="grey", lw=.6); ax.axhline(0, color="grey", lw=.6)
    ax.set_xlabel("trading days around event (0 = yield-move day)"); ax.set_ylabel("cum. S&P return (%) from t-1")
    ax.set_title(f"S&P path around top-{int((1-q)*100)}% daily 10y moves (|d10y| > {thr:.0f}bp) since {since[:4]}")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{OUTDIR}/big_yield_move_paths.png", dpi=130); plt.close(fig)
    tab = pd.DataFrame({k: {"day0": v[pre], "day+5": v[pre + 5], "day+21": v[post + pre]} for k, v in paths.items()}).T
    print(f"\nS&P cumulative return (%) around large daily 10y moves (|d| > {thr:.0f}bp):"); print(tab.round(2).to_string())
    tab.to_csv(f"{OUTDIR}/big_yield_move_table.csv"); return tab


# ============================================================ 7. VALUATION CHANNEL
def valuation_channel(df, shiller):
    """Earnings yield vs real yield -> implied ERP; sensitivity of log P/E to real yields; scenarios."""
    if shiller is None: return None
    print("\n=== Valuation channel: earnings yield, real yield, ERP ===")
    m = df[["y10", "r10", "r10_proxy", "cpi_yoy"]].resample("ME").last().join(shiller, how="inner")
    m["ey_cape"] = 100 / m["CAPE"]                              # cyclically-adjusted earnings yield, %
    m["ey_trail"] = m["E"] / m["P"] * 100                        # trailing 12m earnings yield, %
    m["erp_cape"] = m["ey_cape"] - m["r10_proxy"]                # ERP proxy (CAPE-based, %)
    m["erp_trail"] = m["ey_trail"] - m["r10_proxy"]
    m["log_cape"] = np.log(m["CAPE"])
    m.to_csv(f"{OUTDIR}/valuation_monthly.csv")
    last = m.dropna(subset=["CAPE"]).iloc[-1]
    print(f"  Latest ({m.index[-1].date()}): CAPE {last['CAPE']:.1f}, CAPE earnings yield {last['ey_cape']:.2f}%, "
          f"trailing E/P {last['ey_trail']:.2f}%, real 10y {last['r10_proxy']:.2f}%, "
          f"ERP(CAPE) {last['erp_cape']:+.2f}%, ERP(trailing) {last['erp_trail']:+.2f}%")
    for name, s in (("ERP(CAPE) percentile since 1960", m.loc["1960":, "erp_cape"]),
                    ("ERP(CAPE) percentile since 2003 (TIPS era)", m.loc["2003":, "erp_cape"])):
        s = s.dropna()
        if len(s): print(f"  {name}: {(s < s.iloc[-1]).mean()*100:.0f}th")
    # sensitivity: d log(CAPE) on d real yield, several samples (levels regression on changes, monthly)
    print("\n  Sensitivity of valuation to real yields (12m changes, HAC):  coef = % change in CAPE per +100bp real yield")
    m["dlog_cape_12"] = m["log_cape"].diff(12) * 100; m["dr_12"] = m["r10_proxy"].diff(12) * 100
    for lab, s in (("since 2003 (TIPS)", m.loc["2003":]), ("since 2010", m.loc["2010":]), ("since 2020", m.loc["2020":]),
                   ("1960-2002 (proxy real yield)", m.loc["1960":"2002"])):
        fmt_reg(ols(s["dlog_cape_12"], s[["dr_12"]], 11), f"{lab:<30}", "% per +100bp")
    # level relationship in the TIPS era
    t = m.loc["2003":].dropna(subset=["ey_cape", "r10"])
    fmt_reg(ols(t["ey_cape"], t[["r10"]], 11), "CAPE earnings yield on TIPS real yield (levels, 2003-) ", "pp EY per +1pp real yield", mult=1)
    # scenario table at constant ERP: EY' = EY + d(real) ; P/E' = 100/EY'
    print("\n  Scenario: P/E if the real 10y moves and the ERP stays constant (pure discount-rate channel):")
    rows = []
    for dr in (-100, -50, 0, 50, 100, 150):
        ey_new = last["ey_cape"] + dr / 100; pe_new = 100 / ey_new if ey_new > 0 else np.nan
        rows.append({"d_real_bp": dr, "real_10y": round(last["r10_proxy"] + dr / 100, 2), "CAPE": round(pe_new, 1),
                     "price_impact_%": round((pe_new / last["CAPE"] - 1) * 100, 1)})
    sc = pd.DataFrame(rows); print(sc.to_string(index=False)); sc.to_csv(f"{OUTDIR}/valuation_scenarios.csv", index=False)
    print("  (Empirical sensitivity above is smaller than this because growth expectations co-move with real yields.)")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    m.loc["1960":, ["ey_cape", "r10_proxy", "erp_cape"]].plot(ax=axes[0], lw=1); axes[0].axhline(0, color="k", lw=.6)
    axes[0].set_title("CAPE earnings yield, real 10y (TIPS; CPI-proxy pre-2003), implied ERP (%)")
    tt = m.loc["2003":].dropna(subset=["r10", "CAPE"])
    sc_ = axes[1].scatter(tt["r10"], tt["CAPE"], c=np.arange(len(tt)), cmap="viridis", s=12)
    axes[1].scatter(tt["r10"].iloc[-1], tt["CAPE"].iloc[-1], color="red", s=60, label="latest"); axes[1].legend()
    axes[1].set_xlabel("10y TIPS real yield (%)"); axes[1].set_ylabel("CAPE"); axes[1].set_title("CAPE vs real yield, 2003- (colour = time)")
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/valuation_channel.png", dpi=130); plt.close(fig)
    return m


# ============================================================ 8. CROSS-SECTION
def cross_section(df, skip=False):
    """Rate betas by sector and style. If discounting matters, long-duration assets (growth, utilities,
    REITs) must be more rate-sensitive than short-duration ones (banks, energy) for the SAME yield move."""
    if skip: return None
    print("\n=== Cross-section: rate beta by sector / style (% per +100bp in 10y, daily, with HAC t) ===")
    tick = {**SECTOR_ETFS, **STYLE_ETFS}; closes = {}
    missing = []
    for t in tick:
        s = yahoo_close(t, start="1998-01-01", min_rows=250)
        if s is not None: closes[t] = s
        else: missing.append(t)
    if missing: print(f"  ! no data for: {', '.join(missing)} (excluded)")
    if len(closes) < 3: print("  too few ETF series downloaded; skipping"); return None
    px = pd.DataFrame(closes).reindex(df.index).ffill(limit=5); rets = px.pct_change() * 100
    rows = []
    for t in rets.columns:
        for lb_name, lb in (("all", None), ("3y", 756)):
            r = rets[t] if lb is None else rets[t].iloc[-lb:]
            X = df[["dy10", "dr10"]].loc[r.index]
            for c in ("dy10", "dr10"):
                res = ols(r, X[[c]], 5)
                if res is not None:
                    rows.append({"ticker": t, "name": tick[t], "window": lb_name, "yield": c,
                                 "beta_%_per_100bp": res.loc[c, "coef"] * 100, "t": res.loc[c, "t"], "n": res.attrs["n"]})
    tab = pd.DataFrame(rows); tab.to_csv(f"{OUTDIR}/sector_rate_betas.csv", index=False)
    show = tab[(tab["yield"] == "dy10")].pivot(index="name", columns="window", values="beta_%_per_100bp").sort_values("3y")
    print("  Beta to 10y NOMINAL (sorted by last-3y beta):"); print(show.round(2).to_string())
    show_r = tab[(tab["yield"] == "dr10")].pivot(index="name", columns="window", values="beta_%_per_100bp").sort_values("3y")
    print("\n  Beta to 10y REAL (TIPS) yield:"); print(show_r.round(2).to_string())
    if "IWF" in rets and "IWD" in rets:
        spread = (np.log(px["IWF"]) - np.log(px["IWD"])).diff() * 100
        for lb_name, lb in (("all", None), ("3y", 756)):
            s = spread if lb is None else spread.iloc[-lb:]
            fmt_reg(ols(s, df[["dr10"]].loc[s.index], 5), f"Growth-minus-Value daily spread on d(real 10y) [{lb_name}]", "% per +100bp")
        gv = (np.log(px["IWF"]) - np.log(px["IWD"])).resample("ME").last(); r10m = df["r10"].resample("ME").last()
        c = pd.concat([gv, r10m], axis=1).dropna(); c.columns = ["gv", "r10"]
        print(f"  Corr(level of log Growth/Value, real 10y): all={c['gv'].corr(c['r10']):+.2f}, "
              f"since 2018={c.loc['2018':,'gv'].corr(c.loc['2018':,'r10']):+.2f}")
        fig, ax = plt.subplots(figsize=(12, 4.5)); c["gv"].plot(ax=ax, color="tab:blue", label="log(Growth/Value)")
        ax2 = ax.twinx(); c["r10"].plot(ax=ax2, color="tab:red", label="10y real yield (%)"); ax2.invert_yaxis()
        ax.set_title("Growth vs Value (long vs short duration) against the real 10y yield (inverted)")
        ax.legend(loc="upper left"); ax2.legend(loc="upper right")
        fig.tight_layout(); fig.savefig(f"{OUTDIR}/growth_value_vs_real_yield.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5)); show["3y"].plot.barh(ax=ax, color="tab:blue"); ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("% per +100bp in 10y (last 3y, daily)"); ax.set_title("Rate sensitivity by sector: long-duration sectors should sit at the bottom")
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/sector_rate_betas.png", dpi=130); plt.close(fig)
    return tab


# ============================================================ MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spx-csv"); ap.add_argument("--fomc-dates"); ap.add_argument("--shiller-csv")
    ap.add_argument("--cpi-threshold", type=float, default=3.0); ap.add_argument("--skip-sectors", action="store_true")
    ap.add_argument("--insecure", action="store_true", help="disable SSL verification for market-data downloads")
    ap.add_argument("--ca-bundle", help="path to corporate CA bundle (.pem) for SSL verification")
    a = ap.parse_args()
    NET["insecure"] = a.insecure; NET["ca_bundle"] = a.ca_bundle
    if a.ca_bundle:
        for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"): os.environ[var] = a.ca_bundle
    if a.insecure:
        import urllib3; urllib3.disable_warnings()
        print("!! SSL verification DISABLED for market-data downloads (--insecure)")

    df = build_panel(a.spx_csv); df = add_changes_returns(df)

    print("\n=== 1. Correlations by lookback window ===")
    res = correlation_tables(df)
    for s in ("y2", "y10", "y30", "r10", "be10", "s2s10"): print_corr_summary(res, s, "contemp")
    print_corr_summary(res, "y10", "predict")
    plot_corr_heatmap(res, "contemp"); plot_corr_heatmap(res, "predict"); rolling_stock_bond_corr(df)

    print("\n=== 2. Rate beta ===");                 rolling_rate_beta(df)
    decomposition_regressions(df, a.cpi_threshold)
    print("\n=== 4. Shock classification ===");      shock_classification(df)
    fomc_event_study(df, a.fomc_dates)
    print("\n=== 6. Large yield-move windows ===");  big_move_windows(df, cpi_threshold=a.cpi_threshold)
    shiller = get_shiller(a.shiller_csv);           valuation_channel(df, shiller)
    cross_section(df, a.skip_sectors)
    print(f"\nDone. Outputs in ./{OUTDIR}/")


if __name__ == "__main__":
    main()
