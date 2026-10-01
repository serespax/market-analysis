#!/usr/bin/env python3
"""
Yield vs Equity (Nasdaq-100) analysis
=====================================

Fetches market data (with retry/back-off), then tests the questions raised in
our discussion:

  1. Is the *day-to-day* correlation between Nasdaq-100 returns and Treasury
     yield changes still negative, even while the index makes new highs?
  2. Does it matter *why* yields rise?  Proxy decomposition:
       - "growth-driven" days  : yields up AND cyclicals/equities up
       - "inflation/term-premium" days : yields up AND equities down
     Plus, if FRED is reachable, a cleaner split of the 10y into
     real yield (DFII10) + breakeven (T10YIE).
  3. Is the rally earnings-driven or multiple-driven?  Proxied via
     concentration / breadth (NDX vs equal-weight S&P, SOX vs NDX).
  4. Is it momentum?  12-1m momentum, 50/200-dma, RSI, up-day streaks,
     and how NDX behaves conditional on the yield-change regime.
  5. Multivariate regression: NDX return ~ d10y + d(30y-10y) + dOil + dUSD.
  6. Cross-section (--no-sectors to skip): yield beta of market-cap buckets
     (mega/large/equal-weight/mid/small/micro, growth/value) and sectors;
     rolling betas by cap; driver-correlation heat-map; stress-vs-relief
     asymmetry per bucket; the Small−Mega size trade against the 10y.

Data sources
------------
  - Yahoo Finance via `yfinance` (no key needed):
        ^NDX, ^GSPC, RSP, ^SOX, ^TNX (10y), ^TYX (30y), ^FVX (5y), ^IRX (3m),
        CL=F (WTI), BZ=F (Brent), DX-Y.NYB (DXY), GC=F (gold), TLT, QQQ
  - FRED CSV endpoint (no key needed, optional): DFII10, T10YIE, T10Y2Y

Install
-------
    pip install yfinance pandas numpy matplotlib requests

Usage
-----
    python yield_equity_analysis.py                 # 3y daily history
    python yield_equity_analysis.py --years 5 --out ./charts
    python yield_equity_analysis.py --no-intraday   # skip live 1-min panel

Behind a corporate proxy ("self signed certificate in certificate chain"):
    python yield_equity_analysis.py --ca-bundle C:/certs/corp-root.pem
    pip install truststore && python yield_equity_analysis.py     # OS trust store
    python yield_equity_analysis.py --source fred                  # skip Yahoo; FRED + Stooq
    python yield_equity_analysis.py --insecure                     # last resort
"""

from __future__ import annotations

import argparse
import functools
import io
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except ImportError:  # pragma: no cover
    sys.exit("matplotlib is required: pip install matplotlib")

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    sys.exit("yfinance is required: pip install yfinance")

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("yield_equity")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)   # its "possibly delisted" spam hides the real error

# --------------------------------------------------------------------------- #
# TLS / corporate-proxy handling
# --------------------------------------------------------------------------- #
# "self signed certificate in certificate chain" == a TLS-inspecting proxy.
# Fix, in order of preference:
#   1. --ca-bundle path/to/corporate-root.pem   (export it from your browser /
#      Windows cert store, or ask IT; it is usually a Zscaler/Netskope/BlueCoat root)
#   2. pip install truststore   -> we use the OS trust store automatically
#   3. --insecure               -> verify=False. Last resort, data-only, don't
#      send credentials over it.
VERIFY: object = True          # True | path-to-pem | False   (requests semantics)
YF_SESSION = None              # curl_cffi session handed to yfinance


def configure_tls(ca_bundle: Optional[str], insecure: bool) -> None:
    global VERIFY, YF_SESSION
    if insecure:
        VERIFY = False
        log.warning("TLS verification DISABLED (--insecure). Market data only; fine for this, not for anything with credentials.")
        try:
            import urllib3
            urllib3.disable_warnings()
        except Exception:  # noqa: BLE001
            pass
    elif ca_bundle:
        p = Path(ca_bundle).expanduser()
        if not p.exists():
            sys.exit(f"--ca-bundle {p} does not exist")
        VERIFY = str(p)
        for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
            os.environ[var] = str(p)
        log.info("Using CA bundle %s", p)
    else:
        try:  # use the OS trust store (Windows/macOS/Linux) – picks up corporate roots pushed by IT
            import truststore
            truststore.inject_into_ssl()
            log.info("truststore: using OS certificate store")
        except ImportError:
            pass

    # yfinance >= 0.2.5x speaks through curl_cffi, which keeps its own CA bundle;
    # hand it a session with the same verify setting.
    try:
        from curl_cffi import requests as cffi_requests
        YF_SESSION = cffi_requests.Session(impersonate="chrome", verify=VERIFY)
    except Exception as exc:  # noqa: BLE001
        log.debug("curl_cffi session unavailable (%s) – yfinance default session", exc)
        YF_SESSION = None


class NonRetryable(Exception):
    """Deterministic failure (HTML instead of CSV, unknown symbol) – don't waste retries."""


SOURCES: dict[str, str] = {}   # friendly name -> where the series actually came from


def _is_tls_error(exc: BaseException) -> bool:
    s = str(exc).lower()
    return "certificate" in s or "ssl" in s or "curl: (60)" in s

# --------------------------------------------------------------------------- #
# Retry helper
# --------------------------------------------------------------------------- #
def retry(max_attempts: int = 5, base_delay: float = 1.5, max_delay: float = 30.0,
          exceptions: tuple = (Exception,)) -> Callable:
    """Exponential back-off with jitter. Re-raises after max_attempts."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    result = fn(*args, **kwargs)
                    # Treat an empty DataFrame/Series as a failure worth retrying
                    if isinstance(result, (pd.DataFrame, pd.Series)) and result.empty:
                        raise ValueError(f"{fn.__name__} returned empty data")
                    return result
                except NonRetryable:
                    raise
                except exceptions as exc:
                    if _is_tls_error(exc):
                        log.error("%s: TLS/certificate failure – corporate proxy? Re-run with "
                                  "--ca-bundle <corporate-root.pem>, `pip install truststore`, or --insecure. (%s)",
                                  fn.__name__, exc)
                        raise
                    if attempt == max_attempts:
                        log.error("%s failed after %d attempts: %s", fn.__name__, attempt, exc)
                        raise
                    sleep_for = min(max_delay, delay) * (0.8 + 0.4 * np.random.rand())
                    log.warning("%s attempt %d/%d failed (%s). Retrying in %.1fs",
                                fn.__name__, attempt, max_attempts, exc, sleep_for)
                    time.sleep(sleep_for)
                    delay *= 2
        return wrapper
    return deco


# --------------------------------------------------------------------------- #
# Data fetching – Yahoo first, FRED and Stooq fill whatever Yahoo misses
# --------------------------------------------------------------------------- #
TICKERS = {                      # friendly name -> Yahoo symbol
    "NDX": "^NDX",        # Nasdaq-100
    "SPX": "^GSPC",       # S&P 500
    "RSP": "RSP",         # Equal-weight S&P 500 (breadth proxy)
    "SOX": "^SOX",        # Philly semis
    "QQQ": "QQQ",
    "TLT": "TLT",
    "UST3M": "^IRX",      # 13-week bill discount yield (%)
    "UST5Y": "^FVX",
    "UST10Y": "^TNX",
    "UST30Y": "^TYX",
    "WTI": "CL=F",
    "BRENT": "BZ=F",
    "DXY": "DX-Y.NYB",
    "GOLD": "GC=F",
}
YIELD_COLS = ["UST3M", "UST5Y", "UST10Y", "UST30Y"]

# Alternate Yahoo symbols tried one-by-one when the batch download drops a ticker
YAHOO_ALTS = {
    "NDX": ["^NDX", "QQQ"], "SPX": ["^GSPC", "SPY"], "RSP": ["RSP"], "SOX": ["^SOX", "SOXX"],
    "TLT": ["TLT"], "GOLD": ["GC=F", "GLD", "IAU"], "BRENT": ["BZ=F", "BNO"], "WTI": ["CL=F", "USO"],
    "UST30Y": ["^TYX"], "UST10Y": ["^TNX"], "UST5Y": ["^FVX"], "UST3M": ["^IRX"], "DXY": ["DX-Y.NYB", "UUP"],
}

# FRED equivalents (daily, no API key). DTWEXBGS is the Fed's broad dollar index –
# not DXY, but its *changes* are a fine substitute for the regression.
FRED_FALLBACK = {
    "NDX": "NASDAQ100", "SPX": "SP500",
    "UST3M": "DGS3MO", "UST5Y": "DGS5", "UST10Y": "DGS10", "UST30Y": "DGS30",
    "WTI": "DCOILWTICO", "BRENT": "DCOILBRENTEU", "DXY": "DTWEXBGS",
}
# Stooq (free CSV, no key) for the things FRED doesn't carry
STOOQ_FALLBACK = {
    "NDX": "^ndx", "SPX": "^spx", "SOX": "^sox", "RSP": "rsp.us", "QQQ": "qqq.us",
    "TLT": "tlt.us", "GOLD": "xauusd", "WTI": "cl.f", "BRENT": "cb.f",
    "UST10Y": "10usy.b", "UST30Y": "30usy.b", "UST5Y": "5usy.b",
}

# Cross-section: market-cap buckets and sectors (all ETFs → Yahoo only)
CAP_TICKERS = {
    "Mega (Top50)": "XLG",      # Invesco S&P 500 Top 50
    "Large (S&P500)": "SPY",
    "Large EqWt": "RSP",
    "Mid (S&P400)": "IJH",
    "Small (R2000)": "IWM",
    "Micro": "IWC",
    "Growth (R1000G)": "IWF",
    "Value (R1000V)": "IWD",
}
SECTOR_TICKERS = {
    "Tech": "XLK", "Semis": "SMH", "Comm Svcs": "XLC", "Cons Disc": "XLY", "Cons Staples": "XLP",
    "Energy": "XLE", "Financials": "XLF", "Reg Banks": "KRE", "Health": "XLV", "Industrials": "XLI",
    "Materials": "XLB", "Real Estate": "XLRE", "Utilities": "XLU", "Homebuilders": "ITB",
    "Unprofitable Gr (ARKK)": "ARKK",
}
CROSS_ALTS = {"Mega (Top50)": ["XLG", "OEF", "MGC"], "Mid (S&P400)": ["IJH", "MDY"],
              "Small (R2000)": ["IWM", "IJR"], "Micro": ["IWC"], "Semis": ["SMH", "SOXX"]}

FRED_SERIES = {                   # extras for the real/breakeven decomposition
    "REAL10Y": "DFII10",   # 10y TIPS real yield
    "BE10Y": "T10YIE",     # 10y breakeven inflation
    "CURVE_10s2s": "T10Y2Y",
}


def _yf_kwargs() -> dict:
    return {"session": YF_SESSION} if YF_SESSION is not None else {}


@retry(max_attempts=3)
def fetch_yahoo_history(tickers: dict[str, str], years: int) -> pd.DataFrame:
    """Daily closes for all tickers, columns renamed to friendly keys."""
    raw = yf.download(
        list(tickers.values()),
        period=f"{years}y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
        **_yf_kwargs(),
    )
    if raw is None or raw.empty:
        raise ValueError("yfinance returned nothing")

    # yfinance returns MultiIndex (field, ticker) for multi-ticker requests
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw.xs("Close", axis=1, level=-1)
    else:
        close = raw[["Close"]]
        close.columns = [list(tickers.values())[0]]

    inv = {v: k for k, v in tickers.items()}
    close = close.rename(columns=inv).dropna(axis=1, how="all").sort_index()
    if close.empty:
        raise ValueError("yfinance returned no usable columns")
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


@retry(max_attempts=2, base_delay=1.0)
def fetch_yahoo_single(symbol: str, years: int) -> pd.Series:
    """Single-ticker history – more robust than the batch endpoint behind proxies."""
    h = yf.Ticker(symbol, **_yf_kwargs()).history(period=f"{years}y", interval="1d", auto_adjust=True)
    if h is None or h.empty or "Close" not in h:
        raise NonRetryable(f"no history for {symbol}")
    s = h["Close"].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.rename(symbol)


@retry(max_attempts=3, base_delay=1.0)
def fetch_yahoo_intraday(symbols: list[str]) -> pd.DataFrame:
    """1-minute bars for today (or last session) – the 'real-time' panel."""
    # 5d so that a pre-US-open run still gets the last completed session
    raw = yf.download(symbols, period="5d", interval="1m", progress=False, auto_adjust=False, **_yf_kwargs())
    if raw is None or raw.empty:
        raise ValueError("no intraday bars")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close = close.dropna()                     # rows where every symbol traded
    if close.empty:
        raise ValueError("no overlapping intraday bars")
    last_day = close.index[-1].normalize()
    return close[close.index.normalize() == last_day]


def _http_get(url: str) -> str:
    if requests is None:
        raise RuntimeError("requests not installed")
    r = requests.get(url, timeout=25, verify=VERIFY,
                     headers={"User-Agent": "Mozilla/5.0 (yield-equity-analysis)"})
    r.raise_for_status()
    return r.text


@retry(max_attempts=3, base_delay=1.0)
def fetch_fred_series(series_id: str) -> pd.Series:
    """FRED public CSV endpoint – no API key required."""
    txt = _http_get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}")
    df = pd.read_csv(io.StringIO(txt))
    df.columns = ["date", series_id]
    df["date"] = pd.to_datetime(df["date"])
    s = pd.to_numeric(df[series_id].replace(".", np.nan), errors="coerce")
    s.index = df["date"]
    return s.dropna().rename(series_id)


@retry(max_attempts=3, base_delay=1.0)
def fetch_stooq_series(symbol: str) -> pd.Series:
    """Stooq daily CSV – no key. Symbol examples: ^ndx, rsp.us, 10usy.b, cb.f"""
    txt = _http_get(f"https://stooq.com/q/d/l/?s={symbol}&i=d")
    first = txt.splitlines()[0] if txt else ""
    if txt.lstrip().startswith("<") or "Date" not in first:
        raise NonRetryable(f"stooq returned HTML/no CSV for {symbol} (bot wall or proxy block)")
    if "Exceeded the daily hits limit" in txt:
        raise NonRetryable("stooq daily hit limit reached")
    df = pd.read_csv(io.StringIO(txt))
    df["Date"] = pd.to_datetime(df["Date"])
    s = pd.to_numeric(df["Close"], errors="coerce")
    s.index = df["Date"]
    return s.dropna().rename(symbol)


def fetch_fred_all(series: dict[str, str]) -> pd.DataFrame:
    out = {}
    for name, sid in series.items():
        try:
            out[name] = fetch_fred_series(sid)
        except Exception as exc:  # noqa: BLE001 – optional data
            log.warning("FRED %s unavailable (%s); skipping", sid, exc)
    return pd.DataFrame(out) if out else pd.DataFrame()


def fetch_cross_section(years: int, source: str) -> pd.DataFrame:
    """Cap-bucket and sector ETFs. Yahoo only (batch + single-ticker retry)."""
    YAHOO_ALTS.update(CROSS_ALTS)
    tick = {**CAP_TICKERS, **SECTOR_TICKERS}
    src = "yahoo" if source == "fred" else source     # FRED has no ETFs
    try:
        cs = fetch_history(tick, years, src)
    except SystemExit as exc:
        log.warning("cross-section unavailable: %s", exc)
        return pd.DataFrame()
    return cs


def fetch_history(tickers: dict[str, str], years: int, source: str) -> pd.DataFrame:
    """
    source = 'auto'  : Yahoo, then FRED, then Stooq for anything still missing
             'yahoo' : Yahoo only
             'fred'  : skip Yahoo entirely (FRED + Stooq) – handy behind strict proxies
    """
    start = pd.Timestamp.today().normalize() - pd.DateOffset(years=years)
    close = pd.DataFrame()

    if source in ("auto", "yahoo"):
        try:
            close = fetch_yahoo_history(tickers, years)
            SOURCES.update({c: f"yahoo batch {tickers[c]}" for c in close.columns})
            log.info("Yahoo batch delivered: %s", list(close.columns))
        except Exception as exc:  # noqa: BLE001
            log.warning("Yahoo unavailable (%s)", exc)
            if source == "yahoo":
                raise

    def _fill(fallback: dict[str, str], fetcher: Callable[[str], pd.Series], label: str):
        nonlocal close
        missing = [k for k in tickers if k not in close.columns or close[k].dropna().empty]
        for name in missing:
            sym = fallback.get(name)
            if not sym:
                continue
            try:
                s = fetcher(sym)
                s = s[s.index >= start]
                if s.empty:
                    continue
                close = close.join(s.rename(name), how="outer") if not close.empty else s.rename(name).to_frame()
                SOURCES[name] = f"{label} {sym}"
                log.info("%s filled %s from %s", label, name, sym)
            except Exception as exc:  # noqa: BLE001
                log.warning("%s could not supply %s (%s)", label, name, exc)

    if source in ("auto", "yahoo"):
        # second pass: anything the batch dropped, one symbol at a time, trying alternates
        missing = [k for k in tickers if k not in close.columns or close[k].dropna().empty]
        for name in missing:
            for sym in YAHOO_ALTS.get(name, [tickers[name]]):
                try:
                    s = fetch_yahoo_single(sym, years)
                    close = close.join(s.rename(name), how="outer") if not close.empty else s.rename(name).to_frame()
                    SOURCES[name] = f"yahoo single {sym}"
                    log.info("Yahoo single-ticker filled %s from %s", name, sym)
                    break
                except Exception as exc:  # noqa: BLE001
                    log.debug("yahoo single %s failed: %s", sym, exc)

    if source in ("auto", "fred"):
        _fill(FRED_FALLBACK, fetch_fred_series, "FRED")
        _fill(STOOQ_FALLBACK, fetch_stooq_series, "Stooq")

    if close.empty:
        raise SystemExit(
            "No market data from any source. If the log shows certificate errors, run with "
            "--ca-bundle <corporate-root.pem>, `pip install truststore`, or --insecure."
        )
    close = close.sort_index()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    # FRED/Stooq holidays differ from Yahoo's – keep NYSE-ish calendar, forward-fill yield gaps by 1 day
    close = close[close.index.dayofweek < 5]
    anchor = "NDX" if "NDX" in close else ("Large (S&P500)" if "Large (S&P500)" in close else None)
    if anchor:
        close = close[close[anchor].notna()]
    close = close.ffill(limit=1)

    still_missing = [k for k in tickers if k not in close.columns]
    if still_missing:
        log.warning("No source for: %s (continuing without them)", still_missing)
    return close


def latest_quotes(symbols: dict[str, str], close: pd.DataFrame) -> pd.DataFrame:
    """Best-effort snapshot: yfinance fast_info, else the last two daily closes we already have."""
    rows = []
    for name, sym in symbols.items():
        last = prev = np.nan
        src = "yahoo live"
        try:
            @retry(max_attempts=2, base_delay=0.5)
            def _one(sym=sym):
                fi = yf.Ticker(sym, **_yf_kwargs()).fast_info
                l = fi.get("last_price") or fi.get("lastPrice")
                p = fi.get("previous_close") or fi.get("previousClose")
                if l is None:
                    raise ValueError("no last price")
                return l, p
            last, prev = _one()
        except Exception:  # noqa: BLE001  – fast_info is flaky for ^indices; try a short history instead
            try:
                h = yf.Ticker(sym, **_yf_kwargs()).history(period="5d", interval="1d")["Close"].dropna()
                if len(h) < 2:
                    raise ValueError("short history")
                last, prev, src = float(h.iloc[-1]), float(h.iloc[-2]), f"yahoo daily {h.index[-1]:%Y-%m-%d}"
                rows.append({"name": name, "symbol": sym, "last": last, "prev": prev,
                             "chg_%": (last / prev - 1) * 100, "source": src})
                continue
            except Exception as exc:  # noqa: BLE001
                pass
            if name in close.columns and close[name].dropna().shape[0] >= 2:
                s = close[name].dropna()
                last, prev, src = s.iloc[-1], s.iloc[-2], f"daily close {s.index[-1]:%Y-%m-%d}"
            else:
                src = f"unavailable ({str(exc)[:40]})"
        chg = (last / prev - 1) * 100 if prev and not np.isnan(prev) else np.nan
        rows.append({"name": name, "symbol": sym, "last": last, "prev": prev, "chg_%": chg, "source": src})
    return pd.DataFrame(rows).set_index("name")


# --------------------------------------------------------------------------- #
# Analytics
# --------------------------------------------------------------------------- #
@dataclass
class OLSResult:
    params: pd.Series
    tstats: pd.Series
    r2: float
    n: int

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"coef": self.params, "t_stat": self.tstats})


def ols(y: pd.Series, X: pd.DataFrame, hac_lags: int = 5) -> OLSResult:
    """OLS with Newey–West (HAC) standard errors – no statsmodels dependency."""
    data = pd.concat([y, X], axis=1).dropna()
    keep = [c for c in X.columns if data[c].std() > 1e-12]          # drop degenerate regressors
    X = X[keep]; data = data[[data.columns[0]] + keep]
    yv = data.iloc[:, 0].values
    Xm = np.column_stack([np.ones(len(data)), data.iloc[:, 1:].values])
    beta, *_ = np.linalg.lstsq(Xm, yv, rcond=None)
    resid = yv - Xm @ beta
    n, k = Xm.shape
    XtX_inv = np.linalg.pinv(Xm.T @ Xm)
    # Newey–West
    S = np.zeros((k, k))
    for lag in range(hac_lags + 1):
        w = 1.0 if lag == 0 else 1 - lag / (hac_lags + 1)
        for t in range(lag, n):
            u = Xm[t] * resid[t]
            v = Xm[t - lag] * resid[t - lag]
            S += w * (np.outer(u, v) + (np.outer(v, u) if lag else 0))
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    names = ["const"] + list(X.columns)
    r2 = 1 - resid.var() / yv.var()
    return OLSResult(pd.Series(beta, index=names), pd.Series(beta / se, index=names), r2, n)


def build_panel(close: pd.DataFrame, fred: pd.DataFrame) -> pd.DataFrame:
    """Daily returns (equities/commodities) and bp changes (yields)."""
    df = close.copy()
    if not fred.empty:
        df = df.join(fred.reindex(df.index).ffill(limit=3), how="left")

    out = pd.DataFrame(index=df.index)
    for c in ["NDX", "SPX", "RSP", "SOX", "QQQ", "TLT", "WTI", "BRENT", "DXY", "GOLD"]:
        if c in df:
            out[f"r_{c}"] = np.log(df[c]).diff() * 100          # % log return
    for c in YIELD_COLS + ["REAL10Y", "BE10Y"]:
        if c in df:
            out[f"d_{c}"] = df[c].diff() * 100                  # bp change
            out[f"lvl_{c}"] = df[c]
    if "UST30Y" in df and "UST10Y" in df:
        out["d_30s10s"] = (df["UST30Y"] - df["UST10Y"]).diff() * 100
        out["lvl_30s10s"] = (df["UST30Y"] - df["UST10Y"]) * 100
    if "UST10Y" in df and "UST3M" in df:
        out["lvl_10y3m"] = (df["UST10Y"] - df["UST3M"]) * 100
    if "r_NDX" in out and "r_RSP" in out:
        out["r_NDX_minus_RSP"] = out["r_NDX"] - out["r_RSP"]    # concentration
    if "r_SOX" in out and "r_NDX" in out:
        out["r_SOX_minus_NDX"] = out["r_SOX"] - out["r_NDX"]    # leadership
    return out


def momentum_stats(px: pd.Series) -> dict:
    px = px.dropna()
    ret = px.pct_change()
    dma50, dma200 = px.rolling(50).mean(), px.rolling(200).mean()
    delta = px.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + up / dn)
    sign = np.sign(ret)
    streak = (sign.groupby((sign != sign.shift()).cumsum()).cumcount() + 1) * sign
    mom_12_1 = px.iloc[-1] / px.iloc[-min(len(px), 252)] - 1 - (px.iloc[-1] / px.iloc[-min(len(px), 21)] - 1)
    return {
        "last": px.iloc[-1],
        "1m_%": (px.iloc[-1] / px.iloc[-22] - 1) * 100 if len(px) > 22 else np.nan,
        "3m_%": (px.iloc[-1] / px.iloc[-64] - 1) * 100 if len(px) > 64 else np.nan,
        "12-1m_mom_%": mom_12_1 * 100,
        "vs_50dma_%": (px.iloc[-1] / dma50.iloc[-1] - 1) * 100,
        "vs_200dma_%": (px.iloc[-1] / dma200.iloc[-1] - 1) * 100,
        "RSI14": rsi.iloc[-1],
        "current_streak_days": int(streak.iloc[-1]),
        "drawdown_from_high_%": (px.iloc[-1] / px.cummax().iloc[-1] - 1) * 100,
        "_rsi": rsi, "_dma50": dma50, "_dma200": dma200,
    }


def regime_table(panel: pd.DataFrame, ycol: str = "d_UST10Y", ecol: str = "r_NDX",
                 window_days: Optional[int] = None) -> pd.DataFrame:
    """Mean NDX return by quintile of daily yield change."""
    d = panel[[ycol, ecol]].dropna()
    if window_days:
        d = d.iloc[-window_days:]
    d["bucket"] = pd.qcut(d[ycol], 5, labels=["Δy very -", "Δy -", "flat", "Δy +", "Δy very +"])
    g = d.groupby("bucket", observed=True)
    return pd.DataFrame({
        f"mean_{ycol}_bp": g[ycol].mean(),
        f"mean_{ecol}_%": g[ecol].mean(),
        "hit_rate_%": g[ecol].apply(lambda s: (s > 0).mean() * 100),
        "n": g.size(),
    })


def good_bad_yield_days(panel: pd.DataFrame, lookback: int = 126) -> pd.DataFrame:
    """
    Classify yield-up days:
      growth-driven   : d10y > 0 and r_RSP (broad cyclicals) > 0
      inflation/TP    : d10y > 0 and r_RSP <= 0 (and oil often up)
    Then show how NDX did on each type.
    """
    breadth = "r_RSP" if "r_RSP" in panel else "r_SPX"        # equal-weight preferred, cap-weight fallback
    cols = ["d_UST10Y", "r_NDX", breadth] + (["r_BRENT"] if "r_BRENT" in panel else [])
    d = panel[cols].dropna().iloc[-lookback:].rename(columns={breadth: "r_RSP"})
    up = d[d["d_UST10Y"] > 0].copy()
    up["type"] = np.where(up["r_RSP"] > 0, "growth-driven (yields↑, breadth↑)",
                          "inflation/term-premium (yields↑, breadth↓)")
    agg = {"d_UST10Y": "mean", "r_NDX": "mean", "r_RSP": "mean"}
    if "r_BRENT" in up:
        agg["r_BRENT"] = "mean"
    out = up.groupby("type").agg(agg)
    out["n_days"] = up.groupby("type").size()
    out["NDX_hit_rate_%"] = up.groupby("type")["r_NDX"].apply(lambda s: (s > 0).mean() * 100)
    return out


def yield_days_by_composition(panel: pd.DataFrame, lookback: int = 126) -> Optional[pd.DataFrame]:
    """
    Non-circular version: classify yield-UP days by what moved the yield, not by equities.
      real-driven      : Δreal10y >= Δbreakeven  (growth / term premium / supply)
      inflation-driven : Δbreakeven > Δreal       (oil, inflation repricing)
    Needs FRED DFII10 + T10YIE.
    """
    if "d_REAL10Y" not in panel or "d_BE10Y" not in panel:
        return None
    cols = ["d_UST10Y", "d_REAL10Y", "d_BE10Y", "r_NDX"] + (["r_BRENT"] if "r_BRENT" in panel else [])
    d = panel[cols].dropna().iloc[-lookback:]
    up = d[d["d_UST10Y"] > 0].copy()
    if up.empty:
        return None
    up["type"] = np.where(up["d_BE10Y"] > up["d_REAL10Y"],
                          "inflation-driven (Δbreakeven > Δreal)", "real-yield-driven (Δreal ≥ Δbreakeven)")
    g = up.groupby("type")
    out = g[[c for c in cols if c != "d_UST10Y"] + ["d_UST10Y"]].mean()
    out["n_days"] = g.size()
    out["NDX_hit_rate_%"] = g["r_NDX"].apply(lambda s: (s > 0).mean() * 100)
    return out


# --------------------------------------------------------------------------- #
# Cross-sectional analytics: who is most rate-sensitive?
# --------------------------------------------------------------------------- #
def cross_returns(cs_close: pd.DataFrame) -> pd.DataFrame:
    return (np.log(cs_close).diff() * 100).dropna(how="all")


def yield_beta_table(rets: pd.DataFrame, panel: pd.DataFrame, ycol: str = "d_UST10Y",
                     recent: int = 126) -> pd.DataFrame:
    """
    For each asset: beta to Δ10y (% per +10bp), HAC t-stat and correlation, full vs recent,
    plus the beta of its EXCESS return over the S&P 500 (relative sensitivity – what an
    allocator actually cares about).
    """
    dy = panel[ycol]
    spx = panel["r_SPX"] if "r_SPX" in panel else None
    rows = []
    for a in rets.columns:
        r = rets[a]
        d = pd.concat([r, dy], axis=1).dropna()
        if len(d) < 60:
            continue
        rec = d.iloc[-recent:]
        b_full = ols(d.iloc[:, 0], d.iloc[:, [1]])
        b_rec = ols(rec.iloc[:, 0], rec.iloc[:, [1]])
        row = {"asset": a,
               "beta_full_%/10bp": b_full.params.iloc[1] * 10, "t_full": b_full.tstats.iloc[1],
               "beta_6m_%/10bp": b_rec.params.iloc[1] * 10, "t_6m": b_rec.tstats.iloc[1],
               "corr_6m": rec.corr().iloc[0, 1], "ret_6m_%": r.iloc[-recent:].sum()}
        if spx is not None and a != "Large (S&P500)":
            ex = (r - spx).dropna()
            e = pd.concat([ex, dy], axis=1).dropna().iloc[-recent:]
            row["excess_beta_6m_%/10bp"] = ols(e.iloc[:, 0], e.iloc[:, [1]]).params.iloc[1] * 10
        rows.append(row)
    return pd.DataFrame(rows).set_index("asset").sort_values("beta_6m_%/10bp")


def rolling_yield_beta(rets: pd.DataFrame, dy: pd.Series, window: int = 60) -> pd.DataFrame:
    """Rolling OLS beta (% per +10bp) of each asset to Δ10y."""
    out = {}
    for a in rets.columns:
        d = pd.concat([rets[a], dy], axis=1).dropna()
        cov = d.iloc[:, 0].rolling(window).cov(d.iloc[:, 1])
        var = d.iloc[:, 1].rolling(window).var()
        out[a] = (cov / var) * 10
    return pd.DataFrame(out)


def driver_corr_matrix(rets: pd.DataFrame, panel: pd.DataFrame, recent: int = 126) -> pd.DataFrame:
    drivers = [c for c in ["d_UST5Y", "d_UST10Y", "d_UST30Y", "d_30s10s", "d_REAL10Y", "d_BE10Y", "r_BRENT", "r_DXY"]
               if c in panel]
    d = pd.concat([rets, panel[drivers]], axis=1).dropna().iloc[-recent:]
    m = d.corr().loc[rets.columns, drivers]
    m.columns = [c.replace("d_UST", "Δ").replace("d_", "Δ").replace("r_", "").replace("Y", "y") for c in drivers]
    key = "Δ10y" if "Δ10y" in m.columns else m.columns[0]
    return m.sort_values(key)          # most yield-sensitive at the top


def regime_perf_by_asset(rets: pd.DataFrame, dy: pd.Series, recent: int = 126) -> pd.DataFrame:
    """Avg return on big-yield-UP vs big-yield-DOWN days (top/bottom quintile), last `recent` days."""
    d = pd.concat([rets, dy.rename("dy")], axis=1).dropna().iloc[-recent:]
    hi, lo = d["dy"].quantile(.8), d["dy"].quantile(.2)
    up, dn = d[d["dy"] >= hi], d[d["dy"] <= lo]
    out = pd.DataFrame({"yield_UP_days_%": up[rets.columns].mean(),
                        "yield_DOWN_days_%": dn[rets.columns].mean()})
    out["asymmetry"] = out["yield_DOWN_days_%"] + out["yield_UP_days_%"]   # >0: gains on relief exceed losses on stress
    out["n_up/n_down"] = f"{len(up)}/{len(dn)}"
    return out.sort_values("yield_UP_days_%")


def size_spread(cs_close: pd.DataFrame, panel: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Small-minus-Mega and Mid-minus-Mega cumulative spreads vs 10y level."""
    need = ["Small (R2000)", "Mega (Top50)"]
    if not all(c in cs_close for c in need):
        return None
    r = cross_returns(cs_close)
    out = pd.DataFrame({"Small − Mega (cum %)": (r["Small (R2000)"] - r["Mega (Top50)"]).cumsum()})
    if "Mid (S&P400)" in r:
        out["Mid − Mega (cum %)"] = (r["Mid (S&P400)"] - r["Mega (Top50)"]).cumsum()
    if "Large EqWt" in r:
        out["EqWt − Mega (cum %)"] = (r["Large EqWt"] - r["Mega (Top50)"]).cumsum()
    out["Small − Mega (daily)"] = r["Small (R2000)"] - r["Mega (Top50)"]
    out = out.join(panel[["lvl_UST10Y", "d_UST10Y"]], how="left")
    return out


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _save(fig, out: Path, name: str):
    fig.tight_layout()
    p = out / name
    fig.savefig(p, dpi=140)
    plt.close(fig)
    log.info("saved %s", p)


def plot_levels(close: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    ax = axes[0]
    for c, lab in [("NDX", "Nasdaq-100"), ("SPX", "S&P 500"), ("RSP", "S&P equal-wt"), ("SOX", "SOX")]:
        if c in close:
            ax.plot(close.index, close[c] / close[c].dropna().iloc[0] * 100, label=lab, lw=1.3)
    ax.set_title("Equities, rebased = 100"); ax.legend(loc="upper left"); ax.grid(alpha=.3)

    ax = axes[1]
    for c, lab in [("UST3M", "3m"), ("UST5Y", "5y"), ("UST10Y", "10y"), ("UST30Y", "30y")]:
        if c in close:
            ax.plot(close.index, close[c], label=lab, lw=1.3)
    ax.set_title("US Treasury yields (%)"); ax.legend(loc="upper left"); ax.grid(alpha=.3)

    ax = axes[2]
    if "BRENT" in close:
        ax.plot(close.index, close["BRENT"], color="black", lw=1.3, label="Brent ($)")
    ax.set_ylabel("Brent"); ax.grid(alpha=.3)
    if "DXY" in close:
        ax2 = ax.twinx(); ax2.plot(close.index, close["DXY"], color="tab:green", lw=1.1, label="DXY")
        ax2.set_ylabel("DXY")
    ax.set_title("Oil and dollar – the 'why yields are rising' inputs")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    _save(fig, out, "01_levels.png")


def plot_rolling_corr(panel: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(12, 5))
    for ycol, lab in [("d_UST10Y", "10y"), ("d_UST30Y", "30y"), ("d_UST5Y", "5y")]:
        if ycol in panel:
            for w, ls in [(20, "-"), (60, "--")]:
                rc = panel["r_NDX"].rolling(w).corr(panel[ycol])
                ax.plot(rc.index, rc, ls, lw=1.2, label=f"NDX ret vs Δ{lab}  ({w}d)")
    ax.axhline(0, color="k", lw=.8)
    ax.set_title("Rolling correlation: Nasdaq-100 daily return vs daily yield change\n"
                 "(negative = 'yields up, tech down' is still the day-to-day rule)")
    ax.legend(ncol=3, fontsize=8); ax.grid(alpha=.3)
    _save(fig, out, "02_rolling_corr.png")


def plot_scatter(panel: pd.DataFrame, out: Path, recent_days: int = 126):
    d = panel[["d_UST10Y", "r_NDX"]].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (sub, title) in zip(axes, [(d, "Full sample"), (d.iloc[-recent_days:], f"Last {recent_days} days")]):
        ax.scatter(sub["d_UST10Y"], sub["r_NDX"], s=14, alpha=.55)
        b = np.polyfit(sub["d_UST10Y"], sub["r_NDX"], 1)
        xs = np.linspace(sub["d_UST10Y"].min(), sub["d_UST10Y"].max(), 50)
        ax.plot(xs, np.polyval(b, xs), "r-", lw=1.5,
                label=f"slope {b[0]*10:.2f}% per +10bp,  ρ={sub.corr().iloc[0,1]:.2f}")
        # highlight the last 10 sessions
        tail = sub.iloc[-10:]
        ax.scatter(tail["d_UST10Y"], tail["r_NDX"], s=45, color="orange", edgecolor="k", label="last 10 sessions")
        ax.axhline(0, color="k", lw=.6); ax.axvline(0, color="k", lw=.6)
        ax.set_xlabel("Δ 10y yield (bp)"); ax.set_ylabel("NDX return (%)"); ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(alpha=.3)
    _save(fig, out, "03_scatter_ndx_vs_d10y.png")


def plot_regime_bars(reg_full: pd.DataFrame, reg_recent: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(reg_full)); w = .38
    ax.bar(x - w/2, reg_full.iloc[:, 1], w, label="full sample")
    ax.bar(x + w/2, reg_recent.iloc[:, 1], w, label="last 6 months")
    ax.set_xticks(x); ax.set_xticklabels(reg_full.index, rotation=0)
    ax.axhline(0, color="k", lw=.8)
    ax.set_ylabel("mean NDX daily return (%)")
    ax.set_title("Nasdaq-100 return conditional on the size of the daily 10y yield move")
    ax.legend(); ax.grid(alpha=.3, axis="y")
    _save(fig, out, "04_regime_bars.png")


def plot_good_bad(gb: pd.DataFrame, comp: Optional[pd.DataFrame], out: Path, breadth_label: str):
    ncol = 2 if comp is not None else 1
    fig, axes = plt.subplots(1, ncol, figsize=(7 * ncol, 5.2))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    cols = [c for c in ["r_NDX", "r_RSP", "r_BRENT"] if c in gb]
    gb[cols].rename(columns={"r_NDX": "NDX", "r_RSP": breadth_label, "r_BRENT": "Brent"}).plot(kind="bar", ax=ax, rot=0)
    ax.legend(fontsize=8)
    ax.axhline(0, color="k", lw=.8); ax.set_ylabel("mean daily return (%) on yield-UP days")
    ax.set_title(f"By equity breadth ({breadth_label})\n[partly circular – NDX and breadth co-move]", fontsize=10)
    ax.grid(alpha=.3, axis="y")
    ax.set_xticklabels([t.get_text().replace(" (", "\n(") for t in ax.get_xticklabels()], fontsize=8)
    for i, n in enumerate(gb["n_days"]):
        ax.text(i, ax.get_ylim()[1] * .9, f"n={n}", ha="center", fontsize=9)

    if comp is not None:
        ax = axes[1]
        ccols = [c for c in ["r_NDX", "r_BRENT"] if c in comp]
        comp[ccols].rename(columns={"r_NDX": "NDX", "r_BRENT": "Brent"}).plot(kind="bar", ax=ax, rot=0)
        ax.legend(fontsize=8)
        ax.axhline(0, color="k", lw=.8)
        ax.set_title("By yield composition (real vs breakeven)\n[exogenous to equities]", fontsize=10)
        ax.grid(alpha=.3, axis="y")
        ax.set_xticklabels([t.get_text().replace(" (", "\n(") for t in ax.get_xticklabels()], fontsize=8)
        for i, n in enumerate(comp["n_days"]):
            ax.text(i, ax.get_ylim()[1] * .9, f"n={n}", ha="center", fontsize=9)
    fig.suptitle("Does it matter *why* yields rise?  NDX on yield-UP days, last 6 months", fontsize=12)
    _save(fig, out, "05_good_vs_bad_yield_days.png")


def plot_real_breakeven(panel: pd.DataFrame, out: Path):
    if "lvl_REAL10Y" not in panel or "lvl_BE10Y" not in panel:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(panel.index, panel["lvl_REAL10Y"], label="10y real (TIPS)", lw=1.3)
    axes[0].plot(panel.index, panel["lvl_BE10Y"], label="10y breakeven", lw=1.3)
    if "lvl_UST10Y" in panel:
        axes[0].plot(panel.index, panel["lvl_UST10Y"], "k--", lw=1, label="10y nominal")
    axes[0].set_title("Decomposing the 10y: real yield vs inflation compensation (%)")
    axes[0].legend(); axes[0].grid(alpha=.3)
    for col, lab in [("d_REAL10Y", "Δ real"), ("d_BE10Y", "Δ breakeven")]:
        rc = panel["r_NDX"].rolling(60).corr(panel[col])
        axes[1].plot(rc.index, rc, lw=1.3, label=f"60d corr NDX vs {lab}")
    axes[1].axhline(0, color="k", lw=.8); axes[1].legend(); axes[1].grid(alpha=.3)
    axes[1].set_title("Which component hurts tech more?  (typically real yields >> breakevens)")
    _save(fig, out, "06_real_vs_breakeven.png")


def plot_breadth_momentum(close: pd.DataFrame, mom: dict, out: Path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    ax = axes[0]
    if "RSP" in close and "NDX" in close:
        ratio = close["NDX"] / close["RSP"]; ax.plot(ratio.index, ratio / ratio.dropna().iloc[0] * 100, lw=1.3, label="NDX / S&P equal-weight")
    if "SOX" in close and "NDX" in close:
        ratio = close["SOX"] / close["NDX"]; ax.plot(ratio.index, ratio / ratio.dropna().iloc[0] * 100, lw=1.3, label="SOX / NDX")
    ax.set_title("Concentration & leadership (rebased=100; rising = narrower, semis-led)"); ax.legend(); ax.grid(alpha=.3)

    ax = axes[1]
    px = close["NDX"].dropna()
    ax.plot(px.index, px, lw=1.2, label="NDX"); ax.plot(mom["_dma50"].index, mom["_dma50"], lw=1, label="50dma")
    ax.plot(mom["_dma200"].index, mom["_dma200"], lw=1, label="200dma"); ax.legend(); ax.grid(alpha=.3)
    ax.set_title("Trend / momentum")

    ax = axes[2]
    ax.plot(mom["_rsi"].index, mom["_rsi"], lw=1.1, color="purple"); ax.axhline(70, color="r", ls="--", lw=.8); ax.axhline(30, color="g", ls="--", lw=.8)
    ax.set_ylim(0, 100); ax.set_title("RSI(14)"); ax.grid(alpha=.3)
    _save(fig, out, "07_breadth_momentum.png")


def plot_yield_beta_bars(bt: pd.DataFrame, out: Path):
    caps = [a for a in bt.index if a in CAP_TICKERS]
    secs = [a for a in bt.index if a in SECTOR_TICKERS]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [len(caps), len(secs)]})
    for ax, grp, title in zip(axes, [caps, secs], ["Market-cap / style buckets", "Sectors & rate-sensitive industries"]):
        sub = bt.loc[grp].sort_values("beta_6m_%/10bp")
        y = np.arange(len(sub)); h = .38
        ax.barh(y + h/2, sub["beta_full_%/10bp"], h, label="full sample", color="lightgrey", edgecolor="grey")
        bars = ax.barh(y - h/2, sub["beta_6m_%/10bp"], h, label="last 6 months",
                       color=["tab:red" if v < 0 else "tab:green" for v in sub["beta_6m_%/10bp"]])
        for yi, (v, t) in enumerate(zip(sub["beta_6m_%/10bp"], sub["t_6m"])):
            ax.text(v + (0.02 if v >= 0 else -0.02), yi - h/2, f"t={t:.1f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=7.5)
        ax.set_yticks(y); ax.set_yticklabels(sub.index, fontsize=9)
        ax.axvline(0, color="k", lw=.8); ax.grid(alpha=.3, axis="x")
        ax.set_xlabel("return (%) per +10bp in 10y yield"); ax.set_title(title)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Who is most yield-sensitive?  Daily beta to Δ10y (red = hurt by rising yields)", fontsize=12)
    _save(fig, out, "09_yield_beta_by_asset.png")


def plot_rolling_beta_caps(rb: pd.DataFrame, panel: pd.DataFrame, out: Path):
    cols = [c for c in ["Mega (Top50)", "Large (S&P500)", "Large EqWt", "Mid (S&P400)", "Small (R2000)", "Growth (R1000G)", "Value (R1000V)"] if c in rb]
    fig, ax = plt.subplots(figsize=(13, 6))
    for c in cols:
        ax.plot(rb.index, rb[c], lw=1.4 if "Mega" in c or "Small" in c else 1, label=c)
    ax.axhline(0, color="k", lw=.8); ax.set_ylabel("60d rolling beta (% per +10bp)")
    ax.legend(loc="lower left", fontsize=8, ncol=2); ax.grid(alpha=.3)
    ax2 = ax.twinx(); ax2.plot(panel.index, panel["lvl_UST10Y"], color="black", ls=":", lw=1.2, label="10y yield (rhs)")
    ax2.set_ylabel("10y yield (%)"); ax2.legend(loc="upper left", fontsize=8)
    ax.set_title("Rate sensitivity through time by market cap – does the gap widen when yields are high?")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    _save(fig, out, "10_rolling_beta_by_cap.png")


def plot_driver_heatmap(m: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(m) + 2))
    im = ax.imshow(m.values, cmap="RdBu", vmin=-.7, vmax=.7, aspect="auto")
    ax.set_xticks(range(m.shape[1])); ax.set_xticklabels(m.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(m.shape[0])); ax.set_yticklabels(m.index, fontsize=9)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color="white" if abs(v) > .4 else "black")
    fig.colorbar(im, ax=ax, fraction=.025, pad=.02, label="correlation of daily returns, last 6m")
    ax.set_title("Which macro driver does each bucket/sector actually trade on?")
    _save(fig, out, "11_driver_corr_heatmap.png")


def plot_regime_by_asset(rp: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(13, 6.5))
    y = np.arange(len(rp)); h = .4
    ax.barh(y + h/2, rp["yield_DOWN_days_%"], h, color="tab:green", label="big yield-DOWN days (bottom quintile)")
    ax.barh(y - h/2, rp["yield_UP_days_%"], h, color="tab:red", label="big yield-UP days (top quintile)")
    ax.set_yticks(y); ax.set_yticklabels(rp.index, fontsize=9); ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("average daily return (%)"); ax.grid(alpha=.3, axis="x"); ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Stress vs relief: how each bucket behaves on the biggest yield days (last 6m)\n"
                 "green bar longer than red bar = convex (gains on relief > losses on stress)")
    _save(fig, out, "12_regime_by_asset.png")


def plot_size_spread(sp: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    for c in [c for c in sp.columns if "cum %" in c]:
        ax.plot(sp.index, sp[c], lw=1.3, label=c)
    ax.axhline(0, color="k", lw=.8); ax.set_ylabel("cumulative relative return (%)"); ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=.3)
    ax2 = ax.twinx(); ax2.plot(sp.index, sp["lvl_UST10Y"], "k:", lw=1.2, label="10y (rhs)"); ax2.set_ylabel("10y yield (%)"); ax2.legend(loc="upper left", fontsize=8)
    ax.set_title("Small/mid/equal-weight vs mega-caps: the size trade against the 10y")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    ax = axes[1]
    d = sp[["d_UST10Y", "Small − Mega (daily)"]].dropna().iloc[-126:]
    ax.scatter(d["d_UST10Y"], d["Small − Mega (daily)"], s=14, alpha=.6)
    b = np.polyfit(d["d_UST10Y"], d["Small − Mega (daily)"], 1); xs = np.linspace(d["d_UST10Y"].min(), d["d_UST10Y"].max(), 30)
    ax.plot(xs, np.polyval(b, xs), "r-", label=f"slope {b[0]*10:+.2f}pp per +10bp, ρ={d.corr().iloc[0,1]:.2f}")
    ax.axhline(0, color="k", lw=.6); ax.axvline(0, color="k", lw=.6); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_xlabel("Δ10y (bp)"); ax.set_ylabel("Small − Mega daily (pp)"); ax.set_title("Last 6m: small caps vs mega on yield days")
    _save(fig, out, "13_size_spread_vs_yields.png")


def plot_intraday(intra: pd.DataFrame, out: Path):
    cols = list(intra.columns)
    eq = next((c for c in cols if c in ("QQQ", "^NDX")), None)
    yl = next((c for c in cols if c in ("^TNX",)), None)
    if eq is None or yl is None:
        return
    d = intra[[eq, yl]].dropna()
    if len(d) < 10:
        log.warning("intraday panel skipped: too few overlapping bars")
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(d.index, d[eq] / d[eq].iloc[0] * 100 - 100, color="tab:blue", label=f"{eq} (% from open)")
    ax.set_ylabel("% chg"); ax.grid(alpha=.3)
    ax2 = ax.twinx(); ax2.plot(d.index, (d[yl] - d[yl].iloc[0]) * 100, color="tab:red", label="Δ10y (bp)")
    ax2.set_ylabel("Δ 10y yield (bp)")
    corr = d[eq].pct_change().corr(d[yl].diff())
    ax.set_title(f"Intraday (1-min): {eq} vs 10y yield – session {d.index[0]:%Y-%m-%d}   |   1-min corr = {corr:.2f}")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    _save(fig, out, "08_intraday_live.png")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", type=int, default=3, help="years of daily history (default 3)")
    ap.add_argument("--out", type=Path, default=Path("charts"), help="output folder for PNGs/CSV")
    ap.add_argument("--no-intraday", action="store_true", help="skip live 1-minute panel")
    ap.add_argument("--no-fred", action="store_true", help="skip FRED real/breakeven series")
    ap.add_argument("--no-sectors", action="store_true", help="skip market-cap / sector cross-section")
    ap.add_argument("--source", choices=["auto", "yahoo", "fred"], default="auto",
                    help="auto = Yahoo then FRED/Stooq fill-in; fred = skip Yahoo entirely")
    ap.add_argument("--ca-bundle", metavar="PEM", help="corporate root CA (fixes 'self signed certificate in chain')")
    ap.add_argument("--insecure", action="store_true", help="disable TLS verification (last resort)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configure_tls(args.ca_bundle, args.insecure)
    pd.set_option("display.width", 160, "display.max_columns", 20, "display.float_format", "{:,.2f}".format)

    # ---- 1. Data ----------------------------------------------------------
    log.info("Fetching daily history (%dy) ...", args.years)
    close = fetch_history(TICKERS, args.years, args.source)
    if "NDX" not in close or "UST10Y" not in close:
        raise SystemExit(f"Need at least NDX and UST10Y; got {list(close.columns)}")
    fred = pd.DataFrame() if args.no_fred else fetch_fred_all(FRED_SERIES)
    panel = build_panel(close, fred)
    close.to_csv(args.out / "levels.csv"); panel.to_csv(args.out / "panel_returns_bp.csv")

    print("\n=== LIVE SNAPSHOT ===")
    snap = latest_quotes({k: TICKERS[k] for k in ["NDX", "SPX", "SOX", "UST10Y", "UST30Y", "BRENT", "DXY"]}, close)
    print(snap.to_string())
    def _last(col):
        return panel[col].dropna().iloc[-1] if col in panel and panel[col].notna().any() else np.nan
    print(f"\nLast close 10y = {_last('lvl_UST10Y'):.2f}%   30y = {_last('lvl_UST30Y'):.2f}%   "
          f"30s10s = {_last('lvl_30s10s'):.0f}bp   10y3m = {_last('lvl_10y3m'):.0f}bp")

    # ---- 2. Correlations --------------------------------------------------
    print("\n=== CORRELATION: NDX daily return vs yield changes ===")
    ycols = [c for c in ["d_UST5Y", "d_UST10Y", "d_UST30Y", "d_30s10s", "d_REAL10Y", "d_BE10Y", "r_BRENT", "r_DXY"] if c in panel]
    corr_tbl = pd.DataFrame({
        "full": [panel["r_NDX"].corr(panel[c]) for c in ycols],
        "last_126d": [panel["r_NDX"].iloc[-126:].corr(panel[c].iloc[-126:]) for c in ycols],
        "last_21d": [panel["r_NDX"].iloc[-21:].corr(panel[c].iloc[-21:]) for c in ycols],
    }, index=ycols)
    print(corr_tbl.to_string())

    # ---- 3. Regression ----------------------------------------------------
    print("\n=== OLS (Newey-West t-stats): NDX return ~ Δ10y + Δ(30s10s) + Brent + DXY ===")
    X = panel[[c for c in ["d_UST10Y", "d_30s10s", "r_BRENT", "r_DXY"] if c in panel]]
    for label, sl in [("full sample", slice(None)), ("last 126 days", slice(-126, None))]:
        res = ols(panel["r_NDX"].iloc[sl], X.iloc[sl])
        print(f"\n-- {label}: n={res.n}, R²={res.r2:.3f}")
        print(res.to_frame().to_string())
    if "d_REAL10Y" in panel and "d_BE10Y" in panel:
        print("\n=== OLS: NDX return ~ Δreal10y + Δbreakeven10y (which component bites?) ===")
        res = ols(panel["r_NDX"], panel[["d_REAL10Y", "d_BE10Y"]])
        print(f"n={res.n}, R²={res.r2:.3f}"); print(res.to_frame().to_string())

    # ---- 4. Regimes & good/bad yield days ---------------------------------
    print("\n=== NDX return by quintile of daily Δ10y (full sample) ===")
    reg_full = regime_table(panel); print(reg_full.to_string())
    print("\n=== same, last 126 days ===")
    reg_recent = regime_table(panel, window_days=126); print(reg_recent.to_string())

    breadth_label = "S&P equal-wt" if "r_RSP" in panel else "S&P 500 (cap-wt fallback)"
    print(f"\n=== Yield-UP days, last 126d, split by equity breadth [{breadth_label}] ===")
    gb = good_bad_yield_days(panel); print(gb.to_string())
    comp = yield_days_by_composition(panel)
    if comp is not None:
        print("\n=== Same days, split by what moved the yield (real vs breakeven) – non-circular ===")
        print(comp.to_string())

    # ---- 5. Momentum / concentration -------------------------------------
    print("\n=== Momentum / trend diagnostics (NDX) ===")
    mom = momentum_stats(close["NDX"])
    print(pd.Series({k: v for k, v in mom.items() if not k.startswith("_")}).to_string())
    if "SOX" in close:
        sox_mom = momentum_stats(close["SOX"])
        print(f"\nSOX: 1m {sox_mom['1m_%']:.1f}%, streak {sox_mom['current_streak_days']}d, RSI {sox_mom['RSI14']:.0f}")
    if "r_NDX_minus_RSP" in panel:
        rel = panel["r_NDX_minus_RSP"].iloc[-63:].sum()
        print(f"\nNDX vs equal-weight S&P, last 3m: {rel:+.1f}pp  (positive = rally is concentrated / narrow)")

    # ---- 5b. Cross-section: market caps & sectors -------------------------
    cs_close, bt, rb, cm, rp, sp = pd.DataFrame(), None, None, None, None, None
    if not args.no_sectors:
        log.info("Fetching cap-bucket and sector ETFs ...")
        cs_close = fetch_cross_section(args.years, args.source)
        cs_close = cs_close.reindex(close.index).ffill(limit=1) if not cs_close.empty else cs_close
    if not cs_close.empty and cs_close.shape[1] >= 4:
        rets = cross_returns(cs_close)
        bt = yield_beta_table(rets, panel)
        pd.set_option("display.max_rows", 60)
        print("\n=== YIELD BETA BY MARKET CAP / STYLE (% return per +10bp in 10y; sorted most-hurt first) ===")
        print(bt.loc[[a for a in bt.index if a in CAP_TICKERS]].to_string())
        print("\n=== YIELD BETA BY SECTOR ===")
        print(bt.loc[[a for a in bt.index if a in SECTOR_TICKERS]].to_string())
        print("   excess_beta_6m = beta of (asset − S&P500): relative rate-sensitivity vs the market")

        rb = rolling_yield_beta(rets, panel["d_UST10Y"])
        cm = driver_corr_matrix(rets, panel)
        rp = regime_perf_by_asset(rets, panel["d_UST10Y"])
        print("\n=== BIG YIELD-UP vs BIG YIELD-DOWN DAYS, last 6m (avg daily %; asymmetry>0 = convex) ===")
        print(rp.to_string())
        sp = size_spread(cs_close, panel)
        if sp is not None:
            d = sp[["d_UST10Y", "Small − Mega (daily)"]].dropna().iloc[-126:]
            print(f"\nSmall−Mega spread vs Δ10y, last 6m: corr {d.corr().iloc[0,1]:+.2f}; "
                  f"cum Small−Mega 6m {sp['Small − Mega (cum %)'].iloc[-1]-sp['Small − Mega (cum %)'].iloc[-127]:+.1f}pp")
        cs_close.to_csv(args.out / "cross_section_levels.csv"); bt.to_csv(args.out / "yield_beta_table.csv")
    elif not args.no_sectors:
        log.warning("cross-section skipped – too few ETFs retrieved")

    # ---- 6. Plots ---------------------------------------------------------
    plot_levels(close, args.out)
    plot_rolling_corr(panel, args.out)
    plot_scatter(panel, args.out)
    plot_regime_bars(reg_full, reg_recent, args.out)
    plot_good_bad(gb, comp, args.out, breadth_label)
    plot_real_breakeven(panel, args.out)
    plot_breadth_momentum(close, mom, args.out)

    if bt is not None:
        plot_yield_beta_bars(bt, args.out)
        plot_rolling_beta_caps(rb, panel, args.out)
        plot_driver_heatmap(cm, args.out)
        plot_regime_by_asset(rp, args.out)
        if sp is not None:
            plot_size_spread(sp, args.out)

    if not args.no_intraday:
        try:
            intra = fetch_yahoo_intraday(["QQQ", "^TNX"])
            plot_intraday(intra, args.out)
        except Exception as exc:  # noqa: BLE001
            log.warning("intraday panel skipped: %s", exc)

    # ---- 7. Narrative summary --------------------------------------------
    c_full, c_rec = corr_tbl.loc["d_UST10Y", "full"], corr_tbl.loc["d_UST10Y", "last_126d"]
    print("\n=== READ-ACROSS ===")
    print(f"* Day-to-day link:  corr(NDX, Δ10y) = {c_full:+.2f} full sample, {c_rec:+.2f} last 6m. "
          f"{'Still negative → tech still sells on yield spikes; the index makes highs in the pauses.' if c_rec < -0.1 else 'Weak/positive → yields are being read as growth, not as a discount-rate shock.'}")
    if len(gb) == 2:
        g, b = gb.iloc[0], gb.iloc[1]
        print(f"* Why yields rise matters:  on growth-driven yield-up days NDX avg {gb['r_NDX'].max():+.2f}%, "
              f"on inflation/term-premium days {gb['r_NDX'].min():+.2f}%.")
    print(f"* Momentum:  NDX {mom['vs_50dma_%']:+.1f}% vs 50dma, RSI {mom['RSI14']:.0f}, 12-1m mom {mom['12-1m_mom_%']:+.1f}%, "
          f"drawdown from high {mom['drawdown_from_high_%']:+.1f}%.")
    if comp is not None and len(comp) == 2:
        print(f"* Composition test:  NDX avg {comp['r_NDX'].max():+.2f}% on {comp['r_NDX'].idxmax().split(' (')[0]} days vs "
              f"{comp['r_NDX'].min():+.2f}% on {comp['r_NDX'].idxmin().split(' (')[0]} days.")

    if bt is not None:
        caps = bt.loc[[a for a in bt.index if a in CAP_TICKERS], "beta_6m_%/10bp"]
        secs = bt.loc[[a for a in bt.index if a in SECTOR_TICKERS], "beta_6m_%/10bp"]
        if "Mega (Top50)" in caps and "Small (R2000)" in caps:
            print(f"* Size:  per +10bp in 10y, mega-caps {caps['Mega (Top50)']:+.2f}% vs small-caps {caps['Small (R2000)']:+.2f}% "
                  f"(last 6m) → {'small caps MORE rate-sensitive – classic duration/leverage story' if caps['Small (R2000)'] < caps['Mega (Top50)'] else 'mega-caps MORE rate-sensitive – the discount-rate hit is landing on long-duration growth, not on leverage'}.")
        print(f"* Sectors:  most hurt by rising yields: {secs.idxmin()} ({secs.min():+.2f}%/10bp); "
              f"least hurt / helped: {secs.idxmax()} ({secs.max():+.2f}%/10bp).")
        if "Growth (R1000G)" in caps and "Value (R1000V)" in caps:
            print(f"* Style:  growth {caps['Growth (R1000G)']:+.2f}% vs value {caps['Value (R1000V)']:+.2f}% per +10bp.")

    print("\n=== DATA PROVENANCE ===")
    prov = pd.Series(SOURCES).reindex([c for c in TICKERS if c in close.columns])
    print(prov.to_string())
    dropped = [k for k in TICKERS if k not in close.columns]
    if dropped:
        print(f"unavailable from every source: {dropped}")
    print(f"\nCharts and CSVs written to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
