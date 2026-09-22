"""
Absorption Ratio Monitor v3  --  US equity indices (S&P 500, Nasdaq-100, SOX)
=============================================================================
NEW in v3:
  * Top panel: INDEX PRICE + DRAWDOWN, sharing one time axis with AR & dAR,
    so you can visually test whether AR/dAR spikes PRECEDE index drawdowns.
  * Markers where dAR crosses >= +1 sigma (paper's fragility signal).
  * Shaded drawdown episodes (index >X% below prior peak).
  * Prints a simple lead-lag table: for each drawdown episode, how many days
    earlier the nearest dAR>=+1 signal fired.

Methodology (AR & dAR): Kritzman, Li, Page & Rigobon (2010/2011),
   "Principal Components as a Measure of Systemic Risk."
   SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1633027
   MIT:  https://web.mit.edu/finlunch/Fall10/PCASystemicRisk.pdf
   Paper evidence: median dAR rose ~40 days (~6 wks) before the 10% most
   turbulent periods; ALL of the 1% worst monthly drawdowns were preceded
   by a >= +1 sigma AR spike.

Data: Yahoo Finance via yfinance (free, unofficial, end-of-day; delayed).
Install: pip install yfinance pandas numpy dash plotly
Run:     python absorption_ratio.py  ->  http://127.0.0.1:8050
"""

import numpy as np
import pandas as pd
import yfinance as yf
from dash import Dash, dcc, html, Output, Input, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG (paper-faithful)
# ─────────────────────────────────────────────────────────────────────────────
WINDOWS = {
    "500-day (paper)":  500,
    "750-day (3y)":     750,
    "1250-day (5y)":   1250,
}
EIGEN_FRAC   = 0.20   # n eigenvectors ≈ N/5
SHORT_MA     = 15     # AR 15-day average
LONG_MA      = 252    # AR 1-year average / std
HIST_PERIOD  = "10y"
DD_THRESHOLD = 0.10   # drawdown episode = >=10% below prior peak (tunable)

# Each universe: (constituent tickers, index proxy ticker for the price panel)
UNIVERSES = {
    "S&P 500 (50 largest members)": {
        "index": "^GSPC",
        "members": [
            "AAPL","MSFT","NVDA","AMZN","GOOGL","META","BRK-B","LLY","AVGO","JPM",
            "TSLA","XOM","UNH","V","PG","MA","JNJ","HD","COST","MRK","ABBV","CVX",
            "CRM","WMT","PEP","KO","BAC","NFLX","AMD","ADBE","TMO","LIN","MCD","CSCO",
            "ACN","ABT","WFC","DHR","GE","TXN","DIS","CAT","QCOM","INTU","VZ","IBM",
            "AMGN","PM","CMCSA","NOW",
        ],
    },
    "Nasdaq-100 (40 largest members)": {
        "index": "^NDX",
        "members": [
            "AAPL","MSFT","NVDA","AMZN","AVGO","META","GOOGL","GOOG","TSLA","COST",
            "NFLX","AMD","PEP","ADBE","CSCO","TMUS","INTU","QCOM","TXN","AMGN",
            "ISRG","CMCSA","HON","AMAT","BKNG","VRTX","ADP","GILD","ADI","LRCX",
            "MU","PANW","REGN","KLAC","SBUX","MDLZ","SNPS","CDNS","MELI","CRWD",
        ],
    },
    "PHLX Semiconductor (SOX, 30 members)": {
        "index": "^SOX",
        "members": [
            "NVDA","AVGO","AMD","QCOM","TXN","INTC","AMAT","MU","ADI","LRCX",
            "KLAC","MCHP","MRVL","NXPI","ON","MPWR","SWKS","TER","QRVO","ASML",
            "TSM","ARM","STM","GFS","ENTG","COHR","LSCC","AMKR","UMC","WOLF",
        ],
    },
}

WINDOW_COLOURS = {
    "500-day (paper)": "#1f77b4",
    "750-day (3y)":    "#ff7f0e",
    "1250-day (5y)":   "#2ca02c",
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA LAYER
# ─────────────────────────────────────────────────────────────────────────────
_cache: dict = {}


def fetch_prices(tickers: list[str]) -> pd.DataFrame:
    key = tuple(sorted(tickers))
    if key in _cache:
        return _cache[key]
    px = yf.download(tickers, period=HIST_PERIOD,
                     auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):          # single ticker → make DataFrame
        px = px.to_frame()
    px = px.dropna(axis=1, how="all")
    _cache[key] = px
    return px


# ─────────────────────────────────────────────────────────────────────────────
# CORE MATHS (unchanged from v2 – faithful to Kritzman et al.)
# ─────────────────────────────────────────────────────────────────────────────
def ewma_cov(returns_window: pd.DataFrame, halflife: int) -> np.ndarray:
    T   = len(returns_window)
    lam = 0.5 ** (1.0 / halflife)          # half-life = window/2 (paper)
    w   = lam ** np.arange(T - 1, -1, -1)
    w  /= w.sum()
    X   = returns_window.values
    mu  = np.average(X, axis=0, weights=w)
    Xc  = X - mu
    return (Xc * w[:, None]).T @ Xc


def absorption_ratio_from_cov(cov: np.ndarray, n_eigen: int) -> float:
    eig   = np.clip(np.linalg.eigvalsh(cov), 0, None)
    eig   = np.sort(eig)[::-1]
    total = eig.sum()
    return float(eig[:n_eigen].sum() / total) if total > 0 else np.nan


def compute_ar_series(returns: pd.DataFrame, window: int) -> pd.Series:
    halflife = window // 2
    idx, vals = [], []
    for end in range(window, len(returns) + 1):
        win = returns.iloc[end - window:end].dropna(axis=1)
        if win.shape[1] < 2:
            continue
        n_e = max(1, int(round(EIGEN_FRAC * win.shape[1])))
        vals.append(absorption_ratio_from_cov(ewma_cov(win, halflife), n_e))
        idx.append(returns.index[end - 1])
    return pd.Series(vals, index=idx, name=f"AR_{window}d")


def standardised_shift(ar: pd.Series) -> pd.Series:
    ar15 = ar.rolling(SHORT_MA, min_periods=SHORT_MA).mean()
    ar1y = ar.rolling(LONG_MA,  min_periods=LONG_MA).mean()
    sd1y = ar.rolling(LONG_MA,  min_periods=LONG_MA).std(ddof=1)
    return ((ar15 - ar1y) / sd1y).rename(f"dAR_{ar.name}")


# ─────────────────────────────────────────────────────────────────────────────
# DRAWDOWN & LEAD-LAG ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
def drawdown_series(price: pd.Series) -> pd.Series:
    """Percentage below running peak (0 at highs, negative in drawdown)."""
    return price / price.cummax() - 1.0


def drawdown_episodes(price: pd.Series, threshold: float = DD_THRESHOLD):
    """
    Identify episodes where drawdown breaches -threshold.
    Returns list of dicts: {start, trough, trough_dd}.
    An episode runs from the day drawdown first breaches -threshold until
    price makes a new all-time high (drawdown returns to 0).
    """
    dd = drawdown_series(price)
    episodes, in_ep, start, trough_val, trough_dt = [], False, None, 0.0, None
    for dt, v in dd.items():
        if not in_ep and v <= -threshold:
            in_ep, start, trough_val, trough_dt = True, dt, v, dt
        elif in_ep:
            if v < trough_val:
                trough_val, trough_dt = v, dt
            if v >= -1e-9:               # recovered to prior peak
                episodes.append({"start": start, "trough": trough_dt,
                                 "trough_dd": trough_val})
                in_ep = False
    if in_ep:                            # still ongoing at series end
        episodes.append({"start": start, "trough": trough_dt,
                         "trough_dd": trough_val})
    return episodes


def signal_dates(dar: pd.Series, level: float = 1.0):
    """Dates where dAR crosses UP through +level (fresh fragility signal)."""
    s = dar.dropna()
    prev = s.shift(1)
    crosses = s[(s >= level) & (prev < level)]
    return list(crosses.index)


def lead_lag_table(price: pd.Series, dar: pd.Series):
    """
    For each drawdown episode, find the most recent dAR>=+1 signal at or
    before the episode start, and report the lead time in calendar days.
    Faithful caveat: this is a descriptive lead-lag check, NOT a backtest.
    """
    eps  = drawdown_episodes(price)
    sigs = signal_dates(dar)
    rows = []
    for ep in eps:
        prior = [d for d in sigs if d <= ep["start"]]
        if prior:
            last_sig = max(prior)
            lead = (ep["start"] - last_sig).days
            sig_str, lead_str = last_sig.date().isoformat(), f"{lead}"
        else:
            sig_str, lead_str = "none before", "-"
        rows.append({
            "Drawdown start":   ep["start"].date().isoformat(),
            "Trough date":      ep["trough"].date().isoformat(),
            "Max drawdown":     f"{ep['trough_dd']*100:.1f}%",
            "Prior dAR≥+1 date": sig_str,
            "Lead (days)":      lead_str,
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# BUILD EVERYTHING FOR A UNIVERSE
# ─────────────────────────────────────────────────────────────────────────────
def build(universe_key: str):
    cfg     = UNIVERSES[universe_key]
    members = cfg["members"]
    idx_tkr = cfg["index"]

    member_px = fetch_prices(members)
    rets      = member_px.pct_change().dropna(how="all")

    # robust single-ticker index fetch
    idx_raw = fetch_prices([idx_tkr])
    idx_px  = idx_raw[idx_tkr] if idx_tkr in idx_raw.columns else idx_raw.iloc[:, 0]
    idx_px  = idx_px.dropna()

    windows = {}
    for label, w in WINDOWS.items():
        ar  = compute_ar_series(rets, w)
        dar = standardised_shift(ar)
        windows[label] = {"ar": ar, "dar": dar,
                          "n_assets": rets.shape[1],
                          "n_eigen": max(1, int(round(EIGEN_FRAC * rets.shape[1])))}

    return idx_px, windows


# ─────────────────────────────────────────────────────────────────────────────
# DASH APP
# ─────────────────────────────────────────────────────────────────────────────
app = Dash(__name__)
app.title = "Absorption Ratio Monitor"

app.layout = html.Div(
    style={"fontFamily": "'Segoe UI', Arial, sans-serif",
           "maxWidth": 1250, "margin": "auto", "padding": "20px 24px"},
    children=[
        html.H2("Absorption Ratio vs Index — does AR precede drawdowns?",
                style={"marginBottom": 4}),
        html.P(
            ["Methodology: Kritzman, Li, Page & Rigobon (2010/2011). ",
             "Paper evidence: median dAR rose ~40 days (~6 weeks) before the "
             "10% most turbulent periods; all of the 1% worst monthly "
             "drawdowns were preceded by a ≥+1σ AR spike. ",
             "Data: Yahoo Finance (EOD, unofficial, delayed)."],
            style={"fontSize": 12, "color": "#666", "marginBottom": 14}),
        dcc.Dropdown(
            id="universe",
            options=[{"label": k, "value": k} for k in UNIVERSES],
            value=list(UNIVERSES)[0], clearable=False,
            style={"marginBottom": 8}),
        html.Div([
            html.Label("AR window to overlay with index:",
                       style={"fontSize": 12, "color": "#555",
                              "marginRight": 8}),
            dcc.RadioItems(
                id="window",
                options=[{"label": k, "value": k} for k in WINDOWS],
                value="500-day (paper)", inline=True,
                style={"fontSize": 12}),
        ], style={"marginBottom": 12}),
        dcc.Interval(id="tick", interval=300_000, n_intervals=0),
        dcc.Graph(id="stack-chart"),
        html.H4("Lead-lag check: drawdown episodes vs prior dAR ≥ +1σ signal",
                style={"marginTop": 18, "marginBottom": 6}),
        html.P("Descriptive only — not a backtest. 'Lead' = calendar days "
               "between the last ≥+1σ signal and the drawdown's start.",
               style={"fontSize": 11, "color": "#888"}),
        dash_table.DataTable(
            id="leadlag",
            style_cell={"fontFamily": "Arial", "fontSize": 12,
                        "padding": "6px", "textAlign": "center"},
            style_header={"fontWeight": "bold", "backgroundColor": "#f2f2f2"}),
        html.P(
            "Source: Kritzman, Li, Page & Rigobon (2011), 'Principal "
            "Components as a Measure of Systemic Risk', J. Portfolio "
            "Management 37(4). SSRN 1633027. | Data: Yahoo Finance (yfinance).",
            style={"fontSize": 11, "color": "#aaa", "marginTop": 16}),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACK  (v3.1 — fixes disappearing index/AR panels)
#
# BUG FIXED: previously the drawdown used a MANUAL layout.yaxis2 with
# overlaying="y". make_subplots ALSO auto-creates yaxis2 (for row 2),
# so the manual definition clobbered the AR panel's axis, blanking panel 2
# and pushing the AR line into panel 1. Fix: build the secondary axis the
# subplot-native way via specs=[{"secondary_y": True}] + secondary_y=.
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("stack-chart", "figure"),
    Output("leadlag",     "data"),
    Output("leadlag",     "columns"),
    Input("universe",     "value"),
    Input("window",       "value"),
    Input("tick",         "n_intervals"),
)
def refresh(universe, window_label, _n):
    idx_px, windows = build(universe)
    w       = windows[window_label]
    ar, dar = w["ar"], w["dar"]
    colour  = WINDOW_COLOURS[window_label]

    # ---- Sanity guard: if index download failed, say so instead of hiding it
    if idx_px is None or idx_px.dropna().empty:
        empty = go.Figure().update_layout(
            height=760,
            annotations=[dict(text="Index price unavailable (data fetch "
                                   "returned no rows). Check the index ticker "
                                   "or network.", showarrow=False,
                              font=dict(size=14, color="#c0392b"))])
        return empty, [], []

    # Align index onto the AR window's date span; forward-fill non-trading gaps
    span   = ar.index
    idx_al = idx_px.reindex(idx_px.index.union(span)).ffill().loc[
                 span.min():span.max()]
    dd     = (idx_al / idx_al.cummax() - 1.0) * 100.0

    # ---- 3 stacked panels; ONLY row 1 has a secondary y-axis --------------
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.42, 0.30, 0.28],
        specs=[[{"secondary_y": True}],   # row 1: price (left) + drawdown (right)
               [{"secondary_y": False}],  # row 2: AR
               [{"secondary_y": False}]], # row 3: dAR
        subplot_titles=(
            f"{universe.split(' (')[0]} — index price & drawdown",
            f"Absorption Ratio — {window_label}",
            "Standardised Shift dAR (±1σ)"))

    # Panel 1: index price on primary (left) y-axis
    fig.add_trace(
        go.Scatter(x=idx_al.index, y=idx_al.values, name="Index",
                   line=dict(color="#222", width=1.4)),
        row=1, col=1, secondary_y=False)

    # Panel 1: drawdown on secondary (right) y-axis
    fig.add_trace(
        go.Scatter(x=dd.index, y=dd.values, name="Drawdown %",
                   fill="tozeroy", line=dict(color="#c0392b", width=0.6),
                   fillcolor="rgba(192,57,43,0.12)"),
        row=1, col=1, secondary_y=True)

    fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Drawdown %", row=1, col=1, secondary_y=True,
                     range=[min(dd.min() * 1.1, -5), 5], showgrid=False)

    # Panel 2: AR + 1y average
    ar1y = ar.rolling(LONG_MA, min_periods=LONG_MA).mean()
    fig.add_trace(go.Scatter(x=ar.index, y=ar.values, name="AR",
                             line=dict(color=colour, width=1.4)),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=ar1y.index, y=ar1y.values, name="AR 1y avg",
                             line=dict(color=colour, width=1, dash="dash")),
                  row=2, col=1)
    fig.update_yaxes(title_text="AR", row=2, col=1)

    # Panel 3: dAR + ±1σ + fragility markers
    fig.add_trace(go.Scatter(x=dar.index, y=dar.values, name="dAR",
                             line=dict(color=colour, width=1.4)),
                  row=3, col=1)
    fig.add_hline(y=1,  line_dash="dot", line_color="red",   row=3, col=1)
    fig.add_hline(y=-1, line_dash="dot", line_color="green", row=3, col=1)
    fig.add_hline(y=0,  line_color="#ddd", line_width=0.8,    row=3, col=1)

    sigs = signal_dates(dar)
    if sigs:
        fig.add_trace(go.Scatter(
            x=sigs, y=[dar.loc[d] for d in sigs], mode="markers",
            name="dAR ≥ +1σ signal",
            marker=dict(color="red", size=7, symbol="triangle-up")),
            row=3, col=1)
    fig.update_yaxes(title_text="σ", row=3, col=1)

    # Vertical guides at each fragility signal (all panels)
    for d in sigs:
        fig.add_vline(x=d, line_color="red", line_width=0.5, opacity=0.22)

    # NOTE: do NOT set layout.yaxis2 manually here — make_subplots owns it.
    fig.update_layout(
        height=780, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.03,
                    xanchor="left", x=0),
        margin=dict(t=90, b=30),
        plot_bgcolor="#fff", paper_bgcolor="#fff")

    rows = lead_lag_table(idx_al, dar)
    cols = [{"name": c, "id": c} for c in rows[0].keys()] if rows else []
    return fig, rows, cols


if __name__ == "__main__":
    app.run(debug=True)   # → http://127.0.0.1:8050
