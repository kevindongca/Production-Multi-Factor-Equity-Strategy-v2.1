"""
================================================================================
Production Multi-Factor Equity Strategy v2.1
================================================================================
Author:  Kevin Dong | github.com/kevindongca | UTSC CS + Stats QF

Design:
  - Regime detection (200-day MA bear market filter)
  - 4 price-based factors (momentum, reversal, low vol, 52W high)
  - Minimum variance portfolio weights (closed form, w = S^-1 1 / 1'S^-1 1)
  - Rolling out-of-sample evaluation (6-month test windows, sliding start)
  - Turnover-based transaction costs (0.1% per unit of one-way turnover)
  - Quarterly or monthly rebalancing
  - Full risk metric suite
  - PCA risk decomposition
  - Dashboard

Note on factors: value and quality factors were removed in v2.1. The prior
version sourced trailing P/E and ROE from the current yfinance snapshot and
tiled them backward across all history, which selected 2021 portfolios using
2026 fundamentals. Without a point-in-time fundamentals source, those factors
cannot be used in a backtest.

Note on validation: this is a rolling out-of-sample backtest, not walk-forward
validation. No parameters are fit on a training window; signal definitions are
fixed ex ante. The sliding window measures stability across regimes, not
parameter selection.

Usage:
    python lesson_29_production_strategy.py
================================================================================
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

TICKERS = (
    "AAPL NVDA MSFT GOOGL META TSLA AMD CRM "
    "JPM GS AXP V MA "
    "AMZN WMT COST NKE "
    "JNJ UNH PFE "
    "XOM CVX "
    "BA CAT HON SHOP"
)

START_DATE     = '2020-01-01'
END_DATE       = '2026-01-01'
TOP_N          = 3
COST_PER_TRADE = 0.001      # charged per unit of one-way turnover
WARMUP_YEARS   = 2          # history consumed by signal windows before testing
TEST_MONTHS    = 6
REBALANCE_FREQ = 'QE'       # 'QE' = quarterly, 'ME' = monthly
USE_REGIME     = True       # enable 200-day MA bear market filter
USE_MINVAR     = True       # enable minimum variance weighting
MA_WINDOW      = 200        # regime detection MA window

# signal lookback windows (trading days)
SHORT_WINDOW  = 21    # ~1 month
MEDIUM_WINDOW = 63    # ~1 quarter
LONG_WINDOW   = 252   # ~1 year

# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

def download_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)
    close = data['Close']
    returns = close.pct_change()
    return close, returns

# ══════════════════════════════════════════════════════════════════════════════
# SIGNALS
# ══════════════════════════════════════════════════════════════════════════════

def compute_signals(close, returns):
    momentum = close.shift(SHORT_WINDOW) / close.shift(LONG_WINDOW) - 1
    reversal = -(close / close.shift(SHORT_WINDOW) - 1)
    low_vol  = -(returns.rolling(MEDIUM_WINDOW).std())
    high_52w = close / close.rolling(LONG_WINDOW).max()
    return momentum, reversal, low_vol, high_52w

def rank_signals(momentum, reversal, low_vol, high_52w):
    r1 = momentum.rank(axis=1, pct=True)
    r2 = reversal.rank(axis=1, pct=True)
    r3 = low_vol.rank(axis=1, pct=True)
    r4 = high_52w.rank(axis=1, pct=True)
    return r1, r2, r3, r4

def build_composite(r1, r2, r3, r4):
    """Equal-weighted composite of four price-based factor ranks."""
    return ((r1 + r2 + r3 + r4) / 4).dropna()

# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def min_variance_weights(returns_history, selected_stocks):
    """
    Long-only-ish minimum variance weights, closed form:
        w = S^-1 1 / (1' S^-1 1)

    Note this is the unconstrained solution and may produce negative weights.
    They are clipped and renormalized; if clipping removes everything, falls
    back to equal weight.
    """
    stock_returns = returns_history[selected_stocks].dropna()
    n = len(selected_stocks)
    if len(stock_returns) < 10:
        return np.array([1.0 / n] * n)

    cov = stock_returns.cov().values
    ones = np.ones(n)
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)

    raw = inv_cov @ ones
    denom = ones @ raw
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        return np.array([1.0 / n] * n)

    weights = raw / denom
    weights = np.clip(weights, 0.0, None)
    total = weights.sum()
    if total <= 1e-12:
        return np.array([1.0 / n] * n)
    return weights / total

def is_bull_market(date, spy_ma_flag):
    """Look up precomputed bull-market flag; default True if date missing."""
    try:
        return bool(spy_ma_flag.loc[:date].iloc[-1])
    except (KeyError, IndexError):
        return True

def compute_turnover(prev_weights, new_weights):
    """
    One-way turnover: sum of |w_new - w_old| over the union of names.
    A full switch of all holdings gives 2.0 (sell one side, buy the other).
    """
    names = set(prev_weights) | set(new_weights)
    return sum(abs(new_weights.get(k, 0.0) - prev_weights.get(k, 0.0))
               for k in names)

def run_portfolio(comp, returns, spy_ma_flag, window_start, window_end,
                  cost_per_trade=0.0):
    """
    Run the strategy over [window_start, window_end], rebalancing on the
    configured frequency. Costs are charged on realized turnover.
    """
    comp_win = comp.loc[window_start:window_end]
    if comp_win.empty:
        return None

    rebalance_dates = [d for d in comp_win.resample(REBALANCE_FREQ).first().index
                       if d in comp_win.index]
    if not rebalance_dates:
        rebalance_dates = [comp_win.index[0]]
    if rebalance_dates[0] != comp_win.index[0]:
        rebalance_dates = [comp_win.index[0]] + rebalance_dates

    portfolio_returns = []
    prev_weights = {}

    for i, date in enumerate(rebalance_dates):
        seg_start = date
        seg_end = (rebalance_dates[i + 1] if i + 1 < len(rebalance_dates)
                   else window_end)

        seg_index = returns.loc[seg_start:seg_end].index
        if i + 1 < len(rebalance_dates):
            seg_index = seg_index[:-1]      # avoid double-counting the seam
        if len(seg_index) == 0:
            continue

        stocks = comp_win.loc[date].nlargest(TOP_N).index.tolist()

        # regime filter: go to cash
        if USE_REGIME and spy_ma_flag is not None and not is_bull_market(date, spy_ma_flag):
            new_weights = {}
            pr = pd.Series(0.0, index=seg_index)
        else:
            if USE_MINVAR:
                w = min_variance_weights(returns.loc[:date], stocks)
            else:
                w = np.array([1.0 / len(stocks)] * len(stocks))
            new_weights = dict(zip(stocks, w))
            pr = returns.loc[seg_index, stocks].dot(w)

        turnover = compute_turnover(prev_weights, new_weights)
        if turnover > 0:
            pr = pr.copy()
            pr.iloc[0] -= cost_per_trade * turnover

        portfolio_returns.append(pr)
        prev_weights = new_weights

    if not portfolio_returns:
        return None
    return pd.concat(portfolio_returns)

# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(portfolio):
    portfolio = portfolio.dropna()
    cumulative = (1 + portfolio).cumprod()
    years = (cumulative.index[-1] - cumulative.index[0]).days / 365.25
    growth = cumulative.iloc[-1] / cumulative.iloc[0]
    cagr = growth ** (1 / years) - 1
    sharpe = portfolio.mean() / portfolio.std() * np.sqrt(252)
    drawdown = (cumulative / cumulative.cummax() - 1).min()
    calmar = cagr / abs(drawdown) if drawdown != 0 else np.nan
    downside = portfolio[portfolio < 0].std()
    sortino = portfolio.mean() / downside * np.sqrt(252) if downside > 0 else np.nan
    var_95 = portfolio.quantile(0.05)
    return cumulative, cagr, sharpe, sortino, calmar, drawdown, var_95

def print_metrics(name, cagr, sharpe, sortino, calmar, drawdown, var_95):
    print(f"\n{'─'*45}")
    print(f"  {name}")
    print(f"{'─'*45}")
    print(f"  CAGR:         {cagr:.2%}")
    print(f"  Sharpe:       {sharpe:.2f}")
    print(f"  Sortino:      {sortino:.2f}")
    print(f"  Calmar:       {calmar:.2f}")
    print(f"  Max Drawdown: {drawdown:.2%}")
    print(f"  VaR (95%):    {var_95:.2%}")

# ══════════════════════════════════════════════════════════════════════════════
# PCA
# ══════════════════════════════════════════════════════════════════════════════

def run_pca(returns, n=5):
    clean = returns.dropna()
    scaled = StandardScaler().fit_transform(clean)
    pca = PCA(n_components=n).fit(scaled)
    explained = pca.explained_variance_ratio_
    loadings = pd.DataFrame(pca.components_.T,
                            index=clean.columns,
                            columns=[f'PC{i+1}' for i in range(n)])
    print(f"\n  PCA: PC1={explained[0]:.1%}, PC2={explained[1]:.1%}, "
          f"PC3={explained[2]:.1%} (top 3 explain {sum(explained[:3]):.1%})")
    return explained, loadings

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def plot_dashboard(portfolio, cumulative, spy_cumulative, explained, loadings):
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Production Multi-Factor Strategy v2.1\nKevin Dong — UTSC',
                 fontsize=15, fontweight='bold')

    ax1 = fig.add_subplot(3, 2, (1, 2))
    ax1.plot(cumulative / cumulative.iloc[0], label='Strategy',
             color='royalblue', linewidth=2)
    spy_aligned = spy_cumulative.loc[cumulative.index[0]:cumulative.index[-1]]
    if len(spy_aligned) > 0:
        ax1.plot(spy_aligned / spy_aligned.iloc[0], label='SPY',
                 color='darkorange', linewidth=1.5)
    ax1.set_title('Cumulative Returns (Rolling Out-of-Sample)')
    ax1.set_ylabel('Growth of $1')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(3, 2, 3)
    dd = cumulative / cumulative.cummax() - 1
    spy_dd = spy_cumulative / spy_cumulative.cummax() - 1
    ax2.fill_between(dd.index, dd, 0, color='crimson', alpha=0.4, label='Strategy')
    ax2.fill_between(spy_dd.index, spy_dd, 0, color='orange', alpha=0.2, label='SPY')
    ax2.set_title('Drawdown')
    ax2.set_ylabel('Drawdown %')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(3, 2, 4)
    rs = portfolio.rolling(63).mean() / portfolio.rolling(63).std() * np.sqrt(252)
    ax3.plot(rs, color='seagreen')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.axhline(y=1, color='seagreen', linestyle='--', alpha=0.3)
    ax3.set_title('Rolling Sharpe (63 days)')
    ax3.set_ylabel('Sharpe')
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(3, 2, 5)
    ax4.bar(range(1, len(explained)+1), explained, color='steelblue')
    ax4.plot(range(1, len(explained)+1), np.cumsum(explained), 'ro-')
    ax4.set_title('PCA Variance Explained')
    ax4.set_xlabel('Principal Component')
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(3, 2, 6)
    pc2 = loadings['PC2'].sort_values()
    colors = ['tomato' if v < 0 else 'steelblue' for v in pc2.values]
    ax5.barh(pc2.index, pc2.values, color=colors)
    ax5.set_title('PC2 Loadings')
    ax5.set_xlabel('Loading')
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("="*55)
    print("  Production Multi-Factor Strategy v2.1")
    print("  Kevin Dong — UTSC CS + Stats QF")
    print("="*55)
    print(f"\n  Config: TOP_N={TOP_N}, Regime={USE_REGIME}, MinVar={USE_MINVAR}")
    print(f"  Rebalance: {REBALANCE_FREQ}, Windows: "
          f"{SHORT_WINDOW}/{MEDIUM_WINDOW}/{LONG_WINDOW}")

    print("\nDownloading data...")
    close, returns = download_data(TICKERS, START_DATE, END_DATE)

    momentum, reversal, low_vol, high_52w = compute_signals(close, returns)
    r1, r2, r3, r4 = rank_signals(momentum, reversal, low_vol, high_52w)
    comp = build_composite(r1, r2, r3, r4)

    spy_data = yf.download('SPY', start=START_DATE, end=END_DATE, auto_adjust=True)
    spy_close   = spy_data['Close'].squeeze()
    spy_returns = spy_close.pct_change().dropna()
    spy_cum     = (1 + spy_returns).cumprod()
    spy_ma_flag = (spy_close > spy_close.rolling(MA_WINDOW).mean()).fillna(True)

    explained, loadings = run_pca(returns)

    print("\nRunning rolling out-of-sample backtest...")
    window_start = pd.Timestamp(START_DATE) + pd.DateOffset(years=WARMUP_YEARS)
    all_returns = []

    while True:
        window_end = window_start + pd.DateOffset(months=TEST_MONTHS)
        if window_end > pd.Timestamp(END_DATE):
            break

        port = run_portfolio(comp, returns, spy_ma_flag,
                             window_start, window_end,
                             cost_per_trade=COST_PER_TRADE)
        if port is not None:
            all_returns.append(port)

        window_start = window_end

    if not all_returns:
        raise SystemExit("No out-of-sample windows produced returns.")

    portfolio = pd.concat(all_returns)
    portfolio = portfolio[~portfolio.index.duplicated(keep='first')].sort_index()

    cumulative, *metrics = compute_metrics(portfolio)
    print_metrics("Production Strategy v2.1", *metrics)

    spy_oos = spy_returns.loc[portfolio.index[0]:portfolio.index[-1]]
    _, spy_cagr, spy_sharpe, spy_sortino, spy_calmar, spy_dd, spy_var = \
        compute_metrics(spy_oos)
    print_metrics("SPY Benchmark (same window)", spy_cagr, spy_sharpe,
                  spy_sortino, spy_calmar, spy_dd, spy_var)

    plot_dashboard(portfolio, cumulative, spy_cum, explained, loadings)
