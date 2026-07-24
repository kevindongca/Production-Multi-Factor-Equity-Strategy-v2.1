# factor-tilt

Four-factor equity backtester with minimum-variance weighting, regime filtering, and turnover-based transaction costs.

A cross-sectional factor strategy over a fixed US large-cap universe, evaluated on rolling out-of-sample windows. Built to be correct rather than impressive — see [Limitations](#limitations), which is the part worth reading.

## What it does

Each rebalance date, every stock in the universe is ranked on four price-based signals. The ranks are averaged into a composite score, the top *N* names are selected, and capital is allocated across them by minimum-variance weights estimated from trailing returns.

**Factors** (all cross-sectional percentile ranks, equal-weighted):

| Factor | Definition |
|---|---|
| Momentum | 12-month return, skipping the most recent month |
| Short-term reversal | Negated 1-month return |
| Low volatility | Negated 3-month realized volatility |
| 52-week high proximity | Price relative to trailing 252-day max |

**Portfolio construction.** Weights solve the unconstrained minimum-variance problem in closed form, `w = Σ⁻¹1 / 1'Σ⁻¹1`, using the sample covariance of trailing daily returns. Negative weights are clipped and renormalized, with equal weight as a fallback when the covariance is singular or clipping empties the portfolio.

**Regime filter.** When SPY closes below its 200-day moving average on a rebalance date, the portfolio goes to cash for that period.

**Transaction costs.** Charged on realized one-way turnover, `Σ|w_new − w_old|` over the union of old and new holdings, at 10 bps. A complete switch of every position costs both sides; unchanged positions cost nothing.

**Evaluation.** Non-overlapping 6-month test windows sliding forward from the end of a 2-year warmup period (needed for the 252-day signal lookbacks). Reports CAGR, Sharpe, Sortino, Calmar, max drawdown, and 95% VaR against a SPY benchmark sliced to the identical window. A PCA decomposition of the return covariance is included as a risk-concentration diagnostic.

## Usage

```bash
pip install pandas numpy yfinance matplotlib scikit-learn
python strategy.py
```

Configuration constants sit at the top of `strategy.py`: universe, date range, `TOP_N`, rebalance frequency, cost assumption, and toggles for the regime filter and minimum-variance weighting.

## Limitations

These are real and I'd rather state them than have someone find them.

**Universe selection bias.** The 26 tickers were chosen by hand, today, with full knowledge of which companies survived and did well over the sample period. This is the largest unquantified bias in the project and it cannot be fixed by changing the code — it requires a point-in-time index membership file. Treat the absolute return numbers as uninformative; the relative behavior across regimes is the part with any signal in it.

**No point-in-time fundamentals.** An earlier version included value and quality factors sourced from a current-day API snapshot and tiled backward across all history, which meant 2021 portfolios were selected using 2026 financials. Those factors were removed rather than left in with a caveat. Adding them back requires a vendor with as-reported historical fundamentals.

**This is not walk-forward validation.** No parameters are fit. Signal definitions, lookback windows, and factor weights are fixed ex ante, so the sliding window measures stability across market regimes, not out-of-sample parameter selection. The code and docstrings say "rolling out-of-sample" for this reason.

**Cost model is a simplification.** Flat 10 bps per unit of turnover ignores spread, market impact, and the fact that impact scales with position size relative to volume. At a 3-name portfolio this is optimistic.

**Small-N covariance estimation.** With `TOP_N = 3`, the sample covariance is estimated from very few assets and minimum-variance weights are unstable. Shrinkage (Ledoit-Wolf) would be the natural next step.

**Survivorship in the price data.** Adjusted close series from Yahoo Finance reflect current corporate actions and do not include delisted names.

## Possible extensions

- Point-in-time universe from historical index membership
- Ledoit-Wolf shrinkage on the covariance estimate
- Factor orthogonalization — momentum and 52-week-high proximity are highly correlated by construction
- Sector-neutral ranking
- Sensitivity analysis across `TOP_N`, rebalance frequency, and cost assumptions

## License

MIT
