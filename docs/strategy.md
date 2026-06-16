# Strategy Research

TradeSmart is designed for systematic equity research: define a signal hypothesis, evaluate it against historical data, inspect risk and transaction costs, and only then consider whether it belongs in a portfolio workflow.

## Research Loop

1. **Universe selection**: choose a small ticker set, broad index sample, watchlist, or strategy-specific universe.
2. **Feature generation**: collect price, return, volume, volatility, trend, sentiment, and fundamental context where available.
3. **Signal scoring**: compare heuristic scores with model-assisted GBDT outputs.
4. **Validation**: run historical simulations with realistic costs and benchmark comparison.
5. **Portfolio review**: evaluate sizing, concentration, drawdown behavior, and capacity constraints.
6. **Iteration**: adjust thresholds, exits, re-entry rules, and training windows based on diagnostics.

## Signal Families

| Signal | Purpose |
| --- | --- |
| Momentum and trend | Identify persistent price movement, trend strength, and moving-average structure |
| Volatility and range | Capture risk, sizing pressure, ATR behavior, and drawdown exposure |
| Volume and liquidity | Estimate tradability, relative activity, spread pressure, and implementation cost |
| Fundamental context | Add company-level context through available financial data |
| Sentiment context | Add news or AI-assisted qualitative context where configured |
| ML score | Search for non-linear feature interactions using Gradient Boosted Decision Trees |

## Model Research

The GBDT layer is used for alpha signal exploration, not as a black-box guarantee. The model pipeline supports robust scaling, feature selection, PCA support, regression scoring, trained-artifact metadata, and time-aware validation.

Key research questions:

- Does the signal survive transaction costs?
- Does the edge concentrate in a few names or generalize across the universe?
- Does performance change across market regimes?
- Which features dominate the prediction?
- How sensitive is the strategy to thresholds, exits, and training lookback?

## Output

The strategy workflow produces ranked signals, backtest diagnostics, portfolio context, and model metadata so each idea can be reviewed as a research artifact rather than a one-off trade call.
