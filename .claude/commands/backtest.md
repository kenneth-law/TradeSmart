# /backtest

Run a backtest for a given ticker and date range using `modules/backtesting.py`.

## Usage

```
/backtest AAPL 2024-01-01 2024-12-31
```

Arguments: `<TICKER> <START_DATE> <END_DATE>`

## What this does

1. Fetches historical OHLCV data via yfinance
2. Runs the backtesting engine in `modules/backtesting.py`
3. Reports total return, Sharpe ratio, max drawdown, and trade log

## Steps to run

Parse the arguments from `$ARGUMENTS` (format: `TICKER START END`), then:

```bash
python -c "
from modules.data_retrieval import patch_yfdata_cookie_basic
patch_yfdata_cookie_basic()
from modules.backtesting import run_backtest
args = '$ARGUMENTS'.split()
ticker, start, end = args[0], args[1], args[2]
results = run_backtest(ticker, start, end)
print(results)
"
```

If the backtest module API differs, read `modules/backtesting.py` first to confirm function signatures before running.
