# /analyze

Analyze one or more stock tickers using TradeSmart's technical analysis and ML scoring pipeline.

## Usage

```
/analyze AAPL MSFT TSLA
```

## What this does

1. Calls `modules/analysis_reporting.py:analyze_stocks()` with the given tickers
2. Prints ranked results including `day_trading_score`, strategy, and key indicators
3. Flags any tickers that failed to fetch data

## Steps to run

```bash
python -c "
from modules.data_retrieval import patch_yfdata_cookie_basic
patch_yfdata_cookie_basic()
from modules.analysis_reporting import analyze_stocks, format_results
tickers = '$ARGUMENTS'.split()
ranked, failed = analyze_stocks(tickers)
print(format_results(ranked, failed))
"
```

Replace `$ARGUMENTS` with space-separated ticker symbols.
