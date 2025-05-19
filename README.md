# TradeSmart Analytics

TradeSmart Analytics is an advanced stock analysis tool designed to identify day trading opportunities through comprehensive technical analysis, volatility measures, and news sentiment analysis. The application provides a user-friendly web interface for analyzing stocks, particularly focused on the Australian Securities Exchange (ASX).

## Features

- **Comprehensive Stock Analysis**: Analyzes stocks using multiple technical indicators, volatility measures, and news sentiment
- **Real-time Progress Updates**: Provides real-time feedback during analysis with a terminal-like display
- **Interactive Visualizations**: Generates multiple charts and visualizations to help understand the data
- **Machine Learning-Based Scoring**: Uses gradient boosted trees to discover non-linear relationships in market data
- **Industrial-Grade Backtesting**: Provides realistic backtesting with transaction cost modeling and point-in-time integrity
- **Portfolio Construction**: Implements risk management and position sizing based on expected edge and volatility
- **Low-Slippage Execution**: Integrates with execution algorithms (TWAP, VWAP) to minimize market impact
- **Watchlist Generation**: Creates a curated watchlist of the most promising stocks
- **Detailed Stock Information**: Provides in-depth analysis of individual stocks
- **News Sentiment Analysis**: Incorporates news sentiment into the analysis
- **Data Export**: Allows downloading analysis results as CSV

## Technical Indicators Used

### Short-Term Indicators
- **RSI (Relative Strength Index)**: Measures the speed and change of price movements
- **MACD (Moving Average Convergence Divergence)**: Trend-following momentum indicator
- **Short-Term Moving Averages (5, 10, 20-day)**: Identifies short-term trend direction
- **ATR (Average True Range)**: Measures market volatility
- **Bollinger Bands**: Indicates overbought or oversold conditions
- **Volume Analysis**: Compares current volume to average volume
- **Stochastic Oscillator**: Compares a security's closing price to its price range over a specific period
- **On-Balance Volume (OBV)**: Relates volume to price change to predict trend strength

### Long-Term Performance Metrics
- **Long-Term Moving Averages (50, 200-day)**: Identifies medium and long-term trend direction
- **Golden Cross/Death Cross**: Signals major trend changes (50-day MA crossing 200-day MA)
- **Extended Historical Data (180 days)**: Provides more context for analysis
- **Medium and Long-Term Returns (30-day, 90-day)**: Measures performance over longer periods

## Scoring System

The application offers two scoring systems to evaluate stocks:

### Traditional Heuristic Scoring
- **Technical Score (35%)**: Evaluates short-term technical indicators including RSI, MACD, Bollinger Bands, Stochastic Oscillator, and recent price movements
- **Long-Term Score (15%)**: Assesses long-term performance using 50/200-day moving averages, golden/death crosses, and 30/90-day returns
- **Volatility Score (20%)**: Measures price volatility using ATR and intraday range
- **News Sentiment Score (10%)**: Analyzes recent news sentiment using natural language processing
- **Gap Potential Score (10%)**: Evaluates potential for gap trading based on pre-market movement and gap frequency
- **Volume Score (10%)**: Assesses trading volume relative to average

### Machine Learning-Based Scoring
The new ML-based scoring system uses gradient boosted trees to:
- Discover non-linear relationships in market data
- Automatically adjust to changing market conditions
- Reduce overfitting through feature orthogonalization
- Provide more stable signals with lower turnover
- Generate explainable predictions with feature importance

### Strategy Classification
Stocks are classified into strategies (Strong Buy, Buy, Neutral/Watch, Sell, Strong Sell) based on their overall score, with strategy details that consider both short-term and long-term performance metrics.

## Advanced Trading Features

### Backtesting Framework
- **Point-in-Time Integrity**: Includes delisted tickers to avoid survivorship bias
- **Transaction Cost Modeling**: Models spread, market impact, and fees based on liquidity
- **Walk-Forward Testing**: Tests strategy robustness across different time periods
- **Monte Carlo Simulation**: Assesses strategy performance under various market conditions

### Portfolio Management
- **Position Sizing**: Calculates optimal position sizes based on expected edge and volatility
- **Risk Controls**: Implements sector and single-name concentration limits
- **Market Neutrality**: Maintains market-neutral or beta-targeted exposures
- **Drawdown Protection**: Activates kill switches when drawdowns exceed thresholds

### Execution Algorithms
- **TWAP (Time-Weighted Average Price)**: Divides orders into equal-sized slices over time
- **VWAP (Volume-Weighted Average Price)**: Executes orders based on expected volume profile
- **Smart Order Routing**: Minimizes market impact through intelligent order placement
- **Dynamic Participation**: Adjusts participation rates based on market liquidity

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up your OpenAI API key (for sentiment analysis):
   ```
   set OPENAI_API_KEY=your-api-key
   ```

## Usage

1. Start the Flask application:
   ```
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Enter ticker symbols for analysis (comma-separated) or use the "Load ASX 200" button to populate with Australian stocks.

4. Click "Analyze Stocks" and wait for the analysis to complete.

5. Explore the results, including:
   - Strategy distribution
   - Day trading watchlist
   - Technical indicator charts
   - Detailed metrics table

## Project Structure

- **app.py**: Main Flask application with routes and chart generation
- **stock_analysis.py**: Core stock analysis functionality
- **yfinance_cookie_patch.py**: Patch for the yfinance library to handle cookies properly
- **templates/**: HTML templates for the web interface
- **static/**: CSS, JavaScript, and generated charts
- **cache/**: Cached stock data for improved performance
- **modules/**: Modular components for advanced functionality
  - **ml_scoring.py**: Machine learning-based scoring system
  - **backtesting.py**: Industrial-grade backtesting framework
  - **portfolio_management.py**: Risk management and portfolio construction
  - **execution.py**: Low-slippage execution algorithms
  - **technical_analysis.py**: Technical indicators and analysis
  - **data_retrieval.py**: Data retrieval and caching
  - **utils.py**: Utility functions

## Dependencies

- Flask: Web framework
- Pandas & NumPy: Data manipulation
- yfinance: Yahoo Finance API for stock data
- Plotly: Interactive charts and visualisations
- OpenAI: News sentiment analysis
- Requests & BeautifulSoup: Web scraping and HTTP requests
- scikit-learn: Machine learning algorithms and preprocessing
- matplotlib: Data visualization for backtesting results
- joblib: Model serialization and persistence

## Notes

- The application is optimised for analysing 5-200 stocks at a time
- For ASX stocks, use the format: APX.AX, WBC.AX, etc.
- Re-run analysis before market open for fresh data
- Check news sentiment for potential catalysts

## License

[Specify license information here]

## Acknowledgements

- Yahoo Finance for providing stock data
- OpenAI for sentiment analysis capabilities
