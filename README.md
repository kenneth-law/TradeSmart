# TradeSmart Analytics

TradeSmart Analytics is an advanced stock analysis tool designed to identify day trading opportunities through comprehensive technical analysis, volatility measures, news sentiment analysis, and machine learning. The platform uses Gradient Boosted Decision Trees (GBDT) to find non-linear relationships in market data, provides industrial-grade backtesting capabilities, and includes portfolio management tools to optimize trading performance. The application provides a user-friendly web interface for analysing stocks, particularly focused on the Australian Securities Exchange (ASX).



[System Architecture](https://tradevision.up.railway.app/documentation/system_architecture)
<img width="1113" height="863" alt="image" src="https://github.com/user-attachments/assets/e712bc85-26ab-440d-8da8-5ccf1cd5113e" />


## DISCLAIMER:
This software is provided for educational and informational purposes only.
The author(s) are not registered investment advisors and do not provide financial advice.
This software does not guarantee accuracy of data and should not be the sole basis for any investment decision.
Users run and use this software at their own risk. The author(s) accept no liability for any loss or damage 
resulting from its use. Always consult with a qualified financial professional before making investment decisions.

## Terms of Use

By using this software, you agree:
1. To use it at your own risk
2. Not to hold the author(s) liable for any damages
3. To comply with the terms of the [LICENSE]([LICENSE](https://www.apache.org/licenses/LICENSE-2.0.txt))
4. That this is not financial advice

This project is not affiliated with the ASX or any financial institution.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE]([LICENSE](https://www.apache.org/licenses/LICENSE-2.0.txt)) file for details.

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
The ML-based scoring system uses Gradient Boosted Decision Trees (GBDT) to:
- Discover non-linear relationships in market data
- Automatically adjust to changing market conditions
- Prevent overfitting through model complexity reduction and proper validation
- Provide more stable signals with lower turnover
- Generate explainable predictions with feature importance

#### GBDT Implementation Details
- **Model Types**: Supports both regression (score prediction) and classification (buy/sell signals)
- **Feature Engineering**: Extracts 30+ technical indicators as features
- **Preprocessing Pipeline**:
  - Robust scaling to handle outliers
  - Feature selection using mutual information
  - Dimensionality reduction with PCA to reduce multicollinearity
  - Proper scikit-learn Pipeline to ensure preprocessing is only fit on training data
- **Training Process**:
  - Proper time-series cross-validation with expanding windows and gaps
  - Reduced model complexity to prevent overfitting:
    - Fewer estimators (300 vs 500)
    - Reduced tree depth (3 vs 5)
    - Increased min_samples for splits and leaves
    - Reduced subsample ratio (0.7 vs 0.8)
  - L1 regularization and Huber loss for robustness
- **Per-Ticker Models**: Supports individual models for each ticker for more accurate predictions
- **Automatic Retraining**: Models can be scheduled for periodic retraining to adapt to changing market conditions
- **Data Leakage Prevention**: Strict time separation between training and testing periods with verification

### Strategy Classification
Stocks are classified into strategies (Strong Buy, Buy, Neutral/Watch, Sell, Strong Sell) based on their overall score, with strategy details that consider both short-term and long-term performance metrics.

## Advanced Trading Features

### Backtesting Framework
- **Point-in-Time Integrity**: Includes delisted tickers to avoid survivorship bias
- **Transaction Cost Modeling**: Models spread, market impact, and fees based on liquidity
- **Walk-Forward Testing**: Tests strategy robustness across different time periods
- **Monte Carlo Simulation**: Assesses strategy performance under various market conditions

<img width="1113" height="846" alt="image" src="https://github.com/user-attachments/assets/02f6458e-25ad-48d5-ac7f-e946528e05ff" />


#### Backtesting System Architecture
- **Backtester Class**: Core engine that simulates trading over historical data
  - Day-by-day simulation with realistic order execution
  - Support for custom strategy functions
  - Integration with ML scoring for model-driven strategies
- **Transaction Cost Model**: Realistic modeling of trading costs
  - Spread estimation based on price, volume, and volatility
  - Market impact calculation for larger orders
  - Fee structure modeling for different exchanges
- **BacktestResult Class**: Comprehensive performance analysis
  - Equity curve tracking and drawdown analysis
  - Trade-by-trade record keeping
  - Performance metrics calculation (Sharpe, Sortino, Win Rate, etc.)
  - Detailed reporting with visualizations
- **Strategy Functions**:
  - ML-based strategy using GBDT predictions with risk management
  - Technical indicator strategies for benchmarking
  - Custom strategy support through function interface



### Portfolio Management
- **Position Sizing**: Calculates optimal position sizes based on expected edge and volatility
- **Risk Controls**: Implements sector and single-name concentration limits
- **Market Neutrality**: Maintains market-neutral or beta-targeted exposures
- **Drawdown Protection**: Activates kill switches when drawdowns exceed thresholds

<img width="1113" height="935" alt="image" src="https://github.com/user-attachments/assets/4bed9640-ea3f-4d99-ad20-cdc02931b1bb" />


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

### Local Development

1. Start the Flask application:
   ```
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:8080
   ```

3. Enter ticker symbols for analysis (comma-separated) or use the "Load ASX 200" button to populate with Australian stocks.

4. Click "Analyze Stocks" and wait for the analysis to complete.

5. Explore the results, including:
   - Strategy distribution
   - Day trading watchlist
   - Technical indicator charts
   - Detailed metrics table

## Deployment

### Deploying to Railway

The application is configured to be deployed to Railway, a platform for deploying web applications.

1. Create a Railway account at [railway.app](https://railway.app/)

2. Install the Railway CLI:
   ```
   npm i -g @railway/cli
   ```

3. Login to Railway:
   ```
   railway login
   ```

4. Initialize a new Railway project:
   ```
   railway init
   ```

5. Deploy the application:
   ```
   railway up
   ```

6. Open the deployed application:
   ```
   railway open
   ```

The application uses the following files for deployment:
- `wsgi.py`: Entry point for the WSGI server
- `Procfile`: Tells Railway how to run the application
- `requirements.txt`: Lists all dependencies, including Gunicorn

You can also deploy directly from GitHub by connecting your repository to Railway.

## Project Structure

- **app.py**: Main Flask application with routes and chart generation
- **wsgi.py**: Entry point for WSGI servers in production
- **Procfile**: Configuration file for Railway deployment
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
- Gunicorn: WSGI HTTP server for production deployment
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
- For ASX stocks, use the format: WBC.AX, CBA.AX, etc.
- Re-run analysis before market open for fresh data
- Check news sentiment for potential catalysts


## Acknowledgements

- Yahoo Finance for providing stock data
- OpenAI for sentiment analysis capabilities
