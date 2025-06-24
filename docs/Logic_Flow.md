
# Integrated Trading System Logic Flow

When a user runs the Integrated Trading System with example stocks like APX.AX and ABM.AX, the system follows a comprehensive workflow that combines data retrieval, technical analysis, machine learning, portfolio management, and trade execution. Here's a detailed walkthrough of the entire process:

## 1. User Interface Interaction

1. The user navigates to the `/integrated_system` route in the web application.
2. They enter the ticker symbols (APX.AX and ABM.AX) in the input form.
3. They select options like whether to use ML scoring and whether to execute trades.
4. Upon submission, the form data is sent to the `/run_integrated_system` endpoint.

## 2. System Initialization

1. The application creates a unique session ID based on the current timestamp.
2. It initializes a new `TradingSystem` instance with parameters:
   - `initial_capital=100000.0` (starting with $100,000)
   - `market_neutral=True` (aiming for a beta-neutral portfolio)

## 3. Complete Workflow Execution

The system calls the `run_complete_workflow()` method with the provided tickers (APX.AX and ABM.AX), which executes the following steps:

### Step 1: ML Model Training (if using ML scoring)

1. If ML scoring is enabled and the model isn't already trained, the system calls `train_ml_model()`.
2. This collects historical data for the stocks using `collect_training_data()`.
3. The ML model (GradientBoostingRegressor) is trained to predict future price movements.
4. The trained model is saved to disk for future use.

### Step 2: Stock Analysis

1. For each ticker (APX.AX and ABM.AX), the system:
   - Retrieves stock information using `get_stock_info()` from Yahoo Finance.
   - Fetches historical price data using `get_stock_history()`.
   - Calculates technical indicators using `get_stock_data()`, including:
     - Moving averages (MA5, MA10, MA20, MA50, MA200)
     - Volatility measures (ATR, Bollinger Bands)
     - Momentum indicators (RSI, MACD)
     - Volume analysis
     - Support and resistance levels
     - Pattern recognition (gaps, breakouts)
   - If ML scoring is enabled, applies `score_stock_ml()` to get a predicted return percentage.
   - Converts the predicted return to a 0-100 score.
   - Determines a trading strategy based on the score (Strong Buy, Buy, Neutral, Sell, Strong Sell).
   - Updates market data for execution and portfolio management.

2. The stocks are then ranked by their day trading score, with higher scores indicating more favorable trading opportunities.

### Step 3: Watchlist Generation

1. The system generates a watchlist from the ranked stocks using `generate_watchlist()`.
2. Stocks with scores above the minimum threshold (default 60) are included.
3. The watchlist is limited to a maximum number of stocks (default 20).

### Step 4: Portfolio Update

1. The system updates the current portfolio values using `update_portfolio()`.
2. This retrieves current market prices for all holdings.
3. It calculates the current portfolio value, including cash and invested positions.
4. It updates sector exposures and portfolio beta.

### Step 5: Trade Recommendation Generation

1. The system generates trade recommendations using `generate_trade_recommendations()`.
2. The portfolio manager analyzes the ranked stocks and current portfolio to determine:
   - Buy recommendations for new positions or to increase existing positions.
   - Sell recommendations for underperforming positions.
   - Hold recommendations for adequately sized positions with good scores.
   - Rebalance recommendations to maintain sector and beta targets.

3. For buy recommendations:
   - Stocks with high scores (>60) are considered.
   - Position sizing is calculated based on score, volatility, and sector exposure.
   - Available capital is allocated across candidates.

4. For sell recommendations:
   - Positions with low scores (<40) are flagged for selling.
   - The system may also recommend partial selling to reduce overexposed positions.

### Step 6: Trade Execution (if enabled)

1. If trade execution is enabled, the system calls `execute_trades()` with the recommendations.
2. For each buy recommendation:
   - Creates a market order using `create_order()`.
   - Submits the order through `submit_order()`.
   - If the order is filled, updates the portfolio by adding the new position.

3. For each sell recommendation:
   - Creates a market order to sell.
   - Submits the order.
   - If the order is filled, updates the portfolio by reducing or removing the position.

4. After trades are executed, the portfolio is updated again to reflect the changes.

### Step 7: System State Saving

1. The system saves its current state using `save_system_state()`.
2. This includes portfolio positions, cash balance, trade history, and other system parameters.
3. The state is saved to a JSON file for future reference and to enable system continuity.

### Step 8: Backtest on Watchlist

1. The system runs a backtest on the top watchlist stocks using `run_backtest()`.
2. This simulates how the selected strategy would have performed over the past 180 days.
3. The backtest calculates metrics like total return, Sharpe ratio, and maximum drawdown.

### Step 9: Individual Stock Return Calculation

1. For each stock in the watchlist, the system runs an individual backtest.
2. This calculates the historical return percentage for each stock.
3. The return percentages are stored with the stock data for reference.

## 4. Results Presentation

1. The system creates charts and visualizations using `create_trading_system_charts()`.
2. The results are rendered in the 'integrated_results.html' template, showing:
   - Portfolio summary (total value, cash, invested value)
   - Watchlist of top-ranked stocks
   - Trade recommendations
   - Executed orders (if any)
   - Performance metrics and charts
   - Backtest results

## Example for APX.AX and ABM.AX

For the specific example of APX.AX (Appen Limited) and ABM.AX (ABM Resources):

1. **Data Retrieval**: The system fetches current and historical price data for both stocks from Yahoo Finance.

2. **Technical Analysis**: 
   - Calculates indicators like RSI, MACD, Bollinger Bands for both stocks
   - Identifies patterns like breakouts, support/resistance levels
   - Analyzes volume trends and price momentum

3. **ML Scoring** (if enabled):
   - Predicts potential returns for each stock based on historical patterns
   - Converts predictions to actionable scores (0-100)
   - Determines if each stock is a buy, sell, or hold candidate

4. **Portfolio Management**:
   - If either stock is already in the portfolio, evaluates whether to increase, decrease, or maintain the position
   - If not in portfolio, evaluates whether to add them based on scores and available capital
   - Ensures proper diversification and risk management

5. **Trade Execution** (if enabled):
   - Places market orders for the recommended trades
   - Updates the portfolio with the new positions or adjustments

The entire process provides a comprehensive analysis of APX.AX and ABM.AX, generating actionable trading recommendations based on both technical analysis and machine learning predictions, while managing portfolio risk and tracking performance over time.