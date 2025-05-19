# TradeSmart Analytics

TradeSmart Analytics is an advanced stock analysis tool designed to identify day trading opportunities through comprehensive technical analysis, volatility measures, and news sentiment analysis. The application provides a user-friendly web interface for analyzing stocks, particularly focused on the Australian Securities Exchange (ASX).

## Features

- **Comprehensive Stock Analysis**: Analyzes stocks using multiple technical indicators, volatility measures, and news sentiment
- **Real-time Progress Updates**: Provides real-time feedback during analysis with a terminal-like display
- **Interactive Visualizations**: Generates multiple charts and visualizations to help understand the data
- **Day Trading Scoring System**: Assigns scores to stocks based on their potential as day trading opportunities
- **Watchlist Generation**: Creates a curated watchlist of the most promising stocks
- **Detailed Stock Information**: Provides in-depth analysis of individual stocks
- **News Sentiment Analysis**: Incorporates news sentiment into the analysis
- **Data Export**: Allows downloading analysis results as CSV

## Technical Indicators Used

- **RSI (Relative Strength Index)**: Measures the speed and change of price movements
- **MACD (Moving Average Convergence Divergence)**: Trend-following momentum indicator
- **Moving Averages**: Identifies trend direction
- **ATR (Average True Range)**: Measures market volatility
- **Bollinger Bands**: Indicates overbought or oversold conditions
- **Volume Analysis**: Compares current volume to average volume

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

## Dependencies

- Flask: Web framework
- Pandas & NumPy: Data manipulation
- yfinance: Yahoo Finance API for stock data
- Plotly: Interactive charts and visualisations
- OpenAI: News sentiment analysis
- Requests & BeautifulSoup: Web scraping and HTTP requests

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