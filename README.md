# TradeSmart Analytics

TradeSmart Analytics is an advanced stock analysis tool designed to identify day trading opportunities through comprehensive technical analysis, volatility measures, and news sentiment analysis. The application provides a user-friendly web interface for analyzing stocks, particularly focused on the Australian Securities Exchange (ASX).


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
3. To comply with the terms of the [LICENSE](LICENSE)
4. That this is not financial advice

This project is not affiliated with the ASX or any financial institution.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

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


## Acknowledgements

- Yahoo Finance for providing stock data
- OpenAI for sentiment analysis capabilities
