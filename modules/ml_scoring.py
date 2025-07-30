"""
Machine Learning Scoring Module

This module implements a data-driven approach to stock scoring using machine learning techniques.
It replaces the heuristic weights in the original scoring system with a gradient boosted tree model
that can discover non-linear relationships and automatically adjust to changing market conditions.

The module includes:
1. Feature extraction from stock data
2. Feature orthogonalization and dimensionality reduction
3. Model training with walk-forward cross-validation
4. Prediction and scoring functions
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression, mutual_info_regression
import joblib
import os
from datetime import datetime, timedelta
import logging
from modules.utils import log_message
from modules.data_retrieval import get_stock_history

# Create cache directory for models if it doesn't exist
os.makedirs('cache/models', exist_ok=True)

class MLScorer:
    """
    Machine Learning based stock scorer that replaces heuristic weights with data-driven factors.
    Uses gradient boosted trees with regularisation and cross-validation.

    Supports per-ticker models for more accurate predictions.
    """

    def __init__(self, model_type='regression', retrain_interval_days=0, ticker=None):
        """
        Initialize the ML scorer.

        Parameters:
            model_type (str): 'regression' for score prediction, 'classification' for buy/sell signals
            retrain_interval_days (int): How often to retrain the model (in days)
            ticker (str): Optional ticker symbol for per-ticker models
        """
        self.model_type = model_type
        self.retrain_interval_days = retrain_interval_days
        self.ticker = ticker
        self.model = None
        self.feature_names = None
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.target_scaler = RobustScaler()  # Added scaler for target values

        # Feature selection components
        self.feature_selector = SelectKBest(score_func=mutual_info_regression, k=10)  # Select top 10 features
        self.pca = PCA(n_components=0.98)  # Keep 98% of variance
        self.last_train_date = None

        # Model file paths
        self.model_dir = 'cache/models'

        # Create ticker-specific directory if needed
        if ticker:
            self.ticker_dir = os.path.join(self.model_dir, ticker)
            os.makedirs(self.ticker_dir, exist_ok=True)

            # Ticker-specific model paths
            self.model_path = os.path.join(self.ticker_dir, f'stock_scorer_{model_type}.joblib')
            self.scaler_path = os.path.join(self.ticker_dir, 'feature_scaler.joblib')
            self.target_scaler_path = os.path.join(self.ticker_dir, 'target_scaler.joblib')
            self.feature_selector_path = os.path.join(self.ticker_dir, 'feature_selector.joblib')
            self.pca_path = os.path.join(self.ticker_dir, 'feature_pca.joblib')
            self.feature_names_path = os.path.join(self.ticker_dir, 'feature_names.joblib')
        else:
            # Global model paths (for backward compatibility)
            self.model_path = os.path.join(self.model_dir, f'stock_scorer_{model_type}.joblib')
            self.scaler_path = os.path.join(self.model_dir, 'feature_scaler.joblib')
            self.target_scaler_path = os.path.join(self.model_dir, 'target_scaler.joblib')
            self.feature_selector_path = os.path.join(self.model_dir, 'feature_selector.joblib')
            self.pca_path = os.path.join(self.model_dir, 'feature_pca.joblib')
            self.feature_names_path = os.path.join(self.model_dir, 'feature_names.joblib')

        # Try to load existing model
        self._load_model()

    def _load_model(self):
        """Load model and preprocessors if they exist"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.pca = joblib.load(self.pca_path)
                self.feature_names = joblib.load(self.feature_names_path)

                # Load target scaler if it exists
                if os.path.exists(self.target_scaler_path):
                    self.target_scaler = joblib.load(self.target_scaler_path)

                # Load feature selector if it exists
                if os.path.exists(self.feature_selector_path):
                    self.feature_selector = joblib.load(self.feature_selector_path)

                # Get last modified time of model file
                last_modified = os.path.getmtime(self.model_path)
                self.last_train_date = datetime.fromtimestamp(last_modified)

                log_message(f"Loaded existing model trained on {self.last_train_date}")
                return True
        except Exception as e:
            log_message(f"Error loading model: {e}")

        return False

    def _save_model(self):
        """Save model and preprocessors"""
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            joblib.dump(self.target_scaler, self.target_scaler_path)  # Save target scaler
            joblib.dump(self.pca, self.pca_path)
            joblib.dump(self.feature_names, self.feature_names_path)
            self.last_train_date = datetime.now()
            log_message(f"Model saved successfully on {self.last_train_date}")
            return True
        except Exception as e:
            log_message(f"Error saving model: {e}")
            return False

    def _extract_features(self, stock_data):
        """
        Extract features from stock data for model input.

        Parameters:
            stock_data (dict): Stock data dictionary from get_stock_data function

        Returns:
            pandas.DataFrame: Features dataframe
        """
        # Extract relevant features from stock_data
        features = {}

        # Technical indicators
        features['rsi7'] = stock_data.get('rsi7', 50)
        features['rsi14'] = stock_data.get('rsi14', 50)
        features['bb_position'] = stock_data.get('bb_position', 0.5)
        features['above_ma5'] = int(stock_data.get('above_ma5', False))
        features['above_ma10'] = int(stock_data.get('above_ma10', False))
        features['above_ma20'] = int(stock_data.get('above_ma20', False))
        features['macd'] = stock_data.get('macd', 0)
        features['macd_signal'] = stock_data.get('macd_signal', 0)
        features['macd_hist'] = stock_data.get('macd_hist', 0)

        # Returns
        features['return_1d'] = stock_data.get('return_1d', 0)
        features['return_3d'] = stock_data.get('return_3d', 0)
        features['return_5d'] = stock_data.get('return_5d', 0)

        # Volatility
        features['atr_pct'] = stock_data.get('atr_pct', 0)
        features['avg_intraday_range'] = stock_data.get('avg_intraday_range', 0)

        # Volume
        features['volume_ratio'] = stock_data.get('volume_ratio', 1)

        # Gap patterns
        features['gap_ups_5d'] = stock_data.get('gap_ups_5d', 0)
        features['gap_downs_5d'] = stock_data.get('gap_downs_5d', 0)

        # Sentiment
        features['news_sentiment_score'] = stock_data.get('news_sentiment_score', 0)

        # Market cap (log-transformed)
        market_cap = stock_data.get('market_cap', 0)
        features['log_market_cap'] = np.log1p(market_cap) if market_cap > 0 else 0

        # Convert to DataFrame
        return pd.DataFrame([features])

    def _check_if_retrain_needed(self):
        """Check if model needs retraining based on last train date"""
        if self.last_train_date is None:
            return True

        days_since_last_train = (datetime.now() - self.last_train_date).days
        return days_since_last_train >= self.retrain_interval_days

    def train(self, historical_data, force=False):
        """
        Train the model using historical stock data with proper time-series cross-validation.

        Parameters:
            historical_data (list): List of dictionaries with historical stock data and outcomes
            force (bool): Force retraining even if interval hasn't elapsed

        Returns:
            bool: True if training was successful
        """
        if not force and not self._check_if_retrain_needed():
            log_message(f"Model for {self.ticker or 'all tickers'} is up to date, skipping training")
            return True

        # Filter data for specific ticker if needed
        if self.ticker:
            ticker_data = [item for item in historical_data if item.get('ticker') == self.ticker]
            if not ticker_data:
                log_message(f"No training data available for ticker {self.ticker}")
                return False
            training_data = ticker_data
        else:
            training_data = historical_data

        try:
            log_message("Training ML model with historical data using walk-forward validation...")

            # Ensure data is sorted by date for proper time-series validation
            training_data = sorted(training_data, key=lambda x: x['date'])

            # Extract features and targets from historical data
            features_list = []
            targets = []
            dates = []

            for item in training_data:
                features = self._extract_features(item['stock_data'])
                features_list.append(features)

                if self.model_type == 'regression':
                    targets.append(item['future_return'])
                else:  # classification
                    # Convert to binary signal (1 for positive return, 0 for negative)
                    targets.append(1 if item['future_return'] > 0 else 0)

                dates.append(item['date'])

            if not features_list:
                log_message("No training data available")
                return False

            # Combine all features
            X = pd.concat(features_list, ignore_index=True)
            y = np.array(targets)
            dates = np.array(dates)

            # Save feature names
            self.feature_names = X.columns.tolist()

            # Create a more robust time series cross-validation with expanding window
            # This ensures proper time separation between training and testing periods
            n_splits = 5
            test_size = len(X) // (n_splits + 1)  # Size of each test fold

            # Add a gap between train and test to prevent leakage
            gap_size = 5  # 5 days gap to prevent any leakage

            # Create and train the model with reduced complexity to prevent overfitting
            if self.model_type == 'regression':
                model = GradientBoostingRegressor(
                    n_estimators=300,  # Reduced from 500
                    learning_rate=0.01,
                    max_depth=3,       # Reduced from 5 to prevent overfitting
                    min_samples_split=15,  # Increased from 10
                    min_samples_leaf=10,   # Increased from 5
                    subsample=0.7,         # Reduced from 0.8
                    max_features='sqrt',
                    alpha=0.9,  # L1 regularization
                    loss='huber',  # More robust to outliers
                    random_state=42
                )
            else:  # classification
                model = GradientBoostingClassifier(
                    n_estimators=300,  # Reduced from 500
                    learning_rate=0.01,
                    max_depth=3,       # Reduced from 5
                    min_samples_split=15,  # Increased from 10
                    min_samples_leaf=10,   # Increased from 5
                    subsample=0.7,         # Reduced from 0.8
                    max_features='sqrt',
                    random_state=42
                )

            # Create a pipeline that properly handles preprocessing
            # This ensures that preprocessing is only fit on training data
            from sklearn.pipeline import Pipeline

            pipeline = Pipeline([
                ('scaler', self.scaler),
                ('pca', self.pca),
                ('model', model)
            ])

            # Implement walk-forward validation with expanding windows and gaps
            cv_scores = []

            # Use at least 30% of data for initial training
            min_train_size = max(int(len(X) * 0.3), 30)

            for i in range(n_splits):
                # Calculate split indices for expanding window
                train_end = min_train_size + i * test_size
                test_start = train_end + gap_size  # Add gap between train and test
                test_end = min(test_start + test_size, len(X))

                # Skip if we don't have enough data left for testing
                if test_end <= test_start:
                    continue

                # Get train/test indices
                train_idx = np.arange(0, train_end)
                test_idx = np.arange(test_start, test_end)

                # Get train/test data
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Scale target values for regression model
                if self.model_type == 'regression':
                    # Reshape y for scaler
                    y_train_reshaped = y_train.reshape(-1, 1)
                    # Fit target scaler only on training data
                    y_train_scaled = self.target_scaler.fit_transform(y_train_reshaped).ravel()

                    # Transform test targets using the scaler fit on training data
                    y_test_reshaped = y_test.reshape(-1, 1)
                    y_test_scaled = self.target_scaler.transform(y_test_reshaped).ravel()
                else:
                    # For classification, no need to scale targets
                    y_train_scaled = y_train
                    y_test_scaled = y_test

                # Train pipeline on this fold
                pipeline.fit(X_train, y_train_scaled)

                # Evaluate on test fold
                if self.model_type == 'regression':
                    y_pred_scaled = pipeline.predict(X_test)

                    # Convert predictions back to original scale for RMSE calculation
                    y_pred_orig = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

                    # Calculate RMSE on original scale
                    score = np.sqrt(mean_squared_error(y_test, y_pred_orig))
                    cv_scores.append(score)

                    log_message(f"Fold {i+1}: Train size={len(train_idx)}, Test size={len(test_idx)}, RMSE={score:.4f}")
                else:  # classification
                    y_pred = pipeline.predict(X_test)
                    score = accuracy_score(y_test_scaled, y_pred)
                    cv_scores.append(score)

                    log_message(f"Fold {i+1}: Train size={len(train_idx)}, Test size={len(test_idx)}, Accuracy={score:.4f}")

            # Final fit on all data
            # For the final model, we still need to fit the preprocessors on all data
            # since they'll be used for future predictions
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            self.pca.fit(X_scaled)
            X_pca = self.pca.transform(X_scaled)

            if self.model_type == 'regression':
                y_reshaped = y.reshape(-1, 1)
                y_scaled = self.target_scaler.fit_transform(y_reshaped).ravel()
            else:
                y_scaled = y

            model.fit(X_pca, y_scaled)
            self.model = model

            # Save the model
            self._save_model()

            # Log results
            if self.model_type == 'regression':
                log_message(f"Model training complete. Avg RMSE: {np.mean(cv_scores):.4f}")
            else:
                log_message(f"Model training complete. Avg Accuracy: {np.mean(cv_scores):.4f}")

            return True

        except Exception as e:
            log_message(f"Error training model: {e}")
            return False

    def predict(self, stock_data):
        """
        Generate a prediction for the given stock data.

        Parameters:
            stock_data (dict): Stock data dictionary from get_stock_data function

        Returns:
            float: Predicted score or signal
        """
        if self.model is None:
            log_message("Model not trained yet")
            return None

        try:
            # Extract features
            features = self._extract_features(stock_data)

            # Check if we have all required features
            missing_features = set(self.feature_names) - set(features.columns)
            if missing_features:
                for feature in missing_features:
                    features[feature] = 0  # Fill missing with zeros

            # Ensure features are in the right order
            features = features[self.feature_names]

            # Transform features using the same preprocessing pipeline
            X_scaled = self.scaler.transform(features)
            X_pca = self.pca.transform(X_scaled)

            # Make prediction
            prediction_scaled = self.model.predict(X_pca)[0]

            # For regression models, inverse transform the prediction to get original scale
            if self.model_type == 'regression':
                prediction = self.target_scaler.inverse_transform([[prediction_scaled]])[0][0]
            else:
                prediction = prediction_scaled

            # Log prediction details for debugging
            log_message(f"Prediction for stock: {stock_data.get('ticker', 'Unknown')}, "
                        f"Prediction: {prediction:.2f}, "
                        f"Features used: {len(features.columns)}")

            return prediction

        except Exception as e:
            log_message(f"Error making prediction: {e}")
            return None

    def get_feature_importance(self):
        """
        Get feature importance from the model.

        Returns:
            dict: Feature names and their importance scores
        """
        if self.model is None or self.feature_names is None:
            return {}

        try:
            importances = self.model.feature_importances_
            feature_importance = dict(zip(self.feature_names, importances))
            return {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)}
        except Exception as e:
            log_message(f"Error getting feature importance: {e}")
            return {}

def train_ticker_models(tickers, force=False, lookback_days=180, prediction_horizon=5):
    """
    Train individual ML models for each ticker in the list.

    Parameters:
        tickers (list): List of ticker symbols to train models for
        force (bool): Force retraining even if interval hasn't elapsed
        lookback_days (int): Number of days to look back for historical data
        prediction_horizon (int): Number of days ahead to predict returns for

    Returns:
        dict: Dictionary with ticker symbols as keys and training success status as values
    """
    results = {}

    # Collect training data for all tickers
    training_data = collect_training_data(tickers, lookback_days, prediction_horizon)

    # Train individual models for each ticker
    for ticker in tickers:
        try:
            # Create ticker-specific model
            ml_scorer = MLScorer(model_type='regression', retrain_interval_days=7, ticker=ticker)

            # Train the model
            success = ml_scorer.train(training_data, force=force)
            results[ticker] = success

            log_message(f"Training model for {ticker}: {'Success' if success else 'Failed'}")
        except Exception as e:
            log_message(f"Error training model for {ticker}: {e}")
            results[ticker] = False

    return results

def verify_no_future_data(historical_data, current_idx):
    """
    Verify that no future data is used in feature calculation.

    Parameters:
        historical_data (pandas.DataFrame): Historical price data
        current_idx (int): Current index in the historical data

    Returns:
        bool: True if no future data is used, False otherwise
    """
    # Check if the historical data is properly sliced
    if len(historical_data) > current_idx + 1:
        log_message(f"WARNING: Historical data contains future information. Expected length: {current_idx + 1}, Actual length: {len(historical_data)}")
        return False
    return True

def get_stock_data_point_in_time(ticker, historical_data, current_idx):
    """
    Get stock data for a specific point in time, ensuring no future data is used.

    Parameters:
        ticker (str): Ticker symbol
        historical_data (pandas.DataFrame): Full historical price data
        current_idx (int): Current index in the historical data

    Returns:
        tuple: (stock_data, error) - stock data dictionary and error message if any
    """
    from modules.technical_analysis import get_stock_data

    # Create a slice of history up to this point (including current day)
    hist_slice = historical_data.iloc[:current_idx+1].copy()

    # Verify no future data is used
    verify_no_future_data(hist_slice, current_idx)

    # Get the stock data for this point in time
    stock_data, error = get_stock_data(ticker, historical_data=hist_slice)

    return stock_data, error

def collect_training_data(tickers, lookback_days=180, prediction_horizon=5):
    """
    Collect historical data for model training with proper time-series separation.

    Parameters:
        tickers (list): List of ticker symbols to collect data for
        lookback_days (int): Number of days to look back for historical data
        prediction_horizon (int): Number of days ahead to predict returns for

    Returns:
        list: List of dictionaries with stock data and future returns, sorted by date
    """
    from modules.technical_analysis import get_stock_data

    training_data = []

    try:
        log_message(f"Collecting training data for {len(tickers)} tickers...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + prediction_horizon + 30)  # Add extra buffer for indicators

        # Use a fixed timestamp for caching to ensure consistency
        cache_timestamp = f"training_fixed_{end_date.strftime('%Y%m%d')}"

        for ticker in tickers:
            try:
                # Get full history for the ticker
                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = end_date.strftime('%Y-%m-%d')

                hist = get_stock_history(ticker, start_date_str, end_date_str, "1d", cache_timestamp)

                if hist.empty or len(hist) < prediction_horizon + 30:  # Need more data for indicators
                    continue

                # Process each day in the history (except the last prediction_horizon days)
                # Start from day 30 to ensure enough history for indicators
                for i in range(30, len(hist) - prediction_horizon):
                    # Get the date for this data point
                    current_date = hist.index[i].date()

                    # Get the stock data for this point in time with strict time boundary
                    stock_data, error = get_stock_data_point_in_time(ticker, hist, i)

                    if error or not stock_data:
                        continue

                    # Calculate future return
                    future_price = hist['Close'].iloc[i + prediction_horizon]
                    current_price = hist['Close'].iloc[i]
                    # Avoid division by zero
                    if current_price > 0:
                        future_return = ((future_price / current_price) - 1) * 100
                    else:
                        # Skip this data point by continuing to the next iteration
                        continue

                    # Add to training data
                    training_data.append({
                        'ticker': ticker,
                        'date': current_date,
                        'stock_data': stock_data,
                        'future_return': future_return,
                        'future_signal': 1 if future_return > 0 else 0
                    })

                log_message(f"Collected {len(training_data)} training samples for {ticker}")

            except Exception as e:
                log_message(f"Error collecting data for {ticker}: {e}")
                continue

        # Sort training data by date to ensure proper time-series order
        training_data.sort(key=lambda x: x['date'])

        log_message(f"Total training samples collected: {len(training_data)}")
        return training_data

    except Exception as e:
        log_message(f"Error in collect_training_data: {e}")
        return []

def schedule_model_updates(tickers, schedule_days=7):
    """
    Schedule periodic updates for all ticker models.

    Parameters:
        tickers (list): List of ticker symbols to update models for
        schedule_days (int): Number of days between updates

    Returns:
        None
    """
    import threading
    import time
    from datetime import datetime, timedelta

    def update_models():
        while True:
            try:
                log_message(f"Scheduled update of {len(tickers)} ticker models starting")
                train_ticker_models(tickers, force=False)
                log_message(f"Scheduled update completed. Next update in {schedule_days} days")

                # Sleep for schedule_days
                time.sleep(schedule_days * 24 * 60 * 60)
            except Exception as e:
                log_message(f"Error in scheduled model update: {e}")
                # Sleep for an hour before retrying
                time.sleep(60 * 60)

    # Start the update thread
    update_thread = threading.Thread(target=update_models, daemon=True)
    update_thread.start()

    log_message(f"Model update scheduler started for {len(tickers)} tickers every {schedule_days} days")

def initialize_asx_ticker_models(asx_tickers=None, schedule_updates=True):
    """
    Initialize and optionally schedule updates for ASX ticker models.

    Parameters:
        asx_tickers (list): List of ASX ticker symbols. If None, a default list should be provided.
        schedule_updates (bool): Whether to schedule periodic updates

    Returns:
        dict: Results of initial model training
    """
    # Default to a sample of ASX tickers if none provided
    if asx_tickers is None:
        # 1. Load a list of ASX tickers from a file or database
        # 2. Or use an API to fetch the current list of ASX tickers
        log_message("No ASX tickers provided. Using a placeholder list.")
        asx_tickers = [ticker + ".AX" for ticker in ["BHP", "CBA", "WBC", "NAB", "ANZ"]]
        log_message(f"Using {len(asx_tickers)} placeholder tickers. In production, provide the full list of ~200 ASX tickers.")

    # Train initial models
    log_message(f"Initializing models for {len(asx_tickers)} ASX tickers")
    results = train_ticker_models(asx_tickers, force=True)

    # Schedule updates if requested
    if schedule_updates:
        schedule_model_updates(asx_tickers)

    return results

def score_stock_ml(stock_data, ml_scorer=None, all_predictions=None):
    """
    Score a stock using the ML model instead of heuristic weights.

    Parameters:
        stock_data (dict): Stock data dictionary from get_stock_data function
        ml_scorer (MLScorer): Optional pre-initialized ML scorer
        all_predictions (list): Optional list of all predictions for relative ranking

    Returns:
        dict: Updated stock data with ML-based scores
    """
    try:
        # Get ticker from stock_data
        ticker = stock_data.get('ticker')

        # Create or use existing scorer
        if ml_scorer is None:
            # Use ticker-specific model if ticker is available
            if ticker:
                ml_scorer = MLScorer(model_type='regression', retrain_interval_days=7, ticker=ticker)
            else:
                ml_scorer = MLScorer(model_type='regression')

        # Check if model needs training
        if ml_scorer.model is None:
            log_message("ML model not trained. Using fallback scoring method.")
            return stock_data

        # Get prediction from model
        ml_score = ml_scorer.predict(stock_data)

        if ml_score is not None:
            # Store the raw predicted return percentage
            stock_data['ml_score_raw'] = ml_score
            stock_data['predicted_return_pct'] = ml_score

            # If we have all predictions, use relative ranking
            if all_predictions is not None and len(all_predictions) > 1:
                # Sort all predictions
                sorted_predictions = sorted(all_predictions)
                # Find the percentile rank of this prediction
                rank = sorted_predictions.index(ml_score) / (len(sorted_predictions) - 1) * 100
                # Ensure we have a good spread of scores (20-80 range minimum)
                scaled_score = 20 + (rank * 0.6)  # Maps 0-100 percentile to 20-80 score range
            else:
                # Fallback to absolute scaling if we don't have all predictions
                # Scale the percentage return to a 0-100 score
                # Assuming returns can range from -20% to +20% for typical stocks
                scaled_score = min(100, max(0, (ml_score + 20) * 2.5))

            # Update stock data with ML scores
            stock_data['day_trading_score'] = scaled_score

            # Determine trading strategy based on scaled score
            if scaled_score >= 70:
                stock_data['day_trading_strategy'] = "Strong Buy"
                stock_data['strategy_details'] = f"ML model predicts {ml_score:.2f}% return (Score: {scaled_score:.1f})"
            elif scaled_score >= 60:
                stock_data['day_trading_strategy'] = "Buy"
                stock_data['strategy_details'] = f"ML model predicts {ml_score:.2f}% return (Score: {scaled_score:.1f})"
            elif scaled_score >= 40:
                stock_data['day_trading_strategy'] = "Neutral/Watch"
                stock_data['strategy_details'] = f"ML model predicts {ml_score:.2f}% return (Score: {scaled_score:.1f})"
            elif scaled_score >= 30:
                stock_data['day_trading_strategy'] = "Sell"
                stock_data['strategy_details'] = f"ML model predicts {ml_score:.2f}% return (Score: {scaled_score:.1f})"
            else:
                stock_data['day_trading_strategy'] = "Strong Sell"
                stock_data['strategy_details'] = f"ML model predicts {ml_score:.2f}% return (Score: {scaled_score:.1f})"

            # Get feature importance for explainability
            feature_importance = ml_scorer.get_feature_importance()
            stock_data['ml_feature_importance'] = feature_importance

        return stock_data

    except Exception as e:
        log_message(f"Error in ML scoring: {e}")
        return stock_data
