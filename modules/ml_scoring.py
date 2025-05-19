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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
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
    Uses gradient boosted trees with regularization and cross-validation.
    """

    def __init__(self, model_type='regression', retrain_interval_days=7):
        """
        Initialize the ML scorer.

        Parameters:
            model_type (str): 'regression' for score prediction, 'classification' for buy/sell signals
            retrain_interval_days (int): How often to retrain the model (in days)
        """
        self.model_type = model_type
        self.retrain_interval_days = retrain_interval_days
        self.model = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        self.last_train_date = None

        # Model file paths
        self.model_dir = 'cache/models'
        self.model_path = os.path.join(self.model_dir, f'stock_scorer_{model_type}.joblib')
        self.scaler_path = os.path.join(self.model_dir, 'feature_scaler.joblib')
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
        Train the model using historical stock data.

        Parameters:
            historical_data (list): List of dictionaries with historical stock data and outcomes
            force (bool): Force retraining even if interval hasn't elapsed

        Returns:
            bool: True if training was successful
        """
        if not force and not self._check_if_retrain_needed():
            log_message("Model is up to date, skipping training")
            return True

        try:
            log_message("Training ML model with historical data...")

            # Extract features and targets from historical data
            features_list = []
            targets = []

            for item in historical_data:
                features = self._extract_features(item['stock_data'])
                features_list.append(features)

                if self.model_type == 'regression':
                    targets.append(item['future_return'])
                else:  # classification
                    # Convert to binary signal (1 for positive return, 0 for negative)
                    targets.append(1 if item['future_return'] > 0 else 0)

            if not features_list:
                log_message("No training data available")
                return False

            # Combine all features
            X = pd.concat(features_list, ignore_index=True)
            y = np.array(targets)

            # Save feature names
            self.feature_names = X.columns.tolist()

            # Create time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)

            # Create and train the model
            if self.model_type == 'regression':
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    subsample=0.8,
                    random_state=42
                )
            else:  # classification
                model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    subsample=0.8,
                    random_state=42
                )

            # Fit scaler and PCA on all data
            X_scaled = self.scaler.fit_transform(X)
            X_pca = self.pca.fit_transform(X_scaled)

            # Train with cross-validation
            cv_scores = []
            for train_idx, test_idx in tscv.split(X_pca):
                X_train, X_test = X_pca[train_idx], X_pca[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model.fit(X_train, y_train)

                if self.model_type == 'regression':
                    y_pred = model.predict(X_test)
                    score = np.sqrt(mean_squared_error(y_test, y_pred))  # RMSE
                    cv_scores.append(score)
                else:  # classification
                    y_pred = model.predict(X_test)
                    score = accuracy_score(y_test, y_pred)
                    cv_scores.append(score)

            # Final fit on all data
            model.fit(X_pca, y)
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

            # Transform features
            X_scaled = self.scaler.transform(features)
            X_pca = self.pca.transform(X_scaled)

            # Make prediction
            prediction = self.model.predict(X_pca)[0]

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

def collect_training_data(tickers, lookback_days=180, prediction_horizon=5):
    """
    Collect historical data for model training.

    Parameters:
        tickers (list): List of ticker symbols to collect data for
        lookback_days (int): Number of days to look back for historical data
        prediction_horizon (int): Number of days ahead to predict returns for

    Returns:
        list: List of dictionaries with stock data and future returns
    """
    from modules.technical_analysis import get_stock_data

    training_data = []

    try:
        log_message(f"Collecting training data for {len(tickers)} tickers...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + prediction_horizon)

        for ticker in tickers:
            try:
                # Get full history for the ticker
                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = end_date.strftime('%Y-%m-%d')

                # Use a unique timestamp for caching
                cache_timestamp = f"training_{datetime.now().strftime('%Y%m%d%H%M%S')}"

                hist = get_stock_history(ticker, start_date_str, end_date_str, "1d", cache_timestamp)

                if hist.empty or len(hist) < prediction_horizon + 10:
                    continue

                # Process each day in the history (except the last prediction_horizon days)
                for i in range(10, len(hist) - prediction_horizon):
                    # Get the date for this data point
                    current_date = hist.index[i].date()

                    # Create a slice of history up to this point
                    hist_slice = hist.iloc[:i+1]

                    # Get the stock data for this point in time
                    stock_data, error = get_stock_data(ticker, historical_data=hist_slice)

                    if error or not stock_data:
                        continue

                    # Calculate future return
                    future_price = hist['Close'].iloc[i + prediction_horizon]
                    current_price = hist['Close'].iloc[i]
                    future_return = ((future_price / current_price) - 1) * 100

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

        log_message(f"Total training samples collected: {len(training_data)}")
        return training_data

    except Exception as e:
        log_message(f"Error in collect_training_data: {e}")
        return []

def score_stock_ml(stock_data, ml_scorer=None):
    """
    Score a stock using the ML model instead of heuristic weights.

    Parameters:
        stock_data (dict): Stock data dictionary from get_stock_data function
        ml_scorer (MLScorer): Optional pre-initialized ML scorer

    Returns:
        dict: Updated stock data with ML-based scores
    """
    try:
        # Create or use existing scorer
        if ml_scorer is None:
            ml_scorer = MLScorer(model_type='regression')

        # Check if model needs training
        if ml_scorer.model is None:
            log_message("ML model not trained. Using fallback scoring method.")
            return stock_data

        # Get prediction from model
        ml_score = ml_scorer.predict(stock_data)

        if ml_score is not None:
            # Scale the score to 0-100 range (assuming model output is roughly -10 to +10)
            scaled_score = min(100, max(0, (ml_score + 10) * 5))

            # Update stock data with ML scores
            stock_data['ml_score_raw'] = ml_score
            stock_data['day_trading_score'] = scaled_score

            # Determine trading strategy based on ML score
            if scaled_score >= 70:
                stock_data['day_trading_strategy'] = "Strong Buy"
                stock_data['strategy_details'] = "ML model predicts strong positive return"
            elif scaled_score >= 60:
                stock_data['day_trading_strategy'] = "Buy"
                stock_data['strategy_details'] = "ML model predicts positive return"
            elif scaled_score >= 45:
                stock_data['day_trading_strategy'] = "Neutral/Watch"
                stock_data['strategy_details'] = "ML model predicts flat return"
            elif scaled_score >= 35:
                stock_data['day_trading_strategy'] = "Sell"
                stock_data['strategy_details'] = "ML model predicts negative return"
            else:
                stock_data['day_trading_strategy'] = "Strong Sell"
                stock_data['strategy_details'] = "ML model predicts strong negative return"

            # Get feature importance for explainability
            feature_importance = ml_scorer.get_feature_importance()
            stock_data['ml_feature_importance'] = feature_importance

        return stock_data

    except Exception as e:
        log_message(f"Error in ML scoring: {e}")
        return stock_data
