"""
Visualization Module

This module contains functions for preparing data for charts and visualizations.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from modules.data_retrieval import get_stock_history
from modules.technical_analysis import get_stock_data, calculate_score_contribution
from sklearn.tree import export_graphviz
import os
import json

def get_historical_data_for_chart(ticker_symbol, days=30):
    """
    Retrieves historical stock data for charting purposes.

    Parameters:
        ticker_symbol (str): The stock ticker symbol
        days (int): Number of days of historical data to retrieve

    Returns:
        dict: Dictionary containing historical data formatted for charting
    """
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Format dates
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        # Get historical data
        hist = get_stock_history(ticker_symbol, start_date_str, end_date_str, "1d")

        if len(hist) == 0:
            return {"error": "No historical data available"}

        # Format data for chart
        dates = [d.strftime('%Y-%m-%d') for d in hist.index]
        prices = hist['Close'].tolist()
        volumes = hist['Volume'].tolist()

        # Calculate moving averages
        hist['MA5'] = hist['Close'].rolling(window=5).mean()
        hist['MA20'] = hist['Close'].rolling(window=20).mean()

        ma5 = hist['MA5'].tolist()
        ma20 = hist['MA20'].tolist()

        return {
            "ticker": ticker_symbol,
            "dates": dates,
            "prices": prices,
            "volumes": volumes,
            "ma5": ma5,
            "ma20": ma20
        }
    except Exception as e:
        return {"error": str(e)}

def get_detailed_stock_metrics(stock_data):
    """
    Extracts and formats detailed metrics from stock data for visualization.

    Parameters:
        stock_data (dict): Dictionary containing stock data

    Returns:
        dict: Dictionary with formatted metrics for visualization
    """
    if not stock_data:
        return {"error": "No stock data provided"}

    # Extract key metrics in the original structure
    metrics_original = {
        "ticker": stock_data.get('ticker', ''),
        "company_name": stock_data.get('company_name', ''),
        "current_price": stock_data.get('current_price', 0),
        "day_trading_score": stock_data.get('day_trading_score', 0),
        "strategy": stock_data.get('day_trading_strategy', ''),
        "technical": {
            "rsi7": stock_data.get('rsi7', 0),
            "rsi14": stock_data.get('rsi14', 0),
            "macd": stock_data.get('macd', 0),
            "macd_signal": stock_data.get('macd_signal', 0),
            "macd_hist": stock_data.get('macd_hist', 0),
            "macd_trend": stock_data.get('macd_trend', ''),
            "bb_position": stock_data.get('bb_position', 0),
            "above_ma5": stock_data.get('above_ma5', False),
            "above_ma10": stock_data.get('above_ma10', False),
            "above_ma20": stock_data.get('above_ma20', False),
        },
        "volatility": {
            "atr": stock_data.get('atr', 0),
            "atr_pct": stock_data.get('atr_pct', 0),
            "avg_intraday_range": stock_data.get('avg_intraday_range', 0),
            "gap_ups_5d": stock_data.get('gap_ups_5d', 0),
            "gap_downs_5d": stock_data.get('gap_downs_5d', 0),
        },
        "momentum": {
            "return_1d": stock_data.get('return_1d', 0),
            "return_3d": stock_data.get('return_3d', 0),
            "return_5d": stock_data.get('return_5d', 0),
        },
        "volume": {
            "volume_ratio": stock_data.get('volume_ratio', 0),
        },
        "sentiment": {
            "news_sentiment_score": stock_data.get('news_sentiment_score', 0),
            "news_sentiment_label": stock_data.get('news_sentiment_label', ''),
        }
    }

    # Calculate score contributions
    score_contributions = calculate_score_contribution(stock_data)
    metrics_original["score_contributions"] = score_contributions

    # Create the structure expected by the template
    metrics = {
        # Keep the original structure
        **metrics_original,

        # Add the structure expected by the template
        "Technical Indicators": {
            "RSI (7-day)": f"{stock_data.get('rsi7', 0):.2f}",
            "RSI (14-day)": f"{stock_data.get('rsi14', 0):.2f}",
            "MACD": f"{stock_data.get('macd', 0):.4f}",
            "MACD Signal": f"{stock_data.get('macd_signal', 0):.4f}",
            "MACD Histogram": f"{stock_data.get('macd_hist', 0):.4f}",
            "MACD Trend": stock_data.get('macd_trend', 'neutral').capitalize(),
            "Bollinger Position": f"{stock_data.get('bb_position', 0):.2f}",
            "Above MA5": "Yes" if stock_data.get('above_ma5', False) else "No",
            "Above MA20": "Yes" if stock_data.get('above_ma20', False) else "No"
        },
        "Volatility Metrics": {
            "ATR": f"${stock_data.get('atr', 0):.2f}",
            "ATR %": f"{stock_data.get('atr_pct', 0):.2f}%",
            "Avg Intraday Range": f"{stock_data.get('avg_intraday_range', 0):.2f}%",
            "Gap Ups (5d)": stock_data.get('gap_ups_5d', 0),
            "Gap Downs (5d)": stock_data.get('gap_downs_5d', 0)
        },
        "Performance Metrics": {
            "1-Day Return": f"{stock_data.get('return_1d', 0):.2f}%",
            "3-Day Return": f"{stock_data.get('return_3d', 0):.2f}%",
            "5-Day Return": f"{stock_data.get('return_5d', 0):.2f}%",
            "Volume Ratio": f"{stock_data.get('volume_ratio', 0):.2f}x",
            "News Sentiment": f"{stock_data.get('news_sentiment_label', 'Neutral')} ({stock_data.get('news_sentiment_score', 0):.2f})"
        }
    }

    return metrics

def prepare_price_chart_data(ticker_symbol, days=30):
    """
    Prepares comprehensive price chart data including technical indicators.

    Parameters:
        ticker_symbol (str): The stock ticker symbol
        days (int): Number of days of historical data to retrieve

    Returns:
        dict: Dictionary containing chart data with technical indicators
    """
    try:
        # Get basic historical data
        chart_data = get_historical_data_for_chart(ticker_symbol, days)

        if "error" in chart_data:
            return chart_data

        # Get current stock data for additional metrics
        stock_data, error = get_stock_data(ticker_symbol)

        if error:
            chart_data["error_details"] = error
            return chart_data

        # Add current metrics
        chart_data["current_metrics"] = {
            "price": stock_data.get('current_price', 0),
            "day_trading_score": stock_data.get('day_trading_score', 0),
            "strategy": stock_data.get('day_trading_strategy', ''),
            "rsi7": stock_data.get('rsi7', 0),
            "macd_trend": stock_data.get('macd_trend', ''),
            "atr_pct": stock_data.get('atr_pct', 0),
            "volume_ratio": stock_data.get('volume_ratio', 0),
            "news_sentiment": stock_data.get('news_sentiment_score', 0),
        }

        return chart_data
    except Exception as e:
        return {"error": str(e)}

def get_stock_comparison_data(ticker_symbols, metric='day_trading_score'):
    """
    Retrieves data for comparing multiple stocks based on a specified metric.

    Parameters:
        ticker_symbols (list): List of stock ticker symbols to compare
        metric (str): The metric to use for comparison

    Returns:
        dict: Dictionary containing comparison data
    """
    if not ticker_symbols:
        return {"error": "No ticker symbols provided"}

    comparison_data = {
        "tickers": [],
        "values": [],
        "labels": [],
        "colors": []
    }

    for ticker in ticker_symbols:
        stock_data, error = get_stock_data(ticker)

        if error:
            continue

        # Add data for this ticker
        comparison_data["tickers"].append(ticker)

        # Get the metric value
        value = stock_data.get(metric, 0)
        comparison_data["values"].append(value)

        # Create label
        label = f"{ticker} - {stock_data.get('company_name', '')}"
        comparison_data["labels"].append(label)

        # Determine color based on strategy
        strategy = stock_data.get('day_trading_strategy', '')
        if strategy == "Strong Buy":
            color = "green"
        elif strategy == "Buy":
            color = "lightgreen"
        elif strategy == "Neutral/Watch":
            color = "gray"
        elif strategy == "Sell":
            color = "pink"
        elif strategy == "Strong Sell":
            color = "red"
        else:
            color = "blue"

        comparison_data["colors"].append(color)

    return comparison_data

def create_gbdt_tree_visualization(model, feature_names, output_folder, max_trees=5):
    """
    Creates visualizations for GBDT trees.

    Parameters:
        model: Trained GBDT model (GradientBoostingRegressor or GradientBoostingClassifier)
        feature_names (list): List of feature names
        output_folder (str): Folder to save the visualizations
        max_trees (int): Maximum number of trees to visualize

    Returns:
        dict: Dictionary with filenames of created visualizations
    """
    if not hasattr(model, 'estimators_'):
        return {"error": "Model is not a gradient boosted tree model"}

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Dictionary to store visualization filenames
    visualizations = {}

    # Create a visualization for a single tree (the first one)
    tree_idx = 0
    estimator = model.estimators_[0, 0]  # First tree in the ensemble

    # Create a Plotly figure for the tree
    tree_data = export_tree_to_plotly(estimator, feature_names)

    # Create a tree visualization
    fig = go.Figure(data=[go.Scatter(
        x=tree_data['x'],
        y=tree_data['y'],
        mode='markers+text',
        marker=dict(
            size=20,
            color=tree_data['colors'],
            line=dict(width=2, color='DarkSlateGrey')
        ),
        text=tree_data['labels'],
        textposition="middle center",
        hoverinfo='text',
        hovertext=tree_data['hover_text']
    )])

    # Add edges (connections between nodes)
    for edge in tree_data['edges']:
        fig.add_shape(
            type="line",
            x0=edge['x0'], y0=edge['y0'],
            x1=edge['x1'], y1=edge['y1'],
            line=dict(color="RoyalBlue", width=1)
        )

    # Update layout
    fig.update_layout(
        title=f"GBDT Tree Visualization (Tree #{tree_idx})",
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )

    # Save the figure
    tree_file = "gbdt_tree.html"
    fig.write_html(os.path.join(output_folder, tree_file))
    visualizations['tree'] = tree_file

    # Create an animation of multiple trees
    num_trees = min(max_trees, len(model.estimators_))

    # Prepare data for animation
    frames = []
    for i in range(num_trees):
        estimator = model.estimators_[i, 0]
        tree_data = export_tree_to_plotly(estimator, feature_names)

        # Create a frame for this tree
        frame = go.Frame(
            data=[go.Scatter(
                x=tree_data['x'],
                y=tree_data['y'],
                mode='markers+text',
                marker=dict(
                    size=20,
                    color=tree_data['colors'],
                    line=dict(width=2, color='DarkSlateGrey')
                ),
                text=tree_data['labels'],
                textposition="middle center",
                hoverinfo='text',
                hovertext=tree_data['hover_text']
            )],
            name=f"Tree {i}"
        )
        frames.append(frame)

    # Create the base figure for animation
    tree_data = export_tree_to_plotly(model.estimators_[0, 0], feature_names)

    fig = go.Figure(
        data=[go.Scatter(
            x=tree_data['x'],
            y=tree_data['y'],
            mode='markers+text',
            marker=dict(
                size=20,
                color=tree_data['colors'],
                line=dict(width=2, color='DarkSlateGrey')
            ),
            text=tree_data['labels'],
            textposition="middle center",
            hoverinfo='text',
            hovertext=tree_data['hover_text']
        )],
        frames=frames
    )

    # Add slider and buttons for animation
    fig.update_layout(
        title="GBDT Trees Animation",
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 1000, "redraw": True}, "fromcurrent": True}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }],
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 16},
                "prefix": "Tree: ",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 300, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [
                        [f"Tree {i}"],
                        {"frame": {"duration": 300, "redraw": True}, "mode": "immediate", "transition": {"duration": 300}}
                    ],
                    "label": f"{i}",
                    "method": "animate"
                } for i in range(num_trees)
            ]
        }],
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )

    # Save the animation
    animation_file = "gbdt_trees_animation.html"
    fig.write_html(os.path.join(output_folder, animation_file))
    visualizations['animation'] = animation_file

    return visualizations

def export_tree_to_plotly(tree, feature_names):
    """
    Exports a decision tree to a format suitable for Plotly visualization.

    Parameters:
        tree: A decision tree estimator
        feature_names (list): List of feature names

    Returns:
        dict: Dictionary with tree data for Plotly
    """
    # Get tree structure
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    value = tree.tree_.value

    # Initialize lists to store node positions and labels
    x = []
    y = []
    labels = []
    hover_text = []
    colors = []
    edges = []

    # Function to recursively compute node positions
    def add_node(node_id, depth, parent_x=None, parent_y=None, is_left=True):
        if node_id == -1:  # Leaf node
            return

        # Compute x position based on tree structure
        if parent_x is None:  # Root node
            pos_x = 0
        else:
            # Spread nodes horizontally based on depth
            spread = 2.0 / (2 ** depth)
            if is_left:
                pos_x = parent_x - spread
            else:
                pos_x = parent_x + spread

        # Y position is based on depth
        pos_y = -depth

        # Store node position
        x.append(pos_x)
        y.append(pos_y)

        # Create node label
        if children_left[node_id] == -1 and children_right[node_id] == -1:
            # Leaf node - show the predicted value
            node_val = value[node_id][0, 0]
            label = f"{node_val:.2f}"
            color = 'rgba(255, 165, 0, 0.7)'  # Orange for leaf nodes
            hover = f"Leaf Node<br>Value: {node_val:.4f}"
        else:
            # Decision node - show the feature and threshold
            feat_name = feature_names[feature[node_id]] if feature[node_id] >= 0 else "Unknown"
            label = f"{feat_name}"
            color = 'rgba(100, 149, 237, 0.7)'  # Cornflower blue for decision nodes
            hover = f"Feature: {feat_name}<br>Threshold: {threshold[node_id]:.4f}"

        labels.append(label)
        hover_text.append(hover)
        colors.append(color)

        # Add edge to parent if not root
        if parent_x is not None and parent_y is not None:
            edges.append({
                'x0': parent_x, 'y0': parent_y,
                'x1': pos_x, 'y1': pos_y
            })

        # Process children
        add_node(children_left[node_id], depth + 1, pos_x, pos_y, True)
        add_node(children_right[node_id], depth + 1, pos_x, pos_y, False)

    # Start with the root node
    add_node(0, 0)

    return {
        'x': x,
        'y': y,
        'labels': labels,
        'hover_text': hover_text,
        'colors': colors,
        'edges': edges
    }
