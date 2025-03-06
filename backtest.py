import numpy as np
from model import predict_lstm

def backtest_strategy(historical):
    if historical.empty or 'Close' not in historical:
        return {"profit": 0, "accuracy": 0}

    predictions = predict_lstm(historical['Close'].values, forecast_days=30)
    actual = historical['Close'].values[-30:]  # Last 30 days
    pred = predictions[:30]  # Compare 30-day prediction

    profit = np.mean((pred - actual) / actual) * 100  # Percentage profit
    accuracy = np.mean(np.abs(pred - actual) / actual) < 0.05  # 5% error threshold
    return {"profit": profit, "accuracy": accuracy * 100}