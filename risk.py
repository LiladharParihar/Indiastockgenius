import numpy as np
from data import fetch_stock_data
from model import predict_lstm

def assess_risk(historical, predictions):
    if historical.empty or 'Close' not in historical:
        return {"volatility": 0, "stop_loss": 0, "confidence": 0}

    close = historical['Close'].values
    volatility = np.std(close[-30:])  # 30-day volatility
    last_price = close[-1]
    stop_loss = last_price * 0.95  # 5% below current
    confidence = 0.8  # Mock—train LSTM to estimate real confidence
    return {"volatility": volatility, "stop_loss": stop_loss, "confidence": confidence}