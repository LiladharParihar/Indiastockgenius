import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

def calculate_indicators(data):
    if data.empty or 'Close' not in data:
        return {}

    close = data['Close']
    indicators = {
        'MA50': SMAIndicator(close=close, window=50).sma_indicator().iloc[-1],
        'MA100': SMAIndicator(close=close, window=100).sma_indicator().iloc[-1],
        'MA200': SMAIndicator(close=close, window=200).sma_indicator().iloc[-1],
        'RSI': RSIIndicator(close=close, window=14).rsi().iloc[-1],
        'MACD': MACD(close=close).macd().iloc[-1],
        'Bollinger': BollingerBands(close=close).bollinger_mavg().iloc[-1]
    }
    return indicators