import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from data import fetch_stock_data

def predict_lstm(prices, volumes=None, model=None, scaler=None, seq_length=60, forecast_days=30):
    """
    Predict future stock prices using an LSTM model with optional caching.
    
    Args:
        prices (np.array): Historical closing prices.
        volumes (np.array, optional): Historical volumes (ignored for now, added for compatibility).
        model (tf.keras.Model, optional): Pre-trained model to reuse.
        scaler (MinMaxScaler, optional): Pre-fitted scaler.
        seq_length (int): Sequence length for LSTM input.
        forecast_days (int): Number of days to predict.
    
    Returns:
        dict or list: {'predictions': list, 'model': model, 'scaler': scaler} if training new model,
                      or just predictions list if using cached model/scaler.
    """
    try:
        if model is None or scaler is None:
            # Train a new model if not provided
            prices = np.array(prices, dtype=float)
            if len(prices) < seq_length + 29:  # Need enough data for 30-day forecast
                print("Not enough data for training LSTM")
                return [np.nan] * forecast_days
            model, scaler = train_lstm(prices, seq_length=seq_length)
            if model is None or scaler is None:
                return [np.nan] * forecast_days

        # Ensure prices is a numpy array and has enough data
        prices = np.array(prices, dtype=float)
        if len(prices) < seq_length:
            print("Not enough data for prediction")
            return [np.nan] * forecast_days

        scaled_prices = scaler.transform(prices.reshape(-1, 1))
        last_sequence = scaled_prices[-seq_length:]
        if len(last_sequence) < seq_length:
            return [np.nan] * forecast_days

        # Multi-step prediction
        predictions = []
        current_sequence = last_sequence.copy()
        for _ in range(forecast_days):
            input_data = np.array([current_sequence])
            next_pred = model.predict(input_data, verbose=0)
            next_price = scaler.inverse_transform(next_pred)[0][0]  # Take first day of 30-day output
            predictions.append(next_price)
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = scaler.transform([[next_price]])[0][0]

        return {'predictions': predictions, 'model': model, 'scaler': scaler} if model is None else predictions
    except Exception as e:
        print(f"Error in predict_lstm: {e}")
        return [np.nan] * forecast_days

def train_lstm(data, seq_length=60, epochs=10):
    try:
        scaler = MinMaxScaler()
        if len(data) < 2 or not isinstance(data, (np.ndarray, list)):
            print("Not enough data or invalid data type for training LSTM")
            return None, None
        data = np.array(data, dtype=float)
        scaled_data = scaler.fit_transform(data.reshape(-1, 1))

        X, y = [], []
        for i in range(len(data) - seq_length - 29):  # 30-day forecast
            X.append(scaled_data[i:i + seq_length])
            y.append(scaled_data[i + seq_length:i + seq_length + 30])  # 30 steps ahead
        if len(X) < 2:
            print("Not enough sequences for training LSTM")
            return None, None

        X, y = np.array(X), np.array(y)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dense(30)  # Output 30 days
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test), verbose=1)
        return model, scaler
    except Exception as e:
        print(f"Error in train_lstm: {e}")
        return None, None

def create_sequences(data, seq_length=60):
    try:
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=float)
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)
    except Exception as e:
        print(f"Error in create_sequences: {e}")
        return np.array([]), np.array([])