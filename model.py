import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from data import fetch_stock_data
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_model(sequence_length, forecast_days=30):
    """Create an LSTM model for time series prediction"""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(sequence_length, 1)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(25, activation='relu'),
        tf.keras.layers.Dense(1)  # Predict one step at a time
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def prepare_sequences(data, sequence_length, forecast_days):
    """
    Prepare sequences for training
    
    Args:
        data: Input data array of shape (n_samples, 1)
        sequence_length: Number of time steps to use as input
        forecast_days: Number of days to predict
    
    Returns:
        X: Input sequences of shape (n_samples, sequence_length, 1)
        y: Target values of shape (n_samples, 1)
    """
    if len(data) < sequence_length + forecast_days:
        raise ValueError(f"Not enough data points. Need at least {sequence_length + forecast_days}, got {len(data)}")
    
    X, y = [], []
    for i in range(len(data) - sequence_length - forecast_days + 1):
        X.append(data[i:(i + sequence_length)])
        # For training, we only need the next immediate value
        y.append(data[i + sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Failed to create sequences")
        
    # Ensure correct shapes
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = y.reshape((y.shape[0], 1))
    
    return X, y

def calculate_confidence(predictions, actual, scale_min, scale_max):
    """
    Calculate model confidence based on prediction error and data scale
    
    Args:
        predictions: Model predictions
        actual: Actual values
        scale_min: Minimum value in original data
        scale_max: Maximum value in original data
    
    Returns:
        float: Confidence score between 0 and 1
    """
    if len(predictions) != len(actual):
        raise ValueError("Predictions and actual values must have same length")
        
    mse = np.mean((predictions - actual) ** 2)
    scale_range = max(1e-8, scale_max - scale_min)  # Avoid division by zero
    normalized_mse = mse / (scale_range ** 2)
    confidence = max(0.0, min(1.0, 1.0 - (normalized_mse * 10)))
    return confidence

def predict_lstm(data, forecast_days=30, sequence_length=60, model=None, scaler=None):
    """
    Predict stock prices using LSTM model
    
    Args:
        data: numpy array of closing prices
        forecast_days: number of days to forecast
        sequence_length: number of previous days to use for prediction
        model: pre-trained LSTM model (optional)
        scaler: fitted MinMaxScaler (optional)
    
    Returns:
        dict containing predictions, confidence score, and optionally the model and scaler
    """
    try:
        # Input validation
        data = np.asarray(data, dtype=np.float32)
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
            
        if len(data) < sequence_length + forecast_days:
            raise ValueError(f"Not enough data points. Need at least {sequence_length + forecast_days}, got {len(data)}")

        # Prepare data
        if scaler is None:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)
        else:
            scaled_data = scaler.transform(data)

        # Prepare sequences for training
        X, y = prepare_sequences(scaled_data, sequence_length, forecast_days)
        
        # Split data for training and validation
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # Create or use existing model
        if model is None:
            logger.info("Creating new LSTM model...")
            model = create_model(sequence_length, forecast_days)
            
            # Train the model with early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            logger.info("Training model...")
            history = model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=0
            )
            logger.info("Model training completed")

        # Calculate confidence using validation set
        val_predictions = model.predict(X_val, verbose=0)
        confidence = calculate_confidence(
            val_predictions.flatten(), 
            y_val.flatten(),
            np.min(data),
            np.max(data)
        )

        # Generate predictions iteratively for the forecast period
        predictions = []
        last_sequence = scaled_data[-sequence_length:].copy()
        
        for _ in range(forecast_days):
            # Reshape sequence for prediction
            current_sequence = last_sequence[-sequence_length:].reshape(1, sequence_length, 1)
            # Predict next value
            next_pred = model.predict(current_sequence, verbose=0)[0, 0]
            predictions.append(next_pred)
            # Update sequence with prediction
            last_sequence = np.vstack([last_sequence, next_pred])

        # Convert predictions back to original scale
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions)

        logger.info(f"Generated predictions for next {forecast_days} days with confidence: {confidence:.2%}")
        
        return {
            'predictions': predictions.flatten(),
            'confidence': confidence,
            'model': model,
            'scaler': scaler
        }

    except Exception as e:
        logger.error(f"Error in predict_lstm: {str(e)}")
        raise

def train_lstm(data, forecast_days=30, sequence_length=60, epochs=100):
    """
    Train a new LSTM model
    
    Args:
        data: Input time series data
        forecast_days: Number of days to forecast
        sequence_length: Number of previous days to use for prediction
        epochs: Number of training epochs
    
    Returns:
        tuple: (trained model, fitted scaler) or (None, None) if training fails
    """
    try:
        # Input validation
        data = np.asarray(data, dtype=np.float32)
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
            
        if len(data) < sequence_length + forecast_days:
            raise ValueError(f"Not enough data points. Need at least {sequence_length + forecast_days}, got {len(data)}")
        
        # Prepare data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # Create sequences
        X, y = prepare_sequences(scaled_data, sequence_length, forecast_days)
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:]
        y_val = y[train_size:]
        
        # Create and train model
        logger.info("Creating and training new LSTM model...")
        model = create_model(sequence_length, forecast_days)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        logger.info("Model training completed successfully")
        return model, scaler
        
    except Exception as e:
        logger.error(f"Error in train_lstm: {str(e)}")
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