import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
from typing import Tuple, Dict, Any

class LSTMModel:
    def __init__(self, input_shape: Tuple[int, int]):
        """
        Initialize LSTM model
        :param input_shape: (timesteps, features)
        """
        self.model = self._build_model(input_shape)
        self.scaler = None  # Will be set during training
        self.input_shape = input_shape
        
    def _build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Construct LSTM architecture"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae']
        )
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              epochs: int = 20, batch_size: int = 32) -> Dict[str, Any]:
        """
        Train the LSTM model
        :return: Training history
        """
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = self.model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        return {
            'mse': mean_squared_error(y_test, predictions),
            'mae': np.mean(np.abs(y_test - predictions))
        }
    
    def save(self, path: str):
        """Save model and scaler"""
        self.model.save(path)
        if self.scaler:
            import joblib
            joblib.dump(self.scaler, f"{path}_scaler.pkl")
    
    def load(self, path: str):
        """Load model and scaler"""
        self.model = tf.keras.models.load_model(path)
        try:
            import joblib
            self.scaler = joblib.load(f"{path}_scaler.pkl")
        except:
            self.scaler = None
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Estimate feature importance using permutation"""
        if not hasattr(self, 'X_test'):
            return {}
        
        baseline_score = self.evaluate(self.X_test, self.y_test)['mse']
        importance = {}
        for i in range(self.input_shape[1]):
            X_temp = self.X_test.copy()
            np.random.shuffle(X_temp[:, :, i])
            score = self.evaluate(X_temp, self.y_test)['mse']
            importance[f"feature_{i}"] = baseline_score - score
            
        return importance