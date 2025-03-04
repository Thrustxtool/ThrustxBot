import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict

class DataPreprocessor:
    def __init__(self, lookback: int = 60, primary_symbol: str = 'SPY'):
        self.lookback = lookback
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.primary_symbol = primary_symbol
        self.feature_columns = [
            f'{primary_symbol}_close',
            f'{primary_symbol}_volume',
            f'{primary_symbol}_rsi',
            f'{primary_symbol}_macd'
        ]

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        try:
            df = self._clean_data(df)
            scaled = self.scaler.fit_transform(df[self.feature_columns])
            
            X, y = [], []
            for i in range(self.lookback, len(scaled)):
                X.append(scaled[i-self.lookback:i])
                y.append(scaled[i, 0])
                
            return np.array(X), np.array(y)
        except Exception as e:
            print(f"Data preparation error: {str(e)}")
            return np.empty((0, self.lookback, 4)), np.empty(0)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        dummy = np.zeros((len(data), len(self.feature_columns)))
        dummy[:, 0] = data
        return self.scaler.inverse_transform(dummy)[:, 0]

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            close_col = f"{self.primary_symbol}_close"
            vol_col = f"{self.primary_symbol}_volume"
            
            rsi_col = f"{self.primary_symbol}_rsi"
            if rsi_col not in df.columns:
                df[rsi_col] = self._calculate_rsi(df[close_col])
                
            macd_col = f"{self.primary_symbol}_macd"
            if macd_col not in df.columns:
                df[macd_col] = self._calculate_macd(df[close_col])
                
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0.0
            return df.dropna()
        except Exception as e:
            print(f"Data cleaning failed: {str(e)}")
            return pd.DataFrame()

    def _calculate_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, series: pd.Series, 
                   fast: int = 12, slow: int = 26, 
                   signal: int = 9) -> pd.Series:
        fast_ema = series.ewm(span=fast, adjust=False).mean()
        slow_ema = series.ewm(span=slow, adjust=False).mean()
        macd = fast_ema - slow_ema
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line

    def get_feature_importance(self, model) -> Dict[str, float]:
        if hasattr(model, 'feature_importances_'):
            return {col: imp for col, imp in zip(self.feature_columns, model.feature_importances_)}
        return {col: 1.0/len(self.feature_columns) for col in self.feature_columns}