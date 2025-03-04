import numpy as np
import pandas as pd
import talib
from typing import Dict, Any

class TechnicalFeatures:
    def __init__(self):
        self.indicators = {
            'rsi': {'window': 14},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'atr': {'window': 14},
            'obv': {},
            'sma': {'window': 50},
            'ema': {'window': 20},
            'bollinger': {'window': 20, 'nbdevup': 2, 'nbdevdn': 2}
        }
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        # Price-based indicators
        df['rsi'] = talib.RSI(df['close'], timeperiod=self.indicators['rsi']['window'])
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
            df['close'],
            fastperiod=self.indicators['macd']['fast'],
            slowperiod=self.indicators['macd']['slow'],
            signalperiod=self.indicators['macd']['signal']
        )
        
        # Volume indicators
        df['obv'] = talib.OBV(df['close'], df['volume'])
        df['adl'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
        
        # Volatility
        df['atr'] = talib.ATR(
            df['high'], df['low'], df['close'],
            timeperiod=self.indicators['atr']['window']
        )
        
        # Moving Averages
        df['sma'] = talib.SMA(df['close'], timeperiod=self.indicators['sma']['window'])
        df['ema'] = talib.EMA(df['close'], timeperiod=self.indicators['ema']['window'])
        
        # Bollinger Bands
        df['upper_band'], df['middle_band'], df['lower_band'] = talib.BBANDS(
            df['close'],
            timeperiod=self.indicators['bollinger']['window'],
            nbdevup=self.indicators['bollinger']['nbdevup'],
            nbdevdn=self.indicators['bollinger']['nbdevdn']
        )
        
        # Additional Features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close']).diff()
        df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        return df.dropna()
    
    def get_indicator_config(self) -> Dict[str, Any]:
        """Get current indicator configuration"""
        return self.indicators
    
    def update_indicator_config(self, new_config: Dict[str, Any]):
        """Update indicator parameters"""
        self.indicators.update(new_config)
    
    def calculate_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional custom features"""
        # Price Momentum
        df['momentum'] = df['close'] / df['close'].shift(4) - 1
        
        # Volume Features
        df['volume_change'] = df['volume'].pct_change()
        df['volume_sma'] = talib.SMA(df['volume'], timeperiod=20)
        
        # Price-Volume Relationship
        df['pvr'] = df['returns'] * df['volume_change']
        
        return df.dropna()