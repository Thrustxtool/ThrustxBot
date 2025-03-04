import numpy as np
import pandas as pd
from changefinder import ChangeFinder
from alibi_detect.cd import MMDDrift
from typing import Dict, Any, Optional
from sklearn.decomposition import PCA
from datetime import datetime, timedelta

class MarketRegimeDetector:
    def __init__(self, window_size: int = 60, threshold: float = 3.0, 
                 cooldown_hours: int = 12, r: float = 0.02, order: int = 2, 
                 smooth: int = 14):
        self.cf = ChangeFinder(r=r, order=order, smooth=smooth)
        self.window_size = window_size
        self.threshold = threshold
        self.cooldown = timedelta(hours=cooldown_hours)
        self.last_change = datetime.min
        self.change_scores = []

    def detect_change(self, prices: pd.Series) -> bool:
        try:
            if datetime.now() - self.last_change < self.cooldown:
                return False

            if len(prices) < self.window_size * 2:
                return False

            new_scores = [self.cf.update(p) for p in prices.iloc[-self.window_size:]]
            self.change_scores = self.change_scores[-1000:] + new_scores
            
            recent_scores = np.array(self.change_scores[-self.window_size:])
            if len(self.change_scores) < 100:
                return False

            z_score = (recent_scores.mean() - np.mean(self.change_scores)) / np.std(self.change_scores)
            
            if z_score > self.threshold:
                self.last_change = datetime.now()
                return True
                
            return False
            
        except Exception as e:
            print(f"Regime detection error: {str(e)}")
            return False

    def get_change_scores(self) -> pd.DataFrame:
        return pd.DataFrame({
            'timestamp': pd.date_range(end=datetime.now(), periods=len(self.change_scores), freq='T'),
            'score': self.change_scores
        })

class ConceptDriftDetector:
    def __init__(self, baseline_data: np.ndarray, p_val: float = 0.01, 
                 batch_size: int = 256, min_samples: int = 1000):
        self.p_val = p_val
        self.batch_size = batch_size
        self.min_samples = min_samples
        self.detector = None
        self.pca = PCA(n_components=0.95, whiten=True)
        
        if baseline_data.shape[0] > 100:
            self._initialize_detector(baseline_data)

    def _initialize_detector(self, data: np.ndarray):
        try:
            data_pca = self.pca.fit_transform(data)
            self.detector = MMDDrift(
                data_pca, 
                p_val=self.p_val, 
                backend='pytorch',
                input_shape=(data_pca.shape[1],)
            )
        except Exception as e:
            print(f"Drift detector init failed: {str(e)}")

    def check_drift(self, new_data: np.ndarray) -> Optional[Dict[str, Any]]:
        try:
            if new_data.shape[0] < self.min_samples or self.detector is None:
                return None
                
            new_data_pca = self.pca.transform(new_data[-self.batch_size:])
            return self.detector.predict(new_data_pca)
            
        except Exception as e:
            print(f"Drift check error: {str(e)}")
            return None

    def update_baseline(self, new_baseline: np.ndarray):
        try:
            if new_baseline.shape[0] > 100:
                self._initialize_detector(new_baseline)
        except Exception as e:
            print(f"Baseline update failed: {str(e)}")