import sys
import os
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, Optional, List
from alpaca.trading.client import TradingClient

# Local imports
from src.data.fetcher import DataFetcher
from src.data.preprocessor import DataPreprocessor
from src.features.technical import TechnicalFeatures
from src.models.lstm_model import LSTMModel
from src.models.drift_detection import MarketRegimeDetector, ConceptDriftDetector
from src.trading.execution import TWAPExecutor
from src.trading.risk import RiskManager
from src.compliance.audit import ComplianceLogger

# Load environment variables
load_dotenv('config/.env')

class TradingBot:
    def __init__(self):
        # Dashboard-required attributes
        self.start_time = datetime.now()
        self.order_history = []
        self.market_data = pd.DataFrame()
        
        # Trading components
        self.fetcher = DataFetcher()
        self.symbols = self._get_initial_symbols()
        self.primary_symbol = self.symbols[0]
        self.resolution = '15min'

        # Data processing
        self.preprocessor = DataPreprocessor(
            lookback=60,
            primary_symbol=self.primary_symbol
        )
        self.technical = TechnicalFeatures()

        # Risk management
        self.risk = RiskManager(capital=float(os.getenv('INITIAL_CAPITAL', 10000)))
        self.compliance = ComplianceLogger()
        #Models
        self.regime_detector = MarketRegimeDetector(
            window_size=60,
            threshold=3.5,
            cooldown_hours=12
        )
        self.model = LSTMModel(input_shape=(60, 4))
        self.drift_detector = self._safe_init_drift_detector()

        # Trading infrastructure
        self.trading_client = TradingClient(
            os.getenv('APCA_API_KEY_ID'),
            os.getenv('APCA_API_SECRET_KEY'),
            paper=os.getenv('TRADE_MODE', 'PAPER') == 'PAPER'
        )
        self.executor = TWAPExecutor(self.trading_client)

        # Performance tracking
        self.model_performance = {
            'predictions': [],
            'actuals': [],
            'accuracy_history': []
        }

    def _get_initial_symbols(self) -> List[str]:
        return ['SPY', 'QQQ']

    def _safe_init_drift_detector(self) -> Optional[ConceptDriftDetector]:
        try:
            baseline_data = self._load_baseline_data()
            return ConceptDriftDetector(baseline_data)
        except Exception as e:
            print(f"Drift detector initialization failed: {str(e)}")
            return None

    async def run(self):
        print("Starting trading bot...")
        while True:
            try:
                await self.trading_cycle()
                await asyncio.sleep(60)
            except Exception as e:
                print(f"Critical error: {str(e)}")
                await asyncio.sleep(300)

    async def trading_cycle(self):
        try:
            # 1. Data Collection
            raw_data = await self.fetcher.fetch_market_data(self.symbols, self.resolution)
            self.market_data = raw_data if isinstance(raw_data, pd.DataFrame) else pd.DataFrame()
            
            if len(self.market_data) < 100:
                print(f"Collecting initial data ({len(self.market_data)}/100 samples)")
                return

            # 2. Feature Engineering
            processed_data = self._process_multi_symbol_data()
            
            if await self._check_market_regime(processed_data):
                return
                
            # 4. Concept Drift Detection
            if self._check_concept_drift(processed_data):
                self._retrain_models(processed_data)
            
            # 5. Make Prediction
            prediction = self._make_prediction(processed_data)
            if prediction is None:
                return
            
            # 6. Risk Management
            position_size = self._calculate_position_size(processed_data, prediction)
            
            # 7. Execute Strategy
            await self._execute_trade_strategy(position_size)

        except Exception as e:
            print(f"Trading cycle error: {str(e)}")

    def _process_multi_symbol_data(self) -> pd.DataFrame:
        processed = pd.DataFrame()
        for symbol in self.symbols:
            symbol_cols = [col for col in self.market_data.columns 
                          if col.startswith(f"{symbol}_")]
            if not symbol_cols:
                continue
                
            symbol_data = self.market_data[symbol_cols].copy()
            base_cols = [col.split('_', 1)[1] for col in symbol_data.columns]
            symbol_data.columns = base_cols
            
            symbol_processed = self.technical.calculate_indicators(symbol_data)
            symbol_processed.columns = [f"{symbol}_{col}" for col in symbol_processed.columns]
            processed = pd.concat([processed, symbol_processed], axis=1)
        
        return processed

    async def _check_market_regime(self, data: pd.DataFrame) -> bool:
        try:
            if f"{self.primary_symbol}_close" not in data.columns:
                return False
                
            price_series = data[f"{self.primary_symbol}_close"].dropna()
            
            if len(price_series) < 200:
                return False
                
            if self.regime_detector.detect_change(price_series):
                print("Market regime changed - liquidating positions")
                await self._liquidate_positions()
                return True
            return False
            
        except Exception as e:
            print(f"Regime check error: {str(e)}")
            return False

    def _check_concept_drift(self, data: pd.DataFrame) -> bool:
        if not self.drift_detector:
            return False
        try:
            result = self.drift_detector.check_drift(data.values)
            return result['data']['is_drift'] if result else False
        except Exception as e:
            print(f"Concept drift check error: {str(e)}")
            return False

    def _make_prediction(self, data: pd.DataFrame) -> Optional[float]:
        try:
            X, _ = self.preprocessor.prepare_data(data)
            if len(X) == 0:
                return None
                
            prediction = self.model.predict(X[-1].reshape(1, 60, 4))[0][0]
            self._update_performance_tracking(
                prediction, 
                data[f"{self.primary_symbol}_close"].iloc[-1]
            )
            return prediction
        except Exception as e:
            print(f"Prediction failed: {str(e)}")
            return None

    def _update_performance_tracking(self, prediction: float, actual: float):
        self.model_performance['predictions'].append(prediction)
        self.model_performance['actuals'].append(actual)
        
        if len(self.model_performance['predictions']) >= 10:
            try:
                errors = np.abs(
                    np.array(self.model_performance['predictions']) - 
                    np.array(self.model_performance['actuals'])
                )
                current_accuracy = 1 - np.mean(
                    errors / np.array(self.model_performance['actuals'])
                )
                self.model_performance['accuracy_history'].append(current_accuracy)
            except Exception as e:
                print(f"Performance update error: {str(e)}")

    def _calculate_position_size(self, data: pd.DataFrame, prediction: float) -> float:
        try:
            current_price = data[f"{self.primary_symbol}_close"].iloc[-1]
            atr = data[f"{self.primary_symbol}_atr"].iloc[-1]
            stop_loss = self.risk.volatility_adjusted_stop(current_price, atr)
            return self.risk.calculate_position(current_price, stop_loss)
        except KeyError as e:
            print(f"Missing data for position calculation: {str(e)}")
            return 0.0

    async def _execute_trade_strategy(self, position_size: float):
        if position_size <= 0:
            return

        try:
            current_price = await self._get_current_price()
            if current_price <= 0:
                raise ValueError("Invalid current price")
            
            await self.executor.execute_twap(
                symbol=self.primary_symbol,
                qty=position_size,
                side='long',
                minutes=5
            )
            
            # Update portfolio tracking
            self.risk.update_portfolio(
                symbol=self.primary_symbol,
                action='buy',
                qty=position_size,
                price=current_price
            )
            
            self._log_trade(self.primary_symbol, 'buy', position_size, current_price)
            
            self.order_history.append({
                'timestamp': datetime.now(),
                'symbol': self.primary_symbol,
                'side': 'long',
                'qty': position_size,
                'price': current_price,
                'status': 'executed'
            })
                
        except Exception as e:
            print(f"Trade execution failed: {str(e)}")
            self.order_history.append({
                'timestamp': datetime.now(),
                'error': str(e),
                'status': 'failed'
            })

    async def _get_current_price(self) -> float:
        try:
            data = await self.fetcher.fetch_market_data([self.primary_symbol], '1min')
            return data[f"{self.primary_symbol}_close"].iloc[-1]
        except Exception as e:
            print(f"Price check failed: {str(e)}")
            return 0.0

    def _log_trade(self, symbol: str, action: str, qty: float, price: float):
        self.compliance.log_trade({
            'symbol': symbol,
            'action': action,
            'qty': qty,
            'price': price,
            'strategy': 'LSTM+Sentiment'
        })

    async def _liquidate_positions(self):
        try:
            positions = self.trading_client.get_all_positions()
            for p in positions:
                symbol = p.symbol
                qty = float(p.qty)
                
                try:
                    current_price = float(p.market_value) / qty if qty != 0 else 0.0
                except:
                    current_price = await self._get_current_price_for_symbol(symbol)
                
                await self.executor.execute_twap(
                    symbol=symbol,
                    qty=qty,
                    side='sell',
                    minutes=2
                )
                
                # Update portfolio tracking
                self.risk.update_portfolio(
                    symbol=symbol,
                    action='sell',
                    qty=qty,
                    price=current_price
                )
                
                self._log_trade(symbol, 'sell', qty, current_price)
                
                self.order_history.append({
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'side': 'sell',
                    'qty': qty,
                    'price': current_price,
                    'status': 'executed'
                })
        except Exception as e:
            print(f"Position liquidation failed: {str(e)}")

    async def _get_current_price_for_symbol(self, symbol: str) -> float:
        try:
            data = await self.fetcher.fetch_market_data([symbol], '1min')
            return data[f"{symbol}_close"].iloc[-1]
        except Exception as e:
            print(f"Price check failed for {symbol}: {str(e)}")
            return 0.0

    def _retrain_models(self, new_data: pd.DataFrame):
        try:
            X, y = self.preprocessor.prepare_data(new_data)
            self.model.train(X, y)
            if self.drift_detector:
                self.drift_detector.update_baseline(new_data)
            print("Models retrained successfully")
        except Exception as e:
            print(f"Model retraining failed: {str(e)}")

    def _load_baseline_data(self) -> np.ndarray:
        try:
            print("Loading baseline data...")
            data = self.fetcher.fetch_historical_data(self.primary_symbol)
            if data.empty:
                raise ValueError("Empty historical data")
            
            required_columns = [f"{self.primary_symbol}_close", f"{self.primary_symbol}_volume"]
            for col in required_columns:
                if col not in data.columns:
                    raise ValueError(f"Missing column: {col}")
            
            processed, _ = self.preprocessor.prepare_data(data)
            if processed.size == 0:
                raise ValueError("Empty processed data")
                
            print(f"Loaded baseline data with shape: {processed.shape}")
            return processed.reshape((processed.shape[0], -1))
        except Exception as e:
            print(f"Baseline data error: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        bot = TradingBot()
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"Initialization failed: {str(e)}")