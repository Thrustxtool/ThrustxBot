import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from collections import defaultdict
import yfinance as yf
import time

class RiskManager:
    def __init__(self, capital: float = 10000):
        """
        Enhanced risk management system with position tracking
        :param capital: Initial trading capital in USD
        """
        self.initial_capital = capital
        self.portfolio = {
            'cash': capital,
            'positions': {},
            'total_value': capital
        }
        self.max_drawdown = 0.2  # 20% max drawdown
        self.position_size_limit = 0.1  # 10% max position size
        self.sector_limits = {
            'technology': 0.3,
            'financials': 0.25,
            'healthcare': 0.2,
            'other': 0.25
        }
    
    def calculate_position(self, price: float, stop_loss: float) -> float:
        """
        Calculate position size with volatility-adjusted risk
        :param price: Entry price per share
        :param stop_loss: Stop loss price
        :return: Number of shares to trade
        """
        try:
            risk_per_trade = self.portfolio['total_value'] * self.position_size_limit
            risk_amount = abs(price - stop_loss)
            
            if risk_amount <= 0:
                raise ValueError("Invalid risk calculation parameters")
                
            shares = risk_per_trade / risk_amount
            max_shares = (self.portfolio['cash'] * 0.5) / price
            return min(shares, max_shares)
            
        except Exception as e:
            print(f"Position calculation error: {str(e)}")
            return 0.0

    def volatility_adjusted_stop(self, price: float, atr: float) -> float:
        """
        Calculate dynamic stop loss based on volatility
        :param price: Entry price
        :param atr: Average True Range (volatility measure)
        :return: Stop loss price
        """
        return max(0.01, price - (2 * atr))  # Prevent negative stops

    def update_portfolio(self, symbol: str, action: str, qty: float, price: float):
        """
        Maintain accurate portfolio state with error checking
        :param symbol: Asset ticker
        :param action: 'buy' or 'sell'
        :param qty: Number of shares
        :param price: Execution price per share
        """
        try:
            value = qty * price
            
            if action == 'buy':
                if value > self.portfolio['cash']:
                    raise ValueError("Insufficient funds for purchase")
                    
                self.portfolio['positions'][symbol] = {
                    'quantity': qty,
                    'entry_price': price,
                    'current_price': price,
                    'value': value
                }
                self.portfolio['cash'] -= value
                
            elif action == 'sell':
                position = self.portfolio['positions'].get(symbol)
                if not position:
                    raise KeyError(f"No position found for {symbol}")
                    
                self.portfolio['cash'] += position['value']
                del self.portfolio['positions'][symbol]
                
            self._update_portfolio_value()
            
        except Exception as e:
            print(f"Portfolio update failed: {str(e)}")

    def _update_portfolio_value(self):
        """Recalculate total portfolio value"""
        positions_value = sum(
            p['value'] for p in self.portfolio['positions'].values()
        )
        self.portfolio['total_value'] = self.portfolio['cash'] + positions_value

    def check_drawdown(self) -> bool:
        """Monitor maximum admissible drawdown"""
        current_value = self.portfolio['total_value']
        peak_value = max(self.initial_capital, current_value)
        drawdown = (peak_value - current_value) / peak_value
        return drawdown > self.max_drawdown

    def sector_exposure(self) -> Dict[str, float]:
        """Calculate current sector allocations with retries"""
        sectors = defaultdict(float)
        
        for symbol, position in self.portfolio['positions'].items():
            sector = 'other'
            max_retries = 3
            
            for _ in range(max_retries):
                try:
                    sector = self.get_sector(symbol).lower()
                    break
                except Exception as e:
                    print(f"Sector lookup failed for {symbol}: {str(e)}")
                    time.sleep(1)
            
            sectors[sector] += position['value']
        
        total = sum(sectors.values()) + self.portfolio['cash']
        return {k: v / total for k, v in sectors.items()}

    def get_sector(self, symbol: str) -> str:
        """Robust sector lookup with error handling"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get('sector', 'other').lower() or 'other'
        except Exception as e:
            print(f"Sector lookup error for {symbol}: {str(e)}")
            return 'other'

    def portfolio_heat(self) -> float:
        """Calculate percentage of capital deployed"""
        return 1 - (self.portfolio['cash'] / self.portfolio['total_value'])

    def update_position_values(self, market_data: pd.DataFrame):
        """Mark positions to market with current prices"""
        try:
            for symbol, position in self.portfolio['positions'].items():
                current_price = market_data.get(f"{symbol}_close", np.nan)
                if not np.isnan(current_price):
                    position['current_price'] = current_price
                    position['value'] = position['quantity'] * current_price
            self._update_portfolio_value()
        except Exception as e:
            print(f"Position valuation error: {str(e)}")