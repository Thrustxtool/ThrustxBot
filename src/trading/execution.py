import asyncio
import time
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from typing import Dict,List, Any

class TWAPExecutor:
    def __init__(self, api_client: TradingClient):
        """
        Initialize TWAP execution system
        :param api_client: Alpaca trading client
        """
        self.api = api_client
        self.order_history = []
    
    async def execute_twap(self, symbol: str, qty: float, 
                          side: str, minutes: int = 5) -> Dict[str, Any]:
        """
        Execute TWAP order
        :param symbol: Ticker symbol
        :param qty: Total quantity
        :param side: 'buy' or 'sell'
        :param minutes: Execution period
        :return: Execution summary
        """
        chunks = self._calculate_chunks(qty, minutes)
        results = []
        
        for chunk in chunks:
            order = MarketOrderRequest(
                symbol=symbol,
                qty=chunk,
                side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            
            # Execute chunk
            try:
                result = self.api.submit_order(order)
                results.append(result)
                self.order_history.append({
                    'timestamp': time.time(),
                    'symbol': symbol,
                    'qty': chunk,
                    'side': side,
                    'price': result.filled_avg_price
                })
            except Exception as e:
                print(f"Order failed: {str(e)}")
            
            # Wait between chunks
            await asyncio.sleep((60 * minutes) / len(chunks))
        
        return self._summarize_execution(results)
    
    def _calculate_chunks(self, total_qty: float, minutes: int) -> List[float]:
        """
        Calculate order chunks for TWAP
        :param total_qty: Total quantity
        :param minutes: Execution period
        :return: List of order quantities
        """
        chunks = []
        remaining = total_qty
        while remaining > 0:
            chunk = min(remaining, total_qty / (minutes * 2))
            chunks.append(chunk)
            remaining -= chunk
        return chunks
    
    def _summarize_execution(self, results: List[Any]) -> Dict[str, Any]:
        """
        Summarize TWAP execution results
        :param results: List of order results
        :return: Execution summary
        """
        filled_qty = sum(r.filled_qty for r in results)
        avg_price = sum(r.filled_qty * float(r.filled_avg_price) 
                    for r in results) / filled_qty
        return {
            'total_qty': filled_qty,
            'avg_price': avg_price,
            'num_orders': len(results),
            'success_rate': len(results) / (len(results) + len(self.order_history) - len(results))
        }
    
    async def liquidate_positions(self):
        """Liquidate all positions using TWAP"""
        positions = self.api.get_all_positions()
        for p in positions:
            await self.execute_twap(
                symbol=p.symbol,
                qty=float(p.qty),
                side='sell',
                minutes=2
            )
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get recent execution history"""
        return self.order_history[-100:]  # Last 100 orders