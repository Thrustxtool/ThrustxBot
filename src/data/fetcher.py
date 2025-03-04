import os
import aiohttp
import asyncio
import yfinance as yf
import pandas as pd
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta

class DataFetcher:
    """
    Robust data fetcher with:
    - Polygon.io API v2 compliance
    - Yahoo Finance fallback
    - Symbol prefixing
    - Rate limiting
    - Caching
    """
    
    def __init__(self):
        self.cache_dir = 'data/live_cache/'
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_duration = timedelta(minutes=int(os.getenv('DATA_CACHE_MINUTES', 15)))
        self.last_request_time = datetime.min
        self.rate_limit = int(os.getenv('POLYGON_RATE_LIMIT', 5))

    async def fetch_market_data(self, symbols: List[str], resolution: str = '15min'):
        """Main data fetching method with cache validation"""
        cache_key = f"{'_'.join(symbols)}_{resolution}.pkl"
        cache_path = os.path.join(self.cache_dir, cache_key)
        
        try:
            if self._is_cache_valid(cache_path):
                return pd.read_pickle(cache_path)
        except Exception as e:
            print(f"Cache error: {str(e)}")

        await self._enforce_rate_limit()
        data = await self._fetch_concurrent(symbols, resolution)
        
        try:
            data.to_pickle(cache_path)
        except Exception as e:
            print(f"Cache write failed: {str(e)}")
        
        return data

    def _is_cache_valid(self, cache_path: str) -> bool:
        """Validate cache existence and freshness"""
        if not os.path.exists(cache_path):
            return False
        mod_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        return (datetime.now() - mod_time) < self.cache_duration

    async def _fetch_concurrent(self, symbols: List[str], resolution: str):
        """Orchestrate concurrent data fetching"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for symbol in symbols:
                tasks.append(self._fetch_symbol_data(session, symbol, resolution))
            
            results = await asyncio.gather(*tasks)
            return self._merge_symbol_data(results, symbols)

    async def _fetch_symbol_data(self, session, symbol: str, resolution: str):
        """Fetch data for single symbol with fallback"""
        # Try Polygon first
        multiplier, timespan = self._convert_resolution(resolution)
        if multiplier and timespan:
            try:
                data = await self._fetch_polygon(session, symbol, multiplier, timespan)
                if not data.empty:
                    return data
            except Exception as e:
                print(f"Polygon failed for {symbol}: {str(e)}")

        # Fallback to Yahoo
        return self._fetch_yfinance(symbol, resolution)

    def _merge_symbol_data(self, dataframes: List[pd.DataFrame], symbols: List[str]):
        """Merge symbol data with alignment checks"""
        merged = pd.concat(dataframes, axis=1)
        
        # Forward-fill missing values
        merged.ffill(inplace=True)
        
        # Validate all symbols present
        for symbol in symbols:
            if not any(col.startswith(symbol) for col in merged.columns):
                print(f"Warning: Missing data for {symbol}")
                
        return merged

    async def _fetch_polygon(self, session: aiohttp.ClientSession, 
                           symbol: str, multiplier: int, timespan: str):
        """Polygon.io API v2 implementation with proper parameters"""
        base_url = "https://api.polygon.io/v2/aggs/ticker"
        
        # Calculate date range
        to_date = datetime.utcnow()
        from_date = to_date - timedelta(days=30)
        
        url = f"{base_url}/{symbol}/range/{multiplier}/{timespan}/{from_date:%Y-%m-%d}/{to_date:%Y-%m-%d}"
        
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": "5000",
            "apiKey": os.getenv("POLYGON_API_KEY")
        }

        try:
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    error_data = await response.json()
                    error_msg = error_data.get('error', 'Unknown error')
                    print(f"Polygon API Error ({response.status}): {error_msg}")
                    return pd.DataFrame()
                
                data = await response.json()
                return self._parse_polygon_data(data, symbol)
        except Exception as e:
            print(f"Polygon fetch failed: {str(e)}")
            return pd.DataFrame()

    def _convert_resolution(self, res: str) -> Tuple[int, str]:
        """Convert resolution to (multiplier, timespan) tuple"""
        conversion = {
            '1min': (1, 'minute'),
            '5min': (5, 'minute'),
            '15min': (15, 'minute'),
            '1h': (1, 'hour'),
            '1d': (1, 'day')
        }
        return conversion.get(res, (None, None))

    def _parse_polygon_data(self, data: Dict[str, Any], symbol: str):
        """Parse Polygon response with proper column prefixing"""
        try:
            if data.get('resultsCount', 0) == 0:
                return pd.DataFrame()
            
            df = pd.DataFrame(data['results'])
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms', errors='coerce')
            df.set_index('timestamp', inplace=True)
            
            # Map Polygon fields to standardized columns
            column_map = {
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                'vw': 'vwap',
                'n': 'transactions'
            }
            
            df = df[list(column_map.keys())].rename(columns=column_map)
            df.columns = [f"{symbol}_{col}" for col in df.columns]
            
            return df
        except KeyError as e:
            print(f"Missing Polygon data key: {str(e)}")
            return pd.DataFrame()

    def _fetch_yfinance(self, symbol: str, interval: str):
        """Yahoo Finance fallback with column prefixing"""
        try:
            interval_map = {
                '1min': '1m',
                '5min': '5m',
                '15min': '15m',
                '1h': '60m',
                '1d': '1d'
            }
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                period='2d',
                interval=interval_map.get(interval, '15m'),
                prepost=True
            )
            # Standardize column names
            data.columns = [
                f"{symbol}_open", f"{symbol}_high",
                f"{symbol}_low", f"{symbol}_close",
                f"{symbol}_volume", f"{symbol}_dividends",
                f"{symbol}_stock_splits"
            ]
            return data
        except Exception as e:
            print(f"Yahoo Finance failed: {str(e)}")
            return pd.DataFrame()

    async def _enforce_rate_limit(self):
        """Polygon.io rate limiting (5 requests/minute default)"""
        elapsed = datetime.now() - self.last_request_time
        if elapsed < timedelta(seconds=60/self.rate_limit):
            wait_time = (60/self.rate_limit) - elapsed.total_seconds()
            await asyncio.sleep(wait_time)
        self.last_request_time = datetime.now()

    def fetch_historical_data(self, symbol: str, period: str = '2y', interval: str = '1d') -> pd.DataFrame:
        """Historical data fetcher with dynamic column handling"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                period=period,
                interval=interval,
                auto_adjust=True
            )
            
            # Map Yahoo columns to our naming convention
            column_mapping = {
                'Open': f'{symbol}_open',
                'High': f'{symbol}_high',
                'Low': f'{symbol}_low',
                'Close': f'{symbol}_close',
                'Volume': f'{symbol}_volume',
                'Dividends': f'{symbol}_dividends',
                'Stock Splits': f'{symbol}_stock_splits'
            }
            
            # Filter and rename existing columns
            data = data.rename(columns=column_mapping)
            
            # Add missing columns with default values
            expected_columns = [v for v in column_mapping.values()]
            for col in expected_columns:
                if col not in data.columns:
                    data[col] = 0.0  # Initialize missing columns
                    
            return data[expected_columns]  # Maintain consistent column order
        
        except Exception as e:
            print(f"Historical data error: {str(e)}")
            return pd.DataFrame()