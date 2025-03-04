import sys
import pprint
print("Python Path:")
pprint.pprint(sys.path)

try:
    from alpaca.trading.client import TradingClient
    print("Alpaca module imported successfully!")
except ModuleNotFoundError as e:
    print(f"Error importing Alpaca: {e}")