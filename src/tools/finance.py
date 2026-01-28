import re
import yfinance as yf
import json
import os
import requests

def resolve_symbol(symbol):
    if not symbol: return None
    symbol = symbol.strip().upper()
    # Basic mappings
    # Load mapping from JSON (relative to this file's parent directory if needed, or same dir)
    # Assuming mapping_data.json is in src/ (parent of src/tools/)
    try:
        # Check src/tools/mapping_data.json first, then src/mapping_data.json
        current_dir = os.path.dirname(__file__)
        mapping_path = os.path.join(current_dir, 'mapping_data.json')
        
        if not os.path.exists(mapping_path):
             mapping_path = os.path.join(current_dir, '..', 'mapping_data.json')

        with open(mapping_path, 'r') as f:
            MAPPING = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load mapping_data.json: {e}")
        # Fallback to minimal if file missing
        MAPPING = {
            "BITCOIN": "BTC-USD", "BTC": "BTC-USD",
            "ETHEREUM": "ETH-USD", "ETH": "ETH-USD",
            "NVIDIA": "NVDA", "GOOGLE": "GOOGL", "APPLE": "AAPL",
            "AMAZON": "AMZN", "MICROSOFT": "MSFT", "TESLA": "TSLA"
        }
    # Check mapping
    if symbol in MAPPING: return MAPPING[symbol]
    for k, v in MAPPING.items():
        if k in symbol: return v
    # If it looks like a ticker (3-5 chars), use it
    if re.match(r'^[A-Z]{1,5}$', symbol):
        return symbol
    # Fallback: Try with yfinance search (mocked here for speed, or basic heuristics)
    return symbol

def get_stock_price(symbol):
    resolved = resolve_symbol(symbol)
    try:
        ticker = yf.Ticker(resolved)
        # fast_info is often faster/more reliable than history for current price
        price = ticker.fast_info.last_price
        if price:
            return price
        # Fallback to history
        hist = ticker.history(period="1d")
        if not hist.empty:
            return hist['Close'].iloc[-1]
        return f"No price found for {resolved}"
    except Exception as e:
        return f"Error fetching price for {symbol}: {e}"

def get_crypto_price(symbol):
    symbol = symbol.strip().upper()
    # Map common names to Binance symbols if needed, or assume format like "BTCUSDT"
    # Basic mapping: "BTC" -> "BTCUSDT", "ETH" -> "ETHUSDT"
    if not symbol.endswith("USDT") and not symbol.endswith("BUSD"):
        # If it's just "BTC", append "USDT"
        symbol += "USDT"
    
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    try:
        response = requests.get(url)
        data = response.json()
        if "price" in data:
            return float(data["price"])
        return f"Error: {data.get('msg', 'Unknown error')}"
    except Exception as e:
        return f"Error fetching crypto price for {symbol}: {e}"
