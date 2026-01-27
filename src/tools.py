# src/tools.py
import re
import yfinance as yf
from tavily import TavilyClient
from .config import TAVILY_API_KEY

def resolve_symbol(symbol):
    if not symbol: return None
    symbol = symbol.strip().upper()
    # Basic mappings
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

def arithmetic_tool(op, a, b):
    try:
        a, b = float(a), float(b)
        if op == 'add': return a + b
        if op == 'subtract': return a - b
        if op == 'multiply': return a * b
        if op == 'divide': return a / b if b != 0 else "Error: Div0"
    except: return "Error: Invalid numbers"
    return "Error: Unknown Op"

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

def get_news(query):
    if not TAVILY_API_KEY:
        return "Error: TAVILY_API_KEY not configured."
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(query, search_depth="basic", max_results=3)
        results = response.get('results', [])
        if not results: return "No news found."
        return "\n".join([f"- {r['title']} ({r['url']})" for r in results])
    except Exception as e:
        return f"Error: {e}"

# Tool Definitions for System Prompt
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "arithmetic_tool",
            "description": "Perform basic arithmetic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "op": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["op", "a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get current stock/crypto price.",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Search for latest news.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_knowledge_base",
            "description": "Search the internal investment knowledge base (RAG) for information about finance concepts, risk management, or portfolio theory.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The specific question to search for."}},
                "required": ["query"]
            }
        }
    }
]
