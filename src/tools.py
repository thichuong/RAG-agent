# src/tools.py
import re
from bs4 import BeautifulSoup
import yfinance as yf
import json
import os
import requests
from tavily import TavilyClient
from .config import TAVILY_API_KEY

def resolve_symbol(symbol):
    if not symbol: return None
    symbol = symbol.strip().upper()
    # Basic mappings
    # Load mapping from JSON (memoized or loaded at module level preferred, but here valid too)
    try:
        mapping_path = os.path.join(os.path.dirname(__file__), 'mapping_data.json')
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

    except Exception as e:
        return f"Error: {e}"

def crawl_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
            
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Truncate to avoid context window explosion (simulated 2000 chars)
        return text[:2000] + "..." if len(text) > 2000 else text
    except Exception as e:
        return f"Error crawling {url}: {e}"

def scrape_web_page(url, selector=None):
    """
    Scrape specific elements from a web page using CSS selectors.
    If selector is None, returns the title and meta description.
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        if not selector:
            # Default: Get Title and Description
            title = soup.title.string if soup.title else "No Title"
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            desc = meta_desc['content'] if meta_desc and 'content' in meta_desc.attrs else "No Description"
            return f"Title: {title}\nDescription: {desc}"
        
        # Scrape specific selector
        elements = soup.select(selector)
        if not elements:
            return f"No elements found for selector: {selector}"
        
        results = []
        for i, el in enumerate(elements[:5]): # Limit to 5 results
            text = el.get_text(" ", strip=True)
            results.append(f"Match {i+1}: {text[:500]}") # Truncate each match
            
        return "\n".join(results)

    except Exception as e:
        return f"Error scraping {url}: {e}"

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
            "description": "Get current stock price (e.g., AAPL, TSLA). Do NOT use for crypto.",
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
            "name": "get_crypto_price",
            "description": "Get current crypto price (e.g., BTC, ETH, SOL) using Binance API.",
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
            "name": "crawl_url",
            "description": "Crawl a specific URL to get its full text content. Use this after get_news returns links.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_web_page",
            "description": "Scrape data from a URL using a CSS selector (e.g., 'div.content' or 'table.prices'). Defaults to Title/Description if selector is empty.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "selector": {"type": "string", "description": "CSS Selector to target elements."}
                },
                "required": ["url"]
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
