from .math import arithmetic_tool
from .finance import get_stock_price, get_crypto_price, resolve_symbol
from .web import get_news, crawl_url, scrape_web_page

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

def get_tool_schemas() -> list:
    """Return the list of tool schemas."""
    return TOOLS_SCHEMA

def get_all_tool_names() -> list:
    """Return a list of all available tool names."""
    return [t["function"]["name"] for t in TOOLS_SCHEMA]
