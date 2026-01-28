import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
# Adjust import to point to parent config
try:
    from ..config import TAVILY_API_KEY
except ImportError:
    # Fallback if executed differently (e.g. direct file run)
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.config import TAVILY_API_KEY

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
