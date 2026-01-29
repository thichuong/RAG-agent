from typing import Dict
from ..state import AgentState
from ..summarizer import summarize_text
from ...config import logger
from ...tools import (
    arithmetic_tool,
    get_stock_price,
    get_crypto_price,
    get_news,
    crawl_url,
    scrape_web_page
)

class ExecuteToolsNode:
    def __init__(self, llm, rag):
        self.llm = llm
        self.rag = rag
        self.tool_map = {
            "arithmetic_tool": arithmetic_tool,
            "get_stock_price": get_stock_price,
            "get_crypto_price": get_crypto_price,
            "get_news": get_news,
            "crawl_url": crawl_url,
            "scrape_web_page": scrape_web_page,
            "query_knowledge_base": self.rag.search
        }

    def __call__(self, state: AgentState) -> Dict:
        """Node to execute tools."""
        tool_calls = state.get("tool_calls", [])
        new_messages = []
        new_logs = []
        
        for call in tool_calls:
            func_name = call.get("name")
            args = call.get("arguments")
            
            logger.info(f"Calling Tool: {func_name} with {args}")
            new_logs.append(f"🛠️ **Tool Call**: `{func_name}` | Args: `{args}`")
            
            result = "Error: Tool not found."
            if func_name in self.tool_map:
                try:
                    result = self.tool_map[func_name](**args)
                except Exception as e:
                    result = f"Error executing {func_name}: {e}"
            
            # Special Handling for crawl_url to save context
            if func_name == "crawl_url" and isinstance(result, str) and not result.startswith("Error"):
                logger.info("Summarizing crawled content...")
                summary = summarize_text(self.llm, result)
                url = args.get('url', 'Unknown URL')
                final_tool_output = f"[Source: {url}]\nSummary:\n{summary}"
                new_logs.append(f"📝 **Summary Generated** for {url}")
            else:
                final_tool_output = str(result)

            # Truncate result for log
            short_str = str(final_tool_output)
            short_result = short_str[:200] + "..." if len(short_str) > 200 else short_str
            new_logs.append(f"📄 **Result**: {short_result}")
            
            new_messages.append({
                "role": "tool",
                "content": final_tool_output
            })
            
        return {
            "messages": new_messages,
            "logs": new_logs,
            "tool_calls": [] # Clear tool calls after execution
        }
