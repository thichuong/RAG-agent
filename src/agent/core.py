import json
import re
from typing import List, Dict, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llama_cpp import Llama
else:
    Llama = object

# Absolute imports based on package structure "src.agent.core"
# But typically we run from root, so "src.rag" etc.
try:
    from ..rag import InvestmentRAG
    from ..tools import (
        arithmetic_tool, 
        get_stock_price, 
        get_news, 
        get_crypto_price, 
        crawl_url, 
        scrape_web_page, 
        TOOLS_SCHEMA
    )
    from ..config import logger
    from .planner import analyze_request
    from .parser import parse_tool_calls
    from .summarizer import summarize_text
except ImportError as e:
    # Fallback/Debug
    print(f"Import Error in core.py: {e}")
    raise e

class QwenAgent:
    def __init__(self, llm: Llama, rag: InvestmentRAG):
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

    def execute_tool(self, func_name, args):
        """Helper to execute a tool and format the result."""
        if func_name in self.tool_map:
            try:
                result = self.tool_map[func_name](**args)
                return result
            except Exception as e:
                return f"Error executing {func_name}: {e}"
        else:
            return f"Error: Tool {func_name} not found."

    def generate(self, messages: List[Dict]):
        """Generate response using LlamaCPP raw completion."""
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.1
        )
        return response

    def run(self, user_query: str, history: List[Dict] = []):
        # System prompt with explicit tool definitions
        tools_json = json.dumps(TOOLS_SCHEMA, indent=2)
        system_content = f"""You are a helpful and intelligent assistant. You can use tools to answer questions.
If you need to use a tool, output the function call inside <tool_call> tags.
Do NOT output the result of the tool, just the proper XML tag.

Available Tools:
{tools_json}

### 1. SEARCH & NEWS PROTOCOL:
- **Listicle Format**: When you search for news (using `get_news`), your *immediate* next response should summarize the findings in a structured list (3-5 items) with sources.
    - Example:
      1. **[Source Name]**: Headline or brief summary.
      2. **[Source Name]**: Headline...
- **Deep Dive**: If the user needs more detail, or the task implies a comprehensive report, do NOT rely on just the headlines.
    - **Step 1**: Select the top 2-3 most relevant URLs from the search results.
    - **Step 2**: specificially call `crawl_url` on these URLs.
    - **Step 3**: Synthesize the content from these crawled pages into a single cohesive answer.
    - **Constraint**: If you choose to crawl only *one* source, you MUST explicitly explain why (e.g., "This was the only relevant detailed analysis found...").

### 2. CITATION PROTOCOL (STRICT):
- **MANDATORY**: Your FINAL answer MUST include citations with links for any factual claims.
- Format: [Source: Source Name or Domain] or [Source: Title](URL). 
- Example: "The market crashed due to... [Source: Bloomberg]"

### 3. KNOWLEDGE BASE:
- For general investment concepts, ALWAYS search the knowledge base first using <tool_call>{{"name": "query_knowledge_base", "arguments": {{"query": "your question"}} }}</tool_call>.
"""

        messages = [
            {"role": "system", "content": system_content}
        ]
        
        # Add history
        if history:
             for turn in history:
                 if isinstance(turn, (list, tuple)) and len(turn) == 2:
                     user_msg, bot_msg = turn
                     messages.append({"role": "user", "content": str(user_msg)})
                     messages.append({"role": "assistant", "content": str(bot_msg)})

        messages.append({"role": "user", "content": user_query})

        # --- PLANNING STEP ---
        planning_hint = analyze_request(self.llm, user_query)
        if planning_hint:
            logger.info(f"💡 Injecting Plan: {planning_hint}")
            messages.append({"role": "system", "content": planning_hint})

        MAX_STEPS = 5
        steps_log = []
        tools_executed = False

        for step in range(MAX_STEPS):
            logger.info(f"Step {step+1}: Generating response...")
            
            # Generate
            response = self.generate(messages)
            response_text = response["choices"][0]["message"]["content"]
            logger.info(f"Agent Raw Output: {response_text[:100]}...")
            
            # Parse
            tool_calls = parse_tool_calls(response_text)
            
            if not tool_calls:
                # No new tool calls generated.
                if tools_executed:
                    logger.info("Tools were executed. Triggering Final Synthesis Step...")
                    steps_log.append("🧠 **Final Synthesis**: Generating consolidated answer based on tool outputs...")
                    
                    synthesis_prompt = f"""SYSTEM: PREPARE FINAL ANSWER.
1. Review the User Query: "{user_query}"
2. Review ALL Tool Outputs above (especially Summaries of crawled pages).
3. Synthesize a COMPREHENSIVE final response.
4. CONSTRAINT: Use ONLY the information present in the tool outputs. Do not make up facts. verification_step=TRUE
5. CITATIONS: You MUST include citations [Source: Name](URL) for all factual claims as per protocol.
"""
                    messages.append({"role": "system", "content": synthesis_prompt})
                    
                    final_response = self.generate(messages)
                    final_answer_text = final_response["choices"][0]["message"]["content"]
                    
                    final_answer = re.sub(r"<tool_call>.*?</tool_call>", "", final_answer_text, flags=re.DOTALL).strip()
                    final_answer = final_answer.replace("<|im_end|>", "")
                    
                    steps_log.append(f"🤖 **Final Answer**: {final_answer}")
                    return final_answer, "\n\n".join(steps_log)
                
                else:
                    # Simple Chit-Chat or Direct Knowledge Answer (no tools used)
                    final_answer = re.sub(r"<tool_call>.*?</tool_call>", "", response_text, flags=re.DOTALL).strip()
                    final_answer = final_answer.replace("<|im_end|>", "")
                    steps_log.append(f"🤖 **Answer**: {final_answer}")
                    return final_answer, "\n\n".join(steps_log)

            # Execute tools
            tools_executed = True
            messages.append({"role": "assistant", "content": response_text})
            
            for call in tool_calls:
                func_name = call.get("name")
                args = call.get("arguments")
                
                logger.info(f"Calling Tool: {func_name} with {args}")
                steps_log.append(f"🛠️ **Tool Call**: `{func_name}` | Args: `{args}`")
                
                if func_name in self.tool_map:
                     result = self.execute_tool(func_name, args)
                else:
                    result = f"Error: Tool {func_name} not found."
                
                # Special Handling for crawl_url to save context
                if func_name == "crawl_url" and isinstance(result, str) and not result.startswith("Error"):
                    logger.info("Summarizing crawled content...")
                    summary = summarize_text(self.llm, result)
                    url = args.get('url', 'Unknown URL')
                    final_tool_output = f"[Source: {url}]\nSummary:\n{summary}"
                    steps_log.append(f"📝 **Summary Generated** for {url}")
                else:
                    final_tool_output = str(result)

                # Truncate result for log
                short_result = str(final_tool_output)[:200] + "..." if len(str(final_tool_output)) > 200 else str(final_tool_output)
                steps_log.append(f"📄 **Result**: {short_result}")
                
                # Append tool result to messages
                messages.append({
                    "role": "tool",
                    "content": final_tool_output
                })
        
        return "Max steps reached. See logs for partial progress.", "\n\n".join(steps_log)
