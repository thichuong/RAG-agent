# src/agent.py
import json
import re
from typing import List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from llama_cpp import Llama
else:
    # Create a dummy class for runtime checks if needed, or just set to Any
    Llama = object
from .rag import InvestmentRAG # For type hinting
from .tools import arithmetic_tool, get_stock_price, get_news, get_crypto_price, crawl_url, scrape_web_page, TOOLS_SCHEMA
from .config import logger

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

    def parse_tool_calls(self, text):
        """
        Parse tool calls from Qwen instructions.
        """
        calls = []
        text = str(text)
        
        # 1. Try <tool_call> XML-like tags
        tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
        matches = re.findall(tool_call_pattern, text, re.DOTALL)
        
        for m in matches:
            try:
                call_data = json.loads(m.strip())
                calls.append(call_data)
            except Exception as e:
                logger.warning(f"Failed to parse tool call JSON: {e} | Content: {m}")

        if calls:
            return calls

        # 2. Fallback: Detect standard tool structure
        try:
            json_pattern = r"\{.*?\}"
            potential_jsons = re.findall(json_pattern, text, re.DOTALL)
            for pj in potential_jsons:
                try:
                    data = json.loads(pj)
                    if "name" in data and "arguments" in data:
                        calls.append(data)
                except:
                    continue
        except:
            pass
        
        return calls

    def generate(self, messages: List[Dict]):
        """Generate response using LlamaCPP raw completion."""
        # Convert messages to Qwen prompt format manually or use create_chat_completion without tools
        # Using create_chat_completion without 'tools' arg is safest for GGUF chat template
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.1
        )
        return response

    def analyze_request(self, query: str) -> str:
        """
        Analyze the user query to identify information gaps.
        Returns a planning hint or empty string.
        """
        analysis_prompt = f"""You are a paranoid Query Analyzer. Assume you know NOTHING about the world after 2023.
Your goal is to break down the user query into specific search needs.

User Query: "{query}"

Instructions:
1. Identify specific keywords or named entities.
2. If multiple topics are involved (e.g., comparisons), break them down into separate lines.
3. Output "NEED_SEARCH: [Topic]" for EACH distinct topic needing information.
4. Output "NO_SEARCH" only if the query is trivial chit-chat.

Examples:
Query: "Compare Apple and Tesla"
NEED_SEARCH: Apple stock news analysis
NEED_SEARCH: Tesla stock news analysis

Query: "Bitcoin price"
NEED_SEARCH: Bitcoin price today

Query: "Hello"
NO_SEARCH

Output format:
NEED_SEARCH: <topic 1>
NEED_SEARCH: <topic 2>
...
"""
        try:
            response = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": analysis_prompt}],
                max_tokens=100, # Increased for multi-line
                temperature=0.0
            )
            content = response["choices"][0]["message"]["content"].strip()
            logger.info(f"🔍 Analysis Result:\n{content}")
            
            search_needs = []
            for line in content.split('\n'):
                if "NEED_SEARCH:" in line:
                    topic = line.replace("NEED_SEARCH:", "").strip()
                    if topic:
                        search_needs.append(topic)
            
            if search_needs:
                topics_str = "; ".join(search_needs)
                return f"PLANNING_STEP: You previously analyzed this request and determined you lack internal knowledge. You MUST use tools to find information for the following topics: [{topics_str}]. Do NOT answer 'I don't know' without trying tools for EACH topic first."
            
            return ""
        except Exception as e:
            logger.warning(f"Analysis failed: {e}")
            return ""


    def summarize_text(self, text: str) -> str:
        """
        Summarize crawled text to save context window.
        """
        if len(text) < 500:
            return text # Short enough, no need to summarize

        summary_prompt = f"""Summarize the following text into 3-4 distinct bullet points. Focus on facts, numbers, and key insights relevant to the topic.
        
Text:
{text[:3000]}
"""
        try:
            response = self.llm.create_chat_completion(
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=200,
                temperature=0.1
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
            return text[:500] + "... (Summarization failed)"

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
                 # Ensure history is strictly string
                 # Gradio history is often [user, bot] pairs
                 # If passing raw Gradio history:
                 if isinstance(turn, (list, tuple)) and len(turn) == 2:
                     user_msg, bot_msg = turn
                     messages.append({"role": "user", "content": str(user_msg)})
                     messages.append({"role": "assistant", "content": str(bot_msg)})

        messages.append({"role": "user", "content": user_query})

        # --- PLANNING STEP ---
        planning_hint = self.analyze_request(user_query)
        if planning_hint:
            logger.info(f"💡 Injecting Plan: {planning_hint}")
            # Inject as a system instruction (or user reinforcement) to guide the model
            messages.append({"role": "system", "content": planning_hint})

        MAX_STEPS = 5
        steps_log = []
        tools_executed = False

        for step in range(MAX_STEPS):
            logger.info(f"Step {step+1}: Generating response...")
            
            # Generate
            response = self.generate(messages)
            response_text = response["choices"][0]["message"]["content"]
            logger.info(f"Agent Raw Output: {response_text[:100]}...") # Log first 100 chars
            
            # Parse
            tool_calls = self.parse_tool_calls(response_text)
            
            if not tool_calls:
                # No new tool calls generated.
                # Check if we should synthesize a final answer based on previous tool outputs.
                
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
                    summary = self.summarize_text(result)
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
        
        # Fallback if max steps reached
        return "Max steps reached. See logs for partial progress.", "\n\n".join(steps_log)
