# src/agent.py
import json
import re
from typing import List, Dict
from llama_cpp import Llama  # For type hinting
from .rag import InvestmentRAG # For type hinting
from .tools import arithmetic_tool, get_stock_price, get_news, get_crypto_price, TOOLS_SCHEMA
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
            "query_knowledge_base": self.rag.search
        }

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

    def run(self, user_query: str, history: List[Dict] = []):
        # System prompt with explicit tool definitions
        tools_json = json.dumps(TOOLS_SCHEMA, indent=2)
        system_content = f"""You are a helpful assistant. You can use tools to answer questions.
If you need to use a tool, output the function call inside <tool_call> tags.
Do NOT output the result of the tool, just the proper XML tag.

Available Tools:
{tools_json}

For logical questions, break them down.
For investment questions, ALWAYS search the knowledge base first using <tool_call>{{"name": "query_knowledge_base", "arguments": {{"query": "your question"}} }}</tool_call>."""

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

        MAX_STEPS = 5
        steps_log = []

        for step in range(MAX_STEPS):
            logger.info(f"Step {step+1}: Generating response...")
            
            # Generate
            response = self.generate(messages)
            response_text = response["choices"][0]["message"]["content"]
            logger.info(f"Agent Raw Output: {response_text[:100]}...") # Log first 100 chars
            
            # Parse
            tool_calls = self.parse_tool_calls(response_text)
            
            if not tool_calls:
                # Final Answer
                final_answer = re.sub(r"<tool_call>.*?</tool_call>", "", response_text, flags=re.DOTALL).strip()
                final_answer = final_answer.replace("<|im_end|>", "")
                steps_log.append(f"🤖 **Answer**: {final_answer}")
                return final_answer, "\n\n".join(steps_log)

            # Execute tools
            messages.append({"role": "assistant", "content": response_text})
            
            for call in tool_calls:
                func_name = call.get("name")
                args = call.get("arguments")
                
                logger.info(f"Calling Tool: {func_name} with {args}")
                steps_log.append(f"🛠️ **Tool Call**: `{func_name}` | Args: `{args}`")
                
                if func_name in self.tool_map:
                    try:
                        result = self.tool_map[func_name](**args)
                    except Exception as e:
                        result = f"Error executing {func_name}: {e}"
                else:
                    result = f"Error: Tool {func_name} not found."
                
                # Truncate result for log
                short_result = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
                steps_log.append(f"📄 **Result**: {short_result}")
                
                # Append tool result to messages
                messages.append({
                    "role": "tool",
                    "content": str(result)
                })
        
        return "Max steps reached without final answer.", "\n\n".join(steps_log)
