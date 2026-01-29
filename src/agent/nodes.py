import json
import re
from typing import Dict, Any, List
from .state import AgentState
from .intent_analyzer import analyze_intent
from .planner import analyze_request
from .parser import parse_tool_calls
from .summarizer import summarize_text
from ..config import logger
from ..tools import (
    arithmetic_tool,
    get_stock_price,
    get_crypto_price,
    get_news,
    crawl_url,
    scrape_web_page,
    TOOLS_SCHEMA
)

try:
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
except ImportError:
    # Minimal fallback if not fully installed, but we expect it to be
    pass

class GraphNodes:
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

    def analyze_intent_node(self, state: AgentState) -> Dict:
        """Node to analyze user intent."""
        # Get the first user message (assuming it's the last one added before this starts or the first one)
        # In a persistent chat, we might want the latest user message.
        # Check if messages exist
        messages = state.get("messages", [])
        if not messages:
            return {}
        
        # Find latest user message
        user_query = ""
        for m in reversed(messages):
            if m["role"] == "user":
                user_query = m["content"]
                break
        
        if not user_query:
            return {}

        logger.info("🧠 Analyzing Intent...")
        intent_data = analyze_intent(self.llm, user_query)
        logger.info(f"🎯 Intent Detected: {intent_data}")
        
        return {"intent": intent_data}

    def planning_node(self, state: AgentState) -> Dict:
        """Node to create a plan."""
        messages = state.get("messages", [])
        user_query = ""
        for m in reversed(messages):
            if m["role"] == "user":
                user_query = m["content"]
                break
            
        if not user_query:
            return {}

        planning_hint = analyze_request(self.llm, user_query)
        if planning_hint:
            logger.info(f"💡 Injecting Plan: {planning_hint}")
            return {"plan": planning_hint}
        
        return {"plan": ""}

    def generate_node(self, state: AgentState) -> Dict:
        """Node to generate LLM response."""
        messages = state.get("messages", [])
        intent_data = state.get("intent", {})
        plan = state.get("plan", "")
        
        # Construct dynamic system prompt
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
- For general investment concepts, ALWAYS search the knowledge base first using <tool_call>{{"name": "query_knowledge_base", "arguments": {{ "query": "your question" }} }}</tool_call>.
"""
        
        # Build prompt messages
        prompt_messages = [{"role": "system", "content": system_content}]
        
        # Inject Plan if available
        if plan:
             prompt_messages.append({"role": "system", "content": plan})

        # Add history
        # Simplify: just append all valid history messages
        # Note: state['messages'] accumulates everything. 
        # We need to filter/format for LlamaCPP if needed, or just pass them if format matches.
        # The state['messages'] are defined as List[Dict[str, str]].
        prompt_messages.extend(messages)

        logger.info(f"Step {state.get('step_count', 0) + 1}: Generating response...")
        
        response = self.llm.create_chat_completion(
            messages=prompt_messages,
            max_tokens=512,
            temperature=0.1
        )
        response_text = response["choices"][0]["message"]["content"]
        logger.info(f"Agent Raw Output: {response_text[:100]}...")

        # Parse tools
        tool_calls = parse_tool_calls(response_text)
        
        # Update logs
        new_logs = []
        # Not logging raw output to user logs, only final answer or tool calls
        
        return {
            "messages": [{"role": "assistant", "content": response_text}],
            "tool_calls": tool_calls,
            "step_count": state.get("step_count", 0) + 1
        }
    
    def execute_tools_node(self, state: AgentState) -> Dict:
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

    def synthesis_node(self, state: AgentState) -> Dict:
        """Node to synthesize final answer."""
        messages = state.get("messages", [])
        intent_data = state.get("intent", {})
        
        # Find original query
        user_query = ""
        for m in messages:
            if m["role"] == "user":
                user_query = m["content"]
                # Keep finding the last user query? Or first?
                # Usually the main query is the start of this session.
        
        logger.info("Tools were executed. Triggering Final Synthesis Step...")
        new_logs = ["🧠 **Final Synthesis**: Generating consolidated answer based on tool outputs..."]
        
        synthesis_prompt = f"""SYSTEM: PREPARE FINAL ANSWER.
1. Review the User Query: "{user_query}"
2. Review ALL Tool Outputs above (especially Summaries of crawled pages).
3. Synthesize a COMPREHENSIVE final response.
4. CONSTRAINT: Use ONLY the information present in the tool outputs. Do not make up facts. verification_step=TRUE
5. CITATIONS: You MUST include citations [Source: Name](URL) for all factual claims as per protocol.

CONTEXT FROM INTENT ANALYSIS:
- Goal: {intent_data.get('goal', 'Answer the question')}
- Required Language: {intent_data.get('language', 'Vietnamese')}

Ensure the final answer addresses the Goal and is written in {intent_data.get('language', 'Vietnamese')}.
"""
        # Construct prompt
        prompt_messages = list(messages) # Copy
        prompt_messages.append({"role": "system", "content": synthesis_prompt})
        
        response = self.llm.create_chat_completion(
            messages=prompt_messages,
            max_tokens=612,
            temperature=0.1
        )
        final_answer_text = response["choices"][0]["message"]["content"]
        
        # Cleanup
        final_answer = re.sub(r"<tool_call>.*?</tool_call>", "", final_answer_text, flags=re.DOTALL).strip()
        final_answer = final_answer.replace("<|im_end|>", "")
        
        new_logs.append(f"🤖 **Final Answer**: {final_answer}")
        
        return {
            "final_answer": final_answer,
            "logs": new_logs
        }
