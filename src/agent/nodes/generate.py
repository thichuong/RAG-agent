import json
from typing import Dict
from ..state import AgentState
from ..parser import parse_tool_calls
from ...config import logger
from ...tools import TOOLS_SCHEMA
from .utils import get_history_for_generation

class GenerateNode:
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, state: AgentState) -> Dict:
        """Node to generate LLM response."""
        messages = state.get("messages", [])
        intent_data = state.get("intent", {})
        plan = state.get("plan", "")
        active_tools = state.get("active_tools")
        
        # Filter tools if active_tools is specified, otherwise use all
        if active_tools:
            current_tools_schema = [
                tool for tool in TOOLS_SCHEMA 
                if tool["function"]["name"] in active_tools
            ]
        else:
            current_tools_schema = TOOLS_SCHEMA
        
        # Construct dynamic system prompt
        tools_json = json.dumps(current_tools_schema, indent=2)
        
        # Build Few-Shot Examples based on active tools
        stock_tool_active = False
        if not active_tools: # None means all active
            stock_tool_active = True
        elif "get_stock_price" in active_tools:
            stock_tool_active = True
            
        stock_example = ""
        if stock_tool_active:
            stock_example = """
- Input: "What is the stock price of Apple?"
- Output: <tool_call>{{"name": "get_stock_price", "arguments": {{"symbol": "AAPL"}} }}</tool_call>
"""

        system_content = f"""You are a helpful and intelligent assistant. You can use tools to answer questions.
If you need to use a tool, output the function call inside <tool_call> tags.
Do NOT output the result of the tool, just the proper XML tag.

Available Tools:
{tools_json}

### 1. PLAN ENFORCEMENT (CRITICAL):
- **PRIORITY**: If a system message provides a "PLANNING_STEP" or instructions starting with "You MUST use tools", you must STRICTLY follow it.
- **ACTION**: Generate <tool_call> tags for the requested topics immediately. 
- **PROHIBITION**: Do NOT generate any conversational text, explanations, or internal knowledge answers when a plan involves searching. Output ONLY the tool calls.

### 2. PURE ROUTER/SEARCHER ROLE:
- **Constraint**: You are NOT allowed to generate the final conversational answer. Your ONLY job is to find information using tools.
- **Sufficiency Check**: If you believe you have enough information from the history or if no external tools are needed (e.g., simple greeting), you should output NO tool calls.
- **Output**:
    - If tools needed: <tool_call>...</tool_call>
    - If NO tools needed: (Empty response or just whitespace) - The system will automatically route to the Synthesis step to generate the answer.
    - Do NOT write: "I can answer this..." or "Hello...". Leave that to the Synthesis step.

### 3. FEW-SHOT EXAMPLES:
- Input: "Hello!"
- Output: 
{stock_example}
- Input: "who is the president of the US?" (If you know or no tool needed)
- Output: 

### 4. KNOWLEDGE BASE:
- For general investment concepts, ALWAYS search the knowledge base first using <tool_call>{{"name": "query_knowledge_base", "arguments": {{ "query": "your question" }} }}</tool_call>.
"""
        
        # Build prompt messages
        prompt_messages = [{"role": "system", "content": system_content}]
        
        # Inject Plan if available
        if plan:
             prompt_messages.append({"role": "system", "content": plan})

        # Add history
        # Use filtered history: Clean past + Full current turn
        filtered_messages = get_history_for_generation(messages)
        prompt_messages.extend(filtered_messages)

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
