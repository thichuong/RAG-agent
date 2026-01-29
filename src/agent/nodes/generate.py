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
        
        # Construct dynamic system prompt
        tools_json = json.dumps(TOOLS_SCHEMA, indent=2)
        system_content = f"""You are a helpful and intelligent assistant. You can use tools to answer questions.
If you need to use a tool, output the function call inside <tool_call> tags.
Do NOT output the result of the tool, just the proper XML tag.

Available Tools:
{tools_json}

### 1. PLAN ENFORCEMENT (CRITICAL):
- **PRIORITY**: If a system message provides a "PLANNING_STEP" or instructions starting with "You MUST use tools", you must STRICTLY follow it.
- **ACTION**: Generate <tool_call> tags for the requested topics immediately. 
- **PROHIBITION**: Do NOT generate any conversational text, explanations, or internal knowledge answers when a plan involves searching. Output ONLY the tool calls.

### 2. SEARCH & NEWS PROTOCOL:
- **Listicle Format**: When you search for news (using `get_news`), your *immediate* next response should summarize the findings in a structured list (3-5 items) with sources.
    - Example:
      1. **[Source Name]**: Headline or brief summary.
      2. **[Source Name]**: Headline...
- **Deep Dive**: If the user needs more detail, or the task implies a comprehensive report, do NOT rely on just the headlines.
    - **Step 1**: Select the top 2-3 most relevant URLs from the search results.
    - **Step 2**: specificially call `crawl_url` on these URLs.
    - **Step 3**: Synthesize the content from these crawled pages into a single cohesive answer.
    - **Constraint**: If you choose to crawl only *one* source, you MUST explicitly explain why (e.g., "This was the only relevant detailed analysis found...").

### 3. CITATION PROTOCOL (STRICT):
- **MANDATORY**: Your FINAL answer MUST include citations with links for any factual claims.
- Format: [Source: Source Name or Domain] or [Source: Title](URL). 
- Example: "The market crashed due to... [Source: Bloomberg]"

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
