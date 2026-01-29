from typing import Dict, TYPE_CHECKING
from ..state import AgentState
from ...config import logger

if TYPE_CHECKING:
    from llama_cpp import Llama

def analyze_request(llm, query: str) -> str:
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
        response = llm.create_chat_completion(
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

class PlanningNode:
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, state: AgentState) -> Dict:
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
