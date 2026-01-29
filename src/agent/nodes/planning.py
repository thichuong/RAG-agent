from typing import Dict
from ..state import AgentState
from ..planner import analyze_request
from ...config import logger

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
