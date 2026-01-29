from typing import Dict
from ..state import AgentState
from ..intent_analyzer import analyze_intent
from ...config import logger

class AnalyzeIntentNode:
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, state: AgentState) -> Dict:
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
