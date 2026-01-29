import json
from typing import Dict
from ..state import AgentState
from ...config import logger

def analyze_intent(llm, query: str) -> dict:
    """
    Analyzes the user's query to extract the underlying goal and the expected language.
    
    Returns:
        dict: {
            "goal": "The core objective of the user",
            "language": "The language (e.g., Vietnamese, English)"
        }
    """
    
    prompt = f"""You are an advanced Intent Classifier.
Analyze the following User Query and extract:
1. The **Goal** (What does the user really want? Be specific.)
2. The **Language** (What language is the user using or expecting?)

Output JSON ONLY in this format:
{{
  "goal": "...",
  "language": "..."
}}

User Query: "{query}"
"""

    try:
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.1
        )
        content = response["choices"][0]["message"]["content"].strip()
        
        # Simple cleanup to ensure JSON parsing works if model adds markdown
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "")
        
        intent_data = json.loads(content)
        return intent_data

    except Exception as e:
        logger.warning(f"Intent analysis failed: {e}")
        # Fallback
        return {
            "goal": "Answer the user's question.",
            "language": "Vietnamese" if any(c for c in query if ord(c) > 128) else "English" # Crude heuristic
        }

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
