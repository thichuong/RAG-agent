import json
from typing import Dict, List
from ..state import AgentState
from ...config import logger
from .utils import get_clean_history

def analyze_intent(llm, messages: List[Dict]) -> dict:
    """
    Analyzes the user's query to extract the underlying goal and the expected language,
    considering the conversation history.
    
    Returns:
        dict: {
            "goal": "The core objective of the user",
            "language": "The language (e.g., Vietnamese, English)"
        }
    """
    if not messages:
        return {"goal": "", "language": "English"}

    # Extract current query (last user message)
    # Ideally the last message is the user's new input
    query = messages[-1]["content"] if messages else ""
    
    # Format history
    history_text = ""
    # Use clean history (user questions & final answers only)
    clean_msgs = get_clean_history(messages[:-1])
    # Limit to last 10 relevant messages
    for m in clean_msgs[-10:]:
        role = m.get("role", "unknown")
        content = m.get("content", "")
        history_text += f"{role.upper()}: {content}\n"

    prompt = f"""You are an advanced Intent Classifier.

Conversation History:
{history_text}

Current User Query: "{query}"

Analyze the Current User Query in the context of the History (if any) and extract:
1. The **Goal** (What does the user really want? Be specific. If they ask a follow-up question like "and him?", use history to resolve who "he" is.)
2. The **Language** (What language is the user using or expecting?)

Output JSON ONLY in this format:
{{
  "goal": "...",
  "language": "..."
}}
"""

    try:
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
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
        # Get the messages
        messages = state.get("messages", [])
        if not messages:
            return {}
        
        logger.info("🧠 Analyzing Intent with History...")
        intent_data = analyze_intent(self.llm, messages)
        logger.info(f"🎯 Intent Detected: {intent_data}")
        
        return {"intent": intent_data}
