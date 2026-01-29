
import json
import logging

try:
    from ..config import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

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
