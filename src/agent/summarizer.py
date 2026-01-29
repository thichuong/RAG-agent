try:
    from ..config import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

def summarize_text(llm, text: str) -> str:
    """
    Summarize crawled text to save context window.
    """
    if len(text) < 500:
        return text # Short enough, no need to summarize

    summary_prompt = f"""Summarize the following text into 3-4 distinct bullet points. Focus on facts, numbers, and key insights relevant to the topic.
    
Text:
{text[:8000]}
"""
    try:
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=300,
            temperature=0.1
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"Summarization failed: {e}")
        return text[:500] + "... (Summarization failed)"
