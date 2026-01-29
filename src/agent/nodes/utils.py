from typing import List, Dict

def is_tool_call(message: Dict) -> bool:
    """Check if a message is a tool call or output."""
    role = message.get("role")
    content = message.get("content", "") or ""
    
    if role == "tool":
        return True
    
    # Check for <tool_call> tags in assistant messages
    if role == "assistant" and "<tool_call>" in content:
        return True
        
    return False

def get_clean_history(messages: List[Dict]) -> List[Dict]:
    """
    Get clean history containing only User queries and Assistant final answers.
    Filters out tool calls and tool outputs.
    """
    clean_history = []
    for m in messages:
        if is_tool_call(m):
            continue
        
        # Only keep user and assistant messages
        role = m.get("role")
        if role in ["user", "assistant"]:
            clean_history.append(m)
            
    return clean_history

def get_history_for_generation(messages: List[Dict]) -> List[Dict]:
    """
    Get history for generation: 
    - Clean history (User + Final Answer) for past turns
    - FULL context (including tools) for the current turn (messages after last user query)
    """
    if not messages:
        return []

    # Find the index of the last user message
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break
            
    if last_user_idx == -1:
        # No user message found? Just return clean history of what we have
        return get_clean_history(messages)

    # Split into past and current turn
    past_messages = messages[:last_user_idx]
    current_turn_messages = messages[last_user_idx:]

    # Clean past messages
    clean_past = get_clean_history(past_messages)

    # Combine
    return clean_past + current_turn_messages
