import re
import json
try:
    from ..config import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

def parse_tool_calls(text):
    """
    Parse tool calls from Qwen instructions.
    """
    calls = []
    text = str(text)
    
    # 1. Try <tool_call> XML-like tags
    tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
    matches = re.findall(tool_call_pattern, text, re.DOTALL)
    
    for m in matches:
        try:
            call_data = json.loads(m.strip())
            calls.append(call_data)
        except Exception as e:
            logger.warning(f"Failed to parse tool call JSON: {e} | Content: {m}")

    if calls:
        return calls

    # 2. Fallback: Detect standard tool structure
    try:
        json_pattern = r"\{.*?\}"
        potential_jsons = re.findall(json_pattern, text, re.DOTALL)
        for pj in potential_jsons:
            try:
                data = json.loads(pj)
                if "name" in data and "arguments" in data:
                    calls.append(data)
            except:
                continue
    except:
        pass
    
    return calls
