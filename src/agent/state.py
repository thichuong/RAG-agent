from typing import List, Dict, TypedDict, Any, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[List[Dict[str, str]], operator.add]
    logs: Annotated[List[str], operator.add]
    intent: Dict[str, str]
    plan: str
    step_count: int
    tool_calls: List[Dict[str, Any]]
    final_answer: str
