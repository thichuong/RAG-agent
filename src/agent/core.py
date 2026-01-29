import json
import re
from typing import List, Dict, TYPE_CHECKING, Any, Literal
from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import GraphNodes

if TYPE_CHECKING:
    from llama_cpp import Llama
    from ..rag import InvestmentRAG
else:
    Llama = object
    InvestmentRAG = object

# Absolute imports
try:
    from ..config import logger
except ImportError as e:
    raise e

class QwenAgent:
    def __init__(self, llm: Llama, rag: InvestmentRAG):
        self.llm = llm
        self.rag = rag
        
        # Initialize Nodes
        self.nodes = GraphNodes(llm, rag)
        
        # Build Graph
        workflow = StateGraph(AgentState)
        
        # Add Nodes
        workflow.add_node("analyze_intent", self.nodes.analyze_intent_node)
        workflow.add_node("planning", self.nodes.planning_node)
        workflow.add_node("generate", self.nodes.generate_node)
        workflow.add_node("execute_tools", self.nodes.execute_tools_node)
        workflow.add_node("synthesis", self.nodes.synthesis_node)
        
        # Add Edges
        workflow.set_entry_point("analyze_intent")
        workflow.add_edge("analyze_intent", "planning")
        workflow.add_edge("planning", "generate")
        
        # Conditional Edge from Generate
        workflow.add_conditional_edges(
            "generate",
            self.should_continue,
            {
                "execute_tools": "execute_tools",
                "synthesis": "synthesis"
            }
        )
        
        # Loop back from tools to generate
        workflow.add_edge("execute_tools", "generate")
        
        # End from synthesis
        workflow.add_edge("synthesis", END)
        
        # Compile
        self.app = workflow.compile()

    def should_continue(self, state: AgentState) -> Literal["execute_tools", "synthesis", "end"]:
        """Determine next step based on state."""
        step_count = state.get("step_count", 0)
        calls = state.get("tool_calls", [])
        
        # 1. Max steps reached - Force synthesis if tools were used, else end
        if step_count >= 5:
            # Check if we have gathered anything useful to synthesize?
            # Or just wrap up.
            return "synthesis"

        # 2. Tool calls present -> Execute them
        if calls:
            return "execute_tools"
        
        # 3. No new tool calls. 
        # Check if tools were executed previously in this session (checking for 'tool' role)
        messages = state.get("messages", [])
        # We only care about tools executed in the CURRENT turn (after the user query).
        # But 'messages' contains full history.
        # Ideally we should check if we did any tool calls since the last user message.
        # Heuristic: verify if there is any 'tool' message AFTER the last 'user' message.
        
        last_user_idx = -1
        for i, m in enumerate(messages):
            if m.get("role") == "user":
                last_user_idx = i
        
        has_recent_tool_outputs = False
        if last_user_idx != -1:
            for m in messages[last_user_idx:]:
                if m.get("role") == "tool":
                    has_recent_tool_outputs = True
                    break
                    
        if has_recent_tool_outputs:
            return "synthesis"
        
        # 4. No tools used at all -> Synthesis (to generate the final conversational response)
        # Previously we went to "end", but now 'generate' only does tool calls.
        # So we need synthesis to create the actual text response.
        return "synthesis"

    def run(self, user_query: str, history: List[Dict] = [], active_tools: List[str] = None):
        """
        Run the agent workflow.
        """
        # Prepare initial state
        # Convert tuple history to dict list if needed, matching core.py original expectation
        formatted_history = []
        if history:
             for turn in history:
                 if isinstance(turn, (list, tuple)) and len(turn) == 2:
                     user_msg, bot_msg = turn
                     formatted_history.append({"role": "user", "content": str(user_msg)})
                     formatted_history.append({"role": "assistant", "content": str(bot_msg)})
                 elif isinstance(turn, dict):
                     formatted_history.append(turn)

        # Add current user query
        formatted_history.append({"role": "user", "content": user_query})

        initial_state = {
            "messages": formatted_history,
            "logs": [],
            "step_count": 0,
            "intent": {},
            "plan": "",
            "tool_calls": [],
            "active_tools": active_tools,
        }
        
        # Invoke Graph
        final_state = self.app.invoke(initial_state)
        
        # Extract Output
        logs_text = "\n\n".join(final_state.get("logs", []))
        
        # Determine final answer
        if final_state.get("final_answer"):
            final_answer = final_state["final_answer"]
        else:
            # Fallback to the last assistant message content
            messages = final_state.get("messages", [])
            if messages and messages[-1]["role"] == "assistant":
                final_answer = messages[-1]["content"]
                # Clean up <tool_call> tags if present in direct answer
                final_answer = re.sub(r"<tool_call>.*?</tool_call>", "", final_answer, flags=re.DOTALL).strip()
                final_answer = final_answer.replace("<|im_end|>", "")
            else:
                final_answer = "No response generated."

        return final_answer, logs_text
