import re
from typing import Dict
from ..state import AgentState
from ...config import logger

class SynthesisNode:
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, state: AgentState) -> Dict:
        """Node to synthesize final answer."""
        messages = state.get("messages", [])
        intent_data = state.get("intent", {})
        
        # Find original query
        user_query = ""
        for m in messages:
            if m["role"] == "user":
                user_query = m["content"]
                # Keep finding the last user query? Or first?
                # Usually the main query is the start of this session.
        
        logger.info("Tools were executed. Triggering Final Synthesis Step...")
        new_logs = ["🧠 **Final Synthesis**: Generating consolidated answer based on tool outputs..."]
        
        synthesis_prompt = f"""SYSTEM: PREPARE FINAL ANSWER.
1. Review the User Query: "{user_query}"
2. Review ALL Tool Outputs above (especially Summaries of crawled pages).
3. Synthesize a COMPREHENSIVE final response.
4. CONSTRAINT: Use ONLY the information present in the tool outputs. Do not make up facts. verification_step=TRUE
5. CITATIONS: You MUST include citations [Source: Name](URL) for all factual claims as per protocol.

CONTEXT FROM INTENT ANALYSIS:
- Goal: {intent_data.get('goal', 'Answer the question')}
- Required Language: {intent_data.get('language', 'Vietnamese')}

Ensure the final answer addresses the Goal and is written in {intent_data.get('language', 'Vietnamese')}.
"""
        # Construct prompt
        prompt_messages = list(messages) # Copy
        prompt_messages.append({"role": "system", "content": synthesis_prompt})
        
        response = self.llm.create_chat_completion(
            messages=prompt_messages,
            max_tokens=1024,
            temperature=0.1
        )
        final_answer_text = response["choices"][0]["message"]["content"]
        
        # Cleanup
        final_answer = re.sub(r"<tool_call>.*?</tool_call>", "", final_answer_text, flags=re.DOTALL).strip()
        final_answer = final_answer.replace("<|im_end|>", "")
        
        new_logs.append(f"🤖 **Final Answer**: {final_answer}")
        
        return {
            "final_answer": final_answer,
            "logs": new_logs
        }
