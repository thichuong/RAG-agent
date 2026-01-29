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
2. Review the Conversation History and Tool Outputs (if any).

### MODE SELECTION:
**Scenario A: Tool Outputs are present (Research Mode)**
- Synthesize a COMPREHENSIVE final response based *primarily* on the tool outputs.
- **Listicle Format**: For news/lists, use a structured list (3-5 items).
- **Deep Dive**: For detailed topics, synthesize a cohesive report.
- **Constraint**: Use ONLY the information present in the tool outputs. Do not make up facts.
- **CITATIONS (STRICT)**: You MUST include citations [Source: Name](URL) for all factual claims.

**Scenario B: No Tool Outputs (Conversational Mode)**
- If no tools were called (e.g., greetings, philosophical questions, or general knowledge within your training), answer the user directly and helpfully.
- Be polite and professional.

### CRITICAL FORMATTING RULES:
- **Direct Answer**: Your output must be the FINAL answer to the user. Do NOT include "Here is the answer:" or "System:".
- **Language**: The response MUST be in {intent_data.get('language', 'Vietnamese')}.
- **Fix Formatting**: If the context contains broken markdown or XML tags, clean them up in your response.

### CONTEXT:
- Goal: {intent_data.get('goal', 'Answer the question')}
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
