# main.py
import gradio as gr
from src.config import DATA_DIR
from src.rag import InvestmentRAG
from src.llm import load_model
from src.agent import QwenAgent

# --- Main Execution & UI ---
def main():
    # 1. Load Model
    llm = load_model()
    
    # 2. Initialize RAG
    rag = InvestmentRAG(DATA_DIR)
    rag.initialize(llm=llm)
    
    # 3. Initialize Agent
    agent = QwenAgent(llm, rag)
    
    # 4. Gradio Interface
    def chat_fn(message, history):
        # formatted_history passed to run
        response, steps = agent.run(message, history)
        if steps:
            return f"{response}\n\n<details><summary><b>🛠️ Execution Trace (Click to Expand)</b></summary>\n\n{steps}\n</details>"
        return response

    demo = gr.ChatInterface(
        fn=chat_fn,
        title="Qwen3 Multi-Step Agent + Investment RAG",
        description="Agent capable of reasoning, math, stock checks, and querying internal investment docs.",
        examples=["What is Value at Risk?", "Price of Bitcoin divided by 2?", "Search news for 'AI bubble'"]
    )
    
    demo.launch(server_name="0.0.0.0", share=True)

if __name__ == "__main__":
    main()
