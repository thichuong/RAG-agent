# main.py
import argparse
import os
import sys

# Ensure we are running in a virtual environment
if sys.prefix == sys.base_prefix:
    print("❌ Error: You must run this application within the .venv environment.")
    print("👉 Please run: source .venv/bin/activate")
    sys.exit(1)

import gradio as gr
from src.config import DATA_DIR, logger
from src.rag import InvestmentRAG
from src.llm import load_model
from src.agent import QwenAgent
from src.setup_mapping import download_and_process_mappings


# Global references (initialized in main)
rag_instance = None
llm_instance = None
agent_instance = None


def add_document_handler(files):
    """Handle document upload via Gradio UI."""
    global rag_instance, llm_instance
    
    if not rag_instance or not rag_instance.is_ready:
        return "❌ RAG system not initialized. Please wait..."
    
    if not files:
        return "❌ No files selected."
    
    results = []
    for file in files:
        try:
            # Read file content
            with open(file.name, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Get filename as doc_id
            doc_id = os.path.basename(file.name)
            
            # Add to RAG (with LLM for summary generation)
            success = rag_instance.add_document(doc_id, content, llm=llm_instance)
            
            if success:
                chunk_count = len(rag_instance.doc_store.get(doc_id, []))
                results.append(f"✅ **{doc_id}**: Added with {chunk_count} chunks")
            else:
                results.append(f"❌ **{doc_id}**: Failed to add")
                
        except Exception as e:
            results.append(f"❌ **{os.path.basename(file.name)}**: Error - {e}")
    
    # Save cache after adding documents
    try:
        rag_instance.save_cache()
        results.append("\n💾 Cache saved successfully.")
    except Exception as e:
        results.append(f"\n⚠️ Warning: Failed to save cache - {e}")
    
    return "\n".join(results)


def get_rag_status():
    """Get current RAG status."""
    global rag_instance
    if not rag_instance or not rag_instance.is_ready:
        return "⏳ RAG not initialized"
    
    num_docs = len(rag_instance.doc_store)
    total_chunks = sum(len(chunks) for chunks in rag_instance.doc_store.values())
    doc_names = list(rag_instance.doc_store.keys())
    
    status = f"""### 📊 RAG Status
- **Documents**: {num_docs}
- **Total Chunks**: {total_chunks}
- **Strategy**: Summary Vector (Parent-Document Retrieval)

### 📁 Loaded Documents
"""
    for doc in doc_names:
        chunk_count = len(rag_instance.doc_store[doc])
        status += f"- `{doc}` ({chunk_count} chunks)\n"
    
    return status


def main():
    global rag_instance, llm_instance, agent_instance
    
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Qwen3 Multi-Step Agent + Investment RAG")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild RAG cache, ignoring existing cache"
    )
    args = parser.parse_args()
    
    # 0. Setup/Update Mappings
    print("🔄 Ensuring ticker mappings are up-to-date...")
    download_and_process_mappings()

    # 1. Load Model
    llm_instance = load_model()
    
    # 2. Initialize RAG (with optional force rebuild)
    rag_instance = InvestmentRAG(DATA_DIR)
    rag_instance.initialize(llm=llm_instance, force_rebuild=args.rebuild)
    
    # 3. Initialize Agent
    agent_instance = QwenAgent(llm_instance, rag_instance)
    
    # 4. Gradio Interface with Tabs
    def chat_fn(message, history):
        response, steps = agent_instance.run(message, history)
        if steps:
            return f"{response}\n\n<details><summary><b>🛠️ Execution Trace (Click to Expand)</b></summary>\n\n{steps}\n</details>"
        return response

    with gr.Blocks(title="Qwen3 Agent + RAG") as demo:
        gr.Markdown("# 🤖 Qwen3 Multi-Step Agent + Investment RAG")
        gr.Markdown("Agent capable of reasoning, math, stock checks, and querying internal investment docs.")
        
        with gr.Tabs():
            # Tab 1: Chat
            with gr.TabItem("💬 Chat"):
                chatbot = gr.Chatbot(height=500)
                msg = gr.Textbox(
                    placeholder="Ask me anything about investments...",
                    label="Your Message",
                    lines=2
                )
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear")
                
                gr.Examples(
                    examples=[
                        "What is Value at Risk?",
                        "Price of Bitcoin divided by 2?",
                        "Search news for 'AI bubble'"
                    ],
                    inputs=msg
                )
                
                def respond(message, chat_history):
                    chat_history = chat_history or []
                    # Convert messages format to tuple list for agent
                    history = []
                    for item in chat_history:
                        if isinstance(item, dict):
                            role = item.get("role", "")
                            content = item.get("content", "")
                        else:
                            # Fallback if somehow we get tuples or other format
                            role = "user" if len(history) % 2 == 0 else "assistant" 
                            content = str(item)
                        history.append((role, content))
                    
                    response = chat_fn(message, history)
                    chat_history.append({"role": "user", "content": message})
                    chat_history.append({"role": "assistant", "content": response})
                    return "", chat_history
                
                submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
                msg.submit(respond, [msg, chatbot], [msg, chatbot])
                clear_btn.click(lambda: ("", []), outputs=[msg, chatbot])
            
            # Tab 2: Document Management
            with gr.TabItem("📄 Documents"):
                gr.Markdown("## Upload Documents to RAG")
                gr.Markdown("Upload `.txt` files to add them to the knowledge base in real-time.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        file_upload = gr.File(
                            label="Upload Text Files",
                            file_count="multiple",
                            file_types=[".txt"],
                            type="filepath"
                        )
                        upload_btn = gr.Button("📤 Add to RAG", variant="primary")
                        upload_result = gr.Markdown(label="Upload Result")
                    
                    with gr.Column(scale=1):
                        status_display = gr.Markdown(value=get_rag_status, label="RAG Status")
                        refresh_btn = gr.Button("🔄 Refresh Status")
                
                upload_btn.click(add_document_handler, inputs=[file_upload], outputs=[upload_result])
                upload_btn.click(get_rag_status, outputs=[status_display])
                refresh_btn.click(get_rag_status, outputs=[status_display])
        
    demo.launch(server_name="0.0.0.0", share=True)


if __name__ == "__main__":
    main()
