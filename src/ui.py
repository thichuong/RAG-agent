import os
import gradio as gr
from src.tools import get_all_tool_names

def create_ui(agent_instance, rag_instance, llm_instance):
    """
    Creates and returns the Gradio UI object.
    
    Args:
        agent_instance: The initialized QwenAgent instance.
        rag_instance: The initialized InvestmentRAG instance.
        llm_instance: The initialized LLM instance.
    """

    # --- Handlers ---

    def add_document_handler(files):
        """Handle document upload."""
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

    def chat_fn(message, history, active_tools):
        """Chat function connecting to the agent."""
        response, clusters = agent_instance.run(message, history, active_tools=active_tools)
        if clusters:
            return f"{response}\n\n<details><summary><b>🛠️ Execution Trace (Click to Expand)</b></summary>\n\n{clusters}\n</details>"
        return response

    def respond(message, chat_history, active_tools):
        """Wrapper for the chat function to handle history formatting."""
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
        
        response = chat_fn(message, history, active_tools)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": response})
        return "", chat_history

    # --- UI Layout ---

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
    ).set(
        body_background_fill="*neutral_50",
        block_background_fill="*neutral_100",
        button_primary_background_fill="*primary_600",
        button_primary_text_color="white",
    )
    
    css = """
    .container { max-width: 1200px; margin: auto; padding-top: 20px; }
    .header { text-align: center; margin-bottom: 30px; }
    .header h1 { font-size: 2.5em; font-weight: bold; color: #334155; }
    .header p { font-size: 1.1em; color: #64748b; }
    #chatbot { min-height: 600px; border-radius: 10px; border: 1px solid #e2e8f0; }
    .sidebar { background: white; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0; }
    """

    with gr.Blocks(theme=theme, css=css, title="Qwen3 Agent + RAG") as ui:
        with gr.Column(elem_classes="container"):
            
            with gr.Column(elem_classes="header"):
                gr.Markdown("# 🤖 Qwen3 Multi-Step Agent + Investment RAG")
                gr.Markdown("Agent capable of reasoning, math, stock checks, and querying internal investment docs.")

            with gr.Tabs():
                # --- Tab 1: Chat ---
                with gr.TabItem("💬 Chat", id="tab_chat"):
                    with gr.Row():
                        # Main Chat Area
                        with gr.Column(scale=3):
                            chatbot = gr.Chatbot(
                                elem_id="chatbot",
                                height=600
                            )
                            with gr.Row():
                                msg = gr.Textbox(
                                    placeholder="Type your message here...",
                                    show_label=False,
                                    container=False,
                                    scale=8,
                                    autofocus=True
                                )
                                submit_btn = gr.Button("Send", variant="primary", scale=1, min_width=100)
                            
                            with gr.Row():
                                clear_btn = gr.Button("Clear Chat", variant="secondary", size="sm")

                        # Sidebar / Configuration
                        with gr.Column(scale=1):
                            with gr.Group(elem_classes="sidebar"):
                                gr.Markdown("### ⚙️ Configuration")
                                
                                all_tools = get_all_tool_names()
                                tool_checkboxes = gr.CheckboxGroup(
                                    choices=all_tools,
                                    value=all_tools,
                                    label="Active Tools",
                                    info="Select tools allowed for this session."
                                )
                                
                                gr.Markdown("### 📝 Examples")
                                gr.Examples(
                                    examples=[
                                        "What is Value at Risk?",
                                        "Price of Bitcoin divided by 2?",
                                        "Search news for 'AI bubble'"
                                    ],
                                    inputs=msg,
                                    label="Try these:"
                                )

                    # Event Listeners
                    submit_btn.click(respond, [msg, chatbot, tool_checkboxes], [msg, chatbot])
                    msg.submit(respond, [msg, chatbot, tool_checkboxes], [msg, chatbot])
                    clear_btn.click(lambda: ("", []), outputs=[msg, chatbot])

                # --- Tab 2: Document Management ---
                with gr.TabItem("📄 Knowledge Base", id="tab_docs"):
                    gr.Markdown("## 📚 Knowledge Base Management")
                    gr.Markdown("Upload text documents to expand the agent's knowledge.")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            with gr.Group(elem_classes="sidebar"):
                                gr.Markdown("### 📤 Upload")
                                file_upload = gr.File(
                                    label="Select Files (.txt)",
                                    file_count="multiple",
                                    file_types=[".txt"],
                                    type="filepath"
                                )
                                upload_btn = gr.Button("Add to Knowledge Base", variant="primary")
                                upload_result = gr.Markdown(label="Status")
                        
                        with gr.Column(scale=1):
                            with gr.Group(elem_classes="sidebar"):
                                gr.Markdown("### 📊 System Status")
                                status_display = gr.Markdown(value=get_rag_status)
                                refresh_btn = gr.Button("🔄 Refresh Status")
                    
                    # Event Listeners
                    upload_btn.click(add_document_handler, inputs=[file_upload], outputs=[upload_result])
                    upload_btn.click(get_rag_status, outputs=[status_display])
                    refresh_btn.click(get_rag_status, outputs=[status_display])

    return ui
