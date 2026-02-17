"""
CrediTrust Complaint Analysis Chatbot
FIXED for Gradio 4.44.1 - ULTRA FAST & STABLE
"""

import gradio as gr
import sys
sys.path.append('.')

from src.rag_pipeline import get_rag

# Initialize RAG
print("🚀 Starting CrediTrust Chatbot...")
rag = get_rag()
print("✅ Ready! Ask me anything.\n")

def chat(question, history):
    """Chat function for Gradio 4.x."""
    if not question:
        return "", history
    
    result = rag.ask(question)
    
    # Format sources
    sources_text = "### 📚 Source Complaints\n\n"
    for i, src in enumerate(result['sources'], 1):
        product = src.get('metadata', {}).get('product', 'Unknown')
        text = src.get('text', '')[:150]
        sources_text += f"**{i}. {product}**\n> {text}...\n\n"
    
    # Update history
    history = history or []
    history.append((question, result['answer']))
    
    return result['answer'], sources_text, history

def clear():
    return "", "", []

# Simple CSS
css = """
.gradio-container { max-width: 1200px !important; margin: auto !important; background: white !important; }
h1 { color: #1e3c72 !important; font-weight: 700 !important; }
h1, h2, h3, p, label { color: #333333 !important; }
"""

# Create interface
with gr.Blocks(title="CrediTrust Complaint Analysis", css=css) as demo:
    gr.Markdown("# 🏦 CrediTrust Complaint Analysis Chatbot")
    gr.Markdown("### Ask questions about customer complaints")
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation History", height=400)
            question = gr.Textbox(
                label="Ask a question",
                placeholder="What are credit card complaints?",
                lines=2
            )
            with gr.Row():
                submit = gr.Button("🔍 Analyze", variant="primary")
                clear_btn = gr.Button("🗑️ Clear")
        
        with gr.Column(scale=2):
            answer = gr.Textbox(label="📋 Analysis", lines=12, interactive=False)
            sources = gr.Markdown(label="📚 Sources", value="*Sources will appear here*")
    
    state = gr.State([])
    
    submit.click(chat, [question, state], [answer, sources, state]
    ).then(lambda: "", None, question
    ).then(lambda s: s, state, chatbot)
    
    question.submit(chat, [question, state], [answer, sources, state]
    ).then(lambda: "", None, question
    ).then(lambda s: s, state, chatbot)
    
    clear_btn.click(clear, None, [answer, sources, state]
    ).then(lambda: [], None, chatbot)
    
    gr.Markdown("### 💡 Try these examples")
    examples = ["Credit card complaints", "Loan problems", "Money transfer issues", "Savings account issues"]
    for ex in examples:
        gr.Button(ex, size="sm").click(lambda q=ex: q, None, question)

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🌐 Open: http://localhost:7860")
    print("="*50 + "\n")
    demo.launch(
        server_name="127.0.0.1", cd # Changed from 0.0.0.0 to 127.0.0.1
        server_port=7860,
        quiet=False,
        share=False  # Don't create public link
    )