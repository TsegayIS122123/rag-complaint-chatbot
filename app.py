"""
CrediTrust Complaint Analysis - FAST REAL RAG
Loads in < 30 seconds! - FIXED for Gradio 6.5.1
"""

import gradio as gr
import sys
sys.path.append('.')

from src.rag_pipeline import get_rag

print("🚀 Starting FAST REAL RAG...")
rag = get_rag()
print("✅ Ready! Ask me anything about complaints.\n")

def process(question, history, product_filter):
    """Process question with REAL data - FIXED for Gradio 6.5.1."""
    if not question:
        return "", "", history
    
    # Get REAL answer
    result = rag.ask(question, product_filter)
    
    # Format sources
    sources = "### 📚 REAL Complaint Sources\n\n"
    for i, src in enumerate(result['sources'], 1):
        product = src.get('product', 'Unknown')
        text = src.get('text', '')[:150]
        sources += f"**{i}. {product}**\n> {text}...\n\n"
    
    # FIXED: Initialize history if None
    if history is None:
        history = []
    
    # FIXED: Gradio 6.x uses tuple format (user, assistant)
    history.append((question, result['answer']))
    
    return result['answer'], sources, history

def clear():
    """Clear everything."""
    return "", "", "All Products", []

# Product options
products = ["All Products", "Credit Card", "Personal Loan", "Savings Account", "Money Transfer"]

# UI
with gr.Blocks(title="CrediTrust FAST REAL RAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏦 CrediTrust Financial - FAST REAL RAG")
    gr.Markdown("*Using REAL complaint data - Loads in < 30 seconds!*")
    
    with gr.Row():
        with gr.Column(scale=2):
            # FIXED: No type parameter needed for Gradio 6.x
            chatbot = gr.Chatbot(label="Conversation", height=400)
            
            filter_dropdown = gr.Dropdown(
                choices=products,
                value="All Products",
                label="Filter by Product"
            )
            
            question = gr.Textbox(
                label="Ask about complaints",
                placeholder="What are customers saying about credit cards?",
                lines=2
            )
            
            with gr.Row():
                submit = gr.Button("🔍 Analyze", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
        
        with gr.Column(scale=2):
            answer = gr.Markdown(
                label="📋 REAL Analysis",
                value="*Ask a question to see REAL analysis*",
                line_breaks=True
            )
            
            sources = gr.Markdown(
                label="📚 REAL Sources",
                value="*Sources will appear here*",
                line_breaks=True
            )
    
    state = gr.State([])
    
    # Events
    submit.click(
        process, 
        [question, state, filter_dropdown],
        [answer, sources, state]
    ).then(
        lambda: "", None, question
    ).then(
        lambda s: s, state, chatbot
    )
    
    question.submit(
        process,
        [question, state, filter_dropdown],
        [answer, sources, state]
    ).then(
        lambda: "", None, question
    ).then(
        lambda s: s, state, chatbot
    )
    
    clear_btn.click(
        clear,
        None,
        [answer, sources, filter_dropdown, state]
    ).then(
        lambda: [], None, chatbot
    )
    
    # Examples
    gr.Markdown("### 💡 Try These Questions")
    with gr.Row():
        examples = [
            ("💳 Credit Cards", "What are credit card complaints?"),
            ("💰 Personal Loans", "What issues with personal loans?"),
            ("🏦 Savings Accounts", "Savings account problems?"),
            ("💸 Money Transfers", "Money transfer issues?")
        ]
        
        for label, q in examples:
            # FIXED: Button click sets question properly
            btn = gr.Button(label, size="sm")
            btn.click(
                fn=lambda q=q: q,
                outputs=question
            )

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🌐 Open: http://localhost:7860")
    print("🚀 FAST REAL RAG - Ready in < 30 seconds!")
    print("="*50 + "\n")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        quiet=False,
        show_error=True
    )