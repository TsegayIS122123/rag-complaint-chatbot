"""
CrediTrust Complaint Analysis Chatbot
FIXED: Chatbot format, performance, and styling
"""

import gradio as gr
import time
import logging
import sys
import os
sys.path.append('.')

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from src.rag_pipeline import get_rag_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize RAG
try:
    rag = get_rag_pipeline()
    logger.info("✅ RAG pipeline initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize RAG: {e}")
    rag = None

# Product options
PRODUCTS = ["All Products", "Credit card", "Personal loan", "Savings account", "Money transfers"]

def process_question(question, history, product_filter, temperature):
    """Process user question with RAG pipeline."""
    if not question or not question.strip():
        return "", "", history
    
    if not rag:
        return "⚠️ RAG system not available", "Error: Could not initialize RAG", history
    
    try:
        # Get answer from RAG
        result = rag.ask(question, n_results=5)
        
        answer = result['answer']
        sources = result.get('sources', [])
        
        # Format sources for display
        sources_text = "### 📚 Source Complaints\n\n"
        if sources and len(sources) > 0:
            for i, source in enumerate(sources[:3], 1):
                product = source.get('metadata', {}).get('product', 'Unknown')
                text = source.get('text', '')[:200] + "..." if len(source.get('text', '')) > 200 else source.get('text', '')
                score = source.get('relevance_score', 0)
                sources_text += f"**{i}. {product}** (Relevance: {score:.2f})\n"
                sources_text += f"> {text}\n\n"
        else:
            sources_text += "No specific sources retrieved.\n"
        
        # FIX: Update history in the correct format for Gradio Chatbot
        # Gradio expects list of tuples (user, bot)
        history = history or []
        history.append((question, answer))
        
        return answer, sources_text, history
        
    except Exception as e:
        logger.error(f"Error: {e}")
        error_msg = f"❌ Error: {str(e)}"
        return error_msg, "Check logs for details", history

def clear_conversation():
    """Clear all conversation history."""
    return "", "", "All Products", 0.7, []

# Custom CSS for better UI - FIXED white text issue
css = """
.gradio-container { 
    max-width: 1200px !important; 
    margin: auto !important; 
    background-color: white !important;
}
.chatbot { 
    min-height: 400px; 
}
.sources-box { 
    background-color: #f5f5f5; 
    padding: 15px; 
    border-radius: 8px; 
}
h1, h2, h3, p, label { 
    color: #333333 !important; 
}
h1 { 
    color: #1e3c72 !important; 
    font-weight: 700 !important;
}
footer {
    display: none !important;
}
.gradio-container .prose {
    color: #333333 !important;
}
.message {
    color: #333333 !important;
}
"""

# Create interface
with gr.Blocks(title="CrediTrust Complaint Analysis Chatbot", theme=gr.themes.Soft(), css=css) as demo:
    
    gr.Markdown("""
    # 🏦 CrediTrust Complaint Analysis Chatbot
    ### Ask questions about customer complaints to receive actionable insights
    """)
    
    with gr.Row():
        # Left column - Chat interface
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Conversation History", 
                height=400,
                bubble_full_width=False,
                avatar_images=(None, None)
            )
            
            with gr.Row():
                question_input = gr.Textbox(
                    label="Ask a question",
                    placeholder="What are the main issues with credit cards?",
                    lines=2,
                    scale=4,
                    container=True
                )
                submit_btn = gr.Button("🔍 Analyze", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
        
        # Right column - Controls and sources
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("### ⚙️ Filters")
                product_filter = gr.Dropdown(
                    choices=PRODUCTS,
                    value="All Products",
                    label="Filter by Product",
                    interactive=True
                )
                
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature (Creativity)",
                    interactive=True
                )
            
            # Answer output
            answer_output = gr.Textbox(
                label="📋 AI Analysis & Recommendations",
                lines=8,
                interactive=False,
                show_copy_button=True,
                container=True
            )
            
            # Sources output
            sources_output = gr.Markdown(
                label="📚 Evidence Sources",
                value="*Sources will appear here*",
                line_breaks=True,
                container=True
            )
    
    # State
    history_state = gr.State([])
    
    # Events
    submit_btn.click(
        fn=process_question,
        inputs=[question_input, history_state, product_filter, temperature],
        outputs=[answer_output, sources_output, history_state]
    ).then(
        fn=lambda: "",
        outputs=question_input
    ).then(
        fn=lambda h: h,
        inputs=history_state,
        outputs=chatbot
    )
    
    question_input.submit(
        fn=process_question,
        inputs=[question_input, history_state, product_filter, temperature],
        outputs=[answer_output, sources_output, history_state]
    ).then(
        fn=lambda: "",
        outputs=question_input
    ).then(
        fn=lambda h: h,
        inputs=history_state,
        outputs=chatbot
    )
    
    clear_btn.click(
        fn=clear_conversation,
        outputs=[question_input, answer_output, sources_output, product_filter, history_state]
    ).then(
        fn=lambda: [],
        outputs=chatbot
    )
    
    # Examples
    gr.Markdown("### 💡 Example Questions")
    with gr.Row():
        examples = [
            "What are common credit card complaints?",
            "Why do customers complain about personal loans?",
            "What issues occur with money transfers?",
            "How can we improve savings accounts?"
        ]
        for ex in examples:
            btn = gr.Button(ex, size="sm", variant="secondary")
            btn.click(
                fn=lambda q=ex: q,
                outputs=question_input
            )
    
    # Footer
    gr.Markdown("---\n*This chatbot uses real CFPB complaint data with semantic search.*")

# Launch
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )