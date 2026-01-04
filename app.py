"""
Task 4: Interactive Chat Interface for CrediTrust Complaint Analysis
Simple Gradio implementation that meets all requirements
"""

import gradio as gr
import time

# -----------------------------
# Mock RAG system
# -----------------------------
class MockRAG:
    """Mock RAG system that simulates the behavior of the real system."""

    def ask_question(self, question):
        question_lower = question.lower()

        responses = {
            "credit": {
                "answer": """**Analysis of Credit Card Complaints**

🔴 **Top Issues**
1. Incorrect late fees despite on-time payments
2. Slow fraud resolution
3. Poor customer service experience

✅ **Recommendations**
- Fix late-fee calculation logic
- Enforce 48-hour fraud SLA
- Improve agent training

📊 **Impact:** Up to 40% reduction in complaints""",
                "sources": [
                    "Complaint #1234: Charged late fee despite early payment",
                    "Complaint #5678: Fraud case unresolved for weeks",
                    "Complaint #9012: Long wait times for support"
                ]
            },
            "loan": {
                "answer": """**Personal Loan Complaint Analysis**

🔴 **Problems Identified**
1. Long approval times
2. Confusing documentation
3. Poor communication

✅ **Recommendations**
- Automate approval workflow
- Clear eligibility checklist
- Status notifications

📊 **Impact:** 30% faster processing""",
                "sources": [
                    "Complaint #3456: Application delayed 3 weeks",
                    "Complaint #7890: Conflicting document requests",
                    "Complaint #2345: No status updates"
                ]
            },
            "savings": {
                "answer": """**Savings Account Complaints**

🔴 **Key Issues**
1. App access failures
2. Interest rate mismatch
3. Unexpected fees

✅ **Recommendations**
- Improve system uptime
- Transparent interest display
- Fee alerts

📊 **Impact:** 25% satisfaction improvement""",
                "sources": [
                    "Complaint #4567: App unavailable for days",
                    "Complaint #8901: Lower APY than advertised",
                    "Complaint #1235: Hidden monthly fee"
                ]
            },
            "transfer": {
                "answer": """**Money Transfer Issues**

🔴 **Main Problems**
1. Failed transfers
2. Slow international processing
3. Hidden fees

✅ **Recommendations**
- Fix transaction bugs
- Optimize routing
- Show fees upfront

📊 **Impact:** 35% issue reduction""",
                "sources": [
                    "Complaint #6789: Transfer failed but money deducted",
                    "Complaint #0123: Transfer took 6 days",
                    "Complaint #3457: Unexpected $25 fee"
                ]
            }
        }

        if "credit" in question_lower or "card" in question_lower:
            return responses["credit"]
        elif "loan" in question_lower:
            return responses["loan"]
        elif "saving" in question_lower or "account" in question_lower:
            return responses["savings"]
        elif "transfer" in question_lower or "money" in question_lower:
            return responses["transfer"]
        else:
            return {
                "answer": """**General Complaint Insights**

🔴 **Cross-Cutting Issues**
1. Slow responses
2. Inconsistent information
3. System outages

✅ **Recommendations**
- 24-hour response SLA
- Central knowledge base
- Better monitoring

📊 **Impact:** 20–30% complaint reduction""",
                "sources": [
                    "Complaint #1111: Response took 5 days",
                    "Complaint #2222: Conflicting agent answers",
                    "Complaint #3333: System outage"
                ]
            }


rag_system = MockRAG()

# -----------------------------
# Core logic
# -----------------------------
def process_question(question, history):
    if not question.strip():
        return "", "", history
    
    response = rag_system.ask_question(question)
    answer = response["answer"]
    sources = response["sources"]

    # Format sources
    sources_text = "📚 **Sources Used:**\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(sources))
    
    # Simulate streaming answer (optional)
    streaming_answer = ""
    for char in answer:
        streaming_answer += char
        time.sleep(0.005)
    
    # Append to history in the format Gradio expects
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": streaming_answer})

    return streaming_answer, sources_text, history



def clear_conversation():
    return "", "", [], []


# -----------------------------
# UI
# -----------------------------
with gr.Blocks(title="CrediTrust Complaint Analysis Chatbot") as demo:

    gr.Markdown("""
    # 🏦 CrediTrust Complaint Analysis Chatbot
    *Ask questions about customer complaints to receive actionable insights*
    """)

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation History", height=350)

            question_input = gr.Textbox(
                label="Ask a question",
                placeholder="What are the main issues with credit cards?",
                lines=2
            )

            with gr.Row():
                submit_btn = gr.Button("🔍 Analyze", variant="primary")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")

        with gr.Column(scale=2):
            answer_output = gr.Textbox(
                label="📋 AI Analysis & Recommendations",
                lines=12,
                interactive=False
            )

            sources_output = gr.Textbox(
                label="📚 Evidence Sources",
                lines=8,
                interactive=False
            )

    history_state = gr.State([])

    submit_btn.click(
        fn=process_question,
        inputs=[question_input, history_state],
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
        outputs=[question_input, answer_output, sources_output, history_state]
    ).then(
        fn=lambda: [],
        outputs=chatbot
    )


# -----------------------------
# Launch
# -----------------------------
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
        css="""
        .gradio-container { max-width: 900px !important; }
        .chatbot { min-height: 300px; }
        """,
        show_error=True
    )
