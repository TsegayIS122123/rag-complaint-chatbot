# Intelligent Complaint Analysis for Financial Services

A **Retrieval-Augmented Generation (RAG)** powered chatbot that transforms unstructured customer complaints into **actionable insights** for internal teams at **CrediTrust Financial**.

---

## 🎯 Business Objective

CrediTrust Financial operates a mobile-first digital finance platform serving East African markets. With hundreds of thousands of users across multiple countries, the company receives **thousands of customer complaints every month** across products such as Credit Cards, Personal Loans, Savings Accounts, and Money Transfers.

Currently, valuable customer feedback is locked inside long, unstructured narratives, making it difficult for Product, Support, and Compliance teams to identify trends quickly.

This project delivers an **internal AI intelligence platform** that enables stakeholders to ask **natural-language questions** and receive **evidence-backed answers** in seconds.

---

## 🎯 Strategic Goals

### 1️⃣ Accelerate Insight Generation
- Reduce complaint analysis time from **days to minutes**
- Enable rapid detection of emerging product issues
- Free Product Managers from manual complaint reviews

### 2️⃣ Democratize Data Access
- Enable **self-service analytics** for non-technical teams
- Reduce dependency on data analysts
- Improve decision-making speed across departments

### 3️⃣ Enable Proactive Operations
- Identify recurring issues before escalation
- Support compliance and risk monitoring
- Shift from reactive to **data-driven proactive action**

---

## 🔧 Core Capabilities

### 🔍 Intelligent Retrieval
- Semantic search over complaint narratives
- Multi-product and metadata-based filtering
- Evidence-based retrieval with traceable sources

### 🤖 AI-Powered Analysis
- Trend identification across products and regions
- Root-cause analysis from customer narratives
- Concise, grounded answers generated using real complaints

### 👥 User-Focused Design
- Natural language query interface
- Transparent source attribution
- Simple UI for Product, Support, and Compliance teams

---

## 🏦 Financial Products Covered

| Product Category | Common Issues Analyzed |
|------------------|------------------------|
| Credit Cards | Billing disputes, fraud alerts, APR issues |
| Personal Loans | Approval delays, repayment confusion |
| Savings Accounts | Withdrawal problems, balance errors |
| Money Transfers | Failed transactions, delays, FX issues |

---

## 🧠 How the RAG System Works

1. **User Question**  
   A user asks a question in plain English (e.g., *“Why are customers unhappy with credit cards?”*).

2. **Semantic Retrieval**  
   The question is embedded and compared against a **vector database (ChromaDB/FAISS)** containing complaint text embeddings.

3. **Context Assembly**  
   The most relevant complaint excerpts are retrieved and combined into a structured prompt.

4. **Answer Generation**  
   An open-source LLM generates a concise, grounded response **using only retrieved evidence**.

5. **Source Transparency**  
   Retrieved complaint excerpts are displayed alongside the answer to build trust.

---

## 🛠️ Technology Stack

- **Language**: Python
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Database**: ChromaDB / FAISS
- **LLM**: Open-source models (LLaMA, Mistral)
- **RAG Framework**: LangChain
- **UI**: Gradio / Streamlit
- **CI/CD**: GitHub Actions
- **Testing**: Pytest

---
## 📊 Task 1: EDA & Preprocessing 

**Key Findings:**
- Analyzed 50,000 complaint records (sample)
- 92.5% of complaints are about "Credit reporting"
- Only 1.1% about Credit Cards, 0.7% about Savings Accounts
- **Critical Issue**: 98.7% of complaints lack narrative text!
- Filtered dataset: 50 records with narratives for target products

- Performed fast EDA on a **50,000-record sample** from a large-scale complaints dataset (>5GB)
- Analyzed key fields:
  - Date received
  - Product
  - Issue
  - Consumer complaint narrative
  - Company
  - State
- Identified severe class imbalance and missing narratives (~98%)
- Filtered dataset to **target financial products**:
  - Credit cards
  - Personal loans
  - Savings accounts
  - Money transfers
- Removed empty complaint narratives
- Applied basic text cleaning
- Generated a clean, RAG-ready dataset

**Final Output**
- `data/processed/filtered_complaints.csv` (50 records)
- `data/processed/filtered_complaints_sample.csv`

**Key Insights**
- Complaint data is highly imbalanced by product
- Very few records contain usable narrative text
- Filtering and preprocessing are critical before embedding

➡️ **Next Step**: Task 2 – Text chunking, embeddings, and vector store creation
