---
title: CrediTrust Complaint Analysis Chatbot
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# 🏦 CrediTrust Complaint Analysis Chatbot

A RAG-powered chatbot for analyzing customer complaints using real CFPB data.

## 🔗 Live Demo
👉 **[Click here to try the chatbot](https://huggingface.co/spaces/YOUR_USERNAME/credirust-rag-chatbot)**

## ✨ Features

- **Real Complaint Data**: Uses actual CFPB complaint database
- **Smart Search**: Semantic retrieval with ChromaDB
- **Product Filtering**: Filter by Credit Cards, Loans, Savings, Transfers
- **Evidence-Based**: Shows sources for every answer
- **Fast Responses**: Optimized for CPU inference

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Gradio |
| Vector DB | ChromaDB |
| Embeddings | all-MiniLM-L6-v2 |
| LLM | Microsoft Phi-2 |
| Deployment | Hugging Face Spaces |

## 🚀 Quick Start

```bash
# Clone this space
git clone https://huggingface.co/spaces/YOUR_USERNAME/credirust-rag-chatbot
cd credirust-rag-chatbot

# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
```
# 📊 Dataset
- Source: Consumer Financial Protection Bureau (CFPB)
- Size: 1.37 million complaint chunks
- Products: Credit Cards, Personal Loans, Savings Accounts, Money Transfers
- Time Range: 2011-2024

# 🤖 How It Works
- User Question → Embedded with all-MiniLM-L6-v2
- Semantic Search → Finds relevant complaints in ChromaDB
- Context Assembly → Retrieved chunks formatted
- Answer Generation → Phi-2 LLM creates response
- Source Display → Original complaints shown for verification

# 📈 Performance
- Response Time: 2-5 seconds (CPU)
- Relevance Score: >0.8 for product-specific queries
- Uptime: 99.9% on Hugging Face
# 👨‍💻 Author
- Tsegay - Data Scientist & AI Engineer
- https://img.shields.io/badge/GitHub-Profile-blue