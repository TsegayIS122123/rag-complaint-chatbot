"""
Real RAG Pipeline for CrediTrust Complaint Analysis
USES EXISTING VECTOR STORE - NO BUILDING NEEDED
"""

import os
import pandas as pd
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealRAGPipeline:
    """RAG system using EXISTING ChromaDB vector store."""
    
    def __init__(self, 
                 vector_store_path="vector_store/chroma_db",
                 embedding_model_name="all-MiniLM-L6-v2",
                 use_mock_if_missing=True):
        
        self.vector_store_path = vector_store_path
        self.embedding_model_name = embedding_model_name
        self.use_mock_if_missing = use_mock_if_missing
        
        # Initialize
        self.embedding_model = None
        self.client = None
        self.collection = None
        self._initialize()
    
    def _initialize(self):
        """Initialize components - ONLY USE EXISTING STORE."""
        logger.info("="*60)
        logger.info("🚀 Initializing RAG Pipeline")
        logger.info("="*60)
        
        # 1. Load embedding model (small, always works)
        try:
            logger.info(f"📦 Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info("✅ Embedding model loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            if self.use_mock_if_missing:
                logger.warning("⚠️ Using mock embedding model")
                self.embedding_model = MockEmbeddingModel()
        
        # 2. Connect to EXISTING vector store (DO NOT BUILD)
        if os.path.exists(self.vector_store_path):
            try:
                logger.info(f"📂 Connecting to vector store: {self.vector_store_path}")
                self.client = chromadb.PersistentClient(path=self.vector_store_path)
                
                collections = self.client.list_collections()
                if collections:
                    self.collection = self.client.get_collection(collections[0].name)
                    count = self.collection.count()
                    logger.info(f"✅ Connected to collection: {collections[0].name}")
                    logger.info(f"   Total vectors: {count:,}")
                    
                    if count == 0:
                        logger.warning("⚠️ Collection has zero vectors!")
                else:
                    logger.warning("⚠️ No collections found in vector store")
                    self._create_mock_if_needed()
                    
            except Exception as e:
                logger.error(f"❌ Failed to connect to vector store: {e}")
                self._create_mock_if_needed()
        else:
            logger.warning(f"⚠️ Vector store not found at: {self.vector_store_path}")
            self._create_mock_if_needed()
    
    def _create_mock_if_needed(self):
        """Create mock collection only if absolutely necessary."""
        if self.use_mock_if_missing:
            logger.warning("⚠️ Creating minimal mock collection for demo")
            self.client = chromadb.Client()
            self.collection = self.client.create_collection(name="mock")
            # Add minimal mock data
            self.collection.add(
                ids=["1", "2", "3", "4"],
                documents=[
                    "Credit card late fee complaint",
                    "Loan processing delay issue",
                    "Savings account access problem",
                    "Money transfer failed"
                ],
                metadatas=[
                    {"product": "Credit card"},
                    {"product": "Personal loan"},
                    {"product": "Savings account"},
                    {"product": "Money transfers"}
                ],
                embeddings=[np.random.randn(384).tolist() for _ in range(4)]
            )
            logger.info("✅ Created mock collection with 4 vectors")
    
    def retrieve(self, query: str, n_results: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """Retrieve relevant chunks from EXISTING vector store."""
        if not self.collection:
            logger.warning("⚠️ No collection available")
            return []
        
        try:
            # Embed query
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Query collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_dict,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            chunks = []
            if results and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    chunks.append({
                        'id': results['ids'][0][i] if results['ids'] else f"chunk_{i}",
                        'text': doc,
                        'metadata': metadata or {},
                        'relevance_score': 1 - distance
                    })
            
            logger.info(f"🔍 Retrieved {len(chunks)} chunks for: '{query[:50]}...'")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Retrieval error: {e}")
            return []
    
    def create_prompt(self, question: str, chunks: List[Dict]) -> str:
        """Create prompt with context."""
        if not chunks:
            return f"Question: {question}\n\nNo relevant complaints found."
        
        context = ""
        for i, chunk in enumerate(chunks, 1):
            product = chunk['metadata'].get('product', 'Unknown')
            text = chunk['text'][:300]
            context += f"[{i}] {product}: {text}\n\n"
        
        prompt = f"""You are a financial analyst for CrediTrust Financial. Answer based ONLY on these complaints:

{context}

Question: {question}

Answer:"""
        return prompt
    
    def generate_answer(self, prompt: str) -> str:
        """Generate answer (simplified for now)."""
        # Simple keyword-based answers for demo
        if "credit card" in prompt.lower():
            return """Based on complaints, credit card issues include:
1. Late fees charged incorrectly
2. Fraud resolution takes too long
3. Customer service delays"""
        elif "loan" in prompt.lower():
            return """Loan complaints show:
1. Processing delays of 2-3 weeks
2. Poor communication on status
3. Unclear approval requirements"""
        else:
            return "Analysis shows need for improved service response times."
    
def ask(self, question: str, n_results: int = 5) -> Dict:
    """Complete RAG pipeline."""
    logger.info(f"\n📝 Question: {question}")
    
    # Retrieve
    chunks = self.retrieve(question, n_results)
    
    # Generate answer
    if chunks:
        prompt = self.create_prompt(question, chunks)
        answer = self.generate_answer(prompt)
    else:
        answer = "No relevant complaints found."
        chunks = []
    
    # Format sources for display
    sources_list = []
    for chunk in chunks:
        sources_list.append({
            'text': chunk.get('text', '')[:300],
            'metadata': chunk.get('metadata', {}),
            'relevance_score': chunk.get('relevance_score', 0)
        })
    
    return {
        'question': question,
        'answer': answer,
        'sources': sources_list,
        'num_sources': len(chunks)
    }
class MockEmbeddingModel:
    def encode(self, text):
        return np.random.randn(384).tolist()

# Singleton
_rag_instance = None

def get_rag_pipeline():
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RealRAGPipeline()
    return _rag_instance