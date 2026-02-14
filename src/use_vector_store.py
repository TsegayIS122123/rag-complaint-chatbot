"""
Simple script to check and use existing vector store.
No building needed - just verify and use.
"""

import os
import chromadb
import logging
from sentence_transformers import SentenceTransformer
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreChecker:
    """Check if vector store exists and is usable."""
    
    def __init__(self, vector_store_path="vector_store/chroma_db"):
        self.vector_store_path = vector_store_path
        self.client = None
        self.collection = None
        
    def check_exists(self):
        """Check if vector store exists."""
        if not os.path.exists(self.vector_store_path):
            logger.error(f"❌ Vector store not found at: {self.vector_store_path}")
            return False
        
        # Check for ChromaDB files
        files = os.listdir(self.vector_store_path)
        logger.info(f"📂 Vector store directory contains: {files}")
        
        # Look for ChromaDB SQLite file
        has_chroma = any('chroma' in f.lower() or f.endswith('.sqlite3') for f in files)
        if has_chroma:
            logger.info("✅ ChromaDB files detected")
            return True
        else:
            logger.warning("⚠️ No ChromaDB files found")
            return False
    
    def connect(self):
        """Connect to vector store."""
        try:
            self.client = chromadb.PersistentClient(path=self.vector_store_path)
            
            # List collections
            collections = self.client.list_collections()
            logger.info(f"📚 Found collections: {[c.name for c in collections]}")
            
            if collections:
                self.collection = self.client.get_collection(collections[0].name)
                count = self.collection.count()
                logger.info(f"✅ Connected to collection '{collections[0].name}'")
                logger.info(f"   Total vectors: {count:,}")
                
                # Test query
                test_embedding = np.random.randn(384).tolist()
                results = self.collection.query(
                    query_embeddings=[test_embedding],
                    n_results=1
                )
                logger.info(f"✅ Test query successful")
                return True
            else:
                logger.error("❌ No collections found")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to connect: {e}")
            return False
    
    def get_info(self):
        """Get vector store info."""
        if self.collection:
            return {
                "path": self.vector_store_path,
                "collection": self.collection.name,
                "vectors": self.collection.count(),
                "metadata": self.collection.metadata
            }
        return None

if __name__ == "__main__":
    checker = VectorStoreChecker()
    if checker.check_exists():
        if checker.connect():
            info = checker.get_info()
            print("\n" + "="*60)
            print("✅ VECTOR STORE IS READY TO USE!")
            print("="*60)
            print(f"📍 Path: {info['path']}")
            print(f"📚 Collection: {info['collection']}")
            print(f"🔢 Vectors: {info['vectors']:,}")
            print("="*60)
        else:
            print("\n❌ Vector store exists but cannot connect")
    else:
        print("\n❌ No vector store found. You need to:")
        print("1. Download pre-built store, OR")
        print("2. Build sample store first")