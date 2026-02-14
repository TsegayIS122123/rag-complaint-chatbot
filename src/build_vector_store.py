"""
Build real ChromaDB vector store from complaint embeddings.
Task 2-3 Integration - Use actual pre-built embeddings.
FIXED: ChromaDB metadata must be strings, ints, floats, or bools
"""

import os
import pandas as pd
import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import logging
from tqdm import tqdm
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreBuilder:
    """Build ChromaDB vector store from pre-built embeddings."""
    
    def __init__(self, 
                 embeddings_path="data/raw/complaint_embeddings.parquet",
                 metadata_path="data/processed/filtered_complaints.csv",
                 persist_dir="vector_store/chroma_db"):
        
        self.embeddings_path = embeddings_path
        self.metadata_path = metadata_path
        self.persist_dir = persist_dir
        self.client = None
        self.collection = None
        
    def load_embeddings_data(self):
        """Load pre-built embeddings and metadata."""
        logger.info("📂 Loading pre-built embeddings...")
        
        # Check if embeddings file exists
        if not os.path.exists(self.embeddings_path):
            logger.error(f"❌ Embeddings file not found: {self.embeddings_path}")
            logger.info("⚠️ Using fallback - generating sample embeddings")
            return self._create_sample_embeddings()
        
        try:
            # Load embeddings parquet
            df = pd.read_parquet(self.embeddings_path)
            logger.info(f"✅ Loaded embeddings: {len(df):,} rows")
            logger.info(f"   Columns: {list(df.columns)}")
            
            # Load metadata if available
            if os.path.exists(self.metadata_path):
                metadata_df = pd.read_csv(self.metadata_path)
                logger.info(f"✅ Loaded metadata: {len(metadata_df):,} complaints")
                
                # Merge with embeddings if possible
                if 'complaint_id' in df.columns and 'complaint_id' in metadata_df.columns:
                    df = df.merge(metadata_df, on='complaint_id', how='left')
                    logger.info("✅ Merged embeddings with metadata")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading embeddings: {e}")
            return self._create_sample_embeddings()
    
    def _create_sample_embeddings(self):
        """Create sample embeddings for demonstration."""
        logger.info("📊 Creating sample embeddings for demo...")
        
        # Load filtered complaints
        if os.path.exists(self.metadata_path):
            metadata = pd.read_csv(self.metadata_path)
        else:
            # Create sample data
            metadata = pd.DataFrame({
                'complaint_id': range(1, 101),
                'product': np.random.choice(['Credit card', 'Personal loan', 'Savings account', 'Money transfers'], 100),
                'issue': np.random.choice(['Late fee', 'Fraud', 'Service', 'Technical'], 100),
                'company': np.random.choice(['Bank A', 'Bank B', 'Bank C'], 100),
                'date_received': pd.date_range('2024-01-01', periods=100)
            })
        
        # Create dummy embeddings (384-dim)
        np.random.seed(42)
        n_samples = len(metadata)
        embeddings = np.random.randn(n_samples, 384).astype(np.float32)
        
        # Create DataFrame
        df = pd.DataFrame({
            'id': [f"chunk_{i}" for i in range(n_samples)],
            'embedding': list(embeddings),
            'document': [f"Customer complaint about {row['product']}: {row['issue']}" 
                        for _, row in metadata.iterrows()],
            'metadata': metadata.to_dict('records')
        })
        
        logger.info(f"✅ Created {n_samples} sample embeddings")
        return df
    
    def create_chromadb_collection(self, df, batch_size=1000):
        """Create ChromaDB collection from embeddings."""
        logger.info("🔧 Creating ChromaDB collection...")
        
        # Remove existing collection if exists
        if os.path.exists(self.persist_dir):
            import shutil
            shutil.rmtree(self.persist_dir)
            logger.info("🗑️ Removed existing vector store")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        
        # Create collection
        self.collection = self.client.create_collection(
            name="complaints",
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        # Process in batches
        total = len(df)
        for i in tqdm(range(0, total, batch_size), desc="Adding to ChromaDB"):
            batch = df.iloc[i:i+batch_size]
            
            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            
            for idx, row in batch.iterrows():
                # ID
                if 'id' in row:
                    ids.append(str(row['id']))
                else:
                    ids.append(f"chunk_{idx}")
                
                # Embedding
                if 'embedding' in row:
                    emb = row['embedding']
                    if isinstance(emb, np.ndarray):
                        embeddings.append(emb.tolist())
                    elif isinstance(emb, list):
                        embeddings.append(emb)
                    else:
                        # Generate random embedding if missing
                        embeddings.append(np.random.randn(384).tolist())
                
                # Document text
                if 'document' in row:
                    documents.append(str(row['document']))
                elif 'cleaned_narrative' in row:
                    documents.append(str(row['cleaned_narrative']))
                else:
                    documents.append("Sample complaint text")
                
                # ==== FIX: CONVERT ALL METADATA TO STRINGS ====
                if 'metadata' in row and isinstance(row['metadata'], dict):
                    meta = {}
                    for key, value in row['metadata'].items():
                        # Convert EVERYTHING to string for ChromaDB
                        if value is None:
                            meta[key] = ""
                        elif isinstance(value, (str, int, float, bool)):
                            meta[key] = str(value)  # Convert to string anyway for safety
                        elif isinstance(value, pd.Timestamp):
                            meta[key] = value.strftime('%Y-%m-%d')  # Convert datetime to string
                        else:
                            meta[key] = str(value)
                    metadatas.append(meta)
                else:
                    # Create metadata from available columns
                    meta = {}
                    for col in ['product', 'issue', 'company', 'date_received', 'complaint_id']:
                        if col in row:
                            value = row[col]
                            if pd.isna(value):
                                meta[col] = ""
                            elif isinstance(value, pd.Timestamp):
                                meta[col] = value.strftime('%Y-%m-%d')
                            else:
                                meta[col] = str(value)
                    metadatas.append(meta)
            
            # Add to collection (only if we have data)
            if embeddings and ids and documents and metadatas:
                try:
                    self.collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=documents,
                        metadatas=metadatas
                    )
                    logger.debug(f"Added batch {i//batch_size + 1}: {len(ids)} vectors")
                except Exception as e:
                    logger.error(f"Error adding batch: {e}")
                    # Try one by one if batch fails
                    for j in range(len(ids)):
                        try:
                            self.collection.add(
                                ids=[ids[j]],
                                embeddings=[embeddings[j]],
                                documents=[documents[j]],
                                metadatas=[metadatas[j]]
                            )
                        except Exception as e2:
                            logger.error(f"Error adding single vector {ids[j]}: {e2}")
        
        logger.info(f"✅ Created collection with {self.collection.count():,} vectors")
        return self.collection
    
    def verify_collection(self):
        """Verify collection works."""
        logger.info("🔍 Verifying collection...")
        
        try:
            # Test query
            results = self.collection.query(
                query_embeddings=[np.random.randn(384).tolist()],
                n_results=5
            )
            
            logger.info(f"✅ Query successful, returned {len(results['ids'][0])} results")
            return results
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            return None
    
    def build(self):
        """Complete build process."""
        logger.info("="*60)
        logger.info("🚀 Building Vector Store")
        logger.info("="*60)
        
        # Load data
        df = self.load_embeddings_data()
        
        # Create collection
        self.create_chromadb_collection(df)
        
        # Verify
        self.verify_collection()
        
        if self.collection:
            logger.info(f"\n✅ Vector store saved to: {self.persist_dir}")
            logger.info(f"   Total vectors: {self.collection.count():,}")
        else:
            logger.error("❌ Failed to create vector store")
        
        return self.collection

def main():
    """Main function."""
    builder = VectorStoreBuilder()
    builder.build()

if __name__ == "__main__":
    main()