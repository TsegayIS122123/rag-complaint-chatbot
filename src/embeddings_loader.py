"""
Module to load and inspect the pre-built embeddings.
"""

import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class EmbeddingsLoader:
    """Handles loading and inspection of pre-built embeddings."""
    
    def __init__(self, embeddings_path: str = "../data/raw/complaint_embeddings.parquet"):
        """
        Initialize embeddings loader.
        
        Args:
            embeddings_path: Path to embeddings file
        """
        self.embeddings_path = embeddings_path
        self.df = None
        self.metadata = None
        
    def inspect_parquet_structure(self) -> Dict:
        """
        Inspect the structure of the parquet file without loading all data.
        
        Returns:
            Dictionary with file structure info
        """
        if not os.path.exists(self.embeddings_path):
            return {"error": f"File not found: {self.embeddings_path}"}
        
        try:
            # Read just the schema (fast)
            schema = pd.read_parquet(self.embeddings_path, engine='pyarrow').dtypes
            file_size_mb = os.path.getsize(self.embeddings_path) / (1024**2)
            
            # Try to get a small sample to understand structure
            sample = pd.read_parquet(self.embeddings_path, nrows=5)
            
            info = {
                "file_size_mb": file_size_mb,
                "shape": sample.shape,
                "columns": list(sample.columns),
                "dtypes": str(schema.to_dict()),
                "sample_head": sample.head(2).to_dict(orient='records')
            }
            
            logger.info(f"File structure: {info['shape']}, columns: {info['columns']}")
            return info
            
        except Exception as e:
            logger.error(f"Error inspecting parquet: {e}")
            return {"error": str(e)}
    
    def load_embeddings_metadata(self) -> Optional[pd.DataFrame]:
        """
        Load metadata from embeddings file.
        ChromaDB stores data in a specific format.
        """
        try:
            # Read the entire file
            self.df = pd.read_parquet(self.embeddings_path, engine='pyarrow')
            logger.info(f"Loaded embeddings. Shape: {self.df.shape}")
            
            # ChromaDB format: usually has 'id', 'document', 'embedding', 'metadata' columns
            if 'metadata' in self.df.columns:
                # Extract metadata from the struct column
                logger.info("Extracting metadata from ChromaDB format...")
                
                # Convert metadata to separate columns
                metadata_df = self.df['metadata'].apply(pd.Series)
                self.metadata = metadata_df
                
                # Combine with document text
                result_df = pd.DataFrame()
                if 'document' in self.df.columns:
                    result_df['document'] = self.df['document']
                
                result_df = pd.concat([result_df, metadata_df], axis=1)
                
                logger.info(f"Metadata extracted. Columns: {list(result_df.columns)}")
                return result_df
            else:
                logger.warning("No 'metadata' column found. Using raw dataframe.")
                return self.df
                
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return None
    
    def analyze_chunking(self, df: pd.DataFrame) -> Dict:
        """
        Analyze chunking strategy from metadata.
        
        Args:
            df: DataFrame with metadata
            
        Returns:
            Dictionary with chunking analysis
        """
        analysis = {}
        
        # Check for chunk-related columns
        chunk_cols = [col for col in df.columns if 'chunk' in col.lower()]
        analysis['chunk_columns_found'] = chunk_cols
        
        if 'chunk_index' in df.columns and 'total_chunks' in df.columns:
            analysis['total_chunks'] = len(df)
            analysis['unique_complaints'] = df['complaint_id'].nunique() if 'complaint_id' in df.columns else None
            analysis['avg_chunks_per_complaint'] = df['total_chunks'].mean()
            analysis['max_chunks'] = df['total_chunks'].max()
            analysis['min_chunks'] = df['total_chunks'].min()
        
        # Check product categories
        if 'product_category' in df.columns:
            product_counts = df['product_category'].value_counts()
            analysis['product_distribution'] = product_counts.head(10).to_dict()
            analysis['unique_products'] = df['product_category'].nunique()
        
        return analysis
    
    def create_stratified_sample(self, df: pd.DataFrame, sample_size: int = 15000) -> pd.DataFrame:
        """
        Create a stratified sample by product category.
        
        Args:
            df: Full metadata dataframe
            sample_size: Target sample size
            
        Returns:
            Stratified sample dataframe
        """
        if 'product_category' not in df.columns:
            logger.warning("No product_category column for stratified sampling")
            return df.sample(n=min(sample_size, len(df)), random_state=42)
        
        # Calculate proportional samples
        product_counts = df['product_category'].value_counts()
        total_records = len(df)
        
        sample_dfs = []
        remaining_size = sample_size
        
        for product, count in product_counts.items():
            product_df = df[df['product_category'] == product]
            
            # Calculate proportional sample size
            proportion = count / total_records
            product_sample_size = int(sample_size * proportion)
            
            # Ensure minimum samples
            product_sample_size = max(100, min(product_sample_size, len(product_df)))
            
            # Adjust if we're running out of sample size
            if product_sample_size > remaining_size:
                product_sample_size = remaining_size
            
            if product_sample_size > 0:
                sampled = product_df.sample(n=product_sample_size, random_state=42)
                sample_dfs.append(sampled)
                remaining_size -= product_sample_size
            
            if remaining_size <= 0:
                break
        
        # Combine all samples
        if sample_dfs:
            stratified_sample = pd.concat(sample_dfs, ignore_index=True)
            logger.info(f"Created stratified sample: {len(stratified_sample):,} records")
            return stratified_sample
        else:
            logger.warning("Could not create stratified sample, using random sample")
            return df.sample(n=min(sample_size, len(df)), random_state=42)