"""
Data loading module for CFPB complaints dataset.
"""

import pandas as pd
import numpy as np
import os
from typing import Optional, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComplaintDataLoader:
    """Handles loading of CFPB complaint data."""
    
    def __init__(self, data_dir: str = "../data"):
        """
        Initialize data loader.
        
        Args:
            data_dir: Path to data directory
        """
        self.data_dir = data_dir
        self.raw_path = os.path.join(data_dir, "raw")
        self.processed_path = os.path.join(data_dir, "processed")
        self.df = None
        
    def load_complaints_data(self, file_name: str = "complaints.csv") -> pd.DataFrame:
        """
        Load the main complaints CSV file.
        
        Args:
            file_name: Name of CSV file
            
        Returns:
            Loaded DataFrame
        """
        file_path = os.path.join(self.raw_path, file_name)
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading data from: {file_path}")
        
        # Load with optimized settings
        self.df = pd.read_csv(
            file_path,
            low_memory=False,
            parse_dates=['Date received'],
            infer_datetime_format=True
        )
        
        logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
        return self.df
    
    def load_embeddings_data(self) -> Optional[pd.DataFrame]:
        """
        Load pre-built embeddings.
        
        Returns:
            Embeddings DataFrame if exists
        """
        embeddings_path = os.path.join(self.raw_path, "complaint_embeddings.parquet")
        
        if os.path.exists(embeddings_path):
            logger.info(f"Loading embeddings from: {embeddings_path}")
            embeddings_df = pd.read_parquet(embeddings_path)
            logger.info(f"Embeddings loaded. Shape: {embeddings_df.shape}")
            return embeddings_df
        else:
            logger.warning("Embeddings file not found")
            return None
    
    def get_basic_info(self) -> Dict:
        """
        Get basic information about loaded data.
        
        Returns:
            Dictionary with basic info
        """
        if self.df is None:
            return {"error": "No data loaded"}
        
        info = {
            "shape": self.df.shape,
            "memory_mb": self.df.memory_usage().sum() / (1024 ** 2),
            "columns": list(self.df.columns),
            "dtypes": str(self.df.dtypes.value_counts().to_dict()),
            "date_range": None
        }
        
        # Add date range if date column exists
        date_cols = [col for col in self.df.columns if 'date' in col.lower()]
        if date_cols:
            date_col = date_cols[0]
            info["date_range"] = {
                "min": str(self.df[date_col].min()),
                "max": str(self.df[date_col].max())
            }
        
        return info
    
    def save_data(self, df: pd.DataFrame, file_name: str) -> str:
        """
        Save DataFrame to processed directory.
        
        Args:
            df: DataFrame to save
            file_name: Output file name
            
        Returns:
            Path where file was saved
        """
        os.makedirs(self.processed_path, exist_ok=True)
        output_path = os.path.join(self.processed_path, file_name)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to: {output_path}")
        
        return output_path