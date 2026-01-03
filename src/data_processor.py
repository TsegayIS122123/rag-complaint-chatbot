"""
Data processing module for cleaning and preparing complaint data.
"""

import pandas as pd
import numpy as np
import re
from typing import List, Tuple, Optional
import logging
from dataclasses import dataclass
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Configuration for data processing."""
    target_products: List[str] = None
    min_narrative_length: int = 10
    max_narrative_length: int = 10000
    
    def __post_init__(self):
        if self.target_products is None:
            self.target_products = [
                "Credit card", 
                "Personal loan", 
                "Savings account", 
                "Money transfers"
            ]

class ComplaintDataProcessor:
    """Process and clean complaint data for RAG pipeline."""
    
    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        self.df = None
        self.clean_df = None
    
    def clean_text(self, text: str) -> str:
        """
        Clean individual text narrative.
        
        Args:
            text: Raw complaint narrative
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S*@\S*\s?', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data based on configuration.
        
        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()
        
        # 1. Filter by product
        if 'Product' in filtered_df.columns:
            mask = filtered_df['Product'].isin(self.config.target_products)
            filtered_df = filtered_df[mask]
        
        # 2. Remove empty narratives
        if 'Consumer complaint narrative' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Consumer complaint narrative'].notna()]
        
        return filtered_df
