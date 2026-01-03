"""
Data filtering module for complaint data.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict  # Added Dict import
import logging

logger = logging.getLogger(__name__)


class DataFilter:
    """Handles filtering of complaint data."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize filter with data.
        
        Args:
            df: DataFrame to filter
        """
        self.df = df
        self.filtered_df = None
        
    def filter_by_products(self, target_products: List[str]) -> pd.DataFrame:
        """
        Filter data to include only target products.
        
        Args:
            target_products: List of product names to include
            
        Returns:
            Filtered DataFrame
        """
        if 'Product' not in self.df.columns:
            logger.warning("'Product' column not found")
            return self.df
        
        original_count = len(self.df)
        
        # Case-insensitive filtering
        product_mask = self.df['Product'].str.lower().isin(
            [p.lower() for p in target_products]
        )
        
        self.filtered_df = self.df[product_mask].copy()
        
        logger.info(f"Product filtering: {original_count:,} → {len(self.filtered_df):,} records")
        
        # Show distribution
        if len(self.filtered_df) > 0:
            product_counts = self.filtered_df['Product'].value_counts()
            logger.info("Filtered product distribution:")
            for product, count in product_counts.items():
                percentage = count / len(self.filtered_df) * 100
                logger.info(f"  • {product}: {count:,} ({percentage:.1f}%)")
        
        return self.filtered_df
    
    def filter_by_narrative_presence(self, narrative_col: str = 'Consumer complaint narrative') -> pd.DataFrame:
        """
        Filter out records without narratives.
        
        Args:
            narrative_col: Name of narrative column
            
        Returns:
            Filtered DataFrame
        """
        if self.filtered_df is None:
            df_to_filter = self.df
        else:
            df_to_filter = self.filtered_df
        
        if narrative_col not in df_to_filter.columns:
            # Try to find narrative column
            narrative_cols = [col for col in df_to_filter.columns if 'narrative' in col.lower()]
            if not narrative_cols:
                logger.warning("No narrative column found")
                return df_to_filter
            narrative_col = narrative_cols[0]
        
        before_count = len(df_to_filter)
        has_narrative_before = df_to_filter[narrative_col].notna().sum()
        
        # Filter out rows without narratives
        df_filtered = df_to_filter[df_to_filter[narrative_col].notna()].copy()
        
        has_narrative_after = df_filtered[narrative_col].notna().sum()
        
        logger.info(f"Narrative filtering: {before_count:,} → {len(df_filtered):,} records")
        logger.info(f"Records with narrative: {has_narrative_before:,} → {has_narrative_after:,}")
        
        self.filtered_df = df_filtered
        return self.filtered_df
    
    def get_filtered_data(self) -> pd.DataFrame:
        """
        Get the filtered DataFrame.
        
        Returns:
            Filtered DataFrame or original if no filtering applied
        """
        if self.filtered_df is not None:
            return self.filtered_df
        return self.df
    
    def get_filtering_stats(self) -> Dict:
        """
        Get statistics about filtering operations.
        
        Returns:
            Dictionary with filtering statistics
        """
        stats = {
            "original_count": len(self.df),
            "filtered_count": len(self.filtered_df) if self.filtered_df is not None else len(self.df),
            "records_removed": len(self.df) - (len(self.filtered_df) if self.filtered_df is not None else len(self.df)),
            "reduction_percentage": 0
        }
        
        if stats["original_count"] > 0:
            stats["reduction_percentage"] = (
                stats["records_removed"] / stats["original_count"] * 100
            )
        
        return stats