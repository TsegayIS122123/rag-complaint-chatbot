"""
Text cleaning module for complaint narratives.
"""

import re
import pandas as pd
from typing import Optional, Callable, Dict  # Added Dict import
import logging

logger = logging.getLogger(__name__)


class DataCleaner:
    """Handles cleaning of complaint narrative text."""
    
    # Common boilerplate phrases to remove
    BOILERPLATE_PHRASES = [
        r'i am writing to (file|submit) (a|this) complaint',
        r'this (is|letter) (is )?to file (a|my) complaint',
        r'dear (sir|madam|customer service)',
        r'to whom it may concern',
        r'sincerely,?\s*\w+',
        r'respectfully,?\s*\w+',
        r'best regards,?\s*\w+',
        r'thank you (for your (time|attention)|in advance)',
        r'please (find|see) (attached|below)',
        r'cc:|cc :|cc\s*\w+@\w+\.\w+'
    ]
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize cleaner with data.
        
        Args:
            df: DataFrame containing complaint data
        """
        self.df = df
        self.cleaned_df = None
        
    def clean_text(self, text: str) -> str:
        """
        Clean a single text string.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # 1. Convert to lowercase
        text = text.lower()
        
        # 2. Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # 3. Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # 4. Remove phone numbers
        text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '', text)
        
        # 5. Remove special characters (keep basic punctuation)
        text = re.sub(r'[^\w\s.,!?\'"-]', ' ', text)
        
        # 6. Remove boilerplate phrases
        for phrase in self.BOILERPLATE_PHRASES:
            text = re.sub(phrase, '', text, flags=re.IGNORECASE)
        
        # 7. Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 8. Remove very short texts
        if len(text.split()) < 5:
            return ""
        
        return text
    
    def clean_narrative_column(self, narrative_col: str = 'Consumer complaint narrative', 
                              output_col: str = 'cleaned_narrative') -> pd.DataFrame:
        """
        Clean a narrative column in the DataFrame.
        
        Args:
            narrative_col: Name of input narrative column
            output_col: Name of output cleaned column
            
        Returns:
            DataFrame with cleaned narrative
        """
        if narrative_col not in self.df.columns:
            # Try to find narrative column
            narrative_cols = [col for col in self.df.columns if 'narrative' in col.lower()]
            if not narrative_cols:
                logger.warning("No narrative column found")
                return self.df
            narrative_col = narrative_cols[0]
        
        logger.info(f"Cleaning column: {narrative_col}")
        
        # Store original
        self.df['original_narrative'] = self.df[narrative_col].copy()
        
        # Apply cleaning
        self.df[output_col] = self.df[narrative_col].apply(self.clean_text)
        
        # Count results
        original_lengths = self.df['original_narrative'].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )
        cleaned_lengths = self.df[output_col].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )
        
        empty_after = (self.df[output_col] == '').sum()
        
        logger.info(f"Cleaning complete:")
        logger.info(f"  • Total cleaned: {len(self.df):,}")
        logger.info(f"  • Average words before: {original_lengths.mean():.1f}")
        logger.info(f"  • Average words after: {cleaned_lengths.mean():.1f}")
        logger.info(f"  • Empty after cleaning: {empty_after:,}")
        
        self.cleaned_df = self.df
        return self.cleaned_df
    
    def remove_empty_narratives(self, narrative_col: str = 'cleaned_narrative') -> pd.DataFrame:
        """
        Remove rows with empty narratives after cleaning.
        
        Args:
            narrative_col: Name of cleaned narrative column
            
        Returns:
            DataFrame with empty narratives removed
        """
        if self.cleaned_df is None:
            df_to_filter = self.df
        else:
            df_to_filter = self.cleaned_df
        
        if narrative_col not in df_to_filter.columns:
            logger.warning(f"Column '{narrative_col}' not found")
            return df_to_filter
        
        before_count = len(df_to_filter)
        df_filtered = df_to_filter[df_to_filter[narrative_col] != ''].copy()
        
        logger.info(f"Removing empty narratives: {before_count:,} → {len(df_filtered):,} records")
        
        self.cleaned_df = df_filtered
        return self.cleaned_df
    
    def get_cleaning_stats(self) -> Dict:
        """
        Get statistics about cleaning operations.
        
        Returns:
            Dictionary with cleaning statistics
        """
        if self.cleaned_df is None:
            return {"error": "No cleaning performed"}
        
        stats = {
            "original_count": len(self.df),
            "cleaned_count": len(self.cleaned_df),
            "records_removed": len(self.df) - len(self.cleaned_df),
            "reduction_percentage": ((len(self.df) - len(self.cleaned_df)) / len(self.df)) * 100
        }
        
        return stats
    
    def get_cleaned_data(self) -> pd.DataFrame:
        """
        Get the cleaned DataFrame.
        
        Returns:
            Cleaned DataFrame
        """
        if self.cleaned_df is not None:
            return self.cleaned_df
        return self.df