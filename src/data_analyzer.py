"""
Exploratory Data Analysis for complaint data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List  # Fixed import
import logging

logger = logging.getLogger(__name__)


class DataAnalyzer:
    """Performs EDA on complaint data."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize analyzer with data.
        
        Args:
            df: DataFrame containing complaint data
        """
        self.df = df
        
    def analyze_missing_values(self) -> pd.DataFrame:
        """
        Analyze missing values in the dataset.
        
        Returns:
            DataFrame with missing value statistics
        """
        missing_values = self.df.isnull().sum()
        missing_percentage = (missing_values / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'column': missing_values.index,
            'missing_count': missing_values.values,
            'missing_percentage': missing_percentage.values
        })
        
        missing_df = missing_df.sort_values('missing_percentage', ascending=False)
        
        logger.info(f"Missing values analyzed. Total missing: {missing_values.sum():,}")
        return missing_df
    
    def analyze_product_distribution(self) -> Dict:
        """
        Analyze distribution of complaints across products.
        
        Returns:
            Dictionary with product statistics
        """
        if 'Product' not in self.df.columns:
            logger.warning("'Product' column not found")
            return {}
        
        product_counts = self.df['Product'].value_counts()
        product_percentages = (product_counts / len(self.df)) * 100
        
        # Top 10 products
        top_10 = product_counts.head(10)
        
        result = {
            "total_unique_products": self.df['Product'].nunique(),
            "top_10_products": top_10.to_dict(),
            "distribution_percentage": product_percentages.to_dict(),
            "most_common_product": product_counts.index[0] if len(product_counts) > 0 else None
        }
        
        logger.info(f"Product analysis complete. Found {result['total_unique_products']} unique products")
        return result
    
    def analyze_narrative_length(self, narrative_col: str = 'Consumer complaint narrative') -> Dict:
        """
        Analyze length of complaint narratives.
        
        Args:
            narrative_col: Name of narrative column
            
        Returns:
            Dictionary with narrative statistics
        """
        if narrative_col not in self.df.columns:
            # Try to find narrative column
            narrative_cols = [col for col in self.df.columns if 'narrative' in col.lower()]
            if not narrative_cols:
                logger.warning("No narrative column found")
                return {}
            narrative_col = narrative_cols[0]
        
        # Filter only rows with narratives
        has_narrative = self.df[narrative_col].notna()
        narrative_df = self.df[has_narrative].copy()
        
        if len(narrative_df) == 0:
            logger.warning("No narratives found")
            return {}
        
        # Calculate word counts
        narrative_df['word_count'] = narrative_df[narrative_col].apply(
            lambda x: len(str(x).split())
        )
        
        stats = {
            "total_with_narrative": has_narrative.sum(),
            "total_without_narrative": (~has_narrative).sum(),
            "percentage_with_narrative": (has_narrative.sum() / len(self.df)) * 100,
            "word_count_stats": {
                "min": narrative_df['word_count'].min(),
                "max": narrative_df['word_count'].max(),
                "mean": narrative_df['word_count'].mean(),
                "median": narrative_df['word_count'].median(),
                "std": narrative_df['word_count'].std()
            },
            "very_short_count": (narrative_df['word_count'] < 10).sum(),
            "very_long_count": (narrative_df['word_count'] > 500).sum()
        }
        
        logger.info(f"Narrative analysis complete. {stats['total_with_narrative']:,} narratives found")
        return stats
    
    def plot_product_distribution(self, top_n: int = 15, save_path: str = None):
        """
        Plot distribution of products.
        
        Args:
            top_n: Number of top products to show
            save_path: Optional path to save the plot
        """
        if 'Product' not in self.df.columns:
            return
        
        product_counts = self.df['Product'].value_counts()
        top_products = product_counts.head(top_n)
        
        plt.figure(figsize=(12, 6))
        bars = plt.barh(range(len(top_products)), top_products.values)
        plt.yticks(range(len(top_products)), top_products.index)
        plt.xlabel('Number of Complaints')
        plt.title(f'Top {top_n} Product Categories')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, top_products.values)):
            plt.text(value + max(top_products.values)*0.01, 
                    i, 
                    f'{value:,}', 
                    va='center',
                    fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_narrative_length_distribution(self, narrative_col: str = 'Consumer complaint narrative', 
                                          save_path: str = None):
        """
        Plot distribution of narrative lengths.
        
        Args:
            narrative_col: Name of narrative column
            save_path: Optional path to save the plot
        """
        if narrative_col not in self.df.columns:
            return
        
        # Filter only rows with narratives
        has_narrative = self.df[narrative_col].notna()
        narrative_df = self.df[has_narrative].copy()
        
        if len(narrative_df) == 0:
            return
        
        # Calculate word counts
        narrative_df['word_count'] = narrative_df[narrative_col].apply(
            lambda x: len(str(x).split())
        )
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(narrative_df['word_count'], bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(narrative_df['word_count'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {narrative_df["word_count"].mean():.0f}')
        axes[0].axvline(narrative_df['word_count'].median(), color='green', linestyle='--', 
                       label=f'Median: {narrative_df["word_count"].median():.0f}')
        axes[0].set_xlabel('Word Count')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Complaint Narrative Lengths')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(narrative_df['word_count'], vert=False)
        axes[1].set_xlabel('Word Count')
        axes[1].set_title('Box Plot of Narrative Lengths')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {save_path}")
        
        plt.show()