

import pandas as pd
import numpy as np
import re
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Configuration for data processing."""
    target_products: List[str] = None
    min_narrative_length: int = 10
    max_narrative_length: int = 10000
    data_path: str = "../data/raw/complaints.csv"
    output_path: str = "../data/processed/filtered_complaints.csv"
    report_path: str = "../data/processed/eda_report.md"
    
    def __post_init__(self):
        if self.target_products is None:
            self.target_products = [
                "Credit card", 
                "Personal loan", 
                "Savings account", 
                "Money transfers"
            ]

class ComplaintDataProcessor:
    """Complete data processor for EDA and preprocessing."""
    
    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        self.df = None
        self.clean_df = None
        self.stats = {}
        self.stop_words = set(stopwords.words('english'))
        
        # Common boilerplate phrases in complaints
        self.boilerplate_phrases = [
            'i am writing to file a complaint',
            'i am writing to complain',
            'this is a complaint about',
            'dear sir or madam',
            'to whom it may concern',
            'i would like to file a complaint',
            'i am submitting this complaint'
        ]
    
    def load_data(self) -> pd.DataFrame:
        """
        Load complaint data from CSV file.
        
        Returns:
            DataFrame containing complaint data
        """
        logger.info(f"Loading data from {self.config.data_path}")
        try:
            self.df = pd.read_csv(self.config.data_path, low_memory=False)
            logger.info(f"Loaded {len(self.df):,} records with {len(self.df.columns)} columns")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def perform_eda(self) -> Dict:
        """
        Perform exploratory data analysis.
        
        Returns:
            Dictionary with EDA statistics
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        logger.info("Performing Exploratory Data Analysis...")
        
        # Basic statistics
        self.stats['basic'] = {
            'total_records': len(self.df),
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.astype(str).to_dict(),
            'null_counts': self.df.isnull().sum().to_dict(),
            'duplicates': self.df.duplicated().sum()
        }
        
        # Product distribution
        if 'Product' in self.df.columns:
            product_dist = self.df['Product'].value_counts()
            self.stats['products'] = {
                'distribution': product_dist.to_dict(),
                'unique_products': self.df['Product'].nunique(),
                'top_10_products': product_dist.head(10).to_dict()
            }
        
        # Narrative analysis
        if 'Consumer complaint narrative' in self.df.columns:
            self._analyze_narratives()
        
        # Date analysis
        date_cols = [col for col in self.df.columns if 'date' in col.lower() or 'Date' in col]
        if date_cols:
            self._analyze_dates(date_cols)
        
        # Company analysis
        if 'Company' in self.df.columns:
            company_dist = self.df['Company'].value_counts()
            self.stats['companies'] = {
                'unique_companies': self.df['Company'].nunique(),
                'top_10_companies': company_dist.head(10).to_dict()
            }
        
        # Issue analysis
        if 'Issue' in self.df.columns:
            issue_dist = self.df['Issue'].value_counts()
            self.stats['issues'] = {
                'unique_issues': self.df['Issue'].nunique(),
                'top_10_issues': issue_dist.head(10).to_dict()
            }
        
        return self.stats
    
    def _analyze_narratives(self):
        """Analyze complaint narratives."""
        narratives = self.df['Consumer complaint narrative']
        
        # Count narratives
        with_narrative = narratives.notna().sum()
        without_narrative = narratives.isna().sum()
        
        # Calculate lengths for narratives that exist
        narrative_lengths = narratives.dropna().apply(
            lambda x: len(str(x).split()) if pd.notnull(x) else 0
        )
        
        self.stats['narratives'] = {
            'with_narrative': int(with_narrative),
            'without_narrative': int(without_narrative),
            'with_narrative_percent': (with_narrative / len(self.df)) * 100,
            'length_stats': {
                'mean': float(narrative_lengths.mean()),
                'median': float(narrative_lengths.median()),
                'min': int(narrative_lengths.min()),
                'max': int(narrative_lengths.max()),
                'std': float(narrative_lengths.std())
            },
            'length_quantiles': {
                'q25': float(narrative_lengths.quantile(0.25)),
                'q50': float(narrative_lengths.quantile(0.50)),
                'q75': float(narrative_lengths.quantile(0.75)),
                'q90': float(narrative_lengths.quantile(0.90)),
                'q95': float(narrative_lengths.quantile(0.95))
            }
        }
    
    def _analyze_dates(self, date_cols):
        """Analyze date columns."""
        for date_col in date_cols:
            try:
                self.df[date_col] = pd.to_datetime(self.df[date_col], errors='coerce')
                self.stats['dates'] = {
                    date_col: {
                        'min': self.df[date_col].min().strftime('%Y-%m-%d'),
                        'max': self.df[date_col].max().strftime('%Y-%m-%d'),
                        'null_count': int(self.df[date_col].isna().sum())
                    }
                }
            except:
                continue
    
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
        
        # Remove phone numbers
        text = re.sub(r'[\+\d\-\s]{10,}', '', text)
        
        # Remove special characters and digits (keep letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove boilerplate phrases
        for phrase in self.boilerplate_phrases:
            text = text.replace(phrase, '')
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        text = ' '.join(filtered_tokens)
        
        return text
    
    def filter_data(self) -> pd.DataFrame:
        """
        Filter data based on configuration.
        
        Returns:
            Filtered DataFrame
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        logger.info("Filtering data...")
        
        # Make a copy
        filtered_df = self.df.copy()
        initial_count = len(filtered_df)
        
        # 1. Filter by product
        if 'Product' in filtered_df.columns:
            mask = filtered_df['Product'].isin(self.config.target_products)
            filtered_df = filtered_df[mask]
            logger.info(f"After product filtering: {len(filtered_df):,} records")
        
        # 2. Remove empty narratives
        if 'Consumer complaint narrative' in filtered_df.columns:
            before_narrative = len(filtered_df)
            filtered_df = filtered_df[filtered_df['Consumer complaint narrative'].notna()]
            logger.info(f"Removed {before_narrative - len(filtered_df)} records with empty narratives")
        
        # 3. Clean narratives
        if 'Consumer complaint narrative' in filtered_df.columns:
            logger.info("Cleaning text narratives...")
            filtered_df['cleaned_narrative'] = filtered_df['Consumer complaint narrative'].apply(
                self.clean_text
            )
            
            # Remove narratives that became empty after cleaning
            before_cleaning = len(filtered_df)
            filtered_df = filtered_df[filtered_df['cleaned_narrative'].str.len() > 10]
            logger.info(f"Removed {before_cleaning - len(filtered_df)} records with empty cleaned narratives")
        
        # 4. Add word count column
        filtered_df['word_count'] = filtered_df['cleaned_narrative'].apply(
            lambda x: len(str(x).split())
        )
        
        # 5. Remove very short narratives
        filtered_df = filtered_df[filtered_df['word_count'] >= 5]
        
        logger.info(f"Final filtered dataset: {len(filtered_df):,} records")
        logger.info(f"Removed {initial_count - len(filtered_df):,} total records ({(initial_count - len(filtered_df))/initial_count*100:.1f}%)")
        
        self.clean_df = filtered_df
        return filtered_df
    
    def save_clean_data(self):
        """
        Save cleaned data to CSV.
        """
        if self.clean_df is None:
            raise ValueError("No cleaned data available. Call filter_data() first.")
        
        # Create directory if it doesn't exist
        Path(self.config.output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        self.clean_df.to_csv(self.config.output_path, index=False)
        logger.info(f"Saved cleaned data to {self.config.output_path}")
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive EDA report.
        
        Returns:
            Markdown formatted report
        """
        if self.clean_df is None:
            raise ValueError("No cleaned data available. Call filter_data() first.")
        
        report = []
        report.append("# Exploratory Data Analysis Report")
        report.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        report.append("")
        
        # Dataset Overview
        report.append("## 📊 Dataset Overview")
        report.append(f"- **Total original complaints**: {self.stats['basic']['total_records']:,}")
        report.append(f"- **Total after filtering**: {len(self.clean_df):,}")
        report.append(f"- **Retention rate**: {(len(self.clean_df)/self.stats['basic']['total_records'])*100:.1f}%")
        report.append(f"- **Columns in cleaned data**: {len(self.clean_df.columns)}")
        report.append("")
        
        # Product Distribution
        if 'products' in self.stats:
            report.append("## 📈 Product Distribution")
            report.append(f"- **Unique products in original data**: {self.stats['products']['unique_products']}")
            report.append("")
            report.append("**Target Products (after filtering):**")
            product_counts = self.clean_df['Product'].value_counts()
            for product, count in product_counts.items():
                percentage = (count / len(self.clean_df)) * 100
                report.append(f"- **{product}**: {count:,} complaints ({percentage:.1f}%)")
            report.append("")
        
        # Narrative Analysis
        if 'narratives' in self.stats:
            nar_stats = self.stats['narratives']
            report.append("## 📝 Narrative Analysis")
            report.append(f"- **Complaints with narratives**: {nar_stats['with_narrative']:,} ({nar_stats['with_narrative_percent']:.1f}%)")
            report.append(f"- **Complaints without narratives**: {nar_stats['without_narrative']:,}")
            report.append("")
            
            report.append("**Narrative Length Statistics (words):**")
            length_stats = nar_stats['length_stats']
            report.append(f"- **Mean**: {length_stats['mean']:.1f}")
            report.append(f"- **Median**: {length_stats['median']:.1f}")
            report.append(f"- **Minimum**: {length_stats['min']}")
            report.append(f"- **Maximum**: {length_stats['max']}")
            report.append(f"- **Standard Deviation**: {length_stats['std']:.1f}")
            report.append("")
            
            # Length distribution
            lengths = self.clean_df['word_count']
            bins = [0, 50, 100, 200, 500, 1000, float('inf')]
            labels = ['0-50', '51-100', '101-200', '201-500', '501-1000', '1000+']
            length_dist = pd.cut(lengths, bins=bins, labels=labels).value_counts()
            
            report.append("**Narrative Length Distribution:**")
            for label, count in length_dist.items():
                percentage = (count / len(lengths)) * 100
                report.append(f"- **{label} words**: {count:,} ({percentage:.1f}%)")
            report.append("")
        
        # Sample Data
        report.append("## 🔍 Sample Data")
        report.append("**First 3 cleaned complaints:**")
        sample = self.clean_df.head(3)
        for idx, row in sample.iterrows():
            report.append(f"\n**Complaint {idx}:**")
            report.append(f"- **Product**: {row.get('Product', 'N/A')}")
            report.append(f"- **Issue**: {row.get('Issue', 'N/A')}")
            report.append(f"- **Word Count**: {row.get('word_count', 'N/A')}")
            if 'cleaned_narrative' in row:
                preview = row['cleaned_narrative'][:150] + "..." if len(row['cleaned_narrative']) > 150 else row['cleaned_narrative']
                report.append(f"- **Cleaned Narrative**: `{preview}`")
        report.append("")
        
        # Key Findings
        report.append("## 🎯 Key Findings")
        report.append("1. **Data Quality**: The dataset contains a mix of structured and unstructured data.")
        report.append("2. **Narrative Coverage**: Approximately XX% of complaints include detailed narratives.")
        report.append("3. **Product Focus**: The filtered dataset focuses on our target financial products.")
        report.append("4. **Text Cleanliness**: The cleaning pipeline removes noise while preserving meaning.")
        report.append("")
        
        # Next Steps
        report.append("## 🚀 Next Steps")
        report.append("1. **Task 2**: Create text chunks and embeddings for RAG pipeline")
        report.append("2. **Task 3**: Build and evaluate the RAG system")
        report.append("3. **Task 4**: Develop interactive chatbot interface")
        
        return "\n".join(report)
    
    def save_report(self):
        """Save report to file."""
        report = self.generate_report()
        Path(self.config.report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Saved EDA report to {self.config.report_path}")

def main():
    """Main function to run the data processing pipeline."""
    processor = ComplaintDataProcessor()
    
    # Load data
    processor.load_data()
    
    # Perform EDA
    stats = processor.perform_eda()
    print(f"EDA completed. Statistics collected: {len(stats)} categories")
    
    # Filter and clean data
    clean_df = processor.filter_data()
    print(f"Filtered data shape: {clean_df.shape}")
    
    # Save results
    processor.save_clean_data()
    processor.save_report()
    
    print(f"\n✅ Task 1 completed!")
    print(f"📁 Cleaned data saved to: {processor.config.output_path}")
    print(f"📄 Report saved to: {processor.config.report_path}")
    
    return clean_df

if __name__ == "__main__":
    main()
