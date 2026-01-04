"""
Text chunking module for complaint narratives.
"""

import re
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class TextChunker:
    """Handles text chunking for complaint narratives."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or len(text.strip()) == 0:
            return []
        
        # If text is shorter than chunk size, return as single chunk
        if len(text) <= self.chunk_size:
            return [text.strip()]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If this isn't the last chunk, try to break at sentence boundary
            if end < len(text):
                # Look for a sentence ending or whitespace to break at
                sentence_enders = ['. ', '! ', '? ', '\n\n', '\n']
                for ender in sentence_enders:
                    end_pos = text.rfind(ender, start, end)
                    if end_pos != -1 and end_pos > start + self.chunk_size // 2:
                        end = end_pos + len(ender)
                        break
                else:
                    # No sentence ender found, break at last space
                    space_pos = text.rfind(' ', start, end)
                    if space_pos != -1 and space_pos > start + self.chunk_size // 2:
                        end = space_pos + 1
            
            # Extract chunk and clean it
            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            # Move start position for next chunk (with overlap)
            start = end - self.chunk_overlap
            
            # Prevent infinite loops
            if start <= 0 or start >= len(text):
                break
        
        return chunks
    
    def analyze_chunking_results(self, original_text: str, chunks: List[str]) -> Dict:
        """
        Analyze chunking results.
        
        Args:
            original_text: Original text
            chunks: Generated chunks
            
        Returns:
            Dictionary with analysis metrics
        """
        if not chunks:
            return {"error": "No chunks generated"}
        
        analysis = {
            "original_length_chars": len(original_text),
            "original_length_words": len(original_text.split()),
            "num_chunks": len(chunks),
            "chunk_sizes_chars": [len(chunk) for chunk in chunks],
            "chunk_sizes_words": [len(chunk.split()) for chunk in chunks],
            "avg_chunk_size_chars": sum(len(chunk) for chunk in chunks) / len(chunks),
            "avg_chunk_size_words": sum(len(chunk.split()) for chunk in chunks) / len(chunks),
            "total_chars_in_chunks": sum(len(chunk) for chunk in chunks),
            "overlap_percentage": (self.chunk_overlap / self.chunk_size * 100) if self.chunk_size > 0 else 0
        }
        
        # Calculate coverage (accounting for overlap)
        unique_chars = len(set(''.join(chunks)))
        analysis["coverage_percentage"] = (unique_chars / len(original_text) * 100) if len(original_text) > 0 else 0
        
        return analysis
    
    def demonstrate_chunking(self, sample_text: str) -> Tuple[List[str], Dict]:
        """
        Demonstrate chunking on sample text.
        
        Args:
            sample_text: Text to demonstrate chunking on
            
        Returns:
            Tuple of (chunks, analysis)
        """
        logger.info(f"Demonstrating chunking on text of {len(sample_text)} characters")
        
        chunks = self.chunk_text(sample_text)
        analysis = self.analyze_chunking_results(sample_text, chunks)
        
        return chunks, analysis