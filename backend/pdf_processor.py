"""PDF text extraction and processing utilities with page-aware chunking."""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pdfplumber

from backend.exceptions import DocumentError

logger = logging.getLogger(__name__)

# Sentence boundary pattern for respecting sentence boundaries during chunking
SENTENCE_BOUNDARY_PATTERN = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text content
        
    Raises:
        DocumentError: If PDF extraction fails
    """
    try:
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")
                    continue
        
        if not text_parts:
            raise DocumentError("No text could be extracted from the PDF")
        
        full_text = "\n\n".join(text_parts)
        return clean_text(full_text)
    
    except Exception as e:
        error_msg = str(e)
        if "PDF" in error_msg or "corrupted" in error_msg.lower() or "invalid" in error_msg.lower():
            raise DocumentError(f"Invalid or corrupted PDF file: {e}") from e
        raise DocumentError(f"Failed to extract text from PDF: {e}") from e


def extract_text_from_bytes(file_content: bytes) -> str:
    """Extract text from PDF content as bytes.
    
    Args:
        file_content: PDF file content as bytes
        
    Returns:
        Extracted text content
        
    Raises:
        DocumentError: If PDF extraction fails
    """
    try:
        # Validate PDF header
        if not file_content.startswith(b'%PDF'):
            raise DocumentError("File does not appear to be a valid PDF")
        
        text_parts = []
        # Wrap bytes in BytesIO to create a file-like object
        pdf_file = io.BytesIO(file_content)
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")
                    continue
        
        if not text_parts:
            raise DocumentError("No text could be extracted from the PDF")
        
        full_text = "\n\n".join(text_parts)
        return clean_text(full_text)
    
    except DocumentError:
        raise
    except Exception as e:
        error_msg = str(e)
        if "PDF" in error_msg or "corrupted" in error_msg.lower() or "invalid" in error_msg.lower():
            raise DocumentError(f"Invalid or corrupted PDF file: {e}") from e
        raise DocumentError(f"Failed to extract text from PDF: {e}") from e


def extract_text_with_pages(file_content: bytes) -> List[Dict[str, Any]]:
    """Extract text from PDF with page information preserved.
    
    Args:
        file_content: PDF file content as bytes
        
    Returns:
        List of dicts with 'page' (int) and 'text' (str) keys
        
    Raises:
        DocumentError: If PDF extraction fails
    """
    try:
        # Validate PDF header
        if not file_content.startswith(b'%PDF'):
            raise DocumentError("File does not appear to be a valid PDF")
        
        pages_data = []
        pdf_file = io.BytesIO(file_content)
        
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        cleaned = clean_text(page_text)
                        if cleaned.strip():
                            pages_data.append({
                                'page': page_num,
                                'text': cleaned
                            })
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")
                    continue
        
        if not pages_data:
            raise DocumentError("No text could be extracted from the PDF")
        
        return pages_data
    
    except DocumentError:
        raise
    except Exception as e:
        error_msg = str(e)
        if "PDF" in error_msg or "corrupted" in error_msg.lower() or "invalid" in error_msg.lower():
            raise DocumentError(f"Invalid or corrupted PDF file: {e}") from e
        raise DocumentError(f"Failed to extract text from PDF: {e}") from e


def clean_text(text: str) -> str:
    """Clean and normalize extracted text.
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        cleaned_line = ' '.join(line.split())
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
    
    return '\n'.join(cleaned_lines)


def _split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs based on double newlines or significant breaks.
    
    Args:
        text: Text to split
        
    Returns:
        List of paragraph strings
    """
    # Split on double newlines (paragraph boundaries)
    paragraphs = re.split(r'\n\s*\n', text)
    
    # Clean and filter empty paragraphs
    result = []
    for para in paragraphs:
        cleaned = para.strip()
        if cleaned:
            result.append(cleaned)
    
    return result


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences respecting sentence boundaries.
    
    Args:
        text: Text to split
        
    Returns:
        List of sentences
    """
    # Split on sentence boundaries
    sentences = SENTENCE_BOUNDARY_PATTERN.split(text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 400) -> List[Dict[str, Any]]:
    """Split text into chunks with overlap, respecting paragraph and sentence boundaries.
    
    Uses paragraph-aware chunking that:
    1. Splits on paragraph boundaries first
    2. Respects sentence boundaries within paragraphs
    3. Maintains specified overlap between chunks
    
    Args:
        text: Text to chunk
        chunk_size: Target size of each chunk in characters (default: 1500)
        overlap: Number of characters to overlap between chunks (default: 400)
        
    Returns:
        List of chunk dictionaries with 'text' and 'metadata' keys
    """
    if not text:
        return []
    
    # Split into paragraphs first
    paragraphs = _split_into_paragraphs(text)
    
    chunks = []
    current_chunk_parts = []
    current_length = 0
    chunk_index = 0
    
    for para in paragraphs:
        para_length = len(para)
        
        # If single paragraph is larger than chunk_size, split by sentences
        if para_length > chunk_size:
            # First, save current chunk if not empty
            if current_chunk_parts:
                chunk_text_content = '\n\n'.join(current_chunk_parts)
                chunks.append({
                    'text': chunk_text_content,
                    'metadata': {
                        'chunk_index': chunk_index,
                        'chunk_size': len(chunk_text_content)
                    }
                })
                chunk_index += 1
                
                # Keep overlap from end of current chunk
                overlap_text = _get_overlap_text(chunk_text_content, overlap)
                current_chunk_parts = [overlap_text] if overlap_text else []
                current_length = len(overlap_text) if overlap_text else 0
            
            # Split large paragraph by sentences
            sentences = _split_into_sentences(para)
            for sentence in sentences:
                sentence_length = len(sentence)
                
                if current_length + sentence_length + 1 > chunk_size and current_chunk_parts:
                    # Save current chunk
                    chunk_text_content = ' '.join(current_chunk_parts)
                    chunks.append({
                        'text': chunk_text_content,
                        'metadata': {
                            'chunk_index': chunk_index,
                            'chunk_size': len(chunk_text_content)
                        }
                    })
                    chunk_index += 1
                    
                    # Keep overlap
                    overlap_text = _get_overlap_text(chunk_text_content, overlap)
                    current_chunk_parts = [overlap_text] if overlap_text else []
                    current_length = len(overlap_text) if overlap_text else 0
                
                current_chunk_parts.append(sentence)
                current_length += sentence_length + 1
        else:
            # Check if adding this paragraph exceeds chunk_size
            if current_length + para_length + 2 > chunk_size and current_chunk_parts:
                # Save current chunk
                chunk_text_content = '\n\n'.join(current_chunk_parts)
                chunks.append({
                    'text': chunk_text_content,
                    'metadata': {
                        'chunk_index': chunk_index,
                        'chunk_size': len(chunk_text_content)
                    }
                })
                chunk_index += 1
                
                # Keep overlap from end of current chunk
                overlap_text = _get_overlap_text(chunk_text_content, overlap)
                current_chunk_parts = [overlap_text] if overlap_text else []
                current_length = len(overlap_text) if overlap_text else 0
            
            current_chunk_parts.append(para)
            current_length += para_length + 2  # +2 for paragraph separator
    
    # Add final chunk
    if current_chunk_parts:
        chunk_text_content = '\n\n'.join(current_chunk_parts)
        chunks.append({
            'text': chunk_text_content,
            'metadata': {
                'chunk_index': chunk_index,
                'chunk_size': len(chunk_text_content)
            }
        })
    
    # Add total_chunks to all chunk metadata
    total_chunks = len(chunks)
    for chunk in chunks:
        chunk['metadata']['total_chunks'] = total_chunks
    
    return chunks


def _get_overlap_text(text: str, overlap_chars: int) -> str:
    """Get overlap text from the end of a chunk, respecting word boundaries.
    
    Args:
        text: Source text
        overlap_chars: Target number of characters for overlap
        
    Returns:
        Overlap text string
    """
    if len(text) <= overlap_chars:
        return text
    
    # Get last N characters
    overlap_section = text[-overlap_chars:]
    
    # Find first word boundary to avoid cutting words
    first_space = overlap_section.find(' ')
    if first_space > 0:
        return overlap_section[first_space + 1:]
    
    return overlap_section


def chunk_text_with_pages(
    pages_data: List[Dict[str, Any]], 
    chunk_size: int = 1500, 
    overlap: int = 400
) -> List[Dict[str, Any]]:
    """Split page-aware text into chunks, preserving page number information.
    
    Args:
        pages_data: List of dicts with 'page' and 'text' keys from extract_text_with_pages()
        chunk_size: Target size of each chunk in characters (default: 1500)
        overlap: Number of characters to overlap between chunks (default: 400)
        
    Returns:
        List of chunk dictionaries with 'text' and 'metadata' keys.
        Metadata includes 'page_number' (first page the chunk appears on).
    """
    if not pages_data:
        return []
    
    chunks = []
    current_chunk_parts = []
    current_length = 0
    current_page = pages_data[0]['page']
    chunk_index = 0
    
    for page_data in pages_data:
        page_num = page_data['page']
        page_text = page_data['text']
        
        # Split page into paragraphs
        paragraphs = _split_into_paragraphs(page_text)
        
        for para in paragraphs:
            para_length = len(para)
            
            # If single paragraph is larger than chunk_size, split by sentences
            if para_length > chunk_size:
                # First, save current chunk if not empty
                if current_chunk_parts:
                    chunk_text_content = '\n\n'.join(current_chunk_parts)
                    chunks.append({
                        'text': chunk_text_content,
                        'metadata': {
                            'chunk_index': chunk_index,
                            'chunk_size': len(chunk_text_content),
                            'page_number': current_page
                        }
                    })
                    chunk_index += 1
                    
                    # Keep overlap
                    overlap_text = _get_overlap_text(chunk_text_content, overlap)
                    current_chunk_parts = [overlap_text] if overlap_text else []
                    current_length = len(overlap_text) if overlap_text else 0
                    current_page = page_num
                
                # Split large paragraph by sentences
                sentences = _split_into_sentences(para)
                for sentence in sentences:
                    sentence_length = len(sentence)
                    
                    if current_length + sentence_length + 1 > chunk_size and current_chunk_parts:
                        chunk_text_content = ' '.join(current_chunk_parts)
                        chunks.append({
                            'text': chunk_text_content,
                            'metadata': {
                                'chunk_index': chunk_index,
                                'chunk_size': len(chunk_text_content),
                                'page_number': current_page
                            }
                        })
                        chunk_index += 1
                        
                        overlap_text = _get_overlap_text(chunk_text_content, overlap)
                        current_chunk_parts = [overlap_text] if overlap_text else []
                        current_length = len(overlap_text) if overlap_text else 0
                        current_page = page_num
                    
                    current_chunk_parts.append(sentence)
                    current_length += sentence_length + 1
            else:
                # Check if adding this paragraph exceeds chunk_size
                if current_length + para_length + 2 > chunk_size and current_chunk_parts:
                    chunk_text_content = '\n\n'.join(current_chunk_parts)
                    chunks.append({
                        'text': chunk_text_content,
                        'metadata': {
                            'chunk_index': chunk_index,
                            'chunk_size': len(chunk_text_content),
                            'page_number': current_page
                        }
                    })
                    chunk_index += 1
                    
                    overlap_text = _get_overlap_text(chunk_text_content, overlap)
                    current_chunk_parts = [overlap_text] if overlap_text else []
                    current_length = len(overlap_text) if overlap_text else 0
                    current_page = page_num
                
                current_chunk_parts.append(para)
                current_length += para_length + 2
    
    # Add final chunk
    if current_chunk_parts:
        chunk_text_content = '\n\n'.join(current_chunk_parts)
        chunks.append({
            'text': chunk_text_content,
            'metadata': {
                'chunk_index': chunk_index,
                'chunk_size': len(chunk_text_content),
                'page_number': current_page
            }
        })
    
    # Add total_chunks to all chunk metadata
    total_chunks = len(chunks)
    for chunk in chunks:
        chunk['metadata']['total_chunks'] = total_chunks
    
    return chunks
