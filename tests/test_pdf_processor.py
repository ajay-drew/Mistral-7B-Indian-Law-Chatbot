"""Tests for PDF processor with page-aware extraction and paragraph chunking."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil

from backend.pdf_processor import (
    extract_text_from_pdf,
    extract_text_from_bytes,
    extract_text_with_pages,
    clean_text,
    chunk_text,
    chunk_text_with_pages,
    _split_into_paragraphs,
    _split_into_sentences,
    _get_overlap_text
)
from backend.exceptions import DocumentError


class TestTextCleaning:
    """Tests for text cleaning functions."""
    
    def test_clean_text_removes_extra_spaces(self):
        """Test that extra spaces are removed."""
        dirty = "  This   has   extra   spaces  "
        cleaned = clean_text(dirty)
        assert "  " not in cleaned
    
    def test_clean_text_empty(self):
        """Test cleaning empty text."""
        assert clean_text("") == ""
    
    def test_clean_text_preserves_structure(self):
        """Test that paragraph structure is preserved."""
        text = "Line 1\nLine 2\nLine 3"
        cleaned = clean_text(text)
        assert "Line 1" in cleaned
        assert "Line 2" in cleaned


class TestParagraphSplitting:
    """Tests for paragraph splitting."""
    
    def test_split_into_paragraphs(self):
        """Test splitting text into paragraphs."""
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        paragraphs = _split_into_paragraphs(text)
        
        assert len(paragraphs) == 3
        assert "Paragraph 1" in paragraphs[0]
        assert "Paragraph 2" in paragraphs[1]
    
    def test_split_into_paragraphs_empty(self):
        """Test splitting empty text."""
        assert _split_into_paragraphs("") == []
    
    def test_split_into_paragraphs_single(self):
        """Test splitting single paragraph."""
        text = "Single paragraph with no breaks."
        paragraphs = _split_into_paragraphs(text)
        assert len(paragraphs) == 1


class TestSentenceSplitting:
    """Tests for sentence splitting."""
    
    def test_split_into_sentences(self):
        """Test splitting text into sentences."""
        text = "First sentence. Second sentence. Third sentence."
        sentences = _split_into_sentences(text)
        
        assert len(sentences) >= 1
        assert "First sentence" in sentences[0]
    
    def test_split_into_sentences_empty(self):
        """Test splitting empty text."""
        assert _split_into_sentences("") == []


class TestOverlapText:
    """Tests for overlap text extraction."""
    
    def test_get_overlap_text(self):
        """Test getting overlap text from chunk."""
        text = "This is a long text that we want to overlap."
        overlap = _get_overlap_text(text, overlap_chars=20)
        
        assert len(overlap) <= 20
        assert overlap in text
    
    def test_get_overlap_text_short(self):
        """Test overlap when text is shorter than overlap size."""
        text = "Short"
        overlap = _get_overlap_text(text, overlap_chars=100)
        assert overlap == text


class TestChunkText:
    """Tests for text chunking."""
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing."""
        return """This is the first paragraph. It contains some legal content.

This is the second paragraph. It also has important information.

This is the third paragraph. More content here."""
    
    def test_chunk_text_basic(self, sample_text):
        """Test basic text chunking."""
        chunks = chunk_text(sample_text, chunk_size=100, overlap=20)
        
        assert len(chunks) > 0
        assert all('text' in chunk for chunk in chunks)
        assert all('metadata' in chunk for chunk in chunks)
    
    def test_chunk_text_empty(self):
        """Test chunking empty text."""
        chunks = chunk_text("", chunk_size=100, overlap=20)
        assert chunks == []
    
    def test_chunk_text_metadata(self, sample_text):
        """Test chunk metadata structure."""
        chunks = chunk_text(sample_text, chunk_size=100, overlap=20)
        
        for i, chunk in enumerate(chunks):
            assert chunk['metadata']['chunk_index'] == i
            assert 'chunk_size' in chunk['metadata']
            assert 'total_chunks' in chunk['metadata']
    
    def test_chunk_text_total_chunks(self, sample_text):
        """Test that total_chunks is set correctly."""
        chunks = chunk_text(sample_text, chunk_size=100, overlap=20)
        
        total = len(chunks)
        for chunk in chunks:
            assert chunk['metadata']['total_chunks'] == total
    
    def test_chunk_text_respects_paragraphs(self):
        """Test that chunking respects paragraph boundaries."""
        text = "Short para 1.\n\nShort para 2.\n\nShort para 3."
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        
        # Should fit in one chunk
        assert len(chunks) == 1
        assert "Short para 1" in chunks[0]['text']
        assert "Short para 2" in chunks[0]['text']


class TestChunkTextWithPages:
    """Tests for page-aware text chunking."""
    
    @pytest.fixture
    def pages_data(self):
        """Sample page data for testing."""
        return [
            {'page': 1, 'text': 'Content from page 1. Some legal text here.'},
            {'page': 2, 'text': 'Content from page 2. More legal content.'},
            {'page': 3, 'text': 'Content from page 3. Final content.'}
        ]
    
    def test_chunk_text_with_pages_basic(self, pages_data):
        """Test page-aware chunking."""
        chunks = chunk_text_with_pages(pages_data, chunk_size=200, overlap=50)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert 'text' in chunk
            assert 'metadata' in chunk
            assert 'page_number' in chunk['metadata']
    
    def test_chunk_text_with_pages_empty(self):
        """Test chunking with empty pages data."""
        chunks = chunk_text_with_pages([], chunk_size=100, overlap=20)
        assert chunks == []
    
    def test_chunk_text_with_pages_preserves_page_number(self, pages_data):
        """Test that page numbers are preserved."""
        chunks = chunk_text_with_pages(pages_data, chunk_size=50, overlap=10)
        
        page_numbers = [c['metadata']['page_number'] for c in chunks]
        assert all(p in [1, 2, 3] for p in page_numbers)
    
    def test_chunk_text_with_pages_metadata(self, pages_data):
        """Test metadata structure for page-aware chunks."""
        chunks = chunk_text_with_pages(pages_data, chunk_size=200, overlap=50)
        
        for chunk in chunks:
            assert 'chunk_index' in chunk['metadata']
            assert 'chunk_size' in chunk['metadata']
            assert 'page_number' in chunk['metadata']
            assert 'total_chunks' in chunk['metadata']


class TestExtractTextWithPages:
    """Tests for page-aware PDF text extraction."""
    
    @patch('backend.pdf_processor.pdfplumber')
    def test_extract_text_with_pages_success(self, mock_pdfplumber):
        """Test successful page-aware PDF extraction."""
        mock_pdf = MagicMock()
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1 text"
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2 text"
        mock_pdf.pages = [mock_page1, mock_page2]
        
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        pdf_bytes = b"%PDF-1.4 fake pdf content"
        pages = extract_text_with_pages(pdf_bytes)
        
        assert len(pages) == 2
        assert pages[0]['page'] == 1
        assert pages[1]['page'] == 2
        assert "Page 1" in pages[0]['text']
    
    @patch('backend.pdf_processor.pdfplumber')
    def test_extract_text_with_pages_no_text(self, mock_pdfplumber):
        """Test extraction when no text is found."""
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = None
        mock_pdf.pages = [mock_page]
        
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        pdf_bytes = b"%PDF-1.4 fake pdf content"
        
        with pytest.raises(DocumentError, match="No text"):
            extract_text_with_pages(pdf_bytes)
    
    def test_extract_text_with_pages_invalid_pdf(self):
        """Test extraction with invalid PDF header."""
        invalid_bytes = b"Not a PDF file"
        
        with pytest.raises(DocumentError, match="not.*valid PDF"):
            extract_text_with_pages(invalid_bytes)


class TestExtractTextFromBytes:
    """Tests for PDF extraction from bytes."""
    
    @patch('backend.pdf_processor.pdfplumber')
    def test_extract_text_from_bytes_success(self, mock_pdfplumber):
        """Test successful PDF text extraction from bytes."""
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Extracted text"
        mock_pdf.pages = [mock_page]
        
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        pdf_bytes = b"%PDF-1.4 fake pdf content"
        
        with patch('backend.pdf_processor.clean_text', return_value="Extracted text"):
            text = extract_text_from_bytes(pdf_bytes)
            assert text == "Extracted text"
    
    @patch('backend.pdf_processor.pdfplumber')
    def test_extract_text_from_bytes_no_text(self, mock_pdfplumber):
        """Test PDF extraction from bytes when no text is found."""
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = None
        mock_pdf.pages = [mock_page]
        
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        pdf_bytes = b"%PDF-1.4 fake pdf content"
        
        with pytest.raises(DocumentError, match="No text"):
            extract_text_from_bytes(pdf_bytes)
    
    def test_extract_text_from_bytes_invalid_pdf(self):
        """Test extraction with invalid PDF."""
        invalid_bytes = b"Not a PDF"
        
        with pytest.raises(DocumentError, match="not.*valid PDF"):
            extract_text_from_bytes(invalid_bytes)


class TestExtractTextFromPDF:
    """Tests for PDF extraction from file path."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @patch('backend.pdf_processor.pdfplumber')
    def test_extract_text_from_pdf_success(self, mock_pdfplumber, temp_dir):
        """Test successful PDF text extraction."""
        mock_pdf = MagicMock()
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1 text"
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2 text"
        mock_pdf.pages = [mock_page1, mock_page2]
        
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        pdf_path = temp_dir / "test.pdf"
        pdf_path.touch()
        
        with patch('backend.pdf_processor.clean_text', return_value="Page 1 text\n\nPage 2 text"):
            text = extract_text_from_pdf(pdf_path)
            assert "Page 1 text" in text
            assert "Page 2 text" in text
    
    @patch('backend.pdf_processor.pdfplumber')
    def test_extract_text_from_pdf_no_text(self, mock_pdfplumber, temp_dir):
        """Test PDF extraction when no text is found."""
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = None
        mock_pdf.pages = [mock_page]
        
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        pdf_path = temp_dir / "test.pdf"
        pdf_path.touch()
        
        with pytest.raises(DocumentError, match="No text"):
            extract_text_from_pdf(pdf_path)
    
    @patch('backend.pdf_processor.pdfplumber')
    def test_extract_text_from_pdf_corrupted(self, mock_pdfplumber, temp_dir):
        """Test PDF extraction with corrupted file."""
        mock_pdfplumber.open.side_effect = Exception("Corrupted PDF")
        
        pdf_path = temp_dir / "test.pdf"
        pdf_path.touch()
        
        with pytest.raises(DocumentError):
            extract_text_from_pdf(pdf_path)
