"""Tests for validation utilities."""

import pytest
from pathlib import Path
import tempfile
import shutil

from backend.validation import (
    validate_non_empty,
    validate_pdf_file,
    validate_file_size,
)
from backend.exceptions import ValidationError


class TestValidation:
    """Tests for validation functions."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    def test_validate_non_empty_success(self):
        """Test validating non-empty string succeeds."""
        validate_non_empty("test", "TestField")
        validate_non_empty("  test  ", "TestField")
    
    def test_validate_non_empty_fails(self):
        """Test validating empty string fails."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_non_empty("", "TestField")
        
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_non_empty("   ", "TestField")
    
    def test_validate_pdf_file_by_filename(self):
        """Test validating PDF by filename."""
        validate_pdf_file(filename="test.pdf")
        validate_pdf_file(filename="TEST.PDF")
    
    def test_validate_pdf_file_by_filename_fails(self):
        """Test validating non-PDF filename fails."""
        with pytest.raises(ValidationError, match="must be a PDF"):
            validate_pdf_file(filename="test.txt")
    
    def test_validate_pdf_file_by_content(self):
        """Test validating PDF by content."""
        pdf_content = b"%PDF-1.4 fake pdf content"
        validate_pdf_file(file_content=pdf_content)
    
    def test_validate_pdf_file_by_content_fails(self):
        """Test validating non-PDF content fails."""
        with pytest.raises(ValidationError, match="PDF"):
            validate_pdf_file(file_content=b"not a pdf")
        
        with pytest.raises(ValidationError, match="empty"):
            validate_pdf_file(file_content=b"")
    
    def test_validate_pdf_file_path_not_exists(self, temp_dir):
        """Test validating non-existent file path fails."""
        pdf_path = temp_dir / "nonexistent.pdf"
        with pytest.raises(ValidationError, match="not found"):
            validate_pdf_file(file_path=pdf_path)
    
    def test_validate_file_size_success(self):
        """Test validating file size succeeds."""
        small_content = b"x" * (5 * 1024 * 1024)  # 5MB
        validate_file_size(file_content=small_content, max_size_mb=10)
    
    def test_validate_file_size_fails(self):
        """Test validating file size fails when exceeded."""
        large_content = b"x" * (15 * 1024 * 1024)  # 15MB
        with pytest.raises(ValidationError, match="exceeds"):
            validate_file_size(file_content=large_content, max_size_mb=10)
    
    def test_validate_file_size_no_params(self):
        """Test validating file size without params fails."""
        with pytest.raises(ValidationError, match="must be provided"):
            validate_file_size()
