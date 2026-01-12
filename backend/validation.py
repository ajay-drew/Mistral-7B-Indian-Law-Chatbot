"""Input validation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from backend.exceptions import ValidationError


def validate_non_empty(value: str, field_name: str = "Field") -> None:
    """Validate that a string value is not empty."""
    if not value or not value.strip():
        raise ValidationError(f"{field_name} cannot be empty")


def validate_pdf_file(file_path: Optional[Path] = None, file_content: Optional[bytes] = None, filename: Optional[str] = None) -> None:
    """Validate that a file is a valid PDF."""
    if filename:
        if not filename.lower().endswith('.pdf'):
            raise ValidationError("File must be a PDF (.pdf extension required)")
    
    if file_content is not None:
        if len(file_content) == 0:
            raise ValidationError("File is empty")
        if not file_content.startswith(b'%PDF'):
            raise ValidationError("File does not appear to be a valid PDF")
    
    if file_path:
        if not file_path.exists():
            raise ValidationError(f"File not found: {file_path}")


def validate_file_size(file_path: Optional[Path] = None, file_content: Optional[bytes] = None, max_size_mb: int = 10) -> None:
    """Validate that a file does not exceed maximum size."""
    max_size_bytes = max_size_mb * 1024 * 1024
    
    if file_content:
        size = len(file_content)
    elif file_path:
        size = file_path.stat().st_size
    else:
        raise ValidationError("Either file_path or file_content must be provided")
    
    if size > max_size_bytes:
        raise ValidationError(f"File size exceeds {max_size_mb} MB limit")
