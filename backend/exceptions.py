"""Custom exception classes for the application."""

from __future__ import annotations


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


class DocumentError(Exception):
    """Raised when document processing fails."""
    pass
