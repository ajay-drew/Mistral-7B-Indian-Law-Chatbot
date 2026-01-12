"""Document metadata storage and management."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import uuid

from backend.exceptions import DocumentError

logger = logging.getLogger(__name__)


class DocumentStore:
    """Simple JSON-based document metadata store."""
    
    def __init__(self, store_path: Path):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._documents: Dict[str, Dict[str, Any]] = {}
        self._load()
    
    def _load(self) -> None:
        """Load documents from JSON file."""
        if self.store_path.exists():
            try:
                with open(self.store_path, 'r', encoding='utf-8') as f:
                    self._documents = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load document store: {e}")
                self._documents = {}
    
    def _save(self) -> None:
        """Save documents to JSON file."""
        try:
            with open(self.store_path, 'w', encoding='utf-8') as f:
                json.dump(self._documents, f, indent=2, default=str)
        except Exception as e:
            raise DocumentError(f"Failed to save document store: {e}") from e
    
    def create(self, filename: str, rag_id: str, file_size: int, chunk_count: int) -> str:
        """Create a new document entry."""
        doc_id = str(uuid.uuid4())
        self._documents[doc_id] = {
            'id': doc_id,
            'filename': filename,
            'upload_date': datetime.now().isoformat(),
            'rag_id': rag_id,
            'file_size': file_size,
            'chunk_count': chunk_count,
        }
        self._save()
        return doc_id
    
    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""
        return self._documents.get(doc_id)
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Get all documents."""
        return list(self._documents.values())
    
    def delete(self, doc_id: str) -> bool:
        """Delete a document entry."""
        if doc_id in self._documents:
            del self._documents[doc_id]
            self._save()
            return True
        return False
    
    def count(self) -> int:
        """Get total number of documents."""
        return len(self._documents)
    
    def get_by_rag_id(self, rag_id: str) -> Optional[Dict[str, Any]]:
        """Get document by RAG ID."""
        for doc in self._documents.values():
            if doc.get('rag_id') == rag_id:
                return doc
        return None
