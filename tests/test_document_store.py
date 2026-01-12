"""Tests for document store."""

import pytest
import json
from pathlib import Path
import tempfile
import shutil

from backend.document_store import DocumentStore


class TestDocumentStore:
    """Tests for DocumentStore class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.fixture
    def store_path(self, temp_dir):
        """Create store path for testing."""
        return temp_dir / "test_documents.json"
    
    @pytest.fixture
    def document_store(self, store_path):
        """Create document store instance for testing."""
        return DocumentStore(store_path)
    
    def test_document_store_initialization(self, store_path):
        """Test document store initializes correctly."""
        store = DocumentStore(store_path)
        assert store.count() == 0
    
    def test_document_store_loads_existing(self, store_path):
        """Test document store loads existing data."""
        existing_data = {
            "doc1": {
                "id": "doc1",
                "filename": "test.pdf",
                "upload_date": "2024-01-01T00:00:00",
                "rag_id": "rag1",
                "file_size": 1024,
                "chunk_count": 5
            }
        }
        with open(store_path, 'w') as f:
            json.dump(existing_data, f)
        
        store = DocumentStore(store_path)
        assert store.count() == 1
    
    def test_create_document(self, document_store):
        """Test creating a document entry."""
        doc_id = document_store.create(
            filename="test.pdf",
            rag_id="rag1",
            file_size=1024,
            chunk_count=5
        )
        
        doc = document_store.get(doc_id)
        assert doc['filename'] == "test.pdf"
        assert doc['rag_id'] == "rag1"
    
    def test_get_nonexistent_document(self, document_store):
        """Test getting non-existent document returns None."""
        assert document_store.get("nonexistent") is None
    
    def test_get_all_documents(self, document_store):
        """Test getting all documents."""
        document_store.create("test1.pdf", "rag1", 1024, 5)
        document_store.create("test2.pdf", "rag2", 2048, 10)
        
        all_docs = document_store.get_all()
        assert len(all_docs) == 2
    
    def test_delete_document(self, document_store):
        """Test deleting a document."""
        doc_id = document_store.create("test.pdf", "rag1", 1024, 5)
        
        assert document_store.delete(doc_id) is True
        assert document_store.count() == 0
    
    def test_delete_nonexistent_document(self, document_store):
        """Test deleting non-existent document returns False."""
        assert document_store.delete("nonexistent") is False
    
    def test_get_by_rag_id(self, document_store):
        """Test getting document by RAG ID."""
        doc_id = document_store.create("test.pdf", "rag1", 1024, 5)
        
        doc = document_store.get_by_rag_id("rag1")
        assert doc['id'] == doc_id
    
    def test_get_by_rag_id_nonexistent(self, document_store):
        """Test getting document by non-existent RAG ID."""
        assert document_store.get_by_rag_id("nonexistent") is None
