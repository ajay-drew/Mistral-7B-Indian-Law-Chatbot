"""Tests for RAG system with hybrid search, reranking, and query expansion."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import shutil

from backend.rag import RAGSystem, LEGAL_SYNONYMS
from backend.exceptions import DocumentError


class TestRAGSystem:
    """Tests for RAGSystem class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.fixture
    def rag_system(self, temp_dir):
        """Create RAG system instance for testing."""
        with patch('backend.rag.SentenceTransformer') as mock_transformer:
            mock_model = MagicMock()
            mock_embedding = MagicMock()
            mock_embedding.tolist.return_value = [0.1, 0.2, 0.3]
            mock_model.encode.return_value = [mock_embedding]
            mock_transformer.return_value = mock_model
            
            with patch('backend.rag.chromadb.PersistentClient') as mock_client:
                mock_collection = MagicMock()
                mock_client.return_value.get_or_create_collection.return_value = mock_collection
                
                rag = RAGSystem(
                    embedding_model="test-model",
                    persist_dir=temp_dir,
                    top_k=3,
                    min_relevance_score=0.35,
                    hybrid_alpha=0.6,
                    use_reranker=False
                )
                rag.embedding_model = mock_model
                rag.collection = mock_collection
                yield rag
    
    def test_initialization(self, rag_system):
        """Test RAG system initialization with all parameters."""
        assert rag_system.top_k == 3
        assert rag_system.min_relevance_score == 0.35
        assert rag_system.hybrid_alpha == 0.6
    
    def test_add_documents(self, rag_system):
        """Test adding documents to RAG system with filename."""
        documents = [
            {'text': 'Test chunk 1', 'metadata': {'chunk_index': 0, 'page_number': 1}},
            {'text': 'Test chunk 2', 'metadata': {'chunk_index': 1, 'page_number': 2}}
        ]
        
        mock_embeddings = MagicMock()
        mock_embeddings.tolist.return_value = [[0.1, 0.2], [0.3, 0.4]]
        rag_system.embedding_model.encode.return_value = mock_embeddings
        
        rag_system.add_documents(documents, "test-doc", filename="test.pdf")
        
        assert rag_system.collection.add.called
        assert len(rag_system._bm25_corpus) == 2
    
    def test_add_documents_empty(self, rag_system):
        """Test adding empty documents raises error."""
        with pytest.raises(DocumentError, match="No documents"):
            rag_system.add_documents([], "test-doc")
    
    def test_search_semantic(self, rag_system):
        """Test semantic search for relevant chunks."""
        rag_system.collection.query.return_value = {
            'documents': [['Test chunk 1', 'Test chunk 2']],
            'metadatas': [[
                {'chunk_index': 0, 'document_id': 'doc1'},
                {'chunk_index': 1, 'document_id': 'doc1'}
            ]],
            'ids': [['doc1_chunk_0', 'doc1_chunk_1']],
            'distances': [[0.2, 0.3]]
        }
        mock_query_emb = MagicMock()
        mock_query_emb.tolist.return_value = [0.1, 0.2, 0.3]
        rag_system.embedding_model.encode.return_value = [mock_query_emb]
        
        results = rag_system.search("test query")
        
        assert len(results) == 2
        assert 'score' in results[0]
    
    def test_search_empty_query(self, rag_system):
        """Test searching with empty query returns empty list."""
        assert rag_system.search("") == []
        assert rag_system.search("   ") == []
    
    def test_format_context_with_citations(self, rag_system):
        """Test formatting context with detailed source citations."""
        chunks = [
            {
                'text': 'Chunk 1 content', 
                'metadata': {'filename': 'case.pdf', 'page_number': 5},
                'score': 0.85
            }
        ]
        
        context = rag_system.format_context(chunks)
        
        assert "case.pdf" in context
        assert "Page 5" in context
        assert "[1]" in context
    
    def test_format_context_empty(self, rag_system):
        """Test formatting empty context."""
        assert rag_system.format_context([]) == ""
    
    def test_delete_document(self, rag_system):
        """Test deleting a document from both indices."""
        rag_system._bm25_corpus = [['test']]
        rag_system._bm25_doc_ids = ['doc1_chunk_0']
        rag_system._bm25_texts = ['test']
        
        rag_system.collection.get.return_value = {'ids': ['doc1_chunk_0']}
        
        rag_system.delete_document("doc1")
        
        assert rag_system.collection.delete.called
    
    def test_tokenize(self, rag_system):
        """Test text tokenization for BM25."""
        text = "This is a Test!"
        tokens = rag_system._tokenize(text)
        
        assert all(t.islower() for t in tokens)
        assert "this" in tokens
    
    def test_query_expansion(self, rag_system):
        """Test query expansion with legal synonyms."""
        expanded = rag_system._expand_query("murder case")
        assert len(expanded) > len("murder case")
        
        original = "weather today"
        expanded = rag_system._expand_query(original)
        assert expanded == original


class TestLegalSynonyms:
    """Tests for legal synonym dictionary."""
    
    def test_criminal_law_synonyms(self):
        """Test criminal law terms have synonyms."""
        assert "murder" in LEGAL_SYNONYMS
        assert "theft" in LEGAL_SYNONYMS
        assert len(LEGAL_SYNONYMS["murder"]) > 0
    
    def test_civil_law_synonyms(self):
        """Test civil law terms have synonyms."""
        assert "contract" in LEGAL_SYNONYMS
        assert "damages" in LEGAL_SYNONYMS
    
    def test_constitutional_law_synonyms(self):
        """Test constitutional law terms have synonyms."""
        assert "fundamental rights" in LEGAL_SYNONYMS
        assert "writ" in LEGAL_SYNONYMS
    
    def test_procedural_law_synonyms(self):
        """Test procedural law terms have synonyms."""
        assert "bail" in LEGAL_SYNONYMS
        assert "fir" in LEGAL_SYNONYMS
