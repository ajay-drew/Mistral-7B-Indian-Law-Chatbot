"""RAG (Retrieval-Augmented Generation) system with hybrid search and reranking."""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from backend.exceptions import DocumentError

logger = logging.getLogger(__name__)


# Legal synonym dictionary for query expansion
LEGAL_SYNONYMS = {
    # Criminal Law
    "murder": ["homicide", "killing", "section 302 ipc", "culpable homicide"],
    "theft": ["stealing", "larceny", "section 378 ipc", "dishonest taking"],
    "robbery": ["dacoity", "section 390 ipc", "theft with force"],
    "assault": ["battery", "hurt", "section 351 ipc", "grievous hurt"],
    "cheating": ["fraud", "deception", "section 420 ipc", "misrepresentation"],
    "kidnapping": ["abduction", "section 359 ipc", "wrongful confinement"],
    "rape": ["sexual assault", "section 376 ipc", "sexual offence"],
    "defamation": ["libel", "slander", "section 499 ipc"],
    "forgery": ["fabrication", "section 463 ipc", "false document"],
    
    # Civil Law
    "contract": ["agreement", "deed", "covenant", "obligation"],
    "breach": ["violation", "non-performance", "default", "breaking"],
    "damages": ["compensation", "indemnity", "restitution", "remedy"],
    "negligence": ["carelessness", "breach of duty", "tort"],
    "injunction": ["restraining order", "prohibition", "stay order"],
    "specific performance": ["enforcement", "compel performance"],
    
    # Constitutional Law
    "fundamental rights": ["basic rights", "constitutional rights", "part iii"],
    "writ": ["mandamus", "certiorari", "habeas corpus", "prohibition", "quo warranto"],
    "article 21": ["right to life", "personal liberty", "life and liberty"],
    "article 14": ["equality", "equal protection", "right to equality"],
    "article 19": ["freedom of speech", "free speech", "fundamental freedoms"],
    "article 32": ["constitutional remedy", "supreme court writ"],
    "article 226": ["high court writ", "high court jurisdiction"],
    
    # Procedural Law
    "bail": ["anticipatory bail", "regular bail", "section 437", "section 438", "interim bail"],
    "fir": ["first information report", "police complaint", "section 154"],
    "chargesheet": ["charge sheet", "challan", "section 173"],
    "summons": ["notice", "court notice", "legal notice"],
    "warrant": ["arrest warrant", "search warrant", "non-bailable warrant"],
    "appeal": ["revision", "review", "appellate"],
    "stay": ["interim stay", "stay order", "status quo"],
    
    # Property Law
    "property": ["immovable property", "real estate", "land"],
    "lease": ["tenancy", "rent", "rental agreement"],
    "mortgage": ["hypothecation", "pledge", "security interest"],
    "partition": ["division", "separation of property"],
    "succession": ["inheritance", "heir", "legal heir"],
    "will": ["testament", "testamentary", "bequest"],
    
    # Family Law
    "divorce": ["dissolution of marriage", "matrimonial dispute", "separation"],
    "custody": ["child custody", "guardianship", "visitation"],
    "maintenance": ["alimony", "spousal support", "section 125"],
    "adoption": ["legal adoption", "guardianship"],
    
    # Corporate Law
    "company": ["corporation", "corporate entity", "firm"],
    "director": ["board member", "managing director", "company director"],
    "shareholder": ["stockholder", "member", "equity holder"],
    "insolvency": ["bankruptcy", "winding up", "liquidation"],
    
    # Evidence
    "evidence": ["proof", "testimony", "documentary evidence"],
    "witness": ["deponent", "attestor", "eyewitness"],
    "confession": ["admission", "statement", "section 164"],
    "hearsay": ["indirect evidence", "second-hand evidence"],
}


class RAGSystem:
    """RAG system with hybrid search (semantic + BM25), reranking, and query expansion."""
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        persist_dir: Path = Path("./data/chroma_db"),
        collection_name: str = "indian_law_documents",
        top_k: int = 3,
        min_relevance_score: float = 0.35,
        hybrid_alpha: float = 0.6,
        use_reranker: bool = True
    ):
        """Initialize RAG system.
        
        Args:
            embedding_model: Sentence transformer model for semantic embeddings
            reranker_model: Cross-encoder model for reranking
            persist_dir: Directory for ChromaDB persistence
            collection_name: Name of the ChromaDB collection
            top_k: Number of results to return
            min_relevance_score: Minimum relevance score threshold (0-1)
            hybrid_alpha: Weight for semantic vs BM25 (0.6 = 60% semantic, 40% BM25)
            use_reranker: Whether to use cross-encoder reranking
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.min_relevance_score = min_relevance_score
        self.hybrid_alpha = hybrid_alpha
        self.use_reranker = use_reranker
        self.reranker_model_name = reranker_model
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Lazy-load reranker to save memory
        self._reranker: Optional[CrossEncoder] = None
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"ChromaDB collection '{collection_name}' ready")
        
        # BM25 index storage
        self._bm25_index_path = self.persist_dir / "bm25_index.pkl"
        self._bm25_corpus: List[List[str]] = []  # Tokenized documents
        self._bm25_doc_ids: List[str] = []  # Corresponding chunk IDs
        self._bm25_texts: List[str] = []  # Original texts for lookup
        self._bm25: Optional[BM25Okapi] = None
        
        # Load existing BM25 index if available
        self._load_bm25_index()
    
    def _load_reranker(self) -> CrossEncoder:
        """Lazy-load the cross-encoder reranker model."""
        if self._reranker is None:
            logger.info(f"Loading reranker model: {self.reranker_model_name}")
            self._reranker = CrossEncoder(self.reranker_model_name)
        return self._reranker
    
    def _load_bm25_index(self) -> None:
        """Load BM25 index from disk if available."""
        if self._bm25_index_path.exists():
            try:
                with open(self._bm25_index_path, 'rb') as f:
                    data = pickle.load(f)
                    self._bm25_corpus = data.get('corpus', [])
                    self._bm25_doc_ids = data.get('doc_ids', [])
                    self._bm25_texts = data.get('texts', [])
                    if self._bm25_corpus:
                        self._bm25 = BM25Okapi(self._bm25_corpus)
                        logger.info(f"Loaded BM25 index with {len(self._bm25_corpus)} documents")
            except Exception as e:
                logger.warning(f"Failed to load BM25 index: {e}")
                self._bm25 = None
    
    def _save_bm25_index(self) -> None:
        """Save BM25 index to disk."""
        try:
            with open(self._bm25_index_path, 'wb') as f:
                pickle.dump({
                    'corpus': self._bm25_corpus,
                    'doc_ids': self._bm25_doc_ids,
                    'texts': self._bm25_texts
                }, f)
        except Exception as e:
            logger.warning(f"Failed to save BM25 index: {e}")
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 indexing."""
        # Simple tokenization: lowercase, split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def _expand_query(self, query: str) -> str:
        """Expand query with legal synonyms.
        
        Args:
            query: Original query string
            
        Returns:
            Expanded query with synonyms appended
        """
        query_lower = query.lower()
        expansions = []
        
        for term, synonyms in LEGAL_SYNONYMS.items():
            if term in query_lower:
                # Add synonyms that aren't already in the query
                for syn in synonyms[:2]:  # Limit to first 2 synonyms
                    if syn.lower() not in query_lower:
                        expansions.append(syn)
        
        if expansions:
            expanded = f"{query} {' '.join(expansions)}"
            logger.debug(f"Query expanded: '{query}' -> '{expanded}'")
            return expanded
        
        return query
    
    def add_documents(
        self, 
        documents: List[Dict[str, Any]], 
        document_id: str,
        filename: str = ""
    ) -> None:
        """Add documents to both vector store and BM25 index.
        
        Args:
            documents: List of chunk dicts with 'text' and 'metadata' keys
            document_id: Unique identifier for the document
            filename: Original filename for citation purposes
        """
        if not documents:
            raise DocumentError("No documents to add")
        
        texts = [doc['text'] for doc in documents]
        metadatas = []
        ids = []
        
        total_chunks = len(documents)
        
        for i, doc in enumerate(documents):
            chunk_metadata = doc.get('metadata', {})
            metadata = {
                'document_id': document_id,
                'filename': filename,
                'chunk_index': chunk_metadata.get('chunk_index', i),
                'total_chunks': total_chunks,
                'page_number': chunk_metadata.get('page_number', 1),
            }
            metadatas.append(metadata)
            chunk_id = f"{document_id}_chunk_{i}"
            ids.append(chunk_id)
            
            # Add to BM25 corpus
            tokens = self._tokenize(doc['text'])
            self._bm25_corpus.append(tokens)
            self._bm25_doc_ids.append(chunk_id)
            self._bm25_texts.append(doc['text'])
        
        # Generate embeddings and add to ChromaDB
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        # Rebuild BM25 index
        if self._bm25_corpus:
            self._bm25 = BM25Okapi(self._bm25_corpus)
            self._save_bm25_index()
        
        logger.info(f"Added {len(texts)} chunks for document {document_id} ('{filename}')")
    
    def _semantic_search(self, query: str, n_results: int) -> List[Dict[str, Any]]:
        """Perform semantic search using ChromaDB.
        
        Returns list of dicts with 'id', 'text', 'metadata', 'score' keys.
        """
        query_embedding = self.embedding_model.encode([query], show_progress_bar=False)[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        chunks = []
        if results['documents'] and results['documents'][0]:
            for i, doc_text in enumerate(results['documents'][0]):
                # ChromaDB returns distances (1 - cosine_similarity for cosine space)
                distance = results['distances'][0][i] if results['distances'] else 1.0
                score = 1.0 - distance  # Convert distance to similarity
                
                chunks.append({
                    'id': results['ids'][0][i] if results['ids'] else f"chunk_{i}",
                    'text': doc_text,
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'score': score
                })
        
        return chunks
    
    def _bm25_search(self, query: str, n_results: int) -> List[Dict[str, Any]]:
        """Perform BM25 keyword search.
        
        Returns list of dicts with 'id', 'text', 'metadata', 'score' keys.
        """
        if not self._bm25 or not self._bm25_corpus:
            return []
        
        query_tokens = self._tokenize(query)
        scores = self._bm25.get_scores(query_tokens)
        
        # Get top N indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
        
        # Normalize scores to 0-1 range
        max_score = max(scores) if max(scores) > 0 else 1.0
        
        chunks = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include if there's some match
                chunk_id = self._bm25_doc_ids[idx]
                
                # Get metadata from ChromaDB
                try:
                    result = self.collection.get(ids=[chunk_id], include=['metadatas'])
                    metadata = result['metadatas'][0] if result['metadatas'] else {}
                except Exception:
                    metadata = {}
                
                chunks.append({
                    'id': chunk_id,
                    'text': self._bm25_texts[idx],
                    'metadata': metadata,
                    'score': scores[idx] / max_score  # Normalize to 0-1
                })
        
        return chunks
    
    def _fuse_results(
        self, 
        semantic_results: List[Dict[str, Any]], 
        bm25_results: List[Dict[str, Any]],
        alpha: float
    ) -> List[Dict[str, Any]]:
        """Fuse semantic and BM25 results using weighted scoring.
        
        Args:
            semantic_results: Results from semantic search
            bm25_results: Results from BM25 search
            alpha: Weight for semantic scores (1-alpha for BM25)
            
        Returns:
            Fused and sorted list of unique results
        """
        # Create lookup by ID
        fused = {}
        
        # Add semantic results
        for chunk in semantic_results:
            chunk_id = chunk['id']
            fused[chunk_id] = {
                'id': chunk_id,
                'text': chunk['text'],
                'metadata': chunk['metadata'],
                'semantic_score': chunk['score'],
                'bm25_score': 0.0
            }
        
        # Add/update with BM25 results
        for chunk in bm25_results:
            chunk_id = chunk['id']
            if chunk_id in fused:
                fused[chunk_id]['bm25_score'] = chunk['score']
            else:
                fused[chunk_id] = {
                    'id': chunk_id,
                    'text': chunk['text'],
                    'metadata': chunk['metadata'],
                    'semantic_score': 0.0,
                    'bm25_score': chunk['score']
                }
        
        # Calculate combined scores
        results = []
        for chunk_id, data in fused.items():
            combined_score = (alpha * data['semantic_score']) + ((1 - alpha) * data['bm25_score'])
            results.append({
                'id': chunk_id,
                'text': data['text'],
                'metadata': data['metadata'],
                'score': combined_score,
                'semantic_score': data['semantic_score'],
                'bm25_score': data['bm25_score']
            })
        
        # Sort by combined score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results
    
    def _rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Rerank chunks using cross-encoder.
        
        Args:
            query: Search query
            chunks: Candidate chunks to rerank
            top_k: Number of results to return
            
        Returns:
            Reranked and filtered list of chunks
        """
        if not chunks:
            return []
        
        reranker = self._load_reranker()
        
        # Create query-document pairs
        pairs = [[query, chunk['text']] for chunk in chunks]
        
        # Get reranker scores
        scores = reranker.predict(pairs)
        
        # Normalize scores to 0-1 using sigmoid-like transformation
        # Cross-encoder scores can be negative, so we normalize
        min_score, max_score = min(scores), max(scores)
        score_range = max_score - min_score if max_score != min_score else 1.0
        
        # Update chunks with reranker scores
        for i, chunk in enumerate(chunks):
            normalized_score = (scores[i] - min_score) / score_range if score_range > 0 else 0.5
            chunk['rerank_score'] = normalized_score
            chunk['score'] = normalized_score  # Update main score
        
        # Sort by reranker score and return top_k
        chunks.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return chunks[:top_k]
    
    def search(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        expand_query: bool = True
    ) -> List[Dict[str, Any]]:
        """Search for relevant document chunks using hybrid search with optional reranking.
        
        Pipeline:
        1. Expand query with legal synonyms (optional)
        2. Semantic search (ChromaDB)
        3. BM25 keyword search
        4. Fuse results with weighted scoring
        5. Filter by minimum relevance threshold
        6. Rerank with cross-encoder (optional)
        
        Args:
            query: Search query
            top_k: Number of results to return (default: self.top_k)
            expand_query: Whether to expand query with legal synonyms
            
        Returns:
            List of chunk dicts with 'text', 'metadata', 'score' keys
        """
        if not query or not query.strip():
            return []
        
        top_k = top_k or self.top_k
        
        # Step 1: Query expansion
        search_query = self._expand_query(query) if expand_query else query
        
        # Step 2 & 3: Hybrid search (get more candidates for reranking)
        candidate_count = top_k * 3 if self.use_reranker else top_k * 2
        
        semantic_results = self._semantic_search(search_query, candidate_count)
        bm25_results = self._bm25_search(search_query, candidate_count)
        
        # Step 4: Fuse results
        if bm25_results:
            fused_results = self._fuse_results(semantic_results, bm25_results, self.hybrid_alpha)
        else:
            # Fall back to semantic only if BM25 is empty
            fused_results = semantic_results
        
        # Step 5: Filter by minimum relevance
        filtered_results = [
            chunk for chunk in fused_results 
            if chunk['score'] >= self.min_relevance_score
        ]
        
        # Step 6: Rerank if enabled
        if self.use_reranker and len(filtered_results) > top_k:
            final_results = self._rerank(query, filtered_results, top_k)
        else:
            final_results = filtered_results[:top_k]
        
        # Log search results
        logger.debug(f"Search '{query[:50]}...': {len(semantic_results)} semantic, "
                    f"{len(bm25_results)} BM25, {len(filtered_results)} after filter, "
                    f"{len(final_results)} final")
        
        return final_results
    
    def delete_document(self, document_id: str) -> None:
        """Delete all chunks for a document from both indices."""
        # Delete from ChromaDB
        results = self.collection.get(where={"document_id": document_id})
        if results['ids']:
            deleted_ids = set(results['ids'])
            self.collection.delete(ids=results['ids'])
            logger.info(f"Deleted {len(results['ids'])} chunks from ChromaDB for document {document_id}")
            
            # Remove from BM25 index
            new_corpus = []
            new_doc_ids = []
            new_texts = []
            
            for i, doc_id in enumerate(self._bm25_doc_ids):
                if doc_id not in deleted_ids:
                    new_corpus.append(self._bm25_corpus[i])
                    new_doc_ids.append(self._bm25_doc_ids[i])
                    new_texts.append(self._bm25_texts[i])
            
            self._bm25_corpus = new_corpus
            self._bm25_doc_ids = new_doc_ids
            self._bm25_texts = new_texts
            
            # Rebuild BM25 index
            if self._bm25_corpus:
                self._bm25 = BM25Okapi(self._bm25_corpus)
            else:
                self._bm25 = None
            
            self._save_bm25_index()
            logger.info(f"Deleted chunks from BM25 index for document {document_id}")
    
    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into context string with source citations.
        
        Args:
            chunks: List of chunk dicts from search()
            
        Returns:
            Formatted context string with numbered citations
        """
        if not chunks:
            return ""
        
        context_parts = ["Relevant document excerpts:\n"]
        
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get('metadata', {})
            filename = metadata.get('filename', 'Unknown document')
            page = metadata.get('page_number', 'N/A')
            score = chunk.get('score', 0)
            
            context_parts.append(
                f"[{i}] Source: {filename}, Page {page} (relevance: {score:.2f})\n"
                f"{chunk['text']}\n"
            )
        
        return "\n".join(context_parts)
