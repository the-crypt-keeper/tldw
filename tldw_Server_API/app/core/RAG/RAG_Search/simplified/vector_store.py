"""
Vector store implementations with citations support.

This module provides vector store interfaces and implementations that support
both basic search and search with citations for source attribution.
"""

# Handle numpy as optional dependency
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Create a minimal stub for type hints
    class np:
        class ndarray:
            pass
            
from typing import List, Dict, Optional, Protocol, Tuple, Any, Union
from pathlib import Path
import json
from loguru import logger
from dataclasses import dataclass
from abc import abstractmethod
import time
import psutil

from .citations import Citation, CitationType, SearchResultWithCitations
from tldw_chatbook.Metrics.metrics_logger import log_counter, log_histogram, log_gauge, timeit


# Import constants from rag_service
MIN_SYSTEM_MEMORY_MB = 500  # Minimum system memory to avoid pressure
MEMORY_PRESSURE_REDUCTION = 0.2  # Fraction to reduce on memory pressure


@dataclass
class SearchResult:
    """Basic search result for backward compatibility."""
    id: str
    score: float
    document: str
    metadata: dict

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "score": self.score,
            "document": self.document,
            "metadata": self.metadata
        }


class VectorStore(Protocol):
    """
    Protocol defining vector store interface with citations support.
    
    Using Protocol instead of ABC for lighter weight and better type checking.
    """
    
    def add(self, ids: List[str], embeddings: Union[np.ndarray, List[List[float]]], 
            documents: List[str], metadata: List[dict]) -> None:
        """Add documents with embeddings to the store."""
        ...
    
    def search(self, query_embedding: Union[np.ndarray, List[float]], 
               top_k: int = 10) -> List[SearchResult]:
        """Basic search without citations (backward compatibility)."""
        ...
    
    def search_with_citations(self, query_embedding: Union[np.ndarray, List[float]],
                            query_text: str,
                            top_k: int = 10) -> List[SearchResultWithCitations]:
        """Enhanced search that includes citations."""
        ...
    
    def delete_collection(self, name: str) -> None:
        """Delete a collection by name."""
        ...
    
    def get_collection_stats(self) -> dict:
        """Get statistics about the collection."""
        ...
    
    def clear(self) -> None:
        """Clear all data from the store."""
        ...


class ChromaVectorStore:
    """
    ChromaDB implementation with citations support.
    
    Provides persistent vector storage with metadata support for citations.
    """
    
    def __init__(self, 
                 persist_directory: Union[str, Path], 
                 collection_name: str = "default",
                 distance_metric: str = "cosine"):
        """
        Initialize ChromaDB vector store.
        
        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection to use
            distance_metric: Distance metric for similarity (cosine, l2, ip)
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.distance_metric = distance_metric
        self._client = None
        self._collection = None
        
        # Ensure persist directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Log initialization
        logger.info(f"Initializing ChromaVectorStore: collection={collection_name}, "
                   f"metric={distance_metric}, persist_dir={persist_directory}")
        log_counter("vector_store_initialized", labels={
            "type": "chroma",
            "collection": collection_name,
            "metric": distance_metric
        })
        
        # Metrics
        self._add_count = 0
        self._search_count = 0
        self._last_operation_time = None
        self._embedding_dim = None
        
        # Memory usage cache
        self._memory_cache = {
            'value': None,
            'timestamp': 0,
            'ttl': 5.0  # Cache for 5 seconds
        }
    
    def _get_cached_memory_info(self):
        """
        Get memory info with caching to reduce psutil overhead.
        
        Returns:
            Memory usage in MB
        """
        current_time = time.time()
        
        # Check if cache is valid
        if (self._memory_cache['value'] is not None and 
            current_time - self._memory_cache['timestamp'] < self._memory_cache['ttl']):
            return self._memory_cache['value']
        
        # Get fresh memory info
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        
        # Update cache
        self._memory_cache['value'] = memory_mb
        self._memory_cache['timestamp'] = current_time
        
        return memory_mb
    
    @property
    def client(self):
        """Lazy load ChromaDB client."""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings
                
                settings = Settings(
                    persist_directory=str(self.persist_directory),
                    anonymized_telemetry=False,
                    allow_reset=True
                )
                self._client = chromadb.Client(settings)
                logger.info(f"Initialized ChromaDB client at {self.persist_directory}")
                
            except ImportError:
                raise ImportError(
                    "ChromaDB not installed. "
                    "Install with: pip install chromadb"
                )
        return self._client
    
    @property
    def collection(self):
        """Get or create collection."""
        if self._collection is None:
            # Map distance metrics to ChromaDB's hnsw:space parameter
            metric_map = {
                "cosine": "cosine",
                "l2": "l2",
                "ip": "ip"  # inner product
            }
            
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": metric_map.get(self.distance_metric, "cosine")}
            )
            logger.info(f"Using collection: {self.collection_name}")
        return self._collection
    
    @timeit("vector_store_add_documents")
    def add(self, 
            ids: List[str], 
            embeddings: Union[np.ndarray, List[List[float]]], 
            documents: List[str], 
            metadata: List[dict]) -> None:
        """
        Add documents to the collection.
        
        Metadata should include fields needed for citations:
        - doc_id: Original document ID
        - doc_title: Human-readable document title
        - chunk_start: Character offset in original document
        - chunk_end: End character offset
        - chunk_index: Index of this chunk
        - author, date, url: Optional citation metadata
        """
        if len(ids) == 0:
            logger.warning("add() called with empty documents")
            return
        
        start_time = time.time()
        
        # Log operation metrics
        log_counter("vector_store_add_attempt", labels={"collection": self.collection_name})
        log_histogram("vector_store_batch_size", len(ids))
        
        # Validate inputs
        if len(ids) != len(documents) or len(ids) != len(metadata):
            raise ValueError("ids, documents, and metadata must have same length")
        
        # Handle embeddings conversion
        if NUMPY_AVAILABLE and isinstance(embeddings, np.ndarray):
            if embeddings.shape[0] != len(ids):
                raise ValueError(f"embeddings shape {embeddings.shape} doesn't match {len(ids)} documents")
            
            # Store embedding dimension if not set
            if self._embedding_dim is None and embeddings.shape[1] > 0:
                self._embedding_dim = embeddings.shape[1]
                log_gauge("vector_store_embedding_dim", self._embedding_dim)
                logger.info(f"Detected embedding dimension: {self._embedding_dim}")
            elif self._embedding_dim is not None and embeddings.shape[1] != self._embedding_dim:
                # Validate dimension matches for subsequent adds
                raise ValueError(f"Embedding dimension mismatch: expected {self._embedding_dim}, got {embeddings.shape[1]}. "
                               f"All embeddings must have the same dimension.")
            
            embeddings = embeddings.tolist()
        elif isinstance(embeddings, list):
            # Validate list structure
            if len(embeddings) != len(ids):
                raise ValueError(f"embeddings length {len(embeddings)} doesn't match {len(ids)} documents")
            # Store embedding dimension from first embedding
            if self._embedding_dim is None and embeddings and len(embeddings[0]) > 0:
                self._embedding_dim = len(embeddings[0])
                log_gauge("vector_store_embedding_dim", self._embedding_dim)
                logger.info(f"Detected embedding dimension: {self._embedding_dim}")
            elif self._embedding_dim is not None and embeddings:
                # Validate all embeddings have correct dimension
                for i, emb in enumerate(embeddings):
                    if len(emb) != self._embedding_dim:
                        raise ValueError(f"Embedding dimension mismatch at index {i}: expected {self._embedding_dim}, got {len(emb)}. "
                                       f"All embeddings must have the same dimension.")
        
        # Log document statistics
        doc_lengths = [len(doc) for doc in documents]
        log_histogram("vector_store_document_length", sum(doc_lengths) / len(doc_lengths))
        
        # Ensure metadata is properly formatted
        processed_metadata = []
        metadata_sizes = []
        for i, meta in enumerate(metadata):
            # ChromaDB requires all metadata values to be strings, ints, or floats
            processed_meta = {}
            for key, value in meta.items():
                if value is None:
                    continue
                elif isinstance(value, (str, int, float)):
                    processed_meta[key] = value
                else:
                    # Convert other types to string
                    processed_meta[key] = str(value)
            processed_metadata.append(processed_meta)
            metadata_sizes.append(len(json.dumps(processed_meta)))
        
        # Log metadata statistics
        if metadata_sizes:
            log_histogram("vector_store_metadata_size", sum(metadata_sizes) / len(metadata_sizes))
        
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=processed_metadata
            )
            
            self._add_count += len(ids)
            self._last_operation_time = time.time()
            
            # Log success metrics
            elapsed = time.time() - start_time
            log_counter("vector_store_documents_added", value=len(ids))
            log_histogram("vector_store_add_time", elapsed)
            log_gauge("vector_store_total_documents", self._add_count)
            
            logger.info(f"Added {len(ids)} documents to collection in {elapsed:.3f}s")
            
        except Exception as e:
            log_counter("vector_store_add_error", labels={"error": type(e).__name__})
            logger.error(f"Failed to add to ChromaDB: {e}", exc_info=True)
            raise
    
    @timeit("vector_store_search")
    def search(self, 
               query_embedding: Union[np.ndarray, List[float]], 
               top_k: int = 10) -> List[SearchResult]:
        """Basic search without citations (backward compatibility)."""
        start_time = time.time()
        
        # Log search attempt
        log_counter("vector_store_search_attempt", labels={"collection": self.collection_name})
        log_histogram("vector_store_search_top_k", top_k)
        
        # Convert numpy array to list if needed
        if NUMPY_AVAILABLE and isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        elif not isinstance(query_embedding, list):
            raise TypeError(f"query_embedding must be numpy array or list, got {type(query_embedding)}")
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            search_results = []
            scores = []
            
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    score = 1 - results['distances'][0][i]  # Convert distance to similarity
                    scores.append(score)
                    search_results.append(SearchResult(
                        id=results['ids'][0][i],
                        score=score,
                        document=results['documents'][0][i],
                        metadata=results['metadatas'][0][i] or {}
                    ))
            
            # Log search metrics
            elapsed = time.time() - start_time
            log_histogram("vector_store_search_time", elapsed)
            log_histogram("vector_store_search_results_count", len(search_results))
            
            if scores:
                log_histogram("vector_store_search_avg_score", sum(scores) / len(scores))
                log_histogram("vector_store_search_max_score", max(scores))
                log_histogram("vector_store_search_min_score", min(scores))
            
            self._search_count += 1
            self._last_operation_time = time.time()
            log_gauge("vector_store_total_searches", self._search_count)
            
            logger.info(f"Search completed in {elapsed:.3f}s, found {len(search_results)} results")
            return search_results
            
        except Exception as e:
            log_counter("vector_store_search_error", labels={"error": type(e).__name__})
            logger.error(f"Search failed: {e}", exc_info=True)
            return []
    
    def search_with_citations(self, 
                            query_embedding: Union[np.ndarray, List[float]],
                            query_text: str,
                            top_k: int = 10,
                            score_threshold: float = 0.0) -> List[SearchResultWithCitations]:
        """
        Enhanced search that includes citations.
        
        Args:
            query_embedding: Query vector
            query_text: Original query text (used for citation context)
            top_k: Number of results to return
            score_threshold: Minimum score threshold for results
            
        Returns:
            List of results with citations
        """
        # First, get basic search results
        basic_results = self.search(query_embedding, top_k * 2)  # Get extra for filtering
        
        # Filter by score threshold
        filtered_results = [r for r in basic_results if r.score >= score_threshold]
        
        # Convert to results with citations
        results_with_citations = []
        
        for result in filtered_results[:top_k]:
            # Create citations from metadata
            citations = self._create_citations_from_result(result, query_text)
            
            # Create result with citations
            result_with_citations = SearchResultWithCitations(
                id=result.id,
                score=result.score,
                document=result.document,
                metadata=result.metadata,
                citations=citations
            )
            results_with_citations.append(result_with_citations)
        
        logger.debug(f"Search with citations returned {len(results_with_citations)} results")
        return results_with_citations
    
    def _create_citations_from_result(self, 
                                    result: SearchResult, 
                                    query_text: str) -> List[Citation]:
        """Create citations from a search result."""
        citations = []
        metadata = result.metadata
        
        # Create semantic citation for the chunk
        citation = Citation(
            document_id=metadata.get("doc_id", result.id),
            document_title=metadata.get("doc_title", "Unknown Document"),
            chunk_id=result.id,
            text=result.document[:300] + "..." if len(result.document) > 300 else result.document,
            start_char=int(metadata.get("chunk_start", 0)),
            end_char=int(metadata.get("chunk_end", len(result.document))),
            confidence=result.score,
            match_type=CitationType.SEMANTIC,
            metadata={
                "author": metadata.get("author"),
                "date": metadata.get("date"),
                "url": metadata.get("url"),
                "chunk_index": metadata.get("chunk_index", 0),
                "query": query_text  # Include query for context
            }
        )
        citations.append(citation)
        
        # If we have keyword matches in metadata, add exact citations
        if "keyword_matches" in metadata:
            try:
                keyword_matches = json.loads(metadata["keyword_matches"])
                for match in keyword_matches:
                    exact_citation = Citation(
                        document_id=metadata.get("doc_id", result.id),
                        document_title=metadata.get("doc_title", "Unknown Document"),
                        chunk_id=result.id,
                        text=match.get("text", ""),
                        start_char=int(match.get("start", 0)),
                        end_char=int(match.get("end", 0)),
                        confidence=1.0,  # Exact match
                        match_type=CitationType.EXACT,
                        metadata=citation.metadata
                    )
                    citations.append(exact_citation)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse keyword_matches: {e}")
        
        return citations
    
    def delete_collection(self, name: str) -> None:
        """Delete a collection."""
        try:
            self.client.delete_collection(name)
            if name == self.collection_name:
                self._collection = None
            logger.info(f"Deleted collection: {name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
    
    def clear(self) -> None:
        """Clear all data from the current collection."""
        try:
            # Get all IDs and delete them
            all_data = self.collection.get()
            if all_data['ids']:
                self.collection.delete(ids=all_data['ids'])
                logger.info(f"Cleared {len(all_data['ids'])} documents from collection")
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
    
    @timeit("vector_store_get_stats")
    def get_collection_stats(self) -> dict:
        """Get collection statistics."""
        try:
            count = self.collection.count()
            
            # Log collection statistics
            log_gauge("vector_store_collection_size", count, labels={"collection": self.collection_name})
            
            # Get sample of metadata to show available fields
            sample_data = self.collection.peek(limit=1)
            metadata_fields = set()
            if sample_data['metadatas']:
                for meta in sample_data['metadatas']:
                    metadata_fields.update(meta.keys())
            
            stats = {
                "name": self.collection_name,
                "count": count,
                "persist_directory": str(self.persist_directory),
                "distance_metric": self.distance_metric,
                "metadata_fields": list(metadata_fields),
                "add_count": self._add_count,
                "search_count": self._search_count,
                "last_operation": self._last_operation_time
            }
            
            # Try to get embedding dimension
            if sample_data['embeddings'] and sample_data['embeddings'][0]:
                stats["embedding_dimension"] = len(sample_data['embeddings'][0])
                self._embedding_dim = stats["embedding_dimension"]
            
            # Log comprehensive stats
            logger.info(f"Collection stats: {count} documents, {len(metadata_fields)} metadata fields")
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {
                "name": self.collection_name, 
                "count": 0,
                "error": str(e)
            }
    
    def add_documents(self, 
                     collection_name: str,
                     documents: List[str],
                     embeddings: Union[List[List[float]], np.ndarray],
                     metadatas: List[dict],
                     ids: List[str]) -> bool:
        """
        Add documents to a collection (compatibility method).
        
        Args:
            collection_name: Name of the collection
            documents: List of document texts
            embeddings: Document embeddings
            metadatas: List of metadata dicts
            ids: Document IDs
            
        Returns:
            True if successful
        """
        # Switch to the specified collection if different
        if collection_name != self.collection_name:
            self.collection_name = collection_name
            self._collection = None  # Force reload
        
        # Convert embeddings to numpy array if needed and numpy is available
        if NUMPY_AVAILABLE and isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        # If numpy not available, keep as list - the add method will handle it
        
        try:
            self.add(ids, embeddings, documents, metadatas)
            return True
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    def list_collections(self) -> List[str]:
        """List all collection names."""
        try:
            # Get all collections from ChromaDB
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []


class InMemoryVectorStore:
    """
    Simple in-memory vector store for testing or when persistence isn't needed.
    
    Implements the same interface as ChromaVectorStore but keeps everything in memory.
    """
    
    def __init__(self, distance_metric: str = "cosine", max_documents: int = 10000, max_collections: int = 10,
                 memory_threshold_mb: float = 1024.0):
        """
        Initialize in-memory vector store with memory limits.
        
        Args:
            distance_metric: Distance metric for similarity (cosine, l2, ip)
            max_documents: Maximum number of documents to store (default: 10000)
            max_collections: Maximum number of collections to keep (default: 10)
            memory_threshold_mb: Memory threshold in MB for triggering eviction (default: 1024MB)
        """
        self.distance_metric = distance_metric
        self.max_documents = max_documents
        self.max_collections = max_collections
        self.memory_threshold_mb = memory_threshold_mb
        self.ids: List[str] = []
        self.embeddings: List[Union[np.ndarray, List[float]]] = []
        self.documents: List[str] = []
        self.metadata: List[dict] = []
        
        # Track access order for LRU eviction
        self._access_order: List[str] = []
        
        # Collection support (for compatibility) with access tracking
        self._collections: Dict[str, Dict[str, Any]] = {}
        self._collection_access_time: Dict[str, float] = {}
        self._current_collection = "default"
        
        # Metrics
        self._add_count = 0
        self._search_count = 0
        self._eviction_count = 0
        self._collection_eviction_count = 0
        self._memory_pressure_evictions = 0
        
        # Memory monitoring
        self._last_memory_check = time.time()
        self._memory_check_interval = 10.0  # Check every 10 seconds
        
        # Track embedding dimension
        self._embedding_dim: Optional[int] = None
        
        logger.info(f"InMemoryVectorStore initialized with max_documents={max_documents}, "
                   f"max_collections={max_collections}, memory_threshold_mb={memory_threshold_mb}")
    
    def _check_memory_pressure(self) -> bool:
        """
        Check if memory usage exceeds threshold.
        
        Returns:
            True if memory pressure detected, False otherwise
        """
        # Only check periodically to avoid performance impact
        current_time = time.time()
        if current_time - self._last_memory_check < self._memory_check_interval:
            return False
        
        self._last_memory_check = current_time
        
        try:
            # Get current process memory usage
            memory_mb = self._get_cached_memory_info()
            
            # Also check system-wide memory if available
            vm = psutil.virtual_memory()
            available_mb = vm.available / (1024 * 1024)
            
            # Trigger eviction if:
            # 1. Process memory exceeds threshold, OR
            # 2. System available memory is very low
            if memory_mb > self.memory_threshold_mb or available_mb < MIN_SYSTEM_MEMORY_MB:
                logger.warning(f"Memory pressure detected: process={memory_mb:.1f}MB, "
                             f"available={available_mb:.1f}MB, threshold={self.memory_threshold_mb}MB")
                return True
                
        except Exception as e:
            logger.debug(f"Could not check memory pressure: {e}")
        
        return False
    
    def _evict_for_memory_pressure(self, target_reduction_ratio: float = MEMORY_PRESSURE_REDUCTION):
        """
        Evict documents to reduce memory pressure.
        
        Args:
            target_reduction_ratio: Fraction of documents to evict (default: 20%)
        """
        num_to_evict = max(1, int(len(self.ids) * target_reduction_ratio))
        logger.info(f"Evicting {num_to_evict} documents due to memory pressure")
        
        for _ in range(num_to_evict):
            if self._access_order:
                self._evict_lru()
                self._memory_pressure_evictions += 1
            else:
                break
    
    def add(self, 
            ids: List[str], 
            embeddings: Union[np.ndarray, List[List[float]]], 
            documents: List[str], 
            metadata: List[dict]) -> None:
        """Add documents to memory with LRU eviction and memory pressure handling."""
        if len(ids) == 0:
            return
        
        # Check memory pressure before adding
        if self._check_memory_pressure():
            self._evict_for_memory_pressure()
        
        # Handle embeddings based on numpy availability
        if NUMPY_AVAILABLE:
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
                
            # Validate and store embedding dimension
            if self._embedding_dim is None and embeddings.shape[1] > 0:
                self._embedding_dim = embeddings.shape[1]
                logger.info(f"Detected embedding dimension: {self._embedding_dim}")
            elif self._embedding_dim is not None and embeddings.shape[1] != self._embedding_dim:
                raise ValueError(f"Embedding dimension mismatch: expected {self._embedding_dim}, got {embeddings.shape[1]}. "
                               f"All embeddings must have the same dimension.")
        else:
            # Without numpy, ensure embeddings is a list
            if not isinstance(embeddings, list):
                raise TypeError("Without numpy, embeddings must be provided as a list of lists")
                
            # Validate embedding dimensions for lists
            if embeddings:
                # Check first embedding to establish dimension
                if self._embedding_dim is None and len(embeddings[0]) > 0:
                    self._embedding_dim = len(embeddings[0])
                    logger.info(f"Detected embedding dimension: {self._embedding_dim}")
                
                # Validate all embeddings have the same dimension
                for i, emb in enumerate(embeddings):
                    if len(emb) != self._embedding_dim:
                        raise ValueError(f"Embedding dimension mismatch at index {i}: expected {self._embedding_dim}, got {len(emb)}. "
                                       f"All embeddings must have the same dimension.")
        
        # Add documents, updating existing ones
        for i, id_val in enumerate(ids):
            if id_val in self.ids:
                # Update existing
                idx = self.ids.index(id_val)
                self.embeddings[idx] = embeddings[i]
                self.documents[idx] = documents[i]
                self.metadata[idx] = metadata[i]
                # Update access order
                if id_val in self._access_order:
                    self._access_order.remove(id_val)
                self._access_order.append(id_val)
            else:
                # Check if we need to evict due to document limit
                if len(self.ids) >= self.max_documents:
                    # Evict least recently used
                    self._evict_lru()
                
                # Add new
                self.ids.append(id_val)
                self.embeddings.append(embeddings[i])
                self.documents.append(documents[i])
                self.metadata.append(metadata[i])
                self._access_order.append(id_val)
        
        self._add_count += len(ids)
        logger.debug(f"Added {len(ids)} documents to in-memory store (current size: {len(self.ids)})")
    
    def _evict_lru(self) -> None:
        """Evict the least recently used document."""
        if not self._access_order:
            return
            
        # Get the least recently used ID
        lru_id = self._access_order.pop(0)
        
        # Remove from storage
        idx = self.ids.index(lru_id)
        self.ids.pop(idx)
        self.embeddings.pop(idx)
        self.documents.pop(idx)
        self.metadata.pop(idx)
        
        self._eviction_count += 1
        logger.debug(f"Evicted document {lru_id} (total evictions: {self._eviction_count})")
    
    def _compute_similarity(self, 
                          query_embedding: Union[np.ndarray, List[float]], 
                          doc_embedding: Union[np.ndarray, List[float]]) -> float:
        """Compute similarity based on distance metric with numerical stability."""
        if NUMPY_AVAILABLE:
            # Use numpy for efficiency when available
            if self.distance_metric == "cosine":
                # Cosine similarity with proper zero vector handling
                query_norm = np.linalg.norm(query_embedding)
                doc_norm = np.linalg.norm(doc_embedding)
                
                # Check for zero vectors
                if query_norm < 1e-6 or doc_norm < 1e-6:
                    # Zero vectors have no meaningful similarity
                    return 0.0
                
                # Normalize vectors
                query_normalized = query_embedding / query_norm
                doc_normalized = doc_embedding / doc_norm
                
                # Compute cosine similarity (clamp to [-1, 1] for numerical stability)
                similarity = np.dot(query_normalized, doc_normalized)
                return float(np.clip(similarity, -1.0, 1.0))
                
            elif self.distance_metric == "l2":
                # Negative L2 distance (so higher is more similar)
                distance = np.linalg.norm(query_embedding - doc_embedding)
                # Convert to similarity score (bounded between 0 and 1)
                return float(1.0 / (1.0 + distance))
                
            elif self.distance_metric == "ip":
                # Inner product
                return float(np.dot(query_embedding, doc_embedding))
        else:
            # Pure Python implementations
            # Ensure we're working with lists
            if not isinstance(query_embedding, list):
                query_embedding = list(query_embedding)
            if not isinstance(doc_embedding, list):
                doc_embedding = list(doc_embedding)
                
            if self.distance_metric == "cosine":
                # Compute dot product and norms
                dot_product = sum(q * d for q, d in zip(query_embedding, doc_embedding))
                query_norm = sum(q * q for q in query_embedding) ** 0.5
                doc_norm = sum(d * d for d in doc_embedding) ** 0.5
                
                # Check for zero vectors
                if query_norm < 1e-6 or doc_norm < 1e-6:
                    return 0.0
                
                # Compute cosine similarity
                similarity = dot_product / (query_norm * doc_norm)
                # Clamp to [-1, 1]
                return float(max(-1.0, min(1.0, similarity)))
                
            elif self.distance_metric == "l2":
                # Compute L2 distance
                distance = sum((q - d) ** 2 for q, d in zip(query_embedding, doc_embedding)) ** 0.5
                # Convert to similarity score
                return float(1.0 / (1.0 + distance))
                
            elif self.distance_metric == "ip":
                # Inner product
                return float(sum(q * d for q, d in zip(query_embedding, doc_embedding)))
                
        raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def search(self, 
               query_embedding: Union[np.ndarray, List[float]], 
               top_k: int = 10) -> List[SearchResult]:
        """Search using the specified distance metric."""
        if not self.embeddings:
            return []
        
        # Handle query embedding conversion based on numpy availability
        if NUMPY_AVAILABLE:
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding)
        else:
            # Without numpy, ensure it's a list
            if not isinstance(query_embedding, list):
                raise TypeError("Without numpy, query_embedding must be a list")
        
        # Compute similarities
        similarities = []
        for i, emb in enumerate(self.embeddings):
            similarity = self._compute_similarity(query_embedding, emb)
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        
        # Convert to SearchResult
        results = []
        for idx, score in top_results:
            # Update access order for LRU
            id_val = self.ids[idx]
            if id_val in self._access_order:
                self._access_order.remove(id_val)
            self._access_order.append(id_val)
            
            # Normalize score to [0, 1] range for consistency
            if self.distance_metric == "cosine":
                normalized_score = (score + 1) / 2  # From [-1, 1] to [0, 1]
            elif self.distance_metric == "l2":
                # Convert negative distance to similarity
                normalized_score = 1 / (1 + abs(score))
            else:  # ip
                normalized_score = max(0, min(1, score))  # Clamp to [0, 1]
            
            results.append(SearchResult(
                id=self.ids[idx],
                score=normalized_score,
                document=self.documents[idx],
                metadata=self.metadata[idx]
            ))
        
        self._search_count += 1
        return results
    
    def search_with_citations(self, 
                            query_embedding: Union[np.ndarray, List[float]],
                            query_text: str,
                            top_k: int = 10,
                            score_threshold: float = 0.0) -> List[SearchResultWithCitations]:
        """Enhanced search with citations."""
        # Get basic results (get extra to account for filtering)
        basic_results = self.search(query_embedding, top_k * 2)
        
        # Filter by score threshold
        filtered_results = [r for r in basic_results if r.score >= score_threshold]
        
        # Convert to results with citations
        results_with_citations = []
        for result in filtered_results[:top_k]:
            # Create citation
            citation = Citation(
                document_id=result.metadata.get("doc_id", result.id),
                document_title=result.metadata.get("doc_title", "Unknown Document"),
                chunk_id=result.id,
                text=result.document[:300] + "..." if len(result.document) > 300 else result.document,
                start_char=int(result.metadata.get("chunk_start", 0)),
                end_char=int(result.metadata.get("chunk_end", len(result.document))),
                confidence=result.score,
                match_type=CitationType.SEMANTIC,
                metadata={
                    "author": result.metadata.get("author"),
                    "date": result.metadata.get("date"),
                    "url": result.metadata.get("url"),
                    "chunk_index": result.metadata.get("chunk_index", 0)
                }
            )
            
            result_with_citations = SearchResultWithCitations(
                id=result.id,
                score=result.score,
                document=result.document,
                metadata=result.metadata,
                citations=[citation]
            )
            results_with_citations.append(result_with_citations)
        
        return results_with_citations
    
    def delete_collection(self, name: str) -> None:
        """Clear all data (collection name ignored for in-memory)."""
        self.clear()
    
    def clear(self) -> None:
        """Clear all data."""
        self.ids.clear()
        self.embeddings.clear()
        self.documents.clear()
        self.metadata.clear()
        self._access_order.clear()
        self._eviction_count = 0
        self._memory_pressure_evictions = 0
        logger.info("Cleared in-memory vector store")
    
    def get_collection_stats(self) -> dict:
        """Get stats."""
        stats = {
            "type": "in_memory",
            "count": len(self.ids),
            "max_documents": self.max_documents,
            "max_collections": self.max_collections,
            "distance_metric": self.distance_metric,
            "memory_threshold_mb": self.memory_threshold_mb,
            "add_count": self._add_count,
            "search_count": self._search_count,
            "eviction_count": self._eviction_count,
            "collection_eviction_count": self._collection_eviction_count,
            "memory_pressure_evictions": self._memory_pressure_evictions,
            "collections_count": len(self._collections),
            "memory_usage_pct": (len(self.ids) / self.max_documents * 100) if self.max_documents > 0 else 0
        }
        
        # Add embedding dimension if we have data
        if self.embeddings:
            if NUMPY_AVAILABLE and hasattr(self.embeddings[0], 'shape'):
                stats["embedding_dimension"] = self.embeddings[0].shape[0]
                # Estimate memory usage in MB
                embedding_size = self.embeddings[0].nbytes * len(self.embeddings) / (1024 * 1024)
            else:
                # For lists, get length of first embedding
                stats["embedding_dimension"] = len(self.embeddings[0])
                # Estimate memory usage for lists (8 bytes per float)
                embedding_size = len(self.embeddings) * len(self.embeddings[0]) * 8 / (1024 * 1024)
            
            text_size = sum(len(doc) for doc in self.documents) / (1024 * 1024)
            stats["estimated_memory_mb"] = embedding_size + text_size
        
        # Add current memory status
        try:
            stats["process_memory_mb"] = self._get_cached_memory_info()
            
            vm = psutil.virtual_memory()
            stats["system_available_mb"] = vm.available / (1024 * 1024)
            stats["system_percent_used"] = vm.percent
        except Exception:
            pass
        
        return stats
    
    def add_documents(self, 
                     collection_name: str,
                     documents: List[str],
                     embeddings: Union[List[List[float]], np.ndarray],
                     metadatas: List[dict],
                     ids: List[str]) -> bool:
        """
        Add documents to a collection (compatibility method).
        
        Args:
            collection_name: Name of the collection
            documents: List of document texts
            embeddings: Document embeddings
            metadatas: List of metadata dicts
            ids: Document IDs
            
        Returns:
            True if successful
        """
        # Store current collection
        self._current_collection = collection_name
        
        # Convert embeddings to numpy array if needed and numpy is available
        if NUMPY_AVAILABLE and isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        
        # Initialize collection if it doesn't exist
        if collection_name not in self._collections:
            # Check if we need to evict a collection
            if len(self._collections) >= self.max_collections:
                # Find and remove least recently used collection
                lru_collection = min(self._collection_access_time.keys(), 
                                   key=lambda k: self._collection_access_time.get(k, 0))
                del self._collections[lru_collection]
                del self._collection_access_time[lru_collection]
                self._collection_eviction_count += 1
                logger.debug(f"Evicted collection '{lru_collection}' (total evictions: {self._collection_eviction_count})")
            
            self._collections[collection_name] = {
                "ids": [],
                "embeddings": [],
                "documents": [],
                "metadata": []
            }
        
        # Update access time
        self._collection_access_time[collection_name] = time.time()
        
        # Add to the main store (for backward compatibility)
        self.add(ids, embeddings, documents, metadatas)
        
        # Also track in collections with size limit per collection
        collection = self._collections[collection_name]
        
        # Limit collection size to prevent unbounded growth
        max_per_collection = self.max_documents // max(len(self._collections), 1)
        
        # If adding would exceed limit, remove oldest items
        total_after_add = len(collection["ids"]) + len(ids)
        if total_after_add > max_per_collection:
            items_to_remove = total_after_add - max_per_collection
            collection["ids"] = collection["ids"][items_to_remove:]
            collection["embeddings"] = collection["embeddings"][items_to_remove:]
            collection["documents"] = collection["documents"][items_to_remove:]
            collection["metadata"] = collection["metadata"][items_to_remove:]
        
        # Now add the new items
        collection["ids"].extend(ids)
        collection["embeddings"].extend(embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings)
        collection["documents"].extend(documents)
        collection["metadata"].extend(metadatas)
        
        return True
    
    def list_collections(self) -> List[str]:
        """List all collection names."""
        # Always include default collection if we have data
        collections = set(self._collections.keys())
        if self.ids:
            collections.add("default")
        return list(collections)


# Factory function for creating vector stores

def create_vector_store(store_type: str, 
                       persist_directory: Optional[Union[str, Path]] = None,
                       collection_name: str = "default",
                       distance_metric: str = "cosine",
                       **kwargs) -> VectorStore:
    """
    Factory function to create vector stores.
    
    Args:
        store_type: Type of vector store - "chroma", "memory"
        persist_directory: Directory for persistent stores (required for chroma)
        collection_name: Name of the collection
        distance_metric: Distance metric to use
        **kwargs: Additional arguments passed to the store constructor
        
    Returns:
        Vector store instance
        
    Raises:
        ValueError: If required parameters are missing or store type is unknown
    """
    store_type = store_type.lower()
    
    if store_type == "chroma":
        if persist_directory is None:
            raise ValueError("persist_directory required for ChromaDB")
        return ChromaVectorStore(
            persist_directory=persist_directory,
            collection_name=collection_name,
            distance_metric=distance_metric,
            **kwargs
        )
    
    elif store_type == "memory" or store_type == "inmemory":
        max_documents = kwargs.pop('max_documents', 10000)
        return InMemoryVectorStore(
            distance_metric=distance_metric,
            max_documents=max_documents,
            **kwargs
        )
    
    else:
        # Default to in-memory if unknown type
        logger.warning(f"Unknown vector store type: {store_type}. Using in-memory store.")
        max_documents = kwargs.pop('max_documents', 10000)
        return InMemoryVectorStore(
            distance_metric=distance_metric,
            max_documents=max_documents,
            **kwargs
        )