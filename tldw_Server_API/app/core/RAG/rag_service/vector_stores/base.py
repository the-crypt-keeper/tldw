"""
Base abstract class for vector store adapters.

This module defines the interface that all vector store implementations must follow,
enabling easy switching between different vector databases (ChromaDB, Pinecone, Weaviate, etc.)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class VectorStoreType(Enum):
    """Supported vector store types."""
    CHROMADB = "chromadb"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"
    MILVUS = "milvus"
    FAISS = "faiss"


@dataclass
class VectorSearchResult:
    """Result from a vector search operation."""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float  # Similarity score (higher is better)
    distance: float  # Raw distance from vector store


@dataclass
class VectorStoreConfig:
    """Configuration for vector store initialization."""
    store_type: VectorStoreType
    connection_params: Dict[str, Any]
    embedding_dim: int
    distance_metric: str = "cosine"  # cosine, euclidean, dot_product
    collection_prefix: str = "user"  # Collections will be: user_{user_id}_{type}_embeddings
    user_id: str = "0"


class VectorStoreAdapter(ABC):
    """Abstract base class for vector store adapters."""
    
    def __init__(self, config: VectorStoreConfig):
        """
        Initialize the vector store adapter.
        
        Args:
            config: Configuration for the vector store
        """
        self.config = config
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize connection to the vector store."""
        pass
    
    @abstractmethod
    async def create_collection(self, collection_name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Create a new collection/index in the vector store.
        
        Args:
            collection_name: Name of the collection to create
            metadata: Optional metadata for the collection
        """
        pass
    
    @abstractmethod
    async def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection/index from the vector store.
        
        Args:
            collection_name: Name of the collection to delete
        """
        pass
    
    @abstractmethod
    async def list_collections(self) -> List[str]:
        """
        List all available collections in the vector store.
        
        Returns:
            List of collection names
        """
        pass
    
    @abstractmethod
    async def upsert_vectors(
        self,
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> None:
        """
        Insert or update vectors in the specified collection.
        
        Args:
            collection_name: Target collection name
            ids: Unique identifiers for each vector
            vectors: Embedding vectors
            documents: Original text documents
            metadatas: Metadata for each vector
        """
        pass
    
    @abstractmethod
    async def delete_vectors(self, collection_name: str, ids: List[str]) -> None:
        """
        Delete vectors from the specified collection.
        
        Args:
            collection_name: Target collection name
            ids: IDs of vectors to delete
        """
        pass
    
    @abstractmethod
    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> List[VectorSearchResult]:
        """
        Search for similar vectors in the specified collection.
        
        Args:
            collection_name: Collection to search in
            query_vector: Query embedding vector
            k: Number of results to return
            filter: Optional metadata filters
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of search results ordered by similarity
        """
        pass
    
    @abstractmethod
    async def multi_search(
        self,
        collection_patterns: List[str],
        query_vector: List[float],
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchResult]:
        """
        Search across multiple collections matching the patterns.
        
        Args:
            collection_patterns: Patterns to match collection names (e.g., ["media_*", "notes_*"])
            query_vector: Query embedding vector
            k: Number of results per collection
            filter: Optional metadata filters
            
        Returns:
            Merged list of search results from all matching collections
        """
        pass
    
    @abstractmethod
    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics about a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dictionary with stats (count, dimension, etc.)
        """
        pass
    
    @abstractmethod
    async def optimize_collection(self, collection_name: str) -> None:
        """
        Optimize the collection for better search performance.
        Some vector stores support index optimization.
        
        Args:
            collection_name: Name of the collection to optimize
        """
        pass
    
    async def close(self) -> None:
        """Close connection to the vector store."""
        self._initialized = False
    
    def _validate_vectors(self, vectors: List[List[float]]) -> None:
        """
        Validate that all vectors have the correct dimension.
        
        Args:
            vectors: List of embedding vectors
            
        Raises:
            ValueError: If vectors have incorrect dimensions
        """
        if not vectors:
            return
        
        expected_dim = self.config.embedding_dim
        for i, vec in enumerate(vectors):
            if len(vec) != expected_dim:
                raise ValueError(
                    f"Vector at index {i} has dimension {len(vec)}, "
                    f"expected {expected_dim}"
                )