"""
Semantic caching implementation for RAG service.

This module provides semantic similarity-based caching that can find
cached results for semantically similar queries, not just exact matches.
"""

import time
import hashlib
import pickle
import json
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass, field
import threading
import numpy as np
from pathlib import Path

from loguru import logger

from .advanced_cache import CacheEntry, MemoryCache


@dataclass
class SemanticCacheEntry(CacheEntry):
    """Extended cache entry with semantic information."""
    query: str = ""
    embedding: Optional[np.ndarray] = None
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    
    def access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_access = time.time()


class SemanticCache:
    """
    Semantic similarity-based cache for RAG queries.
    
    This cache can:
    - Find cached results for semantically similar queries
    - Use embeddings to measure query similarity
    - Fall back to exact matching when embeddings unavailable
    - Track usage statistics for cache optimization
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        similarity_threshold: float = 0.85,
        ttl: Optional[int] = 3600,
        persist_path: Optional[str] = None,
        embedding_model: Optional[Any] = None
    ):
        """
        Initialize semantic cache.
        
        Args:
            max_size: Maximum number of cached items
            similarity_threshold: Minimum similarity for semantic match (0-1)
            ttl: Default time-to-live in seconds
            persist_path: Optional path for cache persistence
            embedding_model: Optional model for generating embeddings
        """
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.default_ttl = ttl
        self.persist_path = persist_path
        self.embedding_model = embedding_model
        
        # Main cache storage
        self._cache: Dict[str, SemanticCacheEntry] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
        self._lock = threading.RLock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._semantic_hits = 0
        self._exact_hits = 0
        
        # Load persisted cache if available
        if persist_path:
            self.load()
    
    def _generate_key(self, query: str) -> str:
        """Generate a cache key from query."""
        return hashlib.md5(query.encode()).hexdigest()
    
    async def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if unavailable
        """
        if not self.embedding_model:
            logger.debug("No embedding model configured for semantic cache")
            return None
        
        try:
            # Check if embedding_model has the expected interface
            if hasattr(self.embedding_model, 'encode'):
                # Sentence transformers style
                embedding = self.embedding_model.encode(text)
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
            elif hasattr(self.embedding_model, 'embed'):
                # Generic embedding interface
                embedding = await self.embedding_model.embed(text)
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
            elif callable(self.embedding_model):
                # Function-based embedding model
                embedding = await self.embedding_model(text) if asyncio.iscoroutinefunction(self.embedding_model) else self.embedding_model(text)
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
            else:
                logger.warning(f"Embedding model does not have expected interface: {type(self.embedding_model)}")
                return None
            
            # Normalize the embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (0-1)
        """
        # Cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def find_similar(self, query: str, embedding: Optional[np.ndarray] = None) -> Optional[Tuple[str, float]]:
        """
        Find most similar cached query.
        
        Args:
            query: Query to match
            embedding: Pre-computed embedding (optional)
            
        Returns:
            Tuple of (cache_key, similarity_score) or None
        """
        if embedding is None:
            embedding = await self.get_embedding(query)
        
        if embedding is None or not self._embeddings:
            return None
        
        with self._lock:
            best_key = None
            best_similarity = 0.0
            
            for key, cached_embedding in self._embeddings.items():
                similarity = self._compute_similarity(embedding, cached_embedding)
                
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    # Check if entry is still valid
                    if key in self._cache and not self._cache[key].is_expired():
                        best_similarity = similarity
                        best_key = key
            
            if best_key:
                logger.debug(f"Found similar query with similarity {best_similarity:.3f}")
                return best_key, best_similarity
        
        return None
    
    async def get(self, query: str, use_semantic: bool = True) -> Optional[Any]:
        """
        Get cached result for query.
        
        Args:
            query: Query string
            use_semantic: Whether to use semantic matching
            
        Returns:
            Cached value or None
        """
        key = self._generate_key(query)
        
        with self._lock:
            # Try exact match first
            if key in self._cache:
                entry = self._cache[key]
                if not entry.is_expired():
                    entry.access()
                    self._hits += 1
                    self._exact_hits += 1
                    logger.debug(f"Exact cache hit for query: {query[:50]}...")
                    return entry.value
                else:
                    # Remove expired entry
                    del self._cache[key]
                    if key in self._embeddings:
                        del self._embeddings[key]
        
        # Try semantic match if enabled
        if use_semantic and self.embedding_model:
            embedding = await self.get_embedding(query)
            if embedding is not None:
                similar_result = await self.find_similar(query, embedding)
                
                if similar_result:
                    similar_key, similarity = similar_result
                    with self._lock:
                        if similar_key in self._cache:
                            entry = self._cache[similar_key]
                            entry.access()
                            self._hits += 1
                            self._semantic_hits += 1
                            logger.info(
                                f"Semantic cache hit (similarity={similarity:.3f}) "
                                f"for query: {query[:50]}..."
                            )
                            return entry.value
        
        self._misses += 1
        return None
    
    async def set(
        self,
        query: str,
        value: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Cache a query result.
        
        Args:
            query: Query string
            value: Value to cache
            ttl: Time-to-live in seconds
            metadata: Optional metadata to store
        """
        key = self._generate_key(query)
        
        # Generate embedding for semantic matching
        embedding = None
        if self.embedding_model:
            embedding = await self.get_embedding(query)
        
        with self._lock:
            # Create cache entry
            entry = SemanticCacheEntry(
                value=value,
                timestamp=time.time(),
                ttl=ttl or self.default_ttl,
                query=query,
                embedding=embedding
            )
            
            # Store entry
            self._cache[key] = entry
            if embedding is not None:
                self._embeddings[key] = embedding
            
            # Evict oldest if over capacity
            while len(self._cache) > self.max_size:
                self._evict_lru()
            
            logger.debug(f"Cached result for query: {query[:50]}...")
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return
        
        # Find LRU entry
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_access
        )
        
        # Remove from cache
        del self._cache[lru_key]
        if lru_key in self._embeddings:
            del self._embeddings[lru_key]
        
        logger.debug(f"Evicted LRU cache entry: {lru_key}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._embeddings.clear()
            self._hits = 0
            self._misses = 0
            self._semantic_hits = 0
            self._exact_hits = 0
            logger.info("Semantic cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            semantic_hit_rate = self._semantic_hits / self._hits if self._hits > 0 else 0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "total_hits": self._hits,
                "exact_hits": self._exact_hits,
                "semantic_hits": self._semantic_hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "semantic_hit_rate": semantic_hit_rate,
                "total_requests": total_requests,
                "similarity_threshold": self.similarity_threshold,
                "has_embeddings": len(self._embeddings)
            }
    
    def cleanup_expired(self) -> int:
        """Remove expired entries."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self._cache[key]
                if key in self._embeddings:
                    del self._embeddings[key]
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired entries")
            
            return len(expired_keys)
    
    def save(self) -> None:
        """Save cache state to disk."""
        if not self.persist_path:
            return
        
        try:
            with self._lock:
                # Prepare serializable state
                state = {
                    "cache": {},
                    "stats": {
                        "hits": self._hits,
                        "misses": self._misses,
                        "semantic_hits": self._semantic_hits,
                        "exact_hits": self._exact_hits
                    },
                    "config": {
                        "similarity_threshold": self.similarity_threshold,
                        "max_size": self.max_size,
                        "default_ttl": self.default_ttl
                    }
                }
                
                # Convert cache entries to serializable format
                for key, entry in self._cache.items():
                    state["cache"][key] = {
                        "value": entry.value,  # Note: Complex objects may need special handling
                        "query": entry.query,
                        "timestamp": entry.timestamp,
                        "ttl": entry.ttl,
                        "access_count": entry.access_count,
                        "last_access": entry.last_access
                    }
                
                # Save embeddings separately as binary
                embeddings_path = Path(self.persist_path).with_suffix('.embeddings.npz')
                if self._embeddings:
                    np.savez_compressed(embeddings_path, **self._embeddings)
                
                # Save main state as JSON
                with open(self.persist_path, 'w') as f:
                    json.dump(state, f, indent=2)
                
                logger.info(f"Saved semantic cache state ({len(self._cache)} entries)")
        except Exception as e:
            logger.error(f"Failed to save semantic cache: {e}")
    
    def load(self) -> None:
        """Load cache state from disk."""
        if not self.persist_path:
            return
        
        try:
            # Load main state
            with open(self.persist_path, 'r') as f:
                state = json.load(f)
            
            with self._lock:
                # Restore configuration
                config = state.get("config", {})
                self.similarity_threshold = config.get("similarity_threshold", self.similarity_threshold)
                
                # Restore cache entries
                for key, entry_data in state.get("cache", {}).items():
                    entry = SemanticCacheEntry(
                        value=entry_data["value"],
                        query=entry_data["query"],
                        timestamp=entry_data["timestamp"],
                        ttl=entry_data.get("ttl"),
                        access_count=entry_data.get("access_count", 0),
                        last_access=entry_data.get("last_access", time.time())
                    )
                    
                    if not entry.is_expired():
                        self._cache[key] = entry
                
                # Load embeddings
                embeddings_path = Path(self.persist_path).with_suffix('.embeddings.npz')
                if embeddings_path.exists():
                    embeddings_data = np.load(embeddings_path)
                    self._embeddings = {
                        key: embeddings_data[key]
                        for key in embeddings_data.files
                        if key in self._cache  # Only load embeddings for valid cache entries
                    }
                
                # Restore statistics
                stats = state.get("stats", {})
                self._hits = stats.get("hits", 0)
                self._misses = stats.get("misses", 0)
                self._semantic_hits = stats.get("semantic_hits", 0)
                self._exact_hits = stats.get("exact_hits", 0)
                
                logger.info(f"Loaded semantic cache ({len(self._cache)} entries, {len(self._embeddings)} embeddings)")
        except FileNotFoundError:
            logger.debug(f"No cache file found at {self.persist_path}")
        except Exception as e:
            logger.error(f"Failed to load semantic cache: {e}")


class AdaptiveCache(SemanticCache):
    """
    Adaptive cache that adjusts its behavior based on usage patterns.
    
    Features:
    - Dynamic similarity threshold adjustment
    - Automatic TTL optimization
    - Pattern-based prefetching hints
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize adaptive cache with pattern tracking."""
        super().__init__(*args, **kwargs)
        
        # Pattern tracking
        self._query_patterns: Dict[str, int] = {}
        self._similarity_scores: List[float] = []
        self._ttl_effectiveness: Dict[str, float] = {}
        
        # Adaptive parameters
        self._adaptive_threshold = self.similarity_threshold
        self._adaptive_ttl = self.default_ttl
    
    async def get(self, query: str, use_semantic: bool = True) -> Optional[Any]:
        """Get with pattern tracking."""
        result = await super().get(query, use_semantic)
        
        # Track query patterns
        self._track_pattern(query)
        
        # Adjust parameters periodically
        if self._hits + self._misses > 0 and (self._hits + self._misses) % 100 == 0:
            self._adjust_parameters()
        
        return result
    
    def _track_pattern(self, query: str) -> None:
        """Track query patterns for optimization."""
        # Extract pattern (simplified - in production use NLP)
        words = query.lower().split()[:3]  # First 3 words as pattern
        pattern = " ".join(words)
        
        with self._lock:
            self._query_patterns[pattern] = self._query_patterns.get(pattern, 0) + 1
    
    def _adjust_parameters(self) -> None:
        """Adjust cache parameters based on usage patterns."""
        with self._lock:
            # Adjust similarity threshold based on hit rate
            hit_rate = self._hits / (self._hits + self._misses)
            
            if hit_rate < 0.3 and self._adaptive_threshold > 0.7:
                # Low hit rate - be more lenient with similarity
                self._adaptive_threshold -= 0.05
                self.similarity_threshold = self._adaptive_threshold
                logger.info(f"Lowered similarity threshold to {self._adaptive_threshold:.2f}")
            elif hit_rate > 0.7 and self._adaptive_threshold < 0.95:
                # High hit rate - can be more strict
                self._adaptive_threshold += 0.05
                self.similarity_threshold = self._adaptive_threshold
                logger.info(f"Raised similarity threshold to {self._adaptive_threshold:.2f}")
    
    def get_patterns(self) -> List[Tuple[str, int]]:
        """Get most common query patterns."""
        with self._lock:
            return sorted(
                self._query_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
    
    def suggest_prefetch(self) -> List[str]:
        """Suggest queries to prefetch based on patterns."""
        patterns = self.get_patterns()
        suggestions = []
        
        for pattern, count in patterns[:5]:
            if count > 10:  # Only suggest frequently used patterns
                suggestions.append(pattern)
        
        return suggestions