"""
Simple cache implementation for the RAG service.

This provides a lightweight caching solution for search results,
replacing the complex cache service from the old implementation.
"""

import hashlib
import json
import time
import threading
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import OrderedDict
from loguru import logger
import sys
import asyncio

from tldw_chatbook.Metrics.metrics_logger import log_counter, log_histogram, log_gauge, timeit



@dataclass
class CacheEntry:
    """A single cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    
    def access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()


class SimpleRAGCache:
    """
    Simple LRU cache for RAG search results with async-safe operations.
    
    Features:
    - LRU eviction policy
    - TTL support
    - Size limits
    - Basic metrics
    - Async-safe for single-user Textual app
    """
    
    def __init__(self, 
                 max_size: int = 100,
                 ttl_seconds: float = 3600,  # 1 hour default
                 enabled: bool = True,
                 ttl_by_search_type: Optional[Dict[str, float]] = None,
                 max_memory_mb: float = 100.0):  # Default 100MB max memory
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of entries to cache
            ttl_seconds: Default time-to-live for cache entries in seconds
            enabled: Whether caching is enabled
            ttl_by_search_type: Optional dict mapping search types to specific TTLs
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled
        self.ttl_by_search_type = ttl_by_search_type or {}
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        
        # Use OrderedDict for LRU behavior
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Use threading.RLock for thread-safe operations
        # RLock allows the same thread to acquire the lock multiple times
        self._lock = threading.RLock()
        
        # Keep asyncio.Lock for async methods compatibility
        self._async_lock = asyncio.Lock()
        
        # Metrics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._current_memory_bytes = 0
        self._total_requests = 0
        self._last_prune_time = time.time()
        self._prune_interval = min(ttl_seconds / 2, 1800)  # Prune every half TTL or 30 minutes, whichever is less
        
        # Log initialization
        logger.info(f"Cache initialized: max_size={max_size}, ttl={ttl_seconds}s, enabled={enabled}")
        log_gauge("cache_max_size", max_size)
        log_gauge("cache_ttl_seconds", ttl_seconds)
        log_counter("cache_initialized", labels={"enabled": str(enabled)})
    
    def _make_key(self, 
                  query: str, 
                  search_type: str,
                  top_k: int,
                  filters: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a cache key from search parameters.
        
        Uses xxhash for better performance than MD5.
        
        Args:
            query: The search query
            search_type: Type of search (semantic, hybrid, keyword)
            top_k: Number of results
            filters: Optional metadata filters
            
        Returns:
            A unique cache key
        """
        # Create a stable representation of the parameters
        key_parts = [
            query.lower().strip(),
            search_type,
            str(top_k),
            json.dumps(filters or {}, sort_keys=True)
        ]
        
        # Use a faster hash function - fallback to md5 if xxhash not available
        key_str = "|".join(key_parts)
        try:
            import xxhash
            return xxhash.xxh64(key_str.encode()).hexdigest()
        except ImportError:
            # Fallback to MD5 for stable hashing across processes
            # MD5 is fine for cache keys - we don't need cryptographic security
            return hashlib.md5(key_str.encode()).hexdigest()
    
    async def get_async(self, 
                       query: str,
                       search_type: str,
                       top_k: int,
                       filters: Optional[Dict[str, Any]] = None) -> Optional[Tuple[List[Any], str]]:
        """
        Async-safe get cached search results.
        
        Args:
            query: The search query
            search_type: Type of search
            top_k: Number of results
            filters: Optional metadata filters
            
        Returns:
            Tuple of (results, context) if found and valid, None otherwise
        """
        if not self.enabled:
            return None
        
        async with self._async_lock:
            self._total_requests += 1
            
            # Check if we need to prune expired entries
            current_time = time.time()
            if current_time - self._last_prune_time > self._prune_interval:
                await self._prune_expired_async()
                self._last_prune_time = current_time
            
            key = self._make_key(query, search_type, top_k, filters)
            log_counter("cache_request", labels={"type": search_type})
            
            if key not in self._cache:
                self._misses += 1
                log_counter("cache_miss", labels={"type": search_type})
                logger.debug(f"Cache miss for query: '{query[:50]}...'")
                return None
            
            entry = self._cache[key]
            
            # Check TTL - use search type specific TTL if available
            ttl = self.ttl_by_search_type.get(search_type, self.ttl_seconds)
            age = time.time() - entry.timestamp
            if ttl is not None and age > ttl:
                # Expired
                del self._cache[key]
                self._misses += 1
                log_counter("cache_expired", labels={"type": search_type})
                log_histogram("cache_entry_expired_age_seconds", age)
                logger.debug(f"Cache entry expired for query: '{query[:50]}...' (age: {age:.1f}s, ttl: {ttl}s)")
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.access()
            
            self._hits += 1
            log_counter("cache_hit", labels={"type": search_type})
            log_histogram("cache_entry_age_seconds", age)
            log_histogram("cache_entry_access_count", entry.access_count)
            
            # Update hit rate metric
            hit_rate = self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0
            log_gauge("cache_hit_rate", hit_rate)
            
            logger.debug(f"Cache hit for query: '{query[:50]}...' (age: {age:.1f}s, accesses: {entry.access_count})")
            
            return entry.value
    
    def get(self, 
            query: str,
            search_type: str,
            top_k: int,
            filters: Optional[Dict[str, Any]] = None) -> Optional[Tuple[List[Any], str]]:
        """
        Thread-safe synchronous cache get.
        
        This method is safe to call from any context and will not cause deadlocks.
        For better performance in async contexts, use get_async() directly.
        """
        if not self.enabled:
            return None
        
        # Use a separate thread to avoid event loop conflicts
        import concurrent.futures
        import threading
        
        # Check if we're in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context, use thread pool to avoid blocking
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._sync_get_impl, query, search_type, top_k, filters)
                return future.result(timeout=1.0)  # 1 second timeout for cache operations
        except RuntimeError:
            # No running loop, safe to run directly
            return self._sync_get_impl(query, search_type, top_k, filters)
    
    def _sync_get_impl(self, query: str, search_type: str, top_k: int, 
                       filters: Optional[Dict[str, Any]]) -> Optional[Tuple[List[Any], str]]:
        """Internal synchronous implementation using threading lock."""
        with self._lock:
            self._total_requests += 1
            
            # Check if we need to prune expired entries
            current_time = time.time()
            if current_time - self._last_prune_time > self._prune_interval:
                self._prune_expired_sync()
                self._last_prune_time = current_time
            
            key = self._make_key(query, search_type, top_k, filters)
            log_counter("cache_request", labels={"type": search_type})
            
            if key not in self._cache:
                self._misses += 1
                log_counter("cache_miss", labels={"type": search_type})
                logger.debug(f"Cache miss for query: '{query[:50]}...'")
                return None
            
            entry = self._cache[key]
            
            # Get TTL for this search type
            ttl = self.ttl_by_search_type.get(search_type, self.ttl_seconds)
            
            # Check if entry has expired
            if time.time() - entry.timestamp > ttl:
                self._misses += 1
                log_counter("cache_expired", labels={"type": search_type})
                logger.debug(f"Cache expired for query: '{query[:50]}...'")
                del self._cache[key]
                self._update_memory_sync()
                return None
            
            # Update access stats
            entry.access()
            
            # Move to end for LRU
            self._cache.move_to_end(key)
            
            self._hits += 1
            log_counter("cache_hit", labels={"type": search_type})
            logger.debug(f"Cache hit for query: '{query[:50]}...'")
            
            # Check for corrupted entry (None value)
            if entry.value is None:
                logger.warning(f"Corrupted cache entry detected for key: {key}")
                # Remove corrupted entry
                del self._cache[key]
                self._hits -= 1  # Correct the hit count
                self._misses += 1  # Count as miss
                return None
            
            return entry.value
    
    async def put_async(self,
                       query: str,
                       search_type: str,
                       top_k: int,
                       results: List[Any],
                       context: str,
                       filters: Optional[Dict[str, Any]] = None) -> None:
        """
        Async-safe cache search results.
        
        Args:
            query: The search query
            search_type: Type of search
            top_k: Number of results
            results: Search results to cache
            context: Context string to cache
            filters: Optional metadata filters
        """
        if not self.enabled:
            return
        
        async with self._async_lock:
            key = self._make_key(query, search_type, top_k, filters)
            
            # Calculate memory for new entry
            entry = CacheEntry(
                key=key,
                value=(results, context),
                timestamp=time.time()
            )
            entry_memory = self._deep_getsizeof(entry)
            
            # Evict entries if needed (by size or memory)
            while ((len(self._cache) >= self.max_size and key not in self._cache) or
                   (self._current_memory_bytes + entry_memory > self.max_memory_bytes)):
                
                if not self._cache:
                    # Cache is empty but we still exceed memory - entry is too large
                    logger.warning(f"Entry too large for cache: {entry_memory / 1024 / 1024:.1f}MB")
                    return
                
                # Evict least recently used
                oldest_key = next(iter(self._cache))
                evicted_entry = self._cache[oldest_key]
                evicted_memory = self._deep_getsizeof(evicted_entry)
                
                del self._cache[oldest_key]
                self._evictions += 1
                self._current_memory_bytes -= evicted_memory
                
                # Log eviction details
                log_counter("cache_eviction", labels={"type": search_type, "reason": "memory" if self._current_memory_bytes + entry_memory > self.max_memory_bytes else "size"})
                eviction_age = time.time() - evicted_entry.timestamp
                log_histogram("cache_evicted_entry_age_seconds", eviction_age)
                log_histogram("cache_evicted_entry_access_count", evicted_entry.access_count)
                log_histogram("cache_evicted_entry_memory_mb", evicted_memory / 1024 / 1024)
                logger.debug(f"Evicted cache entry (age: {eviction_age:.1f}s, accesses: {evicted_entry.access_count}, memory: {evicted_memory / 1024 / 1024:.1f}MB)")
            
            # Remove old entry if updating
            if key in self._cache:
                old_entry = self._cache[key]
                self._current_memory_bytes -= self._deep_getsizeof(old_entry)
            
            # Store the entry
            self._cache[key] = entry
            self._current_memory_bytes += entry_memory
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            # Log cache statistics
            log_counter("cache_put", labels={"type": search_type})
            log_histogram("cache_result_count", len(results))
            log_histogram("cache_context_size", len(context))
            log_gauge("cache_current_size", len(self._cache))
            log_gauge("cache_eviction_count", self._evictions)
            log_gauge("cache_memory_usage_mb", self._current_memory_bytes / 1024 / 1024)
            log_histogram("cache_entry_size_mb", entry_memory / 1024 / 1024)
            
            logger.debug(f"Cached results for query: '{query[:50]}...' ({len(results)} results, {entry_memory / 1024 / 1024:.1f}MB)")
    
    def put(self,
            query: str,
            search_type: str,
            top_k: int,
            results: List[Any],
            context: str,
            filters: Optional[Dict[str, Any]] = None) -> None:
        """
        Thread-safe synchronous cache put.
        
        This method is safe to call from any context and will not cause deadlocks.
        For better performance in async contexts, use put_async() directly.
        """
        if not self.enabled:
            return
        
        # Use a separate thread to avoid event loop conflicts
        import concurrent.futures
        
        # Check if we're in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context, use thread pool to avoid blocking
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._sync_put_impl, query, search_type, top_k, results, context, filters)
                future.result(timeout=1.0)  # 1 second timeout for cache operations
        except RuntimeError:
            # No running loop, safe to run directly
            self._sync_put_impl(query, search_type, top_k, results, context, filters)
    
    def _sync_put_impl(self, query: str, search_type: str, top_k: int,
                       results: List[Any], context: str, filters: Optional[Dict[str, Any]]) -> None:
        """Internal synchronous implementation using threading lock."""
        with self._lock:
            key = self._make_key(query, search_type, top_k, filters)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=(results, context),
                timestamp=time.time()
            )
            
            # Check memory pressure before adding
            try:
                entry_memory = self._estimate_entry_size(entry)
            except Exception as e:
                logger.warning(f"Error estimating entry size, using fallback: {e}")
                # Fallback to a reasonable default size (1KB)
                entry_memory = 1024
            
            # Evict if necessary to make room
            while (self._current_memory_bytes + entry_memory > self.max_memory_bytes or 
                   len(self._cache) >= self.max_size) and self._cache:
                self._evict_lru_sync()
            
            # Add new entry
            self._cache[key] = entry
            self._current_memory_bytes += entry_memory
            
            # Update metrics
            log_counter("cache_put", labels={"type": search_type})
            log_gauge("cache_size", len(self._cache))
            log_gauge("cache_memory_mb", self._current_memory_bytes / 1024 / 1024)
            
            logger.debug(f"Cached results for query: '{query[:50]}...' ({len(results)} results, {entry_memory / 1024 / 1024:.1f}MB)")
    
    async def clear_async(self) -> None:
        """Async-safe clear all cache entries."""
        async with self._async_lock:
            size_before = len(self._cache)
            memory_before = self._current_memory_bytes / 1024 / 1024
            self._cache.clear()
            self._current_memory_bytes = 0
            log_counter("cache_cleared")
            log_gauge("cache_current_size", 0)
            log_gauge("cache_memory_usage_mb", 0)
            logger.info(f"Cache cleared ({size_before} entries, {memory_before:.1f}MB removed)")
    
    def clear(self) -> None:
        """Thread-safe synchronous cache clear."""
        import concurrent.futures
        
        # Check if we're in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context, use thread pool to avoid blocking
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._sync_clear_impl)
                future.result(timeout=1.0)  # 1 second timeout
        except RuntimeError:
            # No running loop, safe to run directly
            self._sync_clear_impl()
    
    def _sync_clear_impl(self) -> None:
        """Internal synchronous implementation using threading lock."""
        with self._lock:
            size_before = len(self._cache)
            memory_before = self._current_memory_bytes / 1024 / 1024
            self._cache.clear()
            self._current_memory_bytes = 0
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            log_counter("cache_cleared")
            log_gauge("cache_size", 0)
            log_gauge("cache_memory_mb", 0)
            logger.info(f"Cache cleared ({size_before} entries, {memory_before:.1f}MB removed)")
    
    def _deep_getsizeof(self, obj, seen=None, max_depth=3, current_depth=0):
        """
        Calculate the deep size of an object, including all referenced objects.
        
        Args:
            obj: The object to measure
            seen: Set of already-seen object ids to avoid infinite recursion
            max_depth: Maximum recursion depth to prevent O(n²) complexity
            current_depth: Current recursion depth
            
        Returns:
            Size in bytes
        """
        size = sys.getsizeof(obj)
        if seen is None:
            seen = set()
            
        obj_id = id(obj)
        if obj_id in seen:
            return 0
            
        # Mark this object as seen first
        seen.add(obj_id)
        
        # Stop recursion at max depth to prevent O(n²) complexity
        if current_depth >= max_depth:
            # Return a rough estimate based on string representation
            return size + len(str(obj)) if hasattr(obj, '__str__') else size
        
        if isinstance(obj, dict):
            size += sum(self._deep_getsizeof(k, seen, max_depth, current_depth + 1) + 
                       self._deep_getsizeof(v, seen, max_depth, current_depth + 1) 
                       for k, v in obj.items())
        elif hasattr(obj, '__dict__'):
            size += self._deep_getsizeof(obj.__dict__, seen, max_depth, current_depth + 1)
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
            try:
                size += sum(self._deep_getsizeof(i, seen, max_depth, current_depth + 1) for i in obj)
            except TypeError:
                # Some iterables don't support iteration in all contexts
                pass
                
        return size
    
    def _estimate_entry_size(self, entry: CacheEntry) -> int:
        """
        Estimate the memory size of a cache entry.
        Uses a fast estimation approach instead of deep recursion.
        """
        # Base size of the CacheEntry object
        size = sys.getsizeof(entry)
        
        # Add key size
        size += sys.getsizeof(entry.key)
        
        # Estimate value size (tuple of results and context)
        if isinstance(entry.value, tuple) and len(entry.value) == 2:
            results, context = entry.value
            
            # Estimate results size (list of SearchResult objects)
            if isinstance(results, list):
                # Base list size
                size += sys.getsizeof(results)
                # Estimate 1KB per result (reasonable for SearchResult objects)
                size += len(results) * 1024
            
            # Add context string size
            if isinstance(context, str):
                size += sys.getsizeof(context)
        else:
            # Fallback: use string representation length as rough estimate
            size += len(str(entry.value)) * 2  # 2 bytes per character estimate
        
        # Add overhead for metadata (timestamp, access_count, etc)
        size += 100  # Fixed overhead estimate
        
        return size
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get cache metrics.
        
        Returns:
            Dictionary of cache statistics
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
        
        # Use tracked memory size for efficiency
        size_bytes = self._current_memory_bytes
        
        metrics = {
            "enabled": self.enabled,
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": hit_rate,
            "ttl_seconds": self.ttl_seconds,
            "size_bytes": size_bytes,
            "max_memory_bytes": self.max_memory_bytes,
            "memory_usage_percent": (size_bytes / self.max_memory_bytes * 100) if self.max_memory_bytes > 0 else 0,
            "total_requests": self._total_requests
        }
        
        # Log key metrics
        log_gauge("cache_hit_rate", hit_rate)
        log_gauge("cache_memory_estimate_mb", size_bytes / (1024 * 1024))
        log_gauge("cache_fill_ratio", len(self._cache) / self.max_size if self.max_size > 0 else 0)
        
        return metrics
    
    def log_cache_efficiency(self):
        """Log cache efficiency metrics - should be called periodically."""
        metrics = self.get_metrics()
        
        logger.info(
            f"Cache efficiency: hit_rate={metrics['hit_rate']:.2%}, "
            f"size={metrics['size']}/{metrics['max_size']}, "
            f"memory={metrics['size_bytes']/(1024*1024):.1f}MB, "
            f"evictions={metrics['evictions']}"
        )
    
    async def _prune_expired_async(self) -> int:
        """
        Internal async method to prune expired entries.
        Called automatically during cache operations.
        """
        if not self.enabled:
            return 0
        
        current_time = time.time()
        expired_keys = []
        total_age = 0
        total_accesses = 0
        
        for key, entry in self._cache.items():
            age = current_time - entry.timestamp
            if age > self.ttl_seconds:
                expired_keys.append(key)
                total_age += age
                total_accesses += entry.access_count
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            avg_age = total_age / len(expired_keys)
            avg_accesses = total_accesses / len(expired_keys)
            log_counter("cache_entries_expired", value=len(expired_keys))
            log_histogram("cache_pruned_avg_age_seconds", avg_age)
            log_histogram("cache_pruned_avg_access_count", avg_accesses)
            log_gauge("cache_current_size", len(self._cache))
            logger.debug(f"Auto-pruned {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    @timeit("cache_prune_expired")
    def prune_expired(self) -> int:
        """
        Remove expired entries.
        
        Returns:
            Number of entries removed
        """
        if not self.enabled:
            return 0
        
        current_time = time.time()
        expired_keys = []
        total_age = 0
        total_accesses = 0
        
        for key, entry in self._cache.items():
            age = current_time - entry.timestamp
            if age > self.ttl_seconds:
                expired_keys.append(key)
                total_age += age
                total_accesses += entry.access_count
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            avg_age = total_age / len(expired_keys)
            avg_accesses = total_accesses / len(expired_keys)
            log_counter("cache_entries_expired", value=len(expired_keys))
            log_histogram("cache_pruned_avg_age_seconds", avg_age)
            log_histogram("cache_pruned_avg_access_count", avg_accesses)
            log_gauge("cache_current_size", len(self._cache))
            logger.info(f"Pruned {len(expired_keys)} expired cache entries (avg age: {avg_age:.1f}s)")
        
        return len(expired_keys)
    
    def _prune_expired_sync(self) -> int:
        """
        Internal synchronous method to prune expired entries.
        Assumes lock is already held.
        """
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self._cache.items():
            # Get TTL for the search type (stored in entry if available)
            ttl = self.ttl_seconds
            age = current_time - entry.timestamp
            if age > ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            self._update_memory_sync()
            log_counter("cache_entries_expired", value=len(expired_keys))
        
        return len(expired_keys)
    
    def _evict_lru_sync(self) -> None:
        """
        Evict least recently used entry.
        Assumes lock is already held.
        """
        if not self._cache:
            return
        
        # Get the first item (least recently used)
        key, entry = next(iter(self._cache.items()))
        del self._cache[key]
        
        # Update memory tracking
        entry_size = self._estimate_entry_size(entry)
        self._current_memory_bytes = max(0, self._current_memory_bytes - entry_size)
        self._evictions += 1
        
        log_counter("cache_eviction")
        logger.debug(f"Evicted LRU entry: {key[:16]}...")
    
    def _update_memory_sync(self) -> None:
        """
        Update memory usage tracking.
        Assumes lock is already held.
        """
        total_memory = 0
        for entry in self._cache.values():
            total_memory += self._estimate_entry_size(entry)
        self._current_memory_bytes = total_memory
        log_gauge("cache_memory_mb", self._current_memory_bytes / 1024 / 1024)
    
    def __len__(self) -> int:
        """Get the number of cached entries."""
        return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        return key in self._cache


# Global cache instance (can be replaced with per-service instance if needed)
_global_cache: Optional[SimpleRAGCache] = None


def get_rag_cache(max_size: int = 100,
                  ttl_seconds: float = 3600,
                  enabled: bool = True,
                  ttl_by_search_type: Optional[Dict[str, float]] = None) -> SimpleRAGCache:
    """
    Get or create the global RAG cache instance.
    
    Args:
        max_size: Maximum number of entries
        ttl_seconds: Time-to-live in seconds
        enabled: Whether caching is enabled
        ttl_by_search_type: Optional dict mapping search types to specific TTLs
        
    Returns:
        The cache instance
    """
    global _global_cache
    
    if _global_cache is None:
        _global_cache = SimpleRAGCache(
            max_size=max_size,
            ttl_seconds=ttl_seconds,
            enabled=enabled,
            ttl_by_search_type=ttl_by_search_type
        )
    
    return _global_cache