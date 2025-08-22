# advanced_cache.py
"""
Advanced caching system for the RAG service.

This module provides multi-level caching with TTL management, cache warming,
invalidation strategies, and distributed cache support.
"""

import asyncio
import hashlib
import json
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import OrderedDict, defaultdict
import heapq

from loguru import logger
import numpy as np

from .types import Document


class CacheLevel(Enum):
    """Cache levels with different characteristics."""
    L1_MEMORY = "l1_memory"      # In-memory, fastest, smallest
    L2_DISK = "l2_disk"          # Disk-based, slower, larger
    L3_DISTRIBUTED = "l3_distributed"  # Distributed, slowest, largest


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"              # Least Recently Used
    LFU = "lfu"              # Least Frequently Used
    FIFO = "fifo"            # First In First Out
    TTL = "ttl"              # Time To Live based
    ADAPTIVE = "adaptive"     # Adaptive policy based on access patterns


@dataclass
class CacheEntry:
    """A cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 1
    ttl: Optional[float] = None
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() > self.created_at + self.ttl
    
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def get_age(self) -> float:
        """Get age of entry in seconds."""
        return time.time() - self.created_at
    
    def get_frecency_score(self, half_life: float = 3600) -> float:
        """Calculate frecency score (frequency + recency)."""
        age = self.get_age()
        recency_score = 2 ** (-age / half_life)
        frequency_score = min(self.access_count / 10, 1.0)
        return recency_score * 0.7 + frequency_score * 0.3


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    entry_count: int = 0
    avg_access_time: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def record_hit(self, access_time: float):
        """Record a cache hit."""
        self.hits += 1
        self._update_avg_access_time(access_time)
    
    def record_miss(self, access_time: float):
        """Record a cache miss."""
        self.misses += 1
        self._update_avg_access_time(access_time)
    
    def _update_avg_access_time(self, access_time: float):
        """Update average access time."""
        total = self.hits + self.misses
        if total == 1:
            self.avg_access_time = access_time
        else:
            self.avg_access_time = (
                (self.avg_access_time * (total - 1) + access_time) / total
            )


class BaseCache(ABC):
    """Base class for cache implementations."""
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl: Optional[float] = None,
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    ):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of entries
            ttl: Default TTL in seconds
            eviction_policy: Eviction policy to use
        """
        self.max_size = max_size
        self.default_ttl = ttl
        self.eviction_policy = eviction_policy
        self.stats = CacheStats()
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass


class MemoryCache(BaseCache):
    """In-memory cache implementation."""
    
    def __init__(self, **kwargs):
        """Initialize memory cache."""
        super().__init__(**kwargs)
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.access_heap = []  # For LFU tracking
        self.lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        start_time = time.time()
        
        async with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check expiration
                if entry.is_expired():
                    del self.cache[key]
                    self.stats.record_miss(time.time() - start_time)
                    return None
                
                # Update access info
                entry.update_access()
                
                # Move to end for LRU
                if self.eviction_policy == EvictionPolicy.LRU:
                    self.cache.move_to_end(key)
                
                self.stats.record_hit(time.time() - start_time)
                return entry.value
            
            self.stats.record_miss(time.time() - start_time)
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None
    ) -> None:
        """Set value in cache."""
        async with self.lock:
            # Check if we need to evict
            if len(self.cache) >= self.max_size and key not in self.cache:
                await self._evict()
            
            # Calculate size
            size_bytes = len(pickle.dumps(value))
            
            # Create or update entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                ttl=ttl or self.default_ttl,
                size_bytes=size_bytes
            )
            
            self.cache[key] = entry
            self.stats.entry_count = len(self.cache)
            self.stats.total_size_bytes += size_bytes
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        async with self.lock:
            if key in self.cache:
                entry = self.cache.pop(key)
                self.stats.entry_count = len(self.cache)
                self.stats.total_size_bytes -= entry.size_bytes
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self.lock:
            self.cache.clear()
            self.stats = CacheStats()
    
    async def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        async with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if entry.is_expired():
                    del self.cache[key]
                    return False
                return True
            return False
    
    async def _evict(self) -> None:
        """Evict entries based on policy."""
        if not self.cache:
            return
        
        if self.eviction_policy == EvictionPolicy.LRU:
            # Remove least recently used
            key, entry = self.cache.popitem(last=False)
            
        elif self.eviction_policy == EvictionPolicy.LFU:
            # Remove least frequently used
            min_key = min(self.cache.keys(), key=lambda k: self.cache[k].access_count)
            entry = self.cache.pop(min_key)
            
        elif self.eviction_policy == EvictionPolicy.FIFO:
            # Remove oldest
            key, entry = self.cache.popitem(last=False)
            
        elif self.eviction_policy == EvictionPolicy.TTL:
            # Remove expired or oldest
            expired = [k for k, v in self.cache.items() if v.is_expired()]
            if expired:
                for key in expired:
                    entry = self.cache.pop(key)
            else:
                key, entry = self.cache.popitem(last=False)
                
        elif self.eviction_policy == EvictionPolicy.ADAPTIVE:
            # Use frecency score
            min_key = min(
                self.cache.keys(),
                key=lambda k: self.cache[k].get_frecency_score()
            )
            entry = self.cache.pop(min_key)
        
        self.stats.evictions += 1
        self.stats.entry_count = len(self.cache)
        self.stats.total_size_bytes -= entry.size_bytes


class MultiLevelCache:
    """Multi-level cache with L1, L2, and optional L3 caching."""
    
    def __init__(
        self,
        l1_size: int = 100,
        l2_size: int = 1000,
        l3_enabled: bool = False,
        default_ttl: float = 3600
    ):
        """
        Initialize multi-level cache.
        
        Args:
            l1_size: Size of L1 cache
            l2_size: Size of L2 cache
            l3_enabled: Whether to enable L3 cache
            default_ttl: Default TTL in seconds
        """
        self.l1_cache = MemoryCache(
            max_size=l1_size,
            ttl=default_ttl,
            eviction_policy=EvictionPolicy.LRU
        )
        
        self.l2_cache = MemoryCache(  # Could be disk-based in production
            max_size=l2_size,
            ttl=default_ttl * 2,
            eviction_policy=EvictionPolicy.LFU
        )
        
        self.l3_cache = None
        if l3_enabled:
            # Would be Redis/Memcached in production
            self.l3_cache = MemoryCache(
                max_size=l2_size * 10,
                ttl=default_ttl * 10,
                eviction_policy=EvictionPolicy.TTL
            )
        
        self.stats = CacheStats()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache, checking all levels."""
        start_time = time.time()
        
        # Check L1
        value = await self.l1_cache.get(key)
        if value is not None:
            self.stats.record_hit(time.time() - start_time)
            logger.debug(f"L1 cache hit for key: {key}")
            return value
        
        # Check L2
        value = await self.l2_cache.get(key)
        if value is not None:
            # Promote to L1
            await self.l1_cache.set(key, value)
            self.stats.record_hit(time.time() - start_time)
            logger.debug(f"L2 cache hit for key: {key}")
            return value
        
        # Check L3 if enabled
        if self.l3_cache:
            value = await self.l3_cache.get(key)
            if value is not None:
                # Promote to L2 and L1
                await self.l2_cache.set(key, value)
                await self.l1_cache.set(key, value)
                self.stats.record_hit(time.time() - start_time)
                logger.debug(f"L3 cache hit for key: {key}")
                return value
        
        self.stats.record_miss(time.time() - start_time)
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None
    ) -> None:
        """Set value in all cache levels."""
        # Set in L1
        await self.l1_cache.set(key, value, ttl)
        
        # Set in L2 with longer TTL
        l2_ttl = ttl * 2 if ttl else None
        await self.l2_cache.set(key, value, l2_ttl)
        
        # Set in L3 if enabled
        if self.l3_cache:
            l3_ttl = ttl * 10 if ttl else None
            await self.l3_cache.set(key, value, l3_ttl)
    
    async def delete(self, key: str) -> bool:
        """Delete from all cache levels."""
        results = [await self.l1_cache.delete(key)]
        results.append(await self.l2_cache.delete(key))
        
        if self.l3_cache:
            results.append(await self.l3_cache.delete(key))
        
        return any(results)
    
    async def clear(self) -> None:
        """Clear all cache levels."""
        await self.l1_cache.clear()
        await self.l2_cache.clear()
        
        if self.l3_cache:
            await self.l3_cache.clear()
        
        self.stats = CacheStats()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics from all levels."""
        stats = {
            "overall": {
                "hit_rate": self.stats.hit_rate,
                "hits": self.stats.hits,
                "misses": self.stats.misses
            },
            "l1": {
                "hit_rate": self.l1_cache.stats.hit_rate,
                "size": self.l1_cache.stats.entry_count,
                "evictions": self.l1_cache.stats.evictions
            },
            "l2": {
                "hit_rate": self.l2_cache.stats.hit_rate,
                "size": self.l2_cache.stats.entry_count,
                "evictions": self.l2_cache.stats.evictions
            }
        }
        
        if self.l3_cache:
            stats["l3"] = {
                "hit_rate": self.l3_cache.stats.hit_rate,
                "size": self.l3_cache.stats.entry_count,
                "evictions": self.l3_cache.stats.evictions
            }
        
        return stats


class CacheWarmer:
    """Warms cache with frequently accessed data."""
    
    def __init__(self, cache: Union[BaseCache, MultiLevelCache]):
        """
        Initialize cache warmer.
        
        Args:
            cache: Cache instance to warm
        """
        self.cache = cache
        self.warm_queries: List[Tuple[str, Any]] = []
        self.access_history: defaultdict[str, int] = defaultdict(int)
    
    async def warm_from_history(
        self,
        history: List[Tuple[str, Any]],
        max_items: int = 100
    ) -> None:
        """
        Warm cache from historical data.
        
        Args:
            history: List of (key, value) tuples
            max_items: Maximum items to warm
        """
        logger.info(f"Warming cache with {min(len(history), max_items)} items")
        
        for key, value in history[:max_items]:
            await self.cache.set(key, value)
        
        logger.info("Cache warming completed")
    
    async def warm_from_predictions(
        self,
        predictions: List[str],
        fetch_func: callable
    ) -> None:
        """
        Warm cache with predicted queries.
        
        Args:
            predictions: List of predicted query keys
            fetch_func: Function to fetch values for keys
        """
        logger.info(f"Warming cache with {len(predictions)} predictions")
        
        tasks = []
        for key in predictions:
            task = self._warm_single(key, fetch_func)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = sum(1 for r in results if not isinstance(r, Exception))
        logger.info(f"Successfully warmed {successful}/{len(predictions)} predictions")
    
    async def _warm_single(self, key: str, fetch_func: callable) -> None:
        """Warm a single cache entry."""
        try:
            value = await fetch_func(key)
            if value:
                await self.cache.set(key, value)
        except Exception as e:
            logger.error(f"Error warming cache for key {key}: {e}")
    
    def track_access(self, key: str) -> None:
        """Track query access for predictive warming."""
        self.access_history[key] += 1
    
    def get_top_queries(self, n: int = 50) -> List[str]:
        """Get top N most accessed queries."""
        sorted_queries = sorted(
            self.access_history.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [query for query, _ in sorted_queries[:n]]


class InvalidationStrategy:
    """Cache invalidation strategies."""
    
    def __init__(self, cache: Union[BaseCache, MultiLevelCache]):
        """
        Initialize invalidation strategy.
        
        Args:
            cache: Cache instance
        """
        self.cache = cache
        self.dependencies: Dict[str, Set[str]] = defaultdict(set)
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate cache entries matching pattern.
        
        Args:
            pattern: Pattern to match (supports wildcards)
            
        Returns:
            Number of entries invalidated
        """
        count = 0
        
        if isinstance(self.cache, MultiLevelCache):
            # Invalidate from all levels
            caches = [self.cache.l1_cache, self.cache.l2_cache]
            if self.cache.l3_cache:
                caches.append(self.cache.l3_cache)
        else:
            caches = [self.cache]
        
        for cache in caches:
            if isinstance(cache, MemoryCache):
                keys_to_delete = []
                async with cache.lock:
                    for key in cache.cache.keys():
                        if self._matches_pattern(key, pattern):
                            keys_to_delete.append(key)
                
                for key in keys_to_delete:
                    await cache.delete(key)
                    count += 1
        
        logger.info(f"Invalidated {count} cache entries matching pattern: {pattern}")
        return count
    
    async def invalidate_by_tag(self, tag: str) -> int:
        """
        Invalidate cache entries with specific tag.
        
        Args:
            tag: Tag to match
            
        Returns:
            Number of entries invalidated
        """
        count = 0
        
        if isinstance(self.cache, MultiLevelCache):
            caches = [self.cache.l1_cache, self.cache.l2_cache]
            if self.cache.l3_cache:
                caches.append(self.cache.l3_cache)
        else:
            caches = [self.cache]
        
        for cache in caches:
            if isinstance(cache, MemoryCache):
                keys_to_delete = []
                async with cache.lock:
                    for key, entry in cache.cache.items():
                        if tag in entry.metadata.get("tags", []):
                            keys_to_delete.append(key)
                
                for key in keys_to_delete:
                    await cache.delete(key)
                    count += 1
        
        logger.info(f"Invalidated {count} cache entries with tag: {tag}")
        return count
    
    def add_dependency(self, key: str, depends_on: str) -> None:
        """Add dependency relationship between cache keys."""
        self.dependencies[depends_on].add(key)
    
    async def invalidate_with_dependencies(self, key: str) -> int:
        """Invalidate key and all dependent keys."""
        count = 0
        
        # Delete the key itself
        if await self.cache.delete(key):
            count += 1
        
        # Delete all dependent keys
        if key in self.dependencies:
            for dependent_key in self.dependencies[key]:
                if await self.cache.delete(dependent_key):
                    count += 1
            
            # Clean up dependencies
            del self.dependencies[key]
        
        return count
    
    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Check if key matches pattern with wildcards."""
        import fnmatch
        return fnmatch.fnmatch(key, pattern)


class RAGCache:
    """Specialized cache for RAG operations."""
    
    def __init__(
        self,
        enable_multi_level: bool = True,
        enable_warming: bool = True,
        ttl_query: float = 3600,
        ttl_document: float = 7200
    ):
        """
        Initialize RAG cache.
        
        Args:
            enable_multi_level: Use multi-level caching
            enable_warming: Enable cache warming
            ttl_query: TTL for query results
            ttl_document: TTL for document cache
        """
        if enable_multi_level:
            self.cache = MultiLevelCache(
                l1_size=100,
                l2_size=1000,
                default_ttl=ttl_query
            )
        else:
            self.cache = MemoryCache(
                max_size=500,
                ttl=ttl_query
            )
        
        self.warmer = CacheWarmer(self.cache) if enable_warming else None
        self.invalidator = InvalidationStrategy(self.cache)
        self.ttl_query = ttl_query
        self.ttl_document = ttl_document
    
    def _generate_key(self, query: str, context: Dict[str, Any]) -> str:
        """Generate cache key from query and context."""
        # Include relevant context in key
        key_parts = [
            query,
            str(context.get("top_k", 10)),
            str(context.get("sources", [])),
            str(context.get("filters", {}))
        ]
        
        key_str = json.dumps(key_parts, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def get_cached_results(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> Optional[List[Document]]:
        """Get cached search results."""
        key = self._generate_key(query, context)
        
        # Track access for warming
        if self.warmer:
            self.warmer.track_access(key)
        
        return await self.cache.get(key)
    
    async def cache_results(
        self,
        query: str,
        context: Dict[str, Any],
        documents: List[Document]
    ) -> None:
        """Cache search results."""
        key = self._generate_key(query, context)
        
        # Add metadata for invalidation
        for doc in documents:
            if not hasattr(doc, "metadata"):
                doc.metadata = {}
            doc.metadata["cache_key"] = key
            doc.metadata["cached_at"] = time.time()
        
        await self.cache.set(key, documents, ttl=self.ttl_query)
    
    async def warm_popular_queries(self, fetch_func: callable) -> None:
        """Warm cache with popular queries."""
        if not self.warmer:
            return
        
        top_queries = self.warmer.get_top_queries(n=20)
        await self.warmer.warm_from_predictions(top_queries, fetch_func)
    
    async def invalidate_source(self, source: str) -> int:
        """Invalidate all cache entries from a specific source."""
        # This would need to track source -> key mappings
        pattern = f"*{source}*"
        return await self.invalidator.invalidate_pattern(pattern)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if isinstance(self.cache, MultiLevelCache):
            return self.cache.get_stats()
        else:
            return {
                "hit_rate": self.cache.stats.hit_rate,
                "hits": self.cache.stats.hits,
                "misses": self.cache.stats.misses,
                "size": self.cache.stats.entry_count,
                "evictions": self.cache.stats.evictions
            }


# Pipeline integration functions

async def check_advanced_cache(context: Any, **kwargs) -> Any:
    """Check advanced cache for pipeline context."""
    config = context.config.get("cache", {})
    
    if not config.get("enabled", True):
        return context
    
    # Get or create cache instance
    if not hasattr(context, "_cache_instance"):
        context._cache_instance = RAGCache(
            enable_multi_level=config.get("multi_level", True),
            enable_warming=config.get("warming", False),
            ttl_query=config.get("ttl_query", 3600),
            ttl_document=config.get("ttl_document", 7200)
        )
    
    cache = context._cache_instance
    
    # Check cache
    cached_docs = await cache.get_cached_results(
        context.query,
        context.config
    )
    
    if cached_docs:
        context.documents = cached_docs
        context.cache_hit = True
        context.metadata["cache_level"] = "advanced"
        logger.info(f"Advanced cache hit for query: {context.query[:50]}...")
    
    return context


async def store_in_advanced_cache(context: Any, **kwargs) -> Any:
    """Store results in advanced cache for pipeline context."""
    if context.cache_hit or not context.documents:
        return context
    
    config = context.config.get("cache", {})
    
    if not config.get("enabled", True):
        return context
    
    # Get cache instance
    if hasattr(context, "_cache_instance"):
        cache = context._cache_instance
        
        await cache.cache_results(
            context.query,
            context.config,
            context.documents
        )
        
        context.metadata["cached"] = True
        logger.info(f"Cached {len(context.documents)} documents")
    
    return context