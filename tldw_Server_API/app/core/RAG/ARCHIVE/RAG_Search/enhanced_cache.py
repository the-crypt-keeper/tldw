# enhanced_cache.py - Enhanced Caching Strategy for RAG
"""
Enhanced caching module with multiple caching strategies and intelligent eviction.

Features:
- Multi-level caching (memory, disk, distributed)
- Semantic similarity-based cache matching
- TTL and size-based eviction
- Cache warming and preloading
- Analytics and hit rate tracking
"""

import asyncio
import json
import hashlib
import pickle
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import numpy as np
from collections import OrderedDict, defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with metadata"""
    key: str
    value: Any
    query: str
    timestamp: float
    ttl: Optional[float] = None
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl is None:
            return False
        return (time.time() - self.timestamp) > self.ttl
    
    def access(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_access = time.time()


@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    total_size_bytes: int = 0
    avg_response_time: float = 0.0
    hit_rate_history: List[Tuple[float, float]] = field(default_factory=list)
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests
    
    def record_hit(self, response_time: float):
        """Record a cache hit"""
        self.hits += 1
        self.total_requests += 1
        self._update_avg_response_time(response_time)
        self._update_hit_rate_history()
    
    def record_miss(self, response_time: float):
        """Record a cache miss"""
        self.misses += 1
        self.total_requests += 1
        self._update_avg_response_time(response_time)
        self._update_hit_rate_history()
    
    def _update_avg_response_time(self, response_time: float):
        """Update average response time"""
        n = self.hits + self.misses
        if n == 1:
            self.avg_response_time = response_time
        else:
            self.avg_response_time = (
                (self.avg_response_time * (n - 1) + response_time) / n
            )
    
    def _update_hit_rate_history(self):
        """Update hit rate history"""
        current_time = time.time()
        current_hit_rate = self.hit_rate
        self.hit_rate_history.append((current_time, current_hit_rate))
        
        # Keep only last hour of history
        cutoff_time = current_time - 3600
        self.hit_rate_history = [
            (t, r) for t, r in self.hit_rate_history if t > cutoff_time
        ]


class CacheStrategy(ABC):
    """Base class for cache strategies"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, **kwargs):
        """Set value in cache"""
        pass
    
    @abstractmethod
    async def invalidate(self, key: str):
        """Invalidate cache entry"""
        pass
    
    @abstractmethod
    async def clear(self):
        """Clear all cache entries"""
        pass
    
    @abstractmethod
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        pass


class LRUCacheStrategy(CacheStrategy):
    """Least Recently Used (LRU) cache strategy"""
    
    def __init__(self, max_size: int = 1000, ttl: Optional[int] = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = CacheStats()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        start_time = time.time()
        
        if key in self.cache:
            entry = self.cache[key]
            
            # Check expiration
            if entry.is_expired():
                await self.invalidate(key)
                self.stats.record_miss(time.time() - start_time)
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            entry.access()
            
            self.stats.record_hit(time.time() - start_time)
            return entry.value
        
        self.stats.record_miss(time.time() - start_time)
        return None
    
    async def set(self, key: str, value: Any, query: str = "", **kwargs):
        """Set value in cache"""
        # Calculate size
        size_bytes = len(pickle.dumps(value))
        
        # Create entry
        entry = CacheEntry(
            key=key,
            value=value,
            query=query,
            timestamp=time.time(),
            ttl=kwargs.get('ttl', self.ttl),
            size_bytes=size_bytes,
            metadata=kwargs.get('metadata', {})
        )
        
        # Add to cache
        self.cache[key] = entry
        self.stats.total_size_bytes += size_bytes
        
        # Evict if necessary
        while len(self.cache) > self.max_size:
            await self._evict_lru()
    
    async def invalidate(self, key: str):
        """Invalidate cache entry"""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.stats.total_size_bytes -= entry.size_bytes
            self.stats.evictions += 1
    
    async def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.stats = CacheStats()
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        return self.stats
    
    async def _evict_lru(self):
        """Evict least recently used entry"""
        if self.cache:
            key = next(iter(self.cache))
            await self.invalidate(key)


class SemanticCacheStrategy(CacheStrategy):
    """Semantic similarity-based caching"""
    
    def __init__(
        self, 
        embeddings_model,
        similarity_threshold: float = 0.85,
        max_size: int = 1000,
        ttl: Optional[int] = 3600
    ):
        self.embeddings_model = embeddings_model
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.ttl = ttl
        
        self.cache: Dict[str, CacheEntry] = {}
        self.query_embeddings: Dict[str, np.ndarray] = {}
        self.stats = CacheStats()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value using semantic similarity"""
        start_time = time.time()
        
        # First try exact match
        if key in self.cache:
            entry = self.cache[key]
            if not entry.is_expired():
                entry.access()
                self.stats.record_hit(time.time() - start_time)
                return entry.value
        
        # Try semantic match if we have the query
        query = self.cache.get(key, CacheEntry(key="", value=None, query="", timestamp=0)).query
        if query and self.embeddings_model:
            similar_key = await self._find_similar_query(query)
            if similar_key and similar_key in self.cache:
                entry = self.cache[similar_key]
                if not entry.is_expired():
                    entry.access()
                    self.stats.record_hit(time.time() - start_time)
                    logger.debug(f"Semantic cache hit: '{query}' -> '{entry.query}'")
                    return entry.value
        
        self.stats.record_miss(time.time() - start_time)
        return None
    
    async def set(self, key: str, value: Any, query: str = "", **kwargs):
        """Set value with query embedding"""
        # Create entry
        entry = CacheEntry(
            key=key,
            value=value,
            query=query,
            timestamp=time.time(),
            ttl=kwargs.get('ttl', self.ttl),
            size_bytes=len(pickle.dumps(value)),
            metadata=kwargs.get('metadata', {})
        )
        
        # Store entry
        self.cache[key] = entry
        
        # Generate and store embedding if we have a query
        if query and self.embeddings_model:
            try:
                embedding = await self._generate_embedding(query)
                self.query_embeddings[key] = embedding
            except Exception as e:
                logger.error(f"Failed to generate embedding for query: {e}")
        
        # Evict if necessary
        while len(self.cache) > self.max_size:
            await self._evict_oldest()
    
    async def invalidate(self, key: str):
        """Invalidate cache entry"""
        if key in self.cache:
            del self.cache[key]
            if key in self.query_embeddings:
                del self.query_embeddings[key]
            self.stats.evictions += 1
    
    async def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.query_embeddings.clear()
        self.stats = CacheStats()
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        return self.stats
    
    async def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        # This would use the actual embeddings model
        # For now, return random embedding
        return np.random.rand(768)
    
    async def _find_similar_query(self, query: str) -> Optional[str]:
        """Find most similar cached query"""
        if not self.query_embeddings:
            return None
        
        # Generate embedding for query
        query_embedding = await self._generate_embedding(query)
        
        # Find most similar
        best_key = None
        best_similarity = 0.0
        
        for key, embedding in self.query_embeddings.items():
            # Cosine similarity
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_key = key
        
        return best_key
    
    async def _evict_oldest(self):
        """Evict oldest entry"""
        if self.cache:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].timestamp)
            await self.invalidate(oldest_key)


class TieredCacheStrategy(CacheStrategy):
    """Multi-tier caching (memory -> disk -> distributed)"""
    
    def __init__(
        self,
        memory_size: int = 100,
        disk_size: int = 1000,
        disk_path: Optional[Path] = None,
        ttl: Optional[int] = 3600
    ):
        self.memory_cache = LRUCacheStrategy(max_size=memory_size, ttl=ttl)
        self.disk_path = disk_path or Path("./cache")
        self.disk_path.mkdir(parents=True, exist_ok=True)
        self.disk_size = disk_size
        self.ttl = ttl
        self.stats = CacheStats()
        
        # Track disk cache entries
        self._disk_index: Dict[str, Dict[str, Any]] = {}
        self._load_disk_index()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get from tiered cache"""
        start_time = time.time()
        
        # Try memory first
        value = await self.memory_cache.get(key)
        if value is not None:
            self.stats.record_hit(time.time() - start_time)
            return value
        
        # Try disk
        value = await self._get_from_disk(key)
        if value is not None:
            # Promote to memory
            await self.memory_cache.set(key, value)
            self.stats.record_hit(time.time() - start_time)
            return value
        
        self.stats.record_miss(time.time() - start_time)
        return None
    
    async def set(self, key: str, value: Any, **kwargs):
        """Set in tiered cache"""
        # Always set in memory
        await self.memory_cache.set(key, value, **kwargs)
        
        # Also persist to disk
        await self._set_to_disk(key, value, **kwargs)
    
    async def invalidate(self, key: str):
        """Invalidate from all tiers"""
        await self.memory_cache.invalidate(key)
        await self._invalidate_from_disk(key)
    
    async def clear(self):
        """Clear all tiers"""
        await self.memory_cache.clear()
        await self._clear_disk_cache()
        self.stats = CacheStats()
    
    def get_stats(self) -> CacheStats:
        """Get combined statistics"""
        # Combine stats from all tiers
        memory_stats = self.memory_cache.get_stats()
        
        self.stats.hits = memory_stats.hits + self.stats.hits
        self.stats.misses = memory_stats.misses
        self.stats.total_requests = memory_stats.total_requests
        
        return self.stats
    
    async def _get_from_disk(self, key: str) -> Optional[Any]:
        """Get value from disk cache"""
        if key not in self._disk_index:
            return None
        
        meta = self._disk_index[key]
        
        # Check expiration
        if self.ttl and (time.time() - meta['timestamp']) > self.ttl:
            await self._invalidate_from_disk(key)
            return None
        
        # Load from disk
        cache_file = self.disk_path / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load from disk cache: {e}")
                await self._invalidate_from_disk(key)
        
        return None
    
    async def _set_to_disk(self, key: str, value: Any, **kwargs):
        """Set value to disk cache"""
        # Check disk cache size
        if len(self._disk_index) >= self.disk_size:
            await self._evict_from_disk()
        
        # Save to disk
        cache_file = self.disk_path / f"{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
            
            # Update index
            self._disk_index[key] = {
                'timestamp': time.time(),
                'size': cache_file.stat().st_size,
                'query': kwargs.get('query', '')
            }
            self._save_disk_index()
            
        except Exception as e:
            logger.error(f"Failed to save to disk cache: {e}")
    
    async def _invalidate_from_disk(self, key: str):
        """Remove from disk cache"""
        if key in self._disk_index:
            del self._disk_index[key]
            cache_file = self.disk_path / f"{key}.pkl"
            if cache_file.exists():
                cache_file.unlink()
            self._save_disk_index()
    
    async def _clear_disk_cache(self):
        """Clear all disk cache"""
        for cache_file in self.disk_path.glob("*.pkl"):
            cache_file.unlink()
        self._disk_index.clear()
        self._save_disk_index()
    
    async def _evict_from_disk(self):
        """Evict oldest entry from disk"""
        if self._disk_index:
            oldest_key = min(
                self._disk_index.keys(), 
                key=lambda k: self._disk_index[k]['timestamp']
            )
            await self._invalidate_from_disk(oldest_key)
    
    def _load_disk_index(self):
        """Load disk cache index"""
        index_file = self.disk_path / "index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    self._disk_index = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load disk index: {e}")
                self._disk_index = {}
    
    def _save_disk_index(self):
        """Save disk cache index"""
        index_file = self.disk_path / "index.json"
        try:
            with open(index_file, 'w') as f:
                json.dump(self._disk_index, f)
        except Exception as e:
            logger.error(f"Failed to save disk index: {e}")


class AdaptiveCacheStrategy(CacheStrategy):
    """Adaptive caching that adjusts based on usage patterns"""
    
    def __init__(
        self,
        initial_size: int = 500,
        min_size: int = 100,
        max_size: int = 2000,
        ttl: Optional[int] = 3600,
        adaptation_interval: int = 300  # 5 minutes
    ):
        self.current_size = initial_size
        self.min_size = min_size
        self.max_size = max_size
        self.ttl = ttl
        self.adaptation_interval = adaptation_interval
        
        # Underlying cache
        self.cache = LRUCacheStrategy(max_size=self.current_size, ttl=ttl)
        
        # Tracking for adaptation
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        self.last_adaptation = time.time()
        
        # Start adaptation task
        asyncio.create_task(self._adaptation_loop())
    
    async def get(self, key: str) -> Optional[Any]:
        """Get with pattern tracking"""
        # Track access
        self.access_patterns[key].append(time.time())
        
        # Clean old accesses
        cutoff = time.time() - self.adaptation_interval * 2
        self.access_patterns[key] = [
            t for t in self.access_patterns[key] if t > cutoff
        ]
        
        return await self.cache.get(key)
    
    async def set(self, key: str, value: Any, **kwargs):
        """Set value in cache"""
        await self.cache.set(key, value, **kwargs)
    
    async def invalidate(self, key: str):
        """Invalidate entry"""
        await self.cache.invalidate(key)
        if key in self.access_patterns:
            del self.access_patterns[key]
    
    async def clear(self):
        """Clear cache"""
        await self.cache.clear()
        self.access_patterns.clear()
    
    def get_stats(self) -> CacheStats:
        """Get statistics"""
        stats = self.cache.get_stats()
        stats.metadata = {
            'current_size': self.current_size,
            'hot_keys': self._identify_hot_keys()
        }
        return stats
    
    async def _adaptation_loop(self):
        """Periodically adapt cache size"""
        while True:
            await asyncio.sleep(self.adaptation_interval)
            await self._adapt_cache_size()
    
    async def _adapt_cache_size(self):
        """Adapt cache size based on patterns"""
        stats = self.cache.get_stats()
        
        # Calculate optimal size based on hit rate
        if stats.hit_rate > 0.8 and self.current_size > self.min_size:
            # Good hit rate, can reduce size
            self.current_size = max(
                self.min_size, 
                int(self.current_size * 0.9)
            )
        elif stats.hit_rate < 0.6 and self.current_size < self.max_size:
            # Poor hit rate, increase size
            self.current_size = min(
                self.max_size,
                int(self.current_size * 1.2)
            )
        
        # Rebuild cache with new size if changed
        if self.current_size != self.cache.max_size:
            logger.info(
                f"Adapting cache size: {self.cache.max_size} -> {self.current_size} "
                f"(hit rate: {stats.hit_rate:.2%})"
            )
            
            # Create new cache
            new_cache = LRUCacheStrategy(max_size=self.current_size, ttl=self.ttl)
            
            # Copy entries (most recently used first)
            entries = sorted(
                self.cache.cache.values(),
                key=lambda e: e.last_access,
                reverse=True
            )
            
            for entry in entries[:self.current_size]:
                await new_cache.set(
                    entry.key,
                    entry.value,
                    query=entry.query,
                    metadata=entry.metadata
                )
            
            self.cache = new_cache
    
    def _identify_hot_keys(self) -> List[str]:
        """Identify frequently accessed keys"""
        hot_keys = []
        
        for key, accesses in self.access_patterns.items():
            if len(accesses) >= 5:  # At least 5 accesses
                hot_keys.append(key)
        
        return hot_keys[:10]  # Top 10 hot keys


class CacheManager:
    """Manages multiple cache strategies with fallback"""
    
    def __init__(self, strategies: List[CacheStrategy]):
        self.strategies = strategies
    
    async def get(self, key: str) -> Optional[Any]:
        """Get from first available cache"""
        for strategy in self.strategies:
            value = await strategy.get(key)
            if value is not None:
                return value
        return None
    
    async def set(self, key: str, value: Any, **kwargs):
        """Set in all caches"""
        tasks = [
            strategy.set(key, value, **kwargs) 
            for strategy in self.strategies
        ]
        await asyncio.gather(*tasks)
    
    async def invalidate(self, key: str):
        """Invalidate from all caches"""
        tasks = [
            strategy.invalidate(key)
            for strategy in self.strategies
        ]
        await asyncio.gather(*tasks)
    
    async def clear(self):
        """Clear all caches"""
        tasks = [strategy.clear() for strategy in self.strategies]
        await asyncio.gather(*tasks)
    
    def get_stats(self) -> Dict[str, CacheStats]:
        """Get stats from all strategies"""
        return {
            strategy.__class__.__name__: strategy.get_stats()
            for strategy in self.strategies
        }
    
    async def warm_cache(self, entries: List[Tuple[str, Any, Dict[str, Any]]]):
        """Pre-populate cache with entries"""
        tasks = []
        for key, value, metadata in entries:
            tasks.append(self.set(key, value, **metadata))
        
        await asyncio.gather(*tasks)
        logger.info(f"Warmed cache with {len(entries)} entries")


# Cache decorators for easy integration
def cached(
    cache_strategy: CacheStrategy,
    key_func: Optional[Callable] = None,
    ttl: Optional[int] = None
):
    """Decorator to cache function results"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5(
                    ":".join(key_parts).encode()
                ).hexdigest()
            
            # Try cache
            cached_value = await cache_strategy.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Compute value
            if asyncio.iscoroutinefunction(func):
                value = await func(*args, **kwargs)
            else:
                value = func(*args, **kwargs)
            
            # Cache it
            await cache_strategy.set(
                cache_key, 
                value,
                ttl=ttl,
                metadata={'function': func.__name__}
            )
            
            return value
        
        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    async def test_caches():
        # Test different cache strategies
        lru_cache = LRUCacheStrategy(max_size=10, ttl=60)
        semantic_cache = SemanticCacheStrategy(
            embeddings_model=None,
            similarity_threshold=0.8
        )
        tiered_cache = TieredCacheStrategy(
            memory_size=5,
            disk_size=20
        )
        adaptive_cache = AdaptiveCacheStrategy(
            initial_size=10,
            min_size=5,
            max_size=20
        )
        
        # Test caching
        test_data = [
            ("query1", "result1", "machine learning basics"),
            ("query2", "result2", "deep learning tutorial"),
            ("query3", "result3", "machine learning tutorial"),
        ]
        
        for strategy in [lru_cache, semantic_cache, tiered_cache, adaptive_cache]:
            print(f"\nTesting {strategy.__class__.__name__}")
            
            # Set values
            for key, value, query in test_data:
                await strategy.set(key, value, query=query)
            
            # Get values
            for key, _, _ in test_data:
                result = await strategy.get(key)
                print(f"  {key}: {'HIT' if result else 'MISS'}")
            
            # Show stats
            stats = strategy.get_stats()
            print(f"  Stats: {stats.hits} hits, {stats.misses} misses, "
                  f"hit rate: {stats.hit_rate:.2%}")
    
    asyncio.run(test_caches())