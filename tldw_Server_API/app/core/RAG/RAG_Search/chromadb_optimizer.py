# chromadb_optimizer.py - ChromaDB-specific Optimizations
"""
Optimizations specifically for ChromaDB that work with its existing features.

Focuses on:
- Query result caching (ChromaDB doesn't cache query results)
- Hybrid search optimization (combining with FTS results)
- Collection partitioning strategies
- Batch operation optimization
- Connection pooling for concurrent requests
"""

import asyncio
import time
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from loguru import logger
import chromadb
from chromadb.config import Settings


@dataclass
class ChromaDBOptimizationConfig:
    """Configuration for ChromaDB-specific optimizations"""
    # Caching
    enable_result_cache: bool = True
    cache_size: int = 1000
    cache_ttl_seconds: int = 3600
    
    # Hybrid search
    hybrid_alpha: float = 0.7  # Balance between vector and FTS
    
    # Batch optimization
    batch_size: int = 100
    
    # Connection pooling
    max_connections: int = 10
    
    # Collection strategies
    partition_by_date: bool = False
    partition_by_source: bool = False
    max_collection_size: int = 1_000_000


class QueryResultCache:
    """Cache for ChromaDB query results"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0
    
    def _make_key(self, query: str, collection: str, kwargs: Dict) -> str:
        """Create cache key from query parameters"""
        key_data = {
            "query": query,
            "collection": collection,
            "kwargs": kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def get(self, query: str, collection: str, **kwargs) -> Optional[Any]:
        """Get cached results"""
        key = self._make_key(query, collection, kwargs)
        
        async with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None
            
            result, timestamp = self._cache[key]
            
            # Check TTL
            if time.time() - timestamp > self.ttl:
                del self._cache[key]
                self.misses += 1
                return None
            
            # Move to end (LRU)
            self._cache.move_to_end(key)
            self.hits += 1
            return result
    
    async def set(self, query: str, collection: str, result: Any, **kwargs):
        """Cache query results"""
        key = self._make_key(query, collection, kwargs)
        
        async with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._cache.popitem(last=False)
            
            self._cache[key] = (result, time.time())
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0


class ChromaDBOptimizer:
    """ChromaDB-specific optimizations"""
    
    def __init__(self, config: ChromaDBOptimizationConfig):
        self.config = config
        
        # Result cache
        self.result_cache = QueryResultCache(
            max_size=config.cache_size,
            ttl=config.cache_ttl_seconds
        ) if config.enable_result_cache else None
        
        # Thread pool for concurrent operations
        self.executor = ThreadPoolExecutor(max_workers=config.max_connections)
        
        # Client pool
        self._clients: List[chromadb.Client] = []
        self._client_lock = asyncio.Lock()
        
        logger.info("Initialized ChromaDB optimizer")
    
    async def get_client(self, path: str) -> chromadb.Client:
        """Get client from pool"""
        async with self._client_lock:
            if not self._clients:
                # Create new client
                client = chromadb.PersistentClient(
                    path=path,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=False
                    )
                )
                return client
            return self._clients.pop()
    
    async def return_client(self, client: chromadb.Client):
        """Return client to pool"""
        async with self._client_lock:
            if len(self._clients) < self.config.max_connections:
                self._clients.append(client)
    
    async def search_with_cache(self, collection_name: str, query_text: str,
                               query_embeddings: Optional[List[float]] = None,
                               n_results: int = 10,
                               where: Optional[Dict] = None,
                               **kwargs) -> Dict[str, Any]:
        """Search with result caching"""
        # Check cache first
        if self.result_cache:
            cached = await self.result_cache.get(
                query_text or str(query_embeddings),
                collection_name,
                n_results=n_results,
                where=where,
                **kwargs
            )
            if cached is not None:
                logger.debug(f"Cache hit for query in {collection_name}")
                return cached
        
        # Perform actual search
        # This would be integrated with actual ChromaDB client
        result = {
            "ids": [],
            "embeddings": [],
            "documents": [],
            "metadatas": [],
            "distances": []
        }
        
        # Cache result
        if self.result_cache:
            await self.result_cache.set(
                query_text or str(query_embeddings),
                collection_name,
                result,
                n_results=n_results,
                where=where,
                **kwargs
            )
        
        return result
    
    def optimize_hybrid_search(self, vector_results: Dict[str, Any],
                             fts_results: List[Dict[str, Any]],
                             alpha: Optional[float] = None) -> List[Dict[str, Any]]:
        """Optimize combination of vector and FTS results"""
        alpha = alpha or self.config.hybrid_alpha
        
        # Create score map for vector results
        vector_scores = {}
        if vector_results.get("ids"):
            for i, doc_id in enumerate(vector_results["ids"][0]):
                # Convert distance to similarity score
                distance = vector_results["distances"][0][i]
                similarity = 1 / (1 + distance)
                vector_scores[doc_id] = similarity * alpha
        
        # Create score map for FTS results
        fts_scores = {}
        for result in fts_results:
            doc_id = result.get("id", str(result.get("rowid", "")))
            # Normalize FTS rank to 0-1
            rank = result.get("rank", 0)
            normalized_score = 1 / (1 + abs(rank))
            fts_scores[doc_id] = normalized_score * (1 - alpha)
        
        # Combine scores
        all_ids = set(vector_scores.keys()) | set(fts_scores.keys())
        combined_results = []
        
        for doc_id in all_ids:
            combined_score = vector_scores.get(doc_id, 0) + fts_scores.get(doc_id, 0)
            
            # Get document data from either result set
            doc_data = None
            metadata = {}
            
            # From vector results
            if doc_id in vector_scores and vector_results.get("documents"):
                idx = vector_results["ids"][0].index(doc_id)
                doc_data = vector_results["documents"][0][idx]
                metadata = vector_results["metadatas"][0][idx] if vector_results.get("metadatas") else {}
            
            # From FTS results
            elif doc_id in fts_scores:
                for fts_result in fts_results:
                    if str(fts_result.get("id", fts_result.get("rowid", ""))) == doc_id:
                        doc_data = fts_result.get("content", "")
                        metadata = fts_result.get("metadata", {})
                        break
            
            if doc_data:
                combined_results.append({
                    "id": doc_id,
                    "score": combined_score,
                    "document": doc_data,
                    "metadata": metadata,
                    "vector_score": vector_scores.get(doc_id, 0),
                    "fts_score": fts_scores.get(doc_id, 0)
                })
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        
        return combined_results
    
    async def batch_add_optimized(self, collection, documents: List[str],
                                 embeddings: List[List[float]],
                                 metadatas: List[Dict[str, Any]],
                                 ids: List[str]) -> None:
        """Optimized batch addition with chunking"""
        batch_size = self.config.batch_size
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            
            # Run in executor to avoid blocking
            await asyncio.get_event_loop().run_in_executor(
                self.executor,
                collection.add,
                embeddings[i:batch_end],
                metadatas[i:batch_end],
                documents[i:batch_end],
                ids[i:batch_end]
            )
            
            logger.debug(f"Added batch {i//batch_size + 1} ({batch_end - i} documents)")
    
    def get_collection_strategy(self, num_documents: int,
                              metadata: Optional[Dict[str, Any]] = None) -> str:
        """Determine optimal collection partitioning strategy"""
        if num_documents > self.config.max_collection_size:
            if self.config.partition_by_date and metadata and "date" in metadata:
                # Partition by year-month
                date_str = metadata["date"]
                return f"collection_{date_str[:7]}"  # YYYY-MM
            
            elif self.config.partition_by_source and metadata and "source" in metadata:
                # Partition by source
                source = metadata["source"].lower().replace(" ", "_")
                return f"collection_{source}"
            
            else:
                # Partition by document count
                partition_num = num_documents // self.config.max_collection_size
                return f"collection_part_{partition_num}"
        
        return "main_collection"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        stats = {
            "cache": {
                "enabled": self.config.enable_result_cache,
                "hit_rate": f"{self.result_cache.hit_rate:.2%}" if self.result_cache else "N/A",
                "size": len(self.result_cache._cache) if self.result_cache else 0
            },
            "config": {
                "hybrid_alpha": self.config.hybrid_alpha,
                "batch_size": self.config.batch_size,
                "max_connections": self.config.max_connections
            }
        }
        
        return stats


# Integration helper for existing RAG pipeline
class OptimizedChromaStore:
    """Drop-in replacement for ChromaDB operations with optimizations"""
    
    def __init__(self, path: str, collection_name: str,
                 optimization_config: Optional[ChromaDBOptimizationConfig] = None):
        self.path = path
        self.collection_name = collection_name
        self.config = optimization_config or ChromaDBOptimizationConfig()
        self.optimizer = ChromaDBOptimizer(self.config)
        
        # Initialize client
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(collection_name)
    
    async def search(self, query_text: str = None,
                    query_embeddings: List[float] = None,
                    n_results: int = 10,
                    where: Dict[str, Any] = None) -> Dict[str, Any]:
        """Search with optimizations"""
        return await self.optimizer.search_with_cache(
            self.collection_name,
            query_text,
            query_embeddings,
            n_results,
            where
        )
    
    async def hybrid_search(self, query_text: str,
                          query_embeddings: List[float],
                          fts_results: List[Dict[str, Any]],
                          n_results: int = 10) -> List[Dict[str, Any]]:
        """Perform optimized hybrid search"""
        # Get vector results
        vector_results = await self.search(
            query_embeddings=query_embeddings,
            n_results=n_results * 2  # Get more for merging
        )
        
        # Optimize combination
        combined = self.optimizer.optimize_hybrid_search(
            vector_results,
            fts_results
        )
        
        return combined[:n_results]
    
    async def add_documents(self, documents: List[str],
                          embeddings: List[List[float]],
                          metadatas: List[Dict[str, Any]],
                          ids: List[str]):
        """Add documents with batch optimization"""
        await self.optimizer.batch_add_optimized(
            self.collection,
            documents,
            embeddings,
            metadatas,
            ids
        )


# Future database considerations
"""
For future vector databases, useful optimizations would include:

1. **Pinecone**:
   - Index namespace strategies
   - Metadata filtering optimization
   - Batch upsert optimization
   - Query result caching

2. **Weaviate**:
   - GraphQL query optimization
   - Hybrid search with BM25
   - Filtered vector search
   - Multi-tenancy strategies

3. **Qdrant**:
   - Collection sharding strategies
   - Payload indexing optimization
   - Filtered ANN search
   - Quantization settings

4. **Milvus**:
   - Collection partitioning
   - Index type selection (IVF, HNSW, etc.)
   - Consistency level tuning
   - Resource pool management

5. **pgvector**:
   - Index type selection (ivfflat vs hnsw)
   - Partial index strategies
   - Query planning optimization
   - Connection pooling
"""

if __name__ == "__main__":
    async def test_chromadb_optimization():
        """Test ChromaDB optimizations"""
        config = ChromaDBOptimizationConfig(
            enable_result_cache=True,
            hybrid_alpha=0.7,
            batch_size=50
        )
        
        optimizer = ChromaDBOptimizer(config)
        
        # Test hybrid search optimization
        vector_results = {
            "ids": [["doc1", "doc2", "doc3"]],
            "distances": [[0.1, 0.2, 0.3]],
            "documents": [["Vector doc 1", "Vector doc 2", "Vector doc 3"]],
            "metadatas": [[{"source": "vector"}, {"source": "vector"}, {"source": "vector"}]]
        }
        
        fts_results = [
            {"id": "doc2", "content": "FTS doc 2", "rank": -1.5},
            {"id": "doc4", "content": "FTS doc 4", "rank": -2.0},
            {"id": "doc5", "content": "FTS doc 5", "rank": -2.5}
        ]
        
        combined = optimizer.optimize_hybrid_search(vector_results, fts_results)
        
        print("Hybrid Search Results:")
        for i, result in enumerate(combined[:5]):
            print(f"{i+1}. ID: {result['id']}, Score: {result['score']:.3f}, "
                  f"Vector: {result['vector_score']:.3f}, FTS: {result['fts_score']:.3f}")
        
        print(f"\nStats: {optimizer.get_stats()}")
    
    asyncio.run(test_chromadb_optimization())