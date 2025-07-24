"""
Parallel processing optimization for RAG operations.

This module provides utilities for accelerating batch operations through
multiprocessing and concurrent execution strategies.
"""

import asyncio
import time
import multiprocessing as mp
from multiprocessing import Pool, Queue, Manager
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
from queue import Empty
import signal
import sys
from functools import partial
from loguru import logger
import numpy as np

from ..config import load_settings
from ..Metrics.metrics_logger import log_counter, log_histogram, timeit
from .simplified.data_models import IndexingResult


@dataclass
class ProcessingConfig:
    """Configuration for parallel processing."""
    # Worker settings
    num_workers: Optional[int] = None  # None = use CPU count
    max_workers: int = 16  # Maximum workers to prevent resource exhaustion
    min_workers: int = 2   # Minimum workers
    
    # Batch settings
    batch_size: int = 32
    max_batch_size: int = 256
    dynamic_batching: bool = True
    
    # Performance settings
    prefetch_factor: int = 2  # Number of batches to prefetch
    timeout_seconds: float = 300.0  # 5 minutes default
    show_progress: bool = True
    progress_interval: int = 10  # Report every N items
    
    # Memory management
    max_memory_per_worker_mb: float = 1024.0  # 1GB per worker
    monitor_memory: bool = True
    
    # Error handling
    continue_on_error: bool = True
    max_retries: int = 2
    retry_delay: float = 1.0


class ProgressTracker:
    """Thread-safe progress tracking for parallel operations."""
    
    def __init__(self, total: int, desc: str = "Processing", interval: int = 10):
        self.total = total
        self.desc = desc
        self.interval = interval
        self.completed = 0
        self.failed = 0
        self.start_time = time.time()
        self._lock = threading.Lock()
        self._last_report = 0
        
    def update(self, success: bool = True):
        """Update progress and optionally report."""
        with self._lock:
            if success:
                self.completed += 1
            else:
                self.failed += 1
            
            total_processed = self.completed + self.failed
            
            # Report progress at intervals
            if total_processed % self.interval == 0 or total_processed == self.total:
                self._report_progress()
    
    def _report_progress(self):
        """Report current progress (called under lock)."""
        total_processed = self.completed + self.failed
        elapsed = time.time() - self.start_time
        rate = total_processed / elapsed if elapsed > 0 else 0
        eta = (self.total - total_processed) / rate if rate > 0 else 0
        
        logger.info(
            f"{self.desc}: {total_processed}/{self.total} "
            f"({total_processed/self.total*100:.1f}%) | "
            f"Success: {self.completed} | Failed: {self.failed} | "
            f"Rate: {rate:.1f}/s | ETA: {eta:.1f}s"
        )
        
        # Log metrics
        log_gauge("parallel_processing_progress", total_processed / self.total)
        log_histogram("parallel_processing_rate", rate)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get final statistics."""
        with self._lock:
            elapsed = time.time() - self.start_time
            return {
                "total": self.total,
                "completed": self.completed,
                "failed": self.failed,
                "success_rate": self.completed / self.total if self.total > 0 else 0,
                "elapsed_seconds": elapsed,
                "items_per_second": self.total / elapsed if elapsed > 0 else 0
            }


class BatchProcessor:
    """Optimized batch processing with dynamic batching and prefetching."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self._determine_worker_count()
        
    def _determine_worker_count(self):
        """Determine optimal number of workers."""
        cpu_count = mp.cpu_count()
        
        if self.config.num_workers is None:
            # Use 80% of CPUs, leaving some for the main process
            self.num_workers = max(self.config.min_workers, int(cpu_count * 0.8))
        else:
            self.num_workers = self.config.num_workers
        
        # Apply limits
        self.num_workers = max(self.config.min_workers, 
                              min(self.num_workers, self.config.max_workers))
        
        logger.info(f"Using {self.num_workers} workers (CPUs: {cpu_count})")
    
    @timeit("batch_process_documents")
    async def process_documents_parallel(self,
                                       documents: List[Dict[str, Any]],
                                       process_func: Callable,
                                       desc: str = "Processing documents") -> List[Any]:
        """
        Process documents in parallel with optimized batching.
        
        Args:
            documents: List of documents to process
            process_func: Function to apply to each document
            desc: Description for progress tracking
            
        Returns:
            List of results maintaining input order
        """
        if not documents:
            return []
        
        total = len(documents)
        progress = ProgressTracker(total, desc, self.config.progress_interval) if self.config.show_progress else None
        
        # Determine optimal batch size
        batch_size = self._calculate_optimal_batch_size(total)
        
        # Create batches
        batches = [documents[i:i + batch_size] for i in range(0, total, batch_size)]
        
        logger.info(f"Processing {total} documents in {len(batches)} batches of ~{batch_size} items")
        
        # Process batches
        results = [None] * total  # Pre-allocate results list
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all batches
            future_to_index = {}
            for batch_idx, batch in enumerate(batches):
                future = executor.submit(self._process_batch_wrapper, process_func, batch, batch_idx)
                future_to_index[future] = batch_idx * batch_size
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                start_idx = future_to_index[future]
                
                try:
                    batch_results = future.result(timeout=self.config.timeout_seconds)
                    
                    # Place results in correct positions
                    for i, result in enumerate(batch_results):
                        results[start_idx + i] = result
                        if progress:
                            progress.update(success=result is not None)
                    
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    # Mark batch as failed
                    batch_size_actual = min(batch_size, total - start_idx)
                    for i in range(batch_size_actual):
                        if progress:
                            progress.update(success=False)
                    
                    if not self.config.continue_on_error:
                        raise
        
        # Log final stats
        if progress:
            stats = progress.get_stats()
            logger.info(f"Parallel processing completed: {stats}")
            
            # Log metrics
            log_counter("parallel_documents_processed", value=stats["completed"])
            log_counter("parallel_documents_failed", value=stats["failed"])
            log_histogram("parallel_processing_time", stats["elapsed_seconds"])
            log_histogram("parallel_processing_throughput", stats["items_per_second"])
        
        return results
    
    def _calculate_optimal_batch_size(self, total_items: int) -> int:
        """Calculate optimal batch size based on workload."""
        if not self.config.dynamic_batching:
            return self.config.batch_size
        
        # Base calculation: distribute evenly across workers
        items_per_worker = total_items / self.num_workers
        
        # Adjust based on total size
        if total_items < 100:
            # Small dataset: smaller batches for better distribution
            optimal_size = max(1, int(items_per_worker / 2))
        elif total_items < 1000:
            # Medium dataset: balanced approach
            optimal_size = int(items_per_worker)
        else:
            # Large dataset: larger batches to reduce overhead
            optimal_size = int(items_per_worker * 1.5)
        
        # Apply limits
        optimal_size = max(1, min(optimal_size, self.config.max_batch_size))
        
        # Ensure we don't create too many small batches
        min_batches = max(self.num_workers, total_items // self.config.max_batch_size)
        max_batches = total_items  # One item per batch at most
        
        # Adjust if needed
        if total_items / optimal_size > max_batches:
            optimal_size = max(1, total_items // max_batches)
        elif total_items / optimal_size < min_batches:
            optimal_size = max(1, total_items // min_batches)
        
        return optimal_size
    
    @staticmethod
    def _process_batch_wrapper(process_func: Callable, batch: List[Any], batch_idx: int) -> List[Any]:
        """Wrapper to process a batch with error handling."""
        results = []
        
        for item in batch:
            try:
                result = process_func(item)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process item in batch {batch_idx}: {e}")
                results.append(None)
        
        return results


class EmbeddingBatchProcessor:
    """Specialized processor for embedding generation with batching."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        
    @timeit("batch_generate_embeddings")
    async def generate_embeddings_batch(self,
                                      texts: List[str],
                                      embedding_service,
                                      batch_size: Optional[int] = None) -> Tuple[np.ndarray, List[int]]:
        """
        Generate embeddings in optimized batches.
        
        Args:
            texts: List of texts to embed
            embedding_service: Service with create_embeddings_async method
            batch_size: Override default batch size
            
        Returns:
            Tuple of (embeddings array, list of failed indices)
        """
        if not texts:
            return np.array([]), []
        
        batch_size = batch_size or self.config.batch_size
        total = len(texts)
        all_embeddings = []
        failed_indices = []
        
        # Create progress tracker
        progress = ProgressTracker(total, "Generating embeddings", self.config.progress_interval) if self.config.show_progress else None
        
        # Process in batches with concurrent execution
        tasks = []
        for i in range(0, total, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_start_idx = i
            
            # Create task for this batch
            task = self._process_embedding_batch(
                embedding_service,
                batch_texts,
                batch_start_idx,
                progress
            )
            tasks.append(task)
        
        # Execute batches concurrently (but not in parallel due to model constraints)
        # We use limited concurrency to allow prefetching/preparation
        max_concurrent = min(3, len(tasks))  # Limit concurrent batches
        
        for i in range(0, len(tasks), max_concurrent):
            batch_tasks = tasks[i:i + max_concurrent]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Embedding batch failed: {result}")
                    # Mark entire batch as failed
                    failed_indices.extend(range(i, min(i + batch_size, total)))
                else:
                    embeddings, batch_failed = result
                    all_embeddings.extend(embeddings)
                    failed_indices.extend(batch_failed)
        
        # Convert to numpy array
        if all_embeddings:
            embeddings_array = np.vstack(all_embeddings)
        else:
            embeddings_array = np.array([])
        
        # Log final statistics
        if progress:
            stats = progress.get_stats()
            success_rate = (total - len(failed_indices)) / total if total > 0 else 0
            logger.info(f"Embedding generation completed: {success_rate:.1%} success rate")
            
            log_histogram("embedding_batch_success_rate", success_rate)
            log_counter("embedding_batch_failed", value=len(failed_indices))
        
        return embeddings_array, failed_indices
    
    async def _process_embedding_batch(self,
                                     embedding_service,
                                     batch_texts: List[str],
                                     start_idx: int,
                                     progress: Optional[ProgressTracker]) -> Tuple[List[np.ndarray], List[int]]:
        """Process a single batch of embeddings."""
        embeddings = []
        failed_indices = []
        
        try:
            # Generate embeddings for batch
            batch_embeddings = await embedding_service.create_embeddings_async(batch_texts)
            
            for i, embedding in enumerate(batch_embeddings):
                if embedding is not None:
                    embeddings.append(embedding)
                    if progress:
                        progress.update(success=True)
                else:
                    failed_indices.append(start_idx + i)
                    if progress:
                        progress.update(success=False)
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings for batch starting at {start_idx}: {e}")
            # Mark entire batch as failed
            for i in range(len(batch_texts)):
                failed_indices.append(start_idx + i)
                if progress:
                    progress.update(success=False)
        
        return embeddings, failed_indices


class ChunkingBatchProcessor:
    """Specialized processor for document chunking with parallel execution."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.batch_processor = BatchProcessor(config)
    
    @timeit("batch_chunk_documents")
    async def chunk_documents_batch(self,
                                  documents: List[Dict[str, Any]],
                                  chunking_service,
                                  chunk_size: int,
                                  chunk_overlap: int,
                                  method: str = "words") -> Tuple[List[Dict], Dict[str, Any], List[IndexingResult]]:
        """
        Chunk multiple documents in parallel.
        
        Args:
            documents: List of documents with 'id', 'content', etc.
            chunking_service: Service with chunk_text method
            chunk_size: Size of chunks
            chunk_overlap: Overlap between chunks
            method: Chunking method
            
        Returns:
            Tuple of (all_chunks, doc_chunk_mapping, failed_results)
        """
        # Create processing function
        def process_document(doc: Dict[str, Any]) -> Tuple[str, List[Dict], Optional[IndexingResult]]:
            doc_id = doc.get('id', 'unknown')
            
            try:
                # Chunk the document
                chunks = chunking_service.chunk_text(
                    doc['content'],
                    chunk_size,
                    chunk_overlap,
                    method
                )
                
                # Add document metadata to chunks
                for i, chunk in enumerate(chunks):
                    chunk['doc_id'] = doc_id
                    chunk['doc_title'] = doc.get('title', 'Untitled')
                    chunk['chunk_index'] = i
                
                return doc_id, chunks, None
                
            except Exception as e:
                logger.error(f"Failed to chunk document {doc_id}: {e}")
                failed_result = IndexingResult(
                    doc_id=doc_id,
                    chunks_created=0,
                    time_taken=0,
                    success=False,
                    error=str(e)
                )
                return doc_id, [], failed_result
        
        # Process documents in parallel
        results = await self.batch_processor.process_documents_parallel(
            documents,
            process_document,
            "Chunking documents"
        )
        
        # Compile results
        all_chunks = []
        doc_chunk_info = {}
        failed_results = []
        
        for result in results:
            if result:
                doc_id, chunks, failed = result
                if failed:
                    failed_results.append(failed)
                else:
                    doc_chunk_info[doc_id] = {
                        'start_idx': len(all_chunks),
                        'num_chunks': len(chunks)
                    }
                    all_chunks.extend(chunks)
        
        logger.info(f"Chunking completed: {len(all_chunks)} total chunks from {len(documents)} documents")
        
        return all_chunks, doc_chunk_info, failed_results


# Convenience functions

def create_batch_processor(
    num_workers: Optional[int] = None,
    batch_size: int = 32,
    show_progress: bool = True,
    **kwargs
) -> BatchProcessor:
    """Create a batch processor with common settings."""
    config = ProcessingConfig(
        num_workers=num_workers,
        batch_size=batch_size,
        show_progress=show_progress,
        **kwargs
    )
    return BatchProcessor(config)


def create_embedding_processor(**kwargs) -> EmbeddingBatchProcessor:
    """Create an embedding batch processor."""
    config = ProcessingConfig(**kwargs)
    return EmbeddingBatchProcessor(config)


def create_chunking_processor(**kwargs) -> ChunkingBatchProcessor:
    """Create a chunking batch processor."""
    config = ProcessingConfig(**kwargs)
    return ChunkingBatchProcessor(config)


# Utility for CPU-bound operations
def run_cpu_bound_parallel(
    func: Callable,
    items: List[Any],
    num_workers: Optional[int] = None,
    desc: str = "Processing"
) -> List[Any]:
    """
    Simple parallel execution for CPU-bound operations.
    
    Args:
        func: Function to apply to each item
        items: List of items to process
        num_workers: Number of workers (None = auto)
        desc: Description for logging
        
    Returns:
        List of results maintaining input order
    """
    if not items:
        return []
    
    num_workers = num_workers or min(mp.cpu_count(), len(items))
    
    logger.info(f"{desc}: Processing {len(items)} items with {num_workers} workers")
    
    start_time = time.time()
    
    with Pool(num_workers) as pool:
        results = pool.map(func, items)
    
    elapsed = time.time() - start_time
    rate = len(items) / elapsed if elapsed > 0 else 0
    
    logger.info(f"{desc} completed in {elapsed:.2f}s ({rate:.1f} items/s)")
    
    return results