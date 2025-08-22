# async_chunker.py
"""
Asynchronous chunking implementation for improved I/O performance.
Provides async/await interfaces for chunking operations.
"""

import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from pathlib import Path
import aiofiles
from loguru import logger
from concurrent.futures import ThreadPoolExecutor

from .base import ChunkerConfig, ChunkResult
from .chunker import Chunker
from .templates import TemplateManager
from .utils.metrics import get_metrics, MetricsContext


class AsyncChunker:
    """
    Asynchronous wrapper for the chunking system.
    Provides async interfaces for I/O-bound operations.
    """
    
    def __init__(self, config: Optional[ChunkerConfig] = None):
        """
        Initialize async chunker.
        
        Args:
            config: Chunker configuration
        """
        self.config = config or ChunkerConfig()
        self._chunker = Chunker(config)
        self._metrics = get_metrics()
        
        # Thread pool for CPU-bound operations
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.max_workers if hasattr(self.config, 'max_workers') else 4
        )
        
        # Semaphore for rate limiting
        self._semaphore = asyncio.Semaphore(
            self.config.max_concurrent if hasattr(self.config, 'max_concurrent') else 10
        )
        
        logger.info("AsyncChunker initialized")
    
    async def chunk_text(self,
                        text: str,
                        method: Optional[str] = None,
                        max_size: Optional[int] = None,
                        overlap: Optional[int] = None,
                        **options) -> List[str]:
        """
        Asynchronously chunk text.
        
        Args:
            text: Text to chunk
            method: Chunking method
            max_size: Maximum chunk size
            overlap: Overlap between chunks
            **options: Additional options
            
        Returns:
            List of text chunks
        """
        async with self._semaphore:
            # Run CPU-bound chunking in thread pool
            loop = asyncio.get_event_loop()
            chunks = await loop.run_in_executor(
                self._executor,
                self._chunker.chunk_text,
                text,
                method,
                max_size,
                overlap,
                options
            )
            return chunks
    
    async def chunk_file(self,
                        file_path: Union[str, Path],
                        method: Optional[str] = None,
                        max_size: Optional[int] = None,
                        overlap: Optional[int] = None,
                        encoding: str = 'utf-8',
                        **options) -> List[str]:
        """
        Asynchronously read and chunk a file.
        
        Args:
            file_path: Path to file
            method: Chunking method
            max_size: Maximum chunk size
            overlap: Overlap between chunks
            encoding: File encoding
            **options: Additional options
            
        Returns:
            List of text chunks
        """
        file_path = Path(file_path)
        
        # Read file asynchronously
        async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
            text = await f.read()
        
        # Chunk the text
        return await self.chunk_text(text, method, max_size, overlap, **options)
    
    async def chunk_files(self,
                         file_paths: List[Union[str, Path]],
                         method: Optional[str] = None,
                         max_size: Optional[int] = None,
                         overlap: Optional[int] = None,
                         encoding: str = 'utf-8',
                         **options) -> Dict[str, List[str]]:
        """
        Asynchronously chunk multiple files.
        
        Args:
            file_paths: List of file paths
            method: Chunking method
            max_size: Maximum chunk size
            overlap: Overlap between chunks
            encoding: File encoding
            **options: Additional options
            
        Returns:
            Dictionary mapping file paths to their chunks
        """
        tasks = []
        for file_path in file_paths:
            task = self.chunk_file(
                file_path, method, max_size, overlap, encoding, **options
            )
            tasks.append(task)
        
        # Process files concurrently
        results = await asyncio.gather(*tasks)
        
        # Create result dictionary
        return {
            str(file_path): chunks 
            for file_path, chunks in zip(file_paths, results)
        }
    
    async def chunk_stream(self,
                          text_stream: AsyncGenerator[str, None],
                          method: Optional[str] = None,
                          max_size: Optional[int] = None,
                          overlap: Optional[int] = None,
                          buffer_size: int = 10000,
                          **options) -> AsyncGenerator[str, None]:
        """
        Stream chunks from an async text generator.
        
        Args:
            text_stream: Async generator yielding text
            method: Chunking method
            max_size: Maximum chunk size
            overlap: Overlap between chunks
            buffer_size: Size of text buffer
            **options: Additional options
            
        Yields:
            Text chunks as they are generated
        """
        buffer = ""
        overlap_buffer = ""
        overlap_size = overlap if overlap is not None else 0
        
        async for text_piece in text_stream:
            buffer += text_piece
            
            # Process buffer when it's large enough
            if len(buffer) >= buffer_size:
                # Chunk the buffer
                chunks = await self.chunk_text(
                    overlap_buffer + buffer,
                    method, max_size, overlap, **options
                )
                
                # Yield all but the last chunk
                for chunk in chunks[:-1]:
                    yield chunk
                
                # Keep last chunk as overlap
                if chunks and overlap_size > 0:
                    overlap_buffer = chunks[-1][-overlap_size:] if len(chunks[-1]) > overlap_size else chunks[-1]
                else:
                    overlap_buffer = ""
                
                buffer = ""
        
        # Process remaining buffer
        if buffer or overlap_buffer:
            chunks = await self.chunk_text(
                overlap_buffer + buffer,
                method, max_size, overlap, **options
            )
            for chunk in chunks:
                yield chunk
    
    async def chunk_with_metadata(self,
                                 text: str,
                                 method: Optional[str] = None,
                                 max_size: Optional[int] = None,
                                 overlap: Optional[int] = None,
                                 **options) -> List[ChunkResult]:
        """
        Asynchronously chunk text and return with metadata.
        
        Args:
            text: Text to chunk
            method: Chunking method
            max_size: Maximum chunk size
            overlap: Overlap between chunks
            **options: Additional options
            
        Returns:
            List of ChunkResult objects
        """
        async with self._semaphore:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self._executor,
                self._chunker.chunk_text_with_metadata,
                text,
                method,
                max_size,
                overlap,
                options
            )
            return results
    
    async def chunk_url(self,
                       url: str,
                       method: Optional[str] = None,
                       max_size: Optional[int] = None,
                       overlap: Optional[int] = None,
                       **options) -> List[str]:
        """
        Asynchronously fetch and chunk content from URL.
        
        Args:
            url: URL to fetch content from
            method: Chunking method
            max_size: Maximum chunk size
            overlap: Overlap between chunks
            **options: Additional options
            
        Returns:
            List of text chunks
        """
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                text = await response.text()
        
        return await self.chunk_text(text, method, max_size, overlap, **options)
    
    async def process_with_template(self,
                                   text: str,
                                   template_name: str,
                                   **options) -> List[str]:
        """
        Process text using a template asynchronously.
        
        Args:
            text: Text to process
            template_name: Name of template to use
            **options: Additional options
            
        Returns:
            List of processed chunks
        """
        # Get template manager
        if not hasattr(self, '_template_manager'):
            self._template_manager = TemplateManager()
        
        # Process in executor
        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(
            self._executor,
            self._template_manager.process,
            text,
            template_name,
            options
        )
        return chunks
    
    async def close(self):
        """Clean up resources."""
        self._executor.shutdown(wait=True)
        logger.info("AsyncChunker closed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


class AsyncBatchProcessor:
    """
    Process multiple chunking requests in batches for efficiency.
    """
    
    def __init__(self,
                batch_size: int = 10,
                max_concurrent: int = 5):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Number of items to process per batch
            max_concurrent: Maximum concurrent batches
        """
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self._chunker = AsyncChunker()
        self._queue = asyncio.Queue()
        self._results = {}
        self._processing = False
        
        logger.info(f"AsyncBatchProcessor initialized with batch_size={batch_size}")
    
    async def add_request(self,
                         request_id: str,
                         text: str,
                         method: str = 'words',
                         **options):
        """
        Add a chunking request to the queue.
        
        Args:
            request_id: Unique request identifier
            text: Text to chunk
            method: Chunking method
            **options: Additional options
        """
        await self._queue.put({
            'id': request_id,
            'text': text,
            'method': method,
            'options': options
        })
    
    async def process_batch(self):
        """Process a batch of requests."""
        batch = []
        
        # Collect batch
        for _ in range(self.batch_size):
            if self._queue.empty():
                break
            request = await self._queue.get()
            batch.append(request)
        
        if not batch:
            return
        
        # Process batch concurrently
        tasks = []
        for request in batch:
            task = self._chunker.chunk_text(
                request['text'],
                request['method'],
                **request['options']
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Store results
        for request, result in zip(batch, results):
            if isinstance(result, Exception):
                logger.error(f"Error processing request {request['id']}: {result}")
                self._results[request['id']] = {'error': str(result)}
            else:
                self._results[request['id']] = {'chunks': result}
    
    async def start_processing(self):
        """Start processing requests from the queue."""
        self._processing = True
        
        while self._processing:
            if self._queue.empty():
                await asyncio.sleep(0.1)
                continue
            
            # Process multiple batches concurrently
            batch_tasks = []
            for _ in range(self.max_concurrent):
                if not self._queue.empty():
                    batch_tasks.append(self.process_batch())
            
            if batch_tasks:
                await asyncio.gather(*batch_tasks)
    
    async def stop_processing(self):
        """Stop processing requests."""
        self._processing = False
        
        # Process remaining requests
        while not self._queue.empty():
            await self.process_batch()
        
        await self._chunker.close()
    
    def get_result(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get result for a request.
        
        Args:
            request_id: Request identifier
            
        Returns:
            Result dictionary or None if not ready
        """
        return self._results.get(request_id)
    
    async def wait_for_result(self,
                             request_id: str,
                             timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        """
        Wait for a result to be ready.
        
        Args:
            request_id: Request identifier
            timeout: Maximum wait time in seconds
            
        Returns:
            Result dictionary or None if timeout
        """
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            result = self.get_result(request_id)
            if result is not None:
                return result
            await asyncio.sleep(0.1)
        
        return None


async def chunk_parallel(texts: List[str],
                        method: str = 'words',
                        max_size: int = 400,
                        overlap: int = 50,
                        max_concurrent: int = 10,
                        **options) -> List[List[str]]:
    """
    Chunk multiple texts in parallel.
    
    Args:
        texts: List of texts to chunk
        method: Chunking method
        max_size: Maximum chunk size
        overlap: Overlap between chunks
        max_concurrent: Maximum concurrent operations
        **options: Additional options
        
    Returns:
        List of chunk lists
    """
    async with AsyncChunker() as chunker:
        # Limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def chunk_with_limit(text):
            async with semaphore:
                return await chunker.chunk_text(
                    text, method, max_size, overlap, **options
                )
        
        tasks = [chunk_with_limit(text) for text in texts]
        return await asyncio.gather(*tasks)


async def stream_chunks_from_file(file_path: Union[str, Path],
                                 method: str = 'words',
                                 max_size: int = 400,
                                 overlap: int = 50,
                                 chunk_size: int = 1024 * 1024,  # 1MB
                                 **options) -> AsyncGenerator[str, None]:
    """
    Stream chunks from a large file without loading it entirely into memory.
    
    Args:
        file_path: Path to file
        method: Chunking method
        max_size: Maximum chunk size
        overlap: Overlap between chunks
        chunk_size: Size of file chunks to read
        **options: Additional options
        
    Yields:
        Text chunks
    """
    async def file_reader():
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            while True:
                chunk = await f.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    
    async with AsyncChunker() as chunker:
        async for chunk in chunker.chunk_stream(
            file_reader(), method, max_size, overlap, **options
        ):
            yield chunk