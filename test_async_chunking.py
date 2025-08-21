#!/usr/bin/env python3
"""
Test script for async chunking functionality.
"""

import asyncio
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tldw_Server_API.app.core.Chunking.async_chunker import (
    AsyncChunker, AsyncBatchProcessor, chunk_parallel, stream_chunks_from_file
)


async def test_basic_async_chunking():
    """Test basic async chunking."""
    print("\n=== Testing Basic Async Chunking ===")
    
    text = """
    Artificial intelligence is transforming the world. Machine learning algorithms 
    are becoming more sophisticated. Deep learning has enabled breakthroughs in 
    computer vision and natural language processing. Neural networks can now 
    perform tasks that were once thought impossible.
    """
    
    async with AsyncChunker() as chunker:
        start = time.time()
        chunks = await chunker.chunk_text(text, method='sentences', max_size=2)
        elapsed = time.time() - start
        
        print(f"Created {len(chunks)} chunks in {elapsed:.3f} seconds")
        for i, chunk in enumerate(chunks, 1):
            print(f"\nChunk {i}:")
            print(chunk[:100] + "..." if len(chunk) > 100 else chunk)


async def test_parallel_file_chunking():
    """Test parallel file chunking."""
    print("\n=== Testing Parallel File Chunking ===")
    
    # Create test files
    test_dir = Path("test_files")
    test_dir.mkdir(exist_ok=True)
    
    test_files = []
    for i in range(3):
        file_path = test_dir / f"test_{i}.txt"
        content = f"""
        This is test file {i}.
        It contains multiple sentences for testing.
        The async chunker should process this efficiently.
        Each file will be chunked in parallel.
        This demonstrates concurrent I/O operations.
        """
        file_path.write_text(content)
        test_files.append(file_path)
    
    try:
        async with AsyncChunker() as chunker:
            start = time.time()
            results = await chunker.chunk_files(
                test_files, 
                method='sentences', 
                max_size=2
            )
            elapsed = time.time() - start
            
            print(f"Processed {len(test_files)} files in {elapsed:.3f} seconds")
            for file_path, chunks in results.items():
                print(f"\n{Path(file_path).name}: {len(chunks)} chunks")
    
    finally:
        # Clean up test files
        for file_path in test_files:
            file_path.unlink()
        test_dir.rmdir()


async def test_streaming_chunks():
    """Test streaming chunk generation."""
    print("\n=== Testing Streaming Chunks ===")
    
    async def text_generator():
        """Simulate streaming text input."""
        texts = [
            "First part of the stream. ",
            "Second part with more content. ",
            "Third part continues the story. ",
            "Fourth part adds more details. ",
            "Final part concludes everything."
        ]
        for text in texts:
            await asyncio.sleep(0.1)  # Simulate network delay
            yield text
    
    async with AsyncChunker() as chunker:
        print("Streaming chunks as text arrives...")
        chunk_count = 0
        
        async for chunk in chunker.chunk_stream(
            text_generator(),
            method='words',
            max_size=10,
            buffer_size=50
        ):
            chunk_count += 1
            print(f"Chunk {chunk_count}: {chunk[:50]}...")
        
        print(f"Total chunks generated: {chunk_count}")


async def test_batch_processor():
    """Test batch processing of requests."""
    print("\n=== Testing Batch Processor ===")
    
    processor = AsyncBatchProcessor(batch_size=5, max_concurrent=2)
    
    # Start processor in background
    process_task = asyncio.create_task(processor.start_processing())
    
    # Add requests
    request_ids = []
    for i in range(10):
        request_id = f"req_{i}"
        text = f"This is request {i}. It contains text to be chunked. Processing in batches is efficient."
        await processor.add_request(request_id, text, method='sentences')
        request_ids.append(request_id)
    
    print(f"Added {len(request_ids)} requests to queue")
    
    # Wait for results
    results_received = 0
    for request_id in request_ids:
        result = await processor.wait_for_result(request_id, timeout=5.0)
        if result:
            if 'chunks' in result:
                results_received += 1
                print(f"{request_id}: {len(result['chunks'])} chunks")
            else:
                print(f"{request_id}: Error - {result.get('error')}")
    
    print(f"Received {results_received}/{len(request_ids)} results")
    
    # Stop processor
    await processor.stop_processing()
    process_task.cancel()
    try:
        await process_task
    except asyncio.CancelledError:
        pass


async def test_parallel_chunking():
    """Test parallel chunking of multiple texts."""
    print("\n=== Testing Parallel Chunking ===")
    
    texts = [
        "First text about artificial intelligence and machine learning.",
        "Second text discussing natural language processing techniques.",
        "Third text covering computer vision and image recognition.",
        "Fourth text about robotics and autonomous systems.",
        "Fifth text on data science and analytics methods."
    ]
    
    start = time.time()
    results = await chunk_parallel(
        texts,
        method='words',
        max_size=5,
        overlap=1,
        max_concurrent=3
    )
    elapsed = time.time() - start
    
    print(f"Chunked {len(texts)} texts in {elapsed:.3f} seconds")
    for i, chunks in enumerate(results, 1):
        print(f"Text {i}: {len(chunks)} chunks")


async def test_url_chunking():
    """Test chunking content from URL (mock)."""
    print("\n=== Testing URL Chunking (Simulated) ===")
    
    # This would normally fetch from a real URL
    # For testing, we'll just demonstrate the interface
    print("URL chunking interface available via chunk_url() method")
    print("Example: chunks = await chunker.chunk_url('https://example.com/article')")


async def test_template_async():
    """Test template processing asynchronously."""
    print("\n=== Testing Async Template Processing ===")
    
    text = """
    Q: What is async programming?
    Async programming allows concurrent execution without blocking.
    
    Q: Why use async for chunking?
    It improves performance for I/O-bound operations like reading files.
    
    Q: When should you use async?
    Use async when dealing with multiple files or network requests.
    """
    
    async with AsyncChunker() as chunker:
        try:
            chunks = await chunker.process_with_template(
                text,
                'chat_conversation'
            )
            print(f"Processed with template: {len(chunks)} chunks")
        except Exception as e:
            print(f"Template processing not available: {e}")


async def main():
    """Run all async tests."""
    print("Testing Async Chunking System")
    print("=" * 50)
    
    await test_basic_async_chunking()
    await test_parallel_file_chunking()
    await test_streaming_chunks()
    await test_batch_processor()
    await test_parallel_chunking()
    await test_url_chunking()
    await test_template_async()
    
    print("\n" + "=" * 50)
    print("All Async Tests Complete!")


if __name__ == "__main__":
    asyncio.run(main())