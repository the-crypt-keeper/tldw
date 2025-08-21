#!/usr/bin/env python3
"""
Test script to verify RAG module integration with new chunking API.
"""

import sys
import os
import asyncio
from typing import List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test imports to verify they work
print("Testing imports...")

try:
    from tldw_Server_API.app.core.Chunking import (
        Chunker, ChunkerConfig, EnhancedChunk, ChunkType,
        improved_chunking_process, ChunkingError
    )
    print("✓ Chunking module imports successful")
except ImportError as e:
    print(f"✗ Failed to import from Chunking module: {e}")
    sys.exit(1)

try:
    from tldw_Server_API.app.core.RAG.rag_service.types import Document
    from tldw_Server_API.app.core.RAG.rag_service.document_processing_integration import (
        DocumentProcessor, ProcessingConfig
    )
    print("✓ RAG document processing imports successful")
except ImportError as e:
    print(f"✗ Failed to import RAG document processing: {e}")
    sys.exit(1)

try:
    from tldw_Server_API.app.core.RAG.rag_service.enhanced_chunking_integration import (
        enhanced_chunk_documents
    )
    print("✓ RAG enhanced chunking imports successful")
except ImportError as e:
    print(f"✗ Failed to import RAG enhanced chunking: {e}")
    sys.exit(1)


def test_enhanced_chunk_creation():
    """Test creating EnhancedChunk objects."""
    print("\n=== Testing EnhancedChunk Creation ===")
    
    chunk = EnhancedChunk(
        id="test_chunk_1",
        content="This is a test chunk with some content.",
        chunk_type=ChunkType.TEXT,
        start_char=0,
        end_char=40,
        chunk_index=0,
        metadata={"source": "test"},
        parent_id="doc_1"
    )
    
    print(f"Created chunk: {chunk.id}")
    print(f"Content: {chunk.content[:50]}...")
    print(f"Type: {chunk.chunk_type.value}")
    
    # Test serialization
    chunk_dict = chunk.to_dict()
    print(f"Serialized: {list(chunk_dict.keys())}")
    
    # Test deserialization
    restored = EnhancedChunk.from_dict(chunk_dict)
    print(f"Restored chunk: {restored.id}")
    
    return chunk


def test_improved_chunking():
    """Test the backward-compatible improved_chunking_process function."""
    print("\n=== Testing improved_chunking_process ===")
    
    text = """
    # Introduction to Machine Learning
    
    Machine learning is a subset of artificial intelligence that focuses on
    building systems that learn from data. Instead of being explicitly programmed,
    these systems identify patterns and make decisions.
    
    ## Key Concepts
    
    - Supervised Learning: Learning from labeled data
    - Unsupervised Learning: Finding patterns in unlabeled data
    - Reinforcement Learning: Learning through trial and error
    
    ```python
    # Example code
    import numpy as np
    from sklearn import datasets
    
    data = datasets.load_iris()
    print(data.feature_names)
    ```
    
    This is a fundamental technology driving modern AI applications.
    """
    
    options = {
        "method": "sentences",
        "max_size": 3,
        "overlap": 1
    }
    
    try:
        chunks = improved_chunking_process(text, options)
        print(f"Created {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks[:3]):  # Show first 3
            print(f"\nChunk {i}:")
            print(f"  Text: {chunk['text'][:80]}...")
            print(f"  Metadata keys: {list(chunk.get('metadata', {}).keys())}")
        
        return chunks
    except Exception as e:
        print(f"Error in chunking: {e}")
        return []


async def test_document_processor():
    """Test the DocumentProcessor with new chunking."""
    print("\n=== Testing DocumentProcessor ===")
    
    content = """
    # Technical Documentation
    
    This document explains the system architecture.
    
    ## Components
    
    | Component | Description | Status |
    |-----------|-------------|--------|
    | API       | REST endpoints | Active |
    | Database  | PostgreSQL | Active |
    | Cache     | Redis | Planned |
    
    The system uses microservices architecture.
    """
    
    config = ProcessingConfig(
        target_chunk_size=100,
        detect_structure=True,
        optimize_boundaries=True
    )
    
    processor = DocumentProcessor(config)
    
    try:
        chunks = await processor.process_document(
            content=content,
            source="test_doc",
            metadata={"type": "technical"}
        )
        
        print(f"Processed into {len(chunks)} enhanced chunks")
        
        for chunk in chunks[:3]:  # Show first 3
            print(f"\nChunk {chunk.id}:")
            print(f"  Type: {chunk.chunk_type.value if hasattr(chunk.chunk_type, 'value') else chunk.chunk_type}")
            print(f"  Content: {chunk.content[:60]}...")
            print(f"  Position: {chunk.start_char}-{chunk.end_char}")
        
        return chunks
    except Exception as e:
        print(f"Error processing document: {e}")
        import traceback
        traceback.print_exc()
        return []


def test_new_chunker_api():
    """Test using the new Chunker API directly."""
    print("\n=== Testing New Chunker API ===")
    
    text = """
    Artificial intelligence is transforming industries worldwide.
    Machine learning algorithms can now process vast amounts of data.
    Deep learning has enabled breakthroughs in computer vision.
    Natural language processing allows computers to understand human language.
    """
    
    # Create chunker with configuration
    config = ChunkerConfig(
        default_method="sentences",
        default_max_size=2,
        default_overlap=1
    )
    
    chunker = Chunker(config)
    
    # Test basic chunking
    chunks = chunker.chunk_text(text, method="sentences", max_size=2)
    print(f"Created {len(chunks)} chunks using new API")
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {chunk[:60]}...")
    
    # Test with metadata
    chunk_results = chunker.chunk_text_with_metadata(
        text=text,
        method="sentences",
        max_size=2,
        overlap=1
    )
    
    print(f"\nWith metadata: {len(chunk_results)} chunks")
    for i, result in enumerate(chunk_results[:2]):
        print(f"Chunk {i}: words={result.metadata.word_count}, "
              f"chars={result.metadata.char_count}")
    
    return chunks


class MockRAGContext:
    """Mock context for testing enhanced_chunk_documents."""
    def __init__(self, documents, config=None):
        self.documents = documents
        self.config = config or {}
        self.metadata = {}  # Add metadata attribute


async def test_enhanced_chunking_integration():
    """Test the enhanced_chunk_documents function."""
    print("\n=== Testing Enhanced Chunking Integration ===")
    
    # Create mock documents
    doc1 = Document(
        id="doc1",
        source="test",
        content="""
        # Research Paper
        
        ## Abstract
        This paper presents a new approach to data processing.
        
        ## Methods
        We used machine learning algorithms to analyze patterns.
        
        ```python
        def process_data(data):
            return data.transform()
        ```
        
        ## Results
        The results show significant improvements.
        """,
        metadata={"type": "research"}
    )
    
    # Create mock context
    context = MockRAGContext(
        documents=[doc1],
        config={
            "chunk_size": 100,
            "chunk_overlap": 20,
            "preserve_code_blocks": True,
            "structure_aware": True
        }
    )
    
    try:
        # Process documents
        result = await enhanced_chunk_documents(
            context,
            structure_aware=True,
            preserve_code_blocks=True,
            chunk_size=100,
            overlap=20
        )
        
        print(f"Enhanced chunking completed")
        
        # Check if chunks were added to documents
        if hasattr(result, 'documents') and result.documents:
            print(f"Documents processed: {len(result.documents)}")
            for doc in result.documents:
                if hasattr(doc, 'chunks'):
                    print(f"  Document {doc.id}: {len(doc.chunks)} chunks")
        
        return result
    except Exception as e:
        print(f"Error in enhanced chunking: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    """Run all integration tests."""
    print("=" * 60)
    print("RAG-Chunking Integration Tests")
    print("=" * 60)
    
    # Test 1: EnhancedChunk creation
    test_enhanced_chunk_creation()
    
    # Test 2: Backward compatible function
    test_improved_chunking()
    
    # Test 3: New Chunker API
    test_new_chunker_api()
    
    # Test 4: Document processor
    await test_document_processor()
    
    # Test 5: Enhanced chunking integration
    await test_enhanced_chunking_integration()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Handle both sync and async execution
    try:
        asyncio.run(main())
    except RuntimeError:
        # If already in async context
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())