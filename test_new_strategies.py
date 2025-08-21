#!/usr/bin/env python3
"""
Test script for the new semantic, JSON, and XML chunking strategies.
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tldw_Server_API.app.core.Chunking import Chunker, ChunkerConfig

def test_semantic_chunking():
    """Test semantic chunking strategy."""
    print("\n=== Testing Semantic Chunking ===")
    
    text = """
    Machine learning is a subset of artificial intelligence. It focuses on building systems 
    that learn from data. Neural networks are a key component of machine learning.
    
    The weather today is sunny and warm. Birds are singing in the trees. 
    It's a perfect day for a picnic in the park.
    
    Deep learning is a subset of machine learning. It uses multiple layers of neural networks.
    These systems can recognize patterns in complex data.
    """
    
    try:
        chunker = Chunker()
        chunks = chunker.chunk_text(
            text=text,
            method='semantic',
            max_size=50,  # words
            overlap=1  # sentences
        )
        
        print(f"Created {len(chunks)} semantic chunks:")
        for i, chunk in enumerate(chunks, 1):
            print(f"\nChunk {i}:")
            print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
            
    except Exception as e:
        print(f"Semantic chunking not available: {e}")
        print("This is expected if scikit-learn is not installed")

def test_json_chunking():
    """Test JSON chunking strategy."""
    print("\n=== Testing JSON Chunking ===")
    
    # Test with JSON array
    json_array = json.dumps([
        {"id": 1, "name": "Alice", "age": 30},
        {"id": 2, "name": "Bob", "age": 25},
        {"id": 3, "name": "Charlie", "age": 35},
        {"id": 4, "name": "Diana", "age": 28},
        {"id": 5, "name": "Eve", "age": 32}
    ])
    
    chunker = Chunker()
    chunks = chunker.chunk_text(
        text=json_array,
        method='json',
        max_size=2,  # items per chunk
        overlap=1
    )
    
    print(f"Created {len(chunks)} JSON array chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(chunk)
    
    # Test with JSON object
    json_obj = json.dumps({
        "metadata": {"version": "1.0", "created": "2024-01-01"},
        "data": {
            "user1": {"name": "Alice", "score": 100},
            "user2": {"name": "Bob", "score": 85},
            "user3": {"name": "Charlie", "score": 92},
            "user4": {"name": "Diana", "score": 88}
        }
    })
    
    chunks = chunker.chunk_text(
        text=json_obj,
        method='json',
        max_size=2,  # keys per chunk
        overlap=0,
        chunkable_key='data'
    )
    
    print(f"\n\nCreated {len(chunks)} JSON object chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(chunk)

def test_xml_chunking():
    """Test XML chunking strategy."""
    print("\n=== Testing XML Chunking ===")
    
    xml_text = """<?xml version="1.0"?>
    <library>
        <book id="1">
            <title>Python Programming</title>
            <author>John Doe</author>
            <year>2023</year>
            <description>A comprehensive guide to Python programming covering basics to advanced topics.</description>
        </book>
        <book id="2">
            <title>Machine Learning Basics</title>
            <author>Jane Smith</author>
            <year>2024</year>
            <description>Introduction to machine learning concepts and algorithms.</description>
        </book>
        <book id="3">
            <title>Web Development</title>
            <author>Bob Johnson</author>
            <year>2023</year>
            <description>Modern web development with HTML, CSS, and JavaScript.</description>
        </book>
    </library>"""
    
    chunker = Chunker()
    chunks = chunker.chunk_text(
        text=xml_text,
        method='xml',
        max_size=20,  # words per chunk
        overlap=2,  # XML elements
        output_format='text'
    )
    
    print(f"Created {len(chunks)} XML chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(chunk)

def main():
    """Run all tests."""
    print("Testing new chunking strategies...")
    
    test_semantic_chunking()
    test_json_chunking()
    test_xml_chunking()
    
    print("\n=== All Tests Complete ===")

if __name__ == "__main__":
    main()