#!/usr/bin/env python3
"""
Test script for the chunking template system.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tldw_Server_API.app.core.Chunking.templates import (
    TemplateManager, ChunkingTemplate, TemplateStage
)

def test_builtin_templates():
    """Test built-in templates."""
    print("\n=== Testing Built-in Templates ===")
    
    manager = TemplateManager()
    
    # Test academic paper template
    text = """
    # Introduction
    
    Machine learning has revolutionized many fields. This paper explores new approaches.
    We present a novel algorithm for classification tasks.
    
    # Methods
    
    Our approach uses deep neural networks. We employ convolutional layers.
    The architecture consists of multiple stages.
    
    # Results
    
    We achieved 95% accuracy on the test set. The model generalizes well.
    Performance exceeded baseline methods.
    
    # Conclusion
    
    This work demonstrates the effectiveness of our approach.
    Future work will explore additional architectures.
    """
    
    print("\n--- Academic Paper Template ---")
    chunks = manager.process(text, "academic_paper")
    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i} ({len(chunk)} chars):")
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
    
    # Test code documentation template
    code_doc = """
    # API Reference
    
    ## Installation
    
    ```bash
    pip install mypackage
    ```
    
    ## Quick Start
    
    ```python
    import mypackage
    
    # Initialize the client
    client = mypackage.Client()
    
    # Make a request
    response = client.get_data()
    ```
    
    ## Configuration
    
    You can configure the client with various options:
    - `timeout`: Request timeout in seconds
    - `retries`: Number of retry attempts
    - `api_key`: Your API key
    
    ## Advanced Usage
    
    For advanced use cases, you can customize the behavior.
    """
    
    print("\n--- Code Documentation Template ---")
    chunks = manager.process(code_doc, "code_documentation")
    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i} ({len(chunk)} chars):")
        print(chunk[:300] + "..." if len(chunk) > 300 else chunk)

def test_custom_template():
    """Test creating and using a custom template."""
    print("\n=== Testing Custom Template ===")
    
    manager = TemplateManager()
    
    # Create a custom template for FAQ processing
    custom_template = ChunkingTemplate(
        name="faq_processor",
        description="Process FAQ documents",
        base_method="sentences",
        stages=[
            TemplateStage("preprocess", [
                {"type": "normalize_whitespace", "params": {"max_line_breaks": 1}},
                {"type": "extract_sections", "params": {"pattern": r"^Q:\s*(.+)$"}}
            ]),
            TemplateStage("chunk", [
                {"method": "sentences", "max_size": 3, "overlap": 0}
            ]),
            TemplateStage("postprocess", [
                {"type": "filter_empty", "params": {"min_length": 20}},
                {"type": "add_metadata", "params": {
                    "prefix": "[FAQ {index}/{total}] ",
                    "suffix": ""
                }}
            ])
        ]
    )
    
    # Register the template
    manager.register_template(custom_template)
    
    # Test FAQ text
    faq_text = """
    Q: What is machine learning?
    Machine learning is a type of artificial intelligence that enables computers to learn from data.
    It uses algorithms to identify patterns and make decisions.
    
    Q: How does deep learning differ from machine learning?
    Deep learning is a subset of machine learning that uses neural networks with multiple layers.
    It can automatically learn features from raw data.
    
    Q: What are the applications of AI?
    AI has many applications including image recognition, natural language processing, and robotics.
    It is used in healthcare, finance, transportation, and many other fields.
    """
    
    chunks = manager.process(faq_text, "faq_processor")
    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(chunk)

def test_template_operations():
    """Test individual template operations."""
    print("\n=== Testing Template Operations ===")
    
    from tldw_Server_API.app.core.Chunking.templates import TemplateProcessor
    
    processor = TemplateProcessor()
    
    # Test normalize whitespace
    text = "This   has     multiple    spaces.\n\n\n\nAnd many line breaks."
    normalized = processor._normalize_whitespace(text, {"max_line_breaks": 2})
    print(f"Normalized: '{normalized}'")
    
    # Test filter empty
    chunks = ["", "  ", "Valid chunk", "Another valid", "   \n  "]
    filtered = processor._filter_empty(chunks, {"min_length": 5})
    print(f"Filtered chunks: {filtered}")
    
    # Test merge small
    small_chunks = ["Small", "Another", "This is a longer chunk", "Tiny", "End"]
    merged = processor._merge_small(small_chunks, {"min_size": 20, "separator": " "})
    print(f"Merged chunks: {merged}")

def main():
    """Run all template system tests."""
    print("Testing Chunking Template System")
    print("=" * 50)
    
    test_builtin_templates()
    test_custom_template()
    test_template_operations()
    
    print("\n" + "=" * 50)
    print("All Template Tests Complete!")

if __name__ == "__main__":
    main()