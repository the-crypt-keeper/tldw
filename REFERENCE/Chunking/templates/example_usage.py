#!/usr/bin/env python3
"""
Example usage of the chunking template system.
"""

from tldw_chatbook.Chunking import (
    Chunker, 
    ChunkingTemplateManager,
    ChunkingTemplate,
    ChunkingStage,
    ChunkingOperation,
    improved_chunking_process
)


def example_basic_template_usage():
    """Example of using a built-in template."""
    print("=== Basic Template Usage ===")
    
    # Sample text
    text = """
    Natural language processing (NLP) is a subfield of linguistics, computer science, 
    and artificial intelligence concerned with the interactions between computers and 
    human language. In particular, it focuses on how to program computers to process 
    and analyze large amounts of natural language data. The goal is a computer capable 
    of understanding the contents of documents, including the contextual nuances of 
    the language within them.
    """ * 5
    
    # Use a built-in template
    chunker = Chunker(template="sentences")
    chunks = chunker.chunk_text(text)
    
    print(f"Number of chunks: {len(chunks)}")
    print(f"First chunk: {chunks[0][:100]}...")
    print()


def example_custom_template():
    """Example of creating and using a custom template."""
    print("=== Custom Template Example ===")
    
    # Create a custom template programmatically
    template = ChunkingTemplate(
        name="my_custom",
        description="Custom template for demonstration",
        base_method="words",
        pipeline=[
            ChunkingStage(
                stage="preprocess",
                operations=[
                    ChunkingOperation(
                        type="normalize_whitespace",
                        params={}
                    )
                ]
            ),
            ChunkingStage(
                stage="chunk",
                method="words",
                options={
                    "max_size": 50,
                    "overlap": 10
                }
            ),
            ChunkingStage(
                stage="postprocess",
                operations=[
                    ChunkingOperation(
                        type="add_context",
                        params={"context_size": 1}
                    )
                ]
            )
        ],
        metadata={
            "created_by": "example_script",
            "purpose": "demonstration"
        }
    )
    
    # Save the template
    manager = ChunkingTemplateManager()
    manager.save_template(template, user_template=True)
    
    # Use it
    chunker = Chunker(template="my_custom")
    text = " ".join([f"Word{i}" for i in range(200)])
    chunks = chunker.chunk_text(text)
    
    print(f"Created {len(chunks)} chunks with custom template")
    print(f"First chunk with context: {chunks[0][:100]}...")
    print()


def example_domain_specific_template():
    """Example using a domain-specific template."""
    print("=== Domain-Specific Template Example ===")
    
    # Academic paper example
    academic_text = """
    Abstract: This paper presents a comprehensive study of chunking strategies
    for natural language processing tasks. We evaluate multiple approaches
    and demonstrate their effectiveness on downstream applications.
    
    Introduction
    Text chunking is a fundamental preprocessing step in many NLP pipelines.
    The choice of chunking strategy can significantly impact the performance
    of downstream tasks such as information retrieval and text generation.
    
    Methods
    We implemented and compared five different chunking strategies:
    1. Fixed-size word chunks
    2. Sentence-based chunks
    3. Paragraph-based chunks  
    4. Semantic similarity-based chunks
    5. Hybrid approaches
    
    Results
    Our experiments show that semantic chunking provides the best balance
    between chunk coherence and computational efficiency. The hybrid approach
    showed promising results for specific document types.
    
    Conclusion
    The choice of chunking strategy should be tailored to the specific
    requirements of the downstream task and the nature of the input documents.
    """
    
    # Use academic paper template
    try:
        chunks = improved_chunking_process(
            academic_text,
            template="academic_paper"
        )
        
        print(f"Academic paper chunked into {len(chunks)} chunks")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\nChunk {i+1}:")
            print(f"Text: {chunk['text'][:100]}...")
            print(f"Metadata: {chunk['metadata'].get('template_metadata', {})}")
    except Exception as e:
        print(f"Error using academic template: {e}")
    print()


def example_custom_operation():
    """Example of creating a custom operation."""
    print("=== Custom Operation Example ===")
    
    # Create a template manager
    manager = ChunkingTemplateManager()
    
    # Define a custom operation that adds line numbers
    def add_line_numbers(text: str, chunks: list, options: dict) -> list:
        """Add line numbers to each chunk."""
        numbered_chunks = []
        for i, chunk in enumerate(chunks, 1):
            prefix = options.get("prefix", "Line")
            numbered_chunk = f"[{prefix} {i}] {chunk}"
            numbered_chunks.append(numbered_chunk)
        return numbered_chunks
    
    # Register the operation
    manager.register_operation("add_line_numbers", add_line_numbers)
    
    # Create a template using the custom operation
    template = ChunkingTemplate(
        name="numbered_chunks",
        base_method="sentences",
        pipeline=[
            ChunkingStage(
                stage="chunk",
                method="sentences",
                options={"max_size": 2}
            ),
            ChunkingStage(
                stage="postprocess",
                operations=[
                    ChunkingOperation(
                        type="add_line_numbers",
                        params={"prefix": "Chunk"}
                    )
                ]
            )
        ]
    )
    
    # Use it with the custom operation
    from tldw_chatbook.Chunking.chunking_templates import ChunkingPipeline
    pipeline = ChunkingPipeline(manager)
    chunker = Chunker()
    
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    results = pipeline.execute(text, template, chunker)
    
    print("Chunks with custom line numbers:")
    for result in results:
        print(f"  {result['text']}")
    print()


def example_template_inheritance():
    """Example of template inheritance."""
    print("=== Template Inheritance Example ===")
    
    manager = ChunkingTemplateManager()
    
    # Assume we have a base template (using words template as base)
    # Create a child template that inherits and overrides
    child_template = ChunkingTemplate(
        name="words_large",
        description="Words template with larger chunks",
        parent_template="words",
        pipeline=[
            ChunkingStage(
                stage="chunk",
                options={
                    "max_size": 1000,  # Override the default
                    "overlap": 200
                }
            )
        ]
    )
    
    # Save it
    manager.save_template(child_template, user_template=True)
    
    # Use it
    chunker = Chunker(template="words_large", template_manager=manager)
    text = " ".join([f"word{i}" for i in range(2000)])
    chunks = chunker.chunk_text(text)
    
    print(f"Parent template (words) default max_size: 400")
    print(f"Child template (words_large) chunks: {len(chunks)}")
    print("Successfully demonstrated template inheritance")
    print()


def list_available_templates():
    """List all available templates."""
    print("=== Available Templates ===")
    
    manager = ChunkingTemplateManager()
    templates = manager.get_available_templates()
    
    print("Built-in templates:")
    for template_name in sorted(templates):
        template = manager.load_template(template_name)
        if template:
            print(f"  - {template_name}: {template.description}")
    print()


if __name__ == "__main__":
    print("Chunking Template System Examples\n")
    
    # Run examples
    list_available_templates()
    example_basic_template_usage()
    example_custom_template()
    example_domain_specific_template()
    example_custom_operation()
    example_template_inheritance()
    
    print("Examples completed!")