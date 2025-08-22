# Enhanced RAG Implementation Features

This document describes the advanced RAG features implemented based on analysis of the external RAG pipeline from https://github.com/IlyaRice/RAG-Challenge-2.

## Overview

The enhanced RAG implementation adds several sophisticated features to improve retrieval quality and context understanding:

1. **Enhanced Chunking with Character-Level Position Tracking**
2. **Hierarchical Document Structure Preservation**
3. **Parent Document Retrieval**
4. **Advanced Text Processing**
5. **Table Serialization**

## 1. Enhanced Chunking with Character-Level Position Tracking

### Features
- Accurate character-level position tracking (`start_char`, `end_char`)
- Word count tracking per chunk
- Hierarchical chunk relationships (parent-child)
- Structure-aware chunking that respects document boundaries

### Implementation
```python
from tldw_chatbook.RAG_Search.enhanced_chunking_service import EnhancedChunkingService

service = EnhancedChunkingService()
chunks = service.chunk_text_with_structure(
    content,
    chunk_size=400,
    chunk_overlap=100,
    method="hierarchical",  # or "structural"
    preserve_structure=True,
    clean_artifacts=True,
    serialize_tables=True
)
```

### Benefits
- Precise citation generation
- Better chunk boundary detection
- Maintains document structure integrity

## 2. Hierarchical Document Structure Preservation

### Features
- Identifies document elements (headers, sections, lists, tables, etc.)
- Maintains parent-child relationships between chunks
- Preserves hierarchical levels (h1, h2, h3, etc.)
- Respects structural boundaries when chunking

### Chunk Types
- `HEADER` - Document headers
- `SECTION` - Main content sections
- `LIST` - Bulleted or numbered lists
- `TABLE` - Tabular data
- `CODE_BLOCK` - Code snippets
- `QUOTE` - Quoted text
- `FOOTNOTE` - Footnotes and references

### Implementation
```python
# Each chunk includes structural metadata
chunk = StructuredChunk(
    text="chunk content",
    chunk_type=ChunkType.SECTION,
    level=2,  # Hierarchical level
    parent_index=0,  # Reference to parent chunk
    children_indices=[3, 4, 5]  # Child chunks
)
```

## 3. Parent Document Retrieval

### Features
- Creates smaller chunks for precise retrieval
- Maintains larger parent chunks for context
- Automatic context expansion during search
- Configurable parent size multiplier

### Implementation
```python
from tldw_chatbook.RAG_Search.simplified.enhanced_rag_service import EnhancedRAGService

service = EnhancedRAGService(enable_parent_retrieval=True)

# Index with parent chunks
result = await service.index_document_with_parents(
    doc_id="doc_001",
    content=content,
    parent_size_multiplier=3  # Parent chunks 3x larger
)

# Search with automatic context expansion
results = await service.search_with_context_expansion(
    query="your search query",
    expand_to_parent=True
)
```

### Benefits
- Better context for LLM understanding
- Maintains retrieval precision
- Reduces hallucination by providing more context

## 4. Advanced Text Processing

### PDF Artifact Cleaning
Automatically cleans common PDF parsing artifacts:
- Command replacements (`/period` → `.`, `/comma` → `,`)
- Glyph removal (`glyph<123>` → ``)
- Character normalization (`/A.cap` → `A`)

### Implementation
```python
parser = DocumentStructureParser()
cleaned_text, corrections = parser.clean_text(pdf_text)
```

### Structure-Aware Formatting
- Preserves document hierarchy
- Maintains list and table relationships
- Handles footnotes and references correctly

## 5. Table Serialization

### Features
- Multiple serialization methods (entities, sentences, hybrid)
- Preserves table structure and relationships
- Creates searchable representations of tabular data

### Serialization Methods

#### Entity Blocks
Each table row becomes a structured entity:
```
Row 1; Product: Laptop; Q1 Sales: 1000; Q2 Sales: 1200; Growth: 20%
```

#### Natural Language Sentences
Tables converted to descriptive sentences:
```
This table contains 3 rows and 4 columns with headers: Product, Q1 Sales, Q2 Sales, Growth.
In row 1, Product is Laptop, Q1 Sales is 1000, Q2 Sales is 1200, Growth is 20%.
```

### Implementation
```python
from tldw_chatbook.RAG_Search.table_serializer import serialize_table

result = serialize_table(
    table_text,
    format=TableFormat.MARKDOWN,
    method="hybrid"  # Uses both entities and sentences
)
```

## Usage Examples

### Basic Usage
```python
# Create enhanced RAG service
from tldw_chatbook.RAG_Search.simplified.enhanced_rag_service import create_enhanced_rag_service

service = create_enhanced_rag_service(
    embedding_model="all-MiniLM-L6-v2",
    enable_parent_retrieval=True
)

# Index document with all enhancements
await service.index_document_with_parents(
    doc_id="doc_001",
    content=document_text,
    title="Document Title",
    use_structural_chunking=True
)

# Search with context expansion
results = await service.search_with_context_expansion(
    query="your query",
    top_k=5,
    expand_to_parent=True
)
```

### Batch Processing
```python
# Process multiple documents
results = await service.index_batch_with_parents(
    documents=[
        {"id": "doc1", "content": "...", "title": "..."},
        {"id": "doc2", "content": "...", "title": "..."}
    ],
    use_structural_chunking=True
)
```

## Configuration Options

### Enhanced Chunking
- `chunk_size` - Target size for retrieval chunks
- `chunk_overlap` - Overlap between chunks
- `parent_size_multiplier` - How much larger parent chunks are
- `preserve_structure` - Whether to maintain document structure
- `clean_artifacts` - Whether to clean PDF artifacts
- `serialize_tables` - Whether to serialize tables

### Search Options
- `expand_to_parent` - Automatically expand to parent context
- `include_citations` - Include citation information
- `search_type` - "semantic", "keyword", or "hybrid"

## Performance Considerations

1. **Memory Usage**: Parent chunks increase memory usage by ~3x
2. **Indexing Time**: Enhanced processing adds 20-30% overhead
3. **Search Performance**: Context expansion has minimal impact
4. **Storage**: Requires additional metadata storage

## Migration Guide

### From Basic to Enhanced RAG

1. Replace imports:
```python
# Old
from tldw_chatbook.RAG_Search.simplified.rag_service import RAGService

# New
from tldw_chatbook.RAG_Search.simplified.enhanced_rag_service import EnhancedRAGService
```

2. Update service creation:
```python
# Old
service = RAGService(config)

# New
service = EnhancedRAGService(config, enable_parent_retrieval=True)
```

3. Use enhanced methods:
```python
# Old
await service.index_document(...)

# New
await service.index_document_with_parents(...)
```

## Testing

Run the test script to verify functionality:
```bash
python test_enhanced_rag.py
```

This will test:
- Enhanced chunking with structure preservation
- Parent document retrieval
- Table serialization
- PDF artifact cleaning
- Context expansion during search