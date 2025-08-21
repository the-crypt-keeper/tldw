# Chunking Templates

This directory contains JSON templates that define various text chunking strategies. Templates provide a flexible way to configure complex chunking pipelines with preprocessing and postprocessing stages.

## Template Structure

Each template is a JSON file with the following structure:

```json
{
  "name": "template_name",
  "description": "Description of the template",
  "base_method": "default_chunking_method",
  "pipeline": [
    {
      "stage": "preprocess|chunk|postprocess",
      "operations": [...],
      "method": "chunking_method",
      "options": {...}
    }
  ],
  "metadata": {
    "suitable_for": [...],
    "other_metadata": "..."
  }
}
```

## Available Templates

### Basic Templates
- **words.json** - Word-based chunking with configurable size
- **sentences.json** - Sentence-based chunking preserving grammar
- **paragraphs.json** - Paragraph-based chunking for structured documents
- **tokens.json** - Token-based chunking for LLM processing

### Specialized Templates
- **semantic.json** - Groups related content based on similarity
- **ebook_chapters.json** - Preserves e-book chapter structure
- **json.json** - JSON-aware chunking for structured data
- **xml.json** - XML-aware chunking with path preservation
- **rolling_summarize.json** - Progressive summarization for long documents

### Domain-Specific Templates
- **academic_paper.json** - Optimized for research papers
- **code_documentation.json** - For technical documentation
- **legal_document.json** - Preserves legal document structure
- **conversation.json** - For dialogue and transcripts

## Using Templates

### In Code
```python
from tldw_chatbook.Chunking import Chunker

# Use a template by name
chunker = Chunker(template="academic_paper")
chunks = chunker.chunk_text(text)

# Or with the improved process
from tldw_chatbook.Chunking import improved_chunking_process
chunks = improved_chunking_process(text, template="academic_paper")
```

### In Configuration
Add to your `config.toml`:
```toml
[chunking_config]
template = "academic_paper"
```

## Creating Custom Templates

1. Create a new JSON file in `~/.config/tldw_cli/chunking_templates/`
2. Define your pipeline stages
3. Use it by name in your code

### Example Custom Template
```json
{
  "name": "my_custom_template",
  "description": "Custom chunking for my use case",
  "base_method": "sentences",
  "pipeline": [
    {
      "stage": "preprocess",
      "operations": [
        {"type": "normalize_whitespace"},
        {"type": "extract_metadata", "params": {"patterns": ["title", "author"]}}
      ]
    },
    {
      "stage": "chunk",
      "method": "semantic",
      "options": {
        "max_size": 1000,
        "overlap": 100
      }
    },
    {
      "stage": "postprocess", 
      "operations": [
        {"type": "add_context", "params": {"context_size": 2}}
      ]
    }
  ]
}
```

## Available Operations

### Preprocessing Operations
- **normalize_whitespace** - Normalize spaces and newlines
- **extract_metadata** - Extract metadata by patterns
- **section_detection** - Detect document sections
- **code_block_detection** - Identify code blocks

### Postprocessing Operations
- **add_context** - Add surrounding chunk context
- **add_overlap** - Add text overlap between chunks
- **filter_empty** - Remove empty/short chunks
- **merge_small** - Combine small chunks

## Template Inheritance

Templates can inherit from other templates:

```json
{
  "name": "my_research_paper",
  "parent_template": "academic_paper",
  "pipeline": [
    {
      "stage": "chunk",
      "options": {
        "max_size": 1200
      }
    }
  ]
}
```

This inherits all settings from `academic_paper` but overrides the chunk size.