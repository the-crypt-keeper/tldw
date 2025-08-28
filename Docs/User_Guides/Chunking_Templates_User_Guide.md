# Chunking Templates User Guide

## What are Chunking Templates?

Chunking templates are pre-configured strategies for breaking down documents into smaller, manageable pieces. Think of them as recipes that tell the system exactly how to process different types of content - whether it's an academic paper, a chat conversation, or a legal document.

## Why Use Templates?

- **Consistency**: Process similar documents the same way every time
- **Efficiency**: No need to remember specific settings for different document types
- **Quality**: Built-in templates are optimized for their specific content types
- **Simplicity**: Just specify a template name instead of multiple parameters

## Available Built-in Templates

### 📚 academic_paper
**Best for**: Research papers, scientific articles, thesis documents
- Extracts sections like Abstract, Introduction, Methods
- Preserves academic structure
- Merges small fragments intelligently

### 💻 code_documentation
**Best for**: API docs, README files, technical documentation
- Preserves code blocks intact
- Maintains header hierarchy
- Handles markdown formatting

### 💬 chat_conversation
**Best for**: Chat logs, instant messages, conversation transcripts
- Maintains conversation context
- Adds overlap for continuity
- Preserves speaker information

### 📖 book_chapters
**Best for**: Novels, textbooks, long-form content
- Detects chapter boundaries
- Handles various chapter formats (Chapter 1, Part I, etc.)
- Adds chapter metadata to chunks

### 🎙️ transcript_dialogue
**Best for**: Interview transcripts, meeting notes, podcasts
- Identifies speakers
- Groups dialogue by speaker turns
- Maintains conversation flow

### ⚖️ legal_document
**Best for**: Contracts, agreements, legal briefs
- Preserves section numbering
- Maintains legal structure
- Handles formal formatting

## How to Use Templates

### Using Templates via API

#### Option 1: Using the Chunking Endpoint with a Template

```bash
curl -X POST "http://localhost:8000/api/v1/chunking/chunk_text" \
  -H "Content-Type: application/json" \
  -d '{
    "text_content": "Your document text here...",
    "options": {
      "template_name": "academic_paper"
    }
  }'
```

#### Option 2: Apply Template Directly

```bash
curl -X POST "http://localhost:8000/api/v1/chunking/templates/apply" \
  -H "Content-Type: application/json" \
  -d '{
    "template_name": "academic_paper",
    "text": "Your document text here..."
  }'
```

### Using Templates in Python

```python
import requests

# Example: Process an academic paper
def chunk_with_template(text, template_name):
    response = requests.post(
        "http://localhost:8000/api/v1/chunking/templates/apply",
        json={
            "template_name": template_name,
            "text": text
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        return result["chunks"]
    else:
        print(f"Error: {response.status_code}")
        return None

# Use it
paper_text = """
# Abstract
This study investigates...

# Introduction
Previous research has shown...

# Methods
We conducted experiments...
"""

chunks = chunk_with_template(paper_text, "academic_paper")
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}: {chunk[:100]}...")
```

## Creating Custom Templates

### Step 1: Design Your Template

Decide on:
- What preprocessing do you need? (cleaning, normalizing)
- What chunking method works best? (sentences, paragraphs, custom patterns)
- What postprocessing helps? (filtering, merging, adding metadata)

### Step 2: Create the Template

```bash
curl -X POST "http://localhost:8000/api/v1/chunking/templates" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_custom_template",
    "description": "Template for my specific documents",
    "tags": ["custom", "myproject"],
    "template": {
      "preprocessing": [
        {
          "operation": "normalize_whitespace",
          "config": {
            "max_line_breaks": 2
          }
        }
      ],
      "chunking": {
        "method": "sentences",
        "config": {
          "max_size": 5,
          "overlap": 1
        }
      },
      "postprocessing": [
        {
          "operation": "filter_empty",
          "config": {
            "min_length": 50
          }
        }
      ]
    }
  }'
```

### Step 3: Test Your Template

Before using in production, validate your template:

```bash
curl -X POST "http://localhost:8000/api/v1/chunking/templates/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "chunking": {
      "method": "sentences",
      "config": {
        "max_size": 5
      }
    }
  }'
```

### Step 4: Use Your Template

```bash
curl -X POST "http://localhost:8000/api/v1/chunking/templates/apply" \
  -H "Content-Type: application/json" \
  -d '{
    "template_name": "my_custom_template",
    "text": "Your content here..."
  }'
```

## Template Configuration Options

### Preprocessing Operations

| Operation | Purpose | Config Options |
|-----------|---------|----------------|
| normalize_whitespace | Clean up spacing | max_line_breaks |
| remove_headers | Remove headers/footers | pattern |
| extract_sections | Find document sections | pattern |
| clean_markdown | Remove markdown syntax | remove_images, remove_links |
| detect_language | Auto-detect language | (none) |

### Chunking Methods

| Method | Best For | Key Options |
|--------|----------|-------------|
| words | General text | max_size, overlap |
| sentences | Natural breaks | max_size, overlap |
| paragraphs | Structured text | max_size, overlap |
| tokens | LLM processing | max_size, overlap |
| semantic | Related content | similarity_threshold |
| regex | Custom patterns | pattern |
| markdown | Markdown docs | preserve_headers |

### Postprocessing Operations

| Operation | Purpose | Config Options |
|-----------|---------|----------------|
| filter_empty | Remove short chunks | min_length |
| merge_small | Combine tiny chunks | min_size, separator |
| add_overlap | Add context | size, marker |
| add_metadata | Add labels | prefix, suffix |
| format_chunks | Custom formatting | template |

## Managing Templates

### List All Templates

```bash
curl -X GET "http://localhost:8000/api/v1/chunking/templates"
```

### Get Specific Template

```bash
curl -X GET "http://localhost:8000/api/v1/chunking/templates/academic_paper"
```

### Update a Template

```bash
curl -X PUT "http://localhost:8000/api/v1/chunking/templates/my_custom_template" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Updated description",
    "tags": ["updated", "v2"]
  }'
```

### Delete a Template

```bash
curl -X DELETE "http://localhost:8000/api/v1/chunking/templates/my_custom_template"
```

## Tips and Best Practices

### Choosing the Right Template

1. **Document Type Matters**
   - Academic papers → `academic_paper`
   - Technical docs → `code_documentation`
   - Conversations → `chat_conversation` or `transcript_dialogue`
   - Books/novels → `book_chapters`
   - Legal documents → `legal_document`

2. **Consider Your Use Case**
   - For search: Smaller chunks with overlap
   - For summarization: Larger, complete sections
   - For analysis: Semantic or paragraph-based

3. **Test Before Production**
   - Try templates on sample documents
   - Check chunk sizes and boundaries
   - Verify important content isn't split

### Customizing Templates

1. **Start with Built-in Templates**
   - Copy a similar built-in template
   - Modify settings gradually
   - Test each change

2. **Common Customizations**
   ```json
   {
     "chunking": {
       "method": "sentences",
       "config": {
         "max_size": 10,  // Increase for larger chunks
         "overlap": 2     // Increase for more context
       }
     }
   }
   ```

3. **Override Options**
   You can override template settings without creating a new template:
   ```json
   {
     "template_name": "academic_paper",
     "text": "...",
     "override_options": {
       "max_size": 20  // Override just this setting
     }
   }
   ```

## Common Use Cases

### Processing Research Papers

```python
# Use academic_paper template for PDFs converted to text
with open('research_paper.txt', 'r') as f:
    paper_text = f.read()

response = requests.post(
    "http://localhost:8000/api/v1/chunking/templates/apply",
    json={
        "template_name": "academic_paper",
        "text": paper_text
    }
)
```

### Analyzing Chat Logs

```python
# Use chat_conversation template for chat exports
chat_log = """
User1: Hey, how's the project going?
User2: Pretty good! Just finished the API integration.
User1: Awesome! Any issues?
User2: Just some minor bugs, nothing major.
"""

response = requests.post(
    "http://localhost:8000/api/v1/chunking/templates/apply",
    json={
        "template_name": "chat_conversation",
        "text": chat_log
    }
)
```

### Processing Legal Documents

```python
# Use legal_document template for contracts
contract_text = """
ARTICLE 1. DEFINITIONS
1.1 "Agreement" means this contract...
1.2 "Party" means...

ARTICLE 2. TERMS
2.1 The term of this Agreement shall...
"""

response = requests.post(
    "http://localhost:8000/api/v1/chunking/templates/apply",
    json={
        "template_name": "legal_document",
        "text": contract_text
    }
)
```

## Troubleshooting

### Template Not Found
- Check spelling of template name
- List all templates to see available options
- Ensure template wasn't deleted

### Chunks Too Large/Small
- Adjust `max_size` in chunking config
- Try different chunking method
- Check preprocessing isn't removing content

### Missing Content
- Verify `filter_empty` settings
- Check `min_length` in postprocessing
- Ensure preprocessing isn't too aggressive

### Poor Quality Chunks
- Try different template for your content type
- Adjust overlap for better context
- Consider semantic chunking for related content

## Getting Help

1. **Check Available Templates**
   ```bash
   curl -X GET "http://localhost:8000/api/v1/chunking/templates"
   ```

2. **Validate Your Configuration**
   ```bash
   curl -X POST "http://localhost:8000/api/v1/chunking/templates/validate" \
     -d '{"your": "config"}'
   ```

3. **Review Documentation**
   - This user guide
   - API documentation
   - Developer guide for advanced usage

4. **Test with Small Samples**
   - Start with small text samples
   - Gradually increase complexity
   - Save working configurations

## Examples Repository

For more examples and use cases, check:
- `/Docs/API-related/Chunking_Templates_API_Documentation.md` - Full API reference
- `/Docs/Code_Documentation/Chunking_Templates_Developer_Guide.md` - Technical details
- `/tests/Chunking/test_chunking_templates.py` - Working code examples

---

*Last Updated: January 2025*
*Version: 1.0.0*