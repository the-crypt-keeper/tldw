# Chunking Templates API Documentation

## Overview

The Chunking Templates API provides a powerful way to define, manage, and apply reusable document chunking configurations. Templates allow you to standardize how different types of content are processed, ensuring consistency across your application.

## Key Features

- **Template Management**: Create, read, update, and delete chunking templates
- **Built-in Templates**: Pre-configured templates for common document types
- **Template Application**: Apply templates to text content via API
- **Template Validation**: Validate template configurations before saving
- **Integration**: Use templates with the existing chunking API

## Base URL

```
http://localhost:8000/api/v1/chunking/templates
```

## Authentication

Currently, the API does not require authentication. In production, you should implement proper authentication using the provided auth middleware.

## Endpoints

### 1. List Templates

**GET** `/api/v1/chunking/templates`

List all available chunking templates with optional filtering.

#### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| include_builtin | boolean | true | Include built-in templates |
| include_custom | boolean | true | Include custom templates |
| tags | array[string] | null | Filter by tags |
| user_id | string | null | Filter by user ID |

#### Response

```json
{
  "templates": [
    {
      "id": 1,
      "uuid": "550e8400-e29b-41d4-a716-446655440000",
      "name": "academic_paper",
      "description": "Template for processing academic papers",
      "template_json": "{...}",
      "is_builtin": true,
      "tags": ["academic", "research", "paper"],
      "created_at": "2024-01-24T10:00:00Z",
      "updated_at": "2024-01-24T10:00:00Z",
      "version": 1,
      "user_id": null
    }
  ],
  "total": 1
}
```

#### Example Request

```bash
curl -X GET "http://localhost:8000/api/v1/chunking/templates?tags=academic&include_custom=false"
```

### 2. Get Template

**GET** `/api/v1/chunking/templates/{template_name}`

Retrieve a specific template by name.

#### Path Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| template_name | string | Yes | Name of the template |

#### Response

```json
{
  "id": 1,
  "uuid": "550e8400-e29b-41d4-a716-446655440000",
  "name": "academic_paper",
  "description": "Template for processing academic papers",
  "template_json": "{\"preprocessing\":[...],\"chunking\":{...},\"postprocessing\":[...]}",
  "is_builtin": true,
  "tags": ["academic", "research"],
  "created_at": "2024-01-24T10:00:00Z",
  "updated_at": "2024-01-24T10:00:00Z",
  "version": 1,
  "user_id": null
}
```

#### Example Request

```bash
curl -X GET "http://localhost:8000/api/v1/chunking/templates/academic_paper"
```

### 3. Create Template

**POST** `/api/v1/chunking/templates`

Create a new chunking template.

#### Request Body

```json
{
  "name": "custom_template",
  "description": "My custom chunking template",
  "tags": ["custom", "example"],
  "user_id": "user123",
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
          "min_length": 20
        }
      }
    ]
  }
}
```

#### Response

Status: 201 Created

```json
{
  "id": 2,
  "uuid": "660e8400-e29b-41d4-a716-446655440001",
  "name": "custom_template",
  "description": "My custom chunking template",
  "template_json": "{...}",
  "is_builtin": false,
  "tags": ["custom", "example"],
  "created_at": "2024-01-24T11:00:00Z",
  "updated_at": "2024-01-24T11:00:00Z",
  "version": 1,
  "user_id": "user123"
}
```

### 4. Update Template

**PUT** `/api/v1/chunking/templates/{template_name}`

Update an existing template (cannot update built-in templates).

#### Path Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| template_name | string | Yes | Name of the template |

#### Request Body

```json
{
  "description": "Updated description",
  "tags": ["updated", "modified"],
  "template": {
    "chunking": {
      "method": "sentences",
      "config": {
        "max_size": 10,
        "overlap": 2
      }
    }
  }
}
```

### 5. Delete Template

**DELETE** `/api/v1/chunking/templates/{template_name}`

Delete a template (cannot delete built-in templates).

#### Path Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| template_name | string | Yes | Name of the template |

#### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| hard_delete | boolean | false | Permanently delete instead of soft delete |

### 6. Apply Template

**POST** `/api/v1/chunking/templates/apply`

Apply a template to process text content.

#### Request Body

```json
{
  "template_name": "academic_paper",
  "text": "Your text content to be chunked...",
  "override_options": {
    "max_size": 10
  }
}
```

#### Response

```json
{
  "template_name": "academic_paper",
  "chunks": [
    "First chunk of text...",
    "Second chunk of text...",
    "Third chunk of text..."
  ],
  "metadata": {
    "chunk_count": 3,
    "template_version": 1
  }
}
```

### 7. Validate Template

**POST** `/api/v1/chunking/templates/validate`

Validate a template configuration without saving it.

#### Request Body

```json
{
  "preprocessing": [...],
  "chunking": {
    "method": "words",
    "config": {
      "max_size": 100
    }
  },
  "postprocessing": [...]
}
```

#### Response

```json
{
  "valid": true,
  "errors": null,
  "warnings": null
}
```

## Built-in Templates

The system comes with several pre-configured templates:

### 1. academic_paper
- **Description**: Template for processing academic papers
- **Method**: Sentences
- **Features**: Section extraction, intelligent merging
- **Tags**: academic, research, paper, scientific

### 2. code_documentation
- **Description**: Template for processing code documentation
- **Method**: Markdown-aware
- **Features**: Preserves code blocks and headers
- **Tags**: code, documentation, technical, programming

### 3. chat_conversation
- **Description**: Template for processing chat conversations
- **Method**: Sentences
- **Features**: Context preservation with overlap
- **Tags**: chat, conversation, dialogue, messaging

### 4. book_chapters
- **Description**: Template for processing books by chapters
- **Method**: Regex-based chapter detection
- **Features**: Chapter boundary detection, metadata addition
- **Tags**: book, ebook, chapters, novel, long-form

### 5. transcript_dialogue
- **Description**: Template for processing transcripts with speakers
- **Method**: Regex-based speaker detection
- **Features**: Speaker identification, dialogue preservation
- **Tags**: transcript, dialogue, interview, meeting, podcast

### 6. legal_document
- **Description**: Template for processing legal documents
- **Method**: Paragraph-based with section preservation
- **Features**: Section numbering preservation, formal structure
- **Tags**: legal, contract, law, document, formal

## Template Configuration Structure

### Complete Template Schema

```json
{
  "name": "template_name",
  "description": "Template description",
  "tags": ["tag1", "tag2"],
  "preprocessing": [
    {
      "operation": "operation_name",
      "config": {
        // Operation-specific configuration
      }
    }
  ],
  "chunking": {
    "method": "chunking_method",
    "config": {
      "max_size": 100,
      "overlap": 20,
      // Method-specific options
    }
  },
  "postprocessing": [
    {
      "operation": "operation_name",
      "config": {
        // Operation-specific configuration
      }
    }
  ]
}
```

### Available Operations

#### Preprocessing Operations
- `normalize_whitespace`: Normalize spacing and line breaks
- `remove_headers`: Remove header sections
- `extract_sections`: Extract sections based on patterns
- `clean_markdown`: Clean markdown formatting
- `detect_language`: Auto-detect text language

#### Chunking Methods
- `words`: Chunk by word count
- `sentences`: Chunk by sentence count
- `paragraphs`: Chunk by paragraph count
- `tokens`: Chunk by token count
- `semantic`: Semantic similarity-based chunking
- `regex`: Pattern-based chunking
- `markdown`: Markdown-aware chunking
- `code`: Code-aware chunking

#### Postprocessing Operations
- `add_overlap`: Add overlap between chunks
- `filter_empty`: Remove empty or short chunks
- `merge_small`: Merge small chunks
- `add_metadata`: Add metadata to chunks
- `format_chunks`: Format chunks with templates

## Integration with Existing Chunking API

Templates can be used with the existing chunking endpoint by specifying the `template_name` parameter:

```bash
curl -X POST "http://localhost:8000/api/v1/chunking/chunk_text" \
  -H "Content-Type: application/json" \
  -d '{
    "text_content": "Your text here...",
    "options": {
      "template_name": "academic_paper",
      "max_size": 10
    }
  }'
```

When a template is specified:
1. Template settings are loaded first
2. Any additional options override template defaults
3. Text is processed according to the combined configuration

## Error Handling

### Common Error Responses

#### 400 Bad Request
```json
{
  "detail": "Invalid template configuration: chunking method required"
}
```

#### 404 Not Found
```json
{
  "detail": "Template 'non_existent' not found"
}
```

#### 409 Conflict
```json
{
  "detail": "Template with name 'duplicate_name' already exists"
}
```

#### 500 Internal Server Error
```json
{
  "detail": "Internal server error: Database connection failed"
}
```

## Best Practices

1. **Template Naming**: Use descriptive, lowercase names with underscores
2. **Tags**: Use consistent tags for easy filtering
3. **Versioning**: Templates automatically version on updates
4. **Testing**: Always validate templates before production use
5. **Built-in Templates**: Use built-in templates as starting points
6. **Override Options**: Use override options for temporary adjustments

## Examples

### Example 1: Creating a Custom News Article Template

```bash
curl -X POST "http://localhost:8000/api/v1/chunking/templates" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "news_article",
    "description": "Template for news articles with headline preservation",
    "tags": ["news", "article", "journalism"],
    "template": {
      "preprocessing": [
        {
          "operation": "extract_sections",
          "config": {
            "pattern": "^(HEADLINE|BYLINE|DATELINE):"
          }
        }
      ],
      "chunking": {
        "method": "paragraphs",
        "config": {
          "max_size": 3,
          "overlap": 1
        }
      },
      "postprocessing": [
        {
          "operation": "filter_empty",
          "config": {
            "min_length": 50
          }
        },
        {
          "operation": "add_metadata",
          "config": {
            "prefix": "[Article Section {index}] "
          }
        }
      ]
    }
  }'
```

### Example 2: Processing a Research Paper

```python
import requests

# Apply the academic_paper template
response = requests.post(
    "http://localhost:8000/api/v1/chunking/templates/apply",
    json={
        "template_name": "academic_paper",
        "text": """
        # Abstract
        This paper presents a novel approach...
        
        # Introduction
        Recent advances in machine learning...
        
        # Methodology
        We propose a three-stage approach...
        """
    }
)

chunks = response.json()["chunks"]
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk[:100]}...")
```

### Example 3: Batch Processing with Templates

```python
import requests

documents = [
    {"type": "academic_paper", "text": "..."},
    {"type": "code_documentation", "text": "..."},
    {"type": "legal_document", "text": "..."}
]

for doc in documents:
    response = requests.post(
        "http://localhost:8000/api/v1/chunking/templates/apply",
        json={
            "template_name": doc["type"],
            "text": doc["text"]
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Processed {doc['type']}: {result['metadata']['chunk_count']} chunks")
```

## Migration Guide

If you're currently using the chunking API without templates, here's how to migrate:

### Before (Direct Chunking)
```json
{
  "text_content": "Your text...",
  "options": {
    "method": "sentences",
    "max_size": 5,
    "overlap": 1
  }
}
```

### After (Using Templates)
```json
{
  "text_content": "Your text...",
  "options": {
    "template_name": "academic_paper"
  }
}
```

Or create a custom template matching your current configuration:
```bash
curl -X POST "/api/v1/chunking/templates" \
  -d '{
    "name": "my_config",
    "template": {
      "chunking": {
        "method": "sentences",
        "config": {
          "max_size": 5,
          "overlap": 1
        }
      }
    }
  }'
```

---

*Last Updated: January 2025*
*API Version: 1.0.0*