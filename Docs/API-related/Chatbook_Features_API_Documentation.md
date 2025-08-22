# Chatbook Features API Documentation

## Overview
This document provides comprehensive API documentation for the Chatbook features migrated from the single-user TUI application to the multi-user API architecture. These features include Chat Dictionary, World Book Manager, Document Generator, and Chatbook Import/Export functionality.

## Table of Contents
1. [Chat Dictionary API](#chat-dictionary-api)
2. [World Book Manager API](#world-book-manager-api)
3. [Document Generator API](#document-generator-api)
4. [Chatbooks Import/Export API](#chatbooks-importexport-api)

---

## Chat Dictionary API

The Chat Dictionary API provides pattern-based text replacement functionality for conversations, supporting both literal and regex patterns with probability-based application.

### Base URL
`/api/v1/dictionaries`

### Authentication
All endpoints require a valid JWT token in the Authorization header:
```
Authorization: Bearer <token>
```

### Endpoints

#### 1. Create Dictionary
**POST** `/api/v1/dictionaries/create`

Creates a new chat dictionary for the authenticated user.

**Request Body:**
```json
{
  "name": "Fantasy Terms",
  "description": "Convert modern terms to fantasy equivalents",
  "is_active": true
}
```

**Response:**
```json
{
  "success": true,
  "dictionary_id": 1,
  "message": "Dictionary created successfully"
}
```

#### 2. List Dictionaries
**GET** `/api/v1/dictionaries/list`

Lists all dictionaries for the authenticated user.

**Query Parameters:**
- `include_inactive` (boolean): Include inactive dictionaries (default: false)

**Response:**
```json
{
  "dictionaries": [
    {
      "id": 1,
      "name": "Fantasy Terms",
      "description": "Convert modern terms to fantasy equivalents",
      "is_active": true,
      "entry_count": 25,
      "created_at": "2024-01-01T00:00:00Z",
      "updated_at": "2024-01-01T00:00:00Z"
    }
  ]
}
```

#### 3. Get Dictionary
**GET** `/api/v1/dictionaries/{dictionary_id}`

Retrieves a specific dictionary with all its entries.

**Response:**
```json
{
  "id": 1,
  "name": "Fantasy Terms",
  "description": "Convert modern terms to fantasy equivalents",
  "is_active": true,
  "entries": [
    {
      "id": 1,
      "key_pattern": "car",
      "replacement": "carriage",
      "is_regex": false,
      "probability": 100,
      "max_replacements": 1
    }
  ],
  "created_at": "2024-01-01T00:00:00Z"
}
```

#### 4. Add Entry
**POST** `/api/v1/dictionaries/{dictionary_id}/entries`

Adds a new entry to a dictionary.

**Request Body:**
```json
{
  "key_pattern": "phone",
  "replacement": "sending stone",
  "is_regex": false,
  "probability": 100,
  "max_replacements": 1
}
```

**Response:**
```json
{
  "success": true,
  "entry_id": 2,
  "message": "Entry added successfully"
}
```

#### 5. Update Entry
**PUT** `/api/v1/dictionaries/entries/{entry_id}`

Updates an existing dictionary entry.

**Request Body:**
```json
{
  "key_pattern": "telephone",
  "replacement": "communication crystal",
  "probability": 90
}
```

#### 6. Delete Entry
**DELETE** `/api/v1/dictionaries/entries/{entry_id}`

Deletes a dictionary entry.

#### 7. Process Text
**POST** `/api/v1/dictionaries/process`

Processes text through active dictionaries.

**Request Body:**
```json
{
  "text": "I'll call you on my phone from the car",
  "token_budget": 1000,
  "dictionary_ids": [1]  // Optional, uses all active if not specified
}
```

**Response:**
```json
{
  "processed_text": "I'll call you on my sending stone from the carriage",
  "replacements_made": 2,
  "dictionaries_used": ["Fantasy Terms"],
  "token_budget_exceeded": false
}
```

#### 8. Bulk Add Entries
**POST** `/api/v1/dictionaries/{dictionary_id}/entries/bulk`

Adds multiple entries at once.

**Request Body:**
```json
{
  "entries": [
    {"key_pattern": "sword", "replacement": "blade", "is_regex": false},
    {"key_pattern": "gun", "replacement": "wand", "is_regex": false}
  ]
}
```

#### 9. Import Dictionary
**POST** `/api/v1/dictionaries/import`

Imports a dictionary from markdown format.

**Request Body (multipart/form-data):**
- `file`: Markdown file containing dictionary data

#### 10. Export Dictionary
**GET** `/api/v1/dictionaries/{dictionary_id}/export`

Exports a dictionary to markdown format.

#### 11. Clone Dictionary
**POST** `/api/v1/dictionaries/{dictionary_id}/clone`

Creates a copy of an existing dictionary.

**Request Body:**
```json
{
  "new_name": "Fantasy Terms Copy"
}
```

#### 12. Toggle Active Status
**PUT** `/api/v1/dictionaries/{dictionary_id}/toggle`

Toggles a dictionary's active status.

#### 13. Delete Dictionary
**DELETE** `/api/v1/dictionaries/{dictionary_id}`

Deletes a dictionary and all its entries.

#### 14. Search Entries
**GET** `/api/v1/dictionaries/entries/search`

Searches for entries across all dictionaries.

**Query Parameters:**
- `query` (string): Search term

#### 15. Get Statistics
**GET** `/api/v1/dictionaries/statistics`

Gets dictionary usage statistics.

---

## World Book Manager API

The World Book Manager API provides contextual lore injection for character conversations, supporting keyword-based entry matching and character-specific attachments.

### Base URL
`/api/v1/worldbooks`

### Endpoints

#### 1. Create World Book
**POST** `/api/v1/worldbooks/create`

Creates a new world book.

**Request Body:**
```json
{
  "name": "Fantasy Campaign",
  "description": "D&D campaign setting",
  "scan_depth": 3,
  "token_budget": 1000,
  "recursive_scanning": true,
  "enabled": true
}
```

**Response:**
```json
{
  "success": true,
  "world_book_id": 1,
  "message": "World book created successfully"
}
```

#### 2. List World Books
**GET** `/api/v1/worldbooks/list`

Lists all world books for the user.

**Query Parameters:**
- `include_disabled` (boolean): Include disabled world books

**Response:**
```json
{
  "world_books": [
    {
      "id": 1,
      "name": "Fantasy Campaign",
      "description": "D&D campaign setting",
      "entry_count": 50,
      "enabled": true,
      "created_at": "2024-01-01T00:00:00Z"
    }
  ]
}
```

#### 3. Get World Book
**GET** `/api/v1/worldbooks/{world_book_id}`

Retrieves a world book with all entries.

**Response:**
```json
{
  "id": 1,
  "name": "Fantasy Campaign",
  "entries": [
    {
      "id": 1,
      "keywords": "dragon,wyrm",
      "content": "Dragons are ancient magical beings",
      "priority": 100,
      "enabled": true
    }
  ]
}
```

#### 4. Add Entry
**POST** `/api/v1/worldbooks/{world_book_id}/entries`

Adds an entry to a world book.

**Request Body:**
```json
{
  "keywords": ["magic", "wizard", "spell"],
  "content": "Magic in this world comes from ley lines",
  "priority": 90,
  "enabled": true
}
```

#### 5. Update Entry
**PUT** `/api/v1/worldbooks/entries/{entry_id}`

Updates a world book entry.

#### 6. Delete Entry
**DELETE** `/api/v1/worldbooks/entries/{entry_id}`

Deletes a world book entry.

#### 7. Attach to Character
**POST** `/api/v1/worldbooks/{world_book_id}/attach`

Attaches a world book to a character.

**Request Body:**
```json
{
  "character_id": 1,
  "is_primary": true
}
```

#### 8. Detach from Character
**DELETE** `/api/v1/worldbooks/{world_book_id}/detach/{character_id}`

Detaches a world book from a character.

#### 9. Get Character World Books
**GET** `/api/v1/characters/{character_id}/worldbooks`

Gets all world books attached to a character.

#### 10. Process Context
**POST** `/api/v1/worldbooks/process`

Processes text to inject relevant world book entries.

**Request Body:**
```json
{
  "text": "Tell me about dragons and magic",
  "character_id": 1,
  "max_tokens": 1000
}
```

**Response:**
```json
{
  "processed_context": "Tell me about dragons and magic\n\n[World Info: Dragons are ancient magical beings...]\n[World Info: Magic in this world comes from ley lines...]",
  "entries_applied": 2,
  "token_budget_exceeded": false
}
```

#### 11. Search Entries
**GET** `/api/v1/worldbooks/entries/search`

Searches for entries across world books.

**Query Parameters:**
- `query` (string): Search term
- `world_book_id` (integer, optional): Limit to specific world book

#### 12. Import World Book
**POST** `/api/v1/worldbooks/import`

Imports a world book from JSON.

**Request Body:**
```json
{
  "name": "Imported World",
  "description": "Imported lore",
  "entries": [
    {"keywords": "test", "content": "Test content", "priority": 100}
  ]
}
```

#### 13. Export World Book
**GET** `/api/v1/worldbooks/{world_book_id}/export`

Exports a world book to JSON.

#### 14. Clone World Book
**POST** `/api/v1/worldbooks/{world_book_id}/clone`

Creates a copy of a world book.

#### 15. Delete World Book
**DELETE** `/api/v1/worldbooks/{world_book_id}`

Deletes a world book and all its entries.

#### 16. Bulk Update Entries
**PUT** `/api/v1/worldbooks/{world_book_id}/entries/bulk`

Updates multiple entries at once.

**Request Body:**
```json
{
  "entry_ids": [1, 2, 3],
  "enabled": false
}
```

#### 17. Get Statistics
**GET** `/api/v1/worldbooks/statistics`

Gets world book usage statistics.

---

## Document Generator API

The Document Generator API creates structured documents from conversations, supporting multiple document types and async job management.

### Base URL
`/api/v1/documents`

### Document Types
- `timeline`: Chronological event summary
- `study_guide`: Educational material with key concepts
- `briefing`: Executive summary
- `summary`: General conversation summary
- `qa_pairs`: Question and answer pairs
- `meeting_notes`: Structured meeting notes

### Endpoints

#### 1. Generate Document
**POST** `/api/v1/documents/generate`

Generates a document from a conversation.

**Request Body:**
```json
{
  "conversation_id": "conv123",
  "document_type": "timeline",
  "custom_prompt": "Focus on technical decisions",
  "async": false
}
```

**Response (Synchronous):**
```json
{
  "success": true,
  "document_id": "doc123",
  "document_type": "timeline",
  "title": "Conversation Timeline",
  "content": "Timeline:\n1. Initial discussion...\n2. Decision made...",
  "metadata": {
    "word_count": 250,
    "generation_time": 2.5
  }
}
```

**Response (Asynchronous):**
```json
{
  "success": true,
  "job_id": "job456",
  "status": "pending",
  "message": "Document generation job created"
}
```

#### 2. Get Document
**GET** `/api/v1/documents/{document_id}`

Retrieves a generated document.

#### 3. List Documents
**GET** `/api/v1/documents/list`

Lists documents for a conversation or user.

**Query Parameters:**
- `conversation_id` (string, optional): Filter by conversation
- `document_type` (string, optional): Filter by type
- `limit` (integer): Maximum results (default: 50)

#### 4. Delete Document
**DELETE** `/api/v1/documents/{document_id}`

Deletes a generated document.

#### 5. Bulk Generate
**POST** `/api/v1/documents/bulk`

Generates multiple document types at once.

**Request Body:**
```json
{
  "conversation_id": "conv123",
  "document_types": ["timeline", "summary", "qa_pairs"]
}
```

#### 6. Get Job Status
**GET** `/api/v1/documents/jobs/{job_id}`

Gets the status of an async generation job.

**Response:**
```json
{
  "job_id": "job456",
  "status": "completed",
  "document_id": "doc789",
  "created_at": "2024-01-01T00:00:00Z",
  "completed_at": "2024-01-01T00:00:05Z"
}
```

#### 7. Cancel Job
**DELETE** `/api/v1/documents/jobs/{job_id}`

Cancels a pending generation job.

#### 8. Save Prompt Config
**POST** `/api/v1/documents/prompts/config`

Saves custom prompt configurations.

**Request Body:**
```json
{
  "timeline": "Create a detailed timeline with timestamps",
  "summary": "Provide a concise executive summary"
}
```

#### 9. Get Prompt Config
**GET** `/api/v1/documents/prompts/config`

Gets saved prompt configurations.

#### 10. Get Statistics
**GET** `/api/v1/documents/statistics`

Gets document generation statistics.

**Response:**
```json
{
  "total_documents": 150,
  "documents_by_type": {
    "timeline": 40,
    "summary": 60,
    "qa_pairs": 50
  },
  "total_jobs": 200,
  "success_rate": 0.95
}
```

---

## Chatbooks Import/Export API

The Chatbooks API provides functionality to export and import collections of chat-related content, with conflict resolution and job management.

### Base URL
`/api/v1/chatbooks`

### Endpoints

#### 1. Export Chatbook
**POST** `/api/v1/chatbooks/export`

Exports selected content to a chatbook archive.

**Request Body:**
```json
{
  "name": "My Chatbook Export",
  "description": "Backup of my conversations",
  "content_types": ["conversations", "characters", "world_books", "dictionaries", "notes", "prompts"],
  "filters": {
    "date_from": "2024-01-01",
    "date_to": "2024-12-31",
    "tags": ["important"]
  },
  "async": true
}
```

**Response:**
```json
{
  "success": true,
  "job_id": "export_job_123",
  "status": "pending",
  "message": "Export job created successfully"
}
```

#### 2. Preview Export
**POST** `/api/v1/chatbooks/export/preview`

Previews what would be exported without creating the archive.

**Request Body:**
```json
{
  "content_types": ["conversations", "notes"]
}
```

**Response:**
```json
{
  "conversations": 25,
  "notes": 10,
  "estimated_size": "15.2 MB"
}
```

#### 3. Get Export Job Status
**GET** `/api/v1/chatbooks/export/jobs/{job_id}`

Gets the status of an export job.

**Response:**
```json
{
  "job_id": "export_job_123",
  "status": "completed",
  "file_path": "/exports/my_chatbook_20240101.chatbook",
  "download_url": "/api/v1/chatbooks/download/export_job_123",
  "content_summary": {
    "conversations": 25,
    "characters": 5,
    "world_books": 2,
    "dictionaries": 3,
    "notes": 10,
    "prompts": 15
  },
  "file_size": "15.2 MB",
  "created_at": "2024-01-01T00:00:00Z",
  "completed_at": "2024-01-01T00:00:30Z"
}
```

#### 4. Download Export
**GET** `/api/v1/chatbooks/download/{job_id}`

Downloads the exported chatbook file.

**Response:**
Binary file download with appropriate headers:
```
Content-Type: application/zip
Content-Disposition: attachment; filename="my_chatbook_20240101.chatbook"
```

#### 5. Import Chatbook
**POST** `/api/v1/chatbooks/import`

Imports a chatbook archive.

**Request Body (multipart/form-data):**
- `file`: The chatbook archive file
- `conflict_strategy`: How to handle conflicts ("skip", "replace", "rename")
- `async`: Whether to process asynchronously (boolean)

**Response:**
```json
{
  "success": true,
  "job_id": "import_job_456",
  "status": "pending",
  "message": "Import job created successfully"
}
```

#### 6. Get Import Job Status
**GET** `/api/v1/chatbooks/import/jobs/{job_id}`

Gets the status of an import job.

**Response:**
```json
{
  "job_id": "import_job_456",
  "status": "completed",
  "items_imported": 55,
  "conflicts_found": 5,
  "conflicts_resolved": {
    "skipped": 2,
    "replaced": 2,
    "renamed": 1
  },
  "created_at": "2024-01-01T00:00:00Z",
  "completed_at": "2024-01-01T00:01:00Z"
}
```

#### 7. Validate Chatbook
**POST** `/api/v1/chatbooks/validate`

Validates a chatbook file without importing.

**Request Body (multipart/form-data):**
- `file`: The chatbook archive file

**Response:**
```json
{
  "valid": true,
  "version": "1.0.0",
  "exported_at": "2024-01-01T00:00:00Z",
  "user_id": "original_user",
  "content_types": ["conversations", "notes"],
  "content_summary": {
    "conversations": 10,
    "notes": 5
  }
}
```

#### 8. List Export Jobs
**GET** `/api/v1/chatbooks/export/jobs`

Lists export jobs for the user.

**Query Parameters:**
- `status` (string): Filter by status ("pending", "processing", "completed", "failed", "cancelled")
- `limit` (integer): Maximum results

#### 9. List Import Jobs
**GET** `/api/v1/chatbooks/import/jobs`

Lists import jobs for the user.

#### 10. Cancel Export Job
**DELETE** `/api/v1/chatbooks/export/jobs/{job_id}`

Cancels a pending export job.

#### 11. Cancel Import Job
**DELETE** `/api/v1/chatbooks/import/jobs/{job_id}`

Cancels a pending import job.

#### 12. Clean Old Exports
**DELETE** `/api/v1/chatbooks/export/clean`

Removes old export files.

**Query Parameters:**
- `days_old` (integer): Delete exports older than this many days (default: 30)

#### 13. Get Statistics
**GET** `/api/v1/chatbooks/statistics`

Gets import/export statistics.

**Response:**
```json
{
  "total_exports": 50,
  "export_success_rate": 0.96,
  "total_imports": 30,
  "import_success_rate": 0.93,
  "total_items_exported": 1500,
  "total_items_imported": 900,
  "storage_used": "256.5 MB"
}
```

---

## Error Responses

All APIs use consistent error response format:

```json
{
  "success": false,
  "error": "Detailed error message",
  "error_code": "SPECIFIC_ERROR_CODE",
  "details": {
    // Additional error context
  }
}
```

### Common Error Codes
- `AUTH_REQUIRED`: Authentication token missing
- `AUTH_INVALID`: Invalid authentication token
- `NOT_FOUND`: Resource not found
- `VALIDATION_ERROR`: Request validation failed
- `CONFLICT`: Resource conflict (e.g., duplicate)
- `QUOTA_EXCEEDED`: User quota exceeded
- `RATE_LIMITED`: Too many requests
- `INTERNAL_ERROR`: Server error

## Rate Limiting

All endpoints are subject to rate limiting:
- **Default limit**: 100 requests per minute per user
- **Bulk operations**: 10 requests per minute per user
- **Export/Import**: 5 requests per minute per user

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1704067200
```

## Pagination

List endpoints support pagination:

**Query Parameters:**
- `page` (integer): Page number (default: 1)
- `per_page` (integer): Items per page (default: 20, max: 100)

**Response Headers:**
```
X-Total-Count: 250
X-Page: 1
X-Per-Page: 20
Link: </api/v1/resource?page=2>; rel="next"
```

## Webhooks

The system supports webhooks for async job completion:

**Webhook Payload:**
```json
{
  "event": "export.completed",
  "job_id": "export_job_123",
  "user_id": "user_456",
  "timestamp": "2024-01-01T00:00:30Z",
  "data": {
    "file_path": "/exports/my_chatbook_20240101.chatbook",
    "content_summary": {
      "conversations": 25
    }
  }
}
```

**Supported Events:**
- `export.started`
- `export.completed`
- `export.failed`
- `import.started`
- `import.completed`
- `import.failed`
- `document.generated`
- `document.generation_failed`

## SDK Examples

### Python
```python
import requests

class ChatbookAPI:
    def __init__(self, base_url, token):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {token}"}
    
    def create_dictionary(self, name, description):
        response = requests.post(
            f"{self.base_url}/api/v1/dictionaries/create",
            json={"name": name, "description": description},
            headers=self.headers
        )
        return response.json()
    
    def process_text(self, text, token_budget=1000):
        response = requests.post(
            f"{self.base_url}/api/v1/dictionaries/process",
            json={"text": text, "token_budget": token_budget},
            headers=self.headers
        )
        return response.json()
```

### JavaScript
```javascript
class ChatbookAPI {
  constructor(baseUrl, token) {
    this.baseUrl = baseUrl;
    this.headers = {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    };
  }
  
  async exportChatbook(name, contentTypes) {
    const response = await fetch(`${this.baseUrl}/api/v1/chatbooks/export`, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({
        name: name,
        content_types: contentTypes,
        async: true
      })
    });
    return response.json();
  }
  
  async getExportStatus(jobId) {
    const response = await fetch(
      `${this.baseUrl}/api/v1/chatbooks/export/jobs/${jobId}`,
      { headers: this.headers }
    );
    return response.json();
  }
}
```

## Migration Guide

For users migrating from the TUI application:

1. **Authentication**: All API calls now require authentication tokens
2. **User Isolation**: Content is automatically isolated per user
3. **Async Operations**: Large operations support async processing with job management
4. **Conflict Resolution**: Import operations provide multiple strategies for handling conflicts
5. **Rate Limiting**: API calls are subject to rate limits for stability

## Support

For API support and questions:
- Documentation: [https://docs.tldw-server.com/api](https://docs.tldw-server.com/api)
- Issues: [https://github.com/tldw-server/issues](https://github.com/tldw-server/issues)
- Discord: [https://discord.gg/tldw-server](https://discord.gg/tldw-server)