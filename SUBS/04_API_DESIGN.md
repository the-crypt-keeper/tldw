# Subscriptions Feature - API Design

## API Overview

The Subscriptions API follows the RESTful design patterns established in tldw_server, providing comprehensive endpoints for managing content subscriptions, reviewing discovered items, and controlling the import process.

## Base Path

All subscription endpoints are prefixed with: `/api/v1/subscriptions`

## Authentication

- **Single-User Mode**: API key authentication via `X-API-KEY` header
- **Multi-User Mode** (future): JWT Bearer token authentication
- All endpoints require authentication

## API Endpoints

### 1. Subscription Management

#### Create Subscription
```http
POST /api/v1/subscriptions
```

**Request Body:**
```json
{
  "url": "https://www.youtube.com/@channel-name",
  "name": "Optional custom name",
  "check_interval": 3600,
  "auto_import": false,
  "tags": ["tech", "tutorials"],
  "import_rules": {
    "keywords": ["python", "fastapi"],
    "min_duration": 300,
    "max_age_days": 30
  }
}
```

**Response:** `201 Created`
```json
{
  "id": 1,
  "uuid": "550e8400-e29b-41d4-a716-446655440000",
  "url": "https://www.youtube.com/@channel-name",
  "name": "Channel Name",
  "type": "youtube_channel",
  "check_interval": 3600,
  "auto_import": false,
  "is_active": true,
  "last_checked": null,
  "created_at": "2024-01-20T10:00:00Z",
  "metadata": {
    "channel_id": "UC...",
    "subscriber_count": 150000,
    "video_count": 523,
    "description": "Channel description"
  }
}
```

#### List Subscriptions
```http
GET /api/v1/subscriptions
```

**Query Parameters:**
- `type`: Filter by subscription type (youtube_channel, rss_feed, etc.)
- `is_active`: Filter by active status (true/false)
- `tag`: Filter by tag name
- `search`: Search in name/description
- `page`: Page number (default: 1)
- `limit`: Items per page (default: 20, max: 100)
- `sort`: Sort field (name, created_at, last_checked)
- `order`: Sort order (asc, desc)

**Response:** `200 OK`
```json
{
  "items": [
    {
      "id": 1,
      "uuid": "550e8400-e29b-41d4-a716-446655440000",
      "url": "https://example.com/feed.rss",
      "name": "Example Blog",
      "type": "rss_feed",
      "check_interval": 3600,
      "is_active": true,
      "last_checked": "2024-01-20T09:00:00Z",
      "stats": {
        "total_items": 245,
        "imported_items": 23,
        "new_items": 5
      }
    }
  ],
  "total": 15,
  "page": 1,
  "pages": 1,
  "limit": 20
}
```

#### Get Subscription Details
```http
GET /api/v1/subscriptions/{subscription_id}
```

**Response:** `200 OK`
```json
{
  "id": 1,
  "uuid": "550e8400-e29b-41d4-a716-446655440000",
  "url": "https://example.com/feed.rss",
  "name": "Example Blog",
  "type": "rss_feed",
  "check_interval": 3600,
  "auto_import": false,
  "is_active": true,
  "last_checked": "2024-01-20T09:00:00Z",
  "last_successful_check": "2024-01-20T09:00:00Z",
  "consecutive_failures": 0,
  "created_at": "2024-01-15T10:00:00Z",
  "updated_at": "2024-01-20T09:00:00Z",
  "metadata": {
    "feed_title": "Example Blog",
    "feed_description": "A blog about examples",
    "feed_link": "https://example.com",
    "feed_language": "en",
    "feed_updated": "2024-01-20T08:45:00Z"
  },
  "stats": {
    "total_items_found": 245,
    "total_items_imported": 23,
    "new_items": 5,
    "recent_imports": 3
  },
  "tags": ["tech", "examples"],
  "import_rules": {
    "keywords": ["important", "tutorial"],
    "min_length": 1000
  }
}
```

#### Update Subscription
```http
PUT /api/v1/subscriptions/{subscription_id}
```

**Request Body:**
```json
{
  "name": "Updated Name",
  "check_interval": 7200,
  "auto_import": true,
  "is_active": true,
  "tags": ["tech", "ai"],
  "import_rules": {
    "keywords": ["machine learning", "ai"]
  }
}
```

**Response:** `200 OK`
```json
{
  "id": 1,
  "message": "Subscription updated successfully",
  "updated_fields": ["name", "check_interval", "auto_import", "tags", "import_rules"]
}
```

#### Delete Subscription
```http
DELETE /api/v1/subscriptions/{subscription_id}
```

**Query Parameters:**
- `delete_items`: Also delete discovered items (default: false)
- `delete_imported_media`: Also delete imported media (default: false)

**Response:** `200 OK`
```json
{
  "message": "Subscription deleted successfully",
  "items_deleted": 45,
  "media_deleted": 0
}
```

### 2. Subscription Operations

#### Check Subscription Now
```http
POST /api/v1/subscriptions/{subscription_id}/check
```

**Request Body (optional):**
```json
{
  "full_check": true,
  "import_new_items": false
}
```

**Response:** `202 Accepted`
```json
{
  "message": "Check initiated",
  "check_id": "550e8400-e29b-41d4-a716-446655440001",
  "estimated_duration": 30
}
```

#### Get Check Status
```http
GET /api/v1/subscriptions/{subscription_id}/checks/{check_id}
```

**Response:** `200 OK`
```json
{
  "check_id": "550e8400-e29b-41d4-a716-446655440001",
  "subscription_id": 1,
  "status": "success",
  "started_at": "2024-01-20T10:00:00Z",
  "completed_at": "2024-01-20T10:00:30Z",
  "duration_ms": 30000,
  "items_found": 15,
  "new_items": 3,
  "errors": []
}
```

#### Get Check History
```http
GET /api/v1/subscriptions/{subscription_id}/checks
```

**Query Parameters:**
- `limit`: Number of checks to return (default: 10)
- `status`: Filter by status (success, failed, partial)

**Response:** `200 OK`
```json
{
  "checks": [
    {
      "check_id": "550e8400-e29b-41d4-a716-446655440001",
      "started_at": "2024-01-20T10:00:00Z",
      "status": "success",
      "items_found": 15,
      "new_items": 3,
      "duration_ms": 30000
    }
  ],
  "total": 50
}
```

### 3. Content Discovery & Import

#### Get Discovered Items
```http
GET /api/v1/subscriptions/{subscription_id}/items
```

**Query Parameters:**
- `status`: Filter by status (new, reviewed, imported, failed, skipped)
- `date_from`: Filter by discovered date
- `date_to`: Filter by discovered date
- `search`: Search in title/description
- `page`: Page number
- `limit`: Items per page

**Response:** `200 OK`
```json
{
  "items": [
    {
      "id": 101,
      "uuid": "650e8400-e29b-41d4-a716-446655440002",
      "subscription_id": 1,
      "url": "https://example.com/article-1",
      "title": "Interesting Article Title",
      "description": "Article summary...",
      "author": "John Doe",
      "published_date": "2024-01-19T15:00:00Z",
      "status": "new",
      "discovered_at": "2024-01-20T10:00:00Z",
      "content_type": "article",
      "thumbnail_url": "https://example.com/thumb.jpg",
      "metadata": {
        "categories": ["Tech", "AI"],
        "read_time": "5 min"
      }
    }
  ],
  "total": 25,
  "page": 1,
  "pages": 2,
  "limit": 20
}
```

#### Get Watchlist (All New Items)
```http
GET /api/v1/subscriptions/watchlist
```

**Query Parameters:**
- `subscription_ids`: Comma-separated list of subscription IDs
- `tags`: Filter by subscription tags
- `date_from`: Filter by discovered date
- `date_to`: Filter by discovered date
- `content_type`: Filter by content type
- `sort`: Sort by (discovered_at, published_date, title)
- `page`: Page number
- `limit`: Items per page

**Response:** `200 OK`
```json
{
  "items": [
    {
      "id": 101,
      "subscription": {
        "id": 1,
        "name": "Example Blog",
        "type": "rss_feed"
      },
      "url": "https://example.com/article-1",
      "title": "Interesting Article Title",
      "description": "Article summary...",
      "author": "John Doe",
      "published_date": "2024-01-19T15:00:00Z",
      "discovered_at": "2024-01-20T10:00:00Z",
      "content_type": "article",
      "thumbnail_url": "https://example.com/thumb.jpg",
      "matches_rules": ["keyword:ai"]
    }
  ],
  "total": 45,
  "stats": {
    "by_subscription": {
      "1": 15,
      "2": 30
    },
    "by_content_type": {
      "article": 20,
      "video": 25
    }
  }
}
```

#### Import Items
```http
POST /api/v1/subscriptions/items/import
```

**Request Body:**
```json
{
  "item_ids": [101, 102, 103],
  "tags": ["imported", "to-review"],
  "process_options": {
    "transcribe": true,
    "chunk": true,
    "embed": true
  }
}
```

**Response:** `202 Accepted`
```json
{
  "message": "Import initiated",
  "import_job_id": "750e8400-e29b-41d4-a716-446655440003",
  "items_queued": 3,
  "estimated_duration": 180
}
```

#### Update Item Status
```http
PATCH /api/v1/subscriptions/items/{item_id}
```

**Request Body:**
```json
{
  "status": "reviewed",
  "notes": "Looks interesting, will import later"
}
```

**Response:** `200 OK`
```json
{
  "id": 101,
  "status": "reviewed",
  "updated_at": "2024-01-20T11:00:00Z"
}
```

#### Bulk Update Items
```http
POST /api/v1/subscriptions/items/bulk-update
```

**Request Body:**
```json
{
  "item_ids": [101, 102, 103],
  "updates": {
    "status": "skipped",
    "reason": "Not relevant"
  }
}
```

**Response:** `200 OK`
```json
{
  "updated": 3,
  "failed": 0
}
```

### 4. Import Rules

#### Create Import Rule
```http
POST /api/v1/subscriptions/{subscription_id}/rules
```

**Request Body:**
```json
{
  "name": "Import AI content",
  "type": "keyword",
  "value": {
    "keywords": ["artificial intelligence", "machine learning", "ai"],
    "match_type": "any"
  },
  "action": "import",
  "priority": 100,
  "is_active": true
}
```

**Response:** `201 Created`
```json
{
  "id": 1,
  "uuid": "850e8400-e29b-41d4-a716-446655440004",
  "subscription_id": 1,
  "name": "Import AI content",
  "type": "keyword",
  "created_at": "2024-01-20T11:00:00Z"
}
```

#### List Import Rules
```http
GET /api/v1/subscriptions/{subscription_id}/rules
```

**Response:** `200 OK`
```json
{
  "rules": [
    {
      "id": 1,
      "name": "Import AI content",
      "type": "keyword",
      "value": {
        "keywords": ["ai", "machine learning"],
        "match_type": "any"
      },
      "action": "import",
      "priority": 100,
      "is_active": true,
      "stats": {
        "times_matched": 45,
        "last_matched": "2024-01-20T10:00:00Z"
      }
    }
  ],
  "total": 3
}
```

### 5. Statistics & Monitoring

#### Get Subscription Statistics
```http
GET /api/v1/subscriptions/stats
```

**Response:** `200 OK`
```json
{
  "summary": {
    "total_subscriptions": 15,
    "active_subscriptions": 12,
    "total_items_discovered": 1543,
    "total_items_imported": 234,
    "pending_items": 45
  },
  "by_type": {
    "youtube_channel": {
      "count": 5,
      "items_discovered": 800,
      "items_imported": 150
    },
    "rss_feed": {
      "count": 10,
      "items_discovered": 743,
      "items_imported": 84
    }
  },
  "recent_activity": {
    "last_24h": {
      "checks_performed": 48,
      "items_discovered": 23,
      "items_imported": 5
    },
    "last_7d": {
      "checks_performed": 336,
      "items_discovered": 156,
      "items_imported": 34
    }
  }
}
```

#### Get System Status
```http
GET /api/v1/subscriptions/system/status
```

**Response:** `200 OK`
```json
{
  "scheduler": {
    "status": "running",
    "active_jobs": 12,
    "next_check": "2024-01-20T11:15:00Z"
  },
  "queue": {
    "pending_checks": 3,
    "pending_imports": 5,
    "failed_items": 2
  },
  "performance": {
    "average_check_duration_ms": 2500,
    "average_import_duration_ms": 8000,
    "checks_per_hour": 48
  }
}
```

### 6. Import/Export

#### Export Subscriptions (OPML)
```http
GET /api/v1/subscriptions/export
```

**Query Parameters:**
- `format`: Export format (opml, json, csv)
- `include_rules`: Include import rules (default: true)

**Response:** `200 OK`
```xml
<?xml version="1.0" encoding="UTF-8"?>
<opml version="2.0">
  <head>
    <title>tldw_server Subscriptions</title>
    <dateCreated>2024-01-20T11:00:00Z</dateCreated>
  </head>
  <body>
    <outline text="Tech Blogs" title="Tech Blogs">
      <outline type="rss" text="Example Blog" title="Example Blog" 
               xmlUrl="https://example.com/feed.rss" htmlUrl="https://example.com"/>
    </outline>
    <outline text="YouTube Channels" title="YouTube Channels">
      <outline type="youtube" text="Channel Name" title="Channel Name"
               xmlUrl="https://www.youtube.com/@channel-name"/>
    </outline>
  </body>
</opml>
```

#### Import Subscriptions (OPML)
```http
POST /api/v1/subscriptions/import
```

**Request Body (multipart/form-data):**
- `file`: OPML file
- `check_interval`: Default check interval for imported subscriptions
- `auto_import`: Default auto-import setting
- `tags`: Tags to apply to all imported subscriptions

**Response:** `201 Created`
```json
{
  "message": "Import completed",
  "imported": 15,
  "failed": 2,
  "errors": [
    {
      "url": "https://invalid.com/feed",
      "error": "Invalid feed URL"
    }
  ]
}
```

## Error Responses

All endpoints follow consistent error response format:

```json
{
  "error": {
    "code": "SUBSCRIPTION_NOT_FOUND",
    "message": "Subscription with ID 123 not found",
    "details": {
      "subscription_id": 123
    }
  }
}
```

### Common Error Codes

- `INVALID_URL`: The provided URL is not valid
- `UNSUPPORTED_FEED_TYPE`: Feed type not supported
- `SUBSCRIPTION_NOT_FOUND`: Subscription doesn't exist
- `ITEM_NOT_FOUND`: Item doesn't exist
- `IMPORT_FAILED`: Import operation failed
- `QUOTA_EXCEEDED`: User quota exceeded
- `RATE_LIMIT_EXCEEDED`: API rate limit exceeded
- `VALIDATION_ERROR`: Request validation failed

## Rate Limiting

- Default: 100 requests per minute per API key
- Bulk operations count as multiple requests
- Check operations limited to 10 per hour per subscription

## Webhooks (Future)

```json
{
  "event": "items.discovered",
  "subscription_id": 1,
  "timestamp": "2024-01-20T11:00:00Z",
  "data": {
    "new_items": 5,
    "total_items": 250
  }
}
```

## API Versioning

- Current version: v1
- Version included in URL path
- Deprecation notices via headers
- Backward compatibility for 6 months