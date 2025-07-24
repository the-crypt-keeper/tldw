# Subscriptions Feature - Technical Architecture

## System Architecture Overview

The Subscriptions feature integrates seamlessly with the existing tldw_server architecture while adding new components for feed monitoring and content curation. The design follows established patterns and leverages existing infrastructure where possible.

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   FastAPI       │     │  Background      │     │   External      │
│   Endpoints     │────▶│  Task Manager    │────▶│   Services      │
└────────┬────────┘     └────────┬─────────┘     └─────────────────┘
         │                       │                         │
         ▼                       ▼                         ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Subscription   │     │   RSS/YouTube    │     │    yt-dlp       │
│   Service       │────▶│   Parsers        │────▶│  feedparser     │
└────────┬────────┘     └──────────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│   Database      │     │  Media Pipeline  │
│   (SQLite)      │────▶│  (Existing)      │
└─────────────────┘     └──────────────────┘
```

## Core Components

### 1. Database Layer

#### New Tables
- **Subscriptions**: Stores subscription metadata and configuration
- **SubscriptionItems**: Tracks discovered content items
- **SubscriptionChecks**: Logs check history and statistics
- **ImportRules**: Stores auto-import rules and filters

#### Integration with Existing Schema
- Links to Media table via foreign keys
- Follows UUID/versioning patterns
- Implements soft deletes and sync_log entries
- Uses same transaction patterns

### 2. Service Layer

#### SubscriptionService (`/app/core/Subscriptions/subscription_service.py`)
```python
class SubscriptionService:
    """Manages subscription CRUD operations and orchestration"""
    
    def __init__(self, db_path: str):
        self.db = MediaDatabase(db_path)
        self.parser_factory = ParserFactory()
    
    async def add_subscription(self, url: str, config: SubscriptionConfig) -> Subscription:
        """Add a new subscription with validation"""
        
    async def check_subscription(self, subscription_id: int) -> List[ContentItem]:
        """Check a subscription for new content"""
        
    async def import_items(self, item_ids: List[int]) -> List[int]:
        """Import selected items into media library"""
```

#### ParserFactory (`/app/core/Subscriptions/parsers/factory.py`)
```python
class ParserFactory:
    """Factory for creating appropriate parsers based on URL"""
    
    @staticmethod
    def create_parser(url: str) -> BaseParser:
        if 'youtube.com' in url or 'youtu.be' in url:
            return YouTubeParser()
        elif any(feed_indicator in url for feed_indicator in ['.rss', '.xml', 'feed']):
            return RSSParser()
        else:
            # Attempt to detect feed type
            return AutoDetectParser()
```

### 3. Parser Components

#### Base Parser Interface
```python
class BaseParser(ABC):
    """Abstract base class for content parsers"""
    
    @abstractmethod
    async def parse(self, url: str) -> ParseResult:
        """Parse content from URL"""
        
    @abstractmethod
    async def validate_url(self, url: str) -> bool:
        """Validate if URL is supported"""
```

#### YouTube Parser
- Leverages existing yt-dlp integration
- Supports channels, playlists, and user URLs
- Extracts video metadata efficiently
- Handles pagination for large channels

#### RSS Parser
- Uses feedparser library
- Supports RSS 2.0, Atom, and RDF formats
- Extracts standard feed metadata
- Handles various content encodings

### 4. Background Task System

#### Task Manager (`/app/services/subscription_scheduler.py`)
```python
class SubscriptionScheduler:
    """Manages periodic subscription checks"""
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.active_jobs = {}
    
    async def start(self):
        """Initialize and start the scheduler"""
        await self._load_active_subscriptions()
        self.scheduler.start()
    
    async def schedule_check(self, subscription: Subscription):
        """Schedule periodic checks for a subscription"""
        job = self.scheduler.add_job(
            self._check_subscription,
            'interval',
            seconds=subscription.check_interval,
            args=[subscription.id],
            id=f"subscription_{subscription.id}"
        )
        self.active_jobs[subscription.id] = job
```

#### Integration Options
1. **Option A: Standalone Service**
   - Separate process managed by systemd/supervisor
   - Communicates via database
   - Simple and reliable

2. **Option B: Embedded Service**
   - Runs within FastAPI application
   - Uses FastAPI's lifespan events
   - Easier deployment

3. **Option C: External Scheduler**
   - Use cron or external job scheduler
   - Calls API endpoints
   - Most flexible but complex

### 5. API Layer

#### Endpoint Structure
```
/api/v1/subscriptions/
├── GET    /                    # List subscriptions
├── POST   /                    # Create subscription
├── GET    /{id}                # Get subscription details
├── PUT    /{id}                # Update subscription
├── DELETE /{id}                # Delete subscription
├── POST   /{id}/check          # Manual check
├── GET    /{id}/items          # Get discovered items
├── POST   /{id}/items/import   # Import selected items
├── GET    /stats               # Subscription statistics
└── POST   /import-opml         # Import OPML file
```

#### Request/Response Schemas
```python
class SubscriptionCreate(BaseModel):
    url: str
    name: Optional[str] = None
    check_interval: int = 3600  # seconds
    auto_import: bool = False
    tags: List[str] = []
    import_rules: Optional[ImportRules] = None

class SubscriptionResponse(BaseModel):
    id: int
    uuid: str
    url: str
    name: str
    type: SubscriptionType
    check_interval: int
    last_checked: Optional[datetime]
    is_active: bool
    item_count: int
    unprocessed_count: int
```

### 6. Media Pipeline Integration

#### Content Import Flow
1. User selects items from watchlist
2. System creates import jobs
3. Items processed through existing pipeline:
   - URL validation
   - Media download (if applicable)
   - Metadata extraction
   - Transcription (if applicable)
   - Database storage
4. Original subscription item linked to media entry

#### Metadata Preservation
- Source subscription tracked
- Original publish date preserved
- Author/creator information maintained
- Feed-specific metadata stored in JSON

### 7. Authentication & Authorization

#### Single-User Mode (Current)
- All subscriptions global
- No user-specific limits
- Simple implementation

#### Multi-User Mode (Future)
- Subscriptions linked to user accounts
- Per-user quotas and limits
- Sharing capabilities
- Access control

### 8. Error Handling & Resilience

#### Failure Scenarios
1. **Feed Unavailable**
   - Exponential backoff
   - Mark subscription as failing
   - Notify user after threshold

2. **Parse Errors**
   - Log detailed errors
   - Skip problematic items
   - Continue processing

3. **Import Failures**
   - Retry mechanism
   - Quarantine failed items
   - Manual retry option

#### Rate Limiting
- Respect robots.txt
- Implement per-domain delays
- Use polite crawling practices
- Cache feed results

### 9. Performance Optimization

#### Caching Strategy
- Cache feed contents (15-minute TTL)
- Store parsed results
- Deduplicate requests
- Use ETags when supported

#### Database Optimization
- Indexes on frequently queried fields
- Batch operations where possible
- Connection pooling
- Query optimization

#### Concurrent Processing
- Async/await throughout
- Process multiple feeds concurrently
- Limit concurrent imports
- Queue management

## Integration Points

### Existing Components Used
1. **Media Processing Pipeline**
   - `process_videos()` for YouTube content
   - `process_url()` for web content
   - Existing chunking and embedding

2. **Database Layer**
   - MediaDatabase class
   - Transaction management
   - UUID generation
   - Sync logging

3. **Authentication**
   - Current API key system
   - Future JWT integration
   - User context

4. **Background Tasks**
   - Similar pattern to embedding workers
   - Reuse job management concepts
   - Status tracking

### New Dependencies
1. **feedparser** - RSS/Atom parsing
2. **apscheduler** - Task scheduling
3. **httpx** - Async HTTP client
4. **beautifulsoup4** - HTML parsing (optional)

## Security Considerations

### Input Validation
- Sanitize all URLs
- Validate feed content
- Prevent SSRF attacks
- Check content types

### Resource Limits
- Maximum feed size
- Request timeouts
- Memory limits
- CPU usage caps

### Data Privacy
- No external tracking
- Local storage only
- User-controlled data
- Secure API endpoints

## Monitoring & Logging

### Metrics to Track
- Subscription check frequency
- Success/failure rates
- Import statistics
- Performance metrics

### Logging Strategy
- Use loguru consistently
- Structured logging
- Appropriate log levels
- Rotation and retention

## Testing Strategy

### Unit Tests
- Parser implementations
- Service methods
- Database operations
- Utility functions

### Integration Tests
- API endpoints
- Full import flow
- Background tasks
- Error scenarios

### Performance Tests
- Large feed handling
- Concurrent operations
- Database performance
- Memory usage