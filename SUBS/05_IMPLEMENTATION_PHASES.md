# Subscriptions Feature - Implementation Phases

## Overview

This document outlines a phased approach to implementing the Subscriptions feature, ensuring each phase delivers working functionality while building toward the complete system. The implementation is designed to minimize risk and allow for testing and feedback at each stage.

## Phase 1: Foundation (Week 1-2)

### Goals
- Establish database schema
- Create basic CRUD operations
- Set up project structure

### Tasks

#### 1.1 Database Setup
```python
# Location: /app/core/DB_Management/Subscriptions_DB.py

class SubscriptionsDatabase:
    """Manages all subscription-related database operations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Create subscription tables if they don't exist"""
        # Implementation of schema from 03_DATABASE_SCHEMA.md
```

**Checklist:**
- [ ] Create migration script for new tables
- [ ] Implement SubscriptionsDatabase class
- [ ] Add database initialization to startup
- [ ] Write unit tests for database operations
- [ ] Test migration and rollback procedures

#### 1.2 Basic Models
```python
# Location: /app/api/v1/schemas/subscription_schemas.py

from pydantic import BaseModel, HttpUrl, validator
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum

class SubscriptionType(str, Enum):
    YOUTUBE_CHANNEL = "youtube_channel"
    YOUTUBE_PLAYLIST = "youtube_playlist"
    RSS_FEED = "rss_feed"
    ATOM_FEED = "atom_feed"
    PODCAST = "podcast"
    OTHER = "other"

class SubscriptionCreate(BaseModel):
    url: HttpUrl
    name: Optional[str] = None
    check_interval: int = 3600
    auto_import: bool = False
    tags: List[str] = []
    
    @validator('check_interval')
    def validate_interval(cls, v):
        if v < 300:  # 5 minutes minimum
            raise ValueError('Check interval must be at least 300 seconds')
        return v
```

**Checklist:**
- [ ] Define all Pydantic models
- [ ] Add validation rules
- [ ] Create response schemas
- [ ] Document model fields
- [ ] Add example data

#### 1.3 Basic API Endpoints
```python
# Location: /app/api/v1/endpoints/subscriptions.py

from fastapi import APIRouter, Depends, HTTPException
from typing import List
from app.api.v1.schemas import subscription_schemas as schemas

router = APIRouter(prefix="/subscriptions", tags=["subscriptions"])

@router.post("/", response_model=schemas.SubscriptionResponse)
async def create_subscription(
    subscription: schemas.SubscriptionCreate,
    db: SubscriptionsDatabase = Depends(get_subscriptions_db)
):
    """Create a new subscription"""
    # Basic implementation
    
@router.get("/", response_model=List[schemas.SubscriptionResponse])
async def list_subscriptions(
    skip: int = 0,
    limit: int = 20,
    db: SubscriptionsDatabase = Depends(get_subscriptions_db)
):
    """List all subscriptions"""
    # Basic implementation
```

**Checklist:**
- [ ] Implement CRUD endpoints
- [ ] Add to main router
- [ ] Write API tests
- [ ] Update OpenAPI documentation
- [ ] Test with curl/Postman

### Deliverables
- Working database with subscription tables
- Basic API for creating and listing subscriptions
- Unit tests for all components
- Documentation updates

## Phase 2: Feed Parsing (Week 3-4)

### Goals
- Implement RSS/Atom feed parsing
- Add YouTube channel/playlist support
- Create unified parser interface

### Tasks

#### 2.1 Parser Infrastructure
```python
# Location: /app/core/Subscriptions/parsers/base.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from datetime import datetime

class ContentItem:
    """Represents a single content item from a feed"""
    def __init__(
        self,
        url: str,
        title: str,
        description: Optional[str] = None,
        author: Optional[str] = None,
        published_date: Optional[datetime] = None,
        **kwargs
    ):
        self.url = url
        self.title = title
        self.description = description
        self.author = author
        self.published_date = published_date
        self.metadata = kwargs

class BaseParser(ABC):
    """Abstract base class for content parsers"""
    
    @abstractmethod
    async def parse(self, url: str) -> List[ContentItem]:
        """Parse content from URL and return list of items"""
        pass
    
    @abstractmethod
    async def validate_url(self, url: str) -> bool:
        """Check if this parser can handle the given URL"""
        pass
```

**Checklist:**
- [ ] Define parser interfaces
- [ ] Create ContentItem class
- [ ] Set up parser factory
- [ ] Add error handling
- [ ] Write parser tests

#### 2.2 RSS/Atom Parser
```python
# Location: /app/core/Subscriptions/parsers/rss_parser.py

import feedparser
import httpx
from typing import List

class RSSParser(BaseParser):
    """Parser for RSS and Atom feeds"""
    
    async def parse(self, url: str) -> List[ContentItem]:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            
        feed = feedparser.parse(response.text)
        items = []
        
        for entry in feed.entries:
            item = ContentItem(
                url=entry.get('link', ''),
                title=entry.get('title', 'Untitled'),
                description=entry.get('summary', ''),
                author=entry.get('author', ''),
                published_date=self._parse_date(entry.get('published'))
            )
            items.append(item)
            
        return items
```

**Checklist:**
- [ ] Install feedparser dependency
- [ ] Implement RSS parsing
- [ ] Handle various feed formats
- [ ] Add feed validation
- [ ] Test with real feeds

#### 2.3 YouTube Parser
```python
# Location: /app/core/Subscriptions/parsers/youtube_parser.py

import yt_dlp
from typing import List

class YouTubeParser(BaseParser):
    """Parser for YouTube channels and playlists"""
    
    def __init__(self):
        self.ydl_opts = {
            'extract_flat': True,
            'skip_download': True,
            'quiet': True,
            'no_warnings': True
        }
    
    async def parse(self, url: str) -> List[ContentItem]:
        with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
        items = []
        entries = info.get('entries', [])
        
        for entry in entries[:50]:  # Limit to recent 50
            item = ContentItem(
                url=f"https://youtube.com/watch?v={entry['id']}",
                title=entry.get('title', 'Untitled'),
                description=entry.get('description', ''),
                author=entry.get('uploader', ''),
                published_date=self._parse_date(entry.get('upload_date')),
                duration=entry.get('duration'),
                view_count=entry.get('view_count')
            )
            items.append(item)
            
        return items
```

**Checklist:**
- [ ] Leverage existing yt-dlp integration
- [ ] Support channel URLs
- [ ] Support playlist URLs
- [ ] Handle large channels efficiently
- [ ] Test with various YouTube URLs

### Deliverables
- Working RSS/Atom parser
- Working YouTube parser
- Parser factory and interface
- Integration tests
- Support for feed validation

## Phase 3: Subscription Checking (Week 5-6)

### Goals
- Implement subscription checking logic
- Create background task system
- Add check history tracking

### Tasks

#### 3.1 Subscription Service
```python
# Location: /app/core/Subscriptions/subscription_service.py

class SubscriptionService:
    """Core service for subscription operations"""
    
    def __init__(self, db: SubscriptionsDatabase):
        self.db = db
        self.parser_factory = ParserFactory()
        
    async def check_subscription(self, subscription_id: int) -> CheckResult:
        """Check a subscription for new content"""
        subscription = await self.db.get_subscription(subscription_id)
        if not subscription:
            raise SubscriptionNotFoundError(subscription_id)
            
        # Create check record
        check_id = await self.db.create_check(subscription_id)
        
        try:
            # Get appropriate parser
            parser = self.parser_factory.get_parser(subscription.url)
            
            # Parse content
            items = await parser.parse(subscription.url)
            
            # Process items
            new_items = await self._process_items(subscription_id, items)
            
            # Update check record
            await self.db.complete_check(check_id, len(items), len(new_items))
            
            return CheckResult(
                subscription_id=subscription_id,
                total_items=len(items),
                new_items=len(new_items),
                status='success'
            )
            
        except Exception as e:
            await self.db.fail_check(check_id, str(e))
            raise
```

**Checklist:**
- [ ] Implement check logic
- [ ] Add duplicate detection
- [ ] Create check history
- [ ] Handle failures gracefully
- [ ] Add performance metrics

#### 3.2 Background Task Manager
```python
# Location: /app/services/subscription_scheduler.py

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

class SubscriptionScheduler:
    """Manages periodic subscription checks"""
    
    def __init__(self, subscription_service: SubscriptionService):
        self.service = subscription_service
        self.scheduler = AsyncIOScheduler()
        self.jobs = {}
        
    async def start(self):
        """Start the scheduler"""
        # Load active subscriptions
        subscriptions = await self.service.get_active_subscriptions()
        
        for sub in subscriptions:
            await self.schedule_subscription(sub)
            
        self.scheduler.start()
        
    async def schedule_subscription(self, subscription):
        """Schedule checks for a subscription"""
        job_id = f"sub_{subscription.id}"
        
        # Remove existing job if any
        if job_id in self.jobs:
            self.scheduler.remove_job(job_id)
            
        # Create new job
        trigger = IntervalTrigger(seconds=subscription.check_interval)
        job = self.scheduler.add_job(
            self._check_subscription,
            trigger,
            id=job_id,
            args=[subscription.id],
            max_instances=1
        )
        self.jobs[job_id] = job
```

**Checklist:**
- [ ] Choose scheduler implementation
- [ ] Implement job management
- [ ] Add startup/shutdown hooks
- [ ] Handle job failures
- [ ] Add monitoring/logging

### Deliverables
- Working subscription checking
- Background task system
- Check history in database
- Manual check endpoint
- Performance metrics

## Phase 4: Content Import (Week 7-8)

### Goals
- Integrate with media processing pipeline
- Implement selective import
- Add import rules

### Tasks

#### 4.1 Import Integration
```python
# Location: /app/core/Subscriptions/import_service.py

class ImportService:
    """Handles importing subscription items to media library"""
    
    def __init__(self, media_processor):
        self.media_processor = media_processor
        
    async def import_items(self, item_ids: List[int], options: ImportOptions):
        """Import selected items into media library"""
        results = []
        
        for item_id in item_ids:
            try:
                # Get item details
                item = await self.db.get_subscription_item(item_id)
                
                # Process through media pipeline
                media_id = await self.media_processor.process_url(
                    url=item.url,
                    title=item.title,
                    metadata={
                        'source': 'subscription',
                        'subscription_id': item.subscription_id,
                        'published_date': item.published_date
                    },
                    options=options
                )
                
                # Update item status
                await self.db.mark_item_imported(item_id, media_id)
                
                results.append({
                    'item_id': item_id,
                    'media_id': media_id,
                    'status': 'success'
                })
                
            except Exception as e:
                results.append({
                    'item_id': item_id,
                    'status': 'failed',
                    'error': str(e)
                })
                
        return results
```

**Checklist:**
- [ ] Connect to media pipeline
- [ ] Preserve metadata
- [ ] Handle various content types
- [ ] Implement batch imports
- [ ] Add progress tracking

#### 4.2 Import Rules Engine
```python
# Location: /app/core/Subscriptions/rules_engine.py

class RulesEngine:
    """Evaluates import rules against content items"""
    
    def evaluate_item(self, item: ContentItem, rules: List[ImportRule]) -> bool:
        """Check if item matches any import rules"""
        for rule in rules:
            if rule.type == 'keyword':
                if self._check_keywords(item, rule.value):
                    return rule.action == 'import'
                    
            elif rule.type == 'author':
                if item.author == rule.value.get('author'):
                    return rule.action == 'import'
                    
            elif rule.type == 'date_range':
                if self._check_date_range(item, rule.value):
                    return rule.action == 'import'
                    
        return False  # No rules matched
```

**Checklist:**
- [ ] Design rule types
- [ ] Implement rule evaluation
- [ ] Add rule management API
- [ ] Test rule combinations
- [ ] Add rule statistics

### Deliverables
- Working import functionality
- Integration with media pipeline
- Import rules system
- Batch import support
- Import statistics

## Phase 5: UI and Polish (Week 9-10)

### Goals
- Create user interface mockups
- Implement watchlist view
- Add statistics and monitoring
- Polish and optimize

### Tasks

#### 5.1 API Enhancements
- [ ] Add filtering and search
- [ ] Implement pagination
- [ ] Add sorting options
- [ ] Create aggregate endpoints
- [ ] Optimize query performance

#### 5.2 Monitoring and Statistics
- [ ] Create statistics endpoints
- [ ] Add performance metrics
- [ ] Implement health checks
- [ ] Create admin dashboard data
- [ ] Add usage analytics

#### 5.3 Import/Export
- [ ] Implement OPML export
- [ ] Implement OPML import
- [ ] Add JSON export format
- [ ] Support bulk operations
- [ ] Add backup/restore

#### 5.4 Documentation
- [ ] Update API documentation
- [ ] Create user guides
- [ ] Add configuration guide
- [ ] Write troubleshooting guide
- [ ] Create video tutorials

### Deliverables
- Complete API implementation
- Statistics and monitoring
- Import/export functionality
- Comprehensive documentation
- Performance optimizations

## Phase 6: Testing and Deployment (Week 11-12)

### Goals
- Comprehensive testing
- Performance optimization
- Deployment preparation
- User acceptance testing

### Tasks

#### 6.1 Testing Suite
- [ ] Unit tests (>80% coverage)
- [ ] Integration tests
- [ ] Performance tests
- [ ] Load tests
- [ ] Security tests

#### 6.2 Optimization
- [ ] Database query optimization
- [ ] Caching implementation
- [ ] Connection pooling
- [ ] Memory usage optimization
- [ ] API response time optimization

#### 6.3 Deployment
- [ ] Create deployment guide
- [ ] Update Docker configuration
- [ ] Create migration scripts
- [ ] Set up monitoring
- [ ] Create backup procedures

### Deliverables
- Complete test suite
- Performance benchmarks
- Deployment documentation
- Migration tools
- Monitoring setup

## Risk Mitigation

### Technical Risks
1. **Feed Parsing Complexity**
   - Mitigation: Use well-tested libraries
   - Fallback: Limit initial format support

2. **Performance at Scale**
   - Mitigation: Design for async operations
   - Fallback: Implement rate limiting

3. **External API Limits**
   - Mitigation: Implement caching and quotas
   - Fallback: User-configurable limits

### Implementation Risks
1. **Scope Creep**
   - Mitigation: Strict phase boundaries
   - Fallback: Defer features to future releases

2. **Integration Complexity**
   - Mitigation: Reuse existing components
   - Fallback: Simplified integration points

## Success Criteria

### Phase 1
- [ ] Database migrations work correctly
- [ ] Basic CRUD operations functional
- [ ] All tests passing

### Phase 2
- [ ] Can parse RSS feeds successfully
- [ ] Can parse YouTube channels
- [ ] Parser tests cover edge cases

### Phase 3
- [ ] Subscriptions check automatically
- [ ] Check history is recorded
- [ ] Manual checks work

### Phase 4
- [ ] Items import to media library
- [ ] Metadata is preserved
- [ ] Import rules work correctly

### Phase 5
- [ ] API is feature-complete
- [ ] Documentation is comprehensive
- [ ] Performance meets targets

### Phase 6
- [ ] All tests passing
- [ ] Deployment successful
- [ ] User acceptance complete