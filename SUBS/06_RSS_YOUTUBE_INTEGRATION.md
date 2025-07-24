# RSS and YouTube Integration - Technical Details

## Overview

This document provides detailed technical specifications for integrating RSS feed parsing and YouTube channel/playlist monitoring into the Subscriptions feature. It covers parser implementations, edge cases, performance optimizations, and platform-specific considerations.

## RSS Feed Integration

### Supported Feed Formats

#### RSS 2.0
```xml
<?xml version="1.0" encoding="UTF-8" ?>
<rss version="2.0">
  <channel>
    <title>Example Blog</title>
    <link>https://example.com</link>
    <description>A blog about examples</description>
    <language>en-us</language>
    <pubDate>Mon, 20 Jan 2024 10:00:00 GMT</pubDate>
    <item>
      <title>Article Title</title>
      <link>https://example.com/article-1</link>
      <description>Article summary...</description>
      <author>john@example.com (John Doe)</author>
      <guid>https://example.com/article-1</guid>
      <pubDate>Mon, 20 Jan 2024 09:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
```

#### Atom 1.0
```xml
<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Example Blog</title>
  <link href="https://example.com/"/>
  <updated>2024-01-20T10:00:00Z</updated>
  <author>
    <name>John Doe</name>
  </author>
  <entry>
    <title>Article Title</title>
    <link href="https://example.com/article-1"/>
    <id>https://example.com/article-1</id>
    <updated>2024-01-20T09:00:00Z</updated>
    <summary>Article summary...</summary>
  </entry>
</feed>
```

#### Podcast RSS
```xml
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
  <channel>
    <title>Example Podcast</title>
    <itunes:author>John Doe</itunes:author>
    <item>
      <title>Episode 1</title>
      <enclosure url="https://example.com/episode1.mp3" length="12345678" type="audio/mpeg"/>
      <itunes:duration>45:30</itunes:duration>
    </item>
  </channel>
</rss>
```

### RSS Parser Implementation

```python
# Location: /app/core/Subscriptions/parsers/rss_parser.py

import feedparser
import httpx
from typing import List, Optional, Dict, Any
from datetime import datetime
import hashlib
from urllib.parse import urljoin
import email.utils
import time

class RSSParser(BaseParser):
    """Advanced RSS/Atom feed parser with robust error handling"""
    
    def __init__(self):
        self.supported_formats = ['rss', 'atom', 'rdf']
        self.timeout = 30  # seconds
        self.max_items = 100  # Limit items per feed
        
    async def validate_url(self, url: str) -> bool:
        """Check if URL points to a valid feed"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Use HEAD request first to check content type
                head_response = await client.head(url, follow_redirects=True)
                content_type = head_response.headers.get('content-type', '').lower()
                
                # Check for feed content types
                feed_types = ['application/rss+xml', 'application/atom+xml', 
                             'application/xml', 'text/xml']
                if any(ft in content_type for ft in feed_types):
                    return True
                
                # If content type is HTML, it might still be a feed
                # Do a GET request and check content
                if 'text/html' in content_type:
                    response = await client.get(url, follow_redirects=True)
                    content = response.text[:1000]  # Check first 1KB
                    return self._looks_like_feed(content)
                    
            return False
        except Exception:
            return False
    
    async def parse(self, url: str) -> List[ContentItem]:
        """Parse RSS/Atom feed and return list of content items"""
        try:
            # Fetch feed content
            feed_content = await self._fetch_feed(url)
            
            # Parse with feedparser
            feed = feedparser.parse(feed_content)
            
            # Check for parsing errors
            if feed.bozo and not self._is_acceptable_bozo(feed):
                raise ValueError(f"Feed parsing error: {feed.bozo_exception}")
            
            # Extract feed metadata
            feed_metadata = self._extract_feed_metadata(feed)
            
            # Parse entries
            items = []
            entries = feed.entries[:self.max_items]
            
            for entry in entries:
                item = self._parse_entry(entry, feed_metadata, url)
                if item:
                    items.append(item)
                    
            return items
            
        except Exception as e:
            raise ParseError(f"Failed to parse RSS feed: {str(e)}")
    
    async def _fetch_feed(self, url: str) -> str:
        """Fetch feed content with proper headers and error handling"""
        headers = {
            'User-Agent': 'tldw_server/1.0 (RSS Reader)',
            'Accept': 'application/rss+xml, application/atom+xml, application/xml, text/xml, */*'
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, headers=headers, follow_redirects=True)
            response.raise_for_status()
            
            # Handle different encodings
            content_type = response.headers.get('content-type', '')
            if 'charset=' in content_type:
                encoding = content_type.split('charset=')[-1].strip()
                return response.content.decode(encoding)
            else:
                return response.text
    
    def _parse_entry(self, entry: Dict, feed_metadata: Dict, feed_url: str) -> Optional[ContentItem]:
        """Parse a single feed entry into ContentItem"""
        try:
            # Extract URL (handle relative URLs)
            url = entry.get('link', '')
            if url and not url.startswith(('http://', 'https://')):
                url = urljoin(feed_url, url)
            
            # Extract title
            title = self._clean_text(entry.get('title', 'Untitled'))
            
            # Extract description/summary
            description = self._extract_description(entry)
            
            # Extract author
            author = self._extract_author(entry, feed_metadata)
            
            # Extract publish date
            published_date = self._extract_date(entry)
            
            # Generate unique ID
            guid = entry.get('id') or entry.get('guid') or self._generate_guid(url, title)
            
            # Extract additional metadata
            metadata = {
                'guid': guid,
                'categories': [tag.term for tag in entry.get('tags', [])],
                'enclosures': self._extract_enclosures(entry),
                'feed_title': feed_metadata.get('title'),
                'feed_url': feed_url
            }
            
            # Add podcast-specific metadata
            if 'itunes_duration' in entry:
                metadata['duration'] = self._parse_duration(entry.itunes_duration)
            
            return ContentItem(
                url=url,
                title=title,
                description=description,
                author=author,
                published_date=published_date,
                content_type=self._determine_content_type(entry),
                **metadata
            )
            
        except Exception as e:
            # Log error but continue parsing other entries
            logger.warning(f"Failed to parse entry: {e}")
            return None
    
    def _extract_description(self, entry: Dict) -> str:
        """Extract best available description from entry"""
        # Try different fields in order of preference
        for field in ['summary_detail', 'content', 'summary']:
            if field in entry:
                if field == 'content':
                    # content is usually a list
                    content = entry[field][0] if entry[field] else {}
                    return self._clean_text(content.get('value', ''))
                elif field == 'summary_detail':
                    return self._clean_text(entry[field].get('value', ''))
                else:
                    return self._clean_text(entry[field])
        return ''
    
    def _extract_author(self, entry: Dict, feed_metadata: Dict) -> Optional[str]:
        """Extract author information from entry or feed"""
        # Try entry-level author first
        if 'author_detail' in entry:
            return entry['author_detail'].get('name', '')
        elif 'author' in entry:
            return entry['author']
        
        # Fall back to feed-level author
        return feed_metadata.get('author')
    
    def _extract_date(self, entry: Dict) -> Optional[datetime]:
        """Extract and parse publication date"""
        date_fields = ['published_parsed', 'updated_parsed', 'created_parsed']
        
        for field in date_fields:
            if field in entry and entry[field]:
                try:
                    return datetime.fromtimestamp(time.mktime(entry[field]))
                except Exception:
                    continue
                    
        # Try parsing string dates
        date_strings = ['published', 'updated', 'created']
        for field in date_strings:
            if field in entry and entry[field]:
                try:
                    return datetime.fromtimestamp(
                        email.utils.mktime_tz(email.utils.parsedate_tz(entry[field]))
                    )
                except Exception:
                    continue
                    
        return None
    
    def _extract_enclosures(self, entry: Dict) -> List[Dict]:
        """Extract media enclosures (podcasts, videos, etc.)"""
        enclosures = []
        
        for enclosure in entry.get('enclosures', []):
            enc_data = {
                'url': enclosure.get('href', ''),
                'type': enclosure.get('type', ''),
                'length': enclosure.get('length', 0)
            }
            if enc_data['url']:
                enclosures.append(enc_data)
                
        return enclosures
    
    def _determine_content_type(self, entry: Dict) -> str:
        """Determine the type of content (article, podcast, video, etc.)"""
        # Check for enclosures
        enclosures = entry.get('enclosures', [])
        if enclosures:
            mime_type = enclosures[0].get('type', '').lower()
            if 'audio' in mime_type:
                return 'podcast'
            elif 'video' in mime_type:
                return 'video'
                
        # Check for video platforms in URL
        url = entry.get('link', '')
        if any(platform in url for platform in ['youtube.com', 'vimeo.com', 'dailymotion.com']):
            return 'video'
            
        return 'article'
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ''
            
        # Remove HTML tags if present
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _generate_guid(self, url: str, title: str) -> str:
        """Generate a unique ID for entries without GUID"""
        content = f"{url}:{title}"
        return hashlib.sha256(content.encode()).hexdigest()
```

### Feed Discovery

```python
# Location: /app/core/Subscriptions/parsers/feed_discovery.py

class FeedDiscovery:
    """Discover RSS feeds from web pages"""
    
    async def discover_feeds(self, url: str) -> List[Dict[str, str]]:
        """Find RSS feeds linked from a web page"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                
            soup = BeautifulSoup(response.text, 'html.parser')
            feeds = []
            
            # Look for <link> tags with RSS/Atom types
            for link in soup.find_all('link', type=re.compile('(rss|atom)')):
                feed_url = urljoin(url, link.get('href', ''))
                feeds.append({
                    'url': feed_url,
                    'title': link.get('title', 'RSS Feed'),
                    'type': link.get('type', 'application/rss+xml')
                })
                
            # Look for common feed URLs if none found
            if not feeds:
                common_paths = ['/feed', '/rss', '/atom', '/feed.xml', '/rss.xml']
                for path in common_paths:
                    test_url = urljoin(url, path)
                    if await self._test_feed_url(test_url):
                        feeds.append({
                            'url': test_url,
                            'title': f'Feed at {path}',
                            'type': 'application/rss+xml'
                        })
                        
            return feeds
            
        except Exception as e:
            logger.error(f"Feed discovery failed: {e}")
            return []
```

## YouTube Integration

### YouTube URL Patterns

```python
# Supported YouTube URL patterns
YOUTUBE_PATTERNS = {
    'channel_id': r'youtube\.com/channel/(UC[\w-]+)',
    'channel_custom': r'youtube\.com/c/([\w-]+)',
    'channel_user': r'youtube\.com/user/([\w-]+)',
    'channel_handle': r'youtube\.com/@([\w-]+)',
    'playlist': r'youtube\.com/playlist\?list=([\w-]+)',
    'video': r'youtube\.com/watch\?v=([\w-]+)',
    'short': r'youtu\.be/([\w-]+)'
}
```

### YouTube Parser Implementation

```python
# Location: /app/core/Subscriptions/parsers/youtube_parser.py

import yt_dlp
from typing import List, Dict, Optional
import re
from datetime import datetime, timedelta

class YouTubeParser(BaseParser):
    """YouTube channel and playlist parser using yt-dlp"""
    
    def __init__(self):
        self.ydl_opts = {
            'extract_flat': 'in_playlist',
            'skip_download': True,
            'quiet': True,
            'no_warnings': True,
            'ignoreerrors': True,
            'no_check_certificate': True,
            'prefer_insecure': True
        }
        self.max_items = 50  # Limit items to fetch
        
    async def validate_url(self, url: str) -> bool:
        """Check if URL is a valid YouTube channel or playlist"""
        # Check URL patterns
        for pattern in YOUTUBE_PATTERNS.values():
            if re.search(pattern, url):
                return True
        return False
    
    async def parse(self, url: str) -> List[ContentItem]:
        """Parse YouTube channel or playlist"""
        try:
            # Determine URL type and adjust options
            url_type = self._determine_url_type(url)
            
            if url_type == 'channel':
                return await self._parse_channel(url)
            elif url_type == 'playlist':
                return await self._parse_playlist(url)
            else:
                raise ValueError(f"Unsupported YouTube URL type: {url}")
                
        except Exception as e:
            raise ParseError(f"Failed to parse YouTube URL: {str(e)}")
    
    async def _parse_channel(self, url: str) -> List[ContentItem]:
        """Parse YouTube channel for recent videos"""
        # Convert channel URL to videos tab
        if '@' in url:
            # Handle new @username format
            url = f"{url}/videos"
        elif '/c/' in url or '/user/' in url:
            # Handle custom and user URLs
            url = f"{url}/videos"
        elif '/channel/' in url:
            # Handle channel ID URLs
            url = f"{url}/videos"
            
        # Set options for channel parsing
        opts = self.ydl_opts.copy()
        opts.update({
            'playlistend': self.max_items,
            'extract_flat': True
        })
        
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
        # Extract channel metadata
        channel_metadata = {
            'channel_id': info.get('channel_id'),
            'channel_name': info.get('uploader', info.get('channel')),
            'channel_url': info.get('channel_url', url),
            'subscriber_count': info.get('subscriber_count'),
            'description': info.get('description')
        }
        
        # Parse video entries
        items = []
        entries = info.get('entries', [])[:self.max_items]
        
        for entry in entries:
            item = self._parse_video_entry(entry, channel_metadata)
            if item:
                items.append(item)
                
        return items
    
    async def _parse_playlist(self, url: str) -> List[ContentItem]:
        """Parse YouTube playlist"""
        opts = self.ydl_opts.copy()
        opts['playlistend'] = self.max_items
        
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
        # Extract playlist metadata
        playlist_metadata = {
            'playlist_id': info.get('id'),
            'playlist_title': info.get('title'),
            'playlist_author': info.get('uploader'),
            'playlist_count': info.get('playlist_count')
        }
        
        # Parse video entries
        items = []
        entries = info.get('entries', [])[:self.max_items]
        
        for entry in entries:
            item = self._parse_video_entry(entry, playlist_metadata)
            if item:
                items.append(item)
                
        return items
    
    def _parse_video_entry(self, entry: Dict, metadata: Dict) -> Optional[ContentItem]:
        """Parse individual video entry"""
        try:
            # Skip private/deleted videos
            if entry.get('availability') == 'private':
                return None
                
            video_id = entry.get('id')
            if not video_id:
                return None
                
            # Build video URL
            url = f"https://www.youtube.com/watch?v={video_id}"
            
            # Extract video metadata
            title = entry.get('title', 'Untitled')
            description = entry.get('description', '')
            
            # Handle different uploader field names
            author = (entry.get('uploader') or 
                     entry.get('channel') or 
                     metadata.get('channel_name', ''))
            
            # Parse upload date
            published_date = self._parse_youtube_date(entry)
            
            # Extract duration
            duration = entry.get('duration')
            if duration and isinstance(duration, str):
                # Convert duration string to seconds
                duration = self._parse_duration_string(duration)
                
            # Build metadata
            video_metadata = {
                'video_id': video_id,
                'duration': duration,
                'view_count': entry.get('view_count'),
                'like_count': entry.get('like_count'),
                'comment_count': entry.get('comment_count'),
                'thumbnail_url': self._get_best_thumbnail(entry),
                'channel_id': entry.get('channel_id', metadata.get('channel_id')),
                'channel_url': entry.get('channel_url', metadata.get('channel_url')),
                'is_live': entry.get('live_status') == 'is_live',
                'age_limit': entry.get('age_limit', 0),
                **metadata
            }
            
            return ContentItem(
                url=url,
                title=title,
                description=description,
                author=author,
                published_date=published_date,
                content_type='video',
                estimated_size=self._estimate_video_size(entry),
                **video_metadata
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse video entry: {e}")
            return None
    
    def _parse_youtube_date(self, entry: Dict) -> Optional[datetime]:
        """Parse YouTube date formats"""
        # Try different date fields
        date_fields = ['upload_date', 'release_date', 'timestamp']
        
        for field in date_fields:
            if field in entry:
                if field == 'upload_date' and entry[field]:
                    # Format: YYYYMMDD
                    try:
                        return datetime.strptime(entry[field], '%Y%m%d')
                    except Exception:
                        continue
                elif field == 'timestamp' and entry[field]:
                    try:
                        return datetime.fromtimestamp(entry[field])
                    except Exception:
                        continue
                        
        # Try relative dates (e.g., "2 hours ago")
        if 'release_timestamp' in entry:
            try:
                return datetime.fromtimestamp(entry['release_timestamp'])
            except Exception:
                pass
                
        return None
    
    def _get_best_thumbnail(self, entry: Dict) -> Optional[str]:
        """Get highest quality thumbnail URL"""
        thumbnails = entry.get('thumbnails', [])
        if not thumbnails:
            # Fallback to video ID-based URL
            video_id = entry.get('id')
            if video_id:
                return f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
            return None
            
        # Sort by resolution and get highest
        sorted_thumbs = sorted(
            thumbnails,
            key=lambda x: x.get('width', 0) * x.get('height', 0),
            reverse=True
        )
        
        return sorted_thumbs[0].get('url') if sorted_thumbs else None
    
    def _estimate_video_size(self, entry: Dict) -> int:
        """Estimate video file size in bytes"""
        duration = entry.get('duration', 0)
        if not duration:
            return 0
            
        # Rough estimate: 1MB per minute at 720p
        # Adjust based on resolution if available
        base_rate = 1_000_000  # 1MB per minute
        
        # Check for resolution hints
        formats = entry.get('formats', [])
        if formats:
            # Look for height indicator
            max_height = max(f.get('height', 720) for f in formats if f.get('height'))
            if max_height >= 2160:  # 4K
                base_rate *= 4
            elif max_height >= 1440:  # 2K
                base_rate *= 2.5
            elif max_height >= 1080:  # Full HD
                base_rate *= 1.5
                
        return int((duration / 60) * base_rate)
    
    def _parse_duration_string(self, duration_str: str) -> int:
        """Convert duration string (HH:MM:SS) to seconds"""
        parts = duration_str.split(':')
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        elif len(parts) == 1:
            return int(parts[0])
        return 0
    
    def _determine_url_type(self, url: str) -> str:
        """Determine if URL is channel or playlist"""
        if '/playlist' in url or 'list=' in url:
            return 'playlist'
        return 'channel'
```

### YouTube API Optimization

```python
# Location: /app/core/Subscriptions/parsers/youtube_optimization.py

class YouTubeOptimizer:
    """Optimization strategies for YouTube parsing"""
    
    @staticmethod
    async def get_channel_rss_feed(channel_id: str) -> Optional[str]:
        """Get RSS feed URL for YouTube channel (faster than yt-dlp)"""
        # YouTube provides RSS feeds for channels
        return f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    
    @staticmethod
    async def extract_channel_id(url: str) -> Optional[str]:
        """Extract channel ID from various YouTube URL formats"""
        # Try direct channel ID
        match = re.search(r'/channel/(UC[\w-]+)', url)
        if match:
            return match.group(1)
            
        # For other formats, we need to resolve with yt-dlp
        opts = {
            'skip_download': True,
            'quiet': True,
            'extract_flat': True
        }
        
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return info.get('channel_id')
        except Exception:
            return None
    
    @staticmethod
    def should_use_rss_fallback(url: str) -> bool:
        """Determine if RSS feed is better option"""
        # RSS is faster but has limitations:
        # - Only shows last 15 videos
        # - Less metadata
        # - No playlist support
        return '/channel/' in url and 'videos' not in url
```

## Performance Optimizations

### Caching Strategy

```python
# Location: /app/core/Subscriptions/parsers/cache.py

from typing import Optional, Dict, Any
import hashlib
import json
from datetime import datetime, timedelta

class FeedCache:
    """Cache for feed parsing results"""
    
    def __init__(self, cache_dir: str = "./cache/feeds"):
        self.cache_dir = cache_dir
        self.default_ttl = timedelta(minutes=15)
        
    async def get(self, url: str) -> Optional[Dict[str, Any]]:
        """Get cached feed data if fresh"""
        cache_key = self._get_cache_key(url)
        cache_file = f"{self.cache_dir}/{cache_key}.json"
        
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                
            # Check if cache is fresh
            cached_at = datetime.fromisoformat(data['cached_at'])
            if datetime.now() - cached_at < self.default_ttl:
                return data['content']
                
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass
            
        return None
    
    async def set(self, url: str, content: Dict[str, Any]):
        """Cache feed data"""
        cache_key = self._get_cache_key(url)
        cache_file = f"{self.cache_dir}/{cache_key}.json"
        
        data = {
            'url': url,
            'cached_at': datetime.now().isoformat(),
            'content': content
        }
        
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(data, f)
    
    def _get_cache_key(self, url: str) -> str:
        """Generate cache key from URL"""
        return hashlib.md5(url.encode()).hexdigest()
```

### Concurrent Fetching

```python
# Location: /app/core/Subscriptions/parsers/concurrent.py

import asyncio
from typing import List, Dict, Tuple

class ConcurrentFeedFetcher:
    """Fetch multiple feeds concurrently with rate limiting"""
    
    def __init__(self, max_concurrent: int = 5, per_domain_delay: float = 1.0):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.per_domain_delay = per_domain_delay
        self.domain_locks = {}
        
    async def fetch_many(self, urls: List[str]) -> List[Tuple[str, Any]]:
        """Fetch multiple feeds concurrently"""
        tasks = [self._fetch_with_limit(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return list(zip(urls, results))
    
    async def _fetch_with_limit(self, url: str) -> Any:
        """Fetch single feed with rate limiting"""
        domain = urlparse(url).netloc
        
        # Get or create domain lock
        if domain not in self.domain_locks:
            self.domain_locks[domain] = asyncio.Lock()
            
        async with self.semaphore:
            async with self.domain_locks[domain]:
                try:
                    parser = ParserFactory.create_parser(url)
                    result = await parser.parse(url)
                    
                    # Delay before next request to same domain
                    await asyncio.sleep(self.per_domain_delay)
                    
                    return result
                except Exception as e:
                    return e
```

## Error Handling and Recovery

### Common Issues and Solutions

```python
# Location: /app/core/Subscriptions/parsers/error_handling.py

class FeedErrorHandler:
    """Handle common feed parsing errors"""
    
    @staticmethod
    def handle_parse_error(error: Exception, url: str) -> Dict[str, Any]:
        """Categorize and handle parsing errors"""
        error_type = type(error).__name__
        error_msg = str(error)
        
        if isinstance(error, httpx.HTTPStatusError):
            if error.response.status_code == 404:
                return {
                    'error': 'FEED_NOT_FOUND',
                    'message': 'Feed URL returns 404',
                    'recoverable': False
                }
            elif error.response.status_code == 403:
                return {
                    'error': 'ACCESS_DENIED',
                    'message': 'Access to feed denied',
                    'recoverable': False
                }
            elif error.response.status_code >= 500:
                return {
                    'error': 'SERVER_ERROR',
                    'message': 'Feed server error',
                    'recoverable': True
                }
                
        elif isinstance(error, httpx.TimeoutException):
            return {
                'error': 'TIMEOUT',
                'message': 'Feed request timed out',
                'recoverable': True
            }
            
        elif 'bozo_exception' in error_msg:
            return {
                'error': 'MALFORMED_FEED',
                'message': 'Feed XML is malformed',
                'recoverable': False,
                'suggestion': 'Try feed validation service'
            }
            
        return {
            'error': 'UNKNOWN_ERROR',
            'message': error_msg,
            'recoverable': True
        }
```

## Feed Validation

### Pre-import Validation

```python
# Location: /app/core/Subscriptions/parsers/validators.py

class FeedValidator:
    """Validate feeds before adding as subscription"""
    
    async def validate_feed(self, url: str) -> Dict[str, Any]:
        """Comprehensive feed validation"""
        results = {
            'url': url,
            'valid': False,
            'type': None,
            'issues': [],
            'metadata': {}
        }
        
        try:
            # Check URL accessibility
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.head(url, follow_redirects=True)
                
            if response.status_code != 200:
                results['issues'].append(f"HTTP {response.status_code}")
                return results
                
            # Detect feed type
            parser = ParserFactory.create_parser(url)
            if not parser:
                results['issues'].append("No suitable parser found")
                return results
                
            # Try parsing
            items = await parser.parse(url)
            
            if not items:
                results['issues'].append("No items found in feed")
            else:
                results['valid'] = True
                results['type'] = parser.__class__.__name__.replace('Parser', '').lower()
                results['metadata'] = {
                    'item_count': len(items),
                    'latest_item': items[0].published_date.isoformat() if items[0].published_date else None,
                    'sample_title': items[0].title
                }
                
        except Exception as e:
            results['issues'].append(str(e))
            
        return results
```

## Best Practices

### Rate Limiting and Politeness

1. **Respect robots.txt**
2. **Use appropriate User-Agent**
3. **Implement exponential backoff**
4. **Cache aggressively**
5. **Limit concurrent connections**

### Feed-Specific Considerations

1. **RSS/Atom**
   - Handle malformed XML gracefully
   - Support various date formats
   - Clean HTML in descriptions
   - Handle relative URLs

2. **YouTube**
   - Use RSS feeds when possible (faster)
   - Respect API quotas
   - Handle age-restricted content
   - Cache channel metadata

### Error Recovery

1. **Temporary Failures**
   - Retry with exponential backoff
   - Mark as temporary failure
   - Continue checking

2. **Permanent Failures**
   - After N consecutive failures, reduce check frequency
   - Notify user of issues
   - Provide diagnostic information

3. **Partial Failures**
   - Process items that parsed successfully
   - Log failed items for debugging
   - Continue normal operation