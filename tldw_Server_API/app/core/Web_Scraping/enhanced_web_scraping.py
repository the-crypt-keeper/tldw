# enhanced_web_scraping.py - Production Web Scraping Pipeline
"""
Enhanced web scraping pipeline with production features:
- Concurrent scraping with rate limiting
- Job queue management with priority
- Cookie/session management
- Progress tracking and resumability
- Content deduplication
- Robust error handling and retries
- Support for multiple scraping strategies
"""

import asyncio
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import pickle
from collections import deque, defaultdict
from urllib.parse import urlparse, urljoin
import random

from loguru import logger
import aiohttp
import aiofiles
from bs4 import BeautifulSoup
import trafilatura
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
import redis
from ratelimit import limits, sleep_and_retry

# Import existing components
from tldw_Server_API.app.core.DB_Management.DB_Manager import add_media_with_keywords
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze
from tldw_Server_API.app.core.config import load_and_log_configs


class JobStatus(Enum):
    """Job status enumeration"""
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class JobPriority(Enum):
    """Job priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ScrapingJob:
    """Represents a scraping job"""
    job_id: str
    url: str
    method: str
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retries: int = 0
    max_retries: int = 3
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary"""
        return {
            "job_id": self.job_id,
            "url": self.url,
            "method": self.method,
            "priority": self.priority.value,
            "status": self.status.name,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retries": self.retries,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata
        }


class RateLimiter:
    """Rate limiting for scraping requests"""
    
    def __init__(self, max_requests_per_second: float = 2.0, 
                 max_requests_per_minute: int = 60,
                 max_requests_per_hour: int = 1000):
        self.max_rps = max_requests_per_second
        self.max_rpm = max_requests_per_minute
        self.max_rph = max_requests_per_hour
        self._request_times: deque = deque(maxlen=max_requests_per_hour)
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire permission to make a request"""
        async with self._lock:
            now = time.time()
            
            # Clean old request times
            cutoff_hour = now - 3600
            while self._request_times and self._request_times[0] < cutoff_hour:
                self._request_times.popleft()
            
            # Check hourly limit
            if len(self._request_times) >= self.max_rph:
                wait_time = 3600 - (now - self._request_times[0])
                if wait_time > 0:
                    logger.info(f"Rate limit: waiting {wait_time:.1f}s for hourly limit")
                    await asyncio.sleep(wait_time)
            
            # Check minute limit
            minute_ago = now - 60
            recent_minute = sum(1 for t in self._request_times if t > minute_ago)
            if recent_minute >= self.max_rpm:
                wait_time = 60 - (now - minute_ago)
                if wait_time > 0:
                    logger.info(f"Rate limit: waiting {wait_time:.1f}s for minute limit")
                    await asyncio.sleep(wait_time)
            
            # Check second limit
            if self._request_times and (now - self._request_times[-1]) < (1.0 / self.max_rps):
                wait_time = (1.0 / self.max_rps) - (now - self._request_times[-1])
                await asyncio.sleep(wait_time)
            
            # Record request time
            self._request_times.append(now)


class CookieManager:
    """Manages cookies and sessions for scraping"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./scraping_cookies.json")
        self._cookies: Dict[str, List[Dict[str, Any]]] = {}
        self._sessions: Dict[str, aiohttp.ClientSession] = {}
        self._load_cookies()
    
    def _load_cookies(self):
        """Load cookies from storage"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    self._cookies = json.load(f)
                logger.info(f"Loaded cookies for {len(self._cookies)} domains")
            except Exception as e:
                logger.error(f"Failed to load cookies: {e}")
    
    def _save_cookies(self):
        """Save cookies to storage"""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self._cookies, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cookies: {e}")
    
    def add_cookies(self, domain: str, cookies: List[Dict[str, Any]]):
        """Add cookies for a domain"""
        self._cookies[domain] = cookies
        self._save_cookies()
    
    def get_cookies(self, url: str) -> Optional[List[Dict[str, Any]]]:
        """Get cookies for a URL"""
        domain = urlparse(url).netloc
        return self._cookies.get(domain)
    
    async def get_session(self, url: str) -> aiohttp.ClientSession:
        """Get or create session for URL"""
        domain = urlparse(url).netloc
        
        if domain not in self._sessions:
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=2)
            timeout = aiohttp.ClientTimeout(total=30)
            self._sessions[domain] = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
            )
            
            # Apply cookies if available
            cookies = self.get_cookies(url)
            if cookies:
                for cookie in cookies:
                    self._sessions[domain].cookie_jar.update_cookies(cookie)
        
        return self._sessions[domain]
    
    async def close_all(self):
        """Close all sessions"""
        for session in self._sessions.values():
            await session.close()
        self._sessions.clear()


class ContentDeduplicator:
    """Handles content deduplication"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./content_hashes.db")
        self._hashes: Dict[str, Dict[str, Any]] = {}
        self._load_hashes()
    
    def _load_hashes(self):
        """Load content hashes from storage"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'rb') as f:
                    self._hashes = pickle.load(f)
                logger.info(f"Loaded {len(self._hashes)} content hashes")
            except Exception as e:
                logger.error(f"Failed to load hashes: {e}")
    
    def _save_hashes(self):
        """Save content hashes to storage"""
        try:
            with open(self.storage_path, 'wb') as f:
                pickle.dump(self._hashes, f)
        except Exception as e:
            logger.error(f"Failed to save hashes: {e}")
    
    def compute_hash(self, content: str) -> str:
        """Compute hash for content"""
        # Normalize content before hashing
        normalized = ' '.join(content.lower().split())
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def is_duplicate(self, url: str, content: str) -> bool:
        """Check if content is duplicate"""
        content_hash = self.compute_hash(content)
        
        # Check if exact hash exists
        if content_hash in self._hashes:
            existing = self._hashes[content_hash]
            if existing['url'] != url:
                logger.info(f"Duplicate content found: {url} duplicates {existing['url']}")
                return True
        
        return False
    
    def add_content(self, url: str, content: str, title: str = ""):
        """Add content hash to deduplication store"""
        content_hash = self.compute_hash(content)
        self._hashes[content_hash] = {
            "url": url,
            "title": title,
            "first_seen": datetime.now().isoformat(),
            "last_seen": datetime.now().isoformat()
        }
        self._save_hashes()
    
    def update_seen(self, content_hash: str):
        """Update last seen time for content"""
        if content_hash in self._hashes:
            self._hashes[content_hash]["last_seen"] = datetime.now().isoformat()
            self._save_hashes()


class ScrapingJobQueue:
    """Priority job queue for scraping tasks"""
    
    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self._queues: Dict[JobPriority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in JobPriority
        }
        self._active_jobs: Dict[str, ScrapingJob] = {}
        self._completed_jobs: Dict[str, ScrapingJob] = {}
        self._job_futures: Dict[str, asyncio.Future] = {}
        self._workers: List[asyncio.Task] = []
        self._shutdown = False
        self._lock = asyncio.Lock()
    
    async def start(self):
        """Start worker tasks"""
        for i in range(self.max_concurrent):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self._workers.append(worker)
        logger.info(f"Started {self.max_concurrent} scraping workers")
    
    async def stop(self):
        """Stop all workers"""
        self._shutdown = True
        
        # Cancel all pending jobs
        for queue in self._queues.values():
            while not queue.empty():
                try:
                    job = await queue.get()
                    job.status = JobStatus.CANCELLED
                    if job.job_id in self._job_futures:
                        self._job_futures[job.job_id].set_exception(
                            Exception("Job cancelled due to shutdown")
                        )
                except:
                    pass
        
        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)
        logger.info("All scraping workers stopped")
    
    async def add_job(self, job: ScrapingJob) -> asyncio.Future:
        """Add job to queue and return future for result"""
        async with self._lock:
            # Create future for job result
            future = asyncio.Future()
            self._job_futures[job.job_id] = future
            
            # Add to appropriate priority queue
            await self._queues[job.priority].put(job)
            logger.info(f"Added job {job.job_id} with priority {job.priority.name}")
            
            return future
    
    async def _worker(self, worker_id: str):
        """Worker task to process jobs"""
        logger.info(f"{worker_id} started")
        
        while not self._shutdown:
            job = None
            try:
                # Get highest priority job
                for priority in sorted(JobPriority, key=lambda p: p.value, reverse=True):
                    if not self._queues[priority].empty():
                        job = await asyncio.wait_for(
                            self._queues[priority].get(), 
                            timeout=0.1
                        )
                        break
                
                if not job:
                    await asyncio.sleep(0.1)
                    continue
                
                # Process job
                async with self._lock:
                    job.status = JobStatus.IN_PROGRESS
                    job.started_at = datetime.now()
                    self._active_jobs[job.job_id] = job
                
                logger.info(f"{worker_id} processing job {job.job_id}")
                
                # Execute job (implement actual scraping logic)
                result = await self._execute_job(job)
                
                # Update job status
                async with self._lock:
                    job.completed_at = datetime.now()
                    if result.get("error"):
                        job.status = JobStatus.FAILED
                        job.error = result["error"]
                        
                        # Retry if possible
                        if job.retries < job.max_retries:
                            job.retries += 1
                            job.status = JobStatus.PENDING
                            await self._queues[job.priority].put(job)
                            logger.info(f"Retrying job {job.job_id} ({job.retries}/{job.max_retries})")
                            continue
                    else:
                        job.status = JobStatus.COMPLETED
                        job.result = result
                    
                    # Move to completed
                    del self._active_jobs[job.job_id]
                    self._completed_jobs[job.job_id] = job
                    
                    # Resolve future
                    if job.job_id in self._job_futures:
                        if job.status == JobStatus.COMPLETED:
                            self._job_futures[job.job_id].set_result(job.result)
                        else:
                            self._job_futures[job.job_id].set_exception(
                                Exception(job.error or "Job failed")
                            )
                        del self._job_futures[job.job_id]
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"{worker_id} error: {e}")
                if job and job.job_id in self._job_futures:
                    self._job_futures[job.job_id].set_exception(e)
        
        logger.info(f"{worker_id} stopped")
    
    async def _execute_job(self, job: ScrapingJob) -> Dict[str, Any]:
        """Execute a scraping job"""
        # This will be implemented with actual scraping logic
        # For now, return mock result
        await asyncio.sleep(random.uniform(1, 3))
        return {"content": f"Scraped content for {job.url}", "title": "Test Article"}
    
    def get_status(self) -> Dict[str, Any]:
        """Get queue status"""
        return {
            "active_jobs": len(self._active_jobs),
            "completed_jobs": len(self._completed_jobs),
            "pending_by_priority": {
                priority.name: self._queues[priority].qsize() 
                for priority in JobPriority
            }
        }


class EnhancedWebScraper:
    """Main enhanced web scraping class"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_and_log_configs().get('web_scraper', {})
        
        # Initialize components
        self.rate_limiter = RateLimiter(
            max_requests_per_second=self.config.get('max_rps', 2.0),
            max_requests_per_minute=self.config.get('max_rpm', 60),
            max_requests_per_hour=self.config.get('max_rph', 1000)
        )
        
        self.cookie_manager = CookieManager()
        self.deduplicator = ContentDeduplicator()
        self.job_queue = ScrapingJobQueue(
            max_concurrent=self.config.get('max_concurrent', 5)
        )
        
        # Playwright browser
        self._browser: Optional[Browser] = None
        self._playwright = None
        
        # Progress tracking
        self._progress: Dict[str, Any] = defaultdict(dict)
        
    async def start(self):
        """Start the scraper"""
        await self.job_queue.start()
        
        # Initialize Playwright
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )
        
        logger.info("Enhanced web scraper started")
    
    async def stop(self):
        """Stop the scraper"""
        await self.job_queue.stop()
        await self.cookie_manager.close_all()
        
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        
        logger.info("Enhanced web scraper stopped")
    
    async def scrape_article(self, url: str, method: str = "trafilatura",
                           custom_cookies: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Scrape a single article with specified method"""
        # Apply rate limiting
        await self.rate_limiter.acquire()
        
        try:
            if method == "trafilatura":
                return await self._scrape_with_trafilatura(url, custom_cookies)
            elif method == "playwright":
                return await self._scrape_with_playwright(url, custom_cookies)
            elif method == "beautifulsoup":
                return await self._scrape_with_beautifulsoup(url, custom_cookies)
            else:
                raise ValueError(f"Unknown scraping method: {method}")
        
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            return {
                "url": url,
                "error": str(e),
                "extraction_successful": False
            }
    
    async def _scrape_with_trafilatura(self, url: str, 
                                     custom_cookies: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Scrape using trafilatura"""
        session = await self.cookie_manager.get_session(url)
        
        if custom_cookies:
            for cookie in custom_cookies:
                session.cookie_jar.update_cookies(cookie)
        
        async with session.get(url) as response:
            html = await response.text()
            
            # Extract with trafilatura
            content = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                include_images=False,
                output_format='json'
            )
            
            if content:
                content_dict = json.loads(content)
                
                # Check for duplicates
                if self.deduplicator.is_duplicate(url, content_dict.get('text', '')):
                    return {
                        "url": url,
                        "error": "Duplicate content",
                        "extraction_successful": False,
                        "is_duplicate": True
                    }
                
                # Add to deduplicator
                self.deduplicator.add_content(
                    url, 
                    content_dict.get('text', ''),
                    content_dict.get('title', '')
                )
                
                return {
                    "url": url,
                    "title": content_dict.get('title', 'Untitled'),
                    "author": content_dict.get('author', 'Unknown'),
                    "date": content_dict.get('date', ''),
                    "content": content_dict.get('text', ''),
                    "extraction_successful": True,
                    "method": "trafilatura"
                }
            
            return {
                "url": url,
                "error": "No content extracted",
                "extraction_successful": False
            }
    
    async def _scrape_with_playwright(self, url: str,
                                    custom_cookies: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Scrape using Playwright for JavaScript-heavy sites"""
        context = await self._browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        
        if custom_cookies:
            await context.add_cookies(custom_cookies)
        
        page = await context.new_page()
        
        try:
            # Navigate with timeout
            await page.goto(url, wait_until="networkidle", timeout=30000)
            
            # Wait for content to load
            await page.wait_for_load_state("domcontentloaded")
            
            # Extract content
            title = await page.title()
            
            # Try to find main content
            content = ""
            for selector in ['main', 'article', '[role="main"]', '#content', '.content']:
                elements = await page.query_selector_all(selector)
                if elements:
                    for element in elements:
                        text = await element.inner_text()
                        if len(text) > len(content):
                            content = text
            
            # Fallback to body if no content found
            if not content:
                content = await page.inner_text('body')
            
            # Extract metadata
            author = await page.evaluate('''() => {
                const meta = document.querySelector('meta[name="author"]');
                return meta ? meta.content : "Unknown";
            }''')
            
            date = await page.evaluate('''() => {
                const meta = document.querySelector('meta[property="article:published_time"]');
                return meta ? meta.content : "";
            }''')
            
            # Check for duplicates
            if self.deduplicator.is_duplicate(url, content):
                return {
                    "url": url,
                    "error": "Duplicate content",
                    "extraction_successful": False,
                    "is_duplicate": True
                }
            
            # Add to deduplicator
            self.deduplicator.add_content(url, content, title)
            
            return {
                "url": url,
                "title": title,
                "author": author,
                "date": date,
                "content": content,
                "extraction_successful": True,
                "method": "playwright"
            }
        
        finally:
            await page.close()
            await context.close()
    
    async def _scrape_with_beautifulsoup(self, url: str,
                                       custom_cookies: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Scrape using BeautifulSoup for simple HTML parsing"""
        session = await self.cookie_manager.get_session(url)
        
        if custom_cookies:
            for cookie in custom_cookies:
                session.cookie_jar.update_cookies(cookie)
        
        async with session.get(url) as response:
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract title
            title_tag = soup.find('title')
            title = title_tag.string.strip() if title_tag else "Untitled"
            
            # Extract content
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Try to find main content
            content = ""
            for tag in ['main', 'article', 'div']:
                elements = soup.find_all(tag, class_=lambda x: x and any(
                    keyword in x.lower() for keyword in ['content', 'article', 'post', 'entry']
                ))
                if elements:
                    content = '\n\n'.join(elem.get_text(strip=True) for elem in elements)
                    break
            
            # Fallback to body
            if not content:
                content = soup.get_text(strip=True)
            
            # Extract metadata
            author_meta = soup.find('meta', {'name': 'author'})
            author = author_meta.get('content', 'Unknown') if author_meta else 'Unknown'
            
            date_meta = soup.find('meta', {'property': 'article:published_time'})
            date = date_meta.get('content', '') if date_meta else ''
            
            # Check for duplicates
            if self.deduplicator.is_duplicate(url, content):
                return {
                    "url": url,
                    "error": "Duplicate content",
                    "extraction_successful": False,
                    "is_duplicate": True
                }
            
            # Add to deduplicator
            self.deduplicator.add_content(url, content, title)
            
            return {
                "url": url,
                "title": title,
                "author": author,
                "date": date,
                "content": content,
                "extraction_successful": True,
                "method": "beautifulsoup"
            }
    
    async def scrape_multiple(self, urls: List[str], method: str = "trafilatura",
                            priority: JobPriority = JobPriority.NORMAL,
                            summarize: bool = False,
                            **kwargs) -> List[Dict[str, Any]]:
        """Scrape multiple URLs concurrently"""
        jobs = []
        futures = []
        
        # Create jobs
        for url in urls:
            job = ScrapingJob(
                job_id=f"job_{int(time.time() * 1000)}_{hash(url)}",
                url=url,
                method=method,
                priority=priority,
                metadata=kwargs
            )
            future = await self.job_queue.add_job(job)
            jobs.append(job)
            futures.append(future)
        
        # Wait for all jobs to complete
        results = await asyncio.gather(*futures, return_exceptions=True)
        
        # Process results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append({
                    "url": urls[i],
                    "error": str(result),
                    "extraction_successful": False
                })
            else:
                # Add summarization if requested
                if summarize and result.get('extraction_successful') and result.get('content'):
                    summary = await self._summarize_content(
                        result['content'],
                        **kwargs
                    )
                    result['summary'] = summary
                
                final_results.append(result)
        
        return final_results
    
    async def _summarize_content(self, content: str, **kwargs) -> str:
        """Summarize content using LLM"""
        try:
            summary = analyze(
                input_data=content,
                custom_prompt_arg=kwargs.get('custom_prompt', ''),
                api_name=kwargs.get('api_name', 'openai'),
                api_key=kwargs.get('api_key'),
                temp=kwargs.get('temperature', 0.7),
                system_message=kwargs.get('system_message', 'Summarize this article concisely.')
            )
            return summary
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return "Summary generation failed"
    
    async def scrape_sitemap(self, sitemap_url: str, 
                           filter_func: Optional[Callable[[str], bool]] = None,
                           max_urls: Optional[int] = None) -> List[Dict[str, Any]]:
        """Scrape all URLs from a sitemap"""
        session = await self.cookie_manager.get_session(sitemap_url)
        
        async with session.get(sitemap_url) as response:
            content = await response.text()
            
        # Parse sitemap
        soup = BeautifulSoup(content, 'xml')
        urls = []
        
        for loc in soup.find_all('loc'):
            url = loc.text.strip()
            if filter_func and not filter_func(url):
                continue
            urls.append(url)
            if max_urls and len(urls) >= max_urls:
                break
        
        logger.info(f"Found {len(urls)} URLs in sitemap")
        
        # Scrape all URLs
        return await self.scrape_multiple(urls)
    
    async def recursive_scrape(self, base_url: str, max_pages: int = 100,
                             max_depth: int = 3, 
                             url_filter: Optional[Callable[[str], bool]] = None) -> List[Dict[str, Any]]:
        """Recursively scrape a website"""
        visited = set()
        to_visit = [(base_url, 0)]
        results = []
        
        while to_visit and len(results) < max_pages:
            url, depth = to_visit.pop(0)
            
            if url in visited or depth > max_depth:
                continue
            
            visited.add(url)
            
            # Update progress
            self._progress['recursive_scrape'] = {
                'visited': len(visited),
                'to_visit': len(to_visit),
                'scraped': len(results),
                'current_url': url,
                'current_depth': depth
            }
            
            # Scrape page
            result = await self.scrape_article(url)
            
            if result.get('extraction_successful'):
                results.append(result)
                
                # Extract links if not at max depth
                if depth < max_depth:
                    links = await self._extract_links(url, result.get('content', ''))
                    for link in links:
                        absolute_url = urljoin(base_url, link)
                        if absolute_url.startswith(base_url) and absolute_url not in visited:
                            if not url_filter or url_filter(absolute_url):
                                to_visit.append((absolute_url, depth + 1))
        
        return results
    
    async def _extract_links(self, base_url: str, content: str) -> List[str]:
        """Extract links from content"""
        soup = BeautifulSoup(content, 'html.parser')
        links = []
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if href and not href.startswith('#'):
                links.append(href)
        
        return links
    
    def get_progress(self, task_name: str) -> Dict[str, Any]:
        """Get progress for a specific task"""
        return self._progress.get(task_name, {})
    
    async def save_progress(self, task_name: str, filepath: Path):
        """Save progress to file for resumability"""
        progress_data = self._progress.get(task_name, {})
        async with aiofiles.open(filepath, 'w') as f:
            await f.write(json.dumps(progress_data, indent=2))
    
    async def load_progress(self, task_name: str, filepath: Path):
        """Load progress from file"""
        if filepath.exists():
            async with aiofiles.open(filepath, 'r') as f:
                content = await f.read()
                self._progress[task_name] = json.loads(content)


# Integration with existing system
async def create_enhanced_scraper() -> EnhancedWebScraper:
    """Create and initialize enhanced scraper"""
    scraper = EnhancedWebScraper()
    await scraper.start()
    return scraper


# Example usage
if __name__ == "__main__":
    async def test_enhanced_scraper():
        """Test the enhanced scraper"""
        scraper = await create_enhanced_scraper()
        
        try:
            # Test single article scraping
            print("Testing single article scraping...")
            result = await scraper.scrape_article(
                "https://example.com/article",
                method="trafilatura"
            )
            print(f"Result: {result['title'] if result.get('extraction_successful') else result['error']}")
            
            # Test multiple URL scraping
            print("\nTesting multiple URL scraping...")
            urls = [
                "https://example.com/article1",
                "https://example.com/article2",
                "https://example.com/article3"
            ]
            results = await scraper.scrape_multiple(
                urls,
                priority=JobPriority.HIGH,
                summarize=True
            )
            print(f"Scraped {len(results)} articles")
            
            # Test recursive scraping
            print("\nTesting recursive scraping...")
            results = await scraper.recursive_scrape(
                "https://example.com",
                max_pages=10,
                max_depth=2
            )
            print(f"Recursively scraped {len(results)} pages")
            
            # Get queue status
            status = scraper.job_queue.get_status()
            print(f"\nQueue status: {status}")
            
        finally:
            await scraper.stop()
    
    asyncio.run(test_enhanced_scraper())