# enhanced_web_scraping_service.py - Enhanced Web Scraping Service
"""
Enhanced web scraping service that integrates the production scraping pipeline
with the existing tldw_server API structure.
"""

import asyncio
import uuid
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException
from loguru import logger

# Import the enhanced scraper
from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import (
    EnhancedWebScraper, ScrapingJob, JobPriority, JobStatus,
    create_enhanced_scraper
)

# Import existing components
from tldw_Server_API.app.services.ephemeral_store import ephemeral_storage
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze
from tldw_Server_API.app.core.DB_Management.DB_Manager import add_media_with_keywords
from tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib import (
    is_content_page, ContentMetadataHandler
)


class WebScrapingService:
    """Enhanced web scraping service with production features"""
    
    def __init__(self):
        self.scraper: Optional[EnhancedWebScraper] = None
        self._initialized = False
        self._active_jobs: Dict[str, ScrapingJob] = {}
    
    async def initialize(self):
        """Initialize the scraping service"""
        if not self._initialized:
            self.scraper = await create_enhanced_scraper()
            self._initialized = True
            logger.info("Web scraping service initialized")
    
    async def shutdown(self):
        """Shutdown the scraping service"""
        if self.scraper:
            await self.scraper.stop()
            self._initialized = False
            logger.info("Web scraping service shutdown")
    
    async def process_web_scraping_task(
        self,
        scrape_method: str,
        url_input: str,
        url_level: Optional[int] = None,
        max_pages: int = 100,
        max_depth: int = 3,
        summarize_checkbox: bool = False,
        custom_prompt: Optional[str] = None,
        api_name: Optional[str] = None,
        api_key: Optional[str] = None,
        keywords: str = "",
        custom_titles: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        custom_cookies: Optional[List[Dict[str, Any]]] = None,
        mode: str = "persist",
        priority: str = "normal",
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process web scraping task with enhanced features.
        
        This replaces the placeholder implementation with production features:
        - Concurrent scraping with rate limiting
        - Job queue management
        - Progress tracking
        - Content deduplication
        - Robust error handling
        """
        # Ensure service is initialized
        if not self._initialized:
            await self.initialize()
        
        try:
            # Map priority string to enum
            priority_map = {
                "low": JobPriority.LOW,
                "normal": JobPriority.NORMAL,
                "high": JobPriority.HIGH,
                "critical": JobPriority.CRITICAL
            }
            job_priority = priority_map.get(priority.lower(), JobPriority.NORMAL)
            
            # Create task ID
            task_id = f"scrape_{uuid.uuid4().hex[:8]}"
            
            # Process based on scraping method
            if scrape_method == "Individual URLs":
                result = await self._scrape_individual_urls(
                    url_input, custom_titles, summarize_checkbox,
                    custom_prompt, api_name, api_key, keywords,
                    system_prompt, temperature, custom_cookies,
                    job_priority
                )
                
            elif scrape_method == "Sitemap":
                result = await self._scrape_sitemap(
                    url_input, max_pages, summarize_checkbox,
                    custom_prompt, api_name, api_key, keywords,
                    system_prompt, temperature, job_priority
                )
                
            elif scrape_method == "URL Level":
                if url_level is None:
                    raise ValueError("url_level must be provided for URL Level scraping")
                result = await self._scrape_by_url_level(
                    url_input, url_level, max_pages, summarize_checkbox,
                    custom_prompt, api_name, api_key, keywords,
                    system_prompt, temperature, job_priority
                )
                
            elif scrape_method == "Recursive Scraping":
                result = await self._scrape_recursive(
                    url_input, max_pages, max_depth, summarize_checkbox,
                    custom_prompt, api_name, api_key, keywords,
                    system_prompt, temperature, custom_cookies,
                    job_priority
                )
                
            else:
                raise ValueError(f"Unknown scrape method: {scrape_method}")
            
            # Process results based on mode
            if mode == "ephemeral":
                return await self._store_ephemeral(result, task_id, user_id)
            else:
                return await self._store_persistent(result, keywords, user_id)
                
        except Exception as e:
            logger.error(f"Web scraping task failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _scrape_individual_urls(
        self, url_input: str, custom_titles: Optional[str],
        summarize: bool, custom_prompt: Optional[str],
        api_name: Optional[str], api_key: Optional[str],
        keywords: str, system_prompt: Optional[str],
        temperature: float, custom_cookies: Optional[List[Dict[str, Any]]],
        priority: JobPriority
    ) -> Dict[str, Any]:
        """Scrape individual URLs with enhanced features"""
        # Parse URLs and titles
        urls = [url.strip() for url in url_input.split('\n') if url.strip()]
        titles = custom_titles.split('\n') if custom_titles else []
        
        # Scrape with enhanced scraper
        results = await self.scraper.scrape_multiple(
            urls,
            method="trafilatura",  # Can be made configurable
            priority=priority,
            summarize=summarize,
            custom_prompt=custom_prompt,
            api_name=api_name,
            api_key=api_key,
            system_prompt=system_prompt,
            temperature=temperature,
            custom_cookies=custom_cookies
        )
        
        # Apply custom titles if provided
        for i, result in enumerate(results):
            if i < len(titles) and titles[i]:
                result['title'] = titles[i]
        
        return {
            "method": "Individual URLs",
            "total_articles": len(results),
            "articles": results,
            "keywords": keywords
        }
    
    async def _scrape_sitemap(
        self, sitemap_url: str, max_pages: int,
        summarize: bool, custom_prompt: Optional[str],
        api_name: Optional[str], api_key: Optional[str],
        keywords: str, system_prompt: Optional[str],
        temperature: float, priority: JobPriority
    ) -> Dict[str, Any]:
        """Scrape from sitemap with filtering"""
        # Scrape sitemap with content page filter
        results = await self.scraper.scrape_sitemap(
            sitemap_url,
            filter_func=is_content_page,
            max_urls=max_pages
        )
        
        # Add summarization if requested
        if summarize:
            for result in results:
                if result.get('extraction_successful') and result.get('content'):
                    summary = await self._summarize_content(
                        result['content'],
                        custom_prompt, api_name, api_key,
                        system_prompt, temperature
                    )
                    result['summary'] = summary
        
        return {
            "method": "Sitemap",
            "total_articles": len(results),
            "articles": results,
            "keywords": keywords,
            "sitemap_url": sitemap_url
        }
    
    async def _scrape_by_url_level(
        self, base_url: str, url_level: int, max_pages: int,
        summarize: bool, custom_prompt: Optional[str],
        api_name: Optional[str], api_key: Optional[str],
        keywords: str, system_prompt: Optional[str],
        temperature: float, priority: JobPriority
    ) -> Dict[str, Any]:
        """Scrape by URL level"""
        # Define URL level filter
        def url_level_filter(url: str) -> bool:
            from urllib.parse import urlparse
            path_parts = urlparse(url).path.strip('/').split('/')
            return len(path_parts) <= url_level and is_content_page(url)
        
        # Recursive scrape with level filter
        results = await self.scraper.recursive_scrape(
            base_url,
            max_pages=max_pages,
            max_depth=url_level,
            url_filter=url_level_filter
        )
        
        # Add summarization if requested
        if summarize:
            for result in results:
                if result.get('extraction_successful') and result.get('content'):
                    summary = await self._summarize_content(
                        result['content'],
                        custom_prompt, api_name, api_key,
                        system_prompt, temperature
                    )
                    result['summary'] = summary
        
        return {
            "method": "URL Level",
            "total_articles": len(results),
            "articles": results,
            "keywords": keywords,
            "url_level": url_level
        }
    
    async def _scrape_recursive(
        self, base_url: str, max_pages: int, max_depth: int,
        summarize: bool, custom_prompt: Optional[str],
        api_name: Optional[str], api_key: Optional[str],
        keywords: str, system_prompt: Optional[str],
        temperature: float, custom_cookies: Optional[List[Dict[str, Any]]],
        priority: JobPriority
    ) -> Dict[str, Any]:
        """Recursive scraping with progress tracking"""
        # Create progress file for resumability
        progress_file = Path(f"./scrape_progress_{uuid.uuid4().hex[:8]}.json")
        
        try:
            # Perform recursive scrape
            results = await self.scraper.recursive_scrape(
                base_url,
                max_pages=max_pages,
                max_depth=max_depth,
                url_filter=is_content_page
            )
            
            # Add summarization if requested
            if summarize:
                for result in results:
                    if result.get('extraction_successful') and result.get('content'):
                        summary = await self._summarize_content(
                            result['content'],
                            custom_prompt, api_name, api_key,
                            system_prompt, temperature
                        )
                        result['summary'] = summary
            
            # Save final progress
            await self.scraper.save_progress('recursive_scrape', progress_file)
            
            return {
                "method": "Recursive Scraping",
                "total_articles": len(results),
                "articles": results,
                "keywords": keywords,
                "max_depth": max_depth,
                "progress_file": str(progress_file)
            }
            
        except Exception as e:
            # Save progress on error for resumability
            await self.scraper.save_progress('recursive_scrape', progress_file)
            raise
    
    async def _summarize_content(
        self, content: str, custom_prompt: Optional[str],
        api_name: Optional[str], api_key: Optional[str],
        system_prompt: Optional[str], temperature: float
    ) -> str:
        """Summarize content using LLM"""
        try:
            summary = analyze(
                input_data=content,
                custom_prompt_arg=custom_prompt or "Summarize this article concisely.",
                api_name=api_name or "openai",
                api_key=api_key,
                temp=temperature,
                system_message=system_prompt or "You are a professional summarizer."
            )
            return summary
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return "Summary generation failed"
    
    async def _store_ephemeral(
        self, result: Dict[str, Any], task_id: str, user_id: Optional[int]
    ) -> Dict[str, Any]:
        """Store results in ephemeral storage"""
        ephemeral_data = {
            "task_id": task_id,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
        
        ephemeral_id = ephemeral_storage.store_data(ephemeral_data)
        
        return {
            "status": "ephemeral-ok",
            "ephemeral_id": ephemeral_id,
            "task_id": task_id,
            "total_articles": result.get("total_articles", 0),
            "method": result.get("method"),
            "preview": {
                "articles": len(result.get("articles", [])),
                "first_article": result.get("articles", [{}])[0].get("title", "N/A") if result.get("articles") else None
            }
        }
    
    async def _store_persistent(
        self, result: Dict[str, Any], keywords: str, user_id: Optional[int]
    ) -> Dict[str, Any]:
        """Store results in database"""
        media_ids = []
        errors = []
        
        for article in result.get("articles", []):
            if not article.get("extraction_successful"):
                errors.append(f"Failed to extract: {article.get('url')}")
                continue
            
            try:
                # Prepare data for database
                info_dict = {
                    "title": article.get("title", "Untitled"),
                    "author": article.get("author", "Unknown"),
                    "source": article.get("url", ""),
                    "scrape_method": result.get("method", "Unknown"),
                    "extraction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Format content with metadata
                content_with_metadata = ContentMetadataHandler.format_content_with_metadata(
                    url=article.get("url", ""),
                    content=article.get("content", ""),
                    pipeline=article.get("method", "enhanced"),
                    additional_metadata={
                        "date": article.get("date", ""),
                        "author": article.get("author", "Unknown")
                    }
                )
                
                # Prepare segments
                segments = [{"Text": content_with_metadata}]
                
                # Use summary if available
                summary = article.get("summary", "No summary available")
                
                # Add to database
                media_id = add_media_with_keywords(
                    url=article.get("url", ""),
                    info_dict=info_dict,
                    segments=segments,
                    summary=summary,
                    keywords=keywords.split(",") if keywords else [],
                    custom_prompt_input=None,
                    whisper_model="web-scraping-import",
                    media_type="web_document",
                    overwrite=False,
                    user_id=user_id
                )
                
                media_ids.append(media_id)
                
            except Exception as e:
                logger.error(f"Failed to store article: {e}")
                errors.append(f"Storage failed for {article.get('url')}: {str(e)}")
        
        return {
            "status": "persist-ok",
            "media_ids": media_ids,
            "total_articles": len(result.get("articles", [])),
            "stored_articles": len(media_ids),
            "method": result.get("method"),
            "errors": errors if errors else None
        }
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a scraping job"""
        if not self._initialized:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        # Check active jobs
        if job_id in self._active_jobs:
            job = self._active_jobs[job_id]
            return job.to_dict()
        
        # Check queue status
        queue_status = self.scraper.job_queue.get_status()
        
        return {
            "job_id": job_id,
            "status": "unknown",
            "queue_status": queue_status
        }
    
    async def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel a scraping job"""
        if not self._initialized:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        # Implementation would cancel the job if it's in queue
        # For now, return mock response
        return {
            "job_id": job_id,
            "status": "cancelled",
            "message": "Job cancellation requested"
        }
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get service status and statistics"""
        if not self._initialized:
            return {
                "status": "not_initialized",
                "initialized": False
            }
        
        queue_status = self.scraper.job_queue.get_status()
        
        return {
            "status": "operational",
            "initialized": True,
            "queue": queue_status,
            "active_jobs": len(self._active_jobs),
            "rate_limiter": {
                "max_rps": self.scraper.rate_limiter.max_rps,
                "max_rpm": self.scraper.rate_limiter.max_rpm,
                "max_rph": self.scraper.rate_limiter.max_rph
            }
        }


# Global service instance
_web_scraping_service: Optional[WebScrapingService] = None


def get_web_scraping_service() -> WebScrapingService:
    """Get or create web scraping service instance"""
    global _web_scraping_service
    if _web_scraping_service is None:
        _web_scraping_service = WebScrapingService()
    return _web_scraping_service


# Integration with existing API
async def process_web_scraping_task(**kwargs) -> Dict[str, Any]:
    """
    Process web scraping task - drop-in replacement for the placeholder function.
    
    This function maintains the same interface as the original but uses
    the enhanced scraping service.
    """
    service = get_web_scraping_service()
    return await service.process_web_scraping_task(**kwargs)


# Cleanup on shutdown
async def shutdown_web_scraping_service():
    """Shutdown the web scraping service"""
    global _web_scraping_service
    if _web_scraping_service:
        await _web_scraping_service.shutdown()
        _web_scraping_service = None