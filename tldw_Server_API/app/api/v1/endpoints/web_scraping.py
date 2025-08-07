# web_scraping.py - Web Scraping Management Endpoints
"""
Additional endpoints for managing the enhanced web scraping service.
Provides job management, status checking, and service control.
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from loguru import logger

from tldw_Server_API.app.services.enhanced_web_scraping_service import (
    get_web_scraping_service, WebScrapingService
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User


router = APIRouter(
    prefix="/web-scraping",
    tags=["Web Scraping Management"],
)


@router.get("/status")
async def get_scraping_service_status(
    current_user: User = Depends(get_request_user)
) -> Dict[str, Any]:
    """
    Get the status of the web scraping service including queue statistics.
    
    Returns:
        Service status including:
        - Initialization status
        - Queue statistics (active, pending, completed jobs)
        - Rate limiting configuration
    """
    try:
        service = get_web_scraping_service()
        return service.get_service_status()
    except Exception as e:
        logger.error(f"Failed to get scraping service status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/job/{job_id}")
async def get_scraping_job_status(
    job_id: str,
    current_user: User = Depends(get_request_user)
) -> Dict[str, Any]:
    """
    Get the status of a specific scraping job.
    
    Args:
        job_id: The ID of the scraping job
        
    Returns:
        Job details including status, progress, and results
    """
    try:
        service = get_web_scraping_service()
        return await service.get_job_status(job_id)
    except Exception as e:
        logger.error(f"Failed to get job status for {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/job/{job_id}")
async def cancel_scraping_job(
    job_id: str,
    current_user: User = Depends(get_request_user)
) -> Dict[str, Any]:
    """
    Cancel a pending or active scraping job.
    
    Args:
        job_id: The ID of the scraping job to cancel
        
    Returns:
        Cancellation status
    """
    try:
        service = get_web_scraping_service()
        return await service.cancel_job(job_id)
    except Exception as e:
        logger.error(f"Failed to cancel job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/service/initialize")
async def initialize_scraping_service(
    current_user: User = Depends(get_request_user)
) -> Dict[str, Any]:
    """
    Initialize the web scraping service if not already initialized.
    
    This starts the worker pool and prepares the service for scraping.
    """
    try:
        service = get_web_scraping_service()
        await service.initialize()
        return {
            "status": "success",
            "message": "Web scraping service initialized",
            "service_status": service.get_service_status()
        }
    except Exception as e:
        logger.error(f"Failed to initialize scraping service: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/service/shutdown")
async def shutdown_scraping_service(
    current_user: User = Depends(get_request_user)
) -> Dict[str, Any]:
    """
    Shutdown the web scraping service gracefully.
    
    This stops all workers and cleans up resources.
    Admin only endpoint.
    """
    # Check if user is admin (add your admin check logic)
    # if not current_user.is_admin:
    #     raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        service = get_web_scraping_service()
        await service.shutdown()
        return {
            "status": "success",
            "message": "Web scraping service shutdown completed"
        }
    except Exception as e:
        logger.error(f"Failed to shutdown scraping service: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/progress/{task_id}")
async def get_scraping_progress(
    task_id: str,
    current_user: User = Depends(get_request_user)
) -> Dict[str, Any]:
    """
    Get progress information for a scraping task.
    
    Useful for long-running recursive or sitemap scraping tasks.
    
    Args:
        task_id: The task identifier
        
    Returns:
        Progress information including pages scraped, remaining, current URL, etc.
    """
    try:
        service = get_web_scraping_service()
        if not service._initialized:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        progress = service.scraper.get_progress(task_id)
        if not progress:
            raise HTTPException(status_code=404, detail="Task not found or no progress available")
        
        return {
            "task_id": task_id,
            "progress": progress,
            "status": "in_progress"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get progress for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cookies/{domain}")
async def get_cookies_for_domain(
    domain: str,
    current_user: User = Depends(get_request_user)
) -> Dict[str, Any]:
    """
    Get stored cookies for a specific domain.
    
    Args:
        domain: The domain to get cookies for
        
    Returns:
        List of cookies for the domain
    """
    try:
        service = get_web_scraping_service()
        if not service._initialized:
            await service.initialize()
        
        cookies = service.scraper.cookie_manager.get_cookies(f"https://{domain}")
        
        return {
            "domain": domain,
            "cookies": cookies or [],
            "cookie_count": len(cookies) if cookies else 0
        }
    except Exception as e:
        logger.error(f"Failed to get cookies for {domain}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cookies/{domain}")
async def set_cookies_for_domain(
    domain: str,
    cookies: list[Dict[str, Any]],
    current_user: User = Depends(get_request_user)
) -> Dict[str, Any]:
    """
    Set cookies for a specific domain.
    
    Useful for handling authentication or paywalled content.
    
    Args:
        domain: The domain to set cookies for
        cookies: List of cookie dictionaries
        
    Returns:
        Success status
    """
    try:
        service = get_web_scraping_service()
        if not service._initialized:
            await service.initialize()
        
        service.scraper.cookie_manager.add_cookies(domain, cookies)
        
        return {
            "status": "success",
            "message": f"Added {len(cookies)} cookies for {domain}",
            "domain": domain
        }
    except Exception as e:
        logger.error(f"Failed to set cookies for {domain}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/duplicates/check")
async def check_url_duplicate(
    url: str = Query(..., description="URL to check for duplicate content"),
    current_user: User = Depends(get_request_user)
) -> Dict[str, Any]:
    """
    Check if a URL's content has already been scraped (duplicate detection).
    
    Args:
        url: The URL to check
        
    Returns:
        Duplicate status and information about the original if found
    """
    try:
        service = get_web_scraping_service()
        if not service._initialized:
            await service.initialize()
        
        # For checking, we'd need to get the content first
        # This is a simplified check - in production you might want to
        # check against URL patterns or pre-fetch headers
        
        return {
            "url": url,
            "is_duplicate": False,  # Placeholder
            "message": "Duplicate checking requires content analysis"
        }
    except Exception as e:
        logger.error(f"Failed to check duplicate for {url}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Include this router in your main app
# In main.py or wherever you configure routes:
# app.include_router(web_scraping_router, prefix="/api/v1")