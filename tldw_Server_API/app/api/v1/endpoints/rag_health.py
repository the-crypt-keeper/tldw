# rag_health.py
"""
Health and monitoring endpoints for the RAG service.

Provides health checks, cache statistics, and system monitoring.
"""

from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, status

from loguru import logger

# Import RAG components
from ....core.RAG.rag_service.advanced_cache import RAGCache
from ....core.RAG.rag_service.metrics_collector import get_metrics_collector
from ....core.RAG.rag_service.resilience import get_coordinator, HealthStatus
from ....core.RAG.rag_service.quick_wins import get_cost_tracker
from ....core.RAG.rag_service.batch_processing import BatchProcessor


router = APIRouter(prefix="/api/v1/rag", tags=["RAG Health"])


# Global instances
_rag_cache: Optional[RAGCache] = None
_batch_processor: Optional[BatchProcessor] = None


def get_rag_cache() -> RAGCache:
    """Get or create RAG cache instance."""
    global _rag_cache
    if _rag_cache is None:
        _rag_cache = RAGCache(enable_multi_level=True)
    return _rag_cache


def get_batch_processor() -> BatchProcessor:
    """Get or create batch processor instance."""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = BatchProcessor()
    return _batch_processor


@router.get("/health", summary="RAG service health check")
async def health_check() -> Dict[str, Any]:
    """
    Comprehensive health check for RAG service.
    
    Returns health status of all components.
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {},
        "version": "1.0.0"
    }
    
    try:
        # Check error recovery coordinator
        coordinator = get_coordinator()
        
        # Check circuit breakers
        circuit_breakers_healthy = True
        for name, breaker in coordinator.circuit_breakers.items():
            breaker_stats = breaker.get_stats()
            is_healthy = breaker_stats["state"] != "open"
            circuit_breakers_healthy &= is_healthy
            
            health_status["components"][f"circuit_breaker_{name}"] = {
                "status": "healthy" if is_healthy else "unhealthy",
                "state": breaker_stats["state"],
                "failure_rate": breaker_stats["failure_rate"]
            }
        
        # Check cache
        try:
            cache = get_rag_cache()
            cache_stats = cache.get_stats()
            cache_healthy = True  # Could check hit rate thresholds
            
            health_status["components"]["cache"] = {
                "status": "healthy" if cache_healthy else "degraded",
                "hit_rate": cache_stats.get("hit_rate", 0),
                "size": cache_stats.get("size", 0)
            }
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            health_status["components"]["cache"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check metrics collector
        try:
            metrics = get_metrics_collector()
            current_metrics = metrics.get_current_metrics()
            metrics_healthy = current_metrics is not None
            
            health_status["components"]["metrics"] = {
                "status": "healthy" if metrics_healthy else "unhealthy",
                "recent_queries": current_metrics.get("recent_queries", 0)
            }
        except Exception as e:
            logger.error(f"Metrics health check failed: {e}")
            health_status["components"]["metrics"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check batch processor
        try:
            batch = get_batch_processor()
            batch_stats = batch.get_statistics()
            batch_healthy = True
            
            health_status["components"]["batch_processor"] = {
                "status": "healthy" if batch_healthy else "degraded",
                "active_jobs": len(batch.active_jobs),
                "success_rate": batch_stats.get("job_success_rate", 0)
            }
        except Exception as e:
            logger.error(f"Batch processor health check failed: {e}")
            health_status["components"]["batch_processor"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Overall health determination
        all_healthy = all(
            comp.get("status") == "healthy"
            for comp in health_status["components"].values()
        )
        
        any_unhealthy = any(
            comp.get("status") == "unhealthy"
            for comp in health_status["components"].values()
        )
        
        if any_unhealthy:
            health_status["status"] = "unhealthy"
        elif not all_healthy:
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": "Error occured during RAG health check"
        }


@router.get("/health/live", summary="Simple liveness check")
async def liveness_check() -> Dict[str, str]:
    """
    Simple liveness check for container orchestration.
    
    Returns 200 if service is alive.
    """
    return {"status": "alive"}


@router.get("/health/ready", summary="Readiness check")
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check for container orchestration.
    
    Returns 200 if service is ready to handle requests.
    """
    try:
        # Quick checks for critical components
        cache = get_rag_cache()
        metrics = get_metrics_collector()
        
        return {
            "status": "ready",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service not ready: {str(e)}"
        )


@router.get("/cache/stats", summary="Get cache statistics")
async def get_cache_statistics() -> Dict[str, Any]:
    """
    Get detailed cache statistics.
    
    Returns cache performance metrics and status.
    """
    try:
        cache = get_rag_cache()
        stats = cache.get_stats()
        
        # Add additional computed metrics
        if isinstance(stats, dict):
            # For multi-level cache
            overall_stats = stats.get("overall", {})
            hit_rate = overall_stats.get("hit_rate", 0)
            
            # Determine cache effectiveness
            effectiveness = "excellent" if hit_rate > 0.8 else \
                          "good" if hit_rate > 0.6 else \
                          "fair" if hit_rate > 0.4 else \
                          "poor"
            
            return {
                "timestamp": datetime.now().isoformat(),
                "effectiveness": effectiveness,
                "statistics": stats,
                "recommendations": _get_cache_recommendations(stats)
            }
        else:
            # Simple cache stats
            return {
                "timestamp": datetime.now().isoformat(),
                "statistics": stats
            }
            
    except Exception as e:
        logger.error(f"Failed to get cache statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve cache statistics: {str(e)}"
        )


@router.post("/cache/clear", summary="Clear cache")
async def clear_cache() -> Dict[str, str]:
    """
    Clear all cache entries.
    
    WARNING: This will impact performance until cache is rebuilt.
    """
    try:
        cache = get_rag_cache()
        await cache.cache.clear()
        
        logger.warning("Cache cleared via API endpoint")
        
        return {
            "status": "success",
            "message": "Cache cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}"
        )


@router.get("/cache/warm", summary="Get cache warming status")
async def get_cache_warming_status() -> Dict[str, Any]:
    """Get status of cache warming operations."""
    try:
        cache = get_rag_cache()
        
        if cache.warmer:
            top_queries = cache.warmer.get_top_queries(n=10)
            
            return {
                "warming_enabled": True,
                "top_queries": top_queries,
                "access_history_size": len(cache.warmer.access_history)
            }
        else:
            return {
                "warming_enabled": False,
                "message": "Cache warming not configured"
            }
            
    except Exception as e:
        logger.error(f"Failed to get warming status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/metrics/summary", summary="Get metrics summary")
async def get_metrics_summary() -> Dict[str, Any]:
    """Get summary of RAG pipeline metrics."""
    try:
        metrics = get_metrics_collector()
        current = metrics.get_current_metrics()
        
        # Get aggregated metrics for last hour
        end_time = datetime.now().timestamp()
        start_time = end_time - 3600  # Last hour
        
        aggregated = metrics.aggregate_metrics(start_time, end_time)
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "current": current,
            "last_hour": {
                "query_count": aggregated.query_count if aggregated else 0,
                "avg_duration": aggregated.avg_total_duration if aggregated else 0,
                "p95_duration": aggregated.p95_duration if aggregated else 0,
                "cache_hit_rate": aggregated.cache_hit_rate if aggregated else 0,
                "error_rate": aggregated.error_rate if aggregated else 0
            } if aggregated else None
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Failed to get metrics summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/costs/summary", summary="Get cost tracking summary")
async def get_cost_summary() -> Dict[str, Any]:
    """Get summary of LLM API costs."""
    try:
        tracker = get_cost_tracker()
        summary = tracker.get_summary()
        
        # Add budget warnings if configured
        budget_warnings = []
        daily_budget = 10.0  # Example: $10/day
        
        if summary["total_cost"] > daily_budget:
            budget_warnings.append({
                "level": "warning",
                "message": f"Daily budget exceeded: ${summary['total_cost']:.2f} > ${daily_budget:.2f}"
            })
        
        return {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "warnings": budget_warnings
        }
        
    except Exception as e:
        logger.error(f"Failed to get cost summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/batch/jobs", summary="Get batch job statuses")
async def get_batch_jobs() -> Dict[str, Any]:
    """Get status of all batch processing jobs."""
    try:
        processor = get_batch_processor()
        
        jobs = []
        for job_id, job in processor.jobs.items():
            jobs.append({
                "id": job_id,
                "status": job.status.value,
                "progress": job.progress,
                "total_queries": job.total_queries,
                "completed_queries": job.completed_queries,
                "success_rate": job.success_rate,
                "created_at": job.created_at
            })
        
        # Sort by creation time (most recent first)
        jobs.sort(key=lambda x: x["created_at"], reverse=True)
        
        return {
            "active_jobs": list(processor.active_jobs),
            "total_jobs": len(jobs),
            "jobs": jobs[:20]  # Last 20 jobs
        }
        
    except Exception as e:
        logger.error(f"Failed to get batch jobs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


def _get_cache_recommendations(stats: Dict[str, Any]) -> list:
    """Generate cache recommendations based on statistics."""
    recommendations = []
    
    # Check overall hit rate
    overall = stats.get("overall", {})
    hit_rate = overall.get("hit_rate", 0)
    
    if hit_rate < 0.3:
        recommendations.append({
            "priority": "high",
            "message": "Very low cache hit rate. Consider cache warming or increasing TTL."
        })
    elif hit_rate < 0.5:
        recommendations.append({
            "priority": "medium",
            "message": "Low cache hit rate. Review query patterns and adjust caching strategy."
        })
    
    # Check L1 cache
    l1_stats = stats.get("l1", {})
    if l1_stats.get("evictions", 0) > l1_stats.get("size", 1) * 2:
        recommendations.append({
            "priority": "medium",
            "message": "High L1 eviction rate. Consider increasing L1 cache size."
        })
    
    # Check L2 cache
    l2_stats = stats.get("l2", {})
    l2_hit_rate = l2_stats.get("hit_rate", 0)
    if l2_hit_rate > 0.7 and l1_stats.get("hit_rate", 0) < 0.3:
        recommendations.append({
            "priority": "low",
            "message": "L2 performing better than L1. Consider adjusting promotion strategy."
        })
    
    return recommendations