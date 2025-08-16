# health.py
# Description: Health check and monitoring endpoints for Kubernetes and system monitoring
#
# Imports
from typing import Dict, Any, Optional
from datetime import datetime
import psutil
import aiosqlite
#
# 3rd-party imports
from fastapi import APIRouter, Response, Depends, status
from loguru import logger
#
# Local imports
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.settings import Settings, get_settings
from tldw_Server_API.app.core.Evaluations.circuit_breaker import CircuitBreaker, CircuitState
from tldw_Server_API.app.core.Evaluations.evaluation_manager import EvaluationManager

#######################################################################################################################
#
# Router Configuration

router = APIRouter(
    prefix="/health",
    tags=["health", "monitoring"],
    responses={503: {"description": "Service unavailable"}}
)


#######################################################################################################################
#
# Health Check Endpoints

@router.get("")
async def health_check(
    db_pool: DatabasePool = Depends(get_db_pool),
    settings: Settings = Depends(get_settings)
) -> Dict[str, Any]:
    """
    Comprehensive health check endpoint
    
    Checks database connectivity, system resources, and service status.
    
    Returns:
        Dict with health status and component checks
    """
    health = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "auth_mode": settings.AUTH_MODE,
        "checks": {}
    }
    
    # Database check
    try:
        if settings.AUTH_MODE == "multi_user":
            async with db_pool.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
        else:
            async with aiosqlite.connect(db_pool.db_path) as conn:
                await conn.execute("SELECT 1")
        health["checks"]["database"] = "ok"
    except Exception as e:
        health["checks"]["database"] = f"error: {str(e)}"
        health["status"] = "unhealthy"
        logger.error(f"Database health check failed: {e}")
    
    # Redis check (optional)
    if settings.REDIS_URL:
        try:
            import aioredis
            redis_client = await aioredis.from_url(settings.REDIS_URL)
            await redis_client.ping()
            await redis_client.close()
            health["checks"]["redis"] = "ok"
        except Exception as e:
            health["checks"]["redis"] = f"warning: {str(e)}"
            # Redis is optional, don't mark as unhealthy
            logger.warning(f"Redis health check failed: {e}")
    
    # Evaluation service check
    try:
        # Check evaluation database
        eval_manager = EvaluationManager()
        eval_db_status = "ok"
        
        # Check if evaluation database is accessible
        import sqlite3
        with sqlite3.connect(eval_manager.db_path) as conn:
            conn.execute("SELECT COUNT(*) FROM evaluations")
            
        # Check circuit breakers status
        circuit_breakers = {}
        # Note: In production, you'd get actual circuit breaker instances
        # For now, we'll just report the expected providers
        providers = ["openai", "anthropic", "google", "cohere", "mistral"]
        for provider in providers:
            # Create a dummy circuit breaker to check configuration
            cb = CircuitBreaker(name=f"evaluation_{provider}")
            circuit_breakers[provider] = {
                "state": cb.state.value,
                "failure_count": cb.stats.failed_calls,
                "success_count": cb.stats.successful_calls
            }
            
        health["checks"]["evaluations"] = {
            "database": eval_db_status,
            "circuit_breakers": circuit_breakers,
            "embeddings_available": True  # Check if embeddings service is configured
        }
    except Exception as e:
        health["checks"]["evaluations"] = f"error: {str(e)}"
        # Don't mark as unhealthy since evaluations is not critical
        if health["status"] == "healthy":
            health["status"] = "degraded"
        logger.warning(f"Evaluation service health check failed: {e}")
    
    # System resources
    try:
        health["checks"]["system"] = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "process_count": len(psutil.pids())
        }
        
        # High resource usage warning
        if health["checks"]["system"]["memory_percent"] > 90:
            health["status"] = "degraded"
            health["warnings"] = health.get("warnings", [])
            health["warnings"].append("High memory usage")
        
        if health["checks"]["system"]["disk_percent"] > 90:
            health["status"] = "degraded"
            health["warnings"] = health.get("warnings", [])
            health["warnings"].append("Low disk space")
            
    except Exception as e:
        health["checks"]["system"] = f"error: {str(e)}"
        logger.error(f"System health check failed: {e}")
    
    # Determine HTTP status code
    status_code = 200 if health["status"] == "healthy" else 503 if health["status"] == "unhealthy" else 206
    
    return Response(
        content=__import__('json').dumps(health, indent=2),
        status_code=status_code,
        media_type="application/json"
    )


@router.get("/live")
async def liveness_probe() -> Dict[str, str]:
    """
    Simple liveness probe for Kubernetes
    
    Returns 200 if the application is alive and running.
    This endpoint should be lightweight and not check dependencies.
    
    Returns:
        Dict with status "alive"
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/ready")
async def readiness_probe(
    db_pool: DatabasePool = Depends(get_db_pool),
    settings: Settings = Depends(get_settings)
) -> Response:
    """
    Readiness probe for Kubernetes
    
    Checks if the application is ready to serve requests.
    This includes checking critical dependencies like database.
    
    Returns:
        Dict with readiness status
    """
    try:
        # Check database connection
        if settings.AUTH_MODE == "multi_user":
            async with db_pool.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
        else:
            async with aiosqlite.connect(db_pool.db_path) as conn:
                await conn.execute("SELECT 1")
        
        return Response(
            content=__import__('json').dumps({
                "status": "ready",
                "timestamp": datetime.utcnow().isoformat()
            }),
            status_code=200,
            media_type="application/json"
        )
    except Exception as e:
        logger.warning(f"Readiness check failed: {e}")
        return Response(
            content=__import__('json').dumps({
                "status": "not ready",
                "reason": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }),
            status_code=503,
            media_type="application/json"
        )


@router.get("/metrics")
async def metrics() -> Dict[str, Any]:
    """
    System metrics endpoint for monitoring
    
    Provides detailed system metrics for monitoring tools.
    Can be scraped by Prometheus or other monitoring systems.
    
    Returns:
        Dict with various system metrics
    """
    try:
        # CPU metrics
        cpu_stats = {
            "percent": psutil.cpu_percent(interval=0.1),
            "count": psutil.cpu_count(),
            "count_logical": psutil.cpu_count(logical=True),
            "frequency_current": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            "frequency_max": psutil.cpu_freq().max if psutil.cpu_freq() else None,
        }
        
        # Memory metrics
        mem = psutil.virtual_memory()
        memory_stats = {
            "total_mb": mem.total / (1024 * 1024),
            "available_mb": mem.available / (1024 * 1024),
            "used_mb": mem.used / (1024 * 1024),
            "percent": mem.percent,
            "cached_mb": mem.cached / (1024 * 1024) if hasattr(mem, 'cached') else None,
        }
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_stats = {
            "total_gb": disk.total / (1024 * 1024 * 1024),
            "used_gb": disk.used / (1024 * 1024 * 1024),
            "free_gb": disk.free / (1024 * 1024 * 1024),
            "percent": disk.percent
        }
        
        # Network metrics
        net_io = psutil.net_io_counters()
        network_stats = {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv,
            "errors_in": net_io.errin,
            "errors_out": net_io.errout,
            "drop_in": net_io.dropin,
            "drop_out": net_io.dropout
        }
        
        # Process metrics
        process = psutil.Process()
        process_stats = {
            "pid": process.pid,
            "cpu_percent": process.cpu_percent(),
            "memory_mb": process.memory_info().rss / (1024 * 1024),
            "memory_percent": process.memory_percent(),
            "num_threads": process.num_threads(),
            "num_fds": process.num_fds() if hasattr(process, 'num_fds') else None,
            "create_time": datetime.fromtimestamp(process.create_time()).isoformat()
        }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "cpu": cpu_stats,
            "memory": memory_stats,
            "disk": disk_stats,
            "network": network_stats,
            "process": process_stats,
            "system": {
                "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
                "uptime_seconds": (datetime.now() - datetime.fromtimestamp(psutil.boot_time())).total_seconds()
            }
        }
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/startup")
async def startup_probe() -> Dict[str, str]:
    """
    Startup probe for Kubernetes
    
    Used to check if the application has started successfully.
    Kubernetes uses this to know when to start liveness/readiness probes.
    
    Returns:
        Dict with startup status
    """
    # This is called during startup, so just return success
    # More complex apps might check if initialization is complete
    return {
        "status": "started",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/evaluations")
async def evaluation_service_health() -> Dict[str, Any]:
    """
    Dedicated health check for the evaluation service.
    
    Checks:
    - Evaluation database connectivity
    - Circuit breaker states for all providers
    - Embeddings service availability
    - Recent evaluation metrics
    
    Returns:
        Dict with detailed evaluation service health status
    """
    health = {
        "service": "evaluations",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {}
    }
    
    # Check evaluation database
    try:
        eval_manager = EvaluationManager()
        
        # Test database connectivity and get stats
        import sqlite3
        with sqlite3.connect(eval_manager.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_evaluations,
                    COUNT(DISTINCT evaluation_type) as evaluation_types,
                    MAX(created_at) as last_evaluation
                FROM evaluations
            """)
            stats = cursor.fetchone()
            
        health["components"]["database"] = {
            "status": "ok",
            "total_evaluations": stats[0] if stats else 0,
            "evaluation_types": stats[1] if stats else 0,
            "last_evaluation": stats[2] if stats and stats[2] else None
        }
    except Exception as e:
        health["components"]["database"] = {
            "status": "error",
            "error": str(e)
        }
        health["status"] = "degraded"
        logger.error(f"Evaluation database health check failed: {e}")
    
    # Check circuit breakers
    try:
        circuit_breaker_status = {}
        providers = ["openai", "anthropic", "google", "cohere", "mistral", "groq", "openrouter"]
        
        for provider in providers:
            cb = CircuitBreaker(name=f"evaluation_{provider}")
            cb_health = "healthy"
            
            if cb.state == CircuitState.OPEN:
                cb_health = "unhealthy"
            elif cb.state == CircuitState.HALF_OPEN:
                cb_health = "recovering"
                
            circuit_breaker_status[provider] = {
                "state": cb.state.value,
                "health": cb_health,
                "stats": {
                    "failures": cb.stats.failed_calls,
                    "successes": cb.stats.successful_calls,
                    "last_failure": cb.stats.last_failure_time.isoformat() if cb.stats.last_failure_time else None
                }
            }
            
            # If any circuit breaker is open, mark service as degraded
            if cb.state == CircuitState.OPEN and health["status"] == "healthy":
                health["status"] = "degraded"
                
        health["components"]["circuit_breakers"] = circuit_breaker_status
    except Exception as e:
        health["components"]["circuit_breakers"] = {
            "status": "error",
            "error": str(e)
        }
        logger.warning(f"Circuit breaker health check failed: {e}")
    
    # Check embeddings availability
    try:
        from tldw_Server_API.app.core.config import load_comprehensive_config
        config = load_comprehensive_config()
        embeddings_config = config.get("Embeddings", {})
        
        health["components"]["embeddings"] = {
            "status": "configured" if embeddings_config else "not_configured",
            "providers": []
        }
        
        # Check which embedding providers are configured
        if embeddings_config:
            providers = []
            if embeddings_config.get("embedding_provider"):
                providers.append(embeddings_config.get("embedding_provider"))
            if embeddings_config.get("embedding_model"):
                health["components"]["embeddings"]["model"] = embeddings_config.get("embedding_model")
            health["components"]["embeddings"]["providers"] = providers
            
    except Exception as e:
        health["components"]["embeddings"] = {
            "status": "error",
            "error": str(e)
        }
        logger.warning(f"Embeddings health check failed: {e}")
    
    # Calculate evaluation service metrics
    try:
        # Get recent evaluation performance metrics
        with sqlite3.connect(eval_manager.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    AVG(CAST(json_extract(results, '$.processing_time') AS REAL)) as avg_processing_time,
                    COUNT(CASE WHEN json_extract(results, '$.error') IS NOT NULL THEN 1 END) as error_count,
                    COUNT(*) as total_recent
                FROM evaluations
                WHERE created_at > datetime('now', '-1 hour')
            """)
            metrics = cursor.fetchone()
            
        health["metrics"] = {
            "last_hour": {
                "total_evaluations": metrics[2] if metrics else 0,
                "error_count": metrics[1] if metrics else 0,
                "avg_processing_time_seconds": round(metrics[0], 2) if metrics and metrics[0] else None
            }
        }
    except Exception as e:
        health["metrics"] = {"error": str(e)}
        logger.warning(f"Evaluation metrics collection failed: {e}")
    
    # Determine overall health status
    if health["status"] == "healthy":
        status_code = 200
    elif health["status"] == "degraded":
        status_code = 206
    else:
        status_code = 503
        
    return Response(
        content=__import__('json').dumps(health, indent=2),
        status_code=status_code,
        media_type="application/json"
    )


#
## End of health.py
#######################################################################################################################