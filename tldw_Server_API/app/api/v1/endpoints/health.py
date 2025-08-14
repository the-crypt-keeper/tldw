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


#
## End of health.py
#######################################################################################################################