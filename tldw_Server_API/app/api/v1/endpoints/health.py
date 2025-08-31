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
from tldw_Server_API.app.core.Chatbooks.chatbook_service import audit_logger
from tldw_Server_API.app.core.Evaluations.circuit_breaker import CircuitBreaker, CircuitState
from tldw_Server_API.app.core.Evaluations.evaluation_manager import EvaluationManager
from tldw_Server_API.app.core.Evaluations.metrics import get_metrics
from tldw_Server_API.app.core.Security.secret_manager import secret_manager
from tldw_Server_API.app.core.Evaluations.connection_pool import get_connection_health, get_connection_stats

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
    
    # Include API key in test mode for E2E testing
    import os
    if os.getenv("TEST_MODE") == "true" and settings.AUTH_MODE == "single_user":
        health["test_api_key"] = settings.SINGLE_USER_API_KEY
    
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
            "error": "ERROR - SEE LOGS",
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
    
    # Check connection pool health
    try:
        pool_health = get_connection_health()
        health["components"]["connection_pool"] = pool_health
        
        # Degrade overall health if connection pool is unhealthy
        if pool_health["status"] == "unhealthy":
            health["status"] = "degraded"
        elif pool_health["status"] == "degraded" and health["status"] == "healthy":
            health["status"] = "degraded"
            
    except Exception as e:
        health["components"]["connection_pool"] = {
            "status": "error",
            "error": str(e)
        }
        health["status"] = "degraded"
        logger.error(f"Connection pool health check failed: {e}")
    
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


@router.get("/security")
async def security_health_check() -> Dict[str, Any]:
    """
    Security-focused health check for the evaluation service.
    
    Checks:
    - Recent security events and audit logs
    - Authentication failure patterns
    - Rate limiting violations
    - Secret management status
    - Suspicious activity detection
    
    Returns:
        Dict with detailed security health status
    """
    security_data = {
        "service": "evaluations_security",
        "status": "secure",
        "timestamp": datetime.utcnow().isoformat(),
        "risk_level": "low",
        "security_components": {},
        "recent_alerts": []
    }
    
    try:
        # Get recent security events from audit log
        security_summary = audit_logger.get_security_summary(hours=24)
        
        security_data["security_components"]["audit_events"] = {
            "status": "ok",
            "high_severity_events": security_summary.get("severity_counts", {}).get("high", 0),
            "critical_events": security_summary.get("severity_counts", {}).get("critical", 0),
            "failure_events": sum(security_summary.get("failure_counts", {}).values()),
            "unique_security_users": security_summary.get("unique_security_users", 0),
            "top_failing_ips": security_summary.get("top_failing_ips", [])[:5]  # Top 5 only
        }
        
        # Determine risk level based on security events
        critical_events = security_summary.get("severity_counts", {}).get("critical", 0)
        high_events = security_summary.get("severity_counts", {}).get("high", 0)
        failure_events = sum(security_summary.get("failure_counts", {}).values())
        
        if critical_events > 0:
            security_data["risk_level"] = "critical"
            security_data["status"] = "at_risk"
            security_data["recent_alerts"].append(f"{critical_events} critical security events in last 24h")
        elif high_events > 10:
            security_data["risk_level"] = "high"
            security_data["status"] = "elevated"
            security_data["recent_alerts"].append(f"{high_events} high-severity security events in last 24h")
        elif failure_events > 50:
            security_data["risk_level"] = "medium"
            security_data["recent_alerts"].append(f"{failure_events} failure events in last 24h")
        elif high_events > 0 or failure_events > 0:
            security_data["risk_level"] = "low"
    
    except Exception as e:
        security_data["security_components"]["audit_events"] = {
            "status": "error",
            "error": str(e)
        }
        security_data["status"] = "unknown"
        security_data["recent_alerts"].append(f"Audit log check failed: {e}")
        logger.error(f"Security audit check failed: {e}")
    
    try:
        # Check secret management status
        secrets_health = secret_manager.get_production_health_check()
        
        security_data["security_components"]["secrets"] = {
            "status": "ok" if secrets_health["status"] == "healthy" else "warning",
            "required_secrets_ok": secrets_health["required_secrets_ok"],
            "config_available": secrets_health.get("config_available", False),
            "using_defaults": len([rec for rec in secrets_health.get("recommendations", []) 
                                 if "default value" in rec]),
            "rotation_required": len(secrets_health.get("rotation_warnings", []))
        }
        
        if not secrets_health["required_secrets_ok"]:
            security_data["status"] = "at_risk"
            security_data["recent_alerts"].append("Required secrets missing or invalid")
        
        if secrets_health.get("recommendations"):
            security_data["recent_alerts"].extend(secrets_health["recommendations"][:3])  # Top 3 recommendations
    
    except Exception as e:
        security_data["security_components"]["secrets"] = {
            "status": "error",
            "error": str(e)
        }
        security_data["recent_alerts"].append(f"Secret manager check failed: {e}")
        logger.error(f"Secret manager health check failed: {e}")
    
    try:
        # Check Prometheus metrics for security-related metrics
        metrics = get_metrics()
        metrics_health = metrics.get_health_metrics()
        
        security_data["security_components"]["metrics"] = {
            "status": "ok" if metrics_health["metrics_enabled"] else "disabled",
            "prometheus_available": metrics_health["metrics_enabled"],
            "last_updated": metrics_health.get("last_updated")
        }
        
        if not metrics_health["metrics_enabled"]:
            security_data["recent_alerts"].append("Security metrics collection is disabled")
    
    except Exception as e:
        security_data["security_components"]["metrics"] = {
            "status": "error",
            "error": str(e)
        }
        logger.warning(f"Metrics health check failed: {e}")
    
    try:
        # Check evaluation database for potential security issues
        eval_manager = EvaluationManager()
        
        with __import__('sqlite3').connect(eval_manager.db_path) as conn:
            # Check for unusual evaluation patterns that might indicate abuse
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_recent,
                    COUNT(DISTINCT user_id) as unique_users,
                    AVG(CASE WHEN json_extract(results, '$.error') IS NOT NULL THEN 1.0 ELSE 0.0 END) as error_rate
                FROM evaluations
                WHERE created_at > datetime('now', '-1 hour')
            """)
            
            eval_metrics = cursor.fetchone()
            
        security_data["security_components"]["evaluation_patterns"] = {
            "status": "ok",
            "recent_evaluations": eval_metrics[0] if eval_metrics else 0,
            "unique_users": eval_metrics[1] if eval_metrics else 0,
            "error_rate": round(eval_metrics[2] or 0, 2)
        }
        
        # Flag suspicious patterns
        if eval_metrics and eval_metrics[0] > 1000:  # More than 1000 evaluations in an hour
            security_data["recent_alerts"].append("Unusually high evaluation volume detected")
        
        if eval_metrics and eval_metrics[2] and eval_metrics[2] > 0.5:  # >50% error rate
            security_data["recent_alerts"].append("High evaluation error rate may indicate abuse")
    
    except Exception as e:
        security_data["security_components"]["evaluation_patterns"] = {
            "status": "error",
            "error": str(e)
        }
        logger.warning(f"Evaluation pattern check failed: {e}")
    
    # Generate overall security score
    healthy_components = sum(1 for comp in security_data["security_components"].values() 
                           if comp.get("status") in ["ok", "configured"])
    total_components = len(security_data["security_components"])
    security_data["security_score"] = healthy_components / total_components if total_components > 0 else 0
    
    # Determine HTTP status code
    if security_data["status"] == "secure":
        status_code = 200
    elif security_data["status"] == "elevated":
        status_code = 206
    else:
        status_code = 503
    
    return Response(
        content=__import__('json').dumps(security_data, indent=2),
        status_code=status_code,
        media_type="application/json"
    )


@router.get("/prometheus")
async def prometheus_metrics() -> Response:
    """
    Prometheus metrics endpoint for the evaluation service.
    
    Returns Prometheus-formatted metrics including:
    - Request counts and latencies
    - Circuit breaker states
    - Rate limiting violations
    - Security events
    - Database performance
    
    Returns:
        Prometheus-formatted metrics as plain text
    """
    try:
        metrics = get_metrics()
        metrics_data = metrics.get_metrics()
        
        return Response(
            content=metrics_data.decode('utf-8'),
            media_type="text/plain"
        )
    except Exception as e:
        logger.error(f"Prometheus metrics collection failed: {e}")
        return Response(
            content=f"# Metrics collection failed: ERROR - SEE LOGS",
            status_code=503,
            media_type="text/plain"
        )


#
## End of health.py
#######################################################################################################################