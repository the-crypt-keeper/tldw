"""
Health check system for RAG components.

Provides health status monitoring for all RAG subsystems.
"""

import time
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import psutil
import traceback

from .circuit_breaker import get_all_circuit_breaker_stats



class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health status of a component."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    last_check: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "last_check": self.last_check,
            "age_seconds": time.time() - self.last_check
        }


class RAGHealthChecker:
    """
    Health checker for RAG system components.
    
    Monitors the health of:
    - Embeddings service
    - Vector store
    - Cache
    - Database connections
    - Circuit breakers
    - System resources
    """
    
    def __init__(self, rag_service=None):
        """
        Initialize health checker.
        
        Args:
            rag_service: RAG service instance to monitor
        """
        self.rag_service = rag_service
        self._last_check_time = 0
        self._cached_health: Optional[Dict[str, Any]] = None
        self._cache_duration = 5.0  # Cache health for 5 seconds
    
    async def check_embeddings_health(self) -> ComponentHealth:
        """Check embeddings service health."""
        try:
            if not self.rag_service or not self.rag_service.embeddings:
                return ComponentHealth(
                    name="embeddings",
                    status=HealthStatus.UNHEALTHY,
                    message="Embeddings service not initialized",
                    details={},
                    last_check=time.time()
                )
            
            # Try to get embedding dimension (lightweight check)
            start = time.time()
            dim = self.rag_service.embeddings.get_embedding_dimension()
            latency = time.time() - start
            
            if dim is None:
                status = HealthStatus.UNHEALTHY
                message = "Cannot determine embedding dimension"
            elif latency > 1.0:
                status = HealthStatus.DEGRADED
                message = f"High latency: {latency:.2f}s"
            else:
                status = HealthStatus.HEALTHY
                message = "Embeddings service operational"
            
            # Get service metrics
            metrics = self.rag_service.embeddings.get_metrics()
            
            return ComponentHealth(
                name="embeddings",
                status=status,
                message=message,
                details={
                    "dimension": dim,
                    "latency_seconds": latency,
                    "total_calls": metrics.get("total_calls", 0),
                    "error_rate": metrics.get("error_rate", 0),
                    "cache_hit_rate": metrics.get("cache_hit_rate", 0)
                },
                last_check=time.time()
            )
            
        except Exception as e:
            logger.error(f"Health check failed for embeddings: {e}")
            return ComponentHealth(
                name="embeddings",
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e)},
                last_check=time.time()
            )
    
    async def check_vector_store_health(self) -> ComponentHealth:
        """Check vector store health."""
        try:
            if not self.rag_service or not self.rag_service.vector_store:
                return ComponentHealth(
                    name="vector_store",
                    status=HealthStatus.UNHEALTHY,
                    message="Vector store not initialized",
                    details={},
                    last_check=time.time()
                )
            
            # Get collection stats
            start = time.time()
            stats = self.rag_service.vector_store.get_collection_stats()
            latency = time.time() - start
            
            if "error" in stats:
                status = HealthStatus.UNHEALTHY
                message = f"Vector store error: {stats['error']}"
            elif stats.get("count", 0) == 0:
                status = HealthStatus.DEGRADED
                message = "Vector store empty"
            elif latency > 0.5:
                status = HealthStatus.DEGRADED
                message = f"High latency: {latency:.2f}s"
            else:
                status = HealthStatus.HEALTHY
                message = f"Vector store operational ({stats.get('count', 0)} documents)"
            
            return ComponentHealth(
                name="vector_store",
                status=status,
                message=message,
                details={
                    "document_count": stats.get("count", 0),
                    "latency_seconds": latency,
                    "collection": stats.get("name", "unknown")
                },
                last_check=time.time()
            )
            
        except Exception as e:
            logger.error(f"Health check failed for vector store: {e}")
            return ComponentHealth(
                name="vector_store",
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e)},
                last_check=time.time()
            )
    
    async def check_cache_health(self) -> ComponentHealth:
        """Check cache health."""
        try:
            if not self.rag_service or not self.rag_service.cache:
                return ComponentHealth(
                    name="cache",
                    status=HealthStatus.UNHEALTHY,
                    message="Cache not initialized",
                    details={},
                    last_check=time.time()
                )
            
            metrics = self.rag_service.cache.get_metrics()
            
            if not metrics.get("enabled", False):
                status = HealthStatus.DEGRADED
                message = "Cache disabled"
            elif metrics.get("hit_rate", 0) < 0.1 and metrics.get("total_requests", 0) > 100:
                status = HealthStatus.DEGRADED
                message = f"Low hit rate: {metrics['hit_rate']:.1%}"
            else:
                status = HealthStatus.HEALTHY
                message = f"Cache operational (hit rate: {metrics.get('hit_rate', 0):.1%})"
            
            return ComponentHealth(
                name="cache",
                status=status,
                message=message,
                details={
                    "size": metrics.get("size", 0),
                    "max_size": metrics.get("max_size", 0),
                    "hit_rate": metrics.get("hit_rate", 0),
                    "memory_mb": metrics.get("size_bytes", 0) / (1024 * 1024)
                },
                last_check=time.time()
            )
            
        except Exception as e:
            logger.error(f"Health check failed for cache: {e}")
            return ComponentHealth(
                name="cache",
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e)},
                last_check=time.time()
            )
    
    def check_circuit_breakers_health(self) -> ComponentHealth:
        """Check circuit breakers health."""
        try:
            breakers = get_all_circuit_breaker_stats()
            
            open_breakers = []
            half_open_breakers = []
            
            for name, stats in breakers.items():
                if stats["state"] == "open":
                    open_breakers.append(name)
                elif stats["state"] == "half_open":
                    half_open_breakers.append(name)
            
            if open_breakers:
                status = HealthStatus.UNHEALTHY
                message = f"Circuit breakers open: {', '.join(open_breakers)}"
            elif half_open_breakers:
                status = HealthStatus.DEGRADED
                message = f"Circuit breakers recovering: {', '.join(half_open_breakers)}"
            else:
                status = HealthStatus.HEALTHY
                message = "All circuit breakers closed"
            
            return ComponentHealth(
                name="circuit_breakers",
                status=status,
                message=message,
                details={
                    "total": len(breakers),
                    "open": len(open_breakers),
                    "half_open": len(half_open_breakers),
                    "breakers": breakers
                },
                last_check=time.time()
            )
            
        except Exception as e:
            logger.error(f"Health check failed for circuit breakers: {e}")
            return ComponentHealth(
                name="circuit_breakers",
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e)},
                last_check=time.time()
            )
    
    def check_system_resources_health(self) -> ComponentHealth:
        """Check system resources health."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            issues = []
            
            if cpu_percent > 90:
                issues.append(f"High CPU usage: {cpu_percent}%")
            if memory.percent > 90:
                issues.append(f"High memory usage: {memory.percent}%")
            if disk.percent > 90:
                issues.append(f"Low disk space: {disk.percent}% used")
            
            if len(issues) >= 2:
                status = HealthStatus.UNHEALTHY
                message = "; ".join(issues)
            elif issues:
                status = HealthStatus.DEGRADED
                message = "; ".join(issues)
            else:
                status = HealthStatus.HEALTHY
                message = "System resources normal"
            
            return ComponentHealth(
                name="system_resources",
                status=status,
                message=message,
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": disk.percent,
                    "disk_free_gb": disk.free / (1024**3)
                },
                last_check=time.time()
            )
            
        except Exception as e:
            logger.error(f"Health check failed for system resources: {e}")
            return ComponentHealth(
                name="system_resources",
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e)},
                last_check=time.time()
            )
    
    async def check_all_health(self) -> Dict[str, Any]:
        """
        Check health of all components.
        
        Returns:
            Dictionary with overall health status and component details
        """
        # Check cache
        if (self._cached_health and 
            time.time() - self._last_check_time < self._cache_duration):
            return self._cached_health
        
        # Run all health checks
        checks = await asyncio.gather(
            self.check_embeddings_health(),
            self.check_vector_store_health(),
            self.check_cache_health(),
            return_exceptions=True
        )
        
        # Add synchronous checks
        checks.extend([
            self.check_circuit_breakers_health(),
            self.check_system_resources_health()
        ])
        
        # Process results
        components = []
        unhealthy_count = 0
        degraded_count = 0
        
        for check in checks:
            if isinstance(check, Exception):
                # Handle failed checks
                components.append(ComponentHealth(
                    name="unknown",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {str(check)}",
                    details={"error": traceback.format_exc()},
                    last_check=time.time()
                ))
                unhealthy_count += 1
            else:
                components.append(check)
                if check.status == HealthStatus.UNHEALTHY:
                    unhealthy_count += 1
                elif check.status == HealthStatus.DEGRADED:
                    degraded_count += 1
        
        # Determine overall status
        if unhealthy_count > 0:
            overall_status = HealthStatus.UNHEALTHY
            overall_message = f"{unhealthy_count} components unhealthy"
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED
            overall_message = f"{degraded_count} components degraded"
        else:
            overall_status = HealthStatus.HEALTHY
            overall_message = "All components healthy"
        
        health_report = {
            "status": overall_status.value,
            "message": overall_message,
            "timestamp": time.time(),
            "components": [c.to_dict() for c in components],
            "summary": {
                "total": len(components),
                "healthy": len([c for c in components if c.status == HealthStatus.HEALTHY]),
                "degraded": degraded_count,
                "unhealthy": unhealthy_count
            }
        }
        
        # Cache the result
        self._cached_health = health_report
        self._last_check_time = time.time()
        
        return health_report
    
    def get_health_sync(self) -> Dict[str, Any]:
        """Synchronous wrapper for health check."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.check_all_health())
        finally:
            loop.close()


# Global health checker instance
_health_checker: Optional[RAGHealthChecker] = None


def init_health_checker(rag_service):
    """Initialize the global health checker."""
    global _health_checker
    _health_checker = RAGHealthChecker(rag_service)


def get_health_status() -> Dict[str, Any]:
    """Get current health status."""
    if not _health_checker:
        return {
            "status": "unhealthy",
            "message": "Health checker not initialized",
            "timestamp": time.time(),
            "components": []
        }
    
    return _health_checker.get_health_sync()