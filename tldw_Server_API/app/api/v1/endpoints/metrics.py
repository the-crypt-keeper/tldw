# metrics.py
# Metrics endpoint for Prometheus and health monitoring

from fastapi import APIRouter, Response, HTTPException, status
from typing import Dict, Any, Optional

from loguru import logger

from tldw_Server_API.app.core.Chat.chat_metrics import get_chat_metrics
from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry

router = APIRouter(tags=["Metrics"])


@router.get("/metrics", 
            summary="Get metrics in Prometheus format",
            response_class=Response)
async def get_prometheus_metrics() -> Response:
    """
    Export all metrics in Prometheus text format.
    
    This endpoint provides metrics for monitoring the application's performance,
    including:
    - Request rates and latencies
    - LLM API usage and costs
    - Database operations
    - Chat-specific metrics
    - System resource usage
    
    The format is compatible with Prometheus scrapers.
    """
    try:
        registry = get_metrics_registry()
        prometheus_text = registry.export_prometheus_format()
        
        return Response(
            content=prometheus_text,
            media_type="text/plain; version=0.0.4",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    except Exception as e:
        logger.error(f"Error exporting metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export metrics"
        )


@router.get("/metrics/json",
            summary="Get metrics in JSON format",
            response_model=Dict[str, Any])
async def get_json_metrics() -> Dict[str, Any]:
    """
    Get all metrics in JSON format.
    
    This provides a more detailed view of metrics with statistics,
    useful for debugging and custom monitoring solutions.
    """
    try:
        registry = get_metrics_registry()
        chat_metrics = get_chat_metrics()
        
        all_metrics = registry.get_all_metrics()
        active_operations = chat_metrics.get_active_metrics()
        
        return {
            "metrics": all_metrics,
            "active_operations": active_operations,
            "timestamp": None  # Will be set to current time by FastAPI
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve metrics"
        )


@router.get("/metrics/health",
            summary="Health check with metrics",
            response_model=Dict[str, Any])
async def health_check_with_metrics() -> Dict[str, Any]:
    """
    Health check endpoint with basic metrics.
    
    Returns the health status of the application along with
    key operational metrics.
    """
    try:
        chat_metrics = get_chat_metrics()
        active = chat_metrics.get_active_metrics()
        
        # Determine health status based on active operations
        status = "healthy"
        if active["active_requests"] > 100:
            status = "degraded"
        elif active["active_requests"] > 200:
            status = "unhealthy"
        
        return {
            "status": status,
            "active_requests": active["active_requests"],
            "active_streams": active["active_streams"],
            "active_transactions": active["active_transactions"],
            "message": "Service is operational"
        }
    except Exception as e:
        logger.error(f"Metrics Health check failed: {e}")
        return {
            "status": "unhealthy",
            "message": f"Metrics Health check failed: ERROR - SEE LOGS",
            "active_requests": -1,
            "active_streams": -1,
            "active_transactions": -1
        }


@router.get("/metrics/chat",
            summary="Get chat-specific metrics",
            response_model=Dict[str, Any])
async def get_chat_metrics_endpoint() -> Dict[str, Any]:
    """
    Get detailed chat-specific metrics.
    
    This endpoint provides metrics specifically related to the chat
    functionality, including:
    - Request counts by provider and model
    - Token usage and costs
    - Streaming statistics
    - Character and conversation metrics
    """
    try:
        chat_metrics = get_chat_metrics()
        registry = get_metrics_registry()
        
        # Extract chat-specific metrics
        chat_metric_names = [
            "chat_requests_total",
            "chat_request_duration_seconds",
            "chat_tokens_prompt",
            "chat_tokens_completion",
            "chat_llm_cost_estimate_usd",
            "chat_streaming_duration_seconds",
            "chat_conversations_created_total",
            "chat_messages_saved_total",
            "chat_errors_total"
        ]
        
        chat_stats = {}
        for metric_name in chat_metric_names:
            stats = registry.get_metric_stats(metric_name)
            if stats:
                chat_stats[metric_name] = stats
        
        active = chat_metrics.get_active_metrics()
        
        return {
            "active_operations": active,
            "metrics": chat_stats,
            "token_costs": chat_metrics.token_costs  # Model pricing info
        }
    except Exception as e:
        logger.error(f"Error getting chat metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve chat metrics"
        )


@router.post("/metrics/reset",
             summary="Reset metrics (admin only)",
             response_model=Dict[str, str])
async def reset_metrics() -> Dict[str, str]:
    """
    Reset all metrics to their initial state.
    
    WARNING: This will clear all historical metrics data.
    This endpoint should be protected with admin authentication
    in production.
    """
    try:
        # In production, add authentication check here
        # if not is_admin(request):
        #     raise HTTPException(status_code=403, detail="Admin access required")
        
        # Reinitialize metrics
        registry = get_metrics_registry()
        chat_metrics = get_chat_metrics()
        
        # Clear values
        registry.values.clear()
        
        # Reset active counters
        chat_metrics.active_requests = 0
        chat_metrics.active_streams = 0
        chat_metrics.active_transactions = 0
        
        logger.info("Metrics reset by admin")
        
        return {
            "status": "success",
            "message": "All metrics have been reset"
        }
    except Exception as e:
        logger.error(f"Error resetting metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset metrics"
        )