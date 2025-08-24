"""
Main MCP Server implementation for unified module

Handles WebSocket and HTTP connections with production-ready features.
"""

import asyncio
import json
from typing import Dict, Any, Optional, Set, List
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from loguru import logger

from .config import get_config
from .protocol import MCPProtocol, MCPRequest, MCPResponse, RequestContext
from .modules.registry import get_module_registry
from .auth.jwt_manager import get_jwt_manager
from .auth.rbac import get_rbac_policy
from .auth.rate_limiter import get_rate_limiter, RateLimitExceeded


class WebSocketConnection:
    """Manages a single WebSocket connection"""
    
    def __init__(
        self,
        websocket: WebSocket,
        connection_id: str,
        client_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        self.websocket = websocket
        self.connection_id = connection_id
        self.client_id = client_id
        self.user_id = user_id
        self.connected_at = datetime.now(timezone.utc)
        self.last_activity = self.connected_at
        self.message_count = 0
        self.error_count = 0
    
    async def send_json(self, data: Dict[str, Any]):
        """Send JSON data to client"""
        try:
            await self.websocket.send_json(data)
            self.last_activity = datetime.now(timezone.utc)
        except Exception as e:
            logger.error(f"Error sending to WebSocket {self.connection_id}: {e}")
            self.error_count += 1
            raise
    
    async def receive_json(self) -> Dict[str, Any]:
        """Receive JSON data from client"""
        try:
            data = await self.websocket.receive_json()
            self.last_activity = datetime.now(timezone.utc)
            self.message_count += 1
            return data
        except Exception as e:
            logger.error(f"Error receiving from WebSocket {self.connection_id}: {e}")
            self.error_count += 1
            raise
    
    async def close(self, code: int = 1000, reason: str = ""):
        """Close the WebSocket connection"""
        try:
            await self.websocket.close(code=code, reason=reason)
        except Exception as e:
            logger.error(f"Error closing WebSocket {self.connection_id}: {e}")


class MCPServer:
    """
    Production-ready MCP Server with WebSocket and HTTP support.
    
    Features:
    - WebSocket connection management
    - HTTP request handling
    - Connection pooling
    - Graceful shutdown
    - Health monitoring
    - Metrics collection
    """
    
    def __init__(self):
        self.config = get_config()
        self.protocol = MCPProtocol()
        self.module_registry = get_module_registry()
        self.jwt_manager = get_jwt_manager()
        self.rbac_policy = get_rbac_policy()
        self.rate_limiter = get_rate_limiter()
        
        # Connection management
        self.connections: Dict[str, WebSocketConnection] = {}
        self.connection_lock = asyncio.Lock()
        
        # Server state
        self.initialized = False
        self.startup_time = datetime.now(timezone.utc)
        self.shutdown_event = asyncio.Event()
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        
        logger.info("MCP Server created")
    
    async def initialize(self):
        """Initialize the server and all modules"""
        if self.initialized:
            logger.warning("Server already initialized")
            return
        
        logger.info("Initializing MCP Server")
        
        try:
            # Start module health monitoring
            await self.module_registry.start_health_monitoring()
            
            # Register default modules (will be implemented when migrating modules)
            await self._register_default_modules()
            
            # Start background tasks
            self._start_background_tasks()
            
            self.initialized = True
            logger.info("MCP Server initialized successfully")
            
        except Exception as e:
            logger.error(f"Server initialization failed: {e}")
            raise
    
    async def shutdown(self):
        """Gracefully shutdown the server"""
        logger.info("Shutting down MCP Server")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Close all WebSocket connections
        await self._close_all_connections()
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Shutdown modules
        await self.module_registry.shutdown_all()
        
        self.initialized = False
        logger.info("MCP Server shutdown complete")
    
    async def _register_default_modules(self):
        """Register default modules (placeholder for migration)"""
        # This will be implemented when migrating modules
        logger.info("Default modules registration placeholder")
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        # Connection cleanup task
        task = asyncio.create_task(self._connection_cleanup_loop())
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        
        # Metrics collection task
        task = asyncio.create_task(self._metrics_collection_loop())
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
    
    async def _connection_cleanup_loop(self):
        """Periodically clean up stale connections"""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_stale_connections()
            except Exception as e:
                logger.error(f"Error in connection cleanup: {e}")
    
    async def _cleanup_stale_connections(self):
        """Remove stale WebSocket connections"""
        async with self.connection_lock:
            stale_connections = []
            current_time = datetime.now(timezone.utc)
            
            for conn_id, connection in self.connections.items():
                # Check for stale connections (no activity for 5 minutes)
                if (current_time - connection.last_activity).total_seconds() > 300:
                    stale_connections.append(conn_id)
                
                # Check for error threshold
                elif connection.error_count > 10:
                    stale_connections.append(conn_id)
            
            # Close stale connections
            for conn_id in stale_connections:
                logger.info(f"Closing stale connection: {conn_id}")
                connection = self.connections[conn_id]
                await connection.close(code=1001, reason="Connection timeout")
                del self.connections[conn_id]
    
    async def _metrics_collection_loop(self):
        """Periodically collect and log metrics"""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self._log_metrics()
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
    
    async def _log_metrics(self):
        """Log server metrics"""
        metrics = await self.get_metrics()
        logger.info(f"Server metrics: {metrics}")
    
    async def handle_websocket(
        self,
        websocket: WebSocket,
        client_id: Optional[str] = None,
        auth_token: Optional[str] = None
    ):
        """
        Handle a WebSocket connection.
        
        Args:
            websocket: FastAPI WebSocket instance
            client_id: Optional client identifier
            auth_token: Optional authentication token
        """
        connection_id = f"ws_{client_id or 'anonymous'}_{datetime.now().timestamp()}"
        user_id = None
        
        # Authenticate if token provided
        if auth_token:
            try:
                token_data = self.jwt_manager.verify_token(auth_token)
                user_id = token_data.sub
                logger.info(f"WebSocket authenticated for user: {user_id}")
            except Exception as e:
                logger.warning(f"WebSocket authentication failed: {e}")
                await websocket.close(code=1008, reason="Authentication failed")
                return
        
        # Check connection limit
        async with self.connection_lock:
            if len(self.connections) >= self.config.ws_max_connections:
                logger.warning("Maximum WebSocket connections reached")
                await websocket.close(code=1013, reason="Server at capacity")
                return
            
            # Accept connection
            await websocket.accept()
            
            # Create connection object
            connection = WebSocketConnection(
                websocket=websocket,
                connection_id=connection_id,
                client_id=client_id,
                user_id=user_id
            )
            
            self.connections[connection_id] = connection
        
        logger.info(
            f"WebSocket connected: {connection_id} "
            f"(client={client_id}, user={user_id})"
        )
        
        try:
            # Start ping task
            ping_task = asyncio.create_task(
                self._websocket_ping_loop(connection)
            )
            
            # Handle messages
            await self._handle_websocket_messages(connection)
            
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {connection_id}")
        except Exception as e:
            logger.error(f"WebSocket error for {connection_id}: {e}")
            await connection.close(code=1011, reason="Internal error")
        finally:
            # Cancel ping task
            ping_task.cancel()
            
            # Remove connection
            async with self.connection_lock:
                if connection_id in self.connections:
                    del self.connections[connection_id]
            
            logger.info(f"WebSocket cleanup complete: {connection_id}")
    
    async def _websocket_ping_loop(self, connection: WebSocketConnection):
        """Send periodic pings to keep connection alive"""
        while True:
            try:
                await asyncio.sleep(self.config.ws_ping_interval)
                await connection.websocket.send_json({"type": "ping"})
            except Exception:
                break
    
    async def _handle_websocket_messages(self, connection: WebSocketConnection):
        """Handle incoming WebSocket messages"""
        while True:
            # Receive message
            try:
                data = await connection.receive_json()
            except json.JSONDecodeError as e:
                await connection.send_json({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32700,
                        "message": f"Parse error: {str(e)}"
                    }
                })
                continue
            
            # Check message size
            message_size = len(json.dumps(data))
            if message_size > self.config.ws_max_message_size:
                await connection.send_json({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32600,
                        "message": "Message too large"
                    },
                    "id": data.get("id") if isinstance(data, dict) else None
                })
                continue
            
            # Handle ping/pong
            if isinstance(data, dict) and data.get("type") == "pong":
                continue
            
            # Create request context
            context = RequestContext(
                request_id=data.get("id", "unknown") if isinstance(data, dict) else "unknown",
                user_id=connection.user_id,
                client_id=connection.client_id,
                session_id=connection.connection_id
            )
            
            # Process MCP request
            try:
                response = await self.protocol.process_request(data, context)
                await connection.send_json(response.dict())
            except RateLimitExceeded as e:
                await connection.send_json({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32002,
                        "message": f"Rate limit exceeded. Retry after {e.retry_after} seconds"
                    },
                    "id": data.get("id") if isinstance(data, dict) else None
                })
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                await connection.send_json({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": "Internal error"
                    },
                    "id": data.get("id") if isinstance(data, dict) else None
                })
    
    async def handle_http_request(
        self,
        request: MCPRequest,
        client_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> MCPResponse:
        """
        Handle an HTTP MCP request.
        
        Args:
            request: MCP request
            client_id: Optional client identifier
            user_id: Optional user identifier (from auth)
        
        Returns:
            MCP response
        """
        # Create request context
        context = RequestContext(
            request_id=request.id or "http_request",
            user_id=user_id,
            client_id=client_id
        )
        
        # Process request
        try:
            response = await self.protocol.process_request(request, context)
            return response
        except RateLimitExceeded as e:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Retry after {e.retry_after} seconds"
            )
        except Exception as e:
            logger.error(f"Error processing HTTP request: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
    
    async def _close_all_connections(self):
        """Close all WebSocket connections"""
        async with self.connection_lock:
            tasks = []
            for connection in self.connections.values():
                tasks.append(
                    connection.close(code=1001, reason="Server shutdown")
                )
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            self.connections.clear()
    
    async def get_status(self) -> Dict[str, Any]:
        """Get server status"""
        uptime = (datetime.now(timezone.utc) - self.startup_time).total_seconds()
        
        # Get module health
        health_results = await self.module_registry.check_all_health()
        
        # Get connection stats
        connection_stats = {
            "total": len(self.connections),
            "authenticated": sum(1 for c in self.connections.values() if c.user_id),
            "anonymous": sum(1 for c in self.connections.values() if not c.user_id)
        }
        
        return {
            "status": "healthy" if self.initialized else "initializing",
            "version": "3.0.0",
            "uptime_seconds": uptime,
            "connections": connection_stats,
            "modules": {
                "total": len(health_results),
                "healthy": sum(1 for h in health_results.values() if h.is_healthy),
                "degraded": sum(1 for h in health_results.values() if h.is_operational and not h.is_healthy),
                "unhealthy": sum(1 for h in health_results.values() if not h.is_operational)
            }
        }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics"""
        # Collect module metrics
        module_metrics = {}
        modules = await self.module_registry.get_all_modules()
        
        for module_id, module in modules.items():
            metrics = module.get_metrics()
            module_metrics[module_id] = {
                "requests": metrics.total_requests,
                "errors": metrics.failed_requests,
                "error_rate": metrics.error_rate,
                "avg_latency_ms": metrics.avg_latency_ms
            }
        
        # Connection metrics
        total_messages = sum(c.message_count for c in self.connections.values())
        total_errors = sum(c.error_count for c in self.connections.values())
        
        return {
            "connections": {
                "active": len(self.connections),
                "total_messages": total_messages,
                "total_errors": total_errors
            },
            "modules": module_metrics
        }


# Singleton instance management
_server: Optional[MCPServer] = None


def get_mcp_server() -> MCPServer:
    """Get or create MCP server singleton"""
    global _server
    if _server is None:
        _server = MCPServer()
    return _server


@asynccontextmanager
async def lifespan(app):
    """
    FastAPI lifespan manager for server initialization and shutdown.
    
    Usage in main.py:
    ```python
    from tldw_Server_API.app.core.MCP_unified.server import lifespan
    
    app = FastAPI(lifespan=lifespan)
    ```
    """
    # Startup
    server = get_mcp_server()
    await server.initialize()
    logger.info("MCP Server started")
    
    yield
    
    # Shutdown
    await server.shutdown()
    logger.info("MCP Server stopped")