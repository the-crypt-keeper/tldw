# mcp_server.py - Main MCP Server Implementation
"""
Model Context Protocol (MCP) server implementation.

This module provides the main MCP server that handles:
- WebSocket connections
- Message routing
- Session management
- Integration with tools and context
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, Set, List
from datetime import datetime
import weakref

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from .mcp_protocol import (
    MCPMessage,
    MCPMessageType,
    MCPRequest,
    MCPResponse,
    MCPError,
    MCPErrorCode,
    MCPCapabilities,
    MCPConnectRequest,
    MCPConnectResponse,
    MCPToolExecutionRequest,
    MCPToolExecutionResult
)
from .mcp_tools import ToolRegistry, tool_registry
from .mcp_context import ContextManager, ContextInjector

logger = logging.getLogger(__name__)


class MCPSession:
    """Represents an MCP client session"""
    
    def __init__(
        self,
        session_id: str,
        client_id: str,
        websocket: WebSocket,
        capabilities: MCPCapabilities
    ):
        self.session_id = session_id
        self.client_id = client_id
        self.websocket = websocket
        self.capabilities = capabilities
        self.connected_at = datetime.utcnow()
        self.last_ping = time.time()
        self.context_ids: Set[str] = set()
        self.metadata: Dict[str, Any] = {}
    
    async def send_message(self, message: MCPMessage):
        """Send message to client"""
        try:
            await self.websocket.send_json(message.dict())
        except Exception as e:
            logger.error(f"Failed to send message to {self.client_id}: {e}")
            raise
    
    async def close(self, code: int = 1000, reason: str = "Normal closure"):
        """Close the WebSocket connection"""
        if self.websocket.application_state == WebSocketState.CONNECTED:
            await self.websocket.close(code=code, reason=reason)


class MCPServer:
    """Main MCP server implementation"""
    
    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        context_manager: Optional[ContextManager] = None,
        max_sessions: int = 1000,
        ping_interval: int = 30,
        ping_timeout: int = 60,
        max_message_size: int = 1024 * 1024  # 1MB
    ):
        self.tool_registry = tool_registry or ToolRegistry()
        self.context_manager = context_manager or ContextManager()
        self.max_sessions = max_sessions
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.max_message_size = max_message_size
        
        # Session management
        self._sessions: Dict[str, MCPSession] = {}
        self._client_sessions: Dict[str, Set[str]] = {}  # client_id -> session_ids
        
        # Server capabilities
        self._capabilities = MCPCapabilities(
            tools=True,
            context=True,
            resources=True,
            streaming=True,
            max_context_size=self.context_manager._max_context_size,
            server_info={
                "name": "tldw_server MCP",
                "version": "0.1.0"
            }
        )
        
        # Message handlers
        self._handlers = {
            MCPMessageType.CONNECT: self._handle_connect,
            MCPMessageType.DISCONNECT: self._handle_disconnect,
            MCPMessageType.PING: self._handle_ping,
            MCPMessageType.LIST_TOOLS: self._handle_list_tools,
            MCPMessageType.EXECUTE_TOOL: self._handle_execute_tool,
            MCPMessageType.GET_CONTEXT: self._handle_get_context,
            MCPMessageType.UPDATE_CONTEXT: self._handle_update_context,
            MCPMessageType.CLEAR_CONTEXT: self._handle_clear_context,
            MCPMessageType.CAPABILITIES: self._handle_capabilities,
        }
        
        # Add context injection middleware
        self.tool_registry.add_middleware(
            ContextInjector(self.context_manager)
        )
        
        # Background tasks
        self._ping_task = asyncio.create_task(self._ping_loop())
    
    async def handle_websocket(self, websocket: WebSocket):
        """Handle WebSocket connection"""
        await websocket.accept()
        session = None
        
        try:
            # Wait for connection request
            raw_message = await websocket.receive_json()
            message = MCPRequest(**raw_message)
            
            if message.type != MCPMessageType.CONNECT:
                await self._send_error(
                    websocket,
                    message.id,
                    MCPErrorCode.INVALID_REQUEST,
                    "First message must be CONNECT"
                )
                await websocket.close(code=1002)
                return
            
            # Handle connection
            connect_request = MCPConnectRequest(**message.params)
            session = await self._create_session(
                connect_request.client_id,
                websocket
            )
            
            # Send connection response
            response = MCPResponse(
                type=MCPMessageType.CONNECT,
                request_id=message.id,
                result=MCPConnectResponse(
                    session_id=session.session_id,
                    capabilities=self._capabilities,
                    welcome_message=f"Connected to tldw_server MCP v0.1.0"
                ).dict()
            )
            await session.send_message(response)
            
            # Main message loop
            await self._message_loop(session)
            
        except WebSocketDisconnect:
            logger.info(f"Client disconnected: {session.client_id if session else 'unknown'}")
        except Exception as e:
            logger.error(f"WebSocket error: {e}", exc_info=True)
        finally:
            if session:
                await self._cleanup_session(session)
    
    async def _message_loop(self, session: MCPSession):
        """Main message processing loop"""
        while True:
            try:
                # Receive message with size limit check
                raw_message = await session.websocket.receive_json()
                
                # Parse and validate message
                try:
                    message = MCPRequest(**raw_message)
                except Exception as e:
                    await self._send_error(
                        session.websocket,
                        raw_message.get('id', 'unknown'),
                        MCPErrorCode.INVALID_REQUEST,
                        f"Invalid message format: {e}"
                    )
                    continue
                
                # Update last activity
                session.last_ping = time.time()
                
                # Route message to handler
                handler = self._handlers.get(message.type)
                if handler:
                    await handler(session, message)
                else:
                    await self._send_error(
                        session.websocket,
                        message.id,
                        MCPErrorCode.METHOD_NOT_FOUND,
                        f"Unknown message type: {message.type}"
                    )
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                await self._send_error(
                    session.websocket,
                    'unknown',
                    MCPErrorCode.INTERNAL_ERROR,
                    "Internal server error"
                )
    
    async def _create_session(
        self,
        client_id: str,
        websocket: WebSocket
    ) -> MCPSession:
        """Create a new session"""
        if len(self._sessions) >= self.max_sessions:
            raise ValueError("Maximum sessions reached")
        
        import uuid
        session_id = str(uuid.uuid4())
        
        session = MCPSession(
            session_id=session_id,
            client_id=client_id,
            websocket=websocket,
            capabilities=self._capabilities
        )
        
        self._sessions[session_id] = session
        
        if client_id not in self._client_sessions:
            self._client_sessions[client_id] = set()
        self._client_sessions[client_id].add(session_id)
        
        logger.info(f"Created session {session_id} for client {client_id}")
        return session
    
    async def _cleanup_session(self, session: MCPSession):
        """Clean up session resources"""
        # Remove from registries
        self._sessions.pop(session.session_id, None)
        
        if session.client_id in self._client_sessions:
            self._client_sessions[session.client_id].discard(session.session_id)
            if not self._client_sessions[session.client_id]:
                del self._client_sessions[session.client_id]
        
        # Clean up contexts
        await self.context_manager.cleanup_session(session.session_id)
        
        # Close WebSocket if still open
        await session.close()
        
        logger.info(f"Cleaned up session {session.session_id}")
    
    async def _handle_connect(self, session: MCPSession, message: MCPRequest):
        """Handle connect message (already connected)"""
        response = MCPResponse(
            type=MCPMessageType.CONNECT,
            request_id=message.id,
            result={"status": "already_connected", "session_id": session.session_id}
        )
        await session.send_message(response)
    
    async def _handle_disconnect(self, session: MCPSession, message: MCPRequest):
        """Handle disconnect message"""
        response = MCPResponse(
            type=MCPMessageType.DISCONNECT,
            request_id=message.id,
            result={"status": "disconnecting"}
        )
        await session.send_message(response)
        await session.close()
    
    async def _handle_ping(self, session: MCPSession, message: MCPRequest):
        """Handle ping message"""
        response = MCPResponse(
            type=MCPMessageType.PONG,
            request_id=message.id,
            result={"timestamp": time.time()}
        )
        await session.send_message(response)
    
    async def _handle_list_tools(self, session: MCPSession, message: MCPRequest):
        """Handle list tools request"""
        tags = message.params.get("tags") if message.params else None
        tools = self.tool_registry.list_tools(tags=tags)
        
        response = MCPResponse(
            type=MCPMessageType.LIST_TOOLS,
            request_id=message.id,
            result={
                "tools": [tool.dict() for tool in tools]
            }
        )
        await session.send_message(response)
    
    async def _handle_execute_tool(self, session: MCPSession, message: MCPRequest):
        """Handle tool execution request"""
        try:
            request = MCPToolExecutionRequest(**message.params)
            
            # Add session context
            context = {
                "session_id": session.session_id,
                "client_id": session.client_id,
                "context_id": request.context_id
            }
            
            result = await self.tool_registry.execute(request, context)
            
            response = MCPResponse(
                type=MCPMessageType.TOOL_RESULT,
                request_id=message.id,
                result=result.dict()
            )
            await session.send_message(response)
            
        except Exception as e:
            await self._send_error(
                session.websocket,
                message.id,
                MCPErrorCode.TOOL_EXECUTION_ERROR,
                str(e)
            )
    
    async def _handle_get_context(self, session: MCPSession, message: MCPRequest):
        """Handle get context request"""
        context_id = message.params.get("context_id") if message.params else None
        
        if not context_id:
            await self._send_error(
                session.websocket,
                message.id,
                MCPErrorCode.INVALID_PARAMS,
                "context_id is required"
            )
            return
        
        context = await self.context_manager.get_context(context_id)
        
        if context:
            response = MCPResponse(
                type=MCPMessageType.GET_CONTEXT,
                request_id=message.id,
                result=context.dict()
            )
        else:
            response = MCPResponse(
                type=MCPMessageType.GET_CONTEXT,
                request_id=message.id,
                error=MCPError(
                    code=MCPErrorCode.RESOURCE_NOT_FOUND,
                    message=f"Context not found: {context_id}"
                )
            )
        
        await session.send_message(response)
    
    async def _handle_update_context(self, session: MCPSession, message: MCPRequest):
        """Handle update context request"""
        context_id = message.params.get("context_id") if message.params else None
        updates = message.params.get("updates", {}) if message.params else {}
        merge = message.params.get("merge", True) if message.params else True
        
        if not context_id:
            # Create new context
            context = await self.context_manager.create_context(
                content=updates,
                metadata={"session_id": session.session_id}
            )
            session.context_ids.add(context.id)
            self.context_manager.associate_context(session.session_id, context.id)
        else:
            # Update existing context
            context = await self.context_manager.update_context(
                context_id, updates, merge
            )
        
        if context:
            response = MCPResponse(
                type=MCPMessageType.CONTEXT_UPDATED,
                request_id=message.id,
                result=context.dict()
            )
        else:
            response = MCPResponse(
                type=MCPMessageType.CONTEXT_UPDATED,
                request_id=message.id,
                error=MCPError(
                    code=MCPErrorCode.RESOURCE_NOT_FOUND,
                    message=f"Context not found: {context_id}"
                )
            )
        
        await session.send_message(response)
    
    async def _handle_clear_context(self, session: MCPSession, message: MCPRequest):
        """Handle clear context request"""
        context_id = message.params.get("context_id") if message.params else None
        
        if not context_id:
            await self._send_error(
                session.websocket,
                message.id,
                MCPErrorCode.INVALID_PARAMS,
                "context_id is required"
            )
            return
        
        context = await self.context_manager.clear_context(context_id)
        
        if context:
            response = MCPResponse(
                type=MCPMessageType.CLEAR_CONTEXT,
                request_id=message.id,
                result={"status": "cleared", "context_id": context_id}
            )
        else:
            response = MCPResponse(
                type=MCPMessageType.CLEAR_CONTEXT,
                request_id=message.id,
                error=MCPError(
                    code=MCPErrorCode.RESOURCE_NOT_FOUND,
                    message=f"Context not found: {context_id}"
                )
            )
        
        await session.send_message(response)
    
    async def _handle_capabilities(self, session: MCPSession, message: MCPRequest):
        """Handle capabilities request"""
        response = MCPResponse(
            type=MCPMessageType.CAPABILITIES,
            request_id=message.id,
            result=self._capabilities.dict()
        )
        await session.send_message(response)
    
    async def _send_error(
        self,
        websocket: WebSocket,
        request_id: str,
        code: MCPErrorCode,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """Send error response"""
        response = MCPResponse(
            type=MCPMessageType.ERROR,
            request_id=request_id,
            error=MCPError(
                code=code,
                message=message,
                data=data
            )
        )
        await websocket.send_json(response.dict())
    
    async def _ping_loop(self):
        """Periodically ping clients and check for timeouts"""
        while True:
            try:
                await asyncio.sleep(self.ping_interval)
                await self._check_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in ping loop: {e}")
    
    async def _check_sessions(self):
        """Check sessions for timeouts and send pings"""
        current_time = time.time()
        timeout_sessions = []
        
        for session_id, session in self._sessions.items():
            time_since_last = current_time - session.last_ping
            
            if time_since_last > self.ping_timeout:
                timeout_sessions.append(session)
            elif time_since_last > self.ping_interval:
                # Send ping
                try:
                    ping_message = MCPMessage(
                        type=MCPMessageType.PING
                    )
                    await session.send_message(ping_message)
                except Exception as e:
                    logger.error(f"Failed to ping session {session_id}: {e}")
                    timeout_sessions.append(session)
        
        # Clean up timeout sessions
        for session in timeout_sessions:
            logger.warning(f"Session {session.session_id} timed out")
            await self._cleanup_session(session)
    
    async def shutdown(self):
        """Shutdown the server"""
        # Cancel background tasks
        if self._ping_task:
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass
        
        # Close all sessions
        sessions = list(self._sessions.values())
        for session in sessions:
            await session.close(code=1012, reason="Server shutdown")
            await self._cleanup_session(session)
        
        # Shutdown context manager
        await self.context_manager.shutdown()
        
        logger.info("MCP server shutdown complete")


# Global server instance (created in API endpoint)
_mcp_server: Optional[MCPServer] = None


def get_mcp_server() -> MCPServer:
    """Get or create the global MCP server instance"""
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = MCPServer(
            tool_registry=tool_registry,
            context_manager=ContextManager()
        )
    return _mcp_server


__all__ = [
    'MCPSession',
    'MCPServer',
    'get_mcp_server'
]