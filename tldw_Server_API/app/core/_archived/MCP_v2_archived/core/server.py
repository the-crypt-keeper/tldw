"""
Main MCP v2 server implementation for tldw
"""

from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
from datetime import datetime
from loguru import logger
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.responses import JSONResponse

from ..core.protocol import process_mcp_request
from ..core.registry import get_module_registry, register_module
from ..schemas import (
    MCPRequest,
    MCPResponse,
    ModuleConfig,
    ModuleCapability,
    MCPUser,
    UserRole
)
from ..modules.media_module import MediaModule
from ..modules.rag_module import RAGModule
from ..modules.notes_module import NotesModule
from ..modules.prompts_module import PromptsModule
from ..modules.transcription_module import TranscriptionModule
from ..modules.chat_module import ChatModule


class MCPServer:
    """MCP v2 Server for tldw"""
    
    def __init__(self, app: FastAPI = None):
        self.app = app
        self.registry = get_module_registry()
        self.initialized = False
        self.startup_time = datetime.utcnow()
    
    async def initialize(self):
        """Initialize the MCP server and register modules"""
        if self.initialized:
            return
        
        logger.info("Initializing MCP v2 server for tldw")
        
        try:
            # Register Media Module
            media_config = ModuleConfig(
                name="media",
                description="Media ingestion and management module",
                version="1.0.0",
                department="media",  # Feature area
                enabled=True,
                capabilities=[ModuleCapability.TOOLS, ModuleCapability.RESOURCES],
                settings={
                    "db_path": "./Databases/Media_DB_v2.db"
                }
            )
            await register_module("media", MediaModule, media_config)
            logger.info("Media module registered")
            
            # Register RAG Module
            rag_config = ModuleConfig(
                name="rag",
                description="RAG and vector search module",
                version="1.0.0",
                department="rag",  # Feature area
                enabled=True,
                capabilities=[ModuleCapability.TOOLS, ModuleCapability.RESOURCES],
                settings={
                    "collection_name": "tldw_media",
                    "embedding_model": "all-MiniLM-L6-v2"
                }
            )
            await register_module("rag", RAGModule, rag_config)
            logger.info("RAG module registered")
            
            # Register Notes Module
            notes_config = ModuleConfig(
                name="notes",
                description="Note-taking and knowledge management module",
                version="1.0.0",
                department="notes",
                enabled=True,
                capabilities=[ModuleCapability.TOOLS, ModuleCapability.RESOURCES],
                settings={
                    "db_path": "./Databases/ChaChaNotes.db"
                }
            )
            await register_module("notes", NotesModule, notes_config)
            logger.info("Notes module registered")
            
            # Register Prompts Module
            prompts_config = ModuleConfig(
                name="prompts",
                description="Prompt library and template management",
                version="1.0.0",
                department="prompts",
                enabled=True,
                capabilities=[ModuleCapability.TOOLS, ModuleCapability.RESOURCES, ModuleCapability.PROMPTS],
                settings={
                    "db_path": "./Databases/Prompts.db"
                }
            )
            await register_module("prompts", PromptsModule, prompts_config)
            logger.info("Prompts module registered")
            
            # Register Transcription Module
            transcription_config = ModuleConfig(
                name="transcription",
                description="Advanced transcription services",
                version="1.0.0",
                department="transcription",
                enabled=True,
                capabilities=[ModuleCapability.TOOLS, ModuleCapability.RESOURCES],
                settings={
                    "model": "whisper",
                    "model_size": "base",
                    "device": "cpu",
                    "temp_dir": "/tmp/transcriptions"
                }
            )
            await register_module("transcription", TranscriptionModule, transcription_config)
            logger.info("Transcription module registered")
            
            # Register Chat Module
            chat_config = ModuleConfig(
                name="chat",
                description="Chat completions and conversation management",
                version="1.0.0",
                department="chat",
                enabled=True,
                capabilities=[ModuleCapability.TOOLS, ModuleCapability.RESOURCES],
                settings={
                    "db_path": "./Databases/Chat.db",
                    "default_provider": "openai",
                    "default_model": "gpt-3.5-turbo",
                    "max_context_length": 4000
                }
            )
            await register_module("chat", ChatModule, chat_config)
            logger.info("Chat module registered")
            
            self.initialized = True
            logger.info("MCP v2 server initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP server: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the MCP server"""
        logger.info("Shutting down MCP v2 server")
        await self.registry.shutdown_all()
        self.initialized = False
        logger.info("MCP v2 server shutdown complete")
    
    async def handle_websocket(self, websocket: WebSocket, client_id: str = None):
        """Handle WebSocket connections for MCP protocol"""
        await websocket.accept()
        logger.info(f"WebSocket connection accepted from client: {client_id}")
        
        try:
            while True:
                # Receive message from client
                data = await websocket.receive_json()
                
                # Parse as MCP request
                try:
                    request = MCPRequest(**data)
                except Exception as e:
                    error_response = MCPResponse(
                        error={
                            "code": -32600,
                            "message": f"Invalid request: {str(e)}"
                        },
                        id=data.get("id")
                    )
                    await websocket.send_json(error_response.dict())
                    continue
                
                # Create context for request processing
                context = {
                    "module_registry": self.registry,
                    "client_id": client_id,
                    "websocket": websocket,
                    "user": await self._get_user_context(client_id)
                }
                
                # Process request
                response = await process_mcp_request(request, context)
                
                # Send response
                await websocket.send_json(response.dict())
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {client_id}")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await websocket.close(code=1011, reason=str(e))
    
    async def handle_http_request(self, request: MCPRequest, client_id: str = None) -> MCPResponse:
        """Handle HTTP requests for MCP protocol"""
        # Create context for request processing
        context = {
            "module_registry": self.registry,
            "client_id": client_id,
            "user": await self._get_user_context(client_id)
        }
        
        # Process request
        response = await process_mcp_request(request, context)
        return response
    
    async def get_status(self) -> Dict[str, Any]:
        """Get server status"""
        # Get module health status
        health_results = await self.registry.health_check_all()
        
        # Get registered modules
        registrations = await self.registry.list_registrations()
        
        return {
            "status": "healthy" if self.initialized else "initializing",
            "version": "2.0.0",
            "uptime_seconds": (datetime.utcnow() - self.startup_time).total_seconds(),
            "modules": {
                "total": len(registrations),
                "active": sum(1 for r in registrations if r.status == "active"),
                "registrations": [
                    {
                        "id": r.module_id,
                        "name": r.name,
                        "version": r.version,
                        "status": r.status,
                        "capabilities": r.capabilities
                    }
                    for r in registrations
                ]
            },
            "health": health_results
        }
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools across modules"""
        tools = []
        modules = await self.registry.get_all_modules()
        
        for module_id, module in modules.items():
            try:
                module_tools = await module.get_tools()
                for tool in module_tools:
                    tool_info = tool.copy()
                    tool_info["module"] = module_id
                    tools.append(tool_info)
            except Exception as e:
                logger.error(f"Error getting tools from module {module_id}: {e}")
        
        return tools
    
    async def _get_user_context(self, client_id: str) -> Optional[MCPUser]:
        """Get user context from client ID"""
        # This would integrate with tldw's existing auth system
        # For now, return a basic user context
        if client_id:
            return MCPUser(
                id=client_id,
                username=client_id,
                roles=[UserRole.USER],
                department="general",
                permissions=["tools:execute", "resources:read"]
            )
        return None


# Global server instance
_server: Optional[MCPServer] = None


def get_mcp_server() -> MCPServer:
    """Get the global MCP server instance"""
    global _server
    if _server is None:
        _server = MCPServer()
    return _server


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for FastAPI integration"""
    # Startup
    server = get_mcp_server()
    await server.initialize()
    logger.info("MCP v2 server started")
    
    yield
    
    # Shutdown
    await server.shutdown()
    logger.info("MCP v2 server stopped")