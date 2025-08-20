# prompt_studio_websocket.py
# WebSocket endpoint for real-time Prompt Studio updates

import json
import asyncio
from typing import Dict, Set, Optional
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect, Depends, Query
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps import (
    get_prompt_studio_db, get_prompt_studio_user,
    PromptStudioDatabase
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.job_manager import (
    JobManager, JobStatus
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.event_broadcaster import (
    EventBroadcaster, EventType
)

########################################################################################################################
# Connection Manager

class ConnectionManager:
    """Manages WebSocket connections for Prompt Studio."""
    
    def __init__(self):
        """Initialize connection manager."""
        # Store active connections by client ID
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Store connection metadata
        self.connection_metadata: Dict[WebSocket, Dict] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str, 
                     user_context: Optional[Dict] = None):
        """
        Accept and register a new WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            client_id: Client identifier
            user_context: Optional user context
        """
        await websocket.accept()
        
        # Add to active connections
        if client_id not in self.active_connections:
            self.active_connections[client_id] = set()
        
        self.active_connections[client_id].add(websocket)
        
        # Store metadata
        self.connection_metadata[websocket] = {
            "client_id": client_id,
            "user_context": user_context,
            "connected_at": datetime.utcnow().isoformat()
        }
        
        logger.info(f"WebSocket connected for client {client_id}")
    
    def disconnect(self, websocket: WebSocket):
        """
        Remove a WebSocket connection.
        
        Args:
            websocket: WebSocket connection to remove
        """
        metadata = self.connection_metadata.get(websocket)
        if metadata:
            client_id = metadata["client_id"]
            
            # Remove from active connections
            if client_id in self.active_connections:
                self.active_connections[client_id].discard(websocket)
                
                # Clean up empty sets
                if not self.active_connections[client_id]:
                    del self.active_connections[client_id]
            
            # Remove metadata
            del self.connection_metadata[websocket]
            
            logger.info(f"WebSocket disconnected for client {client_id}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """
        Send a message to a specific WebSocket.
        
        Args:
            message: Message to send
            websocket: Target WebSocket
        """
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Failed to send message to WebSocket: {e}")
            self.disconnect(websocket)
    
    async def broadcast_to_client(self, client_id: str, message: str):
        """
        Broadcast a message to all connections for a client.
        
        Args:
            client_id: Client identifier
            message: Message to broadcast
        """
        if client_id in self.active_connections:
            disconnected = []
            
            for websocket in self.active_connections[client_id]:
                try:
                    await websocket.send_text(message)
                except Exception as e:
                    logger.error(f"Failed to send to WebSocket: {e}")
                    disconnected.append(websocket)
            
            # Clean up disconnected sockets
            for ws in disconnected:
                self.disconnect(ws)
    
    async def broadcast_to_all(self, message: str):
        """
        Broadcast a message to all connected clients.
        
        Args:
            message: Message to broadcast
        """
        for client_id in self.active_connections:
            await self.broadcast_to_client(client_id, message)
    
    def get_connection_count(self) -> int:
        """Get total number of active connections."""
        return sum(len(connections) for connections in self.active_connections.values())
    
    def get_client_count(self) -> int:
        """Get number of unique clients connected."""
        return len(self.active_connections)

# Global connection manager instance
manager = ConnectionManager()

########################################################################################################################
# WebSocket Endpoint

async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str = Query(..., description="Client ID"),
    project_id: Optional[int] = Query(None, description="Project ID to subscribe to"),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db)
):
    """
    WebSocket endpoint for real-time Prompt Studio updates.
    
    Args:
        websocket: WebSocket connection
        client_id: Client identifier
        project_id: Optional project to subscribe to
        db: Database instance
    """
    # Accept connection
    await manager.connect(websocket, client_id)
    
    # Create event broadcaster
    broadcaster = EventBroadcaster(manager, db)
    
    # Send initial connection message
    await websocket.send_json({
        "type": "connection",
        "status": "connected",
        "client_id": client_id,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    # If project specified, send current job status
    if project_id:
        job_manager = JobManager(db)
        jobs = job_manager.list_jobs(limit=10)
        
        await websocket.send_json({
            "type": "initial_state",
            "project_id": project_id,
            "jobs": jobs,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    try:
        # Keep connection alive and handle incoming messages
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                message_type = message.get("type")
                
                if message_type == "ping":
                    # Respond to ping
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                elif message_type == "subscribe":
                    # Subscribe to specific events
                    entity_type = message.get("entity_type")
                    entity_id = message.get("entity_id")
                    
                    await websocket.send_json({
                        "type": "subscribed",
                        "entity_type": entity_type,
                        "entity_id": entity_id,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                elif message_type == "unsubscribe":
                    # Unsubscribe from events
                    entity_type = message.get("entity_type")
                    entity_id = message.get("entity_id")
                    
                    await websocket.send_json({
                        "type": "unsubscribed",
                        "entity_type": entity_type,
                        "entity_id": entity_id,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                elif message_type == "get_job_status":
                    # Get status of specific job
                    job_id = message.get("job_id")
                    job_manager = JobManager(db)
                    job = job_manager.get_job(job_id)
                    
                    await websocket.send_json({
                        "type": "job_status",
                        "job": job,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                elif message_type == "get_stats":
                    # Get connection statistics
                    stats = {
                        "connections": manager.get_connection_count(),
                        "clients": manager.get_client_count(),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    await websocket.send_json({
                        "type": "stats",
                        "data": stats
                    })
                
                else:
                    # Unknown message type
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {message_type}",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON message",
                    "timestamp": datetime.utcnow().isoformat()
                })
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

########################################################################################################################
# SSE (Server-Sent Events) Fallback

from fastapi import Response
from fastapi.responses import StreamingResponse
import asyncio

async def sse_endpoint(
    client_id: str = Query(..., description="Client ID"),
    project_id: Optional[int] = Query(None, description="Project ID to subscribe to"),
    db: PromptStudioDatabase = Depends(get_prompt_studio_db)
):
    """
    Server-Sent Events endpoint as fallback for WebSocket.
    
    Args:
        client_id: Client identifier
        project_id: Optional project to subscribe to
        db: Database instance
    """
    async def event_generator():
        """Generate SSE events."""
        # Send initial connection event
        yield f"data: {json.dumps({'type': 'connection', 'status': 'connected', 'client_id': client_id})}\n\n"
        
        # If project specified, send current state
        if project_id:
            job_manager = JobManager(db)
            jobs = job_manager.list_jobs(limit=10)
            
            yield f"data: {json.dumps({'type': 'initial_state', 'project_id': project_id, 'jobs': jobs})}\n\n"
        
        # Keep connection alive with periodic heartbeats
        try:
            while True:
                # Send heartbeat every 30 seconds
                await asyncio.sleep(30)
                yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.utcnow().isoformat()})}\n\n"
                
        except asyncio.CancelledError:
            logger.info(f"SSE connection closed for client {client_id}")
            raise
        except Exception as e:
            logger.error(f"SSE error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )