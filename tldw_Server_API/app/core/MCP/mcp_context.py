# mcp_context.py - MCP Context Management
"""
Context management system for MCP server.

This module provides:
- Context storage and retrieval
- Context lifecycle management
- Context sharing between tools
- Persistence options
"""

import asyncio
import json
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import weakref

from .mcp_protocol import MCPContext, MCPError, MCPErrorCode

logger = logging.getLogger(__name__)


class ContextStore:
    """Storage backend for contexts"""
    
    async def get(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Get context by ID"""
        raise NotImplementedError
    
    async def set(self, context_id: str, context: Dict[str, Any]):
        """Store context"""
        raise NotImplementedError
    
    async def delete(self, context_id: str):
        """Delete context"""
        raise NotImplementedError
    
    async def list_keys(self, pattern: str = "*") -> List[str]:
        """List context IDs matching pattern"""
        raise NotImplementedError


class InMemoryContextStore(ContextStore):
    """In-memory context storage"""
    
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}
    
    async def get(self, context_id: str) -> Optional[Dict[str, Any]]:
        return self._store.get(context_id)
    
    async def set(self, context_id: str, context: Dict[str, Any]):
        self._store[context_id] = context
    
    async def delete(self, context_id: str):
        self._store.pop(context_id, None)
    
    async def list_keys(self, pattern: str = "*") -> List[str]:
        if pattern == "*":
            return list(self._store.keys())
        # Simple pattern matching (just prefix for now)
        prefix = pattern.rstrip("*")
        return [k for k in self._store.keys() if k.startswith(prefix)]


class ContextManager:
    """Manages MCP contexts with lifecycle and persistence"""
    
    def __init__(
        self,
        store: Optional[ContextStore] = None,
        max_context_size: int = 1024 * 1024,  # 1MB
        default_ttl: Optional[int] = 3600,  # 1 hour
        cleanup_interval: int = 300  # 5 minutes
    ):
        self._store = store or InMemoryContextStore()
        self._max_context_size = max_context_size
        self._default_ttl = default_ttl
        self._cleanup_interval = cleanup_interval
        
        # Track active contexts
        self._contexts: Dict[str, MCPContext] = {}
        self._context_refs: Dict[str, Set[str]] = defaultdict(set)  # session_id -> context_ids
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def create_context(
        self,
        name: Optional[str] = None,
        content: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None
    ) -> MCPContext:
        """Create a new context"""
        context = MCPContext(
            name=name,
            content=content or {},
            metadata=metadata or {}
        )
        
        # Set expiration
        if ttl or self._default_ttl:
            context.expires_at = datetime.utcnow() + timedelta(
                seconds=ttl or self._default_ttl
            )
        
        # Validate size
        self._validate_context_size(context)
        
        # Store context
        self._contexts[context.id] = context
        await self._persist_context(context)
        
        logger.info(f"Created context: {context.id}")
        return context
    
    async def get_context(self, context_id: str) -> Optional[MCPContext]:
        """Get context by ID"""
        # Check in-memory cache first
        if context_id in self._contexts:
            context = self._contexts[context_id]
            if not context.is_expired():
                return context
            else:
                # Remove expired context
                await self.delete_context(context_id)
                return None
        
        # Try to load from store
        context_data = await self._store.get(context_id)
        if context_data:
            try:
                context = MCPContext(**context_data)
                if not context.is_expired():
                    self._contexts[context_id] = context
                    return context
                else:
                    await self._store.delete(context_id)
            except Exception as e:
                logger.error(f"Failed to deserialize context {context_id}: {e}")
        
        return None
    
    async def update_context(
        self,
        context_id: str,
        updates: Dict[str, Any],
        merge: bool = True
    ) -> Optional[MCPContext]:
        """Update context content"""
        context = await self.get_context(context_id)
        if not context:
            return None
        
        if merge:
            context.update(updates)
        else:
            context.content = updates
            context.updated_at = datetime.utcnow()
        
        # Validate size
        self._validate_context_size(context)
        
        # Persist changes
        await self._persist_context(context)
        
        logger.info(f"Updated context: {context_id}")
        return context
    
    async def delete_context(self, context_id: str):
        """Delete a context"""
        self._contexts.pop(context_id, None)
        await self._store.delete(context_id)
        
        # Remove from session references
        for session_contexts in self._context_refs.values():
            session_contexts.discard(context_id)
        
        logger.info(f"Deleted context: {context_id}")
    
    async def clear_context(self, context_id: str) -> Optional[MCPContext]:
        """Clear context content"""
        context = await self.get_context(context_id)
        if not context:
            return None
        
        context.clear()
        await self._persist_context(context)
        
        logger.info(f"Cleared context: {context_id}")
        return context
    
    async def list_contexts(
        self,
        session_id: Optional[str] = None,
        include_expired: bool = False
    ) -> List[MCPContext]:
        """List all contexts, optionally filtered by session"""
        contexts = []
        
        if session_id:
            context_ids = self._context_refs.get(session_id, set())
            for context_id in context_ids:
                context = await self.get_context(context_id)
                if context and (include_expired or not context.is_expired()):
                    contexts.append(context)
        else:
            for context in self._contexts.values():
                if include_expired or not context.is_expired():
                    contexts.append(context)
        
        return contexts
    
    def associate_context(self, session_id: str, context_id: str):
        """Associate a context with a session"""
        self._context_refs[session_id].add(context_id)
    
    async def cleanup_session(self, session_id: str):
        """Clean up contexts associated with a session"""
        context_ids = self._context_refs.pop(session_id, set())
        for context_id in context_ids:
            # Only delete if no other sessions reference it
            referenced = any(
                context_id in contexts
                for sid, contexts in self._context_refs.items()
                if sid != session_id
            )
            if not referenced:
                await self.delete_context(context_id)
    
    def _validate_context_size(self, context: MCPContext):
        """Validate context size"""
        # Rough estimation of context size with datetime handling
        context_data = context.dict()
        # Convert datetime objects to ISO strings for JSON serialization
        for key, value in context_data.items():
            if isinstance(value, datetime):
                context_data[key] = value.isoformat()
        
        context_str = json.dumps(context_data, default=str)
        if len(context_str) > self._max_context_size:
            raise ValueError(
                f"Context size ({len(context_str)} bytes) exceeds "
                f"maximum ({self._max_context_size} bytes)"
            )
    
    async def _persist_context(self, context: MCPContext):
        """Persist context to store"""
        # Serialize with proper datetime handling
        context_data = context.dict()
        # Convert datetime objects to ISO strings
        for key, value in context_data.items():
            if isinstance(value, datetime):
                context_data[key] = value.isoformat()
        await self._store.set(context.id, context_data)
    
    async def _cleanup_loop(self):
        """Periodically clean up expired contexts"""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_expired_contexts()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_expired_contexts(self):
        """Remove expired contexts"""
        expired = []
        for context_id, context in self._contexts.items():
            if context.is_expired():
                expired.append(context_id)
        
        for context_id in expired:
            await self.delete_context(context_id)
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired contexts")
    
    async def shutdown(self):
        """Shutdown context manager"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass


# Context injection for tools
class ContextInjector:
    """Middleware for injecting context into tool execution"""
    
    def __init__(self, context_manager: ContextManager):
        self.context_manager = context_manager
    
    async def __call__(
        self,
        request: Any,
        context: Optional[Dict[str, Any]]
    ) -> tuple:
        """Inject context data into tool execution"""
        if context and "context_id" in context:
            mcp_context = await self.context_manager.get_context(
                context["context_id"]
            )
            if mcp_context:
                context["mcp_context"] = mcp_context.content
        
        return request, context


__all__ = [
    'ContextStore',
    'InMemoryContextStore',
    'ContextManager',
    'ContextInjector'
]