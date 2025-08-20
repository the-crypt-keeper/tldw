# tab_state_manager.py
# Description: Thread-safe state management for chat tabs
#
# Imports
import threading
import asyncio
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from loguru import logger

#######################################################################################################################
#
# Classes:

@dataclass
class TabState:
    """State information for a single tab."""
    tab_id: str
    conversation_id: Optional[str] = None
    is_ephemeral: bool = True
    is_streaming: bool = False
    worker_id: Optional[str] = None
    ai_message_widget_id: Optional[str] = None
    has_unsaved_changes: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class TabStateManager:
    """
    Thread-safe manager for tab states across the application.
    
    This class ensures that tab state is properly synchronized across
    multiple workers and async operations, preventing race conditions
    and data corruption.
    """
    
    def __init__(self):
        """Initialize the state manager with thread-safe storage."""
        # Thread-local storage for current tab context
        self._local = threading.local()
        
        # Global tab states with async lock protection
        self._states: Dict[str, TabState] = {}
        self._states_lock = asyncio.Lock()
        
        # Active tab tracking
        self._active_tab_id: Optional[str] = None
        self._active_tab_lock = asyncio.Lock()
        
        logger.debug("TabStateManager initialized")
    
    # Context Management
    
    @asynccontextmanager
    async def tab_context(self, tab_id: str):
        """
        Context manager for tab-specific operations.
        
        Usage:
            async with state_manager.tab_context(tab_id) as context:
                # Operations here have tab_id in thread-local storage
                pass
        """
        previous_tab = getattr(self._local, 'current_tab_id', None)
        self._local.current_tab_id = tab_id
        
        try:
            yield self
        finally:
            if previous_tab is not None:
                self._local.current_tab_id = previous_tab
            else:
                delattr(self._local, 'current_tab_id')
    
    # State Access Methods
    
    def get_current_tab_id(self) -> Optional[str]:
        """
        Get the current tab ID from thread-local storage.
        
        Returns:
            Current tab ID or None if not in a tab context
        """
        return getattr(self._local, 'current_tab_id', None)
    
    async def get_active_tab_id(self) -> Optional[str]:
        """
        Get the globally active tab ID.
        
        Returns:
            Active tab ID or None
        """
        async with self._active_tab_lock:
            return self._active_tab_id
    
    async def set_active_tab_id(self, tab_id: Optional[str]):
        """
        Set the globally active tab ID.
        
        Args:
            tab_id: Tab ID to set as active, or None
        """
        async with self._active_tab_lock:
            self._active_tab_id = tab_id
            logger.debug(f"Active tab set to: {tab_id}")
    
    # State Management Methods
    
    async def create_tab_state(self, tab_id: str, **kwargs) -> TabState:
        """
        Create a new tab state.
        
        Args:
            tab_id: Unique identifier for the tab
            **kwargs: Additional state attributes
            
        Returns:
            Created TabState instance
        """
        async with self._states_lock:
            if tab_id in self._states:
                logger.warning(f"Tab state already exists for: {tab_id}")
                return self._states[tab_id]
            
            state = TabState(tab_id=tab_id, **kwargs)
            self._states[tab_id] = state
            logger.debug(f"Created tab state: {tab_id}")
            return state
    
    async def get_tab_state(self, tab_id: str) -> Optional[TabState]:
        """
        Get the state for a specific tab.
        
        Args:
            tab_id: Tab identifier
            
        Returns:
            TabState instance or None if not found
        """
        async with self._states_lock:
            return self._states.get(tab_id)
    
    async def update_tab_state(self, tab_id: str, **updates):
        """
        Update specific attributes of a tab state.
        
        Args:
            tab_id: Tab identifier
            **updates: Attributes to update
        """
        async with self._states_lock:
            state = self._states.get(tab_id)
            if not state:
                logger.error(f"No state found for tab: {tab_id}")
                return
            
            for key, value in updates.items():
                if hasattr(state, key):
                    setattr(state, key, value)
                else:
                    state.metadata[key] = value
            
            logger.debug(f"Updated tab state {tab_id}: {updates}")
    
    async def delete_tab_state(self, tab_id: str):
        """
        Delete a tab state.
        
        Args:
            tab_id: Tab identifier
        """
        async with self._states_lock:
            if tab_id in self._states:
                del self._states[tab_id]
                logger.debug(f"Deleted tab state: {tab_id}")
                
                # Clear active tab if it was deleted
                if self._active_tab_id == tab_id:
                    self._active_tab_id = None
    
    # Bulk Operations
    
    async def get_all_tab_states(self) -> Dict[str, TabState]:
        """
        Get all tab states.
        
        Returns:
            Dictionary of tab_id to TabState
        """
        async with self._states_lock:
            return self._states.copy()
    
    async def get_streaming_tabs(self) -> list[str]:
        """
        Get IDs of all tabs currently streaming.
        
        Returns:
            List of tab IDs that are streaming
        """
        async with self._states_lock:
            return [
                tab_id 
                for tab_id, state in self._states.items() 
                if state.is_streaming
            ]
    
    async def has_unsaved_changes(self) -> bool:
        """
        Check if any tab has unsaved changes.
        
        Returns:
            True if any tab has unsaved changes
        """
        async with self._states_lock:
            return any(
                state.has_unsaved_changes 
                for state in self._states.values()
            )
    
    # Worker Management
    
    async def set_tab_worker(self, tab_id: str, worker_id: Optional[str]):
        """
        Associate a worker with a tab.
        
        Args:
            tab_id: Tab identifier
            worker_id: Worker identifier or None to clear
        """
        await self.update_tab_state(tab_id, worker_id=worker_id)
    
    async def get_tab_by_worker(self, worker_id: str) -> Optional[str]:
        """
        Find which tab a worker belongs to.
        
        Args:
            worker_id: Worker identifier
            
        Returns:
            Tab ID or None if not found
        """
        async with self._states_lock:
            for tab_id, state in self._states.items():
                if state.worker_id == worker_id:
                    return tab_id
            return None
    
    # Utility Methods
    
    async def clear_all_states(self):
        """Clear all tab states. Use with caution!"""
        async with self._states_lock:
            self._states.clear()
            self._active_tab_id = None
            logger.warning("All tab states cleared")
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"TabStateManager(tabs={len(self._states)}, active={self._active_tab_id})"


# Singleton instance
_state_manager_instance: Optional[TabStateManager] = None


def get_tab_state_manager() -> TabStateManager:
    """
    Get the singleton TabStateManager instance.
    
    Returns:
        The global TabStateManager instance
    """
    global _state_manager_instance
    if _state_manager_instance is None:
        _state_manager_instance = TabStateManager()
    return _state_manager_instance

#
# End of tab_state_manager.py
#######################################################################################################################