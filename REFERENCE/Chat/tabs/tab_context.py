# tab_context.py
# Description: Tab context management for widget resolution without monkey patching
#
# Imports
from typing import TYPE_CHECKING, Optional, TypeVar, Type, Set
from loguru import logger

if TYPE_CHECKING:
    from ...app import TldwCli
    from ..chat_models import ChatSessionData
    from textual.widget import Widget

# Type variable for widget types
W = TypeVar('W', bound='Widget')

#######################################################################################################################
#
# Classes:

class TabContext:
    """
    Manages tab-specific widget resolution without modifying core framework methods.
    
    This class provides a clean abstraction for querying widgets in a tab-aware manner,
    replacing the dangerous monkey-patching approach with dependency injection.
    """
    
    # Widget IDs that need tab-specific mapping
    TAB_SPECIFIC_WIDGETS: Set[str] = {
        "#chat-log",
        "#chat-input", 
        "#chat-input-area",
        "#send-stop-chat",
        "#respond-for-me-button",
        "#attach-image",
        "#image-attachment-indicator"
    }
    
    # Widget IDs that remain global (not tab-specific)
    GLOBAL_WIDGETS: Set[str] = {
        "#chat-conversation-title-input",
        "#chat-conversation-keywords-input",
        "#chat-conversation-uuid-display",
        "#chat-system-prompt"
    }
    
    def __init__(self, app: 'TldwCli', session_data: Optional['ChatSessionData'] = None):
        """
        Initialize tab context.
        
        Args:
            app: The TldwCli application instance
            session_data: Optional session data for tab-specific operations
        """
        self.app = app
        self.session_data = session_data
        self._widget_cache = {}
        
    def _map_selector_to_tab(self, selector: str) -> str:
        """
        Map a generic selector to a tab-specific selector if needed.
        
        Args:
            selector: The original widget selector
            
        Returns:
            The mapped selector (tab-specific or original)
        """
        # Check if tabs are enabled
        from ...config import get_cli_setting
        if not get_cli_setting("chat_defaults", "enable_tabs", False):
            return selector
            
        # No session data means use default selector
        if not self.session_data:
            return selector
            
        # Check if this is a tab-specific widget
        if selector in self.TAB_SPECIFIC_WIDGETS:
            base_id = selector[1:]  # Remove '#'
            tab_specific_id = f"#{base_id}-{self.session_data.tab_id}"
            logger.debug(f"TabContext: Mapping {selector} to {tab_specific_id}")
            return tab_specific_id
            
        # Global widgets remain unchanged
        if selector in self.GLOBAL_WIDGETS:
            return selector
            
        # Unknown selectors pass through
        return selector
    
    def query_one(self, selector: str, widget_type: Optional[Type[W]] = None) -> W:
        """
        Query for a single widget in a tab-aware manner.
        
        Args:
            selector: CSS selector for the widget
            widget_type: Optional widget type for type checking
            
        Returns:
            The matched widget
        """
        mapped_selector = self._map_selector_to_tab(selector)
        
        # Check cache first
        cache_key = (mapped_selector, widget_type)
        if cache_key in self._widget_cache:
            return self._widget_cache[cache_key]
        
        # Query the widget
        if widget_type:
            widget = self.app.query_one(mapped_selector, widget_type)
        else:
            widget = self.app.query_one(mapped_selector)
            
        # Cache the result
        self._widget_cache[cache_key] = widget
        return widget
    
    def query(self, selector: str, widget_type: Optional[Type[W]] = None):
        """
        Query for multiple widgets in a tab-aware manner.
        
        Args:
            selector: CSS selector for the widgets
            widget_type: Optional widget type for type checking
            
        Returns:
            Query result with matched widgets
        """
        mapped_selector = self._map_selector_to_tab(selector)
        
        if widget_type:
            return self.app.query(mapped_selector, widget_type)
        else:
            return self.app.query(mapped_selector)
    
    def try_query_one(self, selector: str, widget_type: Optional[Type[W]] = None) -> Optional[W]:
        """
        Try to query for a single widget, returning None if not found.
        
        Args:
            selector: CSS selector for the widget
            widget_type: Optional widget type for type checking
            
        Returns:
            The matched widget or None
        """
        try:
            return self.query_one(selector, widget_type)
        except Exception as e:
            logger.debug(f"Widget not found: {selector} - {e}")
            return None
    
    def clear_cache(self):
        """Clear the widget cache."""
        self._widget_cache.clear()
    
    def update_session(self, session_data: 'ChatSessionData'):
        """
        Update the session data for this context.
        
        Args:
            session_data: New session data
        """
        self.session_data = session_data
        self.clear_cache()  # Clear cache when session changes
    
    @classmethod
    def add_tab_specific_widget(cls, widget_id: str):
        """
        Add a widget ID to the tab-specific set.
        
        Args:
            widget_id: The widget ID to add (including #)
        """
        cls.TAB_SPECIFIC_WIDGETS.add(widget_id)
    
    @classmethod
    def add_global_widget(cls, widget_id: str):
        """
        Add a widget ID to the global set.
        
        Args:
            widget_id: The widget ID to add (including #)
        """
        cls.GLOBAL_WIDGETS.add(widget_id)

#
# End of tab_context.py
#######################################################################################################################