# tabs/__init__.py
# Description: Tab management module for chat functionality
#
# This module provides tab-aware functionality for the chat system
# without using dangerous monkey-patching techniques.
#
from .tab_context import TabContext
from .tab_state_manager import TabStateManager

__all__ = ["TabContext", "TabStateManager"]