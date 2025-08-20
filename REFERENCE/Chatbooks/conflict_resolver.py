# conflict_resolver.py
# Description: Handle conflicts when importing chatbook content
#
"""
Conflict Resolver
-----------------

Handles conflict resolution when importing content that may already exist.
"""

from enum import Enum
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from loguru import logger


class ConflictResolution(Enum):
    """Conflict resolution strategies."""
    ASK = "ask"          # Ask user for each conflict (interactive)
    SKIP = "skip"        # Skip conflicting items
    RENAME = "rename"    # Rename imported items
    REPLACE = "replace"  # Replace existing items
    MERGE = "merge"      # Merge with existing (where applicable)


class ConflictResolver:
    """Handle conflicts during chatbook import."""
    
    def __init__(self, ask_callback: Optional[Callable] = None):
        """
        Initialize conflict resolver.
        
        Args:
            ask_callback: Optional callback for interactive conflict resolution
        """
        self.ask_callback = ask_callback
        
    def resolve_conversation_conflict(
        self,
        existing: Dict[str, Any],
        incoming: Dict[str, Any],
        default_resolution: ConflictResolution
    ) -> ConflictResolution:
        """
        Resolve conflict for conversations.
        
        Args:
            existing: Existing conversation data
            incoming: Incoming conversation data
            default_resolution: Default resolution strategy
            
        Returns:
            Resolution to apply
        """
        if default_resolution == ConflictResolution.ASK and self.ask_callback:
            # Show conflict details and ask user
            conflict_info = {
                "type": "conversation",
                "existing_name": existing.get('conversation_name', 'Unknown'),
                "existing_created": existing.get('created_at', 'Unknown'),
                "existing_messages": existing.get('message_count', 0),
                "incoming_name": incoming.get('name', 'Unknown'),
                "incoming_created": incoming.get('created_at', 'Unknown'),
                "incoming_messages": len(incoming.get('messages', []))
            }
            
            return self.ask_callback(conflict_info)
        
        # Apply default resolution
        if default_resolution == ConflictResolution.MERGE:
            # Can't really merge conversations - treat as rename
            return ConflictResolution.RENAME
            
        return default_resolution
    
    def resolve_note_conflict(
        self,
        existing: Dict[str, Any],
        incoming: Dict[str, Any],
        default_resolution: ConflictResolution
    ) -> ConflictResolution:
        """
        Resolve conflict for notes.
        
        Args:
            existing: Existing note data
            incoming: Incoming note data
            default_resolution: Default resolution strategy
            
        Returns:
            Resolution to apply
        """
        if default_resolution == ConflictResolution.ASK and self.ask_callback:
            # Show conflict details and ask user
            conflict_info = {
                "type": "note",
                "existing_title": existing.get('title', 'Unknown'),
                "existing_created": existing.get('created_at', 'Unknown'),
                "existing_updated": existing.get('updated_at', 'Unknown'),
                "existing_length": len(existing.get('content', '')),
                "incoming_title": incoming.get('title', 'Unknown'),
                "incoming_length": len(incoming.get('content', ''))
            }
            
            return self.ask_callback(conflict_info)
        
        # Apply default resolution
        if default_resolution == ConflictResolution.MERGE:
            # For notes, merge could mean appending content
            # But for now, treat as rename
            return ConflictResolution.RENAME
            
        return default_resolution
    
    def resolve_character_conflict(
        self,
        existing: Dict[str, Any],
        incoming: Dict[str, Any],
        default_resolution: ConflictResolution
    ) -> ConflictResolution:
        """
        Resolve conflict for characters.
        
        Args:
            existing: Existing character data
            incoming: Incoming character data
            default_resolution: Default resolution strategy
            
        Returns:
            Resolution to apply
        """
        if default_resolution == ConflictResolution.ASK and self.ask_callback:
            # Show conflict details and ask user
            conflict_info = {
                "type": "character",
                "existing_name": existing.get('name', 'Unknown'),
                "existing_created": existing.get('created_at', 'Unknown'),
                "existing_description": existing.get('description', '')[:100] + '...',
                "incoming_name": incoming.get('name', 'Unknown'),
                "incoming_description": incoming.get('description', '')[:100] + '...'
            }
            
            return self.ask_callback(conflict_info)
        
        # Apply default resolution
        return default_resolution
    
    def resolve_prompt_conflict(
        self,
        existing: Dict[str, Any],
        incoming: Dict[str, Any],
        default_resolution: ConflictResolution
    ) -> ConflictResolution:
        """
        Resolve conflict for prompts.
        
        Args:
            existing: Existing prompt data
            incoming: Incoming prompt data
            default_resolution: Default resolution strategy
            
        Returns:
            Resolution to apply
        """
        if default_resolution == ConflictResolution.ASK and self.ask_callback:
            # Show conflict details and ask user
            conflict_info = {
                "type": "prompt",
                "existing_name": existing.get('name', 'Unknown'),
                "existing_created": existing.get('created_at', 'Unknown'),
                "incoming_name": incoming.get('name', 'Unknown')
            }
            
            return self.ask_callback(conflict_info)
        
        # Apply default resolution
        return default_resolution
    
    def merge_notes(
        self,
        existing: Dict[str, Any],
        incoming: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge two notes together.
        
        Args:
            existing: Existing note data
            incoming: Incoming note data
            
        Returns:
            Merged note data
        """
        # Simple merge strategy: append content with separator
        merged = existing.copy()
        
        separator = f"\n\n---\n[Imported from chatbook on {datetime.now().strftime('%Y-%m-%d %H:%M')}]\n\n"
        
        merged['content'] = existing.get('content', '') + separator + incoming.get('content', '')
        merged['updated_at'] = datetime.now().isoformat()
        
        # Merge tags
        existing_tags = set(existing.get('keywords', '').split(',')) if existing.get('keywords') else set()
        incoming_tags = set(incoming.get('tags', []))
        merged_tags = existing_tags.union(incoming_tags)
        merged['keywords'] = ','.join(sorted(merged_tags))
        
        return merged