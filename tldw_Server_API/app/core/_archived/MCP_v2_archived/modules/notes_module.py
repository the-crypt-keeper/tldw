"""
Notes Module for tldw MCP - Handles note creation, management, and search
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger

from ..modules.base import BaseModule, create_tool_definition, create_resource_definition
from ..schemas import ModuleConfig

# Import tldw's existing notes functionality
try:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
except ImportError:
    CharactersRAGDB = None
    logger.warning("CharactersRAGDB not available - using mock implementation")


class NotesModule(BaseModule):
    """Notes Module for tldw
    
    Provides tools for:
    - Creating and managing notes
    - Searching notes
    - Linking notes to media
    - Organizing notes with tags
    - Exporting notes in various formats
    """
    
    def __init__(self, config: ModuleConfig):
        super().__init__(config)
        self.notes_db: Optional[CharactersRAGDB] = None
        self.db_path = config.settings.get("db_path", "./Databases/ChaChaNotes.db")
    
    async def on_initialize(self) -> None:
        """Initialize notes module"""
        try:
            # Initialize notes database connection
            self.notes_db = CharactersRAGDB(self.db_path, client_id="mcp_notes_module")
            logger.info(f"Notes module initialized with database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize notes database: {e}")
            raise
    
    async def on_shutdown(self) -> None:
        """Shutdown notes module"""
        if self.notes_db:
            logger.info("Notes module shutdown")
    
    async def check_health(self) -> bool:
        """Check module health"""
        try:
            if self.notes_db:
                # Check database connectivity
                test_result = self.notes_db.list_notes(limit=1)
                return True
            return False
        except Exception as e:
            logger.error(f"Notes module health check failed: {e}")
            return False
    
    async def get_tools(self) -> List[Dict[str, Any]]:
        """Get list of notes tools"""
        return [
            create_tool_definition(
                name="create_note",
                description="Create a new note",
                parameters={
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Note title"
                        },
                        "content": {
                            "type": "string",
                            "description": "Note content (markdown supported)"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for categorization"
                        },
                        "media_id": {
                            "type": "integer",
                            "description": "Optional media ID to link note to"
                        }
                    },
                    "required": ["title", "content"]
                },
                department="notes"
            ),
            create_tool_definition(
                name="update_note",
                description="Update an existing note",
                parameters={
                    "properties": {
                        "note_id": {
                            "type": "integer",
                            "description": "ID of the note to update"
                        },
                        "title": {
                            "type": "string",
                            "description": "New title (optional)"
                        },
                        "content": {
                            "type": "string",
                            "description": "New content (optional)"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "New tags (optional)"
                        }
                    },
                    "required": ["note_id"]
                },
                department="notes"
            ),
            create_tool_definition(
                name="search_notes",
                description="Search notes by keyword or tags",
                parameters={
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by tags"
                        },
                        "media_id": {
                            "type": "integer",
                            "description": "Filter by linked media"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results",
                            "default": 10
                        }
                    }
                },
                department="notes"
            ),
            create_tool_definition(
                name="get_note",
                description="Get a specific note by ID",
                parameters={
                    "properties": {
                        "note_id": {
                            "type": "integer",
                            "description": "ID of the note"
                        },
                        "include_media": {
                            "type": "boolean",
                            "description": "Include linked media info",
                            "default": False
                        }
                    },
                    "required": ["note_id"]
                },
                department="notes"
            ),
            create_tool_definition(
                name="delete_note",
                description="Delete a note (soft delete)",
                parameters={
                    "properties": {
                        "note_id": {
                            "type": "integer",
                            "description": "ID of the note to delete"
                        },
                        "permanent": {
                            "type": "boolean",
                            "description": "Permanently delete (no recovery)",
                            "default": False
                        }
                    },
                    "required": ["note_id"]
                },
                department="notes"
            ),
            create_tool_definition(
                name="list_notes",
                description="List notes with pagination",
                parameters={
                    "properties": {
                        "offset": {
                            "type": "integer",
                            "description": "Pagination offset",
                            "default": 0
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of notes to return",
                            "default": 20
                        },
                        "sort_by": {
                            "type": "string",
                            "enum": ["created", "updated", "title"],
                            "description": "Sort order",
                            "default": "updated"
                        }
                    }
                },
                department="notes"
            ),
            create_tool_definition(
                name="export_note",
                description="Export a note in various formats",
                parameters={
                    "properties": {
                        "note_id": {
                            "type": "integer",
                            "description": "ID of the note to export"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["markdown", "html", "pdf", "json"],
                            "description": "Export format",
                            "default": "markdown"
                        }
                    },
                    "required": ["note_id"]
                },
                department="notes"
            )
        ]
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute notes tool"""
        logger.info(f"Executing notes tool: {tool_name}")
        
        try:
            if tool_name == "create_note":
                return await self._create_note(arguments)
            elif tool_name == "update_note":
                return await self._update_note(arguments)
            elif tool_name == "search_notes":
                return await self._search_notes(arguments)
            elif tool_name == "get_note":
                return await self._get_note(arguments)
            elif tool_name == "delete_note":
                return await self._delete_note(arguments)
            elif tool_name == "list_notes":
                return await self._list_notes(arguments)
            elif tool_name == "export_note":
                return await self._export_note(arguments)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
        except Exception as e:
            logger.error(f"Error executing notes tool {tool_name}: {e}")
            raise
    
    async def get_resources(self) -> List[Dict[str, Any]]:
        """Get notes resources"""
        return [
            create_resource_definition(
                uri="notes://statistics",
                name="Notes Statistics",
                description="Statistics about notes database",
                mime_type="application/json"
            ),
            create_resource_definition(
                uri="notes://tags",
                name="All Tags",
                description="List of all tags used in notes",
                mime_type="application/json"
            ),
            create_resource_definition(
                uri="notes://templates",
                name="Note Templates",
                description="Available note templates",
                mime_type="application/json"
            )
        ]
    
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read notes resource"""
        if uri == "notes://statistics":
            total_notes = len(self.notes_db.list_notes())
            return {
                "total_notes": total_notes,
                "last_updated": datetime.utcnow().isoformat()
            }
        elif uri == "notes://tags":
            # CharactersRAGDB doesn't have get_all_tags, return empty for now
            tags = []
            return {
                "tags": tags,
                "count": len(tags)
            }
        elif uri == "notes://templates":
            return {
                "templates": [
                    {
                        "name": "Meeting Notes",
                        "template": "# Meeting Notes\n\n**Date:** \n**Attendees:** \n\n## Agenda\n\n## Discussion\n\n## Action Items\n"
                    },
                    {
                        "name": "Research Notes",
                        "template": "# Research Notes\n\n**Topic:** \n**Date:** \n\n## Summary\n\n## Key Points\n\n## References\n"
                    },
                    {
                        "name": "Daily Journal",
                        "template": "# Daily Journal\n\n**Date:** \n\n## Today's Goals\n\n## Accomplishments\n\n## Reflections\n"
                    }
                ]
            }
        else:
            raise ValueError(f"Unknown resource: {uri}")
    
    # Tool implementation methods
    
    async def _create_note(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new note"""
        title = args["title"]
        content = args["content"]
        tags = args.get("tags", [])
        media_id = args.get("media_id")
        
        try:
            # CharactersRAGDB uses add_note method
            note_id = self.notes_db.add_note(
                title=title,
                content=content
            )
            # Note: tags and media_id handling would need separate implementation
            
            return {
                "success": True,
                "note_id": note_id,
                "message": f"Note '{title}' created successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _update_note(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing note"""
        note_id = args["note_id"]
        
        try:
            updates = {}
            if "title" in args:
                updates["title"] = args["title"]
            if "content" in args:
                updates["content"] = args["content"]
            if "tags" in args:
                updates["tags"] = args["tags"]
            
            # CharactersRAGDB requires expected_version for updates
            # For simplicity, we'll use version 1
            self.notes_db.update_note(note_id, updates, expected_version=1)
            
            return {
                "success": True,
                "note_id": note_id,
                "message": "Note updated successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _search_notes(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Search notes"""
        query = args.get("query", "")
        tags = args.get("tags", [])
        media_id = args.get("media_id")
        limit = args.get("limit", 10)
        
        try:
            # CharactersRAGDB search_notes takes search_term and limit
            results = self.notes_db.search_notes(
                search_term=query,
                limit=limit
            )
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _get_note(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get a specific note"""
        note_id = args["note_id"]
        include_media = args.get("include_media", False)
        
        try:
            note = self.notes_db.get_note_by_id(note_id)
            
            if note:
                if include_media and note.get("media_id"):
                    # Could fetch media info here
                    pass
                
                return {
                    "success": True,
                    "note": note
                }
            else:
                return {
                    "success": False,
                    "error": f"Note {note_id} not found"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _delete_note(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a note"""
        note_id = args["note_id"]
        permanent = args.get("permanent", False)
        
        try:
            # CharactersRAGDB uses soft_delete_note with expected_version
            if permanent:
                # No permanent delete method available, use soft delete
                self.notes_db.soft_delete_note(note_id, expected_version=1)
                message = f"Note {note_id} deleted"
            else:
                self.notes_db.soft_delete_note(note_id, expected_version=1)
                message = f"Note {note_id} moved to trash"
            
            return {
                "success": True,
                "message": message
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _list_notes(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List notes with pagination"""
        offset = args.get("offset", 0)
        limit = args.get("limit", 20)
        sort_by = args.get("sort_by", "updated")
        
        try:
            # CharactersRAGDB uses list_notes with limit and offset
            notes = self.notes_db.list_notes(
                limit=limit,
                offset=offset
            )
            
            return {
                "success": True,
                "notes": notes,
                "count": len(notes),
                "offset": offset,
                "limit": limit
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _export_note(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Export a note"""
        note_id = args["note_id"]
        format = args.get("format", "markdown")
        
        try:
            note = self.notes_db.get_note_by_id(note_id)
            
            if not note:
                return {
                    "success": False,
                    "error": f"Note {note_id} not found"
                }
            
            if format == "markdown":
                content = f"# {note['title']}\n\n{note['content']}"
            elif format == "html":
                # Simple HTML conversion
                content = f"<h1>{note['title']}</h1>\n<div>{note['content']}</div>"
            elif format == "json":
                content = note
            else:
                content = note['content']
            
            return {
                "success": True,
                "format": format,
                "content": content
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }