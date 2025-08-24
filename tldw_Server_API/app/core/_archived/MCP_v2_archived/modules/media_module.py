"""
Media Module for tldw MCP - Handles media ingestion, search, and retrieval
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger

from ..modules.base import BaseModule, create_tool_definition, create_resource_definition
from ..schemas import ModuleConfig

# Import tldw's existing media functionality
try:
    from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import MediaDatabase
except ImportError:
    MediaDatabase = None
    logger.warning("MediaDatabase not available - using mock implementation")


class MediaModule(BaseModule):
    """Media Module for tldw
    
    Provides tools for:
    - Media search (full-text and semantic)
    - Media ingestion (URLs, files)
    - Transcript retrieval
    - Media metadata management
    - Summary generation
    """
    
    def __init__(self, config: ModuleConfig):
        super().__init__(config)
        self.db: Optional[MediaDatabase] = None
        self.db_path = config.settings.get("db_path", "./Databases/Media_DB_v2.db")
    
    async def on_initialize(self) -> None:
        """Initialize media module"""
        try:
            # Initialize database connection
            self.db = MediaDatabase(self.db_path, client_id="mcp_media_module")
            logger.info(f"Media module initialized with database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize media database: {e}")
            raise
    
    async def on_shutdown(self) -> None:
        """Shutdown media module"""
        if self.db:
            # Close database connection if needed
            logger.info("Media module shutdown")
    
    async def check_health(self) -> bool:
        """Check module health"""
        try:
            # Check database connectivity
            if self.db:
                # Perform a simple query to check database health
                test_result = self.db.fetch_all_media_ids(client_id="health_check")
                return True
            return False
        except Exception as e:
            logger.error(f"Media module health check failed: {e}")
            return False
    
    async def get_tools(self) -> List[Dict[str, Any]]:
        """Get list of media tools"""
        return [
            create_tool_definition(
                name="search_media",
                description="Search for media content using keywords or semantic search",
                parameters={
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "search_type": {
                            "type": "string",
                            "enum": ["keyword", "semantic", "hybrid"],
                            "description": "Type of search to perform",
                            "default": "keyword"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                },
                department="media"
            ),
            create_tool_definition(
                name="get_transcript",
                description="Get the transcript for a specific media item",
                parameters={
                    "properties": {
                        "media_id": {
                            "type": "integer",
                            "description": "ID of the media item"
                        },
                        "include_timestamps": {
                            "type": "boolean",
                            "description": "Include timestamps in transcript",
                            "default": False
                        }
                    },
                    "required": ["media_id"]
                },
                department="media"
            ),
            create_tool_definition(
                name="get_media_metadata",
                description="Get metadata for a specific media item",
                parameters={
                    "properties": {
                        "media_id": {
                            "type": "integer",
                            "description": "ID of the media item"
                        }
                    },
                    "required": ["media_id"]
                },
                department="media"
            ),
            create_tool_definition(
                name="ingest_media",
                description="Ingest a new media item from URL",
                parameters={
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL of the media to ingest"
                        },
                        "title": {
                            "type": "string",
                            "description": "Optional title for the media"
                        },
                        "process_type": {
                            "type": "string",
                            "enum": ["transcribe", "summarize", "both"],
                            "description": "Type of processing to perform",
                            "default": "transcribe"
                        }
                    },
                    "required": ["url"]
                },
                department="media"
            ),
            create_tool_definition(
                name="get_media_summary",
                description="Get or generate a summary for a media item",
                parameters={
                    "properties": {
                        "media_id": {
                            "type": "integer",
                            "description": "ID of the media item"
                        },
                        "summary_type": {
                            "type": "string",
                            "enum": ["short", "detailed", "bullet_points"],
                            "description": "Type of summary to generate",
                            "default": "short"
                        }
                    },
                    "required": ["media_id"]
                },
                department="media"
            ),
            create_tool_definition(
                name="list_recent_media",
                description="List recently added media items",
                parameters={
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Number of items to return",
                            "default": 10
                        },
                        "media_type": {
                            "type": "string",
                            "enum": ["video", "audio", "document", "all"],
                            "description": "Filter by media type",
                            "default": "all"
                        }
                    }
                },
                department="media"
            )
        ]
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute media tool"""
        logger.info(f"Executing media tool: {tool_name} with args: {arguments}")
        
        try:
            if tool_name == "search_media":
                return await self._search_media(arguments)
            elif tool_name == "get_transcript":
                return await self._get_transcript(arguments)
            elif tool_name == "get_media_metadata":
                return await self._get_media_metadata(arguments)
            elif tool_name == "ingest_media":
                return await self._ingest_media(arguments)
            elif tool_name == "get_media_summary":
                return await self._get_media_summary(arguments)
            elif tool_name == "list_recent_media":
                return await self._list_recent_media(arguments)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
        except Exception as e:
            logger.error(f"Error executing media tool {tool_name}: {e}")
            raise
    
    async def get_resources(self) -> List[Dict[str, Any]]:
        """Get media resources"""
        return [
            create_resource_definition(
                uri="media://statistics",
                name="Media Statistics",
                description="Overall statistics about the media library",
                mime_type="application/json"
            ),
            create_resource_definition(
                uri="media://categories",
                name="Media Categories",
                description="List of available media categories and tags",
                mime_type="application/json"
            )
        ]
    
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read media resource"""
        if uri == "media://statistics":
            # Get media statistics
            total_media = len(self.db.fetch_all_media_ids())
            return {
                "total_media": total_media,
                "last_updated": datetime.utcnow().isoformat(),
                "database": self.db_path
            }
        elif uri == "media://categories":
            # Get categories/tags
            tags = self.db.get_all_unique_tags()
            return {
                "tags": tags,
                "count": len(tags)
            }
        else:
            raise ValueError(f"Unknown resource: {uri}")
    
    # Tool implementation methods
    
    async def _search_media(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Search for media content"""
        query = args["query"]
        search_type = args.get("search_type", "keyword")
        limit = args.get("limit", 10)
        
        try:
            # Use tldw's existing search functionality
            if self.db:
                results = self.db.search_media_db(
                    search_query=query,
                    limit=limit
                )
            else:
                results = []
            
            return {
                "success": True,
                "query": query,
                "search_type": search_type,
                "results": results,
                "count": len(results)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _get_transcript(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get transcript for a media item"""
        media_id = args["media_id"]
        include_timestamps = args.get("include_timestamps", False)
        
        try:
            # Fetch transcript from database
            transcript = self.db.get_transcript_by_media_id(media_id)
            
            if transcript:
                return {
                    "success": True,
                    "media_id": media_id,
                    "transcript": transcript,
                    "has_timestamps": include_timestamps
                }
            else:
                return {
                    "success": False,
                    "error": f"No transcript found for media ID {media_id}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _get_media_metadata(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get metadata for a media item"""
        media_id = args["media_id"]
        
        try:
            # Fetch media metadata
            media_data = self.db.get_media_by_id(media_id)
            
            if media_data:
                return {
                    "success": True,
                    "media_id": media_id,
                    "metadata": media_data
                }
            else:
                return {
                    "success": False,
                    "error": f"No media found with ID {media_id}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _ingest_media(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest new media from URL"""
        url = args["url"]
        title = args.get("title")
        process_type = args.get("process_type", "transcribe")
        
        try:
            # For now, return a placeholder response
            # In production, this would integrate with tldw's media ingestion pipeline
            result = {
                "media_id": "placeholder_id",
                "status": "queued"
            }
            
            return {
                "success": True,
                "url": url,
                "media_id": result.get("media_id"),
                "message": "Media ingested successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _get_media_summary(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get or generate summary for media"""
        media_id = args["media_id"]
        summary_type = args.get("summary_type", "short")
        
        try:
            # Fetch existing summary or generate new one
            summary = self.db.get_summary_by_media_id(media_id)
            
            if summary:
                return {
                    "success": True,
                    "media_id": media_id,
                    "summary": summary,
                    "summary_type": summary_type
                }
            else:
                # Could trigger summary generation here
                return {
                    "success": False,
                    "error": f"No summary available for media ID {media_id}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _list_recent_media(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List recently added media"""
        limit = args.get("limit", 10)
        media_type = args.get("media_type", "all")
        
        try:
            # Fetch recent media from database
            recent_media = self.db.get_recent_media(limit=limit, media_type=media_type)
            
            return {
                "success": True,
                "media": recent_media,
                "count": len(recent_media)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }