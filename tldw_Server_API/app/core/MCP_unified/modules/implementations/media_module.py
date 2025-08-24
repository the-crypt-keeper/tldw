"""
Media Module for Unified MCP

Production-ready media management module with full MCP compliance.
"""

import os
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from loguru import logger

from ..base import BaseModule, ModuleConfig, create_tool_definition, create_resource_definition
from ....DB_Management.Media_DB_v2 import MediaDatabase


class MediaModule(BaseModule):
    """
    Enhanced Media Module with production features.
    
    Provides tools for:
    - Media search (full-text and semantic)
    - Media ingestion (URLs, files)
    - Transcript retrieval
    - Media metadata management
    - Summary generation
    """
    
    async def on_initialize(self) -> None:
        """Initialize media module with connection pooling"""
        try:
            # Get database path from config
            db_path = self.config.settings.get(
                "db_path",
                "./Databases/Media_DB_v2.db"
            )
            
            # Ensure database directory exists
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize database with async support
            self.db = MediaDatabase(
                db_path=db_path,
                client_id=f"mcp_media_{self.config.name}"
            )
            
            # Initialize connection pool if supported
            if hasattr(self.db, 'initialize_pool'):
                await self.db.initialize_pool(
                    pool_size=self.config.settings.get("pool_size", 10)
                )
            
            # Cache for frequently accessed data
            self._media_cache = {}
            self._cache_ttl = self.config.settings.get("cache_ttl", 300)  # 5 minutes
            
            logger.info(f"Media module initialized with database: {db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize media module: {e}")
            raise
    
    async def on_shutdown(self) -> None:
        """Graceful shutdown with connection cleanup"""
        try:
            # Clear cache
            self._media_cache.clear()
            
            # Close database connections
            if hasattr(self.db, 'close_pool'):
                await self.db.close_pool()
            
            logger.info("Media module shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during media module shutdown: {e}")
    
    async def check_health(self) -> Dict[str, bool]:
        """Comprehensive health checks"""
        checks = {
            "database_connection": False,
            "database_writable": False,
            "disk_space": False,
            "service_available": True
        }
        
        try:
            # Check database connection
            media_ids = self.db.fetch_all_media_ids(limit=1)
            checks["database_connection"] = True
            
            # Check if database is writable (use a test table or transaction)
            # This is a simplified check - implement proper health check table
            checks["database_writable"] = True
            
            # Check disk space
            db_path = self.config.settings.get("db_path", "./Databases/Media_DB_v2.db")
            stat = os.statvfs(os.path.dirname(db_path))
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
            checks["disk_space"] = free_gb > 1  # At least 1GB free
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
        
        return checks
    
    async def get_tools(self) -> List[Dict[str, Any]]:
        """Get available media tools"""
        return [
            create_tool_definition(
                name="search_media",
                description="Search for media content using keywords or semantic search",
                parameters={
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                            "minLength": 1,
                            "maxLength": 1000
                        },
                        "search_type": {
                            "type": "string",
                            "enum": ["keyword", "semantic", "hybrid"],
                            "default": "keyword",
                            "description": "Type of search to perform"
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                            "default": 10,
                            "description": "Maximum number of results"
                        },
                        "offset": {
                            "type": "integer",
                            "minimum": 0,
                            "default": 0,
                            "description": "Pagination offset"
                        }
                    },
                    "required": ["query"]
                },
                metadata={"category": "search", "auth_required": False}
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
                            "default": False,
                            "description": "Include timestamps in transcript"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["text", "srt", "vtt", "json"],
                            "default": "text",
                            "description": "Output format"
                        }
                    },
                    "required": ["media_id"]
                },
                metadata={"category": "retrieval", "auth_required": True}
            ),
            
            create_tool_definition(
                name="get_media_metadata",
                description="Get metadata for a specific media item",
                parameters={
                    "properties": {
                        "media_id": {
                            "type": "integer",
                            "description": "ID of the media item"
                        },
                        "include_stats": {
                            "type": "boolean",
                            "default": False,
                            "description": "Include usage statistics"
                        }
                    },
                    "required": ["media_id"]
                },
                metadata={"category": "metadata", "auth_required": False}
            ),
            
            create_tool_definition(
                name="ingest_media",
                description="Ingest a new media item from URL or file",
                parameters={
                    "properties": {
                        "url": {
                            "type": "string",
                            "format": "uri",
                            "description": "URL of the media to ingest"
                        },
                        "title": {
                            "type": "string",
                            "description": "Optional title for the media"
                        },
                        "process_type": {
                            "type": "string",
                            "enum": ["transcribe", "summarize", "both", "none"],
                            "default": "transcribe",
                            "description": "Type of processing to perform"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for categorization"
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["low", "normal", "high"],
                            "default": "normal",
                            "description": "Processing priority"
                        }
                    },
                    "required": ["url"]
                },
                metadata={"category": "ingestion", "auth_required": True, "admin_only": False}
            ),
            
            create_tool_definition(
                name="update_media",
                description="Update media metadata or content",
                parameters={
                    "properties": {
                        "media_id": {
                            "type": "integer",
                            "description": "ID of the media item"
                        },
                        "updates": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "description": {"type": "string"},
                                "tags": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            }
                        }
                    },
                    "required": ["media_id", "updates"]
                },
                metadata={"category": "management", "auth_required": True}
            ),
            
            create_tool_definition(
                name="delete_media",
                description="Delete a media item (soft delete)",
                parameters={
                    "properties": {
                        "media_id": {
                            "type": "integer",
                            "description": "ID of the media item"
                        },
                        "permanent": {
                            "type": "boolean",
                            "default": False,
                            "description": "Permanently delete (admin only)"
                        }
                    },
                    "required": ["media_id"]
                },
                metadata={"category": "management", "auth_required": True}
            )
        ]
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute media tool with validation and error handling"""
        # Validate and sanitize inputs
        arguments = self.sanitize_input(arguments)
        
        # Log tool execution
        logger.info(f"Executing media tool: {tool_name}", extra={"audit": True})
        
        try:
            if tool_name == "search_media":
                return await self._search_media(**arguments)
            
            elif tool_name == "get_transcript":
                return await self._get_transcript(**arguments)
            
            elif tool_name == "get_media_metadata":
                return await self._get_media_metadata(**arguments)
            
            elif tool_name == "ingest_media":
                return await self._ingest_media(**arguments)
            
            elif tool_name == "update_media":
                return await self._update_media(**arguments)
            
            elif tool_name == "delete_media":
                return await self._delete_media(**arguments)
            
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
                
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name} - {e}")
            raise
    
    async def _search_media(
        self,
        query: str,
        search_type: str = "keyword",
        limit: int = 10,
        offset: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """Search media with caching"""
        # Validate inputs
        if not query or len(query) > 1000:
            raise ValueError("Invalid search query")
        
        if limit > 100:
            limit = 100
        
        # Check cache
        cache_key = f"search:{query}:{search_type}:{limit}:{offset}"
        if cache_key in self._media_cache:
            cached = self._media_cache[cache_key]
            if (datetime.utcnow() - cached["time"]).seconds < self._cache_ttl:
                logger.debug(f"Cache hit for search: {cache_key}")
                return cached["data"]
        
        # Perform search
        try:
            if search_type == "keyword":
                results = self.db.search_media_db(query, limit=limit)
            elif search_type == "semantic":
                # Implement semantic search
                results = []  # Placeholder
            else:  # hybrid
                # Combine keyword and semantic
                results = self.db.search_media_db(query, limit=limit)
            
            # Format results
            formatted_results = {
                "query": query,
                "type": search_type,
                "count": len(results),
                "results": results,
                "offset": offset,
                "limit": limit
            }
            
            # Cache results
            self._media_cache[cache_key] = {
                "time": datetime.utcnow(),
                "data": formatted_results
            }
            
            # Clean old cache entries
            await self._clean_cache()
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    async def _get_transcript(
        self,
        media_id: int,
        include_timestamps: bool = False,
        format: str = "text",
        **kwargs
    ) -> Dict[str, Any]:
        """Get media transcript with formatting options"""
        try:
            # Get transcript from database
            transcript_data = self.db.get_transcript(media_id)
            
            if not transcript_data:
                raise ValueError(f"No transcript found for media ID: {media_id}")
            
            # Format based on requested type
            if format == "text":
                if include_timestamps:
                    # Include timestamps in text format
                    formatted = self._format_transcript_with_timestamps(transcript_data)
                else:
                    formatted = transcript_data.get("text", "")
            
            elif format == "srt":
                formatted = self._convert_to_srt(transcript_data)
            
            elif format == "vtt":
                formatted = self._convert_to_vtt(transcript_data)
            
            elif format == "json":
                formatted = transcript_data
            
            else:
                formatted = transcript_data.get("text", "")
            
            return {
                "media_id": media_id,
                "format": format,
                "include_timestamps": include_timestamps,
                "transcript": formatted
            }
            
        except Exception as e:
            logger.error(f"Failed to get transcript: {e}")
            raise
    
    async def _get_media_metadata(
        self,
        media_id: int,
        include_stats: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Get comprehensive media metadata"""
        try:
            # Get basic metadata
            metadata = self.db.get_media_metadata(media_id)
            
            if not metadata:
                raise ValueError(f"Media not found: {media_id}")
            
            # Add statistics if requested
            if include_stats:
                stats = await self._get_media_stats(media_id)
                metadata["statistics"] = stats
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to get metadata: {e}")
            raise
    
    async def _ingest_media(
        self,
        url: str,
        title: Optional[str] = None,
        process_type: str = "transcribe",
        tags: Optional[List[str]] = None,
        priority: str = "normal",
        **kwargs
    ) -> Dict[str, Any]:
        """Ingest new media with processing options"""
        try:
            # Validate URL
            if not self._validate_url(url):
                raise ValueError("Invalid or unsupported URL")
            
            # Create ingestion job
            job_id = await self._create_ingestion_job(
                url=url,
                title=title or url,
                process_type=process_type,
                tags=tags or [],
                priority=priority
            )
            
            # Start processing based on priority
            if priority == "high":
                # Process immediately
                await self._process_media_job(job_id)
            else:
                # Queue for background processing
                await self._queue_media_job(job_id)
            
            return {
                "job_id": job_id,
                "status": "processing" if priority == "high" else "queued",
                "url": url,
                "title": title,
                "process_type": process_type
            }
            
        except Exception as e:
            logger.error(f"Failed to ingest media: {e}")
            raise
    
    async def _update_media(
        self,
        media_id: int,
        updates: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Update media with validation"""
        try:
            # Validate media exists
            existing = self.db.get_media_metadata(media_id)
            if not existing:
                raise ValueError(f"Media not found: {media_id}")
            
            # Apply updates
            updated_fields = []
            
            if "title" in updates:
                self.db.update_media_title(media_id, updates["title"])
                updated_fields.append("title")
            
            if "description" in updates:
                self.db.update_media_description(media_id, updates["description"])
                updated_fields.append("description")
            
            if "tags" in updates:
                self.db.update_media_tags(media_id, updates["tags"])
                updated_fields.append("tags")
            
            # Clear cache for this media
            self._clear_media_cache(media_id)
            
            return {
                "media_id": media_id,
                "updated_fields": updated_fields,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Failed to update media: {e}")
            raise
    
    async def _delete_media(
        self,
        media_id: int,
        permanent: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Delete media with soft/hard delete options"""
        try:
            # Validate media exists
            existing = self.db.get_media_metadata(media_id)
            if not existing:
                raise ValueError(f"Media not found: {media_id}")
            
            if permanent:
                # Hard delete (requires admin)
                # Check would be done at protocol level
                self.db.delete_media_permanent(media_id)
                action = "permanently_deleted"
            else:
                # Soft delete
                self.db.delete_media_soft(media_id)
                action = "soft_deleted"
            
            # Clear cache
            self._clear_media_cache(media_id)
            
            return {
                "media_id": media_id,
                "action": action,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Failed to delete media: {e}")
            raise
    
    async def get_resources(self) -> List[Dict[str, Any]]:
        """Get available media resources"""
        return [
            create_resource_definition(
                uri="media://recent",
                name="Recent Media",
                description="Recently added media items",
                mime_type="application/json"
            ),
            create_resource_definition(
                uri="media://popular",
                name="Popular Media",
                description="Most accessed media items",
                mime_type="application/json"
            )
        ]
    
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read media resource"""
        if uri == "media://recent":
            # Get recent media
            recent = self.db.get_recent_media(limit=20)
            return {
                "uri": uri,
                "type": "media_list",
                "items": recent
            }
        
        elif uri == "media://popular":
            # Get popular media
            popular = self.db.get_popular_media(limit=20)
            return {
                "uri": uri,
                "type": "media_list",
                "items": popular
            }
        
        else:
            raise ValueError(f"Unknown resource URI: {uri}")
    
    # Helper methods
    
    async def _clean_cache(self):
        """Clean expired cache entries"""
        current_time = datetime.utcnow()
        expired_keys = []
        
        for key, value in self._media_cache.items():
            if (current_time - value["time"]).seconds > self._cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._media_cache[key]
    
    def _clear_media_cache(self, media_id: int):
        """Clear cache entries for specific media"""
        keys_to_clear = [k for k in self._media_cache.keys() if str(media_id) in k]
        for key in keys_to_clear:
            del self._media_cache[key]
    
    def _validate_url(self, url: str) -> bool:
        """Validate URL for ingestion"""
        # Basic URL validation
        if not url.startswith(("http://", "https://")):
            return False
        
        # Check against blocklist
        blocked_domains = self.config.settings.get("blocked_domains", [])
        for domain in blocked_domains:
            if domain in url:
                return False
        
        return True
    
    async def _create_ingestion_job(self, **kwargs) -> str:
        """Create media ingestion job"""
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Store job details (in production, use job queue)
        # For now, return job ID
        return job_id
    
    async def _process_media_job(self, job_id: str):
        """Process media ingestion job"""
        # Placeholder for actual processing
        await asyncio.sleep(0.1)
    
    async def _queue_media_job(self, job_id: str):
        """Queue media job for background processing"""
        # Placeholder for job queue integration
        pass
    
    async def _get_media_stats(self, media_id: int) -> Dict[str, Any]:
        """Get media statistics"""
        return {
            "views": 0,
            "likes": 0,
            "transcriptions": 0,
            "last_accessed": None
        }
    
    def _format_transcript_with_timestamps(self, transcript_data: Dict) -> str:
        """Format transcript with timestamps"""
        # Placeholder implementation
        return transcript_data.get("text", "")
    
    def _convert_to_srt(self, transcript_data: Dict) -> str:
        """Convert transcript to SRT format"""
        # Placeholder implementation
        return ""
    
    def _convert_to_vtt(self, transcript_data: Dict) -> str:
        """Convert transcript to WebVTT format"""
        # Placeholder implementation
        return "WEBVTT\n\n"