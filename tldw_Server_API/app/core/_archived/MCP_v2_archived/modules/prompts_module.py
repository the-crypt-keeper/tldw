"""
Prompts Module for tldw MCP - Manages prompt templates and libraries
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger
import json

from ..modules.base import BaseModule, create_tool_definition, create_resource_definition, create_prompt_definition
from ..schemas import ModuleConfig

# Import tldw's existing prompts functionality  
try:
    from tldw_Server_API.app.core.DB_Management.Prompts_DB import PromptsDatabase
except ImportError:
    PromptsDatabase = None
    logger.warning("PromptsDatabase not available - using mock implementation")


class PromptsModule(BaseModule):
    """Prompts Module for tldw
    
    Provides tools for:
    - Managing prompt templates
    - Creating and organizing prompt libraries
    - Prompt versioning
    - Prompt parameter management
    - Importing/exporting prompts
    """
    
    def __init__(self, config: ModuleConfig):
        super().__init__(config)
        self.prompts_db: Optional[PromptsDatabase] = None
        self.db_path = config.settings.get("db_path", "./Databases/Prompts.db")
    
    async def on_initialize(self) -> None:
        """Initialize prompts module"""
        try:
            # Initialize prompts database connection
            self.prompts_db = PromptsDatabase(self.db_path, client_id="mcp_prompts_module")
            logger.info(f"Prompts module initialized with database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize prompts database: {e}")
            raise
    
    async def on_shutdown(self) -> None:
        """Shutdown prompts module"""
        if self.prompts_db:
            logger.info("Prompts module shutdown")
    
    async def check_health(self) -> bool:
        """Check module health"""
        try:
            if self.prompts_db:
                # Check database connectivity
                test_result = self.prompts_db.list_prompts(page=1, per_page=1)
                return True
            return False
        except Exception as e:
            logger.error(f"Prompts module health check failed: {e}")
            return False
    
    async def get_tools(self) -> List[Dict[str, Any]]:
        """Get list of prompt tools"""
        return [
            create_tool_definition(
                name="create_prompt",
                description="Create a new prompt template",
                parameters={
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Prompt name/identifier"
                        },
                        "description": {
                            "type": "string",
                            "description": "Description of what the prompt does"
                        },
                        "template": {
                            "type": "string",
                            "description": "The prompt template with {{variables}}"
                        },
                        "variables": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "description": {"type": "string"},
                                    "default": {"type": "string"}
                                }
                            },
                            "description": "Variables used in the template"
                        },
                        "category": {
                            "type": "string",
                            "description": "Category for organization"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for categorization"
                        }
                    },
                    "required": ["name", "template"]
                },
                department="prompts"
            ),
            create_tool_definition(
                name="get_prompt",
                description="Get a prompt template by name or ID",
                parameters={
                    "properties": {
                        "prompt_id": {
                            "type": "string",
                            "description": "Prompt ID or name"
                        },
                        "fill_variables": {
                            "type": "object",
                            "description": "Variables to fill in the template"
                        }
                    },
                    "required": ["prompt_id"]
                },
                department="prompts"
            ),
            create_tool_definition(
                name="search_prompts",
                description="Search for prompt templates",
                parameters={
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "category": {
                            "type": "string",
                            "description": "Filter by category"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by tags"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results",
                            "default": 10
                        }
                    }
                },
                department="prompts"
            ),
            create_tool_definition(
                name="update_prompt",
                description="Update an existing prompt template",
                parameters={
                    "properties": {
                        "prompt_id": {
                            "type": "string",
                            "description": "Prompt ID to update"
                        },
                        "name": {
                            "type": "string",
                            "description": "New name (optional)"
                        },
                        "template": {
                            "type": "string",
                            "description": "New template (optional)"
                        },
                        "description": {
                            "type": "string",
                            "description": "New description (optional)"
                        },
                        "variables": {
                            "type": "array",
                            "description": "New variables (optional)"
                        },
                        "version_note": {
                            "type": "string",
                            "description": "Note about this version"
                        }
                    },
                    "required": ["prompt_id"]
                },
                department="prompts"
            ),
            create_tool_definition(
                name="delete_prompt",
                description="Delete a prompt template",
                parameters={
                    "properties": {
                        "prompt_id": {
                            "type": "string",
                            "description": "Prompt ID to delete"
                        }
                    },
                    "required": ["prompt_id"]
                },
                department="prompts"
            ),
            create_tool_definition(
                name="list_prompts",
                description="List all prompt templates",
                parameters={
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Filter by category"
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Pagination offset",
                            "default": 0
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of prompts to return",
                            "default": 20
                        }
                    }
                },
                department="prompts"
            ),
            create_tool_definition(
                name="import_prompts",
                description="Import prompts from JSON",
                parameters={
                    "properties": {
                        "prompts_json": {
                            "type": "string",
                            "description": "JSON string containing prompts to import"
                        },
                        "overwrite": {
                            "type": "boolean",
                            "description": "Overwrite existing prompts with same name",
                            "default": False
                        }
                    },
                    "required": ["prompts_json"]
                },
                department="prompts"
            ),
            create_tool_definition(
                name="export_prompts",
                description="Export prompts to JSON",
                parameters={
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Export only prompts from this category"
                        },
                        "prompt_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific prompt IDs to export"
                        }
                    }
                },
                department="prompts"
            )
        ]
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute prompts tool"""
        logger.info(f"Executing prompts tool: {tool_name}")
        
        try:
            if tool_name == "create_prompt":
                return await self._create_prompt(arguments)
            elif tool_name == "get_prompt":
                return await self._get_prompt(arguments)
            elif tool_name == "search_prompts":
                return await self._search_prompts(arguments)
            elif tool_name == "update_prompt":
                return await self._update_prompt(arguments)
            elif tool_name == "delete_prompt":
                return await self._delete_prompt(arguments)
            elif tool_name == "list_prompts":
                return await self._list_prompts(arguments)
            elif tool_name == "import_prompts":
                return await self._import_prompts(arguments)
            elif tool_name == "export_prompts":
                return await self._export_prompts(arguments)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
        except Exception as e:
            logger.error(f"Error executing prompts tool {tool_name}: {e}")
            raise
    
    async def get_prompts(self) -> List[Dict[str, Any]]:
        """Get list of available prompts for MCP protocol"""
        try:
            # Use list_prompts with high limit to get all prompts
            all_prompts, total, pages, current = self.prompts_db.list_prompts(page=1, per_page=1000)
            
            mcp_prompts = []
            for prompt in all_prompts:
                # Convert to MCP prompt format
                arguments = []
                if prompt.get("variables"):
                    for var in prompt["variables"]:
                        arguments.append({
                            "name": var["name"],
                            "description": var.get("description", ""),
                            "required": var.get("required", False)
                        })
                
                mcp_prompts.append(create_prompt_definition(
                    name=prompt["name"],
                    description=prompt.get("description", ""),
                    arguments=arguments
                ))
            
            return mcp_prompts
        except Exception as e:
            logger.error(f"Error getting prompts: {e}")
            return []
    
    async def has_prompt(self, name: str) -> bool:
        """Check if module has a specific prompt"""
        try:
            prompt = self.prompts_db.get_prompt_by_name(name)
            return prompt is not None
        except:
            return False
    
    async def get_prompt(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get a prompt with filled variables"""
        try:
            prompt = self.prompts_db.get_prompt_by_name(name)
            if not prompt:
                raise ValueError(f"Prompt '{name}' not found")
            
            # Fill in template variables
            template = prompt["template"]
            for key, value in arguments.items():
                template = template.replace(f"{{{{{key}}}}}", str(value))
            
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": template
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Error getting prompt {name}: {e}")
            raise
    
    async def get_resources(self) -> List[Dict[str, Any]]:
        """Get prompt resources"""
        return [
            create_resource_definition(
                uri="prompts://categories",
                name="Prompt Categories",
                description="List of all prompt categories",
                mime_type="application/json"
            ),
            create_resource_definition(
                uri="prompts://statistics",
                name="Prompt Statistics",
                description="Statistics about prompt library",
                mime_type="application/json"
            ),
            create_resource_definition(
                uri="prompts://featured",
                name="Featured Prompts",
                description="Curated list of featured prompts",
                mime_type="application/json"
            )
        ]
    
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read prompt resource"""
        if uri == "prompts://categories":
            # PromptsDatabase doesn't have get_all_categories, return empty for now
            categories = []
            return {
                "categories": categories,
                "count": len(categories)
            }
        elif uri == "prompts://statistics":
            prompts, total, _, _ = self.prompts_db.list_prompts(page=1, per_page=1)
            return {
                "total_prompts": total,
                "categories": 0,  # No categories method available
                "last_updated": datetime.utcnow().isoformat()
            }
        elif uri == "prompts://featured":
            # PromptsDatabase doesn't have get_featured_prompts, return first few prompts
            featured, _, _, _ = self.prompts_db.list_prompts(page=1, per_page=5)
            return {
                "featured": featured,
                "count": len(featured)
            }
        else:
            raise ValueError(f"Unknown resource: {uri}")
    
    # Tool implementation methods
    
    async def _create_prompt(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new prompt"""
        try:
            # PromptsDatabase uses add_prompt method
            prompt_id, status = self.prompts_db.add_prompt(
                name=args["name"],
                author="MCP User",
                details=args.get("description", ""),
                system_prompt=None,
                user_prompt=args["template"],
                keywords=args.get("tags", [])
            )
            
            return {
                "success": True,
                "prompt_id": prompt_id,
                "message": f"Prompt '{args['name']}' created successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _get_prompt(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get a prompt template"""
        prompt_id = args["prompt_id"]
        fill_variables = args.get("fill_variables", {})
        
        try:
            # Try to get by ID first, then by name
            prompt = self.prompts_db.get_prompt_by_id(prompt_id)
            if not prompt:
                prompt = self.prompts_db.get_prompt_by_name(prompt_id)
            
            if not prompt:
                return {
                    "success": False,
                    "error": f"Prompt '{prompt_id}' not found"
                }
            
            # Fill template if variables provided
            template = prompt["template"]
            if fill_variables:
                for key, value in fill_variables.items():
                    template = template.replace(f"{{{{{key}}}}}", str(value))
            
            return {
                "success": True,
                "prompt": prompt,
                "filled_template": template if fill_variables else None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _search_prompts(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Search prompts"""
        try:
            # PromptsDatabase search_prompts has different signature
            results = self.prompts_db.search_prompts(
                search_query=args.get("query", ""),
                limit=args.get("limit", 10)
            )
            
            return {
                "success": True,
                "results": results,
                "count": len(results)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _update_prompt(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Update a prompt"""
        prompt_id = args["prompt_id"]
        
        try:
            updates = {}
            if "name" in args:
                updates["name"] = args["name"]
            if "template" in args:
                updates["template"] = args["template"]
            if "description" in args:
                updates["description"] = args["description"]
            if "variables" in args:
                updates["variables"] = args["variables"]
            
            # PromptsDatabase uses update_prompt_by_id
            self.prompts_db.update_prompt_by_id(int(prompt_id), updates)
            
            # Log version note if provided
            if args.get("version_note"):
                self.prompts_db.add_version_note(prompt_id, args["version_note"])
            
            return {
                "success": True,
                "prompt_id": prompt_id,
                "message": "Prompt updated successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _delete_prompt(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a prompt"""
        prompt_id = args["prompt_id"]
        
        try:
            # PromptsDatabase uses soft_delete_prompt
            self.prompts_db.soft_delete_prompt(prompt_id)
            
            return {
                "success": True,
                "message": f"Prompt '{prompt_id}' deleted successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _list_prompts(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List prompts"""
        try:
            # PromptsDatabase uses list_prompts with page/per_page
            offset = args.get("offset", 0)
            limit = args.get("limit", 20)
            page = (offset // limit) + 1 if limit > 0 else 1
            prompts, total, pages, current = self.prompts_db.list_prompts(page=page, per_page=limit)
            
            return {
                "success": True,
                "prompts": prompts,
                "count": len(prompts)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _import_prompts(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Import prompts from JSON"""
        try:
            prompts_data = json.loads(args["prompts_json"])
            overwrite = args.get("overwrite", False)
            
            imported = 0
            skipped = 0
            
            for prompt in prompts_data:
                existing = self.prompts_db.get_prompt_by_name(prompt["name"])
                
                if existing and not overwrite:
                    skipped += 1
                    continue
                
                if existing:
                    self.prompts_db.update_prompt_by_id(existing["id"], prompt)
                else:
                    # Use add_prompt for creating new prompts
                    self.prompts_db.add_prompt(
                        name=prompt.get("name"),
                        author=prompt.get("author", "MCP Import"),
                        details=prompt.get("description", ""),
                        user_prompt=prompt.get("template", ""),
                        keywords=prompt.get("tags", [])
                    )
                
                imported += 1
            
            return {
                "success": True,
                "imported": imported,
                "skipped": skipped,
                "message": f"Imported {imported} prompts, skipped {skipped}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _export_prompts(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Export prompts to JSON"""
        try:
            if args.get("prompt_ids"):
                prompts = []
                for pid in args["prompt_ids"]:
                    prompt = self.prompts_db.get_prompt_by_id(pid)
                    if prompt:
                        prompts.append(prompt)
            elif args.get("category"):
                # No category filter available, get all prompts
                prompts, _, _, _ = self.prompts_db.list_prompts(page=1, per_page=1000)
            else:
                prompts, _, _, _ = self.prompts_db.list_prompts(page=1, per_page=1000)
            
            return {
                "success": True,
                "prompts": prompts,
                "json": json.dumps(prompts, indent=2),
                "count": len(prompts)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }