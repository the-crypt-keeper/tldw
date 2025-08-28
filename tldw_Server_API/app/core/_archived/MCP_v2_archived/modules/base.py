"""
Base module interface for tldw MCP modules
"""

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from datetime import datetime
from loguru import logger

from ..schemas import (
    ModuleConfig,
    ToolDefinition,
    ResourceDefinition,
    PromptDefinition
)


class BaseModule(ABC):
    """Base class for tldw MCP modules"""
    
    def __init__(self, config: ModuleConfig):
        self.config = config
        self.name = config.name
        self.version = config.version
        self.department = config.department  # Feature area in tldw (media, rag, chat, etc.)
        self._initialized = False
        self._health_status = {"status": "unknown", "last_check": None}
    
    async def initialize(self) -> None:
        """Initialize the module"""
        logger.info(f"Initializing module: {self.name} v{self.version}")
        try:
            await self.on_initialize()
            self._initialized = True
            logger.info(f"Module initialized: {self.name}")
        except Exception as e:
            logger.error(f"Module initialization failed: {self.name} - {str(e)}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the module"""
        logger.info(f"Shutting down module: {self.name}")
        try:
            await self.on_shutdown()
            self._initialized = False
            logger.info(f"Module shutdown: {self.name}")
        except Exception as e:
            logger.error(f"Module shutdown failed: {self.name} - {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check module health"""
        try:
            is_healthy = await self.check_health()
            self._health_status = {
                "status": "healthy" if is_healthy else "unhealthy",
                "last_check": datetime.utcnow().isoformat(),
                "module": self.name,
                "version": self.version
            }
        except Exception as e:
            self._health_status = {
                "status": "unhealthy",
                "last_check": datetime.utcnow().isoformat(),
                "error": str(e),
                "module": self.name,
                "version": self.version
            }
        
        return self._health_status
    
    # Abstract methods that modules must implement
    @abstractmethod
    async def on_initialize(self) -> None:
        """Module-specific initialization"""
        pass
    
    @abstractmethod
    async def on_shutdown(self) -> None:
        """Module-specific shutdown"""
        pass
    
    @abstractmethod
    async def check_health(self) -> bool:
        """Module-specific health check"""
        pass
    
    @abstractmethod
    async def get_tools(self) -> List[Dict[str, Any]]:
        """Get list of tools provided by this module"""
        pass
    
    @abstractmethod
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool"""
        pass
    
    # Optional methods with default implementations
    async def has_tool(self, tool_name: str) -> bool:
        """Check if module provides a tool"""
        tools = await self.get_tools()
        return any(tool["name"] == tool_name for tool in tools)
    
    async def get_resources(self) -> List[Dict[str, Any]]:
        """Get list of resources (optional)"""
        return []
    
    async def has_resource(self, uri: str) -> bool:
        """Check if module provides a resource"""
        resources = await self.get_resources()
        return any(resource["uri"] == uri for resource in resources)
    
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read a resource"""
        raise NotImplementedError(f"Resource reading not implemented for {self.name}")
    
    async def get_prompts(self) -> List[Dict[str, Any]]:
        """Get list of prompts (optional)"""
        return []
    
    async def has_prompt(self, name: str) -> bool:
        """Check if module provides a prompt"""
        prompts = await self.get_prompts()
        return any(prompt["name"] == name for prompt in prompts)
    
    async def get_prompt(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get a prompt with arguments"""
        raise NotImplementedError(f"Prompt not implemented for {self.name}")


# Helper functions for creating MCP-compliant definitions

def create_tool_definition(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    department: str = None
) -> Dict[str, Any]:
    """Helper to create MCP-compliant tool definition"""
    tool_def = {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": parameters.get("properties", {}),
            "required": parameters.get("required", [])
        }
    }
    
    if department:
        tool_def["metadata"] = {"department": department}
    
    return tool_def


def create_resource_definition(
    uri: str,
    name: str,
    description: str,
    mime_type: str = "application/json"
) -> Dict[str, Any]:
    """Helper to create MCP-compliant resource definition"""
    return {
        "uri": uri,
        "name": name,
        "description": description,
        "mimeType": mime_type
    }


def create_prompt_definition(
    name: str,
    description: str,
    arguments: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Helper to create MCP-compliant prompt definition"""
    return {
        "name": name,
        "description": description,
        "arguments": arguments or []
    }