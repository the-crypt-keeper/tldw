"""
Module registry for managing tldw MCP modules
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
from loguru import logger

from ..modules.base import BaseModule
from ..schemas import ModuleRegistration, ModuleConfig


class ModuleRegistry:
    """Registry for managing MCP modules in tldw"""
    
    def __init__(self):
        self._modules: Dict[str, BaseModule] = {}
        self._registrations: Dict[str, ModuleRegistration] = {}
        self._lock = asyncio.Lock()
    
    async def register_module(
        self,
        module_id: str,
        module: BaseModule,
        registration: ModuleRegistration
    ) -> None:
        """Register a new module"""
        async with self._lock:
            if module_id in self._modules:
                raise ValueError(f"Module {module_id} already registered")
            
            logger.info(f"Registering module: {module_id} ({module.name})")
            
            # Initialize module
            try:
                await module.initialize()
                self._modules[module_id] = module
                self._registrations[module_id] = registration
                registration.status = "active"
                logger.info(f"Module registered successfully: {module_id}")
            except Exception as e:
                logger.error(f"Module registration failed: {module_id} - {str(e)}")
                raise
    
    async def unregister_module(self, module_id: str) -> None:
        """Unregister a module"""
        async with self._lock:
            if module_id not in self._modules:
                raise ValueError(f"Module {module_id} not found")
            
            logger.info(f"Unregistering module: {module_id}")
            
            module = self._modules[module_id]
            try:
                await module.shutdown()
                del self._modules[module_id]
                del self._registrations[module_id]
                logger.info(f"Module unregistered: {module_id}")
            except Exception as e:
                logger.error(f"Module unregistration failed: {module_id} - {str(e)}")
                raise
    
    async def get_module(self, module_id: str) -> Optional[BaseModule]:
        """Get a module by ID"""
        return self._modules.get(module_id)
    
    async def get_all_modules(self) -> Dict[str, BaseModule]:
        """Get all registered modules"""
        return self._modules.copy()
    
    async def get_modules_by_department(self, department: str) -> List[BaseModule]:
        """Get modules for a specific department/feature area"""
        return [
            module for module in self._modules.values()
            if module.department == department
        ]
    
    async def get_module_registration(self, module_id: str) -> Optional[ModuleRegistration]:
        """Get module registration info"""
        return self._registrations.get(module_id)
    
    async def list_registrations(self) -> List[ModuleRegistration]:
        """List all module registrations"""
        return list(self._registrations.values())
    
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Perform health checks on all modules"""
        health_results = {}
        
        # Run health checks concurrently
        tasks = {
            module_id: module.health_check()
            for module_id, module in self._modules.items()
        }
        
        if tasks:
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            
            for (module_id, _), result in zip(tasks.items(), results):
                if isinstance(result, Exception):
                    health_results[module_id] = {
                        "status": "error",
                        "error": str(result),
                        "last_check": datetime.utcnow().isoformat()
                    }
                else:
                    health_results[module_id] = result
        
        return health_results
    
    async def find_module_for_tool(self, tool_name: str) -> Optional[BaseModule]:
        """Find which module provides a specific tool"""
        for module in self._modules.values():
            if await module.has_tool(tool_name):
                return module
        return None
    
    async def find_module_for_resource(self, uri: str) -> Optional[BaseModule]:
        """Find which module provides a specific resource"""
        for module in self._modules.values():
            if await module.has_resource(uri):
                return module
        return None
    
    async def find_module_for_prompt(self, name: str) -> Optional[BaseModule]:
        """Find which module provides a specific prompt"""
        for module in self._modules.values():
            if await module.has_prompt(name):
                return module
        return None
    
    async def shutdown_all(self) -> None:
        """Shutdown all modules"""
        logger.info("Shutting down all modules")
        
        # Shutdown modules concurrently
        tasks = [
            module.shutdown()
            for module in self._modules.values()
        ]
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for module_id, result in zip(self._modules.keys(), results):
                if isinstance(result, Exception):
                    logger.error(f"Module shutdown error: {module_id} - {str(result)}")
        
        self._modules.clear()
        self._registrations.clear()
        logger.info("All modules shutdown")


# Global registry instance
_registry: Optional[ModuleRegistry] = None


def get_module_registry() -> ModuleRegistry:
    """Get the global module registry instance"""
    global _registry
    if _registry is None:
        _registry = ModuleRegistry()
    return _registry


async def register_module(
    module_id: str,
    module_class: type[BaseModule],
    config: ModuleConfig
) -> None:
    """Convenience function to register a module"""
    registry = get_module_registry()
    
    # Create module instance
    module = module_class(config)
    
    # Create registration
    registration = ModuleRegistration(
        module_id=module_id,
        name=config.name,
        version=config.version,
        department=config.department,
        capabilities=[cap.value for cap in config.capabilities],
        status="pending",
        module_metadata={
            "description": config.description,
            "settings": config.settings
        }
    )
    
    await registry.register_module(module_id, module, registration)