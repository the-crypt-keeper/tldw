"""
Module registry with health monitoring and circuit breaker support

Manages module lifecycle and provides intelligent routing with failover.
"""

import asyncio
from typing import Dict, Any, Optional, List, Set, Type
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from loguru import logger
import time

from .base import BaseModule, ModuleConfig, HealthStatus, ModuleHealth


class ModuleStatus(str, Enum):
    """Module registration status"""
    PENDING = "pending"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    INACTIVE = "inactive"
    ERROR = "error"


@dataclass
class ModuleRegistration:
    """Module registration information"""
    module_id: str
    module_type: Type[BaseModule]
    module_instance: Optional[BaseModule]
    config: ModuleConfig
    status: ModuleStatus
    registered_at: datetime
    last_health_check: Optional[datetime] = None
    error_message: Optional[str] = None
    capabilities: Set[str] = field(default_factory=set)
    
    def is_operational(self) -> bool:
        """Check if module is operational"""
        return self.status in [ModuleStatus.ACTIVE, ModuleStatus.DEGRADED]


class ModuleRegistry:
    """
    Enhanced module registry with production features.
    
    Features:
    - Automatic health monitoring
    - Circuit breaker protection
    - Intelligent routing with failover
    - Module dependency management
    - Graceful degradation
    """
    
    def __init__(self, health_check_interval: int = 60):
        self._modules: Dict[str, ModuleRegistration] = {}
        self._module_instances: Dict[str, BaseModule] = {}
        self._lock = asyncio.Lock()
        self._health_check_interval = health_check_interval
        self._health_check_task = None
        self._shutdown = False
        
        # Tool/Resource/Prompt to module mapping for fast lookup
        self._tool_registry: Dict[str, str] = {}  # tool_name -> module_id
        self._resource_registry: Dict[str, str] = {}  # resource_uri -> module_id
        self._prompt_registry: Dict[str, str] = {}  # prompt_name -> module_id
        
        logger.info("Module registry initialized")
    
    async def start_health_monitoring(self):
        """Start background health monitoring task"""
        if self._health_check_task is None:
            self._health_check_task = asyncio.create_task(self._health_monitor_loop())
            logger.info("Health monitoring started")
    
    async def stop_health_monitoring(self):
        """Stop health monitoring"""
        self._shutdown = True
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            logger.info("Health monitoring stopped")
    
    async def _health_monitor_loop(self):
        """Background task to monitor module health"""
        while not self._shutdown:
            try:
                await self.check_all_health()
                await asyncio.sleep(self._health_check_interval)
            except Exception as e:
                logger.error(f"Error in health monitor loop: {e}")
                await asyncio.sleep(10)  # Short delay on error
    
    async def register_module(
        self,
        module_id: str,
        module_type: Type[BaseModule],
        config: ModuleConfig
    ) -> None:
        """
        Register a new module.
        
        Args:
            module_id: Unique module identifier
            module_type: Module class type
            config: Module configuration
        """
        async with self._lock:
            if module_id in self._modules:
                raise ValueError(f"Module {module_id} already registered")
            
            logger.info(f"Registering module: {module_id}")
            
            # Create registration entry
            registration = ModuleRegistration(
                module_id=module_id,
                module_type=module_type,
                module_instance=None,
                config=config,
                status=ModuleStatus.PENDING,
                registered_at=datetime.utcnow()
            )
            
            self._modules[module_id] = registration
            
            # Initialize module
            try:
                await self._initialize_module(registration)
            except Exception as e:
                logger.error(f"Failed to initialize module {module_id}: {e}")
                registration.status = ModuleStatus.ERROR
                registration.error_message = str(e)
    
    async def _initialize_module(self, registration: ModuleRegistration):
        """Initialize a module instance"""
        registration.status = ModuleStatus.INITIALIZING
        
        try:
            # Create module instance
            module = registration.module_type(registration.config)
            registration.module_instance = module
            
            # Initialize module
            await module.initialize()
            
            # Store instance
            self._module_instances[registration.module_id] = module
            
            # Update registries
            await self._update_registries(registration.module_id, module)
            
            # Check health
            health = await module.health_check()
            
            if health.is_operational:
                registration.status = ModuleStatus.ACTIVE
                logger.info(f"Module {registration.module_id} initialized successfully")
            else:
                registration.status = ModuleStatus.DEGRADED
                logger.warning(f"Module {registration.module_id} initialized but degraded: {health.message}")
            
            registration.last_health_check = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Module initialization failed: {registration.module_id} - {e}")
            registration.status = ModuleStatus.ERROR
            registration.error_message = str(e)
            raise
    
    async def _update_registries(self, module_id: str, module: BaseModule):
        """Update tool/resource/prompt registries"""
        try:
            # Register tools
            tools = await module.get_tools()
            for tool in tools:
                tool_name = tool.get("name")
                if tool_name:
                    self._tool_registry[tool_name] = module_id
                    logger.debug(f"Registered tool {tool_name} -> {module_id}")
            
            # Register resources
            resources = await module.get_resources()
            for resource in resources:
                resource_uri = resource.get("uri")
                if resource_uri:
                    self._resource_registry[resource_uri] = module_id
                    logger.debug(f"Registered resource {resource_uri} -> {module_id}")
            
            # Register prompts
            prompts = await module.get_prompts()
            for prompt in prompts:
                prompt_name = prompt.get("name")
                if prompt_name:
                    self._prompt_registry[prompt_name] = module_id
                    logger.debug(f"Registered prompt {prompt_name} -> {module_id}")
            
        except Exception as e:
            logger.error(f"Failed to update registries for {module_id}: {e}")
    
    async def unregister_module(self, module_id: str) -> None:
        """Unregister a module"""
        async with self._lock:
            if module_id not in self._modules:
                raise ValueError(f"Module {module_id} not found")
            
            logger.info(f"Unregistering module: {module_id}")
            
            registration = self._modules[module_id]
            
            # Shutdown module if active
            if registration.module_instance:
                try:
                    await registration.module_instance.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down module {module_id}: {e}")
            
            # Remove from registries
            self._tool_registry = {k: v for k, v in self._tool_registry.items() if v != module_id}
            self._resource_registry = {k: v for k, v in self._resource_registry.items() if v != module_id}
            self._prompt_registry = {k: v for k, v in self._prompt_registry.items() if v != module_id}
            
            # Remove from instances
            if module_id in self._module_instances:
                del self._module_instances[module_id]
            
            # Remove registration
            del self._modules[module_id]
            
            logger.info(f"Module {module_id} unregistered")
    
    async def get_module(self, module_id: str) -> Optional[BaseModule]:
        """Get a module by ID"""
        registration = self._modules.get(module_id)
        if registration and registration.is_operational():
            return registration.module_instance
        return None
    
    async def get_all_modules(self) -> Dict[str, BaseModule]:
        """Get all operational modules"""
        result = {}
        for module_id, registration in self._modules.items():
            if registration.is_operational() and registration.module_instance:
                result[module_id] = registration.module_instance
        return result
    
    async def find_module_for_tool(self, tool_name: str) -> Optional[BaseModule]:
        """Find module that provides a specific tool"""
        module_id = self._tool_registry.get(tool_name)
        if module_id:
            return await self.get_module(module_id)
        
        # Fallback: search all modules
        for module_id, module in self._module_instances.items():
            if await module.has_tool(tool_name):
                self._tool_registry[tool_name] = module_id  # Cache for next time
                return module
        
        return None
    
    async def find_module_for_resource(self, uri: str) -> Optional[BaseModule]:
        """Find module that provides a specific resource"""
        module_id = self._resource_registry.get(uri)
        if module_id:
            return await self.get_module(module_id)
        
        # Fallback: search all modules
        for module_id, module in self._module_instances.items():
            if await module.has_resource(uri):
                self._resource_registry[uri] = module_id  # Cache for next time
                return module
        
        return None
    
    async def find_module_for_prompt(self, name: str) -> Optional[BaseModule]:
        """Find module that provides a specific prompt"""
        module_id = self._prompt_registry.get(name)
        if module_id:
            return await self.get_module(module_id)
        
        # Fallback: search all modules
        for module_id, module in self._module_instances.items():
            if await module.has_prompt(name):
                self._prompt_registry[name] = module_id  # Cache for next time
                return module
        
        return None
    
    async def check_all_health(self) -> Dict[str, ModuleHealth]:
        """Check health of all modules"""
        health_results = {}
        
        # Run health checks concurrently
        tasks = {}
        for module_id, registration in self._modules.items():
            if registration.module_instance:
                tasks[module_id] = registration.module_instance.health_check()
        
        if tasks:
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            
            for module_id, result in zip(tasks.keys(), results):
                registration = self._modules[module_id]
                
                if isinstance(result, Exception):
                    health_results[module_id] = ModuleHealth(
                        status=HealthStatus.UNHEALTHY,
                        message=f"Health check failed: {str(result)}"
                    )
                    registration.status = ModuleStatus.ERROR
                    registration.error_message = str(result)
                else:
                    health_results[module_id] = result
                    
                    # Update module status based on health
                    if result.status == HealthStatus.HEALTHY:
                        registration.status = ModuleStatus.ACTIVE
                    elif result.status == HealthStatus.DEGRADED:
                        registration.status = ModuleStatus.DEGRADED
                    else:
                        registration.status = ModuleStatus.INACTIVE
                
                registration.last_health_check = datetime.utcnow()
        
        return health_results
    
    async def get_module_status(self, module_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status for a module"""
        registration = self._modules.get(module_id)
        if not registration:
            return None
        
        status = {
            "module_id": module_id,
            "status": registration.status.value,
            "registered_at": registration.registered_at.isoformat(),
            "last_health_check": registration.last_health_check.isoformat() if registration.last_health_check else None,
            "error_message": registration.error_message,
            "config": {
                "name": registration.config.name,
                "version": registration.config.version,
                "enabled": registration.config.enabled
            }
        }
        
        # Add metrics if available
        if registration.module_instance:
            metrics = registration.module_instance.get_metrics()
            status["metrics"] = {
                "total_requests": metrics.total_requests,
                "error_rate": metrics.error_rate,
                "avg_latency_ms": metrics.avg_latency_ms
            }
        
        return status
    
    async def list_registrations(self) -> List[Dict[str, Any]]:
        """List all module registrations"""
        registrations = []
        
        for module_id, registration in self._modules.items():
            registrations.append(await self.get_module_status(module_id))
        
        return registrations
    
    async def shutdown_all(self) -> None:
        """Shutdown all modules gracefully"""
        logger.info("Shutting down all modules")
        
        # Stop health monitoring first
        await self.stop_health_monitoring()
        
        # Shutdown modules concurrently
        tasks = []
        for module_id, registration in self._modules.items():
            if registration.module_instance:
                tasks.append(registration.module_instance.shutdown())
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for module_id, result in zip(self._modules.keys(), results):
                if isinstance(result, Exception):
                    logger.error(f"Error shutting down module {module_id}: {result}")
        
        # Clear all registrations
        self._modules.clear()
        self._module_instances.clear()
        self._tool_registry.clear()
        self._resource_registry.clear()
        self._prompt_registry.clear()
        
        logger.info("All modules shut down")
    
    async def execute_with_failover(
        self,
        primary_module_id: str,
        fallback_module_ids: List[str],
        operation: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute an operation with automatic failover.
        
        Args:
            primary_module_id: Primary module to try
            fallback_module_ids: Fallback modules in order
            operation: Operation name to execute
            *args, **kwargs: Operation arguments
        
        Returns:
            Operation result
        
        Raises:
            Exception if all modules fail
        """
        errors = []
        
        # Try primary module
        module = await self.get_module(primary_module_id)
        if module:
            try:
                method = getattr(module, operation)
                return await module.execute_with_circuit_breaker(method, *args, **kwargs)
            except Exception as e:
                logger.warning(f"Primary module {primary_module_id} failed: {e}")
                errors.append((primary_module_id, str(e)))
        
        # Try fallback modules
        for module_id in fallback_module_ids:
            module = await self.get_module(module_id)
            if module:
                try:
                    method = getattr(module, operation)
                    return await module.execute_with_circuit_breaker(method, *args, **kwargs)
                except Exception as e:
                    logger.warning(f"Fallback module {module_id} failed: {e}")
                    errors.append((module_id, str(e)))
        
        # All modules failed
        error_msg = "All modules failed: " + "; ".join([f"{m}: {e}" for m, e in errors])
        raise Exception(error_msg)


# Singleton instance
_module_registry = None


def get_module_registry() -> ModuleRegistry:
    """Get or create module registry singleton"""
    global _module_registry
    if _module_registry is None:
        _module_registry = ModuleRegistry()
    return _module_registry


async def register_module(
    module_id: str,
    module_type: Type[BaseModule],
    config: ModuleConfig
) -> None:
    """Convenience function to register a module"""
    registry = get_module_registry()
    await registry.register_module(module_id, module_type, config)