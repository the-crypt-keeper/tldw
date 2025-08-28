"""
Base module interface for unified MCP with production features

Includes health checking, metrics, circuit breaker support, and proper error handling.
"""

from typing import Dict, Any, List, Optional, Set
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from loguru import logger
import asyncio
import time


class HealthStatus(str, Enum):
    """Module health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ModuleHealth:
    """Module health information"""
    status: HealthStatus
    message: str = ""
    last_check: datetime = field(default_factory=datetime.utcnow)
    checks: Dict[str, bool] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY
    
    @property
    def is_operational(self) -> bool:
        return self.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]


@dataclass
class ModuleMetrics:
    """Module performance metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0
    last_request_time: Optional[datetime] = None
    error_rate: float = 0
    avg_latency_ms: float = 0
    
    def record_request(self, success: bool, latency_ms: float):
        """Record a request"""
        self.total_requests += 1
        self.total_latency_ms += latency_ms
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        self.last_request_time = datetime.utcnow()
        self.error_rate = self.failed_requests / max(1, self.total_requests)
        self.avg_latency_ms = self.total_latency_ms / max(1, self.total_requests)


@dataclass
class ModuleConfig:
    """Module configuration"""
    name: str
    version: str = "1.0.0"
    description: str = ""
    department: str = "general"
    enabled: bool = True
    timeout_seconds: int = 30
    max_retries: int = 3
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    settings: Dict[str, Any] = field(default_factory=dict)


class BaseModule(ABC):
    """
    Enhanced base module interface with production features.
    
    All modules must inherit from this class and implement required methods.
    """
    
    def __init__(self, config: ModuleConfig):
        self.config = config
        self.name = config.name
        self.version = config.version
        self.department = config.department
        self.enabled = config.enabled
        
        # Health and metrics
        self._health = ModuleHealth(status=HealthStatus.UNKNOWN)
        self._metrics = ModuleMetrics()
        
        # Circuit breaker state
        self._circuit_breaker_failures = 0
        self._circuit_breaker_open_until = None
        
        # Initialization state
        self._initialized = False
        self._initializing = False
        self._shutdown = False
        
        # Tools, resources, and prompts cache
        self._tools_cache = None
        self._resources_cache = None
        self._prompts_cache = None
        
        logger.info(f"Module created: {self.name} v{self.version}")
    
    async def initialize(self) -> None:
        """
        Initialize the module with error handling and health check.
        
        This method should not be overridden. Override on_initialize instead.
        """
        if self._initialized:
            logger.warning(f"Module {self.name} already initialized")
            return
        
        if self._initializing:
            logger.warning(f"Module {self.name} is already initializing")
            return
        
        self._initializing = True
        logger.info(f"Initializing module: {self.name}")
        
        try:
            # Call module-specific initialization
            await self.on_initialize()
            
            # Perform initial health check
            health = await self.health_check()
            
            if not health.is_operational:
                raise Exception(f"Module failed health check: {health.message}")
            
            self._initialized = True
            logger.info(f"Module initialized successfully: {self.name}")
            
        except Exception as e:
            logger.error(f"Module initialization failed: {self.name} - {str(e)}")
            self._health = ModuleHealth(
                status=HealthStatus.UNHEALTHY,
                message=f"Initialization failed: {str(e)}"
            )
            raise
        finally:
            self._initializing = False
    
    async def shutdown(self) -> None:
        """
        Shutdown the module gracefully.
        
        This method should not be overridden. Override on_shutdown instead.
        """
        if self._shutdown:
            logger.warning(f"Module {self.name} already shut down")
            return
        
        logger.info(f"Shutting down module: {self.name}")
        
        try:
            # Call module-specific shutdown
            await self.on_shutdown()
            
            self._shutdown = True
            self._initialized = False
            self._health = ModuleHealth(
                status=HealthStatus.UNKNOWN,
                message="Module shut down"
            )
            
            logger.info(f"Module shut down successfully: {self.name}")
            
        except Exception as e:
            logger.error(f"Module shutdown failed: {self.name} - {str(e)}")
            # Continue shutdown even if there's an error
    
    async def health_check(self) -> ModuleHealth:
        """
        Perform health check with caching.
        
        This method should not be overridden. Override check_health instead.
        """
        try:
            # Check if we need a new health check
            if self._health.last_check:
                time_since_check = datetime.utcnow() - self._health.last_check
                if time_since_check < timedelta(seconds=10):  # Cache for 10 seconds
                    return self._health
            
            # Perform health check
            checks = await self.check_health()
            
            # Determine overall status
            if all(checks.values()):
                status = HealthStatus.HEALTHY
                message = "All checks passed"
            elif any(checks.values()):
                status = HealthStatus.DEGRADED
                failed = [k for k, v in checks.items() if not v]
                message = f"Some checks failed: {', '.join(failed)}"
            else:
                status = HealthStatus.UNHEALTHY
                message = "All checks failed"
            
            self._health = ModuleHealth(
                status=status,
                message=message,
                checks=checks
            )
            
        except Exception as e:
            logger.error(f"Health check failed for {self.name}: {str(e)}")
            self._health = ModuleHealth(
                status=HealthStatus.UNHEALTHY,
                message=f"Health check error: {str(e)}"
            )
        
        return self._health
    
    def is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open"""
        if self._circuit_breaker_open_until:
            if datetime.utcnow() < self._circuit_breaker_open_until:
                return True
            else:
                # Reset circuit breaker
                self._circuit_breaker_open_until = None
                self._circuit_breaker_failures = 0
        return False
    
    def record_circuit_breaker_failure(self):
        """Record a failure for circuit breaker"""
        self._circuit_breaker_failures += 1
        
        if self._circuit_breaker_failures >= self.config.circuit_breaker_threshold:
            self._circuit_breaker_open_until = (
                datetime.utcnow() + 
                timedelta(seconds=self.config.circuit_breaker_timeout)
            )
            logger.warning(
                f"Circuit breaker opened for module {self.name} "
                f"until {self._circuit_breaker_open_until}"
            )
    
    def record_circuit_breaker_success(self):
        """Record a success for circuit breaker"""
        if self._circuit_breaker_failures > 0:
            self._circuit_breaker_failures -= 1
    
    async def execute_with_circuit_breaker(self, operation, *args, **kwargs):
        """Execute an operation with circuit breaker protection"""
        if self.is_circuit_breaker_open():
            raise Exception(f"Circuit breaker is open for module {self.name}")
        
        start_time = time.time()
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                operation(*args, **kwargs),
                timeout=self.config.timeout_seconds
            )
            
            # Record success
            latency_ms = (time.time() - start_time) * 1000
            self._metrics.record_request(True, latency_ms)
            self.record_circuit_breaker_success()
            
            return result
            
        except asyncio.TimeoutError:
            # Record failure
            latency_ms = (time.time() - start_time) * 1000
            self._metrics.record_request(False, latency_ms)
            self.record_circuit_breaker_failure()
            
            logger.error(f"Operation timeout in module {self.name}")
            raise Exception(f"Operation timeout after {self.config.timeout_seconds}s")
            
        except Exception as e:
            # Record failure
            latency_ms = (time.time() - start_time) * 1000
            self._metrics.record_request(False, latency_ms)
            self.record_circuit_breaker_failure()
            
            logger.error(f"Operation failed in module {self.name}: {str(e)}")
            raise
    
    def get_metrics(self) -> ModuleMetrics:
        """Get module metrics"""
        return self._metrics
    
    # Abstract methods that modules must implement
    
    @abstractmethod
    async def on_initialize(self) -> None:
        """Module-specific initialization logic"""
        pass
    
    @abstractmethod
    async def on_shutdown(self) -> None:
        """Module-specific shutdown logic"""
        pass
    
    @abstractmethod
    async def check_health(self) -> Dict[str, bool]:
        """
        Module-specific health checks.
        
        Returns:
            Dictionary of check_name -> passed (True/False)
        """
        pass
    
    @abstractmethod
    async def get_tools(self) -> List[Dict[str, Any]]:
        """
        Get list of tools provided by this module.
        
        Returns:
            List of tool definitions in MCP format
        """
        pass
    
    @abstractmethod
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
        
        Returns:
            Tool execution result
        """
        pass
    
    # Optional methods with default implementations
    
    async def has_tool(self, tool_name: str) -> bool:
        """Check if module provides a tool"""
        if self._tools_cache is None:
            self._tools_cache = await self.get_tools()
        return any(tool["name"] == tool_name for tool in self._tools_cache)
    
    async def get_resources(self) -> List[Dict[str, Any]]:
        """Get list of resources (optional)"""
        return []
    
    async def has_resource(self, uri: str) -> bool:
        """Check if module provides a resource"""
        if self._resources_cache is None:
            self._resources_cache = await self.get_resources()
        return any(resource["uri"] == uri for resource in self._resources_cache)
    
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read a resource"""
        raise NotImplementedError(f"Resource reading not implemented for {self.name}")
    
    async def get_prompts(self) -> List[Dict[str, Any]]:
        """Get list of prompts (optional)"""
        return []
    
    async def has_prompt(self, name: str) -> bool:
        """Check if module provides a prompt"""
        if self._prompts_cache is None:
            self._prompts_cache = await self.get_prompts()
        return any(prompt["name"] == name for prompt in self._prompts_cache)
    
    async def get_prompt(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Get a prompt with arguments"""
        raise NotImplementedError(f"Prompt not implemented for {self.name}")
    
    # Validation helpers
    
    def validate_tool_arguments(self, tool_name: str, arguments: Dict[str, Any]):
        """
        Validate tool arguments against schema.
        
        Override this to add custom validation.
        """
        # Basic validation - check required fields
        pass
    
    def sanitize_input(self, input_data: Any) -> Any:
        """
        Sanitize user input to prevent injection attacks.
        
        Override this to add custom sanitization.
        """
        # Basic sanitization
        if isinstance(input_data, str):
            # Remove potential SQL injection patterns
            dangerous_patterns = ["';", '";', '--', '/*', '*/', 'xp_', 'sp_']
            for pattern in dangerous_patterns:
                if pattern in input_data.lower():
                    raise ValueError(f"Potentially dangerous input detected: {pattern}")
        
        return input_data


# Helper functions for creating MCP-compliant definitions

def create_tool_definition(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create MCP-compliant tool definition"""
    tool_def = {
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": parameters.get("properties", {}),
            "required": parameters.get("required", [])
        }
    }
    
    if metadata:
        tool_def["metadata"] = metadata
    
    return tool_def


def create_resource_definition(
    uri: str,
    name: str,
    description: str,
    mime_type: str = "application/json",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create MCP-compliant resource definition"""
    resource_def = {
        "uri": uri,
        "name": name,
        "description": description,
        "mimeType": mime_type
    }
    
    if metadata:
        resource_def["metadata"] = metadata
    
    return resource_def


def create_prompt_definition(
    name: str,
    description: str,
    arguments: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create MCP-compliant prompt definition"""
    prompt_def = {
        "name": name,
        "description": description,
        "arguments": arguments or []
    }
    
    if metadata:
        prompt_def["metadata"] = metadata
    
    return prompt_def