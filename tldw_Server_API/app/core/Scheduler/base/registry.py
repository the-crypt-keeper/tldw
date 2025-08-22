"""
Type-safe task handler registry.
Manages registration and validation of task handlers.
"""

from typing import Callable, Dict, Any, Optional, Union, Awaitable, List
import inspect
from functools import wraps
from loguru import logger


class TaskRegistry:
    """
    Type-safe registry for task handlers.
    
    Provides decorator-based registration and validation of task handlers.
    Ensures handlers are properly typed and can be called safely.
    """
    
    def __init__(self):
        """Initialize the registry"""
        self._handlers: Dict[str, Callable] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._validated = False
    
    def task(self, 
             name: Optional[str] = None,
             max_retries: int = 3,
             timeout: int = 300,
             queue: str = "default") -> Callable:
        """
        Decorator to register a task handler.
        
        Args:
            name: Handler name (defaults to function name)
            max_retries: Maximum retry attempts
            timeout: Task timeout in seconds
            queue: Queue name for this task type
            
        Returns:
            Decorator function
            
        Example:
            @registry.task(max_retries=5, timeout=600)
            async def process_video(video_id: int) -> dict:
                # Process video
                return {"status": "completed"}
        """
        def decorator(func: Callable) -> Callable:
            # Determine handler name
            handler_name = name or f"{func.__module__}.{func.__name__}"
            
            # Validate handler signature
            sig = inspect.signature(func)
            if not sig.parameters and not inspect.iscoroutinefunction(func):
                # Allow no-parameter handlers for simple tasks
                pass
            elif len(sig.parameters) < 1:
                raise ValueError(
                    f"Handler {handler_name} must accept at least one parameter (payload)"
                )
            
            # Store handler and metadata
            self._handlers[handler_name] = func
            self._metadata[handler_name] = {
                'max_retries': max_retries,
                'timeout': timeout,
                'queue': queue,
                'is_async': inspect.iscoroutinefunction(func),
                'signature': sig,
                'module': func.__module__,
                'function': func.__name__
            }
            
            # Add handler name as attribute for reference
            func._task_name = handler_name
            func._task_metadata = self._metadata[handler_name]
            
            logger.debug(f"Registered task handler: {handler_name}")
            
            @wraps(func)
            async def wrapper(*args, **kwargs):
                """Wrapper to ensure async execution"""
                if inspect.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    # Run sync function in thread pool to avoid blocking
                    import asyncio
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, func, *args, **kwargs)
            
            return wrapper
        
        return decorator
    
    def get_handler(self, name: str) -> Callable:
        """
        Get a registered handler by name.
        
        Args:
            name: Handler name
            
        Returns:
            Handler function
            
        Raises:
            ValueError: If handler not found
        """
        if name not in self._handlers:
            available = ', '.join(sorted(self._handlers.keys()))
            raise ValueError(
                f"Handler '{name}' not registered. "
                f"Available handlers: {available}"
            )
        return self._handlers[name]
    
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """
        Get metadata for a handler.
        
        Args:
            name: Handler name
            
        Returns:
            Handler metadata
        """
        if name not in self._metadata:
            raise ValueError(f"Handler '{name}' not registered")
        return self._metadata[name]
    
    def list_handlers(self) -> List[str]:
        """
        List all registered handler names.
        
        Returns:
            List of handler names
        """
        return sorted(self._handlers.keys())
    
    def validate_all(self) -> None:
        """
        Validate all registered handlers.
        
        This should be called at startup to ensure all handlers
        are properly configured and callable.
        
        Raises:
            ValueError: If any handler is invalid
        """
        if self._validated:
            return
        
        errors = []
        
        for name, handler in self._handlers.items():
            try:
                # Test that handler is callable
                if not callable(handler):
                    errors.append(f"{name}: Handler is not callable")
                
                # Test metadata
                metadata = self._metadata.get(name)
                if not metadata:
                    errors.append(f"{name}: Missing metadata")
                
                # Validate timeout
                if metadata and metadata['timeout'] <= 0:
                    errors.append(f"{name}: Invalid timeout {metadata['timeout']}")
                
                # Validate max_retries
                if metadata and metadata['max_retries'] < 0:
                    errors.append(f"{name}: Invalid max_retries {metadata['max_retries']}")
                
            except Exception as e:
                errors.append(f"{name}: Validation error: {e}")
        
        if errors:
            error_msg = "Handler validation failed:\n" + "\n".join(errors)
            raise ValueError(error_msg)
        
        self._validated = True
        logger.info(f"Validated {len(self._handlers)} task handlers")
    
    def execute_handler(self, name: str, payload: Any) -> Union[Any, Awaitable[Any]]:
        """
        Execute a handler with payload.
        
        Args:
            name: Handler name
            payload: Task payload
            
        Returns:
            Handler result (may be awaitable)
        """
        handler = self.get_handler(name)
        metadata = self.get_metadata(name)
        
        # Prepare arguments based on handler signature
        sig = metadata['signature']
        
        if not sig.parameters:
            # No parameters - call without arguments
            return handler()
        else:
            # Pass payload as first argument
            # More sophisticated argument mapping could be added here
            return handler(payload)
    
    def clear(self) -> None:
        """Clear all registered handlers (useful for testing)"""
        self._handlers.clear()
        self._metadata.clear()
        self._validated = False
    
    def __len__(self) -> int:
        """Get number of registered handlers"""
        return len(self._handlers)
    
    def __contains__(self, name: str) -> bool:
        """Check if handler is registered"""
        return name in self._handlers
    
    def __repr__(self) -> str:
        """String representation"""
        return f"TaskRegistry({len(self._handlers)} handlers)"


# Global registry instance
_global_registry = TaskRegistry()


def get_registry() -> TaskRegistry:
    """Get the global task registry"""
    return _global_registry


def task(name: Optional[str] = None,
         max_retries: int = 3,
         timeout: int = 300,
         queue: str = "default") -> Callable:
    """
    Convenience decorator using the global registry.
    
    Example:
        from scheduler.base.registry import task
        
        @task(max_retries=5)
        async def send_email(data):
            # Send email
            return {"sent": True}
    """
    return _global_registry.task(name, max_retries, timeout, queue)