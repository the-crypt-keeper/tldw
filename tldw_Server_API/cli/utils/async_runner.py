"""
Async command execution wrapper for CLI commands.
"""

import asyncio
import functools
from typing import Callable, Any


def run_async(func: Callable) -> Callable:
    """Decorator to run async functions in CLI commands."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper


def run_async_safely(coro) -> Any:
    """Safely run async coroutine with proper error handling."""
    try:
        return asyncio.run(coro)
    except KeyboardInterrupt:
        raise
    except Exception:
        raise