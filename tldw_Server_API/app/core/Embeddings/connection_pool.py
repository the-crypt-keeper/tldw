# connection_pool.py
# Connection pooling for API providers to improve performance

import asyncio
import time
from typing import Dict, Optional, Any
from contextlib import asynccontextmanager
import aiohttp
from aiohttp import ClientTimeout, ClientSession, TCPConnector
from loguru import logger
import threading

from tldw_Server_API.app.core.Embeddings.circuit_breaker import circuit_breaker


class ConnectionPool:
    """
    Manages connection pools for different API providers.
    Provides reusable connections with proper lifecycle management.
    """
    
    def __init__(
        self,
        provider: str,
        max_connections: int = 10,
        max_keepalive_connections: int = 5,
        keepalive_timeout: int = 30,
        timeout_seconds: int = 30,
        retry_attempts: int = 3
    ):
        """
        Initialize connection pool for a provider.
        
        Args:
            provider: Name of the provider (e.g., 'openai', 'cohere')
            max_connections: Maximum number of connections in pool
            max_keepalive_connections: Max idle connections to keep alive
            keepalive_timeout: Seconds to keep idle connections alive
            timeout_seconds: Request timeout in seconds
            retry_attempts: Number of retry attempts on failure
        """
        self.provider = provider
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        self.keepalive_timeout = keepalive_timeout
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = retry_attempts
        
        self.session: Optional[ClientSession] = None
        self._lock = threading.RLock()
        self._usage_stats = {
            'requests': 0,
            'failures': 0,
            'total_time': 0,
            'active_connections': 0
        }
        
        logger.info(
            f"ConnectionPool for {provider} initialized: "
            f"max_connections={max_connections}, "
            f"keepalive_timeout={keepalive_timeout}s"
        )
    
    async def _create_session(self) -> ClientSession:
        """Create a new aiohttp session with connection pooling."""
        connector = TCPConnector(
            limit=self.max_connections,
            limit_per_host=self.max_connections,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
            keepalive_timeout=self.keepalive_timeout,
            force_close=False
        )
        
        timeout = ClientTimeout(
            total=self.timeout_seconds,
            connect=5,
            sock_connect=5,
            sock_read=self.timeout_seconds
        )
        
        return ClientSession(
            connector=connector,
            timeout=timeout,
            trust_env=True  # Use system proxy settings if available
        )
    
    async def get_session(self) -> ClientSession:
        """Get or create the session."""
        if self.session is None or self.session.closed:
            self.session = await self._create_session()
        return self.session
    
    @asynccontextmanager
    async def acquire_connection(self):
        """
        Context manager to acquire a connection from the pool.
        
        Yields:
            ClientSession: An aiohttp session for making requests
        """
        session = await self.get_session()
        
        with self._lock:
            self._usage_stats['active_connections'] += 1
        
        try:
            yield session
        finally:
            with self._lock:
                self._usage_stats['active_connections'] -= 1
    
    @circuit_breaker(name="api_request", failure_threshold=5, recovery_timeout=60)
    async def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        params: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Make an HTTP request using the connection pool.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Target URL
            headers: Request headers
            json_data: JSON payload
            data: Form data or raw data
            params: Query parameters
            
        Returns:
            Response data as dictionary
            
        Raises:
            aiohttp.ClientError: On network errors
            ValueError: On invalid responses
        """
        start_time = time.time()
        
        async with self.acquire_connection() as session:
            try:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=json_data,
                    data=data,
                    params=params
                ) as response:
                    # Update statistics
                    elapsed = time.time() - start_time
                    with self._lock:
                        self._usage_stats['requests'] += 1
                        self._usage_stats['total_time'] += elapsed
                    
                    # Check response status
                    if response.status >= 400:
                        error_text = await response.text()
                        logger.error(
                            f"{self.provider} API error: "
                            f"status={response.status}, body={error_text[:500]}"
                        )
                        
                        with self._lock:
                            self._usage_stats['failures'] += 1
                        
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=error_text,
                            headers=response.headers
                        )
                    
                    # Parse response
                    if 'application/json' in response.headers.get('content-type', ''):
                        return await response.json()
                    else:
                        text = await response.text()
                        return {'text': text}
                        
            except asyncio.TimeoutError as e:
                with self._lock:
                    self._usage_stats['failures'] += 1
                logger.error(f"{self.provider} request timeout after {self.timeout_seconds}s")
                raise
            except aiohttp.ClientError as e:
                with self._lock:
                    self._usage_stats['failures'] += 1
                logger.error(f"{self.provider} connection error: {e}")
                raise
            except Exception as e:
                with self._lock:
                    self._usage_stats['failures'] += 1
                logger.error(f"{self.provider} unexpected error: {e}")
                raise
    
    async def close(self):
        """Close the connection pool and cleanup resources."""
        if self.session and not self.session.closed:
            await self.session.close()
            # Allow time for graceful shutdown
            await asyncio.sleep(0.25)
        
        logger.info(f"ConnectionPool for {self.provider} closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        with self._lock:
            avg_time = (
                self._usage_stats['total_time'] / self._usage_stats['requests']
                if self._usage_stats['requests'] > 0 else 0
            )
            
            success_rate = (
                (self._usage_stats['requests'] - self._usage_stats['failures']) 
                / self._usage_stats['requests'] * 100
                if self._usage_stats['requests'] > 0 else 100
            )
            
            return {
                'provider': self.provider,
                'total_requests': self._usage_stats['requests'],
                'failed_requests': self._usage_stats['failures'],
                'success_rate': success_rate,
                'average_response_time': avg_time,
                'active_connections': self._usage_stats['active_connections'],
                'max_connections': self.max_connections
            }
    
    def reset_stats(self):
        """Reset usage statistics."""
        with self._lock:
            self._usage_stats = {
                'requests': 0,
                'failures': 0,
                'total_time': 0,
                'active_connections': self._usage_stats['active_connections']
            }


class ConnectionPoolManager:
    """
    Manages multiple connection pools for different providers.
    """
    
    def __init__(self):
        self.pools: Dict[str, ConnectionPool] = {}
        self._lock = threading.RLock()
    
    def get_pool(
        self,
        provider: str,
        max_connections: int = 10,
        **kwargs
    ) -> ConnectionPool:
        """
        Get or create a connection pool for a provider.
        
        Args:
            provider: Provider name
            max_connections: Max connections in pool
            **kwargs: Additional pool configuration
            
        Returns:
            ConnectionPool instance
        """
        with self._lock:
            if provider not in self.pools:
                self.pools[provider] = ConnectionPool(
                    provider=provider,
                    max_connections=max_connections,
                    **kwargs
                )
            return self.pools[provider]
    
    async def close_all(self):
        """Close all connection pools."""
        close_tasks = []
        
        with self._lock:
            for pool in self.pools.values():
                close_tasks.append(pool.close())
        
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        
        self.pools.clear()
        logger.info("All connection pools closed")
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all connection pools."""
        with self._lock:
            return {
                provider: pool.get_stats()
                for provider, pool in self.pools.items()
            }


# Global connection pool manager
_pool_manager: Optional[ConnectionPoolManager] = None


def get_pool_manager() -> ConnectionPoolManager:
    """Get or create the global connection pool manager."""
    global _pool_manager
    if _pool_manager is None:
        _pool_manager = ConnectionPoolManager()
    return _pool_manager


async def get_connection_pool(
    provider: str,
    max_connections: int = 10,
    **kwargs
) -> ConnectionPool:
    """
    Get a connection pool for the specified provider.
    
    Args:
        provider: Provider name
        max_connections: Maximum connections
        **kwargs: Additional configuration
        
    Returns:
        ConnectionPool instance
    """
    manager = get_pool_manager()
    return manager.get_pool(provider, max_connections, **kwargs)


# Cleanup function for graceful shutdown
async def cleanup_connection_pools():
    """Clean up all connection pools on shutdown."""
    global _pool_manager
    if _pool_manager:
        await _pool_manager.close_all()
        _pool_manager = None