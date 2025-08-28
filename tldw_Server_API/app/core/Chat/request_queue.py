# request_queue.py
# Description: Request queuing system with backpressure and priority management
#
# Imports
import asyncio
import time
from dataclasses import dataclass, field
from enum import IntEnum
from heapq import heappush, heappop
from typing import Any, Dict, Optional
from loguru import logger

#######################################################################################################################
#
# Types:

class RequestPriority(IntEnum):
    """Request priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4

@dataclass(order=True)
class QueuedRequest:
    """Represents a queued request with priority."""
    priority: int
    timestamp: float = field(compare=False)
    request_id: str = field(compare=False)
    request_data: Any = field(compare=False)
    future: asyncio.Future = field(compare=False)
    client_id: str = field(compare=False)
    estimated_tokens: int = field(compare=False, default=0)

#######################################################################################################################
#
# Classes:

class RequestQueue:
    """
    Priority-based request queue with backpressure management.
    """
    
    def __init__(
        self,
        max_queue_size: int = 100,
        max_concurrent: int = 10,
        timeout: float = 300.0
    ):
        """
        Initialize the request queue.
        
        Args:
            max_queue_size: Maximum number of queued requests
            max_concurrent: Maximum concurrent processing
            timeout: Request timeout in seconds
        """
        self.max_queue_size = max_queue_size
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        
        self.queue = []  # Priority queue
        self.processing_count = 0
        self.total_processed = 0
        self.total_rejected = 0
        
        self._lock = asyncio.Lock()
        self._processing_semaphore = asyncio.Semaphore(max_concurrent)
        self._workers = []
        self._running = False
    
    async def start(self, num_workers: int = 4):
        """
        Start the queue workers.
        
        Args:
            num_workers: Number of worker tasks
        """
        if self._running:
            return
        
        self._running = True
        for i in range(num_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self._workers.append(worker)
        
        logger.info(f"Started {num_workers} queue workers")
    
    async def stop(self):
        """Stop the queue workers."""
        self._running = False
        
        # Cancel all workers
        for worker in self._workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        
        logger.info("Stopped queue workers")
    
    async def _worker(self, worker_id: str):
        """
        Worker task that processes queued requests.
        
        Args:
            worker_id: Worker identifier
        """
        logger.debug(f"Worker {worker_id} started")
        
        while self._running:
            try:
                # Get next request from queue
                request = await self._get_next_request()
                if not request:
                    # No requests, wait a bit
                    await asyncio.sleep(0.1)
                    continue
                
                # Check if request has timed out
                if time.time() - request.timestamp > self.timeout:
                    logger.warning(f"Request {request.request_id} timed out in queue")
                    request.future.set_exception(
                        TimeoutError(f"Request timed out after {self.timeout}s in queue")
                    )
                    continue
                
                # Process request
                async with self._processing_semaphore:
                    self.processing_count += 1
                    try:
                        # Execute the actual request processing
                        # This would be replaced with actual chat processing
                        result = await self._process_request(request)
                        request.future.set_result(result)
                        self.total_processed += 1
                    except Exception as e:
                        logger.error(f"Error processing request {request.request_id}: {e}")
                        request.future.set_exception(e)
                    finally:
                        self.processing_count -= 1
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1)
        
        logger.debug(f"Worker {worker_id} stopped")
    
    async def _get_next_request(self) -> Optional[QueuedRequest]:
        """Get the next request from the priority queue."""
        async with self._lock:
            if self.queue:
                return heappop(self.queue)
        return None
    
    async def _process_request(self, request: QueuedRequest) -> Any:
        """
        Process a request (placeholder for actual processing).
        
        Args:
            request: The request to process
            
        Returns:
            Processing result
        """
        # This would be replaced with actual chat processing logic
        logger.debug(f"Processing request {request.request_id}")
        
        # Simulate processing time based on estimated tokens
        processing_time = 0.001 * request.estimated_tokens
        await asyncio.sleep(processing_time)
        
        return {"status": "completed", "request_id": request.request_id}
    
    async def enqueue(
        self,
        request_id: str,
        request_data: Any,
        client_id: str,
        priority: RequestPriority = RequestPriority.NORMAL,
        estimated_tokens: int = 0
    ) -> asyncio.Future:
        """
        Add a request to the queue.
        
        Args:
            request_id: Unique request identifier
            request_data: The request data
            client_id: Client identifier
            priority: Request priority
            estimated_tokens: Estimated token count for the request
            
        Returns:
            Future that will contain the result
            
        Raises:
            ValueError: If queue is full
        """
        async with self._lock:
            # Check queue size (backpressure)
            if len(self.queue) >= self.max_queue_size:
                self.total_rejected += 1
                raise ValueError(f"Queue full: {len(self.queue)} requests pending")
            
            # Create queued request
            future = asyncio.Future()
            request = QueuedRequest(
                priority=priority.value,
                timestamp=time.time(),
                request_id=request_id,
                request_data=request_data,
                future=future,
                client_id=client_id,
                estimated_tokens=estimated_tokens
            )
            
            # Add to priority queue
            heappush(self.queue, request)
            
            logger.debug(f"Enqueued request {request_id} with priority {priority.name}")
            
        return future
    
    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get current queue status.
        
        Returns:
            Dictionary with queue statistics
        """
        return {
            "queue_size": len(self.queue),
            "processing_count": self.processing_count,
            "max_queue_size": self.max_queue_size,
            "max_concurrent": self.max_concurrent,
            "total_processed": self.total_processed,
            "total_rejected": self.total_rejected,
            "is_running": self._running
        }
    
    async def clear_queue(self):
        """Clear all pending requests."""
        async with self._lock:
            # Cancel all pending requests
            for request in self.queue:
                request.future.cancel()
            self.queue.clear()
            logger.info("Cleared request queue")


class RateLimitedQueue(RequestQueue):
    """
    Request queue with rate limiting per client and globally.
    """
    
    def __init__(
        self,
        max_queue_size: int = 100,
        max_concurrent: int = 10,
        timeout: float = 300.0,
        global_rate_limit: int = 60,  # requests per minute
        per_client_rate_limit: int = 20  # requests per minute per client
    ):
        """
        Initialize rate-limited queue.
        
        Args:
            max_queue_size: Maximum queue size
            max_concurrent: Maximum concurrent processing
            timeout: Request timeout
            global_rate_limit: Global requests per minute
            per_client_rate_limit: Per-client requests per minute
        """
        super().__init__(max_queue_size, max_concurrent, timeout)
        
        self.global_rate_limit = global_rate_limit
        self.per_client_rate_limit = per_client_rate_limit
        
        # Track request times for rate limiting
        self.global_request_times = []
        self.client_request_times = {}
    
    def _check_rate_limit(self, client_id: str) -> bool:
        """
        Check if request is within rate limits.
        
        Args:
            client_id: Client identifier
            
        Returns:
            True if within limits, False otherwise
        """
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Clean old entries
        self.global_request_times = [
            t for t in self.global_request_times if t > minute_ago
        ]
        
        if client_id in self.client_request_times:
            self.client_request_times[client_id] = [
                t for t in self.client_request_times[client_id] if t > minute_ago
            ]
        
        # Check global rate limit
        if len(self.global_request_times) >= self.global_rate_limit:
            return False
        
        # Check per-client rate limit
        client_requests = self.client_request_times.get(client_id, [])
        if len(client_requests) >= self.per_client_rate_limit:
            return False
        
        # Record request time
        self.global_request_times.append(current_time)
        if client_id not in self.client_request_times:
            self.client_request_times[client_id] = []
        self.client_request_times[client_id].append(current_time)
        
        return True
    
    async def enqueue(
        self,
        request_id: str,
        request_data: Any,
        client_id: str,
        priority: RequestPriority = RequestPriority.NORMAL,
        estimated_tokens: int = 0
    ) -> asyncio.Future:
        """
        Add a request to the queue with rate limiting.
        
        Args:
            request_id: Unique request identifier
            request_data: The request data
            client_id: Client identifier
            priority: Request priority
            estimated_tokens: Estimated token count
            
        Returns:
            Future that will contain the result
            
        Raises:
            ValueError: If queue is full or rate limit exceeded
        """
        # Check rate limits
        if not self._check_rate_limit(client_id):
            raise ValueError(f"Rate limit exceeded for client {client_id}")
        
        # Proceed with normal enqueue
        return await super().enqueue(
            request_id, request_data, client_id, priority, estimated_tokens
        )


# Global queue instance
_request_queue: Optional[RateLimitedQueue] = None

def get_request_queue() -> Optional[RateLimitedQueue]:
    """Get the global request queue instance."""
    return _request_queue

def initialize_request_queue(
    max_queue_size: int = 100,
    max_concurrent: int = 10,
    global_rate_limit: int = 60,
    per_client_rate_limit: int = 20
) -> RateLimitedQueue:
    """
    Initialize the global request queue.
    
    Args:
        max_queue_size: Maximum queue size
        max_concurrent: Maximum concurrent processing
        global_rate_limit: Global rate limit
        per_client_rate_limit: Per-client rate limit
        
    Returns:
        The initialized queue
    """
    global _request_queue
    _request_queue = RateLimitedQueue(
        max_queue_size=max_queue_size,
        max_concurrent=max_concurrent,
        global_rate_limit=global_rate_limit,
        per_client_rate_limit=per_client_rate_limit
    )
    return _request_queue