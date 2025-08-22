# streaming_utils.py
# Description: Utilities for handling streaming responses safely
#
# Imports
import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Union
from loguru import logger

#######################################################################################################################
#
# Constants:

# Load configuration values
from tldw_Server_API.app.core.config import load_comprehensive_config

_config = load_comprehensive_config()
# ConfigParser uses sections, check if Chat-Module section exists
_chat_config = {}
if _config and _config.has_section('Chat-Module'):
    _chat_config = dict(_config.items('Chat-Module'))

# Timeout for idle connections (seconds)
STREAMING_IDLE_TIMEOUT = int(_chat_config.get('streaming_idle_timeout_seconds', 300))  # Default 5 minutes

# Heartbeat interval for long-running streams (seconds)
HEARTBEAT_INTERVAL = int(_chat_config.get('streaming_heartbeat_interval_seconds', 30))

#######################################################################################################################
#
# Functions:

class StreamingResponseHandler:
    """
    Handles streaming responses with proper error handling, cleanup, and timeouts.
    """
    
    def __init__(
        self,
        conversation_id: str,
        model_name: str,
        idle_timeout: int = STREAMING_IDLE_TIMEOUT,
        heartbeat_interval: int = HEARTBEAT_INTERVAL,
        max_response_size: int = 10 * 1024 * 1024  # 10MB default
    ):
        """
        Initialize the streaming response handler.
        
        Args:
            conversation_id: ID of the conversation
            model_name: Name of the model being used
            idle_timeout: Timeout for idle connections in seconds
            heartbeat_interval: Interval for sending heartbeat messages
            max_response_size: Maximum response size in bytes
        """
        self.conversation_id = conversation_id
        self.model_name = model_name
        self.idle_timeout = idle_timeout
        self.heartbeat_interval = heartbeat_interval
        self.max_response_size = max_response_size
        self.last_activity = time.time()
        self.is_cancelled = False
        self.full_response = []
        self.response_size = 0
        self.error_occurred = False
        
    def update_activity(self):
        """Update the last activity timestamp."""
        self.last_activity = time.time()
    
    def is_timed_out(self) -> bool:
        """Check if the stream has timed out due to inactivity."""
        return (time.time() - self.last_activity) > self.idle_timeout
    
    def cancel(self):
        """Mark the stream as cancelled."""
        self.is_cancelled = True
        logger.info(f"Stream cancelled for conversation {self.conversation_id}")
    
    async def heartbeat_generator(self) -> AsyncIterator[str]:
        """
        Generate heartbeat messages to keep the connection alive.
        
        Yields:
            SSE heartbeat messages
        """
        while not self.is_cancelled and not self.error_occurred:
            await asyncio.sleep(self.heartbeat_interval)
            if self.is_timed_out():
                logger.warning(f"Stream timeout for conversation {self.conversation_id}")
                self.cancel()
                yield f"data: {json.dumps({'error': {'message': 'Stream timeout - no activity'}})}\n\n"
                break
            yield f": heartbeat {datetime.now(timezone.utc).isoformat()}\n\n"
    
    async def safe_stream_generator(
        self,
        stream: Union[Iterator, AsyncIterator],
        save_callback: Optional[callable] = None
    ) -> AsyncIterator[str]:
        """
        Safely generate streaming responses with error handling and cleanup.
        
        Args:
            stream: The stream to process (sync or async iterator)
            save_callback: Optional callback to save the full response
            
        Yields:
            SSE formatted messages
        """
        try:
            # Send initial metadata
            yield f"event: stream_start\ndata: {json.dumps({'conversation_id': self.conversation_id, 'model': self.model_name, 'timestamp': datetime.now(timezone.utc).isoformat()})}\n\n"
            self.update_activity()
            
            # Process the stream
            if hasattr(stream, '__aiter__'):
                # Async iterator
                async for chunk in stream:
                    if self.is_cancelled:
                        logger.info(f"Stream processing cancelled for {self.conversation_id}")
                        break
                    
                    if self.is_timed_out():
                        logger.warning(f"Stream timeout during processing for {self.conversation_id}")
                        yield f"data: {json.dumps({'error': {'message': 'Stream timeout'}})}\n\n"
                        break
                    
                    try:
                        # Process chunk
                        text_content = chunk.decode('utf-8', errors='replace') if isinstance(chunk, bytes) else str(chunk)
                        if text_content:
                            # Check response size limit
                            chunk_size = len(text_content.encode('utf-8'))
                            if self.response_size + chunk_size > self.max_response_size:
                                logger.warning(f"Stream response size limit exceeded for {self.conversation_id}")
                                yield f"data: {json.dumps({'error': {'message': 'Response size limit exceeded'}})}\n\n"
                                break
                            
                            self.full_response.append(text_content)
                            self.response_size += chunk_size
                            self.update_activity()
                            
                            # Send chunk to client
                            yield f"data: {json.dumps({'choices': [{'delta': {'content': text_content}}]})}\n\n"
                            
                    except Exception as e:
                        logger.error(f"Error processing stream chunk for {self.conversation_id}: {e}")
                        self.error_occurred = True
                        yield f"data: {json.dumps({'error': {'message': f'Error processing chunk: {str(e)}'}})}\n\n"
                        break
            else:
                # Sync iterator - run in executor
                loop = asyncio.get_running_loop()
                
                def sync_iterator():
                    try:
                        for chunk in stream:
                            if self.is_cancelled:
                                break
                            yield chunk
                    except StopIteration:
                        pass
                
                for chunk in sync_iterator():
                    if self.is_cancelled:
                        break
                    
                    if self.is_timed_out():
                        logger.warning(f"Stream timeout during sync processing for {self.conversation_id}")
                        yield f"data: {json.dumps({'error': {'message': 'Stream timeout'}})}\n\n"
                        break
                    
                    try:
                        text_content = chunk.decode('utf-8', errors='replace') if isinstance(chunk, bytes) else str(chunk)
                        if text_content:
                            self.full_response.append(text_content)
                            self.update_activity()
                            yield f"data: {json.dumps({'choices': [{'delta': {'content': text_content}}]})}\n\n"
                    except Exception as e:
                        logger.error(f"Error processing sync stream chunk for {self.conversation_id}: {e}")
                        self.error_occurred = True
                        yield f"data: {json.dumps({'error': {'message': f'Error processing chunk: {str(e)}'}})}\n\n"
                        break
            
        except asyncio.CancelledError:
            # Client disconnected
            logger.info(f"Client disconnected from stream for {self.conversation_id}")
            self.cancel()
            
        except Exception as e:
            # Unexpected error
            logger.error(f"Unexpected error in stream for {self.conversation_id}: {e}", exc_info=True)
            self.error_occurred = True
            yield f"data: {json.dumps({'error': {'message': f'Stream error: {str(e)}'}})}\n\n"
            
        finally:
            # Cleanup and final message
            try:
                if not self.error_occurred and not self.is_cancelled:
                    # Send completion message
                    done_payload = {
                        "id": f"chatcmpl-{datetime.now(timezone.utc).timestamp()}",
                        "object": "chat.completion.chunk",
                        "created": int(datetime.now(timezone.utc).timestamp()),
                        "model": self.model_name,
                        "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
                        "conversation_id": self.conversation_id
                    }
                    yield f"data: {json.dumps(done_payload)}\n\n"
                    yield "data: [DONE]\n\n"
                
                # Save the full response if callback provided
                if save_callback and self.full_response and not self.error_occurred:
                    full_text = "".join(self.full_response)
                    try:
                        await save_callback(full_text)
                        logger.info(f"Saved streaming response for {self.conversation_id} (length: {len(full_text)})")
                    except Exception as e:
                        logger.error(f"Failed to save streaming response for {self.conversation_id}: {e}")
                
                # Send stream end event
                yield f"event: stream_end\ndata: {json.dumps({'conversation_id': self.conversation_id, 'success': not self.error_occurred, 'timestamp': datetime.now(timezone.utc).isoformat()})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in stream cleanup for {self.conversation_id}: {e}")


async def create_streaming_response_with_timeout(
    stream: Union[Iterator, AsyncIterator],
    conversation_id: str,
    model_name: str,
    save_callback: Optional[callable] = None,
    idle_timeout: int = STREAMING_IDLE_TIMEOUT,
    heartbeat_interval: int = HEARTBEAT_INTERVAL
) -> AsyncIterator[str]:
    """
    Create a streaming response with timeout and error handling.
    
    Args:
        stream: The stream to process
        conversation_id: ID of the conversation
        model_name: Name of the model
        save_callback: Optional callback to save the response
        idle_timeout: Timeout for idle connections
        heartbeat_interval: Interval for heartbeat messages
        
    Yields:
        SSE formatted messages
    """
    handler = StreamingResponseHandler(
        conversation_id=conversation_id,
        model_name=model_name,
        idle_timeout=idle_timeout,
        heartbeat_interval=heartbeat_interval
    )
    
    # Create tasks for streaming and heartbeat
    async def stream_with_heartbeat():
        stream_task = asyncio.create_task(
            handler.safe_stream_generator(stream, save_callback).__aiter__().__anext__()
        )
        heartbeat_task = asyncio.create_task(
            handler.heartbeat_generator().__aiter__().__anext__()
        )
        
        while not handler.is_cancelled and not handler.error_occurred:
            done, pending = await asyncio.wait(
                {stream_task, heartbeat_task},
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for task in done:
                try:
                    result = task.result()
                    if task == stream_task:
                        yield result
                        # Create new stream task
                        stream_task = asyncio.create_task(
                            handler.safe_stream_generator(stream, save_callback).__aiter__().__anext__()
                        )
                    else:
                        # Heartbeat
                        yield result
                        # Create new heartbeat task
                        heartbeat_task = asyncio.create_task(
                            handler.heartbeat_generator().__aiter__().__anext__()
                        )
                except StopAsyncIteration:
                    # Stream ended
                    handler.cancel()
                    break
                except Exception as e:
                    logger.error(f"Error in streaming task: {e}")
                    handler.error_occurred = True
                    break
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()
    
    async for message in stream_with_heartbeat():
        yield message


#
# End of streaming_utils.py
#######################################################################################################################