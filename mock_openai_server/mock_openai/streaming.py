"""
Streaming support for Server-Sent Events (SSE) in the mock OpenAI server.

Handles streaming responses for chat completions.
"""

import asyncio
import json
import time
import uuid
from typing import AsyncIterator, Dict, Any, List, Optional

from .models import (
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
    ChatCompletionStreamResponseDelta
)


class StreamingResponseGenerator:
    """Generates streaming responses for chat completions."""
    
    def __init__(self, chunk_delay_ms: int = 50, words_per_chunk: int = 5):
        """Initialize the streaming response generator."""
        self.chunk_delay_ms = chunk_delay_ms
        self.words_per_chunk = words_per_chunk
    
    async def generate_stream(
        self,
        content: str,
        model: str = "gpt-4",
        role: str = "assistant",
        request_id: Optional[str] = None
    ) -> AsyncIterator[str]:
        """Generate a streaming response as Server-Sent Events."""
        request_id = request_id or f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        
        # Split content into words for chunking
        words = content.split()
        chunks = []
        
        # Group words into chunks
        for i in range(0, len(words), self.words_per_chunk):
            chunk_words = words[i:i + self.words_per_chunk]
            chunk_text = " ".join(chunk_words)
            if i > 0:
                chunk_text = " " + chunk_text  # Add leading space for continuation
            chunks.append(chunk_text)
        
        # Send initial chunk with role
        initial_response = ChatCompletionStreamResponse(
            id=request_id,
            created=created,
            model=model,
            choices=[
                ChatCompletionStreamResponseChoice(
                    index=0,
                    delta=ChatCompletionStreamResponseDelta(role=role),
                    finish_reason=None
                )
            ]
        )
        yield f"data: {json.dumps(initial_response.model_dump(exclude_none=True))}\n\n"
        
        # Send content chunks
        for i, chunk in enumerate(chunks):
            response = ChatCompletionStreamResponse(
                id=request_id,
                created=created,
                model=model,
                choices=[
                    ChatCompletionStreamResponseChoice(
                        index=0,
                        delta=ChatCompletionStreamResponseDelta(content=chunk),
                        finish_reason=None
                    )
                ]
            )
            
            yield f"data: {json.dumps(response.model_dump(exclude_none=True))}\n\n"
            
            # Add delay between chunks
            if self.chunk_delay_ms > 0 and i < len(chunks) - 1:
                await asyncio.sleep(self.chunk_delay_ms / 1000.0)
        
        # Send final chunk with finish_reason
        final_response = ChatCompletionStreamResponse(
            id=request_id,
            created=created,
            model=model,
            choices=[
                ChatCompletionStreamResponseChoice(
                    index=0,
                    delta=ChatCompletionStreamResponseDelta(),
                    finish_reason="stop"
                )
            ]
        )
        yield f"data: {json.dumps(final_response.model_dump(exclude_none=True))}\n\n"
        
        # Send [DONE] marker
        yield "data: [DONE]\n\n"
    
    async def generate_stream_from_response(
        self,
        response_data: Dict[str, Any],
        request_id: Optional[str] = None
    ) -> AsyncIterator[str]:
        """Generate a streaming response from a complete response object."""
        request_id = request_id or response_data.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}")
        created = response_data.get("created", int(time.time()))
        model = response_data.get("model", "gpt-4")
        
        # Process each choice in the response
        for choice in response_data.get("choices", []):
            message = choice.get("message", {})
            content = message.get("content", "")
            role = message.get("role", "assistant")
            
            # Send initial chunk with role
            initial_response = ChatCompletionStreamResponse(
                id=request_id,
                created=created,
                model=model,
                choices=[
                    ChatCompletionStreamResponseChoice(
                        index=choice.get("index", 0),
                        delta=ChatCompletionStreamResponseDelta(role=role),
                        finish_reason=None
                    )
                ]
            )
            yield f"data: {json.dumps(initial_response.model_dump(exclude_none=True))}\n\n"
            
            # Handle function calls if present
            if message.get("function_call"):
                function_response = ChatCompletionStreamResponse(
                    id=request_id,
                    created=created,
                    model=model,
                    choices=[
                        ChatCompletionStreamResponseChoice(
                            index=choice.get("index", 0),
                            delta=ChatCompletionStreamResponseDelta(
                                function_call=message["function_call"]
                            ),
                            finish_reason=None
                        )
                    ]
                )
                yield f"data: {json.dumps(function_response.model_dump(exclude_none=True))}\n\n"
            
            # Handle tool calls if present
            if message.get("tool_calls"):
                for tool_call in message["tool_calls"]:
                    tool_response = ChatCompletionStreamResponse(
                        id=request_id,
                        created=created,
                        model=model,
                        choices=[
                            ChatCompletionStreamResponseChoice(
                                index=choice.get("index", 0),
                                delta=ChatCompletionStreamResponseDelta(
                                    tool_calls=[tool_call]
                                ),
                                finish_reason=None
                            )
                        ]
                    )
                    yield f"data: {json.dumps(tool_response.model_dump(exclude_none=True))}\n\n"
            
            # Stream content if present
            if content:
                # Split content into chunks
                words = content.split()
                chunks = []
                
                for i in range(0, len(words), self.words_per_chunk):
                    chunk_words = words[i:i + self.words_per_chunk]
                    chunk_text = " ".join(chunk_words)
                    if i > 0:
                        chunk_text = " " + chunk_text
                    chunks.append(chunk_text)
                
                # Send content chunks
                for i, chunk in enumerate(chunks):
                    response = ChatCompletionStreamResponse(
                        id=request_id,
                        created=created,
                        model=model,
                        choices=[
                            ChatCompletionStreamResponseChoice(
                                index=choice.get("index", 0),
                                delta=ChatCompletionStreamResponseDelta(content=chunk),
                                finish_reason=None
                            )
                        ]
                    )
                    
                    yield f"data: {json.dumps(response.model_dump(exclude_none=True))}\n\n"
                    
                    # Add delay between chunks
                    if self.chunk_delay_ms > 0 and i < len(chunks) - 1:
                        await asyncio.sleep(self.chunk_delay_ms / 1000.0)
            
            # Send final chunk with finish_reason
            final_response = ChatCompletionStreamResponse(
                id=request_id,
                created=created,
                model=model,
                choices=[
                    ChatCompletionStreamResponseChoice(
                        index=choice.get("index", 0),
                        delta=ChatCompletionStreamResponseDelta(),
                        finish_reason=choice.get("finish_reason", "stop")
                    )
                ]
            )
            yield f"data: {json.dumps(final_response.model_dump(exclude_none=True))}\n\n"
        
        # Send [DONE] marker
        yield "data: [DONE]\n\n"
    
    async def generate_error_stream(
        self,
        error_message: str,
        error_type: str = "invalid_request_error"
    ) -> AsyncIterator[str]:
        """Generate an error response as a stream."""
        error_data = {
            "error": {
                "message": error_message,
                "type": error_type,
                "param": None,
                "code": None
            }
        }
        yield f"data: {json.dumps(error_data)}\n\n"
        yield "data: [DONE]\n\n"