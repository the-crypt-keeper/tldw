"""
Pydantic models matching OpenAI API specification.

These models define the request and response formats for the mock OpenAI API server.
"""

from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import time


# Chat Completion Models

class ChatMessage(BaseModel):
    """A single message in a chat conversation."""
    role: Literal["system", "user", "assistant", "function", "tool"]
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatCompletionRequest(BaseModel):
    """Request format for chat completions endpoint."""
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    response_format: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    stream_options: Optional[Dict[str, Any]] = None


class ChatCompletionResponseChoice(BaseModel):
    """A single completion choice in the response."""
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "function_call", "tool_calls", "content_filter", "null"]]
    logprobs: Optional[Dict[str, Any]] = None


class ChatCompletionUsage(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Response format for chat completions endpoint."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time() * 1000)}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Optional[ChatCompletionUsage] = None
    system_fingerprint: Optional[str] = None


# Streaming Response Models

class ChatCompletionStreamResponseDelta(BaseModel):
    """Delta content for streaming responses."""
    content: Optional[str] = None
    role: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatCompletionStreamResponseChoice(BaseModel):
    """A single streaming completion choice."""
    index: int
    delta: ChatCompletionStreamResponseDelta
    finish_reason: Optional[Literal["stop", "length", "function_call", "tool_calls", "content_filter", "null"]] = None
    logprobs: Optional[Dict[str, Any]] = None


class ChatCompletionStreamResponse(BaseModel):
    """Response format for streaming chat completions."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time() * 1000)}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionStreamResponseChoice]
    system_fingerprint: Optional[str] = None


# Embeddings Models

class EmbeddingRequest(BaseModel):
    """Request format for embeddings endpoint."""
    input: Union[str, List[str], List[int], List[List[int]]]
    model: str
    encoding_format: Optional[Literal["float", "base64"]] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None


class EmbeddingData(BaseModel):
    """A single embedding in the response."""
    index: int
    embedding: Union[List[float], str]  # float list or base64 string
    object: str = "embedding"


class EmbeddingUsage(BaseModel):
    """Token usage for embeddings."""
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    """Response format for embeddings endpoint."""
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage


# Models List

class ModelInfo(BaseModel):
    """Information about a single model."""
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "openai"
    permission: Optional[List[Dict[str, Any]]] = None
    root: Optional[str] = None
    parent: Optional[str] = None


class ModelsResponse(BaseModel):
    """Response format for models list endpoint."""
    object: str = "list"
    data: List[ModelInfo]


# Completions (Legacy) Models

class CompletionRequest(BaseModel):
    """Request format for legacy completions endpoint."""
    model: str
    prompt: Union[str, List[str], List[int], List[List[int]]]
    suffix: Optional[str] = None
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1)
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    best_of: Optional[int] = Field(default=1, ge=1)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None


class CompletionChoice(BaseModel):
    """A single completion choice."""
    text: str
    index: int
    logprobs: Optional[Dict[str, Any]] = None
    finish_reason: Optional[Literal["stop", "length", "content_filter", "null"]]


class CompletionUsage(BaseModel):
    """Token usage for completions."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    """Response format for legacy completions endpoint."""
    id: str = Field(default_factory=lambda: f"cmpl-{int(time.time() * 1000)}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionChoice]
    usage: Optional[CompletionUsage] = None
    system_fingerprint: Optional[str] = None


# Error Models

class ErrorDetail(BaseModel):
    """Error detail information."""
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard error response format."""
    error: ErrorDetail