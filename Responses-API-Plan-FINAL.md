# OpenAI Responses API Implementation Plan - FINAL

## Executive Summary

This document provides the definitive implementation plan for OpenAI's Responses API based on the official documentation. The Responses API is fundamentally different from what was previously understood - it's a new API primitive that uses `client.responses.create()` with an `input` array and `output` array structure, supporting both JSON schema-based functions and custom tools with optional context-free grammars.

## Key API Characteristics (From Official Docs)

### Core Differences from Chat Completions API:
1. **Different client method**: `client.responses.create()` instead of `client.chat.completions.create()`
2. **Input structure**: Takes `input` array instead of `messages` array
3. **Output structure**: Returns `output` array containing multiple item types
4. **Function calling flow**: Uses `type: "function_call_output"` with `call_id` for results
5. **Tool types**: Supports both `function` tools (JSON schema) and `custom` tools (free text/grammar)

### Input/Output Flow:
```python
# Step 1: Initial request with tools
input = [{"role": "user", "content": "What's my horoscope? I'm an Aquarius"}]
response = client.responses.create(model="gpt-5", tools=tools, input=input)

# Step 2: Process function calls from output
for item in response.output:
    if item.type == "function_call":
        # Execute function with arguments
        result = execute_function(item.name, json.loads(item.arguments))
        
# Step 3: Send function results back
input.append({
    "type": "function_call_output",
    "call_id": item.call_id,
    "output": json.dumps(result)
})

# Step 4: Get final response
final_response = client.responses.create(model="gpt-5", tools=tools, input=input)
```

## Database Schema Design

### Decision: Extend ChaChaNotes_DB with Response Tables

```sql
-- Add response-specific fields to conversations
ALTER TABLE conversations ADD COLUMN api_type TEXT DEFAULT 'chat'; -- 'chat' or 'response'
ALTER TABLE conversations ADD COLUMN response_id TEXT UNIQUE; -- resp_xxx format

-- New table for response sessions
CREATE TABLE IF NOT EXISTS response_sessions (
    id TEXT PRIMARY KEY,              -- resp_<uuid>
    conversation_id TEXT REFERENCES conversations(id),
    model TEXT NOT NULL,
    input_array TEXT NOT NULL,        -- JSON array of input items
    output_array TEXT,                -- JSON array of output items
    tools_config TEXT,                -- JSON array of tool definitions
    tool_choice TEXT,                 -- auto/required/none/specific
    parallel_tool_calls BOOLEAN DEFAULT TRUE,
    instructions TEXT,                -- System instructions
    store BOOLEAN DEFAULT FALSE,      -- Store for retrieval
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    usage TEXT,                       -- Token usage stats
    metadata TEXT
);

-- Track function calls
CREATE TABLE IF NOT EXISTS response_function_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    response_id TEXT NOT NULL,
    call_id TEXT UNIQUE NOT NULL,     -- call_xxx format from OpenAI
    item_id TEXT NOT NULL,            -- fc_xxx format
    function_name TEXT NOT NULL,
    arguments TEXT NOT NULL,          -- JSON string
    output TEXT,                      -- Function result (always string)
    output_index INTEGER,             -- Position in output array
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (response_id) REFERENCES response_sessions(id)
);

-- Track custom tool calls  
CREATE TABLE IF NOT EXISTS response_custom_tools (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    response_id TEXT NOT NULL,
    call_id TEXT UNIQUE NOT NULL,
    tool_name TEXT NOT NULL,
    input TEXT NOT NULL,              -- Free-form text input
    grammar_type TEXT,                -- 'lark' or 'regex'
    grammar_definition TEXT,          -- Grammar specification
    status TEXT NOT NULL,             -- 'completed', 'failed'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (response_id) REFERENCES response_sessions(id)
);

-- Indexes for performance
CREATE INDEX idx_response_sessions_created ON response_sessions(created_at DESC);
CREATE INDEX idx_function_calls_response ON response_function_calls(response_id);
CREATE INDEX idx_custom_tools_response ON response_custom_tools(response_id);
```

## Pydantic Schemas (Exact OpenAI Format)

```python
# openai_responses_schemas.py
from typing import Dict, List, Optional, Any, Literal, Union
from pydantic import BaseModel, Field, field_validator
from enum import Enum

# Tool Definitions
class FunctionTool(BaseModel):
    type: Literal["function"] = "function"
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    strict: bool = False

class CustomTool(BaseModel):
    type: Literal["custom"] = "custom"
    name: str
    description: str
    format: Optional[Dict[str, Any]] = None  # Grammar spec

class GrammarFormat(BaseModel):
    type: Literal["grammar"] = "grammar"
    syntax: Literal["lark", "regex"]
    definition: str

# Input Types
class UserInput(BaseModel):
    role: Literal["user"] = "user"
    content: str

class AssistantInput(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str

class FunctionCallOutput(BaseModel):
    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str  # Always string, even for JSON

# Output Types
class FunctionCallItem(BaseModel):
    id: str  # fc_xxx format
    call_id: str  # call_xxx format
    type: Literal["function_call"] = "function_call"
    name: str
    arguments: str  # JSON-encoded string

class CustomToolCallItem(BaseModel):
    id: str  # ctc_xxx format
    type: Literal["custom_tool_call"] = "custom_tool_call"
    status: Literal["completed", "failed"]
    call_id: str
    input: str  # Free-form text
    name: str

class TextItem(BaseModel):
    id: str
    type: Literal["text"] = "text"
    content: str

class ReasoningItem(BaseModel):
    """For reasoning models like GPT-5/o4"""
    id: str  # rs_xxx format
    type: Literal["reasoning"] = "reasoning"
    content: List[Any]
    summary: List[Any]

# Request/Response
class CreateResponseRequest(BaseModel):
    model: str
    input: List[Union[UserInput, AssistantInput, FunctionCallOutput, Dict]]
    tools: Optional[List[Union[FunctionTool, CustomTool]]] = None
    tool_choice: Optional[Union[str, Dict]] = "auto"
    parallel_tool_calls: bool = True
    instructions: Optional[str] = None
    stream: bool = False
    store: bool = False
    metadata: Optional[Dict[str, Any]] = None

class ResponseObject(BaseModel):
    id: str  # resp_xxx format
    object: Literal["response"] = "response"
    model: str
    created_at: int  # Unix timestamp
    output: List[Union[FunctionCallItem, CustomToolCallItem, TextItem, ReasoningItem]]
    output_text: Optional[str] = None  # Aggregated text
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None

# Streaming Events
class StreamEventType(str, Enum):
    OUTPUT_ITEM_ADDED = "response.output_item.added"
    FUNCTION_CALL_DELTA = "response.function_call_arguments.delta"
    FUNCTION_CALL_DONE = "response.function_call_arguments.done"
    OUTPUT_ITEM_DONE = "response.output_item.done"
```

## Core Implementation

### ResponseRunner Class

```python
# response_runner.py
import asyncio
import json
import uuid
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from loguru import logger

class ResponseRunner:
    """Executes Responses API requests following OpenAI specification"""
    
    def __init__(self, db_path: str):
        self.db = CharactersRAGDB(db_path)
        self.function_executor = FunctionExecutor()
        self.custom_tool_executor = CustomToolExecutor()
    
    async def process_response(
        self,
        request: CreateResponseRequest
    ) -> ResponseObject:
        """Main entry point for processing a response"""
        
        response_id = f"resp_{uuid.uuid4().hex[:12]}"
        start_time = time.time()
        
        # Save initial request
        await self._save_request(response_id, request)
        
        # Build messages from input array
        messages = self._build_messages(request.input)
        
        # Prepare tools for LLM call
        functions = self._extract_functions(request.tools)
        
        # Call LLM
        llm_response = await self._call_llm(
            model=request.model,
            messages=messages,
            functions=functions,
            instructions=request.instructions
        )
        
        # Process LLM response into output array
        output = []
        pending_function_calls = []
        
        # Handle different response types
        if llm_response.get("function_call"):
            # Single function call (non-parallel)
            fc_item = self._create_function_call_item(llm_response["function_call"])
            output.append(fc_item)
            pending_function_calls.append(fc_item)
            
        elif llm_response.get("tool_calls"):
            # Multiple parallel function calls
            for tool_call in llm_response["tool_calls"]:
                fc_item = self._create_function_call_item(tool_call)
                output.append(fc_item)
                pending_function_calls.append(fc_item)
        
        # Add text content if present
        if llm_response.get("content"):
            output.append({
                "id": f"txt_{uuid.uuid4().hex[:12]}",
                "type": "text",
                "content": llm_response["content"]
            })
        
        # Execute pending function calls if any
        if pending_function_calls and request.tool_choice != "none":
            function_results = await self._execute_functions(pending_function_calls)
            
            # If we have function results, we need another LLM call
            if function_results:
                # Add function results to input
                new_input = request.input.copy()
                for call_id, result in function_results.items():
                    new_input.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": str(result)
                    })
                
                # Make final call
                final_messages = self._build_messages(new_input)
                final_response = await self._call_llm(
                    model=request.model,
                    messages=final_messages,
                    instructions=request.instructions
                )
                
                # Add final text response
                if final_response.get("content"):
                    output.append({
                        "id": f"txt_{uuid.uuid4().hex[:12]}",
                        "type": "text",
                        "content": final_response["content"]
                    })
        
        # Calculate aggregated text
        output_text = " ".join([
            item["content"] for item in output 
            if item.get("type") == "text"
        ])
        
        # Save and return response
        response_obj = ResponseObject(
            id=response_id,
            object="response",
            model=request.model,
            created_at=int(start_time),
            output=output,
            output_text=output_text,
            usage=llm_response.get("usage")
        )
        
        await self._save_response(response_id, response_obj)
        
        return response_obj
    
    def _create_function_call_item(self, function_call: Dict) -> Dict:
        """Create a function call item for the output array"""
        return {
            "id": f"fc_{uuid.uuid4().hex[:12]}",
            "call_id": f"call_{uuid.uuid4().hex[:12]}",
            "type": "function_call",
            "name": function_call["name"],
            "arguments": function_call.get("arguments", "{}")
        }
    
    async def _execute_functions(self, function_calls: List[Dict]) -> Dict[str, Any]:
        """Execute function calls and return results keyed by call_id"""
        results = {}
        
        for fc in function_calls:
            try:
                func_name = fc["name"]
                func_args = json.loads(fc["arguments"])
                
                # Execute function based on name
                if func_name == "get_weather":
                    result = await self._get_weather(**func_args)
                elif func_name == "file_search":
                    result = await self._file_search(**func_args)
                elif func_name == "web_search":
                    result = await self._web_search(**func_args)
                else:
                    # Custom function handling
                    result = await self.function_executor.execute(func_name, func_args)
                
                results[fc["call_id"]] = result
                
            except Exception as e:
                logger.error(f"Function {func_name} failed: {e}")
                results[fc["call_id"]] = {"error": str(e)}
        
        return results
    
    async def _file_search(self, query: str, **kwargs) -> Dict:
        """Integrate with existing RAG service"""
        from tldw_Server_API.app.core.RAG.RAG_Search.simplified.enhanced_rag_service_v2 import EnhancedRAGServiceV2
        
        rag_service = EnhancedRAGServiceV2()
        results = await rag_service.search(
            query=query,
            top_k=kwargs.get("top_k", 5),
            return_citations=True
        )
        
        return {"results": results}
```

### API Endpoints

```python
# responses_openai.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
import json
import asyncio
from typing import AsyncGenerator

router = APIRouter()

@router.post("/v1/responses", response_model=ResponseObject)
async def create_response(
    request: CreateResponseRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    api_key: str = Depends(verify_api_key)
):
    """
    Create a response using OpenAI Responses API format.
    
    This endpoint:
    1. Accepts an input array (not messages)
    2. Returns an output array with function calls and text
    3. Handles function/custom tool execution
    4. Supports streaming
    """
    
    runner = ResponseRunner(db.db_path)
    
    if request.stream:
        # Return streaming response
        return StreamingResponse(
            stream_response(runner, request),
            media_type="text/event-stream"
        )
    else:
        # Return complete response
        response = await runner.process_response(request)
        return response

async def stream_response(
    runner: ResponseRunner,
    request: CreateResponseRequest
) -> AsyncGenerator[str, None]:
    """Stream response events"""
    
    response_id = f"resp_{uuid.uuid4().hex[:12]}"
    
    # Process with streaming
    async for event in runner.process_response_stream(request, response_id):
        if event["type"] == "response.output_item.added":
            # New function call started
            yield f"data: {json.dumps(event)}\n\n"
            
        elif event["type"] == "response.function_call_arguments.delta":
            # Function arguments streaming
            yield f"data: {json.dumps(event)}\n\n"
            
        elif event["type"] == "response.function_call_arguments.done":
            # Function call complete
            yield f"data: {json.dumps(event)}\n\n"
    
    yield "data: [DONE]\n\n"
```

### Custom Tool Support

```python
# custom_tool_executor.py
import re
from lark import Lark
from typing import Dict, Any, Optional

class CustomToolExecutor:
    """Handles custom tool execution with grammar validation"""
    
    def __init__(self):
        self.lark_parsers = {}
        self.regex_patterns = {}
    
    async def execute_custom_tool(
        self,
        tool_name: str,
        tool_config: CustomTool,
        model_output: str
    ) -> Dict[str, Any]:
        """Execute a custom tool with optional grammar validation"""
        
        # Validate against grammar if specified
        if tool_config.format:
            syntax = tool_config.format.get("syntax")
            definition = tool_config.format.get("definition")
            
            if syntax == "lark":
                if not self._validate_lark(definition, model_output):
                    raise ValueError(f"Output does not match Lark grammar")
                    
            elif syntax == "regex":
                if not self._validate_regex(definition, model_output):
                    raise ValueError(f"Output does not match regex pattern")
        
        # Execute tool-specific logic
        if tool_name == "code_exec":
            return await self._execute_code(model_output)
        elif tool_name == "timestamp":
            return {"timestamp": model_output}
        else:
            # Generic custom tool handling
            return {"result": model_output}
    
    def _validate_lark(self, grammar: str, text: str) -> bool:
        """Validate text against Lark grammar"""
        try:
            if grammar not in self.lark_parsers:
                self.lark_parsers[grammar] = Lark(grammar)
            
            parser = self.lark_parsers[grammar]
            parser.parse(text)
            return True
        except Exception:
            return False
    
    def _validate_regex(self, pattern: str, text: str) -> bool:
        """Validate text against regex pattern"""
        try:
            if pattern not in self.regex_patterns:
                # Use Rust regex syntax compatibility
                self.regex_patterns[pattern] = re.compile(pattern)
            
            regex = self.regex_patterns[pattern]
            return bool(regex.match(text))
        except Exception:
            return False
```

## Testing Strategy

### Unit Tests
```python
# test_responses_api.py
import pytest
from unittest.mock import Mock, AsyncMock, patch

@pytest.mark.asyncio
async def test_function_call_flow():
    """Test complete function calling flow"""
    # Test input with user message
    # Verify function call in output
    # Test function result handling
    # Verify final text response

@pytest.mark.asyncio
async def test_custom_tool_with_grammar():
    """Test custom tool with grammar validation"""
    # Test Lark grammar validation
    # Test regex validation
    # Verify grammar enforcement

@pytest.mark.asyncio
async def test_parallel_function_calls():
    """Test multiple function calls in parallel"""
    # Test parallel_tool_calls=true
    # Verify multiple function calls
    # Test result aggregation

@pytest.mark.asyncio
async def test_streaming_events():
    """Test streaming response events"""
    # Test event types
    # Test delta aggregation
    # Test completion events
```

## Migration Plan

### Phase 1: Database Setup
1. Create migration script for new tables
2. Test migration on development database
3. Add rollback capability

### Phase 2: Core Implementation
1. Implement ResponseRunner
2. Add function execution logic
3. Integrate with existing RAG service

### Phase 3: API Endpoints
1. Add /v1/responses endpoint
2. Implement streaming support
3. Add function call handling

### Phase 4: Custom Tools
1. Implement grammar validation
2. Add Lark parser support
3. Add regex validation

### Phase 5: Testing & Deployment
1. Complete test suite
2. Performance testing
3. Documentation
4. Gradual rollout

## Key Differences from Previous Understanding

1. **NOT a background task API** - It's a synchronous API with a different structure
2. **Input/Output arrays** - Not messages/response format
3. **Function call flow** - Requires explicit function_call_output items
4. **Custom tools** - Support free-text with optional grammars
5. **Streaming format** - Specific event types for function calls

## Critical Implementation Notes

1. **Must maintain call_id** - Essential for matching function results
2. **Arguments are JSON strings** - Not objects
3. **Output is always string** - Even for JSON results
4. **Reasoning items** - Must be preserved for reasoning models
5. **Strict mode** - Requires additionalProperties: false

## Success Criteria

- [ ] Exact OpenAI request/response format
- [ ] Function calling with call_id matching
- [ ] Custom tool support with grammars
- [ ] Streaming with correct event types
- [ ] Integration with existing RAG for file_search
- [ ] Comprehensive test coverage
- [ ] Performance within targets

## Conclusion

The OpenAI Responses API is a sophisticated new API primitive that requires careful implementation to match its unique input/output structure and function calling flow. By extending our existing infrastructure and following the specification exactly, we can provide a fully compatible implementation that leverages our existing RAG and LLM capabilities.