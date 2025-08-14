# rag_schemas_simple_compat.py - Compatibility layer for RAG schemas
"""
Compatibility layer to support old test imports while using new simplified schemas.
This file provides the missing classes and enums that tests expect.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

# Import everything from the main schema file
from tldw_Server_API.app.api.v1.schemas.rag_schemas_simple import *

# ============= Compatibility Enums =============

class SearchModeEnum(str, Enum):
    """Search modes for backward compatibility"""
    BASIC = "basic"
    ADVANCED = "advanced"
    CUSTOM = "custom"


class AgentModeEnum(str, Enum):
    """Agent modes for backward compatibility"""
    RAG = "rag"
    RESEARCH = "research"


class MessageRole(str, Enum):
    """Message roles for chat context"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# ============= Compatibility Classes =============

class Message(BaseModel):
    """Chat message for conversations"""
    role: MessageRole
    content: str


class SearchApiRequest(BaseModel):
    """Compatibility wrapper for search requests"""
    query: str = Field(..., description="Search query string")
    mode: Optional[SearchModeEnum] = Field(default=SearchModeEnum.BASIC, description="Search mode")
    top_k: Optional[int] = Field(default=10, ge=1, le=100, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional filters")
    data_sources: Optional[List[str]] = Field(default=None, description="Databases to search")
    
    # Additional fields expected by tests
    search_databases: Optional[List[str]] = Field(default=None, description="Databases to search (alias)")
    offset: Optional[int] = Field(default=0, ge=0, description="Pagination offset")
    date_range_start: Optional[str] = Field(default=None, description="Start date for filtering")
    date_range_end: Optional[str] = Field(default=None, description="End date for filtering")
    use_semantic_search: Optional[bool] = Field(default=False, description="Enable semantic search")
    use_hybrid_search: Optional[bool] = Field(default=True, description="Enable hybrid search")
    
    class Config:
        # Allow both 'query' and 'querystring' field names
        fields = {
            'query': {'alias': 'querystring'}
        }
        populate_by_name = True


class RetrievalAgentRequest(BaseModel):
    """Compatibility wrapper for agent requests"""
    message: Optional[Message] = Field(default=None, description="Single message")
    messages: Optional[List[Message]] = Field(default=None, description="Conversation history")
    mode: Optional[AgentModeEnum] = Field(default=AgentModeEnum.RAG, description="Agent mode")
    rag_generation_config: Optional[GenerationConfig] = Field(default=None, description="Generation config")
    api_config: Optional[Dict[str, Any]] = Field(default=None, description="API configuration")
    
    # Additional fields for compatibility
    search_config: Optional[Dict[str, Any]] = Field(default=None, description="Search configuration")
    
    @field_validator('messages', mode='before')
    def ensure_messages_list(cls, v, values):
        """Ensure we have either message or messages"""
        if v is None and 'message' in values and values['message'] is not None:
            return [values['message']]
        return v


# Re-export existing GenerationConfig for convenience
# (It's already in the main schema file)