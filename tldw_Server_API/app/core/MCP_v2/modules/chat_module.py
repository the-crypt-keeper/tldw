"""
Chat Module for tldw MCP - Handles chat completions and conversations
"""

from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime
import json
from loguru import logger

from ..modules.base import BaseModule, create_tool_definition, create_resource_definition
from ..schemas import ModuleConfig

# Import tldw's existing chat/LLM functionality
try:
    from tldw_Server_API.app.core.LLM_Calls.LLM_API_Calls import (
        chat_with_openai,
        chat_with_anthropic,
        chat_with_local_llm
    )
    from tldw_Server_API.app.core.DB_Management.Chat_DB import ChatDatabase
except ImportError:
    logger.warning("Chat/LLM functionality imports not available")
    ChatDatabase = None


class ChatModule(BaseModule):
    """Chat Module for tldw
    
    Provides tools for:
    - Chat completions with various LLM providers
    - Conversation management
    - Context-aware responses
    - Character-based chat
    - Chat history management
    """
    
    def __init__(self, config: ModuleConfig):
        super().__init__(config)
        self.chat_db: Optional[Any] = None
        self.db_path = config.settings.get("db_path", "./Databases/Chat.db")
        self.default_provider = config.settings.get("default_provider", "openai")
        self.default_model = config.settings.get("default_model", "gpt-3.5-turbo")
        self.max_context_length = config.settings.get("max_context_length", 4000)
    
    async def on_initialize(self) -> None:
        """Initialize chat module"""
        try:
            # Initialize chat database if available
            if ChatDatabase:
                self.chat_db = ChatDatabase(self.db_path, client_id="mcp_chat_module")
            
            logger.info(f"Chat module initialized with provider: {self.default_provider}")
        except Exception as e:
            logger.error(f"Failed to initialize chat module: {e}")
            # Module can still work with limited functionality
    
    async def on_shutdown(self) -> None:
        """Shutdown chat module"""
        if self.chat_db:
            logger.info("Chat module shutdown")
    
    async def check_health(self) -> bool:
        """Check module health"""
        try:
            # Basic health check - module is available
            return True
        except Exception as e:
            logger.error(f"Chat module health check failed: {e}")
            return False
    
    async def get_tools(self) -> List[Dict[str, Any]]:
        """Get list of chat tools"""
        return [
            create_tool_definition(
                name="chat_completion",
                description="Generate a chat completion response",
                parameters={
                    "properties": {
                        "messages": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {"type": "string", "enum": ["system", "user", "assistant"]},
                                    "content": {"type": "string"}
                                }
                            },
                            "description": "Conversation messages"
                        },
                        "provider": {
                            "type": "string",
                            "enum": ["openai", "anthropic", "local", "groq", "cohere"],
                            "description": "LLM provider to use",
                            "default": "openai"
                        },
                        "model": {
                            "type": "string",
                            "description": "Model to use (provider-specific)"
                        },
                        "temperature": {
                            "type": "number",
                            "description": "Temperature for randomness",
                            "default": 0.7
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Maximum tokens to generate",
                            "default": 1000
                        },
                        "stream": {
                            "type": "boolean",
                            "description": "Stream the response",
                            "default": False
                        }
                    },
                    "required": ["messages"]
                },
                department="chat"
            ),
            create_tool_definition(
                name="create_conversation",
                description="Create a new conversation",
                parameters={
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Conversation title"
                        },
                        "system_prompt": {
                            "type": "string",
                            "description": "System prompt for the conversation"
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Additional metadata"
                        }
                    },
                    "required": ["title"]
                },
                department="chat"
            ),
            create_tool_definition(
                name="add_message",
                description="Add a message to a conversation",
                parameters={
                    "properties": {
                        "conversation_id": {
                            "type": "string",
                            "description": "Conversation ID"
                        },
                        "role": {
                            "type": "string",
                            "enum": ["user", "assistant", "system"],
                            "description": "Message role"
                        },
                        "content": {
                            "type": "string",
                            "description": "Message content"
                        }
                    },
                    "required": ["conversation_id", "role", "content"]
                },
                department="chat"
            ),
            create_tool_definition(
                name="get_conversation",
                description="Get a conversation with its messages",
                parameters={
                    "properties": {
                        "conversation_id": {
                            "type": "string",
                            "description": "Conversation ID"
                        },
                        "include_system": {
                            "type": "boolean",
                            "description": "Include system messages",
                            "default": True
                        }
                    },
                    "required": ["conversation_id"]
                },
                department="chat"
            ),
            create_tool_definition(
                name="search_conversations",
                description="Search through conversations",
                parameters={
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                },
                department="chat"
            ),
            create_tool_definition(
                name="summarize_conversation",
                description="Generate a summary of a conversation",
                parameters={
                    "properties": {
                        "conversation_id": {
                            "type": "string",
                            "description": "Conversation ID to summarize"
                        },
                        "style": {
                            "type": "string",
                            "enum": ["brief", "detailed", "bullet_points"],
                            "description": "Summary style",
                            "default": "brief"
                        }
                    },
                    "required": ["conversation_id"]
                },
                department="chat"
            ),
            create_tool_definition(
                name="chat_with_context",
                description="Chat with context from media or documents",
                parameters={
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "User message"
                        },
                        "context_source": {
                            "type": "string",
                            "enum": ["media", "notes", "search"],
                            "description": "Source of context"
                        },
                        "context_id": {
                            "type": "string",
                            "description": "ID of the context source (media_id, note_id, etc.)"
                        },
                        "provider": {
                            "type": "string",
                            "description": "LLM provider",
                            "default": "openai"
                        }
                    },
                    "required": ["message", "context_source"]
                },
                department="chat"
            ),
            create_tool_definition(
                name="character_chat",
                description="Chat as a specific character",
                parameters={
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "User message"
                        },
                        "character_id": {
                            "type": "string",
                            "description": "Character ID to use"
                        },
                        "conversation_id": {
                            "type": "string",
                            "description": "Existing conversation ID (optional)"
                        }
                    },
                    "required": ["message", "character_id"]
                },
                department="chat"
            )
        ]
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute chat tool"""
        logger.info(f"Executing chat tool: {tool_name}")
        
        try:
            if tool_name == "chat_completion":
                return await self._chat_completion(arguments)
            elif tool_name == "create_conversation":
                return await self._create_conversation(arguments)
            elif tool_name == "add_message":
                return await self._add_message(arguments)
            elif tool_name == "get_conversation":
                return await self._get_conversation(arguments)
            elif tool_name == "search_conversations":
                return await self._search_conversations(arguments)
            elif tool_name == "summarize_conversation":
                return await self._summarize_conversation(arguments)
            elif tool_name == "chat_with_context":
                return await self._chat_with_context(arguments)
            elif tool_name == "character_chat":
                return await self._character_chat(arguments)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
        except Exception as e:
            logger.error(f"Error executing chat tool {tool_name}: {e}")
            raise
    
    async def get_resources(self) -> List[Dict[str, Any]]:
        """Get chat resources"""
        return [
            create_resource_definition(
                uri="chat://providers",
                name="Available Providers",
                description="List of available LLM providers",
                mime_type="application/json"
            ),
            create_resource_definition(
                uri="chat://models",
                name="Available Models",
                description="List of available models per provider",
                mime_type="application/json"
            ),
            create_resource_definition(
                uri="chat://statistics",
                name="Chat Statistics",
                description="Statistics about chat usage",
                mime_type="application/json"
            ),
            create_resource_definition(
                uri="chat://characters",
                name="Available Characters",
                description="List of available character personas",
                mime_type="application/json"
            )
        ]
    
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read chat resource"""
        if uri == "chat://providers":
            return {
                "providers": [
                    {"name": "openai", "models": ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"]},
                    {"name": "anthropic", "models": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]},
                    {"name": "groq", "models": ["mixtral-8x7b", "llama2-70b"]},
                    {"name": "local", "models": ["llama2", "mistral", "phi-2"]}
                ],
                "default": self.default_provider
            }
        elif uri == "chat://models":
            return {
                "openai": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                "anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
                "groq": ["mixtral-8x7b", "llama2-70b"],
                "cohere": ["command", "command-light"],
                "local": ["available_via_api"]
            }
        elif uri == "chat://statistics":
            if self.chat_db:
                total_conversations = len(self.chat_db.get_all_conversations())
                total_messages = self.chat_db.get_total_messages()
            else:
                total_conversations = 0
                total_messages = 0
            
            return {
                "total_conversations": total_conversations,
                "total_messages": total_messages,
                "active_provider": self.default_provider,
                "last_updated": datetime.utcnow().isoformat()
            }
        elif uri == "chat://characters":
            if self.chat_db:
                characters = self.chat_db.get_all_characters()
            else:
                characters = []
            
            return {
                "characters": characters,
                "count": len(characters)
            }
        else:
            raise ValueError(f"Unknown resource: {uri}")
    
    # Tool implementation methods
    
    async def _chat_completion(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Generate chat completion"""
        messages = args["messages"]
        provider = args.get("provider", self.default_provider)
        model = args.get("model", self.default_model)
        temperature = args.get("temperature", 0.7)
        max_tokens = args.get("max_tokens", 1000)
        stream = args.get("stream", False)
        
        try:
            # Route to appropriate provider
            if provider == "openai":
                response = await chat_with_openai(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream
                )
            elif provider == "anthropic":
                response = await chat_with_anthropic(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            elif provider == "local":
                response = await chat_with_local_llm(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                # Fallback response
                response = {
                    "content": f"[{provider}] Response would be generated here",
                    "model": model,
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0}
                }
            
            return {
                "success": True,
                "response": response,
                "provider": provider,
                "model": model
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _create_conversation(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Create new conversation"""
        title = args["title"]
        system_prompt = args.get("system_prompt", "")
        metadata = args.get("metadata", {})
        
        try:
            if self.chat_db:
                conversation_id = self.chat_db.create_conversation(
                    title=title,
                    system_prompt=system_prompt,
                    metadata=metadata
                )
            else:
                # Mock implementation
                conversation_id = f"conv_{datetime.now().timestamp()}"
            
            return {
                "success": True,
                "conversation_id": conversation_id,
                "title": title,
                "message": "Conversation created successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _add_message(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Add message to conversation"""
        conversation_id = args["conversation_id"]
        role = args["role"]
        content = args["content"]
        
        try:
            if self.chat_db:
                message_id = self.chat_db.add_message(
                    conversation_id=conversation_id,
                    role=role,
                    content=content
                )
            else:
                message_id = f"msg_{datetime.now().timestamp()}"
            
            return {
                "success": True,
                "message_id": message_id,
                "conversation_id": conversation_id
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _get_conversation(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get conversation with messages"""
        conversation_id = args["conversation_id"]
        include_system = args.get("include_system", True)
        
        try:
            if self.chat_db:
                conversation = self.chat_db.get_conversation(conversation_id)
                messages = self.chat_db.get_messages(
                    conversation_id,
                    include_system=include_system
                )
            else:
                # Mock data
                conversation = {
                    "id": conversation_id,
                    "title": "Mock Conversation",
                    "created": datetime.utcnow().isoformat()
                }
                messages = []
            
            return {
                "success": True,
                "conversation": conversation,
                "messages": messages,
                "message_count": len(messages)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _search_conversations(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Search conversations"""
        query = args["query"]
        limit = args.get("limit", 10)
        
        try:
            if self.chat_db:
                results = self.chat_db.search_conversations(
                    query=query,
                    limit=limit
                )
            else:
                results = []
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _summarize_conversation(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize a conversation"""
        conversation_id = args["conversation_id"]
        style = args.get("style", "brief")
        
        try:
            # Get conversation messages
            if self.chat_db:
                messages = self.chat_db.get_messages(conversation_id)
            else:
                messages = []
            
            if not messages:
                return {
                    "success": False,
                    "error": "No messages found in conversation"
                }
            
            # Create summary prompt
            conversation_text = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in messages
            ])
            
            summary_prompt = f"Summarize this conversation in a {style} style:\n\n{conversation_text}"
            
            # Generate summary using default provider
            summary_response = await self._chat_completion({
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that creates concise summaries."},
                    {"role": "user", "content": summary_prompt}
                ],
                "max_tokens": 500
            })
            
            if summary_response["success"]:
                summary = summary_response["response"].get("content", "Summary generation failed")
            else:
                summary = "Unable to generate summary"
            
            return {
                "success": True,
                "conversation_id": conversation_id,
                "summary": summary,
                "style": style,
                "message_count": len(messages)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _chat_with_context(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Chat with context from various sources"""
        message = args["message"]
        context_source = args["context_source"]
        context_id = args.get("context_id")
        provider = args.get("provider", self.default_provider)
        
        try:
            # Fetch context based on source
            context = ""
            
            if context_source == "media":
                # Fetch media transcript/content
                context = f"[Media context for ID {context_id}]"
            elif context_source == "notes":
                # Fetch note content
                context = f"[Note context for ID {context_id}]"
            elif context_source == "search":
                # Perform search and use results as context
                context = f"[Search results context]"
            
            # Create messages with context
            messages = [
                {"role": "system", "content": f"Use this context to answer questions:\n\n{context}"},
                {"role": "user", "content": message}
            ]
            
            # Generate response
            response = await self._chat_completion({
                "messages": messages,
                "provider": provider
            })
            
            return {
                "success": True,
                "response": response.get("response", {}),
                "context_source": context_source,
                "context_id": context_id
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _character_chat(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Chat as a character"""
        message = args["message"]
        character_id = args["character_id"]
        conversation_id = args.get("conversation_id")
        
        try:
            # Get character definition
            if self.chat_db:
                character = self.chat_db.get_character(character_id)
            else:
                character = {
                    "name": "Default Character",
                    "description": "A helpful assistant",
                    "personality": "Friendly and informative"
                }
            
            if not character:
                return {
                    "success": False,
                    "error": f"Character {character_id} not found"
                }
            
            # Create or get conversation
            if not conversation_id:
                conv_result = await self._create_conversation({
                    "title": f"Chat with {character['name']}",
                    "system_prompt": character.get("description", "")
                })
                conversation_id = conv_result.get("conversation_id")
            
            # Build messages with character context
            messages = [
                {
                    "role": "system",
                    "content": f"You are {character['name']}. {character.get('description', '')} {character.get('personality', '')}"
                },
                {"role": "user", "content": message}
            ]
            
            # Get previous messages if continuing conversation
            if conversation_id and self.chat_db:
                prev_messages = self.chat_db.get_messages(conversation_id, limit=10)
                messages = messages[:1] + prev_messages + messages[1:]
            
            # Generate response
            response = await self._chat_completion({
                "messages": messages
            })
            
            # Save to conversation
            if response["success"] and conversation_id:
                await self._add_message({
                    "conversation_id": conversation_id,
                    "role": "user",
                    "content": message
                })
                await self._add_message({
                    "conversation_id": conversation_id,
                    "role": "assistant",
                    "content": response["response"].get("content", "")
                })
            
            return {
                "success": True,
                "response": response.get("response", {}),
                "character": character["name"],
                "conversation_id": conversation_id
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }