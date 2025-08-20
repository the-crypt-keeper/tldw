# document_generator.py
# Description: Service for generating different document types from chat conversations
#
"""
Document Generator Service
--------------------------

Provides functionality to generate various document types from chat conversations:
- Timeline documents
- Study guides
- Briefing documents

Uses configurable prompts and LLM APIs to generate content based on conversation context.
"""

import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Generator, Union
import pyperclip
from loguru import logger

# Local imports
from ..config import get_cli_setting
from ..Metrics.metrics_logger import log_counter, log_histogram
from ..LLM_Calls.LLM_API_Calls import chat_with_openai, chat_with_anthropic, chat_with_cohere, \
    chat_with_groq, chat_with_openrouter, chat_with_deepseek, chat_with_mistral, \
    chat_with_huggingface, chat_with_google
from ..LLM_Calls.LLM_API_Calls_Local import chat_with_aphrodite, chat_with_local_llm, \
    chat_with_ollama, chat_with_kobold, chat_with_llama, chat_with_oobabooga, \
    chat_with_tabbyapi, chat_with_vllm, chat_with_custom_openai, chat_with_custom_openai_2, \
    chat_with_mlx_lm
from ..DB.ChaChaNotes_DB import CharactersRAGDB
from .Chat_Deps import ChatAPIError

# Configure logger
logger = logger.bind(module="DocumentGenerator")


class DocumentGenerator:
    """Service for generating documents from chat conversations."""
    
    def __init__(self, db_path: str, client_id: str = "document_generator"):
        """
        Initialize the document generator.
        
        Args:
            db_path: Path to the ChaChaNotes database
            client_id: Client identifier for database operations
        """
        self.db = CharactersRAGDB(db_path, client_id)
        
        # Load prompt configurations
        self.timeline_config = get_cli_setting("prompts.document_generation.timeline", {
            "prompt": "Create a detailed text-based timeline based on our conversation/materials being referenced.",
            "temperature": 0.3,
            "max_tokens": 2000
        })
        
        self.study_guide_config = get_cli_setting("prompts.document_generation.study_guide", {
            "prompt": "Create a detailed and well produced study guide based on the current focus of our conversation/materials in reference.",
            "temperature": 0.5,
            "max_tokens": 3000
        })
        
        self.briefing_config = get_cli_setting("prompts.document_generation.briefing", {
            "prompt": "Create a detailed and well produced executive briefing document regarding this conversation and the subject material.",
            "temperature": 0.4,
            "max_tokens": 2500
        })
        
        # Provider mapping
        self.provider_functions = {
            "openai": chat_with_openai,
            "anthropic": chat_with_anthropic,
            "cohere": chat_with_cohere,
            "groq": chat_with_groq,
            "openrouter": chat_with_openrouter,
            "deepseek": chat_with_deepseek,
            "mistral": chat_with_mistral,
            "mistralai": chat_with_mistral,
            "huggingface": chat_with_huggingface,
            "google": chat_with_google,
            "aphrodite": chat_with_aphrodite,
            "local-llm": chat_with_local_llm,
            "ollama": chat_with_ollama,
            "koboldcpp": chat_with_kobold,
            "llama.cpp": chat_with_llama,
            "llama_cpp": chat_with_llama,
            "oobabooga": chat_with_oobabooga,
            "tabbyapi": chat_with_tabbyapi,
            "vllm": chat_with_vllm,
            "custom": chat_with_custom_openai,
            "custom_openai": chat_with_custom_openai,
            "custom_2": chat_with_custom_openai_2,
            "custom_openai_2": chat_with_custom_openai_2,
            "mlx": chat_with_mlx_lm,
            "mlx_lm": chat_with_mlx_lm,
        }
    
    def get_conversation_context(self, conversation_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get conversation context including recent messages.
        
        Args:
            conversation_id: ID of the conversation
            limit: Maximum number of messages to include
            
        Returns:
            List of message dictionaries
        """
        try:
            messages = self.db.get_messages_by_conversation_id(conversation_id, limit=limit)
            return [msg.to_dict() for msg in messages]
        except Exception as e:
            logger.error(f"Failed to get conversation context: {e}")
            return []
    
    def format_context_for_llm(self, messages: List[Dict[str, Any]], 
                              specific_message: Optional[str] = None) -> str:
        """
        Format conversation context for LLM processing.
        
        Args:
            messages: List of message dictionaries
            specific_message: Optional specific message to highlight
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Add conversation history
        for msg in messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            timestamp = msg.get('timestamp', '')
            
            context_parts.append(f"[{timestamp}] {role.upper()}: {content}")
        
        # Add specific message if provided
        if specific_message:
            context_parts.append("\n--- SPECIFIC MESSAGE TO FOCUS ON ---")
            context_parts.append(specific_message)
            context_parts.append("--- END SPECIFIC MESSAGE ---\n")
        
        return "\n".join(context_parts)
    
    def generate_timeline(self, conversation_id: str, provider: str, model: str, 
                         api_key: str, specific_message: Optional[str] = None,
                         stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """
        Generate a timeline document from conversation.
        
        Args:
            conversation_id: ID of the conversation
            provider: LLM provider name
            model: Model name
            api_key: API key for the provider
            specific_message: Optional specific message to focus on
            stream: Whether to stream the response
            
        Returns:
            Generated timeline content or stream generator
        """
        start_time = time.time()
        logger.info(f"Generating timeline for conversation {conversation_id}")
        log_counter("document_generator_request", labels={
            "document_type": "timeline",
            "provider": provider,
            "model": model,
            "streaming": str(stream)
        })
        
        # Get conversation context
        messages = self.get_conversation_context(conversation_id)
        context = self.format_context_for_llm(messages, specific_message)
        
        # Build prompt
        system_prompt = "You are an expert at creating clear, chronological timelines from conversations and content."
        user_prompt = f"{self.timeline_config['prompt']}\n\nConversation Context:\n{context}"
        
        # Call LLM
        try:
            result = self._call_llm(
                provider, model, api_key,
                system_prompt, user_prompt,
                temperature=self.timeline_config.get('temperature', 0.3),
                max_tokens=self.timeline_config.get('max_tokens', 2000),
                stream=stream
            )
            
            # Log success metrics (only for non-streaming)
            if not stream:
                duration = time.time() - start_time
                log_histogram("document_generator_duration", duration, labels={
                    "document_type": "timeline",
                    "provider": provider,
                    "model": model
                })
                log_counter("document_generator_success", labels={
                    "document_type": "timeline",
                    "provider": provider
                })
                
            return result
        except Exception as e:
            # Log error metrics
            duration = time.time() - start_time
            log_histogram("document_generator_duration", duration, labels={
                "document_type": "timeline",
                "provider": provider,
                "model": model
            })
            log_counter("document_generator_error", labels={
                "document_type": "timeline",
                "provider": provider,
                "error_type": type(e).__name__
            })
            raise
    
    def generate_study_guide(self, conversation_id: str, provider: str, model: str,
                                  api_key: str, specific_message: Optional[str] = None,
                                  stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """
        Generate a study guide from conversation.
        
        Args:
            conversation_id: ID of the conversation
            provider: LLM provider name
            model: Model name
            api_key: API key for the provider
            specific_message: Optional specific message to focus on
            stream: Whether to stream the response
            
        Returns:
            Generated study guide content or stream generator
        """
        start_time = time.time()
        logger.info(f"Generating study guide for conversation {conversation_id}")
        log_counter("document_generator_request", labels={
            "document_type": "study_guide",
            "provider": provider,
            "model": model,
            "streaming": str(stream)
        })
        
        # Get conversation context
        messages = self.get_conversation_context(conversation_id)
        context = self.format_context_for_llm(messages, specific_message)
        
        # Build prompt
        system_prompt = "You are an educational expert specializing in creating comprehensive study guides."
        user_prompt = f"{self.study_guide_config['prompt']}\n\nConversation Context:\n{context}"
        
        # Call LLM
        try:
            result = self._call_llm(
                provider, model, api_key,
                system_prompt, user_prompt,
                temperature=self.study_guide_config.get('temperature', 0.5),
                max_tokens=self.study_guide_config.get('max_tokens', 3000),
                stream=stream
            )
            
            # Log success metrics (only for non-streaming)
            if not stream:
                duration = time.time() - start_time
                log_histogram("document_generator_duration", duration, labels={
                    "document_type": "study_guide",
                    "provider": provider,
                    "model": model
                })
                log_counter("document_generator_success", labels={
                    "document_type": "study_guide",
                    "provider": provider
                })
                
            return result
        except Exception as e:
            # Log error metrics
            duration = time.time() - start_time
            log_histogram("document_generator_duration", duration, labels={
                "document_type": "study_guide",
                "provider": provider,
                "model": model
            })
            log_counter("document_generator_error", labels={
                "document_type": "study_guide",
                "provider": provider,
                "error_type": type(e).__name__
            })
            raise
    
    def generate_briefing(self, conversation_id: str, provider: str, model: str,
                               api_key: str, specific_message: Optional[str] = None,
                               stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """
        Generate an executive briefing from conversation.
        
        Args:
            conversation_id: ID of the conversation
            provider: LLM provider name
            model: Model name
            api_key: API key for the provider
            specific_message: Optional specific message to focus on
            stream: Whether to stream the response
            
        Returns:
            Generated briefing content or stream generator
        """
        start_time = time.time()
        logger.info(f"Generating briefing for conversation {conversation_id}")
        log_counter("document_generator_request", labels={
            "document_type": "briefing",
            "provider": provider,
            "model": model,
            "streaming": str(stream)
        })
        
        # Get conversation context
        messages = self.get_conversation_context(conversation_id)
        context = self.format_context_for_llm(messages, specific_message)
        
        # Build prompt
        system_prompt = "You are an expert at creating executive briefing documents with actionable insights."
        user_prompt = f"{self.briefing_config['prompt']}\n\nConversation Context:\n{context}"
        
        # Call LLM
        try:
            result = self._call_llm(
                provider, model, api_key,
                system_prompt, user_prompt,
                temperature=self.briefing_config.get('temperature', 0.4),
                max_tokens=self.briefing_config.get('max_tokens', 2500),
                stream=stream
            )
            
            # Log success metrics (only for non-streaming)
            if not stream:
                duration = time.time() - start_time
                log_histogram("document_generator_duration", duration, labels={
                    "document_type": "briefing",
                    "provider": provider,
                    "model": model
                })
                log_counter("document_generator_success", labels={
                    "document_type": "briefing",
                    "provider": provider
                })
                
            return result
        except Exception as e:
            # Log error metrics
            duration = time.time() - start_time
            log_histogram("document_generator_duration", duration, labels={
                "document_type": "briefing",
                "provider": provider,
                "model": model
            })
            log_counter("document_generator_error", labels={
                "document_type": "briefing",
                "provider": provider,
                "error_type": type(e).__name__
            })
            raise
    
    def _call_llm(self, provider: str, model: str, api_key: str,
                       system_prompt: str, user_prompt: str,
                       temperature: float = 0.7, max_tokens: int = 2000,
                       stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """
        Call the appropriate LLM provider.
        
        Args:
            provider: Provider name
            model: Model name
            api_key: API key
            system_prompt: System prompt
            user_prompt: User prompt
            temperature: Generation temperature
            max_tokens: Maximum tokens
            stream: Whether to stream
            
        Returns:
            Generated content or stream generator
        """
        provider_lower = provider.lower()
        chat_function = self.provider_functions.get(provider_lower)
        
        if not chat_function:
            raise ChatAPIError(f"Unknown provider: {provider}")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = chat_function(
                messages=messages,
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                system_prompt=system_prompt
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error calling {provider} API: {e}")
            raise ChatAPIError(f"Failed to generate document: {str(e)}")
    
    def copy_to_clipboard(self, content: str) -> bool:
        """
        Copy content to clipboard.
        
        Args:
            content: Content to copy
            
        Returns:
            True if successful, False otherwise
        """
        try:
            pyperclip.copy(content)
            logger.debug("Content copied to clipboard")
            return True
        except Exception as e:
            logger.error(f"Failed to copy to clipboard: {e}")
            return False
    
    def create_note_with_metadata(self, title: str, content: str, 
                                 document_type: str, conversation_id: str) -> str:
        """
        Create a note with metadata about the generation.
        
        Args:
            title: Note title
            content: Note content
            document_type: Type of document generated
            conversation_id: Source conversation ID
            
        Returns:
            Note ID
        """
        metadata = {
            "document_type": document_type,
            "conversation_id": conversation_id,
            "generated_at": datetime.now().isoformat()
        }
        
        # Add metadata to content
        full_content = f"---\nDocument Type: {document_type}\nGenerated: {metadata['generated_at']}\nConversation ID: {conversation_id}\n---\n\n{content}"
        
        try:
            note_id = self.db.add_note(title, full_content, conversation_id)
            logger.info(f"Created note {note_id} for {document_type}")
            return note_id
        except Exception as e:
            logger.error(f"Failed to create note: {e}")
            raise