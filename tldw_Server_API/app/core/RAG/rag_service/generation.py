# generation.py
"""
Response generation strategies for the RAG service.

This module provides LLM integration for generating responses using retrieved context,
with support for multiple providers, streaming, and fallback strategies.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, AsyncIterator, List, Union, Protocol
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger

# Import LLM infrastructure
from ...LLM_Calls.LLM_API_Calls import (
    chat_with_openai,
    chat_with_anthropic,
    chat_with_groq,
    chat_with_openrouter,
    chat_with_deepseek,
    chat_with_huggingface,
    chat_with_cohere
)

from .types import Document


class GenerationStrategy(Protocol):
    """Protocol for response generation strategies."""
    
    async def generate(
        self,
        context: Any,  # RAGPipelineContext
        query: str,
        **kwargs
    ) -> str:
        """Generate a response using the context and query."""
        ...
    
    async def generate_stream(
        self,
        context: Any,
        query: str,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate a streaming response."""
        ...


@dataclass
class GenerationConfig:
    """Configuration for response generation."""
    provider: str = "openai"
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 1024
    streaming: bool = False
    fallback_enabled: bool = True
    prompt_template: str = "default"
    system_prompt: Optional[str] = None
    api_key: Optional[str] = None
    timeout: int = 60
    retry_attempts: int = 3
    retry_delay: int = 2


@dataclass
class GenerationResult:
    """Result from response generation."""
    response: str
    tokens_used: int
    generation_time: float
    provider: str
    model: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class PromptTemplates:
    """Collection of prompt templates for different use cases."""
    
    DEFAULT = """You are a helpful AI assistant. Use the following context to answer the user's question.
If the context doesn't contain relevant information, say so clearly.

Context:
{context}

Question: {question}

Answer:"""
    
    DETAILED = """You are an expert research assistant. Analyze the following context carefully and provide a comprehensive answer to the user's question.

Context Documents:
{context}

User Question: {question}

Instructions:
1. Provide a detailed answer based on the context
2. Cite specific information from the context when possible
3. If information is missing, clearly state what is not available
4. Structure your response with clear sections if appropriate

Answer:"""
    
    CONCISE = """Based on the context below, provide a brief, direct answer to the question.

Context: {context}

Question: {question}

Brief Answer:"""
    
    ACADEMIC = """You are an academic researcher. Use the provided sources to answer the question with scholarly precision.

Research Sources:
{context}

Research Question: {question}

Provide a well-referenced answer with clear attribution to sources:"""
    
    CONVERSATIONAL = """Hey! I've found some information that might help answer your question.

Here's what I found:
{context}

Your question: {question}

Let me explain:"""
    
    @classmethod
    def get_template(cls, name: str) -> str:
        """Get a template by name."""
        templates = {
            "default": cls.DEFAULT,
            "detailed": cls.DETAILED,
            "concise": cls.CONCISE,
            "academic": cls.ACADEMIC,
            "conversational": cls.CONVERSATIONAL
        }
        return templates.get(name, cls.DEFAULT)


class BaseGenerator(ABC):
    """Base class for response generators."""
    
    def __init__(self, config: GenerationConfig):
        """Initialize generator with configuration."""
        self.config = config
        self.prompt_template = PromptTemplates.get_template(config.prompt_template)
    
    def format_context(self, documents: List[Document]) -> str:
        """Format documents into context string."""
        if not documents:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            # Format each document with metadata
            source = doc.metadata.get("source", "Unknown")
            title = doc.metadata.get("title", f"Document {i}")
            
            context_parts.append(f"[Source {i}: {title} ({source})]")
            context_parts.append(doc.content)
            context_parts.append("")  # Empty line between documents
        
        return "\n".join(context_parts)
    
    def build_prompt(self, context_text: str, query: str) -> str:
        """Build the final prompt from template."""
        return self.prompt_template.format(
            context=context_text,
            question=query
        )
    
    @abstractmethod
    async def generate(
        self,
        context: Any,
        query: str,
        **kwargs
    ) -> GenerationResult:
        """Generate a response."""
        pass
    
    async def generate_stream(
        self,
        context: Any,
        query: str,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate a streaming response."""
        # Default implementation: yield complete response
        result = await self.generate(context, query, **kwargs)
        yield result.response


class LLMGenerator(BaseGenerator):
    """Generator using LLM API calls."""
    
    async def generate(
        self,
        context: Any,
        query: str,
        **kwargs
    ) -> GenerationResult:
        """Generate response using configured LLM provider."""
        start_time = time.time()
        
        try:
            # Extract documents from context
            documents = context.documents if hasattr(context, 'documents') else []
            
            # Format context
            context_text = self.format_context(documents)
            
            # Build prompt
            prompt = self.build_prompt(context_text, query)
            
            # Add system prompt if configured
            if self.config.system_prompt:
                full_prompt = f"{self.config.system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            # Call appropriate LLM provider
            response = await self._call_llm(full_prompt, **kwargs)
            
            # Extract text from response
            if isinstance(response, dict):
                response_text = response.get("content", response.get("text", str(response)))
                tokens_used = response.get("usage", {}).get("total_tokens", 0)
            else:
                response_text = str(response)
                tokens_used = len(response_text.split()) * 1.3  # Rough estimate
            
            generation_time = time.time() - start_time
            
            logger.info(
                f"Generated response using {self.config.provider}/{self.config.model} "
                f"in {generation_time:.2f}s ({tokens_used} tokens)"
            )
            
            return GenerationResult(
                response=response_text,
                tokens_used=int(tokens_used),
                generation_time=generation_time,
                provider=self.config.provider,
                model=self.config.model,
                metadata={
                    "prompt_length": len(full_prompt),
                    "context_documents": len(documents)
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            
            # Try fallback if enabled
            if self.config.fallback_enabled:
                logger.info("Attempting fallback generation")
                fallback_gen = FallbackGenerator(self.config)
                return await fallback_gen.generate(context, query, **kwargs)
            
            raise
    
    async def _call_llm(self, prompt: str, **kwargs) -> Any:
        """Call the appropriate LLM provider."""
        provider = self.config.provider.lower()
        
        # Prepare common parameters
        params = {
            "custom_prompt_arg": prompt,
            "streaming": self.config.streaming,
            "system_prompt": self.config.system_prompt
        }
        
        # Add API key if provided
        if self.config.api_key:
            params["api_key"] = self.config.api_key
        
        # Override with kwargs
        params.update(kwargs)
        
        # Call appropriate provider
        if provider == "openai":
            return await asyncio.to_thread(
                chat_with_openai,
                api_key=params.get("api_key"),
                file_path="",  # Not used for direct prompt
                custom_prompt_arg=prompt,
                streaming=self.config.streaming
            )
        
        elif provider == "anthropic":
            return await asyncio.to_thread(
                chat_with_anthropic,
                api_key=params.get("api_key"),
                file_path="",
                model=self.config.model,
                custom_prompt_arg=prompt,
                streaming=self.config.streaming
            )
        
        elif provider == "groq":
            return await asyncio.to_thread(
                chat_with_groq,
                api_key=params.get("api_key"),
                input_data=prompt,
                custom_prompt_arg="",
                system_prompt=self.config.system_prompt,
                streaming=self.config.streaming
            )
        
        elif provider == "openrouter":
            return await asyncio.to_thread(
                chat_with_openrouter,
                api_key=params.get("api_key"),
                input_data=prompt,
                custom_prompt_arg="",
                system_prompt=self.config.system_prompt,
                streaming=self.config.streaming
            )
        
        elif provider == "deepseek":
            return await asyncio.to_thread(
                chat_with_deepseek,
                api_key=params.get("api_key"),
                input_data=prompt,
                custom_prompt_arg="",
                system_prompt=self.config.system_prompt,
                streaming=self.config.streaming
            )
        
        elif provider == "huggingface":
            return await asyncio.to_thread(
                chat_with_huggingface,
                api_key=params.get("api_key"),
                input_data=prompt,
                custom_prompt_arg="",
                system_prompt=self.config.system_prompt,
                streaming=self.config.streaming
            )
        
        elif provider == "cohere":
            return await asyncio.to_thread(
                chat_with_cohere,
                api_key=params.get("api_key"),
                file_path="",
                model=self.config.model,
                custom_prompt_arg=prompt,
                streaming=self.config.streaming
            )
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")


class StreamingGenerator(LLMGenerator):
    """Generator with streaming response support."""
    
    async def generate_stream(
        self,
        context: Any,
        query: str,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate a streaming response."""
        # Enable streaming in config
        original_streaming = self.config.streaming
        self.config.streaming = True
        
        try:
            # Extract documents from context
            documents = context.documents if hasattr(context, 'documents') else []
            
            # Format context
            context_text = self.format_context(documents)
            
            # Build prompt
            prompt = self.build_prompt(context_text, query)
            
            # Add system prompt if configured
            if self.config.system_prompt:
                full_prompt = f"{self.config.system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt
            
            # Call LLM with streaming
            response = await self._call_llm(full_prompt, **kwargs)
            
            # Handle streaming response
            if hasattr(response, '__aiter__'):
                # Async iterator
                async for chunk in response:
                    if isinstance(chunk, dict):
                        text = chunk.get("content", chunk.get("text", ""))
                    else:
                        text = str(chunk)
                    if text:
                        yield text
            elif hasattr(response, '__iter__'):
                # Sync iterator - convert to async
                for chunk in response:
                    if isinstance(chunk, dict):
                        text = chunk.get("content", chunk.get("text", ""))
                    else:
                        text = str(chunk)
                    if text:
                        yield text
                    await asyncio.sleep(0)  # Allow other tasks
            else:
                # Non-streaming response
                if isinstance(response, dict):
                    text = response.get("content", response.get("text", str(response)))
                else:
                    text = str(response)
                
                # Simulate streaming by yielding in chunks
                chunk_size = 50  # characters
                for i in range(0, len(text), chunk_size):
                    yield text[i:i+chunk_size]
                    await asyncio.sleep(0.01)  # Small delay for streaming effect
                    
        finally:
            # Restore original streaming setting
            self.config.streaming = original_streaming


class FallbackGenerator(BaseGenerator):
    """Fallback generator when LLM is unavailable."""
    
    async def generate(
        self,
        context: Any,
        query: str,
        **kwargs
    ) -> GenerationResult:
        """Generate a simple response without LLM."""
        start_time = time.time()
        
        # Extract documents from context
        documents = context.documents if hasattr(context, 'documents') else []
        
        if not documents:
            response = (
                f"I couldn't find any relevant information to answer your question: '{query}'. "
                "Please try rephrasing your question or providing more context."
            )
        else:
            # Build a simple response from the context
            response_parts = [
                f"Based on the available information, here's what I found regarding: '{query}'",
                "",
                "Relevant Information:"
            ]
            
            for i, doc in enumerate(documents[:3], 1):  # Limit to top 3 documents
                title = doc.metadata.get("title", f"Source {i}")
                content_preview = doc.content[:500] + "..." if len(doc.content) > 500 else doc.content
                
                response_parts.append(f"\n{i}. From {title}:")
                response_parts.append(content_preview)
            
            response_parts.append(
                "\nNote: This is a simplified response. For a more detailed answer, "
                "please ensure the AI service is properly configured."
            )
            
            response = "\n".join(response_parts)
        
        generation_time = time.time() - start_time
        
        logger.info(f"Generated fallback response in {generation_time:.2f}s")
        
        return GenerationResult(
            response=response,
            tokens_used=len(response.split()),
            generation_time=generation_time,
            provider="fallback",
            model="none",
            metadata={
                "context_documents": len(documents),
                "fallback_reason": "LLM unavailable or error"
            }
        )


def create_generator(config: Union[GenerationConfig, Dict[str, Any]]) -> GenerationStrategy:
    """Factory function to create appropriate generator."""
    if isinstance(config, dict):
        config = GenerationConfig(**config)
    
    if config.streaming:
        logger.debug(f"Creating StreamingGenerator with provider: {config.provider}")
        return StreamingGenerator(config)
    elif config.provider == "fallback":
        logger.debug("Creating FallbackGenerator")
        return FallbackGenerator(config)
    else:
        logger.debug(f"Creating LLMGenerator with provider: {config.provider}")
        return LLMGenerator(config)


# Pipeline integration functions

async def generate_response(context: Any, **kwargs) -> Any:
    """Generate response for pipeline context."""
    config_dict = context.config.get("generation", {})
    
    # Override with kwargs
    config_dict.update(kwargs)
    
    # Create generator
    generator = create_generator(config_dict)
    
    # Generate response
    result = await generator.generate(context, context.query)
    
    # Add to context
    context.response = result.response
    context.metadata["generation"] = {
        "provider": result.provider,
        "model": result.model,
        "tokens_used": result.tokens_used,
        "generation_time": result.generation_time
    }
    
    return context


async def generate_streaming_response(context: Any, **kwargs) -> Any:
    """Generate streaming response for pipeline context."""
    config_dict = context.config.get("generation", {})
    config_dict["streaming"] = True
    
    # Override with kwargs
    config_dict.update(kwargs)
    
    # Create generator
    generator = create_generator(config_dict)
    
    # Store generator in context for streaming
    context.stream_generator = generator.generate_stream(context, context.query)
    context.metadata["streaming"] = True
    
    return context