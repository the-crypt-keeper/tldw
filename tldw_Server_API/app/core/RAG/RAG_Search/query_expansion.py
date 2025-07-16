"""
Query expansion module for RAG search.

This module provides query expansion functionality to improve search accuracy
by generating alternative phrasings, sub-queries, and related terms.
"""

import asyncio
import hashlib
import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache
from datetime import datetime, timedelta

from tldw_chatbook.RAG_Search.simplified.config import QueryExpansionConfig
from tldw_chatbook.LLM_Calls.LLM_API_Calls import chat_with_openai, chat_with_anthropic, chat_with_deepseek
from tldw_chatbook.LLM_Calls.LLM_API_Calls_Local import chat_with_ollama, chat_with_local_llm
from tldw_chatbook.config import get_cli_setting

logger = logging.getLogger(__name__)


class QueryExpansionCache:
    """Simple in-memory cache for expanded queries."""
    
    def __init__(self, ttl_seconds: int = 3600):
        self._cache: Dict[str, Tuple[List[str], datetime]] = {}
        self._ttl = timedelta(seconds=ttl_seconds)
    
    def get(self, query: str, config_hash: str) -> Optional[List[str]]:
        """Get cached expansion if available and not expired."""
        cache_key = f"{query}:{config_hash}"
        if cache_key in self._cache:
            expansions, timestamp = self._cache[cache_key]
            if datetime.now() - timestamp < self._ttl:
                return expansions
            else:
                del self._cache[cache_key]
        return None
    
    def set(self, query: str, config_hash: str, expansions: List[str]):
        """Cache the expansions."""
        cache_key = f"{query}:{config_hash}"
        self._cache[cache_key] = (expansions, datetime.now())
    
    def clear(self):
        """Clear all cached expansions."""
        self._cache.clear()


class QueryExpander:
    """Handles query expansion for improved RAG search."""
    
    # Default prompt templates
    DEFAULT_PROMPTS = {
        "default": """Generate {max_queries} alternative search queries for the following query. 
Each alternative should capture the same intent but use different wording or focus on different aspects.
Return ONLY the alternative queries, one per line, without numbering or explanations.

Original query: {query}""",
        
        "contextual": """Break down this search query into {max_queries} specific sub-queries that would help find comprehensive information.
Each sub-query should focus on a different aspect or component of the original query.
Return ONLY the sub-queries, one per line, without numbering or explanations.

Original query: {query}""",
        
        "synonyms": """Generate {max_queries} alternative search queries using synonyms and related terms for the following query.
Each alternative should use different vocabulary while maintaining the same meaning.
Return ONLY the alternative queries, one per line, without numbering or explanations.

Original query: {query}"""
    }
    
    def __init__(self, app: Any, config: QueryExpansionConfig):
        """Initialize the query expander."""
        self.app = app
        self.config = config
        self._cache = QueryExpansionCache() if config.cache_expansions else None
        
    def _get_config_hash(self) -> str:
        """Get a hash of the current configuration for cache invalidation."""
        config_str = f"{self.config.method}:{self.config.llm_provider}:{self.config.llm_model}:{self.config.local_model}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    async def expand_query(self, query: str) -> List[str]:
        """
        Expand a query into alternative formulations.
        
        Args:
            query: The original search query
            
        Returns:
            List of alternative queries (not including the original)
        """
        if not self.config.enabled:
            return []
        
        # Check cache first
        if self._cache:
            config_hash = self._get_config_hash()
            cached = self._cache.get(query, config_hash)
            if cached is not None:
                logger.debug(f"Using cached expansion for query: {query}")
                return cached
        
        try:
            # Generate expansions based on method
            if self.config.method == "llm":
                expansions = await self._expand_with_llm(query)
            elif self.config.method == "local_llm":
                expansions = await self._expand_with_local_llm(query)
            elif self.config.method == "llamafile":
                expansions = await self._expand_with_llamafile(query)
            elif self.config.method == "keywords":
                expansions = self._expand_with_keywords(query)
            else:
                logger.warning(f"Unknown expansion method: {self.config.method}")
                return []
            
            # Validate and limit to max_sub_queries (ensure it's between 1 and 5)
            max_queries = min(max(1, self.config.max_sub_queries), 5)
            expansions = expansions[:max_queries]
            
            # Cache the result
            if self._cache and expansions:
                self._cache.set(query, self._get_config_hash(), expansions)
            
            logger.info(f"Generated {len(expansions)} query expansions for: {query}")
            return expansions
            
        except Exception as e:
            logger.error(f"Error expanding query '{query}': {e}")
            return []
    
    async def _expand_with_llm(self, query: str) -> List[str]:
        """Expand query using a remote LLM API."""
        # Get the prompt template
        prompt_template = self._get_prompt_template()
        prompt = prompt_template.format(
            query=query,
            max_queries=self.config.max_sub_queries
        )
        
        # Prepare messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates alternative search queries."},
            {"role": "user", "content": prompt}
        ]
        
        # Get API key and call the appropriate provider
        provider = self.config.llm_provider.lower()
        api_key = get_cli_setting("API", f"{provider}_api_key", "")
        
        if not api_key:
            logger.error(f"No API key found for provider: {provider}")
            return []
        
        try:
            # Call the LLM based on provider
            if provider == "openai":
                response = await asyncio.to_thread(
                    chat_with_openai,
                    messages,
                    api_key,
                    self.config.llm_model,
                    temperature=0.7,
                    max_tokens=200
                )
            elif provider == "anthropic":
                response = await asyncio.to_thread(
                    chat_with_anthropic,
                    messages,
                    api_key,
                    self.config.llm_model,
                    temperature=0.7,
                    max_tokens=200
                )
            elif provider == "deepseek":
                # DeepSeek has a different function signature
                response = await asyncio.to_thread(
                    chat_with_deepseek,
                    messages,  # input_data
                    self.config.llm_model,  # model
                    api_key,  # api_key
                    None,  # system_message (already in messages)
                    0.7,  # temp
                    None,  # top_p
                    200  # max_tokens
                )
            else:
                # Try as a generic OpenAI-compatible endpoint
                response = await asyncio.to_thread(
                    chat_with_openai,
                    messages,
                    api_key,
                    self.config.llm_model,
                    temperature=0.7,
                    max_tokens=200
                )
            
            # Parse the response
            if response:
                return self._parse_llm_response(response)
            
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
        
        return []
    
    async def _expand_with_local_llm(self, query: str) -> List[str]:
        """Expand query using a local LLM (Ollama)."""
        # Get the prompt template
        prompt_template = self._get_prompt_template()
        prompt = prompt_template.format(
            query=query,
            max_queries=self.config.max_sub_queries
        )
        
        # Prepare messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates alternative search queries."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Call Ollama
            response = await asyncio.to_thread(
                chat_with_ollama,
                messages,
                api_key="",  # Ollama doesn't need an API key
                model=self.config.local_model,
                temperature=0.7,
                max_tokens=200
            )
            
            # Parse the response
            if response:
                return self._parse_llm_response(response)
                
        except Exception as e:
            logger.error(f"Local LLM call failed: {e}")
        
        return []
    
    async def _expand_with_llamafile(self, query: str) -> List[str]:
        """Expand query using llamafile local server."""
        # Get the prompt template
        prompt_template = self._get_prompt_template()
        prompt = prompt_template.format(
            query=query,
            max_queries=self.config.max_sub_queries
        )
        
        # Prepare messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant that generates alternative search queries."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Call llamafile via chat_with_local_llm
            # Use the local_model field for the model name
            response = await asyncio.to_thread(
                chat_with_local_llm,
                messages,
                model=self.config.local_model,  # Use local_model for llamafile
                temp=0.7,
                max_tokens=200
            )
            
            # Parse the response
            if response:
                return self._parse_llm_response(response)
                
        except Exception as e:
            logger.error(f"Llamafile call failed: {e}")
        
        return []
    
    def _expand_with_keywords(self, query: str) -> List[str]:
        """Expand query by extracting and recombining keywords."""
        # Simple keyword-based expansion
        # This is a basic implementation that can be enhanced
        
        # Clean and tokenize the query
        words = re.findall(r'\w+', query.lower())
        if len(words) < 2:
            return []
        
        expansions = []
        
        # Strategy 1: Remove stop words and reorder
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        content_words = [w for w in words if w not in stop_words]
        
        if len(content_words) >= 2:
            # Reorder words
            if len(content_words) > 2:
                expansions.append(' '.join(content_words[1:] + [content_words[0]]))
            expansions.append(' '.join(reversed(content_words)))
        
        # Strategy 2: Focus on different parts
        if len(words) > 3:
            # First half
            expansions.append(' '.join(words[:len(words)//2 + 1]))
            # Second half
            expansions.append(' '.join(words[len(words)//2:]))
        
        # Strategy 3: Add question words for different intents
        question_variants = []
        if not any(query.lower().startswith(q) for q in ['what', 'how', 'why', 'when', 'where', 'who']):
            base_query = ' '.join(content_words) if content_words else query
            question_variants = [
                f"what is {base_query}",
                f"how to {base_query}",
                f"why {base_query}"
            ]
        
        expansions.extend(question_variants)
        
        # Remove duplicates and limit
        seen = set()
        unique_expansions = []
        for exp in expansions:
            if exp and exp != query.lower() and exp not in seen:
                seen.add(exp)
                unique_expansions.append(exp)
        
        return unique_expansions[:self.config.max_sub_queries]
    
    def _parse_llm_response(self, response: str) -> List[str]:
        """Parse LLM response into a list of queries."""
        # Split by newlines and clean up
        lines = response.strip().split('\n')
        queries = []
        
        for line in lines:
            # Remove numbering (1., 2., etc.) and bullet points
            cleaned = re.sub(r'^[\d\.\-\*\s]+', '', line.strip())
            if cleaned and len(cleaned) > 3:  # Minimum query length
                queries.append(cleaned)
        
        return queries
    
    def _get_prompt_template(self) -> str:
        """Get the prompt template for query expansion."""
        template_name = self.config.expansion_prompt_template
        
        # Check if it's a built-in template
        if template_name in self.DEFAULT_PROMPTS:
            return self.DEFAULT_PROMPTS[template_name]
        
        # TODO: In the future, load custom templates from the prompts database
        # For now, fall back to default
        logger.warning(f"Unknown prompt template: {template_name}, using default")
        return self.DEFAULT_PROMPTS["default"]
    
    def combine_search_results(self, results_list: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Combine results from multiple queries, removing duplicates.
        
        Args:
            results_list: List of result sets from different queries
            
        Returns:
            Combined and deduplicated results
        """
        if not self.config.combine_results:
            # Just return the first set if not combining
            return results_list[0] if results_list else []
        
        # Track seen content by ID or content hash
        seen_ids = set()
        combined_results = []
        
        for results in results_list:
            for result in results:
                # Use ID if available, otherwise hash the content
                result_id = result.get('id')
                if not result_id:
                    content = result.get('content', '')
                    result_id = hashlib.md5(content.encode()).hexdigest()
                
                if result_id not in seen_ids:
                    seen_ids.add(result_id)
                    combined_results.append(result)
        
        # Sort by score if available
        if combined_results and 'score' in combined_results[0]:
            combined_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return combined_results