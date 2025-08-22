"""
LLM-based reranking for improved search result relevance.

This module implements various reranking strategies using language models
to evaluate and reorder search results based on their relevance to the query.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Union, Literal, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
from loguru import logger
import numpy as np

from ..Chat.Chat_Functions import chat_api_call
from ..config import load_settings
from ..Metrics.metrics_logger import log_counter, log_histogram, timeit
from .simplified.vector_store import SearchResult, SearchResultWithCitations


@dataclass
class RerankingResult:
    """Result from reranking operation."""
    original_rank: int
    new_rank: int
    original_score: float
    rerank_score: float
    reasoning: Optional[str] = None
    
    @property
    def rank_change(self) -> int:
        """Calculate rank change (negative means improved)."""
        return self.new_rank - self.original_rank


@dataclass
class RerankingConfig:
    """Configuration for reranking operations."""
    # Model settings
    model_provider: str = "openai"
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.0  # Use deterministic scoring
    max_tokens: int = 100
    
    # Reranking settings
    strategy: Literal["pairwise", "listwise", "pointwise"] = "pointwise"
    top_k_to_rerank: int = 20  # Only rerank top K results
    batch_size: int = 5  # Number of results to evaluate at once
    include_reasoning: bool = False  # Whether to generate explanations
    
    # Scoring settings
    score_scale: Tuple[float, float] = (0.0, 1.0)  # Min and max scores
    combine_original_score: bool = True  # Combine with original retrieval score
    original_score_weight: float = 0.3  # Weight for original score (0-1)
    
    # Prompts
    system_prompt: Optional[str] = None
    scoring_prompt_template: Optional[str] = None
    
    # Performance settings
    cache_results: bool = True
    timeout_seconds: float = 30.0
    retry_on_failure: bool = True
    max_retries: int = 2


class BaseReranker(ABC):
    """Base class for reranking strategies."""
    
    def __init__(self, config: RerankingConfig):
        self.config = config
        self._cache = {} if config.cache_results else None
        self._settings = load_settings()
        
    @abstractmethod
    async def rerank(self, 
                     query: str, 
                     results: List[Union[SearchResult, SearchResultWithCitations]],
                     **kwargs) -> List[Union[SearchResult, SearchResultWithCitations]]:
        """Rerank search results based on relevance to query."""
        pass
    
    def _get_cache_key(self, query: str, result_ids: List[str]) -> str:
        """Generate cache key for reranking operation."""
        import hashlib
        key_str = f"{query}|{'|'.join(sorted(result_ids))}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call LLM with retry logic."""
        retries = 0
        while retries <= self.config.max_retries:
            try:
                response = await asyncio.wait_for(
                    self._call_llm_impl(prompt, system_prompt),
                    timeout=self.config.timeout_seconds
                )
                return response
            except asyncio.TimeoutError:
                logger.warning(f"LLM call timed out after {self.config.timeout_seconds}s")
                if not self.config.retry_on_failure or retries >= self.config.max_retries:
                    raise
                retries += 1
                await asyncio.sleep(1 * retries)  # Exponential backoff
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                if not self.config.retry_on_failure or retries >= self.config.max_retries:
                    raise
                retries += 1
                await asyncio.sleep(1 * retries)
    
    async def _call_llm_impl(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Implementation of LLM call."""
        # Get API key from settings
        api_key = None
        if self.config.model_provider == "openai":
            api_key = self._settings.get("API", {}).get("openai_api_key")
        elif self.config.model_provider == "anthropic":
            api_key = self._settings.get("API", {}).get("anthropic_api_key")
        elif self.config.model_provider == "groq":
            api_key = self._settings.get("API", {}).get("groq_api_key")
        elif self.config.model_provider == "deepseek":
            api_key = self._settings.get("api_settings", {}).get("deepseek", {}).get("api_key")
        # Add other providers as needed
        
        if not api_key:
            raise ValueError(f"No API key found for provider: {self.config.model_provider}")
        
        # Prepare messages
        messages_payload = []
        if system_prompt or self.config.system_prompt:
            messages_payload.append({
                "role": "system", 
                "content": system_prompt or self.config.system_prompt or "You are a search result relevance evaluator."
            })
        messages_payload.append({"role": "user", "content": prompt})
        
        # Call using chat_api_call
        try:
            # Run in executor since chat_api_call is sync
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                chat_api_call,
                api_key,                    # api_key
                messages_payload,           # messages_payload  
                self.config.model_provider, # provider
                self.config.model_name,     # model
                self.config.temperature,    # temp
                self.config.max_tokens      # maxp
            )
            
            # Extract the text response
            if isinstance(response, dict):
                # Handle standard OpenAI-style response
                if 'choices' in response and len(response['choices']) > 0:
                    return response['choices'][0]['message']['content']
                # Handle other response formats
                elif 'content' in response:
                    return response['content']
                elif 'text' in response:
                    return response['text']
                else:
                    logger.warning(f"Unexpected response format: {response}")
                    return str(response)
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise


class PointwiseReranker(BaseReranker):
    """Reranks each result independently with a relevance score."""
    
    def __init__(self, config: RerankingConfig):
        super().__init__(config)
        
        # Default system prompt for pointwise reranking
        if not config.system_prompt:
            self.config.system_prompt = """You are a search result relevance evaluator. 
Your task is to score how relevant a search result is to a given query.
Return only a JSON object with a 'score' field (0.0 to 1.0) and optionally a 'reasoning' field.
Higher scores indicate better relevance."""
        
        # Default scoring template
        if not config.scoring_prompt_template:
            self.config.scoring_prompt_template = """Query: {query}

Search Result:
Title: {title}
Content: {content}

How relevant is this search result to the query? Consider:
1. Direct answer to the query
2. Topical relevance
3. Information quality
4. Completeness

Return JSON: {{"score": 0.0-1.0{reasoning}}}"""
    
    @timeit("reranker_pointwise")
    async def rerank(self, 
                     query: str, 
                     results: List[Union[SearchResult, SearchResultWithCitations]],
                     **kwargs) -> List[Union[SearchResult, SearchResultWithCitations]]:
        """Rerank results using pointwise scoring."""
        if not results:
            return results
        
        # Limit to top K results
        results_to_rerank = results[:self.config.top_k_to_rerank]
        remaining_results = results[self.config.top_k_to_rerank:]
        
        # Check cache if enabled
        cache_key = None
        if self._cache is not None:
            cache_key = self._get_cache_key(query, [r.id for r in results_to_rerank])
            if cache_key in self._cache:
                log_counter("reranker_cache_hit", labels={"strategy": "pointwise"})
                cached_scores = self._cache[cache_key]
                return self._apply_scores(results_to_rerank, cached_scores) + remaining_results
        
        log_counter("reranker_cache_miss", labels={"strategy": "pointwise"})
        
        # Score each result
        scoring_tasks = []
        for i, result in enumerate(results_to_rerank):
            task = self._score_result(query, result, i)
            scoring_tasks.append(task)
        
        # Process in batches
        all_scores = []
        for i in range(0, len(scoring_tasks), self.config.batch_size):
            batch = scoring_tasks[i:i + self.config.batch_size]
            batch_scores = await asyncio.gather(*batch, return_exceptions=True)
            all_scores.extend(batch_scores)
        
        # Handle errors and compile results
        reranking_results = []
        for i, score_result in enumerate(all_scores):
            if isinstance(score_result, Exception):
                logger.error(f"Failed to score result {i}: {score_result}")
                # Keep original score on failure
                reranking_results.append(RerankingResult(
                    original_rank=i,
                    new_rank=i,
                    original_score=results_to_rerank[i].score,
                    rerank_score=results_to_rerank[i].score
                ))
            else:
                reranking_results.append(score_result)
        
        # Cache results if enabled
        if self._cache is not None and cache_key:
            self._cache[cache_key] = reranking_results
        
        # Apply scores and reorder
        reranked = self._apply_scores(results_to_rerank, reranking_results)
        
        # Log metrics
        self._log_reranking_metrics(reranking_results)
        
        return reranked + remaining_results
    
    async def _score_result(self, query: str, result: Union[SearchResult, SearchResultWithCitations], 
                           original_rank: int) -> RerankingResult:
        """Score a single result."""
        # Prepare content
        title = result.metadata.get('doc_title', 'Untitled')
        content = result.document[:500]  # Limit content length
        
        # Format prompt
        reasoning_part = ', "reasoning": "explanation"' if self.config.include_reasoning else ''
        prompt = self.config.scoring_prompt_template.format(
            query=query,
            title=title,
            content=content,
            reasoning=reasoning_part
        )
        
        try:
            # Get LLM response
            response = await self._call_llm(prompt)
            
            # Parse JSON response
            result_json = json.loads(response)
            score = float(result_json.get('score', 0.5))
            reasoning = result_json.get('reasoning') if self.config.include_reasoning else None
            
            # Clamp score to configured range
            score = max(self.config.score_scale[0], min(score, self.config.score_scale[1]))
            
            return RerankingResult(
                original_rank=original_rank,
                new_rank=original_rank,  # Will be updated later
                original_score=result.score,
                rerank_score=score,
                reasoning=reasoning
            )
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse LLM response: {e}, Response: {response[:200]}")
            # Return original score on parse error
            return RerankingResult(
                original_rank=original_rank,
                new_rank=original_rank,
                original_score=result.score,
                rerank_score=result.score
            )
    
    def _apply_scores(self, results: List[Union[SearchResult, SearchResultWithCitations]], 
                     reranking_results: List[RerankingResult]) -> List[Union[SearchResult, SearchResultWithCitations]]:
        """Apply reranking scores and reorder results."""
        # Calculate final scores
        scored_results = []
        for result, rerank_result in zip(results, reranking_results):
            if self.config.combine_original_score:
                # Weighted combination
                final_score = (
                    self.config.original_score_weight * rerank_result.original_score +
                    (1 - self.config.original_score_weight) * rerank_result.rerank_score
                )
            else:
                final_score = rerank_result.rerank_score
            
            # Create a copy of the result with new score
            result_copy = type(result)(
                id=result.id,
                score=final_score,
                document=result.document,
                metadata={**result.metadata, 'rerank_score': rerank_result.rerank_score}
            )
            
            # Preserve citations if present
            if hasattr(result, 'citations'):
                result_copy.citations = result.citations
            
            scored_results.append((final_score, result_copy, rerank_result))
        
        # Sort by final score (descending)
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        # Update ranks in reranking results
        for new_rank, (_, _, rerank_result) in enumerate(scored_results):
            rerank_result.new_rank = new_rank
        
        # Return reordered results
        return [result for _, result, _ in scored_results]
    
    def _log_reranking_metrics(self, reranking_results: List[RerankingResult]):
        """Log metrics about reranking performance."""
        if not reranking_results:
            return
        
        # Calculate rank changes
        rank_changes = [r.rank_change for r in reranking_results]
        avg_rank_change = sum(rank_changes) / len(rank_changes)
        
        # Calculate score changes
        score_changes = [r.rerank_score - r.original_score for r in reranking_results]
        avg_score_change = sum(score_changes) / len(score_changes)
        
        # Log metrics
        log_histogram("reranker_avg_rank_change", avg_rank_change)
        log_histogram("reranker_avg_score_change", avg_score_change)
        log_counter("reranker_results_processed", value=len(reranking_results))
        
        # Log significant reorderings
        significant_changes = sum(1 for r in reranking_results if abs(r.rank_change) >= 3)
        log_counter("reranker_significant_changes", value=significant_changes)


class PairwiseReranker(BaseReranker):
    """Reranks by comparing pairs of results."""
    
    def __init__(self, config: RerankingConfig):
        super().__init__(config)
        
        if not config.system_prompt:
            self.config.system_prompt = """You are a search result comparator.
Given a query and two search results, determine which one is more relevant.
Return only a JSON object with 'choice' (1 or 2) and optionally 'reasoning'."""
        
        if not config.scoring_prompt_template:
            self.config.scoring_prompt_template = """Query: {query}

Result 1:
Title: {title1}
Content: {content1}

Result 2:
Title: {title2}
Content: {content2}

Which result better answers the query? Consider relevance, accuracy, and completeness.

Return JSON: {{"choice": 1 or 2{reasoning}}}"""
    
    @timeit("reranker_pairwise")
    async def rerank(self, 
                     query: str, 
                     results: List[Union[SearchResult, SearchResultWithCitations]],
                     **kwargs) -> List[Union[SearchResult, SearchResultWithCitations]]:
        """Rerank using pairwise comparisons with tournament-style ranking."""
        if len(results) <= 1:
            return results
        
        # Limit to top K
        results_to_rerank = results[:self.config.top_k_to_rerank]
        remaining_results = results[self.config.top_k_to_rerank:]
        
        # Perform tournament-style comparisons
        reranked = await self._tournament_rank(query, results_to_rerank)
        
        log_counter("reranker_pairwise_complete", labels={"results": len(results_to_rerank)})
        
        return reranked + remaining_results
    
    async def _tournament_rank(self, query: str, results: List[Union[SearchResult, SearchResultWithCitations]]) -> List[Union[SearchResult, SearchResultWithCitations]]:
        """Use tournament-style ranking with pairwise comparisons."""
        # Implementation of merge sort with async comparisons
        if len(results) <= 1:
            return results
        
        mid = len(results) // 2
        left_half = results[:mid]
        right_half = results[mid:]
        
        # Recursively sort both halves
        left_sorted = await self._tournament_rank(query, left_half)
        right_sorted = await self._tournament_rank(query, right_half)
        
        # Merge with pairwise comparisons
        return await self._merge_with_comparisons(query, left_sorted, right_sorted)
    
    async def _merge_with_comparisons(self, query: str, left: List, right: List) -> List:
        """Merge two sorted lists using pairwise comparisons."""
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            # Compare current elements
            is_left_better = await self._compare_pair(query, left[i], right[j])
            
            if is_left_better:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        # Add remaining elements
        result.extend(left[i:])
        result.extend(right[j:])
        
        return result
    
    async def _compare_pair(self, query: str, result1: Union[SearchResult, SearchResultWithCitations], 
                           result2: Union[SearchResult, SearchResultWithCitations]) -> bool:
        """Compare two results and return True if result1 is better."""
        # Format prompt
        reasoning_part = ', "reasoning": "explanation"' if self.config.include_reasoning else ''
        prompt = self.config.scoring_prompt_template.format(
            query=query,
            title1=result1.metadata.get('doc_title', 'Untitled'),
            content1=result1.document[:300],
            title2=result2.metadata.get('doc_title', 'Untitled'),
            content2=result2.document[:300],
            reasoning=reasoning_part
        )
        
        try:
            response = await self._call_llm(prompt)
            result_json = json.loads(response)
            choice = int(result_json.get('choice', 1))
            
            log_counter("reranker_pairwise_comparison")
            
            return choice == 1
            
        except Exception as e:
            logger.error(f"Pairwise comparison failed: {e}")
            # Fall back to original scores
            return result1.score > result2.score


class ListwiseReranker(BaseReranker):
    """Reranks all results together in a single prompt."""
    
    def __init__(self, config: RerankingConfig):
        super().__init__(config)
        
        if not config.system_prompt:
            self.config.system_prompt = """You are a search result ranker.
Given a query and a list of search results, reorder them by relevance.
Return a JSON object with 'ranking' as an array of result indices in order of relevance."""
        
        if not config.scoring_prompt_template:
            self.config.scoring_prompt_template = """Query: {query}

Search Results:
{results_list}

Reorder these results by relevance to the query (most relevant first).
Return the indices in order.

Return JSON: {{"ranking": [indices in order]{reasoning}}}"""
    
    @timeit("reranker_listwise")
    async def rerank(self, 
                     query: str, 
                     results: List[Union[SearchResult, SearchResultWithCitations]],
                     **kwargs) -> List[Union[SearchResult, SearchResultWithCitations]]:
        """Rerank all results together."""
        if len(results) <= 1:
            return results
        
        # Limit to top K
        results_to_rerank = results[:min(self.config.top_k_to_rerank, 10)]  # Limit for prompt size
        remaining_results = results[len(results_to_rerank):]
        
        # Format results for prompt
        results_text = []
        for i, result in enumerate(results_to_rerank):
            title = result.metadata.get('doc_title', 'Untitled')
            content = result.document[:200]
            results_text.append(f"{i}. Title: {title}\n   Content: {content}...")
        
        results_list = "\n\n".join(results_text)
        
        # Format prompt
        reasoning_part = ', "reasoning": "explanation"' if self.config.include_reasoning else ''
        prompt = self.config.scoring_prompt_template.format(
            query=query,
            results_list=results_list,
            reasoning=reasoning_part
        )
        
        try:
            response = await self._call_llm(prompt)
            result_json = json.loads(response)
            ranking = result_json.get('ranking', list(range(len(results_to_rerank))))
            
            # Validate ranking
            if not self._validate_ranking(ranking, len(results_to_rerank)):
                logger.warning("Invalid ranking returned, using original order")
                return results
            
            # Reorder results
            reranked = [results_to_rerank[i] for i in ranking]
            
            log_counter("reranker_listwise_complete")
            
            return reranked + remaining_results
            
        except Exception as e:
            logger.error(f"Listwise reranking failed: {e}")
            return results
    
    def _validate_ranking(self, ranking: List[int], expected_length: int) -> bool:
        """Validate that ranking contains all indices exactly once."""
        if len(ranking) != expected_length:
            return False
        return set(ranking) == set(range(expected_length))


def create_reranker(strategy: str = "pointwise", **kwargs) -> BaseReranker:
    """Factory function to create a reranker with the specified strategy."""
    config = RerankingConfig(strategy=strategy, **kwargs)
    
    if strategy == "pointwise":
        return PointwiseReranker(config)
    elif strategy == "pairwise":
        return PairwiseReranker(config)
    elif strategy == "listwise":
        return ListwiseReranker(config)
    else:
        raise ValueError(f"Unknown reranking strategy: {strategy}")


# Convenience function for one-shot reranking
async def rerank_results(
    query: str,
    results: List[Union[SearchResult, SearchResultWithCitations]],
    strategy: str = "pointwise",
    **kwargs
) -> List[Union[SearchResult, SearchResultWithCitations]]:
    """Convenience function to rerank results without creating a reranker instance."""
    reranker = create_reranker(strategy, **kwargs)
    return await reranker.rerank(query, results, **kwargs)