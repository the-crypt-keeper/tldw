"""
Cached LLM responses for testing.

This module provides cached/mocked LLM responses to avoid making real API calls
during testing while still maintaining realistic response patterns.
"""

import json
from typing import Dict, Any, List, Optional
import hashlib


class LLMResponseCache:
    """Cache for LLM responses to enable deterministic testing."""
    
    # Pre-defined responses for common evaluation scenarios
    CACHED_RESPONSES = {
        "geval_coherence": {
            "role": "assistant",
            "content": json.dumps({
                "score": 4.5,
                "reasoning": "The summary maintains logical flow and consistency with the source material.",
                "details": {
                    "strengths": ["Clear structure", "Logical progression", "Good transitions"],
                    "weaknesses": ["Minor detail omission"],
                    "confidence": 0.85
                }
            })
        },
        
        "geval_consistency": {
            "role": "assistant",
            "content": json.dumps({
                "score": 4.8,
                "reasoning": "The summary accurately reflects the facts from the source text.",
                "details": {
                    "factual_accuracy": 0.96,
                    "no_hallucinations": True,
                    "confidence": 0.90
                }
            })
        },
        
        "geval_fluency": {
            "role": "assistant",
            "content": json.dumps({
                "score": 4.2,
                "reasoning": "The text is well-written with good grammar and readability.",
                "details": {
                    "grammar_score": 0.95,
                    "readability": "high",
                    "confidence": 0.88
                }
            })
        },
        
        "geval_relevance": {
            "role": "assistant",
            "content": json.dumps({
                "score": 4.6,
                "reasoning": "The summary captures the most important information from the source.",
                "details": {
                    "key_points_covered": 0.92,
                    "irrelevant_content": 0.02,
                    "confidence": 0.87
                }
            })
        },
        
        "rag_context_relevance": {
            "role": "assistant",
            "content": json.dumps({
                "score": 4.3,
                "reasoning": "The retrieved context is highly relevant to the query.",
                "analysis": {
                    "relevant_chunks": 4,
                    "total_chunks": 5,
                    "relevance_scores": [0.9, 0.85, 0.88, 0.92, 0.6]
                }
            })
        },
        
        "rag_answer_faithfulness": {
            "role": "assistant",
            "content": json.dumps({
                "score": 4.7,
                "reasoning": "The answer is well-grounded in the provided context.",
                "analysis": {
                    "supported_claims": 8,
                    "total_claims": 9,
                    "unsupported_claims": ["minor detail about implementation"]
                }
            })
        },
        
        "rag_answer_relevance": {
            "role": "assistant",
            "content": json.dumps({
                "score": 4.5,
                "reasoning": "The answer directly addresses the query.",
                "analysis": {
                    "query_coverage": 0.90,
                    "irrelevant_content": 0.05,
                    "missing_aspects": ["performance metrics"]
                }
            })
        },
        
        "response_quality_coherence": {
            "role": "assistant",
            "content": json.dumps({
                "score": 4.4,
                "reasoning": "The response is logically structured and easy to follow.",
                "breakdown": {
                    "logical_flow": 0.88,
                    "consistency": 0.90,
                    "clarity": 0.85
                }
            })
        },
        
        "response_quality_helpfulness": {
            "role": "assistant",
            "content": json.dumps({
                "score": 4.6,
                "reasoning": "The response provides valuable and actionable information.",
                "breakdown": {
                    "informativeness": 0.92,
                    "actionability": 0.88,
                    "completeness": 0.85
                }
            })
        },
        
        "custom_metric_evaluation": {
            "role": "assistant",
            "content": json.dumps({
                "score": 0.83,
                "reasoning": "Custom metric evaluation completed successfully.",
                "custom_analysis": {
                    "metric_value": 0.83,
                    "confidence_interval": [0.78, 0.88],
                    "sample_size": 100
                }
            })
        }
    }
    
    @classmethod
    def get_cached_response(cls, prompt_type: str, prompt_content: str = None) -> Dict[str, Any]:
        """Get a cached response based on prompt type."""
        # If we have a direct match, return it
        if prompt_type in cls.CACHED_RESPONSES:
            return cls.CACHED_RESPONSES[prompt_type]
        
        # Generate a deterministic response based on prompt content
        if prompt_content:
            # Use hash to generate consistent scores
            hash_val = int(hashlib.md5(prompt_content.encode()).hexdigest()[:8], 16)
            score = 3.0 + (hash_val % 20) / 10  # Score between 3.0 and 5.0
            
            return {
                "role": "assistant",
                "content": json.dumps({
                    "score": round(score, 1),
                    "reasoning": f"Evaluation completed for: {prompt_type}",
                    "confidence": 0.75 + (hash_val % 20) / 100
                })
            }
        
        # Default response
        return {
            "role": "assistant",
            "content": json.dumps({
                "score": 4.0,
                "reasoning": "Default evaluation response",
                "confidence": 0.80
            })
        }
    
    @classmethod
    def get_embedding_response(cls, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """Get a deterministic embedding based on text."""
        # Generate deterministic embedding based on text hash
        hash_val = hashlib.md5(text.encode()).hexdigest()
        
        # Create embedding vector (1536 dimensions for OpenAI embeddings)
        embedding_size = 1536 if "3" in model else 1024
        embedding = []
        
        for i in range(embedding_size):
            # Use hash chunks to generate values
            chunk = hash_val[i % len(hash_val):(i % len(hash_val)) + 2]
            value = int(chunk, 16) / 255.0 - 0.5  # Normalize to [-0.5, 0.5]
            embedding.append(value)
        
        return embedding
    
    @classmethod
    def get_batch_responses(cls, prompts: List[str], response_type: str = "evaluation") -> List[Dict[str, Any]]:
        """Get batch responses for multiple prompts."""
        responses = []
        
        for i, prompt in enumerate(prompts):
            if response_type == "evaluation":
                # Vary scores slightly for each prompt
                base_score = 4.0 + (i % 10) / 10
                responses.append({
                    "role": "assistant",
                    "content": json.dumps({
                        "score": round(base_score, 1),
                        "reasoning": f"Batch evaluation {i+1} of {len(prompts)}",
                        "batch_index": i,
                        "confidence": 0.80 + (i % 20) / 100
                    })
                })
            else:
                responses.append(cls.get_cached_response("custom_metric_evaluation", prompt))
        
        return responses
    
    @classmethod
    def get_streaming_response(cls, prompt_type: str, chunk_size: int = 20) -> List[str]:
        """Get a streaming response as chunks."""
        full_response = cls.get_cached_response(prompt_type)
        content = full_response["content"]
        
        # Split into chunks
        chunks = []
        for i in range(0, len(content), chunk_size):
            chunks.append(content[i:i + chunk_size])
        
        return chunks


class MockLLMClient:
    """Mock LLM client for testing without real API calls."""
    
    def __init__(self, model: str = "gpt-4", fail_after: Optional[int] = None):
        """
        Initialize mock LLM client.
        
        Args:
            model: Model name to simulate
            fail_after: Fail after this many calls (for testing error handling)
        """
        self.model = model
        self.fail_after = fail_after
        self.call_count = 0
        self.cache = LLMResponseCache()
    
    async def create_completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """Mock completion creation."""
        self.call_count += 1
        
        # Simulate failure if configured
        if self.fail_after and self.call_count > self.fail_after:
            raise Exception("Mock LLM API error")
        
        # Determine response type from prompt
        prompt_lower = prompt.lower()
        
        if "coherence" in prompt_lower:
            response_type = "geval_coherence"
        elif "consistency" in prompt_lower:
            response_type = "geval_consistency"
        elif "fluency" in prompt_lower:
            response_type = "geval_fluency"
        elif "relevance" in prompt_lower and "context" in prompt_lower:
            response_type = "rag_context_relevance"
        elif "faithfulness" in prompt_lower:
            response_type = "rag_answer_faithfulness"
        elif "quality" in prompt_lower:
            response_type = "response_quality_coherence"
        else:
            response_type = "custom_metric_evaluation"
        
        response = self.cache.get_cached_response(response_type, prompt)
        
        return {
            "id": f"mock-{self.call_count}",
            "model": self.model,
            "choices": [{
                "message": response,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response["content"].split()),
                "total_tokens": len(prompt.split()) + len(response["content"].split())
            }
        }
    
    async def create_embedding(self, text: str, model: str = None) -> List[float]:
        """Mock embedding creation."""
        self.call_count += 1
        
        if self.fail_after and self.call_count > self.fail_after:
            raise Exception("Mock embedding API error")
        
        return self.cache.get_embedding_response(text, model or self.model)


# Helper functions for testing
def create_mock_llm_client(**kwargs) -> MockLLMClient:
    """Create a mock LLM client for testing."""
    return MockLLMClient(**kwargs)


def get_deterministic_response(seed: str) -> Dict[str, Any]:
    """Get a deterministic response based on a seed string."""
    cache = LLMResponseCache()
    return cache.get_cached_response("custom_metric_evaluation", seed)