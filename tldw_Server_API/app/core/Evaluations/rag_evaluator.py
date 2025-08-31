# rag_evaluator.py - RAG System Evaluation Module
"""
Evaluation module for RAG (Retrieval-Augmented Generation) systems.

Implements metrics:
- Relevance: How relevant is the response to the query
- Faithfulness: Is the response grounded in retrieved contexts
- Answer Similarity: Similarity to ground truth (if available)
- Context Precision: Precision of retrieved contexts
- Context Recall: Coverage of necessary information
"""

import asyncio
from typing import List, Dict, Any, Optional
from loguru import logger
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze
from tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create import (
    create_embedding, 
    get_embedding_config
)
from tldw_Server_API.app.core.Evaluations.circuit_breaker import (
    llm_circuit_breaker,
    CircuitOpenError
)


class RAGEvaluator:
    """Evaluator for RAG system performance"""
    
    def __init__(self, embedding_provider: str = "openai", embedding_model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        """
        Initialize RAG evaluator with production embeddings.
        
        Args:
            embedding_provider: Provider for embeddings (openai, huggingface, cohere)
            embedding_model: Model to use for embeddings
            api_key: Optional API key for the embedding provider
        """
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.api_key = api_key
        
        # Get embedding configuration
        self.embedding_config = self._setup_embedding_config()
        
        # Lazy check - don't test embeddings on init, only when first used
        self._embedding_available = None  # Will be set on first use
        logger.info(f"RAG evaluator initialized with {embedding_provider}/{embedding_model} configuration")
    
    def _setup_embedding_config(self) -> Dict[str, Any]:
        """Setup embedding configuration."""
        from tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create import OpenAIModelCfg, HFModelCfg
        
        config = get_embedding_config()
        
        # Override model if specified
        if self.embedding_model:
            config["embedding_config"]["default_model_id"] = self.embedding_model
        
        # Add API key if provided - create proper model instance
        if self.api_key and self.embedding_provider == "openai":
            if "models" not in config["embedding_config"]:
                config["embedding_config"]["models"] = {}
            if self.embedding_model not in config["embedding_config"]["models"]:
                config["embedding_config"]["models"][self.embedding_model] = OpenAIModelCfg(
                    provider=self.embedding_provider,
                    model_name_or_path=self.embedding_model,
                    api_key=self.api_key
                )
        
        return config
    
    @property
    def embedding_available(self) -> bool:
        """Check if embeddings are available (lazy evaluation)."""
        if self._embedding_available is None:
            self._embedding_available = self._test_embeddings()
            if not self._embedding_available:
                logger.warning(f"Embeddings not available for {self.embedding_provider}/{self.embedding_model}. Using LLM-based fallback.")
        return self._embedding_available
    
    @embedding_available.setter
    def embedding_available(self, value: bool):
        """Set embedding availability (for testing)."""
        self._embedding_available = value
    
    def _test_embeddings(self) -> bool:
        """Test if embeddings are available."""
        try:
            # Try to get a test embedding
            result = create_embedding("test", self.embedding_config, model_id_override=self.embedding_model)
            return result is not None and len(result) > 0
        except Exception as e:
            logger.debug(f"Embeddings test failed: {e}")
            return False
        
    async def evaluate(
        self,
        query: str,
        contexts: List[str],
        response: str,
        ground_truth: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        api_name: str = "openai"
    ) -> Dict[str, Any]:
        """
        Evaluate RAG system performance.
        
        Args:
            query: User query
            contexts: Retrieved context chunks
            response: Generated response
            ground_truth: Optional ground truth answer
            metrics: Metrics to compute
            api_name: LLM API to use
            
        Returns:
            Evaluation results with metrics and suggestions
        """
        if metrics is None:
            metrics = ["relevance", "faithfulness", "answer_similarity"]
        
        results = {
            "metrics": {},
            "suggestions": []
        }
        
        # Evaluate each metric
        tasks = []
        if "relevance" in metrics:
            tasks.append(self._evaluate_relevance(query, response, api_name))
        
        if "faithfulness" in metrics:
            tasks.append(self._evaluate_faithfulness(response, contexts, api_name))
        
        if "answer_similarity" in metrics and ground_truth:
            tasks.append(self._evaluate_answer_similarity(response, ground_truth))
        
        if "context_precision" in metrics:
            tasks.append(self._evaluate_context_precision(query, contexts, api_name))
        
        if "context_recall" in metrics and ground_truth:
            tasks.append(self._evaluate_context_recall(ground_truth, contexts, api_name))
        
        # Run evaluations in parallel with error handling
        try:
            # Use return_exceptions=True to handle individual failures
            metric_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Compile results and handle exceptions
            failed_metrics = []
            for i, result in enumerate(metric_results):
                if isinstance(result, Exception):
                    # Log the failure but continue with other metrics
                    metric_name = metrics[i] if i < len(metrics) else f"metric_{i}"
                    logger.error(f"Metric {metric_name} failed: {result}")
                    failed_metrics.append(metric_name)
                else:
                    metric_name, metric_result = result
                    results["metrics"][metric_name] = metric_result
            
            # Add information about failed metrics
            if failed_metrics:
                results["failed_metrics"] = failed_metrics
                results["partial_results"] = True
        except Exception as e:
            logger.error(f"Critical failure in evaluation: {e}")
            raise ValueError(f"Evaluation failed: {str(e)}")
            
        # Generate suggestions based on scores
        results["suggestions"] = self._generate_suggestions(results["metrics"])
        
        return results
    
    async def _evaluate_relevance(self, query: str, response: str, api_name: str) -> tuple:
        """Evaluate relevance of response to query"""
        prompt = f"""
        Evaluate how relevant the following response is to the given query.
        
        Query: {query}
        
        Response: {response}
        
        Rate the relevance on a scale of 1-5 where:
        1 = Completely irrelevant
        2 = Mostly irrelevant with minor relevant points
        3 = Partially relevant
        4 = Mostly relevant with minor gaps
        5 = Highly relevant and comprehensive
        
        Provide only the numeric score.
        """
        
        try:
            # Use circuit breaker for LLM call
            score_str = await llm_circuit_breaker.call_with_breaker(
                api_name,
                analyze,
                api_name,  # First param to analyze
                query,     # input_data
                prompt,    # custom_prompt_arg
                "",       # api_key
                "You are an evaluation expert. Provide only numeric scores.",  # system_message
                0.1        # temp
            )
            
            score = float(score_str.strip()) / 5.0  # Normalize to 0-1
            
            return ("relevance", {
                "name": "relevance",
                "score": score,
                "raw_score": float(score_str.strip()),
                "explanation": "Measures how well the response addresses the query"
            })
            
        except CircuitOpenError as e:
            logger.warning(f"Circuit breaker open for relevance evaluation: {e}")
            # Re-raise with more context
            raise ValueError(f"Service temporarily unavailable for relevance evaluation: {str(e)}")
        except Exception as e:
            logger.error(f"Relevance evaluation failed: {e}")
            # Raise exception instead of returning 0.0
            raise ValueError(f"Relevance evaluation failed: {str(e)}")
    
    async def _evaluate_faithfulness(self, response: str, contexts: List[str], api_name: str) -> tuple:
        """Evaluate if response is grounded in contexts"""
        combined_context = "\n\n".join(contexts)
        
        prompt = f"""
        Evaluate if the following response is faithful to the provided contexts.
        Check if all claims in the response are supported by the contexts.
        
        Contexts:
        {combined_context}
        
        Response:
        {response}
        
        Rate faithfulness on a scale of 1-5 where:
        1 = Contains many unsupported claims
        2 = Some major unsupported claims
        3 = Mix of supported and unsupported claims
        4 = Mostly supported with minor unsupported details
        5 = Fully supported by contexts
        
        Provide only the numeric score.
        """
        
        try:
            score_str = await asyncio.to_thread(
                analyze,
                api_name,  # First param
                response,  # input_data
                prompt,    # custom_prompt_arg
                "",       # api_key
                "You are an evaluation expert. Provide only numeric scores.",  # system_message
                0.1        # temp
            )
            
            score = float(score_str.strip()) / 5.0
            
            return ("faithfulness", {
                "name": "faithfulness",
                "score": score,
                "raw_score": float(score_str.strip()),
                "explanation": "Measures if response is grounded in retrieved contexts"
            })
            
        except Exception as e:
            logger.error(f"Faithfulness evaluation failed: {e}")
            # Raise exception instead of returning 0.0
            raise ValueError(f"Faithfulness evaluation failed: {str(e)}")
    
    async def _evaluate_answer_similarity(self, response: str, ground_truth: str) -> tuple:
        """Evaluate similarity between response and ground truth"""
        
        # Use embedding-based similarity if available
        if self.embedding_available:
            try:
                # Get embeddings for both texts (create_embedding is synchronous, so run in executor)
                loop = asyncio.get_event_loop()
                response_embedding = await loop.run_in_executor(
                    None, create_embedding, response, self.embedding_config, self.embedding_model
                )
                ground_truth_embedding = await loop.run_in_executor(
                    None, create_embedding, ground_truth, self.embedding_config, self.embedding_model
                )
                
                # Convert to numpy arrays and calculate cosine similarity
                response_embedding = np.array(response_embedding)
                ground_truth_embedding = np.array(ground_truth_embedding)
                
                # Reshape for sklearn if needed
                if response_embedding.ndim == 1:
                    response_embedding = response_embedding.reshape(1, -1)
                if ground_truth_embedding.ndim == 1:
                    ground_truth_embedding = ground_truth_embedding.reshape(1, -1)
                
                similarity = cosine_similarity(response_embedding, ground_truth_embedding)[0][0]
                
                # Convert similarity to 1-5 scale for consistency
                # Cosine similarity ranges from -1 to 1, but typically 0 to 1 for text
                # Map [0, 1] to [1, 5]
                raw_score = 1 + (similarity * 4)  # Maps 0->1, 0.5->3, 1->5
                score = similarity  # Keep normalized 0-1 score
                
                return ("answer_similarity", {
                    "name": "answer_similarity",
                    "score": score,
                    "raw_score": raw_score,
                    "explanation": "Embedding-based semantic similarity to ground truth answer",
                    "method": "embeddings"
                })
                
            except Exception as e:
                logger.warning(f"Embedding-based similarity failed: {e}. Falling back to LLM-based evaluation.")
        
        # Fallback to LLM-based similarity evaluation
        prompt = f"""
        Compare the similarity between the following response and ground truth answer.
        
        Response: {response}
        
        Ground Truth: {ground_truth}
        
        Rate similarity on a scale of 1-5 where:
        1 = Completely different meaning
        2 = Some shared concepts but mostly different
        3 = Moderately similar with key differences
        4 = Very similar with minor differences
        5 = Nearly identical meaning
        
        Provide only the numeric score.
        """
        
        try:
            score_str = await asyncio.to_thread(
                analyze,
                "openai",  # api_name - first param
                response,  # input_data
                prompt,    # custom_prompt_arg
                "",       # api_key
                "You are an evaluation expert. Provide only numeric scores.",  # system_message
                0.1        # temp
            )
            
            score = float(score_str.strip()) / 5.0
            
            return ("answer_similarity", {
                "name": "answer_similarity",
                "score": score,
                "raw_score": float(score_str.strip()),
                "explanation": "LLM-based semantic similarity to ground truth answer",
                "method": "llm"
            })
            
        except Exception as e:
            logger.error(f"Answer similarity evaluation failed: {e}")
            # Raise exception instead of returning 0.0 (fixing error handling issue)
            raise ValueError(f"Answer similarity evaluation failed: {str(e)}")
    
    async def _evaluate_context_precision(self, query: str, contexts: List[str], api_name: str) -> tuple:
        """Evaluate precision of retrieved contexts"""
        # Check each context for relevance
        relevance_scores = []
        
        for i, context in enumerate(contexts):
            prompt = f"""
            Rate how relevant this context is to the query on a scale of 1-5.
            
            Query: {query}
            
            Context: {context}
            
            Provide only the numeric score.
            """
            
            try:
                score_str = await asyncio.to_thread(
                    analyze,
                    api_name,  # First param
                    context,   # input_data
                    prompt,    # custom_prompt_arg
                    "",       # api_key
                    "You are an evaluation expert. Provide only numeric scores.",  # system_message
                    0.1        # temp
                )
                
                relevance_scores.append(float(score_str.strip()) / 5.0)
                
            except:
                relevance_scores.append(0.0)
        
        # Calculate precision as average relevance
        precision = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        
        return ("context_precision", {
            "name": "context_precision",
            "score": precision,
            "explanation": "Average relevance of retrieved contexts",
            "metadata": {"individual_scores": relevance_scores}
        })
    
    async def _evaluate_context_recall(self, ground_truth: str, contexts: List[str], api_name: str) -> tuple:
        """Evaluate if contexts contain necessary information"""
        combined_context = "\n\n".join(contexts)
        
        prompt = f"""
        Evaluate if the provided contexts contain all the information needed to answer with the ground truth.
        
        Ground Truth Answer: {ground_truth}
        
        Contexts:
        {combined_context}
        
        Rate coverage on a scale of 1-5 where:
        1 = Missing most key information
        2 = Missing several important points
        3 = Contains about half the needed information
        4 = Contains most information with minor gaps
        5 = Contains all necessary information
        
        Provide only the numeric score.
        """
        
        try:
            score_str = await asyncio.to_thread(
                analyze,
                api_name,  # First param
                combined_context,  # input_data
                prompt,    # custom_prompt_arg
                "",       # api_key
                "You are an evaluation expert. Provide only numeric scores.",  # system_message
                0.1        # temp
            )
            
            score = float(score_str.strip()) / 5.0
            
            return ("context_recall", {
                "name": "context_recall",
                "score": score,
                "raw_score": float(score_str.strip()),
                "explanation": "Coverage of necessary information in contexts"
            })
            
        except Exception as e:
            logger.error(f"Context recall evaluation failed: {e}")
            # Raise exception instead of returning 0.0
            raise ValueError(f"Context recall evaluation failed: {str(e)}")
    
    def _generate_suggestions(self, metrics: Dict[str, Dict]) -> List[str]:
        """Generate improvement suggestions based on metrics"""
        suggestions = []
        
        # Check relevance
        if "relevance" in metrics and metrics["relevance"]["score"] < 0.7:
            suggestions.append("Consider improving query understanding or response generation to better address user queries")
        
        # Check faithfulness
        if "faithfulness" in metrics and metrics["faithfulness"]["score"] < 0.7:
            suggestions.append("Improve grounding of responses in retrieved contexts to reduce hallucinations")
        
        # Check context precision
        if "context_precision" in metrics and metrics["context_precision"]["score"] < 0.6:
            suggestions.append("Enhance retrieval system to fetch more relevant contexts")
        
        # Check context recall
        if "context_recall" in metrics and metrics["context_recall"]["score"] < 0.7:
            suggestions.append("Expand retrieval to capture more comprehensive information")
        
        # Check answer similarity
        if "answer_similarity" in metrics and metrics["answer_similarity"]["score"] < 0.6:
            suggestions.append("Fine-tune generation to produce more accurate responses")
        
        return suggestions
    
    def close(self):
        """Clean up resources."""
        # No resources to clean up with direct embeddings API
        pass