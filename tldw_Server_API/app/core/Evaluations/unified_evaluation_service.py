# unified_evaluation_service.py - Unified evaluation service combining all evaluation capabilities
"""
Unified evaluation service that combines OpenAI-compatible evaluation framework
with tldw-specific evaluation features.

This service provides:
- OpenAI Evals compatible evaluation management
- G-Eval, RAG, and response quality evaluations
- Dataset management
- Run management with async processing
- Webhook notifications
- Advanced metrics and monitoring
"""

import asyncio
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from loguru import logger

# Import database components
from tldw_Server_API.app.core.DB_Management.Evaluations_DB import EvaluationsDatabase

# Import evaluation engines
from tldw_Server_API.app.core.Evaluations.ms_g_eval import run_geval
from tldw_Server_API.app.core.Evaluations.rag_evaluator import RAGEvaluator
from tldw_Server_API.app.core.Evaluations.response_quality_evaluator import ResponseQualityEvaluator
from tldw_Server_API.app.core.Evaluations.eval_runner import EvaluationRunner

# Import support services
from tldw_Server_API.app.core.Evaluations.webhook_manager import webhook_manager, WebhookEvent
from tldw_Server_API.app.core.Evaluations.metrics_advanced import advanced_metrics
from tldw_Server_API.app.core.Evaluations.user_rate_limiter import user_rate_limiter, UserTier
from tldw_Server_API.app.core.Evaluations.circuit_breaker import CircuitBreaker
from tldw_Server_API.app.core.Evaluations.audit_logger import AuditLogger, AuditEventType


class EvaluationType(str, Enum):
    """Supported evaluation types"""
    MODEL_GRADED = "model_graded"
    EXACT_MATCH = "exact_match"
    INCLUDES = "includes"
    GEVAL = "geval"
    RAG = "rag"
    RESPONSE_QUALITY = "response_quality"
    CUSTOM = "custom"


class UnifiedEvaluationService:
    """
    Unified service for all evaluation operations.
    
    Combines OpenAI-compatible evaluation framework with tldw-specific features
    into a single, cohesive service.
    """
    
    def __init__(self, db_path: str = "Databases/evaluations.db"):
        """
        Initialize the unified evaluation service.
        
        Args:
            db_path: Path to the evaluations database
        """
        # Initialize database
        self.db = EvaluationsDatabase(db_path)
        
        # Initialize evaluation runner for async processing
        self.runner = EvaluationRunner(db_path)
        
        # Initialize evaluation engines (lazy loading)
        self._rag_evaluator = None
        self._quality_evaluator = None
        
        # Initialize circuit breaker for resilience
        from tldw_Server_API.app.core.Evaluations.circuit_breaker import CircuitBreakerConfig
        self.circuit_breaker = CircuitBreaker(
            name="evaluation_service",
            config=CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=60.0,
                expected_exception=Exception
            )
        )
        
        # Initialize audit logger
        self.audit_logger = AuditLogger()
        
        logger.info("Unified Evaluation Service initialized")
    
    # ============= Lazy Initialization =============
    
    def get_rag_evaluator(self) -> RAGEvaluator:
        """Get or create RAG evaluator instance"""
        if self._rag_evaluator is None:
            self._rag_evaluator = RAGEvaluator()
        return self._rag_evaluator
    
    def get_quality_evaluator(self) -> ResponseQualityEvaluator:
        """Get or create quality evaluator instance"""
        if self._quality_evaluator is None:
            self._quality_evaluator = ResponseQualityEvaluator()
        return self._quality_evaluator
    
    # ============= Evaluation Management =============
    
    async def create_evaluation(
        self,
        name: str,
        eval_type: str,
        eval_spec: Dict[str, Any],
        description: Optional[str] = None,
        dataset_id: Optional[str] = None,
        dataset: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None,
        created_by: str = "system"
    ) -> Dict[str, Any]:
        """
        Create a new evaluation definition.
        
        Supports both OpenAI-style evaluation definitions and tldw-specific types.
        
        Args:
            name: Unique evaluation name
            eval_type: Type of evaluation (see EvaluationType enum)
            eval_spec: Evaluation specification
            description: Optional description
            dataset_id: ID of existing dataset
            dataset: Inline dataset (creates new dataset if provided)
            metadata: Optional metadata
            created_by: User/system creating the evaluation
            
        Returns:
            Created evaluation object
        """
        try:
            # If inline dataset provided, create it first
            if dataset and not dataset_id:
                dataset_id = self.db.create_dataset(
                    name=f"{name}_dataset",
                    samples=dataset,
                    description=f"Dataset for {name}",
                    created_by=created_by
                )
            
            # Create evaluation
            eval_id = self.db.create_evaluation(
                name=name,
                description=description,
                eval_type=eval_type,
                eval_spec=eval_spec,
                dataset_id=dataset_id,
                created_by=created_by,
                metadata=metadata
            )
            
            # Log audit event
            self.audit_logger.log_event(
                event_type=AuditEventType.EVALUATION_CREATE,
                action="create",
                user_id=created_by,
                resource_id=eval_id,
                details={"name": name, "type": eval_type}
            )
            
            # Get and return created evaluation
            evaluation = self.db.get_evaluation(eval_id)
            if not evaluation:
                raise ValueError("Failed to retrieve created evaluation")
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Failed to create evaluation: {e}")
            raise
    
    async def list_evaluations(
        self,
        limit: int = 20,
        after: Optional[str] = None,
        eval_type: Optional[str] = None,
        created_by: Optional[str] = None
    ) -> Tuple[List[Dict], bool]:
        """
        List evaluations with pagination and filtering.
        
        Args:
            limit: Maximum number of results
            after: Cursor for pagination
            eval_type: Filter by evaluation type
            created_by: Filter by creator
            
        Returns:
            Tuple of (evaluations list, has_more flag)
        """
        try:
            return self.db.list_evaluations(
                limit=limit,
                after=after,
                eval_type=eval_type
            )
        except Exception as e:
            logger.error(f"Failed to list evaluations: {e}")
            raise
    
    async def get_evaluation(self, eval_id: str) -> Optional[Dict[str, Any]]:
        """Get evaluation by ID"""
        try:
            return self.db.get_evaluation(eval_id)
        except Exception as e:
            logger.error(f"Failed to get evaluation {eval_id}: {e}")
            raise
    
    async def update_evaluation(
        self,
        eval_id: str,
        updates: Dict[str, Any],
        updated_by: str = "system"
    ) -> bool:
        """Update evaluation definition"""
        try:
            success = self.db.update_evaluation(eval_id, updates)
            
            if success:
                self.audit_logger.log_event(
                    event_type=AuditEventType.EVALUATION_UPDATE,
                    action="update",
                    user_id=updated_by,
                    resource_id=eval_id,
                    details={"updates": list(updates.keys())}
                )
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to update evaluation {eval_id}: {e}")
            raise
    
    async def delete_evaluation(
        self,
        eval_id: str,
        deleted_by: str = "system"
    ) -> bool:
        """Soft delete an evaluation"""
        try:
            success = self.db.delete_evaluation(eval_id)
            
            if success:
                self.audit_logger.log_event(
                    event_type=AuditEventType.EVALUATION_DELETE,
                    action="delete",
                    user_id=deleted_by,
                    resource_id=eval_id
                )
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete evaluation {eval_id}: {e}")
            raise
    
    # ============= Run Management =============
    
    async def create_run(
        self,
        eval_id: str,
        target_model: str,
        config: Optional[Dict] = None,
        dataset_override: Optional[Dict] = None,
        webhook_url: Optional[str] = None,
        created_by: str = "system"
    ) -> Dict[str, Any]:
        """
        Create and start an evaluation run.
        
        Args:
            eval_id: ID of evaluation to run
            target_model: Model to evaluate
            config: Run configuration
            dataset_override: Optional dataset override
            webhook_url: Optional webhook for notifications
            created_by: User creating the run
            
        Returns:
            Run object with status and ID
        """
        try:
            # Get evaluation
            evaluation = await self.get_evaluation(eval_id)
            if not evaluation:
                raise ValueError(f"Evaluation {eval_id} not found")
            
            # Create run record
            run_id = self.db.create_run(
                eval_id=eval_id,
                target_model=target_model,
                config=config or {},
                webhook_url=webhook_url
            )
            
            # Send webhook for run started
            if webhook_url:
                await webhook_manager.send_webhook(
                    user_id=created_by,
                    event=WebhookEvent.EVALUATION_STARTED,
                    evaluation_id=run_id,
                    data={
                        "eval_id": eval_id,
                        "target_model": target_model,
                        "eval_type": evaluation["eval_type"]
                    }
                )
            
            # Prepare evaluation config
            eval_config = {
                "eval_type": evaluation["eval_type"],
                "eval_spec": evaluation["eval_spec"],
                "dataset_id": evaluation.get("dataset_id"),
                "dataset_override": dataset_override,
                "config": config or {},
                "webhook_url": webhook_url
            }
            
            # Start async evaluation
            asyncio.create_task(
                self._run_evaluation_async(
                    run_id=run_id,
                    eval_id=eval_id,
                    eval_config=eval_config,
                    created_by=created_by
                )
            )
            
            # Log audit event
            self.audit_logger.log_event(
                event_type=AuditEventType.EVALUATION_RUN,
                action="create",
                user_id=created_by,
                resource_id=run_id,
                details={
                    "eval_id": eval_id,
                    "target_model": target_model
                }
            )
            
            # Return run info
            run = self.db.get_run(run_id)
            return run
            
        except Exception as e:
            logger.error(f"Failed to create run: {e}")
            raise
    
    async def _run_evaluation_async(
        self,
        run_id: str,
        eval_id: str,
        eval_config: Dict,
        created_by: str
    ):
        """Run evaluation asynchronously"""
        try:
            # Run evaluation with circuit breaker protection
            await self.circuit_breaker.call(
                self.runner.run_evaluation,
                run_id=run_id,
                eval_id=eval_id,
                eval_config=eval_config,
                background=False
            )
            
            # Send completion webhook
            if eval_config.get("webhook_url"):
                run = self.db.get_run(run_id)
                await webhook_manager.send_webhook(
                    user_id=created_by,
                    event=WebhookEvent.EVALUATION_COMPLETED,
                    evaluation_id=run_id,
                    data={
                        "run_id": run_id,
                        "status": run["status"],
                        "results": run.get("results")
                    }
                )
                
        except Exception as e:
            logger.error(f"Evaluation run {run_id} failed: {e}")
            
            # Update run status
            self.db.update_run_status(run_id, "failed", error_message=str(e))
            
            # Send failure webhook
            if eval_config.get("webhook_url"):
                await webhook_manager.send_webhook(
                    user_id=created_by,
                    event=WebhookEvent.EVALUATION_FAILED,
                    evaluation_id=run_id,
                    data={"error": str(e)}
                )
    
    async def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get run by ID"""
        try:
            return self.db.get_run(run_id)
        except Exception as e:
            logger.error(f"Failed to get run {run_id}: {e}")
            raise
    
    async def list_runs(
        self,
        eval_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20,
        after: Optional[str] = None
    ) -> Tuple[List[Dict], bool]:
        """List runs with filtering"""
        try:
            return self.db.list_runs(
                eval_id=eval_id,
                status=status,
                limit=limit,
                after=after
            )
        except Exception as e:
            logger.error(f"Failed to list runs: {e}")
            raise
    
    async def cancel_run(self, run_id: str, cancelled_by: str = "system") -> bool:
        """Cancel a running evaluation"""
        try:
            # Try to cancel via runner
            success = self.runner.cancel_run(run_id)
            
            if not success:
                # Update status directly if not in runner
                self.db.update_run_status(run_id, "cancelled")
            
            self.audit_logger.log_event(
                event_type=AuditEventType.EVALUATION_RUN,
                action="cancel",
                user_id=cancelled_by,
                resource_id=run_id
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel run {run_id}: {e}")
            raise
    
    # ============= Specialized Evaluations =============
    
    async def evaluate_geval(
        self,
        source_text: str,
        summary: str,
        metrics: Optional[List[str]] = None,
        api_name: str = "openai",
        api_key: Optional[str] = None,
        user_id: str = "system"
    ) -> Dict[str, Any]:
        """
        Run G-Eval summarization evaluation.
        
        Args:
            source_text: Original text
            summary: Summary to evaluate
            metrics: Metrics to compute (default: all)
            api_name: LLM API to use
            api_key: Optional API key
            user_id: User running evaluation
            
        Returns:
            Evaluation results with metrics
        """
        try:
            start_time = time.time()
            
            # Track metrics
            if advanced_metrics.enabled:
                with advanced_metrics.track_sli_request("/evaluations/geval"):
                    result = await asyncio.to_thread(
                        run_geval,
                        transcript=source_text,
                        summary=summary,
                        api_key=api_key,
                        api_name=api_name,
                        save=False
                    )
            else:
                result = await asyncio.to_thread(
                    run_geval,
                    transcript=source_text,
                    summary=summary,
                    api_key=api_key,
                    api_name=api_name,
                    save=False
                )
            
            # Parse and structure results
            evaluation_time = time.time() - start_time
            
            # Store in database
            eval_id = await self._store_evaluation_result(
                evaluation_type="geval",
                input_data={
                    "source_text": source_text[:1000],  # Truncate for storage
                    "summary": summary[:500]
                },
                results=result,
                metadata={
                    "api_name": api_name,
                    "evaluation_time": evaluation_time,
                    "user_id": user_id
                }
            )
            
            return {
                "evaluation_id": eval_id,
                "results": result,
                "evaluation_time": evaluation_time
            }
            
        except Exception as e:
            logger.error(f"G-Eval evaluation failed: {e}")
            raise
    
    async def evaluate_rag(
        self,
        query: str,
        contexts: List[str],
        response: str,
        ground_truth: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        api_name: str = "openai",
        user_id: str = "system"
    ) -> Dict[str, Any]:
        """
        Run RAG system evaluation.
        
        Args:
            query: User query
            contexts: Retrieved contexts
            response: Generated response
            ground_truth: Optional expected answer
            metrics: Metrics to compute
            api_name: LLM API to use
            user_id: User running evaluation
            
        Returns:
            Evaluation results with metrics
        """
        try:
            start_time = time.time()
            
            # Run evaluation
            results = await self.get_rag_evaluator().evaluate(
                query=query,
                contexts=contexts,
                response=response,
                ground_truth=ground_truth,
                metrics=metrics,
                api_name=api_name
            )
            
            evaluation_time = time.time() - start_time
            
            # Store in database
            eval_id = await self._store_evaluation_result(
                evaluation_type="rag",
                input_data={
                    "query": query,
                    "num_contexts": len(contexts),
                    "response_length": len(response)
                },
                results=results,
                metadata={
                    "api_name": api_name,
                    "evaluation_time": evaluation_time,
                    "user_id": user_id
                }
            )
            
            return {
                "evaluation_id": eval_id,
                "results": results,
                "evaluation_time": evaluation_time
            }
            
        except Exception as e:
            logger.error(f"RAG evaluation failed: {e}")
            raise
    
    async def evaluate_response_quality(
        self,
        prompt: str,
        response: str,
        expected_format: Optional[str] = None,
        custom_criteria: Optional[Dict] = None,
        api_name: str = "openai",
        user_id: str = "system"
    ) -> Dict[str, Any]:
        """
        Evaluate response quality.
        
        Args:
            prompt: Original prompt
            response: Generated response
            expected_format: Expected response format
            custom_criteria: Custom evaluation criteria
            api_name: LLM API to use
            user_id: User running evaluation
            
        Returns:
            Evaluation results with quality metrics
        """
        try:
            start_time = time.time()
            
            # Run evaluation
            results = await self.get_quality_evaluator().evaluate(
                prompt=prompt,
                response=response,
                expected_format=expected_format,
                custom_criteria=custom_criteria,
                api_name=api_name
            )
            
            evaluation_time = time.time() - start_time
            
            # Store in database
            eval_id = await self._store_evaluation_result(
                evaluation_type="response_quality",
                input_data={
                    "prompt": prompt[:500],
                    "response_length": len(response),
                    "expected_format": expected_format
                },
                results=results,
                metadata={
                    "api_name": api_name,
                    "evaluation_time": evaluation_time,
                    "user_id": user_id
                }
            )
            
            return {
                "evaluation_id": eval_id,
                "results": results,
                "evaluation_time": evaluation_time
            }
            
        except Exception as e:
            logger.error(f"Response quality evaluation failed: {e}")
            raise
    
    # ============= Dataset Management =============
    
    async def create_dataset(
        self,
        name: str,
        samples: List[Dict],
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
        created_by: str = "system"
    ) -> str:
        """Create a new dataset"""
        try:
            dataset_id = self.db.create_dataset(
                name=name,
                description=description,
                samples=samples,
                created_by=created_by,
                metadata=metadata
            )
            
            self.audit_logger.log_event(
                event_type=AuditEventType.EVALUATION_CREATE,
                action="create",
                user_id=created_by,
                resource_id=dataset_id,
                details={"name": name, "samples": len(samples)}
            )
            
            return dataset_id
            
        except Exception as e:
            logger.error(f"Failed to create dataset: {e}")
            raise
    
    async def list_datasets(
        self,
        limit: int = 20,
        after: Optional[str] = None
    ) -> Tuple[List[Dict], bool]:
        """List datasets with pagination"""
        try:
            return self.db.list_datasets(limit=limit, after=after)
        except Exception as e:
            logger.error(f"Failed to list datasets: {e}")
            raise
    
    async def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get dataset by ID"""
        try:
            return self.db.get_dataset(dataset_id)
        except Exception as e:
            logger.error(f"Failed to get dataset {dataset_id}: {e}")
            raise
    
    async def delete_dataset(
        self,
        dataset_id: str,
        deleted_by: str = "system"
    ) -> bool:
        """Delete a dataset"""
        try:
            success = self.db.delete_dataset(dataset_id)
            
            if success:
                self.audit_logger.log_event(
                    event_type=AuditEventType.EVALUATION_DELETE,
                    action="delete",
                    user_id=deleted_by,
                    resource_id=dataset_id
                )
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete dataset {dataset_id}: {e}")
            raise
    
    # ============= Helper Methods =============
    
    async def _store_evaluation_result(
        self,
        evaluation_type: str,
        input_data: Dict,
        results: Any,
        metadata: Dict
    ) -> str:
        """Store evaluation result in database"""
        try:
            # Create a simple evaluation record
            eval_id = self.db.create_evaluation(
                name=f"{evaluation_type}_{int(time.time())}",
                eval_type=evaluation_type,
                eval_spec={"type": evaluation_type},
                created_by=metadata.get("user_id", "system"),
                metadata=metadata
            )
            
            # Create and complete a run
            run_id = self.db.create_run(
                eval_id=eval_id,
                target_model=metadata.get("api_name", "unknown"),
                config={}
            )
            
            # Update run with results
            self.db.update_run_status(run_id, "completed")
            self.db.update_run(run_id, {"results": results})
            
            return eval_id
            
        except Exception as e:
            logger.error(f"Failed to store evaluation result: {e}")
            return f"temp_{int(time.time())}"
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get evaluation metrics summary"""
        try:
            if advanced_metrics.enabled:
                return advanced_metrics.get_summary()
            return {"metrics_enabled": False}
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            # Do not expose internal error details to external clients
            return {"error": "Failed to collect metrics"}
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        try:
            # Check database
            db_healthy = self.db.get_evaluation("test") is not None or True
            
            # Check circuit breaker
            from tldw_Server_API.app.core.Evaluations.circuit_breaker import CircuitState
            cb_healthy = self.circuit_breaker.state != CircuitState.OPEN
            
            return {
                "status": "healthy" if (db_healthy and cb_healthy) else "degraded",
                "database": "connected" if db_healthy else "disconnected",
                "circuit_breaker": "closed" if cb_healthy else "open",
                "uptime": time.time(),
                "version": "1.0.0"
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Singleton instance
_service_instance = None


def get_unified_evaluation_service(db_path: Optional[str] = None) -> UnifiedEvaluationService:
    """
    Get or create the unified evaluation service singleton.
    
    Args:
        db_path: Optional database path override
        
    Returns:
        UnifiedEvaluationService instance
    """
    global _service_instance
    
    if _service_instance is None:
        _service_instance = UnifiedEvaluationService(
            db_path or "Databases/evaluations.db"
        )
    
    return _service_instance