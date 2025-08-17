# eval_runner.py - Async evaluation runner for OpenAI-compatible API
"""
Orchestrates evaluation runs asynchronously.

Handles:
- Async task execution
- Progress tracking
- Result aggregation
- Integration with existing evaluation backends
"""

import asyncio
import time
import statistics
import httpx
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from loguru import logger

# Import existing evaluation modules
from tldw_Server_API.app.core.Evaluations.ms_g_eval import run_geval
from tldw_Server_API.app.core.Evaluations.rag_evaluator import RAGEvaluator
from tldw_Server_API.app.core.Evaluations.response_quality_evaluator import ResponseQualityEvaluator
from tldw_Server_API.app.core.DB_Management.Evaluations_DB import EvaluationsDatabase


class EvaluationRunner:
    """Manages asynchronous evaluation execution"""
    
    def __init__(self, db_path: str, max_concurrent_evals: int = 10, eval_timeout: int = 60):
        """
        Initialize evaluation runner with concurrency controls.
        
        Args:
            db_path: Path to evaluations database
            max_concurrent_evals: Maximum number of concurrent evaluations
            eval_timeout: Timeout in seconds for each evaluation
        """
        self.db = EvaluationsDatabase(db_path)
        self.rag_evaluator = RAGEvaluator()
        self.quality_evaluator = ResponseQualityEvaluator()
        self.running_tasks = {}  # Track running evaluations
        
        # Concurrency control
        self.semaphore = asyncio.Semaphore(max_concurrent_evals)
        self.eval_timeout = eval_timeout
    
    async def run_evaluation(
        self,
        run_id: str,
        eval_id: str,
        eval_config: Dict[str, Any],
        background: bool = True
    ):
        """
        Run an evaluation asynchronously.
        
        Args:
            run_id: The run ID
            eval_id: The evaluation ID
            eval_config: Configuration including eval_type, samples, etc.
            background: If True, run in background task
        """
        if background:
            # Start background task
            task = asyncio.create_task(self._execute_evaluation(run_id, eval_id, eval_config))
            self.running_tasks[run_id] = task
            return {"status": "started", "run_id": run_id}
        else:
            # Run synchronously (for testing)
            return await self._execute_evaluation(run_id, eval_id, eval_config)
    
    async def _execute_evaluation(
        self,
        run_id: str,
        eval_id: str,
        eval_config: Dict[str, Any]
    ):
        """Execute the evaluation"""
        try:
            # Update status to running
            self.db.update_run_status(run_id, "running")
            start_time = time.time()
            
            # Get evaluation details
            evaluation = self.db.get_evaluation(eval_id)
            if not evaluation:
                raise ValueError(f"Evaluation {eval_id} not found")
            
            eval_type = evaluation["eval_type"]
            eval_spec = evaluation["eval_spec"]
            
            # Get samples
            samples = await self._get_samples(evaluation, eval_config)
            total_samples = len(samples)
            
            # Initialize progress
            progress = {
                "total_samples": total_samples,
                "completed_samples": 0,
                "failed_samples": 0,
                "current_batch": 0
            }
            self.db.update_run_progress(run_id, progress)
            
            # Get evaluation function
            eval_fn = self._get_evaluation_function(eval_type, eval_spec)
            
            # Process samples in batches
            batch_size = eval_config.get("config", {}).get("batch_size", 10)
            max_workers = eval_config.get("config", {}).get("max_workers", 4)
            
            all_results = []
            sample_results = []
            
            for i in range(0, total_samples, batch_size):
                batch = samples[i:i + batch_size]
                progress["current_batch"] = i // batch_size + 1
                
                # Process batch
                batch_results = await self._process_batch(
                    batch,
                    eval_fn,
                    eval_spec,
                    eval_config,
                    max_workers
                )
                
                # Update progress
                for result in batch_results:
                    if result.get("error"):
                        progress["failed_samples"] += 1
                    else:
                        progress["completed_samples"] += 1
                        all_results.append(result)
                    
                    sample_results.append(result)
                
                self.db.update_run_progress(run_id, progress)
            
            # Calculate aggregate results
            aggregate = self._calculate_aggregate_results(
                all_results,
                eval_spec.get("metrics", []),
                eval_spec.get("threshold", 0.7)
            )
            
            # Calculate token usage
            usage = self._calculate_usage(all_results)
            
            # Prepare final results
            duration = time.time() - start_time
            results = {
                "aggregate": aggregate,
                "by_metric": self._calculate_metric_stats(all_results, eval_spec.get("metrics", [])),
                "sample_results": sample_results,
                "failed_samples": [r for r in sample_results if r.get("error")]
            }
            
            # Store results
            self.db.store_run_results(run_id, results, usage)
            
            # Send webhook if configured
            webhook_url = eval_config.get("webhook_url")
            if webhook_url:
                await self._send_webhook(webhook_url, run_id, eval_id, "completed", results)
            
            logger.info(f"Evaluation run {run_id} completed in {duration:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Evaluation run {run_id} failed: {e}")
            self.db.update_run_status(run_id, "failed", error_message=str(e))
            
            # Send failure webhook
            webhook_url = eval_config.get("webhook_url")
            if webhook_url:
                await self._send_webhook(webhook_url, run_id, eval_id, "failed", {"error": str(e)})
            
            raise
        finally:
            # Clean up running task
            if run_id in self.running_tasks:
                del self.running_tasks[run_id]
    
    async def _get_samples(
        self,
        evaluation: Dict[str, Any],
        eval_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get samples for evaluation"""
        # Check for dataset override
        if eval_config.get("dataset_override"):
            return eval_config["dataset_override"]["samples"]
        
        # Get from evaluation's dataset
        dataset_id = evaluation.get("dataset_id")
        if dataset_id:
            dataset = self.db.get_dataset(dataset_id)
            if dataset:
                return dataset["samples"]
        
        # Check if evaluation has inline dataset
        if evaluation.get("dataset"):
            return evaluation["dataset"]
        
        raise ValueError("No samples found for evaluation")
    
    def _get_evaluation_function(
        self,
        eval_type: str,
        eval_spec: Dict[str, Any]
    ) -> Callable:
        """Get the appropriate evaluation function"""
        if eval_type == "model_graded":
            sub_type = eval_spec.get("sub_type")
            # Default to summarization if sub_type is None or empty
            if not sub_type:
                sub_type = "summarization"
            
            if sub_type == "summarization":
                return self._eval_summarization
            elif sub_type == "rag":
                return self._eval_rag
            elif sub_type == "response_quality":
                return self._eval_response_quality
            else:
                raise ValueError(f"Unknown model_graded sub_type: {sub_type}")
        
        elif eval_type == "exact_match":
            return self._eval_exact_match
        
        elif eval_type == "includes":
            return self._eval_includes
        
        elif eval_type == "fuzzy_match":
            return self._eval_fuzzy_match
        
        else:
            raise ValueError(f"Unknown evaluation type: {eval_type}")
    
    async def _process_batch(
        self,
        batch: List[Dict[str, Any]],
        eval_fn: Callable,
        eval_spec: Dict[str, Any],
        eval_config: Dict[str, Any],
        max_workers: int
    ) -> List[Dict[str, Any]]:
        """Process a batch of samples with proper concurrency control"""
        
        async def eval_with_timeout_and_semaphore(sample, sample_id):
            """Evaluate a single sample with timeout and semaphore control"""
            async with self.semaphore:
                try:
                    # Apply timeout to individual evaluation
                    result = await asyncio.wait_for(
                        eval_fn(
                            sample=sample,
                            eval_spec=eval_spec,
                            config=eval_config,
                            sample_id=sample_id
                        ),
                        timeout=self.eval_timeout
                    )
                    return result
                except asyncio.TimeoutError:
                    logger.error(f"Evaluation timeout for {sample_id}")
                    return {"sample_id": sample_id, "error": f"Timeout after {self.eval_timeout}s"}
                except Exception as e:
                    logger.error(f"Evaluation failed for {sample_id}: {e}")
                    return {"sample_id": sample_id, "error": str(e)}
        
        # Create tasks with proper error handling
        tasks = []
        for i, sample in enumerate(batch):
            sample_id = f"sample_{i:04d}"
            task = eval_with_timeout_and_semaphore(sample, sample_id)
            tasks.append(task)
        
        # Process all tasks concurrently (semaphore will limit actual concurrency)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any unexpected exceptions from gather
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "sample_id": f"sample_{i:04d}",
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    # ============= Evaluation Functions =============
    
    async def _eval_summarization(
        self,
        sample: Dict[str, Any],
        eval_spec: Dict[str, Any],
        config: Dict[str, Any],
        sample_id: str
    ) -> Dict[str, Any]:
        """Evaluate summarization using G-Eval"""
        try:
            # Extract required fields
            source_text = sample["input"].get("source_text", "")
            summary = sample["input"].get("summary", "")
            
            # Run G-Eval with controlled thread pool usage
            # Use a dedicated executor to avoid exhausting the default thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,  # Use default executor but with timeout control
                run_geval,
                source_text,
                summary,
                eval_spec.get("evaluator_model", "openai"),
                False  # save=False
            )
            
            # Parse results
            scores = {}
            for metric in eval_spec.get("metrics", ["fluency", "consistency", "relevance", "coherence"]):
                # Extract score from result string (G-Eval returns formatted text)
                import re
                pattern = f"{metric}.*?([0-9.]+)"
                match = re.search(pattern, result, re.IGNORECASE)
                if match:
                    scores[metric] = float(match.group(1)) / 5.0  # Normalize to 0-1
            
            # Calculate pass/fail
            avg_score = statistics.mean(scores.values()) if scores else 0
            passed = avg_score >= eval_spec.get("threshold", 0.7)
            
            return {
                "sample_id": sample_id,
                "scores": scores,
                "passed": passed,
                "avg_score": avg_score
            }
            
        except Exception as e:
            logger.error(f"Summarization eval failed for {sample_id}: {e}")
            return {"sample_id": sample_id, "error": str(e)}
    
    async def _eval_rag(
        self,
        sample: Dict[str, Any],
        eval_spec: Dict[str, Any],
        config: Dict[str, Any],
        sample_id: str
    ) -> Dict[str, Any]:
        """Evaluate RAG system"""
        try:
            # Extract fields
            query = sample["input"].get("query", "")
            contexts = sample["input"].get("contexts", [])
            response = sample["input"].get("response", "")
            ground_truth = sample.get("expected", {}).get("answer", "")
            
            # Run RAG evaluation
            result = await self.rag_evaluator.evaluate(
                query=query,
                contexts=contexts,
                response=response,
                ground_truth=ground_truth,
                metrics=eval_spec.get("metrics", ["relevance", "faithfulness"]),
                api_name=eval_spec.get("evaluator_model", "openai")
            )
            
            # Extract scores
            scores = {}
            for metric_name, metric_data in result.get("metrics", {}).items():
                scores[metric_name] = metric_data.score
            
            # Calculate pass/fail
            avg_score = statistics.mean(scores.values()) if scores else 0
            passed = avg_score >= eval_spec.get("threshold", 0.7)
            
            return {
                "sample_id": sample_id,
                "scores": scores,
                "passed": passed,
                "avg_score": avg_score
            }
            
        except Exception as e:
            logger.error(f"RAG eval failed for {sample_id}: {e}")
            return {"sample_id": sample_id, "error": str(e)}
    
    async def _eval_response_quality(
        self,
        sample: Dict[str, Any],
        eval_spec: Dict[str, Any],
        config: Dict[str, Any],
        sample_id: str
    ) -> Dict[str, Any]:
        """Evaluate response quality"""
        try:
            # Extract fields
            prompt = sample["input"].get("prompt", "")
            response = sample["input"].get("response", "")
            expected_format = sample["input"].get("expected_format")
            
            # Run quality evaluation
            result = await self.quality_evaluator.evaluate(
                prompt=prompt,
                response=response,
                expected_format=expected_format,
                custom_criteria=eval_spec.get("custom_criteria"),
                api_name=eval_spec.get("evaluator_model", "openai")
            )
            
            # Extract scores
            scores = {}
            for metric_name, metric_data in result.get("metrics", {}).items():
                scores[metric_name] = metric_data.score
            
            # Add overall quality
            scores["overall_quality"] = result.get("overall_quality", 0)
            
            # Calculate pass/fail
            avg_score = scores.get("overall_quality", 0)
            passed = avg_score >= eval_spec.get("threshold", 0.7)
            
            return {
                "sample_id": sample_id,
                "scores": scores,
                "passed": passed,
                "avg_score": avg_score,
                "format_compliance": result.get("format_compliance", True)
            }
            
        except Exception as e:
            logger.error(f"Quality eval failed for {sample_id}: {e}")
            return {"sample_id": sample_id, "error": str(e)}
    
    async def _eval_exact_match(
        self,
        sample: Dict[str, Any],
        eval_spec: Dict[str, Any],
        config: Dict[str, Any],
        sample_id: str
    ) -> Dict[str, Any]:
        """Evaluate exact match"""
        try:
            output = sample["input"].get("output", "")
            expected = sample.get("expected", {}).get("output", "")
            
            # Normalize strings
            output = str(output).strip().lower()
            expected = str(expected).strip().lower()
            
            passed = output == expected
            score = 1.0 if passed else 0.0
            
            return {
                "sample_id": sample_id,
                "scores": {"exact_match": score},
                "passed": passed,
                "avg_score": score
            }
            
        except Exception as e:
            logger.error(f"Exact match eval failed for {sample_id}: {e}")
            return {"sample_id": sample_id, "error": str(e)}
    
    async def _eval_includes(
        self,
        sample: Dict[str, Any],
        eval_spec: Dict[str, Any],
        config: Dict[str, Any],
        sample_id: str
    ) -> Dict[str, Any]:
        """Evaluate if output includes expected content"""
        try:
            output = str(sample["input"].get("output", ""))
            expected_items = sample.get("expected", {}).get("includes", [])
            
            if isinstance(expected_items, str):
                expected_items = [expected_items]
            
            # Check each expected item
            found_count = 0
            for item in expected_items:
                if str(item).lower() in output.lower():
                    found_count += 1
            
            score = found_count / len(expected_items) if expected_items else 0
            passed = score >= eval_spec.get("threshold", 0.7)
            
            return {
                "sample_id": sample_id,
                "scores": {"includes": score},
                "passed": passed,
                "avg_score": score,
                "found": found_count,
                "total": len(expected_items)
            }
            
        except Exception as e:
            logger.error(f"Includes eval failed for {sample_id}: {e}")
            return {"sample_id": sample_id, "error": str(e)}
    
    async def _eval_fuzzy_match(
        self,
        sample: Dict[str, Any],
        eval_spec: Dict[str, Any],
        config: Dict[str, Any],
        sample_id: str
    ) -> Dict[str, Any]:
        """Evaluate fuzzy string matching"""
        try:
            from difflib import SequenceMatcher
            
            output = str(sample["input"].get("output", ""))
            expected = str(sample.get("expected", {}).get("output", ""))
            
            # Calculate similarity
            similarity = SequenceMatcher(None, output, expected).ratio()
            passed = similarity >= eval_spec.get("threshold", 0.7)
            
            return {
                "sample_id": sample_id,
                "scores": {"fuzzy_match": similarity},
                "passed": passed,
                "avg_score": similarity
            }
            
        except Exception as e:
            logger.error(f"Fuzzy match eval failed for {sample_id}: {e}")
            return {"sample_id": sample_id, "error": str(e)}
    
    # ============= Helper Methods =============
    
    def _calculate_aggregate_results(
        self,
        results: List[Dict[str, Any]],
        metrics: List[str],
        threshold: float
    ) -> Dict[str, Any]:
        """Calculate aggregate statistics"""
        if not results:
            return {
                "mean_score": 0,
                "std_dev": 0,
                "min_score": 0,
                "max_score": 0,
                "pass_rate": 0,
                "total_samples": 0,
                "failed_samples": 0
            }
        
        # Get all scores
        all_scores = []
        passed_count = 0
        
        for result in results:
            if "avg_score" in result:
                all_scores.append(result["avg_score"])
                if result.get("passed", False):
                    passed_count += 1
        
        if not all_scores:
            return {
                "mean_score": 0,
                "std_dev": 0,
                "min_score": 0,
                "max_score": 0,
                "pass_rate": 0,
                "total_samples": len(results),
                "failed_samples": len(results)
            }
        
        return {
            "mean_score": statistics.mean(all_scores),
            "std_dev": statistics.stdev(all_scores) if len(all_scores) > 1 else 0,
            "min_score": min(all_scores),
            "max_score": max(all_scores),
            "pass_rate": passed_count / len(results),
            "total_samples": len(results),
            "failed_samples": len(results) - len(all_scores)
        }
    
    def _calculate_metric_stats(
        self,
        results: List[Dict[str, Any]],
        metrics: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate per-metric statistics"""
        metric_scores = {metric: [] for metric in metrics}
        
        for result in results:
            if "scores" in result:
                for metric, score in result["scores"].items():
                    if metric in metric_scores:
                        metric_scores[metric].append(score)
        
        metric_stats = {}
        for metric, scores in metric_scores.items():
            if scores:
                metric_stats[metric] = {
                    "mean": statistics.mean(scores),
                    "std": statistics.stdev(scores) if len(scores) > 1 else 0,
                    "min": min(scores),
                    "max": max(scores)
                }
            else:
                metric_stats[metric] = {"mean": 0, "std": 0, "min": 0, "max": 0}
        
        return metric_stats
    
    def _calculate_usage(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate token usage"""
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        
        for result in results:
            if "usage" in result:
                total_tokens += result["usage"].get("total_tokens", 0)
                prompt_tokens += result["usage"].get("prompt_tokens", 0)
                completion_tokens += result["usage"].get("completion_tokens", 0)
        
        return {
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        }
    
    async def _send_webhook(
        self,
        webhook_url: str,
        run_id: str,
        eval_id: str,
        status: str,
        summary: Dict[str, Any]
    ):
        """Send webhook notification"""
        try:
            async with httpx.AsyncClient() as client:
                payload = {
                    "event": f"run.{status}",
                    "run_id": run_id,
                    "eval_id": eval_id,
                    "status": status,
                    "completed_at": int(datetime.utcnow().timestamp()),
                    "results_url": f"/v1/runs/{run_id}/results",
                    "summary": summary
                }
                
                response = await client.post(webhook_url, json=payload, timeout=10)
                response.raise_for_status()
                logger.info(f"Webhook sent to {webhook_url} for run {run_id}")
                
        except Exception as e:
            logger.error(f"Failed to send webhook to {webhook_url}: {e}")
    
    def cancel_run(self, run_id: str) -> bool:
        """Cancel a running evaluation"""
        if run_id in self.running_tasks:
            task = self.running_tasks[run_id]
            task.cancel()
            self.db.update_run_status(run_id, "cancelled")
            del self.running_tasks[run_id]
            return True
        return False
    
    def get_run_status(self, run_id: str) -> Optional[str]:
        """Get the status of a run"""
        run = self.db.get_run(run_id)
        if run:
            return run["status"]
        return None
    
    # Alias for compatibility with tests
    async def run_evaluation_async(self, run_id: str, eval_config: Dict[str, Any]):
        """Alias for run_evaluation to match test expectations"""
        eval_id = eval_config.get("eval_id")
        if not eval_id:
            raise ValueError("eval_id required in eval_config")
        return await self.run_evaluation(run_id, eval_id, eval_config, background=True)