# test_runner.py
# Test runner for executing evaluations in Prompt Studio

import json
import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from loguru import logger

from .prompt_executor import PromptExecutor, PromptValidator
from .evaluation_metrics import EvaluationMetrics, MetricType, EvaluationAggregator
from .test_case_manager import TestCaseManager
from .event_broadcaster import EventBroadcaster, EventType
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase

########################################################################################################################
# Test Runner

class TestRunner:
    """Runs test cases against prompts and evaluates results."""
    
    def __init__(self, db: PromptStudioDatabase, 
                 event_broadcaster: Optional[EventBroadcaster] = None):
        """
        Initialize TestRunner.
        
        Args:
            db: Database instance
            event_broadcaster: Optional event broadcaster for progress updates
        """
        self.db = db
        self.client_id = db.client_id
        self.executor = PromptExecutor(db)
        self.metrics_evaluator = EvaluationMetrics()
        self.aggregator = EvaluationAggregator()
        self.test_manager = TestCaseManager(db)
        self.broadcaster = event_broadcaster
    
    ####################################################################################################################
    # Single Test Execution
    
    async def run_single_test(self, prompt_id: int, test_case_id: int,
                             model_config: Dict[str, Any],
                             metrics: Optional[List[MetricType]] = None) -> Dict[str, Any]:
        """
        Run a single test case against a prompt.
        
        Args:
            prompt_id: Prompt ID
            test_case_id: Test case ID
            model_config: Model configuration
            metrics: Optional list of metrics to evaluate
            
        Returns:
            Test run result
        """
        start_time = time.time()
        
        try:
            # Get test case
            test_case = self.test_manager.get_test_case(test_case_id)
            if not test_case:
                raise ValueError(f"Test case {test_case_id} not found")
            
            # Execute prompt
            execution_result = await self.executor.execute_prompt(
                prompt_id=prompt_id,
                test_inputs=test_case.get("inputs", {}),
                model_config=model_config
            )
            
            # Evaluate output if expected outputs provided
            scores = {}
            if test_case.get("expected_outputs") and execution_result.get("success"):
                scores = self.metrics_evaluator.evaluate(
                    output=execution_result.get("parsed_output", execution_result.get("raw_output")),
                    expected=test_case["expected_outputs"],
                    metrics=metrics
                )
            
            # Prepare test run result
            test_run = {
                "prompt_id": prompt_id,
                "test_case_id": test_case_id,
                "test_case_name": test_case.get("name"),
                "model": model_config.get("model"),
                "provider": model_config.get("provider"),
                "inputs": test_case.get("inputs"),
                "expected_outputs": test_case.get("expected_outputs"),
                "actual_output": execution_result.get("parsed_output", execution_result.get("raw_output")),
                "raw_output": execution_result.get("raw_output"),
                "scores": scores,
                "success": execution_result.get("success", False),
                "error": execution_result.get("error"),
                "execution_time_ms": execution_result.get("execution_time_ms", 0),
                "tokens_used": execution_result.get("tokens_used", 0),
                "cost_estimate": execution_result.get("cost_estimate", 0),
                "metadata": execution_result.get("metadata", {}),
                "total_time_ms": (time.time() - start_time) * 1000
            }
            
            # Store in database
            test_run_id = await self._save_test_run(test_run)
            test_run["id"] = test_run_id
            
            return test_run
            
        except Exception as e:
            logger.error(f"Test run failed: {e}")
            return {
                "prompt_id": prompt_id,
                "test_case_id": test_case_id,
                "success": False,
                "error": str(e),
                "total_time_ms": (time.time() - start_time) * 1000
            }
    
    ####################################################################################################################
    # Batch Test Execution
    
    async def run_evaluation(self, evaluation_id: int, 
                            max_concurrent: int = 5,
                            progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Run a complete evaluation with multiple test cases and models.
        
        Args:
            evaluation_id: Evaluation ID
            max_concurrent: Maximum concurrent test runs
            progress_callback: Optional callback for progress updates
            
        Returns:
            Evaluation results
        """
        start_time = time.time()
        
        try:
            # Get evaluation details
            evaluation = self._get_evaluation(evaluation_id)
            if not evaluation:
                raise ValueError(f"Evaluation {evaluation_id} not found")
            
            # Parse configuration
            prompt_id = evaluation["prompt_id"]
            test_case_ids = json.loads(evaluation["test_case_ids"])
            model_configs = json.loads(evaluation["model_configs"])
            evaluation_config = json.loads(evaluation.get("evaluation_config", "{}"))
            
            # Get metrics to use
            metrics = None
            if evaluation_config.get("metrics"):
                metrics = [MetricType(m) for m in evaluation_config["metrics"]]
            
            # Update status to running
            self._update_evaluation_status(evaluation_id, "running")
            
            # Broadcast start event
            if self.broadcaster:
                await self.broadcaster.broadcast_event(
                    EventType.EVALUATION_STARTED,
                    {"evaluation_id": evaluation_id, "total_tests": len(test_case_ids) * len(model_configs)},
                    project_id=evaluation["project_id"]
                )
            
            # Create all test tasks
            tasks = []
            total_tests = len(test_case_ids) * len(model_configs)
            
            for test_case_id in test_case_ids:
                for model_config in model_configs:
                    task = self.run_single_test(
                        prompt_id=prompt_id,
                        test_case_id=test_case_id,
                        model_config=model_config,
                        metrics=metrics
                    )
                    tasks.append(task)
            
            # Run tests in batches
            test_runs = []
            completed = 0
            
            for i in range(0, len(tasks), max_concurrent):
                batch = tasks[i:i + max_concurrent]
                batch_results = await asyncio.gather(*batch, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"Test execution error: {result}")
                        test_runs.append({
                            "success": False,
                            "error": str(result),
                            "scores": {}
                        })
                    else:
                        test_runs.append(result)
                    
                    completed += 1
                    
                    # Progress update
                    if progress_callback:
                        progress_callback(completed, total_tests)
                    
                    # Broadcast progress
                    if self.broadcaster:
                        await self.broadcaster.broadcast_evaluation_progress(
                            evaluation_id=evaluation_id,
                            tests_completed=completed,
                            total_tests=total_tests
                        )
            
            # Aggregate results
            aggregated_metrics = self.aggregator.aggregate_results(test_runs)
            
            # Group by model for comparison
            results_by_model = {}
            for run in test_runs:
                model_key = f"{run.get('provider', 'unknown')}/{run.get('model', 'unknown')}"
                if model_key not in results_by_model:
                    results_by_model[model_key] = []
                results_by_model[model_key].append(run)
            
            # Compare models
            model_comparison = self.aggregator.compare_models(results_by_model)
            
            # Calculate success rate
            successful_runs = sum(1 for run in test_runs if run.get("success", False))
            success_rate = (successful_runs / len(test_runs) * 100) if test_runs else 0
            
            # Prepare final results
            evaluation_results = {
                "evaluation_id": evaluation_id,
                "status": "completed",
                "total_tests": total_tests,
                "successful_tests": successful_runs,
                "success_rate": success_rate,
                "test_runs": test_runs,
                "aggregated_metrics": aggregated_metrics,
                "model_comparison": model_comparison,
                "execution_time_seconds": time.time() - start_time,
                "completed_at": datetime.utcnow().isoformat()
            }
            
            # Update evaluation in database
            self._update_evaluation_results(evaluation_id, evaluation_results)
            
            # Broadcast completion
            if self.broadcaster:
                await self.broadcaster.broadcast_event(
                    EventType.EVALUATION_COMPLETED,
                    {
                        "evaluation_id": evaluation_id,
                        "success_rate": success_rate,
                        "overall_score": aggregated_metrics.get("overall_score", 0)
                    },
                    project_id=evaluation["project_id"]
                )
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            
            # Update status to failed
            self._update_evaluation_status(evaluation_id, "failed", str(e))
            
            return {
                "evaluation_id": evaluation_id,
                "status": "failed",
                "error": str(e),
                "execution_time_seconds": time.time() - start_time
            }
    
    ####################################################################################################################
    # Parallel Execution Utilities
    
    async def run_parallel_tests(self, test_configs: List[Dict[str, Any]],
                                max_concurrent: int = 10) -> List[Dict[str, Any]]:
        """
        Run multiple test configurations in parallel.
        
        Args:
            test_configs: List of test configurations
            max_concurrent: Maximum concurrent executions
            
        Returns:
            List of test results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_with_semaphore(config):
            async with semaphore:
                return await self.run_single_test(
                    prompt_id=config["prompt_id"],
                    test_case_id=config["test_case_id"],
                    model_config=config["model_config"],
                    metrics=config.get("metrics")
                )
        
        tasks = [run_with_semaphore(config) for config in test_configs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Test {i} failed: {result}")
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "config": test_configs[i]
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    ####################################################################################################################
    # Comparison and Analysis
    
    async def compare_prompts(self, prompt_ids: List[int], test_case_ids: List[int],
                             model_config: Dict[str, Any],
                             metrics: Optional[List[MetricType]] = None) -> Dict[str, Any]:
        """
        Compare multiple prompts on the same test cases.
        
        Args:
            prompt_ids: List of prompt IDs to compare
            test_case_ids: List of test case IDs
            model_config: Model configuration
            metrics: Optional metrics to evaluate
            
        Returns:
            Comparison results
        """
        results_by_prompt = {}
        
        for prompt_id in prompt_ids:
            # Run tests for this prompt
            test_runs = []
            for test_case_id in test_case_ids:
                result = await self.run_single_test(
                    prompt_id=prompt_id,
                    test_case_id=test_case_id,
                    model_config=model_config,
                    metrics=metrics
                )
                test_runs.append(result)
            
            # Get prompt info
            prompt = self._get_prompt(prompt_id)
            prompt_key = f"Prompt {prompt_id}: {prompt.get('name', 'Unnamed')}"
            results_by_prompt[prompt_key] = test_runs
        
        # Compare results
        comparison = self.aggregator.compare_models(results_by_prompt)
        
        # Rename "models" to "prompts" for clarity
        comparison["prompts"] = comparison.pop("models", {})
        
        return comparison
    
    async def run_regression_test(self, prompt_id: int, baseline_prompt_id: int,
                                 test_case_ids: List[int],
                                 model_config: Dict[str, Any],
                                 threshold: float = 0.95) -> Dict[str, Any]:
        """
        Run regression test comparing a prompt against a baseline.
        
        Args:
            prompt_id: New prompt ID
            baseline_prompt_id: Baseline prompt ID
            test_case_ids: Test cases to run
            model_config: Model configuration
            threshold: Minimum score ratio to pass (new/baseline)
            
        Returns:
            Regression test results
        """
        # Run tests for both prompts
        comparison = await self.compare_prompts(
            prompt_ids=[baseline_prompt_id, prompt_id],
            test_case_ids=test_case_ids,
            model_config=model_config
        )
        
        # Extract scores
        baseline_key = f"Prompt {baseline_prompt_id}"
        new_key = f"Prompt {prompt_id}"
        
        baseline_score = comparison["prompts"].get(baseline_key, {}).get("overall_score", 0)
        new_score = comparison["prompts"].get(new_key, {}).get("overall_score", 0)
        
        # Calculate regression
        if baseline_score > 0:
            score_ratio = new_score / baseline_score
            passed = score_ratio >= threshold
        else:
            score_ratio = 1.0 if new_score >= baseline_score else 0.0
            passed = new_score >= baseline_score
        
        return {
            "passed": passed,
            "baseline_prompt_id": baseline_prompt_id,
            "new_prompt_id": prompt_id,
            "baseline_score": baseline_score,
            "new_score": new_score,
            "score_ratio": score_ratio,
            "threshold": threshold,
            "comparison": comparison
        }
    
    ####################################################################################################################
    # Database Operations
    
    async def _save_test_run(self, test_run: Dict[str, Any]) -> int:
        """Save test run to database."""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        # Get project ID from prompt
        cursor.execute(
            "SELECT project_id FROM prompt_studio_prompts WHERE id = ?",
            (test_run["prompt_id"],)
        )
        row = cursor.fetchone()
        project_id = row[0] if row else None
        
        # Insert test run
        cursor.execute("""
            INSERT INTO prompt_studio_test_runs (
                uuid, project_id, prompt_id, test_case_id,
                model_name, model_params, inputs, outputs,
                expected_outputs, scores, execution_time_ms,
                tokens_used, cost_estimate, error_message, client_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            f"run-{datetime.utcnow().timestamp()}",
            project_id,
            test_run["prompt_id"],
            test_run["test_case_id"],
            test_run.get("model"),
            json.dumps({"provider": test_run.get("provider")}),
            json.dumps(test_run.get("inputs", {})),
            json.dumps(test_run.get("actual_output", {})),
            json.dumps(test_run.get("expected_outputs")),
            json.dumps(test_run.get("scores", {})),
            test_run.get("execution_time_ms", 0),
            test_run.get("tokens_used", 0),
            test_run.get("cost_estimate", 0),
            test_run.get("error"),
            self.client_id
        ))
        
        test_run_id = cursor.lastrowid
        conn.commit()
        
        return test_run_id
    
    def _get_evaluation(self, evaluation_id: int) -> Optional[Dict[str, Any]]:
        """Get evaluation from database."""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM prompt_studio_evaluations WHERE id = ?",
            (evaluation_id,)
        )
        
        row = cursor.fetchone()
        if row:
            return self.db._row_to_dict(cursor, row)
        return None
    
    def _get_prompt(self, prompt_id: int) -> Optional[Dict[str, Any]]:
        """Get prompt from database."""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM prompt_studio_prompts WHERE id = ?",
            (prompt_id,)
        )
        
        row = cursor.fetchone()
        if row:
            return self.db._row_to_dict(cursor, row)
        return None
    
    def _update_evaluation_status(self, evaluation_id: int, status: str,
                                 error_message: Optional[str] = None):
        """Update evaluation status in database."""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        if status == "running":
            cursor.execute("""
                UPDATE prompt_studio_evaluations
                SET status = ?, started_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (status, evaluation_id))
        elif status in ["completed", "failed"]:
            cursor.execute("""
                UPDATE prompt_studio_evaluations
                SET status = ?, completed_at = CURRENT_TIMESTAMP, error_message = ?
                WHERE id = ?
            """, (status, error_message, evaluation_id))
        else:
            cursor.execute("""
                UPDATE prompt_studio_evaluations
                SET status = ?
                WHERE id = ?
            """, (status, evaluation_id))
        
        conn.commit()
    
    def _update_evaluation_results(self, evaluation_id: int, results: Dict[str, Any]):
        """Update evaluation with final results."""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        # Extract test run IDs
        test_run_ids = [run.get("id") for run in results.get("test_runs", []) if run.get("id")]
        
        cursor.execute("""
            UPDATE prompt_studio_evaluations
            SET status = 'completed',
                completed_at = CURRENT_TIMESTAMP,
                test_run_ids = ?,
                aggregate_metrics = ?,
                total_tokens = ?,
                total_cost = ?
            WHERE id = ?
        """, (
            json.dumps(test_run_ids),
            json.dumps(results.get("aggregated_metrics", {})),
            results.get("aggregated_metrics", {}).get("tokens", {}).get("total", 0),
            results.get("aggregated_metrics", {}).get("cost", {}).get("total", 0),
            evaluation_id
        ))
        
        conn.commit()