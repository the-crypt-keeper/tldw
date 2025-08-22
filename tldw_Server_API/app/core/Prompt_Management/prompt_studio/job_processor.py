# job_processor.py
# Job processing handlers for Prompt Studio

import json
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger

from .job_manager import JobManager, JobType, JobStatus
from .test_case_manager import TestCaseManager
from .test_case_generator import TestCaseGenerator
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase

########################################################################################################################
# Job Processor

class JobProcessor:
    """Processes different types of Prompt Studio jobs."""
    
    def __init__(self, db: PromptStudioDatabase, job_manager: JobManager):
        """
        Initialize JobProcessor.
        
        Args:
            db: Database instance
            job_manager: Job manager instance
        """
        self.db = db
        self.job_manager = job_manager
        self.test_manager = TestCaseManager(db)
        self.test_generator = TestCaseGenerator(self.test_manager)
        
        # Register handlers
        self._register_handlers()
    
    def _register_handlers(self):
        """Register job handlers with the job manager."""
        self.job_manager.register_handler(JobType.GENERATION, self.process_generation_job)
        self.job_manager.register_handler(JobType.EVALUATION, self.process_evaluation_job)
        self.job_manager.register_handler(JobType.OPTIMIZATION, self.process_optimization_job)
    
    ####################################################################################################################
    # Generation Jobs
    
    async def process_generation_job(self, payload: Dict[str, Any], entity_id: int) -> Dict[str, Any]:
        """
        Process a test case generation job.
        
        Args:
            payload: Job payload with generation parameters
            entity_id: Project ID
            
        Returns:
            Generation results
        """
        try:
            project_id = entity_id
            generation_type = payload.get("type", "description")
            
            logger.info(f"Processing generation job for project {project_id}")
            
            if generation_type == "diverse":
                # Generate diverse test cases
                generated = await self._generate_diverse_cases(project_id, payload)
            elif generation_type == "description":
                # Generate from description
                generated = await self._generate_from_description(project_id, payload)
            elif generation_type == "data":
                # Generate from existing data
                generated = await self._generate_from_data(project_id, payload)
            else:
                raise ValueError(f"Unknown generation type: {generation_type}")
            
            result = {
                "generated_count": len(generated),
                "test_case_ids": [tc["id"] for tc in generated],
                "generation_type": generation_type,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Generated {len(generated)} test cases for project {project_id}")
            return result
            
        except Exception as e:
            logger.error(f"Generation job failed: {e}")
            raise
    
    async def _generate_diverse_cases(self, project_id: int, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate diverse test cases."""
        signature_id = payload.get("signature_id")
        num_cases = payload.get("num_cases", 5)
        
        if not signature_id:
            raise ValueError("signature_id required for diverse generation")
        
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.test_generator.generate_diverse_cases,
            project_id, signature_id, num_cases
        )
    
    async def _generate_from_description(self, project_id: int, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate test cases from description."""
        description = payload.get("description")
        num_cases = payload.get("num_cases", 5)
        signature_id = payload.get("signature_id")
        prompt_id = payload.get("prompt_id")
        
        if not description:
            raise ValueError("description required for description-based generation")
        
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.test_generator.generate_from_description,
            project_id, description, num_cases, signature_id, prompt_id
        )
    
    async def _generate_from_data(self, project_id: int, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate test cases from existing data."""
        source_data = payload.get("source_data", [])
        signature_id = payload.get("signature_id")
        
        if not source_data:
            raise ValueError("source_data required for data-based generation")
        
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.test_generator.generate_from_existing_data,
            project_id, source_data, signature_id
        )
    
    ####################################################################################################################
    # Evaluation Jobs
    
    async def process_evaluation_job(self, payload: Dict[str, Any], entity_id: int) -> Dict[str, Any]:
        """
        Process an evaluation job.
        
        Args:
            payload: Job payload with evaluation parameters
            entity_id: Evaluation ID
            
        Returns:
            Evaluation results
        """
        try:
            evaluation_id = entity_id
            prompt_id = payload.get("prompt_id")
            test_case_ids = payload.get("test_case_ids", [])
            model_configs = payload.get("model_configs", [])
            
            logger.info(f"Processing evaluation job {evaluation_id}")
            
            # Update evaluation status
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE prompt_studio_evaluations
                SET status = 'running', started_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (evaluation_id,))
            conn.commit()
            
            # Process test cases
            test_runs = []
            total_tokens = 0
            total_cost = 0.0
            
            for test_case_id in test_case_ids:
                for model_config in model_configs:
                    # Simulate test execution (would call actual LLM here)
                    test_run = await self._execute_test_case(
                        prompt_id, test_case_id, model_config
                    )
                    test_runs.append(test_run)
                    total_tokens += test_run.get("tokens_used", 0)
                    total_cost += test_run.get("cost_estimate", 0.0)
                
                # Add small delay to avoid rate limiting
                await asyncio.sleep(0.1)
            
            # Calculate aggregate metrics
            aggregate_metrics = self._calculate_aggregate_metrics(test_runs)
            
            # Update evaluation with results
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
                json.dumps([tr["id"] for tr in test_runs]),
                json.dumps(aggregate_metrics),
                total_tokens,
                total_cost,
                evaluation_id
            ))
            conn.commit()
            
            result = {
                "evaluation_id": evaluation_id,
                "test_runs": len(test_runs),
                "aggregate_metrics": aggregate_metrics,
                "total_tokens": total_tokens,
                "total_cost": total_cost,
                "status": "completed"
            }
            
            logger.info(f"Completed evaluation {evaluation_id} with {len(test_runs)} test runs")
            return result
            
        except Exception as e:
            logger.error(f"Evaluation job failed: {e}")
            
            # Update evaluation status to failed
            conn = self.db.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE prompt_studio_evaluations
                SET status = 'failed',
                    error_message = ?,
                    completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (str(e), entity_id))
            conn.commit()
            
            raise
    
    async def _execute_test_case(self, prompt_id: int, test_case_id: int, 
                                 model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single test case (simulation).
        
        In production, this would call the actual LLM API.
        """
        import random
        import uuid
        
        # Simulate execution delay
        await asyncio.sleep(random.uniform(0.5, 1.5))
        
        # Get test case
        test_case = self.test_manager.get_test_case(test_case_id)
        
        # Simulate test run result
        test_run = {
            "id": random.randint(1000, 9999),
            "uuid": str(uuid.uuid4()),
            "prompt_id": prompt_id,
            "test_case_id": test_case_id,
            "model_name": model_config.get("model", "gpt-3.5-turbo"),
            "inputs": test_case["inputs"],
            "outputs": {
                "result": f"Simulated output for {test_case.get('name', 'test')}"
            },
            "expected_outputs": test_case.get("expected_outputs"),
            "scores": {
                "accuracy": random.uniform(0.7, 1.0),
                "relevance": random.uniform(0.6, 1.0)
            },
            "execution_time_ms": random.randint(100, 2000),
            "tokens_used": random.randint(50, 500),
            "cost_estimate": random.uniform(0.001, 0.01)
        }
        
        # Store test run in database
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO prompt_studio_test_runs (
                uuid, project_id, prompt_id, test_case_id,
                model_name, model_params, inputs, outputs,
                expected_outputs, scores, execution_time_ms,
                tokens_used, cost_estimate, client_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            test_run["uuid"],
            test_case["project_id"],
            prompt_id,
            test_case_id,
            test_run["model_name"],
            json.dumps(model_config),
            json.dumps(test_run["inputs"]),
            json.dumps(test_run["outputs"]),
            json.dumps(test_run["expected_outputs"]),
            json.dumps(test_run["scores"]),
            test_run["execution_time_ms"],
            test_run["tokens_used"],
            test_run["cost_estimate"],
            self.db.client_id
        ))
        
        test_run["id"] = cursor.lastrowid
        conn.commit()
        
        return test_run
    
    def _calculate_aggregate_metrics(self, test_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate metrics from test runs."""
        if not test_runs:
            return {}
        
        # Calculate averages
        total_accuracy = sum(tr.get("scores", {}).get("accuracy", 0) for tr in test_runs)
        total_relevance = sum(tr.get("scores", {}).get("relevance", 0) for tr in test_runs)
        
        metrics = {
            "total_runs": len(test_runs),
            "avg_accuracy": total_accuracy / len(test_runs),
            "avg_relevance": total_relevance / len(test_runs),
            "avg_execution_time_ms": sum(tr.get("execution_time_ms", 0) for tr in test_runs) / len(test_runs),
            "total_tokens": sum(tr.get("tokens_used", 0) for tr in test_runs),
            "total_cost": sum(tr.get("cost_estimate", 0) for tr in test_runs)
        }
        
        return metrics
    
    ####################################################################################################################
    # Optimization Jobs
    
    async def process_optimization_job(self, payload: Dict[str, Any], entity_id: int) -> Dict[str, Any]:
        """
        Process an optimization job.
        
        Args:
            payload: Job payload with optimization parameters
            entity_id: Optimization ID
            
        Returns:
            Optimization results
        """
        try:
            optimization_id = entity_id
            initial_prompt_id = payload.get("initial_prompt_id")
            optimizer_type = payload.get("optimizer_type", "basic")
            max_iterations = payload.get("max_iterations", 20)
            
            logger.info(f"Processing optimization job {optimization_id}")
            
            # Update optimization status
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE prompt_studio_optimizations
                SET status = 'running', started_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (optimization_id,))
            conn.commit()
            
            # Simulate optimization iterations
            best_prompt_id = initial_prompt_id
            best_metric = 0.5
            iterations = []
            
            for i in range(min(max_iterations, 5)):  # Limit to 5 for simulation
                # Simulate iteration
                iteration_result = await self._run_optimization_iteration(
                    optimization_id, initial_prompt_id, i + 1
                )
                iterations.append(iteration_result)
                
                if iteration_result["metric"] > best_metric:
                    best_metric = iteration_result["metric"]
                    best_prompt_id = iteration_result.get("prompt_id", initial_prompt_id)
                
                # Check for early stopping
                if best_metric > 0.95:
                    logger.info(f"Early stopping at iteration {i + 1} with metric {best_metric}")
                    break
                
                await asyncio.sleep(0.5)  # Simulate processing time
            
            # Calculate improvement
            initial_metric = 0.5  # Simulated initial metric
            improvement = ((best_metric - initial_metric) / initial_metric) * 100
            
            # Update optimization with results
            cursor.execute("""
                UPDATE prompt_studio_optimizations
                SET status = 'completed',
                    completed_at = CURRENT_TIMESTAMP,
                    optimized_prompt_id = ?,
                    iterations_completed = ?,
                    initial_metrics = ?,
                    final_metrics = ?,
                    improvement_percentage = ?
                WHERE id = ?
            """, (
                best_prompt_id,
                len(iterations),
                json.dumps({"accuracy": initial_metric}),
                json.dumps({"accuracy": best_metric}),
                improvement,
                optimization_id
            ))
            conn.commit()
            
            result = {
                "optimization_id": optimization_id,
                "iterations_completed": len(iterations),
                "best_prompt_id": best_prompt_id,
                "best_metric": best_metric,
                "improvement_percentage": improvement,
                "status": "completed"
            }
            
            logger.info(f"Completed optimization {optimization_id} with {improvement:.1f}% improvement")
            return result
            
        except Exception as e:
            logger.error(f"Optimization job failed: {e}")
            
            # Update optimization status to failed
            conn = self.db.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE prompt_studio_optimizations
                SET status = 'failed',
                    error_message = ?,
                    completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (str(e), entity_id))
            conn.commit()
            
            raise
    
    async def _run_optimization_iteration(self, optimization_id: int, 
                                         prompt_id: int, iteration: int) -> Dict[str, Any]:
        """
        Run a single optimization iteration (simulation).
        
        In production, this would implement actual optimization logic.
        """
        import random
        
        # Simulate iteration processing
        await asyncio.sleep(random.uniform(1, 2))
        
        # Simulate metric improvement
        metric = 0.5 + (iteration * 0.1) + random.uniform(-0.05, 0.1)
        metric = min(1.0, max(0.0, metric))  # Clamp to [0, 1]
        
        return {
            "iteration": iteration,
            "prompt_id": prompt_id,
            "metric": metric,
            "tokens_used": random.randint(100, 1000),
            "cost": random.uniform(0.01, 0.1)
        }