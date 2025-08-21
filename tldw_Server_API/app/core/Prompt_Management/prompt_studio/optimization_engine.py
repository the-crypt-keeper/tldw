# optimization_engine.py
# MIPRO-style optimization engine for Prompt Studio

import json
import random
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import numpy as np
from loguru import logger

from .prompt_executor import PromptExecutor
from .test_runner import TestRunner
from .evaluation_metrics import EvaluationMetrics
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase

########################################################################################################################
# Metric Types

class MetricType(str, Enum):
    """Types of metrics for evaluation."""
    ACCURACY = "accuracy"
    F1_SCORE = "f1_score"
    PRECISION = "precision"
    RECALL = "recall"
    EXACT_MATCH = "exact_match"
    SIMILARITY = "similarity"

########################################################################################################################
# Optimization Strategies

class OptimizationStrategy(str, Enum):
    """Optimization strategies available."""
    
    MIPRO = "mipro"  # Multi-Instruction Prompt Optimization
    BOOTSTRAP = "bootstrap"  # Bootstrap few-shot examples
    ITERATIVE = "iterative"  # Iterative refinement
    GENETIC = "genetic"  # Genetic algorithm
    BAYESIAN = "bayesian"  # Bayesian optimization
    GRID_SEARCH = "grid_search"  # Grid search
    RANDOM_SEARCH = "random_search"  # Random search

########################################################################################################################
# MIPRO Optimizer

class MIPROOptimizer:
    """
    Multi-Instruction Prompt Optimization (MIPRO) implementation.
    Based on DSPy's MIPRO approach for optimizing prompt instructions.
    """
    
    def __init__(self, db: PromptStudioDatabase, test_runner: TestRunner):
        """
        Initialize MIPRO optimizer.
        
        Args:
            db: Database instance
            test_runner: Test runner for evaluations
        """
        self.db = db
        self.test_runner = test_runner
        self.executor = PromptExecutor(db)
        self.metrics = EvaluationMetrics()
    
    async def optimize(self, initial_prompt_id: int, test_case_ids: List[int],
                       model_config: Dict[str, Any], max_iterations: int = 20,
                       target_metric: MetricType = MetricType.ACCURACY,
                       min_improvement: float = 0.01) -> Dict[str, Any]:
        """
        Optimize a prompt using MIPRO strategy.
        
        Args:
            initial_prompt_id: Starting prompt ID
            test_case_ids: Test cases for evaluation
            model_config: Model configuration
            max_iterations: Maximum optimization iterations
            target_metric: Metric to optimize
            min_improvement: Minimum improvement to continue
            
        Returns:
            Optimization results
        """
        logger.info(f"Starting MIPRO optimization for prompt {initial_prompt_id}")
        
        # Initialize
        current_prompt_id = initial_prompt_id
        best_prompt_id = initial_prompt_id
        best_score = await self._evaluate_prompt(
            initial_prompt_id, test_case_ids, model_config, target_metric
        )
        
        iteration_history = []
        no_improvement_count = 0
        
        for iteration in range(max_iterations):
            logger.info(f"MIPRO iteration {iteration + 1}/{max_iterations}")
            
            # Generate instruction candidates
            candidates = await self._generate_instruction_candidates(
                current_prompt_id, best_score, iteration
            )
            
            # Evaluate candidates
            candidate_scores = []
            for candidate in candidates:
                # Create new prompt with candidate instruction
                new_prompt_id = await self._create_prompt_variant(
                    current_prompt_id, candidate
                )
                
                # Evaluate
                score = await self._evaluate_prompt(
                    new_prompt_id, test_case_ids, model_config, target_metric
                )
                
                candidate_scores.append((new_prompt_id, score, candidate))
            
            # Select best candidate
            if candidate_scores:
                candidate_scores.sort(key=lambda x: x[1], reverse=True)
                new_prompt_id, new_score, best_candidate = candidate_scores[0]
                
                # Track iteration
                iteration_data = {
                    "iteration": iteration + 1,
                    "prompt_id": new_prompt_id,
                    "score": new_score,
                    "improvement": new_score - best_score,
                    "instruction": best_candidate
                }
                iteration_history.append(iteration_data)
                
                # Check for improvement
                if new_score > best_score + min_improvement:
                    logger.info(f"Improvement found: {best_score:.3f} -> {new_score:.3f}")
                    best_score = new_score
                    best_prompt_id = new_prompt_id
                    current_prompt_id = new_prompt_id
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    
                    # Early stopping
                    if no_improvement_count >= 3:
                        logger.info("Early stopping: No improvement for 3 iterations")
                        break
            
            # Check convergence
            if await self._check_convergence(iteration_history):
                logger.info("Convergence detected")
                break
        
        # Return optimization results
        return {
            "initial_prompt_id": initial_prompt_id,
            "optimized_prompt_id": best_prompt_id,
            "initial_score": iteration_history[0]["score"] if iteration_history else best_score,
            "final_score": best_score,
            "improvement": best_score - (iteration_history[0]["score"] if iteration_history else best_score),
            "iterations": len(iteration_history),
            "iteration_history": iteration_history,
            "strategy": "MIPRO"
        }
    
    async def _generate_instruction_candidates(self, prompt_id: int, 
                                              current_score: float,
                                              iteration: int) -> List[str]:
        """
        Generate instruction candidates for MIPRO.
        
        Args:
            prompt_id: Current prompt ID
            current_score: Current best score
            iteration: Current iteration number
            
        Returns:
            List of instruction candidates
        """
        # Get current prompt
        prompt = self._get_prompt(prompt_id)
        current_instruction = prompt.get("system_prompt", "")
        
        candidates = []
        
        # Strategy 1: Rephrase current instruction
        rephrased = await self._rephrase_instruction(current_instruction)
        if rephrased:
            candidates.append(rephrased)
        
        # Strategy 2: Add clarifying details
        detailed = await self._add_details(current_instruction, current_score)
        if detailed:
            candidates.append(detailed)
        
        # Strategy 3: Simplify instruction
        if iteration > 5:  # Try simplification after some iterations
            simplified = await self._simplify_instruction(current_instruction)
            if simplified:
                candidates.append(simplified)
        
        # Strategy 4: Add examples
        with_examples = await self._add_examples(current_instruction)
        if with_examples:
            candidates.append(with_examples)
        
        # Strategy 5: Combine successful patterns
        if iteration > 10:
            combined = await self._combine_patterns(current_instruction)
            if combined:
                candidates.append(combined)
        
        return candidates[:5]  # Limit to 5 candidates per iteration
    
    async def _rephrase_instruction(self, instruction: str) -> Optional[str]:
        """Rephrase instruction using LLM."""
        prompt = f"""Rephrase the following instruction to be clearer and more effective.
Keep the same intent but improve clarity and specificity.

Original instruction:
{instruction}

Rephrased instruction:"""
        
        try:
            result = await self.executor._call_llm(
                provider="openai",
                model="gpt-3.5-turbo",
                prompt=prompt,
                parameters={"temperature": 0.7, "max_tokens": 500}
            )
            return result["content"].strip()
        except:
            return None
    
    async def _add_details(self, instruction: str, current_score: float) -> Optional[str]:
        """Add clarifying details to instruction."""
        prompt = f"""The following instruction achieves {current_score:.1%} accuracy.
Add specific details and constraints to improve its effectiveness.

Current instruction:
{instruction}

Enhanced instruction with more details:"""
        
        try:
            result = await self.executor._call_llm(
                provider="openai",
                model="gpt-3.5-turbo",
                prompt=prompt,
                parameters={"temperature": 0.8, "max_tokens": 500}
            )
            return result["content"].strip()
        except:
            return None
    
    async def _simplify_instruction(self, instruction: str) -> Optional[str]:
        """Simplify instruction by removing unnecessary complexity."""
        prompt = f"""Simplify the following instruction while keeping its core requirements.
Remove unnecessary words and complexity.

Complex instruction:
{instruction}

Simplified instruction:"""
        
        try:
            result = await self.executor._call_llm(
                provider="openai",
                model="gpt-3.5-turbo",
                prompt=prompt,
                parameters={"temperature": 0.5, "max_tokens": 300}
            )
            return result["content"].strip()
        except:
            return None
    
    async def _add_examples(self, instruction: str) -> Optional[str]:
        """Add few-shot examples to instruction."""
        # This would ideally use actual successful examples from test runs
        enhanced = f"""{instruction}

Here are some examples of the expected format:
- Example 1: [Input] -> [Expected Output]
- Example 2: [Input] -> [Expected Output]

Follow these examples for consistency."""
        
        return enhanced
    
    async def _combine_patterns(self, instruction: str) -> Optional[str]:
        """Combine successful patterns from history."""
        # This would analyze successful prompts and combine patterns
        # For now, return a variation
        return f"""Follow these guidelines:
1. {instruction}
2. Be precise and consistent
3. Validate your output before responding"""
    
    async def _evaluate_prompt(self, prompt_id: int, test_case_ids: List[int],
                              model_config: Dict[str, Any],
                              target_metric: MetricType) -> float:
        """Evaluate a prompt and return target metric score."""
        scores = []
        
        for test_case_id in test_case_ids:
            result = await self.test_runner.run_single_test(
                prompt_id=prompt_id,
                test_case_id=test_case_id,
                model_config=model_config,
                metrics=[target_metric]
            )
            
            if result.get("success") and "scores" in result:
                score = result["scores"].get(target_metric.value, 0)
                scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    
    async def _create_prompt_variant(self, base_prompt_id: int, 
                                    new_instruction: str) -> int:
        """Create a new prompt variant with updated instruction."""
        # Get base prompt
        prompt = self._get_prompt(base_prompt_id)
        
        # Create new prompt
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO prompt_studio_prompts (
                uuid, project_id, signature_id, name, content,
                system_prompt, version, parent_version_id, client_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            f"opt-{datetime.utcnow().timestamp()}",
            prompt["project_id"],
            prompt.get("signature_id"),
            f"{prompt['name']} (Optimized)",
            prompt["content"],
            new_instruction,
            prompt["version"] + 1,
            base_prompt_id,
            self.db.client_id
        ))
        
        new_prompt_id = cursor.lastrowid
        conn.commit()
        
        return new_prompt_id
    
    async def _check_convergence(self, history: List[Dict[str, Any]]) -> bool:
        """Check if optimization has converged."""
        if len(history) < 5:
            return False
        
        # Check if last 5 scores are very similar
        recent_scores = [h["score"] for h in history[-5:]]
        std_dev = np.std(recent_scores)
        
        return std_dev < 0.001
    
    def _get_prompt(self, prompt_id: int) -> Dict[str, Any]:
        """Get prompt from database."""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM prompt_studio_prompts WHERE id = ?", (prompt_id,))
        row = cursor.fetchone()
        
        if row:
            return self.db._row_to_dict(cursor, row)
        return {}

########################################################################################################################
# Bootstrap Optimizer

class BootstrapOptimizer:
    """
    Bootstrap optimization for few-shot learning.
    Automatically selects best examples from test runs.
    """
    
    def __init__(self, db: PromptStudioDatabase, test_runner: TestRunner):
        """Initialize Bootstrap optimizer."""
        self.db = db
        self.test_runner = test_runner
        self.executor = PromptExecutor(db)
    
    async def optimize(self, prompt_id: int, test_case_ids: List[int],
                       model_config: Dict[str, Any], 
                       num_examples: int = 3,
                       selection_strategy: str = "diverse") -> Dict[str, Any]:
        """
        Optimize prompt by bootstrapping few-shot examples.
        
        Args:
            prompt_id: Base prompt ID
            test_case_ids: Test cases to use
            model_config: Model configuration
            num_examples: Number of examples to include
            selection_strategy: How to select examples (best, diverse, random)
            
        Returns:
            Optimization results
        """
        logger.info(f"Starting Bootstrap optimization for prompt {prompt_id}")
        
        # Run initial evaluation to get examples
        test_runs = []
        for test_case_id in test_case_ids:
            result = await self.test_runner.run_single_test(
                prompt_id=prompt_id,
                test_case_id=test_case_id,
                model_config=model_config
            )
            test_runs.append(result)
        
        # Select best examples
        examples = self._select_examples(test_runs, num_examples, selection_strategy)
        
        # Create new prompt with examples
        new_prompt_id = await self._create_prompt_with_examples(prompt_id, examples)
        
        # Evaluate new prompt
        new_scores = []
        for test_case_id in test_case_ids:
            result = await self.test_runner.run_single_test(
                prompt_id=new_prompt_id,
                test_case_id=test_case_id,
                model_config=model_config
            )
            
            if result.get("success") and "scores" in result:
                new_scores.append(result["scores"].get("aggregate_score", 0))
        
        # Calculate improvement
        original_scores = [
            run.get("scores", {}).get("aggregate_score", 0)
            for run in test_runs if run.get("success")
        ]
        
        original_mean = np.mean(original_scores) if original_scores else 0
        new_mean = np.mean(new_scores) if new_scores else 0
        
        return {
            "initial_prompt_id": prompt_id,
            "optimized_prompt_id": new_prompt_id,
            "initial_score": original_mean,
            "final_score": new_mean,
            "improvement": new_mean - original_mean,
            "num_examples": len(examples),
            "strategy": "Bootstrap"
        }
    
    def _select_examples(self, test_runs: List[Dict[str, Any]], 
                        num_examples: int,
                        strategy: str) -> List[Dict[str, Any]]:
        """Select examples based on strategy."""
        # Filter successful runs with good scores
        good_runs = [
            run for run in test_runs
            if run.get("success") and 
            run.get("scores", {}).get("aggregate_score", 0) > 0.8
        ]
        
        if not good_runs:
            good_runs = [run for run in test_runs if run.get("success")]
        
        if strategy == "best":
            # Select top scoring examples
            good_runs.sort(
                key=lambda x: x.get("scores", {}).get("aggregate_score", 0),
                reverse=True
            )
            return good_runs[:num_examples]
        
        elif strategy == "diverse":
            # Select diverse examples
            selected = []
            remaining = good_runs.copy()
            
            while len(selected) < num_examples and remaining:
                # Pick one at random
                if not selected:
                    selected.append(remaining.pop(random.randint(0, len(remaining)-1)))
                else:
                    # Pick most different from selected
                    max_diff = -1
                    max_idx = 0
                    
                    for i, candidate in enumerate(remaining):
                        diff = self._calculate_diversity(candidate, selected)
                        if diff > max_diff:
                            max_diff = diff
                            max_idx = i
                    
                    selected.append(remaining.pop(max_idx))
            
            return selected
        
        else:  # random
            random.shuffle(good_runs)
            return good_runs[:num_examples]
    
    def _calculate_diversity(self, candidate: Dict[str, Any], 
                           selected: List[Dict[str, Any]]) -> float:
        """Calculate diversity score for candidate."""
        # Simple diversity based on input/output differences
        diversity_scores = []
        
        for s in selected:
            # Compare inputs
            input_diff = 0
            for key in candidate.get("inputs", {}):
                if str(candidate["inputs"][key]) != str(s.get("inputs", {}).get(key)):
                    input_diff += 1
            
            # Compare outputs
            output_diff = 0
            for key in candidate.get("actual_output", {}):
                if str(candidate["actual_output"][key]) != str(s.get("actual_output", {}).get(key)):
                    output_diff += 1
            
            diversity_scores.append(input_diff + output_diff)
        
        return min(diversity_scores) if diversity_scores else float('inf')
    
    async def _create_prompt_with_examples(self, base_prompt_id: int,
                                          examples: List[Dict[str, Any]]) -> int:
        """Create new prompt with few-shot examples."""
        # Get base prompt
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM prompt_studio_prompts WHERE id = ?", (base_prompt_id,))
        row = cursor.fetchone()
        prompt = self.db._row_to_dict(cursor, row)
        
        # Build examples text
        examples_text = "Here are some examples:\n\n"
        for i, example in enumerate(examples, 1):
            examples_text += f"Example {i}:\n"
            examples_text += f"Input: {json.dumps(example.get('inputs', {}), indent=2)}\n"
            examples_text += f"Output: {json.dumps(example.get('actual_output', {}), indent=2)}\n\n"
        
        # Add examples to prompt
        new_content = f"{examples_text}{prompt['content']}"
        
        # Create new prompt
        cursor.execute("""
            INSERT INTO prompt_studio_prompts (
                uuid, project_id, signature_id, name, content,
                system_prompt, version, parent_version_id, client_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            f"bootstrap-{datetime.utcnow().timestamp()}",
            prompt["project_id"],
            prompt.get("signature_id"),
            f"{prompt['name']} (Bootstrap)",
            new_content,
            prompt.get("system_prompt"),
            prompt["version"] + 1,
            base_prompt_id,
            self.db.client_id
        ))
        
        new_prompt_id = cursor.lastrowid
        conn.commit()
        
        return new_prompt_id

########################################################################################################################
# Main Optimization Engine

class OptimizationEngine:
    """Main optimization engine coordinating different strategies."""
    
    def __init__(self, db: PromptStudioDatabase):
        """
        Initialize OptimizationEngine.
        
        Args:
            db: Database instance
        """
        self.db = db
        self.test_runner = TestRunner(db)
        
        # Initialize optimizers
        self.mipro = MIPROOptimizer(db, self.test_runner)
        self.bootstrap = BootstrapOptimizer(db, self.test_runner)
    
    async def optimize(self, optimization_id: int) -> Dict[str, Any]:
        """
        Run optimization based on configuration.
        
        Args:
            optimization_id: Optimization ID from database
            
        Returns:
            Optimization results
        """
        # Get optimization config
        optimization = self._get_optimization(optimization_id)
        if not optimization:
            raise ValueError(f"Optimization {optimization_id} not found")
        
        # Parse configuration
        config = json.loads(optimization.get("optimizer_config", "{}"))
        strategy = config.get("strategy", "mipro")
        
        # Update status
        self._update_optimization_status(optimization_id, "running")
        
        try:
            # Run optimization based on strategy
            if strategy == "mipro":
                results = await self.mipro.optimize(
                    initial_prompt_id=optimization["initial_prompt_id"],
                    test_case_ids=json.loads(optimization["test_case_ids"]),
                    model_config=json.loads(optimization["model_config"]),
                    max_iterations=optimization["max_iterations"],
                    target_metric=MetricType(config.get("target_metric", "accuracy"))
                )
            
            elif strategy == "bootstrap":
                results = await self.bootstrap.optimize(
                    prompt_id=optimization["initial_prompt_id"],
                    test_case_ids=json.loads(optimization["test_case_ids"]),
                    model_config=json.loads(optimization["model_config"]),
                    num_examples=config.get("num_examples", 3),
                    selection_strategy=config.get("selection_strategy", "diverse")
                )
            
            else:
                raise ValueError(f"Unknown optimization strategy: {strategy}")
            
            # Update optimization with results
            self._update_optimization_results(optimization_id, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            self._update_optimization_status(optimization_id, "failed", str(e))
            raise
    
    def _get_optimization(self, optimization_id: int) -> Optional[Dict[str, Any]]:
        """Get optimization from database."""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM prompt_studio_optimizations WHERE id = ?",
            (optimization_id,)
        )
        
        row = cursor.fetchone()
        if row:
            return self.db._row_to_dict(cursor, row)
        return None
    
    def _update_optimization_status(self, optimization_id: int, status: str,
                                   error_message: Optional[str] = None):
        """Update optimization status."""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        if status == "running":
            cursor.execute("""
                UPDATE prompt_studio_optimizations
                SET status = ?, started_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (status, optimization_id))
        else:
            cursor.execute("""
                UPDATE prompt_studio_optimizations
                SET status = ?, error_message = ?
                WHERE id = ?
            """, (status, error_message, optimization_id))
        
        conn.commit()
    
    def _update_optimization_results(self, optimization_id: int, results: Dict[str, Any]):
        """Update optimization with results."""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
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
            results.get("optimized_prompt_id"),
            results.get("iterations", 1),
            json.dumps({"score": results.get("initial_score", 0)}),
            json.dumps({"score": results.get("final_score", 0)}),
            results.get("improvement", 0) * 100,
            optimization_id
        ))
        
        conn.commit()