# optimization_strategies.py
# Additional optimization strategies for Prompt Studio

import json
import random
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np
from scipy import stats
from loguru import logger

from .prompt_executor import PromptExecutor
from .test_runner import TestRunner
from .evaluation_metrics import MetricType
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase

########################################################################################################################
# Hyperparameter Optimizer

class HyperparameterOptimizer:
    """
    Optimize model hyperparameters (temperature, max_tokens, etc.).
    Uses Bayesian optimization approach.
    """
    
    def __init__(self, db: PromptStudioDatabase, test_runner: TestRunner):
        """Initialize hyperparameter optimizer."""
        self.db = db
        self.test_runner = test_runner
        self.executor = PromptExecutor(db)
        
        # Define parameter search spaces
        self.param_spaces = {
            "temperature": (0.0, 2.0),
            "max_tokens": (50, 2000),
            "top_p": (0.1, 1.0),
            "frequency_penalty": (0.0, 2.0),
            "presence_penalty": (0.0, 2.0)
        }
    
    async def optimize(self, prompt_id: int, test_case_ids: List[int],
                       base_model_config: Dict[str, Any],
                       max_iterations: int = 20,
                       params_to_optimize: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Bayesian optimization.
        
        Args:
            prompt_id: Prompt to optimize
            test_case_ids: Test cases for evaluation
            base_model_config: Base model configuration
            max_iterations: Maximum iterations
            params_to_optimize: Specific parameters to optimize
            
        Returns:
            Optimization results with best parameters
        """
        logger.info(f"Starting hyperparameter optimization for prompt {prompt_id}")
        
        if params_to_optimize is None:
            params_to_optimize = ["temperature", "max_tokens", "top_p"]
        
        # Initialize with random samples
        observations = []
        
        # Random exploration phase
        for i in range(min(5, max_iterations)):
            params = self._sample_random_params(params_to_optimize)
            score = await self._evaluate_params(
                prompt_id, test_case_ids, base_model_config, params
            )
            observations.append((params, score))
            logger.info(f"Random sample {i+1}: score={score:.3f}, params={params}")
        
        # Bayesian optimization phase
        best_params = observations[0][0]
        best_score = observations[0][1]
        
        for i in range(5, max_iterations):
            # Use Gaussian Process to predict next point
            next_params = self._get_next_params(observations, params_to_optimize)
            
            # Evaluate
            score = await self._evaluate_params(
                prompt_id, test_case_ids, base_model_config, next_params
            )
            
            observations.append((next_params, score))
            logger.info(f"Iteration {i+1}: score={score:.3f}, params={next_params}")
            
            # Update best
            if score > best_score:
                best_score = score
                best_params = next_params
                logger.info(f"New best score: {best_score:.3f}")
            
            # Early stopping if converged
            if self._has_converged(observations):
                logger.info("Convergence detected")
                break
        
        return {
            "best_params": best_params,
            "best_score": best_score,
            "iterations": len(observations),
            "all_observations": observations,
            "improvement": best_score - observations[0][1]
        }
    
    def _sample_random_params(self, params_to_optimize: List[str]) -> Dict[str, Any]:
        """Sample random parameters from search space."""
        params = {}
        
        for param in params_to_optimize:
            if param in self.param_spaces:
                min_val, max_val = self.param_spaces[param]
                
                if param in ["max_tokens"]:
                    # Integer parameters
                    params[param] = random.randint(int(min_val), int(max_val))
                else:
                    # Float parameters
                    params[param] = random.uniform(min_val, max_val)
        
        return params
    
    def _get_next_params(self, observations: List[Tuple[Dict, float]],
                        params_to_optimize: List[str]) -> Dict[str, Any]:
        """
        Use acquisition function to get next parameters to try.
        Simplified Expected Improvement approach.
        """
        if len(observations) < 10:
            # Not enough data, sample randomly
            return self._sample_random_params(params_to_optimize)
        
        # Extract parameter values and scores
        X = []
        y = []
        
        for params, score in observations:
            x = [params.get(p, 0) for p in params_to_optimize]
            X.append(x)
            y.append(score)
        
        X = np.array(X)
        y = np.array(y)
        
        # Simple acquisition: explore areas with high uncertainty
        # Sample candidates
        candidates = []
        for _ in range(100):
            candidate = self._sample_random_params(params_to_optimize)
            
            # Calculate distance to observed points
            x_cand = [candidate.get(p, 0) for p in params_to_optimize]
            distances = np.linalg.norm(X - x_cand, axis=1)
            min_distance = np.min(distances)
            
            # Prefer points far from observations (exploration)
            exploration_score = min_distance
            
            # Also consider exploitation (near good points)
            best_idx = np.argmax(y)
            exploitation_score = -distances[best_idx]
            
            # Balance exploration and exploitation
            acquisition_score = 0.5 * exploration_score + 0.5 * exploitation_score
            
            candidates.append((candidate, acquisition_score))
        
        # Select best candidate
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    async def _evaluate_params(self, prompt_id: int, test_case_ids: List[int],
                              base_model_config: Dict[str, Any],
                              params: Dict[str, Any]) -> float:
        """Evaluate parameters on test cases."""
        # Merge parameters
        model_config = base_model_config.copy()
        model_config["parameters"] = params
        
        # Run evaluation
        scores = []
        for test_case_id in test_case_ids[:5]:  # Limit for speed
            result = await self.test_runner.run_single_test(
                prompt_id=prompt_id,
                test_case_id=test_case_id,
                model_config=model_config
            )
            
            if result.get("success") and "scores" in result:
                scores.append(result["scores"].get("aggregate_score", 0))
        
        return np.mean(scores) if scores else 0.0
    
    def _has_converged(self, observations: List[Tuple[Dict, float]]) -> bool:
        """Check if optimization has converged."""
        if len(observations) < 10:
            return False
        
        # Check if recent scores are stable
        recent_scores = [score for _, score in observations[-5:]]
        return np.std(recent_scores) < 0.01

########################################################################################################################
# Iterative Refinement Optimizer

class IterativeRefinementOptimizer:
    """
    Iteratively refine prompts based on error analysis.
    """
    
    def __init__(self, db: PromptStudioDatabase, test_runner: TestRunner):
        """Initialize iterative refinement optimizer."""
        self.db = db
        self.test_runner = test_runner
        self.executor = PromptExecutor(db)
    
    async def optimize(self, prompt_id: int, test_case_ids: List[int],
                       model_config: Dict[str, Any],
                       max_iterations: int = 10) -> Dict[str, Any]:
        """
        Iteratively refine prompt based on errors.
        
        Args:
            prompt_id: Initial prompt ID
            test_case_ids: Test cases
            model_config: Model configuration
            max_iterations: Maximum refinement iterations
            
        Returns:
            Optimization results
        """
        logger.info(f"Starting iterative refinement for prompt {prompt_id}")
        
        current_prompt_id = prompt_id
        iteration_history = []
        
        for iteration in range(max_iterations):
            logger.info(f"Refinement iteration {iteration + 1}")
            
            # Run evaluation
            test_runs = []
            for test_case_id in test_case_ids:
                result = await self.test_runner.run_single_test(
                    prompt_id=current_prompt_id,
                    test_case_id=test_case_id,
                    model_config=model_config
                )
                test_runs.append(result)
            
            # Analyze errors
            errors = self._analyze_errors(test_runs)
            
            if not errors:
                logger.info("No errors found, optimization complete")
                break
            
            # Generate refinement based on errors
            refinement = await self._generate_refinement(current_prompt_id, errors)
            
            if not refinement:
                logger.warning("Could not generate refinement")
                break
            
            # Create refined prompt
            new_prompt_id = await self._create_refined_prompt(
                current_prompt_id, refinement
            )
            
            # Evaluate refined prompt
            new_score = await self._evaluate_prompt(
                new_prompt_id, test_case_ids, model_config
            )
            
            iteration_history.append({
                "iteration": iteration + 1,
                "prompt_id": new_prompt_id,
                "score": new_score,
                "errors_addressed": len(errors),
                "refinement": refinement
            })
            
            current_prompt_id = new_prompt_id
            
            # Check if errors are decreasing
            if len(errors) <= 2:
                logger.info("Few errors remaining, stopping refinement")
                break
        
        # Calculate final improvement
        initial_score = await self._evaluate_prompt(prompt_id, test_case_ids, model_config)
        final_score = iteration_history[-1]["score"] if iteration_history else initial_score
        
        return {
            "initial_prompt_id": prompt_id,
            "optimized_prompt_id": current_prompt_id,
            "initial_score": initial_score,
            "final_score": final_score,
            "improvement": final_score - initial_score,
            "iterations": len(iteration_history),
            "iteration_history": iteration_history
        }
    
    def _analyze_errors(self, test_runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze test runs to identify error patterns."""
        errors = []
        
        for run in test_runs:
            if not run.get("success") or run.get("scores", {}).get("aggregate_score", 0) < 0.8:
                error_info = {
                    "test_case": run.get("test_case_name"),
                    "input": run.get("inputs"),
                    "expected": run.get("expected_outputs"),
                    "actual": run.get("actual_output"),
                    "score": run.get("scores", {}).get("aggregate_score", 0),
                    "error": run.get("error")
                }
                errors.append(error_info)
        
        return errors
    
    async def _generate_refinement(self, prompt_id: int, 
                                  errors: List[Dict[str, Any]]) -> Optional[str]:
        """Generate refinement based on error analysis."""
        # Get current prompt
        prompt = self._get_prompt(prompt_id)
        
        # Build error summary
        error_summary = "Common errors found:\n"
        for i, error in enumerate(errors[:5], 1):  # Limit to 5 errors
            error_summary += f"{i}. Input: {error['input']}\n"
            error_summary += f"   Expected: {error['expected']}\n"
            error_summary += f"   Got: {error['actual']}\n\n"
        
        # Generate refinement
        refinement_prompt = f"""Analyze these errors and suggest how to refine the prompt to address them.

Current prompt:
{prompt.get('content', '')}

{error_summary}

Suggest specific refinements to fix these errors:"""
        
        try:
            result = await self.executor._call_llm(
                provider="openai",
                model="gpt-3.5-turbo",
                prompt=refinement_prompt,
                parameters={"temperature": 0.7, "max_tokens": 500}
            )
            return result["content"].strip()
        except:
            return None
    
    async def _create_refined_prompt(self, base_prompt_id: int,
                                    refinement: str) -> int:
        """Create refined version of prompt."""
        # Get base prompt
        prompt = self._get_prompt(base_prompt_id)
        
        # Apply refinement
        refined_content = f"{prompt['content']}\n\n{refinement}"
        
        # Create new prompt
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO prompt_studio_prompts (
                uuid, project_id, signature_id, name, content,
                system_prompt, version, parent_version_id, client_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            f"refined-{datetime.utcnow().timestamp()}",
            prompt["project_id"],
            prompt.get("signature_id"),
            f"{prompt['name']} (Refined)",
            refined_content,
            prompt.get("system_prompt"),
            prompt["version"] + 1,
            base_prompt_id,
            self.db.client_id
        ))
        
        new_prompt_id = cursor.lastrowid
        conn.commit()
        
        return new_prompt_id
    
    async def _evaluate_prompt(self, prompt_id: int, test_case_ids: List[int],
                              model_config: Dict[str, Any]) -> float:
        """Evaluate prompt and return average score."""
        scores = []
        
        for test_case_id in test_case_ids:
            result = await self.test_runner.run_single_test(
                prompt_id=prompt_id,
                test_case_id=test_case_id,
                model_config=model_config
            )
            
            if result.get("success") and "scores" in result:
                scores.append(result["scores"].get("aggregate_score", 0))
        
        return np.mean(scores) if scores else 0.0
    
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
# Genetic Algorithm Optimizer

class GeneticOptimizer:
    """
    Use genetic algorithm to evolve prompts.
    """
    
    def __init__(self, db: PromptStudioDatabase, test_runner: TestRunner):
        """Initialize genetic optimizer."""
        self.db = db
        self.test_runner = test_runner
        self.executor = PromptExecutor(db)
    
    async def optimize(self, prompt_id: int, test_case_ids: List[int],
                       model_config: Dict[str, Any],
                       population_size: int = 10,
                       generations: int = 10,
                       mutation_rate: float = 0.1) -> Dict[str, Any]:
        """
        Optimize prompt using genetic algorithm.
        
        Args:
            prompt_id: Initial prompt ID
            test_case_ids: Test cases
            model_config: Model configuration
            population_size: Population size
            generations: Number of generations
            mutation_rate: Mutation probability
            
        Returns:
            Optimization results
        """
        logger.info(f"Starting genetic optimization for prompt {prompt_id}")
        
        # Initialize population
        population = await self._initialize_population(prompt_id, population_size)
        
        best_individual = None
        best_score = -1
        generation_history = []
        
        for generation in range(generations):
            logger.info(f"Generation {generation + 1}/{generations}")
            
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                score = await self._evaluate_fitness(
                    individual, test_case_ids, model_config
                )
                fitness_scores.append((individual, score))
            
            # Sort by fitness
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Track best
            if fitness_scores[0][1] > best_score:
                best_score = fitness_scores[0][1]
                best_individual = fitness_scores[0][0]
            
            generation_history.append({
                "generation": generation + 1,
                "best_score": fitness_scores[0][1],
                "avg_score": np.mean([s for _, s in fitness_scores]),
                "population_size": len(population)
            })
            
            # Check convergence
            if generation > 5 and self._has_converged_genetic(generation_history):
                logger.info("Population converged")
                break
            
            # Selection and reproduction
            new_population = []
            
            # Elitism: keep top 2
            new_population.extend([ind for ind, _ in fitness_scores[:2]])
            
            # Crossover and mutation
            while len(new_population) < population_size:
                # Tournament selection
                parent1 = self._tournament_select(fitness_scores)
                parent2 = self._tournament_select(fitness_scores)
                
                # Crossover
                child = await self._crossover(parent1, parent2)
                
                # Mutation
                if random.random() < mutation_rate:
                    child = await self._mutate(child)
                
                new_population.append(child)
            
            population = new_population
        
        # Create final prompt from best individual
        final_prompt_id = await self._individual_to_prompt(best_individual, prompt_id)
        
        return {
            "initial_prompt_id": prompt_id,
            "optimized_prompt_id": final_prompt_id,
            "best_score": best_score,
            "generations": len(generation_history),
            "generation_history": generation_history
        }
    
    async def _initialize_population(self, prompt_id: int, 
                                    size: int) -> List[Dict[str, Any]]:
        """Initialize population with variations of base prompt."""
        prompt = self._get_prompt(prompt_id)
        population = []
        
        # Add original
        population.append({
            "content": prompt["content"],
            "system_prompt": prompt.get("system_prompt", "")
        })
        
        # Generate variations
        for _ in range(size - 1):
            variation = await self._generate_variation(prompt)
            population.append(variation)
        
        return population
    
    async def _generate_variation(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a variation of the prompt."""
        variation_prompt = f"""Create a variation of this prompt that maintains the same goal but uses different wording or structure:

{prompt['content']}

Variation:"""
        
        try:
            result = await self.executor._call_llm(
                provider="openai",
                model="gpt-3.5-turbo",
                prompt=variation_prompt,
                parameters={"temperature": 0.9, "max_tokens": 500}
            )
            
            return {
                "content": result["content"].strip(),
                "system_prompt": prompt.get("system_prompt", "")
            }
        except:
            # Fallback to original with minor changes
            return {
                "content": prompt["content"] + "\nBe precise and clear.",
                "system_prompt": prompt.get("system_prompt", "")
            }
    
    async def _evaluate_fitness(self, individual: Dict[str, Any],
                               test_case_ids: List[int],
                               model_config: Dict[str, Any]) -> float:
        """Evaluate fitness of an individual."""
        # Create temporary prompt
        prompt_id = await self._individual_to_prompt(individual, None)
        
        # Evaluate
        scores = []
        for test_case_id in test_case_ids[:3]:  # Limit for speed
            result = await self.test_runner.run_single_test(
                prompt_id=prompt_id,
                test_case_id=test_case_id,
                model_config=model_config
            )
            
            if result.get("success") and "scores" in result:
                scores.append(result["scores"].get("aggregate_score", 0))
        
        return np.mean(scores) if scores else 0.0
    
    def _tournament_select(self, fitness_scores: List[Tuple[Dict, float]],
                          tournament_size: int = 3) -> Dict[str, Any]:
        """Tournament selection."""
        tournament = random.sample(fitness_scores, min(tournament_size, len(fitness_scores)))
        tournament.sort(key=lambda x: x[1], reverse=True)
        return tournament[0][0]
    
    async def _crossover(self, parent1: Dict[str, Any],
                        parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Crossover two individuals."""
        # Simple approach: combine parts of prompts
        crossover_prompt = f"""Combine the best aspects of these two prompts into a new one:

Prompt 1:
{parent1['content']}

Prompt 2:
{parent2['content']}

Combined prompt:"""
        
        try:
            result = await self.executor._call_llm(
                provider="openai",
                model="gpt-3.5-turbo",
                prompt=crossover_prompt,
                parameters={"temperature": 0.7, "max_tokens": 500}
            )
            
            return {
                "content": result["content"].strip(),
                "system_prompt": parent1.get("system_prompt", "")
            }
        except:
            # Fallback: return parent1
            return parent1.copy()
    
    async def _mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate an individual."""
        mutation_prompt = f"""Make a small random change to this prompt while keeping its core purpose:

{individual['content']}

Mutated prompt:"""
        
        try:
            result = await self.executor._call_llm(
                provider="openai",
                model="gpt-3.5-turbo",
                prompt=mutation_prompt,
                parameters={"temperature": 1.0, "max_tokens": 500}
            )
            
            return {
                "content": result["content"].strip(),
                "system_prompt": individual.get("system_prompt", "")
            }
        except:
            # Fallback: add random instruction
            mutations = [
                "\nBe concise.",
                "\nProvide detailed explanations.",
                "\nFocus on accuracy.",
                "\nConsider edge cases."
            ]
            
            return {
                "content": individual["content"] + random.choice(mutations),
                "system_prompt": individual.get("system_prompt", "")
            }
    
    def _has_converged_genetic(self, history: List[Dict[str, Any]]) -> bool:
        """Check if population has converged."""
        if len(history) < 5:
            return False
        
        # Check if best scores are stable
        recent_best = [h["best_score"] for h in history[-5:]]
        return np.std(recent_best) < 0.001
    
    async def _individual_to_prompt(self, individual: Dict[str, Any],
                                   base_prompt_id: Optional[int]) -> int:
        """Convert individual to prompt in database."""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        # Get project ID if base prompt provided
        project_id = 1  # Default
        if base_prompt_id:
            cursor.execute(
                "SELECT project_id FROM prompt_studio_prompts WHERE id = ?",
                (base_prompt_id,)
            )
            row = cursor.fetchone()
            if row:
                project_id = row[0]
        
        # Create prompt
        cursor.execute("""
            INSERT INTO prompt_studio_prompts (
                uuid, project_id, name, content, system_prompt,
                version, parent_version_id, client_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            f"genetic-{datetime.utcnow().timestamp()}",
            project_id,
            "Genetic Prompt",
            individual["content"],
            individual.get("system_prompt"),
            1,
            base_prompt_id,
            self.db.client_id
        ))
        
        prompt_id = cursor.lastrowid
        conn.commit()
        
        return prompt_id
    
    def _get_prompt(self, prompt_id: int) -> Dict[str, Any]:
        """Get prompt from database."""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM prompt_studio_prompts WHERE id = ?", (prompt_id,))
        row = cursor.fetchone()
        
        if row:
            return self.db._row_to_dict(cursor, row)
        return {}