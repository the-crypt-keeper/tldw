# bootstrap_manager.py
# Bootstrap and example management for Prompt Studio optimization

import json
import random
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict
from loguru import logger

from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import (
    PromptStudioDatabase, DatabaseError
)

########################################################################################################################
# Bootstrap Manager Class

class BootstrapManager:
    """Manages bootstrap examples and traces for prompt optimization."""
    
    def __init__(self, db: PromptStudioDatabase):
        """
        Initialize BootstrapManager.
        
        Args:
            db: PromptStudioDatabase instance
        """
        self.db = db
        self.client_id = db.client_id
    
    ####################################################################################################################
    # Trace Collection
    
    def collect_trace(self, prompt_id: int, test_case_id: int,
                      inputs: Dict[str, Any], outputs: Dict[str, Any],
                      score: float, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Collect an execution trace for bootstrapping.
        
        Args:
            prompt_id: ID of the prompt used
            test_case_id: ID of the test case
            inputs: Input data
            outputs: Output data
            score: Quality score (0-1)
            metadata: Additional metadata
            
        Returns:
            Trace ID
        """
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            # Store as a test run with bootstrap metadata
            cursor.execute("""
                INSERT INTO prompt_studio_test_runs (
                    uuid, project_id, prompt_id, test_case_id,
                    model_name, inputs, outputs, scores,
                    client_id
                ) VALUES (
                    lower(hex(randomblob(16))),
                    (SELECT project_id FROM prompt_studio_prompts WHERE id = ?),
                    ?, ?, 'bootstrap', ?, ?, ?, ?
                )
            """, (
                prompt_id, prompt_id, test_case_id,
                json.dumps(inputs), json.dumps(outputs),
                json.dumps({"quality": score, "metadata": metadata}),
                self.client_id
            ))
            
            trace_id = cursor.lastrowid
            conn.commit()
            
            logger.debug(f"Collected trace {trace_id} for prompt {prompt_id}")
            return trace_id
            
        except Exception as e:
            logger.error(f"Failed to collect trace: {e}")
            raise DatabaseError(f"Failed to collect trace: {e}")
    
    def get_traces(self, prompt_id: int, min_score: float = 0.7,
                  limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get high-quality traces for a prompt.
        
        Args:
            prompt_id: Prompt ID
            min_score: Minimum quality score
            limit: Maximum number of traces
            
        Returns:
            List of trace data
        """
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, test_case_id, inputs, outputs, scores
                FROM prompt_studio_test_runs
                WHERE prompt_id = ? 
                AND model_name = 'bootstrap'
                AND json_extract(scores, '$.quality') >= ?
                ORDER BY json_extract(scores, '$.quality') DESC
                LIMIT ?
            """, (prompt_id, min_score, limit))
            
            traces = []
            for row in cursor.fetchall():
                traces.append({
                    "id": row[0],
                    "test_case_id": row[1],
                    "inputs": json.loads(row[2]),
                    "outputs": json.loads(row[3]),
                    "score": json.loads(row[4]).get("quality", 0)
                })
            
            return traces
            
        except Exception as e:
            logger.error(f"Failed to get traces: {e}")
            raise DatabaseError(f"Failed to get traces: {e}")
    
    ####################################################################################################################
    # Example Selection
    
    def select_bootstrap_examples(self, prompt_id: int, n_examples: int = 5,
                                 strategy: str = "diverse") -> List[Dict[str, Any]]:
        """
        Select bootstrap examples for few-shot prompting.
        
        Args:
            prompt_id: Prompt ID
            n_examples: Number of examples to select
            strategy: Selection strategy ('diverse', 'top', 'random')
            
        Returns:
            Selected examples
        """
        traces = self.get_traces(prompt_id)
        
        if not traces:
            return []
        
        if strategy == "top":
            # Select top-scoring examples
            return traces[:n_examples]
        
        elif strategy == "random":
            # Random selection
            return random.sample(traces, min(n_examples, len(traces)))
        
        elif strategy == "diverse":
            # Select diverse examples based on input/output characteristics
            return self._select_diverse_examples(traces, n_examples)
        
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")
    
    def _select_diverse_examples(self, traces: List[Dict[str, Any]], 
                                n_examples: int) -> List[Dict[str, Any]]:
        """
        Select diverse examples using clustering.
        
        Args:
            traces: All available traces
            n_examples: Number to select
            
        Returns:
            Diverse selection of examples
        """
        if len(traces) <= n_examples:
            return traces
        
        # Simple diversity selection based on input/output characteristics
        selected = []
        remaining = traces.copy()
        
        # Start with highest scoring
        selected.append(remaining.pop(0))
        
        # Select examples that are most different from already selected
        while len(selected) < n_examples and remaining:
            max_min_distance = -1
            most_diverse_idx = 0
            
            for i, candidate in enumerate(remaining):
                # Calculate minimum distance to selected examples
                min_distance = float('inf')
                for selected_example in selected:
                    distance = self._calculate_example_distance(
                        candidate, selected_example
                    )
                    min_distance = min(min_distance, distance)
                
                # Track example with maximum minimum distance
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    most_diverse_idx = i
            
            selected.append(remaining.pop(most_diverse_idx))
        
        return selected
    
    def _calculate_example_distance(self, ex1: Dict[str, Any], 
                                   ex2: Dict[str, Any]) -> float:
        """
        Calculate distance between two examples.
        
        Args:
            ex1: First example
            ex2: Second example
            
        Returns:
            Distance score
        """
        # Simple distance based on string similarity
        # Could be enhanced with embeddings
        
        input1_str = json.dumps(ex1["inputs"], sort_keys=True)
        input2_str = json.dumps(ex2["inputs"], sort_keys=True)
        
        output1_str = json.dumps(ex1["outputs"], sort_keys=True)
        output2_str = json.dumps(ex2["outputs"], sort_keys=True)
        
        # Jaccard distance for simple similarity
        def jaccard_distance(s1: str, s2: str) -> float:
            set1 = set(s1.split())
            set2 = set(s2.split())
            intersection = set1.intersection(set2)
            union = set1.union(set2)
            if not union:
                return 1.0
            return 1.0 - len(intersection) / len(union)
        
        input_distance = jaccard_distance(input1_str, input2_str)
        output_distance = jaccard_distance(output1_str, output2_str)
        
        return (input_distance + output_distance) / 2
    
    ####################################################################################################################
    # Bootstrap Optimization
    
    def create_bootstrapped_prompt(self, prompt_id: int, n_examples: int = 5,
                                  selection_strategy: str = "diverse") -> Dict[str, Any]:
        """
        Create a bootstrapped version of a prompt with examples.
        
        Args:
            prompt_id: Original prompt ID
            n_examples: Number of examples to include
            selection_strategy: Example selection strategy
            
        Returns:
            New prompt with bootstrap examples
        """
        try:
            # Get original prompt
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT project_id, signature_id, name, system_prompt, user_prompt
                FROM prompt_studio_prompts
                WHERE id = ? AND deleted = 0
            """, (prompt_id,))
            
            prompt_data = cursor.fetchone()
            if not prompt_data:
                raise ValueError(f"Prompt {prompt_id} not found")
            
            # Select bootstrap examples
            examples = self.select_bootstrap_examples(
                prompt_id, n_examples, selection_strategy
            )
            
            # Format examples for few-shot prompt
            formatted_examples = self._format_examples_for_prompt(examples)
            
            # Create new bootstrapped prompt
            cursor.execute("""
                INSERT INTO prompt_studio_prompts (
                    uuid, project_id, signature_id, name,
                    system_prompt, user_prompt, few_shot_examples,
                    parent_version_id, change_description,
                    version_number, client_id
                ) VALUES (
                    lower(hex(randomblob(16))), ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    (SELECT COALESCE(MAX(version_number), 0) + 1 
                     FROM prompt_studio_prompts 
                     WHERE project_id = ? AND name = ?),
                    ?
                )
            """, (
                prompt_data[0], prompt_data[1],
                f"{prompt_data[2]} (Bootstrapped)",
                prompt_data[3], prompt_data[4],
                json.dumps(formatted_examples),
                prompt_id,
                f"Bootstrapped with {len(examples)} examples using {selection_strategy} strategy",
                prompt_data[0], prompt_data[2],
                self.client_id
            ))
            
            new_prompt_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Created bootstrapped prompt {new_prompt_id} from {prompt_id}")
            
            return {
                "id": new_prompt_id,
                "parent_id": prompt_id,
                "n_examples": len(examples),
                "selection_strategy": selection_strategy,
                "examples": formatted_examples
            }
            
        except Exception as e:
            logger.error(f"Failed to create bootstrapped prompt: {e}")
            raise DatabaseError(f"Failed to create bootstrapped prompt: {e}")
    
    def _format_examples_for_prompt(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format examples for inclusion in a few-shot prompt.
        
        Args:
            examples: Raw example traces
            
        Returns:
            Formatted examples
        """
        formatted = []
        
        for i, example in enumerate(examples, 1):
            formatted.append({
                "index": i,
                "input": example["inputs"],
                "output": example["outputs"],
                "quality_score": example["score"]
            })
        
        return formatted
    
    ####################################################################################################################
    # Analysis Methods
    
    def analyze_bootstrap_performance(self, original_prompt_id: int,
                                     bootstrapped_prompt_id: int) -> Dict[str, Any]:
        """
        Analyze performance improvement from bootstrapping.
        
        Args:
            original_prompt_id: Original prompt ID
            bootstrapped_prompt_id: Bootstrapped prompt ID
            
        Returns:
            Performance analysis
        """
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            # Get test run statistics for both prompts
            def get_stats(prompt_id):
                cursor.execute("""
                    SELECT 
                        COUNT(*) as n_runs,
                        AVG(json_extract(scores, '$.quality')) as avg_score,
                        MIN(json_extract(scores, '$.quality')) as min_score,
                        MAX(json_extract(scores, '$.quality')) as max_score
                    FROM prompt_studio_test_runs
                    WHERE prompt_id = ? AND model_name != 'bootstrap'
                """, (prompt_id,))
                
                row = cursor.fetchone()
                return {
                    "n_runs": row[0] or 0,
                    "avg_score": row[1] or 0,
                    "min_score": row[2] or 0,
                    "max_score": row[3] or 0
                }
            
            original_stats = get_stats(original_prompt_id)
            bootstrapped_stats = get_stats(bootstrapped_prompt_id)
            
            # Calculate improvements
            improvement = {
                "avg_score_change": bootstrapped_stats["avg_score"] - original_stats["avg_score"],
                "avg_score_improvement_pct": (
                    ((bootstrapped_stats["avg_score"] - original_stats["avg_score"]) / 
                     max(original_stats["avg_score"], 0.01)) * 100
                    if original_stats["avg_score"] > 0 else 0
                ),
                "consistency_improvement": (
                    original_stats["max_score"] - original_stats["min_score"] -
                    (bootstrapped_stats["max_score"] - bootstrapped_stats["min_score"])
                )
            }
            
            return {
                "original": original_stats,
                "bootstrapped": bootstrapped_stats,
                "improvement": improvement
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze bootstrap performance: {e}")
            raise DatabaseError(f"Failed to analyze bootstrap performance: {e}")