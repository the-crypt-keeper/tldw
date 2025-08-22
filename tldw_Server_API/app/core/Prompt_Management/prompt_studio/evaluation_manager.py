# evaluation_manager.py
# Manages evaluation runs for prompt testing

import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from loguru import logger

from ....core.Chat.Chat_Functions import chat_api_call

class EvaluationManager:
    """Manages prompt evaluation runs and metrics calculation."""
    
    def __init__(self, db_manager):
        """
        Initialize evaluation manager.
        
        Args:
            db_manager: Database manager instance
        """
        self.db = db_manager
    
    def run_evaluation(
        self,
        prompt_id: int,
        test_case_ids: List[int],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Run evaluation for a prompt against test cases.
        
        Args:
            prompt_id: ID of the prompt to evaluate
            test_case_ids: List of test case IDs to run
            model: LLM model to use
            temperature: Temperature setting
            max_tokens: Maximum tokens for response
            
        Returns:
            Evaluation results with metrics
        """
        # Get prompt details
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, system_prompt, user_prompt, name, project_id
            FROM prompt_studio_prompts
            WHERE id = ? AND deleted = 0
        """, (prompt_id,))
        
        prompt = cursor.fetchone()
        if not prompt:
            raise ValueError(f"Prompt {prompt_id} not found")
        
        # Get test cases
        placeholders = ','.join('?' * len(test_case_ids))
        cursor.execute(f"""
            SELECT id, inputs, expected_outputs, name
            FROM prompt_studio_test_cases
            WHERE id IN ({placeholders}) AND deleted = 0
        """, test_case_ids)
        
        test_cases = cursor.fetchall()
        
        # Create evaluation record
        eval_uuid = str(uuid.uuid4())
        model_configs = json.dumps({
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens
        })
        cursor.execute("""
            INSERT INTO prompt_studio_evaluations (
                uuid, prompt_id, project_id, model_configs, status, 
                test_case_ids, started_at, client_id
            ) VALUES (?, ?, ?, ?, 'running', ?, CURRENT_TIMESTAMP, ?)
        """, (eval_uuid, prompt_id, prompt[4], model_configs, json.dumps(test_case_ids), self.db.client_id))
        
        eval_id = cursor.lastrowid
        conn.commit()
        
        # Run each test case
        results = []
        total_score = 0
        
        for test_case in test_cases:
            test_id = test_case[0]
            inputs = json.loads(test_case[1]) if test_case[1] else {}
            expected = json.loads(test_case[2]) if test_case[2] else {}
            
            # Format prompt with inputs
            formatted_user_prompt = prompt[2]
            for key, value in inputs.items():
                formatted_user_prompt = formatted_user_prompt.replace(f"{{{key}}}", str(value))
            
            # Call LLM
            try:
                response = chat_api_call(
                    api_endpoint="openai",
                    model=model,
                    messages=[
                        {"role": "system", "content": prompt[1]},
                        {"role": "user", "content": formatted_user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                actual_output = response[0] if response else ""
                
                # Simple scoring - exact match = 1.0, partial match = 0.5, no match = 0.0
                score = self._calculate_score(expected, {"response": actual_output})
                total_score += score
                
                results.append({
                    "test_case_id": test_id,
                    "inputs": inputs,
                    "expected": expected,
                    "actual": {"response": actual_output},
                    "score": score,
                    "passed": score >= 0.5
                })
                
            except Exception as e:
                logger.error(f"Error running test case {test_id}: {e}")
                results.append({
                    "test_case_id": test_id,
                    "inputs": inputs,
                    "expected": expected,
                    "actual": {"error": str(e)},
                    "score": 0.0,
                    "passed": False
                })
        
        # Calculate metrics
        avg_score = total_score / len(results) if results else 0.0
        passed_count = sum(1 for r in results if r["passed"])
        
        aggregate_metrics = {
            "average_score": avg_score,
            "total_tests": len(results),
            "passed": passed_count,
            "failed": len(results) - passed_count,
            "pass_rate": passed_count / len(results) if results else 0.0
        }
        
        # Update evaluation record
        cursor.execute("""
            UPDATE prompt_studio_evaluations
            SET status = 'completed',
                completed_at = CURRENT_TIMESTAMP,
                test_run_ids = ?,
                aggregate_metrics = ?
            WHERE id = ?
        """, (
            json.dumps([r["test_case_id"] for r in results]),
            json.dumps(aggregate_metrics),
            eval_id
        ))
        conn.commit()
        
        return {
            "id": eval_id,
            "uuid": eval_uuid,
            "project_id": prompt[4],
            "prompt_id": prompt_id,
            "model": model,
            "status": "completed",
            "results": results,
            "metrics": aggregate_metrics
        }
    
    def _calculate_score(self, expected: Dict, actual: Dict) -> float:
        """
        Calculate similarity score between expected and actual outputs.
        
        Args:
            expected: Expected output dictionary
            actual: Actual output dictionary
            
        Returns:
            Score between 0.0 and 1.0
        """
        if not expected:
            return 1.0  # No expected output means any output is valid
        
        # Simple implementation - can be enhanced with better similarity metrics
        expected_str = str(expected.get("response", "")).lower().strip()
        actual_str = str(actual.get("response", "")).lower().strip()
        
        if expected_str == actual_str:
            return 1.0
        elif expected_str in actual_str or actual_str in expected_str:
            return 0.5
        else:
            # Calculate word overlap
            expected_words = set(expected_str.split())
            actual_words = set(actual_str.split())
            
            if not expected_words:
                return 0.0
                
            overlap = len(expected_words & actual_words)
            return overlap / len(expected_words)
    
    def get_evaluation(self, eval_id: int) -> Optional[Dict[str, Any]]:
        """
        Get evaluation details by ID.
        
        Args:
            eval_id: Evaluation ID
            
        Returns:
            Evaluation record or None
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM prompt_studio_evaluations
            WHERE id = ?
        """, (eval_id,))
        
        row = cursor.fetchone()
        if row:
            return self.db._row_to_dict(cursor, row)
        return None
    
    def list_evaluations(
        self,
        project_id: Optional[int] = None,
        prompt_id: Optional[int] = None,
        status: Optional[str] = None,
        page: int = 1,
        per_page: int = 20
    ) -> Dict[str, Any]:
        """
        List evaluations with filtering.
        
        Args:
            project_id: Filter by project
            prompt_id: Filter by prompt
            status: Filter by status
            page: Page number
            per_page: Items per page
            
        Returns:
            Dictionary with evaluations and pagination
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        # Build query
        conditions = []
        params = []
        
        if project_id:
            conditions.append("p.project_id = ?")
            params.append(project_id)
        
        if prompt_id:
            conditions.append("e.prompt_id = ?")
            params.append(prompt_id)
        
        if status:
            conditions.append("e.status = ?")
            params.append(status)
        
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        
        # Count total
        count_query = f"""
            SELECT COUNT(*) 
            FROM prompt_studio_evaluations e
            LEFT JOIN prompt_studio_prompts p ON e.prompt_id = p.id
            {where_clause}
        """
        cursor.execute(count_query, params)
        total = cursor.fetchone()[0]
        
        # Get evaluations
        offset = (page - 1) * per_page
        query = f"""
            SELECT e.*, p.name as prompt_name
            FROM prompt_studio_evaluations e
            LEFT JOIN prompt_studio_prompts p ON e.prompt_id = p.id
            {where_clause}
            ORDER BY e.created_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([per_page, offset])
        
        cursor.execute(query, params)
        evaluations = []
        for row in cursor.fetchall():
            eval_dict = self.db._row_to_dict(cursor, row)
            evaluations.append(eval_dict)
        
        return {
            "evaluations": evaluations,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "total_pages": (total + per_page - 1) // per_page
            }
        }
    
    def compare_evaluations(self, eval_ids: List[int]) -> Dict[str, Any]:
        """
        Compare multiple evaluation runs.
        
        Args:
            eval_ids: List of evaluation IDs to compare
            
        Returns:
            Comparison results
        """
        evaluations = []
        for eval_id in eval_ids:
            eval_data = self.get_evaluation(eval_id)
            if eval_data:
                evaluations.append(eval_data)
        
        if not evaluations:
            return {"error": "No evaluations found"}
        
        # Compare metrics
        comparison = {
            "evaluations": evaluations,
            "metrics_comparison": {
                "average_scores": [
                    e.get("aggregate_metrics", {}).get("average_score", 0)
                    for e in evaluations
                ],
                "pass_rates": [
                    e.get("aggregate_metrics", {}).get("pass_rate", 0)
                    for e in evaluations
                ],
                "best_performer": max(
                    evaluations,
                    key=lambda e: e.get("aggregate_metrics", {}).get("average_score", 0)
                )["id"] if evaluations else None
            }
        }
        
        return comparison