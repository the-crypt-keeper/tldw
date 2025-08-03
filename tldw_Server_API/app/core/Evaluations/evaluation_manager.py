# evaluation_manager.py - Evaluation Management Module
"""
Central manager for evaluation operations.

Handles:
- Evaluation storage and retrieval
- History tracking
- Comparison operations
- Custom metric evaluation
- Statistical analysis
"""

import json
import sqlite3
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
from loguru import logger
import numpy as np

from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze
from tldw_Server_API.app.core.config import load_and_log_configs


class EvaluationManager:
    """Manages evaluation operations and persistence"""
    
    def __init__(self):
        self.config = load_and_log_configs()
        self.db_path = self._get_db_path()
        self._init_database()
    
    def _get_db_path(self) -> Path:
        """Get evaluation database path"""
        # FIXME: Update to use proper config structure once config API is standardized
        database_config = self.config.get('database', {})
        if isinstance(database_config, dict):
            base_path = Path(database_config.get('evaluation_db_path', 'user_databases'))
        else:
            base_path = Path('user_databases')
        db_file = base_path / "evaluations.db"
        db_file.parent.mkdir(parents=True, exist_ok=True)
        return db_file
    
    def _init_database(self):
        """Initialize evaluation database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    evaluation_id TEXT UNIQUE NOT NULL,
                    evaluation_type TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    input_data TEXT NOT NULL,
                    results TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            # Create indexes separately in SQLite
            conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON evaluations(evaluation_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON evaluations(created_at)")
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    evaluation_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    score REAL NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (evaluation_id) REFERENCES evaluations(evaluation_id)
                )
            """)
            
            # Create indexes for metrics table
            conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_id ON evaluation_metrics(evaluation_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metric ON evaluation_metrics(metric_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metric_created ON evaluation_metrics(created_at)")
            
            conn.commit()
    
    async def store_evaluation(
        self,
        evaluation_type: str,
        input_data: Dict[str, Any],
        results: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store evaluation results"""
        import uuid
        
        evaluation_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)
        
        # Store main evaluation record
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO evaluations (
                    evaluation_id, evaluation_type, created_at,
                    input_data, results, metadata
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                evaluation_id,
                evaluation_type,
                created_at,
                json.dumps(input_data),
                json.dumps(results),
                json.dumps(metadata or {})
            ))
            
            # Store individual metrics for easier querying
            if "metrics" in results:
                for metric_name, metric_data in results["metrics"].items():
                    score = metric_data.get("score", 0.0)
                    conn.execute("""
                        INSERT INTO evaluation_metrics (
                            evaluation_id, metric_name, score, created_at
                        ) VALUES (?, ?, ?, ?)
                    """, (evaluation_id, metric_name, score, created_at))
            
            conn.commit()
        
        logger.info(f"Stored evaluation {evaluation_id} of type {evaluation_type}")
        return evaluation_id
    
    async def get_history(
        self,
        evaluation_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Retrieve evaluation history with filtering"""
        query = "SELECT * FROM evaluations WHERE 1=1"
        params = []
        
        if evaluation_type and evaluation_type != "all":
            query += " AND evaluation_type = ?"
            params.append(evaluation_type)
        
        if start_date:
            query += " AND created_at >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND created_at <= ?"
            params.append(end_date)
        
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get total count
            count_query = query.replace("SELECT *", "SELECT COUNT(*)").split("LIMIT")[0]
            total_count = conn.execute(count_query, params[:-2]).fetchone()[0]
            
            # Get records
            cursor = conn.execute(query, params)
            items = []
            
            for row in cursor:
                item = dict(row)
                item["input_data"] = json.loads(item["input_data"])
                item["results"] = json.loads(item["results"])
                item["metadata"] = json.loads(item["metadata"] if item["metadata"] else "{}")
                items.append(item)
            
            # Calculate average scores
            avg_query = """
                SELECT metric_name, AVG(score) as avg_score
                FROM evaluation_metrics
                WHERE evaluation_id IN (
                    SELECT evaluation_id FROM evaluations WHERE 1=1
            """
            
            if evaluation_type and evaluation_type != "all":
                avg_query += " AND evaluation_type = ?"
            
            avg_query += ") GROUP BY metric_name"
            
            avg_cursor = conn.execute(avg_query, params[:1] if evaluation_type and evaluation_type != "all" else [])
            average_scores = {row[0]: row[1] for row in avg_cursor}
        
        # Calculate trends if we have enough data
        trends = None
        if len(items) > 10:
            trends = self._calculate_trends(items)
        
        return {
            "total_count": total_count,
            "items": items,
            "average_scores": average_scores,
            "trends": trends
        }
    
    async def compare_evaluations(
        self,
        evaluation_ids: List[str],
        metrics_to_compare: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare multiple evaluations"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get evaluations
            placeholders = ",".join("?" * len(evaluation_ids))
            cursor = conn.execute(
                f"SELECT * FROM evaluations WHERE evaluation_id IN ({placeholders})",
                evaluation_ids
            )
            
            evaluations = []
            for row in cursor:
                eval_dict = dict(row)
                eval_dict["results"] = json.loads(eval_dict["results"])
                evaluations.append(eval_dict)
            
            if len(evaluations) != len(evaluation_ids):
                raise ValueError("Some evaluation IDs not found")
            
            # Get metrics for comparison
            metric_cursor = conn.execute(
                f"""
                SELECT evaluation_id, metric_name, score
                FROM evaluation_metrics
                WHERE evaluation_id IN ({placeholders})
                """,
                evaluation_ids
            )
            
            metric_data = {}
            for row in metric_cursor:
                eval_id = row[0]
                metric_name = row[1]
                score = row[2]
                
                if metric_name not in metric_data:
                    metric_data[metric_name] = {}
                metric_data[metric_name][eval_id] = score
        
        # Filter metrics if specified
        if metrics_to_compare:
            metric_data = {k: v for k, v in metric_data.items() if k in metrics_to_compare}
        
        # Format comparison data
        metric_comparisons = {}
        best_performing = {}
        
        for metric_name, scores in metric_data.items():
            # Order scores by evaluation ID order
            ordered_scores = [scores.get(eval_id, 0.0) for eval_id in evaluation_ids]
            metric_comparisons[metric_name] = ordered_scores
            
            # Find best performing
            best_eval_id = max(scores.items(), key=lambda x: x[1])[0]
            best_performing[metric_name] = best_eval_id
        
        # Generate comparison summary
        summary_parts = []
        for metric, best_id in best_performing.items():
            summary_parts.append(f"{metric}: {best_id} performs best")
        
        comparison_summary = "Comparison Results:\n" + "\n".join(summary_parts)
        
        # Statistical analysis
        statistical_analysis = None
        if len(evaluation_ids) > 2:
            statistical_analysis = self._perform_statistical_analysis(metric_comparisons)
        
        return {
            "comparison_summary": comparison_summary,
            "metric_comparisons": metric_comparisons,
            "best_performing": best_performing,
            "statistical_analysis": statistical_analysis
        }
    
    async def evaluate_custom_metric(
        self,
        metric_name: str,
        description: str,
        evaluation_prompt: str,
        input_data: Dict[str, Any],
        scoring_criteria: Dict[str, Any],
        api_name: str = "openai"
    ) -> Dict[str, Any]:
        """Evaluate using custom metric definition"""
        # Format the evaluation prompt with input data
        formatted_prompt = evaluation_prompt
        for key, value in input_data.items():
            formatted_prompt = formatted_prompt.replace(f"{{{key}}}", str(value))
        
        # Add scoring criteria to prompt
        criteria_text = "\n".join([f"- {k}: {v}" for k, v in scoring_criteria.items()])
        full_prompt = f"{formatted_prompt}\n\nScoring Criteria:\n{criteria_text}\n\nProvide a score from 1-10 and explanation."
        
        try:
            # Get evaluation from LLM
            response = await asyncio.to_thread(
                analyze,
                json.dumps(input_data),
                full_prompt,
                api_name,
                "",
                temp=0.3,
                system_message="You are an expert evaluator. Provide scores and detailed explanations."
            )
            
            # Parse response
            import re
            score_match = re.search(r'\b(\d+)\b', response)
            score = float(score_match.group(1)) / 10.0 if score_match else 0.5
            
            # Extract explanation
            explanation = response.split('\n', 1)[1] if '\n' in response else response
            
            return {
                "metric_name": metric_name,
                "score": score,
                "explanation": explanation.strip(),
                "raw_output": response
            }
            
        except Exception as e:
            logger.error(f"Custom metric evaluation failed: {e}")
            return {
                "metric_name": metric_name,
                "score": 0.0,
                "explanation": f"Evaluation failed: {str(e)}",
                "raw_output": None
            }
    
    def _calculate_trends(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metric trends over time"""
        trends = {}
        
        # Group by metric
        metric_scores = {}
        for item in items:
            if "metrics" in item["results"]:
                for metric_name, metric_data in item["results"]["metrics"].items():
                    if metric_name not in metric_scores:
                        metric_scores[metric_name] = []
                    metric_scores[metric_name].append({
                        "timestamp": item["created_at"],
                        "score": metric_data.get("score", 0.0)
                    })
        
        # Calculate trends
        for metric_name, scores in metric_scores.items():
            if len(scores) > 1:
                # Sort by timestamp
                scores.sort(key=lambda x: x["timestamp"])
                
                # Calculate simple linear trend
                values = [s["score"] for s in scores]
                x = np.arange(len(values))
                
                # Linear regression
                coeffs = np.polyfit(x, values, 1)
                trend_direction = "improving" if coeffs[0] > 0 else "declining" if coeffs[0] < 0 else "stable"
                
                trends[metric_name] = {
                    "direction": trend_direction,
                    "slope": float(coeffs[0]),
                    "recent_average": np.mean(values[-5:]) if len(values) >= 5 else np.mean(values),
                    "overall_average": np.mean(values)
                }
        
        return trends
    
    def _perform_statistical_analysis(self, metric_comparisons: Dict[str, List[float]]) -> Dict[str, Any]:
        """Perform statistical analysis on comparison data"""
        analysis = {}
        
        for metric_name, scores in metric_comparisons.items():
            if len(scores) > 1:
                analysis[metric_name] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                    "range": float(np.max(scores) - np.min(scores)),
                    "cv": float(np.std(scores) / np.mean(scores)) if np.mean(scores) > 0 else 0
                }
        
        return analysis