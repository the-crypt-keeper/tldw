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
from tldw_Server_API.app.core.DB_Management.migrations import migrate_evaluations_database

from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze
from tldw_Server_API.app.core.config import load_comprehensive_config
import os


class EvaluationManager:
    """Manages evaluation operations and persistence"""
    
    def __init__(self):
        self.config = load_comprehensive_config()
        self.db_path = self._get_db_path()
        self._init_database()
    
    def _get_db_path(self) -> Path:
        """Get evaluation database path with security validation"""
        # Use the same path as the OpenAI-compatible evaluations DB
        if self.config and self.config.has_section("Database"):
            db_path = self.config.get("Database", "evaluations_db_path", fallback="Databases/evaluations.db")
        else:
            db_path = "Databases/evaluations.db"
        
        # Sanitize path to prevent directory traversal and null byte injection
        # Remove any directory traversal attempts
        db_path = db_path.replace("..", "")
        # Remove null bytes to prevent null byte injection attacks
        db_path = db_path.replace("\x00", "")
        db_path = os.path.normpath(db_path)
        
        # Make absolute if relative
        if not os.path.isabs(db_path):
            # Get the project root (4 levels up from this file)
            project_root = Path(__file__).parent.parent.parent.parent
            db_path = project_root / db_path
        else:
            db_path = Path(db_path)
        
        # Resolve to absolute path and check it's within project boundaries
        db_path = db_path.resolve()
        project_root = Path(__file__).parent.parent.parent.parent.resolve()
        
        # Ensure the path is within the project directory
        try:
            db_path.relative_to(project_root)
        except ValueError:
            # Path is outside project directory - use default safe path
            logger.warning(f"Attempted to use database path outside project: {db_path}")
            db_path = project_root / "Databases" / "evaluations.db"
        
        # Ensure directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return db_path
    
    def _init_database(self):
        """Initialize evaluation database using migration system"""
        try:
            # Apply all pending migrations
            migrate_evaluations_database(self.db_path)
            logger.info("Database migrations applied successfully")
        except Exception as e:
            # In production, database migration failures should be fatal
            # This ensures consistency and prevents silent data corruption
            error_msg = f"CRITICAL: Failed to apply database migrations to {self.db_path}: {e}"
            logger.critical(error_msg)
            
            # Check if we're in a production environment
            env = os.getenv('ENVIRONMENT', 'development').lower()
            
            if env in ['production', 'staging']:
                # Fail loudly in production/staging
                raise RuntimeError(error_msg) from e
            else:
                # In development, log a warning but continue with fallback
                logger.warning("Running in development mode - using fallback database initialization")
                self._init_database_fallback()
    
    def _init_database_fallback(self):
        """Fallback database initialization without migrations
        
        WARNING: This method should ONLY be used in development environments.
        Production deployments must use the migration system to ensure
        database schema consistency.
        """
        with sqlite3.connect(self.db_path) as conn:
            # Create basic tables if they don't exist
            # Use internal_evaluations table to avoid conflict with OpenAI evaluations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS internal_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    evaluation_id TEXT UNIQUE NOT NULL,
                    evaluation_type TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    input_data TEXT NOT NULL,
                    results TEXT NOT NULL,
                    metadata TEXT,
                    user_id TEXT,
                    status TEXT DEFAULT 'completed',
                    error_message TEXT,
                    completed_at TIMESTAMP,
                    embedding_provider TEXT,
                    embedding_model TEXT
                )
            """)
            
            # Create indexes
            for index_sql in [
                "CREATE INDEX IF NOT EXISTS idx_type ON internal_evaluations(evaluation_type)",
                "CREATE INDEX IF NOT EXISTS idx_created ON internal_evaluations(created_at)",
                "CREATE INDEX IF NOT EXISTS idx_user_id ON internal_evaluations(user_id)",
                "CREATE INDEX IF NOT EXISTS idx_status ON internal_evaluations(status)"
            ]:
                try:
                    conn.execute(index_sql)
                except sqlite3.OperationalError:
                    pass  # Index might already exist
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    evaluation_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    score REAL NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (evaluation_id) REFERENCES internal_evaluations(evaluation_id)
                )
            """)
            
            # Create indexes for metrics table
            for index_sql in [
                "CREATE INDEX IF NOT EXISTS idx_eval_id ON evaluation_metrics(evaluation_id)",
                "CREATE INDEX IF NOT EXISTS idx_metric ON evaluation_metrics(metric_name)",
                "CREATE INDEX IF NOT EXISTS idx_metric_created ON evaluation_metrics(created_at)"
            ]:
                try:
                    conn.execute(index_sql)
                except sqlite3.OperationalError:
                    pass
            
            # Create webhook registrations table (needed for webhook tests)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS webhook_registrations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    url TEXT NOT NULL,
                    secret TEXT NOT NULL,
                    events TEXT NOT NULL,
                    active BOOLEAN DEFAULT 1,
                    retry_count INTEGER DEFAULT 3,
                    timeout_seconds INTEGER DEFAULT 30,
                    total_deliveries INTEGER DEFAULT 0,
                    successful_deliveries INTEGER DEFAULT 0,
                    failed_deliveries INTEGER DEFAULT 0,
                    last_delivery_at TIMESTAMP,
                    last_error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, url)
                )
            """)
            
            # Create webhook deliveries table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS webhook_deliveries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    webhook_id INTEGER NOT NULL,
                    evaluation_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    signature TEXT NOT NULL,
                    status_code INTEGER,
                    response_body TEXT,
                    response_time_ms INTEGER,
                    delivered BOOLEAN DEFAULT 0,
                    retry_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    delivered_at TIMESTAMP,
                    next_retry_at TIMESTAMP,
                    FOREIGN KEY (webhook_id) REFERENCES webhook_registrations(id),
                    FOREIGN KEY (evaluation_id) REFERENCES internal_evaluations(evaluation_id)
                )
            """)
            
            # Create indexes for webhook tables
            for index_sql in [
                "CREATE INDEX IF NOT EXISTS idx_webhook_user ON webhook_registrations(user_id)",
                "CREATE INDEX IF NOT EXISTS idx_webhook_active ON webhook_registrations(active)",
                "CREATE INDEX IF NOT EXISTS idx_delivery_webhook ON webhook_deliveries(webhook_id)",
                "CREATE INDEX IF NOT EXISTS idx_delivery_eval ON webhook_deliveries(evaluation_id)",
                "CREATE INDEX IF NOT EXISTS idx_delivery_status ON webhook_deliveries(delivered)",
                "CREATE INDEX IF NOT EXISTS idx_delivery_retry ON webhook_deliveries(next_retry_at)"
            ]:
                try:
                    conn.execute(index_sql)
                except sqlite3.OperationalError:
                    pass  # Index might already exist
            
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
                INSERT INTO internal_evaluations (
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
        query = "SELECT * FROM internal_evaluations WHERE 1=1"
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
                    SELECT evaluation_id FROM internal_evaluations WHERE 1=1
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
                f"SELECT * FROM internal_evaluations WHERE evaluation_id IN ({placeholders})",
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
            
            # Parse response with strict validation
            import re
            import json as json_module
            
            # Try to parse as JSON first (most reliable)
            score = None
            explanation = response
            
            try:
                # Attempt to parse JSON response
                parsed = json_module.loads(response)
                if isinstance(parsed, dict):
                    if 'score' in parsed:
                        raw_score = parsed['score']
                        # Validate score is a number between 0 and 10
                        if isinstance(raw_score, (int, float)) and 0 <= raw_score <= 10:
                            score = float(raw_score) / 10.0
                    if 'explanation' in parsed:
                        explanation = str(parsed['explanation'])
            except (json_module.JSONDecodeError, ValueError):
                # Fallback to regex parsing with strict validation
                # Only accept scores that are clearly delimited (e.g., "Score: 8")
                score_patterns = [
                    r'[Ss]core[:\s]+(\d+(?:\.\d+)?)\s*(?:/\s*10)?',  # Score: 8 or Score: 8/10
                    r'(\d+(?:\.\d+)?)\s*(?:/\s*10)\s+points?',  # 8/10 points
                    r'^(\d+(?:\.\d+)?)\s*$'  # Just a number on its own line
                ]
                
                for pattern in score_patterns:
                    match = re.search(pattern, response, re.MULTILINE)
                    if match:
                        try:
                            raw_score = float(match.group(1))
                            # Validate range
                            if 0 <= raw_score <= 10:
                                score = raw_score / 10.0 if raw_score > 1 else raw_score
                                break
                        except (ValueError, IndexError):
                            continue
                
                # Extract explanation (everything after first score mention or first newline)
                if '\n' in response:
                    lines = response.split('\n')
                    # Skip lines that contain just the score
                    explanation_lines = [l for l in lines if not re.match(r'^\s*\d+(?:\.\d+)?\s*(?:/\s*10)?\s*$', l)]
                    explanation = '\n'.join(explanation_lines).strip()
            
            # Default to 0.5 if no valid score found
            if score is None:
                logger.warning(f"Could not parse valid score from response: {response[:100]}...")
                score = 0.5
            
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