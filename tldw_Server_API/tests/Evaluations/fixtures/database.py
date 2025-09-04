"""
Database fixtures for testing.

Provides utilities for creating and managing test databases with realistic data.
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import uuid


class TestDatabaseHelper:
    """Helper class for managing test databases."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def seed_evaluations(self, count: int = 10) -> List[str]:
        """Seed the database with test evaluations."""
        eval_ids = []
        
        with sqlite3.connect(self.db_path) as conn:
            for i in range(count):
                eval_id = f"eval_{uuid.uuid4().hex[:8]}"
                eval_ids.append(eval_id)
                
                conn.execute("""
                    INSERT INTO evaluations (
                        id, name, description, eval_type, eval_spec, 
                        dataset_id, created_at, created_by, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    eval_id,
                    f"test_evaluation_{i}",
                    f"Test evaluation {i} for testing",
                    "model_graded" if i % 2 == 0 else "g_eval",
                    json.dumps({
                        "evaluator_model": "gpt-4",
                        "metrics": ["accuracy", "relevance"],
                        "threshold": 0.7 + (i * 0.01)
                    }),
                    f"dataset_{i}",
                    (datetime.utcnow() - timedelta(days=i)).isoformat(),
                    f"test_user_{i % 3}",
                    json.dumps({"test": True, "index": i})
                ))
            
            conn.commit()
        
        return eval_ids
    
    def seed_runs(self, eval_ids: List[str], runs_per_eval: int = 3) -> List[str]:
        """Seed the database with test runs."""
        run_ids = []
        
        with sqlite3.connect(self.db_path) as conn:
            for eval_id in eval_ids:
                for i in range(runs_per_eval):
                    run_id = f"run_{uuid.uuid4().hex[:8]}"
                    run_ids.append(run_id)
                    
                    status = ["pending", "running", "completed", "failed"][i % 4]
                    
                    conn.execute("""
                        INSERT INTO evaluation_runs (
                            id, eval_id, status, target_model, config,
                            results, started_at, completed_at,
                            error_message
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        run_id,
                        eval_id,
                        status,
                        "gpt-4",
                        json.dumps({"temperature": 0.7, "max_tokens": 1000}),
                        json.dumps({"scores": [0.8, 0.9, 0.85]}) if status == "completed" else None,
                        datetime.utcnow().isoformat(),
                        datetime.utcnow().isoformat() if status in ["completed", "failed"] else None,
                        "Test error" if status == "failed" else None
                    ))
            
            conn.commit()
        
        return run_ids
    
    def seed_datasets(self, count: int = 5) -> List[str]:
        """Seed the database with test datasets."""
        dataset_ids = []
        
        with sqlite3.connect(self.db_path) as conn:
            for i in range(count):
                dataset_id = f"dataset_{uuid.uuid4().hex[:8]}"
                dataset_ids.append(dataset_id)
                
                samples = [
                    {
                        "input": {"question": f"Test question {j}"},
                        "expected": {"answer": f"Test answer {j}"},
                        "context": f"Test context {j}"
                    }
                    for j in range(5)
                ]
                
                conn.execute("""
                    INSERT INTO datasets (
                        id, name, description, samples,
                        created_at, created_by, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    dataset_id,
                    f"test_dataset_{i}",
                    f"Test dataset {i}",
                    json.dumps(samples),
                    datetime.utcnow().isoformat(),
                    f"test_user_{i % 2}",
                    json.dumps({"test": True, "size": len(samples)})
                ))
            
            conn.commit()
        
        return dataset_ids
    
    def seed_internal_evaluations(self, count: int = 10) -> List[str]:
        """Seed internal evaluations table."""
        eval_ids = []
        
        with sqlite3.connect(self.db_path) as conn:
            for i in range(count):
                eval_id = f"internal_{uuid.uuid4().hex[:8]}"
                eval_ids.append(eval_id)
                
                conn.execute("""
                    INSERT INTO internal_evaluations (
                        evaluation_id, evaluation_type, created_at,
                        input_data, results, metadata, user_id,
                        status, completed_at, embedding_provider, embedding_model
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    eval_id,
                    ["g_eval", "rag", "response_quality"][i % 3],
                    datetime.utcnow().isoformat(),
                    json.dumps({"text": f"Test input {i}"}),
                    json.dumps({"score": 0.8 + (i * 0.01)}),
                    json.dumps({"test": True}),
                    f"user_{i % 3}",
                    "completed",
                    datetime.utcnow().isoformat(),
                    "openai" if i % 2 == 0 else None,
                    "text-embedding-3-small" if i % 2 == 0 else None
                ))
            
            conn.commit()
        
        return eval_ids
    
    def seed_metrics(self, eval_ids: List[str]) -> None:
        """Seed evaluation metrics."""
        metrics = ["accuracy", "relevance", "coherence", "fluency", "factuality"]
        
        with sqlite3.connect(self.db_path) as conn:
            for eval_id in eval_ids:
                for metric in metrics[:3]:  # Add 3 metrics per evaluation
                    conn.execute("""
                        INSERT INTO evaluation_metrics (
                            evaluation_id, metric_name, score, created_at
                        ) VALUES (?, ?, ?, ?)
                    """, (
                        eval_id,
                        metric,
                        0.7 + (hash(eval_id + metric) % 30) / 100,
                        datetime.utcnow().isoformat()
                    ))
            
            conn.commit()
    
    def seed_webhooks(self, count: int = 3) -> List[str]:
        """Seed webhook registrations."""
        webhook_ids = []
        
        with sqlite3.connect(self.db_path) as conn:
            for i in range(count):
                webhook_id = f"webhook_{uuid.uuid4().hex[:8]}"
                webhook_ids.append(webhook_id)
                
                conn.execute("""
                    INSERT INTO webhook_registrations (
                        webhook_id, url, events,
                        active, created_at, user_id, secret
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    webhook_id,
                    f"https://example.com/webhook/{i}",
                    json.dumps(["evaluation.completed", "run.started"]),
                    i != 1,  # Second webhook is inactive
                    datetime.utcnow().isoformat(),
                    f"test_user_{i}",
                    f"secret_{i}"  # Add a secret for each webhook
                ))
            
            conn.commit()
        
        return webhook_ids
    
    def seed_audit_log(self, count: int = 20) -> None:
        """Seed audit log entries."""
        event_types = ["evaluation.create", "evaluation.update", "run.create", "webhook.register"]
        actions = ["create", "update", "delete", "view"]
        
        with sqlite3.connect(self.db_path) as conn:
            for i in range(count):
                conn.execute("""
                    INSERT INTO audit_log (
                        event_type, action, user_id, resource_id,
                        details, ip_address, user_agent, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event_types[i % len(event_types)],
                    actions[i % len(actions)],
                    f"user_{i % 5}",
                    f"resource_{i}",
                    json.dumps({"test": True, "index": i}),
                    f"192.168.1.{i % 255}",
                    "TestClient/1.0",
                    (datetime.utcnow() - timedelta(hours=i)).isoformat()
                ))
            
            conn.commit()
    
    def clear_all_tables(self) -> None:
        """Clear all data from tables (for cleanup)."""
        tables = [
            "evaluations", "runs", "run_samples", "datasets",
            "internal_evaluations", "evaluation_metrics",
            "webhook_registrations", "webhook_deliveries",
            "rate_limit_tracking", "audit_log"
        ]
        
        with sqlite3.connect(self.db_path) as conn:
            for table in tables:
                conn.execute(f"DELETE FROM {table}")
            conn.commit()
    
    def get_statistics(self) -> Dict[str, int]:
        """Get counts of records in each table."""
        stats = {}
        
        tables = [
            "evaluations", "runs", "datasets",
            "internal_evaluations", "evaluation_metrics",
            "webhook_registrations", "audit_log"
        ]
        
        with sqlite3.connect(self.db_path) as conn:
            for table in tables:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]
        
        return stats


def create_test_database_with_data(db_path: str) -> TestDatabaseHelper:
    """Create and seed a test database with sample data."""
    # First, ensure the database is properly initialized with all tables
    from tldw_Server_API.app.core.DB_Management.Evaluations_DB import EvaluationsDatabase
    db = EvaluationsDatabase(db_path)  # This creates all necessary tables
    
    helper = TestDatabaseHelper(db_path)
    
    # Seed all tables
    dataset_ids = helper.seed_datasets(5)
    eval_ids = helper.seed_evaluations(10)
    run_ids = helper.seed_runs(eval_ids[:5], 3)  # Create runs for first 5 evaluations
    internal_eval_ids = helper.seed_internal_evaluations(10)
    helper.seed_metrics(internal_eval_ids)
    webhook_ids = helper.seed_webhooks(3)
    # helper.seed_audit_log(20)  # Commented out - audit_log table doesn't exist
    
    return helper