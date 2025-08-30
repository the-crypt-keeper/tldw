# Evaluations_DB.py - Database management for OpenAI-compatible evaluations API
"""
Database operations for evaluations, runs, and datasets.

Provides CRUD operations and query methods for:
- Evaluation definitions
- Evaluation runs
- Datasets
"""

import json
import sqlite3
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager
from loguru import logger

class EvaluationsDatabase:
    """Database manager for evaluations system"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._initialize_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _initialize_database(self):
        """Create database tables if they don't exist"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Evaluations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    eval_type TEXT NOT NULL,
                    eval_spec TEXT NOT NULL,
                    dataset_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_by TEXT,
                    metadata TEXT,
                    deleted_at TIMESTAMP NULL
                )
            """)
            
            # Evaluation runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    id TEXT PRIMARY KEY,
                    eval_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    target_model TEXT,
                    config TEXT,
                    progress TEXT,
                    results TEXT,
                    error_message TEXT,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    webhook_url TEXT,
                    usage TEXT,
                    FOREIGN KEY (eval_id) REFERENCES evaluations(id)
                )
            """)
            
            # Datasets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    samples TEXT NOT NULL,
                    sample_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_by TEXT,
                    metadata TEXT
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_evals_created ON evaluations(created_at DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_eval ON evaluation_runs(eval_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON evaluation_runs(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_datasets_created ON datasets(created_at DESC)")
            
            conn.commit()
            logger.info("Evaluations database initialized")
    
    # ============= Evaluation CRUD Operations =============
    
    def create_evaluation(
        self,
        name: str,
        eval_type: str,
        eval_spec: Dict[str, Any],
        description: Optional[str] = None,
        dataset_id: Optional[str] = None,
        created_by: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new evaluation definition"""
        eval_id = f"eval_{uuid.uuid4().hex[:12]}"
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO evaluations (id, name, description, eval_type, eval_spec, 
                                       dataset_id, created_by, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                eval_id,
                name,
                description,
                eval_type,
                json.dumps(eval_spec),
                dataset_id,
                created_by,
                json.dumps(metadata) if metadata else None
            ))
            conn.commit()
        
        logger.info(f"Created evaluation: {eval_id}")
        return eval_id
    
    def get_evaluation(self, eval_id: str) -> Optional[Dict[str, Any]]:
        """Get evaluation by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM evaluations 
                WHERE id = ? AND deleted_at IS NULL
            """, (eval_id,))
            
            row = cursor.fetchone()
            if row:
                return self._row_to_eval_dict(row)
        return None
    
    def list_evaluations(
        self,
        limit: int = 20,
        after: Optional[str] = None,
        eval_type: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """List evaluations with pagination"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM evaluations WHERE deleted_at IS NULL"
            params = []
            
            if eval_type:
                query += " AND eval_type = ?"
                params.append(eval_type)
            
            if after:
                query += " AND created_at < (SELECT created_at FROM evaluations WHERE id = ?)"
                params.append(after)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit + 1)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            has_more = len(rows) > limit
            evaluations = [self._row_to_eval_dict(row) for row in rows[:limit]]
            
            return evaluations, has_more
    
    def update_evaluation(self, eval_id: str, updates: Dict[str, Any]) -> bool:
        """Update evaluation definition"""
        allowed_fields = {"name", "description", "eval_spec", "dataset_id", "metadata"}
        updates = {k: v for k, v in updates.items() if k in allowed_fields}
        
        if not updates:
            return False
        
        # Handle metadata merging
        if "metadata" in updates:
            # Get existing evaluation to merge metadata
            existing = self.get_evaluation(eval_id)
            if existing and existing.get("metadata"):
                # Merge existing metadata with updates
                merged_metadata = existing["metadata"].copy()
                merged_metadata.update(updates["metadata"])
                updates["metadata"] = merged_metadata
        
        # JSON serialize complex fields
        if "eval_spec" in updates:
            updates["eval_spec"] = json.dumps(updates["eval_spec"])
        if "metadata" in updates:
            updates["metadata"] = json.dumps(updates["metadata"])
        
        updates["updated_at"] = datetime.utcnow().isoformat()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
            values = list(updates.values()) + [eval_id]
            
            cursor.execute(f"""
                UPDATE evaluations 
                SET {set_clause}
                WHERE id = ? AND deleted_at IS NULL
            """, values)
            
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_evaluation(self, eval_id: str) -> bool:
        """Soft delete evaluation"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE evaluations 
                SET deleted_at = CURRENT_TIMESTAMP
                WHERE id = ? AND deleted_at IS NULL
            """, (eval_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    # ============= Run CRUD Operations =============
    
    def create_run(
        self,
        eval_id: str,
        target_model: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        webhook_url: Optional[str] = None
    ) -> str:
        """Create a new evaluation run"""
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO evaluation_runs (id, eval_id, status, target_model, config, webhook_url)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                eval_id,
                "pending",
                target_model,
                json.dumps(config) if config else None,
                webhook_url
            ))
            conn.commit()
        
        logger.info(f"Created run: {run_id} for evaluation: {eval_id}")
        return run_id
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get run by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM evaluation_runs WHERE id = ?", (run_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_run_dict(row)
        return None
    
    def list_runs(
        self,
        eval_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20,
        after: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """List runs with optional filtering"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM evaluation_runs WHERE 1=1"
            params = []
            
            if eval_id:
                query += " AND eval_id = ?"
                params.append(eval_id)
            
            if status:
                query += " AND status = ?"
                params.append(status)
            
            if after:
                query += " AND created_at < (SELECT created_at FROM evaluation_runs WHERE id = ?)"
                params.append(after)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit + 1)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            has_more = len(rows) > limit
            runs = [self._row_to_run_dict(row) for row in rows[:limit]]
            
            return runs, has_more
    
    def update_run_status(
        self,
        run_id: str,
        status: str,
        error_message: Optional[str] = None
    ) -> bool:
        """Update run status"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            updates = {"status": status}
            
            if status == "running" and "started_at" not in updates:
                updates["started_at"] = datetime.utcnow().isoformat()
            elif status in ["completed", "failed", "cancelled"]:
                updates["completed_at"] = datetime.utcnow().isoformat()
            
            if error_message:
                updates["error_message"] = error_message
            
            set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
            values = list(updates.values()) + [run_id]
            
            cursor.execute(f"""
                UPDATE evaluation_runs 
                SET {set_clause}
                WHERE id = ?
            """, values)
            
            conn.commit()
            return cursor.rowcount > 0
    
    def update_run_progress(self, run_id: str, progress: Dict[str, Any]) -> bool:
        """Update run progress"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE evaluation_runs 
                SET progress = ?
                WHERE id = ?
            """, (json.dumps(progress), run_id))
            conn.commit()
            return cursor.rowcount > 0
    
    def store_run_results(
        self,
        run_id: str,
        results: Dict[str, Any],
        usage: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store run results"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE evaluation_runs 
                SET results = ?, usage = ?, status = 'completed', completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (
                json.dumps(results),
                json.dumps(usage) if usage else None,
                run_id
            ))
            conn.commit()
            return cursor.rowcount > 0
    
    # ============= Dataset CRUD Operations =============
    
    def create_dataset(
        self,
        name: str,
        samples: List[Dict[str, Any]],
        description: Optional[str] = None,
        created_by: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new dataset"""
        dataset_id = f"dataset_{uuid.uuid4().hex[:12]}"
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO datasets (id, name, description, samples, sample_count, created_by, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                dataset_id,
                name,
                description,
                json.dumps(samples),
                len(samples),
                created_by,
                json.dumps(metadata) if metadata else None
            ))
            conn.commit()
        
        logger.info(f"Created dataset: {dataset_id} with {len(samples)} samples")
        return dataset_id
    
    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get dataset by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_dataset_dict(row)
        return None
    
    def list_datasets(
        self,
        limit: int = 20,
        after: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """List datasets with pagination"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM datasets WHERE 1=1"
            params = []
            
            if after:
                query += " AND created_at < (SELECT created_at FROM datasets WHERE id = ?)"
                params.append(after)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit + 1)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            has_more = len(rows) > limit
            datasets = [self._row_to_dataset_dict(row, include_samples=False) 
                       for row in rows[:limit]]
            
            return datasets, has_more
    
    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete dataset"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    # ============= Helper Methods =============
    
    def _row_to_eval_dict(self, row) -> Dict[str, Any]:
        """Convert database row to evaluation dictionary"""
        # Parse created_at timestamp
        if row["created_at"]:
            if "T" in row["created_at"]:
                # ISO format
                created_timestamp = int(datetime.fromisoformat(row["created_at"].replace("Z", "+00:00")).timestamp())
            else:
                # SQLite timestamp format
                created_timestamp = int(datetime.strptime(row["created_at"], "%Y-%m-%d %H:%M:%S").timestamp())
        else:
            created_timestamp = int(datetime.now().timestamp())
        
        return {
            "id": row["id"],
            "object": "evaluation",
            "created_at": created_timestamp,  # Fixed: use created_at instead of created
            "name": row["name"],
            "description": row["description"],
            "eval_type": row["eval_type"],
            "eval_spec": json.loads(row["eval_spec"]) if row["eval_spec"] else {},
            "dataset_id": row["dataset_id"],
            "created_by": row["created_by"] or "unknown",  # Fixed: add created_by field
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
        }
    
    def _row_to_run_dict(self, row) -> Dict[str, Any]:
        """Convert database row to run dictionary"""
        # Parse created_at timestamp
        if row["created_at"]:
            if "T" in row["created_at"]:
                # ISO format
                created_timestamp = int(datetime.fromisoformat(row["created_at"].replace("Z", "+00:00")).timestamp())
            else:
                # SQLite timestamp format
                created_timestamp = int(datetime.strptime(row["created_at"], "%Y-%m-%d %H:%M:%S").timestamp())
        else:
            created_timestamp = int(datetime.now().timestamp())
            
        # Parse optional timestamps
        started_at = None
        if row["started_at"]:
            try:
                started_at = int(datetime.fromisoformat(row["started_at"].replace("Z", "+00:00")).timestamp())
            except:
                started_at = None
                
        completed_at = None
        if row["completed_at"]:
            try:
                completed_at = int(datetime.fromisoformat(row["completed_at"].replace("Z", "+00:00")).timestamp())
            except:
                completed_at = None
        
        return {
            "id": row["id"],
            "object": "run",  # Fixed: use "run" to match schema
            "created_at": created_timestamp,  # Fixed: use created_at instead of created
            "eval_id": row["eval_id"],
            "status": row["status"],
            "target_model": row["target_model"] or "",  # Ensure not None
            "config": json.loads(row["config"]) if row["config"] else {},
            "progress": json.loads(row["progress"]) if row["progress"] else None,
            "results": json.loads(row["results"]) if row["results"] else None,
            "error_message": row["error_message"],
            "started_at": started_at,
            "completed_at": completed_at,
            "usage": json.loads(row["usage"]) if row["usage"] else None
        }
    
    def _row_to_dataset_dict(self, row, include_samples: bool = True) -> Dict[str, Any]:
        """Convert database row to dataset dictionary"""
        # Parse created_at timestamp
        if row["created_at"]:
            if "T" in row["created_at"]:
                # ISO format
                created_timestamp = int(datetime.fromisoformat(row["created_at"].replace("Z", "+00:00")).timestamp())
            else:
                # SQLite timestamp format
                created_timestamp = int(datetime.strptime(row["created_at"], "%Y-%m-%d %H:%M:%S").timestamp())
        else:
            created_timestamp = int(datetime.now().timestamp())
            
        result = {
            "id": row["id"],
            "object": "dataset",
            "created_at": created_timestamp,  # Fixed: use created_at instead of created
            "name": row["name"],
            "description": row["description"],
            "sample_count": row["sample_count"] or 0,
            "created_by": row["created_by"] or "unknown",  # Fixed: add created_by field
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
        }
        
        if include_samples:
            result["samples"] = json.loads(row["samples"]) if row["samples"] else []
        
        return result