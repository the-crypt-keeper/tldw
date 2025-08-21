# test_case_manager.py
# Test case management for Prompt Studio

import json
import sqlite3
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from loguru import logger

from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import (
    PromptStudioDatabase, DatabaseError, InputError, ConflictError
)

########################################################################################################################
# Data Classes

@dataclass
class TestCase:
    """Test case for prompt evaluation."""
    name: str
    inputs: Dict[str, Any]
    expected_outputs: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    tags: List[str] = field(default_factory=list)
    is_golden: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "inputs": self.inputs,
            "expected_outputs": self.expected_outputs,
            "tags": self.tags,
            "is_golden": self.is_golden,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create from dictionary."""
        return cls(**data)

@dataclass
class TestResult:
    """Result of running a test case."""
    test_case_id: str
    actual_outputs: Dict[str, Any]
    passed: bool
    execution_time: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

########################################################################################################################
# Test Case Manager Class

class TestCaseManager:
    """Manages test cases for Prompt Studio projects."""
    
    def __init__(self, db: PromptStudioDatabase):
        """
        Initialize TestCaseManager with database instance.
        
        Args:
            db: PromptStudioDatabase instance
        """
        self.db = db
        self.client_id = db.client_id
        self.test_cases = {}
        self.results = {}
    
    ####################################################################################################################
    # CRUD Operations
    
    def create_test_case(self, project_id: int, name: Optional[str], description: Optional[str],
                        inputs: Dict[str, Any], expected_outputs: Optional[Dict[str, Any]] = None,
                        tags: Optional[List[str]] = None, is_golden: bool = False,
                        signature_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Create a new test case.
        
        Args:
            project_id: Parent project ID
            name: Test case name
            description: Test case description
            inputs: Input data
            expected_outputs: Expected output data
            tags: Tags for categorization
            is_golden: Whether this is a golden test case
            signature_id: Associated signature ID
            
        Returns:
            Created test case record
        """
        # Validate inputs
        if not name or not name.strip():
            raise InputError("Test case name cannot be empty")
        
        try:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            # Generate UUID
            test_case_uuid = str(uuid.uuid4())
            
            # Convert tags to string
            tags_str = ",".join(tags) if tags else None
            
            # Insert test case
            cursor.execute("""
                INSERT INTO prompt_studio_test_cases (
                    uuid, project_id, signature_id, name, description,
                    inputs, expected_outputs, tags, is_golden, client_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                test_case_uuid, project_id, signature_id, name, description,
                json.dumps(inputs), json.dumps(expected_outputs) if expected_outputs else None,
                tags_str, int(is_golden), self.client_id
            ))
            
            test_case_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Created test case: {name or 'Unnamed'} (ID: {test_case_id})")
            
            return self.get_test_case(test_case_id)
            
        except sqlite3.IntegrityError as e:
            raise ConflictError(f"Failed to create test case: {e}")
        except Exception as e:
            raise DatabaseError(f"Failed to create test case: {e}")
    
    def get_test_case(self, test_case_id: int, include_deleted: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get a test case by ID.
        
        Args:
            test_case_id: Test case ID
            include_deleted: Include soft-deleted test cases
            
        Returns:
            Test case record or None
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        query = """
            SELECT * FROM prompt_studio_test_cases
            WHERE id = ?
        """
        
        if not include_deleted:
            query += " AND deleted = 0"
        
        cursor.execute(query, (test_case_id,))
        row = cursor.fetchone()
        
        if row:
            test_case = self.db._row_to_dict(cursor, row)
            # Parse tags back to list
            if test_case.get("tags"):
                test_case["tags"] = test_case["tags"].split(",")
            return test_case
        return None
    
    def list_test_cases(self, project_id: int, signature_id: Optional[int] = None,
                       is_golden: Optional[bool] = None, tags: Optional[List[str]] = None,
                       search: Optional[str] = None, include_deleted: bool = False,
                       page: int = 1, per_page: int = 20) -> Dict[str, Any]:
        """
        List test cases with filtering.
        
        Args:
            project_id: Project ID
            signature_id: Filter by signature
            is_golden: Filter by golden status
            tags: Filter by tags
            search: Search in name and description
            include_deleted: Include soft-deleted test cases
            page: Page number
            per_page: Items per page
            
        Returns:
            Dictionary with test cases and pagination
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        # Build query
        conditions = ["project_id = ?"]
        params = [project_id]
        
        if not include_deleted:
            conditions.append("deleted = 0")
        
        if signature_id is not None:
            conditions.append("signature_id = ?")
            params.append(signature_id)
        
        if is_golden is not None:
            conditions.append("is_golden = ?")
            params.append(int(is_golden))
        
        if tags:
            tag_conditions = []
            for tag in tags:
                tag_conditions.append("tags LIKE ?")
                params.append(f"%{tag}%")
            conditions.append(f"({' OR '.join(tag_conditions)})")
        
        where_clause = " WHERE " + " AND ".join(conditions)
        
        # Add search if provided
        if search:
            where_clause += " AND (name LIKE ? OR description LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%"])
        
        # Count total
        count_query = f"SELECT COUNT(*) FROM prompt_studio_test_cases{where_clause}"
        cursor.execute(count_query, params)
        total = cursor.fetchone()[0]
        
        # Get test cases with pagination
        offset = (page - 1) * per_page
        query = f"""
            SELECT * FROM prompt_studio_test_cases
            {where_clause}
            ORDER BY is_golden DESC, created_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([per_page, offset])
        
        cursor.execute(query, params)
        test_cases = []
        for row in cursor.fetchall():
            test_case = self.db._row_to_dict(cursor, row)
            if test_case.get("tags"):
                test_case["tags"] = test_case["tags"].split(",")
            test_cases.append(test_case)
        
        return {
            "test_cases": test_cases,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "total_pages": (total + per_page - 1) // per_page
            }
        }
    
    def update_test_case(self, test_case_id: int, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a test case.
        
        Args:
            test_case_id: Test case ID
            updates: Fields to update
            
        Returns:
            Updated test case record
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        # Build update query
        allowed_fields = ["name", "description", "inputs", "expected_outputs", 
                         "actual_outputs", "tags", "is_golden"]
        set_clauses = []
        params = []
        
        for field in allowed_fields:
            if field in updates:
                set_clauses.append(f"{field} = ?")
                value = updates[field]
                
                # Handle special fields
                if field in ["inputs", "expected_outputs", "actual_outputs"] and value is not None:
                    value = json.dumps(value)
                elif field == "tags" and isinstance(value, list):
                    value = ",".join(value)
                elif field == "is_golden":
                    value = int(value)
                
                params.append(value)
        
        if not set_clauses:
            return self.get_test_case(test_case_id)
        
        # Add updated_at
        set_clauses.append("updated_at = CURRENT_TIMESTAMP")
        params.append(test_case_id)
        
        query = f"""
            UPDATE prompt_studio_test_cases
            SET {', '.join(set_clauses)}
            WHERE id = ? AND deleted = 0
        """
        
        cursor.execute(query, params)
        
        if cursor.rowcount == 0:
            raise InputError(f"Test case {test_case_id} not found or already deleted")
        
        conn.commit()
        
        return self.get_test_case(test_case_id)
    
    def delete_test_case(self, test_case_id: int, hard_delete: bool = False) -> bool:
        """
        Delete a test case.
        
        Args:
            test_case_id: Test case ID
            hard_delete: Permanently delete if True
            
        Returns:
            True if deleted
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        if hard_delete:
            cursor.execute("DELETE FROM prompt_studio_test_cases WHERE id = ?", (test_case_id,))
        else:
            cursor.execute("""
                UPDATE prompt_studio_test_cases
                SET deleted = 1, deleted_at = CURRENT_TIMESTAMP
                WHERE id = ? AND deleted = 0
            """, (test_case_id,))
        
        success = cursor.rowcount > 0
        if success:
            conn.commit()
            logger.info(f"{'Hard' if hard_delete else 'Soft'} deleted test case {test_case_id}")
        
        return success
    
    ####################################################################################################################
    # Bulk Operations
    
    def create_bulk_test_cases(self, project_id: int, test_cases: List[Dict[str, Any]],
                              signature_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Create multiple test cases at once.
        
        Args:
            project_id: Project ID
            test_cases: List of test case data
            signature_id: Optional signature ID for all test cases
            
        Returns:
            List of created test case records
        """
        created_cases = []
        
        try:
            with self.db.transaction() as conn:
                for test_case_data in test_cases:
                    test_case = self.create_test_case(
                        project_id=project_id,
                        name=test_case_data.get("name"),
                        description=test_case_data.get("description"),
                        inputs=test_case_data["inputs"],
                        expected_outputs=test_case_data.get("expected_outputs"),
                        tags=test_case_data.get("tags"),
                        is_golden=test_case_data.get("is_golden", False),
                        signature_id=signature_id or test_case_data.get("signature_id")
                    )
                    created_cases.append(test_case)
            
            logger.info(f"Created {len(created_cases)} test cases in bulk")
            return created_cases
            
        except Exception as e:
            logger.error(f"Failed to create test cases in bulk: {e}")
            raise DatabaseError(f"Bulk creation failed: {e}")
    
    ####################################################################################################################
    # Search and Filter
    
    def search_test_cases(self, project_id: int, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search test cases using FTS.
        
        Args:
            project_id: Project ID
            query: Search query
            limit: Maximum results
            
        Returns:
            List of matching test cases
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        # Use FTS for search
        cursor.execute("""
            SELECT tc.*
            FROM prompt_studio_test_cases tc
            JOIN prompt_studio_test_cases_fts fts ON tc.id = fts.rowid
            WHERE tc.project_id = ? 
                AND tc.deleted = 0
                AND fts.prompt_studio_test_cases_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (project_id, query, limit))
        
        results = []
        for row in cursor.fetchall():
            test_case = self.db._row_to_dict(cursor, row)
            if test_case.get("tags"):
                test_case["tags"] = test_case["tags"].split(",")
            results.append(test_case)
        
        return results
    
    def get_golden_test_cases(self, project_id: int) -> List[Dict[str, Any]]:
        """
        Get all golden test cases for a project.
        
        Args:
            project_id: Project ID
            
        Returns:
            List of golden test cases
        """
        result = self.list_test_cases(
            project_id=project_id,
            is_golden=True,
            per_page=1000  # Get all golden cases
        )
        return result["test_cases"]
    
    def get_test_cases_by_signature(self, signature_id: int) -> List[Dict[str, Any]]:
        """
        Get all test cases for a signature.
        
        Args:
            signature_id: Signature ID
            
        Returns:
            List of test cases
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM prompt_studio_test_cases
            WHERE signature_id = ? AND deleted = 0
            ORDER BY is_golden DESC, created_at DESC
        """, (signature_id,))
        
        test_cases = []
        for row in cursor.fetchall():
            test_case = self.db._row_to_dict(cursor, row)
            if test_case.get("tags"):
                test_case["tags"] = test_case["tags"].split(",")
            test_cases.append(test_case)
        
        return test_cases
    
    ####################################################################################################################
    # Statistics
    
    def get_test_case_stats(self, project_id: int) -> Dict[str, Any]:
        """
        Get statistics for test cases in a project.
        
        Args:
            project_id: Project ID
            
        Returns:
            Statistics dictionary
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        # Total test cases
        cursor.execute("""
            SELECT COUNT(*) FROM prompt_studio_test_cases
            WHERE project_id = ? AND deleted = 0
        """, (project_id,))
        stats["total"] = cursor.fetchone()[0]
        
        # Golden test cases
        cursor.execute("""
            SELECT COUNT(*) FROM prompt_studio_test_cases
            WHERE project_id = ? AND deleted = 0 AND is_golden = 1
        """, (project_id,))
        stats["golden"] = cursor.fetchone()[0]
        
        # Generated test cases
        cursor.execute("""
            SELECT COUNT(*) FROM prompt_studio_test_cases
            WHERE project_id = ? AND deleted = 0 AND is_generated = 1
        """, (project_id,))
        stats["generated"] = cursor.fetchone()[0]
        
        # Test cases with expected outputs
        cursor.execute("""
            SELECT COUNT(*) FROM prompt_studio_test_cases
            WHERE project_id = ? AND deleted = 0 AND expected_outputs IS NOT NULL
        """, (project_id,))
        stats["with_expected"] = cursor.fetchone()[0]
        
        # Test cases by signature
        cursor.execute("""
            SELECT signature_id, COUNT(*) as count
            FROM prompt_studio_test_cases
            WHERE project_id = ? AND deleted = 0 AND signature_id IS NOT NULL
            GROUP BY signature_id
        """, (project_id,))
        stats["by_signature"] = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Most used tags
        cursor.execute("""
            SELECT tags FROM prompt_studio_test_cases
            WHERE project_id = ? AND deleted = 0 AND tags IS NOT NULL
        """, (project_id,))
        
        tag_counts = {}
        for row in cursor.fetchall():
            tags = row[0].split(",")
            for tag in tags:
                tag = tag.strip()
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        stats["top_tags"] = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return stats