# test_case_io.py
# Import/Export functionality for test cases

import csv
import json
import base64
from io import StringIO
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from loguru import logger

from .test_case_manager import TestCaseManager

########################################################################################################################
# Test Case I/O Manager

class TestCaseIO:
    """Handles import and export of test cases in various formats."""
    
    def __init__(self, test_case_manager: TestCaseManager):
        """
        Initialize TestCaseIO with a test case manager.
        
        Args:
            test_case_manager: TestCaseManager instance
        """
        self.manager = test_case_manager
    
    ####################################################################################################################
    # Export Functions
    
    def export_to_json(self, project_id: int, include_golden_only: bool = False,
                      tag_filter: Optional[List[str]] = None) -> str:
        """
        Export test cases to JSON format.
        
        Args:
            project_id: Project ID
            include_golden_only: Export only golden test cases
            tag_filter: Filter by tags
            
        Returns:
            JSON string of test cases
        """
        # Get test cases
        result = self.manager.list_test_cases(
            project_id=project_id,
            is_golden=True if include_golden_only else None,
            tags=tag_filter,
            per_page=10000  # Get all
        )
        
        test_cases = result["test_cases"]
        
        # Clean up for export
        export_data = []
        for tc in test_cases:
            export_case = {
                "name": tc.get("name"),
                "description": tc.get("description"),
                "inputs": tc.get("inputs"),
                "expected_outputs": tc.get("expected_outputs"),
                "tags": tc.get("tags", []),
                "is_golden": tc.get("is_golden", False)
            }
            # Remove None values
            export_case = {k: v for k, v in export_case.items() if v is not None}
            export_data.append(export_case)
        
        return json.dumps({
            "version": "1.0",
            "export_date": datetime.utcnow().isoformat(),
            "project_id": project_id,
            "test_cases": export_data
        }, indent=2)
    
    def export_to_csv(self, project_id: int, include_golden_only: bool = False,
                     tag_filter: Optional[List[str]] = None) -> str:
        """
        Export test cases to CSV format.
        
        Args:
            project_id: Project ID
            include_golden_only: Export only golden test cases
            tag_filter: Filter by tags
            
        Returns:
            CSV string of test cases
        """
        # Get test cases
        result = self.manager.list_test_cases(
            project_id=project_id,
            is_golden=True if include_golden_only else None,
            tags=tag_filter,
            per_page=10000  # Get all
        )
        
        test_cases = result["test_cases"]
        
        # Create CSV
        output = StringIO()
        
        if not test_cases:
            return ""
        
        # Determine all unique input and output keys
        all_input_keys = set()
        all_output_keys = set()
        
        for tc in test_cases:
            if tc.get("inputs"):
                all_input_keys.update(tc["inputs"].keys())
            if tc.get("expected_outputs"):
                all_output_keys.update(tc["expected_outputs"].keys())
        
        # Create header
        fieldnames = ["name", "description"]
        fieldnames.extend([f"input.{key}" for key in sorted(all_input_keys)])
        fieldnames.extend([f"expected.{key}" for key in sorted(all_output_keys)])
        fieldnames.extend(["tags", "is_golden"])
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write test cases
        for tc in test_cases:
            row = {
                "name": tc.get("name", ""),
                "description": tc.get("description", ""),
                "tags": ",".join(tc.get("tags", [])),
                "is_golden": "yes" if tc.get("is_golden") else "no"
            }
            
            # Add input fields
            if tc.get("inputs"):
                for key, value in tc["inputs"].items():
                    row[f"input.{key}"] = json.dumps(value) if not isinstance(value, str) else value
            
            # Add expected output fields
            if tc.get("expected_outputs"):
                for key, value in tc["expected_outputs"].items():
                    row[f"expected.{key}"] = json.dumps(value) if not isinstance(value, str) else value
            
            writer.writerow(row)
        
        return output.getvalue()
    
    ####################################################################################################################
    # Import Functions
    
    def import_from_json(self, project_id: int, json_data: str, 
                        signature_id: Optional[int] = None,
                        auto_generate_names: bool = True) -> Tuple[int, List[str]]:
        """
        Import test cases from JSON format.
        
        Args:
            project_id: Project ID
            json_data: JSON string or base64 encoded JSON
            signature_id: Optional signature ID for all test cases
            auto_generate_names: Generate names if missing
            
        Returns:
            Tuple of (number imported, list of errors)
        """
        errors = []
        imported_count = 0
        
        try:
            # Try to decode if base64
            try:
                json_bytes = base64.b64decode(json_data)
                json_data = json_bytes.decode('utf-8')
            except:
                pass  # Not base64, use as is
            
            # Parse JSON
            data = json.loads(json_data)
            
            # Handle both raw array and structured format
            if isinstance(data, list):
                test_cases = data
            elif isinstance(data, dict) and "test_cases" in data:
                test_cases = data["test_cases"]
            else:
                errors.append("Invalid JSON format: expected array or object with 'test_cases'")
                return 0, errors
            
            # Import each test case
            for idx, tc_data in enumerate(test_cases):
                try:
                    # Validate required fields
                    if "inputs" not in tc_data:
                        errors.append(f"Test case {idx}: missing 'inputs' field")
                        continue
                    
                    # Generate name if needed
                    name = tc_data.get("name")
                    if not name and auto_generate_names:
                        name = f"Imported Test {idx + 1}"
                    
                    # Create test case
                    self.manager.create_test_case(
                        project_id=project_id,
                        name=name,
                        description=tc_data.get("description"),
                        inputs=tc_data["inputs"],
                        expected_outputs=tc_data.get("expected_outputs"),
                        tags=tc_data.get("tags"),
                        is_golden=tc_data.get("is_golden", False),
                        signature_id=signature_id
                    )
                    imported_count += 1
                    
                except Exception as e:
                    errors.append(f"Test case {idx}: {str(e)}")
            
            logger.info(f"Imported {imported_count} test cases from JSON")
            
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON: {str(e)}")
        except Exception as e:
            errors.append(f"Import failed: {str(e)}")
        
        return imported_count, errors
    
    def import_from_csv(self, project_id: int, csv_data: str,
                       signature_id: Optional[int] = None,
                       auto_generate_names: bool = True) -> Tuple[int, List[str]]:
        """
        Import test cases from CSV format.
        
        Args:
            project_id: Project ID
            csv_data: CSV string or base64 encoded CSV
            signature_id: Optional signature ID for all test cases
            auto_generate_names: Generate names if missing
            
        Returns:
            Tuple of (number imported, list of errors)
        """
        errors = []
        imported_count = 0
        
        try:
            # Try to decode if base64
            try:
                csv_bytes = base64.b64decode(csv_data)
                csv_data = csv_bytes.decode('utf-8')
            except:
                pass  # Not base64, use as is
            
            # Parse CSV
            reader = csv.DictReader(StringIO(csv_data))
            
            for row_idx, row in enumerate(reader):
                try:
                    # Extract inputs and expected outputs
                    inputs = {}
                    expected_outputs = {}
                    
                    for key, value in row.items():
                        if key.startswith("input."):
                            field_name = key[6:]  # Remove "input." prefix
                            # Try to parse as JSON, otherwise use as string
                            try:
                                inputs[field_name] = json.loads(value) if value else None
                            except:
                                inputs[field_name] = value
                        elif key.startswith("expected."):
                            field_name = key[9:]  # Remove "expected." prefix
                            try:
                                expected_outputs[field_name] = json.loads(value) if value else None
                            except:
                                expected_outputs[field_name] = value
                    
                    # Skip if no inputs
                    if not inputs:
                        errors.append(f"Row {row_idx + 2}: no input fields found")
                        continue
                    
                    # Generate name if needed
                    name = row.get("name", "").strip()
                    if not name and auto_generate_names:
                        name = f"Imported Test {row_idx + 1}"
                    
                    # Parse tags
                    tags = []
                    if row.get("tags"):
                        tags = [t.strip() for t in row["tags"].split(",") if t.strip()]
                    
                    # Parse is_golden
                    is_golden = row.get("is_golden", "").lower() in ["yes", "true", "1"]
                    
                    # Create test case
                    self.manager.create_test_case(
                        project_id=project_id,
                        name=name,
                        description=row.get("description", "").strip() or None,
                        inputs=inputs,
                        expected_outputs=expected_outputs if expected_outputs else None,
                        tags=tags if tags else None,
                        is_golden=is_golden,
                        signature_id=signature_id
                    )
                    imported_count += 1
                    
                except Exception as e:
                    errors.append(f"Row {row_idx + 2}: {str(e)}")
            
            logger.info(f"Imported {imported_count} test cases from CSV")
            
        except Exception as e:
            errors.append(f"CSV parsing failed: {str(e)}")
        
        return imported_count, errors
    
    ####################################################################################################################
    # Template Generation
    
    def generate_csv_template(self, signature_id: Optional[int] = None) -> str:
        """
        Generate a CSV template for test case import.
        
        Args:
            signature_id: Optional signature ID to base template on
            
        Returns:
            CSV template string
        """
        output = StringIO()
        
        # Default fields
        fieldnames = ["name", "description"]
        
        if signature_id:
            # Get signature to determine fields
            conn = self.manager.db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT input_schema, output_schema
                FROM prompt_studio_signatures
                WHERE id = ? AND deleted = 0
            """, (signature_id,))
            
            row = cursor.fetchone()
            if row:
                try:
                    input_schema = json.loads(row[0]) if row[0] else []
                    output_schema = json.loads(row[1]) if row[1] else []
                    
                    # Add input fields
                    for field in input_schema:
                        if isinstance(field, dict) and "name" in field:
                            fieldnames.append(f"input.{field['name']}")
                    
                    # Add output fields
                    for field in output_schema:
                        if isinstance(field, dict) and "name" in field:
                            fieldnames.append(f"expected.{field['name']}")
                except:
                    # Fall back to generic template
                    fieldnames.extend(["input.text", "expected.result"])
        else:
            # Generic template
            fieldnames.extend(["input.text", "expected.result"])
        
        fieldnames.extend(["tags", "is_golden"])
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        # Add example rows
        example_row = {
            "name": "Example Test 1",
            "description": "This is an example test case",
            "tags": "example,template",
            "is_golden": "no"
        }
        
        # Add placeholder values for input/output fields
        for field in fieldnames:
            if field.startswith("input."):
                example_row[field] = "Sample input value"
            elif field.startswith("expected."):
                example_row[field] = "Expected output value"
        
        writer.writerow(example_row)
        
        return output.getvalue()
    
    def generate_json_template(self, signature_id: Optional[int] = None) -> str:
        """
        Generate a JSON template for test case import.
        
        Args:
            signature_id: Optional signature ID to base template on
            
        Returns:
            JSON template string
        """
        template = {
            "version": "1.0",
            "test_cases": []
        }
        
        example_case = {
            "name": "Example Test 1",
            "description": "This is an example test case",
            "inputs": {},
            "expected_outputs": {},
            "tags": ["example", "template"],
            "is_golden": False
        }
        
        if signature_id:
            # Get signature to determine fields
            conn = self.manager.db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT input_schema, output_schema
                FROM prompt_studio_signatures
                WHERE id = ? AND deleted = 0
            """, (signature_id,))
            
            row = cursor.fetchone()
            if row:
                try:
                    input_schema = json.loads(row[0]) if row[0] else []
                    output_schema = json.loads(row[1]) if row[1] else []
                    
                    # Add input fields
                    for field in input_schema:
                        if isinstance(field, dict) and "name" in field:
                            field_type = field.get("type", "string")
                            if field_type == "integer":
                                example_case["inputs"][field["name"]] = 0
                            elif field_type == "boolean":
                                example_case["inputs"][field["name"]] = False
                            elif field_type == "array":
                                example_case["inputs"][field["name"]] = []
                            elif field_type == "object":
                                example_case["inputs"][field["name"]] = {}
                            else:
                                example_case["inputs"][field["name"]] = "Sample value"
                    
                    # Add output fields
                    for field in output_schema:
                        if isinstance(field, dict) and "name" in field:
                            field_type = field.get("type", "string")
                            if field_type == "integer":
                                example_case["expected_outputs"][field["name"]] = 0
                            elif field_type == "boolean":
                                example_case["expected_outputs"][field["name"]] = False
                            elif field_type == "array":
                                example_case["expected_outputs"][field["name"]] = []
                            elif field_type == "object":
                                example_case["expected_outputs"][field["name"]] = {}
                            else:
                                example_case["expected_outputs"][field["name"]] = "Expected value"
                except:
                    # Fall back to generic template
                    example_case["inputs"] = {"text": "Sample input"}
                    example_case["expected_outputs"] = {"result": "Expected output"}
        else:
            # Generic template
            example_case["inputs"] = {"text": "Sample input"}
            example_case["expected_outputs"] = {"result": "Expected output"}
        
        template["test_cases"].append(example_case)
        
        return json.dumps(template, indent=2)