# test_case_generator.py
# Test case generation for Prompt Studio

import json
import random
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from loguru import logger

from .test_case_manager import TestCaseManager

########################################################################################################################
# Test Case Generator

class TestCaseGenerator:
    """Generates test cases automatically for Prompt Studio projects."""
    
    def __init__(self, test_case_manager: TestCaseManager):
        """
        Initialize TestCaseGenerator.
        
        Args:
            test_case_manager: TestCaseManager instance
        """
        self.manager = test_case_manager
        self.db = test_case_manager.db
    
    ####################################################################################################################
    # Generation Strategies
    
    def generate_from_description(self, project_id: int, description: str, 
                                 num_cases: int = 5, signature_id: Optional[int] = None,
                                 prompt_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Generate test cases from a task description.
        
        Args:
            project_id: Project ID
            description: Task description to base generation on
            num_cases: Number of test cases to generate
            signature_id: Optional signature to follow
            prompt_id: Optional prompt to test against
            
        Returns:
            List of generated test cases
        """
        generated_cases = []
        
        # Get signature schema if provided
        input_schema = None
        output_schema = None
        
        if signature_id:
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT input_schema, output_schema
                FROM prompt_studio_signatures
                WHERE id = ? AND deleted = 0
            """, (signature_id,))
            
            row = cursor.fetchone()
            if row:
                try:
                    input_schema = json.loads(row[0]) if row[0] else None
                    output_schema = json.loads(row[1]) if row[1] else None
                except:
                    pass
        
        # Generate test cases based on description
        for i in range(num_cases):
            test_case = self._generate_single_case_from_description(
                description, i + 1, input_schema, output_schema
            )
            
            # Create the test case
            created = self.manager.create_test_case(
                project_id=project_id,
                name=test_case["name"],
                description=test_case["description"],
                inputs=test_case["inputs"],
                expected_outputs=test_case.get("expected_outputs"),
                tags=test_case.get("tags", ["generated", "from_description"]),
                is_golden=False,
                signature_id=signature_id
            )
            
            # Mark as generated
            conn = self.db.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE prompt_studio_test_cases
                SET is_generated = 1
                WHERE id = ?
            """, (created["id"],))
            conn.commit()
            
            generated_cases.append(created)
        
        logger.info(f"Generated {len(generated_cases)} test cases from description")
        return generated_cases
    
    def generate_diverse_cases(self, project_id: int, signature_id: int,
                              num_cases: int = 5) -> List[Dict[str, Any]]:
        """
        Generate diverse test cases based on signature schema.
        
        Args:
            project_id: Project ID
            signature_id: Signature ID to follow
            num_cases: Number of test cases to generate
            
        Returns:
            List of generated test cases
        """
        # Get signature schema
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT name, input_schema, output_schema
            FROM prompt_studio_signatures
            WHERE id = ? AND deleted = 0
        """, (signature_id,))
        
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Signature {signature_id} not found")
        
        sig_name, input_schema_str, output_schema_str = row
        
        try:
            input_schema = json.loads(input_schema_str) if input_schema_str else []
            output_schema = json.loads(output_schema_str) if output_schema_str else []
        except:
            input_schema = []
            output_schema = []
        
        generated_cases = []
        
        # Generate diverse cases
        strategies = ["edge_case", "typical", "complex", "minimal", "maximal"]
        
        for i in range(num_cases):
            strategy = strategies[i % len(strategies)]
            
            test_case_data = self._generate_case_by_strategy(
                input_schema, output_schema, strategy, i + 1
            )
            
            # Create the test case
            created = self.manager.create_test_case(
                project_id=project_id,
                name=f"{sig_name} - {strategy.replace('_', ' ').title()} Case {i + 1}",
                description=f"Automatically generated {strategy} test case",
                inputs=test_case_data["inputs"],
                expected_outputs=test_case_data.get("expected_outputs"),
                tags=["generated", "diverse", strategy],
                is_golden=False,
                signature_id=signature_id
            )
            
            # Mark as generated
            cursor.execute("""
                UPDATE prompt_studio_test_cases
                SET is_generated = 1
                WHERE id = ?
            """, (created["id"],))
            conn.commit()
            
            generated_cases.append(created)
        
        logger.info(f"Generated {len(generated_cases)} diverse test cases")
        return generated_cases
    
    def generate_from_existing_data(self, project_id: int, source_data: List[Dict[str, Any]],
                                   signature_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Generate test cases from existing data samples.
        
        Args:
            project_id: Project ID
            source_data: List of data samples to convert to test cases
            signature_id: Optional signature ID
            
        Returns:
            List of generated test cases
        """
        generated_cases = []
        
        for idx, data in enumerate(source_data):
            # Extract inputs and expected outputs
            inputs = data.get("inputs", data)
            expected_outputs = data.get("outputs") or data.get("expected_outputs")
            
            # Generate name
            name = data.get("name") or f"Test from Data {idx + 1}"
            
            # Create test case
            created = self.manager.create_test_case(
                project_id=project_id,
                name=name,
                description=data.get("description", "Generated from existing data"),
                inputs=inputs,
                expected_outputs=expected_outputs,
                tags=data.get("tags", ["generated", "from_data"]),
                is_golden=data.get("is_golden", False),
                signature_id=signature_id
            )
            
            # Mark as generated
            conn = self.db.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE prompt_studio_test_cases
                SET is_generated = 1
                WHERE id = ?
            """, (created["id"],))
            conn.commit()
            
            generated_cases.append(created)
        
        logger.info(f"Generated {len(generated_cases)} test cases from existing data")
        return generated_cases
    
    ####################################################################################################################
    # Helper Methods
    
    def _generate_single_case_from_description(self, description: str, index: int,
                                              input_schema: Optional[List] = None,
                                              output_schema: Optional[List] = None) -> Dict[str, Any]:
        """
        Generate a single test case from description.
        
        Args:
            description: Task description
            index: Test case index
            input_schema: Optional input schema
            output_schema: Optional output schema
            
        Returns:
            Test case data
        """
        # Parse description to identify task type
        description_lower = description.lower()
        
        # Determine task type
        if "summariz" in description_lower:
            return self._generate_summarization_case(index, input_schema, output_schema)
        elif "classif" in description_lower:
            return self._generate_classification_case(index, input_schema, output_schema)
        elif "extract" in description_lower:
            return self._generate_extraction_case(index, input_schema, output_schema)
        elif "translat" in description_lower:
            return self._generate_translation_case(index, input_schema, output_schema)
        elif "question" in description_lower or "answer" in description_lower:
            return self._generate_qa_case(index, input_schema, output_schema)
        else:
            return self._generate_generic_case(index, input_schema, output_schema)
    
    def _generate_summarization_case(self, index: int, input_schema: Optional[List] = None,
                                    output_schema: Optional[List] = None) -> Dict[str, Any]:
        """Generate a summarization test case."""
        texts = [
            "The quick brown fox jumps over the lazy dog. This pangram contains all letters of the alphabet.",
            "Artificial intelligence is transforming industries worldwide. From healthcare to finance, AI applications are becoming increasingly sophisticated.",
            "Climate change represents one of the most pressing challenges of our time. Rising temperatures and extreme weather events demand immediate action.",
            "The human brain contains approximately 86 billion neurons. These neurons communicate through trillions of synapses.",
            "Space exploration has captivated humanity for decades. Recent advances in technology have made Mars colonization a realistic possibility."
        ]
        
        text = texts[index % len(texts)]
        
        return {
            "name": f"Summarization Test {index}",
            "description": "Test case for text summarization",
            "inputs": {"text": text, "max_length": 50},
            "expected_outputs": {"summary": f"Summary of text about {text.split('.')[0][:20]}..."},
            "tags": ["summarization", "generated"]
        }
    
    def _generate_classification_case(self, index: int, input_schema: Optional[List] = None,
                                     output_schema: Optional[List] = None) -> Dict[str, Any]:
        """Generate a classification test case."""
        samples = [
            ("I love this product! Best purchase ever!", "positive"),
            ("This is terrible. Complete waste of money.", "negative"),
            ("It's okay, nothing special.", "neutral"),
            ("Amazing quality and fast shipping!", "positive"),
            ("Disappointed with the results.", "negative")
        ]
        
        text, label = samples[index % len(samples)]
        
        return {
            "name": f"Classification Test {index}",
            "description": "Test case for sentiment classification",
            "inputs": {"text": text},
            "expected_outputs": {"sentiment": label, "confidence": 0.85},
            "tags": ["classification", "sentiment", "generated"]
        }
    
    def _generate_extraction_case(self, index: int, input_schema: Optional[List] = None,
                                 output_schema: Optional[List] = None) -> Dict[str, Any]:
        """Generate an extraction test case."""
        samples = [
            "John Smith was born on January 15, 1990 in New York City.",
            "The meeting is scheduled for March 20, 2024 at 2:00 PM.",
            "Contact us at info@example.com or call 555-1234.",
            "The product costs $99.99 and ships within 3-5 business days.",
            "Dr. Jane Doe published her research in Nature journal in 2023."
        ]
        
        text = samples[index % len(samples)]
        
        return {
            "name": f"Extraction Test {index}",
            "description": "Test case for information extraction",
            "inputs": {"text": text},
            "expected_outputs": {"entities": ["person", "date", "location"]},
            "tags": ["extraction", "NER", "generated"]
        }
    
    def _generate_translation_case(self, index: int, input_schema: Optional[List] = None,
                                  output_schema: Optional[List] = None) -> Dict[str, Any]:
        """Generate a translation test case."""
        samples = [
            ("Hello, how are you?", "Bonjour, comment allez-vous?"),
            ("Thank you very much.", "Merci beaucoup."),
            ("Good morning!", "Bonjour!"),
            ("See you later.", "À plus tard."),
            ("I love learning languages.", "J'aime apprendre les langues.")
        ]
        
        source, target = samples[index % len(samples)]
        
        return {
            "name": f"Translation Test {index}",
            "description": "Test case for language translation",
            "inputs": {"text": source, "target_language": "French"},
            "expected_outputs": {"translation": target},
            "tags": ["translation", "generated"]
        }
    
    def _generate_qa_case(self, index: int, input_schema: Optional[List] = None,
                         output_schema: Optional[List] = None) -> Dict[str, Any]:
        """Generate a Q&A test case."""
        samples = [
            ("What is the capital of France?", "Paris"),
            ("How many planets are in our solar system?", "Eight"),
            ("Who wrote Romeo and Juliet?", "William Shakespeare"),
            ("What is the speed of light?", "299,792,458 meters per second"),
            ("When did World War II end?", "1945")
        ]
        
        question, answer = samples[index % len(samples)]
        
        return {
            "name": f"Q&A Test {index}",
            "description": "Test case for question answering",
            "inputs": {"question": question},
            "expected_outputs": {"answer": answer},
            "tags": ["qa", "generated"]
        }
    
    def _generate_generic_case(self, index: int, input_schema: Optional[List] = None,
                              output_schema: Optional[List] = None) -> Dict[str, Any]:
        """Generate a generic test case based on schema."""
        inputs = {}
        expected_outputs = {}
        
        # Generate inputs based on schema
        if input_schema:
            for field in input_schema:
                if isinstance(field, dict):
                    field_name = field.get("name", f"field_{len(inputs)}")
                    field_type = field.get("type", "string")
                    inputs[field_name] = self._generate_value_for_type(field_type, index)
        else:
            # Default generic inputs
            inputs = {
                "input": f"Test input {index}",
                "parameter": index,
                "flag": index % 2 == 0
            }
        
        # Generate outputs based on schema
        if output_schema:
            for field in output_schema:
                if isinstance(field, dict):
                    field_name = field.get("name", f"output_{len(expected_outputs)}")
                    field_type = field.get("type", "string")
                    expected_outputs[field_name] = self._generate_value_for_type(field_type, index)
        else:
            # Default generic outputs
            expected_outputs = {
                "result": f"Expected result {index}",
                "status": "success"
            }
        
        return {
            "name": f"Generic Test {index}",
            "description": "Automatically generated generic test case",
            "inputs": inputs,
            "expected_outputs": expected_outputs,
            "tags": ["generic", "generated"]
        }
    
    def _generate_case_by_strategy(self, input_schema: List, output_schema: List,
                                  strategy: str, index: int) -> Dict[str, Any]:
        """
        Generate a test case using a specific strategy.
        
        Args:
            input_schema: Input field schema
            output_schema: Output field schema
            strategy: Generation strategy
            index: Test case index
            
        Returns:
            Test case data
        """
        inputs = {}
        expected_outputs = {}
        
        # Generate inputs based on strategy
        for field in input_schema:
            if isinstance(field, dict):
                field_name = field.get("name", f"field_{len(inputs)}")
                field_type = field.get("type", "string")
                
                if strategy == "edge_case":
                    inputs[field_name] = self._generate_edge_case_value(field_type)
                elif strategy == "minimal":
                    inputs[field_name] = self._generate_minimal_value(field_type)
                elif strategy == "maximal":
                    inputs[field_name] = self._generate_maximal_value(field_type)
                elif strategy == "complex":
                    inputs[field_name] = self._generate_complex_value(field_type)
                else:  # typical
                    inputs[field_name] = self._generate_typical_value(field_type, index)
        
        # Generate expected outputs
        for field in output_schema:
            if isinstance(field, dict):
                field_name = field.get("name", f"output_{len(expected_outputs)}")
                field_type = field.get("type", "string")
                expected_outputs[field_name] = self._generate_value_for_type(field_type, index)
        
        return {
            "inputs": inputs,
            "expected_outputs": expected_outputs if expected_outputs else None
        }
    
    def _generate_value_for_type(self, field_type: str, index: int) -> Any:
        """Generate a value based on field type."""
        if field_type == "integer":
            return index * 10
        elif field_type == "boolean":
            return index % 2 == 0
        elif field_type == "array":
            return [f"item_{i}" for i in range(min(3, index))]
        elif field_type == "object":
            return {"key": f"value_{index}"}
        elif field_type == "number":
            return index * 1.5
        else:  # string or default
            return f"Sample value {index}"
    
    def _generate_edge_case_value(self, field_type: str) -> Any:
        """Generate edge case value for testing."""
        if field_type == "integer":
            return random.choice([0, -1, 2147483647, -2147483648])
        elif field_type == "string":
            return random.choice(["", " ", "x" * 1000, "🎉emoji🎉", "null", "undefined"])
        elif field_type == "boolean":
            return random.choice([True, False])
        elif field_type == "array":
            return random.choice([[], [""], [None], list(range(100))])
        elif field_type == "object":
            return random.choice([{}, {"": ""}, {"null": None}])
        else:
            return None
    
    def _generate_minimal_value(self, field_type: str) -> Any:
        """Generate minimal valid value."""
        if field_type == "integer":
            return 0
        elif field_type == "string":
            return "a"
        elif field_type == "boolean":
            return False
        elif field_type == "array":
            return []
        elif field_type == "object":
            return {}
        else:
            return None
    
    def _generate_maximal_value(self, field_type: str) -> Any:
        """Generate maximal valid value."""
        if field_type == "integer":
            return 999999
        elif field_type == "string":
            return "x" * 500
        elif field_type == "boolean":
            return True
        elif field_type == "array":
            return list(range(50))
        elif field_type == "object":
            return {f"key_{i}": f"value_{i}" for i in range(20)}
        else:
            return "maximal"
    
    def _generate_complex_value(self, field_type: str) -> Any:
        """Generate complex realistic value."""
        if field_type == "integer":
            return random.randint(100, 10000)
        elif field_type == "string":
            return "This is a complex string with multiple words, punctuation, and numbers: 12345!"
        elif field_type == "boolean":
            return random.choice([True, False])
        elif field_type == "array":
            return [
                {"id": i, "name": f"Item {i}", "value": random.random()}
                for i in range(5)
            ]
        elif field_type == "object":
            return {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "data": {"nested": True, "values": [1, 2, 3]}
            }
        else:
            return "complex"
    
    def _generate_typical_value(self, field_type: str, index: int) -> Any:
        """Generate typical value for normal testing."""
        if field_type == "integer":
            return 100 + index
        elif field_type == "string":
            return f"Typical string {index}"
        elif field_type == "boolean":
            return index % 2 == 0
        elif field_type == "array":
            return [f"item_{i}" for i in range(3)]
        elif field_type == "object":
            return {"id": index, "name": f"Object {index}"}
        else:
            return f"typical_{index}"