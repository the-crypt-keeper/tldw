"""
Sample data generators for testing.

Provides functions to generate various types of test data for evaluations.
"""

import random
import string
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta


class SampleDataGenerator:
    """Generate sample data for testing."""
    
    @staticmethod
    def generate_text(min_length: int = 10, max_length: int = 100) -> str:
        """Generate random text of specified length."""
        words = [
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
            "evaluation", "system", "metrics", "performance", "quality", "response",
            "analysis", "data", "model", "test", "sample", "content", "result"
        ]
        
        num_words = random.randint(min_length, max_length) // 5
        return " ".join(random.choices(words, k=num_words))
    
    @staticmethod
    def generate_evaluation_spec() -> Dict[str, Any]:
        """Generate a random evaluation specification."""
        eval_types = ["model_graded", "g_eval", "rag", "response_quality", "custom"]
        metrics = ["accuracy", "relevance", "coherence", "fluency", "factuality", 
                  "completeness", "consistency", "helpfulness"]
        
        return {
            "evaluator_model": random.choice(["gpt-4", "gpt-3.5-turbo", "claude-2", "llama-2"]),
            "eval_type": random.choice(eval_types),
            "metrics": random.sample(metrics, k=random.randint(2, 4)),
            "threshold": round(random.uniform(0.5, 0.9), 2),
            "temperature": round(random.uniform(0.0, 1.0), 1),
            "max_tokens": random.choice([500, 1000, 1500, 2000]),
            "top_p": round(random.uniform(0.8, 1.0), 2)
        }
    
    @staticmethod
    def generate_dataset_samples(count: int = 5) -> List[Dict[str, Any]]:
        """Generate dataset samples."""
        samples = []
        
        for i in range(count):
            samples.append({
                "input": {
                    "question": f"Test question {i}: {SampleDataGenerator.generate_text(5, 20)}",
                    "context": SampleDataGenerator.generate_text(20, 50)
                },
                "expected": {
                    "answer": f"Expected answer {i}: {SampleDataGenerator.generate_text(10, 30)}",
                    "score": round(random.uniform(0.0, 1.0), 2)
                },
                "metadata": {
                    "index": i,
                    "difficulty": random.choice(["easy", "medium", "hard"]),
                    "category": random.choice(["factual", "reasoning", "creative"])
                }
            })
        
        return samples
    
    @staticmethod
    def generate_rag_evaluation_data() -> Dict[str, Any]:
        """Generate RAG evaluation test data."""
        return {
            "query": SampleDataGenerator.generate_text(5, 15),
            "retrieved_contexts": [
                SampleDataGenerator.generate_text(20, 40)
                for _ in range(random.randint(2, 5))
            ],
            "generated_response": SampleDataGenerator.generate_text(30, 60),
            "ground_truth": SampleDataGenerator.generate_text(20, 40),
            "api_name": "openai",
            "metrics": ["relevance", "faithfulness", "answer_similarity", "context_precision"]
        }
    
    @staticmethod
    def generate_geval_data() -> Dict[str, Any]:
        """Generate G-Eval test data."""
        source_text = SampleDataGenerator.generate_text(100, 200)
        
        return {
            "source_text": source_text,
            "summary": SampleDataGenerator.generate_text(20, 40),
            "criteria": random.choice(["coherence", "consistency", "fluency", "relevance"]),
            "model": random.choice(["gpt-4", "gpt-3.5-turbo"]),
            "metadata": {
                "document_type": random.choice(["article", "report", "essay", "paper"]),
                "word_count": len(source_text.split()),
                "language": "en"
            }
        }
    
    @staticmethod
    def generate_response_quality_data() -> Dict[str, Any]:
        """Generate response quality evaluation data."""
        return {
            "prompt": SampleDataGenerator.generate_text(10, 30),
            "response": SampleDataGenerator.generate_text(50, 100),
            "expected_format": "A clear and concise answer",
            "evaluation_criteria": {
                "coherence": "The response should be logically structured",
                "relevance": "The response should directly address the prompt",
                "fluency": "The response should be grammatically correct",
                "factuality": "The response should be accurate",
                "completeness": "The response should fully answer the question"
            },
            "api_name": "openai"
        }
    
    @staticmethod
    def generate_batch_evaluation_request(size: int = 3) -> Dict[str, Any]:
        """Generate batch evaluation request."""
        evaluations = []
        
        for i in range(size):
            eval_type = random.choice(["g_eval", "rag", "response_quality"])
            
            if eval_type == "g_eval":
                data = SampleDataGenerator.generate_geval_data()
            elif eval_type == "rag":
                data = SampleDataGenerator.generate_rag_evaluation_data()
            else:
                data = SampleDataGenerator.generate_response_quality_data()
            
            evaluations.append({
                "name": f"batch_eval_{i}_{uuid.uuid4().hex[:6]}",
                "eval_type": eval_type,
                "data": data,
                "priority": random.choice(["low", "medium", "high"])
            })
        
        return {
            "evaluation_type": "geval",  # Default to one type for batch
            "items": evaluations,
            "parallel_workers": random.randint(1, 4),
            "continue_on_error": not random.choice([True, False]),  # Opposite of fail_fast
            "metadata": {
                "batch_id": f"batch_{uuid.uuid4().hex[:8]}",
                "submitted_at": datetime.utcnow().isoformat(),
                "user": f"test_user_{random.randint(1, 10)}"
            }
        }
    
    @staticmethod
    def generate_webhook_payload() -> Dict[str, Any]:
        """Generate webhook payload."""
        event_types = [
            "evaluation.created", "evaluation.completed", "evaluation.failed",
            "run.started", "run.completed", "run.failed",
            "dataset.created", "dataset.updated"
        ]
        
        return {
            "event": random.choice(event_types),
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "id": f"{uuid.uuid4().hex[:8]}",
                "type": random.choice(["evaluation", "run", "dataset"]),
                "status": random.choice(["pending", "running", "completed", "failed"]),
                "result": {
                    "score": round(random.uniform(0.0, 1.0), 2),
                    "metrics": {
                        "accuracy": round(random.uniform(0.0, 1.0), 2),
                        "relevance": round(random.uniform(0.0, 1.0), 2)
                    }
                } if random.choice([True, False]) else None
            },
            "metadata": {
                "source": "test_suite",
                "version": "1.0.0"
            }
        }
    
    @staticmethod
    def generate_error_scenarios() -> List[Dict[str, Any]]:
        """Generate various error scenarios for testing."""
        return [
            {
                "type": "invalid_input",
                "data": {
                    "eval_type": "invalid_type",
                    "eval_spec": {}
                },
                "expected_error": "Invalid evaluation type"
            },
            {
                "type": "missing_required",
                "data": {
                    "eval_type": "g_eval"
                    # Missing required fields
                },
                "expected_error": "Missing required field"
            },
            {
                "type": "rate_limit",
                "data": {
                    "requests_count": 100,
                    "time_window": 60
                },
                "expected_error": "Rate limit exceeded"
            },
            {
                "type": "timeout",
                "data": {
                    "timeout_seconds": 0.001,
                    "operation": "evaluation"
                },
                "expected_error": "Operation timed out"
            },
            {
                "type": "llm_failure",
                "data": {
                    "error": "API key invalid",
                    "provider": "openai"
                },
                "expected_error": "LLM API error"
            }
        ]


# Convenience functions for quick data generation
def generate_evaluation_request() -> Dict[str, Any]:
    """Generate a complete evaluation request."""
    generator = SampleDataGenerator()
    
    return {
        "name": f"test_eval_{uuid.uuid4().hex[:8]}",
        "description": generator.generate_text(10, 30),
        "eval_type": random.choice(["model_graded", "g_eval", "rag", "response_quality"]),
        "eval_spec": generator.generate_evaluation_spec(),
        "dataset": generator.generate_dataset_samples(random.randint(2, 5)),
        "metadata": {
            "created_by": f"test_user_{random.randint(1, 10)}",
            "tags": random.sample(["test", "unit", "integration", "sample", "qa"], k=2),
            "version": f"{random.randint(1, 3)}.{random.randint(0, 9)}.{random.randint(0, 9)}"
        }
    }


def generate_run_request(eval_id: str) -> Dict[str, Any]:
    """Generate a run request for an evaluation."""
    generator = SampleDataGenerator()
    
    return {
        "eval_id": eval_id,
        "model": random.choice(["gpt-4", "gpt-3.5-turbo", "claude-2"]),
        "parameters": {
            "temperature": round(random.uniform(0.0, 1.0), 1),
            "max_tokens": random.choice([500, 1000, 1500]),
            "top_p": round(random.uniform(0.8, 1.0), 2)
        },
        "dataset_sample_indices": list(range(random.randint(1, 10))),
        "metadata": {
            "run_type": random.choice(["test", "production", "validation"]),
            "triggered_by": random.choice(["manual", "scheduled", "webhook"])
        }
    }