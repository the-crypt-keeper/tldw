# evaluation_metrics.py
# Calculate evaluation metrics for prompt testing

import json
from typing import Dict, List, Any, Optional
from difflib import SequenceMatcher
from loguru import logger

class EvaluationMetrics:
    """Calculate various evaluation metrics for prompt testing."""
    
    @staticmethod
    def calculate_exact_match(expected: str, actual: str) -> float:
        """
        Calculate exact match score.
        
        Args:
            expected: Expected output
            actual: Actual output
            
        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        return 1.0 if expected.strip() == actual.strip() else 0.0
    
    @staticmethod
    def calculate_similarity(expected: str, actual: str) -> float:
        """
        Calculate similarity score using sequence matching.
        
        Args:
            expected: Expected output
            actual: Actual output
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        return SequenceMatcher(None, expected, actual).ratio()
    
    @staticmethod
    def calculate_word_overlap(expected: str, actual: str) -> float:
        """
        Calculate word overlap score.
        
        Args:
            expected: Expected output
            actual: Actual output
            
        Returns:
            Overlap score between 0.0 and 1.0
        """
        expected_words = set(expected.lower().split())
        actual_words = set(actual.lower().split())
        
        if not expected_words:
            return 0.0
        
        overlap = len(expected_words & actual_words)
        return overlap / len(expected_words)
    
    @staticmethod
    def calculate_contains_score(expected: str, actual: str) -> float:
        """
        Check if actual output contains expected output.
        
        Args:
            expected: Expected output
            actual: Actual output
            
        Returns:
            1.0 if actual contains expected, 0.0 otherwise
        """
        return 1.0 if expected.lower() in actual.lower() else 0.0
    
    @staticmethod
    def calculate_composite_score(
        expected: Dict[str, Any],
        actual: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate composite score using multiple metrics.
        
        Args:
            expected: Expected output dictionary
            actual: Actual output dictionary
            weights: Weights for different metrics
            
        Returns:
            Composite score between 0.0 and 1.0
        """
        if weights is None:
            weights = {
                "exact_match": 0.3,
                "similarity": 0.3,
                "word_overlap": 0.2,
                "contains": 0.2
            }
        
        expected_str = str(expected.get("response", ""))
        actual_str = str(actual.get("response", ""))
        
        if not expected_str:
            return 1.0  # No expected output means any output is valid
        
        scores = {
            "exact_match": EvaluationMetrics.calculate_exact_match(expected_str, actual_str),
            "similarity": EvaluationMetrics.calculate_similarity(expected_str, actual_str),
            "word_overlap": EvaluationMetrics.calculate_word_overlap(expected_str, actual_str),
            "contains": EvaluationMetrics.calculate_contains_score(expected_str, actual_str)
        }
        
        # Calculate weighted average
        total_score = sum(scores[metric] * weights.get(metric, 0) for metric in scores)
        total_weight = sum(weights.get(metric, 0) for metric in scores)
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    @staticmethod
    def calculate_aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate aggregate metrics for multiple test results.
        
        Args:
            results: List of test results
            
        Returns:
            Aggregate metrics dictionary
        """
        if not results:
            return {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "average_score": 0.0,
                "pass_rate": 0.0,
                "min_score": 0.0,
                "max_score": 0.0
            }
        
        scores = []
        passed = 0
        failed = 0
        
        for result in results:
            score = result.get("score", 0.0)
            scores.append(score)
            
            if score >= 0.5:  # Pass threshold
                passed += 1
            else:
                failed += 1
        
        return {
            "total_tests": len(results),
            "passed": passed,
            "failed": failed,
            "average_score": sum(scores) / len(scores),
            "pass_rate": passed / len(results),
            "min_score": min(scores),
            "max_score": max(scores),
            "median_score": sorted(scores)[len(scores) // 2] if scores else 0.0
        }
    
    @staticmethod
    def compare_evaluations(eval1: Dict[str, Any], eval2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare two evaluation runs.
        
        Args:
            eval1: First evaluation metrics
            eval2: Second evaluation metrics
            
        Returns:
            Comparison results
        """
        return {
            "score_improvement": eval2.get("average_score", 0) - eval1.get("average_score", 0),
            "pass_rate_improvement": eval2.get("pass_rate", 0) - eval1.get("pass_rate", 0),
            "tests_passed_change": eval2.get("passed", 0) - eval1.get("passed", 0),
            "better_evaluation": "eval2" if eval2.get("average_score", 0) > eval1.get("average_score", 0) else "eval1"
        }