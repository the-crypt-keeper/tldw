# evaluation_metrics.py
# Evaluation metrics and scoring for Prompt Studio

import json
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import numpy as np
from loguru import logger

# Import for text similarity
try:
    from difflib import SequenceMatcher
    from rapidfuzz import fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    logger.warning("rapidfuzz not available, using basic similarity metrics")

########################################################################################################################
# Metric Types

class MetricType(str, Enum):
    """Types of evaluation metrics."""
    
    # Exact matching
    EXACT_MATCH = "exact_match"
    CASE_INSENSITIVE_MATCH = "case_insensitive_match"
    
    # Fuzzy matching
    FUZZY_MATCH = "fuzzy_match"
    LEVENSHTEIN = "levenshtein"
    TOKEN_OVERLAP = "token_overlap"
    
    # Semantic similarity
    SEMANTIC_SIMILARITY = "semantic_similarity"
    COSINE_SIMILARITY = "cosine_similarity"
    
    # Structured data
    JSON_MATCH = "json_match"
    JSON_SCHEMA_VALID = "json_schema_valid"
    
    # Classification
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    
    # Regression
    MAE = "mean_absolute_error"
    MSE = "mean_squared_error"
    RMSE = "root_mean_squared_error"
    
    # Custom
    CUSTOM = "custom"
    REGEX_MATCH = "regex_match"
    CONTAINS = "contains"
    LENGTH_MATCH = "length_match"

########################################################################################################################
# Evaluation Metrics

class EvaluationMetrics:
    """Calculate various evaluation metrics for prompt outputs."""
    
    def __init__(self):
        """Initialize EvaluationMetrics."""
        self.metrics = {}
        self._register_metrics()
    
    def _register_metrics(self):
        """Register available metrics."""
        self.metrics = {
            MetricType.EXACT_MATCH: self.exact_match,
            MetricType.CASE_INSENSITIVE_MATCH: self.case_insensitive_match,
            MetricType.FUZZY_MATCH: self.fuzzy_match,
            MetricType.LEVENSHTEIN: self.levenshtein_similarity,
            MetricType.TOKEN_OVERLAP: self.token_overlap,
            MetricType.JSON_MATCH: self.json_match,
            MetricType.JSON_SCHEMA_VALID: self.json_schema_valid,
            MetricType.ACCURACY: self.accuracy,
            MetricType.CONTAINS: self.contains,
            MetricType.REGEX_MATCH: self.regex_match,
            MetricType.LENGTH_MATCH: self.length_match
        }
    
    ####################################################################################################################
    # Main Evaluation Method
    
    def evaluate(self, output: Any, expected: Any, 
                metrics: List[MetricType] = None) -> Dict[str, float]:
        """
        Evaluate output against expected value using specified metrics.
        
        Args:
            output: Model output
            expected: Expected output
            metrics: List of metrics to calculate (None = auto-detect)
            
        Returns:
            Dictionary of metric scores
        """
        if metrics is None:
            metrics = self._auto_detect_metrics(output, expected)
        
        results = {}
        for metric in metrics:
            if metric in self.metrics:
                try:
                    score = self.metrics[metric](output, expected)
                    results[metric.value] = score
                except Exception as e:
                    logger.error(f"Error calculating {metric}: {e}")
                    results[metric.value] = 0.0
            else:
                logger.warning(f"Unknown metric: {metric}")
        
        # Add aggregate score
        if results:
            results["aggregate_score"] = np.mean(list(results.values()))
        
        return results
    
    def _auto_detect_metrics(self, output: Any, expected: Any) -> List[MetricType]:
        """Auto-detect appropriate metrics based on data types."""
        metrics = []
        
        # Convert to strings for comparison
        output_str = str(output)
        expected_str = str(expected)
        
        # Check data types
        if isinstance(expected, dict) or isinstance(output, dict):
            metrics.append(MetricType.JSON_MATCH)
        elif isinstance(expected, (int, float)) and isinstance(output, (int, float)):
            metrics.append(MetricType.MAE)
        elif isinstance(expected, bool) or expected_str.lower() in ["true", "false"]:
            metrics.append(MetricType.EXACT_MATCH)
        else:
            # Text comparison
            metrics.extend([
                MetricType.EXACT_MATCH,
                MetricType.FUZZY_MATCH,
                MetricType.TOKEN_OVERLAP
            ])
        
        return metrics
    
    ####################################################################################################################
    # Text Matching Metrics
    
    def exact_match(self, output: Any, expected: Any) -> float:
        """Exact string match."""
        return 1.0 if str(output) == str(expected) else 0.0
    
    def case_insensitive_match(self, output: Any, expected: Any) -> float:
        """Case-insensitive string match."""
        return 1.0 if str(output).lower() == str(expected).lower() else 0.0
    
    def fuzzy_match(self, output: Any, expected: Any) -> float:
        """Fuzzy string matching using token sort ratio."""
        if FUZZY_AVAILABLE:
            return fuzz.token_sort_ratio(str(output), str(expected)) / 100.0
        else:
            # Fallback to SequenceMatcher
            return SequenceMatcher(None, str(output), str(expected)).ratio()
    
    def levenshtein_similarity(self, output: Any, expected: Any) -> float:
        """Levenshtein distance normalized to similarity score."""
        if FUZZY_AVAILABLE:
            return fuzz.ratio(str(output), str(expected)) / 100.0
        else:
            # Simple character-level similarity
            s1, s2 = str(output), str(expected)
            longer = max(len(s1), len(s2))
            if longer == 0:
                return 1.0
            
            # Use SequenceMatcher as approximation
            return SequenceMatcher(None, s1, s2).ratio()
    
    def token_overlap(self, output: Any, expected: Any) -> float:
        """Token overlap between output and expected."""
        # Tokenize
        output_tokens = set(str(output).lower().split())
        expected_tokens = set(str(expected).lower().split())
        
        if not expected_tokens:
            return 1.0 if not output_tokens else 0.0
        
        # Calculate Jaccard similarity
        intersection = output_tokens & expected_tokens
        union = output_tokens | expected_tokens
        
        if not union:
            return 1.0
        
        return len(intersection) / len(union)
    
    def contains(self, output: Any, expected: Any) -> float:
        """Check if output contains expected substring."""
        return 1.0 if str(expected) in str(output) else 0.0
    
    def regex_match(self, output: Any, pattern: str) -> float:
        """Match output against regex pattern."""
        try:
            if re.search(pattern, str(output)):
                return 1.0
        except:
            pass
        return 0.0
    
    def length_match(self, output: Any, expected: Any, tolerance: float = 0.2) -> float:
        """Compare output length to expected length."""
        output_len = len(str(output))
        expected_len = len(str(expected))
        
        if expected_len == 0:
            return 1.0 if output_len == 0 else 0.0
        
        diff_ratio = abs(output_len - expected_len) / expected_len
        
        if diff_ratio <= tolerance:
            return 1.0 - (diff_ratio / tolerance)
        return 0.0
    
    ####################################################################################################################
    # Structured Data Metrics
    
    def json_match(self, output: Any, expected: Any) -> float:
        """Compare JSON structures."""
        try:
            # Parse if strings
            if isinstance(output, str):
                output = json.loads(output)
            if isinstance(expected, str):
                expected = json.loads(expected)
            
            # Deep comparison
            return self._deep_dict_compare(output, expected)
            
        except json.JSONDecodeError:
            return 0.0
        except Exception as e:
            logger.error(f"JSON comparison error: {e}")
            return 0.0
    
    def _deep_dict_compare(self, d1: Any, d2: Any, path: str = "") -> float:
        """Deep comparison of dictionaries/lists."""
        if type(d1) != type(d2):
            return 0.0
        
        if isinstance(d1, dict):
            if set(d1.keys()) != set(d2.keys()):
                # Partial credit for matching keys
                common_keys = set(d1.keys()) & set(d2.keys())
                all_keys = set(d1.keys()) | set(d2.keys())
                key_score = len(common_keys) / len(all_keys) if all_keys else 0
                
                # Compare values for common keys
                value_scores = []
                for key in common_keys:
                    value_scores.append(
                        self._deep_dict_compare(d1[key], d2[key], f"{path}.{key}")
                    )
                
                if value_scores:
                    return (key_score + np.mean(value_scores)) / 2
                return key_score
            
            # All keys match, compare values
            scores = []
            for key in d1:
                scores.append(
                    self._deep_dict_compare(d1[key], d2[key], f"{path}.{key}")
                )
            return np.mean(scores) if scores else 1.0
        
        elif isinstance(d1, list):
            if len(d1) != len(d2):
                # Partial credit based on length similarity
                len_score = 1.0 - abs(len(d1) - len(d2)) / max(len(d1), len(d2), 1)
                
                # Compare overlapping elements
                min_len = min(len(d1), len(d2))
                if min_len > 0:
                    scores = []
                    for i in range(min_len):
                        scores.append(
                            self._deep_dict_compare(d1[i], d2[i], f"{path}[{i}]")
                        )
                    return (len_score + np.mean(scores)) / 2
                return len_score
            
            # Same length, compare elements
            scores = []
            for i, (item1, item2) in enumerate(zip(d1, d2)):
                scores.append(
                    self._deep_dict_compare(item1, item2, f"{path}[{i}]")
                )
            return np.mean(scores) if scores else 1.0
        
        else:
            # Primitive comparison
            return 1.0 if d1 == d2 else 0.0
    
    def json_schema_valid(self, output: Any, schema: Dict[str, Any]) -> float:
        """Validate output against JSON schema."""
        try:
            import jsonschema
            
            if isinstance(output, str):
                output = json.loads(output)
            
            jsonschema.validate(output, schema)
            return 1.0
            
        except ImportError:
            logger.warning("jsonschema not available")
            return 0.0
        except jsonschema.ValidationError:
            return 0.0
        except Exception:
            return 0.0
    
    ####################################################################################################################
    # Classification Metrics
    
    def accuracy(self, output: Any, expected: Any) -> float:
        """Classification accuracy."""
        return self.exact_match(output, expected)
    
    def precision_recall_f1(self, outputs: List[Any], expected: List[Any], 
                           positive_class: Any = True) -> Dict[str, float]:
        """Calculate precision, recall, and F1 score for binary classification."""
        if len(outputs) != len(expected):
            return {"precision": 0, "recall": 0, "f1": 0}
        
        true_positives = sum(1 for o, e in zip(outputs, expected) 
                            if o == positive_class and e == positive_class)
        false_positives = sum(1 for o, e in zip(outputs, expected) 
                             if o == positive_class and e != positive_class)
        false_negatives = sum(1 for o, e in zip(outputs, expected) 
                             if o != positive_class and e == positive_class)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    
    ####################################################################################################################
    # Numerical Metrics
    
    def mean_absolute_error(self, output: Any, expected: Any) -> float:
        """Mean absolute error for numerical values."""
        try:
            output_val = float(output)
            expected_val = float(expected)
            
            error = abs(output_val - expected_val)
            
            # Normalize to 0-1 score (inverse of error)
            # Assume max acceptable error is 100
            max_error = 100
            score = max(0, 1 - (error / max_error))
            
            return score
            
        except (ValueError, TypeError):
            return 0.0
    
    def mean_squared_error(self, outputs: List[float], expected: List[float]) -> float:
        """Mean squared error for numerical predictions."""
        if len(outputs) != len(expected):
            return 0.0
        
        try:
            mse = np.mean([(o - e) ** 2 for o, e in zip(outputs, expected)])
            # Convert to 0-1 score
            return 1.0 / (1.0 + mse)
        except:
            return 0.0

########################################################################################################################
# Evaluation Aggregator

class EvaluationAggregator:
    """Aggregates evaluation results across multiple test runs."""
    
    def __init__(self):
        """Initialize EvaluationAggregator."""
        self.evaluator = EvaluationMetrics()
    
    def aggregate_results(self, test_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate results from multiple test runs.
        
        Args:
            test_runs: List of test run results
            
        Returns:
            Aggregated metrics and statistics
        """
        if not test_runs:
            return {}
        
        # Collect all scores
        all_scores = {}
        execution_times = []
        token_counts = []
        costs = []
        
        for run in test_runs:
            # Get scores
            scores = run.get("scores", {})
            for metric, score in scores.items():
                if metric not in all_scores:
                    all_scores[metric] = []
                all_scores[metric].append(score)
            
            # Collect performance metrics
            if "execution_time_ms" in run:
                execution_times.append(run["execution_time_ms"])
            if "tokens_used" in run:
                token_counts.append(run["tokens_used"])
            if "cost_estimate" in run:
                costs.append(run["cost_estimate"])
        
        # Calculate statistics
        aggregated = {
            "total_runs": len(test_runs),
            "metrics": {}
        }
        
        # Aggregate each metric
        for metric, scores in all_scores.items():
            aggregated["metrics"][metric] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "median": np.median(scores)
            }
        
        # Performance statistics
        if execution_times:
            aggregated["execution_time"] = {
                "mean_ms": np.mean(execution_times),
                "std_ms": np.std(execution_times),
                "min_ms": np.min(execution_times),
                "max_ms": np.max(execution_times),
                "total_ms": np.sum(execution_times)
            }
        
        if token_counts:
            aggregated["tokens"] = {
                "mean": np.mean(token_counts),
                "total": np.sum(token_counts)
            }
        
        if costs:
            aggregated["cost"] = {
                "mean": np.mean(costs),
                "total": np.sum(costs)
            }
        
        # Overall score
        if all_scores:
            all_values = [score for scores in all_scores.values() for score in scores]
            aggregated["overall_score"] = np.mean(all_values)
        
        return aggregated
    
    def compare_models(self, results_by_model: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Compare evaluation results across different models.
        
        Args:
            results_by_model: Dictionary mapping model names to test run results
            
        Returns:
            Comparison statistics
        """
        comparison = {
            "models": {},
            "rankings": {}
        }
        
        # Aggregate for each model
        for model_name, test_runs in results_by_model.items():
            comparison["models"][model_name] = self.aggregate_results(test_runs)
        
        # Rank models by different criteria
        if comparison["models"]:
            # Rank by overall score
            scores = {
                model: data.get("overall_score", 0)
                for model, data in comparison["models"].items()
            }
            comparison["rankings"]["by_score"] = sorted(
                scores.items(), key=lambda x: x[1], reverse=True
            )
            
            # Rank by speed
            speeds = {
                model: data.get("execution_time", {}).get("mean_ms", float('inf'))
                for model, data in comparison["models"].items()
            }
            comparison["rankings"]["by_speed"] = sorted(
                speeds.items(), key=lambda x: x[1]
            )
            
            # Rank by cost
            costs = {
                model: data.get("cost", {}).get("total", float('inf'))
                for model, data in comparison["models"].items()
            }
            comparison["rankings"]["by_cost"] = sorted(
                costs.items(), key=lambda x: x[1]
            )
            
            # Best value (score per dollar)
            values = {}
            for model in comparison["models"]:
                score = scores.get(model, 0)
                cost = costs.get(model, 1)
                values[model] = score / cost if cost > 0 else 0
            
            comparison["rankings"]["by_value"] = sorted(
                values.items(), key=lambda x: x[1], reverse=True
            )
        
        return comparison