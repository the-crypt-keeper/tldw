"""
Dataset loaders for various benchmark formats.

Supports loading from:
- Local files (JSON, JSONL, CSV)
- URLs
- HuggingFace datasets
- Custom formats
"""

import json
import csv
import requests
from typing import List, Dict, Any, Optional, Generator, Tuple
from pathlib import Path
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Base class for dataset loading."""
    
    @staticmethod
    def load_json(source: str) -> List[Dict[str, Any]]:
        """Load JSON dataset from file or URL."""
        if source.startswith(('http://', 'https://')):
            response = requests.get(source)
            response.raise_for_status()
            data = response.json()
        else:
            with open(source, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Try common keys
            for key in ['data', 'items', 'questions', 'examples', 'samples', 'dataset']:
                if key in data and isinstance(data[key], list):
                    return data[key]
            # If no common key, wrap in list
            return [data]
        else:
            raise ValueError(f"Unexpected JSON structure: {type(data)}")
    
    @staticmethod
    def load_jsonl(source: str) -> List[Dict[str, Any]]:
        """Load JSONL dataset from file or URL."""
        data = []
        
        if source.startswith(('http://', 'https://')):
            response = requests.get(source, stream=True)
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    data.append(json.loads(line))
        else:
            with open(source, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        
        return data
    
    @staticmethod
    def load_csv(source: str, delimiter: str = ',') -> List[Dict[str, Any]]:
        """Load CSV dataset from file or URL."""
        data = []
        
        if source.startswith(('http://', 'https://')):
            response = requests.get(source)
            response.raise_for_status()
            lines = response.text.strip().split('\n')
            reader = csv.DictReader(lines, delimiter=delimiter)
        else:
            with open(source, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=delimiter)
        
        for row in reader:
            data.append(dict(row))
        
        return data
    
    @staticmethod
    def load_huggingface(dataset_id: str, split: str = 'test', 
                        limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load dataset from HuggingFace.
        
        Note: This requires the 'datasets' library to be installed.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("HuggingFace datasets library not installed. "
                        "Install with: pip install datasets")
            return []
        
        try:
            # Load dataset
            dataset = load_dataset(dataset_id, split=split)
            
            # Convert to list of dicts
            data = []
            for i, item in enumerate(dataset):
                if limit and i >= limit:
                    break
                data.append(dict(item))
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load HuggingFace dataset {dataset_id}: {e}")
            return []
    
    @staticmethod
    def stream_large_file(source: str, format: str = 'jsonl', 
                         chunk_size: int = 1000) -> Generator[List[Dict[str, Any]], None, None]:
        """Stream large datasets in chunks."""
        if format == 'jsonl':
            chunk = []
            
            if source.startswith(('http://', 'https://')):
                response = requests.get(source, stream=True)
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        chunk.append(json.loads(line))
                        if len(chunk) >= chunk_size:
                            yield chunk
                            chunk = []
            else:
                with open(source, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            chunk.append(json.loads(line))
                            if len(chunk) >= chunk_size:
                                yield chunk
                                chunk = []
            
            # Yield remaining items
            if chunk:
                yield chunk
        
        else:
            # For non-streaming formats, load all and chunk
            if format == 'json':
                data = DatasetLoader.load_json(source)
            elif format == 'csv':
                data = DatasetLoader.load_csv(source)
            else:
                raise ValueError(f"Unsupported format for streaming: {format}")
            
            for i in range(0, len(data), chunk_size):
                yield data[i:i + chunk_size]


class BenchmarkDatasetLoader:
    """Specialized loaders for specific benchmarks."""
    
    @staticmethod
    def load_mmlu_pro(source: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load MMLU Pro dataset."""
        if not source:
            # Default to HuggingFace
            return DatasetLoader.load_huggingface("TIGER-Lab/MMLU-Pro", split="test")
        
        if source.startswith("huggingface://"):
            dataset_id = source.replace("huggingface://", "")
            return DatasetLoader.load_huggingface(dataset_id, split="test")
        else:
            # Load from file/URL
            if source.endswith('.jsonl'):
                return DatasetLoader.load_jsonl(source)
            else:
                return DatasetLoader.load_json(source)
    
    @staticmethod
    def load_simple_bench(source: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load Simple Bench dataset."""
        if not source:
            source = "https://raw.githubusercontent.com/simple-bench/SimpleBench/main/simple_bench_public.json"
        
        data = DatasetLoader.load_json(source)
        
        # Simple Bench has questions embedded in prompt with choices
        processed = []
        for item in data:
            # Extract choices from prompt (they're embedded as A., B., etc.)
            prompt = item.get("prompt", "")
            
            # Basic parsing - would need more sophisticated parsing for production
            import re
            choices_pattern = r'([A-F])\.\s*([^\n]+)'
            matches = re.findall(choices_pattern, prompt)
            
            choices = [match[1].strip() for match in matches]
            
            processed.append({
                "question": prompt.split('\n\n')[0] if '\n\n' in prompt else prompt,
                "choices": choices,
                "correct_answer": item.get("answer", ""),
                "question_id": item.get("question_id", ""),
                "_original": item
            })
        
        return processed
    
    @staticmethod
    def load_aider_polyglot(source: Optional[str] = None, 
                           language: str = "python") -> List[Dict[str, Any]]:
        """Load Aider Polyglot benchmark for a specific language."""
        if not source:
            source = f"https://raw.githubusercontent.com/Aider-AI/polyglot-benchmark/main/{language}"
        
        # This would need custom parsing based on Aider's format
        # For now, return empty list as placeholder
        logger.warning("Aider Polyglot loader not fully implemented")
        return []
    
    @staticmethod
    def load_swe_bench(source: Optional[str] = None, 
                      subset: str = "lite") -> List[Dict[str, Any]]:
        """Load SWE-bench dataset."""
        if not source:
            dataset_id = f"princeton-nlp/SWE-bench_{subset}" if subset else "princeton-nlp/SWE-bench"
            return DatasetLoader.load_huggingface(dataset_id, split="test")
        
        if source.startswith("huggingface://"):
            dataset_id = source.replace("huggingface://", "")
            return DatasetLoader.load_huggingface(dataset_id, split="test")
        else:
            # Load from file/URL
            if source.endswith('.jsonl'):
                return DatasetLoader.load_jsonl(source)
            else:
                return DatasetLoader.load_json(source)
    
    @staticmethod
    def load_gpqa(source: Optional[str] = None, 
                 subset: str = "diamond") -> List[Dict[str, Any]]:
        """Load GPQA dataset."""
        if not source:
            # Default to HuggingFace
            dataset_id = "Idavidrein/gpqa"
            data = DatasetLoader.load_huggingface(dataset_id, split="train")
            
            # Filter by subset if specified
            if subset and subset != "all":
                data = [item for item in data if item.get("subset") == subset]
            
            return data
        
        if source.startswith("huggingface://"):
            dataset_id = source.replace("huggingface://", "")
            return DatasetLoader.load_huggingface(dataset_id)
        else:
            # Load from file/URL
            if source.endswith('.jsonl'):
                return DatasetLoader.load_jsonl(source)
            else:
                return DatasetLoader.load_json(source)
    
    @staticmethod
    def load_bfcl(source: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load Berkeley Function Calling Leaderboard dataset."""
        if not source:
            return DatasetLoader.load_huggingface(
                "gorilla-llm/berkeley-function-calling-leaderboard", 
                split="test"
            )
        
        if source.startswith("huggingface://"):
            dataset_id = source.replace("huggingface://", "")
            return DatasetLoader.load_huggingface(dataset_id)
        else:
            # Load from file/URL
            if source.endswith('.jsonl'):
                return DatasetLoader.load_jsonl(source)
            else:
                return DatasetLoader.load_json(source)
    
    @staticmethod
    def load_simpleqa(source: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load SimpleQA dataset."""
        if not source:
            # Try to load from HuggingFace if available
            try:
                return DatasetLoader.load_huggingface("openai/simple-qa", split="test")
            except:
                # Fallback to sample data
                logger.warning("SimpleQA dataset not found, using sample data")
                from tldw_Server_API.app.core.Evaluations.simpleqa_eval import SimpleQADataset
                return SimpleQADataset.create_sample_dataset()
        
        if source.startswith("huggingface://"):
            dataset_id = source.replace("huggingface://", "")
            return DatasetLoader.load_huggingface(dataset_id)
        elif source.startswith("http"):
            # Load from URL
            if source.endswith('.jsonl'):
                return DatasetLoader.load_jsonl(source)
            else:
                return DatasetLoader.load_json(source)
        else:
            # Load from local file
            from tldw_Server_API.app.core.Evaluations.simpleqa_eval import SimpleQADataset
            return SimpleQADataset.load_from_file(source)


def load_benchmark_dataset(benchmark_name: str, 
                          source: Optional[str] = None,
                          limit: Optional[int] = None,
                          **kwargs) -> List[Dict[str, Any]]:
    """Load dataset for a specific benchmark.
    
    Args:
        benchmark_name: Name of the benchmark
        source: Optional custom source (file, URL, or HF dataset)
        limit: Maximum number of samples to load
        **kwargs: Additional benchmark-specific parameters
    
    Returns:
        List of dataset items
    """
    # Map benchmark names to loaders
    loaders = {
        "mmlu_pro": BenchmarkDatasetLoader.load_mmlu_pro,
        "simple_bench": BenchmarkDatasetLoader.load_simple_bench,
        "simpleqa": BenchmarkDatasetLoader.load_simpleqa,
        "aider_polyglot": BenchmarkDatasetLoader.load_aider_polyglot,
        "swe_bench": BenchmarkDatasetLoader.load_swe_bench,
        "gpqa": BenchmarkDatasetLoader.load_gpqa,
        "gpqa_diamond": lambda s: BenchmarkDatasetLoader.load_gpqa(s, subset="diamond"),
        "bfcl": BenchmarkDatasetLoader.load_bfcl,
    }
    
    if benchmark_name in loaders:
        data = loaders[benchmark_name](source)
    else:
        # Try generic loading based on source format
        if not source:
            logger.error(f"No source specified for unknown benchmark: {benchmark_name}")
            return []
        
        if source.startswith("huggingface://"):
            dataset_id = source.replace("huggingface://", "")
            data = DatasetLoader.load_huggingface(dataset_id)
        elif source.endswith('.jsonl'):
            data = DatasetLoader.load_jsonl(source)
        elif source.endswith('.json'):
            data = DatasetLoader.load_json(source)
        elif source.endswith('.csv'):
            data = DatasetLoader.load_csv(source)
        else:
            logger.error(f"Cannot determine format for source: {source}")
            return []
    
    # Apply limit if specified
    if limit and limit > 0:
        data = data[:limit]
    
    return data


def validate_dataset_format(data: List[Dict[str, Any]], 
                           required_fields: List[str]) -> Tuple[bool, List[str]]:
    """Validate that dataset has required fields.
    
    Args:
        data: Dataset to validate
        required_fields: List of required field names
    
    Returns:
        Tuple of (is_valid, list_of_missing_fields)
    """
    if not data:
        return False, ["Dataset is empty"]
    
    # Check first item for fields
    first_item = data[0]
    missing_fields = []
    
    for field in required_fields:
        if field not in first_item:
            missing_fields.append(field)
    
    return len(missing_fields) == 0, missing_fields