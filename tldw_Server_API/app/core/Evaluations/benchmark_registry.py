"""
Benchmark Registry for managing and running evaluation benchmarks.

Provides a centralized registry for benchmark configurations and mappings
to appropriate evaluation types.
"""

import json
import yaml
from typing import Dict, Any, List, Optional, Type
from pathlib import Path
from dataclasses import dataclass, field
import logging

from tldw_Server_API.app.core.Evaluations.benchmark_utils import (
    BaseEvaluation,
    MultipleChoiceEvaluation,
    CodeGenerationEvaluation,
    InstructionFollowingEvaluation,
    HonestyEvaluation,
    load_dataset_from_json,
    load_dataset_from_jsonl,
    load_dataset_from_url
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark."""
    name: str
    description: str
    evaluation_type: str
    dataset_source: str  # URL, file path, or HuggingFace dataset ID
    dataset_format: str  # json, jsonl, csv, huggingface
    field_mappings: Dict[str, str] = field(default_factory=dict)
    evaluation_params: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "evaluation_type": self.evaluation_type,
            "dataset_source": self.dataset_source,
            "dataset_format": self.dataset_format,
            "field_mappings": self.field_mappings,
            "evaluation_params": self.evaluation_params,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkConfig':
        """Create from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'BenchmarkConfig':
        """Load from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


class BenchmarkRegistry:
    """Registry for managing benchmarks."""
    
    # Mapping of evaluation types to classes
    EVALUATION_TYPES = {
        "multiple_choice": MultipleChoiceEvaluation,
        "code_generation": CodeGenerationEvaluation,
        "instruction_following": InstructionFollowingEvaluation,
        "honesty": HonestyEvaluation,
        "simpleqa": None,  # Loaded dynamically
        "function_calling": None,  # To be implemented
        "multi_turn": None,  # To be implemented
    }
    
    def __init__(self):
        self.benchmarks: Dict[str, BenchmarkConfig] = {}
        self._load_default_benchmarks()
    
    def _load_default_benchmarks(self):
        """Load default benchmark configurations."""
        # SimpleQA
        self.register(BenchmarkConfig(
            name="simpleqa",
            description="OpenAI SimpleQA - Measuring short-form factuality",
            evaluation_type="simpleqa",
            dataset_source="https://github.com/openai/simple-evals",
            dataset_format="jsonl",
            field_mappings={
                "question": "question",
                "answer": "answer",
                "topic": "topic"
            },
            evaluation_params={
                "grading_model": "openai",
                "strict_grading": True,
                "max_tokens": 50
            },
            metadata={
                "paper": "https://openai.com/index/introducing-simpleqa/",
                "size": 4326,
                "topics": ["history", "science", "technology", "art", "geography"]
            }
        ))
        
        # MMLU Pro
        self.register(BenchmarkConfig(
            name="mmlu_pro",
            description="MMLU Pro - Enhanced multiple choice with 10 options and reasoning",
            evaluation_type="multiple_choice",
            dataset_source="https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro",
            dataset_format="huggingface",
            field_mappings={
                "question": "question",
                "choices": "options",
                "correct_answer": "answer_idx",
                "category": "category"
            },
            evaluation_params={
                "num_choices": 10,
                "require_reasoning": True
            },
            metadata={
                "paper": "https://arxiv.org/abs/2406.01574",
                "domains": 14,
                "size": 12000
            }
        ))
        
        # GPQA Diamond
        self.register(BenchmarkConfig(
            name="gpqa_diamond",
            description="Graduate-level science questions (physics, chemistry, biology)",
            evaluation_type="multiple_choice",
            dataset_source="https://huggingface.co/datasets/Idavidrein/gpqa",
            dataset_format="huggingface",
            field_mappings={
                "question": "question",
                "choices": "choices",
                "correct_answer": "correct_answer",
                "category": "subject"
            },
            evaluation_params={
                "num_choices": 4,
                "require_reasoning": True
            },
            metadata={
                "difficulty": "graduate",
                "domains": ["physics", "chemistry", "biology"]
            }
        ))
        
        # Simple Bench
        self.register(BenchmarkConfig(
            name="simple_bench",
            description="Reasoning problems with detailed scenarios",
            evaluation_type="multiple_choice",
            dataset_source="https://github.com/simple-bench/SimpleBench/blob/main/simple_bench_public.json",
            dataset_format="json",
            field_mappings={
                "question": "prompt",
                "correct_answer": "answer"
            },
            evaluation_params={
                "num_choices": 6,
                "require_reasoning": False
            }
        ))
        
        # Aider Polyglot
        self.register(BenchmarkConfig(
            name="aider_polyglot",
            description="Multi-language code generation from Exercism",
            evaluation_type="code_generation",
            dataset_source="https://github.com/Aider-AI/polyglot-benchmark",
            dataset_format="custom",
            field_mappings={
                "problem": "description",
                "test_cases": "tests",
                "language": "language"
            },
            evaluation_params={
                "languages": ["python", "javascript", "rust", "go", "java", "cpp"],
                "run_tests": True
            }
        ))
        
        # BFCL (Berkeley Function Calling Leaderboard)
        self.register(BenchmarkConfig(
            name="bfcl",
            description="Function calling benchmark",
            evaluation_type="function_calling",
            dataset_source="https://huggingface.co/datasets/gorilla-llm/berkeley-function-calling-leaderboard",
            dataset_format="huggingface",
            field_mappings={
                "query": "question",
                "functions": "functions",
                "expected_call": "answers"
            },
            evaluation_params={
                "strict_params": True
            }
        ))
        
        # MASK Benchmark
        self.register(BenchmarkConfig(
            name="mask",
            description="Honesty evaluation benchmark",
            evaluation_type="honesty",
            dataset_source="https://huggingface.co/datasets/mask-benchmark/MASK",
            dataset_format="huggingface",
            field_mappings={
                "question": "question",
                "variations": "variations",
                "expected_consistency": "expected_consistency"
            },
            evaluation_params={
                "check_consistency": True
            }
        ))
        
        # Vending Bench
        self.register(BenchmarkConfig(
            name="vending_bench",
            description="Long-term business simulation",
            evaluation_type="multi_turn",
            dataset_source="https://andonlabs.com/evals/vending-bench",
            dataset_format="custom",
            field_mappings={
                "scenario": "scenario",
                "initial_state": "initial_state",
                "objectives": "objectives"
            },
            evaluation_params={
                "max_turns": 100,
                "track_metrics": ["net_worth", "units_sold", "days_operational"]
            }
        ))
        
        # SWE-bench
        self.register(BenchmarkConfig(
            name="swe_bench",
            description="Software engineering tasks from GitHub issues",
            evaluation_type="code_generation",
            dataset_source="https://huggingface.co/datasets/princeton-nlp/SWE-bench",
            dataset_format="huggingface",
            field_mappings={
                "problem": "problem_statement",
                "test_cases": "test_patch",
                "repo": "repo"
            },
            evaluation_params={
                "language": "python",
                "run_tests": True,
                "use_repo_context": True
            }
        ))
    
    def register(self, config: BenchmarkConfig) -> None:
        """Register a benchmark configuration."""
        self.benchmarks[config.name] = config
        logger.info(f"Registered benchmark: {config.name}")
    
    def unregister(self, name: str) -> None:
        """Unregister a benchmark."""
        if name in self.benchmarks:
            del self.benchmarks[name]
            logger.info(f"Unregistered benchmark: {name}")
    
    def get(self, name: str) -> Optional[BenchmarkConfig]:
        """Get a benchmark configuration."""
        return self.benchmarks.get(name)
    
    def list_benchmarks(self) -> List[str]:
        """List all registered benchmark names."""
        return list(self.benchmarks.keys())
    
    def get_benchmark_info(self, name: str) -> Dict[str, Any]:
        """Get detailed information about a benchmark."""
        config = self.get(name)
        if not config:
            return {}
        
        return {
            "name": config.name,
            "description": config.description,
            "evaluation_type": config.evaluation_type,
            "dataset_source": config.dataset_source,
            "metadata": config.metadata
        }
    
    def create_evaluator(self, benchmark_name: str) -> Optional[BaseEvaluation]:
        """Create an evaluator instance for a benchmark."""
        config = self.get(benchmark_name)
        if not config:
            logger.error(f"Benchmark not found: {benchmark_name}")
            return None
        
        # Special handling for SimpleQA
        if config.evaluation_type == "simpleqa":
            try:
                from tldw_Server_API.app.core.Evaluations.simpleqa_eval import SimpleQAEvaluation
                evaluator = SimpleQAEvaluation(
                    name=config.name,
                    **config.evaluation_params
                )
                return evaluator
            except Exception as e:
                logger.error(f"Failed to create SimpleQA evaluator: {e}")
                return None
        
        eval_class = self.EVALUATION_TYPES.get(config.evaluation_type)
        if not eval_class:
            logger.error(f"Evaluation type not implemented: {config.evaluation_type}")
            return None
        
        # Create evaluator with benchmark-specific parameters
        try:
            evaluator = eval_class(
                name=config.name,
                **config.evaluation_params
            )
            return evaluator
        except Exception as e:
            logger.error(f"Failed to create evaluator for {benchmark_name}: {e}")
            return None
    
    def load_dataset(self, benchmark_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load dataset for a benchmark."""
        config = self.get(benchmark_name)
        if not config:
            logger.error(f"Benchmark not found: {benchmark_name}")
            return []
        
        try:
            # Load based on format
            if config.dataset_format == "json":
                if config.dataset_source.startswith("http"):
                    data = load_dataset_from_url(config.dataset_source, format="json")
                else:
                    data = load_dataset_from_json(config.dataset_source)
            
            elif config.dataset_format == "jsonl":
                if config.dataset_source.startswith("http"):
                    data = load_dataset_from_url(config.dataset_source, format="jsonl")
                else:
                    data = load_dataset_from_jsonl(config.dataset_source)
            
            elif config.dataset_format == "huggingface":
                # Would need to implement HuggingFace dataset loading
                logger.warning(f"HuggingFace datasets not yet implemented for {benchmark_name}")
                return []
            
            elif config.dataset_format == "custom":
                # Custom loaders for specific benchmarks
                logger.warning(f"Custom loader not yet implemented for {benchmark_name}")
                return []
            
            else:
                logger.error(f"Unknown dataset format: {config.dataset_format}")
                return []
            
            # Apply field mappings to normalize data
            normalized_data = []
            for item in data:
                normalized = {}
                for target_field, source_field in config.field_mappings.items():
                    if source_field in item:
                        normalized[target_field] = item[source_field]
                normalized["_original"] = item  # Keep original for reference
                normalized_data.append(normalized)
            
            # Apply limit if specified
            if limit and limit > 0:
                normalized_data = normalized_data[:limit]
            
            logger.info(f"Loaded {len(normalized_data)} items for {benchmark_name}")
            return normalized_data
            
        except Exception as e:
            logger.error(f"Failed to load dataset for {benchmark_name}: {e}")
            return []
    
    def save_config(self, config: BenchmarkConfig, path: str) -> None:
        """Save benchmark configuration to file."""
        path = Path(path)
        
        if path.suffix == '.yaml' or path.suffix == '.yml':
            with open(path, 'w') as f:
                yaml.dump(config.to_dict(), f, default_flow_style=False)
        else:
            with open(path, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
        
        logger.info(f"Saved benchmark config to {path}")
    
    def load_config(self, path: str) -> Optional[BenchmarkConfig]:
        """Load benchmark configuration from file."""
        path = Path(path)
        
        if not path.exists():
            logger.error(f"Config file not found: {path}")
            return None
        
        try:
            if path.suffix == '.yaml' or path.suffix == '.yml':
                config = BenchmarkConfig.from_yaml(str(path))
            else:
                with open(path, 'r') as f:
                    data = json.load(f)
                config = BenchmarkConfig.from_dict(data)
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config from {path}: {e}")
            return None


# Global registry instance
_registry = BenchmarkRegistry()

def get_registry() -> BenchmarkRegistry:
    """Get the global benchmark registry."""
    return _registry