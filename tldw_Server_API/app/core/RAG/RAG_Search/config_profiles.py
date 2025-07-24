"""
Configuration profiles for different RAG use cases.

This module provides preset configurations optimized for various scenarios,
along with utilities for experiment tracking and A/B testing.
"""

import json
import time
from typing import Dict, Any, Optional, List, Literal, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
import hashlib
from loguru import logger

from .simplified.config import RAGConfig
from .reranker import RerankingConfig
from .parallel_processor import ProcessingConfig
from ..config import get_user_data_dir
from ..Metrics.metrics_logger import log_counter, log_histogram


# Profile types
ProfileType = Literal[
    "fast_search",      # Optimized for speed
    "high_accuracy",    # Optimized for accuracy
    "balanced",         # Balance between speed and accuracy
    "long_context",     # For documents requiring large context
    "technical_docs",   # For technical documentation
    "conversational",   # For chat/conversation data
    "research_papers",  # For academic papers
    "code_search",      # For code repositories
    "multimodal",       # For mixed text/image content
    "custom"           # User-defined configuration
]


@dataclass
class ExperimentConfig:
    """Configuration for A/B testing and experiment tracking."""
    experiment_id: str = field(default_factory=lambda: f"exp_{int(time.time())}")
    name: str = "Unnamed Experiment"
    description: str = ""
    
    # A/B test settings
    enable_ab_testing: bool = False
    control_profile: str = "balanced"
    test_profiles: List[str] = field(default_factory=list)
    traffic_split: Dict[str, float] = field(default_factory=dict)  # Profile -> percentage
    
    # Tracking settings
    track_metrics: bool = True
    metrics_to_track: List[str] = field(default_factory=lambda: [
        "search_latency", "retrieval_accuracy", "reranking_improvement",
        "user_satisfaction", "context_quality"
    ])
    
    # Persistence
    save_results: bool = True
    results_dir: Optional[Path] = None


@dataclass
class ProfileConfig:
    """Complete configuration profile including all components."""
    name: str
    description: str
    profile_type: ProfileType
    
    # Component configurations
    rag_config: RAGConfig
    reranking_config: Optional[RerankingConfig] = None
    processing_config: Optional[ProcessingConfig] = None
    
    # Profile metadata
    created_at: float = field(default_factory=time.time)
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    
    # Performance expectations
    expected_latency_ms: Optional[float] = None
    expected_accuracy: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "profile_type": self.profile_type,
            "rag_config": asdict(self.rag_config),
            "reranking_config": asdict(self.reranking_config) if self.reranking_config else None,
            "processing_config": asdict(self.processing_config) if self.processing_config else None,
            "created_at": self.created_at,
            "version": self.version,
            "tags": self.tags,
            "expected_latency_ms": self.expected_latency_ms,
            "expected_accuracy": self.expected_accuracy
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProfileConfig":
        """Create profile from dictionary."""
        rag_config = RAGConfig(**data["rag_config"]) if isinstance(data["rag_config"], dict) else data["rag_config"]
        
        reranking_config = None
        if data.get("reranking_config"):
            reranking_config = RerankingConfig(**data["reranking_config"]) if isinstance(data["reranking_config"], dict) else data["reranking_config"]
        
        processing_config = None
        if data.get("processing_config"):
            processing_config = ProcessingConfig(**data["processing_config"]) if isinstance(data["processing_config"], dict) else data["processing_config"]
        
        return cls(
            name=data["name"],
            description=data["description"],
            profile_type=data["profile_type"],
            rag_config=rag_config,
            reranking_config=reranking_config,
            processing_config=processing_config,
            created_at=data.get("created_at", time.time()),
            version=data.get("version", "1.0"),
            tags=data.get("tags", []),
            expected_latency_ms=data.get("expected_latency_ms"),
            expected_accuracy=data.get("expected_accuracy")
        )


class ConfigProfileManager:
    """Manages configuration profiles and experiments."""
    
    def __init__(self, profiles_dir: Optional[Path] = None):
        self.profiles_dir = profiles_dir or (get_user_data_dir() / "rag_profiles")
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        
        self._profiles: Dict[str, ProfileConfig] = {}
        self._current_experiment: Optional[ExperimentConfig] = None
        self._experiment_results: Dict[str, List[Dict[str, Any]]] = {}
        
        # Load built-in profiles
        self._load_builtin_profiles()
        
        # Load custom profiles
        self._load_custom_profiles()
    
    def _load_builtin_profiles(self):
        """Load predefined configuration profiles."""
        
        # Fast Search Profile
        fast_rag = RAGConfig()
        fast_rag.embedding.model = "all-MiniLM-L6-v2"  # Smaller, faster model
        fast_rag.embedding.batch_size = 64
        fast_rag.chunking.size = 256  # Smaller chunks
        fast_rag.chunking.overlap = 32
        fast_rag.search.top_k = 5
        fast_rag.search.enable_cache = True
        fast_rag.search.cache_size = 200
        
        self._profiles["fast_search"] = ProfileConfig(
            name="Fast Search",
            description="Optimized for low latency with acceptable accuracy",
            profile_type="fast_search",
            rag_config=fast_rag,
            processing_config=ProcessingConfig(batch_size=64, num_workers=None),
            expected_latency_ms=100,
            expected_accuracy=0.8,
            tags=["speed", "low-latency"]
        )
        
        # High Accuracy Profile
        accurate_rag = RAGConfig()
        accurate_rag.embedding.model = "BAAI/bge-large-en-v1.5"  # Larger, more accurate
        accurate_rag.embedding.batch_size = 16
        accurate_rag.chunking.size = 512
        accurate_rag.chunking.overlap = 128
        accurate_rag.chunking.method = "hierarchical"
        accurate_rag.search.top_k = 20
        accurate_rag.search.include_citations = True
        accurate_rag.search.score_threshold = 0.7
        
        accurate_rerank = RerankingConfig(
            model_provider="openai",
            model_name="gpt-3.5-turbo",
            strategy="pointwise",
            top_k_to_rerank=15,
            include_reasoning=True
        )
        
        self._profiles["high_accuracy"] = ProfileConfig(
            name="High Accuracy",
            description="Optimized for maximum retrieval accuracy",
            profile_type="high_accuracy",
            rag_config=accurate_rag,
            reranking_config=accurate_rerank,
            processing_config=ProcessingConfig(batch_size=16, show_progress=True),
            expected_latency_ms=500,
            expected_accuracy=0.95,
            tags=["accuracy", "quality"]
        )
        
        # Balanced Profile
        balanced_rag = RAGConfig()
        balanced_rag.embedding.model = "sentence-transformers/all-mpnet-base-v2"
        balanced_rag.chunking.size = 384
        balanced_rag.chunking.overlap = 64
        balanced_rag.search.top_k = 10
        
        self._profiles["balanced"] = ProfileConfig(
            name="Balanced",
            description="Balance between speed and accuracy",
            profile_type="balanced",
            rag_config=balanced_rag,
            processing_config=ProcessingConfig(batch_size=32),
            expected_latency_ms=250,
            expected_accuracy=0.88,
            tags=["balanced", "general-purpose"]
        )
        
        # Long Context Profile
        long_context_rag = RAGConfig()
        long_context_rag.embedding.model = "BAAI/bge-base-en-v1.5"
        long_context_rag.chunking.size = 1024  # Larger chunks
        long_context_rag.chunking.overlap = 256
        long_context_rag.chunking.enable_parent_retrieval = True
        long_context_rag.chunking.parent_size_multiplier = 3
        long_context_rag.search.top_k = 5  # Fewer but larger chunks
        
        self._profiles["long_context"] = ProfileConfig(
            name="Long Context",
            description="For documents requiring extended context",
            profile_type="long_context",
            rag_config=long_context_rag,
            expected_latency_ms=300,
            expected_accuracy=0.9,
            tags=["context", "long-form"]
        )
        
        # Technical Documentation Profile
        tech_rag = RAGConfig()
        tech_rag.embedding.model = "sentence-transformers/all-mpnet-base-v2"
        tech_rag.chunking.size = 512
        tech_rag.chunking.method = "structural"  # Preserve document structure
        tech_rag.chunking.clean_artifacts = True
        tech_rag.chunking.preserve_tables = True
        tech_rag.search.include_citations = True
        
        self._profiles["technical_docs"] = ProfileConfig(
            name="Technical Documentation",
            description="Optimized for technical content with tables and code",
            profile_type="technical_docs",
            rag_config=tech_rag,
            expected_accuracy=0.92,
            tags=["technical", "documentation"]
        )
        
        # Research Papers Profile
        research_rag = RAGConfig()
        research_rag.embedding.model = "allenai-specter"  # Specialized for scientific text
        research_rag.chunking.size = 512
        research_rag.chunking.method = "hierarchical"
        research_rag.chunking.clean_artifacts = True  # Clean PDF artifacts
        research_rag.search.include_citations = True
        research_rag.search.top_k = 15
        
        research_rerank = RerankingConfig(
            strategy="listwise",
            top_k_to_rerank=10,
            include_reasoning=True
        )
        
        self._profiles["research_papers"] = ProfileConfig(
            name="Research Papers",
            description="Optimized for academic papers and citations",
            profile_type="research_papers",
            rag_config=research_rag,
            reranking_config=research_rerank,
            expected_accuracy=0.94,
            tags=["academic", "research", "citations"]
        )
        
        # Code Search Profile
        code_rag = RAGConfig()
        code_rag.embedding.model = "microsoft/codebert-base"  # Code-specific embeddings
        code_rag.chunking.size = 256  # Smaller chunks for code
        code_rag.chunking.method = "words"  # Preserve code structure
        code_rag.search.top_k = 20  # More results for code search
        
        self._profiles["code_search"] = ProfileConfig(
            name="Code Search",
            description="Optimized for searching code repositories",
            profile_type="code_search",
            rag_config=code_rag,
            expected_accuracy=0.85,
            tags=["code", "programming"]
        )
        
        logger.info(f"Loaded {len(self._profiles)} built-in profiles")
    
    def _load_custom_profiles(self):
        """Load user-defined profiles from disk."""
        custom_profiles_file = self.profiles_dir / "custom_profiles.json"
        
        if custom_profiles_file.exists():
            try:
                with open(custom_profiles_file, 'r') as f:
                    custom_data = json.load(f)
                
                for profile_data in custom_data.get("profiles", []):
                    profile = ProfileConfig.from_dict(profile_data)
                    self._profiles[profile.name.lower().replace(" ", "_")] = profile
                
                logger.info(f"Loaded {len(custom_data.get('profiles', []))} custom profiles")
                
            except Exception as e:
                logger.error(f"Failed to load custom profiles: {e}")
    
    def get_profile(self, name: str) -> Optional[ProfileConfig]:
        """Get a configuration profile by name."""
        return self._profiles.get(name)
    
    def list_profiles(self) -> List[str]:
        """List all available profile names."""
        return list(self._profiles.keys())
    
    def create_custom_profile(self,
                            name: str,
                            base_profile: str = "balanced",
                            **overrides) -> ProfileConfig:
        """
        Create a custom profile based on an existing one.
        
        Args:
            name: Name for the custom profile
            base_profile: Base profile to start from
            **overrides: Configuration overrides
            
        Returns:
            New custom profile
        """
        base = self.get_profile(base_profile)
        if not base:
            raise ValueError(f"Base profile '{base_profile}' not found")
        
        # Create a copy
        custom_config = ProfileConfig(
            name=name,
            description=f"Custom profile based on {base_profile}",
            profile_type="custom",
            rag_config=RAGConfig(**asdict(base.rag_config)),
            reranking_config=RerankingConfig(**asdict(base.reranking_config)) if base.reranking_config else None,
            processing_config=ProcessingConfig(**asdict(base.processing_config)) if base.processing_config else None,
            tags=["custom"] + base.tags
        )
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(custom_config, key):
                setattr(custom_config, key, value)
            elif hasattr(custom_config.rag_config, key):
                setattr(custom_config.rag_config, key, value)
            # Add more override logic as needed
        
        # Save the custom profile
        profile_key = name.lower().replace(" ", "_")
        self._profiles[profile_key] = custom_config
        self._save_custom_profiles()
        
        logger.info(f"Created custom profile: {name}")
        
        return custom_config
    
    def _save_custom_profiles(self):
        """Save custom profiles to disk."""
        custom_profiles = {
            name: profile for name, profile in self._profiles.items()
            if profile.profile_type == "custom"
        }
        
        custom_profiles_file = self.profiles_dir / "custom_profiles.json"
        
        with open(custom_profiles_file, 'w') as f:
            json.dump({
                "profiles": [p.to_dict() for p in custom_profiles.values()],
                "saved_at": datetime.now().isoformat()
            }, f, indent=2)
    
    def start_experiment(self, config: ExperimentConfig):
        """Start an A/B testing experiment."""
        self._current_experiment = config
        self._experiment_results[config.experiment_id] = []
        
        if not config.results_dir:
            config.results_dir = self.profiles_dir / "experiments" / config.experiment_id
        
        config.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save experiment configuration
        with open(config.results_dir / "config.json", 'w') as f:
            json.dump(asdict(config), f, indent=2)
        
        logger.info(f"Started experiment: {config.name} (ID: {config.experiment_id})")
        
        log_counter("rag_experiment_started", labels={"experiment_id": config.experiment_id})
    
    def select_profile_for_experiment(self, user_id: Optional[str] = None) -> Tuple[str, ProfileConfig]:
        """
        Select a profile for the current experiment based on traffic split.
        
        Args:
            user_id: Optional user ID for consistent assignment
            
        Returns:
            Tuple of (profile_name, profile_config)
        """
        if not self._current_experiment or not self._current_experiment.enable_ab_testing:
            # No experiment running, use default
            return "balanced", self.get_profile("balanced")
        
        # Ensure consistent assignment for user
        if user_id:
            hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            rand_val = (hash_val % 100) / 100.0
        else:
            import random
            rand_val = random.random()
        
        # Select profile based on traffic split
        cumulative = 0.0
        for profile_name, percentage in self._current_experiment.traffic_split.items():
            cumulative += percentage
            if rand_val < cumulative:
                profile = self.get_profile(profile_name)
                if profile:
                    log_counter("rag_experiment_profile_selected", 
                              labels={"experiment_id": self._current_experiment.experiment_id,
                                    "profile": profile_name})
                    return profile_name, profile
        
        # Fallback to control
        return self._current_experiment.control_profile, self.get_profile(self._current_experiment.control_profile)
    
    def record_experiment_result(self,
                               profile_name: str,
                               query: str,
                               metrics: Dict[str, Any]):
        """Record results from an experiment run."""
        if not self._current_experiment or not self._current_experiment.track_metrics:
            return
        
        result = {
            "timestamp": time.time(),
            "profile": profile_name,
            "query": query,
            "metrics": metrics
        }
        
        self._experiment_results[self._current_experiment.experiment_id].append(result)
        
        # Log metrics
        for metric_name, value in metrics.items():
            if metric_name in self._current_experiment.metrics_to_track:
                log_histogram(f"rag_experiment_{metric_name}", value,
                            labels={"experiment_id": self._current_experiment.experiment_id,
                                  "profile": profile_name})
    
    def end_experiment(self) -> Dict[str, Any]:
        """End the current experiment and return results summary."""
        if not self._current_experiment:
            return {}
        
        exp_id = self._current_experiment.experiment_id
        results = self._experiment_results.get(exp_id, [])
        
        # Calculate summary statistics
        summary = {
            "experiment_id": exp_id,
            "name": self._current_experiment.name,
            "total_queries": len(results),
            "profiles": {}
        }
        
        # Group by profile
        profile_results = {}
        for result in results:
            profile = result["profile"]
            if profile not in profile_results:
                profile_results[profile] = []
            profile_results[profile].append(result["metrics"])
        
        # Calculate statistics for each profile
        for profile, metrics_list in profile_results.items():
            profile_stats = {}
            
            # Calculate mean for each metric
            for metric_name in self._current_experiment.metrics_to_track:
                values = [m.get(metric_name, 0) for m in metrics_list if metric_name in m]
                if values:
                    profile_stats[metric_name] = {
                        "mean": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "count": len(values)
                    }
            
            summary["profiles"][profile] = {
                "query_count": len(metrics_list),
                "metrics": profile_stats
            }
        
        # Save results if configured
        if self._current_experiment.save_results:
            results_file = self._current_experiment.results_dir / "results.json"
            with open(results_file, 'w') as f:
                json.dump({
                    "summary": summary,
                    "detailed_results": results,
                    "completed_at": datetime.now().isoformat()
                }, f, indent=2)
            
            logger.info(f"Saved experiment results to {results_file}")
        
        # Clear experiment
        self._current_experiment = None
        
        log_counter("rag_experiment_completed", labels={"experiment_id": exp_id})
        
        return summary
    
    def validate_profile(self, profile: ProfileConfig) -> List[str]:
        """
        Validate a configuration profile for compatibility and correctness.
        
        Returns:
            List of validation warnings/errors (empty if valid)
        """
        warnings = []
        
        # Check embedding model availability
        if profile.rag_config.embedding.model not in self._get_available_models():
            warnings.append(f"Embedding model '{profile.rag_config.embedding.model}' may not be available")
        
        # Check chunk size vs overlap
        if profile.rag_config.chunking.overlap >= profile.rag_config.chunking.size:
            warnings.append("Chunk overlap should be less than chunk size")
        
        # Check reranking configuration
        if profile.reranking_config:
            if profile.reranking_config.top_k_to_rerank > profile.rag_config.search.top_k:
                warnings.append("Reranking top_k should not exceed search top_k")
        
        # Check processing configuration
        if profile.processing_config:
            if profile.processing_config.num_workers and profile.processing_config.num_workers > 32:
                warnings.append("Using more than 32 workers may cause resource issues")
        
        return warnings
    
    def _get_available_models(self) -> List[str]:
        """Get list of known available embedding models."""
        # This is a simplified list - in practice, you'd check actual availability
        return [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "BAAI/bge-base-en-v1.5",
            "BAAI/bge-large-en-v1.5",
            "sentence-transformers/all-mpnet-base-v2",
            "allenai-specter",
            "microsoft/codebert-base"
        ]


# Convenience functions

def get_profile_manager(profiles_dir: Optional[Path] = None) -> ConfigProfileManager:
    """Get or create the global profile manager."""
    return ConfigProfileManager(profiles_dir)


def quick_profile(use_case: ProfileType) -> ProfileConfig:
    """Get a profile for a specific use case."""
    manager = get_profile_manager()
    profile = manager.get_profile(use_case)
    
    if not profile:
        # Fall back to balanced
        logger.warning(f"Profile '{use_case}' not found, using 'balanced'")
        profile = manager.get_profile("balanced")
    
    return profile