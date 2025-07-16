"""
Pipeline loader for TOML-based RAG pipeline configurations.

This module provides functionality to load, validate, and instantiate
RAG pipelines from TOML configuration files.
"""

import os
import sys
if sys.version_info < (3, 11):
    import tomli as tomllib
else:
    import tomllib
import toml
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from functools import wraps
import asyncio
from loguru import logger

from tldw_chatbook.Event_Handlers.Chat_Events import chat_rag_events


@dataclass
class MiddlewareConfig:
    """Configuration for a middleware component."""
    name: str
    type: str  # before_search, after_search, error_handler
    enabled: bool = True
    description: str = ""
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Configuration for a RAG pipeline."""
    id: str
    name: str
    description: str
    type: str  # built-in, custom, composite, wrapper
    enabled: bool = True
    function: Optional[str] = None
    base_pipeline: Optional[str] = None
    profile: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    middleware: Dict[str, List[str]] = field(default_factory=dict)
    strategy: Optional[Dict[str, Any]] = None
    components: Optional[Dict[str, Dict[str, Any]]] = None


class PipelineLoader:
    """Loads and manages pipeline configurations from TOML files."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path(__file__).parent.parent / "Config_Files"
        self.pipelines: Dict[str, PipelineConfig] = {}
        self.middleware: Dict[str, MiddlewareConfig] = {}
        self.global_config: Dict[str, Any] = {}
        self._function_cache: Dict[str, Callable] = {}
        
        # Map of built-in pipeline functions
        self._builtin_functions = {
            "perform_plain_rag_search": chat_rag_events.perform_plain_rag_search,
            "perform_full_rag_pipeline": chat_rag_events.perform_full_rag_pipeline,
            "perform_hybrid_rag_search": chat_rag_events.perform_hybrid_rag_search,
        }
    
    def load_pipeline_config(self, config_file: Optional[Path] = None) -> None:
        """Load pipeline configurations from TOML file."""
        if config_file is None:
            config_file = self.config_dir / "rag_pipelines.toml"
        
        if not config_file.exists():
            logger.warning(f"Pipeline config file not found: {config_file}")
            return
        
        try:
            with open(config_file, "rb") as f:
                config_data = tomllib.load(f)
            
            # Load global configuration
            self.global_config = config_data.get("global", {})
            
            # Load middleware definitions
            middleware_configs = config_data.get("middleware", {})
            for mid_id, mid_config in middleware_configs.items():
                self.middleware[mid_id] = MiddlewareConfig(
                    name=mid_config.get("name", mid_id),
                    type=mid_config.get("type", "before_search"),
                    enabled=mid_config.get("enabled", True),
                    description=mid_config.get("description", ""),
                    config=mid_config.get("config", {})
                )
            
            # Load pipeline configurations
            pipeline_configs = config_data.get("pipelines", {})
            for pipeline_id, pipeline_config in pipeline_configs.items():
                if self._validate_pipeline_config(pipeline_id, pipeline_config):
                    self.pipelines[pipeline_id] = PipelineConfig(
                        id=pipeline_id,
                        name=pipeline_config.get("name", pipeline_id),
                        description=pipeline_config.get("description", ""),
                        type=pipeline_config.get("type", "custom"),
                        enabled=pipeline_config.get("enabled", True),
                        function=pipeline_config.get("function"),
                        base_pipeline=pipeline_config.get("base_pipeline"),
                        profile=pipeline_config.get("profile"),
                        tags=pipeline_config.get("tags", []),
                        parameters=pipeline_config.get("parameters", {}),
                        middleware=pipeline_config.get("middleware", {}),
                        strategy=pipeline_config.get("strategy"),
                        components=pipeline_config.get("components")
                    )
            
            logger.info(f"Loaded {len(self.pipelines)} pipelines and {len(self.middleware)} middleware from {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load pipeline config: {e}")
    
    def _validate_pipeline_config(self, pipeline_id: str, config: Dict[str, Any]) -> bool:
        """Validate a pipeline configuration."""
        required_fields = ["name", "description", "type"]
        valid_types = ["built-in", "custom", "composite", "wrapper", "functional"]
        
        # Check required fields
        for field in required_fields:
            if field not in config:
                logger.error(f"Pipeline '{pipeline_id}' missing required field: {field}")
                return False
        
        # Check valid type
        if config["type"] not in valid_types:
            logger.error(f"Pipeline '{pipeline_id}' has invalid type: {config['type']}")
            return False
        
        # Check type-specific requirements
        if config["type"] == "built-in" and "function" not in config:
            logger.error(f"Built-in pipeline '{pipeline_id}' missing function")
            return False
        
        if config["type"] == "composite" and not (config.get("strategy") or config.get("components")):
            logger.error(f"Composite pipeline '{pipeline_id}' missing strategy or components")
            return False
        
        return True
    
    def get_pipeline_function(self, pipeline_id: str) -> Optional[Callable]:
        """Get the executable function for a pipeline."""
        if pipeline_id not in self.pipelines:
            logger.error(f"Pipeline '{pipeline_id}' not found")
            return None
        
        pipeline = self.pipelines[pipeline_id]
        
        if not pipeline.enabled:
            logger.warning(f"Pipeline '{pipeline_id}' is disabled")
            return None
        
        # Check cache
        if pipeline_id in self._function_cache:
            return self._function_cache[pipeline_id]
        
        # Build function based on type
        if pipeline.type == "built-in":
            func = self._get_builtin_function(pipeline)
        elif pipeline.type == "custom":
            func = self._create_custom_function(pipeline)
        elif pipeline.type == "composite":
            func = self._create_composite_function(pipeline)
        elif pipeline.type == "wrapper":
            func = self._create_wrapper_function(pipeline)
        elif pipeline.type == "functional":
            func = self._create_functional_pipeline(pipeline)
        else:
            logger.error(f"Unknown pipeline type: {pipeline.type}")
            return None
        
        # Apply middleware if configured
        if pipeline.middleware:
            func = self._apply_middleware(func, pipeline)
        
        # Cache the function
        self._function_cache[pipeline_id] = func
        
        return func
    
    def _get_builtin_function(self, pipeline: PipelineConfig) -> Optional[Callable]:
        """Get a built-in pipeline function."""
        if pipeline.function not in self._builtin_functions:
            logger.error(f"Built-in function '{pipeline.function}' not found")
            return None
        
        base_func = self._builtin_functions[pipeline.function]
        
        # Wrap to apply default parameters
        @wraps(base_func)
        async def wrapped_func(app, query, sources, **kwargs):
            # Merge default parameters
            merged_params = {**pipeline.parameters, **kwargs}
            return await base_func(app, query, sources, **merged_params)
        
        return wrapped_func
    
    def _create_custom_function(self, pipeline: PipelineConfig) -> Callable:
        """Create a custom pipeline function."""
        
        async def custom_pipeline(app, query, sources, **kwargs):
            # Merge parameters
            params = {**pipeline.parameters, **kwargs}
            
            # If based on another pipeline, use it
            if pipeline.base_pipeline:
                base_func = self.get_pipeline_function(pipeline.base_pipeline)
                if base_func:
                    return await base_func(app, query, sources, **params)
            
            # Otherwise, use the profile-based approach
            if pipeline.profile:
                from tldw_chatbook.RAG_Search.config_profiles import get_profile_manager
                
                manager = get_profile_manager()
                profile_config = manager.get_profile(pipeline.profile)
                
                if profile_config:
                    # Apply profile configuration
                    # This would need integration with the RAG service
                    logger.info(f"Using profile '{pipeline.profile}' for pipeline '{pipeline.id}'")
                    
                    # For now, default to semantic search with profile
                    from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import perform_full_rag_pipeline
                    return await perform_full_rag_pipeline(app, query, sources, **params)
            
            # Fallback to basic search
            logger.warning(f"Custom pipeline '{pipeline.id}' has no implementation, using default")
            from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import perform_plain_rag_search
            return await perform_plain_rag_search(app, query, sources, **params)
        
        return custom_pipeline
    
    def _create_composite_function(self, pipeline: PipelineConfig) -> Callable:
        """Create a composite pipeline function."""
        
        async def composite_pipeline(app, query, sources, **kwargs):
            params = {**pipeline.parameters, **kwargs}
            
            # Strategy-based composite
            if pipeline.strategy:
                strategy_type = pipeline.strategy.get("type")
                
                if strategy_type == "query_analysis":
                    # Analyze query and select pipeline
                    selected_pipeline = self._select_pipeline_by_query(query, pipeline.strategy)
                    if selected_pipeline:
                        func = self.get_pipeline_function(selected_pipeline)
                        if func:
                            return await func(app, query, sources, **params)
                
            # Component-based composite (ensemble)
            elif pipeline.components:
                return await self._run_ensemble_pipeline(
                    app, query, sources, pipeline.components, **params
                )
            
            # Fallback
            logger.warning(f"Composite pipeline '{pipeline.id}' has no valid strategy")
            from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import perform_hybrid_rag_search
            return await perform_hybrid_rag_search(app, query, sources, **params)
        
        return composite_pipeline
    
    def _create_wrapper_function(self, pipeline: PipelineConfig) -> Callable:
        """Create a wrapper pipeline function."""
        # Similar to custom, but specifically for wrapping other pipelines
        return self._create_custom_function(pipeline)
    
    def _create_functional_pipeline(self, pipeline: PipelineConfig) -> Optional[Callable]:
        """Create a functional pipeline using the new pipeline builder."""
        try:
            # Check if functional pipeline system is available
            from .pipeline_adapter import is_using_v2_pipelines
            if not is_using_v2_pipelines():
                logger.warning(f"Functional pipeline '{pipeline.id}' requested but v2 pipelines not enabled")
                # Fall back to built-in if possible
                if pipeline.id.endswith('_v2'):
                    base_id = pipeline.id[:-3]  # Remove '_v2' suffix
                    if base_id in self.pipelines:
                        return self.get_pipeline_function(base_id)
                return None
            
            # Import the functional pipeline adapter
            from .pipeline_adapter import (
                perform_plain_rag_search_v2,
                perform_full_rag_pipeline_v2,
                perform_hybrid_rag_search_v2
            )
            
            # Map known v2 pipelines to their adapter functions
            if pipeline.id == 'plain_v2':
                return perform_plain_rag_search_v2
            elif pipeline.id == 'semantic_v2':
                return perform_full_rag_pipeline_v2
            elif pipeline.id == 'hybrid_v2':
                return perform_hybrid_rag_search_v2
            else:
                # For other functional pipelines, build from configuration
                from .pipeline_builder import build_pipeline_from_dict
                from .pipeline_resources import get_global_resource_manager
                
                # Convert pipeline config to dict format expected by builder
                pipeline_dict = {
                    'id': pipeline.id,
                    'name': pipeline.name,
                    'description': pipeline.description,
                    'steps': [],
                    'version': getattr(pipeline, 'version', '1.0'),
                    'cache_results': pipeline.parameters.get('cache_results', True),
                    'cache_ttl_seconds': pipeline.parameters.get('cache_ttl_seconds', 3600),
                    'on_error': pipeline.parameters.get('on_error', 'fail'),
                    'timeout_seconds': pipeline.parameters.get('timeout_seconds', 30.0)
                }
                
                # Parse steps from the pipeline config (would need the actual TOML structure)
                # For now, return None as we need the full TOML structure
                logger.warning(f"Custom functional pipeline '{pipeline.id}' not yet fully supported")
                return None
                
        except ImportError as e:
            logger.error(f"Failed to import functional pipeline components: {e}")
            return None
    
    def _apply_middleware(self, func: Callable, pipeline: PipelineConfig) -> Callable:
        """Apply middleware to a pipeline function."""
        
        @wraps(func)
        async def wrapped_with_middleware(app, query, sources, **kwargs):
            # Apply before_search middleware
            if "before" in pipeline.middleware:
                for mid_id in pipeline.middleware["before"]:
                    if mid_id in self.middleware and self.middleware[mid_id].enabled:
                        query, kwargs = await self._apply_before_middleware(
                            mid_id, query, kwargs
                        )
            
            try:
                # Execute main function
                results, context = await func(app, query, sources, **kwargs)
                
                # Apply after_search middleware
                if "after" in pipeline.middleware:
                    for mid_id in pipeline.middleware["after"]:
                        if mid_id in self.middleware and self.middleware[mid_id].enabled:
                            results = await self._apply_after_middleware(
                                mid_id, results
                            )
                
                return results, context
                
            except Exception as e:
                # Apply error_handler middleware
                if "error" in pipeline.middleware:
                    for mid_id in pipeline.middleware["error"]:
                        if mid_id in self.middleware and self.middleware[mid_id].enabled:
                            fallback = await self._apply_error_middleware(mid_id, e)
                            if fallback:
                                return fallback
                raise
        
        return wrapped_with_middleware
    
    async def _apply_before_middleware(
        self, middleware_id: str, query: str, params: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Apply before_search middleware."""
        middleware = self.middleware[middleware_id]
        
        # Implement specific middleware logic based on ID
        if middleware_id == "query_expansion":
            # Example: expand query
            expanded_terms = await self._expand_query(query, middleware.config)
            if expanded_terms:
                query = f"{query} {' '.join(expanded_terms)}"
        
        elif middleware_id == "technical_term_detector":
            # Example: boost technical terms
            if "boost_factor" in middleware.config:
                params["technical_boost"] = middleware.config["boost_factor"]
        
        # Add more middleware implementations as needed
        
        return query, params
    
    async def _apply_after_middleware(
        self, middleware_id: str, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply after_search middleware."""
        middleware = self.middleware[middleware_id]
        
        # Implement specific middleware logic
        if middleware_id == "result_reranking":
            # Example: re-rank results
            # This would integrate with the re-ranking system
            pass
        
        elif middleware_id == "citation_enhancement":
            # Example: enhance citations
            for result in results:
                if "metadata" not in result:
                    result["metadata"] = {}
                result["metadata"]["citation_format"] = middleware.config.get("format", "inline")
        
        return results
    
    async def _apply_error_middleware(
        self, middleware_id: str, error: Exception
    ) -> Optional[Tuple[List[Dict[str, Any]], str]]:
        """Apply error_handler middleware."""
        # Implement error recovery logic
        logger.error(f"Pipeline error handled by {middleware_id}: {error}")
        return None
    
    def _select_pipeline_by_query(self, query: str, strategy: Dict[str, Any]) -> Optional[str]:
        """Select a pipeline based on query analysis."""
        rules = strategy.get("rules", [])
        
        for rule in rules:
            if "condition" in rule:
                condition = rule["condition"]
                
                # Simple condition checks
                if condition == "is_question" and query.strip().endswith("?"):
                    return rule.get("pipeline")
                elif condition == "has_technical_terms":
                    technical_terms = ["api", "function", "class", "error", "bug", "code"]
                    if any(term in query.lower() for term in technical_terms):
                        return rule.get("pipeline")
                elif condition == "is_short_query" and len(query.split()) < 4:
                    return rule.get("pipeline")
            
            elif rule.get("default"):
                return rule.get("pipeline")
        
        return None
    
    async def _run_ensemble_pipeline(
        self, app, query, sources, components: Dict[str, Dict[str, Any]], **kwargs
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Run an ensemble of pipelines."""
        from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import format_results_for_llm
        
        all_results = []
        tasks = []
        
        # Create tasks for each component
        for component_id, component_config in components.items():
            if component_config.get("enabled", True):
                func = self.get_pipeline_function(component_id)
                if func:
                    weight = component_config.get("weight", 1.0)
                    tasks.append((component_id, weight, func(app, query, sources, **kwargs)))
        
        # Run all pipelines in parallel
        if tasks:
            results = await asyncio.gather(*[task[2] for task in tasks], return_exceptions=True)
            
            # Process results
            for (component_id, weight, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    logger.error(f"Ensemble component '{component_id}' failed: {result}")
                    continue
                
                component_results, _ = result
                
                # Apply weighting
                for res in component_results:
                    res["_original_score"] = res.get("score", 0)
                    res["_source_pipeline"] = component_id
                    res["score"] = res["_original_score"] * weight
                    all_results.append(res)
        
        # Merge and deduplicate
        final_results = self._merge_and_deduplicate(all_results)
        
        # Sort by score and limit
        final_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        top_k = kwargs.get("top_k", 10)
        final_results = final_results[:top_k]
        
        # Format context
        context = format_results_for_llm(final_results, kwargs.get("max_context_length", 10000))
        
        return final_results, context
    
    def _merge_and_deduplicate(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge and deduplicate results from multiple pipelines."""
        seen = {}
        
        for result in results:
            # Create a unique key based on content
            key = result.get("content", "")[:200]  # Use first 200 chars as key
            
            if key not in seen or result.get("score", 0) > seen[key].get("score", 0):
                seen[key] = result
        
        return list(seen.values())
    
    async def _expand_query(self, query: str, config: Dict[str, Any]) -> List[str]:
        """Expand query with additional terms."""
        # This is a placeholder - implement actual query expansion
        return []
    
    def list_pipelines(self) -> List[Dict[str, Any]]:
        """List all available pipelines."""
        return [
            {
                "id": pipeline_id,
                "name": pipeline.name,
                "description": pipeline.description,
                "type": pipeline.type,
                "enabled": pipeline.enabled,
                "tags": pipeline.tags
            }
            for pipeline_id, pipeline in self.pipelines.items()
        ]
    
    def get_pipeline_config(self, pipeline_id: str) -> Optional[PipelineConfig]:
        """Get configuration for a specific pipeline."""
        return self.pipelines.get(pipeline_id)
    
    def reload_configs(self) -> None:
        """Reload pipeline configurations from disk."""
        self.pipelines.clear()
        self.middleware.clear()
        self._function_cache.clear()
        self.load_pipeline_config()
    
    def export_pipeline_config(self, pipeline_id: str, output_file: Path) -> bool:
        """Export a pipeline configuration to a TOML file."""
        if pipeline_id not in self.pipelines:
            logger.error(f"Pipeline '{pipeline_id}' not found")
            return False
        
        pipeline = self.pipelines[pipeline_id]
        
        config_data = {
            "pipelines": {
                pipeline_id: {
                    "name": pipeline.name,
                    "description": pipeline.description,
                    "type": pipeline.type,
                    "enabled": pipeline.enabled,
                    "tags": pipeline.tags,
                    "parameters": pipeline.parameters
                }
            }
        }
        
        if pipeline.function:
            config_data["pipelines"][pipeline_id]["function"] = pipeline.function
        if pipeline.base_pipeline:
            config_data["pipelines"][pipeline_id]["base_pipeline"] = pipeline.base_pipeline
        if pipeline.profile:
            config_data["pipelines"][pipeline_id]["profile"] = pipeline.profile
        if pipeline.middleware:
            config_data["pipelines"][pipeline_id]["middleware"] = pipeline.middleware
        if pipeline.strategy:
            config_data["pipelines"][pipeline_id]["strategy"] = pipeline.strategy
        if pipeline.components:
            config_data["pipelines"][pipeline_id]["components"] = pipeline.components
        
        try:
            with open(output_file, "w") as f:
                toml.dump(config_data, f)
            logger.info(f"Exported pipeline '{pipeline_id}' to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to export pipeline: {e}")
            return False


# Global pipeline loader instance
_pipeline_loader: Optional[PipelineLoader] = None


def get_pipeline_loader() -> PipelineLoader:
    """Get or create the global pipeline loader."""
    global _pipeline_loader
    if _pipeline_loader is None:
        _pipeline_loader = PipelineLoader()
        _pipeline_loader.load_pipeline_config()
    return _pipeline_loader


def get_pipeline_function(pipeline_id: str) -> Optional[Callable]:
    """Convenience function to get a pipeline function."""
    loader = get_pipeline_loader()
    return loader.get_pipeline_function(pipeline_id)


def list_available_pipelines() -> List[Dict[str, Any]]:
    """Convenience function to list all pipelines."""
    loader = get_pipeline_loader()
    return loader.list_pipelines()