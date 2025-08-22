"""
Integration module for TOML-based pipeline configuration with the RAG system.

This module provides the bridge between the TOML pipeline loader and the existing
RAG search functions, allowing pipelines to be selected by ID instead of hardcoded
search modes.
"""

from typing import Optional, Dict, Any, List, Tuple, Callable
from pathlib import Path
from loguru import logger

from .pipeline_loader import get_pipeline_loader, get_pipeline_function
from .simplified.config import RAGConfig


class PipelineManager:
    """Manages the integration of TOML-based pipelines with the RAG system."""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig.from_settings()
        self.loader = get_pipeline_loader()
        
        # Map legacy search modes to pipeline IDs
        self.legacy_mode_mapping = {
            "plain": "plain",
            "semantic": "semantic",
            "full": "semantic",  # Alias for semantic
            "hybrid": "hybrid",
            # V2 functional pipelines
            "plain_v2": "plain_v2",
            "semantic_v2": "semantic_v2",
            "hybrid_v2": "hybrid_v2",
            "research_focused_v2": "research_focused_v2",
            "speed_optimized_v2": "speed_optimized_v2",
            "adaptive_v2": "adaptive_v2"
        }
    
    def get_pipeline_by_mode(self, search_mode: str) -> Optional[Callable]:
        """Get a pipeline function by search mode (for backward compatibility)."""
        pipeline_id = self.legacy_mode_mapping.get(search_mode, search_mode)
        return self.get_pipeline(pipeline_id)
    
    def get_pipeline(self, pipeline_id: str) -> Optional[Callable]:
        """Get a pipeline function by ID."""
        # Check if pipeline is configured in TOML
        func = get_pipeline_function(pipeline_id)
        if func:
            return func
        
        # Fallback to default pipelines
        if pipeline_id in self.legacy_mode_mapping.values():
            logger.warning(f"Pipeline '{pipeline_id}' not found in TOML, using built-in")
            # Return None to signal fallback to existing implementation
            return None
        
        logger.error(f"Unknown pipeline ID: {pipeline_id}")
        return None
    
    def get_default_pipeline(self) -> str:
        """Get the default pipeline ID from configuration."""
        return self.config.pipeline.default_pipeline
    
    def list_available_pipelines(self) -> List[Dict[str, Any]]:
        """List all available pipelines."""
        return self.loader.list_pipelines()
    
    def get_pipeline_config(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific pipeline."""
        pipeline_config = self.loader.get_pipeline_config(pipeline_id)
        if pipeline_config:
            return {
                "id": pipeline_config.id,
                "name": pipeline_config.name,
                "description": pipeline_config.description,
                "type": pipeline_config.type,
                "enabled": pipeline_config.enabled,
                "parameters": pipeline_config.parameters,
                "tags": pipeline_config.tags
            }
        return None
    
    def reload_pipelines(self) -> None:
        """Reload pipeline configurations from TOML files."""
        self.loader.reload_configs()
        logger.info("Pipeline configurations reloaded")
    
    async def execute_pipeline(
        self,
        pipeline_id: str,
        app: Any,
        query: str,
        sources: Dict[str, bool],
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Execute a pipeline by ID."""
        func = self.get_pipeline(pipeline_id)
        
        if func:
            # Use TOML-configured pipeline
            try:
                return await func(app, query, sources, **kwargs)
            except Exception as e:
                logger.error(f"Error executing pipeline '{pipeline_id}': {e}")
                raise
        else:
            # Fallback to legacy implementation
            from ..Event_Handlers.Chat_Events import chat_rag_events
            
            if pipeline_id == "plain":
                return await chat_rag_events.perform_plain_rag_search(
                    app, query, sources, **kwargs
                )
            elif pipeline_id in ["semantic", "full"]:
                return await chat_rag_events.perform_full_rag_pipeline(
                    app, query, sources, **kwargs
                )
            elif pipeline_id == "hybrid":
                return await chat_rag_events.perform_hybrid_rag_search(
                    app, query, sources, **kwargs
                )
            # Check if it's a v2 pipeline and v2 is available
            elif pipeline_id.endswith('_v2'):
                # Try to use v2 adapter functions if available
                try:
                    from .pipeline_adapter import (
                        is_using_v2_pipelines,
                        perform_plain_rag_search_v2,
                        perform_full_rag_pipeline_v2,
                        perform_hybrid_rag_search_v2
                    )
                    
                    if is_using_v2_pipelines():
                        if pipeline_id == "plain_v2":
                            return await perform_plain_rag_search_v2(
                                app, query, sources, **kwargs
                            )
                        elif pipeline_id == "semantic_v2":
                            return await perform_full_rag_pipeline_v2(
                                app, query, sources, **kwargs
                            )
                        elif pipeline_id == "hybrid_v2":
                            return await perform_hybrid_rag_search_v2(
                                app, query, sources, **kwargs
                            )
                except ImportError:
                    logger.warning(f"V2 pipelines not available for '{pipeline_id}'")
                    
                # Fall back to base version without _v2
                base_id = pipeline_id[:-3]
                if base_id in self.legacy_mode_mapping:
                    return await self.execute_pipeline(
                        base_id, app, query, sources, **kwargs
                    )
            else:
                raise ValueError(f"Unknown pipeline: {pipeline_id}")
    
    def get_pipeline_parameters(self, pipeline_id: str) -> Dict[str, Any]:
        """Get default parameters for a pipeline."""
        config = self.loader.get_pipeline_config(pipeline_id)
        if config:
            return config.parameters
        
        # Return empty dict for legacy pipelines
        return {}
    
    def validate_pipeline_id(self, pipeline_id: str) -> bool:
        """Check if a pipeline ID is valid."""
        # Check TOML pipelines
        if self.loader.get_pipeline_config(pipeline_id):
            return True
        
        # Check legacy modes
        return pipeline_id in self.legacy_mode_mapping.values()
    
    def get_pipelines_by_tag(self, tag: str) -> List[str]:
        """Get all pipeline IDs that have a specific tag."""
        pipelines = []
        for pipeline_id, config in self.loader.pipelines.items():
            if tag in config.tags and config.enabled:
                pipelines.append(pipeline_id)
        return pipelines
    
    def export_pipeline(self, pipeline_id: str, output_file: Path) -> bool:
        """Export a pipeline configuration to a TOML file."""
        return self.loader.export_pipeline_config(pipeline_id, output_file)


# Global pipeline manager instance
_pipeline_manager: Optional[PipelineManager] = None


def get_pipeline_manager(config: Optional[RAGConfig] = None) -> PipelineManager:
    """Get or create the global pipeline manager."""
    global _pipeline_manager
    if _pipeline_manager is None:
        _pipeline_manager = PipelineManager(config)
    return _pipeline_manager


def reload_all_pipelines() -> None:
    """Reload all pipeline configurations."""
    manager = get_pipeline_manager()
    manager.reload_pipelines()


def get_available_pipeline_ids() -> List[str]:
    """Get list of all available pipeline IDs."""
    manager = get_pipeline_manager()
    return [p["id"] for p in manager.list_available_pipelines()]