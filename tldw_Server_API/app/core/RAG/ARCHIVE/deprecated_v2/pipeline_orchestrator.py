"""
Modular Pipeline Orchestrator for RAG Service

This module provides a plug-n-play architecture for the RAG pipeline,
allowing easy expansion and runtime configuration of components.

Key Features:
- Component registration system
- Pipeline stage management
- Runtime configuration
- Hook system for extensibility
- Performance monitoring integration
"""

import asyncio
from typing import Dict, List, Any, Optional, Callable, Type, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import inspect
from collections import OrderedDict
from contextlib import asynccontextmanager

from loguru import logger

from .types import Document, SearchResult, RAGContext, DataSource
from .config import RAGConfig
from .performance_monitor import PerformanceMonitor, QueryProfiler
from .semantic_cache import SemanticCache
from .chromadb_optimizer import ChromaDBOptimizer, ChromaDBOptimizationConfig


class PipelineStage(Enum):
    """Pipeline stages for the RAG system."""
    PRE_PROCESS = "pre_process"
    QUERY_EXPANSION = "query_expansion"
    CACHE_LOOKUP = "cache_lookup"
    RETRIEVAL = "retrieval"
    POST_RETRIEVAL = "post_retrieval"
    RERANKING = "reranking"
    PROCESSING = "processing"
    GENERATION = "generation"
    POST_PROCESS = "post_process"
    CACHE_STORE = "cache_store"


@dataclass
class PipelineContext:
    """Context passed through the pipeline stages."""
    query: str
    original_query: str
    expanded_queries: List[str] = field(default_factory=list)
    documents: List[Document] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    config: Optional[RAGConfig] = None
    cache_hit: bool = False
    stage_results: Dict[PipelineStage, Any] = field(default_factory=dict)
    performance_data: Dict[str, float] = field(default_factory=dict)
    profiler: Optional[QueryProfiler] = None


class PipelineComponent(ABC):
    """Base class for pipeline components."""
    
    def __init__(self, name: str, stage: PipelineStage, priority: int = 100):
        """
        Initialize pipeline component.
        
        Args:
            name: Component name
            stage: Pipeline stage this component belongs to
            priority: Execution priority (lower = earlier)
        """
        self.name = name
        self.stage = stage
        self.priority = priority
        self.enabled = True
        self.config = {}
    
    @abstractmethod
    async def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute the component logic.
        
        Args:
            context: Pipeline context
            
        Returns:
            Modified pipeline context
        """
        pass
    
    def configure(self, **kwargs) -> None:
        """Configure the component."""
        self.config.update(kwargs)
    
    def enable(self) -> None:
        """Enable the component."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable the component."""
        self.enabled = False


class PipelineOrchestrator:
    """
    Main orchestrator for the modular RAG pipeline.
    
    Features:
    - Plug-n-play component registration
    - Dynamic pipeline configuration
    - Hook system for extensibility
    - Performance monitoring
    - Error handling and recovery
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize the pipeline orchestrator.
        
        Args:
            config: RAG configuration
        """
        self.config = config or RAGConfig()
        
        # Component registry: stage -> list of components
        self._components: Dict[PipelineStage, List[PipelineComponent]] = {
            stage: [] for stage in PipelineStage
        }
        
        # Hook registry: stage -> list of hooks
        self._hooks: Dict[str, List[Callable]] = {
            "before_stage": [],
            "after_stage": [],
            "on_error": [],
            "on_success": []
        }
        
        # Performance monitoring
        self.monitor = PerformanceMonitor() if config and config.log_performance_metrics else None
        
        # Component instances (for stateful components)
        self._component_instances = {}
        
        # Pipeline metadata
        self.pipeline_metadata = {
            "version": "2.0",
            "components_registered": 0,
            "executions": 0
        }
        
        logger.info("Pipeline Orchestrator initialized")
    
    def register_component(
        self,
        component: Union[PipelineComponent, Type[PipelineComponent]],
        **init_kwargs
    ) -> None:
        """
        Register a component in the pipeline.
        
        Args:
            component: Component instance or class
            **init_kwargs: Arguments for component initialization
        """
        # Handle class vs instance
        if inspect.isclass(component):
            component_instance = component(**init_kwargs)
        else:
            component_instance = component
        
        # Add to appropriate stage
        stage_components = self._components[component_instance.stage]
        stage_components.append(component_instance)
        
        # Sort by priority
        stage_components.sort(key=lambda x: x.priority)
        
        # Store instance
        self._component_instances[component_instance.name] = component_instance
        
        self.pipeline_metadata["components_registered"] += 1
        
        logger.debug(f"Registered component '{component_instance.name}' "
                    f"for stage {component_instance.stage.value}")
    
    def unregister_component(self, name: str) -> bool:
        """
        Unregister a component from the pipeline.
        
        Args:
            name: Component name
            
        Returns:
            True if component was removed
        """
        for stage_components in self._components.values():
            for component in stage_components:
                if component.name == name:
                    stage_components.remove(component)
                    del self._component_instances[name]
                    self.pipeline_metadata["components_registered"] -= 1
                    logger.debug(f"Unregistered component '{name}'")
                    return True
        return False
    
    def get_component(self, name: str) -> Optional[PipelineComponent]:
        """Get a component by name."""
        return self._component_instances.get(name)
    
    def configure_component(self, name: str, **kwargs) -> bool:
        """
        Configure a specific component.
        
        Args:
            name: Component name
            **kwargs: Configuration parameters
            
        Returns:
            True if component was configured
        """
        component = self.get_component(name)
        if component:
            component.configure(**kwargs)
            logger.debug(f"Configured component '{name}' with {kwargs}")
            return True
        return False
    
    def enable_component(self, name: str) -> bool:
        """Enable a specific component."""
        component = self.get_component(name)
        if component:
            component.enable()
            logger.debug(f"Enabled component '{name}'")
            return True
        return False
    
    def disable_component(self, name: str) -> bool:
        """Disable a specific component."""
        component = self.get_component(name)
        if component:
            component.disable()
            logger.debug(f"Disabled component '{name}'")
            return True
        return False
    
    def register_hook(self, hook_type: str, callback: Callable) -> None:
        """
        Register a hook callback.
        
        Args:
            hook_type: Type of hook (before_stage, after_stage, on_error, on_success)
            callback: Callback function
        """
        if hook_type in self._hooks:
            self._hooks[hook_type].append(callback)
            logger.debug(f"Registered hook for '{hook_type}'")
        else:
            logger.warning(f"Unknown hook type: {hook_type}")
    
    async def _execute_stage(
        self,
        stage: PipelineStage,
        context: PipelineContext
    ) -> PipelineContext:
        """
        Execute all components for a given stage.
        
        Args:
            stage: Pipeline stage
            context: Pipeline context
            
        Returns:
            Modified context
        """
        components = self._components[stage]
        enabled_components = [c for c in components if c.enabled]
        
        if not enabled_components:
            return context
        
        # Execute before_stage hooks
        for hook in self._hooks["before_stage"]:
            await self._safe_call_hook(hook, stage, context)
        
        # Execute components
        for component in enabled_components:
            try:
                # Monitor performance
                if self.monitor:
                    with self.monitor.timer(f"{stage.value}.{component.name}"):
                        context = await component.execute(context)
                        
                        # Record in profiler
                        if context.profiler:
                            context.profiler.add_event(
                                f"{stage.value}.{component.name}",
                                {"component": component.name, "stage": stage.value}
                            )
                else:
                    context = await component.execute(context)
                
                # Store stage result
                context.stage_results[stage] = {
                    "component": component.name,
                    "success": True
                }
                
            except Exception as e:
                logger.error(f"Error in component '{component.name}': {e}")
                
                # Execute error hooks
                for hook in self._hooks["on_error"]:
                    await self._safe_call_hook(hook, component, e, context)
                
                # Store error in context
                context.stage_results[stage] = {
                    "component": component.name,
                    "success": False,
                    "error": str(e)
                }
                
                # Continue or raise based on configuration
                if self.config and self.config.get("stop_on_error", False):
                    raise
        
        # Execute after_stage hooks
        for hook in self._hooks["after_stage"]:
            await self._safe_call_hook(hook, stage, context)
        
        return context
    
    async def _safe_call_hook(self, hook: Callable, *args, **kwargs) -> None:
        """Safely call a hook function."""
        try:
            if asyncio.iscoroutinefunction(hook):
                await hook(*args, **kwargs)
            else:
                hook(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in hook: {e}")
    
    async def execute(
        self,
        query: str,
        context: Optional[PipelineContext] = None,
        **kwargs
    ) -> PipelineContext:
        """
        Execute the full pipeline.
        
        Args:
            query: Search query
            context: Optional existing context
            **kwargs: Additional parameters
            
        Returns:
            Final pipeline context
        """
        # Create or update context
        if context is None:
            context = PipelineContext(
                query=query,
                original_query=query,
                config=self.config,
                metadata=kwargs
            )
        
        # Create profiler if monitoring is enabled
        if self.monitor:
            context.profiler = QueryProfiler()
            context.profiler.start()
        
        self.pipeline_metadata["executions"] += 1
        
        try:
            # Execute pipeline stages in order
            for stage in PipelineStage:
                if self.monitor:
                    with self.monitor.timer(f"stage.{stage.value}"):
                        context = await self._execute_stage(stage, context)
                else:
                    context = await self._execute_stage(stage, context)
            
            # Execute success hooks
            for hook in self._hooks["on_success"]:
                await self._safe_call_hook(hook, context)
            
            # Record performance metrics
            if self.monitor and context.profiler:
                profile = context.profiler.get_profile()
                self.monitor.record_query(
                    query=query,
                    total_duration=profile.get("total_duration", 0),
                    component_timings=profile.get("summary", {}),
                    cache_hit=context.cache_hit
                )
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            # Re-raise after logging
            raise
        
        return context
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the current pipeline configuration."""
        info = {
            "metadata": self.pipeline_metadata,
            "stages": {}
        }
        
        for stage in PipelineStage:
            components = self._components[stage]
            info["stages"][stage.value] = {
                "components": [
                    {
                        "name": c.name,
                        "enabled": c.enabled,
                        "priority": c.priority,
                        "config": c.config
                    }
                    for c in components
                ],
                "count": len(components),
                "enabled_count": sum(1 for c in components if c.enabled)
            }
        
        return info
    
    def reset_pipeline(self) -> None:
        """Reset the pipeline to initial state."""
        for stage_components in self._components.values():
            stage_components.clear()
        self._component_instances.clear()
        self.pipeline_metadata["components_registered"] = 0
        logger.info("Pipeline reset")


# Example component implementations

class QueryExpansionComponent(PipelineComponent):
    """Component for query expansion using all available strategies."""
    
    def __init__(self, strategies: Optional[List[str]] = None):
        super().__init__(
            name="query_expansion",
            stage=PipelineStage.QUERY_EXPANSION,
            priority=10
        )
        self.strategies = strategies or ["acronym", "semantic", "domain", "entity"]
        self._expander = None
    
    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Expand the query using configured strategies."""
        from .query_expansion import (
            HybridQueryExpansion, AcronymExpansion, SemanticExpansion,
            DomainExpansion, EntityExpansion
        )
        
        # Initialize expander with all strategies
        if self._expander is None:
            self._expander = HybridQueryExpansion()
            
            # Add configured strategies
            if "acronym" in self.strategies:
                self._expander.add_strategy(AcronymExpansion())
            if "semantic" in self.strategies:
                self._expander.add_strategy(SemanticExpansion())
            if "domain" in self.strategies:
                self._expander.add_strategy(DomainExpansion())
            if "entity" in self.strategies:
                self._expander.add_strategy(EntityExpansion())
        
        # Expand query
        expanded = await self._expander.expand(context.query)
        
        context.expanded_queries = expanded
        context.metadata["query_expanded"] = True
        context.metadata["expansion_strategies"] = self.strategies
        
        logger.debug(f"Expanded query from '{context.query}' to {len(expanded)} variants")
        
        return context


class CacheLookupComponent(PipelineComponent):
    """Component for semantic cache lookup with adaptive thresholds."""
    
    def __init__(self, cache: Optional[SemanticCache] = None,
                 similarity_threshold: float = 0.85):
        super().__init__(
            name="cache_lookup",
            stage=PipelineStage.CACHE_LOOKUP,
            priority=5
        )
        self.similarity_threshold = similarity_threshold
        self._cache = cache
    
    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Check cache for similar queries using semantic similarity."""
        if not self.config.get("enable_cache", True):
            return context
        
        # Initialize cache if needed
        if self._cache is None:
            from .semantic_cache import SemanticCache, AdaptiveCache
            
            # Use adaptive cache for better performance
            if self.config.get("use_adaptive_cache", True):
                self._cache = AdaptiveCache(
                    initial_threshold=self.similarity_threshold
                )
            else:
                self._cache = SemanticCache(
                    similarity_threshold=self.similarity_threshold
                )
        
        # Check for similar cached queries
        cached_result = await self._cache.find_similar(context.query)
        
        if cached_result:
            cached_query, similarity = cached_result
            
            # Get cached documents if stored
            cached_docs = await self._cache.get(cached_query)
            if cached_docs:
                context.cache_hit = True
                context.documents = cached_docs
                context.metadata["cache_similarity"] = similarity
                context.metadata["cached_query"] = cached_query
                logger.info(f"Cache hit with similarity {similarity:.3f} for query: {cached_query[:50]}...")
                
                # Skip retrieval stage
                context.metadata["skip_retrieval"] = True
        
        return context


class RetrievalComponent(PipelineComponent):
    """Component for document retrieval."""
    
    def __init__(self, sources: Optional[List[DataSource]] = None):
        super().__init__(
            name="retrieval",
            stage=PipelineStage.RETRIEVAL,
            priority=50
        )
        self.sources = sources or [DataSource.MEDIA_DB]
    
    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Retrieve documents from configured sources."""
        if context.cache_hit:
            return context  # Skip retrieval if cache hit
        
        # This would integrate with actual retrieval logic
        # For now, placeholder
        context.metadata["sources_searched"] = len(self.sources)
        
        return context


class RerankingComponent(PipelineComponent):
    """Component for advanced document reranking with multiple strategies."""
    
    def __init__(self, strategy: str = "hybrid", top_k: int = 10):
        super().__init__(
            name="reranking",
            stage=PipelineStage.RERANKING,
            priority=70
        )
        self.strategy = strategy
        self.top_k = top_k
        self._reranker = None
    
    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Rerank documents using configured strategy."""
        if not context.documents:
            return context
        
        from .advanced_reranking import (
            create_reranker, RerankingStrategy, RerankingConfig
        )
        
        # Initialize reranker if needed
        if self._reranker is None:
            # Create config with settings
            config = RerankingConfig(
                strategy=RerankingStrategy[self.strategy.upper()],
                top_k=self.top_k,
                diversity_weight=self.config.get("diversity_weight", 0.3),
                relevance_weight=self.config.get("relevance_weight", 0.7),
                min_similarity_threshold=self.config.get("min_similarity", 0.3)
            )
            
            self._reranker = create_reranker(
                RerankingStrategy[self.strategy.upper()],
                config
            )
        
        # Get original scores if available
        original_scores = [doc.score for doc in context.documents]
        
        # Rerank documents
        scored_docs = await self._reranker.rerank(
            context.query,
            context.documents,
            original_scores
        )
        
        # Update documents with reranked results
        context.documents = [sd.document for sd in scored_docs[:self.top_k]]
        
        # Store reranking metadata
        context.metadata["reranking_applied"] = True
        context.metadata["reranking_strategy"] = self.strategy
        context.metadata["documents_before_rerank"] = len(original_scores)
        context.metadata["documents_after_rerank"] = len(context.documents)
        
        logger.debug(f"Reranked {len(original_scores)} docs to {len(context.documents)} using {self.strategy}")
        
        return context


class PerformanceMonitoringComponent(PipelineComponent):
    """Component for performance monitoring."""
    
    def __init__(self, monitor: Optional[PerformanceMonitor] = None):
        super().__init__(
            name="performance_monitoring",
            stage=PipelineStage.POST_PROCESS,
            priority=200
        )
        self.monitor = monitor or PerformanceMonitor()
    
    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Record performance metrics."""
        if context.profiler:
            profile = context.profiler.get_profile()
            context.metadata["performance_profile"] = profile
            
            # Log slow queries
            total_duration = profile.get("total_duration", 0)
            if total_duration > 1.0:
                logger.warning(f"Slow query detected: {total_duration:.2f}s")
        
        return context


def create_default_pipeline(config: Optional[RAGConfig] = None) -> PipelineOrchestrator:
    """
    Create a default pipeline with all available components.
    
    This creates a fully-featured pipeline with:
    - Query expansion (acronym, semantic, domain, entity)
    - Semantic caching with adaptive thresholds
    - ChromaDB optimization for hybrid search (100k+ docs)
    - Table serialization for structured data
    - Advanced reranking (hybrid strategy by default)
    - Performance monitoring and profiling
    
    Args:
        config: RAG configuration
        
    Returns:
        Configured pipeline orchestrator with all modules
    """
    orchestrator = PipelineOrchestrator(config)
    
    # Query Expansion - All strategies
    orchestrator.register_component(QueryExpansionComponent(
        strategies=["acronym", "semantic", "domain", "entity"]
    ))
    
    # Semantic Cache Lookup
    orchestrator.register_component(CacheLookupComponent(
        similarity_threshold=0.85
    ))
    
    # ChromaDB Optimization for hybrid search
    orchestrator.register_component(ChromaDBOptimizationComponent(
        enable_hybrid=True,
        alpha=0.7  # Balance between vector and FTS
    ))
    
    # Document Retrieval
    orchestrator.register_component(RetrievalComponent(
        sources=[DataSource.MEDIA_DB, DataSource.NOTES, DataSource.CHAT_HISTORY]
    ))
    
    # Table Serialization
    orchestrator.register_component(TableSerializationComponent(
        serialize_method="hybrid"
    ))
    
    # Advanced Reranking
    orchestrator.register_component(RerankingComponent(
        strategy="hybrid",  # Uses FlashRank + Diversity + MultiCriteria
        top_k=10
    ))
    
    # Cache Storage for future queries
    orchestrator.register_component(CacheStoreComponent())
    
    # Performance Monitoring
    orchestrator.register_component(PerformanceMonitoringComponent())
    
    # Register hooks for monitoring and debugging
    def log_stage_start(stage: PipelineStage, context: PipelineContext):
        if context.profiler:
            context.profiler.add_event(f"stage_{stage.value}_start")
        logger.debug(f"Starting stage: {stage.value}")
    
    def log_stage_end(stage: PipelineStage, context: PipelineContext):
        if context.profiler:
            context.profiler.add_event(f"stage_{stage.value}_end")
        logger.debug(f"Completed stage: {stage.value}")
    
    async def log_success(context: PipelineContext):
        logger.info(f"Pipeline completed successfully for query: {context.query[:50]}...")
        if context.metadata:
            logger.debug(f"Metadata: {context.metadata}")
    
    def log_error(component: PipelineComponent, error: Exception, context: PipelineContext):
        logger.error(f"Error in {component.name}: {error}")
        # Could implement retry logic or fallback here
    
    orchestrator.register_hook("before_stage", log_stage_start)
    orchestrator.register_hook("after_stage", log_stage_end)
    orchestrator.register_hook("on_success", log_success)
    orchestrator.register_hook("on_error", log_error)
    
    logger.info(f"Created default pipeline with {orchestrator.pipeline_metadata['components_registered']} components")
    
    return orchestrator


def create_minimal_pipeline(config: Optional[RAGConfig] = None) -> PipelineOrchestrator:
    """
    Create a minimal pipeline for fast, simple queries.
    
    Args:
        config: RAG configuration
        
    Returns:
        Minimal pipeline orchestrator
    """
    orchestrator = PipelineOrchestrator(config)
    
    # Just retrieval and basic reranking
    orchestrator.register_component(RetrievalComponent())
    orchestrator.register_component(RerankingComponent(strategy="flashrank"))
    
    return orchestrator


def create_quality_pipeline(config: Optional[RAGConfig] = None) -> PipelineOrchestrator:
    """
    Create a quality-focused pipeline with all enhancement features.
    
    Args:
        config: RAG configuration
        
    Returns:
        Quality-optimized pipeline orchestrator
    """
    orchestrator = PipelineOrchestrator(config)
    
    # All expansion strategies
    orchestrator.register_component(QueryExpansionComponent(
        strategies=["acronym", "semantic", "domain", "entity"]
    ))
    
    # Adaptive cache
    cache_component = CacheLookupComponent(similarity_threshold=0.9)
    cache_component.configure(use_adaptive_cache=True)
    orchestrator.register_component(cache_component)
    
    # Hybrid search optimization
    orchestrator.register_component(ChromaDBOptimizationComponent(
        enable_hybrid=True,
        alpha=0.6  # Slightly favor vector search for quality
    ))
    
    # All data sources
    orchestrator.register_component(RetrievalComponent(
        sources=[DataSource.MEDIA_DB, DataSource.NOTES, 
                DataSource.CHAT_HISTORY, DataSource.CHARACTER_CARDS]
    ))
    
    # Advanced table processing
    orchestrator.register_component(TableSerializationComponent(
        serialize_method="hybrid"
    ))
    
    # Multi-stage reranking for best quality
    rerank = RerankingComponent(strategy="hybrid", top_k=20)
    rerank.configure(
        diversity_weight=0.4,  # More diversity
        relevance_weight=0.6
    )
    orchestrator.register_component(rerank)
    
    # Cache results
    orchestrator.register_component(CacheStoreComponent())
    
    # Full monitoring
    orchestrator.register_component(PerformanceMonitoringComponent())
    
    return orchestrator


# Example usage patterns

async def example_basic_usage():
    """Example of basic pipeline usage."""
    # Create pipeline
    pipeline = create_default_pipeline()
    
    # Execute query
    context = await pipeline.execute("What is machine learning?")
    
    # Access results
    documents = context.documents
    metadata = context.metadata
    
    return context


async def example_custom_pipeline():
    """Example of customizing the pipeline."""
    # Create empty pipeline
    pipeline = PipelineOrchestrator()
    
    # Add only the components you need
    pipeline.register_component(QueryExpansionComponent())
    pipeline.register_component(RetrievalComponent())
    
    # Configure components
    pipeline.configure_component("query_expansion", strategies=["semantic"])
    
    # Disable a component temporarily
    pipeline.disable_component("query_expansion")
    
    # Execute
    context = await pipeline.execute("test query")
    
    return context


async def example_with_hooks():
    """Example of using hooks for extensibility."""
    pipeline = create_default_pipeline()
    
    # Add custom hook
    async def custom_preprocessor(stage: PipelineStage, context: PipelineContext):
        if stage == PipelineStage.PRE_PROCESS:
            # Custom preprocessing logic
            context.query = context.query.lower().strip()
    
    pipeline.register_hook("before_stage", custom_preprocessor)
    
    # Add error handler
    def error_handler(component: PipelineComponent, error: Exception, context: PipelineContext):
        logger.error(f"Component {component.name} failed: {error}")
        # Could implement retry logic here
    
    pipeline.register_hook("on_error", error_handler)
    
    # Execute
    context = await pipeline.execute("Test Query")
    
    return context


class TableSerializationComponent(PipelineComponent):
    """Component for advanced table serialization and processing."""
    
    def __init__(self, serialize_method: str = "hybrid"):
        super().__init__(
            name="table_serialization",
            stage=PipelineStage.PROCESSING,
            priority=60
        )
        self.serialize_method = serialize_method
        self._processor = None
    
    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Process and serialize tables in documents."""
        if not context.documents:
            return context
        
        from .table_serialization import TableProcessor
        
        # Initialize processor
        if self._processor is None:
            self._processor = TableProcessor()
        
        total_tables = 0
        
        for doc in context.documents:
            # Check for potential tables
            if any(indicator in doc.content.lower() 
                   for indicator in ["table", "|", "csv", "tsv", "<table"]):
                
                # Process tables
                processed_content, tables = self._processor.process_document_tables(
                    doc.content,
                    serialize_method=self.serialize_method
                )
                
                if tables:
                    doc.content = processed_content
                    doc.metadata["tables_processed"] = len(tables)
                    doc.metadata["table_types"] = [t.get("type") for t in tables]
                    total_tables += len(tables)
        
        if total_tables > 0:
            context.metadata["total_tables_processed"] = total_tables
            logger.debug(f"Processed {total_tables} tables using {self.serialize_method} serialization")
        
        return context


class ChromaDBOptimizationComponent(PipelineComponent):
    """Component for ChromaDB hybrid search optimization."""
    
    def __init__(self, enable_hybrid: bool = True, alpha: float = 0.7):
        super().__init__(
            name="chromadb_optimization",
            stage=PipelineStage.RETRIEVAL,
            priority=55
        )
        self.enable_hybrid = enable_hybrid
        self.alpha = alpha
        self._optimizer = None
    
    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Apply ChromaDB optimizations for hybrid search."""
        # Skip if cache hit
        if context.cache_hit or context.metadata.get("skip_retrieval"):
            return context
        
        from .chromadb_optimizer import (
            ChromaDBOptimizer, ChromaDBOptimizationConfig,
            OptimizedChromaStore
        )
        
        # Initialize optimizer if needed
        if self._optimizer is None and self.enable_hybrid:
            config = ChromaDBOptimizationConfig(
                enable_hybrid_search=True,
                hybrid_alpha=self.alpha,
                cache_size=5000,  # Large cache for 100k+ docs
                batch_size=500,
                parallel_batch_workers=4,
                max_collection_size=100_000
            )
            self._optimizer = ChromaDBOptimizer(config)
        
        # This would integrate with actual ChromaDB retrieval
        # For now, we'll just add metadata
        if self._optimizer:
            context.metadata["chromadb_optimization"] = True
            context.metadata["hybrid_search_enabled"] = self.enable_hybrid
            context.metadata["hybrid_alpha"] = self.alpha
            
            # Log optimization stats
            stats = self._optimizer.get_stats()
            logger.debug(f"ChromaDB optimization active: {stats}")
        
        return context


class CacheStoreComponent(PipelineComponent):
    """Component for storing results in cache."""
    
    def __init__(self, cache: Optional[SemanticCache] = None):
        super().__init__(
            name="cache_store",
            stage=PipelineStage.CACHE_STORE,
            priority=100
        )
        self._cache = cache
    
    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Store query results in cache."""
        if context.cache_hit or not context.documents:
            return context  # Don't cache if already from cache or no results
        
        if self._cache is None:
            from .semantic_cache import SemanticCache
            self._cache = SemanticCache()
        
        # Store documents in cache
        await self._cache.add(
            context.query,
            context.documents
        )
        
        context.metadata["cached_for_future"] = True
        logger.debug(f"Cached {len(context.documents)} documents for query: {context.query[:50]}...")
        
        return context


if __name__ == "__main__":
    # Run example
    async def main():
        # Create and inspect pipeline
        pipeline = create_default_pipeline()
        
        # Get pipeline info
        info = pipeline.get_pipeline_info()
        print("Pipeline Configuration:")
        for stage, details in info["stages"].items():
            print(f"  {stage}: {details['enabled_count']} components enabled")
        
        # Execute query
        context = await pipeline.execute("What is RAG?")
        
        print(f"\nExecution complete:")
        print(f"  Cache hit: {context.cache_hit}")
        print(f"  Documents found: {len(context.documents)}")
        print(f"  Stages completed: {len(context.stage_results)}")
    
    asyncio.run(main())