"""
Tests for the modular pipeline orchestrator.
"""

import pytest
import asyncio
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, patch

# Skip all tests - pipeline_orchestrator module no longer exists in v3
import pytest
pytestmark = pytest.mark.skip(reason="pipeline_orchestrator module deprecated - v3 uses functional_pipeline")

# Original import commented out as module no longer exists
# from tldw_Server_API.app.core.RAG.rag_service.pipeline_orchestrator import (
    PipelineOrchestrator, PipelineComponent, PipelineStage, PipelineContext,
    QueryExpansionComponent, CacheLookupComponent, RetrievalComponent,
    RerankingComponent, PerformanceMonitoringComponent,
    create_default_pipeline, CustomTableSerializationComponent
)
from tldw_Server_API.app.core.RAG.rag_service.types import Document, DataSource
from tldw_Server_API.app.core.RAG.rag_service.config import RAGConfig


class TestPipelineComponent:
    """Test base pipeline component functionality."""
    
    def test_component_initialization(self):
        """Test component initialization."""
        component = QueryExpansionComponent()
        
        assert component.name == "query_expansion"
        assert component.stage == PipelineStage.QUERY_EXPANSION
        assert component.priority == 10
        assert component.enabled is True
    
    def test_component_configuration(self):
        """Test component configuration."""
        component = QueryExpansionComponent()
        
        component.configure(strategies=["test"], custom_param="value")
        
        assert component.config["strategies"] == ["test"]
        assert component.config["custom_param"] == "value"
    
    def test_component_enable_disable(self):
        """Test enabling and disabling components."""
        component = QueryExpansionComponent()
        
        component.disable()
        assert component.enabled is False
        
        component.enable()
        assert component.enabled is True


class TestPipelineOrchestrator:
    """Test pipeline orchestrator functionality."""
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        config = RAGConfig()
        orchestrator = PipelineOrchestrator(config)
        
        assert orchestrator.config == config
        assert len(orchestrator._components) == len(PipelineStage)
        assert orchestrator.pipeline_metadata["components_registered"] == 0
    
    def test_component_registration(self):
        """Test registering components."""
        orchestrator = PipelineOrchestrator()
        
        # Register component instance
        component = QueryExpansionComponent()
        orchestrator.register_component(component)
        
        assert orchestrator.pipeline_metadata["components_registered"] == 1
        assert "query_expansion" in orchestrator._component_instances
        assert component in orchestrator._components[PipelineStage.QUERY_EXPANSION]
    
    def test_component_registration_by_class(self):
        """Test registering component by class."""
        orchestrator = PipelineOrchestrator()
        
        # Register by class with init kwargs
        orchestrator.register_component(
            QueryExpansionComponent,
            strategies=["custom"]
        )
        
        component = orchestrator.get_component("query_expansion")
        assert component is not None
        assert component.strategies == ["custom"]
    
    def test_component_unregistration(self):
        """Test unregistering components."""
        orchestrator = PipelineOrchestrator()
        
        component = QueryExpansionComponent()
        orchestrator.register_component(component)
        
        # Unregister
        result = orchestrator.unregister_component("query_expansion")
        
        assert result is True
        assert orchestrator.pipeline_metadata["components_registered"] == 0
        assert orchestrator.get_component("query_expansion") is None
    
    def test_component_priority_ordering(self):
        """Test components are ordered by priority."""
        orchestrator = PipelineOrchestrator()
        
        # Create mock components with different priorities
        class HighPriorityComponent(PipelineComponent):
            def __init__(self):
                super().__init__("high", PipelineStage.PRE_PROCESS, priority=1)
            
            async def execute(self, context):
                return context
        
        class LowPriorityComponent(PipelineComponent):
            def __init__(self):
                super().__init__("low", PipelineStage.PRE_PROCESS, priority=100)
            
            async def execute(self, context):
                return context
        
        orchestrator.register_component(LowPriorityComponent())
        orchestrator.register_component(HighPriorityComponent())
        
        components = orchestrator._components[PipelineStage.PRE_PROCESS]
        assert components[0].name == "high"
        assert components[1].name == "low"
    
    def test_component_configuration_via_orchestrator(self):
        """Test configuring components through orchestrator."""
        orchestrator = PipelineOrchestrator()
        component = QueryExpansionComponent()
        orchestrator.register_component(component)
        
        # Configure
        result = orchestrator.configure_component(
            "query_expansion",
            strategies=["modified"]
        )
        
        assert result is True
        assert component.config["strategies"] == ["modified"]
    
    def test_component_enable_disable_via_orchestrator(self):
        """Test enabling/disabling components through orchestrator."""
        orchestrator = PipelineOrchestrator()
        component = QueryExpansionComponent()
        orchestrator.register_component(component)
        
        # Disable
        result = orchestrator.disable_component("query_expansion")
        assert result is True
        assert component.enabled is False
        
        # Enable
        result = orchestrator.enable_component("query_expansion")
        assert result is True
        assert component.enabled is True
    
    def test_hook_registration(self):
        """Test hook registration."""
        orchestrator = PipelineOrchestrator()
        
        mock_hook = Mock()
        orchestrator.register_hook("before_stage", mock_hook)
        
        assert mock_hook in orchestrator._hooks["before_stage"]
    
    @pytest.mark.asyncio
    async def test_pipeline_execution(self):
        """Test basic pipeline execution."""
        orchestrator = PipelineOrchestrator()
        
        # Create mock component
        class MockComponent(PipelineComponent):
            def __init__(self):
                super().__init__("mock", PipelineStage.PROCESSING, priority=50)
            
            async def execute(self, context: PipelineContext) -> PipelineContext:
                context.metadata["mock_executed"] = True
                return context
        
        orchestrator.register_component(MockComponent())
        
        # Execute pipeline
        context = await orchestrator.execute("test query")
        
        assert context.query == "test query"
        assert context.metadata["mock_executed"] is True
        assert orchestrator.pipeline_metadata["executions"] == 1
    
    @pytest.mark.asyncio
    async def test_pipeline_with_disabled_component(self):
        """Test pipeline skips disabled components."""
        orchestrator = PipelineOrchestrator()
        
        # Create and register component
        class TestComponent(PipelineComponent):
            def __init__(self):
                super().__init__("test", PipelineStage.PROCESSING, priority=50)
            
            async def execute(self, context: PipelineContext) -> PipelineContext:
                context.metadata["should_not_execute"] = True
                return context
        
        orchestrator.register_component(TestComponent())
        orchestrator.disable_component("test")
        
        # Execute
        context = await orchestrator.execute("test")
        
        assert "should_not_execute" not in context.metadata
    
    @pytest.mark.asyncio
    async def test_pipeline_hooks_execution(self):
        """Test hooks are executed during pipeline."""
        orchestrator = PipelineOrchestrator()
        
        # Track hook calls
        hook_calls = []
        
        def before_hook(stage, context):
            hook_calls.append(("before", stage))
        
        def after_hook(stage, context):
            hook_calls.append(("after", stage))
        
        orchestrator.register_hook("before_stage", before_hook)
        orchestrator.register_hook("after_stage", after_hook)
        
        # Add a component
        orchestrator.register_component(QueryExpansionComponent())
        
        # Execute
        with patch.object(QueryExpansionComponent, 'execute', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = PipelineContext(query="test", original_query="test")
            await orchestrator.execute("test")
        
        # Check hooks were called
        assert any(call[0] == "before" for call in hook_calls)
        assert any(call[0] == "after" for call in hook_calls)
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in pipeline."""
        orchestrator = PipelineOrchestrator()
        
        # Create failing component
        class FailingComponent(PipelineComponent):
            def __init__(self):
                super().__init__("failing", PipelineStage.PROCESSING, priority=50)
            
            async def execute(self, context: PipelineContext) -> PipelineContext:
                raise ValueError("Test error")
        
        orchestrator.register_component(FailingComponent())
        
        # Track error hooks
        error_caught = []
        
        def error_hook(component, error, context):
            error_caught.append((component.name, str(error)))
        
        orchestrator.register_hook("on_error", error_hook)
        
        # Execute (should not raise by default)
        context = await orchestrator.execute("test")
        
        # Check error was caught
        assert len(error_caught) == 1
        assert error_caught[0][0] == "failing"
        assert "Test error" in error_caught[0][1]
        
        # Check stage result recorded error
        assert PipelineStage.PROCESSING in context.stage_results
        assert context.stage_results[PipelineStage.PROCESSING]["success"] is False
    
    def test_get_pipeline_info(self):
        """Test getting pipeline information."""
        orchestrator = create_default_pipeline()
        
        info = orchestrator.get_pipeline_info()
        
        assert "metadata" in info
        assert "stages" in info
        assert info["metadata"]["components_registered"] > 0
        
        # Check stage info
        for stage_name, stage_info in info["stages"].items():
            assert "components" in stage_info
            assert "count" in stage_info
            assert "enabled_count" in stage_info
    
    def test_reset_pipeline(self):
        """Test resetting pipeline."""
        orchestrator = PipelineOrchestrator()
        
        # Add components
        orchestrator.register_component(QueryExpansionComponent())
        orchestrator.register_component(CacheLookupComponent())
        
        assert orchestrator.pipeline_metadata["components_registered"] == 2
        
        # Reset
        orchestrator.reset_pipeline()
        
        assert orchestrator.pipeline_metadata["components_registered"] == 0
        assert len(orchestrator._component_instances) == 0


class TestDefaultPipeline:
    """Test default pipeline creation."""
    
    def test_create_default_pipeline(self):
        """Test creating default pipeline."""
        pipeline = create_default_pipeline()
        
        assert pipeline.pipeline_metadata["components_registered"] > 0
        
        # Check expected components are registered
        assert pipeline.get_component("query_expansion") is not None
        assert pipeline.get_component("cache_lookup") is not None
        assert pipeline.get_component("retrieval") is not None
        assert pipeline.get_component("reranking") is not None
        assert pipeline.get_component("performance_monitoring") is not None
    
    @pytest.mark.asyncio
    async def test_default_pipeline_execution(self):
        """Test executing default pipeline."""
        pipeline = create_default_pipeline()
        
        # Mock the actual execution methods
        with patch.object(QueryExpansionComponent, 'execute', new_callable=AsyncMock) as mock_expand:
            mock_expand.return_value = PipelineContext(
                query="test",
                original_query="test",
                expanded_queries=["test", "testing"]
            )
            
            context = await pipeline.execute("test query")
            
            assert context.query == "test query"
            assert pipeline.pipeline_metadata["executions"] == 1


class TestPipelineComponents:
    """Test individual pipeline components."""
    
    @pytest.mark.asyncio
    async def test_query_expansion_component(self):
        """Test query expansion component."""
        component = QueryExpansionComponent(strategies=["acronym"])
        context = PipelineContext(query="ML algorithms", original_query="ML algorithms")
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.query_expansion.HybridQueryExpansion') as mock_expander:
            mock_instance = mock_expander.return_value
            mock_instance.expand = AsyncMock(return_value=["ML algorithms", "machine learning algorithms"])
            
            result = await component.execute(context)
            
            assert len(result.expanded_queries) > 0
            assert result.metadata["query_expanded"] is True
    
    @pytest.mark.asyncio
    async def test_cache_lookup_component(self):
        """Test cache lookup component."""
        from tldw_Server_API.app.core.RAG.rag_service.semantic_cache import SemanticCache
        
        # Create mock cache
        mock_cache = Mock(spec=SemanticCache)
        mock_cache.find_similar = AsyncMock(return_value=None)
        
        component = CacheLookupComponent(cache=mock_cache)
        component.configure(enable_cache=True)
        
        context = PipelineContext(query="test", original_query="test")
        
        result = await component.execute(context)
        
        assert result.cache_hit is False
        mock_cache.find_similar.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_reranking_component(self):
        """Test reranking component."""
        component = RerankingComponent(strategy="flashrank")
        
        # Create test documents
        docs = [
            Document(id="1", content="Test 1", metadata={}, score=0.5, source=DataSource.MEDIA_DB),
            Document(id="2", content="Test 2", metadata={}, score=0.7, source=DataSource.MEDIA_DB),
        ]
        
        context = PipelineContext(
            query="test",
            original_query="test",
            documents=docs
        )
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.advanced_reranking.create_reranker') as mock_create:
            mock_reranker = Mock()
            mock_reranker.rerank = AsyncMock(return_value=[
                Mock(document=docs[1]),
                Mock(document=docs[0])
            ])
            mock_create.return_value = mock_reranker
            
            result = await component.execute(context)
            
            assert result.metadata["reranking_applied"] is True
            assert len(result.documents) == 2
    
    @pytest.mark.asyncio
    async def test_custom_table_serialization_component(self):
        """Test custom table serialization component."""
        component = CustomTableSerializationComponent()
        
        # Create documents with tables
        docs = [
            Document(
                id="1",
                content="Here is a table: | A | B |\n|---|---|\n| 1 | 2 |",
                metadata={},
                score=0.5,
                source=DataSource.MEDIA_DB
            )
        ]
        
        context = PipelineContext(
            query="test",
            original_query="test",
            documents=docs
        )
        
        with patch('tldw_Server_API.app.core.RAG.rag_service.table_serialization.TableProcessor') as mock_processor:
            mock_instance = mock_processor.return_value
            mock_instance.process_document_tables.return_value = (
                "Processed content",
                [{"type": "markdown", "headers": ["A", "B"]}]
            )
            
            result = await component.execute(context)
            
            assert result.documents[0].metadata.get("tables_processed", 0) > 0


class TestPipelineIntegration:
    """Test pipeline integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_flow(self):
        """Test complete pipeline flow with all components."""
        pipeline = create_default_pipeline()
        
        # Add custom component
        pipeline.register_component(CustomTableSerializationComponent())
        
        # Configure components
        pipeline.configure_component("query_expansion", strategies=["semantic"])
        pipeline.configure_component("reranking", strategy="diversity")
        
        # Get pipeline info before execution
        info_before = pipeline.get_pipeline_info()
        components_before = info_before["metadata"]["components_registered"]
        
        # Execute with mocked components
        with patch.object(QueryExpansionComponent, 'execute', new_callable=AsyncMock) as mock_expand:
            mock_expand.return_value = PipelineContext(
                query="test",
                original_query="test",
                expanded_queries=["test", "testing"]
            )
            
            context = await pipeline.execute("test query")
            
            assert context is not None
            assert pipeline.pipeline_metadata["executions"] == 1
            
            # Verify component count didn't change
            info_after = pipeline.get_pipeline_info()
            assert info_after["metadata"]["components_registered"] == components_before
    
    @pytest.mark.asyncio
    async def test_pipeline_with_performance_monitoring(self):
        """Test pipeline with performance monitoring enabled."""
        config = RAGConfig()
        config.log_performance_metrics = True
        
        pipeline = PipelineOrchestrator(config)
        
        # Add components
        pipeline.register_component(QueryExpansionComponent())
        pipeline.register_component(PerformanceMonitoringComponent())
        
        # Execute
        with patch.object(QueryExpansionComponent, 'execute', new_callable=AsyncMock) as mock_expand:
            mock_expand.return_value = PipelineContext(
                query="test",
                original_query="test"
            )
            
            context = await pipeline.execute("test")
            
            # Should have profiler
            assert context.profiler is not None
    
    @pytest.mark.asyncio
    async def test_dynamic_pipeline_modification(self):
        """Test dynamically modifying pipeline during runtime."""
        pipeline = PipelineOrchestrator()
        
        # Start with minimal pipeline
        pipeline.register_component(RetrievalComponent())
        
        # Execute once
        context1 = await pipeline.execute("query 1")
        assert pipeline.pipeline_metadata["executions"] == 1
        
        # Add more components
        pipeline.register_component(QueryExpansionComponent())
        pipeline.register_component(RerankingComponent())
        
        # Execute again with expanded pipeline
        with patch.object(QueryExpansionComponent, 'execute', new_callable=AsyncMock) as mock_expand:
            mock_expand.return_value = PipelineContext(
                query="query 2",
                original_query="query 2",
                expanded_queries=["query 2", "query two"]
            )
            
            context2 = await pipeline.execute("query 2")
            assert pipeline.pipeline_metadata["executions"] == 2
            
            # Should have more components now
            info = pipeline.get_pipeline_info()
            assert info["metadata"]["components_registered"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])