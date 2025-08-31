"""
Unit tests for the analytics and feedback system.

Tests both the Analytics.db storage and ChaChaNotes_DB feedback integration.
"""

import pytest
import tempfile
import os
import time
from pathlib import Path
from datetime import datetime, timedelta

from tldw_Server_API.app.core.RAG.rag_service.analytics_db import (
    AnalyticsDatabase,
    get_analytics_db
)
from tldw_Server_API.app.core.RAG.rag_service.analytics_system import (
    AnalyticsStore,
    UserFeedbackStore,
    UnifiedFeedbackSystem,
    AnalyticsEvent,
    AnalyticsEventType,
    UserFeedback,
    FeedbackType
)


class TestAnalyticsDatabase:
    """Test the AnalyticsDatabase class."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        db = AnalyticsDatabase(db_path)
        yield db
        
        # Cleanup
        db.close()
        os.unlink(db_path)
    
    def test_database_initialization(self, temp_db):
        """Test that database tables are created properly."""
        # Check that we can query tables without error
        conn = temp_db.connection
        cursor = conn.cursor()
        
        # Check all expected tables exist
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' 
            ORDER BY name
        """)
        
        tables = [row[0] for row in cursor.fetchall()]
        expected_tables = [
            'ab_testing',
            'citation_analytics',
            'document_performance',
            'error_tracking',
            'feature_usage',
            'feedback_analytics',
            'query_patterns',
            'search_analytics',
            'system_performance'
        ]
        
        for table in expected_tables:
            assert table in tables
    
    def test_record_search(self, temp_db):
        """Test recording search analytics."""
        search_data = {
            'query': 'test query',
            'query_length': 10,
            'query_complexity': 'simple',
            'search_type': 'hybrid',
            'results_count': 5,
            'max_score': 0.95,
            'avg_score': 0.75,
            'response_time_ms': 150,
            'cache_hit': False,
            'reranking_used': True,
            'expansion_used': False,
            'filters_used': ['date', 'type']
        }
        
        temp_db.record_search(search_data)
        
        # Verify it was recorded
        cursor = temp_db.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM search_analytics")
        count = cursor.fetchone()[0]
        assert count == 1
        
        # Verify query was hashed
        cursor.execute("SELECT query_hash FROM search_analytics")
        query_hash = cursor.fetchone()[0]
        assert query_hash != 'test query'  # Should be hashed
        assert len(query_hash) == 16  # SHA256 truncated to 16 chars
    
    def test_record_document_performance(self, temp_db):
        """Test recording document performance metrics."""
        doc_data = {
            'document_id': 'doc123',
            'document_type': 'article',
            'chunk_size': 500,
            'retrieved': True,
            'relevance_score': 0.85,
            'cited': True,
            'feedback': 'positive'
        }
        
        # First retrieval
        temp_db.record_document_performance(doc_data)
        
        # Second retrieval of same document
        doc_data['feedback'] = 'negative'
        temp_db.record_document_performance(doc_data)
        
        # Check aggregated metrics
        cursor = temp_db.connection.cursor()
        cursor.execute("""
            SELECT retrieval_count, citation_count, 
                   feedback_positive, feedback_negative
            FROM document_performance
        """)
        
        row = cursor.fetchone()
        assert row[0] == 2  # retrieval_count
        assert row[1] == 2  # citation_count
        assert row[2] == 1  # feedback_positive
        assert row[3] == 1  # feedback_negative
    
    def test_record_feedback(self, temp_db):
        """Test recording anonymized feedback."""
        feedback_data = {
            'session_id': 'session123',
            'query': 'machine learning',
            'feedback_type': 'relevance',
            'rating': 4,
            'response_quality': 'good',
            'retrieval_accuracy': 'high',
            'response_time_acceptable': True,
            'categories': ['search', 'ml'],
            'improvement_areas': ['speed', 'accuracy']
        }
        
        temp_db.record_feedback(feedback_data)
        
        # Verify recording
        cursor = temp_db.connection.cursor()
        cursor.execute("""
            SELECT session_hash, rating, response_quality
            FROM feedback_analytics
        """)
        
        row = cursor.fetchone()
        assert row[0] != 'session123'  # Should be hashed
        assert row[1] == 4
        assert row[2] == 'good'
    
    def test_record_error(self, temp_db):
        """Test error tracking."""
        error_data = {
            'error_type': 'ConnectionError',
            'error_category': 'network',
            'component': 'retriever',
            'severity': 'high',
            'stack_trace': 'Traceback...'
        }
        
        # Record same error multiple times
        temp_db.record_error(error_data)
        temp_db.record_error(error_data)
        temp_db.record_error(error_data)
        
        # Check frequency tracking
        cursor = temp_db.connection.cursor()
        cursor.execute("""
            SELECT frequency FROM error_tracking
            WHERE error_type = ?
        """, ('ConnectionError',))
        
        frequency = cursor.fetchone()[0]
        assert frequency == 3
    
    def test_record_feature_usage(self, temp_db):
        """Test feature usage tracking."""
        feature_data = {
            'feature_name': 'semantic_search',
            'success': True,
            'execution_time_ms': 250
        }
        
        # Record multiple uses
        temp_db.record_feature_usage(feature_data)
        feature_data['success'] = False
        temp_db.record_feature_usage(feature_data)
        
        # Check aggregation
        cursor = temp_db.connection.cursor()
        cursor.execute("""
            SELECT usage_count, success_count, failure_count
            FROM feature_usage
            WHERE feature_name = ?
            AND DATE(timestamp) = DATE('now')
        """, ('semantic_search',))
        
        row = cursor.fetchone()
        assert row[0] == 2  # usage_count
        assert row[1] == 1  # success_count
        assert row[2] == 1  # failure_count
    
    def test_analytics_summary(self, temp_db):
        """Test getting analytics summary."""
        # Add some test data
        temp_db.record_search({
            'query': 'test',
            'results_count': 10,
            'response_time_ms': 100,
            'cache_hit': True,
            'error_occurred': False
        })
        
        temp_db.record_document_performance({
            'document_id': 'doc1',
            'retrieved': True,
            'cited': True,
            'feedback': 'positive'
        })
        
        # Get summary
        summary = temp_db.get_analytics_summary(days=7)
        
        assert 'search_analytics' in summary
        assert 'document_performance' in summary
        assert 'top_features' in summary
        assert 'top_errors' in summary
        
        # Check search stats
        search_stats = summary['search_analytics']
        assert search_stats['total_searches'] == 1
        assert search_stats['cache_hit_rate'] == 100.0
    
    def test_cleanup_old_data(self, temp_db):
        """Test cleaning up old analytics data."""
        # Add old data (simulated by direct SQL)
        conn = temp_db.connection
        cursor = conn.cursor()
        
        # Add old search record
        old_timestamp = (datetime.now() - timedelta(days=100)).isoformat()
        cursor.execute("""
            INSERT INTO search_analytics 
            (timestamp, query_hash, search_type, results_count)
            VALUES (?, ?, ?, ?)
        """, (old_timestamp, 'oldhash', 'fts', 5))
        
        # Add recent record
        temp_db.record_search({'query': 'recent', 'results_count': 10})
        
        # Run cleanup
        deleted = temp_db.cleanup_old_data(days_to_keep=90)
        
        assert deleted >= 1
        
        # Verify old data is gone
        cursor.execute("SELECT COUNT(*) FROM search_analytics")
        count = cursor.fetchone()[0]
        assert count == 1  # Only recent record remains


class TestAnalyticsStore:
    """Test the AnalyticsStore class."""
    
    @pytest.fixture
    def temp_store(self):
        """Create a temporary analytics store."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        store = AnalyticsStore(db_path)
        yield store
        
        # Cleanup
        os.unlink(db_path)
    
    @pytest.mark.asyncio
    async def test_record_search(self, temp_store):
        """Test async search recording."""
        search_data = {
            'query': 'async test',
            'results_count': 15,
            'response_time_ms': 200
        }
        
        result = await temp_store.record_search(search_data)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_record_feedback(self, temp_store):
        """Test async feedback recording."""
        feedback_data = {
            'session_id': 'test_session',
            'query': 'test query',
            'rating': 5
        }
        
        result = await temp_store.record_feedback(feedback_data)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_get_analytics_summary(self, temp_store):
        """Test async analytics summary retrieval."""
        summary = await temp_store.get_analytics_summary(days=30)
        assert isinstance(summary, dict)


class TestUnifiedFeedbackSystem:
    """Test the UnifiedFeedbackSystem class."""
    
    @pytest.fixture
    def temp_system(self):
        """Create a temporary unified feedback system."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            analytics_path = f.name
        
        system = UnifiedFeedbackSystem(
            analytics_db_path=analytics_path,
            chacha_db=None,  # Mock or skip ChaChaNotes for unit test
            enable_analytics=True
        )
        
        yield system
        
        # Cleanup
        os.unlink(analytics_path)
    
    @pytest.mark.asyncio
    async def test_record_search_with_analytics(self, temp_system):
        """Test recording search with analytics enabled."""
        result = await temp_system.record_search(
            query="test search",
            results_count=10,
            response_time_ms=150,
            search_type="hybrid"
        )
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_record_feedback_dual_storage(self, temp_system):
        """Test recording feedback to both stores."""
        feedback_id = await temp_system.record_feedback(
            query="test query",
            document_ids=["doc1", "doc2"],
            chunk_ids=["chunk1", "chunk2"],
            relevance_score=4,
            helpful=True,
            conversation_id=None  # No conversation in unit test
        )
        
        assert feedback_id is not None
        assert isinstance(feedback_id, str)


class TestConnectionPooling:
    """Test connection pooling functionality."""
    
    def test_connection_pool_import(self):
        """Test that connection pool can be imported."""
        from tldw_Server_API.app.core.RAG.rag_service.connection_pool import (
            ConnectionPool,
            MultiDatabasePool,
            get_global_pool_manager
        )
        
        assert ConnectionPool is not None
        assert MultiDatabasePool is not None
        assert get_global_pool_manager is not None
    
    def test_basic_pool_creation(self):
        """Test creating a basic connection pool."""
        from tldw_Server_API.app.core.RAG.rag_service.connection_pool import ConnectionPool
        
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        try:
            pool = ConnectionPool(
                db_path=db_path,
                min_connections=1,
                max_connections=5
            )
            
            # Test getting a connection
            with pool.get_connection() as conn:
                assert conn is not None
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                assert result[0] == 1
            
            # Get stats
            stats = pool.get_stats()
            assert stats['connections_created'] >= 1
            assert stats['connections_reused'] >= 0
            
            pool.close()
        finally:
            os.unlink(db_path)


class TestEmbeddingCache:
    """Test embedding cache functionality."""
    
    def test_embedding_cache_import(self):
        """Test that embedding cache can be imported."""
        from tldw_Server_API.app.core.RAG.rag_service.embedding_cache import (
            EmbeddingCache,
            EmbeddingCacheManager,
            get_global_cache_manager
        )
        
        assert EmbeddingCache is not None
        assert EmbeddingCacheManager is not None
        assert get_global_cache_manager is not None
    
    def test_basic_cache_operations(self):
        """Test basic cache get/put operations."""
        import numpy as np
        from tldw_Server_API.app.core.RAG.rag_service.embedding_cache import EmbeddingCache
        
        cache = EmbeddingCache(
            max_size=10,
            max_memory_mb=1,
            ttl_seconds=3600
        )
        
        # Test put and get
        text = "test text"
        embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        cache.put(text, embedding)
        retrieved = cache.get(text)
        
        assert retrieved is not None
        assert np.array_equal(retrieved, embedding)
        
        # Test cache miss
        missing = cache.get("non-existent text")
        assert missing is None
        
        # Test stats
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5
    
    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        import numpy as np
        from tldw_Server_API.app.core.RAG.rag_service.embedding_cache import EmbeddingCache
        
        cache = EmbeddingCache(
            max_size=3,  # Small cache for testing
            max_memory_mb=1
        )
        
        # Fill cache
        for i in range(4):
            text = f"text_{i}"
            embedding = np.array([float(i)])
            cache.put(text, embedding)
        
        # First item should be evicted
        assert cache.get("text_0") is None
        # Last three should still be there
        assert cache.get("text_1") is not None
        assert cache.get("text_2") is not None
        assert cache.get("text_3") is not None
        
        stats = cache.get_stats()
        assert stats['evictions'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])