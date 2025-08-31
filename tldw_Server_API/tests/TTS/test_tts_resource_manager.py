# test_tts_resource_manager.py
# Description: Comprehensive tests for TTS resource management system
#
# Imports
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import Dict, Any
import psutil
import httpx
#
# Local Imports
from tldw_Server_API.app.core.TTS.tts_resource_manager import (
    TTSResourceManager,
    HTTPConnectionPool,
    MemoryMonitor,
    StreamingSession,
    StreamingSessionManager,
    get_resource_manager,
    reset_resource_manager
)
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSResourceError,
    TTSInsufficientMemoryError
)
#
#######################################################################################################################
#
# Test Memory Monitor

class TestMemoryMonitor:
    """Test the MemoryMonitor class"""
    
    def test_memory_monitor_initialization(self):
        """Test MemoryMonitor initialization"""
        monitor = MemoryMonitor()
        
        assert monitor.critical_threshold == 90
        assert monitor.warning_threshold == 80
        assert monitor._last_check_time == 0
        assert monitor._last_memory_usage is None
    
    def test_get_memory_usage(self):
        """Test getting memory usage statistics"""
        monitor = MemoryMonitor()
        usage = monitor.get_memory_usage()
        
        assert "total_mb" in usage
        assert "available_mb" in usage
        assert "used_mb" in usage
        assert "percent" in usage
        assert "process_mb" in usage
        
        assert usage["total_mb"] > 0
        assert usage["percent"] >= 0
        assert usage["percent"] <= 100
    
    def test_memory_thresholds(self):
        """Test memory threshold checks"""
        monitor = MemoryMonitor(critical_threshold=50, warning_threshold=30)
        
        # Mock psutil to control memory values
        with patch('psutil.virtual_memory') as mock_memory:
            # Test critical threshold
            mock_memory.return_value = Mock(
                total=1024 * 1024 * 1024,  # 1GB
                available=400 * 1024 * 1024,  # 400MB (60% used - over critical)
                percent=60
            )
            assert monitor.is_memory_critical() is True
            assert monitor.is_memory_warning() is True
            
            # Test warning threshold
            mock_memory.return_value = Mock(
                total=1024 * 1024 * 1024,
                available=600 * 1024 * 1024,  # 600MB (40% used - between warning and critical)
                percent=40
            )
            assert monitor.is_memory_critical() is False
            assert monitor.is_memory_warning() is True
            
            # Test normal operation
            mock_memory.return_value = Mock(
                total=1024 * 1024 * 1024,
                available=800 * 1024 * 1024,  # 800MB (20% used - below warning)
                percent=20
            )
            assert monitor.is_memory_critical() is False
            assert monitor.is_memory_warning() is False
    
    def test_memory_check_caching(self):
        """Test that memory checks are cached"""
        monitor = MemoryMonitor(check_interval=1.0)
        
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = Mock(
                total=1024 * 1024 * 1024,
                available=500 * 1024 * 1024,
                percent=50
            )
            
            # First call should check memory
            usage1 = monitor.get_memory_usage()
            assert mock_memory.call_count == 1
            
            # Immediate second call should use cache
            usage2 = monitor.get_memory_usage()
            assert mock_memory.call_count == 1
            assert usage1 == usage2
            
            # After interval, should check again
            monitor._last_check_time = 0  # Force cache expiry
            usage3 = monitor.get_memory_usage()
            assert mock_memory.call_count == 2


class TestHTTPConnectionPool:
    """Test the HTTPConnectionPool class"""
    
    @pytest.fixture
    def pool(self):
        """Create a connection pool instance"""
        return HTTPConnectionPool(max_connections=5, timeout=30.0)
    
    @pytest.mark.asyncio
    async def test_get_client_creates_new(self, pool):
        """Test getting a new HTTP client"""
        client = await pool.get_client("provider1", "https://api.example.com")
        
        assert client is not None
        assert isinstance(client, httpx.AsyncClient)
        assert "provider1" in pool._clients
        assert pool._clients["provider1"] == client
    
    @pytest.mark.asyncio
    async def test_get_client_reuses_existing(self, pool):
        """Test reusing existing HTTP client"""
        client1 = await pool.get_client("provider1", "https://api.example.com")
        client2 = await pool.get_client("provider1", "https://api.example.com")
        
        assert client1 is client2
        assert len(pool._clients) == 1
    
    @pytest.mark.asyncio
    async def test_get_client_different_providers(self, pool):
        """Test different clients for different providers"""
        client1 = await pool.get_client("provider1", "https://api1.example.com")
        client2 = await pool.get_client("provider2", "https://api2.example.com")
        
        assert client1 is not client2
        assert len(pool._clients) == 2
        assert "provider1" in pool._clients
        assert "provider2" in pool._clients
    
    @pytest.mark.asyncio
    async def test_close_client(self, pool):
        """Test closing a specific client"""
        client = await pool.get_client("provider1", "https://api.example.com")
        
        # Mock the aclose method
        client.aclose = AsyncMock()
        
        await pool.close_client("provider1")
        
        client.aclose.assert_called_once()
        assert "provider1" not in pool._clients
    
    @pytest.mark.asyncio
    async def test_close_all_clients(self, pool):
        """Test closing all clients"""
        # Create multiple clients
        client1 = await pool.get_client("provider1", "https://api1.example.com")
        client2 = await pool.get_client("provider2", "https://api2.example.com")
        
        # Mock aclose methods
        client1.aclose = AsyncMock()
        client2.aclose = AsyncMock()
        
        await pool.close_all()
        
        client1.aclose.assert_called_once()
        client2.aclose.assert_called_once()
        assert len(pool._clients) == 0
    
    @pytest.mark.asyncio
    async def test_connection_limits(self, pool):
        """Test connection pool limits"""
        pool = HTTPConnectionPool(max_connections=2)
        
        # Create clients up to the limit
        client1 = await pool.get_client("provider1", "https://api.example.com")
        client2 = await pool.get_client("provider2", "https://api.example.com")
        
        assert len(pool._clients) == 2
        
        # Creating another should still work (reuses or creates based on provider)
        client3 = await pool.get_client("provider3", "https://api.example.com")
        assert client3 is not None


class TestStreamingSession:
    """Test the StreamingSession class"""
    
    def test_streaming_session_creation(self):
        """Test creating a streaming session"""
        session = StreamingSession(
            session_id="test123",
            provider="openai",
            start_time=1234567890.0
        )
        
        assert session.session_id == "test123"
        assert session.provider == "openai"
        assert session.start_time == 1234567890.0
        assert session.bytes_streamed == 0
        assert session.chunks_sent == 0
        assert session.is_active is True
    
    def test_streaming_session_update(self):
        """Test updating streaming session stats"""
        session = StreamingSession(
            session_id="test123",
            provider="openai"
        )
        
        session.bytes_streamed += 1024
        session.chunks_sent += 1
        
        assert session.bytes_streamed == 1024
        assert session.chunks_sent == 1
        
        session.bytes_streamed += 512
        session.chunks_sent += 1
        
        assert session.bytes_streamed == 1536
        assert session.chunks_sent == 2


class TestStreamingSessionManager:
    """Test the StreamingSessionManager class"""
    
    @pytest.fixture
    def manager(self):
        """Create a session manager instance"""
        return StreamingSessionManager()
    
    @pytest.mark.asyncio
    async def test_create_session(self, manager):
        """Test creating a new streaming session"""
        session_id = await manager.create_session("openai")
        
        assert session_id is not None
        assert session_id in manager._sessions
        assert manager._sessions[session_id].provider == "openai"
        assert manager._sessions[session_id].is_active is True
    
    @pytest.mark.asyncio
    async def test_get_session(self, manager):
        """Test getting an existing session"""
        session_id = await manager.create_session("kokoro")
        session = await manager.get_session(session_id)
        
        assert session is not None
        assert session.session_id == session_id
        assert session.provider == "kokoro"
        
        # Non-existent session
        none_session = await manager.get_session("non_existent")
        assert none_session is None
    
    @pytest.mark.asyncio
    async def test_update_session(self, manager):
        """Test updating session statistics"""
        session_id = await manager.create_session("elevenlabs")
        
        await manager.update_session(session_id, bytes_sent=1024, chunks_sent=1)
        
        session = await manager.get_session(session_id)
        assert session.bytes_streamed == 1024
        assert session.chunks_sent == 1
        
        # Update again
        await manager.update_session(session_id, bytes_sent=512, chunks_sent=1)
        
        session = await manager.get_session(session_id)
        assert session.bytes_streamed == 1536
        assert session.chunks_sent == 2
    
    @pytest.mark.asyncio
    async def test_close_session(self, manager):
        """Test closing a session"""
        session_id = await manager.create_session("higgs")
        
        # Close the session
        stats = await manager.close_session(session_id)
        
        assert stats is not None
        assert stats["session_id"] == session_id
        assert stats["provider"] == "higgs"
        assert "duration" in stats
        assert "bytes_streamed"] in stats
        assert "chunks_sent"] in stats
        
        # Session should be removed
        assert session_id not in manager._sessions
        
        # Closing non-existent session
        none_stats = await manager.close_session("non_existent")
        assert none_stats is None
    
    @pytest.mark.asyncio
    async def test_cleanup_inactive_sessions(self, manager):
        """Test cleaning up inactive sessions"""
        # Create sessions
        session1 = await manager.create_session("provider1")
        session2 = await manager.create_session("provider2")
        
        # Make session1 inactive and old
        manager._sessions[session1].is_active = False
        manager._sessions[session1].start_time -= 7200  # 2 hours ago
        
        # Keep session2 active
        manager._sessions[session2].is_active = True
        
        # Run cleanup
        await manager.cleanup_inactive(max_age_seconds=3600)
        
        # Session1 should be removed, session2 should remain
        assert session1 not in manager._sessions
        assert session2 in manager._sessions
    
    @pytest.mark.asyncio
    async def test_get_active_sessions(self, manager):
        """Test getting active sessions"""
        # Create multiple sessions
        session1 = await manager.create_session("provider1")
        session2 = await manager.create_session("provider2")
        session3 = await manager.create_session("provider3")
        
        # Make one inactive
        manager._sessions[session2].is_active = False
        
        active = await manager.get_active_sessions()
        
        assert len(active) == 2
        assert session1 in [s.session_id for s in active]
        assert session3 in [s.session_id for s in active]
        assert session2 not in [s.session_id for s in active]


class TestTTSResourceManager:
    """Test the main TTSResourceManager class"""
    
    @pytest.fixture
    def resource_manager(self):
        """Create a resource manager instance"""
        config = {
            "max_connections": 10,
            "connection_timeout": 60,
            "memory_critical_threshold": 90,
            "memory_warning_threshold": 80
        }
        return TTSResourceManager(config)
    
    def test_resource_manager_initialization(self, resource_manager):
        """Test ResourceManager initialization"""
        assert resource_manager.connection_pool is not None
        assert resource_manager.memory_monitor is not None
        assert resource_manager.session_manager is not None
        assert len(resource_manager._registered_models) == 0
    
    @pytest.mark.asyncio
    async def test_get_http_client(self, resource_manager):
        """Test getting HTTP client through resource manager"""
        client = await resource_manager.get_http_client(
            provider="openai",
            base_url="https://api.openai.com"
        )
        
        assert client is not None
        assert isinstance(client, httpx.AsyncClient)
    
    def test_register_model(self, resource_manager):
        """Test registering a model"""
        mock_model = Mock()
        mock_cleanup = Mock()
        
        resource_manager.register_model(
            provider="kokoro",
            model_instance=mock_model,
            cleanup_callback=mock_cleanup
        )
        
        assert "kokoro" in resource_manager._registered_models
        assert resource_manager._registered_models["kokoro"]["model"] == mock_model
        assert resource_manager._registered_models["kokoro"]["cleanup"] == mock_cleanup
    
    @pytest.mark.asyncio
    async def test_unregister_model(self, resource_manager):
        """Test unregistering a model"""
        mock_model = Mock()
        mock_cleanup = AsyncMock()
        
        resource_manager.register_model(
            provider="higgs",
            model_instance=mock_model,
            cleanup_callback=mock_cleanup
        )
        
        await resource_manager.unregister_model("higgs")
        
        mock_cleanup.assert_called_once()
        assert "higgs" not in resource_manager._registered_models
    
    @pytest.mark.asyncio
    async def test_create_streaming_session(self, resource_manager):
        """Test creating a streaming session"""
        session_id = await resource_manager.create_streaming_session("elevenlabs")
        
        assert session_id is not None
        session = await resource_manager.session_manager.get_session(session_id)
        assert session.provider == "elevenlabs"
    
    @pytest.mark.asyncio
    async def test_cleanup_all(self, resource_manager):
        """Test cleaning up all resources"""
        # Register a model
        mock_model = Mock()
        mock_cleanup = AsyncMock()
        resource_manager.register_model("test", mock_model, mock_cleanup)
        
        # Create a client
        client = await resource_manager.get_http_client("test", "https://test.com")
        client.aclose = AsyncMock()
        
        # Create a session
        session_id = await resource_manager.create_streaming_session("test")
        
        # Cleanup all
        await resource_manager.cleanup_all()
        
        # Check everything was cleaned
        mock_cleanup.assert_called_once()
        client.aclose.assert_called()
        assert len(resource_manager._registered_models) == 0
        assert session_id not in resource_manager.session_manager._sessions
    
    def test_get_resource_statistics(self, resource_manager):
        """Test getting resource statistics"""
        # Register a model
        resource_manager.register_model("test", Mock(), Mock())
        
        stats = resource_manager.get_statistics()
        
        assert "memory" in stats
        assert "connections" in stats
        assert "models" in stats
        assert "sessions" in stats
        
        assert stats["models"]["registered"] == ["test"]
        assert stats["connections"]["active"] == 0  # No connections created yet


class TestResourceManagerSingleton:
    """Test the singleton resource manager functions"""
    
    @pytest.mark.asyncio
    async def test_get_resource_manager_singleton(self):
        """Test that get_resource_manager returns singleton"""
        manager1 = await get_resource_manager()
        manager2 = await get_resource_manager()
        
        assert manager1 is manager2
    
    @pytest.mark.asyncio
    async def test_reset_resource_manager(self):
        """Test resetting the resource manager"""
        manager1 = await get_resource_manager()
        
        await reset_resource_manager()
        
        manager2 = await get_resource_manager()
        
        # Should be different instances after reset
        assert manager1 is not manager2


class TestResourceManagerIntegration:
    """Integration tests for resource management"""
    
    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self):
        """Test handling memory pressure scenarios"""
        config = {
            "memory_critical_threshold": 50,  # Low threshold for testing
            "memory_warning_threshold": 30
        }
        manager = TTSResourceManager(config)
        
        with patch('psutil.virtual_memory') as mock_memory:
            # Simulate high memory usage
            mock_memory.return_value = Mock(
                total=1024 * 1024 * 1024,
                available=400 * 1024 * 1024,  # 60% used
                percent=60
            )
            
            # Should detect critical memory
            assert manager.memory_monitor.is_memory_critical() is True
            
            # Get memory stats through manager
            stats = manager.get_statistics()
            assert stats["memory"]["percent"] == 60
            assert stats["memory"]["is_critical"] is True
    
    @pytest.mark.asyncio
    async def test_concurrent_session_management(self):
        """Test managing multiple concurrent sessions"""
        manager = TTSResourceManager()
        
        # Create multiple sessions concurrently
        tasks = [
            manager.create_streaming_session(f"provider{i}")
            for i in range(5)
        ]
        
        session_ids = await asyncio.gather(*tasks)
        
        assert len(session_ids) == 5
        assert len(set(session_ids)) == 5  # All unique
        
        # Get active sessions
        active = await manager.session_manager.get_active_sessions()
        assert len(active) == 5
        
        # Close all sessions
        close_tasks = [
            manager.session_manager.close_session(sid)
            for sid in session_ids
        ]
        
        await asyncio.gather(*close_tasks)
        
        # No active sessions remaining
        active = await manager.session_manager.get_active_sessions()
        assert len(active) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])