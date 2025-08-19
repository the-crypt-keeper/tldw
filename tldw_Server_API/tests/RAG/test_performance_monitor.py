"""
Tests for performance monitoring functionality.
"""

import pytest
import time
import json
import tempfile
from pathlib import Path

from tldw_Server_API.app.core.RAG.rag_service.performance_monitor import (
    PerformanceMonitor, QueryProfiler, TimingMetric, PerformanceMetrics,
    get_monitor, set_monitor
)


class TestPerformanceMonitor:
    """Test performance monitoring functionality."""
    
    def test_timer_context_manager(self):
        """Test timing with context manager."""
        monitor = PerformanceMonitor(history_size=10)
        
        with monitor.timer("test_operation", {"key": "value"}):
            time.sleep(0.01)  # Simulate work
        
        # Check timing was recorded
        assert len(monitor._timings) == 1
        timing = monitor._timings[0]
        assert timing.name == "test_operation"
        assert timing.duration >= 0.01
        assert timing.metadata == {"key": "value"}
    
    def test_start_stop_timer(self):
        """Test manual timer start/stop."""
        monitor = PerformanceMonitor()
        
        monitor.start_timer("manual_timer")
        time.sleep(0.01)
        duration = monitor.stop_timer("manual_timer")
        
        assert duration >= 0.01
        assert len(monitor._timings) == 1
        assert monitor._timings[0].name == "manual_timer"
    
    def test_record_query_metrics(self):
        """Test recording complete query metrics."""
        monitor = PerformanceMonitor()
        
        # Record a successful query
        monitor.record_query(
            query="test query",
            total_duration=0.5,
            component_timings={
                "retrieval": 0.2,
                "reranking": 0.1,
                "generation": 0.2
            },
            cache_hit=False
        )
        
        # Record a cached query
        monitor.record_query(
            query="cached query",
            total_duration=0.01,
            component_timings={"cache_lookup": 0.01},
            cache_hit=True
        )
        
        # Check metrics
        assert monitor.total_queries == 2
        assert monitor.cache_hits == 1
        assert monitor.cache_misses == 1
        assert monitor.get_cache_hit_rate() == 0.5
    
    def test_component_statistics(self):
        """Test component-level statistics."""
        monitor = PerformanceMonitor()
        
        # Record multiple timings for a component
        for i in range(10):
            monitor.record_timing("retrieval", 0.1 + i * 0.01)
        
        stats = monitor.get_component_stats("retrieval")
        
        assert stats["count"] == 10
        assert stats["min"] == pytest.approx(0.1, rel=0.01)
        assert stats["max"] == pytest.approx(0.19, rel=0.01)
        assert stats["mean"] == pytest.approx(0.145, rel=0.01)
        # Median of 10 elements at index 5 (0-indexed) = 0.15
        assert stats["p50"] == pytest.approx(0.15, rel=0.01)
    
    def test_bottleneck_identification(self):
        """Test identifying performance bottlenecks."""
        monitor = PerformanceMonitor()
        
        # Create normal component timings
        for _ in range(50):
            monitor.record_timing("fast_component", 0.01)
        
        # Create component with high variance (potential bottleneck)
        for i in range(50):
            # Most are fast, but some are slow
            duration = 0.01 if i < 45 else 0.5
            monitor.record_timing("slow_component", duration)
        
        bottlenecks = monitor.identify_bottlenecks()
        
        # Should identify slow_component as bottleneck
        assert len(bottlenecks) > 0
        assert "slow_component" in bottlenecks[0]
    
    def test_summary_statistics(self):
        """Test summary statistics generation."""
        monitor = PerformanceMonitor()
        
        # Record some queries
        for i in range(5):
            monitor.record_query(
                query=f"query {i}",
                total_duration=0.1 * (i + 1),
                component_timings={"component": 0.05},
                cache_hit=(i % 2 == 0)
            )
        
        summary = monitor.get_summary_stats()
        
        assert summary["total_queries"] == 5
        assert summary["cache_hit_rate"] == 0.6  # 3 hits, 2 misses
        assert "avg_query_time" in summary
        assert "component_stats" in summary
    
    def test_export_metrics(self):
        """Test exporting metrics to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "metrics.json"
            monitor = PerformanceMonitor(export_path=export_path)
            
            # Record some data
            monitor.record_query(
                query="test",
                total_duration=0.5,
                component_timings={"test": 0.5}
            )
            
            # Export
            monitor.export_metrics()
            
            # Check file exists and is valid JSON
            assert export_path.exists()
            with open(export_path) as f:
                data = json.load(f)
            
            assert "summary" in data
            assert "recent_queries" in data
            assert data["summary"]["total_queries"] == 1
    
    def test_history_size_limit(self):
        """Test that history size is respected."""
        monitor = PerformanceMonitor(history_size=5)
        
        # Record more than history size
        for i in range(10):
            monitor.record_timing(f"operation_{i}", 0.1)
        
        # Should only keep last 5
        assert len(monitor._timings) == 5
        # First 5 should be gone
        timing_names = [t.name for t in monitor._timings]
        assert "operation_0" not in timing_names
        assert "operation_9" in timing_names
    
    def test_reset_metrics(self):
        """Test resetting all metrics."""
        monitor = PerformanceMonitor()
        
        # Add some data
        monitor.record_query("test", 0.1, {})
        monitor.record_timing("test", 0.1)
        
        # Reset
        monitor.reset()
        
        assert monitor.total_queries == 0
        assert len(monitor._timings) == 0
        assert len(monitor._query_metrics) == 0


class TestQueryProfiler:
    """Test query profiler functionality."""
    
    def test_profiling_events(self):
        """Test adding profiling events."""
        profiler = QueryProfiler()
        
        profiler.start()
        time.sleep(0.01)
        profiler.add_event("step_1", {"data": "value"})
        time.sleep(0.01)
        profiler.add_event("step_2", duration=0.005)
        
        profile = profiler.get_profile()
        
        assert profile["event_count"] == 3  # start + 2 events
        assert len(profile["events"]) == 3
        
        # Check events have correct data
        events = profile["events"]
        assert events[0]["name"] == "profiling_started"
        assert events[1]["name"] == "step_1"
        assert events[1]["data"] == {"data": "value"}
        assert events[2]["duration"] == 0.005
    
    def test_event_timing(self):
        """Test event timing calculations."""
        profiler = QueryProfiler()
        
        profiler.start()
        time.sleep(0.01)
        profiler.add_event("event_1")
        time.sleep(0.02)
        profiler.add_event("event_2")
        
        profile = profiler.get_profile()
        
        # Check elapsed times
        events = profile["events"]
        assert events[1]["elapsed"] >= 0.01
        assert events[2]["elapsed"] >= 0.03
        
        # Check time_since_last
        assert events[2]["time_since_last"] >= 0.02
    
    def test_profiling_summary(self):
        """Test profiling summary generation."""
        profiler = QueryProfiler()
        
        profiler.start()
        
        # Add multiple events of same type
        for i in range(3):
            profiler.add_event("repeated_event", duration=0.01 * (i + 1))
        
        profiler.add_event("single_event", duration=0.05)
        
        profile = profiler.get_profile()
        summary = profile["summary"]
        
        assert "repeated_event" in summary
        assert summary["repeated_event"]["count"] == 3
        assert summary["repeated_event"]["total_duration"] == pytest.approx(0.06, rel=0.01)
        assert summary["repeated_event"]["avg_duration"] == pytest.approx(0.02, rel=0.01)
        
        assert summary["single_event"]["count"] == 1
        assert summary["single_event"]["total_duration"] == 0.05


class TestGlobalMonitor:
    """Test global monitor instance."""
    
    def test_get_monitor(self):
        """Test getting global monitor."""
        monitor1 = get_monitor()
        monitor2 = get_monitor()
        
        # Should return same instance
        assert monitor1 is monitor2
    
    def test_set_monitor(self):
        """Test setting global monitor."""
        custom_monitor = PerformanceMonitor(history_size=100)
        set_monitor(custom_monitor)
        
        retrieved = get_monitor()
        assert retrieved is custom_monitor
        assert retrieved.history_size == 100


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""
    
    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = PerformanceMetrics(
            total_duration=1.0,
            component_timings={"retrieval": 0.5, "generation": 0.5},
            query_count=10,
            cache_hit_rate=0.7,
            memory_usage_mb=100.0,
            cpu_percent=25.0,
            timestamp=time.time()
        )
        
        data = metrics.to_dict()
        
        assert data["total_duration"] == 1.0
        assert data["component_timings"]["retrieval"] == 0.5
        assert data["cache_hit_rate"] == 0.7
    
    def test_metrics_to_json(self):
        """Test converting metrics to JSON."""
        metrics = PerformanceMetrics(
            total_duration=1.0,
            component_timings={"test": 1.0},
            query_count=1,
            cache_hit_rate=0.0,
            memory_usage_mb=50.0,
            cpu_percent=10.0,
            timestamp=time.time()
        )
        
        json_str = metrics.to_json()
        data = json.loads(json_str)
        
        assert data["total_duration"] == 1.0
        assert "component_timings" in data


class TestResourceMonitoring:
    """Test resource usage monitoring."""
    
    def test_resource_monitoring_disabled(self):
        """Test with resource monitoring disabled."""
        monitor = PerformanceMonitor(enable_resource_monitoring=False)
        
        usage = monitor.get_resource_usage()
        
        assert usage["memory_mb"] == 0.0
        assert usage["cpu_percent"] == 0.0
    
    def test_resource_monitoring_enabled(self):
        """Test with resource monitoring enabled."""
        monitor = PerformanceMonitor(enable_resource_monitoring=True)
        
        usage = monitor.get_resource_usage()
        
        # Should have non-zero memory usage
        assert usage["memory_mb"] > 0
        # CPU percent might be 0 if not doing much
        assert usage["cpu_percent"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])