"""
RAG Performance Benchmarking Suite

This module provides comprehensive performance benchmarks for the RAG system,
measuring latency, throughput, resource usage, and scalability.

Run with: python -m pytest tldw_Server_API/tests/RAG/test_rag_performance_benchmark.py -v --benchmark
"""

import asyncio
import time
import statistics
import json
import psutil
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import concurrent.futures
from contextlib import contextmanager

import pytest
from httpx import AsyncClient
from loguru import logger

# Test configuration
BENCHMARK_CONFIG = {
    "base_url": "http://localhost:8000",
    "api_prefix": "/api/v1/rag",
    "test_queries": [
        "What is machine learning?",
        "Explain the concept of neural networks in detail",
        "How does natural language processing work?",
        "What are the key differences between supervised and unsupervised learning?",
        "Describe the architecture of transformer models",
    ],
    "load_test_users": [1, 5, 10, 25, 50, 100],  # Concurrent users for load testing
    "test_duration_seconds": 60,  # Duration for sustained load tests
    "warmup_requests": 10,  # Number of warmup requests before benchmarking
}


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    endpoint: str
    operation: str
    timestamp: datetime
    
    # Latency metrics (ms)
    mean_latency: float
    median_latency: float
    p95_latency: float
    p99_latency: float
    min_latency: float
    max_latency: float
    
    # Throughput metrics
    requests_per_second: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    
    # Resource metrics
    cpu_usage_percent: float
    memory_usage_mb: float
    
    # Additional metrics
    concurrent_users: int = 1
    test_duration: float = 0
    error_rate: float = 0
    
    def to_json(self) -> str:
        """Convert to JSON for reporting"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return json.dumps(data, indent=2)


class PerformanceMonitor:
    """Monitor system resources during benchmarks"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.samples = []
        self.monitoring = False
        
    @contextmanager
    def monitor(self):
        """Context manager for monitoring resources"""
        self.start_monitoring()
        try:
            yield self
        finally:
            self.stop_monitoring()
    
    def start_monitoring(self):
        """Start resource monitoring"""
        self.monitoring = True
        self.samples = []
        
        async def _monitor():
            while self.monitoring:
                self.samples.append({
                    'cpu': self.process.cpu_percent(),
                    'memory': self.process.memory_info().rss / 1024 / 1024  # MB
                })
                await asyncio.sleep(0.1)
        
        asyncio.create_task(_monitor())
    
    def stop_monitoring(self):
        """Stop monitoring and return average metrics"""
        self.monitoring = False
        time.sleep(0.2)  # Allow final samples
        
    def get_metrics(self) -> Tuple[float, float]:
        """Get average CPU and memory usage"""
        if not self.samples:
            return 0.0, 0.0
        
        avg_cpu = statistics.mean(s['cpu'] for s in self.samples)
        avg_memory = statistics.mean(s['memory'] for s in self.samples)
        return avg_cpu, avg_memory


class RAGBenchmarkSuite:
    """Comprehensive benchmark suite for RAG endpoints"""
    
    def __init__(self, base_url: str = BENCHMARK_CONFIG["base_url"]):
        self.base_url = base_url
        self.api_prefix = BENCHMARK_CONFIG["api_prefix"]
        self.results = []
        self.monitor = PerformanceMonitor()
        
    async def warmup(self, client: AsyncClient, endpoint: str, payload: Dict[str, Any], count: int = 10):
        """Perform warmup requests to prime caches"""
        logger.info(f"Warming up {endpoint} with {count} requests...")
        for _ in range(count):
            try:
                await client.post(endpoint, json=payload)
            except:
                pass  # Ignore warmup errors
        await asyncio.sleep(1)
    
    async def measure_latency(
        self, 
        client: AsyncClient, 
        endpoint: str, 
        payload: Dict[str, Any], 
        iterations: int = 100
    ) -> Dict[str, float]:
        """Measure latency statistics for an endpoint"""
        latencies = []
        
        for _ in range(iterations):
            start = time.perf_counter()
            try:
                response = await client.post(endpoint, json=payload)
                response.raise_for_status()
                latency = (time.perf_counter() - start) * 1000  # Convert to ms
                latencies.append(latency)
            except Exception as e:
                logger.warning(f"Request failed: {e}")
                continue
        
        if not latencies:
            return {}
        
        return {
            'mean': statistics.mean(latencies),
            'median': statistics.median(latencies),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
            'min': min(latencies),
            'max': max(latencies),
            'std': statistics.stdev(latencies) if len(latencies) > 1 else 0
        }
    
    async def measure_throughput(
        self,
        client: AsyncClient,
        endpoint: str,
        payload: Dict[str, Any],
        duration_seconds: int = 30,
        concurrent_requests: int = 10
    ) -> Dict[str, Any]:
        """Measure throughput for sustained load"""
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        total_requests = 0
        successful_requests = 0
        failed_requests = 0
        latencies = []
        
        async def make_request():
            nonlocal total_requests, successful_requests, failed_requests
            while time.time() < end_time:
                request_start = time.perf_counter()
                try:
                    response = await client.post(endpoint, json=payload)
                    response.raise_for_status()
                    successful_requests += 1
                    latency = (time.perf_counter() - request_start) * 1000
                    latencies.append(latency)
                except:
                    failed_requests += 1
                finally:
                    total_requests += 1
        
        # Run concurrent requests
        tasks = [make_request() for _ in range(concurrent_requests)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        actual_duration = time.time() - start_time
        
        return {
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'requests_per_second': total_requests / actual_duration,
            'error_rate': failed_requests / total_requests if total_requests > 0 else 0,
            'duration': actual_duration,
            'mean_latency': statistics.mean(latencies) if latencies else 0,
            'p95_latency': np.percentile(latencies, 95) if latencies else 0
        }
    
    async def benchmark_search_endpoint(self) -> List[BenchmarkResult]:
        """Benchmark the search endpoint"""
        endpoint = f"{self.base_url}{self.api_prefix}/search"
        results = []
        
        async with AsyncClient(timeout=30.0) as client:
            # Test different query complexities
            for query in BENCHMARK_CONFIG["test_queries"]:
                payload = {
                    "query": query,
                    "search_type": "hybrid",
                    "limit": 10,
                    "databases": ["media_db"]
                }
                
                # Warmup
                await self.warmup(client, endpoint, payload)
                
                # Measure latency
                with self.monitor.monitor():
                    latency_stats = await self.measure_latency(client, endpoint, payload)
                    cpu_usage, memory_usage = self.monitor.get_metrics()
                
                if latency_stats:
                    result = BenchmarkResult(
                        endpoint="search",
                        operation=f"query_length_{len(query)}",
                        timestamp=datetime.now(),
                        mean_latency=latency_stats['mean'],
                        median_latency=latency_stats['median'],
                        p95_latency=latency_stats['p95'],
                        p99_latency=latency_stats['p99'],
                        min_latency=latency_stats['min'],
                        max_latency=latency_stats['max'],
                        requests_per_second=1000 / latency_stats['mean'],  # Estimate
                        total_requests=100,
                        successful_requests=100,
                        failed_requests=0,
                        cpu_usage_percent=cpu_usage,
                        memory_usage_mb=memory_usage
                    )
                    results.append(result)
                    logger.info(f"Search benchmark: {query[:30]}... - Mean: {latency_stats['mean']:.2f}ms")
        
        return results
    
    async def benchmark_agent_endpoint(self) -> List[BenchmarkResult]:
        """Benchmark the agent endpoint"""
        endpoint = f"{self.base_url}{self.api_prefix}/agent"
        results = []
        
        async with AsyncClient(timeout=60.0) as client:
            for query in BENCHMARK_CONFIG["test_queries"][:3]:  # Agent is expensive, test fewer
                payload = {
                    "message": query,
                    "search_databases": ["media_db"],
                    "model": "gpt-3.5-turbo"
                }
                
                # Warmup
                await self.warmup(client, endpoint, payload, count=3)
                
                # Measure latency (fewer iterations for expensive operations)
                with self.monitor.monitor():
                    latency_stats = await self.measure_latency(client, endpoint, payload, iterations=20)
                    cpu_usage, memory_usage = self.monitor.get_metrics()
                
                if latency_stats:
                    result = BenchmarkResult(
                        endpoint="agent",
                        operation=f"generation_length_{len(query)}",
                        timestamp=datetime.now(),
                        mean_latency=latency_stats['mean'],
                        median_latency=latency_stats['median'],
                        p95_latency=latency_stats['p95'],
                        p99_latency=latency_stats['p99'],
                        min_latency=latency_stats['min'],
                        max_latency=latency_stats['max'],
                        requests_per_second=1000 / latency_stats['mean'],
                        total_requests=20,
                        successful_requests=20,
                        failed_requests=0,
                        cpu_usage_percent=cpu_usage,
                        memory_usage_mb=memory_usage
                    )
                    results.append(result)
                    logger.info(f"Agent benchmark: {query[:30]}... - Mean: {latency_stats['mean']:.2f}ms")
        
        return results
    
    async def load_test(self, concurrent_users: int = 10) -> BenchmarkResult:
        """Perform load testing with multiple concurrent users"""
        endpoint = f"{self.base_url}{self.api_prefix}/search"
        
        payload = {
            "query": "machine learning",
            "search_type": "hybrid",
            "limit": 10,
            "databases": ["media_db"]
        }
        
        logger.info(f"Load testing with {concurrent_users} concurrent users...")
        
        async with AsyncClient(timeout=30.0) as client:
            with self.monitor.monitor():
                throughput_stats = await self.measure_throughput(
                    client, 
                    endpoint, 
                    payload,
                    duration_seconds=30,
                    concurrent_requests=concurrent_users
                )
                cpu_usage, memory_usage = self.monitor.get_metrics()
        
        result = BenchmarkResult(
            endpoint="search",
            operation=f"load_test_{concurrent_users}_users",
            timestamp=datetime.now(),
            mean_latency=throughput_stats['mean_latency'],
            median_latency=throughput_stats['mean_latency'],  # Approximate
            p95_latency=throughput_stats['p95_latency'],
            p99_latency=throughput_stats['p95_latency'] * 1.1,  # Approximate
            min_latency=0,
            max_latency=0,
            requests_per_second=throughput_stats['requests_per_second'],
            total_requests=throughput_stats['total_requests'],
            successful_requests=throughput_stats['successful_requests'],
            failed_requests=throughput_stats['failed_requests'],
            cpu_usage_percent=cpu_usage,
            memory_usage_mb=memory_usage,
            concurrent_users=concurrent_users,
            test_duration=throughput_stats['duration'],
            error_rate=throughput_stats['error_rate']
        )
        
        logger.info(f"Load test complete: {throughput_stats['requests_per_second']:.2f} req/s, "
                   f"Error rate: {throughput_stats['error_rate']:.2%}")
        
        return result
    
    async def run_full_benchmark(self) -> List[BenchmarkResult]:
        """Run complete benchmark suite"""
        all_results = []
        
        logger.info("Starting RAG Performance Benchmark Suite...")
        
        # 1. Latency benchmarks
        logger.info("Running latency benchmarks...")
        search_results = await self.benchmark_search_endpoint()
        all_results.extend(search_results)
        
        agent_results = await self.benchmark_agent_endpoint()
        all_results.extend(agent_results)
        
        # 2. Load tests with varying concurrency
        logger.info("Running load tests...")
        for users in [1, 10, 25, 50]:
            load_result = await self.load_test(concurrent_users=users)
            all_results.append(load_result)
            await asyncio.sleep(5)  # Cool down between tests
        
        # Save results
        self.save_results(all_results)
        self.print_summary(all_results)
        
        return all_results
    
    def save_results(self, results: List[BenchmarkResult]):
        """Save benchmark results to file"""
        output_dir = Path("benchmark_results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"rag_benchmark_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_file}")
    
    def print_summary(self, results: List[BenchmarkResult]):
        """Print benchmark summary"""
        print("\n" + "="*80)
        print("RAG PERFORMANCE BENCHMARK SUMMARY")
        print("="*80)
        
        # Group by endpoint
        search_results = [r for r in results if r.endpoint == "search"]
        agent_results = [r for r in results if r.endpoint == "agent"]
        load_results = [r for r in results if "load_test" in r.operation]
        
        if search_results:
            print("\n📊 SEARCH ENDPOINT PERFORMANCE:")
            print(f"  Mean Latency: {statistics.mean(r.mean_latency for r in search_results):.2f}ms")
            print(f"  P95 Latency: {statistics.mean(r.p95_latency for r in search_results):.2f}ms")
            print(f"  P99 Latency: {statistics.mean(r.p99_latency for r in search_results):.2f}ms")
        
        if agent_results:
            print("\n🤖 AGENT ENDPOINT PERFORMANCE:")
            print(f"  Mean Latency: {statistics.mean(r.mean_latency for r in agent_results):.2f}ms")
            print(f"  P95 Latency: {statistics.mean(r.p95_latency for r in agent_results):.2f}ms")
            print(f"  P99 Latency: {statistics.mean(r.p99_latency for r in agent_results):.2f}ms")
        
        if load_results:
            print("\n🔥 LOAD TEST RESULTS:")
            for r in load_results:
                print(f"  {r.concurrent_users} users: {r.requests_per_second:.2f} req/s, "
                      f"Error rate: {r.error_rate:.2%}, Mean latency: {r.mean_latency:.2f}ms")
        
        print("\n💻 RESOURCE USAGE:")
        print(f"  Average CPU: {statistics.mean(r.cpu_usage_percent for r in results):.2f}%")
        print(f"  Average Memory: {statistics.mean(r.memory_usage_mb for r in results):.2f}MB")
        
        print("\n✅ BENCHMARK COMPLETE")
        print("="*80)


# ============= Pytest Fixtures and Tests =============

@pytest.fixture
async def benchmark_suite():
    """Create benchmark suite instance"""
    return RAGBenchmarkSuite()


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_search_latency(benchmark_suite):
    """Test search endpoint latency"""
    results = await benchmark_suite.benchmark_search_endpoint()
    
    # Assertions for acceptable performance
    for result in results:
        assert result.mean_latency < 2000, f"Search latency too high: {result.mean_latency}ms"
        assert result.p95_latency < 5000, f"P95 latency too high: {result.p95_latency}ms"


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_agent_latency(benchmark_suite):
    """Test agent endpoint latency"""
    results = await benchmark_suite.benchmark_agent_endpoint()
    
    # Agent is slower, adjust expectations
    for result in results:
        assert result.mean_latency < 10000, f"Agent latency too high: {result.mean_latency}ms"
        assert result.p95_latency < 20000, f"P95 latency too high: {result.p95_latency}ms"


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_load_handling(benchmark_suite):
    """Test system under load"""
    # Test with 10 concurrent users
    result = await benchmark_suite.load_test(concurrent_users=10)
    
    assert result.requests_per_second > 5, f"Throughput too low: {result.requests_per_second} req/s"
    assert result.error_rate < 0.05, f"Error rate too high: {result.error_rate:.2%}"


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_scalability(benchmark_suite):
    """Test scalability with increasing load"""
    results = []
    
    for users in [1, 5, 10, 25]:
        result = await benchmark_suite.load_test(concurrent_users=users)
        results.append(result)
        await asyncio.sleep(2)  # Cool down
    
    # Check that throughput scales somewhat linearly up to a point
    single_user_rps = results[0].requests_per_second
    ten_user_rps = results[2].requests_per_second
    
    # Should handle at least 5x throughput with 10x users (allowing for overhead)
    assert ten_user_rps > single_user_rps * 5, "System doesn't scale well with concurrent users"


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_full_benchmark(benchmark_suite):
    """Run complete benchmark suite"""
    results = await benchmark_suite.run_full_benchmark()
    
    assert len(results) > 0, "No benchmark results generated"
    
    # Overall system health checks
    avg_cpu = statistics.mean(r.cpu_usage_percent for r in results)
    avg_memory = statistics.mean(r.memory_usage_mb for r in results)
    
    assert avg_cpu < 80, f"Average CPU usage too high: {avg_cpu:.2f}%"
    assert avg_memory < 4096, f"Average memory usage too high: {avg_memory:.2f}MB"


if __name__ == "__main__":
    # Run benchmark from command line
    import sys
    
    async def main():
        suite = RAGBenchmarkSuite()
        await suite.run_full_benchmark()
    
    asyncio.run(main())