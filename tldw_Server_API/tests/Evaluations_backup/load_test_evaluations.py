#!/usr/bin/env python3
"""
Load testing script for evaluation endpoints using Locust.

Usage:
    # Run with web UI
    locust -f load_test_evaluations.py --host=http://localhost:8000
    
    # Run headless
    locust -f load_test_evaluations.py --host=http://localhost:8000 --headless \
        --users 100 --spawn-rate 10 --run-time 5m
    
Requirements:
    pip install locust
"""

import json
import random
import time
from typing import Dict, List
from locust import HttpUser, task, between, events
from locust.stats import stats_printer, stats_history
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationUser(HttpUser):
    """Simulated user for evaluation API load testing"""
    
    # Wait between 1 and 3 seconds between tasks
    wait_time = between(1, 3)
    
    def on_start(self):
        """Initialize test data on user spawn"""
        self.sample_texts = [
            "Climate change is one of the most pressing issues of our time.",
            "Artificial intelligence is revolutionizing various industries.",
            "The global economy faces numerous challenges in the coming years.",
            "Space exploration continues to push the boundaries of human knowledge.",
            "Healthcare systems worldwide are adapting to new technologies."
        ]
        
        self.sample_summaries = [
            "Climate change is a critical global issue.",
            "AI is transforming industries.",
            "Economic challenges lie ahead globally.",
            "Space exploration advances human knowledge.",
            "Healthcare embraces new tech."
        ]
        
        self.sample_queries = [
            "What is climate change?",
            "How does AI work?",
            "What are economic indicators?",
            "Why explore space?",
            "What is telemedicine?"
        ]
        
        self.sample_contexts = [
            ["Climate change refers to long-term shifts in temperatures and weather patterns."],
            ["AI systems use algorithms to process data and make decisions."],
            ["Economic indicators include GDP, inflation, and unemployment rates."],
            ["Space exploration helps us understand our universe and develop new technologies."],
            ["Telemedicine allows remote healthcare delivery through technology."]
        ]
        
        self.sample_responses = [
            "Climate change is the long-term alteration of temperature and weather patterns.",
            "AI works by processing large amounts of data through algorithms.",
            "Economic indicators are statistics that show economic performance.",
            "We explore space to expand knowledge and develop technology.",
            "Telemedicine is remote healthcare using digital technology."
        ]
    
    @task(3)
    def test_geval(self):
        """Test G-Eval endpoint (most common)"""
        payload = {
            "source_text": random.choice(self.sample_texts),
            "summary": random.choice(self.sample_summaries),
            "metrics": ["fluency", "consistency", "relevance", "coherence"],
            "api_name": "openai",
            "save_results": False
        }
        
        with self.client.post(
            "/api/v1/evaluations/geval",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code == 429:
                # Rate limited - this is expected under load
                response.success()
            elif response.status_code != 200:
                response.failure(f"Got status code {response.status_code}")
    
    @task(2)
    def test_rag_evaluation(self):
        """Test RAG evaluation endpoint"""
        idx = random.randint(0, len(self.sample_queries) - 1)
        
        payload = {
            "query": self.sample_queries[idx],
            "retrieved_contexts": self.sample_contexts[idx],
            "generated_response": self.sample_responses[idx],
            "metrics": ["relevance", "faithfulness"],
            "api_name": "openai"
        }
        
        with self.client.post(
            "/api/v1/evaluations/rag",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code == 429:
                response.success()
            elif response.status_code != 200:
                response.failure(f"Got status code {response.status_code}")
    
    @task(1)
    def test_response_quality(self):
        """Test response quality endpoint"""
        payload = {
            "prompt": random.choice(self.sample_queries),
            "response": random.choice(self.sample_responses),
            "expected_format": "concise answer",
            "api_name": "openai"
        }
        
        with self.client.post(
            "/api/v1/evaluations/response-quality",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code == 429:
                response.success()
            elif response.status_code != 200:
                response.failure(f"Got status code {response.status_code}")
    
    @task(1)
    def test_batch_evaluation(self):
        """Test batch evaluation endpoint (resource intensive)"""
        items = []
        for i in range(random.randint(2, 5)):  # Small batch for load test
            items.append({
                "source_text": random.choice(self.sample_texts),
                "summary": random.choice(self.sample_summaries)
            })
        
        payload = {
            "evaluation_type": "geval",
            "items": items,
            "api_name": "openai",
            "parallel_workers": 2
        }
        
        with self.client.post(
            "/api/v1/evaluations/batch",
            json=payload,
            catch_response=True,
            timeout=60  # Longer timeout for batch
        ) as response:
            if response.status_code == 429:
                response.success()
            elif response.status_code != 200:
                response.failure(f"Got status code {response.status_code}")
    
    @task(1)
    def test_health_check(self):
        """Test health endpoint (should always work)"""
        with self.client.get(
            "/api/v1/health/evaluations",
            catch_response=True
        ) as response:
            if response.status_code in [200, 206]:  # 206 = degraded but working
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(1)
    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        with self.client.get(
            "/api/v1/evaluations/metrics",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Metrics endpoint failed: {response.status_code}")


class StressTestUser(HttpUser):
    """Aggressive user for stress testing"""
    
    # No wait time - hammer the API
    wait_time = between(0.1, 0.5)
    
    def on_start(self):
        """Initialize with minimal data"""
        self.test_text = "This is a test text for stress testing."
        self.test_summary = "Test summary."
    
    @task
    def stress_test_geval(self):
        """Rapidly hit the G-Eval endpoint"""
        payload = {
            "source_text": self.test_text,
            "summary": self.test_summary,
            "metrics": ["fluency"],  # Minimal metrics
            "api_name": "openai"
        }
        
        with self.client.post(
            "/api/v1/evaluations/geval",
            json=payload,
            catch_response=True
        ) as response:
            # Accept rate limiting as success during stress test
            if response.status_code in [200, 429]:
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")


# Custom event handlers for better reporting
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Log test start"""
    logger.info(f"Load test starting with host: {environment.host}")
    logger.info(f"Target thresholds:")
    logger.info("  - 100 concurrent users")
    logger.info("  - 1000 requests/minute sustained")
    logger.info("  - <2s response time p99")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Generate test report"""
    logger.info("\n" + "="*60)
    logger.info("LOAD TEST RESULTS")
    logger.info("="*60)
    
    stats = environment.stats
    
    # Overall statistics
    logger.info(f"\nTotal Requests: {stats.total.num_requests}")
    logger.info(f"Total Failures: {stats.total.num_failures}")
    logger.info(f"Failure Rate: {stats.total.fail_ratio:.2%}")
    logger.info(f"Average Response Time: {stats.total.avg_response_time:.2f}ms")
    logger.info(f"Median Response Time: {stats.total.median_response_time:.2f}ms")
    
    # Check if we met our targets
    success_criteria = {
        "Users": environment.runner.user_count >= 100,
        "RPS": stats.total.current_rps >= 16.67,  # 1000/minute = 16.67/second
        "P99 < 2s": stats.total.get_response_time_percentile(0.99) < 2000,
        "Failure Rate < 5%": stats.total.fail_ratio < 0.05
    }
    
    logger.info("\nSuccess Criteria:")
    for criterion, passed in success_criteria.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {status}: {criterion}")
    
    # Per-endpoint statistics
    logger.info("\nPer-Endpoint Statistics:")
    for name, entry in stats.entries.items():
        if entry.num_requests > 0:
            logger.info(f"\n  {name}:")
            logger.info(f"    Requests: {entry.num_requests}")
            logger.info(f"    Failures: {entry.num_failures}")
            logger.info(f"    Avg Time: {entry.avg_response_time:.2f}ms")
            logger.info(f"    P95 Time: {entry.get_response_time_percentile(0.95):.2f}ms")
            logger.info(f"    P99 Time: {entry.get_response_time_percentile(0.99):.2f}ms")
    
    # Overall pass/fail
    all_passed = all(success_criteria.values())
    logger.info("\n" + "="*60)
    if all_passed:
        logger.info("✅ LOAD TEST PASSED - All criteria met!")
    else:
        logger.info("❌ LOAD TEST FAILED - Some criteria not met")
    logger.info("="*60 + "\n")


if __name__ == "__main__":
    # Can also run directly with Python for basic testing
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        print("Running quick load test...")
        import subprocess
        
        cmd = [
            "locust",
            "-f", __file__,
            "--host", "http://localhost:8000",
            "--headless",
            "--users", "10",
            "--spawn-rate", "2",
            "--run-time", "30s",
            "--only-summary"
        ]
        
        subprocess.run(cmd)
    else:
        print("Usage:")
        print("  locust -f load_test_evaluations.py --host=http://localhost:8000")
        print("  python load_test_evaluations.py --quick  # Quick 30s test")