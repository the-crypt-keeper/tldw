#!/usr/bin/env python3
"""
Load testing script for the chat endpoint.

This script tests the chat endpoint under various load conditions to ensure:
- The security modules handle concurrent requests properly
- Database transactions work under load
- Streaming responses handle multiple connections
- Rate limiting works as expected
- Memory usage stays reasonable

Usage:
    python load_test_chat_endpoint.py [options]
    
Options:
    --url URL           API endpoint URL (default: http://localhost:8000)
    --users N           Number of concurrent users (default: 10)
    --duration S        Test duration in seconds (default: 60)
    --rate R            Requests per second per user (default: 1)
    --api-key KEY       API key for authentication
    --streaming         Test streaming responses
    --with-images       Include image inputs in requests
    --with-transactions Use database transactions
    --report FILE       Save report to file (default: load_test_report.json)
"""

import asyncio
import aiohttp
import argparse
import json
import time
import random
import statistics
import sys
import base64
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import psutil
import os


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    start_time: float
    end_time: float
    duration: float
    status_code: int
    error: Optional[str] = None
    response_size: int = 0
    streaming: bool = False
    conversation_id: Optional[str] = None


@dataclass
class LoadTestResults:
    """Aggregated results from load test."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_duration: float = 0
    requests_per_second: float = 0
    average_response_time: float = 0
    median_response_time: float = 0
    min_response_time: float = 0
    max_response_time: float = 0
    p95_response_time: float = 0
    p99_response_time: float = 0
    error_rate: float = 0
    status_codes: Dict[int, int] = None
    errors: Dict[str, int] = None
    memory_usage_mb: Dict[str, float] = None
    cpu_usage_percent: float = 0


class ChatEndpointLoadTester:
    """Load tester for chat endpoint."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        with_images: bool = False,
        with_transactions: bool = False
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key or "test-api-key"
        self.with_images = with_images
        self.with_transactions = with_transactions
        self.metrics: List[RequestMetrics] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        # Sample data for requests
        self.sample_messages = [
            "Hello, how are you today?",
            "Can you explain quantum computing?",
            "What's the weather like?",
            "Tell me a joke",
            "How do I make a REST API?",
            "What are the benefits of exercise?",
            "Explain machine learning in simple terms",
            "What's the capital of France?",
            "How do I debug Python code?",
            "What are best practices for API design?"
        ]
        
        self.sample_characters = [
            "Assistant",
            "Teacher",
            "Developer",
            "Scientist",
            None  # No character specified
        ]
        
        # Small 1x1 pixel images for testing (red, green, blue)
        self.sample_images = [
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==",
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPj/HwADBwIAMCbHYQAAAABJRU5ErkJggg=="
        ]
    
    def generate_request_payload(self, streaming: bool = False) -> Dict[str, Any]:
        """Generate a random request payload."""
        message_content = random.choice(self.sample_messages)
        
        # Build message content
        if self.with_images and random.random() < 0.3:  # 30% chance of including image
            content = [
                {"type": "text", "text": message_content},
                {"type": "image_url", "image_url": {"url": random.choice(self.sample_images)}}
            ]
        else:
            content = message_content
        
        payload = {
            "messages": [
                {"role": "user", "content": content}
            ],
            "model": random.choice(["gpt-4", "gpt-3.5-turbo", "claude-3"]),
            "provider": random.choice(["openai", "anthropic", "local"]),
            "temperature": random.uniform(0.3, 1.0),
            "max_tokens": random.randint(100, 500),
            "stream": streaming
        }
        
        # Optionally add character
        character = random.choice(self.sample_characters)
        if character:
            payload["character_id"] = character
        
        # Optionally use transactions
        if self.with_transactions:
            payload["use_transaction"] = True
        
        # Sometimes include conversation history
        if random.random() < 0.2:  # 20% chance
            payload["messages"].insert(0, {
                "role": "system",
                "content": "You are a helpful assistant."
            })
        
        return payload
    
    async def make_request(
        self,
        session: aiohttp.ClientSession,
        streaming: bool = False
    ) -> RequestMetrics:
        """Make a single request to the chat endpoint."""
        url = f"{self.base_url}/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = self.generate_request_payload(streaming)
        start_time = time.time()
        
        try:
            async with session.post(url, json=payload, headers=headers) as response:
                status_code = response.status
                
                if streaming and status_code == 200:
                    # Read streaming response
                    chunks = []
                    async for line in response.content:
                        chunks.append(len(line))
                        if b"[DONE]" in line:
                            break
                    response_size = sum(chunks)
                else:
                    # Read non-streaming response
                    content = await response.read()
                    response_size = len(content)
                    
                    if status_code == 200:
                        data = json.loads(content)
                        conversation_id = data.get("conversation_id")
                    else:
                        conversation_id = None
                
                end_time = time.time()
                duration = end_time - start_time
                
                return RequestMetrics(
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    status_code=status_code,
                    response_size=response_size,
                    streaming=streaming,
                    conversation_id=conversation_id
                )
                
        except asyncio.TimeoutError:
            end_time = time.time()
            return RequestMetrics(
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                status_code=0,
                error="Timeout"
            )
        except Exception as e:
            end_time = time.time()
            return RequestMetrics(
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                status_code=0,
                error=str(e)
            )
    
    async def run_user_simulation(
        self,
        session: aiohttp.ClientSession,
        duration: int,
        requests_per_second: float,
        streaming_ratio: float = 0.3
    ):
        """Simulate a single user making requests."""
        end_time = time.time() + duration
        request_interval = 1.0 / requests_per_second if requests_per_second > 0 else 1.0
        
        while time.time() < end_time:
            # Decide if this request should be streaming
            streaming = random.random() < streaming_ratio
            
            # Make request
            metric = await self.make_request(session, streaming)
            self.metrics.append(metric)
            
            # Wait before next request
            await asyncio.sleep(request_interval)
    
    async def run_load_test(
        self,
        num_users: int = 10,
        duration: int = 60,
        requests_per_second: float = 1.0,
        streaming: bool = False
    ) -> LoadTestResults:
        """Run the load test with specified parameters."""
        print(f"Starting load test with {num_users} users for {duration} seconds...")
        print(f"Target rate: {requests_per_second} requests/second/user")
        print(f"Features: images={self.with_images}, transactions={self.with_transactions}, streaming={streaming}")
        
        self.metrics = []
        self.start_time = time.time()
        
        # Get initial system metrics
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create session with connection pooling
        connector = aiohttp.TCPConnector(limit=num_users * 2)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Create user tasks
            streaming_ratio = 0.5 if streaming else 0.0
            tasks = [
                self.run_user_simulation(session, duration, requests_per_second, streaming_ratio)
                for _ in range(num_users)
            ]
            
            # Run all users concurrently
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self.end_time = time.time()
        
        # Get final system metrics
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        cpu_percent = process.cpu_percent()
        
        # Calculate results
        return self.calculate_results(initial_memory, final_memory, cpu_percent)
    
    def calculate_results(
        self,
        initial_memory: float,
        final_memory: float,
        cpu_percent: float
    ) -> LoadTestResults:
        """Calculate aggregated results from metrics."""
        if not self.metrics:
            return LoadTestResults()
        
        total_duration = self.end_time - self.start_time
        successful = [m for m in self.metrics if 200 <= m.status_code < 300]
        failed = [m for m in self.metrics if m.status_code == 0 or m.status_code >= 400]
        
        response_times = [m.duration for m in self.metrics if m.status_code > 0]
        
        # Count status codes
        status_codes = defaultdict(int)
        for metric in self.metrics:
            status_codes[metric.status_code] += 1
        
        # Count errors
        errors = defaultdict(int)
        for metric in self.metrics:
            if metric.error:
                errors[metric.error] += 1
        
        # Calculate percentiles
        if response_times:
            response_times.sort()
            p95_index = int(len(response_times) * 0.95)
            p99_index = int(len(response_times) * 0.99)
            
            results = LoadTestResults(
                total_requests=len(self.metrics),
                successful_requests=len(successful),
                failed_requests=len(failed),
                total_duration=total_duration,
                requests_per_second=len(self.metrics) / total_duration,
                average_response_time=statistics.mean(response_times),
                median_response_time=statistics.median(response_times),
                min_response_time=min(response_times),
                max_response_time=max(response_times),
                p95_response_time=response_times[p95_index] if p95_index < len(response_times) else response_times[-1],
                p99_response_time=response_times[p99_index] if p99_index < len(response_times) else response_times[-1],
                error_rate=(len(failed) / len(self.metrics)) * 100,
                status_codes=dict(status_codes),
                errors=dict(errors),
                memory_usage_mb={
                    "initial": initial_memory,
                    "final": final_memory,
                    "increase": final_memory - initial_memory
                },
                cpu_usage_percent=cpu_percent
            )
        else:
            results = LoadTestResults(
                total_requests=len(self.metrics),
                failed_requests=len(self.metrics),
                error_rate=100.0,
                errors=dict(errors)
            )
        
        return results
    
    def print_results(self, results: LoadTestResults):
        """Print load test results to console."""
        print("\n" + "="*60)
        print("LOAD TEST RESULTS")
        print("="*60)
        
        print(f"\nTotal Requests: {results.total_requests}")
        print(f"Successful: {results.successful_requests}")
        print(f"Failed: {results.failed_requests}")
        print(f"Error Rate: {results.error_rate:.2f}%")
        print(f"Duration: {results.total_duration:.2f} seconds")
        print(f"Requests/Second: {results.requests_per_second:.2f}")
        
        if results.successful_requests > 0:
            print(f"\nResponse Times (seconds):")
            print(f"  Average: {results.average_response_time:.3f}")
            print(f"  Median: {results.median_response_time:.3f}")
            print(f"  Min: {results.min_response_time:.3f}")
            print(f"  Max: {results.max_response_time:.3f}")
            print(f"  95th percentile: {results.p95_response_time:.3f}")
            print(f"  99th percentile: {results.p99_response_time:.3f}")
        
        if results.status_codes:
            print(f"\nStatus Codes:")
            for code, count in sorted(results.status_codes.items()):
                print(f"  {code}: {count}")
        
        if results.errors:
            print(f"\nErrors:")
            for error, count in sorted(results.errors.items()):
                print(f"  {error}: {count}")
        
        if results.memory_usage_mb:
            print(f"\nMemory Usage:")
            print(f"  Initial: {results.memory_usage_mb['initial']:.2f} MB")
            print(f"  Final: {results.memory_usage_mb['final']:.2f} MB")
            print(f"  Increase: {results.memory_usage_mb['increase']:.2f} MB")
        
        print(f"\nCPU Usage: {results.cpu_usage_percent:.2f}%")
        print("="*60)
    
    def save_report(self, results: LoadTestResults, filename: str):
        """Save results to JSON file."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "base_url": self.base_url,
                "with_images": self.with_images,
                "with_transactions": self.with_transactions
            },
            "results": asdict(results),
            "detailed_metrics": [asdict(m) for m in self.metrics[:100]]  # First 100 for detail
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed report saved to: {filename}")


async def main():
    """Main function to run load tests."""
    parser = argparse.ArgumentParser(description="Load test the chat endpoint")
    parser.add_argument("--url", default="http://localhost:8000", help="API endpoint URL")
    parser.add_argument("--users", type=int, default=10, help="Number of concurrent users")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--rate", type=float, default=1.0, help="Requests per second per user")
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument("--streaming", action="store_true", help="Test streaming responses")
    parser.add_argument("--with-images", action="store_true", help="Include image inputs")
    parser.add_argument("--with-transactions", action="store_true", help="Use database transactions")
    parser.add_argument("--report", default="load_test_report.json", help="Report file name")
    
    args = parser.parse_args()
    
    # Create tester
    tester = ChatEndpointLoadTester(
        base_url=args.url,
        api_key=args.api_key,
        with_images=args.with_images,
        with_transactions=args.with_transactions
    )
    
    # Run load test
    results = await tester.run_load_test(
        num_users=args.users,
        duration=args.duration,
        requests_per_second=args.rate,
        streaming=args.streaming
    )
    
    # Print and save results
    tester.print_results(results)
    tester.save_report(results, args.report)
    
    # Return exit code based on error rate
    if results.error_rate > 10:
        print("\n⚠️  High error rate detected!")
        return 1
    else:
        print("\n✅ Load test completed successfully!")
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)