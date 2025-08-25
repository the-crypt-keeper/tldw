#!/usr/bin/env python3
"""Debug script to identify what's failing in sustained load test"""

import os
import httpx
import time
import random
from collections import Counter

os.environ["TEST_MODE"] = "true"

BASE_URL = "http://localhost:8000"
API_PREFIX = "/api/v1"

client = httpx.Client(base_url=BASE_URL, timeout=30)
# Set auth for single-user mode
client.headers.update({
    "X-API-KEY": "test-api-key-12345",
    "Token": "test-api-key-12345",
    "Authorization": "Bearer test-api-key-12345"
})

error_types = Counter()
success_count = 0
total_count = 0

print("Testing different endpoint types...")
print("-" * 50)

# Simulate 10 RPS for 5 seconds
start_time = time.time()
end_time = start_time + 5  # 5 seconds
target_rps = 10

while time.time() < end_time:
    request_start = time.time()
    request_type = random.choice(['health', 'list', 'search'])
    total_count += 1
    
    try:
        if request_type == 'health':
            response = client.get(f"{API_PREFIX}/health")
            response.raise_for_status()
        elif request_type == 'list':
            response = client.get(f"{API_PREFIX}/media/", params={"limit": 10})
            response.raise_for_status()
        else:  # search
            response = client.post(
                f"{API_PREFIX}/media/search",
                json={"query": "test"},
                params={"limit": 5}
            )
            response.raise_for_status()
        
        success_count += 1
        print(f"✓ {request_type}: Success")
        
    except httpx.HTTPStatusError as e:
        error_key = f"{request_type}_{e.response.status_code}"
        error_types[error_key] += 1
        print(f"✗ {request_type}: HTTP {e.response.status_code}")
        if e.response.status_code == 429:
            print(f"  Rate limited!")
        elif e.response.status_code == 500:
            print(f"  Server error: {e.response.text[:200]}")
    except Exception as e:
        error_key = f"{request_type}_exception"
        error_types[error_key] += 1
        print(f"✗ {request_type}: {type(e).__name__}: {str(e)[:100]}")
    
    # Try to maintain target RPS
    response_time = time.time() - request_start
    sleep_time = max(0, (1.0 / target_rps) - response_time)
    if sleep_time > 0:
        time.sleep(sleep_time)

print("\n" + "=" * 50)
print(f"Results: {success_count}/{total_count} successful ({success_count/total_count*100:.1f}%)")
print("\nError breakdown:")
for error_type, count in error_types.most_common():
    print(f"  {error_type}: {count}")