#!/usr/bin/env python3
"""
Test the character rate limiter implementation.
"""

import asyncio
from tldw_Server_API.app.core.Character_Chat.character_rate_limiter import CharacterRateLimiter
from fastapi import HTTPException

async def test_rate_limiter():
    """Test rate limiter functionality."""
    print("Testing Character Rate Limiter")
    print("=" * 50)
    
    # Create rate limiter with low limits for testing
    limiter = CharacterRateLimiter(
        redis_client=None,  # Use in-memory for testing
        max_operations=5,
        window_seconds=10,
        max_characters=10,
        max_import_size_mb=1
    )
    
    user_id = 123
    
    # Test 1: Normal operations within limit
    print("\n1. Testing operations within limit:")
    for i in range(4):
        allowed, remaining = await limiter.check_rate_limit(user_id, f"op_{i}")
        print(f"   Operation {i+1}: Allowed={allowed}, Remaining={remaining}")
    
    # Test 2: Hit the rate limit
    print("\n2. Testing rate limit exceeded:")
    try:
        for i in range(3):
            allowed, remaining = await limiter.check_rate_limit(user_id, f"exceed_{i}")
            print(f"   Operation {i+5}: Allowed={allowed}, Remaining={remaining}")
    except HTTPException as e:
        print(f"   Rate limit hit: {e.status_code} - {e.detail}")
    
    # Test 3: Check usage stats
    print("\n3. Getting usage statistics:")
    stats = await limiter.get_usage_stats(user_id)
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test 4: Character limit check
    print("\n4. Testing character limit:")
    try:
        await limiter.check_character_limit(user_id, 5)
        print("   ✓ Under character limit (5/10)")
    except HTTPException as e:
        print(f"   ✗ Character limit exceeded: {e.detail}")
    
    try:
        await limiter.check_character_limit(user_id, 11)
        print("   ✓ Under character limit (11/10) - shouldn't see this")
    except HTTPException as e:
        print(f"   ✓ Character limit exceeded as expected: {e.detail}")
    
    # Test 5: File size check
    print("\n5. Testing file size limit:")
    try:
        limiter.check_import_size(500_000)  # 500KB
        print("   ✓ File size OK (500KB/1MB)")
    except HTTPException as e:
        print(f"   ✗ File too large: {e.detail}")
    
    try:
        limiter.check_import_size(2_000_000)  # 2MB
        print("   ✓ File size OK (2MB/1MB) - shouldn't see this")
    except HTTPException as e:
        print(f"   ✓ File size limit exceeded as expected: {e.detail}")
    
    print("\n" + "=" * 50)
    print("Rate limiter tests completed!")

if __name__ == "__main__":
    asyncio.run(test_rate_limiter())