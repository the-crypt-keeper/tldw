"""
Tests for middleware components (rate limiting, circuit breaking).
"""

import pytest
import time
import asyncio
from unittest.mock import Mock, patch, MagicMock

from tldw_Server_API.app.core.LLM_Calls.middleware import (
    # Rate limiting
    RateLimiter,
    RateLimitConfig,
    TokenBucket,
    get_rate_limiter,
    # Circuit breaking
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerManager,
    CircuitState,
    CircuitOpenError,
    get_circuit_breaker_manager,
)


class TestTokenBucket:
    """Tests for token bucket implementation."""
    
    def test_token_bucket_initialization(self):
        """Test token bucket initialization."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0)
        assert bucket.capacity == 10
        assert bucket.refill_rate == 2.0
        assert bucket.tokens == 10.0
    
    def test_consume_tokens(self):
        """Test consuming tokens from bucket."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        
        # Consume some tokens
        assert bucket.consume(5) is True
        assert bucket.tokens == 5.0
        
        # Try to consume more than available
        assert bucket.consume(6) is False
        assert bucket.tokens == 5.0  # Unchanged
        
        # Consume remaining
        assert bucket.consume(5) is True
        assert bucket.tokens == 0.0
    
    def test_token_refill(self):
        """Test token refilling over time."""
        bucket = TokenBucket(capacity=10, refill_rate=10.0)  # 10 tokens/second
        
        # Consume all tokens
        bucket.consume(10)
        assert bucket.tokens == 0.0
        
        # Wait for refill
        time.sleep(0.5)  # Should refill 5 tokens
        assert bucket.consume(5) is True
        assert bucket.consume(1) is False  # Not enough yet
    
    def test_wait_time_calculation(self):
        """Test wait time calculation."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0)  # 2 tokens/second
        
        # Consume all tokens
        bucket.consume(10)
        
        # Check wait time for different token amounts
        wait_time = bucket.wait_time(4)
        assert 1.9 <= wait_time <= 2.1  # ~2 seconds for 4 tokens at 2/sec
        
        # No wait if tokens available
        bucket.consume(-5)  # Hack to add tokens for testing
        bucket.tokens = 5
        assert bucket.wait_time(3) == 0.0


class TestRateLimiter:
    """Tests for rate limiter."""
    
    @pytest.fixture
    def rate_limiter(self):
        """Create a fresh rate limiter instance."""
        return RateLimiter()
    
    def test_rate_limiter_initialization(self, rate_limiter):
        """Test rate limiter initialization with defaults."""
        assert 'openai' in rate_limiter.provider_configs
        assert 'anthropic' in rate_limiter.provider_configs
        assert 'ollama' in rate_limiter.provider_configs
    
    def test_set_provider_config(self, rate_limiter):
        """Test setting provider configuration."""
        config = RateLimitConfig(
            requests_per_minute=100,
            tokens_per_minute=10000,
            burst_size=5
        )
        
        rate_limiter.set_provider_config('test_provider', config)
        assert 'test_provider' in rate_limiter.provider_configs
        assert rate_limiter.provider_configs['test_provider'] == config
    
    def test_check_rate_limit_allowed(self, rate_limiter):
        """Test rate limit check when allowed."""
        # Configure a test provider
        config = RateLimitConfig(requests_per_minute=60, burst_size=10)
        rate_limiter.set_provider_config('test', config)
        
        # Should allow initial requests
        allowed, wait_time = rate_limiter.check_rate_limit('test')
        assert allowed is True
        assert wait_time is None
    
    def test_check_rate_limit_exceeded(self, rate_limiter):
        """Test rate limit check when exceeded."""
        # Configure a very restrictive limit
        config = RateLimitConfig(requests_per_minute=1, burst_size=1)
        rate_limiter.set_provider_config('test', config)
        
        # First request should succeed
        allowed1, _ = rate_limiter.check_rate_limit('test')
        assert allowed1 is True
        
        # Immediate second request should fail
        allowed2, wait_time = rate_limiter.check_rate_limit('test')
        assert allowed2 is False
        assert wait_time > 0
    
    def test_token_limiting(self, rate_limiter):
        """Test token-based rate limiting."""
        config = RateLimitConfig(
            requests_per_minute=100,
            tokens_per_minute=1000,
            burst_size=10
        )
        rate_limiter.set_provider_config('test', config)
        
        # Large token request should be limited
        allowed, wait_time = rate_limiter.check_rate_limit('test', token_count=2000)
        assert allowed is False
        assert wait_time > 0
        
        # Small token request should succeed
        allowed, wait_time = rate_limiter.check_rate_limit('test', token_count=100)
        assert allowed is True
    
    def test_wait_if_needed(self, rate_limiter):
        """Test waiting for rate limit."""
        config = RateLimitConfig(requests_per_minute=60, burst_size=1)
        rate_limiter.set_provider_config('test', config)
        
        # Consume the burst
        rate_limiter.check_rate_limit('test')
        
        # Should wait and then succeed
        start_time = time.time()
        result = rate_limiter.wait_if_needed('test', max_wait=2.0)
        elapsed = time.time() - start_time
        
        assert result is True
        assert elapsed > 0  # Some waiting occurred
    
    def test_wait_if_needed_timeout(self, rate_limiter):
        """Test waiting timeout."""
        config = RateLimitConfig(requests_per_minute=1, burst_size=1)
        rate_limiter.set_provider_config('test', config)
        
        # Consume the limit
        rate_limiter.check_rate_limit('test')
        
        # Should fail if wait exceeds max
        result = rate_limiter.wait_if_needed('test', max_wait=0.01)
        assert result is False
    
    def test_get_stats(self, rate_limiter):
        """Test getting rate limiter statistics."""
        config = RateLimitConfig(requests_per_minute=100, burst_size=10)
        rate_limiter.set_provider_config('test', config)
        
        # Make some requests
        rate_limiter.check_rate_limit('test')
        rate_limiter.check_rate_limit('test', token_count=100)
        
        stats = rate_limiter.get_stats('test')
        assert stats['requests'] == 2
        assert stats['tokens'] == 100
    
    def test_get_limits(self, rate_limiter):
        """Test getting configured limits."""
        limits = rate_limiter.get_limits('openai')
        assert 'requests_per_minute' in limits
        assert 'burst_size' in limits
        
        # Non-existent provider
        assert rate_limiter.get_limits('nonexistent') is None


class TestCircuitBreaker:
    """Tests for circuit breaker."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout=1.0,  # 1 second for faster tests
            window_size=5,
            failure_rate_threshold=0.6
        )
    
    @pytest.fixture
    def breaker(self, config):
        """Create a circuit breaker."""
        return CircuitBreaker("test_service", config)
    
    def test_circuit_breaker_initialization(self, breaker):
        """Test circuit breaker initialization."""
        assert breaker.name == "test_service"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
        assert breaker.success_count == 0
    
    def test_successful_calls(self, breaker):
        """Test successful calls through circuit breaker."""
        def success_func():
            return "success"
        
        result = breaker.call(success_func)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED
        
        stats = breaker.get_stats()
        assert stats['successful_calls'] == 1
        assert stats['failed_calls'] == 0
    
    def test_circuit_opens_on_failures(self, breaker):
        """Test circuit opens after threshold failures."""
        def failing_func():
            raise Exception("Test failure")
        
        # Make failures up to threshold
        for i in range(3):
            with pytest.raises(Exception):
                breaker.call(failing_func)
        
        # Circuit should now be open
        assert breaker.state == CircuitState.OPEN
        
        # Further calls should be blocked
        with pytest.raises(CircuitOpenError):
            breaker.call(failing_func)
        
        stats = breaker.get_stats()
        assert stats['failed_calls'] == 3
        assert stats['blocked_calls'] == 1
    
    def test_circuit_half_open_after_timeout(self, breaker):
        """Test circuit transitions to half-open after timeout."""
        def failing_func():
            raise Exception("Test failure")
        
        # Open the circuit
        for i in range(3):
            with pytest.raises(Exception):
                breaker.call(failing_func)
        
        assert breaker.state == CircuitState.OPEN
        
        # Wait for timeout
        time.sleep(1.1)
        
        # Next call should transition to half-open
        def success_func():
            return "success"
        
        result = breaker.call(success_func)
        assert result == "success"
        assert breaker.state == CircuitState.HALF_OPEN
    
    def test_circuit_closes_after_successes(self, breaker):
        """Test circuit closes after success threshold in half-open."""
        def failing_func():
            raise Exception("Test failure")
        
        def success_func():
            return "success"
        
        # Open the circuit
        for i in range(3):
            with pytest.raises(Exception):
                breaker.call(failing_func)
        
        # Wait and transition to half-open
        time.sleep(1.1)
        breaker.call(success_func)
        assert breaker.state == CircuitState.HALF_OPEN
        
        # More successes to close circuit
        breaker.call(success_func)
        assert breaker.state == CircuitState.CLOSED
    
    def test_circuit_reopens_on_half_open_failure(self, breaker):
        """Test circuit reopens on failure in half-open state."""
        def failing_func():
            raise Exception("Test failure")
        
        def success_func():
            return "success"
        
        # Open the circuit
        for i in range(3):
            with pytest.raises(Exception):
                breaker.call(failing_func)
        
        # Wait and transition to half-open
        time.sleep(1.1)
        breaker.call(success_func)
        assert breaker.state == CircuitState.HALF_OPEN
        
        # Failure in half-open should reopen
        with pytest.raises(Exception):
            breaker.call(failing_func)
        
        assert breaker.state == CircuitState.OPEN
    
    def test_failure_rate_threshold(self, config):
        """Test circuit opens based on failure rate."""
        config.failure_threshold = 10  # High threshold
        config.failure_rate_threshold = 0.5  # 50% failure rate
        config.window_size = 4
        
        breaker = CircuitBreaker("test", config)
        
        def failing_func():
            raise Exception("Test")
        
        def success_func():
            return "success"
        
        # Create pattern: success, fail, fail, success (50% failure)
        breaker.call(success_func)
        with pytest.raises(Exception):
            breaker.call(failing_func)
        with pytest.raises(Exception):
            breaker.call(failing_func)
        breaker.call(success_func)
        
        # Should still be closed (exactly at threshold)
        assert breaker.state == CircuitState.CLOSED
        
        # One more failure tips it over
        with pytest.raises(Exception):
            breaker.call(failing_func)
        
        assert breaker.state == CircuitState.OPEN
    
    def test_manual_reset(self, breaker):
        """Test manual circuit reset."""
        def failing_func():
            raise Exception("Test failure")
        
        # Open the circuit
        for i in range(3):
            with pytest.raises(Exception):
                breaker.call(failing_func)
        
        assert breaker.state == CircuitState.OPEN
        
        # Manual reset
        breaker.reset()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_async_circuit_breaker(self, breaker):
        """Test async function calls through circuit breaker."""
        async def async_success():
            await asyncio.sleep(0.01)
            return "async_success"
        
        async def async_failure():
            await asyncio.sleep(0.01)
            raise Exception("Async failure")
        
        # Test successful async call
        result = await breaker.async_call(async_success)
        assert result == "async_success"
        
        # Test failures open circuit
        for i in range(3):
            with pytest.raises(Exception):
                await breaker.async_call(async_failure)
        
        assert breaker.state == CircuitState.OPEN
        
        # Test blocking when open
        with pytest.raises(CircuitOpenError):
            await breaker.async_call(async_success)


class TestCircuitBreakerManager:
    """Tests for circuit breaker manager."""
    
    @pytest.fixture
    def manager(self):
        """Create a circuit breaker manager."""
        return CircuitBreakerManager()
    
    def test_manager_initialization(self, manager):
        """Test manager initialization with defaults."""
        assert 'openai' in manager.breakers
        assert 'anthropic' in manager.breakers
        assert 'ollama' in manager.breakers
    
    def test_add_breaker(self, manager):
        """Test adding a new circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=5)
        breaker = manager.add_breaker('test_service', config)
        
        assert breaker is not None
        assert 'test_service' in manager.breakers
        assert manager.get_breaker('test_service') == breaker
    
    def test_call_through_manager(self, manager):
        """Test calling functions through manager."""
        config = CircuitBreakerConfig(failure_threshold=2)
        manager.add_breaker('test', config)
        
        def test_func(x):
            return x * 2
        
        result = manager.call('test', test_func, 5)
        assert result == 10
    
    def test_call_without_breaker(self, manager):
        """Test calling when no breaker configured."""
        def test_func():
            return "direct"
        
        # Should call directly without circuit breaker
        result = manager.call('nonexistent', test_func)
        assert result == "direct"
    
    def test_get_all_stats(self, manager):
        """Test getting statistics for all breakers."""
        stats = manager.get_all_stats()
        
        assert 'openai' in stats
        assert 'anthropic' in stats
        assert 'state' in stats['openai']
        assert 'total_calls' in stats['openai']
    
    def test_get_open_circuits(self, manager):
        """Test getting list of open circuits."""
        config = CircuitBreakerConfig(failure_threshold=1)
        manager.add_breaker('test1', config)
        manager.add_breaker('test2', config)
        
        def failing_func():
            raise Exception("Test")
        
        # Open test1 circuit
        with pytest.raises(Exception):
            manager.call('test1', failing_func)
        
        open_circuits = manager.get_open_circuits()
        assert 'test1' in open_circuits
        assert 'test2' not in open_circuits
    
    def test_reset_all_breakers(self, manager):
        """Test resetting all circuit breakers."""
        config = CircuitBreakerConfig(failure_threshold=1)
        manager.add_breaker('test1', config)
        manager.add_breaker('test2', config)
        
        def failing_func():
            raise Exception("Test")
        
        # Open both circuits
        with pytest.raises(Exception):
            manager.call('test1', failing_func)
        with pytest.raises(Exception):
            manager.call('test2', failing_func)
        
        assert len(manager.get_open_circuits()) == 2
        
        # Reset all
        manager.reset()
        assert len(manager.get_open_circuits()) == 0
    
    def test_reset_specific_breaker(self, manager):
        """Test resetting specific circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=1)
        manager.add_breaker('test', config)
        
        def failing_func():
            raise Exception("Test")
        
        # Open circuit
        with pytest.raises(Exception):
            manager.call('test', failing_func)
        
        assert 'test' in manager.get_open_circuits()
        
        # Reset specific breaker
        manager.reset('test')
        assert 'test' not in manager.get_open_circuits()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])