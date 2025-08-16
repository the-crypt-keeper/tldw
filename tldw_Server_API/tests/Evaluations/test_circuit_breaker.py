"""
Tests for circuit breaker implementation.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock

from tldw_Server_API.app.core.Evaluations.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CircuitOpenError,
    LLMCircuitBreaker,
    with_circuit_breaker
)


class TestCircuitBreaker:
    """Test basic circuit breaker functionality."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state allows calls."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker("test", config)
        
        async def success_func():
            return "success"
        
        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.successful_calls == 1
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after threshold failures."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker("test", config)
        
        async def failing_func():
            raise Exception("Test failure")
        
        # Fail threshold times
        for i in range(3):
            with pytest.raises(Exception):
                await breaker.call(failing_func)
        
        assert breaker.state == CircuitState.OPEN
        assert breaker.stats.failed_calls == 3
        
        # Next call should be rejected
        with pytest.raises(CircuitOpenError):
            await breaker.call(failing_func)
        
        assert breaker.stats.rejected_calls == 1
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker recovery through half-open state."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=2,
            recovery_timeout=0.1  # Short for testing
        )
        breaker = CircuitBreaker("test", config)
        
        async def conditional_func(should_fail):
            if should_fail:
                raise Exception("Test failure")
            return "success"
        
        # Open the circuit
        for i in range(2):
            with pytest.raises(Exception):
                await breaker.call(conditional_func, True)
        
        assert breaker.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        await asyncio.sleep(0.15)
        
        # Should transition to half-open and allow call
        result = await breaker.call(conditional_func, False)
        assert result == "success"
        assert breaker.state == CircuitState.HALF_OPEN
        
        # One more success should close the circuit
        result = await breaker.call(conditional_func, False)
        assert breaker.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_timeout(self):
        """Test circuit breaker handles timeouts."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            timeout=0.1
        )
        breaker = CircuitBreaker("test", config)
        
        async def slow_func():
            await asyncio.sleep(1)
            return "success"
        
        # Should timeout
        with pytest.raises(TimeoutError):
            await breaker.call(slow_func)
        
        assert breaker.stats.timeouts == 1
        assert breaker.stats.failed_calls == 1
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_sync_function(self):
        """Test circuit breaker with synchronous functions."""
        config = CircuitBreakerConfig()
        breaker = CircuitBreaker("test", config)
        
        def sync_func(x, y):
            return x + y
        
        result = await breaker.call(sync_func, 2, 3)
        assert result == 5
        assert breaker.stats.successful_calls == 1
    
    def test_circuit_breaker_state_reporting(self):
        """Test circuit breaker state reporting."""
        config = CircuitBreakerConfig(failure_threshold=5)
        breaker = CircuitBreaker("test", config)
        
        state = breaker.get_state()
        assert state["name"] == "test"
        assert state["state"] == "closed"
        assert state["stats"]["total_calls"] == 0
        assert state["config"]["failure_threshold"] == 5
    
    def test_circuit_breaker_reset(self):
        """Test circuit breaker reset."""
        breaker = CircuitBreaker("test")
        breaker.stats.total_calls = 100
        breaker.state = CircuitState.OPEN
        
        breaker.reset()
        
        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.total_calls == 0


class TestLLMCircuitBreaker:
    """Test LLM-specific circuit breaker."""
    
    def test_provider_specific_configs(self):
        """Test that different providers get different configs."""
        llm_breaker = LLMCircuitBreaker()
        
        openai_breaker = llm_breaker.get_breaker("openai")
        anthropic_breaker = llm_breaker.get_breaker("anthropic")
        
        assert openai_breaker.config.timeout == 30.0
        assert anthropic_breaker.config.timeout == 45.0
    
    @pytest.mark.asyncio
    async def test_call_with_breaker(self):
        """Test calling function through LLM breaker."""
        llm_breaker = LLMCircuitBreaker()
        
        async def llm_func(prompt):
            return f"Response to: {prompt}"
        
        result = await llm_breaker.call_with_breaker(
            "openai",
            llm_func,
            "test prompt"
        )
        
        assert result == "Response to: test prompt"
        
        # Check state
        states = llm_breaker.get_all_states()
        assert "openai" in states
        assert states["openai"]["stats"]["successful_calls"] == 1
    
    def test_reset_all_breakers(self):
        """Test resetting all circuit breakers."""
        llm_breaker = LLMCircuitBreaker()
        
        # Create and use some breakers
        openai_breaker = llm_breaker.get_breaker("openai")
        openai_breaker.stats.total_calls = 10
        
        anthropic_breaker = llm_breaker.get_breaker("anthropic")
        anthropic_breaker.stats.total_calls = 20
        
        # Reset all
        llm_breaker.reset_all()
        
        assert openai_breaker.stats.total_calls == 0
        assert anthropic_breaker.stats.total_calls == 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_decorator(self):
        """Test circuit breaker decorator."""
        call_count = 0
        
        @with_circuit_breaker("test_provider")
        async def decorated_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await decorated_func()
        assert result == "success"
        assert call_count == 1


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker with real scenarios."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_protects_downstream(self):
        """Test that circuit breaker protects downstream services."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=0.5
        )
        breaker = CircuitBreaker("downstream", config)
        
        call_count = 0
        
        async def flaky_service():
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise Exception("Service error")
            return "recovered"
        
        # First 3 calls fail and open circuit
        for i in range(3):
            with pytest.raises(Exception):
                await breaker.call(flaky_service)
        
        assert breaker.state == CircuitState.OPEN
        
        # Next calls are rejected without calling service
        for i in range(5):
            with pytest.raises(CircuitOpenError):
                await breaker.call(flaky_service)
        
        # Service wasn't called during open state
        assert call_count == 3
        
        # Wait for recovery
        await asyncio.sleep(0.6)
        
        # Service should be called again and succeed
        result = await breaker.call(flaky_service)
        assert result == "recovered"
        assert call_count == 4
    
    @pytest.mark.asyncio
    async def test_concurrent_calls_with_circuit_breaker(self):
        """Test circuit breaker handles concurrent calls correctly."""
        config = CircuitBreakerConfig(failure_threshold=5)
        breaker = CircuitBreaker("concurrent", config)
        
        async def async_func(value):
            await asyncio.sleep(0.01)
            return value * 2
        
        # Launch concurrent calls
        tasks = [
            breaker.call(async_func, i)
            for i in range(10)
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert results == [i * 2 for i in range(10)]
        assert breaker.stats.successful_calls == 10
        assert breaker.state == CircuitState.CLOSED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])