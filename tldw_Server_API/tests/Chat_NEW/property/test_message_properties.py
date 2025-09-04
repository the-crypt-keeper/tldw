"""
Property-based tests for chat message invariants.

Uses Hypothesis to verify that message handling maintains required properties
across all valid inputs. Tests invariants that must hold true regardless of
specific message content or structure.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.strategies import composite
from typing import List, Dict, Any
import json

from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    ChatCompletionRequest,
    ChatCompletionUserMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionAssistantMessageParam,
)

# ========================================================================
# Custom Hypothesis Strategies
# ========================================================================

@composite
def valid_message_content(draw):
    """Generate valid message content."""
    return draw(st.one_of(
        st.text(min_size=1, max_size=1000),  # Regular text
        st.text(min_size=0, max_size=0).map(lambda _: " "),  # Single space
        st.text(alphabet=st.characters(categories=["Lu", "Ll", "Nd", "P"]), min_size=1, max_size=500),  # Alphanumeric + punctuation
    ))

@composite
def valid_role(draw):
    """Generate valid message roles."""
    return draw(st.sampled_from(["system", "user", "assistant"]))

@composite
def valid_message(draw):
    """Generate a valid message dictionary."""
    role = draw(valid_role())
    content = draw(valid_message_content())
    message = {"role": role, "content": content}
    
    # Optionally add name field
    if draw(st.booleans()):
        name = draw(st.text(alphabet=st.characters(categories=["Lu", "Ll", "Nd"]), min_size=1, max_size=50))
        message["name"] = name
    
    return message

@composite
def valid_message_list(draw, min_size=1, max_size=10):
    """Generate a valid list of messages."""
    messages = draw(st.lists(valid_message(), min_size=min_size, max_size=max_size))
    
    # Ensure at least one user message
    has_user = any(msg["role"] == "user" for msg in messages)
    if not has_user and messages:
        messages[0]["role"] = "user"
    
    return messages

@composite
def valid_temperature(draw):
    """Generate valid temperature values."""
    return draw(st.floats(min_value=0.0, max_value=2.0))

@composite
def valid_max_tokens(draw):
    """Generate valid max_tokens values."""
    return draw(st.one_of(
        st.none(),  # Optional parameter
        st.integers(min_value=1, max_value=4096)
    ))

# ========================================================================
# Message Role Invariants
# ========================================================================

class TestMessageRoleInvariants:
    """Test invariants related to message roles."""
    
    @pytest.mark.property
    @given(messages=valid_message_list())
    def test_role_preservation(self, messages):
        """Property: Message roles are preserved through processing."""
        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        # Roles should be preserved exactly
        for i, msg in enumerate(messages):
            assert request.messages[i].role == msg["role"]
    
    @pytest.mark.property
    @given(messages=valid_message_list())
    def test_role_constraints(self, messages):
        """Property: Only valid roles are accepted."""
        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        valid_roles = {"system", "user", "assistant", "tool"}
        for msg in request.messages:
            assert msg.role in valid_roles
    
    @pytest.mark.property
    @given(messages=valid_message_list(min_size=2, max_size=10))
    def test_conversation_flow_validity(self, messages):
        """Property: Conversation flow follows valid patterns."""
        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        # Check that we have at least one user message (required)
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        assert len(user_messages) > 0, "At least one user message is required"
        
        # Check that roles are valid
        valid_roles = {"system", "user", "assistant", "tool"}
        for msg in request.messages:
            assert msg.role in valid_roles

# ========================================================================
# Message Content Invariants
# ========================================================================

class TestMessageContentInvariants:
    """Test invariants related to message content."""
    
    @pytest.mark.property
    @given(content=valid_message_content())
    def test_content_preservation(self, content):
        """Property: Message content is preserved exactly."""
        msg = ChatCompletionUserMessageParam(
            role="user",
            content=content
        )
        assert msg.content == content
    
    @pytest.mark.property
    @given(messages=valid_message_list())
    def test_content_never_null(self, messages):
        """Property: Message content is never null after validation."""
        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        for msg in request.messages:
            assert msg.content is not None
    
    @pytest.mark.property
    @given(
        content=st.text(min_size=0, max_size=10000),
        role=valid_role()
    )
    def test_content_length_handling(self, content, role):
        """Property: Any length content is accepted (within reasonable bounds)."""
        if role == "system":
            msg = ChatCompletionSystemMessageParam(role=role, content=content)
        elif role == "user":
            msg = ChatCompletionUserMessageParam(role=role, content=content)
        else:
            msg = ChatCompletionAssistantMessageParam(role=role, content=content)
        
        assert isinstance(msg.content, (str, list))
    
    @pytest.mark.property
    @given(special_chars=st.text(alphabet=st.characters(categories=["Cc", "Cf", "Co", "Cs"])))
    def test_special_character_handling(self, special_chars):
        """Property: Special/control characters are handled safely."""
        msg = ChatCompletionUserMessageParam(
            role="user",
            content=f"Message with special chars: {special_chars}"
        )
        assert msg.role == "user"
        # Should not raise an exception

# ========================================================================
# Request Structure Invariants
# ========================================================================

class TestRequestStructureInvariants:
    """Test invariants for the overall request structure."""
    
    @pytest.mark.property
    @given(
        messages=valid_message_list(),
        temperature=valid_temperature(),
        max_tokens=valid_max_tokens()
    )
    def test_request_completeness(self, messages, temperature, max_tokens):
        """Property: Valid requests always have required fields."""
        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Required fields must exist
        assert request.model is not None
        assert request.messages is not None
        assert len(request.messages) >= 1
    
    @pytest.mark.property
    @given(messages=valid_message_list(min_size=1, max_size=100))
    def test_message_count_preservation(self, messages):
        """Property: Message count is preserved."""
        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        assert len(request.messages) == len(messages)
    
    @pytest.mark.property
    @given(messages=valid_message_list())
    def test_message_order_preservation(self, messages):
        """Property: Message order is preserved."""
        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        for i in range(len(messages)):
            assert request.messages[i].content == messages[i]["content"]
            assert request.messages[i].role == messages[i]["role"]

# ========================================================================
# Parameter Boundary Invariants
# ========================================================================

class TestParameterBoundaryInvariants:
    """Test invariants at parameter boundaries."""
    
    @pytest.mark.property
    @given(temperature=st.floats(min_value=0.0, max_value=2.0, allow_nan=False))
    def test_temperature_bounds_invariant(self, temperature):
        """Property: Temperature is always within valid bounds."""
        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            temperature=temperature
        )
        
        assert 0.0 <= request.temperature <= 2.0
    
    @pytest.mark.property
    @given(n=st.integers(min_value=1, max_value=128))
    def test_n_parameter_bounds(self, n):
        """Property: n parameter is always within valid bounds."""
        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            n=n
        )
        
        assert 1 <= request.n <= 128
    
    @pytest.mark.property
    @given(
        frequency_penalty=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False),
        presence_penalty=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False)
    )
    def test_penalty_bounds(self, frequency_penalty, presence_penalty):
        """Property: Penalty parameters are within bounds."""
        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )
        
        assert -2.0 <= request.frequency_penalty <= 2.0
        assert -2.0 <= request.presence_penalty <= 2.0

# ========================================================================
# JSON Serialization Invariants
# ========================================================================

class TestSerializationInvariants:
    """Test invariants for JSON serialization."""
    
    @pytest.mark.property
    @given(messages=valid_message_list())
    def test_json_round_trip(self, messages):
        """Property: Messages survive JSON round-trip."""
        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        # Serialize to JSON and back
        json_str = request.model_dump_json()
        parsed = json.loads(json_str)
        reconstructed = ChatCompletionRequest(**parsed)
        
        # Should be equivalent
        assert len(reconstructed.messages) == len(request.messages)
        for orig, recon in zip(request.messages, reconstructed.messages):
            assert orig.role == recon.role
            assert orig.content == recon.content
    
    @pytest.mark.property
    @given(messages=valid_message_list())
    def test_json_serialization_always_valid(self, messages):
        """Property: Serialization always produces valid JSON."""
        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        json_str = request.model_dump_json()
        
        # Should be valid JSON
        try:
            parsed = json.loads(json_str)
            assert isinstance(parsed, dict)
            assert "messages" in parsed
        except json.JSONDecodeError:
            pytest.fail("Failed to produce valid JSON")

# ========================================================================
# Error Handling Invariants
# ========================================================================

class TestErrorHandlingInvariants:
    """Test invariants for error handling."""
    
    @pytest.mark.property
    @given(invalid_role=st.text(min_size=1, max_size=20).filter(lambda x: x not in ["system", "user", "assistant", "tool"]))
    def test_invalid_role_always_rejected(self, invalid_role):
        """Property: Invalid roles are always rejected."""
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            ChatCompletionUserMessageParam(
                role=invalid_role,  # This should fail validation
                content="test"
            )
    
    @pytest.mark.property
    @given(messages=st.lists(st.dictionaries(
        keys=st.text(min_size=1, max_size=10),
        values=st.text(min_size=1, max_size=100),
        min_size=1,
        max_size=5
    ), min_size=1, max_size=5))
    def test_malformed_messages_handled_gracefully(self, messages):
        """Property: Malformed messages are handled gracefully."""
        # Ensure messages don't accidentally have correct structure
        assume(not all("role" in msg and "content" in msg for msg in messages))
        
        from pydantic import ValidationError
        
        try:
            request = ChatCompletionRequest(
                model="gpt-3.5-turbo",
                messages=messages
            )
            # If it succeeds, the messages must have been valid
            assert all(hasattr(msg, "role") and hasattr(msg, "content") for msg in request.messages)
        except ValidationError:
            # Expected for truly malformed messages
            pass