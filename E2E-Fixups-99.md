# E2E Test Expansion Plan - Implementation Guide

## Overview
This document outlines the comprehensive expansion of end-to-end tests for the tldw_server API, focusing on character management, character-based chat, and RAG functionality improvements.

## Current State Analysis
- Weak assertions checking only field existence, not values
- Limited character testing (only import and list)
- No character editing tests
- No character-based chat tests
- Basic RAG search test without depth
- Missing multi-database RAG testing
- No performance validation

## Implementation Phases

### Phase 1: Strengthen Existing Validations
**Priority: HIGH**
**Files to modify: `test_full_user_workflow.py`, `fixtures.py`**

#### 1.1 Replace Weak Assertions
- Change all `assert "field" in response` to `assert response.get("field") == expected_value`
- Add type validation: `assert isinstance(response["field"], expected_type)`
- Validate ranges: `assert min_val <= response["numeric_field"] <= max_val`
- Check non-empty strings: `assert response["text_field"] and len(response["text_field"]) > 0`

#### 1.2 Improve Health Check Test
```python
def test_01_health_check(self, api_client):
    response = api_client.health_check()
    # Strong assertions
    assert response.get("status") == "healthy"
    assert response.get("auth_mode") in ["single_user", "multi_user"]
    assert isinstance(response.get("timestamp"), str)
    # Validate timestamp format
    from datetime import datetime
    datetime.fromisoformat(response["timestamp"].replace('Z', '+00:00'))
```

### Phase 2: Character Management Expansion
**Priority: HIGH**
**New tests in: `test_full_user_workflow.py`**

#### 2.1 Character Editing Test
```python
def test_62_edit_existing_character(self, api_client):
    """Test updating an existing character with proper validation."""
    if not TestFullUserWorkflow.characters:
        pytest.skip("No characters available to edit")
    
    character = TestFullUserWorkflow.characters[0]
    character_id = character.get("id") or character.get("character_id")
    current_version = character.get("version", 1)
    
    # Prepare update data
    updated_data = {
        "name": character.get("name"),  # Keep same name
        "description": "Updated description during E2E testing",
        "personality": "Updated personality: more enthusiastic and helpful",
        "scenario": "Updated scenario for testing",
        "system_prompt": "You are an updated test character with new traits",
        "tags": ["updated", "e2e-test", "modified"],
        "version": current_version
    }
    
    # Perform update
    response = api_client.update_character(
        character_id=character_id,
        expected_version=current_version,
        **updated_data
    )
    
    # Strong validations
    assert response.get("success") == True or response.get("id") == character_id
    assert response.get("version") == current_version + 1
    
    # Verify changes persisted
    retrieved = api_client.get_character(character_id)
    assert retrieved.get("description") == updated_data["description"]
    assert retrieved.get("personality") == updated_data["personality"]
    assert retrieved.get("version") == current_version + 1
    assert set(retrieved.get("tags", [])) == set(updated_data["tags"])
```

#### 2.2 Character Version Conflict Test
```python
def test_63_character_version_conflict(self, api_client):
    """Test optimistic locking with version mismatch."""
    if not TestFullUserWorkflow.characters:
        pytest.skip("No characters available")
    
    character = TestFullUserWorkflow.characters[0]
    character_id = character.get("id")
    
    # Try update with wrong version
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        api_client.update_character(
            character_id=character_id,
            expected_version=999,  # Wrong version
            description="This should fail"
        )
    
    assert exc_info.value.response.status_code == 409  # Conflict
    error_detail = exc_info.value.response.json().get("detail", "")
    assert "version" in error_detail.lower()
```

### Phase 3: Character Chat Integration
**Priority: HIGH**
**New tests in: `test_full_user_workflow.py`**

#### 3.1 Chat with Character Card
```python
def test_65_chat_with_character_card(self, api_client, data_tracker):
    """Test chat using a character card for personality."""
    if not TestFullUserWorkflow.characters:
        pytest.skip("No characters available")
    
    character = TestFullUserWorkflow.characters[0]
    character_id = character.get("id") or character.get("character_id")
    character_name = character.get("name", "TestChar")
    
    messages = [
        {"role": "user", "content": "Hello! Who are you?"}
    ]
    
    response = api_client.chat_completion(
        messages=messages,
        model="gpt-3.5-turbo",
        character_id=character_id,
        temperature=0.7
    )
    
    # Validate response structure
    assert "choices" in response
    assert len(response["choices"]) > 0
    
    # Extract assistant message
    assistant_msg = response["choices"][0].get("message", {})
    assert assistant_msg.get("role") == "assistant"
    assert assistant_msg.get("content") and len(assistant_msg["content"]) > 0
    
    # Character name should be in response metadata
    if "name" in assistant_msg:
        assert assistant_msg["name"] == character_name
    
    # Store for history test
    if "conversation_id" in response:
        TestFullUserWorkflow.chats.append({
            "conversation_id": response["conversation_id"],
            "character_id": character_id,
            "messages": messages + [assistant_msg]
        })
        data_tracker.add_chat(response["conversation_id"])
```

#### 3.2 Character Chat with History
```python
def test_66_character_chat_history(self, api_client):
    """Test maintaining conversation context with character."""
    if not TestFullUserWorkflow.chats:
        pytest.skip("No character chats available")
    
    # Find a character chat
    char_chat = next((c for c in TestFullUserWorkflow.chats if c.get("character_id")), None)
    if not char_chat:
        pytest.skip("No character chat found")
    
    conversation_id = char_chat["conversation_id"]
    character_id = char_chat["character_id"]
    
    # Continue conversation
    follow_up = [
        {"role": "user", "content": "What did we just talk about?"}
    ]
    
    response = api_client.chat_completion(
        messages=follow_up,
        model="gpt-3.5-turbo",
        character_id=character_id,
        conversation_id=conversation_id,
        temperature=0.7
    )
    
    # Should reference previous context
    assert "choices" in response
    content = response["choices"][0]["message"]["content"]
    assert len(content) > 20  # Should have substantial response
    
    # Verify conversation continuity
    assert response.get("conversation_id") == conversation_id
```

### Phase 4: RAG System Testing
**Priority: HIGH**
**New tests in: `test_full_user_workflow.py`**

#### 4.1 Simple RAG Search
```python
def test_71_simple_rag_search(self, api_client):
    """Test simple RAG search with value validation."""
    if not TestFullUserWorkflow.media_items:
        pytest.skip("No media items for RAG search")
    
    # Search for known content
    response = api_client.rag_simple_search(
        query="machine learning artificial intelligence",
        databases=["media"],
        max_context_size=4000,
        top_k=5,
        enable_reranking=True,
        enable_citations=True
    )
    
    # Strong validation
    assert response.get("success") == True
    assert "results" in response
    results = response["results"]
    assert isinstance(results, list)
    assert len(results) <= 5  # Respect top_k
    
    if results:
        # Validate first result structure
        first = results[0]
        assert "content" in first
        assert "score" in first
        assert isinstance(first["score"], (int, float))
        assert 0.0 <= first["score"] <= 1.0
        assert "source" in first
        assert first["source"].get("type") in ["media", "note", "character", "chat"]
        assert "id" in first["source"]
        
        # Check citations if enabled
        if enable_citations:
            assert "citation" in first
            assert first["citation"].get("title")
            assert first["citation"].get("source_id")
    
    # Verify total context size
    total_size = sum(len(r.get("content", "")) for r in results)
    assert total_size <= 4000
```

#### 4.2 Multi-Database RAG Search
```python
def test_72_multi_database_rag_search(self, api_client):
    """Test searching across multiple databases."""
    # Ensure we have content in multiple databases
    has_media = len(TestFullUserWorkflow.media_items) > 0
    has_notes = len(TestFullUserWorkflow.notes) > 0
    has_chars = len(TestFullUserWorkflow.characters) > 0
    
    if not (has_media or has_notes or has_chars):
        pytest.skip("Need content in at least one database")
    
    databases = []
    if has_media:
        databases.append("media")
    if has_notes:
        databases.append("notes")
    if has_chars:
        databases.append("characters")
    
    response = api_client.rag_simple_search(
        query="test content information",
        databases=databases,
        max_context_size=8000,
        top_k=10,
        enable_reranking=True
    )
    
    assert response.get("success") == True
    results = response.get("results", [])
    
    # Verify we got results from multiple sources if available
    source_types = set()
    for result in results:
        source_type = result.get("source", {}).get("type")
        if source_type:
            source_types.add(source_type)
    
    # Should have results from each database we searched
    for db in databases:
        db_type_map = {"media": "media", "notes": "note", "characters": "character"}
        expected_type = db_type_map.get(db, db)
        # It's okay if a database has no matching results
        # but log it for debugging
        if expected_type not in source_types:
            print(f"Note: No results from {db} database")
```

#### 4.3 RAG with Advanced Options
```python
def test_73_rag_advanced_configuration(self, api_client):
    """Test RAG with various configuration options."""
    if not TestFullUserWorkflow.media_items:
        pytest.skip("No content for RAG testing")
    
    # Test different configurations
    configs = [
        {"top_k": 3, "enable_reranking": False},
        {"top_k": 20, "enable_reranking": True},
        {"max_context_size": 1000},
        {"keywords": ["AI", "machine learning"]}
    ]
    
    for config in configs:
        response = api_client.rag_simple_search(
            query="technology and innovation",
            databases=["media"],
            **config
        )
        
        assert response.get("success") == True
        results = response.get("results", [])
        
        # Validate config was respected
        if "top_k" in config:
            assert len(results) <= config["top_k"]
        
        if "max_context_size" in config:
            total = sum(len(r.get("content", "")) for r in results)
            assert total <= config["max_context_size"]
        
        if "keywords" in config:
            # At least one result should contain a keyword
            has_keyword = False
            for result in results:
                content = result.get("content", "").lower()
                if any(kw.lower() in content for kw in config["keywords"]):
                    has_keyword = True
                    break
            # Note: keywords are filters, so might have no results
            if results and not has_keyword:
                print(f"Warning: Keyword filter may not be working correctly")
```

#### 4.4 RAG Performance Validation
```python
def test_74_rag_performance_metrics(self, api_client):
    """Test RAG search performance and metrics."""
    if not TestFullUserWorkflow.media_items:
        pytest.skip("No content for performance testing")
    
    import time
    
    # Measure search latency
    queries = [
        "artificial intelligence",
        "machine learning algorithms",
        "natural language processing",
        "deep learning neural networks"
    ]
    
    latencies = []
    for query in queries:
        start = time.time()
        response = api_client.rag_simple_search(
            query=query,
            databases=["media"],
            top_k=10,
            enable_reranking=True
        )
        latency = time.time() - start
        latencies.append(latency)
        
        assert response.get("success") == True
        
        # Check if performance metrics included
        if "metrics" in response:
            metrics = response["metrics"]
            assert "search_time" in metrics
            assert "rerank_time" in metrics
            assert metrics["search_time"] >= 0
            assert metrics["rerank_time"] >= 0
    
    # Performance assertions
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    
    assert avg_latency < 5.0, f"Average latency too high: {avg_latency:.2f}s"
    assert max_latency < 10.0, f"Max latency too high: {max_latency:.2f}s"
    
    print(f"RAG Performance - Avg: {avg_latency:.2f}s, Max: {max_latency:.2f}s")
```

### Phase 5: Enhanced Fixtures
**File: `fixtures.py` additions**

```python
class StrongAssertionHelpers:
    """Enhanced assertion helpers with value checking."""
    
    @staticmethod
    def assert_exact_value(actual, expected, field_name="field"):
        """Assert exact value match with helpful error message."""
        assert actual == expected, \
            f"{field_name}: expected '{expected}', got '{actual}'"
    
    @staticmethod
    def assert_value_in_range(value, min_val, max_val, field_name="value"):
        """Assert numeric value is within range."""
        assert isinstance(value, (int, float)), \
            f"{field_name} must be numeric, got {type(value)}"
        assert min_val <= value <= max_val, \
            f"{field_name} {value} not in range [{min_val}, {max_val}]"
    
    @staticmethod
    def assert_non_empty_string(value, field_name="field", min_length=1):
        """Assert string is non-empty with minimum length."""
        assert isinstance(value, str), \
            f"{field_name} must be string, got {type(value)}"
        assert len(value) >= min_length, \
            f"{field_name} too short: {len(value)} < {min_length}"
    
    @staticmethod
    def assert_valid_timestamp(timestamp_str, field_name="timestamp"):
        """Assert valid ISO format timestamp."""
        from datetime import datetime
        try:
            datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError) as e:
            pytest.fail(f"Invalid {field_name}: {timestamp_str} - {e}")
    
    @staticmethod
    def assert_character_response(character_data):
        """Validate character response structure and values."""
        required = ["id", "name", "version"]
        for field in required:
            assert field in character_data, f"Missing required field: {field}"
        
        # Type validation
        assert isinstance(character_data["id"], int)
        assert isinstance(character_data["name"], str)
        assert isinstance(character_data["version"], int)
        
        # Value validation
        assert character_data["id"] > 0
        assert len(character_data["name"]) > 0
        assert character_data["version"] >= 1
        
        # Optional fields validation when present
        if "tags" in character_data:
            assert isinstance(character_data["tags"], list)
        if "description" in character_data:
            assert isinstance(character_data["description"], str)
    
    @staticmethod
    def assert_rag_result_quality(result, query_terms=None):
        """Validate RAG search result quality."""
        assert "content" in result, "Result missing content"
        assert "score" in result, "Result missing score"
        assert "source" in result, "Result missing source"
        
        # Content validation
        content = result["content"]
        assert isinstance(content, str)
        assert len(content) > 10, "Content too short"
        
        # Score validation
        score = result["score"]
        assert isinstance(score, (int, float))
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
        
        # Source validation
        source = result["source"]
        assert "type" in source
        assert "id" in source
        assert source["type"] in ["media", "note", "character", "chat"]
        
        # Relevance check if query terms provided
        if query_terms:
            content_lower = content.lower()
            has_term = any(term.lower() in content_lower for term in query_terms)
            if not has_term and score > 0.5:
                print(f"Warning: High score {score} but no query terms found")
```

### Phase 6: API Client Extensions
**File: `fixtures.py` - APIClient class additions**

```python
def update_character(self, character_id: int, expected_version: int, **kwargs) -> Dict[str, Any]:
    """Update a character with optimistic locking."""
    response = self.client.put(
        f"{API_PREFIX}/characters/{character_id}",
        json=kwargs,
        params={"expected_version": expected_version}
    )
    response.raise_for_status()
    return response.json()

def get_character(self, character_id: int) -> Dict[str, Any]:
    """Get a specific character by ID."""
    response = self.client.get(f"{API_PREFIX}/characters/{character_id}")
    response.raise_for_status()
    return response.json()

def rag_simple_search(self, query: str, databases: List[str] = None, **kwargs) -> Dict[str, Any]:
    """Perform simple RAG search."""
    data = {
        "query": query,
        "databases": databases or ["media"],
        **kwargs
    }
    response = self.client.post(
        f"{API_PREFIX}/rag/simple/search",
        json=data
    )
    response.raise_for_status()
    return response.json()

def rag_advanced_search(self, config: Dict[str, Any]) -> Dict[str, Any]:
    """Perform advanced RAG search with full configuration."""
    response = self.client.post(
        f"{API_PREFIX}/rag/search",
        json=config
    )
    response.raise_for_status()
    return response.json()

def chat_completion(self, messages: List[Dict[str, str]], 
                   model: str = "gpt-3.5-turbo",
                   temperature: float = 0.7,
                   character_id: Optional[int] = None,
                   conversation_id: Optional[str] = None,
                   stream: bool = False) -> Dict[str, Any]:
    """Send chat completion request with optional character context."""
    data = {
        "messages": messages,
        "model": model,
        "temperature": temperature,
        "stream": stream
    }
    if character_id is not None:
        data["character_id"] = character_id
    if conversation_id is not None:
        data["conversation_id"] = conversation_id
    
    response = self.client.post(
        f"{API_PREFIX}/chat/completions",
        json=data
    )
    response.raise_for_status()
    return response.json()
```

## Test Execution Order
1. Run existing tests with strengthened assertions
2. Add character editing tests
3. Add character chat tests
4. Add RAG search tests
5. Run full suite for regression
6. Collect performance metrics
7. Generate coverage report

## Success Metrics
- Zero weak assertions (no simple existence checks)
- 100% of value checks verify exact expected values
- Character CRUD operations fully tested
- Character chat personality validation working
- RAG search validates relevance and structure
- Multi-database search properly tested
- Performance benchmarks established
- All tests pass consistently across runs

## Notes
- Use `pytest.mark.slow` for performance tests
- Consider parallel execution for independent tests
- Mock LLM responses for deterministic testing where needed
- Ensure proper cleanup in all test teardowns
- Add retry logic for flaky network operations