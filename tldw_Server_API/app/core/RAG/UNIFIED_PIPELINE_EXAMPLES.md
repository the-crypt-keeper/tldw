# Unified RAG Pipeline - Usage Examples

## Overview

The unified RAG pipeline provides a single function where ALL features are controlled by explicit parameters. No configuration files, no presets - just set the parameters you need.

## API Endpoint

```
POST /api/v1/rag/unified/search
```

## Basic Examples

### 1. Simple Search (Minimal Parameters)
```python
# Just search with defaults
{
    "query": "What is machine learning?"
}
```

### 2. Search with Query Expansion
```python
{
    "query": "What is ML?",
    "expand_query": true,
    "expansion_strategies": ["acronym", "synonym"]
}
```

### 3. Multi-Database Search
```python
{
    "query": "neural networks",
    "sources": ["media_db", "notes", "characters"]
}
```

## Feature-Specific Examples

### Query Expansion & Spell Check
```python
{
    "query": "machne lerning algoritms",  # Misspelled
    "spell_check": true,
    "expand_query": true,
    "expansion_strategies": ["acronym", "synonym", "domain"]
}
```

### Citation Generation
```python
{
    "query": "climate change effects",
    "enable_citations": true,
    "citation_style": "apa",
    "include_page_numbers": true
}
```

### Answer Generation
```python
{
    "query": "How do neural networks work?",
    "enable_generation": true,
    "generation_model": "gpt-3.5-turbo",
    "max_generation_tokens": 500
}
```

### Security & PII Detection
```python
{
    "query": "user data privacy",
    "enable_security_filter": true,
    "detect_pii": true,
    "redact_pii": true,
    "sensitivity_level": "internal"
}
```

### Advanced Reranking
```python
{
    "query": "quantum computing",
    "enable_reranking": true,
    "reranking_strategy": "hybrid",
    "rerank_top_k": 20,
    "top_k": 50  # Get 50, then rerank to top 20
}
```

### Table Processing
```python
{
    "query": "financial data tables",
    "enable_table_processing": true,
    "table_method": "markdown"
}
```

### Enhanced Chunking with Parent Context
```python
{
    "query": "code implementation details",
    "enable_enhanced_chunking": true,
    "chunk_type_filter": ["code", "text"],
    "enable_parent_expansion": true,
    "parent_context_size": 1000,
    "include_sibling_chunks": true
}
```

### User Feedback Collection
```python
{
    "query": "database optimization",
    "collect_feedback": true,
    "feedback_user_id": "user123",
    "apply_feedback_boost": true  # Use previous feedback to improve results
}
```

## Combined Feature Examples

### Research Mode (Everything Enabled)
```python
{
    "query": "artificial intelligence applications in healthcare",
    
    # Query Enhancement
    "spell_check": true,
    "expand_query": true,
    "expansion_strategies": ["acronym", "synonym", "domain", "entity"],
    
    # Search Configuration
    "sources": ["media_db", "notes"],
    "search_mode": "hybrid",
    "hybrid_alpha": 0.7,
    "top_k": 30,
    
    # Processing
    "enable_table_processing": true,
    "enable_enhanced_chunking": true,
    "enable_parent_expansion": true,
    
    # Quality
    "enable_reranking": true,
    "reranking_strategy": "hybrid",
    
    # Output
    "enable_citations": true,
    "citation_style": "apa",
    "enable_generation": true,
    "generation_model": "gpt-4",
    
    # Monitoring
    "enable_monitoring": true,
    "enable_performance_analysis": true,
    
    # User Experience
    "highlight_results": true,
    "highlight_query_terms": true,
    "collect_feedback": true
}
```

### Fast Cached Search
```python
{
    "query": "common programming patterns",
    "enable_cache": true,
    "cache_threshold": 0.85,
    "adaptive_cache": true,
    "enable_reranking": false,  # Skip for speed
    "enable_generation": false   # Skip for speed
}
```

### Secure Enterprise Search
```python
{
    "query": "company policies",
    
    # Security
    "enable_security_filter": true,
    "detect_pii": true,
    "redact_pii": true,
    "sensitivity_level": "confidential",
    "content_filter": true,
    
    # Resilience
    "enable_resilience": true,
    "retry_attempts": 3,
    "circuit_breaker": true,
    
    # Monitoring
    "enable_monitoring": true,
    "enable_observability": true,
    "trace_id": "trace-123",
    
    # User Context
    "user_id": "employee123",
    "session_id": "session-456"
}
```

### Academic Research
```python
{
    "query": "quantum entanglement experiments",
    
    # Comprehensive search
    "expand_query": true,
    "expansion_strategies": ["domain", "entity"],
    "sources": ["media_db", "notes"],
    
    # Quality focus
    "enable_reranking": true,
    "reranking_strategy": "cross_encoder",
    "top_k": 50,
    "rerank_top_k": 10,
    
    # Academic output
    "enable_citations": true,
    "citation_style": "chicago",
    "include_page_numbers": true,
    
    # No generation - want original sources
    "enable_generation": false
}
```

## Batch Processing

### Process Multiple Queries
```python
POST /api/v1/rag/unified/batch

{
    "queries": [
        "What is AI?",
        "Explain machine learning",
        "Define neural networks"
    ],
    "max_concurrent": 5,
    
    # These settings apply to ALL queries
    "expand_query": true,
    "enable_citations": true,
    "enable_reranking": true
}
```

## Performance Tips

### For Speed
- Set `enable_cache: true` with high `cache_threshold`
- Disable `enable_generation` and `enable_citations`
- Use `reranking_strategy: "flashrank"` or disable reranking
- Limit `top_k` to smaller values (5-10)

### For Quality
- Enable all expansion strategies
- Use `reranking_strategy: "hybrid"` or `"cross_encoder"`
- Set higher `top_k` with lower `rerank_top_k`
- Enable `enable_enhanced_chunking` and `enable_parent_expansion`

### For Security
- Always set `enable_security_filter: true`
- Use appropriate `sensitivity_level`
- Enable `detect_pii` and `redact_pii` for sensitive data
- Set `enable_resilience: true` for production

## Response Structure

```json
{
    "documents": [...],           // Retrieved and processed documents
    "query": "...",               // Original query
    "expanded_queries": [...],    // If expansion was enabled
    "metadata": {...},            // Additional information
    "timings": {...},            // Performance metrics
    "citations": [...],          // If citations were enabled
    "generated_answer": "...",   // If generation was enabled
    "feedback_id": "...",        // If feedback collection was enabled
    "cache_hit": false,          // Whether cache was used
    "errors": [],                // Any non-fatal errors
    "security_report": {...},    // If security filter was enabled
    "total_time": 0.35           // Total execution time
}
```

## Common Use Cases

### 1. Customer Support
```python
{
    "query": "How do I reset my password?",
    "sources": ["media_db"],
    "enable_generation": true,  # Generate helpful answer
    "collect_feedback": true,   # Track helpfulness
    "highlight_results": true   # Highlight relevant parts
}
```

### 2. Code Search
```python
{
    "query": "async function implementation",
    "enable_enhanced_chunking": true,
    "chunk_type_filter": ["code"],
    "enable_parent_expansion": true,
    "enable_reranking": true
}
```

### 3. Legal Document Search
```python
{
    "query": "liability clauses",
    "enable_security_filter": true,
    "sensitivity_level": "confidential",
    "enable_citations": true,
    "citation_style": "harvard",
    "enable_table_processing": true
}
```

### 4. Medical Research
```python
{
    "query": "COVID-19 treatment protocols",
    "expand_query": true,
    "expansion_strategies": ["domain", "entity"],
    "enable_citations": true,
    "detect_pii": true,
    "redact_pii": true
}
```

## Debugging

### Enable Debug Mode
```python
{
    "query": "test query",
    "debug_mode": true,              # Verbose logging
    "enable_monitoring": true,        # Track performance
    "enable_performance_analysis": true,  # Detailed analysis
    "track_cost": true               # Estimate API costs
}
```

## Notes

- All parameters are optional except `query`
- Features are only executed if explicitly enabled
- Parameters can be mixed and matched as needed
- No configuration files or presets required
- Every feature is directly accessible