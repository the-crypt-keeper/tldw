# Embeddings Module Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [API Reference](#api-reference)
4. [Configuration](#configuration)
5. [Security Features](#security-features)
6. [Provider Integration](#provider-integration)
7. [Performance & Scaling](#performance--scaling)
8. [Monitoring & Observability](#monitoring--observability)
9. [Troubleshooting](#troubleshooting)
10. [Development Guide](#development-guide)

---

## Overview

The Embeddings Module provides a unified interface for generating text embeddings across multiple providers (OpenAI, HuggingFace, Cohere, etc.) with built-in caching, rate limiting, and resource management.

### Key Features
- **Multi-provider support** with automatic fallback
- **TTL-based caching** for performance optimization
- **Resource management** with LRU eviction
- **Security hardening** with input validation and audit logging
- **Rate limiting** per user/tier
- **Circuit breaker** pattern for fault tolerance
- **OpenAI-compatible API** for easy integration

### Current Version
- **Production System**: Single-user synchronous API (`embeddings_v5_production.py`)
- **Future System**: Multi-user scale-out architecture (implemented, not yet integrated)

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                     API Layer                            │
│  /api/v1/embeddings (OpenAI-compatible REST API)        │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│              Service Layer                               │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐ │
│  │Rate Limiter │  │Circuit Break │  │  Auth Check   │ │
│  └─────────────┘  └──────────────┘  └───────────────┘ │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│           Embedding Engine                               │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐   │
│  │  Cache   │  │ Provider │  │ Resource Manager   │   │
│  │  (TTL)   │  │  Router  │  │  (LRU Eviction)    │   │
│  └──────────┘  └──────────┘  └────────────────────┘   │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│              Provider Adapters                           │
│  ┌──────────┐ ┌────────────┐ ┌──────────┐ ┌─────────┐ │
│  │ OpenAI   │ │HuggingFace │ │  Cohere  │ │  Local  │ │
│  └──────────┘ └────────────┘ └──────────┘ └─────────┘ │
└──────────────────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│              Storage Layer                               │
│  ┌──────────────┐           ┌─────────────────────┐    │
│  │  ChromaDB    │           │   Audit Logger      │    │
│  │(Vector Store)│           │  (Security Events)  │    │
│  └──────────────┘           └─────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

### File Structure

```
app/core/Embeddings/
├── __init__.py
├── README.md                          # Architecture overview
├── ChromaDB_Library.py                # Vector storage management
├── circuit_breaker.py                 # Fault tolerance
├── audit_logger.py                    # Security audit logging
├── rate_limiter.py                    # Per-user rate limiting
├── embeddings_config.yaml             # Configuration file
├── job_manager.py                     # Job lifecycle management
├── queue_schemas.py                   # Message schemas
├── worker_config.py                   # Worker configuration
├── worker_orchestrator.py             # Worker pool management
├── Embeddings_Server/
│   ├── __init__.py
│   └── Embeddings_Create.py          # Core embedding logic
└── workers/
    ├── __init__.py
    ├── base_worker.py                 # Abstract base class
    ├── chunking_worker.py             # Text chunking
    ├── embedding_worker.py            # Embedding generation
    └── storage_worker.py              # Storage operations
```

---

## API Reference

### Create Embeddings

**Endpoint:** `POST /api/v1/embeddings`

**Request:**
```json
{
  "input": "string or array of strings",
  "model": "text-embedding-3-small",
  "encoding_format": "float",
  "user": "user_123"
}
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.123, -0.456, ...],
      "metadata": {
        "model": "text-embedding-3-small",
        "dimensions": 1536,
        "provider": "openai"
      }
    }
  ],
  "model": "text-embedding-3-small",
  "usage": {
    "prompt_tokens": 8,
    "total_tokens": 8
  }
}
```

**Error Responses:**
- `400 Bad Request`: Invalid input or parameters
- `401 Unauthorized`: Missing or invalid API key
- `403 Forbidden`: Rate limit exceeded
- `500 Internal Server Error`: Provider failure

### Health Check

**Endpoint:** `GET /api/v1/embeddings/health`

**Response:**
```json
{
  "status": "healthy",
  "providers": {
    "openai": "available",
    "huggingface": "available",
    "cohere": "unavailable"
  },
  "cache": {
    "size": 1234,
    "hit_rate": 0.85
  },
  "models_loaded": 2,
  "uptime_seconds": 3600
}
```

### Admin Endpoints

#### Clear Cache (Admin Only)
**Endpoint:** `DELETE /api/v1/embeddings/cache`

#### Get Metrics (Admin Only)
**Endpoint:** `GET /api/v1/embeddings/metrics`

#### Reset Circuit Breaker (Admin Only)
**Endpoint:** `POST /api/v1/embeddings/circuit-breakers/{provider}/reset`

---

## Configuration

### Main Configuration File

Edit `Config_Files/config.txt`:

```ini
[Embeddings]
# Provider settings
embedding_provider = openai
embedding_model = text-embedding-3-small
embedding_api_url = http://localhost:8080/v1/embeddings
embedding_api_key = your_api_key_here

# Chunking settings
chunk_size = 400
overlap = 200

# Contextual chunking (optional)
enable_contextual_chunking = false
contextual_llm_model = gpt-3.5-turbo
contextual_chunk_method = situate_context

# Resource Management Settings
max_models_in_memory = 3
max_model_memory_gb = 8
model_lru_ttl_seconds = 3600
```

### Provider Configuration

#### OpenAI
```python
{
    "provider": "openai",
    "model": "text-embedding-3-small",
    "api_key": "sk-...",
    "dimensions": 1536,
    "max_batch_size": 2048
}
```

#### HuggingFace
```python
{
    "provider": "huggingface",
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "cache_dir": "./models/huggingface_cache",
    "device": "cuda",  # or "cpu"
    "max_length": 512
}
```

#### Local API
```python
{
    "provider": "local_api",
    "api_url": "http://localhost:8080/v1/embeddings",
    "api_key": "optional_key",
    "timeout": 30
}
```

### Environment Variables

```bash
# Optional overrides
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export COHERE_API_KEY="..."
export EMBEDDING_CACHE_SIZE="10000"
export EMBEDDING_CACHE_TTL="3600"
```

---

## Security Features

### Input Validation

All user inputs are validated to prevent security vulnerabilities:

```python
# User ID validation
- Alphanumeric + underscore/hyphen only
- Maximum 255 characters
- No path traversal patterns

# Model ID validation  
- Prevents injection attacks
- Validates against known models
- Sanitizes special characters
```

### Path Traversal Protection

```python
# Secure path construction
user_path = validate_user_id(user_id)
base_path = Path(base_dir).resolve()
final_path = (base_path / user_path).resolve()

# Verify path is within base directory
final_path.relative_to(base_path)  # Raises if outside
```

### Audit Logging

Security events are logged to `logs/embeddings_audit.jsonl`:

```json
{
  "timestamp": "2024-01-20T10:30:00Z",
  "event_type": "path_traversal_attempt",
  "user_id": "user_123",
  "severity": "WARNING",
  "details": {
    "attempted_value": "../../../etc/passwd"
  }
}
```

### Rate Limiting

Per-user rate limiting with configurable tiers:

```python
# Default limits
FREE_TIER: 60 requests/minute
PREMIUM_TIER: 200 requests/minute
ENTERPRISE_TIER: 1000 requests/minute

# Burst allowance: 1.5x normal limit
```

---

## Provider Integration

### Adding a New Provider

1. **Create Provider Adapter:**
```python
# app/core/Embeddings/providers/new_provider.py
class NewProviderEmbedder:
    def __init__(self, config: dict):
        self.api_key = config.get("api_key")
        self.model = config.get("model")
    
    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        # Implementation
        pass
```

2. **Register Provider:**
```python
# In Embeddings_Create.py
PROVIDERS = {
    "openai": OpenAIEmbedder,
    "new_provider": NewProviderEmbedder,
}
```

3. **Add Configuration:**
```ini
[Embeddings]
embedding_provider = new_provider
new_provider_api_key = xxx
new_provider_model = model-name
```

### Provider Fallback Chain

Configure automatic fallback when primary provider fails:

```python
FALLBACK_CHAIN = [
    "openai",        # Primary
    "huggingface",   # First fallback
    "local",         # Last resort
]
```

---

## Performance & Scaling

### Caching Strategy

**TTL-based LRU Cache:**
- Default TTL: 1 hour
- Max size: 10,000 entries
- Cache key: SHA256(text + model)

```python
# Cache hit example
Input: "Hello world" + "text-embedding-3-small"
Key: "a2f3b8c9..."
Result: [0.123, -0.456, ...] (from cache)
```

### Resource Management

**Model Memory Limits:**
- Maximum models in memory: 3 (configurable)
- Maximum total memory: 8GB (configurable)
- LRU eviction when limits exceeded
- Automatic cleanup after TTL (1 hour)

**Model Loading Strategy:**
```python
1. Check if model in memory → Use it
2. Check if at capacity → Evict LRU model
3. Check memory limit → Evict until under limit
4. Load new model
5. Track usage for future eviction
```

### Batch Processing

Optimize for throughput with batching:

```python
# Automatic batching
texts = ["text1", "text2", ..., "text100"]
# Processed in batches of 32 (configurable)
```

### Connection Pooling

Reuse connections for better performance:

```python
# HTTP connection pooling
max_connections = 10
keepalive_timeout = 30
```

---

## Monitoring & Observability

### Prometheus Metrics

Available at `http://localhost:9090/metrics`:

```
# Active models
embedding_models_active{provider="openai",model="text-embedding-3-small"} 1

# Request rate
embedding_requests_total{provider="openai",status="success"} 12345

# Cache performance
embedding_cache_hit_rate 0.85
embedding_cache_size 5678

# Latency
embedding_latency_seconds{quantile="0.99"} 0.250
```

### Logging

Structured logging with Loguru:

```python
# Log levels
DEBUG: Detailed diagnostic information
INFO: General operational messages
WARNING: Warning messages (degraded performance)
ERROR: Error conditions (recoverable)
CRITICAL: Critical failures (non-recoverable)

# Log format
2024-01-20 10:30:00.123 | INFO | embeddings.create | Created embedding for user_123
```

### Health Monitoring

```python
# Health check intervals
PROVIDER_CHECK: Every 30 seconds
CACHE_CLEANUP: Every 5 minutes
METRIC_COLLECTION: Every 60 seconds
```

---

## Troubleshooting

### Common Issues

#### 1. Model Loading Failures
```
ERROR: Failed to load model "text-embedding-3-small"
SOLUTION: 
- Check API keys in config.txt
- Verify network connectivity
- Check disk space for model downloads
```

#### 2. Rate Limit Exceeded
```
ERROR: 429 Rate limit exceeded for user_123
SOLUTION:
- Implement request batching
- Upgrade user tier
- Add retry with exponential backoff
```

#### 3. Memory Issues
```
ERROR: Cannot load model - memory limit exceeded
SOLUTION:
- Reduce max_models_in_memory
- Increase max_model_memory_gb
- Use smaller models
```

#### 4. Cache Misses
```
WARNING: Cache hit rate below 50%
SOLUTION:
- Increase cache size
- Extend TTL for stable content
- Review cache key generation
```

### Debug Mode

Enable debug logging:

```python
# In config.txt
[Logging]
log_level = DEBUG

# Or via environment
export LOG_LEVEL=DEBUG
```

### Performance Profiling

```python
# Enable profiling
python -m cProfile -o embeddings.prof app.py

# Analyze results
python -m pstats embeddings.prof
```

---

## Development Guide

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/your-org/tldw_server
cd tldw_server

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest tests/Embeddings/ -v
```

### Running Tests

```bash
# Unit tests
pytest tests/Embeddings/test_embeddings_v5_unit.py

# Integration tests
pytest tests/Embeddings/test_embeddings_v5_integration.py

# Performance tests
pytest tests/Embeddings/test_embeddings_v5_production.py::TestLoadTesting

# With coverage
pytest --cov=app.core.Embeddings --cov-report=html
```

### Code Style

Follow PEP 8 with these additions:
- Type hints for all functions
- Docstrings for all public methods
- Maximum line length: 120 characters

```python
def create_embedding(
    text: str,
    model: str = "text-embedding-3-small",
    user_id: Optional[str] = None
) -> List[float]:
    """
    Create an embedding for the given text.
    
    Args:
        text: Input text to embed
        model: Model identifier
        user_id: Optional user ID for tracking
        
    Returns:
        List of embedding values
        
    Raises:
        ValueError: If text is empty or invalid
        RateLimitError: If user exceeds rate limit
    """
    pass
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run tests and linting
6. Submit a pull request

### API Client Examples

#### Python
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/embeddings",
    json={
        "input": "Hello world",
        "model": "text-embedding-3-small"
    },
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)

embedding = response.json()["data"][0]["embedding"]
```

#### JavaScript
```javascript
const response = await fetch('http://localhost:8000/api/v1/embeddings', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer YOUR_API_KEY'
    },
    body: JSON.stringify({
        input: 'Hello world',
        model: 'text-embedding-3-small'
    })
});

const data = await response.json();
const embedding = data.data[0].embedding;
```

#### cURL
```bash
curl -X POST http://localhost:8000/api/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "input": "Hello world",
    "model": "text-embedding-3-small"
  }'
```

---

## Appendix

### Supported Models

| Provider | Model | Dimensions | Max Tokens | Cost/1K tokens |
|----------|-------|------------|------------|----------------|
| OpenAI | text-embedding-3-small | 1536 | 8191 | $0.00002 |
| OpenAI | text-embedding-3-large | 3072 | 8191 | $0.00013 |
| HuggingFace | all-MiniLM-L6-v2 | 384 | 512 | Free |
| HuggingFace | all-mpnet-base-v2 | 768 | 512 | Free |
| Cohere | embed-english-v3.0 | 1024 | 512 | $0.00010 |

### Performance Benchmarks

| Operation | Latency (p50) | Latency (p99) | Throughput |
|-----------|---------------|---------------|------------|
| Single embedding (cached) | 2ms | 5ms | 500 req/s |
| Single embedding (uncached) | 50ms | 200ms | 20 req/s |
| Batch (32 texts) | 100ms | 500ms | 320 texts/s |
| Model loading | 2s | 5s | - |

### Security Checklist

- [ ] API keys stored securely (environment variables or secrets manager)
- [ ] Input validation enabled
- [ ] Rate limiting configured
- [ ] Audit logging enabled
- [ ] HTTPS/TLS for API endpoints
- [ ] Regular security updates
- [ ] Backup and recovery procedures
- [ ] Incident response plan

### Support

- **GitHub Issues**: https://github.com/your-org/tldw_server/issues
- **Documentation**: https://docs.your-org.com/embeddings
- **Community Discord**: https://discord.gg/your-org
- **Email Support**: support@your-org.com

---

*Last Updated: January 2025*
*Version: 1.0.0*