# Embeddings Service Deployment Guide

**Version**: 1.0  
**Last Updated**: 2025-08-16  
**Service Version**: v5 Enhanced (with Circuit Breaker)

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Deployment Decision Tree](#deployment-decision-tree)
3. [Single-User Deployment](#single-user-deployment)
4. [Enterprise Deployment](#enterprise-deployment)
5. [Configuration Reference](#configuration-reference)
6. [Monitoring & Observability](#monitoring--observability)
7. [Troubleshooting](#troubleshooting)
8. [Migration Guide](#migration-guide)

---

## Architecture Overview

The embeddings service implements a **dual-architecture design** to support different deployment scales:

### Single-User Architecture (<5 concurrent users)
- **Implementation**: `embeddings_v5_production_enhanced.py`
- **Processing**: Synchronous with async wrapper
- **Infrastructure**: Minimal (no Redis/queues required)
- **Best for**: Personal use, small teams, development

### Enterprise Architecture (>5 concurrent users)
- **Implementation**: Worker-based architecture (job_manager, workers)
- **Processing**: Queue-based, distributed
- **Infrastructure**: Redis, multiple workers, job tracking
- **Best for**: Production, multi-tenant, high volume

---

## Deployment Decision Tree

```
┌─────────────────────────┐
│ How many concurrent     │
│ users will you have?    │
└───────────┬─────────────┘
            │
    ┌───────┴───────┐
    │               │
   <5             ≥5
    │               │
    v               v
┌──────────┐  ┌──────────┐
│ Single-  │  │Enterprise│
│  User    │  │  Mode    │
└──────────┘  └──────────┘
```

### Choose Single-User Mode if:
- [ ] Less than 5 concurrent users
- [ ] Simple deployment preferred
- [ ] No Redis infrastructure available
- [ ] Development/testing environment
- [ ] Personal or small team use

### Choose Enterprise Mode if:
- [ ] 5+ concurrent users expected
- [ ] Need job tracking and queuing
- [ ] Multi-tenant requirements
- [ ] High availability required
- [ ] Need horizontal scaling

---

## Single-User Deployment

### Prerequisites
```bash
# Required packages
pip install -r requirements.txt

# Optional but recommended
pip install torch  # For GPU acceleration
pip install onnxruntime-gpu  # For ONNX models on GPU
```

### Environment Variables
```bash
# Basic configuration
export EMBEDDINGS_MODE="single_user"
export API_KEY="your-secret-key"
export JWT_SECRET_KEY="your-jwt-secret"

# Provider API keys (as needed)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export COHERE_API_KEY="..."

# Optional: Performance tuning
export MAX_BATCH_SIZE=100
export CACHE_TTL_SECONDS=3600
export CONNECTION_POOL_SIZE=20
```

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY tldw_Server_API/ ./tldw_Server_API/

# Configuration
ENV EMBEDDINGS_MODE=single_user
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/embeddings/health || exit 1

# Run service
CMD ["uvicorn", "tldw_Server_API.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose
```yaml
version: '3.8'

services:
  embeddings:
    build: .
    ports:
      - "8000:8000"
    environment:
      - EMBEDDINGS_MODE=single_user
      - API_KEY=${API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - MAX_BATCH_SIZE=100
      - CACHE_TTL_SECONDS=3600
    volumes:
      - ./user_databases:/app/user_databases
      - ./embedding_models_data:/app/embedding_models_data
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
```

### Systemd Service
```ini
[Unit]
Description=TLDW Embeddings Service (Single-User)
After=network.target

[Service]
Type=simple
User=tldw
Group=tldw
WorkingDirectory=/opt/tldw_server
Environment="EMBEDDINGS_MODE=single_user"
Environment="PYTHONPATH=/opt/tldw_server"
ExecStart=/usr/bin/python3 -m uvicorn tldw_Server_API.app.main:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

---

## Enterprise Deployment

### Prerequisites
```bash
# Required packages
pip install -r requirements-enterprise.txt

# Redis server
sudo apt-get install redis-server
# OR
docker run -d --name redis -p 6379:6379 redis:alpine

# Optional: GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Environment Variables
```bash
# Mode selection
export EMBEDDINGS_MODE="enterprise"

# Redis configuration
export REDIS_URL="redis://localhost:6379"
export REDIS_QUEUE_PREFIX="embeddings"

# Worker configuration
export CHUNKING_WORKERS=2
export EMBEDDING_WORKERS=4
export STORAGE_WORKERS=2

# Job limits
export MAX_CONCURRENT_JOBS_FREE=2
export MAX_CONCURRENT_JOBS_PREMIUM=5
export MAX_CONCURRENT_JOBS_ENTERPRISE=20

# Daily quotas (chunks)
export DAILY_QUOTA_FREE=1000
export DAILY_QUOTA_PREMIUM=10000
export DAILY_QUOTA_ENTERPRISE=100000
```

### Docker Compose (Enterprise)
```yaml
version: '3.8'

services:
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - EMBEDDINGS_MODE=enterprise
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    volumes:
      - ./user_databases:/app/user_databases
      
  chunking-worker:
    build: .
    command: python -m tldw_Server_API.app.core.Embeddings.workers.chunking_worker
    environment:
      - WORKER_TYPE=chunking
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    deploy:
      replicas: 2
      
  embedding-worker:
    build: .
    command: python -m tldw_Server_API.app.core.Embeddings.workers.embedding_worker
    environment:
      - WORKER_TYPE=embedding
      - REDIS_URL=redis://redis:6379
      - CUDA_VISIBLE_DEVICES=0,1
    depends_on:
      - redis
    deploy:
      replicas: 4
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
              
  storage-worker:
    build: .
    command: python -m tldw_Server_API.app.core.Embeddings.workers.storage_worker
    environment:
      - WORKER_TYPE=storage
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    deploy:
      replicas: 2

  orchestrator:
    build: .
    command: python -m tldw_Server_API.app.core.Embeddings.worker_orchestrator
    environment:
      - REDIS_URL=redis://redis:6379
      - PROMETHEUS_PORT=9090
    depends_on:
      - redis
    ports:
      - "9090:9090"

volumes:
  redis_data:
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: embeddings-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: embeddings-api
  template:
    metadata:
      labels:
        app: embeddings-api
    spec:
      containers:
      - name: api
        image: tldw/embeddings:v5
        ports:
        - containerPort: 8000
        env:
        - name: EMBEDDINGS_MODE
          value: "enterprise"
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /api/v1/embeddings/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: embeddings-service
spec:
  selector:
    app: embeddings-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## Configuration Reference

### Core Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `EMBEDDINGS_MODE` | `single_user` | Deployment mode |
| `MAX_BATCH_SIZE` | 100 | Max texts per batch |
| `CACHE_TTL_SECONDS` | 3600 | Cache expiry time |
| `CACHE_MAX_SIZE` | 5000 | Max cache entries |
| `CONNECTION_POOL_SIZE` | 20 | HTTP connections per provider |
| `REQUEST_TIMEOUT` | 30 | Request timeout (seconds) |

### Circuit Breaker Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `CIRCUIT_BREAKER_FAILURE_THRESHOLD` | 5 | Failures before opening |
| `CIRCUIT_BREAKER_RECOVERY_TIMEOUT` | 60 | Seconds before half-open |
| `CIRCUIT_BREAKER_SUCCESS_THRESHOLD` | 2 | Successes to close |
| `CIRCUIT_BREAKER_HALF_OPEN_CALLS` | 3 | Max calls in half-open |

### Provider Configuration

```yaml
# embeddings_config.yaml
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
    models:
      - text-embedding-ada-002
      - text-embedding-3-small
      - text-embedding-3-large
    rate_limit: 3000  # requests per minute
    
  huggingface:
    models:
      - sentence-transformers/all-MiniLM-L6-v2
      - sentence-transformers/all-mpnet-base-v2
    cache_dir: ./huggingface_cache
    device: cuda  # or cpu
    
  local_api:
    url: http://localhost:11434/api/embeddings
    models:
      - nomic-embed-text
      - all-minilm
```

---

## Monitoring & Observability

### Prometheus Metrics

The service exposes metrics at `/metrics`:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'embeddings'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

Key metrics:
- `embedding_requests_total` - Request counter
- `embedding_request_duration_seconds` - Latency histogram
- `embedding_cache_hits_total` - Cache hit rate
- `circuit_breaker_state` - Circuit breaker status
- `active_embedding_requests` - Current load

### Grafana Dashboard

Import the provided dashboard from `monitoring/grafana-dashboard.json`:

1. Embeddings throughput
2. Latency percentiles (p50, p95, p99)
3. Cache hit rate
4. Circuit breaker status per provider
5. Error rate by provider
6. Active requests

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/api/v1/embeddings/health

# Detailed metrics (requires admin)
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/embeddings/metrics

# Circuit breaker status (requires admin)
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/embeddings/circuit-breakers
```

### Logging

Configure log level and format:

```python
# config.py
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "json": {
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "embeddings.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "json"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"]
    }
}
```

---

## Troubleshooting

### Common Issues

#### 1. Circuit Breaker Open
**Symptom**: 503 Service Unavailable errors  
**Solution**:
```bash
# Check circuit breaker status
curl http://localhost:8000/api/v1/embeddings/circuit-breakers

# Reset if needed (admin only)
curl -X POST -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/embeddings/circuit-breakers/openai/reset
```

#### 2. High Memory Usage
**Symptom**: OOM errors, slow responses  
**Solution**:
```bash
# Reduce cache size
export CACHE_MAX_SIZE=1000

# Reduce batch size
export MAX_BATCH_SIZE=50

# Enable model unloading (HuggingFace)
export MODEL_UNLOAD_TIMEOUT=300
```

#### 3. Slow Embeddings
**Symptom**: High latency  
**Solution**:
```bash
# Check GPU availability
nvidia-smi

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Use ONNX for CPU inference
export EMBEDDING_PROVIDER=onnx
```

#### 4. Rate Limiting
**Symptom**: 429 Too Many Requests  
**Solution**:
```python
# Increase rate limit in code
@limiter.limit("120/minute")  # Double the limit
```

### Debug Mode

Enable detailed logging:

```bash
export LOG_LEVEL=DEBUG
export SHOW_SQL_QUERIES=true
export TRACE_REQUESTS=true
```

### Performance Tuning

#### GPU Optimization
```bash
# Use mixed precision
export CUDA_ALLOW_TF32=1

# Optimize batch size for GPU
export MAX_BATCH_SIZE=256  # For V100
export MAX_BATCH_SIZE=128  # For T4
```

#### CPU Optimization
```bash
# Use ONNX Runtime
export EMBEDDING_PROVIDER=onnx
export OMP_NUM_THREADS=8

# Reduce model precision
export USE_INT8_QUANTIZATION=true
```

---

## Migration Guide

### From v4 to v5

1. **Update imports**:
```python
# Old
from embeddings_v4 import router

# New
from embeddings_v5_production_enhanced import router
```

2. **Update configuration**:
```bash
# Remove insecure settings
unset ALLOW_FAKE_EMBEDDINGS
unset SKIP_AUTH_CHECK

# Add new settings
export CIRCUIT_BREAKER_ENABLED=true
export ENHANCED_ERROR_RECOVERY=true
```

3. **Database migration** (if using job tracking):
```sql
-- Add circuit breaker state tracking
ALTER TABLE embedding_jobs 
ADD COLUMN circuit_breaker_state VARCHAR(20) DEFAULT 'closed';

-- Add retry tracking
ALTER TABLE embedding_jobs 
ADD COLUMN retry_count INT DEFAULT 0;
```

4. **Test the migration**:
```bash
# Run test suite
pytest tests/embeddings/test_migration.py

# Verify endpoints
./scripts/verify_embeddings_api.sh
```

### Rollback Procedure

If issues occur:

1. **Immediate rollback**:
```bash
# Switch back to v4
export USE_EMBEDDINGS_V4=true
systemctl restart embeddings-service
```

2. **Data validation**:
```python
# Verify embeddings integrity
python scripts/validate_embeddings.py --check-all
```

3. **Clear corrupted cache**:
```bash
curl -X DELETE -H "Authorization: Bearer $ADMIN_TOKEN" \
  http://localhost:8000/api/v1/embeddings/cache
```

---

## Support

For issues or questions:
- GitHub Issues: [tldw_server/issues](https://github.com/your-repo/issues)
- Documentation: [/Docs/Embeddings/](./Docs/Embeddings/)
- Logs: Check `/var/log/tldw/embeddings.log`

---

**Document Version**: 1.0  
**Last Updated**: 2025-08-16  
**Maintained By**: TLDW Development Team