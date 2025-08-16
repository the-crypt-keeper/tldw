# Embeddings System Dual-Mode Deployment Plan
## Supporting Both Synchronous and Job-Based Architectures

**Document Version**: 2.0  
**Last Updated**: 2025-08-16  
**Status**: Ready for Implementation

---

## Executive Summary

This document outlines the deployment strategy for supporting **both** the synchronous embeddings API (`embeddings_v4.py`) and the new job-based architecture side-by-side. This dual-mode approach allows single users and small installations to use the simpler synchronous system, while large deployments can leverage the scalable job-based architecture.

**Key Principle**: Both systems will coexist permanently, not as a migration but as complementary solutions for different use cases.

---

## System Comparison & Use Cases

### Synchronous System (embeddings_v4)
**Best For**: Single users, small teams, simple deployments
- **Architecture**: Synchronous, request-response
- **Endpoint**: `/api/v1/embeddings`
- **Processing**: In-process, blocking
- **Setup**: Zero infrastructure required
- **Response**: Immediate results
- **Use Cases**:
  - Personal installations
  - Small teams (< 10 users)
  - Development/testing environments
  - Low-latency requirements
  - Simple deployment scenarios

### Job-Based System (New)
**Best For**: Multi-user deployments, high throughput, enterprise
- **Architecture**: Asynchronous, job-based with Redis queues
- **Endpoints**: `/api/v1/embeddings/jobs/*`
- **Processing**: Distributed workers
- **Setup**: Requires Redis and worker infrastructure
- **Response**: Job ID with status polling
- **Use Cases**:
  - Multi-tenant deployments
  - High-volume processing
  - Resource-intensive workloads
  - Enterprise installations
  - Batch processing scenarios

---

## Deployment Options

### Option 1: Synchronous Only (Default)
**For**: Single users, small installations

No additional setup required. The system works out of the box with:
```bash
python -m uvicorn tldw_Server_API.app.main:app
```

### Option 2: Job-Based Only
**For**: High-scale deployments that don't need backward compatibility

Deploy the full infrastructure:
```bash
docker-compose -f docker-compose.embeddings.yml up -d
```

Set configuration:
```bash
export EMBEDDINGS_MODE=jobs_only
```

### Option 3: Dual-Mode (Recommended for transition)
**For**: Organizations wanting to offer both options

Enable both systems and let users/applications choose based on their needs.

## Implementation Phases

### Phase 1: Infrastructure Setup (Week 1)
**Goal**: Deploy supporting infrastructure without affecting current system

#### Tasks:
1. **Deploy Redis**
   ```bash
   docker-compose -f docker-compose.embeddings.yml up -d redis
   ```

2. **Initialize Databases**
   ```python
   from tldw_Server_API.app.core.DB_Management.Embeddings_Jobs_DB import EmbeddingsJobsDatabase
   db = EmbeddingsJobsDatabase()
   # Tables created automatically on first connection
   ```

3. **Deploy Workers** (in shadow mode)
   ```bash
   docker-compose -f docker-compose.embeddings.yml up -d chunking-workers embedding-workers storage-workers
   ```

4. **Setup Monitoring**
   ```bash
   docker-compose -f docker-compose.embeddings.yml --profile monitoring up -d
   ```

#### Validation:
- [ ] Redis accessible at localhost:6379
- [ ] Workers running without errors
- [ ] Prometheus collecting metrics
- [ ] Database tables created

### Phase 2: Parallel Deployment (Week 2)
**Goal**: Run both systems in parallel with feature flag control

#### Tasks:
1. **Add Feature Flag**
   ```python
   # In config.txt or environment
   EMBEDDINGS_USE_JOBS=false  # Start with job system disabled
   ```

2. **Update main.py Router**
   ```python
   # main.py
   from tldw_Server_API.app.api.v1.endpoints import embeddings_v4, embeddings_jobs
   
   # Include both routers
   app.include_router(embeddings_v4.router, prefix=f"{API_V1_PREFIX}", tags=["embeddings"])
   app.include_router(embeddings_jobs.router, prefix=f"{API_V1_PREFIX}", tags=["embeddings-jobs"])
   ```

3. **Create Adapter Layer**
   ```python
   # embeddings_adapter.py
   async def create_embedding_adaptive(request, use_jobs=False):
       if use_jobs:
           return await create_embedding_job(request)
       else:
           return await create_embedding_sync(request)
   ```

4. **Add Backward Compatibility Wrapper**
   ```python
   # Wrap old endpoint to optionally use new system
   @router.post("/embeddings")
   async def create_embeddings_compatible(request):
       if settings.EMBEDDINGS_USE_JOBS:
           # Convert to job and poll for completion
           job = await create_job(request)
           result = await wait_for_completion(job.job_id, timeout=30)
           return convert_to_v4_response(result)
       else:
           return await original_v4_handler(request)
   ```

#### Validation:
- [ ] Both endpoints accessible
- [ ] Feature flag controls routing
- [ ] No impact on existing clients

### Phase 3: Gradual Migration (Weeks 3-4)
**Goal**: Progressively migrate traffic to new system

#### Tasks:
1. **Enable for Internal Testing**
   ```python
   # Route specific users to new system
   if user.tier == "internal_test":
       use_jobs = True
   ```

2. **Implement Traffic Splitting**
   ```python
   # Percentage-based routing
   import random
   use_jobs = random.random() < settings.JOBS_TRAFFIC_PERCENTAGE
   ```

3. **Monitor Performance**
   - Compare latency: old vs new
   - Track error rates
   - Monitor queue depths
   - Check resource utilization

4. **Gradual Rollout Schedule**
   - Day 1-3: 5% traffic
   - Day 4-7: 25% traffic
   - Week 2: 50% traffic
   - Week 3: 75% traffic
   - Week 4: 100% traffic

#### Validation:
- [ ] Performance metrics comparable or better
- [ ] Error rate < 0.1%
- [ ] Queue processing time < 5s for small jobs
- [ ] User quotas enforced correctly

### Phase 4: Client Migration (Weeks 5-6)
**Goal**: Update clients to use new endpoints directly

#### Tasks:
1. **Update Documentation**
   - API documentation
   - Integration guides
   - Migration guide for clients

2. **Client SDK Updates**
   ```python
   # Old client code
   response = client.create_embeddings(text)
   
   # New client code
   job = client.create_embedding_job(text)
   result = client.wait_for_job(job.job_id)
   ```

3. **Deprecation Notices**
   ```python
   @deprecated(version='2.0', reason='Use /embeddings/jobs instead')
   @router.post("/embeddings")
   ```

4. **Communication Plan**
   - Email notification to API users
   - Deprecation timeline announcement
   - Support for migration questions

#### Validation:
- [ ] Documentation updated
- [ ] Client libraries updated
- [ ] Deprecation warnings in place
- [ ] Users notified

### Phase 5: Optimization & Documentation (Week 7)
**Goal**: Optimize both systems and provide clear documentation

#### Tasks:
1. **Document System Selection**
   ```markdown
   # In README or docs
   ## Choosing an Embeddings System
   
   Use **Synchronous** (`/api/v1/embeddings`) when:
   - Running single-user installation
   - Need immediate responses
   - Don't want Redis dependency
   
   Use **Job-Based** (`/api/v1/embeddings/jobs`) when:
   - Supporting multiple concurrent users
   - Processing large batches
   - Need quota management
   ```

2. **Optimize Worker Configuration** (for job-based deployments)
   - Tune worker counts based on load
   - Adjust batch sizes
   - Optimize GPU allocation

3. **Database Cleanup** (for job-based deployments)
   ```python
   # Archive old job data
   db.cleanup_old_jobs(days_to_keep=30)
   ```

4. **Configuration Templates**
   ```yaml
   # config_single_user.yaml
   embeddings:
     mode: synchronous
     
   # config_enterprise.yaml
   embeddings:
     mode: dual  # or jobs_only
     worker_pools:
       chunking:
         num_workers: 4
       embedding:
         num_workers: 6
   ```

---

## Configuration for Dual-Mode Operation

### Environment Variables
```bash
# Embeddings system mode
EMBEDDINGS_MODE=dual  # Options: synchronous, jobs_only, dual (default: synchronous)

# Job system configuration (only needed if mode includes jobs)
REDIS_URL=redis://localhost:6379
EMBEDDINGS_WORKERS_ENABLED=true
EMBEDDINGS_DEFAULT_MODE=synchronous  # Which to use when both available

# Auto-selection based on load
EMBEDDINGS_AUTO_SELECT=true  # Automatically choose based on request size
EMBEDDINGS_SYNC_MAX_CHUNKS=10  # Use sync for <= 10 chunks
```

### API Behavior in Dual Mode

#### Smart Routing
```python
# Automatic selection based on request characteristics
@router.post("/embeddings/auto")
async def create_embeddings_auto(request):
    # Small requests -> synchronous
    if len(request.input) < 10:
        return await embeddings_v4.create_embeddings(request)
    
    # Large requests -> job-based
    else:
        job = await embeddings_jobs.create_job(request)
        return {"job_id": job.job_id, "mode": "async"}
```

#### Client Headers
```http
# Client can request specific mode
POST /api/v1/embeddings
X-Embeddings-Mode: synchronous  # or "jobs"
```

---

## Rollback Plan

### Immediate Rollback (Any Phase)
1. **Feature Flag Disable**
   ```bash
   export EMBEDDINGS_USE_JOBS=false
   ```

2. **Stop Workers** (if needed)
   ```bash
   docker-compose -f docker-compose.embeddings.yml stop chunking-workers embedding-workers storage-workers
   ```

3. **Clear Queues** (if corrupted)
   ```bash
   redis-cli FLUSHDB
   ```

### Data Recovery
1. **Job Status Recovery**
   ```sql
   -- Mark incomplete jobs as failed
   UPDATE embedding_jobs 
   SET status = 'failed', 
       error_message = 'System rollback',
       completed_at = CURRENT_TIMESTAMP
   WHERE status IN ('pending', 'chunking', 'embedding', 'storing');
   ```

2. **Reprocess Failed Jobs**
   ```python
   # Script to reprocess using old system
   failed_jobs = db.get_failed_jobs()
   for job in failed_jobs:
       create_embedding_sync(job.media_id)
   ```

---

## Monitoring & Success Metrics

### Key Performance Indicators
| Metric | Current (v4) | Target (Jobs) | Acceptable Range |
|--------|-------------|---------------|------------------|
| P50 Latency | 2s | 1s | 0.5-2s |
| P99 Latency | 10s | 5s | 3-8s |
| Throughput | 100 req/min | 1000 req/min | 500+ req/min |
| Error Rate | 0.5% | 0.1% | < 0.5% |
| GPU Utilization | 30% | 80% | 60-90% |

### Monitoring Dashboards
1. **Job Processing Dashboard**
   - Jobs created/completed per minute
   - Queue depths by stage
   - Average processing time by stage
   - Error rates by stage

2. **User Quota Dashboard**
   - Daily quota usage by tier
   - Concurrent jobs by user
   - Quota exceeded events
   - Top users by usage

3. **Worker Performance Dashboard**
   - Worker health status
   - Processing rate per worker
   - GPU/CPU utilization
   - Memory usage

### Alerts
```yaml
# Prometheus alert rules
- alert: HighQueueDepth
  expr: embedding_queue_depth > 1000
  for: 5m
  
- alert: WorkerDown
  expr: up{job="worker"} == 0
  for: 1m
  
- alert: HighErrorRate
  expr: rate(embedding_jobs_failed[5m]) > 0.01
  for: 5m
```

---

## Risk Mitigation

### Identified Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Redis failure | Low | High | Redis persistence + backup Redis instance |
| Worker memory leak | Medium | Medium | Memory limits + automatic restart |
| Queue backlog | Medium | Medium | Auto-scaling + circuit breaker |
| Database lock contention | Low | Medium | Connection pooling + query optimization |
| Model loading delays | Medium | Low | Model preloading + caching |

---

## Testing Strategy

### Pre-Production Testing
1. **Load Testing**
   ```bash
   # Using locust or similar
   locust -f tests/load_test_embeddings.py --users 100 --spawn-rate 10
   ```

2. **Chaos Testing**
   - Kill random workers
   - Simulate Redis outage
   - Fill queues to capacity
   - Corrupt messages

3. **Integration Testing**
   ```bash
   python -m pytest tests/Embeddings/test_integration.py -v
   ```

### Production Testing
1. **Canary Deployment**
   - Deploy to single instance
   - Monitor for 24 hours
   - Gradual rollout

2. **A/B Testing**
   - Compare performance metrics
   - User experience feedback
   - Error rate comparison

---

## Communication Plan

### Stakeholders
- **Engineering Team**: Technical details, implementation timeline
- **DevOps Team**: Infrastructure requirements, monitoring setup
- **Product Team**: Feature changes, user impact
- **Support Team**: Known issues, troubleshooting guide
- **API Users**: Migration guide, deprecation timeline

### Timeline Communication
- **T-4 weeks**: Initial announcement
- **T-2 weeks**: Detailed migration guide
- **T-0**: Go-live announcement
- **T+2 weeks**: Deprecation notice for old endpoints
- **T+8 weeks**: Final removal notice

---

## Post-Migration Optimization

### Week 8+: Optimization Phase
1. **Performance Tuning**
   - Analyze metrics from production load
   - Optimize batch sizes
   - Tune worker counts
   - Adjust queue priorities

2. **Cost Optimization**
   - Right-size infrastructure
   - Optimize GPU usage
   - Review Redis memory usage

3. **Feature Enhancements**
   - Add batch job submission
   - Implement job chaining
   - Add scheduling capabilities
   - Enhanced quota management

---

## Appendix

### A. Configuration Files

#### Redis Configuration
```conf
# redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
appendonly yes
appendfsync everysec
```

#### Worker Configuration
```yaml
# embeddings_config.yaml
orchestration:
  redis_url: redis://localhost:6379
  prometheus_port: 9090
  
worker_pools:
  chunking:
    num_workers: 2
    queue_name: embeddings:chunking
    
  embedding:
    num_workers: 4
    queue_name: embeddings:embedding
    gpu_allocation:
      - worker_0: 0
      - worker_1: 0
      - worker_2: 1
      - worker_3: 1
```

### B. Scripts

#### Health Check Script
```python
# check_health.py
import redis
import requests

def check_system_health():
    # Check Redis
    r = redis.Redis(host='localhost', port=6379)
    assert r.ping()
    
    # Check API
    response = requests.get('http://localhost:8000/health')
    assert response.status_code == 200
    
    # Check workers
    metrics = requests.get('http://localhost:9090/metrics')
    assert 'worker_count' in metrics.text
    
    print("All systems operational")
```

### C. Troubleshooting Guide

| Issue | Symptoms | Solution |
|-------|----------|----------|
| Jobs stuck in pending | Queue depth increasing, no progress | Restart workers, check Redis connection |
| High memory usage | OOM errors, slow processing | Reduce batch sizes, add memory limits |
| Slow embedding generation | High latency, GPU underutilized | Increase batch size, check model loading |
| Database locks | Timeout errors, slow queries | Optimize queries, increase connection pool |

---

## Sign-off

This migration plan has been reviewed and approved by:

- [ ] Engineering Lead
- [ ] DevOps Lead
- [ ] Product Manager
- [ ] CTO/Technical Director

**Next Steps**: Begin Phase 1 infrastructure setup upon approval.