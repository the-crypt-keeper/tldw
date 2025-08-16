# Embeddings System Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [System Selection Guide](#system-selection-guide)
4. [Components](#components)
5. [Data Flow](#data-flow)
6. [Deployment Options](#deployment-options)
7. [Monitoring & Operations](#monitoring--operations)

## System Overview

The tldw_server Embeddings System provides a comprehensive solution for generating text embeddings through multiple providers. The system offers two distinct architectures to accommodate different use cases:

1. **Synchronous API** (`embeddings_v5_production.py`) - Direct request-response model ideal for single users and small deployments
2. **Job-Based System** (Worker Architecture) - Distributed, queue-based processing for enterprise and multi-tenant deployments

### Key Features
- 🌐 Multi-provider support (OpenAI, HuggingFace, Cohere, Google, Mistral, etc.)
- ⚡ High-performance caching with TTL
- 🔄 Automatic retry logic with circuit breakers
- 📊 Comprehensive monitoring and metrics
- 🔒 Security-first design with proper authorization
- 🎯 Production-ready with extensive testing

## Architecture

### Overall System Architecture

```mermaid
graph TB
    subgraph "Client Applications"
        CA[Client App]
        SDK[SDK/Library]
        CLI[CLI Tool]
    end
    
    subgraph "API Gateway"
        AUTH[Authentication]
        RL[Rate Limiter]
        ROUTE[Router]
    end
    
    subgraph "Embeddings System"
        direction TB
        subgraph "Synchronous Path"
            SYNC[Synchronous API<br/>embeddings_v5]
            CACHE1[TTL Cache]
            POOL1[Connection Pool]
        end
        
        subgraph "Asynchronous Path"
            JOBS[Job API<br/>embeddings_jobs]
            REDIS[(Redis Queue)]
            WORKERS[Worker Pool]
            CACHE2[Distributed Cache]
        end
        
        CONFIG[Configuration<br/>Manager]
    end
    
    subgraph "Embedding Providers"
        OPENAI[OpenAI API]
        HF[HuggingFace]
        COHERE[Cohere API]
        LOCAL[Local Models]
        OTHER[Other Providers]
    end
    
    subgraph "Storage"
        DB[(PostgreSQL/<br/>SQLite)]
        CHROMA[(ChromaDB)]
        S3[Object Storage]
    end
    
    subgraph "Monitoring"
        PROM[Prometheus]
        GRAF[Grafana]
        LOGS[Log Aggregation]
    end
    
    CA --> AUTH
    SDK --> AUTH
    CLI --> AUTH
    
    AUTH --> RL
    RL --> ROUTE
    
    ROUTE -->|Small Request| SYNC
    ROUTE -->|Large Request| JOBS
    
    SYNC --> CACHE1
    CACHE1 --> POOL1
    POOL1 --> OPENAI
    POOL1 --> HF
    POOL1 --> COHERE
    POOL1 --> LOCAL
    POOL1 --> OTHER
    
    JOBS --> REDIS
    REDIS --> WORKERS
    WORKERS --> CACHE2
    CACHE2 --> OPENAI
    CACHE2 --> HF
    CACHE2 --> COHERE
    
    SYNC --> DB
    WORKERS --> DB
    WORKERS --> CHROMA
    
    SYNC --> PROM
    WORKERS --> PROM
    PROM --> GRAF
    
    CONFIG -.->|Configure| SYNC
    CONFIG -.->|Configure| JOBS
    CONFIG -.->|Configure| WORKERS
```

### Synchronous System Architecture

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Auth
    participant Cache
    participant Pool
    participant Provider
    participant Metrics
    
    Client->>API: POST /embeddings
    API->>Auth: Validate Token
    Auth-->>API: User Context
    
    API->>Cache: Check Cache
    alt Cache Hit
        Cache-->>API: Return Embeddings
        API-->>Client: 200 OK (from cache)
    else Cache Miss
        API->>Pool: Get Connection
        Pool->>Provider: Create Embeddings
        Provider-->>Pool: Embeddings
        Pool-->>API: Embeddings
        
        par Cache Update
            API->>Cache: Store Result
        and Metrics
            API->>Metrics: Record Request
        end
        
        API-->>Client: 200 OK
    end
```

### Job-Based System Architecture

```mermaid
graph LR
    subgraph "Client Layer"
        C[Client]
    end
    
    subgraph "API Layer"
        API[Job API]
        WS[WebSocket]
    end
    
    subgraph "Queue Layer"
        JM[Job Manager]
        RQ[(Redis Queues)]
        PUB[Pub/Sub]
    end
    
    subgraph "Worker Layer"
        subgraph "Chunking Workers"
            CW1[Worker 1]
            CW2[Worker 2]
        end
        
        subgraph "Embedding Workers"
            EW1[Worker 1]
            EW2[Worker 2]
            EW3[Worker 3]
        end
        
        subgraph "Storage Workers"
            SW1[Worker 1]
            SW2[Worker 2]
        end
    end
    
    subgraph "Storage Layer"
        JDB[(Jobs DB)]
        CDB[(ChromaDB)]
    end
    
    C -->|1. Submit Job| API
    API -->|2. Create Job| JM
    JM -->|3. Queue| RQ
    
    RQ -->|4. Chunk Task| CW1
    CW1 -->|5. Chunks| RQ
    
    RQ -->|6. Embed Task| EW1
    EW1 -->|7. Embeddings| RQ
    
    RQ -->|8. Store Task| SW1
    SW1 -->|9. Save| CDB
    SW1 -->|10. Update| JDB
    
    JDB -->|11. Status| API
    API -->|12. Updates| WS
    WS -.->|Real-time| C
    
    PUB -.->|Events| WS
    SW1 -.->|Complete| PUB
```

## System Selection Guide

```mermaid
flowchart TD
    START([Start]) --> Q1{Number of Users?}
    
    Q1 -->|Single User| SYNC[Use Synchronous API]
    Q1 -->|< 10 Users| Q2{Concurrent Requests?}
    Q1 -->|10+ Users| Q3{Infrastructure Available?}
    
    Q2 -->|Low| SYNC
    Q2 -->|High| Q3
    
    Q3 -->|Basic Server| Q4{Performance Requirements?}
    Q3 -->|Kubernetes/Docker| JOBS[Use Job-Based System]
    
    Q4 -->|Low Latency<br/>< 200ms| SYNC
    Q4 -->|High Throughput<br/>Batch Processing| Q5{Redis Available?}
    
    Q5 -->|No| SYNC
    Q5 -->|Yes| JOBS
    
    SYNC --> END1([Synchronous System<br/>Simple, Direct, Fast])
    JOBS --> END2([Job-Based System<br/>Scalable, Resilient, Observable])
    
    style SYNC fill:#90EE90
    style JOBS fill:#87CEEB
    style END1 fill:#90EE90
    style END2 fill:#87CEEB
```

## Components

### Core Components

```mermaid
graph TB
    subgraph "Synchronous Components"
        API[FastAPI Router]
        TTL[TTL Cache]
        CPM[Connection Pool Manager]
        RL[Rate Limiter]
        AUTH[Authorization]
    end
    
    subgraph "Job-Based Components"
        JM[Job Manager]
        WO[Worker Orchestrator]
        CW[Chunking Worker]
        EW[Embedding Worker]
        SW[Storage Worker]
    end
    
    subgraph "Shared Components"
        PROV[Provider Factory]
        CONFIG[Configuration Manager]
        METRICS[Metrics Collector]
        HEALTH[Health Monitor]
    end
    
    API --> TTL
    API --> CPM
    API --> RL
    API --> AUTH
    
    JM --> WO
    WO --> CW
    WO --> EW
    WO --> SW
    
    API --> PROV
    EW --> PROV
    
    CONFIG --> API
    CONFIG --> JM
    CONFIG --> WO
    
    API --> METRICS
    WO --> METRICS
    METRICS --> HEALTH
```

### Cache Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Empty
    Empty --> Populated: Set(key, value)
    Populated --> Accessed: Get(key)
    Accessed --> Populated: Update access time
    Populated --> Expired: TTL exceeded
    Expired --> Empty: Cleanup task
    Populated --> Evicted: Cache full (LRU)
    Evicted --> Empty: Remove entry
    Accessed --> Evicted: LRU eviction
    
    note right of Populated
        Entry stored with:
        - Value
        - Timestamp
        - Last access time
    end note
    
    note right of Expired
        Cleanup runs every 5 minutes
        Removes all expired entries
    end note
```

## Data Flow

### Request Processing Flow

```mermaid
flowchart LR
    subgraph "Input Processing"
        IN[Request Input]
        VAL{Validate}
        PARSE[Parse Input]
    end
    
    subgraph "Provider Selection"
        PSEL{Select Provider}
        PCONF[Load Config]
        PCRED[Get Credentials]
    end
    
    subgraph "Embedding Generation"
        CACHE{Cache Check}
        BATCH[Batch Processing]
        RETRY[Retry Logic]
        CREATE[Create Embeddings]
    end
    
    subgraph "Output Processing"
        FORMAT[Format Response]
        ENCODE{Encoding?}
        B64[Base64 Encode]
        RESP[Response]
    end
    
    IN --> VAL
    VAL -->|Valid| PARSE
    VAL -->|Invalid| ERR1[400 Error]
    
    PARSE --> PSEL
    PSEL --> PCONF
    PCONF --> PCRED
    
    PCRED --> CACHE
    CACHE -->|Hit| FORMAT
    CACHE -->|Miss| BATCH
    
    BATCH --> RETRY
    RETRY --> CREATE
    CREATE -->|Success| FORMAT
    CREATE -->|Fail| RETRY
    RETRY -->|Max Retries| ERR2[503 Error]
    
    FORMAT --> ENCODE
    ENCODE -->|base64| B64
    ENCODE -->|float| RESP
    B64 --> RESP
```

### Error Handling Flow

```mermaid
flowchart TD
    REQ[Request] --> PROC{Process}
    
    PROC -->|Success| RESP[Response]
    PROC -->|Error| ETYPE{Error Type}
    
    ETYPE -->|Validation| E400[400 Bad Request]
    ETYPE -->|Auth| E401[401 Unauthorized]
    ETYPE -->|Permission| E403[403 Forbidden]
    ETYPE -->|Not Found| E404[404 Not Found]
    ETYPE -->|Rate Limit| E429[429 Too Many Requests]
    ETYPE -->|Server Error| RETRY{Retryable?}
    
    RETRY -->|Yes| BACKOFF[Exponential Backoff]
    RETRY -->|No| E500[500 Internal Error]
    
    BACKOFF --> ATTEMPT{Retry Attempt}
    ATTEMPT -->|< Max| PROC
    ATTEMPT -->|>= Max| E503[503 Service Unavailable]
    
    E400 --> LOG1[Log Warning]
    E401 --> LOG1
    E403 --> LOG1
    E404 --> LOG1
    E429 --> LOG2[Log Info]
    E500 --> LOG3[Log Error]
    E503 --> LOG3
    
    LOG1 --> CLIENT[Return to Client]
    LOG2 --> CLIENT
    LOG3 --> ALERT[Alert Ops]
    ALERT --> CLIENT
```

## Deployment Options

### Deployment Architecture Comparison

```mermaid
graph TB
    subgraph "Single Server Deployment"
        SS_API[API Server]
        SS_CACHE[In-Memory Cache]
        SS_DB[(SQLite)]
        SS_API --> SS_CACHE
        SS_API --> SS_DB
    end
    
    subgraph "Docker Compose Deployment"
        DC_API[API Container]
        DC_REDIS[(Redis)]
        DC_WORK[Worker Container]
        DC_DB[(PostgreSQL)]
        DC_PROM[Prometheus]
        
        DC_API --> DC_REDIS
        DC_REDIS --> DC_WORK
        DC_WORK --> DC_DB
        DC_API --> DC_PROM
        DC_WORK --> DC_PROM
    end
    
    subgraph "Kubernetes Deployment"
        subgraph "API Pods"
            K_API1[API Pod 1]
            K_API2[API Pod 2]
        end
        
        subgraph "Worker Pods"
            K_CW[Chunking Workers]
            K_EW[Embedding Workers]
            K_SW[Storage Workers]
        end
        
        K_REDIS[(Redis Cluster)]
        K_DB[(PostgreSQL HA)]
        K_S3[Object Storage]
        
        K_API1 --> K_REDIS
        K_API2 --> K_REDIS
        K_REDIS --> K_CW
        K_REDIS --> K_EW
        K_REDIS --> K_SW
        K_SW --> K_DB
        K_SW --> K_S3
    end
```

## Monitoring & Operations

### Metrics Collection Flow

```mermaid
graph LR
    subgraph "Application Metrics"
        APP[Application]
        COUNT[Request Counter]
        HIST[Latency Histogram]
        GAUGE[Active Requests]
    end
    
    subgraph "System Metrics"
        CPU[CPU Usage]
        MEM[Memory Usage]
        DISK[Disk I/O]
        NET[Network I/O]
    end
    
    subgraph "Business Metrics"
        USERS[Active Users]
        QUOTA[Quota Usage]
        PROVIDERS[Provider Usage]
        CACHE_HIT[Cache Hit Rate]
    end
    
    subgraph "Monitoring Stack"
        PROM[Prometheus]
        ALERT[AlertManager]
        GRAF[Grafana]
    end
    
    APP --> COUNT
    APP --> HIST
    APP --> GAUGE
    
    COUNT --> PROM
    HIST --> PROM
    GAUGE --> PROM
    
    CPU --> PROM
    MEM --> PROM
    DISK --> PROM
    NET --> PROM
    
    USERS --> PROM
    QUOTA --> PROM
    PROVIDERS --> PROM
    CACHE_HIT --> PROM
    
    PROM --> ALERT
    PROM --> GRAF
    
    ALERT -->|Critical| ONCALL[On-Call]
    ALERT -->|Warning| SLACK[Slack]
    GRAF -->|Dashboards| OPS[Operations]
```

### Health Check System

```mermaid
stateDiagram-v2
    [*] --> Initializing
    
    Initializing --> Healthy: All checks pass
    Initializing --> Degraded: Some checks fail
    Initializing --> Unhealthy: Critical checks fail
    
    Healthy --> Degraded: Non-critical failure
    Healthy --> Unhealthy: Critical failure
    
    Degraded --> Healthy: Issues resolved
    Degraded --> Unhealthy: More failures
    
    Unhealthy --> Degraded: Partial recovery
    Unhealthy --> Healthy: Full recovery
    Unhealthy --> [*]: Service shutdown
    
    note right of Healthy
        - All providers accessible
        - Cache operational
        - Database connected
        - Queue accessible (if job-based)
    end note
    
    note right of Degraded
        - Some providers unavailable
        - Cache issues (non-critical)
        - High latency
        - Queue backlog
    end note
    
    note right of Unhealthy
        - Database unreachable
        - All providers down
        - Redis disconnected (if job-based)
        - OOM conditions
    end note
```

## Security Model

```mermaid
graph TB
    subgraph "Authentication Layer"
        TOKEN[JWT/API Key]
        VALIDATE[Validate Token]
        EXTRACT[Extract User]
    end
    
    subgraph "Authorization Layer"
        ROLE{Check Role}
        QUOTA{Check Quota}
        RATE{Rate Limit}
    end
    
    subgraph "Request Processing"
        SANITIZE[Input Sanitization]
        VALIDATE_REQ[Validate Request]
        PROCESS[Process Request]
    end
    
    subgraph "Audit Layer"
        LOG[Audit Log]
        METRICS[Security Metrics]
        ALERT[Security Alerts]
    end
    
    TOKEN --> VALIDATE
    VALIDATE -->|Valid| EXTRACT
    VALIDATE -->|Invalid| REJECT1[401 Unauthorized]
    
    EXTRACT --> ROLE
    ROLE -->|Admin| PROCESS
    ROLE -->|User| QUOTA
    ROLE -->|None| REJECT2[403 Forbidden]
    
    QUOTA -->|Available| RATE
    QUOTA -->|Exceeded| REJECT3[429 Too Many Requests]
    
    RATE -->|OK| SANITIZE
    RATE -->|Limited| REJECT3
    
    SANITIZE --> VALIDATE_REQ
    VALIDATE_REQ --> PROCESS
    
    PROCESS --> LOG
    LOG --> METRICS
    METRICS --> ALERT
    
    REJECT1 --> LOG
    REJECT2 --> LOG
    REJECT3 --> LOG
```

## Performance Optimization

```mermaid
graph LR
    subgraph "Optimization Layers"
        subgraph "Caching"
            L1[L1: In-Memory]
            L2[L2: Redis]
            L3[L3: CDN]
        end
        
        subgraph "Batching"
            BATCH[Request Batching]
            PBATCH[Provider Batching]
            DBATCH[DB Batching]
        end
        
        subgraph "Pooling"
            CPOOL[Connection Pool]
            TPOOL[Thread Pool]
            WPOOL[Worker Pool]
        end
        
        subgraph "Scaling"
            HSCALE[Horizontal Scaling]
            VSCALE[Vertical Scaling]
            AUTO[Auto-scaling]
        end
    end
    
    L1 --> L2
    L2 --> L3
    
    BATCH --> PBATCH
    PBATCH --> DBATCH
    
    CPOOL --> TPOOL
    TPOOL --> WPOOL
    
    HSCALE --> AUTO
    VSCALE --> AUTO
```

---

## Next Steps

For detailed implementation guidance, see:
- [Developer Guide](./Embeddings-Developer-Guide.md) - For developers working on the codebase
- [API Consumer Guide](./Embeddings-API-Guide.md) - For users consuming the API

For specific deployment scenarios, refer to:
- [Single Server Setup](../Deployment/single-server.md)
- [Docker Compose Setup](../Deployment/docker-compose.md)
- [Kubernetes Deployment](../Deployment/kubernetes.md)