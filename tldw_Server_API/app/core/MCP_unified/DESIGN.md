# Unified MCP Module Design

## Overview
This document outlines the design for consolidating MCP v1 and MCP v2 into a single, secure, production-ready module.

## Architecture Goals
1. **Security First**: Fix all identified vulnerabilities
2. **Modular Design**: Keep the module system from v2
3. **Performance**: Add proper connection pooling and caching
4. **Testability**: Design for comprehensive testing
5. **Production Ready**: Include monitoring, metrics, and proper error handling

## Core Components

### 1. Configuration Management
```python
# config.py - Secure configuration management
class MCPConfig:
    jwt_secret: str = Field(default_factory=lambda: os.environ.get("MCP_JWT_SECRET"))
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30
    
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    
    max_websocket_connections: int = 1000
    max_message_size: int = 1024 * 1024  # 1MB
    
    database_pool_size: int = 20
    database_pool_timeout: int = 30
    
    redis_url: Optional[str] = Field(default_factory=lambda: os.environ.get("REDIS_URL"))
    
    class Config:
        env_file = ".env"
```

### 2. Unified Server Architecture
```
MCP_unified/
├── __init__.py
├── config.py                 # Secure configuration
├── server.py                 # Main server implementation
├── protocol.py               # MCP protocol handler
├── auth/
│   ├── __init__.py
│   ├── jwt_manager.py        # JWT with env-based secrets
│   ├── rbac.py              # Role-based access control
│   └── rate_limiter.py      # Redis-backed rate limiting
├── modules/
│   ├── __init__.py
│   ├── base.py              # Base module interface
│   ├── registry.py          # Module registry with health checks
│   └── implementations/     # Actual module implementations
├── storage/
│   ├── __init__.py
│   ├── database.py          # Connection pooling
│   └── cache.py             # Redis caching
├── monitoring/
│   ├── __init__.py
│   ├── metrics.py           # Prometheus metrics
│   └── health.py            # Health check endpoints
└── tests/
    ├── unit/
    ├── integration/
    └── security/
```

## Security Improvements

### 1. Authentication & Authorization
- JWT secrets from environment variables
- Refresh token rotation
- Session management with Redis
- RBAC with fine-grained permissions
- API key management with encryption

### 2. Input Validation
- Pydantic models for all inputs
- Request size limits
- Parameter sanitization
- SQL injection prevention

### 3. Rate Limiting
- Redis-backed distributed rate limiting
- Per-user and per-endpoint limits
- Sliding window algorithm
- DDoS protection

### 4. Security Headers
- CORS configuration
- CSP headers
- XSS protection
- HSTS enforcement

## Module System (Best of v2)

### Base Module Interface
```python
class BaseModule(ABC):
    """Enhanced base module with production features"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize with health check"""
        
    @abstractmethod
    async def shutdown(self) -> None:
        """Graceful shutdown"""
        
    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """Detailed health status"""
        
    @abstractmethod
    async def get_metrics(self) -> Dict[str, Any]:
        """Module metrics for monitoring"""
```

### Module Registry with Circuit Breaker
```python
class ModuleRegistry:
    """Enhanced registry with circuit breaker pattern"""
    
    async def register_module(self, module: BaseModule) -> None:
        """Register with health verification"""
        
    async def execute_with_circuit_breaker(
        self, module_id: str, operation: Callable
    ) -> Any:
        """Execute with circuit breaker protection"""
```

## Performance Enhancements

### 1. Database Connection Pooling
```python
class DatabasePool:
    """SQLAlchemy async connection pool"""
    
    def __init__(self, config: MCPConfig):
        self.engine = create_async_engine(
            config.database_url,
            pool_size=config.database_pool_size,
            max_overflow=20,
            pool_timeout=config.database_pool_timeout,
            pool_recycle=3600
        )
```

### 2. Caching Strategy
- Redis for distributed cache
- LRU memory cache for hot data
- Cache invalidation strategy
- TTL-based expiration

### 3. Async Processing
- Proper async context managers
- Background task queue
- WebSocket connection pooling
- Graceful shutdown handling

## Error Handling

### 1. Custom Exception Hierarchy
```python
class MCPException(Exception):
    """Base MCP exception"""
    
class MCPAuthenticationError(MCPException):
    """Authentication failed"""
    
class MCPAuthorizationError(MCPException):
    """Authorization failed"""
    
class MCPRateLimitError(MCPException):
    """Rate limit exceeded"""
    
class MCPModuleError(MCPException):
    """Module execution error"""
```

### 2. Error Response Format
```python
class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[Dict[str, Any]]
    request_id: str
    timestamp: datetime
```

## Monitoring & Observability

### 1. Metrics (Prometheus)
- Request rate and latency
- Error rates by type
- Module performance metrics
- WebSocket connection metrics
- Cache hit/miss rates

### 2. Logging
- Structured logging with context
- Request correlation IDs
- Security event logging
- Performance logging

### 3. Health Checks
- Liveness probe
- Readiness probe
- Module health aggregation
- Dependency health checks

## Testing Strategy

### 1. Unit Tests (>80% coverage)
- Module tests
- Protocol handler tests
- Authentication tests
- Rate limiter tests

### 2. Integration Tests
- End-to-end API tests
- WebSocket communication tests
- Module interaction tests
- Database integration tests

### 3. Security Tests
- Authentication bypass attempts
- SQL injection tests
- XSS prevention tests
- Rate limit bypass tests

### 4. Performance Tests
- Load testing
- Stress testing
- Memory leak detection
- Connection pool testing

## Migration Plan

### Phase 1: Core Infrastructure (Week 1)
1. Set up unified module structure
2. Implement secure configuration
3. Create base module interface
4. Set up connection pooling

### Phase 2: Security Implementation (Week 2)
1. Implement JWT with env secrets
2. Add RBAC system
3. Implement rate limiting
4. Add input validation

### Phase 3: Module Migration (Week 3)
1. Migrate existing modules to new base
2. Add health checks to all modules
3. Implement circuit breakers
4. Add module metrics

### Phase 4: Testing & Documentation (Week 4)
1. Write comprehensive tests
2. Security testing
3. Performance testing
4. Update documentation

## Backwards Compatibility

### API Compatibility Layer
```python
class MCPCompatibilityLayer:
    """Maintains compatibility with existing clients"""
    
    async def handle_v1_request(self, request: Dict) -> Dict:
        """Convert v1 requests to unified format"""
        
    async def handle_v2_request(self, request: Dict) -> Dict:
        """Convert v2 requests to unified format"""
```

## Configuration Examples

### Production Configuration (.env)
```bash
# Security
MCP_JWT_SECRET=<strong-random-secret>
MCP_JWT_ALGORITHM=HS256
MCP_JWT_EXPIRE_MINUTES=30

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/tldw
DATABASE_POOL_SIZE=20

# Redis
REDIS_URL=redis://localhost:6379/0

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Monitoring
METRICS_ENABLED=true
METRICS_PORT=9090
```

## Success Criteria

1. All security vulnerabilities fixed
2. >80% test coverage
3. <100ms p95 latency for tool execution
4. Zero-downtime deployments
5. Comprehensive monitoring and alerting
6. Complete API documentation
7. Security audit passed