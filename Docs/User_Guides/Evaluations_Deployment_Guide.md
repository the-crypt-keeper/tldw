# Evaluations Module - Production Deployment Guide

## Overview
This guide covers the deployment of the tldw_server Evaluations module for production environments. Follow these steps to ensure a secure, scalable, and monitored deployment.

## Prerequisites

- Python 3.9+
- PostgreSQL or SQLite database
- Redis (optional, for caching)
- Monitoring infrastructure (Prometheus/Grafana recommended)
- SSL certificates for HTTPS

## Security Configuration

### 1. API Key Configuration

**Never use default keys in production!**

Generate secure API keys:
```bash
# Generate a secure API key
openssl rand -hex 32
```

Set environment variables:
```bash
# Single-user mode
export AUTH_MODE="single_user"
export API_BEARER="your-secure-api-key-here"

# Multi-user mode
export AUTH_MODE="multi_user"
export JWT_SECRET_KEY="your-jwt-secret-here"
```

### 2. LLM Provider Keys

Configure at least one LLM provider:
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
export COHERE_API_KEY="..."
```

### 3. Database Configuration

```bash
# Production database path
export EVALUATIONS_DB_PATH="/var/lib/tldw/evaluations.db"

# Ensure proper permissions
chmod 600 /var/lib/tldw/evaluations.db
```

### 4. Session Security

```bash
# Generate session encryption key
export SESSION_ENCRYPTION_KEY=$(openssl rand -hex 32)

# Force HTTPS
export FORCE_HTTPS=true

# Configure CORS
export CORS_ORIGINS="https://yourdomain.com,https://app.yourdomain.com"
```

## Performance Configuration

### 1. Rate Limiting

```bash
# Configure rate limits
export RATE_LIMIT_PER_MINUTE=30  # For evaluations
export RATE_LIMIT_BURST=5        # Burst allowance
```

### 2. Connection Pooling

```bash
# Database connections
export DB_POOL_SIZE=20
export DB_MAX_OVERFLOW=10

# Redis configuration
export REDIS_URL="redis://localhost:6379/0"
export REDIS_MAX_CONNECTIONS=50
```

### 3. Worker Configuration

```bash
# Async workers
export EVALUATION_WORKERS=4
export MAX_CONCURRENT_EVALUATIONS=10
```

## Monitoring Setup

### 1. Enable Metrics

```bash
export METRICS_ENABLED=true
export METRICS_PORT=9090
```

### 2. Logging Configuration

```bash
export LOG_LEVEL=INFO  # Use WARNING for production
export LOG_FILE=/var/log/tldw/evaluations.log
export LOG_ROTATION="1 day"
export LOG_RETENTION="30 days"
```

### 3. Health Checks

The module provides health check endpoints:
- `/api/v1/evaluations/health` - Basic health check
- `/api/v1/evaluations/health/config` - Configuration validation
- `/api/v1/evaluations/metrics` - Prometheus metrics

## Deployment Steps

### 1. Pre-Deployment Checklist

Run configuration validation:
```python
python -c "
from tldw_Server_API.app.core.Evaluations.config_validator import check_production_readiness
if not check_production_readiness():
    print('Configuration issues detected! Fix before deployment.')
    exit(1)
print('Configuration validated successfully!')
"
```

### 2. Database Migration

```bash
# Backup existing database
cp $EVALUATIONS_DB_PATH $EVALUATIONS_DB_PATH.backup

# Run migrations
python -m tldw_Server_API.app.core.DB_Management.migrations \
    --database evaluations \
    --apply
```

### 3. Start the Service

Using systemd (recommended):

Create `/etc/systemd/system/tldw-evaluations.service`:
```ini
[Unit]
Description=TLDW Evaluations Service
After=network.target

[Service]
Type=simple
User=tldw
Group=tldw
WorkingDirectory=/opt/tldw_server
Environment="PATH=/opt/tldw_server/venv/bin"
EnvironmentFile=/etc/tldw/evaluations.env
ExecStart=/opt/tldw_server/venv/bin/uvicorn \
    tldw_Server_API.app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable tldw-evaluations
sudo systemctl start tldw-evaluations
```

### 4. Configure Reverse Proxy (Nginx)

```nginx
server {
    listen 443 ssl http2;
    server_name evaluations.yourdomain.com;

    ssl_certificate /etc/ssl/certs/your-cert.pem;
    ssl_certificate_key /etc/ssl/private/your-key.pem;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    location /api/v1/evaluations/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts for long evaluations
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
```

## Post-Deployment Verification

### 1. Check Service Health

```bash
# Basic health check
curl https://evaluations.yourdomain.com/api/v1/evaluations/health

# Configuration validation
curl https://evaluations.yourdomain.com/api/v1/evaluations/health/config

# Test authentication
curl -H "Authorization: Bearer YOUR_API_KEY" \
    https://evaluations.yourdomain.com/api/v1/evals
```

### 2. Monitor Metrics

Set up Prometheus to scrape:
```yaml
scrape_configs:
  - job_name: 'tldw-evaluations'
    scrape_interval: 30s
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/api/v1/evaluations/metrics'
```

### 3. Set Up Alerts

Example alert rules for Prometheus:
```yaml
groups:
  - name: evaluations
    rules:
      - alert: HighErrorRate
        expr: rate(evaluation_errors_total[5m]) > 0.1
        for: 5m
        annotations:
          summary: High evaluation error rate
          
      - alert: SlowEvaluations
        expr: histogram_quantile(0.95, evaluation_duration_seconds) > 30
        for: 10m
        annotations:
          summary: Evaluations taking too long
```

## Backup and Recovery

### 1. Automated Backups

Create backup script `/opt/tldw/backup-evaluations.sh`:
```bash
#!/bin/bash
BACKUP_DIR="/backup/evaluations"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Backup database
sqlite3 $EVALUATIONS_DB_PATH ".backup $BACKUP_DIR/evaluations_$TIMESTAMP.db"

# Backup audit logs
cp /var/lib/tldw/evaluation_audit.db $BACKUP_DIR/audit_$TIMESTAMP.db

# Keep only last 30 days
find $BACKUP_DIR -type f -mtime +30 -delete
```

Add to crontab:
```bash
0 2 * * * /opt/tldw/backup-evaluations.sh
```

### 2. Recovery Procedure

```bash
# Stop service
sudo systemctl stop tldw-evaluations

# Restore database
cp /backup/evaluations/evaluations_TIMESTAMP.db $EVALUATIONS_DB_PATH

# Restart service
sudo systemctl start tldw-evaluations
```

## Security Best Practices

1. **API Key Rotation**: Rotate API keys every 90 days
2. **Audit Logs**: Review audit logs weekly for suspicious activity
3. **Rate Limiting**: Adjust limits based on usage patterns
4. **Updates**: Keep dependencies updated monthly
5. **Monitoring**: Set up alerts for authentication failures
6. **Encryption**: Use TLS 1.2+ for all connections
7. **Firewall**: Restrict database ports to localhost only

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| "Authentication failed" | Check API_BEARER environment variable |
| "Database locked" | Check file permissions and concurrent access |
| "Rate limit exceeded" | Increase RATE_LIMIT_PER_MINUTE |
| "Evaluation timeout" | Increase worker timeout settings |
| "High memory usage" | Reduce MAX_CONCURRENT_EVALUATIONS |

### Debug Mode

Enable debug logging temporarily:
```bash
export LOG_LEVEL=DEBUG
export DEBUG_SQL=true
sudo systemctl restart tldw-evaluations
```

**Remember to disable debug mode after troubleshooting!**

## Maintenance

### Weekly Tasks
- Review audit logs
- Check disk space
- Monitor API costs
- Review error logs

### Monthly Tasks
- Update dependencies
- Rotate API keys
- Review and optimize database
- Update documentation

### Quarterly Tasks
- Security audit
- Performance review
- Capacity planning
- Disaster recovery test

## Support

For issues or questions:
1. Check logs: `/var/log/tldw/evaluations.log`
2. Run health check: `/api/v1/evaluations/health/config`
3. Review metrics: `/api/v1/evaluations/metrics`
4. Contact support with audit log excerpts

## Appendix: Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| AUTH_MODE | Yes | single_user | Authentication mode |
| API_BEARER | Yes* | - | API key for single-user mode |
| JWT_SECRET_KEY | Yes* | - | JWT secret for multi-user mode |
| OPENAI_API_KEY | No | - | OpenAI API key |
| EVALUATIONS_DB_PATH | No | ./Databases/evaluations.db | Database path |
| RATE_LIMIT_PER_MINUTE | No | 60 | Rate limit per minute |
| METRICS_ENABLED | No | false | Enable metrics collection |
| LOG_LEVEL | No | INFO | Logging level |
| FORCE_HTTPS | No | false | Force HTTPS connections |
| CORS_ORIGINS | No | * | Allowed CORS origins |

*Required depending on AUTH_MODE setting