# Multi-stage Dockerfile for MCP Unified Module
# Optimized for production deployment with security best practices

# Stage 1: Builder
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements
COPY requirements.txt /tmp/
COPY requirements-mcp.txt /tmp/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir -r /tmp/requirements-mcp.txt

# Stage 2: Runtime
FROM python:3.11-slim

# Security: Run as non-root user
RUN useradd -m -u 1000 -s /bin/bash mcp && \
    mkdir -p /app /data/databases /data/logs /data/cache && \
    chown -R mcp:mcp /app /data

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=mcp:mcp tldw_Server_API/app /app

# Create necessary directories
RUN mkdir -p \
    /data/databases \
    /data/logs \
    /data/cache \
    /data/media \
    /data/transcripts

# Environment variables (defaults - override in deployment)
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    MCP_LOG_LEVEL=INFO \
    MCP_LOG_FILE=/data/logs/mcp.log \
    MCP_AUDIT_LOG_FILE=/data/logs/audit.log \
    MCP_DATABASE_URL=sqlite+aiosqlite:////data/databases/mcp_unified.db \
    MCP_CORS_ORIGINS=http://localhost:3000 \
    MCP_METRICS_ENABLED=true \
    MCP_METRICS_PORT=9090 \
    MCP_HEALTH_PATH=/health

# Security: Don't expose secrets in image
# These MUST be provided at runtime
# ENV MCP_JWT_SECRET=
# ENV MCP_API_KEY_SALT=

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/mcp/health || exit 1

# Switch to non-root user
USER mcp

# Entry point script
COPY --chown=mcp:mcp docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["uvicorn", "core.MCP_unified.main:app", "--host", "0.0.0.0", "--port", "8000"]