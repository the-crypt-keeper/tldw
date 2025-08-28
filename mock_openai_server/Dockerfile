# Dockerfile for Mock OpenAI API Server

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY mock_openai/ ./mock_openai/
COPY responses/ ./responses/
COPY config.json .
COPY setup.py .

# Install the package
RUN pip install -e .

# Create a non-root user
RUN useradd -m -u 1000 mockserver && chown -R mockserver:mockserver /app
USER mockserver

# Expose the default port
EXPOSE 8080

# Default command
CMD ["python", "-m", "mock_openai.server"]