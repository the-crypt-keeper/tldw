# Dockerfile to test GitHub Actions workflow
FROM ubuntu:22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    git \
    build-essential \
    portaudio19-dev \
    python3-all-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Add deadsnakes PPA for multiple Python versions
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update

# Install Python versions
RUN apt-get install -y \
    python3.10 python3.10-dev python3.10-venv python3.10-distutils \
    python3.11 python3.11-dev python3.11-venv python3.11-distutils \
    python3.12 python3.12-dev python3.12-venv python3.12-distutils \
    && rm -rf /var/lib/apt/lists/*

# Install pip for each Python version
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.10 get-pip.py && \
    python3.11 get-pip.py && \
    python3.12 get-pip.py && \
    rm get-pip.py

# Set working directory
WORKDIR /app

# Copy the project files
COPY . .

# Set environment variables
ENV PYTHONPATH=/app/tldw_Server_API
ENV OPENAI_API_KEY=test-key
ENV ANTHROPIC_API_KEY=test-key