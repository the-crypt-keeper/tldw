# Mock OpenAI API Server

A standalone mock server that implements the OpenAI API specification for testing purposes. This server can be used across different projects to test OpenAI API integrations without making actual API calls or incurring costs.

Alternative: https://github.com/StacklokLabs/mockllm

## Features

- ✅ **OpenAI-Compatible Endpoints**
  - `/v1/chat/completions` - Chat completions (streaming & non-streaming)
  - `/v1/embeddings` - Generate embeddings
  - `/v1/models` - List available models
  - `/v1/completions` - Legacy completions endpoint

- ✅ **Streaming Support**
  - Server-Sent Events (SSE) for chat completions
  - Configurable chunk delays
  - Proper streaming format with `data: [DONE]` termination

- ✅ **Flexible Response Management**
  - JSON/YAML configuration files
  - Pattern matching for requests
  - Template variables for dynamic responses
  - Response file caching

- ✅ **Authentication**
  - Mock Bearer token validation
  - Supports OpenAI's `sk-*` key format

- ✅ **Error Simulation**
  - Configurable error rates
  - Different error types
  - Useful for testing error handling

## Installation

### From Source

```bash
git clone https://github.com/yourusername/mock-openai-server.git
cd mock-openai-server
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"

# Or with test dependencies
pip install -e ".[test]"

# Or with all optional dependencies
pip install -e ".[all]"
```

### Using pip

```bash
pip install mock-openai-server

# Or with optional dependencies
pip install "mock-openai-server[test]"
```

### Using Docker

```bash
docker build -t mock-openai-server .
docker run -p 8080:8080 mock-openai-server
```

## Quick Start

1. **Start the server:**

```bash
# Using the installed package
mock-openai-server

# Or using Python module
python -m mock_openai.server

# With custom configuration
mock-openai-server --config config.json --port 8080
```

2. **Test with curl:**

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-mock-key-12345" \
  -d '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

3. **Use with OpenAI Python client:**

```python
import openai

# Point to mock server
openai.api_base = "http://localhost:8080/v1"
openai.api_key = "sk-mock-key-12345"

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## Configuration

### Configuration File Structure

Create a `config.json` file:

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8080,
    "cors_origins": ["*"],
    "log_requests": true,
    "simulate_errors": false,
    "error_rate": 0.0
  },
  "streaming": {
    "enabled": true,
    "chunk_delay_ms": 50,
    "tokens_per_chunk": 5
  },
  "responses": {
    "chat_completions": {
      "patterns": [
        {
          "match": {
            "model": "gpt-4",
            "content_regex": ".*test.*"
          },
          "response_file": "responses/chat/test_response.json",
          "priority": 10
        }
      ],
      "default": "responses/chat/default.json"
    },
    "embeddings": {
      "default": "responses/embeddings/default.json"
    }
  }
}
```

### Response Files

Response files support template variables:

```json
{
  "id": "$chat_id",
  "object": "chat.completion",
  "created": $timestamp,
  "model": "$model",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "This is a mock response"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 15,
    "total_tokens": 25
  }
}
```

### Pattern Matching

Match requests based on:
- **Model name**: `"model": "gpt-4"`
- **Content regex**: `"content_regex": ".*test.*"`
- **System prompt**: `"system_prompt": "You are a helpful assistant"`

### Environment Variables

Override configuration with environment variables:

```bash
export MOCK_OPENAI_HOST=0.0.0.0
export MOCK_OPENAI_PORT=8080
export MOCK_OPENAI_CORS_ORIGINS=http://localhost:3000,http://localhost:3001
export MOCK_OPENAI_LOG_REQUESTS=true
export MOCK_OPENAI_STREAMING_ENABLED=true
export MOCK_OPENAI_CHUNK_DELAY=100
```

## Usage Examples

### Basic Chat Completion

```python
import requests

response = requests.post(
    "http://localhost:8080/v1/chat/completions",
    headers={"Authorization": "Bearer sk-mock-key"},
    json={
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello!"}
        ]
    }
)
print(response.json())
```

### Streaming Response

```python
import requests
import json

response = requests.post(
    "http://localhost:8080/v1/chat/completions",
    headers={"Authorization": "Bearer sk-mock-key"},
    json={
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Tell me a story"}],
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line and line != b'data: [DONE]':
        data = json.loads(line.decode('utf-8').replace('data: ', ''))
        if 'choices' in data:
            delta = data['choices'][0].get('delta', {})
            if 'content' in delta:
                print(delta['content'], end='', flush=True)
```

### Embeddings

```python
response = requests.post(
    "http://localhost:8080/v1/embeddings",
    headers={"Authorization": "Bearer sk-mock-key"},
    json={
        "model": "text-embedding-ada-002",
        "input": "Sample text for embedding"
    }
)
embedding = response.json()['data'][0]['embedding']
print(f"Embedding dimension: {len(embedding)}")
```

## Testing

Run the test suite:

```bash
# Install test dependencies
pip install -e ".[test]"

# Run all tests
pytest

# Run with coverage (already configured in pyproject.toml)
pytest --cov

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run tests with verbose output
pytest -v

# Run specific test file
pytest tests/test_server.py

# Generate HTML coverage report
pytest --cov-report=html
# Open htmlcov/index.html in your browser
```

## Docker Support

### Build and Run with Docker

```bash
# Build image
docker build -t mock-openai-server .

# Run container
docker run -p 8080:8080 \
  -v $(pwd)/config.json:/app/config.json \
  -v $(pwd)/responses:/app/responses \
  mock-openai-server

# With environment variables
docker run -p 8080:8080 \
  -e MOCK_OPENAI_PORT=8080 \
  -e MOCK_OPENAI_LOG_REQUESTS=true \
  mock-openai-server
```

### Docker Compose

```yaml
version: '3.8'

services:
  mock-openai:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./config.json:/app/config.json
      - ./responses:/app/responses
    environment:
      - MOCK_OPENAI_LOG_REQUESTS=true
      - MOCK_OPENAI_STREAMING_ENABLED=true
```

## Advanced Features

### Custom Response Logic

Extend the `ResponseManager` class for custom logic:

```python
from mock_openai.responses import ResponseManager

class CustomResponseManager(ResponseManager):
    def generate_chat_response(self, request_data, response_file=None):
        # Custom logic here
        if "special" in str(request_data):
            return self.special_response()
        return super().generate_chat_response(request_data, response_file)
```

### Error Simulation

Enable error simulation for testing error handling:

```json
{
  "server": {
    "simulate_errors": true,
    "error_rate": 0.1
  }
}
```

### Multiple Response Patterns

Define multiple patterns with priorities:

```json
{
  "responses": {
    "chat_completions": {
      "patterns": [
        {
          "match": {"model": "gpt-4", "content_regex": "urgent"},
          "response_file": "responses/urgent.json",
          "priority": 100
        },
        {
          "match": {"model": "gpt-4"},
          "response_file": "responses/gpt4.json",
          "priority": 50
        },
        {
          "match": {"content_regex": ".*"},
          "response_file": "responses/generic.json",
          "priority": 1
        }
      ]
    }
  }
}
```

## API Compatibility

This mock server implements the OpenAI API v1 specification and is compatible with:

- OpenAI Python SDK (v0.27.0+)
- OpenAI Node.js SDK
- LangChain
- Any OpenAI-compatible client library

## Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/mock-openai-server.git
cd mock-openai-server

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[all]"
```

### Code Quality Tools

```bash
# Format code with black
black mock_openai tests

# Sort imports with isort
isort mock_openai tests

# Type checking with mypy
mypy mock_openai

# Linting with ruff
ruff check mock_openai tests

# Run all quality checks
black mock_openai tests && isort mock_openai tests && mypy mock_openai && ruff check mock_openai tests
```

### Pre-commit Hooks (Optional)

```bash
# Install pre-commit
pip install pre-commit

# Set up the git hook scripts
pre-commit install

# Run against all files
pre-commit run --all-files
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Contributing Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and ensure they pass (`pytest`)
5. Run code quality tools (`black`, `isort`, `mypy`, `ruff`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Inspired by the OpenAI API specification
- Built with FastAPI for high performance
- Uses Pydantic for robust data validation