# CLAUDE.md - tldw_server Project Guide

This file provides comprehensive guidance to Claude Code (claude.ai/code) when working with the tldw_server codebase.

## Project Overview

**tldw_server** (Too Long; Didn't Watch Server) is an ambitious open-source project building a comprehensive research assistant and media analysis platform. It's designed to help users ingest, transcribe, analyze, and interact with various media formats through both a powerful API and a web interface.

### Project Vision
The long-term goal is to create something akin to "The Young Lady's Illustrated Primer" from Neal Stephenson's "The Diamond Age" - a personal knowledge assistant that helps users learn and research at their own pace. While acknowledging the inherent difficulties in replicating such a device, this project serves as a practical step toward that vision.

### Current Status (v0.1.0)
The project has transitioned from a Gradio-based UI to a robust FastAPI backend, with the Gradio application remaining available but deprecated. The focus is now on building a stable, scalable API that can support various frontend implementations.

## Repository Structure

```
tldw_server/
├── tldw_Server_API/          # Main API server implementation
│   ├── app/                  # FastAPI application
│   │   ├── api/v1/          # API endpoints and schemas
│   │   ├── core/            # Core business logic
│   │   ├── services/        # Background services
│   │   └── main.py          # FastAPI entry point
│   ├── tests/               # Comprehensive test suite
│   ├── Config_Files/        # Configuration templates
│   ├── chrome-extension/    # Browser extension (WIP)
│   └── requirements.txt     # Python dependencies
├── Docs/                    # Project documentation
│   ├── API-related/         # API design and notes
│   ├── Code_Documentation/  # Technical documentation
│   ├── Design/              # Feature design documents
│   ├── Prompts/             # LLM prompt library
│   └── User_Guides/         # User documentation
├── Helper_Scripts/          # Utility and installation scripts
│   ├── Installer_Scripts/   # Platform-specific installers
│   ├── Dockerfiles/         # Docker configurations
│   └── Prompts/             # Additional prompts
├── user_databases/          # User data storage (gitignored)
├── pyproject.toml          # Project configuration
├── README.md               # Project README
├── LICENSE.txt             # Dual license (AGPL/Commercial)
└── Project_Guidelines.md   # Development philosophy
```

## Core Features

### Implemented Features
1. **Media Ingestion & Processing**
   - Supports: Video, Audio, PDF, EPUB, DOCX, HTML, Markdown, XML, MediaWiki dumps
   - Uses yt-dlp for video/audio downloads from 1000+ sites
   - Automatic metadata extraction and storage

2. **Transcription & Analysis**
   - Audio/video transcription via faster_whisper
   - Content analysis using multiple LLM providers
   - Chunked processing for long-form content
   - Support for diarization (speaker identification)

3. **Search & Retrieval (RAG)**
   - Full-text search using SQLite FTS5
   - Vector embeddings with ChromaDB
   - BM25 + vector search + re-ranking pipeline
   - Contextual retrieval for improved accuracy

4. **Chat & Interaction**
   - OpenAI-compatible chat API (`/chat/completions`)
   - Support for 16+ LLM providers (commercial & local)
   - Character card support (SillyTavern compatible)
   - Chat history management and search

5. **Knowledge Management**
   - Note-taking system (NotebookLM-style)
   - Prompt library with import/export
   - Tagging and categorization
   - Soft delete with recovery options

6. **API Providers Supported**
   - **Commercial**: OpenAI, Anthropic, Cohere, DeepSeek, Google, Groq, HuggingFace, Mistral, OpenRouter
   - **Local**: Llama.cpp, Kobold.cpp, Oobabooga, TabbyAPI, vLLM, Ollama, Aphrodite, Custom OpenAI-compatible

### Work-in-Progress Features
- Embeddings API endpoint
- Enhanced character chat functionality
- Research tools (Arxiv, Semantic Scholar integration)
- TTS (Text-to-Speech) support
- Writing assistance tools
- Browser extension for web content capture
- Sync server for multi-device support
- **Evaluation Module**: Currently undergoing unification (combining OpenAI-compatible and tldw-specific implementations)

## Technical Architecture

### Database Design
- **SQLite Databases**:
  - `Media_DB_v2`: Main content database with FTS5
  - `ChaChaNotes_DB`: Character cards and chats
  - `Prompts_DB`: Prompt management
  - Implements soft deletes, versioning, and sync logging

- **Vector Storage**:
  - ChromaDB for embeddings
  - Supports multiple embedding models

### API Design
- RESTful API following OpenAPI 3.0 specification
- Consistent endpoint naming: `/api/v1/{resource}/{action}`
- Pydantic models for request/response validation
- Comprehensive error handling with meaningful messages

### Key Technologies
- **Backend**: FastAPI, SQLite, ChromaDB
- **ML/AI**: faster_whisper, sentence-transformers, various LLM SDKs
- **Audio/Video**: ffmpeg, yt-dlp, pyannote
- **Document Processing**: pymupdf, docling, ebooklib, pandoc
- **Testing**: pytest, httpx
- **Logging**: loguru

### Key Architectural Components

- **Media Processing Pipeline**: 
  - Location: `/app/core/Ingestion_Media_Processing/`
  - Handles: Video, Audio, PDF, EPUB, DOCX, HTML, Markdown, XML
  - Features: Chunking, metadata extraction, format conversion
  
- **LLM Integration**: 
  - Location: `/app/core/LLM_Calls/`
  - Supports streaming responses
  - Unified interface for all providers
  
- **Embeddings System**:
  - Location: `/app/core/Embeddings/`
  - Features: Job queue, worker orchestration, ChromaDB integration
  - Supports batch processing and async operations
  
- **Authentication**: 
  - Location: `/app/core/AuthNZ/`
  - JWT-based authentication (WIP)
  - User management system
  
- **RAG Service**:
  - Location: `/app/core/RAG/rag_service/`
  - Implements BM25 + vector search + re-ranking
  - Configurable retrieval strategies

## Development Guidelines

### Code Style
- Follow PEP 8 for Python code
- Use type hints for function parameters and returns
- Implement comprehensive docstrings for modules, classes, and functions
- Prefer async/await for I/O operations

### Adding New Features
1. **Design First**: Create a design document in `/Docs/Design/`
2. **Core Implementation**: Add business logic to `/app/core/{feature}/`
3. **API Endpoint**: Create endpoint in `/app/api/v1/endpoints/`
4. **Schemas**: Define Pydantic models in `/app/api/v1/schemas/`
5. **Tests**: Write comprehensive tests in `/tests/{feature}/`
6. **Documentation**: Update relevant documentation

### Common Development Patterns
- **Pydantic Models**: Use for all API request/response validation
- **Dependency Injection**: For database connections and service instances
- **Background Tasks**: Use FastAPI's background tasks or services layer
- **Streaming**: Support streaming responses where applicable
- **Error Responses**: Follow consistent HTTP status codes and error formats
- **Async/Await**: Use async patterns for I/O operations

### Important Implementation Notes
- **Logging**: Use Loguru throughout (`from loguru import logger`)
- **Error Handling**: All endpoints should handle errors gracefully with meaningful messages
- **Rate Limiting**: Implemented via slowapi
- **Database Operations**: Always use the abstraction layer in `/app/core/DB_Management/`
- **File Handling**: Process uploads through the ingestion pipeline, not directly in endpoints
- **API Keys**: Managed through config files and validated in schemas

### Testing Requirements
- Write unit tests for all new functions
- Include integration tests for API endpoints
- Use pytest fixtures for common test data
- Mock external services (LLMs, APIs)
- Aim for >80% code coverage

### Testing Strategy
- **Test Structure**: Tests mirror source code structure
- **Test Types**:
  - Unit tests for individual components
  - Integration tests for API endpoints
  - Property-based tests for complex logic
- **Fixtures**: Use pytest fixtures for database and dependency injection
- **Mocking**: Mock external services (LLMs, transcription services)
- **Test Markers**: `unit`, `integration`, `external_api`, `local_llm_service`

### Error Handling
- Use custom exceptions in `/app/core/exceptions.py`
- Return appropriate HTTP status codes
- Provide meaningful error messages
- Log errors with context using loguru

### Security Best Practices
- Validate all user input
- Use parameterized queries for database operations
- Never log sensitive information (API keys, passwords)
- Implement rate limiting for API endpoints
- Validate file uploads (type, size, content)
- **Input Validation**: Always validate and sanitize user input
- **File Uploads**: Validate file types and sizes
- **SQL Injection**: Use parameterized queries
- **CORS**: Configured in `main.py`, adjust for production

## Configuration

### Configuration Files
- `config.txt`: Main configuration (API keys, settings)
- `mediawiki_import_config.yaml`: MediaWiki import settings
- Environment variables override file settings
- Database paths configurable (default: `./Databases/`)

### Required Setup
1. **Dependencies**: `pip install -r tldw_Server_API/requirements.txt`
2. **FFmpeg**: Required for audio/video processing
3. **API Keys**: Add to config.txt for LLM providers
4. **Optional**: CUDA for faster transcription

## Common Tasks

### Starting the Server
```bash
cd tldw_server
python -m uvicorn tldw_Server_API.app.main:app --reload
# API docs available at http://127.0.0.1:8000/docs
```

### Running Tests
```bash
# All tests
python -m pytest -v

# Specific module
python -m pytest tests/Media_Ingestion_Modification/ -v

# With coverage
python -m pytest --cov=tldw_Server_API --cov-report=html

# Run tests with markers
python -m pytest -m "unit" -v  # Unit tests only
python -m pytest -m "integration" -v  # Integration tests only
```

### Database Operations
```python
# Use the MediaDatabase class for all operations
from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import MediaDatabase

db = MediaDatabase(db_path="path/to/media.db", client_id="api_client")
# Always use context managers for transactions
```

### Adding a New LLM Provider
1. Add provider configuration to `config.txt`
2. Implement provider in `/app/core/LLM_Calls/LLM_API_Calls.py`
3. Add to provider list in chat schemas
4. Write tests for the new provider

## Performance & Deployment

### Performance Optimization
- **Database**: Use indexes, FTS5 for search
- **Chunking**: Process large files in chunks
- **Caching**: Implement caching for expensive operations
- **Connection Pooling**: For database connections
- **Async Operations**: For I/O-bound tasks

### Deployment Considerations
- **Docker**: Dockerfiles available in tldw_Server_API/Dockerfiles/
- **Environment**: Supports Linux, macOS, Windows
- **Dependencies**: CUDA support optional for transcription
- **Backup**: Built-in backup management for databases
- **CORS**: Configured in `main.py`, adjust for production

## Debugging Tips

### Common Issues
1. **Import Errors**: Check PYTHONPATH and virtual environment
2. **Database Locks**: Ensure proper connection management
3. **Transcription Failures**: Verify ffmpeg installation and CUDA setup
4. **API Key Errors**: Check config.txt formatting

### Logging
- Logs use loguru with color-coded output
- Debug level includes detailed request/response info
- Check logs for stack traces on errors

## Project Philosophy

The project follows these core principles (from Project_Guidelines.md):
1. Keep the project actively developed with clear progress
2. Be respectful to all contributors and users
3. Remain open to criticism and new ideas
4. Balance expertise with newcomer perspectives
5. Be kind and answer questions
6. Acknowledge contributions
7. Compensate significant contributions when possible

## Important Notes

### Licensing
- Dual-licensed: AGPL-3.0 for open source, commercial license available
- Respect the license terms when contributing

### Privacy & Security
- Designed for local/self-hosted deployment
- No telemetry or data collection
- Users own and control their data

### Contributing
- Follow the existing code patterns
- Write tests for new features
- Update documentation
- Be respectful in discussions

## Quick Reference

### Key Endpoints
- `POST /api/v1/media/process` - Ingest and process media
- `POST /api/v1/chat/completions` - OpenAI-compatible chat
- `GET /api/v1/media/search` - Search ingested content
- `POST /api/v1/notes/create` - Create a note
- `GET /api/v1/prompts/list` - List prompts
- `POST /api/v1/evaluations` - Create evaluation definitions (OpenAI-compatible)
- `POST /api/v1/evaluations/geval` - G-Eval summarization evaluation
- `POST /api/v1/evaluations/rag` - RAG system evaluation

### Environment Variables
- `TLDW_CONFIG_PATH`: Path to config.txt
- `OPENAI_API_KEY`: OpenAI API key (can be in config.txt)
- `ANTHROPIC_API_KEY`: Anthropic API key (can be in config.txt)
- `DATABASE_PATH`: Override default database location

### Useful Commands
```bash
# Install with all optional dependencies
pip install -e ".[all]"

# Run specific test markers
python -m pytest -m "unit" -v
python -m pytest -m "integration" -v

# Check code coverage
python -m pytest --cov=tldw_Server_API --cov-report=term-missing

# Format code (if black is installed)
black tldw_Server_API/

# Type checking (if mypy is installed)
mypy tldw_Server_API/
```

---

This guide is maintained to help Claude Code understand the project structure, conventions, and best practices. When in doubt, look at existing code patterns and tests for guidance.

# Development Guidelines

## Philosophy

### Core Beliefs

- **Incremental progress over big bangs** - Small changes that compile and pass tests
- **Learning from existing code** - Study and plan before implementing
- **Pragmatic over dogmatic** - Adapt to project reality
- **Clear intent over clever code** - Be boring and obvious

### Simplicity Means

- Single responsibility per function/class
- Avoid premature abstractions
- No clever tricks - choose the boring solution
- If you need to explain it, it's too complex

## Process

### 1. Planning & Staging

Break complex work into 3-5 stages. Document in `IMPLEMENTATION_PLAN.md`:

```markdown
## Stage N: [Name]
**Goal**: [Specific deliverable]
**Success Criteria**: [Testable outcomes]
**Tests**: [Specific test cases]
**Status**: [Not Started|In Progress|Complete]
```
- Update status as you progress
- Remove file when all stages are done

### 2. Implementation Flow

1. **Understand** - Study existing patterns in codebase
2. **Test** - Write test first (red)
3. **Implement** - Minimal code to pass (green)
4. **Refactor** - Clean up with tests passing
5. **Commit** - With clear message linking to plan

### 3. When Stuck (After 3 Attempts)

**CRITICAL**: Maximum 3 attempts per issue, then STOP.

1. **Document what failed**:
   - What you tried
   - Specific error messages
   - Why you think it failed

2. **Research alternatives**:
   - Find 2-3 similar implementations
   - Note different approaches used

3. **Question fundamentals**:
   - Is this the right abstraction level?
   - Can this be split into smaller problems?
   - Is there a simpler approach entirely?

4. **Try different angle**:
   - Different library/framework feature?
   - Different architectural pattern?
   - Remove abstraction instead of adding?

## Technical Standards

### Architecture Principles

- **Composition over inheritance** - Use dependency injection
- **Interfaces over singletons** - Enable testing and flexibility
- **Explicit over implicit** - Clear data flow and dependencies
- **Test-driven when possible** - Never disable tests, fix them

### Code Quality

- **Every commit must**:
  - Compile successfully
  - Pass all existing tests
  - Include tests for new functionality
  - Follow project formatting/linting

- **Before committing**:
  - Run formatters/linters
  - Self-review changes
  - Ensure commit message explains "why"

### Error Handling

- Fail fast with descriptive messages
- Include context for debugging
- Handle errors at appropriate level
- Never silently swallow exceptions

## Decision Framework

When multiple valid approaches exist, choose based on:

1. **Testability** - Can I easily test this?
2. **Readability** - Will someone understand this in 6 months?
3. **Consistency** - Does this match project patterns?
4. **Simplicity** - Is this the simplest solution that works?
5. **Reversibility** - How hard to change later?

## Project Integration

### Learning the Codebase

- Find 3 similar features/components
- Identify common patterns and conventions
- Use same libraries/utilities when possible
- Follow existing test patterns

### Tooling

- Use project's existing build system
- Use project's test framework
- Use project's formatter/linter settings
- Don't introduce new tools without strong justification

## Quality Gates

### Definition of Done

- [ ] Tests written and passing
- [ ] Code follows project conventions
- [ ] No linter/formatter warnings
- [ ] Commit messages are clear
- [ ] Implementation matches plan
- [ ] No TODOs without issue numbers

### Test Guidelines

- Test behavior, not implementation
- One assertion per test when possible
- Clear test names describing scenario
- Use existing test utilities/helpers
- Tests should be deterministic

## Important Reminders

**NEVER**:
- Use `--no-verify` to bypass commit hooks
- Disable tests instead of fixing them
- Commit code that doesn't compile
- Make assumptions - verify with existing code

**ALWAYS**:
- Commit working code incrementally
- Update plan documentation as you go
- Learn from existing implementations
- Stop after 3 failed attempts and reassess