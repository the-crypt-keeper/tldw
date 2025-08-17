# API Documentation

## Overview

API uses FastAPI framework.
Designed to be simple and easy to use.
Generative endpoints follow openai API spec where possible.
See [API Design](API_Design.md) for more details.

### URLs
- **URLs**
    - Main page: http://127.0.0.1:8000
    - API Documentation page: http://127.0.0.1:8000/docs




### Endpoints

#### RAG (Retrieval-Augmented Generation) - `/api/v1/rag`

The RAG module provides advanced search and question-answering capabilities across your content. It achieves 100% test coverage and is production-ready.

##### Search Endpoints

- **`POST /api/v1/rag/search`** - Simple search with hybrid (BM25 + semantic) capabilities
  - Search across multiple databases (media, notes, characters, chat history)
  - Support for keyword filtering
  - Configurable search types (hybrid, semantic, fulltext)
  
- **`POST /api/v1/rag/search/advanced`** - Advanced search with full configuration control
  - Multiple search strategies (vanilla, query_fusion, HyDE)
  - Fine-tuned hybrid weights and similarity thresholds
  - Metadata filters and date range queries
  - Reranking and score inclusion options

##### Agent Endpoints

- **`POST /api/v1/rag/agent`** - Simple Q&A agent with automatic context retrieval
  - Conversational interface with memory
  - Automatic search across specified databases
  - Returns response with source citations
  
- **`POST /api/v1/rag/agent/advanced`** - Research agent with advanced capabilities
  - Multiple modes (RAG, research)
  - Tool support (web_search, reasoning, calculator, code_execution)
  - Custom system prompts
  - Streaming support via Server-Sent Events
  - Detailed statistics and metrics

##### Health Check

- **`GET /api/v1/rag/health`** - Service health status
  - Monitor RAG service availability
  - Used for load balancer health checks

For comprehensive documentation, see:
- [RAG API Consumer Guide](RAG-API-Guide.md) - Complete API reference with examples
- [RAG Developer Guide](../Development/RAG-Developer-Guide.md) - Architecture and implementation details


