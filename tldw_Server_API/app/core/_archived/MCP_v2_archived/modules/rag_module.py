"""
RAG Module for tldw MCP - Handles RAG operations, vector search, and retrieval
"""

from typing import Dict, Any, List, Optional
from loguru import logger

from ..modules.base import BaseModule, create_tool_definition, create_resource_definition
from ..schemas import ModuleConfig

# Import tldw's existing RAG functionality
try:
    from tldw_Server_API.app.core.RAG.rag_service.integration import RAGService
    from tldw_Server_API.app.core.Embeddings.embedding_providers import get_embeddings
    EmbeddingsService = None  # Will use get_embeddings function instead
except ImportError:
    logger.warning("RAG service imports not available - module will have limited functionality")
    RAGService = None
    get_embeddings = None
    EmbeddingsService = None


class RAGModule(BaseModule):
    """RAG Module for tldw
    
    Provides tools for:
    - Vector search using embeddings
    - Hybrid search (BM25 + vector)
    - Context retrieval for LLMs
    - Re-ranking of search results
    - Embedding generation and management
    """
    
    def __init__(self, config: ModuleConfig):
        super().__init__(config)
        self.rag_service: Optional[Any] = None
        self.embeddings_service: Optional[Any] = None
        self.collection_name = config.settings.get("collection_name", "tldw_media")
        self.embedding_model = config.settings.get("embedding_model", "all-MiniLM-L6-v2")
    
    async def on_initialize(self) -> None:
        """Initialize RAG module"""
        try:
            # Initialize RAG and embeddings services if available
            if RAGService:
                self.rag_service = RAGService()
            if EmbeddingsService:
                self.embeddings_service = EmbeddingsService(model_name=self.embedding_model)
            
            logger.info(f"RAG module initialized with model: {self.embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize RAG services: {e}")
            # Module can still work with limited functionality
    
    async def on_shutdown(self) -> None:
        """Shutdown RAG module"""
        if self.rag_service:
            # Cleanup RAG service if needed
            pass
        if self.embeddings_service:
            # Cleanup embeddings service if needed
            pass
        logger.info("RAG module shutdown")
    
    async def check_health(self) -> bool:
        """Check module health"""
        try:
            # Check if services are available
            if self.rag_service or self.embeddings_service:
                return True
            return False
        except Exception as e:
            logger.error(f"RAG module health check failed: {e}")
            return False
    
    async def get_tools(self) -> List[Dict[str, Any]]:
        """Get list of RAG tools"""
        return [
            create_tool_definition(
                name="vector_search",
                description="Perform vector similarity search using embeddings",
                parameters={
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query to embed and search"
                        },
                        "collection": {
                            "type": "string",
                            "description": "Collection name to search in",
                            "default": "tldw_media"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return",
                            "default": 5
                        },
                        "filter": {
                            "type": "object",
                            "description": "Optional metadata filters"
                        }
                    },
                    "required": ["query"]
                },
                department="rag"
            ),
            create_tool_definition(
                name="hybrid_search",
                description="Perform hybrid search combining BM25 and vector search",
                parameters={
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "bm25_weight": {
                            "type": "number",
                            "description": "Weight for BM25 search (0-1)",
                            "default": 0.5
                        },
                        "vector_weight": {
                            "type": "number",
                            "description": "Weight for vector search (0-1)",
                            "default": 0.5
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                },
                department="rag"
            ),
            create_tool_definition(
                name="get_context",
                description="Retrieve relevant context for a query",
                parameters={
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query to find context for"
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Maximum tokens to return",
                            "default": 2000
                        },
                        "include_metadata": {
                            "type": "boolean",
                            "description": "Include source metadata",
                            "default": True
                        }
                    },
                    "required": ["query"]
                },
                department="rag"
            ),
            create_tool_definition(
                name="rerank_results",
                description="Re-rank search results using a reranking model",
                parameters={
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Original query"
                        },
                        "results": {
                            "type": "array",
                            "description": "Search results to rerank"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return after reranking",
                            "default": 5
                        }
                    },
                    "required": ["query", "results"]
                },
                department="rag"
            ),
            create_tool_definition(
                name="generate_embedding",
                description="Generate embedding vector for text",
                parameters={
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to generate embedding for"
                        },
                        "model": {
                            "type": "string",
                            "description": "Embedding model to use",
                            "default": "all-MiniLM-L6-v2"
                        }
                    },
                    "required": ["text"]
                },
                department="rag"
            ),
            create_tool_definition(
                name="index_content",
                description="Index content for vector search",
                parameters={
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Content to index"
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Metadata to associate with content"
                        },
                        "collection": {
                            "type": "string",
                            "description": "Collection to index into",
                            "default": "tldw_media"
                        }
                    },
                    "required": ["content"]
                },
                department="rag"
            )
        ]
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute RAG tool"""
        logger.info(f"Executing RAG tool: {tool_name} with args: {arguments}")
        
        try:
            if tool_name == "vector_search":
                return await self._vector_search(arguments)
            elif tool_name == "hybrid_search":
                return await self._hybrid_search(arguments)
            elif tool_name == "get_context":
                return await self._get_context(arguments)
            elif tool_name == "rerank_results":
                return await self._rerank_results(arguments)
            elif tool_name == "generate_embedding":
                return await self._generate_embedding(arguments)
            elif tool_name == "index_content":
                return await self._index_content(arguments)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
        except Exception as e:
            logger.error(f"Error executing RAG tool {tool_name}: {e}")
            raise
    
    async def get_resources(self) -> List[Dict[str, Any]]:
        """Get RAG resources"""
        return [
            create_resource_definition(
                uri="rag://collections",
                name="Vector Collections",
                description="List of available vector collections",
                mime_type="application/json"
            ),
            create_resource_definition(
                uri="rag://models",
                name="Embedding Models",
                description="Available embedding models",
                mime_type="application/json"
            ),
            create_resource_definition(
                uri="rag://statistics",
                name="RAG Statistics",
                description="Statistics about indexed content",
                mime_type="application/json"
            )
        ]
    
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read RAG resource"""
        if uri == "rag://collections":
            # Return available collections
            return {
                "collections": [self.collection_name],
                "default": self.collection_name
            }
        elif uri == "rag://models":
            # Return available embedding models
            return {
                "models": [
                    "all-MiniLM-L6-v2",
                    "all-mpnet-base-v2",
                    "text-embedding-ada-002"
                ],
                "current": self.embedding_model
            }
        elif uri == "rag://statistics":
            # Return RAG statistics
            return {
                "indexed_documents": 0,  # Would need to query actual stats
                "embedding_model": self.embedding_model,
                "collection": self.collection_name
            }
        else:
            raise ValueError(f"Unknown resource: {uri}")
    
    # Tool implementation methods
    
    async def _vector_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Perform vector similarity search"""
        query = args["query"]
        collection = args.get("collection", self.collection_name)
        top_k = args.get("top_k", 5)
        filter_dict = args.get("filter")
        
        try:
            if not self.rag_service:
                return {
                    "success": False,
                    "error": "RAG service not available"
                }
            
            # Perform vector search
            results = await self.rag_service.vector_search(
                query=query,
                collection=collection,
                top_k=top_k,
                filter_dict=filter_dict
            )
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _hybrid_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Perform hybrid search"""
        query = args["query"]
        bm25_weight = args.get("bm25_weight", 0.5)
        vector_weight = args.get("vector_weight", 0.5)
        top_k = args.get("top_k", 10)
        
        try:
            if not self.rag_service:
                return {
                    "success": False,
                    "error": "RAG service not available"
                }
            
            # Perform hybrid search
            results = await self.rag_service.hybrid_search(
                query=query,
                bm25_weight=bm25_weight,
                vector_weight=vector_weight,
                top_k=top_k
            )
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _get_context(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant context"""
        query = args["query"]
        max_tokens = args.get("max_tokens", 2000)
        include_metadata = args.get("include_metadata", True)
        
        try:
            if not self.rag_service:
                return {
                    "success": False,
                    "error": "RAG service not available"
                }
            
            # Get relevant context
            context = await self.rag_service.get_context(
                query=query,
                max_tokens=max_tokens,
                include_metadata=include_metadata
            )
            
            return {
                "success": True,
                "query": query,
                "context": context
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _rerank_results(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Re-rank search results"""
        query = args["query"]
        results = args["results"]
        top_k = args.get("top_k", 5)
        
        try:
            if not self.rag_service:
                return {
                    "success": False,
                    "error": "RAG service not available"
                }
            
            # Rerank results
            reranked = await self.rag_service.rerank(
                query=query,
                results=results,
                top_k=top_k
            )
            
            return {
                "success": True,
                "query": query,
                "reranked_results": reranked,
                "count": len(reranked)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _generate_embedding(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Generate embedding for text"""
        text = args["text"]
        model = args.get("model", self.embedding_model)
        
        try:
            if not self.embeddings_service:
                return {
                    "success": False,
                    "error": "Embeddings service not available"
                }
            
            # Generate embedding
            embedding = await self.embeddings_service.generate_embedding(
                text=text,
                model=model
            )
            
            return {
                "success": True,
                "text": text[:100] + "..." if len(text) > 100 else text,
                "embedding": embedding,
                "model": model,
                "dimension": len(embedding)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _index_content(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Index content for vector search"""
        content = args["content"]
        metadata = args.get("metadata", {})
        collection = args.get("collection", self.collection_name)
        
        try:
            if not self.embeddings_service:
                return {
                    "success": False,
                    "error": "Embeddings service not available"
                }
            
            # Index the content
            doc_id = await self.embeddings_service.index_content(
                content=content,
                metadata=metadata,
                collection=collection
            )
            
            return {
                "success": True,
                "document_id": doc_id,
                "collection": collection,
                "message": "Content indexed successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }