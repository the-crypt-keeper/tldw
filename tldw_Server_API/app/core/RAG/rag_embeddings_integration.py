"""
RAG-Embeddings Integration Module

This module provides proper integration between the RAG service and the embeddings system,
ensuring that all embeddings are created using the actual production embeddings service
rather than any mocked or default implementations.

NO MOCKING - This is production code that uses real embeddings throughout.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import asyncio
import numpy as np
from loguru import logger

import chromadb
from chromadb.config import Settings
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

# Import the actual embeddings service
from tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create import (
    create_embeddings_batch,
    create_embeddings_batch_async,
    create_embedding,
    get_embedding_config,
    EmbeddingConfigSchema
)

# Import the embeddings wrapper for simplified interface
from tldw_Server_API.app.core.RAG.RAG_Search.simplified.embeddings_wrapper import (
    EmbeddingsServiceWrapper,
    create_embeddings_service
)

from tldw_Server_API.app.core.RAG.rag_service.types import (
    DataSource, Document, SearchResult, RetrievalError
)


class ProductionEmbeddingFunction(EmbeddingFunction):
    """
    ChromaDB-compatible embedding function that uses the actual production embeddings service.
    
    This ensures ChromaDB uses our real embeddings instead of its default implementation.
    NO MOCKING - This uses the actual embeddings service from Embeddings_Create.
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model_id: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        use_async: bool = False
    ):
        """
        Initialize the production embedding function.
        
        Args:
            provider: Embedding provider (openai, huggingface, cohere, etc.)
            model_id: Model identifier
            api_key: Optional API key (uses env var if not provided)
            api_url: Optional API URL for custom endpoints
            use_async: Whether to use async embedding generation
        """
        self.provider = provider
        self.model_id = model_id
        self.api_key = api_key
        self.api_url = api_url
        self.use_async = use_async
        
        # Get embedding configuration
        self.config = self._build_config()
        
        logger.info(f"Initialized ProductionEmbeddingFunction with provider={provider}, model={model_id}")
    
    def _build_config(self) -> Dict[str, Any]:
        """Build configuration for the embeddings service."""
        config = {
            "provider": self.provider,
            "model_id": self.model_id,
            "model_name_or_path": self.model_id
        }
        
        if self.api_key:
            config["api_key"] = self.api_key
        
        if self.api_url:
            config["api_url"] = self.api_url
            
        return config
    
    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for documents using the production service.
        
        This is called by ChromaDB when adding or querying documents.
        
        Args:
            input: List of document strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not input:
            return []
        
        logger.debug(f"Creating embeddings for {len(input)} documents using production service")
        
        try:
            if self.use_async:
                # Run async function in sync context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    embeddings = loop.run_until_complete(
                        create_embeddings_batch_async(
                            texts=input,
                            provider=self.provider,
                            model_id=self.model_id,
                            api_key=self.api_key,
                            api_url=self.api_url
                        )
                    )
                finally:
                    loop.close()
            else:
                # Use synchronous version
                embeddings = create_embeddings_batch(
                    texts=input,
                    provider=self.provider,
                    model_id=self.model_id,
                    api_key=self.api_key,
                    api_url=self.api_url
                )
            
            # Convert numpy array to list if needed
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            
            logger.debug(f"Successfully created {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}") from e


class EnhancedVectorRetriever:
    """
    Enhanced vector retriever that uses production embeddings service.
    
    This replaces the default ChromaDB embeddings with our production service,
    ensuring consistency and quality across the RAG pipeline.
    """
    
    def __init__(
        self,
        source: DataSource,
        chroma_path: Path,
        collection_name: str,
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize enhanced vector retriever with production embeddings.
        
        Args:
            source: Data source type
            chroma_path: Path to ChromaDB storage
            collection_name: Name of the collection
            embedding_provider: Provider for embeddings
            embedding_model: Model to use for embeddings
            api_key: Optional API key
            api_url: Optional custom API URL
            config: Additional configuration
        """
        self.source = source
        self.collection_name = collection_name
        self.config = config or {}
        
        # Create production embedding function
        self.embedding_function = ProductionEmbeddingFunction(
            provider=embedding_provider,
            model_id=embedding_model,
            api_key=api_key,
            api_url=api_url
        )
        
        # Initialize ChromaDB client with production embeddings
        self.chroma_client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection with production embedding function
        try:
            # Try to get existing collection
            self.collection = self.chroma_client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Loaded existing collection '{collection_name}' with production embeddings")
        except:
            # Create new collection with production embeddings
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"source": source.name, "embedding_provider": embedding_provider}
            )
            logger.info(f"Created new collection '{collection_name}' with production embeddings")
    
    async def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10
    ) -> SearchResult:
        """
        Retrieve documents using production embeddings for query.
        
        Args:
            query: Search query
            filters: Optional filters for the search
            top_k: Number of results to return
            
        Returns:
            SearchResult with retrieved documents
        """
        try:
            # Build ChromaDB where clause from filters
            where = None
            if filters:
                where = {}
                for key, value in filters.items():
                    if isinstance(value, list):
                        where[key] = {"$in": value}
                    else:
                        where[key] = value
            
            # Query uses production embeddings via embedding_function
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            documents = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    doc = Document(
                        id=doc_id,
                        content=results["documents"][0][i],
                        metadata=results["metadatas"][0][i],
                        source=self.source,
                        score=1.0 - results["distances"][0][i]  # Convert distance to similarity
                    )
                    documents.append(doc)
            
            logger.debug(f"Retrieved {len(documents)} documents using production embeddings")
            
            return SearchResult(
                documents=documents,
                query=query,
                search_type="vector",
                metadata={
                    "collection": self.collection_name,
                    "embedding_provider": self.embedding_function.provider,
                    "embedding_model": self.embedding_function.model_id
                }
            )
            
        except Exception as e:
            logger.error(f"Error in vector retrieval with production embeddings: {e}")
            raise RetrievalError(f"Failed to retrieve from vector store: {e}")
    
    async def embed_and_store(self, documents: List[Document]) -> None:
        """
        Embed documents using production service and store in vector database.
        
        Args:
            documents: List of documents to embed and store
        """
        if not documents:
            return
        
        try:
            # Prepare data for ChromaDB
            ids = []
            contents = []
            metadatas = []
            
            for doc in documents:
                ids.append(doc.id)
                contents.append(doc.content)
                metadatas.append({
                    **doc.metadata,
                    "source": self.source.name
                })
            
            # Add to collection - uses production embeddings via embedding_function
            self.collection.add(
                ids=ids,
                documents=contents,
                metadatas=metadatas
            )
            
            logger.info(f"Stored {len(documents)} documents with production embeddings")
            
        except Exception as e:
            logger.error(f"Error storing documents with production embeddings: {e}")
            raise RetrievalError(f"Failed to store documents: {e}")
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about the embeddings in this collection."""
        try:
            count = self.collection.count()
            
            # Get a sample to determine embedding dimension
            if count > 0:
                sample = self.collection.peek(1)
                if sample and sample.get("embeddings"):
                    dimension = len(sample["embeddings"][0])
                else:
                    dimension = None
            else:
                dimension = None
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_dimension": dimension,
                "embedding_provider": self.embedding_function.provider,
                "embedding_model": self.embedding_function.model_id,
                "source": self.source.name
            }
        except Exception as e:
            logger.error(f"Error getting embedding stats: {e}")
            return {}


class RAGEmbeddingsIntegration:
    """
    Main integration class for RAG with production embeddings.
    
    This class ensures all RAG operations use the actual embeddings service
    without any mocking or placeholder implementations.
    """
    
    def __init__(
        self,
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        cache_embeddings: bool = True
    ):
        """
        Initialize RAG-Embeddings integration.
        
        Args:
            embedding_provider: Provider for embeddings
            embedding_model: Model to use
            api_key: Optional API key
            api_url: Optional custom API URL
            cache_embeddings: Whether to cache embeddings
        """
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.api_key = api_key
        self.api_url = api_url
        self.cache_embeddings = cache_embeddings
        
        # Initialize embeddings service wrapper for direct use
        self.embeddings_service = EmbeddingsServiceWrapper(
            model_name=f"{embedding_provider}/{embedding_model}" if embedding_provider == "openai" else embedding_model,
            api_key=api_key,
            base_url=api_url
        )
        
        logger.info(f"Initialized RAG-Embeddings integration with {embedding_provider}/{embedding_model}")
    
    def create_vector_retriever(
        self,
        source: DataSource,
        chroma_path: Path,
        collection_name: str,
        **kwargs
    ) -> EnhancedVectorRetriever:
        """
        Create a vector retriever with production embeddings.
        
        Args:
            source: Data source type
            chroma_path: Path to ChromaDB storage
            collection_name: Name of the collection
            **kwargs: Additional configuration
            
        Returns:
            EnhancedVectorRetriever configured with production embeddings
        """
        return EnhancedVectorRetriever(
            source=source,
            chroma_path=chroma_path,
            collection_name=collection_name,
            embedding_provider=self.embedding_provider,
            embedding_model=self.embedding_model,
            api_key=self.api_key,
            api_url=self.api_url,
            config=kwargs
        )
    
    async def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query using the production embeddings service.
        
        Args:
            query: Query text to embed
            
        Returns:
            Query embedding vector
        """
        return await self.embeddings_service.create_embedding_async(query)
    
    async def embed_documents(self, documents: List[str]) -> np.ndarray:
        """
        Embed multiple documents using the production embeddings service.
        
        Args:
            documents: List of document texts
            
        Returns:
            Document embeddings matrix
        """
        return await self.embeddings_service.create_embeddings_async(documents)
    
    def get_embedding_dimension(self) -> Optional[int]:
        """Get the dimension of embeddings produced by the current model."""
        return self.embeddings_service.get_embedding_dimension()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics from the embeddings service."""
        return self.embeddings_service.get_metrics()
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'embeddings_service'):
            self.embeddings_service.close()


# Factory function for easy creation
def create_rag_embeddings_integration(
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    **kwargs
) -> RAGEmbeddingsIntegration:
    """
    Factory function to create RAG-Embeddings integration.
    
    Args:
        provider: Embedding provider (openai, huggingface, cohere, etc.)
        model: Model name (defaults based on provider)
        api_key: Optional API key
        api_url: Optional custom API URL
        **kwargs: Additional configuration
        
    Returns:
        Configured RAGEmbeddingsIntegration instance
    """
    # Default models for providers
    default_models = {
        "openai": "text-embedding-3-small",
        "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
        "cohere": "embed-english-v3.0",
        "voyage": "voyage-2"
    }
    
    if model is None:
        model = default_models.get(provider, "text-embedding-3-small")
    
    return RAGEmbeddingsIntegration(
        embedding_provider=provider,
        embedding_model=model,
        api_key=api_key,
        api_url=api_url,
        **kwargs
    )


# Example usage demonstrating real integration
if __name__ == "__main__":
    async def test_integration():
        """Test the RAG-Embeddings integration with real embeddings."""
        
        # Create integration with real OpenAI embeddings
        integration = create_rag_embeddings_integration(
            provider="openai",
            model="text-embedding-3-small"
        )
        
        # Test query embedding
        query = "What is machine learning?"
        query_embedding = await integration.embed_query(query)
        print(f"Query embedding shape: {query_embedding.shape}")
        
        # Test document embedding
        documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing helps computers understand human language."
        ]
        doc_embeddings = await integration.embed_documents(documents)
        print(f"Document embeddings shape: {doc_embeddings.shape}")
        
        # Create vector retriever with real embeddings
        retriever = integration.create_vector_retriever(
            source=DataSource.MEDIA_DB,
            chroma_path=Path("/tmp/test_chroma"),
            collection_name="test_collection"
        )
        
        # Store documents with real embeddings
        test_docs = [
            Document(
                id=f"doc_{i}",
                content=doc,
                metadata={"index": i},
                source=DataSource.MEDIA_DB,
                score=0.0
            )
            for i, doc in enumerate(documents)
        ]
        
        await retriever.embed_and_store(test_docs)
        
        # Retrieve using real embeddings
        results = await retriever.retrieve(query, top_k=2)
        print(f"Retrieved {len(results.documents)} documents")
        for doc in results.documents:
            print(f"  - Score: {doc.score:.3f}, Content: {doc.content[:50]}...")
        
        # Get metrics
        metrics = integration.get_metrics()
        print(f"Embeddings metrics: {metrics}")
        
        # Clean up
        integration.close()
    
    # Run the test
    asyncio.run(test_integration())