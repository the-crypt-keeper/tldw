"""
Factory for creating vector store adapters.

This module provides a factory pattern for instantiating the appropriate
vector store adapter based on configuration.
"""

from typing import Dict, Type, Optional
from loguru import logger

from .base import VectorStoreAdapter, VectorStoreConfig, VectorStoreType
from .chromadb_adapter import ChromaDBAdapter


class VectorStoreFactory:
    """Factory for creating vector store adapters."""
    
    # Registry of available adapters
    _adapters: Dict[VectorStoreType, Type[VectorStoreAdapter]] = {
        VectorStoreType.CHROMADB: ChromaDBAdapter,
        # Future adapters can be registered here:
        # VectorStoreType.PINECONE: PineconeAdapter,
        # VectorStoreType.WEAVIATE: WeaviateAdapter,
        # VectorStoreType.QDRANT: QdrantAdapter,
        # VectorStoreType.FAISS: FAISSAdapter,
    }
    
    @classmethod
    def create_adapter(
        cls,
        config: VectorStoreConfig,
        initialize: bool = True
    ) -> VectorStoreAdapter:
        """
        Create a vector store adapter based on configuration.
        
        Args:
            config: Vector store configuration
            initialize: Whether to initialize the adapter immediately
            
        Returns:
            Configured vector store adapter instance
            
        Raises:
            ValueError: If the requested vector store type is not supported
        """
        adapter_class = cls._adapters.get(config.store_type)
        
        if not adapter_class:
            available = ", ".join([t.value for t in cls._adapters.keys()])
            raise ValueError(
                f"Unsupported vector store type: {config.store_type}. "
                f"Available types: {available}"
            )
        
        logger.info(f"Creating {config.store_type.value} adapter")
        adapter = adapter_class(config)
        
        if initialize:
            # If async initialization is needed, it should be done separately
            logger.info(f"Adapter created (initialization pending)")
        
        return adapter
    
    @classmethod
    def register_adapter(
        cls,
        store_type: VectorStoreType,
        adapter_class: Type[VectorStoreAdapter]
    ) -> None:
        """
        Register a new adapter type.
        
        This allows for dynamic registration of custom adapters.
        
        Args:
            store_type: Vector store type identifier
            adapter_class: Adapter class to register
        """
        cls._adapters[store_type] = adapter_class
        logger.info(f"Registered adapter for {store_type.value}")
    
    @classmethod
    def get_available_stores(cls) -> list[str]:
        """
        Get list of available vector store types.
        
        Returns:
            List of available store type names
        """
        return [store_type.value for store_type in cls._adapters.keys()]
    
    @classmethod
    def create_from_settings(
        cls,
        settings: Dict,
        user_id: str = "0"
    ) -> Optional[VectorStoreAdapter]:
        """
        Create an adapter from application settings.
        
        Args:
            settings: Application settings dictionary
            user_id: User identifier for multi-tenant scenarios
            
        Returns:
            Configured adapter or None if not configured
        """
        # Extract vector store configuration from settings
        vector_store_type = settings.get("RAG", {}).get("vector_store_type", "chromadb")
        
        # Get embedding configuration
        embedding_config = settings.get("EMBEDDING_CONFIG", {})
        embedding_dim = embedding_config.get("embedding_dimension", 384)
        
        # Get connection parameters based on store type
        connection_params = {}
        if vector_store_type == "chromadb":
            connection_params = {
                "use_default": True,  # Use existing singleton
                "embedding_config": embedding_config
            }
        # Add other store types here as needed
        
        # Create configuration
        try:
            store_type_enum = VectorStoreType(vector_store_type)
        except ValueError:
            logger.warning(f"Unknown vector store type: {vector_store_type}, defaulting to ChromaDB")
            store_type_enum = VectorStoreType.CHROMADB
        
        config = VectorStoreConfig(
            store_type=store_type_enum,
            connection_params=connection_params,
            embedding_dim=embedding_dim,
            distance_metric=settings.get("RAG", {}).get("distance_metric", "cosine"),
            collection_prefix=settings.get("RAG", {}).get("collection_prefix", "unified"),
            user_id=user_id
        )
        
        return cls.create_adapter(config, initialize=False)