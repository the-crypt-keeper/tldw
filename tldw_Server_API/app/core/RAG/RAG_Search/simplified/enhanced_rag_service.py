"""
Enhanced RAG service with parent document retrieval and advanced chunking.

This module extends the basic RAG service with:
- Parent document retrieval for context expansion
- Enhanced chunking with structure preservation
- Advanced text processing and cleaning
- Improved search with context expansion
"""

import asyncio
from typing import List, Optional, Dict, Any, Union, Tuple, Literal
from pathlib import Path
from loguru import logger
from dataclasses import dataclass
import time
import uuid

from .rag_service import RAGService, SearchResult, SearchResultWithCitations
from .citations import Citation, CitationType
from .config import RAGConfig
from .data_models import IndexingResult
from ..enhanced_chunking_service import EnhancedChunkingService
from .enhanced_indexing_helpers import (
    chunk_documents_with_parents,
    generate_embeddings_for_parent_retrieval,
    store_documents_with_parents
)
from tldw_chatbook.Metrics.metrics_logger import log_counter, log_histogram, timeit


class EnhancedRAGService(RAGService):
    """
    Enhanced RAG service with parent document retrieval and advanced features.
    
    Additional features:
    - Parent document retrieval for expanded context
    - Enhanced chunking with structure preservation
    - Advanced text cleaning from PDF artifacts
    - Improved search with automatic context expansion
    """
    
    def __init__(self, config: Optional[RAGConfig] = None, enable_parent_retrieval: bool = True):
        """
        Initialize enhanced RAG service.
        
        Args:
            config: RAG configuration
            enable_parent_retrieval: Whether to enable parent document retrieval
        """
        super().__init__(config)
        
        self.enable_parent_retrieval = enable_parent_retrieval
        self.enhanced_chunking = EnhancedChunkingService()
        
        # Additional configuration
        self.parent_size_multiplier = getattr(config, 'parent_size_multiplier', 3)
        self.expand_context_on_retrieval = getattr(config, 'expand_context_on_retrieval', True)
        self.clean_pdf_artifacts = getattr(config, 'clean_pdf_artifacts', True)
        
        logger.info(f"Initialized EnhancedRAGService with parent_retrieval={enable_parent_retrieval}")
    
    @timeit("enhanced_rag_indexing_document")
    async def index_document_with_parents(self,
                                         doc_id: str,
                                         content: str,
                                         title: Optional[str] = None,
                                         metadata: Optional[Dict[str, Any]] = None,
                                         chunk_size: Optional[int] = None,
                                         chunk_overlap: Optional[int] = None,
                                         parent_size_multiplier: Optional[int] = None,
                                         use_structural_chunking: bool = True) -> IndexingResult:
        """
        Index a document with parent document retrieval support.
        
        Args:
            doc_id: Unique document identifier
            content: Document content to index
            title: Document title
            metadata: Optional metadata
            chunk_size: Override default chunk size
            chunk_overlap: Override default chunk overlap
            parent_size_multiplier: Override parent size multiplier
            use_structural_chunking: Whether to use structural chunking
            
        Returns:
            IndexingResult with enhanced metadata
        """
        start_time = time.time()
        metadata = metadata or {}
        title = title or doc_id
        parent_multiplier = parent_size_multiplier or self.parent_size_multiplier
        
        # Input validation
        if not doc_id or not isinstance(doc_id, str):
            raise ValueError("doc_id must be a non-empty string")
        
        if not content or not isinstance(content, str):
            raise ValueError("content must be a non-empty string")
        
        # Log metrics
        log_counter("enhanced_rag_document_index_attempt")
        log_histogram("enhanced_rag_document_size_chars", len(content))
        
        try:
            # Clean text if enabled
            if self.clean_pdf_artifacts:
                content, corrections = self.enhanced_chunking.structure_parser.clean_text(content)
                if corrections:
                    logger.info(f"Cleaned {len(corrections)} PDF artifacts from document {doc_id}")
                    metadata['pdf_artifacts_cleaned'] = len(corrections)
            
            # Use enhanced chunking with parent retrieval
            if use_structural_chunking:
                result = self.enhanced_chunking.chunk_with_parent_retrieval(
                    content,
                    chunk_size=chunk_size or self.config.chunk_size,
                    chunk_overlap=chunk_overlap or self.config.chunk_overlap,
                    parent_size_multiplier=parent_multiplier
                )
                
                retrieval_chunks = result['chunks']
                parent_chunks = result['parent_chunks']
                chunking_metadata = result['metadata']
                
            else:
                # Fallback to standard chunking
                retrieval_chunks = await self._chunk_document(
                    content,
                    chunk_size or self.config.chunk_size,
                    chunk_overlap or self.config.chunk_overlap,
                    self.config.chunking_method
                )
                
                # Create parent chunks
                parent_chunk_size = (chunk_size or self.config.chunk_size) * parent_multiplier
                parent_overlap = (chunk_overlap or self.config.chunk_overlap) * parent_multiplier
                
                parent_chunks = await self._chunk_document(
                    content,
                    parent_chunk_size,
                    parent_overlap,
                    self.config.chunking_method
                )
                
                chunking_metadata = {
                    'method': 'standard',
                    'total_chunks': len(retrieval_chunks),
                    'total_parent_chunks': len(parent_chunks)
                }
            
            # Generate embeddings for retrieval chunks
            chunk_texts = [chunk['text'] for chunk in retrieval_chunks]
            embeddings = await self.embeddings.create_embeddings_async(chunk_texts)
            
            # Prepare for storage with enhanced metadata
            chunk_ids = []
            chunk_metadata = []
            
            for i, chunk in enumerate(retrieval_chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                chunk_ids.append(chunk_id)
                
                # Create metadata with parent chunk reference
                meta = {
                    **metadata,
                    "doc_id": doc_id,
                    "doc_title": title,
                    "chunk_index": i,
                    "chunk_start": chunk.get("start_char", 0),
                    "chunk_end": chunk.get("end_char", len(chunk["text"])),
                    "chunk_size": len(chunk["text"]),
                    "word_count": chunk.get("word_count", len(chunk["text"].split())),
                    "text_preview": chunk["text"][:200],
                    "chunk_type": chunk.get("chunk_type", "text"),
                    "chunk_level": chunk.get("level", 0)
                }
                
                # Add parent chunk reference
                parent_idx = chunk.get('parent_chunk_index') or chunk.get('metadata', {}).get('parent_chunk_index')
                if parent_idx is not None and parent_idx < len(parent_chunks):
                    parent_chunk = parent_chunks[parent_idx]
                    meta['parent_chunk_text'] = parent_chunk['text']
                    meta['parent_chunk_index'] = parent_idx
                    meta['has_parent_chunk'] = True
                
                chunk_metadata.append(meta)
            
            # Store in vector database
            await self._store_chunks(chunk_ids, embeddings, chunk_texts, chunk_metadata)
            
            # Update metrics
            self._docs_indexed += 1
            self._total_chunks_created += len(retrieval_chunks)
            
            elapsed = time.time() - start_time
            
            # Log success metrics
            log_counter("enhanced_rag_document_index_success")
            log_histogram("enhanced_rag_retrieval_chunks_created", len(retrieval_chunks))
            log_histogram("enhanced_rag_parent_chunks_created", len(parent_chunks))
            
            logger.info(
                f"Indexed document {doc_id} with {len(retrieval_chunks)} retrieval chunks "
                f"and {len(parent_chunks)} parent chunks in {elapsed:.2f}s"
            )
            
            return IndexingResult(
                doc_id=doc_id,
                chunks_created=len(retrieval_chunks),
                time_taken=elapsed,
                success=True,
                metadata={
                    'parent_chunks_created': len(parent_chunks),
                    'chunking_method': chunking_metadata.get('method', 'enhanced'),
                    'parent_retrieval_enabled': True,
                    'pdf_artifacts_cleaned': metadata.get('pdf_artifacts_cleaned', 0)
                }
            )
            
        except Exception as e:
            log_counter("enhanced_rag_document_index_error", labels={"error": type(e).__name__})
            logger.error(f"Failed to index document {doc_id}: {e}", exc_info=True)
            return IndexingResult(
                doc_id=doc_id,
                chunks_created=0,
                time_taken=time.time() - start_time,
                success=False,
                error=str(e)
            )
    
    async def index_batch_with_parents(self,
                                      documents: List[Dict[str, Any]],
                                      show_progress: bool = True,
                                      batch_size: int = 32,
                                      use_structural_chunking: bool = True) -> List[IndexingResult]:
        """
        Index multiple documents with parent document retrieval support.
        
        Args:
            documents: List of documents to index
            show_progress: Whether to show progress
            batch_size: Batch size for embedding generation
            use_structural_chunking: Whether to use structural chunking
            
        Returns:
            List of IndexingResult for each document
        """
        total = len(documents)
        if not documents:
            return []
        
        logger.info(f"Starting enhanced batch indexing for {total} documents with parent retrieval")
        batch_start_time = time.time()
        
        # Phase 1: Enhanced chunking with parent chunks
        chunk_start = time.time()
        retrieval_chunks, parent_chunks, doc_chunk_info, failed_results = await chunk_documents_with_parents(
            self, documents, self.parent_size_multiplier, show_progress, use_structural_chunking
        )
        chunk_time = time.time() - chunk_start
        logger.info(
            f"Enhanced chunking completed in {chunk_time:.2f}s: "
            f"{len(retrieval_chunks)} retrieval chunks, {len(parent_chunks)} parent chunks"
        )
        
        if not retrieval_chunks:
            logger.warning("No chunks created from any documents")
            return failed_results
        
        # Phase 2: Generate embeddings for retrieval chunks
        embed_start = time.time()
        chunk_texts = [chunk['text'] if isinstance(chunk, dict) else str(chunk) for chunk in retrieval_chunks]
        retrieval_embeddings, _, failed_indices, _ = await generate_embeddings_for_parent_retrieval(
            self, retrieval_chunks, parent_chunks, batch_size, show_progress, embed_parents=False
        )
        embed_time = time.time() - embed_start
        logger.info(f"Embedding generation completed in {embed_time:.2f}s")
        
        # Phase 3: Store documents with parent chunk references
        store_start = time.time()
        storage_results = await store_documents_with_parents(
            self, documents, doc_chunk_info, retrieval_embeddings, parent_chunks,
            batch_start_time, failed_indices, store_parent_chunks=True
        )
        store_time = time.time() - store_start
        
        # Combine results
        results = failed_results + storage_results
        total_time = time.time() - batch_start_time
        
        # Summary
        successful = sum(1 for r in results if r and r.success)
        logger.info(
            f"Enhanced batch indexing completed: {successful}/{total} documents, "
            f"total time: {total_time:.2f}s "
            f"(chunk: {chunk_time:.2f}s, embed: {embed_time:.2f}s, store: {store_time:.2f}s)"
        )
        
        # Update metrics
        log_counter("enhanced_rag_batch_index_completed", value=successful)
        log_histogram("enhanced_rag_batch_index_total_time", total_time)
        
        return results
    
    @timeit("enhanced_rag_search_with_expansion")
    async def search_with_context_expansion(self,
                                          query: str,
                                          top_k: Optional[int] = None,
                                          search_type: Literal["semantic", "hybrid", "keyword"] = "semantic",
                                          filter_metadata: Optional[Dict[str, Any]] = None,
                                          expand_to_parent: bool = True,
                                          include_citations: Optional[bool] = None,
                                          score_threshold: Optional[float] = None) -> List[Union[SearchResult, SearchResultWithCitations]]:
        """
        Search with automatic parent document context expansion.
        
        Args:
            query: Search query
            top_k: Number of results to return
            search_type: Type of search
            filter_metadata: Metadata filters
            expand_to_parent: Whether to expand results to parent chunks
            include_citations: Whether to include citations
            score_threshold: Minimum score threshold
            
        Returns:
            Search results with expanded context when available
        """
        # Perform initial search
        results = await self.search(
            query=query,
            top_k=top_k,
            search_type=search_type,
            filter_metadata=filter_metadata,
            include_citations=include_citations,
            score_threshold=score_threshold
        )
        
        if not expand_to_parent or not self.enable_parent_retrieval:
            return results
        
        # Expand results to include parent chunk context
        expanded_results = []
        
        for result in results:
            # Check if this result has parent chunk information
            has_parent = result.metadata.get('has_parent_chunk', False)
            parent_text = result.metadata.get('parent_chunk_text')
            
            if has_parent and parent_text:
                # Create expanded result
                if isinstance(result, SearchResultWithCitations):
                    expanded_result = SearchResultWithCitations(
                        id=result.id,
                        score=result.score,
                        document=parent_text,  # Use parent text as main document
                        metadata={
                            **result.metadata,
                            'original_chunk_text': result.document,
                            'context_expanded': True,
                            'expansion_type': 'parent_chunk'
                        },
                        citations=result.citations
                    )
                else:
                    expanded_result = SearchResult(
                        id=result.id,
                        score=result.score,
                        document=parent_text,  # Use parent text as main document
                        metadata={
                            **result.metadata,
                            'original_chunk_text': result.document,
                            'context_expanded': True,
                            'expansion_type': 'parent_chunk'
                        }
                    )
                
                expanded_results.append(expanded_result)
                
                # Log expansion
                log_counter("enhanced_rag_context_expansion", labels={"type": "parent_chunk"})
            else:
                # No parent chunk available, use original result
                expanded_results.append(result)
        
        logger.info(f"Expanded {sum(1 for r in expanded_results if r.metadata.get('context_expanded'))} results to parent context")
        
        return expanded_results
    
    async def get_chunk_with_context(self,
                                   chunk_id: str,
                                   context_type: Literal["parent", "surrounding", "both"] = "parent") -> Optional[Dict[str, Any]]:
        """
        Retrieve a chunk with its surrounding context.
        
        Args:
            chunk_id: ID of the chunk to retrieve
            context_type: Type of context to include
            
        Returns:
            Dictionary with chunk and context information
        """
        # This would typically query the vector store directly
        # For now, we'll implement a placeholder
        logger.warning(f"get_chunk_with_context not fully implemented for chunk_id: {chunk_id}")
        return None


# Convenience function
def create_enhanced_rag_service(
    embedding_model: Optional[str] = None,
    enable_parent_retrieval: bool = True,
    vector_store: Optional[str] = None,
    **kwargs
) -> EnhancedRAGService:
    """
    Create an enhanced RAG service with common configurations.
    
    Args:
        embedding_model: Embedding model to use
        enable_parent_retrieval: Whether to enable parent document retrieval
        vector_store: Vector store type ("chroma" or "memory")
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured EnhancedRAGService instance
    """
    from .rag_service import RAGConfig
    
    config = RAGConfig()
    
    if embedding_model:
        config.embedding.model = embedding_model
    
    if vector_store:
        config.vector_store.type = vector_store
    
    # Apply any additional kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return EnhancedRAGService(config, enable_parent_retrieval)