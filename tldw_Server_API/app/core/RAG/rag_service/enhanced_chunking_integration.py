# enhanced_chunking_integration.py
"""
Integration module for enhanced chunking in the RAG functional pipeline.

This module provides functions to integrate the enhanced chunking capabilities
into the functional pipeline, making them composable with other pipeline functions.
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from .types import Document, DataSource
# Enhanced features are now integrated into the main Chunk_Lib
from ...Chunking.Chunk_Lib import Chunker as EnhancedChunker, EnhancedChunk, ChunkType


async def enhanced_chunk_documents(
    context,  # RAGPipelineContext
    enable_pdf_cleaning: bool = None,
    preserve_code_blocks: bool = None,
    preserve_tables: bool = None,
    structure_aware: bool = None,
    track_positions: bool = None,
    chunk_size: int = None,
    overlap: int = None
):
    """
    Apply enhanced chunking to documents in the pipeline context.
    
    This function can be composed in the functional pipeline to provide:
    - PDF artifact cleaning
    - Code block preservation
    - Table extraction with serialization
    - Structure-aware chunking
    - Character position tracking
    
    Args:
        context: RAGPipelineContext with documents to chunk
        enable_pdf_cleaning: Clean PDF artifacts
        preserve_code_blocks: Preserve code as separate chunks
        preserve_tables: Extract and serialize tables
        structure_aware: Use structure-aware chunking
        track_positions: Track character positions
        chunk_size: Target chunk size (default from config)
        overlap: Chunk overlap (default from config)
    
    Returns:
        Updated context with chunked documents
    """
    if not context.documents:
        return context
    
    # Get configuration
    config = context.config or {}
    
    # Build enhanced chunking options
    chunking_options = {
        "clean_pdf_artifacts": enable_pdf_cleaning if enable_pdf_cleaning is not None 
                               else config.get("clean_pdf_artifacts", True),
        "preserve_code_blocks": preserve_code_blocks if preserve_code_blocks is not None
                               else config.get("preserve_code_blocks", True),
        "preserve_tables": preserve_tables if preserve_tables is not None
                          else config.get("preserve_tables", True),
        "structure_aware": structure_aware if structure_aware is not None
                          else config.get("structure_aware", True),
        "track_positions": track_positions if track_positions is not None
                          else config.get("track_positions", True),
        "table_serialize_method": config.get("table_serialize_method", "hybrid")
    }
    
    chunk_size = chunk_size or config.get("chunk_size", 512)
    overlap = overlap or config.get("chunk_overlap", 128)
    
    # Initialize enhanced chunker
    chunker = EnhancedChunker(chunking_options)
    
    # Process each document
    chunked_documents = []
    total_chunks = 0
    chunk_type_stats = {}
    
    for doc in context.documents:
        try:
            # Chunk the document
            enhanced_chunks = chunker.chunk_text_enhanced(
                text=doc.content,
                chunk_size=chunk_size,
                overlap=overlap,
                doc_id=doc.id
            )
            
            # Convert enhanced chunks to documents
            for chunk in enhanced_chunks:
                # Build chunk metadata
                chunk_metadata = {
                    **doc.metadata,  # Inherit document metadata
                    **chunk.metadata,  # Add chunk-specific metadata
                    "chunk_type": chunk.chunk_type.value,
                    "chunk_index": chunk.chunk_index,
                    "parent_id": chunk.parent_id or doc.id,
                    "original_doc_id": doc.id,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char
                }
                
                # Create new document for chunk
                chunk_doc = Document(
                    id=chunk.id,
                    content=chunk.content,
                    metadata=chunk_metadata,
                    source=doc.source,
                    score=doc.score  # Preserve original score
                )
                
                chunked_documents.append(chunk_doc)
                
                # Track statistics
                chunk_type = chunk.chunk_type.value
                chunk_type_stats[chunk_type] = chunk_type_stats.get(chunk_type, 0) + 1
                total_chunks += 1
                
        except Exception as e:
            logger.error(f"Failed to chunk document {doc.id}: {e}")
            # Fall back to including original document
            chunked_documents.append(doc)
    
    # Update context
    context.documents = chunked_documents
    
    # Add chunking metadata to context
    context.metadata["enhanced_chunking_applied"] = True
    context.metadata["total_chunks_created"] = total_chunks
    context.metadata["chunk_type_distribution"] = chunk_type_stats
    context.metadata["chunking_options"] = chunking_options
    
    logger.info(
        f"Enhanced chunking created {total_chunks} chunks from {len(context.documents)} documents. "
        f"Types: {chunk_type_stats}"
    )
    
    return context


async def filter_chunks_by_type(
    context,  # RAGPipelineContext
    include_types: Optional[List[str]] = None,
    exclude_types: Optional[List[str]] = None
):
    """
    Filter chunks by their type.
    
    Useful for:
    - Including only code chunks for code search
    - Excluding tables from general text search
    - Focusing on headers for navigation
    
    Args:
        context: RAGPipelineContext with chunked documents
        include_types: List of chunk types to include (e.g., ["code", "text"])
        exclude_types: List of chunk types to exclude (e.g., ["table"])
    
    Returns:
        Updated context with filtered documents
    """
    if not context.documents:
        return context
    
    if not include_types and not exclude_types:
        return context  # No filtering needed
    
    filtered_docs = []
    
    for doc in context.documents:
        chunk_type = doc.metadata.get("chunk_type", "text")
        
        # Apply inclusion filter
        if include_types and chunk_type not in include_types:
            continue
        
        # Apply exclusion filter
        if exclude_types and chunk_type in exclude_types:
            continue
        
        filtered_docs.append(doc)
    
    original_count = len(context.documents)
    context.documents = filtered_docs
    
    context.metadata["chunk_filtering_applied"] = True
    context.metadata["chunks_before_filter"] = original_count
    context.metadata["chunks_after_filter"] = len(filtered_docs)
    
    logger.debug(f"Filtered chunks: {original_count} -> {len(filtered_docs)}")
    
    return context


async def expand_with_parent_context(
    context,  # RAGPipelineContext
    expansion_size: int = None,
    include_siblings: bool = None
):
    """
    Expand chunks with parent document context.
    
    For each chunk, fetches:
    - Parent document context
    - Sibling chunks (adjacent chunks from same document)
    
    Args:
        context: RAGPipelineContext with chunked documents
        expansion_size: Characters to expand around chunk
        include_siblings: Include adjacent chunks
    
    Returns:
        Updated context with expanded documents
    """
    if not context.documents:
        return context
    
    config = context.config or {}
    expansion_size = expansion_size or config.get("parent_expansion_size", 500)
    include_siblings = include_siblings if include_siblings is not None else config.get("include_siblings", True)
    
    # Group chunks by parent document
    parent_groups = {}
    for doc in context.documents:
        parent_id = doc.metadata.get("parent_id") or doc.metadata.get("original_doc_id")
        if parent_id:
            if parent_id not in parent_groups:
                parent_groups[parent_id] = []
            parent_groups[parent_id].append(doc)
    
    expanded_docs = []
    
    for parent_id, chunks in parent_groups.items():
        # Sort chunks by index
        chunks.sort(key=lambda d: d.metadata.get("chunk_index", 0))
        
        for i, chunk in enumerate(chunks):
            # Build expanded content
            expanded_parts = []
            
            # Add previous sibling if requested
            if include_siblings and i > 0:
                prev_chunk = chunks[i - 1]
                expanded_parts.append(f"[Previous context: ...{prev_chunk.content[-100:]}]")
            
            # Add main chunk
            expanded_parts.append(chunk.content)
            
            # Add next sibling if requested
            if include_siblings and i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                expanded_parts.append(f"[Following context: {next_chunk.content[:100]}...]")
            
            # Create expanded document
            expanded_doc = Document(
                id=chunk.id,
                content="\n".join(expanded_parts),
                metadata={
                    **chunk.metadata,
                    "expanded": True,
                    "has_siblings": include_siblings and (i > 0 or i < len(chunks) - 1)
                },
                source=chunk.source,
                score=chunk.score
            )
            
            expanded_docs.append(expanded_doc)
    
    context.documents = expanded_docs
    context.metadata["parent_expansion_applied"] = True
    
    logger.debug(f"Expanded {len(expanded_docs)} chunks with parent context")
    
    return context


async def prioritize_by_chunk_type(
    context,  # RAGPipelineContext
    type_priorities: Optional[Dict[str, float]] = None
):
    """
    Adjust document scores based on chunk type.
    
    Useful for:
    - Boosting code chunks for programming queries
    - Prioritizing headers for navigation queries
    - Reducing table weight for narrative queries
    
    Args:
        context: RAGPipelineContext with chunked documents
        type_priorities: Dict mapping chunk types to score multipliers
                        e.g., {"code": 1.5, "table": 0.8, "header": 1.2}
    
    Returns:
        Updated context with adjusted scores
    """
    if not context.documents:
        return context
    
    # Default priorities
    default_priorities = {
        "code": 1.0,
        "table": 1.0,
        "header": 1.0,
        "text": 1.0,
        "list": 1.0
    }
    
    priorities = type_priorities or context.config.get("chunk_type_priorities", default_priorities)
    
    for doc in context.documents:
        chunk_type = doc.metadata.get("chunk_type", "text")
        
        if chunk_type in priorities:
            multiplier = priorities[chunk_type]
            doc.score = doc.score * multiplier
            doc.metadata["score_adjusted"] = True
            doc.metadata["score_multiplier"] = multiplier
    
    # Re-sort by adjusted scores
    context.documents.sort(key=lambda d: d.score, reverse=True)
    
    context.metadata["chunk_type_prioritization_applied"] = True
    context.metadata["type_priorities"] = priorities
    
    logger.debug(f"Applied chunk type priorities: {priorities}")
    
    return context


# Function registry for pipeline building
ENHANCED_CHUNKING_FUNCTIONS = {
    "enhanced_chunk_documents": enhanced_chunk_documents,
    "filter_chunks_by_type": filter_chunks_by_type,
    "expand_with_parent_context": expand_with_parent_context,
    "prioritize_by_chunk_type": prioritize_by_chunk_type
}