"""
Enhanced indexing helpers with parent document retrieval support.

This module extends the basic indexing helpers with advanced features:
- Parent document retrieval support
- Enhanced chunking with structure preservation
- Improved metadata tracking
"""

from loguru import logger
import time
from typing import List, Dict, Any, Tuple, Optional, Union
import hashlib

# Handle numpy as optional dependency
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Create a minimal stub for type hints
    class np:
        class ndarray:
            pass

from .data_models import IndexingResult
from ..enhanced_chunking_service import EnhancedChunkingService, StructuredChunk

# Constants
CHUNK_PROGRESS_INTERVAL = 10
EMBEDDING_PROGRESS_INTERVAL = 5
DEFAULT_PARENT_SIZE_MULTIPLIER = 3


async def chunk_documents_with_parents(
    rag_service,
    documents: List[Dict[str, Any]],
    parent_size_multiplier: int = DEFAULT_PARENT_SIZE_MULTIPLIER,
    show_progress: bool = True,
    use_enhanced_chunking: bool = True
) -> Tuple[List[Any], List[Any], List[Dict], List[IndexingResult]]:
    """
    Enhanced chunking that creates both retrieval chunks and parent chunks.
    
    Args:
        rag_service: RAG service instance
        documents: List of documents to chunk
        parent_size_multiplier: Parent chunks are this many times larger
        show_progress: Whether to show progress
        use_enhanced_chunking: Whether to use enhanced chunking service
        
    Returns:
        Tuple of (retrieval_chunks, parent_chunks, doc_chunk_info, failed_results)
    """
    logger.info(f"Chunking {len(documents)} documents with parent document support...")
    
    all_retrieval_chunks = []
    all_parent_chunks = []
    doc_chunk_info = []
    failed_results = []
    
    # Initialize enhanced chunking service if requested
    chunking_service = EnhancedChunkingService() if use_enhanced_chunking else None
    
    for i, doc in enumerate(documents):
        if show_progress and i % CHUNK_PROGRESS_INTERVAL == 0 and i > 0:
            logger.info(f"Chunking progress: {i}/{len(documents)} documents")
        
        try:
            if use_enhanced_chunking and chunking_service:
                # Use enhanced chunking with parent retrieval
                result = chunking_service.chunk_with_parent_retrieval(
                    doc['content'],
                    chunk_size=rag_service.config.chunk_size,
                    chunk_overlap=rag_service.config.chunk_overlap,
                    parent_size_multiplier=parent_size_multiplier
                )
                
                retrieval_chunks = result['chunks']
                parent_chunks = result['parent_chunks']
                
            else:
                # Fallback to basic chunking
                # Create retrieval chunks
                retrieval_chunks = await rag_service._chunk_document(
                    doc['content'],
                    rag_service.config.chunk_size,
                    rag_service.config.chunk_overlap,
                    rag_service.config.chunking_method
                )
                
                # Create parent chunks
                parent_chunk_size = rag_service.config.chunk_size * parent_size_multiplier
                parent_overlap = rag_service.config.chunk_overlap * parent_size_multiplier
                
                parent_chunks = await rag_service._chunk_document(
                    doc['content'],
                    parent_chunk_size,
                    parent_overlap,
                    rag_service.config.chunking_method
                )
                
                # Map retrieval chunks to parent chunks
                for r_idx, r_chunk in enumerate(retrieval_chunks):
                    r_start = r_chunk.get('start_char', 0)
                    r_end = r_chunk.get('end_char', len(r_chunk['text']))
                    
                    # Find parent chunk that contains this retrieval chunk
                    for p_idx, p_chunk in enumerate(parent_chunks):
                        p_start = p_chunk.get('start_char', 0)
                        p_end = p_chunk.get('end_char', len(p_chunk['text']))
                        
                        if p_start <= r_start and p_end >= r_end:
                            if 'metadata' not in r_chunk:
                                r_chunk['metadata'] = {}
                            r_chunk['metadata']['parent_chunk_index'] = p_idx
                            break
            
            # Store chunk information
            retrieval_start_idx = len(all_retrieval_chunks)
            parent_start_idx = len(all_parent_chunks)
            
            all_retrieval_chunks.extend(retrieval_chunks)
            all_parent_chunks.extend(parent_chunks)
            
            doc_chunk_info.append({
                'doc_idx': i,
                'retrieval_chunk_start': retrieval_start_idx,
                'retrieval_chunk_count': len(retrieval_chunks),
                'parent_chunk_start': parent_start_idx,
                'parent_chunk_count': len(parent_chunks),
                'retrieval_chunks': retrieval_chunks,
                'parent_chunks': parent_chunks
            })
            
        except Exception as e:
            logger.error(f"Failed to chunk document {doc.get('id', 'unknown')}: {e}")
            failed_results.append(IndexingResult(
                doc_id=doc.get('id', 'unknown'),
                chunks_created=0,
                time_taken=0,
                success=False,
                error=str(e)
            ))
            doc_chunk_info.append(None)
    
    logger.info(f"Created {len(all_retrieval_chunks)} retrieval chunks and {len(all_parent_chunks)} parent chunks")
    return all_retrieval_chunks, all_parent_chunks, doc_chunk_info, failed_results


async def generate_embeddings_for_parent_retrieval(
    rag_service,
    retrieval_chunks: List[Any],
    parent_chunks: List[Any],
    batch_size: int = 32,
    show_progress: bool = True,
    embed_parents: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray], List[int], List[int]]:
    """
    Generate embeddings for retrieval chunks and optionally parent chunks.
    
    Args:
        rag_service: RAG service instance
        retrieval_chunks: List of retrieval chunks
        parent_chunks: List of parent chunks
        batch_size: Batch size for embedding generation
        show_progress: Whether to show progress
        embed_parents: Whether to also embed parent chunks
        
    Returns:
        Tuple of (retrieval_embeddings, parent_embeddings, failed_retrieval_indices, failed_parent_indices)
    """
    # Extract text from chunks
    retrieval_texts = []
    for chunk in retrieval_chunks:
        if isinstance(chunk, dict):
            retrieval_texts.append(chunk.get('text', ''))
        else:
            retrieval_texts.append(str(chunk))
    
    # Generate embeddings for retrieval chunks
    logger.info(f"Generating embeddings for {len(retrieval_texts)} retrieval chunks...")
    retrieval_embeddings, failed_retrieval_indices = await generate_embeddings_batch(
        rag_service, retrieval_texts, batch_size, show_progress
    )
    
    # Generate embeddings for parent chunks if requested
    parent_embeddings = None
    failed_parent_indices = []
    
    if embed_parents and parent_chunks:
        parent_texts = []
        for chunk in parent_chunks:
            if isinstance(chunk, dict):
                parent_texts.append(chunk.get('text', ''))
            else:
                parent_texts.append(str(chunk))
        
        logger.info(f"Generating embeddings for {len(parent_texts)} parent chunks...")
        parent_embeddings, failed_parent_indices = await generate_embeddings_batch(
            rag_service, parent_texts, batch_size, show_progress
        )
    
    return retrieval_embeddings, parent_embeddings, failed_retrieval_indices, failed_parent_indices


async def store_documents_with_parents(
    rag_service,
    documents: List[Dict[str, Any]],
    doc_chunk_info: List[Dict],
    retrieval_embeddings: Union[np.ndarray, List[List[float]]],
    parent_chunks: List[Any],
    batch_start_time: float,
    failed_retrieval_indices: Optional[List[int]] = None,
    store_parent_chunks: bool = True
) -> List[IndexingResult]:
    """
    Store documents with parent document retrieval support.
    
    Args:
        rag_service: RAG service instance
        documents: Original documents
        doc_chunk_info: Chunking information for each document
        retrieval_embeddings: Embeddings for retrieval chunks
        parent_chunks: All parent chunks (for reference)
        batch_start_time: Start time of the batch operation
        failed_retrieval_indices: Indices of chunks that failed embedding
        store_parent_chunks: Whether to store parent chunk information
        
    Returns:
        List of IndexingResult for each document
    """
    logger.info("Storing documents with parent retrieval support...")
    
    results = []
    failed_indices_set = set(failed_retrieval_indices or [])
    
    # Create a global parent chunk lookup
    parent_chunk_lookup = {}
    if store_parent_chunks:
        for i, parent_chunk in enumerate(parent_chunks):
            # Create a unique ID for the parent chunk
            if isinstance(parent_chunk, dict):
                parent_text = parent_chunk.get('text', '')
            else:
                parent_text = str(parent_chunk)
            
            parent_id = hashlib.md5(parent_text.encode()).hexdigest()[:16]
            parent_chunk_lookup[i] = {
                'id': parent_id,
                'text': parent_text,
                'metadata': parent_chunk.get('metadata', {}) if isinstance(parent_chunk, dict) else {}
            }
    
    for doc_info in doc_chunk_info:
        if doc_info is None:
            continue
        
        doc_idx = doc_info['doc_idx']
        doc = documents[doc_idx]
        
        try:
            # Extract embeddings for this document's retrieval chunks
            chunk_start = doc_info['retrieval_chunk_start']
            chunk_count = doc_info['retrieval_chunk_count']
            doc_embeddings = retrieval_embeddings[chunk_start:chunk_start + chunk_count]
            
            # Filter out failed chunks
            valid_chunks = []
            valid_embeddings = []
            valid_chunk_indices = []
            failed_chunk_count = 0
            
            for j in range(chunk_count):
                global_chunk_idx = chunk_start + j
                if global_chunk_idx not in failed_indices_set:
                    chunk = doc_info['retrieval_chunks'][j]
                    valid_chunks.append(chunk)
                    valid_embeddings.append(doc_embeddings[j])
                    valid_chunk_indices.append(j)
                else:
                    failed_chunk_count += 1
            
            if not valid_chunks:
                logger.warning(f"All chunks failed for document {doc['id']}")
                results.append(IndexingResult(
                    doc_id=doc['id'],
                    chunks_created=0,
                    time_taken=time.time() - batch_start_time,
                    success=False,
                    error="All chunks failed embedding generation"
                ))
                continue
            
            # Prepare for storage
            chunk_ids = []
            chunk_texts = []
            chunk_metadata = []
            
            for j, chunk in zip(valid_chunk_indices, valid_chunks):
                chunk_id = f"{doc['id']}_chunk_{j}"
                chunk_ids.append(chunk_id)
                
                chunk_text = chunk.get('text', '') if isinstance(chunk, dict) else str(chunk)
                chunk_texts.append(chunk_text)
                
                # Create enriched metadata
                meta = {
                    **(doc.get('metadata', {})),
                    "doc_id": doc['id'],
                    "doc_title": doc.get('title', 'Untitled'),
                    "chunk_index": j,
                    "chunk_start": chunk.get("start_char", 0) if isinstance(chunk, dict) else 0,
                    "chunk_end": chunk.get("end_char", len(chunk_text)) if isinstance(chunk, dict) else len(chunk_text),
                    "chunk_size": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                    "text_preview": chunk_text[:200]
                }
                
                # Add parent chunk information
                if store_parent_chunks and isinstance(chunk, dict):
                    parent_idx = chunk.get('metadata', {}).get('parent_chunk_index')
                    if parent_idx is not None:
                        # Calculate global parent index
                        global_parent_idx = doc_info['parent_chunk_start'] + parent_idx
                        if global_parent_idx in parent_chunk_lookup:
                            parent_info = parent_chunk_lookup[global_parent_idx]
                            meta['parent_chunk_id'] = parent_info['id']
                            meta['parent_chunk_text'] = parent_info['text']
                            meta['has_parent_chunk'] = True
                
                # Add chunk type and structure information if available
                if isinstance(chunk, dict) and 'chunk_type' in chunk:
                    meta['chunk_type'] = chunk['chunk_type']
                    meta['chunk_level'] = chunk.get('level', 0)
                
                chunk_metadata.append(meta)
            
            # Store in vector database
            if NUMPY_AVAILABLE:
                embeddings_to_store = np.array(valid_embeddings)
            else:
                embeddings_to_store = valid_embeddings
            
            await rag_service._store_chunks(
                chunk_ids,
                embeddings_to_store,
                chunk_texts,
                chunk_metadata
            )
            
            # Update metrics
            rag_service._docs_indexed += 1
            rag_service._total_chunks_created += len(valid_chunks)
            
            elapsed = time.time() - batch_start_time
            
            # Create result
            error_msg = f"{failed_chunk_count} chunks failed" if failed_chunk_count > 0 else None
            
            results.append(IndexingResult(
                doc_id=doc['id'],
                chunks_created=len(valid_chunks),
                time_taken=elapsed,
                success=True,
                error=error_msg,
                metadata={
                    'parent_chunks_created': doc_info['parent_chunk_count'],
                    'parent_retrieval_enabled': store_parent_chunks
                }
            ))
            
        except Exception as e:
            logger.error(f"Failed to store document {doc.get('id', 'unknown')}: {e}")
            results.append(IndexingResult(
                doc_id=doc.get('id', 'unknown'),
                chunks_created=0,
                time_taken=time.time() - batch_start_time,
                success=False,
                error=str(e)
            ))
    
    return results


# Re-export the original helper for backward compatibility
from .indexing_helpers import generate_embeddings_batch