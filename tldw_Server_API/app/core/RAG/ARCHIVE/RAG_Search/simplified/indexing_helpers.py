"""
Helper functions for RAG indexing operations.

This module contains extracted functions to reduce complexity in the main RAG service.
"""

from loguru import logger
import time
from typing import List, Dict, Any, Tuple, Optional, Union

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


# Constants from rag_service
CHUNK_PROGRESS_INTERVAL = 10  # Show progress every N documents
EMBEDDING_PROGRESS_INTERVAL = 5  # Show progress every N batches


async def chunk_documents_batch(rag_service, documents: List[Dict[str, Any]], 
                               show_progress: bool = True) -> Tuple[List[Any], List[Dict]]:
    """
    Phase 1: Chunk all documents in a batch.
    
    Args:
        rag_service: RAG service instance with _chunk_document method
        documents: List of documents to chunk
        show_progress: Whether to show progress
        
    Returns:
        Tuple of (all_chunks, doc_chunk_info)
    """
    logger.info(f"Phase 1: Chunking {len(documents)} documents...")
    
    all_chunks = []
    doc_chunk_info = []
    failed_results = []
    
    for i, doc in enumerate(documents):
        if show_progress and i % CHUNK_PROGRESS_INTERVAL == 0 and i > 0:
            logger.info(f"Chunking progress: {i}/{len(documents)} documents")
        
        try:
            chunks = await rag_service._chunk_document(
                doc['content'],
                rag_service.config.chunk_size,
                rag_service.config.chunk_overlap,
                rag_service.config.chunking_method
            )
            
            chunk_start_idx = len(all_chunks)
            all_chunks.extend(chunks)
            
            doc_chunk_info.append({
                'doc_idx': i,
                'chunk_start': chunk_start_idx,
                'chunk_count': len(chunks),
                'chunks': chunks
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
            doc_chunk_info.append(None)  # Placeholder for failed document
    
    return all_chunks, doc_chunk_info, failed_results


async def generate_embeddings_batch(rag_service, chunk_texts: List[str], 
                                   batch_size: int = 32,
                                   show_progress: bool = True,
                                   retry_failed: bool = True) -> Tuple[np.ndarray, List[int]]:
    """
    Phase 2: Generate embeddings for all chunks in batches with partial failure recovery.
    
    Args:
        rag_service: RAG service instance with embeddings wrapper
        chunk_texts: List of chunk texts to embed
        batch_size: Batch size for embedding generation
        show_progress: Whether to show progress
        retry_failed: Whether to retry failed batches with smaller sizes
        
    Returns:
        Tuple of (embeddings array, list of failed chunk indices)
    """
    logger.info(f"Phase 2: Generating embeddings for {len(chunk_texts)} chunks in batches of {batch_size}...")
    
    all_embeddings = []
    failed_indices = []
    
    for i in range(0, len(chunk_texts), batch_size):
        batch = chunk_texts[i:i + batch_size]
        batch_indices = list(range(i, min(i + batch_size, len(chunk_texts))))
        
        if show_progress and (i // batch_size) % EMBEDDING_PROGRESS_INTERVAL == 0:
            progress = (i + len(batch)) / len(chunk_texts) * 100
            logger.info(f"Embedding progress: {progress:.1f}%")
        
        try:
            batch_embeddings = await rag_service.embeddings.create_embeddings_async(batch)
            all_embeddings.extend(batch_embeddings)
            
        except Exception as e:
            logger.error(f"Failed to embed batch at index {i}: {e}")
            
            if retry_failed and len(batch) > 1:
                # Try to process the batch in smaller chunks or individually
                logger.info(f"Retrying failed batch with individual processing...")
                
                for j, (text, idx) in enumerate(zip(batch, batch_indices)):
                    try:
                        # Try individual embedding
                        single_embedding = await rag_service.embeddings.create_embeddings_async([text])
                        all_embeddings.extend(single_embedding)
                    except Exception as e2:
                        logger.error(f"Failed to embed chunk {idx} individually: {e2}")
                        # Use zero embedding as fallback
                        if NUMPY_AVAILABLE:
                            all_embeddings.append(np.zeros(rag_service._embedding_dim))
                        else:
                            all_embeddings.append([0.0] * rag_service._embedding_dim)
                        failed_indices.append(idx)
            else:
                # Create zero embeddings as fallback for the entire batch
                if NUMPY_AVAILABLE:
                    fallback_embeddings = [np.zeros(rag_service._embedding_dim) for _ in batch]
                else:
                    fallback_embeddings = [[0.0] * rag_service._embedding_dim for _ in batch]
                all_embeddings.extend(fallback_embeddings)
                failed_indices.extend(batch_indices)
    
    if failed_indices:
        logger.warning(f"Failed to generate embeddings for {len(failed_indices)} chunks out of {len(chunk_texts)}")
    
    # Return as numpy array if available, otherwise as list
    if NUMPY_AVAILABLE:
        return np.array(all_embeddings), failed_indices
    else:
        return all_embeddings, failed_indices


async def store_documents_batch(rag_service, documents: List[Dict[str, Any]], 
                               doc_chunk_info: List[Dict], all_embeddings: Union[np.ndarray, List[List[float]]],
                               batch_start_time: float,
                               failed_embedding_indices: Optional[List[int]] = None) -> List[IndexingResult]:
    """
    Phase 3: Store documents with their embeddings in the vector database.
    
    Args:
        rag_service: RAG service instance
        documents: Original documents
        doc_chunk_info: Chunking information for each document
        all_embeddings: All generated embeddings
        batch_start_time: Start time of the batch operation
        failed_embedding_indices: List of chunk indices that failed embedding generation
        
    Returns:
        List of IndexingResult for each document
    """
    logger.info("Phase 3: Storing documents in vector database...")
    
    results = []
    failed_indices_set = set(failed_embedding_indices or [])
    
    for doc_info in doc_chunk_info:
        if doc_info is None:
            continue  # Skip failed documents
        
        doc_idx = doc_info['doc_idx']
        doc = documents[doc_idx]
        
        try:
            # Extract embeddings for this document
            chunk_start = doc_info['chunk_start']
            chunk_count = doc_info['chunk_count']
            doc_embeddings = all_embeddings[chunk_start:chunk_start + chunk_count]
            
            # Filter out chunks with failed embeddings
            valid_chunks = []
            valid_embeddings = []
            valid_chunk_indices = []
            failed_chunk_count = 0
            
            for j in range(chunk_count):
                global_chunk_idx = chunk_start + j
                if global_chunk_idx not in failed_indices_set:
                    valid_chunks.append(doc_info['chunks'][j])
                    valid_embeddings.append(doc_embeddings[j])
                    valid_chunk_indices.append(j)
                else:
                    failed_chunk_count += 1
            
            if not valid_chunks:
                logger.warning(f"All chunks failed for document {doc['id']}, skipping storage")
                results.append(IndexingResult(
                    doc_id=doc['id'],
                    chunks_created=0,
                    time_taken=time.time() - batch_start_time,
                    success=False,
                    error="All chunks failed embedding generation"
                ))
                continue
            
            # Prepare for storage (only valid chunks)
            chunk_ids = [f"{doc['id']}_chunk_{j}" for j in valid_chunk_indices]
            chunk_texts = [chunk['text'] for chunk in valid_chunks]
            chunk_metadata = []
            
            for j, chunk in zip(valid_chunk_indices, valid_chunks):
                meta = {
                    **(doc.get('metadata', {})),
                    "doc_id": doc['id'],
                    "doc_title": doc.get('title', 'Untitled'),
                    "chunk_index": j,
                    "chunk_start": chunk.get("start_char", 0),
                    "chunk_end": chunk.get("end_char", len(chunk["text"])),
                    "chunk_size": len(chunk["text"]),
                    "word_count": chunk.get("word_count", 0),
                    "text_preview": chunk["text"][:200]
                }
                chunk_metadata.append(meta)
            
            # Store in vector database
            # Convert embeddings if numpy is available
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
            
            # Create result with partial success indication if some chunks failed
            if failed_chunk_count > 0:
                logger.warning(f"Document {doc['id']}: {failed_chunk_count} out of {chunk_count} chunks failed")
                error_msg = f"{failed_chunk_count} chunks failed embedding generation"
            else:
                error_msg = None
            
            results.append(IndexingResult(
                doc_id=doc['id'],
                chunks_created=len(valid_chunks),
                time_taken=elapsed,
                success=True,  # Partial success is still success
                error=error_msg
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