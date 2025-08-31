"""
Unified RAG Pipeline - Single Function with All Features

This module provides a single, unified RAG pipeline function where ALL features
are accessible via explicit parameters. No configuration files, no presets,
just direct parameter control.

Design Philosophy:
- One function to rule them all
- Every feature is an optional parameter
- No hidden configuration
- Transparent execution flow
- Mix and match any features
"""

import asyncio
import time
import uuid
from typing import Dict, List, Any, Optional, Union, Literal
from dataclasses import dataclass, field
from loguru import logger

# Core types
from .types import Document, SearchResult, DataSource
from .metrics_collector import MetricsCollector, QueryMetrics

# Import all modules at module level to avoid 500ms overhead
try:
    from .quick_wins import spell_check_query, highlight_results as highlight_func, track_llm_cost
except ImportError:
    spell_check_query = None
    highlight_func = None
    track_llm_cost = None

try:
    from .query_expansion import (
        expand_acronyms,
        expand_synonyms,
        entity_recognition_expansion,
        domain_specific_expansion,
        multi_strategy_expansion
    )
except ImportError:
    expand_acronyms = None
    expand_synonyms = None
    entity_recognition_expansion = None
    domain_specific_expansion = None
    multi_strategy_expansion = None

try:
    from .semantic_cache import SemanticCache, AdaptiveCache
except ImportError:
    SemanticCache = None
    AdaptiveCache = None

try:
    from .database_retrievers import MultiDatabaseRetriever, RetrievalConfig
except ImportError:
    MultiDatabaseRetriever = None
    RetrievalConfig = None

try:
    from .security_filters import SecurityFilter, SensitivityLevel
except ImportError:
    SecurityFilter = None
    SensitivityLevel = None

try:
    from .table_serialization import TableProcessor
except ImportError:
    TableProcessor = None

try:
    from .enhanced_chunking_integration import (
        ChunkTypeFilter,
        ParentChunkExpander,
        SiblingChunkRetriever,
        HierarchicalChunkProcessor
    )
except ImportError:
    ChunkTypeFilter = None
    ParentChunkExpander = None
    SiblingChunkRetriever = None
    HierarchicalChunkProcessor = None

try:
    from .advanced_reranking import create_reranker, RerankingStrategy, RerankingConfig
except ImportError:
    create_reranker = None
    RerankingStrategy = None
    RerankingConfig = None

try:
    from .citations import DualCitationGenerator
except ImportError:
    DualCitationGenerator = None

try:
    from .generation import AnswerGenerator
except ImportError:
    AnswerGenerator = None

try:
    from .analytics_system import UnifiedFeedbackSystem
except ImportError:
    UnifiedFeedbackSystem = None

try:
    from .observability import Tracer
except ImportError:
    Tracer = None

try:
    from .performance_monitor import PerformanceMonitor
except ImportError:
    PerformanceMonitor = None


@dataclass
class UnifiedSearchResult:
    """Unified result structure for all RAG queries."""
    documents: List[Document]
    query: str
    expanded_queries: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    feedback_id: Optional[str] = None
    generated_answer: Optional[str] = None
    cache_hit: bool = False
    errors: List[str] = field(default_factory=list)
    security_report: Optional[Dict[str, Any]] = None
    total_time: float = 0.0


async def unified_rag_pipeline(
    # ========== REQUIRED PARAMETERS ==========
    query: str,
    
    # ========== DATA SOURCES ==========
    sources: List[str] = None,  # ["media_db", "notes", "characters", "chats"]
    media_db_path: Optional[str] = None,
    notes_db_path: Optional[str] = None,
    character_db_path: Optional[str] = None,
    
    # ========== SEARCH CONFIGURATION ==========
    search_mode: Literal["fts", "vector", "hybrid"] = "hybrid",
    hybrid_alpha: float = 0.7,  # 0=FTS only, 1=Vector only
    top_k: int = 10,
    min_score: float = 0.0,
    
    # ========== QUERY EXPANSION ==========
    expand_query: bool = False,
    expansion_strategies: List[str] = None,  # ["acronym", "synonym", "domain", "entity"]
    spell_check: bool = False,
    
    # ========== CACHING ==========
    enable_cache: bool = True,
    cache_threshold: float = 0.85,
    adaptive_cache: bool = True,
    
    # ========== FILTERING ==========
    keyword_filter: List[str] = None,  # Filter by these keywords
    
    # ========== SECURITY & PRIVACY ==========
    enable_security_filter: bool = False,
    detect_pii: bool = False,
    redact_pii: bool = False,
    sensitivity_level: Literal["public", "internal", "confidential", "restricted"] = "public",
    content_filter: bool = False,
    
    # ========== DOCUMENT PROCESSING ==========
    enable_table_processing: bool = False,
    table_method: Literal["markdown", "html", "hybrid"] = "markdown",
    
    # ========== CHUNKING & CONTEXT ==========
    enable_enhanced_chunking: bool = False,
    chunk_type_filter: List[str] = None,  # ["text", "code", "table", "list"]
    enable_parent_expansion: bool = False,
    parent_context_size: int = 500,
    include_sibling_chunks: bool = False,
    
    # ========== RERANKING ==========
    enable_reranking: bool = True,
    reranking_strategy: Literal["flashrank", "cross_encoder", "hybrid", "none"] = "flashrank",
    rerank_top_k: Optional[int] = None,  # Defaults to top_k if not specified
    
    # ========== CITATIONS ==========
    enable_citations: bool = False,
    citation_style: Literal["apa", "mla", "chicago", "harvard"] = "apa",
    include_page_numbers: bool = False,
    
    # ========== ANSWER GENERATION ==========
    enable_generation: bool = False,
    generation_model: Optional[str] = None,
    generation_prompt: Optional[str] = None,
    max_generation_tokens: int = 500,
    
    # ========== FEEDBACK ==========
    collect_feedback: bool = False,
    feedback_user_id: Optional[str] = None,
    apply_feedback_boost: bool = False,
    
    # ========== MONITORING & OBSERVABILITY ==========
    enable_monitoring: bool = False,
    enable_observability: bool = False,
    trace_id: Optional[str] = None,
    
    # ========== PERFORMANCE ==========
    enable_performance_analysis: bool = False,
    timeout_seconds: Optional[float] = None,
    
    # ========== QUICK WINS ==========
    highlight_results: bool = False,
    highlight_query_terms: bool = False,
    track_cost: bool = False,
    debug_mode: bool = False,
    
    # ========== BATCH PROCESSING ==========
    enable_batch: bool = False,
    batch_queries: List[str] = None,
    batch_concurrent: int = 5,
    
    # ========== RESILIENCE ==========
    enable_resilience: bool = False,
    retry_attempts: int = 3,
    circuit_breaker: bool = False,
    
    # ========== USER CONTEXT ==========
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    
    # ========== ADDITIONAL PARAMETERS ==========
    **kwargs: Any
) -> UnifiedSearchResult:
    """
    Unified RAG Pipeline - All features accessible via parameters.
    
    This is the ONE function for all RAG operations. Every feature is controlled
    by explicit parameters. No configuration files, no presets, just parameters.
    
    Args:
        query: The search query (required)
        sources: List of databases to search
        ... (see parameters above for all options)
        
    Returns:
        UnifiedSearchResult with all requested data
        
    Example:
        result = await unified_rag_pipeline(
            query="What is machine learning?",
            expand_query=True,
            expansion_strategies=["synonym", "acronym"],
            enable_citations=True,
            enable_reranking=True,
            reranking_strategy="hybrid"
        )
    """
    
    # Initialize result and timing
    start_time = time.time()
    result = UnifiedSearchResult(
        documents=[],
        query=query,
        metadata={"original_query": query}
    )
    
    # Initialize monitoring if requested
    metrics = None
    if enable_monitoring:
        metrics = QueryMetrics(query=query)
        metrics.start_time = start_time
    
    try:
        # ========== SPELL CHECK ==========
        if spell_check:
            if spell_check_query:
                spell_start = time.time()
                corrected = await spell_check_query(query)
                if corrected != query:
                    result.metadata["original_query"] = query
                    result.metadata["corrected_query"] = corrected
                    query = corrected
                result.timings["spell_check"] = time.time() - spell_start
            else:
                result.errors.append("Spell check module not available")
                logger.warning("Spell check requested but module not available")
        
        # ========== QUERY EXPANSION ==========
        expanded_queries = [query]
        if expand_query:
            expansion_start = time.time()
            if multi_strategy_expansion:
                strategies = expansion_strategies or ["acronym", "synonym"]
                
                # Use the imported expansion functions
                for strategy in strategies:
                    if strategy == "acronym" and expand_acronyms:
                        expanded = await expand_acronyms(query)
                        expanded_queries.extend(expanded)
                    elif strategy == "synonym" and expand_synonyms:
                        expanded = await expand_synonyms(query)
                        expanded_queries.extend(expanded)
                    elif strategy == "domain" and domain_specific_expansion:
                        expanded = await domain_specific_expansion(query)
                        expanded_queries.extend(expanded)
                if "entity" in strategies:
                    strategy_objects.append(EntityExpansion())
                
                # Remove duplicates
                expanded_queries = list(set(expanded_queries))
                result.expanded_queries = expanded_queries[1:]  # Exclude original query
                    
                result.timings["query_expansion"] = time.time() - expansion_start
                if metrics:
                    metrics.expansion_time = result.timings["query_expansion"]
            else:
                result.errors.append("Query expansion modules not available")
                logger.warning("Query expansion requested but modules not available")
        
        # ========== CACHE CHECK ==========
        cached_documents = None
        if enable_cache:
            cache_start = time.time()
            if AdaptiveCache and adaptive_cache:
                cache = AdaptiveCache(similarity_threshold=cache_threshold)
            elif SemanticCache:
                cache = SemanticCache(similarity_threshold=cache_threshold)
            else:
                cache = None
            
            if cache:
                
                # Check cache for all query variations
                for q in expanded_queries:
                    cached_result = await cache.find_similar(q)
                    if cached_result:
                        cached_query, similarity = cached_result
                        cached_documents = await cache.get(cached_query)
                        if cached_documents:
                            result.cache_hit = True
                            result.documents = cached_documents
                            result.metadata["cache_similarity"] = similarity
                            result.metadata["cached_query"] = cached_query
                            break
                            
            result.timings["cache_check"] = time.time() - cache_start
            if metrics:
                metrics.cache_lookup_time = result.timings["cache_check"]
        
        # ========== DOCUMENT RETRIEVAL ==========
        if not result.cache_hit:
            retrieval_start = time.time()
            try:
                if MultiDatabaseRetriever and RetrievalConfig:
                    
                    # Set up database paths
                    db_paths = {}
                    if media_db_path:
                        db_paths["media_db"] = media_db_path
                    if notes_db_path:
                        db_paths["notes_db"] = notes_db_path
                    if character_db_path:
                        db_paths["character_cards_db"] = character_db_path
                    
                    # Initialize retriever
                    retriever = MultiDatabaseRetriever(db_paths, user_id=user_id or "0")
                    
                    # Configure retrieval
                    config = RetrievalConfig(
                        max_results=top_k,
                        min_score=min_score,
                        use_fts=(search_mode in ["fts", "hybrid"]),
                        use_vector=(search_mode in ["vector", "hybrid"]),
                        include_metadata=True
                    )
                    
                    # Determine sources
                    if sources is None:
                        sources = ["media_db"]
                    
                    source_map = {
                        "media_db": DataSource.MEDIA_DB,
                        "media": DataSource.MEDIA_DB,
                        "notes": DataSource.NOTES,
                        "characters": DataSource.CHARACTER_CARDS,
                        "chats": DataSource.CHARACTER_CARDS
                    }
                    
                    data_sources = [source_map.get(s, DataSource.MEDIA_DB) for s in sources]
                    
                    # Retrieve documents
                    if search_mode == "hybrid" and hasattr(retriever, 'retrieve_hybrid'):
                        documents = await retriever.retrieve_hybrid(
                            query=query,
                            alpha=hybrid_alpha
                        )
                    else:
                        documents = await retriever.retrieve(
                            query=query,
                            sources=data_sources,
                            config=config
                        )
                        
                    result.documents = documents
                    result.metadata["sources_searched"] = sources
                    result.metadata["documents_retrieved"] = len(documents)
                    
                    result.timings["retrieval"] = time.time() - retrieval_start
                    if metrics:
                        metrics.retrieval_time = result.timings["retrieval"]
                        
            except Exception as e:
                result.errors.append(f"Document retrieval failed: {str(e)}")
                logger.error(f"Retrieval error: {e}")
        
        # ========== KEYWORD FILTERING ==========
        if keyword_filter and result.documents:
            filter_start = time.time()
            filtered_docs = []
            for doc in result.documents:
                content_lower = doc.content.lower()
                if any(keyword.lower() in content_lower for keyword in keyword_filter):
                    filtered_docs.append(doc)
            
            result.metadata["pre_filter_count"] = len(result.documents)
            result.documents = filtered_docs
            result.metadata["post_filter_count"] = len(filtered_docs)
            result.timings["keyword_filter"] = time.time() - filter_start
        
        # ========== SECURITY FILTERING ==========
        if enable_security_filter and result.documents:
            security_start = time.time()
            try:
                if SecurityFilter and SensitivityLevel:
                    security_filter = SecurityFilter()
                    
                    # Detect PII if requested
                    if detect_pii:
                        pii_report = await security_filter.detect_pii_batch(
                            [doc.content for doc in result.documents]
                        )
                        result.security_report = {"pii_detected": pii_report}
                    
                    # Filter by sensitivity
                    sensitivity_map = {
                        "public": SensitivityLevel.PUBLIC,
                        "internal": SensitivityLevel.INTERNAL,
                        "confidential": SensitivityLevel.CONFIDENTIAL,
                        "restricted": SensitivityLevel.RESTRICTED
                    }
                    
                    filtered_docs = await security_filter.filter_by_sensitivity(
                        result.documents,
                        max_level=sensitivity_map[sensitivity_level]
                    )
                    
                    # Redact PII if requested
                    if redact_pii:
                        for doc in filtered_docs:
                            doc.content = await security_filter.redact_pii(doc.content)
                    
                    result.documents = filtered_docs
                    result.timings["security_filter"] = time.time() - security_start
                    
            except ImportError:
                result.errors.append("Security filter module not available")
                logger.warning("Security filter requested but module not available")
            except Exception as e:
                result.errors.append(f"Security filter failed: {str(e)}")
                logger.error(f"Security filter error: {e}")
        
        # ========== TABLE PROCESSING ==========
        if enable_table_processing and result.documents:
            table_start = time.time()
            try:
                if TableProcessor:
                    processor = TableProcessor()
                    processed_docs = []
                    
                    for doc in result.documents:
                        processed = await processor.process_document(
                            doc.content,
                            method=table_method
                        )
                        doc.content = processed
                        processed_docs.append(doc)
                    
                    result.documents = processed_docs
                    result.timings["table_processing"] = time.time() - table_start
                    
            except ImportError:
                result.errors.append("Table processing module not available")
                logger.warning("Table processing requested but module not available")
        
        # ========== ENHANCED CHUNKING ==========
        if enable_enhanced_chunking and result.documents:
            chunking_start = time.time()
            try:
                if ChunkTypeFilter and ParentChunkExpander:
                    # Use the imported chunking modules
                    # TODO: Fix this incomplete code block
                    pass
                    
                result.timings["enhanced_chunking"] = time.time() - chunking_start
                
            except ImportError:
                result.errors.append("Enhanced chunking module not available")
                logger.warning("Enhanced chunking requested but module not available")
        
        # ========== RERANKING ==========
        if enable_reranking and result.documents and reranking_strategy != "none":
            rerank_start = time.time()
            try:
                if create_reranker and RerankingStrategy and RerankingConfig:
                    strategy_map = {
                        "flashrank": RerankingStrategy.FLASHRANK,
                        "cross_encoder": RerankingStrategy.CROSS_ENCODER,
                        "hybrid": RerankingStrategy.HYBRID
                    }
                    
                    rerank_config = RerankingConfig(
                        strategy=strategy_map[reranking_strategy],
                        top_k=rerank_top_k or top_k,
                        model_name=None  # Use default
                    )
                    
                    reranker = create_reranker(rerank_config)
                    reranked = await reranker.rerank(query, result.documents)
                    result.documents = reranked[:rerank_top_k or top_k]
                    
                    result.timings["reranking"] = time.time() - rerank_start
                    if metrics:
                        metrics.reranking_time = result.timings["reranking"]
                        
                else:
                    result.errors.append("Reranking module not available")
                    logger.warning("Reranking requested but module not available")
            except Exception as e:
                result.errors.append(f"Reranking failed: {str(e)}")
                logger.error(f"Reranking error: {e}")
        
        # ========== CITATION GENERATION ==========
        if enable_citations and result.documents:
            citation_start = time.time()
            try:
                if DualCitationGenerator:
                    generator = DualCitationGenerator()
                    citations = await generator.generate_citations(
                        documents=result.documents,
                        query=query,
                        style=citation_style,
                        include_metadata=include_page_numbers
                    )
                    
                    result.citations = [
                        {
                            "text": c.text,
                            "source": c.document_title,
                            "confidence": c.confidence,
                            "type": c.match_type.value
                        }
                        for c in citations
                    ]
                    
                    result.timings["citation_generation"] = time.time() - citation_start
                    
            except ImportError:
                result.errors.append("Citation module not available")
                logger.warning("Citations requested but module not available")
            except Exception as e:
                result.errors.append(f"Citation generation failed: {str(e)}")
                logger.error(f"Citation error: {e}")
        
        # ========== ANSWER GENERATION ==========
        if enable_generation and result.documents:
            generation_start = time.time()
            try:
                if AnswerGenerator:
                    generator = AnswerGenerator(model=generation_model)
                    
                    # Prepare context from documents
                    context = "\n\n".join([doc.content for doc in result.documents[:5]])
                    
                    answer = await generator.generate(
                        query=query,
                        context=context,
                        prompt_template=generation_prompt,
                        max_tokens=max_generation_tokens
                    )
                    
                    result.generated_answer = answer
                    result.timings["answer_generation"] = time.time() - generation_start
                    if metrics:
                        metrics.generation_time = result.timings["answer_generation"]
                        
            except ImportError:
                result.errors.append("Generation module not available")
                logger.warning("Answer generation requested but module not available")
            except Exception as e:
                result.errors.append(f"Answer generation failed: {str(e)}")
                logger.error(f"Generation error: {e}")
        
        # ========== USER FEEDBACK ==========
        if collect_feedback:
            feedback_start = time.time()
            try:
                if UnifiedFeedbackSystem:
                    collector = UnifiedFeedbackSystem()
                    feedback_id = str(uuid.uuid4())
                    
                    # Record query for feedback
                    await collector.record_query(
                        query_id=feedback_id,
                        query=query,
                        results=[doc.id for doc in result.documents] if result.documents else [],
                        user_id=feedback_user_id
                    )
                    
                    result.feedback_id = feedback_id
                    result.metadata["feedback_enabled"] = True
                    
                    # Apply feedback boost if requested
                    if apply_feedback_boost and result.documents:
                        boosted = await collector.apply_feedback_boost(
                            result.documents,
                            query
                        )
                        result.documents = boosted
                    
                    result.timings["feedback"] = time.time() - feedback_start
                    
            except ImportError:
                result.errors.append("Feedback module not available")
                logger.warning("Feedback requested but module not available")
            except Exception as e:
                result.errors.append(f"Feedback system failed: {str(e)}")
                logger.error(f"Feedback error: {e}")
        
        # ========== RESULT HIGHLIGHTING ==========
        if highlight_results and result.documents:
            highlight_start = time.time()
            try:
                if highlight_func:
                    for doc in result.documents:
                        doc.content = await highlight_func(
                            doc.content,
                            query if highlight_query_terms else None
                        )
                    
                    result.timings["highlighting"] = time.time() - highlight_start
                    
            except ImportError:
                result.errors.append("Highlighting module not available")
                logger.warning("Highlighting requested but module not available")
        
        # ========== COST TRACKING ==========
        if track_cost:
            try:
                if track_llm_cost:
                    # Calculate estimated cost
                    total_tokens = sum(len(doc.content.split()) for doc in result.documents)
                    cost = await track_llm_cost(
                        model=generation_model or "gpt-3.5-turbo",
                        input_tokens=total_tokens,
                        output_tokens=len(result.generated_answer.split()) if result.generated_answer else 0
                    )
                    
                    result.metadata["estimated_cost"] = cost
                    
            except ImportError:
                result.errors.append("Cost tracking module not available")
        
        # ========== CACHE STORAGE ==========
        if enable_cache and not result.cache_hit and result.documents:
            try:
                # Store in cache for future use
                if adaptive_cache and AdaptiveCache:
                    cache = AdaptiveCache(similarity_threshold=cache_threshold)
                elif SemanticCache:
                    cache = SemanticCache(similarity_threshold=cache_threshold)
                else:
                    cache = None
                
                await cache.add(query, result.documents)
                
            except Exception as e:
                logger.error(f"Cache storage error: {e}")
        
        # ========== OBSERVABILITY ==========
        if enable_observability:
            try:
                if Tracer:
                    tracer = Tracer()
                    await tracer.trace(
                        trace_id=trace_id or str(uuid.uuid4()),
                        operation="unified_rag_pipeline",
                        query=query,
                        timings=result.timings,
                        metadata=result.metadata
                    )
                    
            except ImportError:
                result.errors.append("Observability module not available")
        
        # ========== PERFORMANCE ANALYSIS ==========
        if enable_performance_analysis:
            try:
                if PerformanceMonitor:
                    monitor = PerformanceMonitor()
                    analysis = await monitor.analyze(
                        timings=result.timings,
                        document_count=len(result.documents),
                        cache_hit=result.cache_hit
                    )
                    
                    result.metadata["performance_analysis"] = analysis
                    
            except ImportError:
                result.errors.append("Performance monitor not available")
        
    except Exception as e:
        result.errors.append(f"Pipeline error: {str(e)}")
        logger.error(f"Unified pipeline error: {e}")
    
    finally:
        # Calculate total time
        result.total_time = time.time() - start_time
        result.timings["total"] = result.total_time
        
        # Finalize metrics if monitoring
        if metrics:
            metrics.total_time = result.total_time
            metrics.cache_hit = result.cache_hit
            metrics.documents_retrieved = len(result.documents)
            
            if enable_monitoring:
                try:
                    collector = MetricsCollector()
                    await collector.record_query_metrics(metrics)
                except Exception as e:
                    logger.error(f"Metrics recording error: {e}")
        
        # Debug output if requested
        if debug_mode:
            logger.debug(f"Query: {query}")
            logger.debug(f"Documents found: {len(result.documents)}")
            logger.debug(f"Cache hit: {result.cache_hit}")
            logger.debug(f"Timings: {result.timings}")
            logger.debug(f"Errors: {result.errors}")
    
    return result


# ========== BATCH PROCESSING WRAPPER ==========
async def unified_batch_pipeline(
    queries: List[str],
    max_concurrent: int = 5,
    **kwargs
) -> List[UnifiedSearchResult]:
    """
    Process multiple queries concurrently using the unified pipeline.
    
    Args:
        queries: List of queries to process
        max_concurrent: Maximum concurrent executions
        **kwargs: All parameters supported by unified_rag_pipeline
        
    Returns:
        List of results in the same order as queries
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(query: str) -> UnifiedSearchResult:
        async with semaphore:
            return await unified_rag_pipeline(query=query, **kwargs)
    
    tasks = [process_with_semaphore(q) for q in queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Convert exceptions to error results
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            error_result = UnifiedSearchResult(
                documents=[],
                query=queries[i],
                errors=[str(result)]
            )
            final_results.append(error_result)
        else:
            final_results.append(result)
    
    return final_results


# ========== SIMPLE CONVENIENCE WRAPPERS ==========

async def simple_search(query: str, top_k: int = 10) -> List[Document]:
    """
    Simple search wrapper for basic use cases.
    
    Args:
        query: Search query
        top_k: Number of results
        
    Returns:
        List of documents
    """
    result = await unified_rag_pipeline(
        query=query,
        top_k=top_k,
        expand_query=False,
        enable_cache=True,
        enable_reranking=True
    )
    return result.documents


async def advanced_search(
    query: str,
    with_citations: bool = True,
    with_answer: bool = True,
    **kwargs
) -> UnifiedSearchResult:
    """
    Advanced search with commonly used features enabled.
    
    Args:
        query: Search query
        with_citations: Enable citation generation
        with_answer: Enable answer generation
        **kwargs: Additional parameters
        
    Returns:
        Full search result
    """
    return await unified_rag_pipeline(
        query=query,
        expand_query=True,
        expansion_strategies=["acronym", "synonym", "domain"],
        enable_cache=True,
        enable_reranking=True,
        reranking_strategy="hybrid",
        enable_citations=with_citations,
        enable_generation=with_answer,
        enable_table_processing=True,
        enable_performance_analysis=True,
        **kwargs
    )