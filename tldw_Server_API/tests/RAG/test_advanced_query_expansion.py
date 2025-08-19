"""
Tests for advanced query expansion functionality.
"""

import pytest
import asyncio
from typing import List, Dict, Any

from tldw_Server_API.app.core.RAG.rag_service.query_expansion import (
    ExpandedQuery, ExpansionStrategy,
    SynonymExpansion, MultiQueryGeneration,
    AcronymExpansion, DomainExpansion, EntityExpansion,
    HybridQueryExpansion, QueryExpansionRetriever
)
from tldw_Server_API.app.core.RAG.rag_service.types import (
    SearchResult, Document, DataSource
)


class TestAcronymExpansion:
    """Test acronym expansion and contraction."""
    
    @pytest.mark.asyncio
    async def test_expand_acronyms(self):
        """Test expanding acronyms to full forms."""
        expander = AcronymExpansion()
        query = "How to train ML models"
        
        result = await expander.expand(query)
        
        assert "ml" in result.acronym_expansions
        assert "machine learning" in result.acronym_expansions["ml"]
        
        # Check variations contain expanded form
        expanded_found = False
        for variation in result.variations:
            if "machine learning" in variation.lower():
                expanded_found = True
                break
        assert expanded_found
    
    @pytest.mark.asyncio
    async def test_contract_phrases(self):
        """Test contracting full phrases to acronyms."""
        expander = AcronymExpansion()
        query = "artificial intelligence research papers"
        
        result = await expander.expand(query)
        
        # Should find contractable phrase
        contracted_found = False
        for variation in result.variations:
            if "ai research" in variation.lower():
                contracted_found = True
                break
        assert contracted_found
    
    @pytest.mark.asyncio
    async def test_multiple_acronyms(self):
        """Test handling multiple acronyms in one query."""
        expander = AcronymExpansion()
        query = "RAG pipeline with LLM and API"
        
        result = await expander.expand(query)
        
        # Should expand multiple acronyms
        assert len(result.acronym_expansions) >= 3
        assert "rag" in result.acronym_expansions
        assert "llm" in result.acronym_expansions
        assert "api" in result.acronym_expansions
    
    @pytest.mark.asyncio
    async def test_custom_acronyms(self):
        """Test adding custom acronyms."""
        custom = {"tldw": ["too long didn't watch"]}
        expander = AcronymExpansion(custom_acronyms=custom)
        query = "TLDW summary generator"
        
        result = await expander.expand(query)
        
        assert "tldw" in result.acronym_expansions
        expanded_found = any("too long didn't watch" in v.lower() for v in result.variations)
        assert expanded_found


class TestDomainExpansion:
    """Test domain-specific term expansion."""
    
    @pytest.mark.asyncio
    async def test_domain_term_expansion(self):
        """Test expanding domain-specific terms."""
        expander = DomainExpansion()
        query = "document chunking strategies"
        
        result = await expander.expand(query)
        
        assert "chunking" in result.domain_terms
        assert any("segmentation" in v for v in result.variations)
    
    @pytest.mark.asyncio
    async def test_multiple_domain_terms(self):
        """Test multiple domain terms in query."""
        expander = DomainExpansion()
        query = "embedding generation for retrieval"
        
        result = await expander.expand(query)
        
        # Should find multiple domain terms
        assert len(result.domain_terms) >= 2
        assert "embedding" in result.domain_terms
        assert "retrieval" in result.domain_terms
    
    @pytest.mark.asyncio
    async def test_custom_domain_terms(self):
        """Test adding custom domain vocabulary."""
        custom = {"rag": ["retrieval augmented generation", "RAG system"]}
        expander = DomainExpansion(custom_terms=custom)
        query = "optimize rag performance"
        
        result = await expander.expand(query)
        
        assert "rag" in result.domain_terms
        assert len(result.variations) > 0


class TestEntityExpansion:
    """Test entity recognition and expansion."""
    
    @pytest.mark.asyncio
    async def test_date_entity_expansion(self):
        """Test recognizing and expanding date entities."""
        expander = EntityExpansion()
        query = "events on 2024-01-15"
        
        result = await expander.expand(query)
        
        assert len(result.entities) > 0
        assert "2024-01-15" in result.entities
        assert any("timeline" in v for v in result.variations)
    
    @pytest.mark.asyncio
    async def test_version_entity_expansion(self):
        """Test recognizing version numbers."""
        expander = EntityExpansion()
        query = "changes in v2.5.1"
        
        result = await expander.expand(query)
        
        assert len(result.entities) > 0
        assert any("changelog" in v for v in result.variations)
    
    @pytest.mark.asyncio
    async def test_email_entity_expansion(self):
        """Test recognizing email addresses."""
        expander = EntityExpansion()
        query = "messages from user@example.com"
        
        result = await expander.expand(query)
        
        assert "user@example.com" in result.entities
        assert any("conversation" in v for v in result.variations)
    
    @pytest.mark.asyncio
    async def test_named_entity_extraction(self):
        """Test extracting capitalized named entities."""
        expander = EntityExpansion()
        query = "OpenAI GPT models comparison"
        
        result = await expander.expand(query)
        
        assert "OpenAI" in result.entities
        assert "GPT" in result.entities


class TestHybridExpansion:
    """Test hybrid query expansion combining multiple strategies."""
    
    @pytest.mark.asyncio
    async def test_default_hybrid_expansion(self):
        """Test hybrid expansion with default strategies."""
        expander = HybridQueryExpansion()
        query = "how to create ML embeddings"
        
        result = await expander.expand(query)
        
        # Should have results from multiple strategies
        assert len(result.variations) > 0
        assert len(result.synonyms) > 0  # From synonym expansion
        assert "ml" in result.acronym_expansions  # From acronym expansion
        assert len(result.metadata["strategies_used"]) >= 3
    
    @pytest.mark.asyncio
    async def test_custom_strategies_hybrid(self):
        """Test hybrid expansion with custom strategy list."""
        strategies = [
            AcronymExpansion(),
            DomainExpansion()
        ]
        expander = HybridQueryExpansion(strategies=strategies)
        query = "RAG retrieval optimization"
        
        result = await expander.expand(query)
        
        assert len(result.metadata["strategies_used"]) == 2
        assert "AcronymExpansion" in result.metadata["strategies_used"]
        assert "DomainExpansion" in result.metadata["strategies_used"]
    
    @pytest.mark.asyncio
    async def test_deduplication(self):
        """Test that hybrid expansion deduplicates variations."""
        expander = HybridQueryExpansion()
        query = "search database"
        
        result = await expander.expand(query)
        
        # Check no duplicate variations
        variations_set = set(result.variations)
        assert len(variations_set) == len(result.variations)
    
    @pytest.mark.asyncio
    async def test_variation_limit(self):
        """Test that variations are limited appropriately."""
        expander = HybridQueryExpansion()
        query = "machine learning model training optimization techniques"
        
        result = await expander.expand(query)
        
        # Should respect the 10 variation limit
        assert len(result.variations) <= 10


class TestQueryExpansionRetriever:
    """Test query expansion retriever wrapper."""
    
    class MockRetriever:
        """Mock retriever for testing."""
        
        @property
        def source_type(self):
            return DataSource.MEDIA_DB
        
        async def retrieve(self, query: str, filters=None, top_k=10):
            # Return different results for different queries
            if "machine learning" in query.lower():
                docs = [Document(
                    id="doc1_ml",
                    content=f"ML content for: {query}",
                    metadata={},
                    score=0.9,
                    source=DataSource.MEDIA_DB
                )]
            elif "ml" in query.lower():
                docs = [Document(
                    id="doc2_ml",
                    content=f"ML acronym content for: {query}",
                    metadata={},
                    score=0.85,
                    source=DataSource.MEDIA_DB
                )]
            else:
                docs = [Document(
                    id="doc_generic",
                    content=f"Generic content for: {query}",
                    metadata={},
                    score=0.7,
                    source=DataSource.MEDIA_DB
                )]
            
            return SearchResult(
                documents=docs,
                query=query,
                search_type="mock"
            )
    
    @pytest.mark.asyncio
    async def test_expansion_retriever_basic(self):
        """Test basic query expansion retrieval."""
        base_retriever = self.MockRetriever()
        expansion_strategy = AcronymExpansion()
        
        retriever = QueryExpansionRetriever(
            base_retriever=base_retriever,
            expansion_strategy=expansion_strategy,
            max_variations=2
        )
        
        result = await retriever.retrieve("ML models", top_k=10)
        
        assert len(result.documents) > 0
        assert result.query_variations is not None
        assert len(result.query_variations) > 1
        assert "expansion" in result.metadata
    
    @pytest.mark.asyncio
    async def test_expansion_retriever_merge_union(self):
        """Test union merge strategy."""
        base_retriever = self.MockRetriever()
        expansion_strategy = AcronymExpansion()
        
        retriever = QueryExpansionRetriever(
            base_retriever=base_retriever,
            expansion_strategy=expansion_strategy,
            max_variations=2,
            merge_strategy="union"
        )
        
        result = await retriever.retrieve("ML research", top_k=10)
        
        # Should have results from both original and expanded queries
        assert len(result.documents) >= 2
        doc_ids = [doc.id for doc in result.documents]
        assert "doc2_ml" in doc_ids or "doc1_ml" in doc_ids
    
    @pytest.mark.asyncio
    async def test_expansion_retriever_scoring(self):
        """Test that documents appearing in multiple results get boosted scores."""
        base_retriever = self.MockRetriever()
        expansion_strategy = HybridQueryExpansion()
        
        retriever = QueryExpansionRetriever(
            base_retriever=base_retriever,
            expansion_strategy=expansion_strategy,
            max_variations=3
        )
        
        result = await retriever.retrieve("database search", top_k=10)
        
        # Documents should be sorted by score
        for i in range(len(result.documents) - 1):
            assert result.documents[i].score >= result.documents[i + 1].score


class TestIntegration:
    """Integration tests for query expansion."""
    
    @pytest.mark.asyncio
    async def test_complex_query_expansion(self):
        """Test expansion of a complex technical query."""
        expander = HybridQueryExpansion()
        query = "How to optimize RAG pipeline with LLM fine-tuning for v2.0"
        
        result = await expander.expand(query)
        
        # Should extract multiple types of information
        assert len(result.variations) > 0
        assert len(result.acronym_expansions) > 0  # RAG, LLM
        assert len(result.domain_terms) > 0  # fine-tuning, pipeline
        assert len(result.entities) > 0  # v2.0
        
        # Check specific expansions
        assert "rag" in result.acronym_expansions
        assert "llm" in result.acronym_expansions
        
        # Verify variations contain expanded forms
        variations_text = " ".join(result.variations).lower()
        assert "retrieval augmented generation" in variations_text or "large language model" in variations_text
    
    @pytest.mark.asyncio
    async def test_expansion_performance(self):
        """Test that expansion completes in reasonable time."""
        import time
        
        expander = HybridQueryExpansion()
        query = "Complex query with multiple ML, AI, NLP terms and entities like 2024-01-01"
        
        start = time.time()
        result = await expander.expand(query)
        duration = time.time() - start
        
        # Should complete quickly (under 1 second)
        assert duration < 1.0
        assert len(result.variations) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])