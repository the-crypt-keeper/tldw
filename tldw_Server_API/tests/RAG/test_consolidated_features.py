"""
Comprehensive tests for consolidated RAG features.

NOTE: This test file is for deprecated v2 functionality.
The v3 functional pipeline handles these features differently.
Tests are kept for reference but skipped in normal runs.

Tests for:
- Citation generation
- Parent document retrieval
- Connection pooling
- Query expansion
- Enhanced chunking
"""

import pytest

from tldw_Server_API.app.core.RAG import Document, DataSource
from tldw_Server_API.app.core.RAG.rag_service.citations import Citation
from tldw_Server_API.app.core.RAG.rag_service.types import CitationType
from tldw_Server_API.tests.types import SearchResult

# Skip all tests - deprecated v2 functionality
pytestmark = pytest.mark.skip(reason="Tests deprecated v2 features - v3 uses functional pipeline")
import asyncio
import sqlite3
from pathlib import Path
from typing import List, Dict, Any
import tempfile



# Mock retriever for testing wrappers
class MockRetriever:
    """Mock retriever for testing wrapper functionality."""
    
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.source = DataSource.MEDIA_DB
    
    @property
    def source_type(self) -> DataSource:
        return self.source
    
    async def retrieve(self, query: str, filters=None, top_k=10) -> SearchResult:
        # Simple mock: return all documents with mock scores
        for i, doc in enumerate(self.documents):
            doc.score = 1.0 - (i * 0.1)
        
        return SearchResult(
            documents=self.documents[:top_k],
            query=query,
            search_type="mock"
        )


class TestCitationGeneration:
    """Test citation generation functionality."""
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                id="doc1",
                content="Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
                metadata={"title": "ML Basics"},
                source=DataSource.MEDIA_DB,
                score=0.9
            ),
            Document(
                id="doc2",
                content="Deep learning uses neural networks with multiple layers to process complex patterns.",
                metadata={"title": "Deep Learning"},
                source=DataSource.MEDIA_DB,
                score=0.8
            )
        ]
    
    @pytest.mark.asyncio
    async def test_citation_aware_retriever(self, sample_documents):
        """Test CitationAwareRetriever wrapper."""
        base_retriever = MockRetriever(sample_documents)
        citation_retriever = CitationAwareRetriever(
            base_retriever,
            max_citation_length=100,
            citation_context_chars=20
        )
        
        result = await citation_retriever.retrieve("machine learning", top_k=5)
        
        assert len(result.documents) > 0
        assert len(result.citations) > 0
        
        # Check that citations were added to documents
        for doc in result.documents:
            if "machine learning" in doc.content.lower():
                assert len(doc.citations) > 0
                
                # Check citation properties
                for citation in doc.citations:
                    assert citation.document_id == doc.id
                    assert citation.confidence >= 0 and citation.confidence <= 1
                    assert citation.start_char >= 0
                    assert citation.end_char > citation.start_char
    
    def test_citation_types(self):
        """Test different citation types."""
        doc = Document(
            id="test",
            content="This is a test document with machine learning content.",
            metadata={},
            source=DataSource.MEDIA_DB
        )
        
        # Create different types of citations
        exact_citation = Citation(
            document_id="test",
            document_title="Test",
            chunk_id="test",
            text="machine learning",
            start_char=30,
            end_char=46,
            confidence=1.0,
            match_type=CitationType.EXACT
        )
        
        fuzzy_citation = Citation(
            document_id="test",
            document_title="Test",
            chunk_id="test",
            text="machine lerning",  # Typo
            start_char=30,
            end_char=45,
            confidence=0.85,
            match_type=CitationType.FUZZY
        )
        
        doc.add_citation(exact_citation)
        doc.add_citation(fuzzy_citation)
        
        assert len(doc.citations) == 2
        assert len(doc.get_citations_by_type(CitationType.EXACT)) == 1
        assert len(doc.get_citations_by_type(CitationType.FUZZY)) == 1
    
    def test_merge_citations(self):
        """Test citation merging functionality."""
        citations = [
            Citation(
                document_id="doc1",
                document_title="Doc 1",
                chunk_id="chunk1",
                text="test",
                start_char=0,
                end_char=4,
                confidence=0.9,
                match_type=CitationType.EXACT
            ),
            Citation(
                document_id="doc1",
                document_title="Doc 1",
                chunk_id="chunk1",
                text="test",
                start_char=0,
                end_char=4,
                confidence=0.95,  # Higher confidence
                match_type=CitationType.EXACT
            ),
            Citation(
                document_id="doc2",
                document_title="Doc 2",
                chunk_id="chunk2",
                text="another",
                start_char=10,
                end_char=17,
                confidence=0.8,
                match_type=CitationType.KEYWORD
            )
        ]
        
        merged = merge_citations(citations)
        
        assert len(merged) == 2  # Should deduplicate
        assert merged[0].confidence == 0.95  # Should keep higher confidence


class TestParentDocumentRetrieval:
    """Test parent document retrieval functionality."""
    
    @pytest.fixture
    def parent_child_documents(self):
        """Create parent and child documents."""
        parent = Document(
            id="parent1",
            content="This is a long parent document with multiple sections about machine learning and AI.",
            metadata={"title": "Complete Document"},
            source=DataSource.MEDIA_DB,
            children_ids=["child1", "child2"]
        )
        
        child1 = Document(
            id="child1",
            content="machine learning section",
            metadata={"title": "ML Section"},
            source=DataSource.MEDIA_DB,
            parent_id="parent1",
            chunk_index=0
        )
        
        child2 = Document(
            id="child2",
            content="AI section",
            metadata={"title": "AI Section"},
            source=DataSource.MEDIA_DB,
            parent_id="parent1",
            chunk_index=1
        )
        
        return parent, [child1, child2]
    
    @pytest.mark.asyncio
    async def test_parent_document_retriever(self, parent_child_documents):
        """Test ParentDocumentRetriever wrapper."""
        parent, children = parent_child_documents
        
        # Create parent store
        parent_store = {
            parent.id: parent,
            children[0].id: children[0],
            children[1].id: children[1]
        }
        
        # Base retriever returns children
        base_retriever = MockRetriever(children)
        
        # Wrap with parent retriever
        parent_retriever = ParentDocumentRetriever(
            base_retriever,
            parent_store,
            parent_size_multiplier=3,
            expand_to_siblings=True
        )
        
        result = await parent_retriever.retrieve("machine learning", top_k=5)
        
        # Should be EnhancedSearchResult
        assert isinstance(result, EnhancedSearchResult)
        assert len(result.parent_documents) > 0
        assert result.expanded_context is not None
        
        # Check parent document was retrieved
        parent_ids = [p.id for p in result.parent_documents]
        assert "parent1" in parent_ids
    
    @pytest.mark.asyncio
    async def test_hierarchical_retriever(self):
        """Test HierarchicalRetriever functionality."""
        # Create hierarchy
        hierarchy_store = {
            "chapter1": {
                "content": "Chapter 1: Introduction",
                "metadata": {"level": "chapter"},
                "ancestors": [],
                "descendants": ["section1.1", "section1.2"]
            },
            "section1.1": {
                "content": "Section 1.1: Basics",
                "metadata": {"level": "section"},
                "ancestors": ["chapter1"],
                "descendants": ["para1.1.1"]
            },
            "para1.1.1": {
                "content": "Detailed content about basics",
                "metadata": {"level": "paragraph"},
                "ancestors": ["chapter1", "section1.1"],
                "descendants": []
            }
        }
        
        # Create base retriever that returns paragraph
        para_doc = Document(
            id="para1.1.1",
            content="Detailed content about basics",
            metadata={"level": "paragraph"},
            source=DataSource.MEDIA_DB
        )
        base_retriever = MockRetriever([para_doc])
        
        # Wrap with hierarchical retriever
        hierarchical_retriever = HierarchicalRetriever(
            base_retriever,
            hierarchy_store,
            retrieval_depth=2
        )
        
        result = await hierarchical_retriever.retrieve("basics", top_k=5)
        
        # Should include ancestors
        doc_ids = [d.id for d in result.documents]
        assert "para1.1.1" in doc_ids  # Original
        # Note: In full implementation, ancestors would be fetched


class TestConnectionPooling:
    """Test database connection pooling."""
    
    def test_sqlite_connection_pool(self):
        """Test SQLite connection pool functionality."""
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
            pool = SQLiteConnectionPool(
                database_path=tmp.name,
                min_connections=2,
                max_connections=5,
                connection_timeout=1.0
            )
            
            # Test getting connections
            connections_used = []
            for i in range(3):
                with pool.get_connection() as conn:
                    connections_used.append(conn)
                    # Use connection
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    assert result[0] == 1
            
            # Check pool stats
            stats = pool.get_stats()
            assert stats["created_connections"] >= 2  # Min connections
            assert stats["created_connections"] <= 5  # Max connections
            
            pool.close_all()
    
    def test_connection_pool_manager(self):
        """Test ConnectionPoolManager for multiple databases."""
        manager = ConnectionPoolManager()
        
        with tempfile.NamedTemporaryFile(suffix=".db") as tmp1, \
             tempfile.NamedTemporaryFile(suffix=".db") as tmp2:
            
            # Get connections from different pools
            with manager.get_connection(tmp1.name) as conn1:
                cursor1 = conn1.cursor()
                cursor1.execute("CREATE TABLE test1 (id INTEGER)")
            
            with manager.get_connection(tmp2.name) as conn2:
                cursor2 = conn2.cursor()
                cursor2.execute("CREATE TABLE test2 (id INTEGER)")
            
            # Check that we have two pools
            stats = manager.get_all_stats()
            assert len(stats) == 2
            
            manager.close_all()


class TestQueryExpansion:
    """Test query expansion strategies."""
    
    def test_synonym_expansion(self):
        """Test synonym-based query expansion."""
        expander = SynonymExpansion()
        
        # Run expansion synchronously for testing
        expanded = asyncio.run(expander.expand("search document database"))
        
        assert expanded.original_query == "search document database"
        assert len(expanded.variations) > 0
        assert len(expanded.synonyms) > 0
        
        # Check that synonyms were found
        assert "search" in expanded.synonyms or "document" in expanded.synonyms or "database" in expanded.synonyms
    
    def test_multi_query_generation(self):
        """Test multi-query generation."""
        generator = MultiQueryGeneration()
        
        expanded = asyncio.run(generator.expand("how to implement RAG"))
        
        assert expanded.original_query == "how to implement RAG"
        assert len(expanded.variations) > 0
        
        # Check that variations were generated
        variations_text = " ".join(expanded.variations).lower()
        assert "implement" in variations_text or "rag" in variations_text
    
    @pytest.mark.asyncio
    async def test_query_expansion_retriever(self):
        """Test QueryExpansionRetriever wrapper."""
        docs = [
            Document(
                id="doc1",
                content="Information retrieval and search systems",
                metadata={},
                source=DataSource.MEDIA_DB
            )
        ]
        
        base_retriever = MockRetriever(docs)
        expansion_retriever = QueryExpansionRetriever(
            base_retriever,
            expansion_strategy=SynonymExpansion(),
            max_variations=2
        )
        
        result = await expansion_retriever.retrieve("search data", top_k=5)
        
        assert len(result.query_variations) >= 1  # At least original
        assert result.query_variations[0] == "search data"  # Original first
        assert "expansion" in result.metadata


class TestEnhancedChunking:
    """Test enhanced chunking functionality."""
    
    def test_structure_aware_chunking(self):
        """Test structure-aware chunking."""
        service = EnhancedChunkingService({
            'preserve_structure': True,
            'use_sentence_boundaries': True
        })
        
        text = """# Header 1
        This is the first section with some content.
        
        ## Header 2
        This is the second section with more detailed content that might need to be split.
        It has multiple sentences. Each sentence should be preserved when possible.
        """
        
        chunks = service.chunk_text(text, chunk_size=100, overlap=20)
        
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        
        # Check chunk properties
        for chunk in chunks:
            assert chunk.content
            assert chunk.chunk_index >= 0
            assert chunk.end_char >= chunk.start_char
    
    def test_pdf_artifact_cleaning(self):
        """Test PDF artifact cleaning."""
        service = EnhancedChunkingService({
            'clean_pdf_artifacts': True
        })
        
        # Text with PDF artifacts (no leading spaces for proper regex matching)
        pdf_text = """Page 1 of 10
This is some text with hyphen-
ation at line breaks.



Too many blank lines.

Common OCR errors: ﬁrst ﬂight ﬀort"""
        
        cleaned = service._clean_pdf_artifacts(pdf_text)
        
        assert "Page 1 of 10" not in cleaned
        assert "hyphenation" in cleaned  # Should be joined
        assert "first flight ffort" in cleaned  # OCR fixes (ligatures replaced)
        assert "\n\n\n" not in cleaned  # Excessive whitespace removed
    
    def test_code_block_preservation(self):
        """Test code block preservation."""
        service = EnhancedChunkingService({
            'preserve_code_blocks': True
        })
        
        text = """Here is some text.

```python
def hello():
    print("Hello, World!")
```

More text after code."""
        
        processed, code_blocks = service._extract_code_blocks(text)
        
        assert len(code_blocks) > 0
        assert "__CODE_BLOCK_0__" in processed
        assert "def hello():" in code_blocks[0]['content']
    
    def test_table_preservation(self):
        """Test table preservation."""
        service = EnhancedChunkingService({
            'preserve_tables': True
        })
        
        text = """Some text before table.
        
        | Column 1 | Column 2 |
        |----------|----------|
        | Data 1   | Data 2   |
        | Data 3   | Data 4   |
        
        Text after table."""
        
        processed, tables = service._extract_tables(text)
        
        assert len(tables) > 0
        assert "__TABLE_0__" in processed
        assert "Column 1" in tables[0]['content']