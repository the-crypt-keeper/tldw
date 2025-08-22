"""
Shared fixtures and utilities for contextual retrieval testing.

Provides reusable test data, mocks, and helper functions.
"""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
import json

from tldw_Server_API.app.core.RAG.rag_service.types import Document, DataSource
from tldw_Server_API.app.core.RAG.rag_service.functional_pipeline import RAGPipelineContext


# Sample document content for testing
SAMPLE_DOCUMENTS = {
    "machine_learning": """
    Machine learning is a subset of artificial intelligence (AI) that provides systems 
    the ability to automatically learn and improve from experience without being explicitly 
    programmed. Machine learning focuses on the development of computer programs that can 
    access data and use it to learn for themselves.
    """,
    
    "neural_networks": """
    Neural networks are a series of algorithms that endeavor to recognize underlying 
    relationships in a set of data through a process that mimics the way the human brain 
    operates. Neural networks can adapt to changing input so the network generates the 
    best possible result without needing to redesign the output criteria.
    """,
    
    "deep_learning": """
    Deep learning is a subset of machine learning that uses multi-layered neural networks, 
    called deep neural networks, to simulate the complex decision-making power of the human 
    brain. Some form of deep learning powers most of the artificial intelligence (AI) 
    applications in our lives today.
    """,
    
    "code_sample": """
    def train_model(data, labels, epochs=10):
        model = NeuralNetwork()
        for epoch in range(epochs):
            predictions = model.forward(data)
            loss = calculate_loss(predictions, labels)
            model.backward(loss)
        return model
    """,
    
    "table_data": """
    | Model | Accuracy | Speed |
    |-------|----------|-------|
    | GPT-4 | 95.2%    | Slow  |
    | BERT  | 89.1%    | Fast  |
    | T5    | 91.3%    | Med   |
    """
}


@pytest.fixture
def sample_config():
    """Provide a sample configuration for testing."""
    return {
        "embedding_config": {
            "enable_contextual_chunking": False,
            "contextual_llm_model": "gpt-3.5-turbo",
            "contextual_chunk_method": "situate_context",
            "default_model_id": "text-embedding-3-small",
            "models": {
                "text-embedding-3-small": {
                    "provider": "openai",
                    "dimension": 1536
                }
            }
        },
        "chunking_config": {
            "chunk_size": 500,
            "chunk_overlap": 100,
            "enable_contextual_retrieval": False,
            "context_window_size": 500,
            "include_parent_context": False
        },
        "rag_config": {
            "enable_parent_expansion": False,
            "parent_expansion_size": 500,
            "include_sibling_chunks": False,
            "semantic_cache_enabled": True,
            "cache_similarity_threshold": 0.85
        }
    }


@pytest.fixture
def mock_documents():
    """Create a set of mock documents with parent-child relationships."""
    documents = []
    
    # Create documents from ML content
    for i, (key, content) in enumerate(SAMPLE_DOCUMENTS.items()):
        # Split content into chunks
        chunks = content.split(". ")
        for j, chunk in enumerate(chunks):
            if chunk.strip():
                doc = Document(
                    id=f"{key}_chunk_{j}",
                    content=chunk.strip() + ".",
                    metadata={
                        "parent_id": key,
                        "chunk_index": j,
                        "total_chunks": len(chunks),
                        "chunk_type": "code" if key == "code_sample" else "table" if key == "table_data" else "text",
                        "title": key.replace("_", " ").title(),
                        "source_file": f"{key}.txt"
                    },
                    source=DataSource.MEDIA_DB,
                    score=0.9 - (j * 0.05)  # Decreasing scores for later chunks
                )
                documents.append(doc)
    
    return documents


@pytest.fixture
def mock_pipeline_context(mock_documents):
    """Create a mock RAG pipeline context."""
    context = RAGPipelineContext(
        query="What is machine learning?",
        config={
            "top_k": 10,
            "enable_cache": True,
            "expansion_strategies": ["acronym", "semantic"],
            "parent_expansion_size": 500,
            "include_siblings": False
        }
    )
    context.documents = mock_documents[:5]  # Start with subset
    return context


@pytest.fixture
def mock_chromadb_manager(sample_config):
    """Create a mock ChromaDBManager for testing."""
    manager = Mock()
    manager.user_id = "test_user"
    manager.embedding_config = sample_config["embedding_config"]
    manager.user_embedding_config = sample_config
    
    # Mock methods
    manager.get_or_create_collection = MagicMock(return_value=Mock())
    manager.situate_context = MagicMock(return_value="Contextual summary of the chunk")
    manager.process_and_store_content = AsyncMock()
    
    return manager


@pytest.fixture
def temp_config_file(sample_config):
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        # Write config in INI format
        f.write("[Embeddings]\n")
        f.write(f"enable_contextual_chunking = {str(sample_config['embedding_config']['enable_contextual_chunking']).lower()}\n")
        f.write(f"contextual_llm_model = {sample_config['embedding_config']['contextual_llm_model']}\n")
        f.write("\n[Chunking]\n")
        f.write(f"enable_contextual_retrieval = {str(sample_config['chunking_config']['enable_contextual_retrieval']).lower()}\n")
        f.write(f"context_window_size = {sample_config['chunking_config']['context_window_size']}\n")
        f.write("\n[RAG]\n")
        f.write(f"enable_parent_expansion = {str(sample_config['rag_config']['enable_parent_expansion']).lower()}\n")
        f.write(f"parent_expansion_size = {sample_config['rag_config']['parent_expansion_size']}\n")
        
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


class MockLLMAnalyzer:
    """Mock LLM analyzer for testing contextualization."""
    
    def __init__(self, response_template: str = "This chunk discusses {topic}"):
        self.response_template = response_template
        self.call_count = 0
        self.last_inputs = []
    
    def __call__(self, api_name: str, input_data: str, prompt: str, context: str, **kwargs):
        """Mock the analyze function."""
        self.call_count += 1
        self.last_inputs.append({
            "api_name": api_name,
            "input_data": input_data,
            "prompt": prompt,
            "context": context
        })
        
        # Generate a mock contextual response
        if "machine learning" in input_data.lower():
            topic = "machine learning concepts"
        elif "neural" in input_data.lower():
            topic = "neural network architecture"
        elif "deep learning" in input_data.lower():
            topic = "deep learning techniques"
        else:
            topic = "general AI concepts"
        
        return self.response_template.format(topic=topic)


@pytest.fixture
def mock_llm_analyzer():
    """Provide a mock LLM analyzer."""
    return MockLLMAnalyzer()


def create_test_documents(
    count: int = 10,
    with_parents: bool = True,
    chunk_types: Optional[List[str]] = None
) -> List[Document]:
    """
    Helper function to create test documents.
    
    Args:
        count: Number of documents to create
        with_parents: Whether to include parent_id metadata
        chunk_types: List of chunk types to use (cycles through them)
    
    Returns:
        List of test documents
    """
    if chunk_types is None:
        chunk_types = ["text", "code", "table", "header", "list"]
    
    documents = []
    for i in range(count):
        parent_id = f"parent_{i // 3}" if with_parents else None
        chunk_type = chunk_types[i % len(chunk_types)]
        
        doc = Document(
            id=f"doc_{i}",
            content=f"Test content {i}. " * 10,  # Some reasonable content
            metadata={
                "parent_id": parent_id,
                "chunk_index": i % 3 if with_parents else i,
                "chunk_type": chunk_type,
                "timestamp": f"2024-01-{i+1:02d}",
                "author": f"Author {i % 3}"
            },
            source=DataSource.MEDIA_DB,
            score=1.0 - (i * 0.05)  # Decreasing scores
        )
        documents.append(doc)
    
    return documents


def assert_documents_equal(doc1: Document, doc2: Document, ignore_score: bool = False):
    """
    Helper to assert two documents are equal.
    
    Args:
        doc1: First document
        doc2: Second document
        ignore_score: Whether to ignore score differences
    """
    assert doc1.id == doc2.id
    assert doc1.content == doc2.content
    assert doc1.metadata == doc2.metadata
    assert doc1.source == doc2.source
    if not ignore_score:
        assert abs(doc1.score - doc2.score) < 0.0001  # Float comparison tolerance


def create_mock_api_response(
    success: bool = True,
    media_id: int = 123,
    error: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a mock API response for testing.
    
    Args:
        success: Whether the operation succeeded
        media_id: The media ID to include
        error: Error message if not successful
    
    Returns:
        Mock API response dictionary
    """
    if success:
        return {
            "success": True,
            "media_id": media_id,
            "message": "Processing completed successfully",
            "chunks_created": 10,
            "contextualized": True
        }
    else:
        return {
            "success": False,
            "error": error or "Processing failed",
            "media_id": None
        }


class ConfigContextManager:
    """Context manager for temporarily modifying config values."""
    
    def __init__(self, config_dict: Dict[str, Any], updates: Dict[str, Any]):
        self.config_dict = config_dict
        self.updates = updates
        self.original_values = {}
    
    def __enter__(self):
        """Apply temporary config updates."""
        for key_path, value in self.updates.items():
            keys = key_path.split(".")
            current = self.config_dict
            
            # Navigate to the parent of the target key
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Store original value
            if keys[-1] in current:
                self.original_values[key_path] = current[keys[-1]]
            else:
                self.original_values[key_path] = None
            
            # Set new value
            current[keys[-1]] = value
        
        return self.config_dict
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original config values."""
        for key_path, original_value in self.original_values.items():
            keys = key_path.split(".")
            current = self.config_dict
            
            # Navigate to the parent of the target key
            for key in keys[:-1]:
                current = current[key]
            
            # Restore original value
            if original_value is None and keys[-1] in current:
                del current[keys[-1]]
            else:
                current[keys[-1]] = original_value


@pytest.fixture
def config_context_manager(sample_config):
    """Provide a context manager for config modifications."""
    def _create_manager(updates: Dict[str, Any]):
        return ConfigContextManager(sample_config, updates)
    return _create_manager