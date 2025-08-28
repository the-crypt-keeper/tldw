"""
Common types and type hints for test suite.
"""

from typing import Dict, List, Any, Optional, Union, TypedDict, Protocol
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class TestStatus(str, Enum):
    """Test execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class SearchMode(str, Enum):
    """Search modes for RAG"""
    FTS = "fts"
    VECTOR = "vector"
    HYBRID = "hybrid"


@dataclass
class TestDocument:
    """Test document for RAG tests"""
    id: str
    title: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    

@dataclass
class SearchResult:
    """Search result from RAG"""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str


class TestConfig(TypedDict):
    """Test configuration"""
    api_url: str
    api_key: str
    timeout: int
    retry_count: int
    test_data_path: str


class MockDatabase(Protocol):
    """Protocol for mock database implementations"""
    
    def connect(self) -> None:
        """Connect to database"""
        ...
    
    def execute(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute query"""
        ...
    
    def close(self) -> None:
        """Close connection"""
        ...


@dataclass
class RAGTestCase:
    """Test case for RAG functionality"""
    name: str
    query: str
    expected_results: int
    search_mode: SearchMode
    keywords: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None
    

@dataclass
class PerformanceMetrics:
    """Performance metrics for tests"""
    query_time: float
    total_results: int
    relevance_score: float
    memory_usage: float
    cpu_usage: float


class TestFixture:
    """Base class for test fixtures"""
    
    def __init__(self):
        self.setup_complete = False
        self.teardown_complete = False
    
    def setup(self) -> None:
        """Setup test fixture"""
        self.setup_complete = True
    
    def teardown(self) -> None:
        """Teardown test fixture"""
        self.teardown_complete = True


# Re-export commonly used types
__all__ = [
    'TestStatus',
    'SearchMode', 
    'TestDocument',
    'SearchResult',
    'TestConfig',
    'MockDatabase',
    'RAGTestCase',
    'PerformanceMetrics',
    'TestFixture',
]