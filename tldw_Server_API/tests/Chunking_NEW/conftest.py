"""
Chunking Module Test Configuration and Fixtures

Provides fixtures for testing chunking functionality including
various text formats, chunking strategies, and multilingual support.
"""

import os
import json
import xml.etree.ElementTree as ET
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Generator, Optional
from unittest.mock import MagicMock, AsyncMock, Mock
import numpy as np
from datetime import datetime
import uuid

import pytest
from fastapi.testclient import TestClient

# Import actual chunking components for integration tests
from tldw_Server_API.app.core.Chunking.Chunk_Lib import (
    Chunker,
    DEFAULT_CHUNK_OPTIONS,
    improved_chunking_process
)
from tldw_Server_API.app.core.Chunking.chunker import ChunkerV2
from tldw_Server_API.app.core.Chunking.async_chunker import AsyncChunker
from tldw_Server_API.app.core.Chunking.multilingual import MultilingualChunker
from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import MediaDatabase

# =====================================================================
# Test Markers
# =====================================================================

def pytest_configure(config):
    """Register custom markers for test categorization."""
    config.addinivalue_line("markers", "unit: Unit tests with minimal mocking")
    config.addinivalue_line("markers", "integration: Integration tests with real components")
    config.addinivalue_line("markers", "property: Property-based tests")
    config.addinivalue_line("markers", "slow: Tests that take > 1 second")
    config.addinivalue_line("markers", "requires_tokenizer: Tests requiring tokenizer models")
    config.addinivalue_line("markers", "multilingual: Tests for multilingual support")
    config.addinivalue_line("markers", "strategy: Tests for specific chunking strategies")

# =====================================================================
# Environment Configuration
# =====================================================================

@pytest.fixture(scope="session")
def test_env_vars():
    """Set up test environment variables."""
    original_env = os.environ.copy()
    
    # Set test mode
    os.environ["TEST_MODE"] = "true"
    os.environ["DEFAULT_CHUNK_METHOD"] = "words"
    os.environ["DEFAULT_CHUNK_SIZE"] = "500"
    os.environ["DEFAULT_CHUNK_OVERLAP"] = "50"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

# =====================================================================
# Text Content Fixtures
# =====================================================================

@pytest.fixture
def sample_text_short():
    """Short text for basic testing."""
    return "This is a short test text."

@pytest.fixture
def sample_text_medium():
    """Medium-length text for standard testing."""
    return """This is the first sentence of our test document. It contains multiple sentences 
    for testing chunking functionality. The third sentence adds more content. 
    Here's a fourth sentence with different punctuation! And a fifth one with a question?
    
    This is a new paragraph. It should be detected by paragraph-based chunking. 
    The paragraph contains several sentences. Each sentence adds to the content.
    
    Finally, we have a third paragraph. This helps test multi-paragraph documents.
    The chunking should handle this appropriately."""

@pytest.fixture
def sample_text_long():
    """Long text for performance testing."""
    paragraphs = []
    for i in range(50):
        sentences = []
        for j in range(5):
            sentences.append(f"Paragraph {i}, sentence {j} contains test content.")
        paragraphs.append(" ".join(sentences))
    return "\n\n".join(paragraphs)

@pytest.fixture
def sample_text_code():
    """Text with code blocks for structure-aware testing."""
    return """# Documentation Title

This is a paragraph explaining the code below.

```python
def hello_world():
    print("Hello, World!")
    return True
```

After the code block, we have more text.
This tests structure-aware chunking.

```javascript
function greet(name) {
    console.log(`Hello, ${name}!`);
}
```

Final paragraph after the second code block."""

@pytest.fixture
def sample_text_markdown():
    """Markdown-formatted text."""
    return """# Main Title

## Section 1
This is the first section with some content.
It has multiple sentences for testing.

### Subsection 1.1
- Bullet point 1
- Bullet point 2
- Bullet point 3

## Section 2
Another section with different content.

1. Numbered item 1
2. Numbered item 2
3. Numbered item 3

### Subsection 2.1
**Bold text** and *italic text* for formatting.

> A blockquote for testing
> Multi-line blockquotes"""

# =====================================================================
# Multilingual Text Fixtures
# =====================================================================

@pytest.fixture
def multilingual_texts():
    """Text samples in different languages."""
    return {
        "en": "This is an English text for testing chunking functionality.",
        "es": "Este es un texto en español para probar la funcionalidad de fragmentación.",
        "fr": "Ceci est un texte en français pour tester la fonctionnalité de découpage.",
        "de": "Dies ist ein deutscher Text zum Testen der Chunking-Funktionalität.",
        "zh": "这是用于测试分块功能的中文文本。",
        "ja": "これはチャンキング機能をテストするための日本語のテキストです。",
        "ar": "هذا نص عربي لاختبار وظيفة التقسيم.",
        "ru": "Это русский текст для тестирования функции разбиения на части.",
        "ko": "청킹 기능을 테스트하기 위한 한국어 텍스트입니다.",
        "hi": "यह चंकिंग कार्यक्षमता का परीक्षण करने के लिए एक हिंदी पाठ है।"
    }

@pytest.fixture
def mixed_language_text():
    """Text with mixed languages."""
    return """This is an English introduction. 
    Voici une phrase en français. 
    这里有一些中文内容。
    Back to English for the conclusion."""

# =====================================================================
# Structured Data Fixtures
# =====================================================================

@pytest.fixture
def sample_json_data():
    """JSON data for structured chunking."""
    return {
        "metadata": {
            "title": "Test Document",
            "author": "Test Author",
            "date": "2024-01-01"
        },
        "chapters": [
            {
                "id": 1,
                "title": "Introduction",
                "content": "This is the introduction chapter with some content."
            },
            {
                "id": 2,
                "title": "Main Content",
                "content": "The main content chapter has more detailed information."
            },
            {
                "id": 3,
                "title": "Conclusion",
                "content": "The conclusion wraps up all the key points."
            }
        ]
    }

@pytest.fixture
def sample_xml_data():
    """XML data for structured chunking."""
    return """<?xml version="1.0" encoding="UTF-8"?>
    <document>
        <metadata>
            <title>XML Test Document</title>
            <author>Test Author</author>
        </metadata>
        <content>
            <section id="1">
                <title>First Section</title>
                <paragraph>This is the first paragraph.</paragraph>
                <paragraph>This is the second paragraph.</paragraph>
            </section>
            <section id="2">
                <title>Second Section</title>
                <paragraph>Content of the second section.</paragraph>
            </section>
        </content>
    </document>"""

@pytest.fixture
def sample_html_data():
    """HTML data for structured chunking."""
    return """<!DOCTYPE html>
    <html>
    <head><title>Test Page</title></head>
    <body>
        <h1>Main Title</h1>
        <p>First paragraph of content.</p>
        <h2>Subtitle</h2>
        <p>Second paragraph under subtitle.</p>
        <ul>
            <li>List item 1</li>
            <li>List item 2</li>
        </ul>
    </body>
    </html>"""

# =====================================================================
# Chunking Configuration Fixtures
# =====================================================================

@pytest.fixture
def basic_chunk_options():
    """Basic chunking options."""
    return {
        'method': 'words',
        'max_size': 100,
        'overlap': 20,
        'language': 'en',
        'adaptive': False
    }

@pytest.fixture
def advanced_chunk_options():
    """Advanced chunking options with all features."""
    return {
        'method': 'semantic',
        'max_size': 500,
        'overlap': 50,
        'language': 'auto',
        'adaptive': True,
        'multi_level': True,
        'semantic_similarity_threshold': 0.7,
        'semantic_overlap_sentences': 2,
        'preserve_structure': True,
        'tokenizer_name_or_path': 'gpt2'
    }

@pytest.fixture
def chunking_strategies():
    """All available chunking strategies."""
    return [
        'words',
        'sentences', 
        'paragraphs',
        'tokens',
        'semantic',
        'structure_aware',
        'rolling_summarize',
        'json',
        'xml',
        'ebook_chapters',
        'html_based',
        'markdown_structure',
        'code_blocks'
    ]

# =====================================================================
# Chunker Instance Fixtures
# =====================================================================

@pytest.fixture
def basic_chunker():
    """Create a basic Chunker instance."""
    return Chunker(
        chunk_method='words',
        max_chunk_size=100,
        chunk_overlap=20,
        language='en'
    )

@pytest.fixture
def advanced_chunker():
    """Create an advanced Chunker with all features."""
    return Chunker(
        chunk_method='semantic',
        max_chunk_size=500,
        chunk_overlap=50,
        language='auto',
        use_adaptive=True,
        multi_level=True
    )

@pytest.fixture
def async_chunker():
    """Create an AsyncChunker instance."""
    return AsyncChunker(
        chunk_method='words',
        max_chunk_size=100,
        chunk_overlap=20
    )

@pytest.fixture
def multilingual_chunker():
    """Create a MultilingualChunker instance."""
    return MultilingualChunker(
        default_language='en',
        auto_detect=True
    )

# =====================================================================
# Mock Tokenizer Fixtures
# =====================================================================

@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for token-based chunking."""
    tokenizer = MagicMock()
    tokenizer.encode = Mock(side_effect=lambda text: text.split())
    tokenizer.decode = Mock(side_effect=lambda tokens: ' '.join(tokens))
    tokenizer.tokenize = Mock(side_effect=lambda text: text.split())
    tokenizer.convert_tokens_to_string = Mock(side_effect=lambda tokens: ' '.join(tokens))
    return tokenizer

@pytest.fixture
def mock_sentence_transformer():
    """Mock sentence transformer for semantic chunking."""
    model = MagicMock()
    
    def mock_encode(texts, **kwargs):
        # Return random embeddings of appropriate dimension
        if isinstance(texts, str):
            return np.random.randn(384)
        return np.random.randn(len(texts), 384)
    
    model.encode = Mock(side_effect=mock_encode)
    return model

# =====================================================================
# Expected Output Fixtures
# =====================================================================

@pytest.fixture
def expected_word_chunks():
    """Expected output for word-based chunking."""
    return [
        {"text": "This is the first chunk of words that fits within the size limit.", 
         "metadata": {"chunk_index": 0, "method": "words"}},
        {"text": "The second chunk overlaps with the first and continues the text.",
         "metadata": {"chunk_index": 1, "method": "words"}},
    ]

@pytest.fixture
def expected_sentence_chunks():
    """Expected output for sentence-based chunking."""
    return [
        {"text": "This is the first sentence. This is the second sentence.",
         "metadata": {"chunk_index": 0, "method": "sentences"}},
        {"text": "The third sentence. The fourth sentence.",
         "metadata": {"chunk_index": 1, "method": "sentences"}},
    ]

# =====================================================================
# Database Fixtures
# =====================================================================

@pytest.fixture
def media_database() -> Generator[MediaDatabase, None, None]:
    """Create a test media database."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_media.db"
        db = MediaDatabase(db_path=str(db_path), client_id="test_client")
        db.initialize_db()
        yield db
        db.close()

@pytest.fixture
def populated_media_database(media_database) -> MediaDatabase:
    """Create a media database with test data."""
    # Add various media items for chunking
    media_database.add_media(
        title="Short Document",
        content="Short content for testing.",
        media_type="document"
    )
    
    media_database.add_media(
        title="Long Document",
        content=" ".join(["This is sentence number {}.".format(i) for i in range(100)]),
        media_type="document"
    )
    
    media_database.add_media(
        title="Code Document",
        content="```python\ndef test():\n    pass\n```\n\nSome text after code.",
        media_type="document"
    )
    
    return media_database

# =====================================================================
# Performance Testing Fixtures
# =====================================================================

@pytest.fixture
def large_document():
    """Generate a large document for performance testing."""
    # Generate ~1MB of text
    words = ["word"] * 200000
    return " ".join(words)

@pytest.fixture
def performance_metrics():
    """Track performance metrics during tests."""
    return {
        "chunking_time": [],
        "memory_usage": [],
        "chunk_counts": [],
        "processing_speed": []  # chunks per second
    }

# =====================================================================
# Security Testing Fixtures
# =====================================================================

@pytest.fixture
def malicious_inputs():
    """Various malicious inputs for security testing."""
    return {
        "script_injection": "<script>alert('XSS')</script>",
        "sql_injection": "'; DROP TABLE users; --",
        "path_traversal": "../../etc/passwd",
        "xxe_attack": '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><foo>&xxe;</foo>',
        "billion_laughs": '<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol"><!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">]><lolz>&lol2;</lolz>',
        "unicode_exploit": "\u202e\u0041\u0042\u0043",  # Right-to-left override
        "null_bytes": "test\x00hidden",
        "control_chars": "test\x1b[31mred\x1b[0m"
    }

# =====================================================================
# Template Fixtures
# =====================================================================

@pytest.fixture
def chunking_templates():
    """Predefined chunking templates."""
    return {
        "research_paper": {
            "method": "structure_aware",
            "preserve_sections": True,
            "preserve_citations": True
        },
        "code_documentation": {
            "method": "code_blocks",
            "preserve_structure": True,
            "language_detection": True
        },
        "ebook": {
            "method": "ebook_chapters",
            "preserve_chapters": True,
            "max_size": 1000
        },
        "conversation": {
            "method": "sentences",
            "preserve_speaker": True,
            "max_size": 500
        }
    }

# =====================================================================
# API Client Fixtures
# =====================================================================

@pytest.fixture
def test_client(test_env_vars):
    """Create a test client for the FastAPI app."""
    from tldw_Server_API.app.main import app
    return TestClient(app)

@pytest.fixture
def auth_headers():
    """Authentication headers for API requests."""
    return {
        "Authorization": "Bearer test-api-key",
        "Content-Type": "application/json"
    }

# =====================================================================
# Cleanup Fixtures
# =====================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    # Cleanup any temporary files or resources
    import gc
    gc.collect()