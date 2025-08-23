# test_data.py
# Description: Test data generators for end-to-end tests
#
"""
Test Data Generators
--------------------

Provides sample data, content generators, and templates for comprehensive
end-to-end testing.
"""

import random
import string
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional


class TestDataGenerator:
    """Generate test data for various API endpoints."""
    
    @staticmethod
    def random_string(length: int = 10, prefix: str = "") -> str:
        """Generate a random string."""
        chars = string.ascii_lowercase + string.digits
        random_part = ''.join(random.choice(chars) for _ in range(length))
        return f"{prefix}{random_part}" if prefix else random_part
    
    @staticmethod
    def random_email() -> str:
        """Generate a random email address."""
        return f"test_{TestDataGenerator.random_string(8)}@example.com"
    
    @staticmethod
    def random_username() -> str:
        """Generate a random username."""
        return f"user_{TestDataGenerator.random_string(8)}"
    
    # Sample content generators
    @staticmethod
    def sample_text_content() -> str:
        """Generate sample text content for testing."""
        return """
        # Sample Document for E2E Testing
        
        This is a test document created for end-to-end testing of the tldw_server API.
        
        ## Introduction
        The purpose of this document is to provide realistic content for testing various
        features including transcription, analysis, and search functionality.
        
        ## Key Topics
        - Natural Language Processing
        - Machine Learning Applications
        - Document Analysis
        - Information Retrieval
        
        ## Technical Details
        This system uses advanced AI techniques to process and understand content.
        The main components include:
        1. Text extraction and parsing
        2. Semantic analysis
        3. Entity recognition
        4. Summarization
        
        ## Conclusion
        This test content should be sufficient for validating the core functionality
        of the system during end-to-end testing.
        
        Keywords: testing, documentation, AI, NLP, machine learning
        """
    
    @staticmethod
    def sample_transcript() -> str:
        """Generate sample transcript content."""
        return """
        [00:00:00] Speaker 1: Welcome to our discussion about artificial intelligence.
        [00:00:05] Speaker 2: Thank you for having me. I'm excited to talk about recent developments.
        [00:00:10] Speaker 1: Let's start with machine learning. What are the latest trends?
        [00:00:15] Speaker 2: Well, we're seeing significant advances in transformer models.
        [00:00:20] Speaker 2: These models are revolutionizing natural language processing.
        [00:00:25] Speaker 1: How do they compare to traditional approaches?
        [00:00:30] Speaker 2: The key difference is their ability to understand context.
        [00:00:35] Speaker 2: They can process entire sequences simultaneously.
        [00:00:40] Speaker 1: What about practical applications?
        [00:00:45] Speaker 2: We're seeing them used in translation, summarization, and question answering.
        [00:00:50] Speaker 1: Fascinating. Thank you for this insight.
        """
    
    @staticmethod
    def sample_web_content() -> Dict[str, str]:
        """Generate sample web content data."""
        return {
            "url": "https://example.com/test-article",
            "title": "Understanding AI and Machine Learning",
            "content": """
            <h1>Understanding AI and Machine Learning</h1>
            <p>Artificial Intelligence (AI) and Machine Learning (ML) are transforming industries.</p>
            <h2>What is AI?</h2>
            <p>AI refers to systems that can perform tasks that typically require human intelligence.</p>
            <h2>Machine Learning Basics</h2>
            <p>ML is a subset of AI that enables systems to learn from data.</p>
            <ul>
                <li>Supervised Learning</li>
                <li>Unsupervised Learning</li>
                <li>Reinforcement Learning</li>
            </ul>
            """,
            "metadata": {
                "author": "Test Author",
                "date": datetime.now().isoformat(),
                "tags": ["AI", "ML", "Technology"]
            }
        }
    
    @staticmethod
    def sample_chat_messages() -> List[Dict[str, str]]:
        """Generate sample chat messages."""
        return [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."},
            {"role": "user", "content": "Can you give me an example?"},
            {"role": "assistant", "content": "Sure! Email spam filters are a common example. They learn to identify spam by analyzing patterns in emails marked as spam by users."}
        ]
    
    @staticmethod
    def sample_character_card() -> Dict[str, Any]:
        """Generate a sample character card."""
        return {
            "name": "Professor AI",
            "description": "An knowledgeable AI professor who loves teaching about technology",
            "personality": "Friendly, patient, and enthusiastic about learning",
            "scenario": "You are discussing advanced AI concepts with a professor",
            "first_mes": "Hello! I'm Professor AI. I'm here to help you understand artificial intelligence and machine learning. What would you like to know?",
            "mes_example": """
            {{user}}: What is deep learning?
            {{char}}: Excellent question! Deep learning is a subset of machine learning that uses neural networks with multiple layers - hence 'deep' - to progressively extract higher-level features from raw input. Think of it like teaching a computer to see: first it learns edges, then shapes, then objects, and finally complex scenes!
            """,
            "creator": "E2E Test Suite",
            "version": "1.0",
            "tags": ["educational", "AI", "technology"],
            "system_prompt": "You are Professor AI, an enthusiastic and knowledgeable educator specializing in artificial intelligence and machine learning.",
            "post_history_instructions": "Remember to maintain an educational and encouraging tone.",
            "alternate_greetings": [
                "Welcome to AI 101! Ready to explore the fascinating world of artificial intelligence?",
                "Greetings! I'm Professor AI. What aspect of machine learning shall we explore today?"
            ],
            "avatar": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        }
    
    @staticmethod
    def sample_note() -> Dict[str, Any]:
        """Generate a sample note."""
        return {
            "title": f"Test Note {TestDataGenerator.random_string(5)}",
            "content": """
            # Research Notes
            
            ## Key Findings
            - AI systems are becoming more sophisticated
            - Natural language processing has made significant advances
            - Machine learning models require large amounts of data
            
            ## Important Concepts
            1. **Neural Networks**: Computational models inspired by the human brain
            2. **Deep Learning**: ML using multi-layered neural networks
            3. **Transformers**: Architecture that revolutionized NLP
            
            ## References
            - Paper 1: "Attention Is All You Need"
            - Paper 2: "BERT: Pre-training of Deep Bidirectional Transformers"
            
            ## Next Steps
            - Review transformer architecture in detail
            - Experiment with fine-tuning models
            - Document findings
            """,
            "keywords": ["AI", "research", "machine learning", "notes"],
            "metadata": {
                "created": datetime.now().isoformat(),
                "category": "research",
                "importance": "high"
            }
        }
    
    @staticmethod
    def sample_prompt_template() -> Dict[str, str]:
        """Generate a sample prompt template."""
        return {
            "name": f"Test Prompt {TestDataGenerator.random_string(5)}",
            "description": "A test prompt for summarization tasks",
            "content": """
            Please provide a comprehensive summary of the following text:

            {text}

            Summary requirements:
            - Keep it under 200 words
            - Include main points
            - Maintain factual accuracy
            - Use clear, concise language
            
            Summary:
            """,
            "category": "summarization",
            "tags": ["summary", "analysis", "test"]
        }
    
    @staticmethod
    def sample_evaluation_criteria() -> Dict[str, Any]:
        """Generate sample evaluation criteria."""
        return {
            "name": "Response Quality Evaluation",
            "metrics": [
                {
                    "name": "relevance",
                    "description": "How relevant is the response to the query",
                    "weight": 0.3,
                    "scale": [1, 5]
                },
                {
                    "name": "accuracy",
                    "description": "Factual accuracy of the response",
                    "weight": 0.3,
                    "scale": [1, 5]
                },
                {
                    "name": "completeness",
                    "description": "How complete is the response",
                    "weight": 0.2,
                    "scale": [1, 5]
                },
                {
                    "name": "clarity",
                    "description": "Clarity and readability",
                    "weight": 0.2,
                    "scale": [1, 5]
                }
            ]
        }
    
    @staticmethod
    def sample_media_metadata() -> Dict[str, Any]:
        """Generate sample media metadata."""
        return {
            "title": f"Test Media {TestDataGenerator.random_string(5)}",
            "description": "Test media file for E2E testing",
            "author": "E2E Test Suite",
            "duration": random.randint(60, 600),  # seconds
            "tags": ["test", "e2e", "sample"],
            "metadata": {
                "format": "mp4",
                "resolution": "1920x1080",
                "bitrate": "5000kbps",
                "created": datetime.now().isoformat()
            }
        }
    
    @staticmethod
    def sample_search_queries() -> List[str]:
        """Generate sample search queries."""
        return [
            "machine learning",
            "artificial intelligence applications",
            "neural network architecture",
            "deep learning transformer models",
            "natural language processing",
            "computer vision techniques",
            "reinforcement learning algorithms",
            "data preprocessing methods",
            "model evaluation metrics",
            "AI ethics and bias"
        ]
    
    @staticmethod
    def sample_keywords() -> List[str]:
        """Generate sample keywords."""
        return [
            "AI", "ML", "NLP", "deep learning", "neural networks",
            "transformers", "BERT", "GPT", "computer vision", "robotics",
            "data science", "algorithms", "optimization", "statistics",
            "python", "tensorflow", "pytorch", "research", "innovation"
        ]
    
    @staticmethod
    def sample_urls() -> List[str]:
        """Generate sample URLs for web scraping tests."""
        return [
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "https://example.com/test-article",
            "https://www.example.org/sample-content",
            "https://test.example.net/documentation"
        ]
    
    @staticmethod
    def generate_test_batch(count: int = 5) -> List[Dict[str, Any]]:
        """Generate a batch of test items."""
        batch = []
        for i in range(count):
            batch.append({
                "id": f"test_{i}_{TestDataGenerator.random_string(5)}",
                "title": f"Test Item {i}",
                "content": TestDataGenerator.sample_text_content(),
                "metadata": {
                    "index": i,
                    "created": datetime.now().isoformat(),
                    "type": random.choice(["document", "transcript", "note"])
                }
            })
        return batch


class TestScenarios:
    """Pre-defined test scenarios for comprehensive testing."""
    
    @staticmethod
    def research_workflow() -> Dict[str, Any]:
        """Scenario: Research workflow with multiple sources."""
        return {
            "name": "Research Workflow",
            "description": "User researching AI topics from multiple sources",
            "steps": [
                {
                    "action": "upload_document",
                    "data": {
                        "title": "AI Research Paper",
                        "content": TestDataGenerator.sample_text_content()
                    }
                },
                {
                    "action": "web_scrape",
                    "data": {
                        "url": "https://en.wikipedia.org/wiki/Machine_learning"
                    }
                },
                {
                    "action": "create_note",
                    "data": TestDataGenerator.sample_note()
                },
                {
                    "action": "search",
                    "data": {
                        "query": "transformer architecture"
                    }
                },
                {
                    "action": "chat",
                    "data": {
                        "messages": TestDataGenerator.sample_chat_messages(),
                        "use_context": True
                    }
                }
            ]
        }
    
    @staticmethod
    def content_creation_workflow() -> Dict[str, Any]:
        """Scenario: Content creation and organization."""
        return {
            "name": "Content Creation",
            "description": "User creating and organizing content",
            "steps": [
                {
                    "action": "create_prompt",
                    "data": TestDataGenerator.sample_prompt_template()
                },
                {
                    "action": "import_character",
                    "data": TestDataGenerator.sample_character_card()
                },
                {
                    "action": "character_chat",
                    "data": {
                        "character_id": None,  # Will be filled during test
                        "messages": TestDataGenerator.sample_chat_messages()
                    }
                },
                {
                    "action": "export_chat",
                    "data": {
                        "format": "markdown"
                    }
                }
            ]
        }
    
    @staticmethod
    def media_processing_workflow() -> Dict[str, Any]:
        """Scenario: Media processing and analysis."""
        return {
            "name": "Media Processing",
            "description": "User processing various media types",
            "steps": [
                {
                    "action": "upload_audio",
                    "data": {
                        "title": "Test Audio",
                        "transcribe": True
                    }
                },
                {
                    "action": "upload_video",
                    "data": {
                        "title": "Test Video",
                        "extract_audio": True,
                        "transcribe": True
                    }
                },
                {
                    "action": "upload_pdf",
                    "data": {
                        "title": "Test PDF",
                        "extract_text": True
                    }
                },
                {
                    "action": "analyze_content",
                    "data": {
                        "generate_summary": True,
                        "extract_keywords": True
                    }
                }
            ]
        }


# Utility functions for test data
def generate_unique_id(prefix: str = "test") -> str:
    """Generate a unique ID for testing."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    random_part = TestDataGenerator.random_string(5)
    return f"{prefix}_{timestamp}_{random_part}"


def generate_test_user() -> Dict[str, str]:
    """Generate a complete test user profile."""
    username = TestDataGenerator.random_username()
    return {
        "username": username,
        "email": f"{username}@test.example.com",
        "password": f"Test@{TestDataGenerator.random_string(8)}123",
        "full_name": f"Test User {username.split('_')[1].title()}"
    }


def generate_batch_data(data_type: str, count: int = 5) -> List[Dict[str, Any]]:
    """Generate batch data of specific type."""
    generators = {
        "notes": lambda: TestDataGenerator.sample_note(),
        "prompts": lambda: TestDataGenerator.sample_prompt_template(),
        "messages": lambda: {"messages": TestDataGenerator.sample_chat_messages()},
        "keywords": lambda: {"keywords": random.sample(TestDataGenerator.sample_keywords(), 5)}
    }
    
    generator = generators.get(data_type, lambda: {})
    return [generator() for _ in range(count)]