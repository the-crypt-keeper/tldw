# Chunker Module Security Documentation

## Overview
This document outlines the security measures implemented in the Chunker module and provides guidelines for secure usage.

## Security Features

### 1. XML External Entity (XXE) Protection
The XML chunking strategy includes protection against XXE attacks:
- **Entity resolution disabled** to prevent file disclosure attacks
- **Network access blocked** to prevent SSRF attacks
- **DTD processing disabled** to prevent billion laughs attacks
- **Secure parser configuration** with multiple safety layers

### 2. Regular Expression Denial of Service (ReDoS) Protection
Custom regex patterns are validated and protected:
- **Pattern complexity validation** to reject dangerous constructs
- **Maximum pattern length** enforcement (500 characters)
- **Timeout mechanism** (2 seconds) to prevent infinite loops
- **Dangerous pattern detection** for nested quantifiers and exponential complexity

### 3. Input Sanitization
All text inputs are sanitized before processing:
- **Null byte removal** to prevent injection attacks
- **Unicode normalization** (NFC) to prevent homograph attacks
- **Control character filtering** (except \n, \t, \r, \f)
- **Bidirectional override removal** to prevent text spoofing

### 4. Resource Limits
Strict limits are enforced to prevent resource exhaustion:
- **Maximum text size**: 100MB default
- **Maximum JSON size**: 50MB for JSON processing
- **Maximum XML size**: 10MB for XML processing
- **Configurable cache limits** to prevent memory exhaustion

## Threat Model

### Protected Against
✅ XML External Entity (XXE) attacks
✅ Regular Expression Denial of Service (ReDoS)
✅ Null byte injection
✅ Unicode-based attacks
✅ Resource exhaustion attacks
✅ Billion laughs / XML bombs
✅ Server-Side Request Forgery (SSRF) via XXE

### Known Limitations
⚠️ Large file processing may still consume significant memory
⚠️ Regex timeout only works on Unix-like systems (uses SIGALRM)
⚠️ Some advanced unicode attacks may not be fully prevented

## Usage Guidelines

### Safe Usage
```python
from tldw_Server_API.app.core.Chunking import Chunker

# Safe: Using built-in methods
chunker = Chunker()
chunks = chunker.chunk_text(text, method='words', max_size=100)

# Safe: Validated custom patterns are allowed
chunks = chunker.chunk_text(
    text, 
    method='ebook_chapters',
    custom_chapter_pattern=r'Chapter \d+'  # Simple, safe pattern
)
```

### Unsafe Usage (Will Be Rejected)
```python
# UNSAFE: XXE attempt - will be blocked
malicious_xml = '''<!DOCTYPE foo [
    <!ENTITY xxe SYSTEM "file:///etc/passwd">
]><root>&xxe;</root>'''
chunks = chunker.chunk_text(malicious_xml, method='xml')  # Raises InvalidInputError

# UNSAFE: ReDoS pattern - will be blocked
evil_pattern = r'(a+)+b'  # Exponential complexity
chunks = chunker.chunk_text(
    text,
    method='ebook_chapters', 
    custom_chapter_pattern=evil_pattern  # Raises InvalidInputError
)

# UNSAFE: Oversized input - will be blocked
huge_text = 'a' * 100_000_001  # Over 100MB
chunks = chunker.chunk_text(huge_text)  # Raises InvalidInputError
```

## Security Configuration

### Recommended Settings
```python
from tldw_Server_API.app.core.Chunking import ChunkerConfig, Chunker

config = ChunkerConfig(
    max_text_size=50_000_000,  # 50MB limit
    enable_cache=True,          # Enable caching
    cache_size=100,            # Limit cache entries
)

chunker = Chunker(config=config)
```

### Environment Variables
- `CHUNKER_MAX_TEXT_SIZE`: Override maximum text size
- `CHUNKER_DISABLE_CACHE`: Disable caching if needed
- `CHUNKER_REGEX_TIMEOUT`: Adjust regex timeout (seconds)

## Reporting Security Issues

If you discover a security vulnerability in the Chunker module:

1. **DO NOT** create a public GitHub issue
2. Email the security team with details
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if available)

## Testing Security

Run the security test suite:
```bash
python -m pytest tests/Chunking/test_security.py -v
```

Key test coverage:
- XXE attack prevention
- ReDoS protection
- Input sanitization
- Resource limit enforcement

## Audit Log

### Security Fixes Applied
- **2024-01-XX**: Fixed XXE vulnerability in XML parser
- **2024-01-XX**: Added ReDoS protection for custom regex patterns
- **2024-01-XX**: Implemented comprehensive input sanitization
- **2024-01-XX**: Added resource limits and timeout mechanisms

## Best Practices for Developers

1. **Always validate user input** before passing to chunker
2. **Use built-in chunking methods** when possible
3. **Avoid custom regex patterns** unless absolutely necessary
4. **Monitor resource usage** in production
5. **Keep dependencies updated** (especially xml libraries)
6. **Run security tests** before deployment
7. **Review logs** for security warnings

## Dependencies

Security-critical dependencies:
- `xml.etree.ElementTree`: XML parsing (hardened configuration)
- `re`: Regex processing (with timeout protection)
- `unicodedata`: Unicode normalization

Consider using these alternatives for enhanced security:
- `defusedxml`: Drop-in replacement for xml with security by default
- `regex`: Alternative regex library with timeout support

## Compliance

This module follows security best practices aligned with:
- OWASP Top 10
- CWE/SANS Top 25
- NIST Cybersecurity Framework

Regular security audits should be performed to maintain compliance.