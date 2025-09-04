# Patch for ebook_chapters.py to better detect dangerous regex patterns

# This would replace the DANGEROUS_PATTERNS list in ebook_chapters.py
# to catch more ReDoS patterns including nested quantifiers

DANGEROUS_PATTERNS = [
    r'\(\*',                          # Possessive quantifiers  
    r'\(\?R\)',                       # Recursive patterns
    r'\(\?\(DEFINE\)',                # DEFINE patterns
    r'{\d{4,}}',                      # Large repetition ranges
    r'[*+]{2,}',                      # Consecutive quantifiers
    r'\([^)]*[*+].*[*+].*\)',        # Multiple quantifiers in group
    
    # Additional patterns to catch nested quantifiers like (a+)+
    r'\([^)]*[+*]\)[+*]',            # Group with quantifier followed by quantifier
    r'\([^)]*[+*]\)\+',              # (something+)+ pattern
    r'\([^)]*[+*]\)\*',              # (something*)* pattern  
    r'\([^)]*[+*]\){',               # (something+){n,m} pattern
    r'\(\([^)]*[+*]\)[^)]*\)[+*]',   # Nested groups with quantifiers
    r'(\w\+)+\+',                     # Explicit (x+)+ pattern
    r'(\w\*)+\*',                     # Explicit (x*)* pattern
]

# Additional function to detect nested quantifier patterns more thoroughly
def has_nested_quantifiers(pattern: str) -> bool:
    """
    Check if a regex pattern has nested quantifiers that could cause ReDoS.
    
    Args:
        pattern: Regex pattern to check
        
    Returns:
        True if pattern appears to have nested quantifiers
    """
    import re
    
    # Check for patterns like (a+)+, (a*)*, ((a+)+)+, etc.
    nested_patterns = [
        # Direct nested quantifiers
        r'\([^)]*[+*?]\)[+*?]',          # (x+)+ or (x*)* or (x?)?
        r'\([^)]*\{[^}]+\}\)[+*?]',      # (x{n,m})+ 
        r'\([^)]*[+*?]\)\{[^}]+\}',      # (x+){n,m}
        
        # Nested groups with quantifiers
        r'\(\([^)]+\)[+*?]\)[+*?]',      # ((x)+)+
        
        # Alternative nested patterns
        r'\([^)|]*\|[^)]*[+*?]\)[+*?]',  # (a|b+)+
    ]
    
    for nested in nested_patterns:
        if re.search(nested, pattern):
            return True
    
    # Check for sequential groups with quantifiers that could interact badly
    # e.g., (a+)(b+) where backtracking could occur
    if re.search(r'\([^)]*[+*]\)[^(]*\([^)]*[+*]\)', pattern):
        # Check if there's potential for backtracking between groups
        return True
        
    return False

# Enhanced validation function
def validate_regex_pattern_enhanced(pattern: str, max_length: int = 500, timeout: float = 1.0) -> bool:
    """
    Enhanced validation of regex patterns for ReDoS vulnerabilities.
    
    Args:
        pattern: Regex pattern to validate
        max_length: Maximum allowed pattern length
        timeout: Timeout for test execution
        
    Returns:
        True if pattern is safe
        
    Raises:
        InvalidInputError: If pattern is dangerous
    """
    import re
    import time
    from tldw_Server_API.app.core.Chunking.exceptions import InvalidInputError
    
    # Check pattern length
    if len(pattern) > max_length:
        raise InvalidInputError(
            f"Regex pattern too long ({len(pattern)} chars). "
            f"Maximum allowed: {max_length}"
        )
    
    # Check for dangerous patterns
    for dangerous in DANGEROUS_PATTERNS:
        if re.search(dangerous, pattern):
            raise InvalidInputError(
                f"Regex pattern contains potentially dangerous construct"
            )
    
    # Check for nested quantifiers
    if has_nested_quantifiers(pattern):
        raise InvalidInputError(
            "Regex pattern contains nested quantifiers that could cause exponential backtracking"
        )
    
    # Test pattern compilation
    try:
        compiled_pattern = re.compile(pattern)
    except re.error as e:
        raise InvalidInputError(f"Invalid regex pattern: {e}")
    
    # Test for exponential complexity with multiple test inputs
    test_inputs = [
        "a" * 20,      # Repetitive characters
        "a" * 30,      # Longer repetitive  
        "ab" * 15,     # Alternating pattern
        "abc" * 10,    # More complex pattern
    ]
    
    for test_input in test_inputs:
        start_time = time.time()
        try:
            # Use compiled pattern for better performance
            result = compiled_pattern.search(test_input)
            elapsed = time.time() - start_time
            
            if elapsed > timeout:
                raise InvalidInputError(
                    f"Regex pattern appears to have exponential complexity "
                    f"(took {elapsed:.2f}s on test input)"
                )
        except Exception as e:
            if "timeout" in str(e).lower():
                raise InvalidInputError(
                    "Regex pattern appears to have exponential complexity"
                )
            # Re-raise other exceptions
            raise
    
    return True

# Example of how to use in the EbookChapterChunkingStrategy class:
"""
def _validate_regex_pattern(self, pattern: str) -> bool:
    # Use the enhanced validation
    return validate_regex_pattern_enhanced(
        pattern, 
        max_length=self.MAX_REGEX_LENGTH,
        timeout=1.0  # 1 second timeout for validation
    )
"""