#!/usr/bin/env python3
"""Test that legitimate regex patterns still work after our ReDoS fix."""

from tldw_Server_API.app.core.Chunking import Chunker

# Test legitimate patterns
legitimate_patterns = [
    r"Chapter \d+",           # Simple chapter pattern
    r"^Chapter [IVX]+$",      # Roman numerals
    r"Section \w+",           # Section with word characters
    r"Part [A-Z]",            # Part with single letter
    r"\d+\.\s+\w+",          # Numbered sections
]

text = """Chapter 1
This is the content of chapter 1.

Chapter 2
This is the content of chapter 2.

Chapter 3
This is the content of chapter 3."""

chunker = Chunker()

print("Testing legitimate regex patterns after ReDoS fix:\n")

for pattern in legitimate_patterns:
    print(f"Testing pattern: {pattern}")
    try:
        result = chunker.chunk_text(
            text,
            method='ebook_chapters',
            custom_chapter_pattern=pattern,
            max_size=100
        )
        print(f"  ✅ SUCCESS - Created {len(result)} chunks")
    except Exception as e:
        print(f"  ❌ FAILED - {e}")

print("\nAll legitimate patterns should work correctly.")