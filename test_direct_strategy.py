#!/usr/bin/env python3
"""Test the strategy directly without going through Chunker."""

from tldw_Server_API.app.core.Chunking.strategies.ebook_chapters import EbookChapterChunkingStrategy

strategy = EbookChapterChunkingStrategy()

# Test legitimate patterns
patterns = [
    r"Chapter \d+",
    r"Section \w+",
]

text = """Chapter 1
Content of chapter 1.

Chapter 2
Content of chapter 2."""

print("Testing patterns directly with strategy:\n")

for pattern in patterns:
    print(f"Testing: {pattern}")
    try:
        chunks = strategy.chunk(text, max_size=100, custom_chapter_pattern=pattern)
        print(f"  ✅ Created {len(chunks)} chunks")
    except Exception as e:
        print(f"  ❌ Error: {e}")

# Test a dangerous pattern
print("\nTesting dangerous pattern:")
dangerous = r"(a+)+b"
print(f"Testing: {dangerous}")
try:
    chunks = strategy.chunk(text, max_size=100, custom_chapter_pattern=dangerous)
    print(f"  ❌ SHOULD NOT REACH HERE - Created {len(chunks)} chunks")
except Exception as e:
    print(f"  ✅ Correctly rejected: {e}")