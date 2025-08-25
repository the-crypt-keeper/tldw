#!/usr/bin/env python3
"""Test Pydantic validation for Optional fields with max_length"""

from pydantic import BaseModel, Field, ValidationError
from typing import Optional

class TestModel(BaseModel):
    query: Optional[str] = Field(None, max_length=10)

# Test 1: None value (should work)
try:
    model = TestModel(query=None)
    print("✓ None value works")
except ValidationError as e:
    print(f"✗ None value failed: {e}")

# Test 2: Short string (should work)
try:
    model = TestModel(query="short")
    print("✓ Short string works")
except ValidationError as e:
    print(f"✗ Short string failed: {e}")

# Test 3: Long string (should fail)
try:
    model = TestModel(query="a" * 100)
    print("✗ Long string should have failed but didn't!")
except ValidationError as e:
    print(f"✓ Long string correctly failed: {e.errors()[0]['msg']}")

# Test 4: Without Field, just Optional
class TestModel2(BaseModel):
    query: Optional[str] = None

try:
    model = TestModel2(query="a" * 10000)
    print(f"Without Field: accepts long string (length={len(model.query)})")
except ValidationError as e:
    print(f"Without Field: failed with {e}")