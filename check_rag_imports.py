#!/usr/bin/env python3
"""Check for import issues in RAG test files."""

import sys
import importlib
from pathlib import Path

# Add the project to path
sys.path.insert(0, str(Path(__file__).parent))

test_files = [
    "tldw_Server_API.tests.RAG.test_rag_endpoints_integration_real",
    "tldw_Server_API.tests.RAG.test_rag_endpoints_integration", 
    "tldw_Server_API.tests.RAG.test_rag_simple_integration",
    "tldw_Server_API.tests.RAG.test_rag_full_integration",
    "tldw_Server_API.tests.RAG.test_rag_endpoints_unit",
]

print("Checking RAG test imports...")
for module_name in test_files:
    try:
        importlib.import_module(module_name)
        print(f"✓ {module_name} - OK")
    except ImportError as e:
        print(f"✗ {module_name} - ImportError: {e}")
    except Exception as e:
        print(f"✗ {module_name} - Error: {e}")