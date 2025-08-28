#!/usr/bin/env python3
"""
Quick script to run the Mock OpenAI API Server with default settings.
"""

import sys
import os

# Add the parent directory to path so we can import mock_openai
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mock_openai.server import main

if __name__ == "__main__":
    print("Starting Mock OpenAI API Server...")
    print("Server will be available at: http://localhost:8080")
    print("API Documentation: http://localhost:8080/docs")
    print("\nPress Ctrl+C to stop the server")
    print("-" * 50)
    
    main()