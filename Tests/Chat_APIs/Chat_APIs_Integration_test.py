# Chat_APIs_Integration_test.py
# Test file for testing the integration of the LLM API calls with the Chat APIs.
#
# Usage:
# First setup api keys as env variables:
#   export OPENAI_API_KEY=your_openai_key
#   export ANTHROPIC_API_KEY=your_anthropic_key
#   ... set other API keys similarly
# then run it:
#   python -m unittest test_llm_api_calls_integration.py

import unittest
import os
from dotenv import load_dotenv
from App_Function_Libraries.LLM_API_Calls import (
    chat_with_openai,
    chat_with_anthropic,
    chat_with_cohere,
    chat_with_groq,
    chat_with_openrouter,
    chat_with_huggingface,
    chat_with_deepseek,
    chat_with_mistral
)

class TestLLMAPICallsIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load environment variables from .env file
        load_dotenv()

        # Load API keys from environment variables
        cls.openai_api_key = os.getenv('OPENAI_API_KEY')
        cls.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        cls.cohere_api_key = os.getenv('COHERE_API_KEY')
        cls.groq_api_key = os.getenv('GROQ_API_KEY')
        cls.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        cls.huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')
        cls.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
        cls.mistral_api_key = os.getenv('MISTRAL_API_KEY')

    def test_chat_with_openai(self):
        if not self.openai_api_key:
            self.skipTest("OpenAI API key not available")
        response = chat_with_openai(self.openai_api_key, "Hello, how are you?", "Respond briefly")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_chat_with_anthropic(self):
        if not self.anthropic_api_key:
            self.skipTest("Anthropic API key not available")
        response = chat_with_anthropic(self.anthropic_api_key, "Hello, how are you?", "claude-2", "Respond briefly")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_chat_with_cohere(self):
        if not self.cohere_api_key:
            self.skipTest("Cohere API key not available")
        response = chat_with_cohere(self.cohere_api_key, "Hello, how are you?", "command", "Respond briefly")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_chat_with_groq(self):
        if not self.groq_api_key:
            self.skipTest("Groq API key not available")
        response = chat_with_groq(self.groq_api_key, "Hello, how are you?", "Respond briefly")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_chat_with_openrouter(self):
        if not self.openrouter_api_key:
            self.skipTest("OpenRouter API key not available")
        response = chat_with_openrouter(self.openrouter_api_key, "Hello, how are you?", "Respond briefly")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_chat_with_huggingface(self):
        if not self.huggingface_api_key:
            self.skipTest("HuggingFace API key not available")
        response = chat_with_huggingface(self.huggingface_api_key, "Hello, how are you?", "Respond briefly")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_chat_with_deepseek(self):
        if not self.deepseek_api_key:
            self.skipTest("DeepSeek API key not available")
        response = chat_with_deepseek(self.deepseek_api_key, "Hello, how are you?", "Respond briefly")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_chat_with_mistral(self):
        if not self.mistral_api_key:
            self.skipTest("Mistral API key not available")
        response = chat_with_mistral(self.mistral_api_key, "Hello, how are you?", "Respond briefly")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

if __name__ == '__main__':
    unittest.main()