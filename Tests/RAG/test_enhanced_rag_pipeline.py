import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Adjust the path to the parent directory of App_Function_Libraries
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(parent_dir)

# Now import the necessary modules
from App_Function_Libraries.RAG.RAG_Library_2 import enhanced_rag_pipeline
from App_Function_Libraries.RAG.Embeddings_Create import create_embeddings_batch
from App_Function_Libraries.RAG.ChromaDB_Library import vector_search

class TestEnhancedRagPipeline(unittest.TestCase):

    @patch('App_Function_Libraries.RAG.RAG_Libary_2.fetch_relevant_media_ids')
    @patch('App_Function_Libraries.RAG.RAG_Libary_2.perform_vector_search')
    @patch('App_Function_Libraries.RAG.RAG_Libary_2.perform_full_text_search')
    @patch('App_Function_Libraries.RAG.RAG_Libary_2.generate_answer')
    def test_enhanced_rag_pipeline(self, mock_generate_answer, mock_fts_search, mock_vector_search, mock_fetch_keywords):
        """
        Test the enhanced_rag_pipeline function by mocking the dependent functions such as
        vector search, full-text search, and external API calls.
        """

        # Setup mock data
        query = "What is the capital of France?"
        keywords = "geography, cities"
        api_choice = "OpenAI"

        # Mock the relevant_media_ids fetched by keywords
        mock_fetch_keywords.return_value = [1, 2, 3]

        # Mock vector search results
        mock_vector_search.return_value = [
            {"content": "Paris is the capital of France."}
        ]

        # Mock full-text search results
        mock_fts_search.return_value = [
            {"content": "The capital of France is Paris."}
        ]

        # Mock the API response from the generate_answer function
        mock_generate_answer.return_value = "Paris is the capital of France."

        # Call the enhanced_rag_pipeline function
        result = enhanced_rag_pipeline(query=query, api_choice=api_choice, keywords=keywords)

        # Validate that the vector search and full-text search are called with expected arguments
        mock_vector_search.assert_called_once_with(query, [1, 2, 3])
        mock_fts_search.assert_called_once_with(query, [1, 2, 3])

        # Check that generate_answer was called with the correct context and query
        expected_context = "Paris is the capital of France.\nThe capital of France is Paris."
        mock_generate_answer.assert_called_once_with(api_choice, expected_context, query)

        # Validate the result structure
        self.assertIn("answer", result)
        self.assertIn("context", result)
        self.assertEqual(result["answer"], "Paris is the capital of France.")
        self.assertEqual(result["context"], expected_context)

if __name__ == '__main__':
    unittest.main()
