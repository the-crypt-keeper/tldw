# Tests/RAG/test_rag_functions.py

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any

# Adjust the path to the parent directory of App_Function_Libraries
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(parent_dir)

# Import the functions to test
from App_Function_Libraries.RAG.RAG_Library_2 import (
    fetch_relevant_media_ids,
    perform_vector_search,
    perform_full_text_search
)


class TestRAGFunctions(unittest.TestCase):
    """
    Unit tests for RAG-related functions.
    """

    @patch('App_Function_Libraries.RAG.RAG_Library_2.fetch_keywords_for_media')
    def test_fetch_relevant_media_ids_success(self, mock_fetch_keywords_for_media):
        """
        Test fetch_relevant_media_ids with valid keywords.
        """
        # Setup mock return values
        mock_fetch_keywords_for_media.side_effect = lambda keyword: {
            'geography': [1, 2],
            'cities': [2, 3, 4]
        }.get(keyword, [])

        # Input keywords
        keywords = ['geography', 'cities']

        # Call the function
        result = fetch_relevant_media_ids(keywords)

        # Expected result is the union of media_ids: [1,2,3,4]
        self.assertEqual(sorted(result), [1, 2, 3, 4])

        # Assert fetch_keywords_for_media was called correctly
        mock_fetch_keywords_for_media.assert_any_call('geography')
        mock_fetch_keywords_for_media.assert_any_call('cities')
        self.assertEqual(mock_fetch_keywords_for_media.call_count, 2)

    @patch('App_Function_Libraries.RAG.RAG_Library_2.fetch_keywords_for_media')
    def test_fetch_relevant_media_ids_empty_keywords(self, mock_fetch_keywords_for_media):
        """
        Test fetch_relevant_media_ids with an empty keywords list.
        """
        keywords = []
        result = fetch_relevant_media_ids(keywords)
        self.assertEqual(result, [])
        mock_fetch_keywords_for_media.assert_not_called()

    @patch('App_Function_Libraries.RAG.RAG_Library_2.fetch_keywords_for_media')
    @patch('App_Function_Libraries.RAG.RAG_Library_2.logging')
    def test_fetch_relevant_media_ids_exception(self, mock_logging, mock_fetch_keywords_for_media):
        """
        Test fetch_relevant_media_ids when fetch_keywords_for_media raises an exception.
        """
        # Configure the mock to raise an exception
        mock_fetch_keywords_for_media.side_effect = Exception("Database error")

        keywords = ['geography', 'cities']
        result = fetch_relevant_media_ids(keywords)

        # The function should return an empty list upon exception
        self.assertEqual(result, [])

        # Assert that an error was logged
        mock_logging.error.assert_called_once_with("Error fetching relevant media IDs: Database error")

    @patch('App_Function_Libraries.RAG.RAG_Library_2.vector_search')
    @patch('App_Function_Libraries.RAG.RAG_Library_2.chroma_client')
    def test_perform_vector_search_with_relevant_media_ids(self, mock_chroma_client, mock_vector_search):
        """
        Test perform_vector_search with relevant_media_ids provided.
        """
        # Setup mock chroma_client to return a list of collections
        mock_collection = MagicMock()
        mock_collection.name = 'collection1'
        mock_chroma_client.list_collections.return_value = [mock_collection]

        # Setup mock vector_search to return search results
        mock_vector_search.return_value = [
            {'content': 'Document 1', 'metadata': {'media_id': 1}},
            {'content': 'Document 2', 'metadata': {'media_id': 2}},
            {'content': 'Document 3', 'metadata': {'media_id': 3}},
        ]

        # Input parameters
        query = 'sample query'
        relevant_media_ids = [1, 3]

        # Call the function
        result = perform_vector_search(query, relevant_media_ids)

        # Expected to filter out media_id 2
        expected = [
            {'content': 'Document 1', 'metadata': {'media_id': 1}},
            {'content': 'Document 3', 'metadata': {'media_id': 3}},
        ]
        self.assertEqual(result, expected)

        # Assert chroma_client.list_collections was called once
        mock_chroma_client.list_collections.assert_called_once()

        # Assert vector_search was called with correct arguments
        mock_vector_search.assert_called_once_with('collection1', query, k=5)

    @patch('App_Function_Libraries.RAG.RAG_Library_2.vector_search')
    @patch('App_Function_Libraries.RAG.RAG_Library_2.chroma_client')
    def test_perform_vector_search_without_relevant_media_ids(self, mock_chroma_client, mock_vector_search):
        """
        Test perform_vector_search without relevant_media_ids (None).
        """
        # Setup mock chroma_client to return a list of collections
        mock_collection = MagicMock()
        mock_collection.name = 'collection1'
        mock_chroma_client.list_collections.return_value = [mock_collection]

        # Setup mock vector_search to return search results
        mock_vector_search.return_value = [
            {'content': 'Document 1', 'metadata': {'media_id': 1}},
            {'content': 'Document 2', 'metadata': {'media_id': 2}},
        ]

        # Input parameters
        query = 'sample query'
        relevant_media_ids = None

        # Call the function
        result = perform_vector_search(query, relevant_media_ids)

        # Expected to return all results
        expected = [
            {'content': 'Document 1', 'metadata': {'media_id': 1}},
            {'content': 'Document 2', 'metadata': {'media_id': 2}},
        ]
        self.assertEqual(result, expected)

        # Assert chroma_client.list_collections was called once
        mock_chroma_client.list_collections.assert_called_once()

        # Assert vector_search was called with correct arguments
        mock_vector_search.assert_called_once_with('collection1', query, k=5)

    @patch('App_Function_Libraries.RAG.RAG_Library_2.search_db')
    def test_perform_full_text_search_with_relevant_media_ids(self, mock_search_db):
        """
        Test perform_full_text_search with relevant_media_ids provided.
        """
        # Setup mock search_db to return search results
        mock_search_db.return_value = [
            {'content': 'Full text document 1', 'id': 1},
            {'content': 'Full text document 2', 'id': 2},
            {'content': 'Full text document 3', 'id': 3},
        ]

        # Input parameters
        query = 'full text query'
        relevant_media_ids = [1, 3]

        # Call the function
        result = perform_full_text_search(query, relevant_media_ids)

        # Expected to filter out id 2
        expected = [
            {'content': 'Full text document 1', 'metadata': {'media_id': 1}},
            {'content': 'Full text document 3', 'metadata': {'media_id': 3}},
        ]
        self.assertEqual(result, expected)

        # Assert search_db was called with correct arguments
        mock_search_db.assert_called_once_with(
            query, ['content'], '', page=1, results_per_page=5)

    @patch('App_Function_Libraries.RAG.RAG_Library_2.search_db')
    def test_perform_full_text_search_without_relevant_media_ids(self, mock_search_db):
        """
        Test perform_full_text_search without relevant_media_ids (None).
        """
        # Setup mock search_db to return search results
        mock_search_db.return_value = [
            {'content': 'Full text document 1', 'id': 1},
            {'content': 'Full text document 2', 'id': 2},
        ]

        # Input parameters
        query = 'full text query'
        relevant_media_ids = None

        # Call the function
        result = perform_full_text_search(query, relevant_media_ids)

        # Expected to return all results
        expected = [
            {'content': 'Full text document 1', 'metadata': {'media_id': 1}},
            {'content': 'Full text document 2', 'metadata': {'media_id': 2}},
        ]
        self.assertEqual(result, expected)

        # Assert search_db was called with correct arguments
        mock_search_db.assert_called_once_with(
            query, ['content'], '', page=1, results_per_page=5)

    @patch('App_Function_Libraries.RAG.RAG_Library_2.search_db')
    def test_perform_full_text_search_empty_results(self, mock_search_db):
        """
        Test perform_full_text_search when search_db returns no results.
        """
        # Setup mock search_db to return empty list
        mock_search_db.return_value = []

        # Input parameters
        query = 'full text query'
        relevant_media_ids = [1, 2]

        # Call the function
        result = perform_full_text_search(query, relevant_media_ids)

        # Expected to return an empty list
        expected = []
        self.assertEqual(result, expected)

        # Assert search_db was called with correct arguments
        mock_search_db.assert_called_once_with(
            query, ['content'], '', page=1, results_per_page=5)

    @patch('App_Function_Libraries.RAG.RAG_Library_2.fetch_keywords_for_media')
    @patch('App_Function_Libraries.RAG.RAG_Library_2.logging')
    def test_fetch_relevant_media_ids_partial_failure(self, mock_logging, mock_fetch_keywords_for_media):
        """
        Test fetch_relevant_media_ids when fetch_keywords_for_media partially fails.
        """

        # Configure the mock to raise an exception for one keyword
        def side_effect(keyword):
            if keyword == 'geography':
                return [1, 2]
            elif keyword == 'cities':
                raise Exception("Database error")
            return []

        mock_fetch_keywords_for_media.side_effect = side_effect

        keywords = ['geography', 'cities']
        result = fetch_relevant_media_ids(keywords)

        # The function should still return media_ids for 'geography' and skip 'cities'
        self.assertEqual(sorted(result), [1, 2])

        # Assert that an error was logged for 'cities'
        mock_logging.error.assert_called_once_with("Error fetching relevant media IDs: Database error")

    @patch('App_Function_Libraries.RAG.RAG_Library_2.chroma_client')
    @patch('App_Function_Libraries.RAG.RAG_Library_2.vector_search')
    def test_perform_vector_search_no_collections(self, mock_vector_search, mock_chroma_client):
        """
        Test perform_vector_search when there are no collections.
        """
        # Setup mock chroma_client to return an empty list of collections
        mock_chroma_client.list_collections.return_value = []

        # Input parameters
        query = 'sample query'
        relevant_media_ids = [1, 2]

        # Call the function
        result = perform_vector_search(query, relevant_media_ids)

        # Expected to return an empty list since there are no collections
        expected = []
        self.assertEqual(result, expected)

        # Assert chroma_client.list_collections was called once
        mock_chroma_client.list_collections.assert_called_once()

        # Assert vector_search was not called since there are no collections
        mock_vector_search.assert_not_called()

    @patch('App_Function_Libraries.RAG.RAG_Library_2.fetch_keywords_for_media')
    def test_fetch_relevant_media_ids_duplicate_media_ids(self, mock_fetch_keywords_for_media):
        """
        Test fetch_relevant_media_ids with duplicate media_ids across keywords.
        """
        # Setup mock return values with overlapping media_ids
        mock_fetch_keywords_for_media.side_effect = lambda keyword: {
            'science': [1, 2, 3],
            'technology': [3, 4, 5],
            'engineering': [5, 6],
        }.get(keyword, [])

        # Input keywords
        keywords = ['science', 'technology', 'engineering']

        # Call the function
        result = fetch_relevant_media_ids(keywords)

        # Expected result is the unique union of media_ids: [1,2,3,4,5,6]
        self.assertEqual(sorted(result), [1, 2, 3, 4, 5, 6])

        # Assert fetch_keywords_for_media was called correctly
        mock_fetch_keywords_for_media.assert_any_call('science')
        mock_fetch_keywords_for_media.assert_any_call('technology')
        mock_fetch_keywords_for_media.assert_any_call('engineering')
        self.assertEqual(mock_fetch_keywords_for_media.call_count, 3)

    @patch('App_Function_Libraries.RAG.RAG_Library_2.search_db')
    def test_perform_full_text_search_case_insensitive_filtering(self, mock_search_db):
        """
        Test perform_full_text_search with case-insensitive filtering of media_ids.
        """
        # Setup mock search_db to return mixed-case media_ids
        mock_search_db.return_value = [
            {'content': 'Full text document 1', 'id': '1'},
            {'content': 'Full text document 2', 'id': '2'},
            {'content': 'Full text document 3', 'id': '3'},
        ]

        # Input parameters with media_ids as strings
        query = 'full text query'
        relevant_media_ids = ['1', '3']

        # Call the function
        result = perform_full_text_search(query, relevant_media_ids)

        # Expected to filter out id '2'
        expected = [
            {'content': 'Full text document 1', 'metadata': {'media_id': '1'}},
            {'content': 'Full text document 3', 'metadata': {'media_id': '3'}},
        ]
        self.assertEqual(result, expected)

        # Assert search_db was called with correct arguments
        mock_search_db.assert_called_once_with(
            query, ['content'], '', page=1, results_per_page=5)

    @patch('App_Function_Libraries.RAG.RAG_Library_2.search_db')
    def test_perform_full_text_search_multiple_pages(self, mock_search_db):
        """
        Test perform_full_text_search with multiple pages of results.
        Note: The current implementation fetches only the first page.
        """
        # Setup mock search_db to return results from the first page
        mock_search_db.return_value = [
            {'content': 'Full text document 1', 'id': 1},
            {'content': 'Full text document 2', 'id': 2},
            {'content': 'Full text document 3', 'id': 3},
            {'content': 'Full text document 4', 'id': 4},
            {'content': 'Full text document 5', 'id': 5},
        ]

        # Input parameters
        query = 'full text query'
        relevant_media_ids = [1, 2, 3, 4, 5]

        # Call the function
        result = perform_full_text_search(query, relevant_media_ids)

        # Expected to return all results
        expected = [
            {'content': 'Full text document 1', 'metadata': {'media_id': 1}},
            {'content': 'Full text document 2', 'metadata': {'media_id': 2}},
            {'content': 'Full text document 3', 'metadata': {'media_id': 3}},
            {'content': 'Full text document 4', 'metadata': {'media_id': 4}},
            {'content': 'Full text document 5', 'metadata': {'media_id': 5}},
        ]
        self.assertEqual(result, expected)

        # Assert search_db was called with correct arguments
        mock_search_db.assert_called_once_with(
            query, ['content'], '', page=1, results_per_page=5)

    @patch('App_Function_Libraries.RAG.RAG_Library_2.chroma_client')
    @patch('App_Function_Libraries.RAG.RAG_Library_2.vector_search')
    def test_perform_vector_search_multiple_collections(self, mock_vector_search, mock_chroma_client):
        """
        Test perform_vector_search with multiple collections.
        """
        # Setup mock chroma_client to return multiple collections
        mock_collection1 = MagicMock()
        mock_collection1.name = 'collection1'
        mock_collection2 = MagicMock()
        mock_collection2.name = 'collection2'
        mock_chroma_client.list_collections.return_value = [mock_collection1, mock_collection2]

        # Setup mock vector_search to return different results for each collection
        def vector_search_side_effect(collection_name, query, k):
            if collection_name == 'collection1':
                return [
                    {'content': 'Collection1 Document 1', 'metadata': {'media_id': 1}},
                    {'content': 'Collection1 Document 2', 'metadata': {'media_id': 2}},
                ]
            elif collection_name == 'collection2':
                return [
                    {'content': 'Collection2 Document 1', 'metadata': {'media_id': 3}},
                    {'content': 'Collection2 Document 2', 'metadata': {'media_id': 4}},
                ]
            return []

        mock_vector_search.side_effect = vector_search_side_effect

        # Input parameters
        query = 'sample query'
        relevant_media_ids = [2, 3]

        # Call the function
        result = perform_vector_search(query, relevant_media_ids)

        # Expected to filter and include media_id 2 and 3
        expected = [
            {'content': 'Collection1 Document 2', 'metadata': {'media_id': 2}},
            {'content': 'Collection2 Document 1', 'metadata': {'media_id': 3}},
        ]
        self.assertEqual(result, expected)

        # Assert chroma_client.list_collections was called once
        mock_chroma_client.list_collections.assert_called_once()

        # Assert vector_search was called twice with correct arguments
        mock_vector_search.assert_any_call('collection1', query, k=5)
        mock_vector_search.assert_any_call('collection2', query, k=5)
        self.assertEqual(mock_vector_search.call_count, 2)

    @patch('App_Function_Libraries.RAG.RAG_Library_2.search_db')
    def test_perform_full_text_search_partial_matches(self, mock_search_db):
        """
        Test perform_full_text_search where some media_ids do not match the relevant_media_ids.
        """
        # Setup mock search_db to return search results
        mock_search_db.return_value = [
            {'content': 'Full text document 1', 'id': 1},
            {'content': 'Full text document 2', 'id': 2},
            {'content': 'Full text document 3', 'id': 3},
            {'content': 'Full text document 4', 'id': 4},
        ]

        # Input parameters
        query = 'full text query'
        relevant_media_ids = [2, 4]

        # Call the function
        result = perform_full_text_search(query, relevant_media_ids)

        # Expected to include only media_id 2 and 4
        expected = [
            {'content': 'Full text document 2', 'metadata': {'media_id': 2}},
            {'content': 'Full text document 4', 'metadata': {'media_id': 4}},
        ]
        self.assertEqual(result, expected)

        # Assert search_db was called with correct arguments
        mock_search_db.assert_called_once_with(
            query, ['content'], '', page=1, results_per_page=5)


if __name__ == '__main__':
    unittest.main()