# Tests/RAG/test_rag_functions.py
import configparser
import os
import sys
import pytest
from unittest.mock import MagicMock
from typing import List, Dict, Any

# Adjust the path to the parent directory of App_Function_Libraries
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(parent_dir)

# Import the functions to test
from App_Function_Libraries.RAG.RAG_Library_2 import (
    fetch_relevant_media_ids,
    perform_vector_search,
    perform_full_text_search,
    enhanced_rag_pipeline,
    enhanced_rag_pipeline_chat,
    generate_answer,
    fetch_relevant_chat_ids,
    fetch_all_chat_ids,
    filter_results_by_keywords,
    extract_media_id_from_result
)


def test_fetch_relevant_media_ids_success(mocker):
    """Test fetch_relevant_media_ids with valid keywords."""
    mock_fetch_keywords_for_media = mocker.patch(
        'App_Function_Libraries.RAG.RAG_Library_2.fetch_keywords_for_media',
        side_effect=lambda keyword: {
            'geography': [1, 2],
            'cities': [2, 3, 4]
        }.get(keyword, [])
    )

    keywords = ['geography', 'cities']
    result = fetch_relevant_media_ids(keywords)
    assert sorted(result) == [1, 2, 3, 4]

    mock_fetch_keywords_for_media.assert_any_call('geography')
    mock_fetch_keywords_for_media.assert_any_call('cities')
    assert mock_fetch_keywords_for_media.call_count == 2


def test_perform_full_text_search_with_relevant_ids(mocker):
    """Test perform_full_text_search with relevant_ids provided."""
    # Create a transformed response matching the expected format
    transformed_response = [
        {'content': 'Full text document 1', 'metadata': {'media_id': 1}},
        {'content': 'Full text document 3', 'metadata': {'media_id': 3}},
    ]

    # Mock the search functions mapping
    search_function_mock = lambda query, fts_top_k, relevant_ids: transformed_response
    search_functions_mock = {
        "Media DB": search_function_mock
    }
    mocker.patch('App_Function_Libraries.RAG.RAG_Library_2.search_functions', search_functions_mock)

    query = 'full text query'
    database_type = "Media DB"
    relevant_ids = "1,3"

    result = perform_full_text_search(query, database_type, relevant_ids)

    expected = [
        {'content': 'Full text document 1', 'metadata': {'media_id': 1}},
        {'content': 'Full text document 3', 'metadata': {'media_id': 3}},
    ]
    assert result == expected


def test_perform_full_text_search_without_relevant_ids(mocker):
    """Test perform_full_text_search without relevant_ids."""
    # Create a transformed response matching the expected format
    transformed_response = [
        {'content': 'Full text document 1', 'metadata': {'media_id': 1}},
        {'content': 'Full text document 2', 'metadata': {'media_id': 2}},
    ]

    # Mock the search functions mapping
    search_function_mock = lambda query, fts_top_k, relevant_ids: transformed_response
    search_functions_mock = {
        "Media DB": search_function_mock
    }
    mocker.patch('App_Function_Libraries.RAG.RAG_Library_2.search_functions', search_functions_mock)

    query = 'full text query'
    database_type = "Media DB"
    relevant_ids = ""

    result = perform_full_text_search(query, database_type, relevant_ids)

    expected = [
        {'content': 'Full text document 1', 'metadata': {'media_id': 1}},
        {'content': 'Full text document 2', 'metadata': {'media_id': 2}},
    ]
    assert result == expected


@pytest.mark.parametrize("database_type,search_module_path,mock_response", [
    (
        "Media DB",
        'App_Function_Libraries.DB.SQLite_DB.search_media_db',
        [{'content': 'Media DB document 1', 'metadata': {'media_id': '1'}}]
    ),
    (
        "RAG Chat",
        'App_Function_Libraries.DB.RAG_QA_Chat_DB.search_rag_chat',
        [{'content': 'RAG Chat document 1', 'metadata': {'media_id': '1'}}]
    ),
    (
        "RAG Notes",
        'App_Function_Libraries.DB.RAG_QA_Chat_DB.search_rag_notes',
        [{'content': 'RAG Notes document 1', 'metadata': {'media_id': '1'}}]
    ),
    (
        "Character Chat",
        'App_Function_Libraries.DB.Character_Chat_DB.search_character_chat',
        [{'content': 'Character Chat document 1', 'metadata': {'media_id': '1'}}]
    ),
    (
        "Character Cards",
        'App_Function_Libraries.DB.Character_Chat_DB.search_character_cards',
        [{'content': 'Character Cards document 1', 'metadata': {'media_id': '1'}}]
    )
])
def test_perform_full_text_search_different_db_types(mocker, database_type, search_module_path, mock_response):
    """Test perform_full_text_search with different database types."""
    # Mock the search functions mapping with already transformed response
    search_functions_mock = {
        database_type: lambda query, fts_top_k, relevant_ids: mock_response
    }
    mocker.patch('App_Function_Libraries.RAG.RAG_Library_2.search_functions', search_functions_mock)

    query = 'test query'
    relevant_ids = "1"

    result = perform_full_text_search(query, database_type, relevant_ids)
    assert result == mock_response


def test_enhanced_rag_pipeline_success(mocker):
    """Test enhanced_rag_pipeline with a successful flow."""
    # Mock config
    mock_config = configparser.ConfigParser()
    mock_config['Embeddings'] = {'provider': 'openai'}
    mocker.patch('App_Function_Libraries.RAG.RAG_Library_2.config', mock_config)

    # Mock search functions
    fts_result = [{'content': 'FTS result', 'id': 1}]
    vector_result = [{'content': 'Vector result'}]

    mock_search = lambda *args, **kwargs: fts_result
    search_functions_mock = {
        "Media DB": mock_search
    }
    mocker.patch('App_Function_Libraries.RAG.RAG_Library_2.search_functions', search_functions_mock)

    mocker.patch(
        'App_Function_Libraries.RAG.RAG_Library_2.perform_vector_search',
        return_value=vector_result
    )

    mocker.patch(
        'App_Function_Libraries.RAG.RAG_Library_2.generate_answer',
        return_value='Generated answer'
    )

    # Mock relevant media IDs
    mocker.patch(
        'App_Function_Libraries.RAG.RAG_Library_2.fetch_relevant_media_ids',
        return_value=[1, 2, 3]
    )

    result = enhanced_rag_pipeline(
        query='test query',
        api_choice='OpenAI',
        keywords='keyword1,keyword2',
        database_types=["Media DB"]
    )

    # Check both vector and FTS results are in context
    assert result['answer'] == 'Generated answer'
    assert 'Vector result' in result['context']
    assert 'FTS result' in result['context']


def test_enhanced_rag_pipeline_error_handling(mocker):
    """Test enhanced_rag_pipeline error handling."""
    mock_config = configparser.ConfigParser()
    mock_config['Embeddings'] = {'provider': 'openai'}
    mocker.patch('App_Function_Libraries.RAG.RAG_Library_2.config', mock_config)

    mock_fetch_keywords_for_media = mocker.patch(
        'App_Function_Libraries.RAG.RAG_Library_2.fetch_relevant_media_ids',
        side_effect=Exception("Fetch error")
    )

    result = enhanced_rag_pipeline(
        query='test query',
        api_choice='OpenAI',
        keywords='keyword1',
        database_types=["Media DB"]
    )

    assert "An error occurred" in result['answer']
    assert result['context'] == ""


def test_generate_answer_success(mocker):
    """Test generate_answer with successful API call."""
    # Mock config
    mock_config = configparser.ConfigParser()
    mock_config['API'] = {'openai_api_key': 'test_key'}
    mocker.patch(
        'App_Function_Libraries.RAG.RAG_Library_2.load_comprehensive_config',
        return_value=mock_config
    )

    # Mock the summarization function
    mock_summarize = mocker.patch(
        'App_Function_Libraries.Summarization.Summarization_General_Lib.summarize_with_openai',
        return_value='API response'
    )

    result = generate_answer('OpenAI', 'Test context', 'Test query')
    assert result == 'API response'


if __name__ == '__main__':
    pytest.main(['-v'])