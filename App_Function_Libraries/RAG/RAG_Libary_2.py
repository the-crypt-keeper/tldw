# RAG_Library_2.py
# Description: This script contains the main RAG pipeline function and related functions for the RAG pipeline.
#
# Import necessary modules and functions
import configparser
import logging
import os
from typing import Dict, Any, List, Optional
# Local Imports
from App_Function_Libraries.RAG.ChromaDB_Library import process_and_store_content, vector_search, chroma_client
from App_Function_Libraries.Summarization.Local_Summarization_Lib import summarize_with_custom_openai
from App_Function_Libraries.Web_Scraping.Article_Extractor_Lib import scrape_article
from App_Function_Libraries.DB.DB_Manager import search_db, fetch_keywords_for_media
from App_Function_Libraries.Utils.Utils import load_comprehensive_config
#
# 3rd-Party Imports
import openai
#
########################################################################################################################
#
# Functions:

# Initialize OpenAI client (adjust this based on your API key management)
openai.api_key = "your-openai-api-key"

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the config file
config_path = os.path.join(current_dir, 'Config_Files', 'config.txt')
# Read the config file
config = configparser.ConfigParser()
# Read the configuration file
config.read('config.txt')

# RAG pipeline function for web scraping
# def rag_web_scraping_pipeline(url: str, query: str, api_choice=None) -> Dict[str, Any]:
#     try:
#         # Extract content
#         try:
#             article_data = scrape_article(url)
#             content = article_data['content']
#             title = article_data['title']
#         except Exception as e:
#             logging.error(f"Error scraping article: {str(e)}")
#             return {"error": "Failed to scrape article", "details": str(e)}
#
#         # Store the article in the database and get the media_id
#         try:
#             media_id = add_media_to_database(url, title, 'article', content)
#         except Exception as e:
#             logging.error(f"Error adding article to database: {str(e)}")
#             return {"error": "Failed to store article in database", "details": str(e)}
#
#         # Process and store content
#         collection_name = f"article_{media_id}"
#         try:
#             # Assuming you have a database object available, let's call it 'db'
#             db = get_database_connection()
#
#             process_and_store_content(
#                 database=db,
#                 content=content,
#                 collection_name=collection_name,
#                 media_id=media_id,
#                 file_name=title,
#                 create_embeddings=True,
#                 create_contextualized=True,
#                 api_name=api_choice
#             )
#         except Exception as e:
#             logging.error(f"Error processing and storing content: {str(e)}")
#             return {"error": "Failed to process and store content", "details": str(e)}
#
#         # Perform searches
#         try:
#             vector_results = vector_search(collection_name, query, k=5)
#             fts_results = search_db(query, ["content"], "", page=1, results_per_page=5)
#         except Exception as e:
#             logging.error(f"Error performing searches: {str(e)}")
#             return {"error": "Failed to perform searches", "details": str(e)}
#
#         # Combine results with error handling for missing 'content' key
#         all_results = []
#         for result in vector_results + fts_results:
#             if isinstance(result, dict) and 'content' in result:
#                 all_results.append(result['content'])
#             else:
#                 logging.warning(f"Unexpected result format: {result}")
#                 all_results.append(str(result))
#
#         context = "\n".join(all_results)
#
#         # Generate answer using the selected API
#         try:
#             answer = generate_answer(api_choice, context, query)
#         except Exception as e:
#             logging.error(f"Error generating answer: {str(e)}")
#             return {"error": "Failed to generate answer", "details": str(e)}
#
#         return {
#             "answer": answer,
#             "context": context
#         }
#
#     except Exception as e:
#         logging.error(f"Unexpected error in rag_pipeline: {str(e)}")
#         return {"error": "An unexpected error occurred", "details": str(e)}



# RAG Search with keyword filtering
def enhanced_rag_pipeline(query: str, api_choice: str, keywords: str = None) -> Dict[str, Any]:
    try:
        # Load embedding provider from config, or fallback to 'openai'
        embedding_provider = config.get('Embeddings', 'provider', fallback='openai')

        # Log the provider used
        logging.debug(f"Using embedding provider: {embedding_provider}")

        # Process keywords if provided
        keyword_list = [k.strip().lower() for k in keywords.split(',')] if keywords else []
        logging.debug(f"enhanced_rag_pipeline - Keywords: {keyword_list}")

        # Fetch relevant media IDs based on keywords if keywords are provided
        relevant_media_ids = fetch_relevant_media_ids(keyword_list) if keyword_list else None
        logging.debug(f"enhanced_rag_pipeline - relevant media IDs: {relevant_media_ids}")

        # Perform vector search
        vector_results = perform_vector_search(query, relevant_media_ids)
        logging.debug(f"enhanced_rag_pipeline - Vector search results: {vector_results}")

        # Perform full-text search
        fts_results = perform_full_text_search(query, relevant_media_ids)
        logging.debug(f"enhanced_rag_pipeline - Full-text search results: {fts_results}")

        # Combine results
        all_results = vector_results + fts_results

        # FIXME - Apply Re-Ranking of results here
        apply_re_ranking = False
        if apply_re_ranking:
            # Implement re-ranking logic here
            pass
        # Extract content from results
        context = "\n".join([result['content'] for result in all_results[:10]])  # Limit to top 10 results
        logging.debug(f"Context length: {len(context)}")
        logging.debug(f"Context: {context[:200]}")
        # Generate answer using the selected API
        answer = generate_answer(api_choice, context, query)

        if not all_results:
            logging.info(f"No results found. Query: {query}, Keywords: {keywords}")
            return {
                "answer": "No relevant information based on your query and keywords were found in the database. Your query has been directly passed to the LLM, and here is its answer: \n\n" + answer,
                "context": "No relevant information based on your query and keywords were found in the database. The only context used was your query: \n\n" + query
            }

        return {
            "answer": answer,
            "context": context
        }

    except Exception as e:
        logging.error(f"Error in enhanced_rag_pipeline: {str(e)}")
        return {
            "answer": "An error occurred while processing your request.",
            "context": ""
        }

# Need to write a test for this function FIXME
def generate_answer(api_choice: str, context: str, query: str) -> str:
    logging.debug("Entering generate_answer function")
    config = load_comprehensive_config()
    logging.debug(f"Config sections: {config.sections()}")
    prompt = f"Context: {context}\n\nQuestion: {query}"
    if api_choice == "OpenAI":
        from App_Function_Libraries.Summarization.Summarization_General_Lib import summarize_with_openai
        return summarize_with_openai(config['API']['openai_api_key'], prompt, "")

    elif api_choice == "Anthropic":
        from App_Function_Libraries.Summarization.Summarization_General_Lib import summarize_with_anthropic
        return summarize_with_anthropic(config['API']['anthropic_api_key'], prompt, "")

    elif api_choice == "Cohere":
        from App_Function_Libraries.Summarization.Summarization_General_Lib import summarize_with_cohere
        return summarize_with_cohere(config['API']['cohere_api_key'], prompt, "")

    elif api_choice == "Groq":
        from App_Function_Libraries.Summarization.Summarization_General_Lib import summarize_with_groq
        return summarize_with_groq(config['API']['groq_api_key'], prompt, "")

    elif api_choice == "OpenRouter":
        from App_Function_Libraries.Summarization.Summarization_General_Lib import summarize_with_openrouter
        return summarize_with_openrouter(config['API']['openrouter_api_key'], prompt, "")

    elif api_choice == "HuggingFace":
        from App_Function_Libraries.Summarization.Summarization_General_Lib import summarize_with_huggingface
        return summarize_with_huggingface(config['API']['huggingface_api_key'], prompt, "")

    elif api_choice == "DeepSeek":
        from App_Function_Libraries.Summarization.Summarization_General_Lib import summarize_with_deepseek
        return summarize_with_deepseek(config['API']['deepseek_api_key'], prompt, "")

    elif api_choice == "Mistral":
        from App_Function_Libraries.Summarization.Summarization_General_Lib import summarize_with_mistral
        return summarize_with_mistral(config['API']['mistral_api_key'], prompt, "")

    # Local LLM APIs
    elif api_choice == "Local-LLM":
        from App_Function_Libraries.Summarization.Local_Summarization_Lib import summarize_with_local_llm
        # FIXME
        return summarize_with_local_llm(config['Local-API']['local_llm_path'], prompt, "")

    elif api_choice == "Llama.cpp":
        from App_Function_Libraries.Summarization.Local_Summarization_Lib import summarize_with_llama
        return summarize_with_llama(prompt, "", config['Local-API']['llama_api_key'], None, None)

    elif api_choice == "Kobold":
        from App_Function_Libraries.Summarization.Local_Summarization_Lib import summarize_with_kobold
        return summarize_with_kobold(prompt, config['Local-API']['kobold_api_key'], "", system_message=None, temp=None)

    elif api_choice == "Ooba":
        from App_Function_Libraries.Summarization.Local_Summarization_Lib import summarize_with_oobabooga
        return summarize_with_oobabooga(prompt, config['Local-API']['ooba_api_key'], custom_prompt="", system_message=None, temp=None)

    elif api_choice == "TabbyAPI":
        from App_Function_Libraries.Summarization.Local_Summarization_Lib import summarize_with_tabbyapi
        return summarize_with_tabbyapi(prompt, None, None, None, None, )

    elif api_choice == "vLLM":
        from App_Function_Libraries.Summarization.Local_Summarization_Lib import summarize_with_vllm
        return summarize_with_vllm(prompt, "", config['Local-API']['vllm_api_key'], None, None)

    elif api_choice.lower() == "ollama":
        from App_Function_Libraries.Summarization.Local_Summarization_Lib import summarize_with_ollama
        return summarize_with_ollama(prompt, "", config['Local-API']['ollama_api_key'], None, None, None)

    elif api_choice.lower() == "custom_openai_api":
        logging.debug(f"RAG Answer Gen: Trying with Custom_OpenAI API")
        summary = summarize_with_custom_openai(prompt, "", config['API']['custom_openai_api_key'], None,
                                               None)

    else:
        raise ValueError(f"Unsupported API choice: {api_choice}")


def perform_vector_search(query: str, relevant_media_ids: List[str] = None) -> List[Dict[str, Any]]:
    all_collections = chroma_client.list_collections()
    vector_results = []
    for collection in all_collections:
        collection_results = vector_search(collection.name, query, k=5)
        filtered_results = [
            result for result in collection_results
            if relevant_media_ids is None or result['metadata'].get('media_id') in relevant_media_ids
        ]
        vector_results.extend(filtered_results)
    return vector_results


def perform_full_text_search(query: str, relevant_media_ids: List[str] = None) -> List[Dict[str, Any]]:
    fts_results = search_db(query, ["content"], "", page=1, results_per_page=5)
    filtered_fts_results = [
        {
            "content": result['content'],
            "metadata": {"media_id": result['id']}
        }
        for result in fts_results
        if relevant_media_ids is None or result['id'] in relevant_media_ids
    ]
    return filtered_fts_results


def fetch_relevant_media_ids(keywords: List[str]) -> List[int]:
    relevant_ids = set()
    try:
        for keyword in keywords:
            media_ids = fetch_keywords_for_media(keyword)
            relevant_ids.update(media_ids)
    except Exception as e:
        logging.error(f"Error fetching relevant media IDs: {str(e)}")
    return list(relevant_ids)


def filter_results_by_keywords(results: List[Dict[str, Any]], keywords: List[str]) -> List[Dict[str, Any]]:
    if not keywords:
        return results

    filtered_results = []
    for result in results:
        try:
            metadata = result.get('metadata', {})
            if metadata is None:
                logging.warning(f"No metadata found for result: {result}")
                continue
            if not isinstance(metadata, dict):
                logging.warning(f"Unexpected metadata type: {type(metadata)}. Expected dict.")
                continue

            media_id = metadata.get('media_id')
            if media_id is None:
                logging.warning(f"No media_id found in metadata: {metadata}")
                continue

            media_keywords = fetch_keywords_for_media(media_id)
            if any(keyword.lower() in [mk.lower() for mk in media_keywords] for keyword in keywords):
                filtered_results.append(result)
        except Exception as e:
            logging.error(f"Error processing result: {result}. Error: {str(e)}")

    return filtered_results

# FIXME: to be implememted
def extract_media_id_from_result(result: str) -> Optional[int]:
    # Implement this function based on how you store the media_id in your results
    # For example, if it's stored at the beginning of each result:
    try:
        return int(result.split('_')[0])
    except (IndexError, ValueError):
        logging.error(f"Failed to extract media_id from result: {result}")
        return None

#
#
########################################################################################################################


# Function to preprocess and store all existing content in the database
# def preprocess_all_content(database, create_contextualized=True, api_name="gpt-3.5-turbo"):
#     unprocessed_media = get_unprocessed_media()
#     total_media = len(unprocessed_media)
#
#     for index, row in enumerate(unprocessed_media, 1):
#         media_id, content, media_type, file_name = row
#         collection_name = f"{media_type}_{media_id}"
#
#         logger.info(f"Processing media {index} of {total_media}: ID {media_id}, Type {media_type}")
#
#         try:
#             process_and_store_content(
#                 database=database,
#                 content=content,
#                 collection_name=collection_name,
#                 media_id=media_id,
#                 file_name=file_name or f"{media_type}_{media_id}",
#                 create_embeddings=True,
#                 create_contextualized=create_contextualized,
#                 api_name=api_name
#             )
#
#             # Mark the media as processed in the database
#             mark_media_as_processed(database, media_id)
#
#             logger.info(f"Successfully processed media ID {media_id}")
#         except Exception as e:
#             logger.error(f"Error processing media ID {media_id}: {str(e)}")
#
#     logger.info("Finished preprocessing all unprocessed content")

############################################################################################################
#
# ElasticSearch Retriever

# https://github.com/langchain-ai/langchain/tree/44e3e2391c48bfd0a8e6a20adde0b6567f4f43c3/templates/rag-elasticsearch
#
# https://github.com/langchain-ai/langchain/tree/44e3e2391c48bfd0a8e6a20adde0b6567f4f43c3/templates/rag-self-query

#
# End of RAG_Library_2.py
############################################################################################################
