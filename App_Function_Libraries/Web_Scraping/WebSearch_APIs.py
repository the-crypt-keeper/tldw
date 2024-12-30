# WebSearch_APIs.py
# Description: This file contains the functions that are used for performing queries against various Search Engine APIs
#
# Imports
import json
import logging
import re
from html import unescape
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse, urlencode, unquote
#
# 3rd-Party Imports
import requests
from lxml.etree import _Element
from lxml.html import document_fromstring
from requests import RequestException
from requests.adapters import HTTPAdapter
from urllib3 import Retry

from App_Function_Libraries.Chat.Chat_Functions import chat_api_call
#
# Local Imports
from App_Function_Libraries.Utils.Utils import loaded_config_data
#
#######################################################################################################################
#
# Functions:
# 1. analyze_question
#
#######################################################################################################################
#
# Functions:

######################### Main Orchestration Workflow #########################
#
def process_question(question: str, search_params: Dict) -> Dict:
    """
    Orchestrates the entire pipeline:
      1. Optionally generate sub-queries (if subquery_generation=True).
      2. Perform web searches for each query (the original and sub-queries).
      3. Score results for relevance; filter them.
      4. Aggregate final answer from relevant results.

    Args:
        question (str): The user's original question or query.
        search_params (Dict): A dictionary containing parameters for performing web searches
                              and specifying LLM endpoints.

    Returns:
        Dict: A dictionary containing all relevant data, including results from each sub-query.

        A dictionary containing parameters for performing a web search and processing the results.

        Dict Parameters:
            engine (str): The search engine to use (e.g., "google", "bing", "brave", "duckduckgo", etc.).
            content_country (str): The country code for content localization (e.g., "US", "UK", "DE").
            search_lang (str): The language for the search query (e.g., "en" for English).
            output_lang (str): The desired language for the search results.
            result_count (int): The number of results to return.
            date_range (str or None): The time range for the search results (e.g., "y" for the past year).
            safesearch (str or None): The safe search setting (e.g., "moderate", "strict", "off").
            site_blacklist (list or None): A list of sites to exclude from the search results.
            exactTerms (str or None): Terms that must appear in the search results.
            excludeTerms (str or None): Terms that must not appear in the search results.
            filter (str or None): Any additional filtering criteria.
            geolocation (str or None): Geographical location for search results.
            search_result_language (str or None): The language of the search results.
            sort_results_by (str or None): Criteria for sorting the results.
            subquery_generation (bool): Whether to generate sub-queries.
            subquery_generation_api (str): The API to use for sub-query generation (e.g., "openai", "anthropic", "deepseek").
            relevance_analysis_llm (str): The LLM model to use for relevance analysis (e.g., "openai", "anthropic", "deepseek").
            final_answer_llm (str): The LLM model to use for generating the final answer (e.g., "openai", "anthropic", "deepseek").

        Example:
            search_params = {
                "engine": "google",
                "content_country": "US",
                "search_lang": "en",
                "output_lang": "en",
                "result_count": 10,
                "date_range": "y",
                "safesearch": "moderate",
                "site_blacklist": ["example.com", "spam-site.com"],
                "exactTerms": None,
                "excludeTerms": None,
                "filter": None,
                "geolocation": None,
                "search_result_language": None,
                "sort_results_by": None,
                "subquery_generation": True,
                "subquery_generation_llm": "openai",
                "relevance_analysis_llm": "openai",
                "final_answer_llm": "openai"
            }
    """
    logging.info(f"Starting process_question with query: {question}")

    # 1. Generate sub-queries if requested
    sub_queries = []
    sub_query_dict = {
        "main_goal": question,
        "sub_questions": [],
        "search_queries": [],
        "analysis_prompt": None
    }

    if search_params.get("subquery_generation", False):
        api_endpoint = search_params.get("subquery_generation_llm", "openai")
        sub_query_dict = analyze_question(question, api_endpoint)
        sub_queries = sub_query_dict.get("sub_questions", [])

    # Merge original question with sub-queries
    all_queries = [question] + sub_queries

    # 2. Perform searches and accumulate all raw results
    all_results: List[Dict] = []
    for q in all_queries:
        # FIXME - change raw_results to proper dict format
        raw_results = perform_websearch(
            search_engine=search_params.get('engine', 'google'),
            search_query=q,
            content_country=search_params.get('content_country', 'US'),
            search_lang=search_params.get('search_lang', 'en'),
            output_lang=search_params.get('output_lang', 'en'),
            result_count=search_params.get('result_count', 10),
            date_range=search_params.get('date_range'),
            safesearch=search_params.get('safesearch', 'moderate'),
            site_blacklist=search_params.get('site_blacklist', []),
            exactTerms=search_params.get('exactTerms'),
            excludeTerms=search_params.get('excludeTerms'),
            filter=search_params.get('filter'),
            geolocation=search_params.get('geolocation'),
            search_result_language=search_params.get('search_result_language'),
            sort_results_by=search_params.get('sort_results_by')
        )

        # Validate raw_results
        # FIXME - Account for proper structure of returned web_search_results_dict dictionary
        if not isinstance(raw_results, dict) or "processing_error" in raw_results:
            logging.warning(f"Error or invalid data returned for query '{q}': {raw_results}")
            continue

        results_list = raw_results.get("results", [])
        all_results.extend(results_list)

    # 3. Score/filter (placeholder)
    relevant_results = {}
    for r in all_results:
        # FIXME - Put in proper args / Ensure this works
        # search_results: List[Dict],
        # original_question: str,
        # sub_questions: List[str],
        # api_endpoint: str
        list_of_relevant_articles = search_result_relevance(
            all_results,
            question,
            sub_query_dict['sub_questions'],
            search_params.get('relevance_analysis_llm')
        )
        if list_of_relevant_articles:
            relevant_results[r] = list_of_relevant_articles

    # 4. Summarize/aggregate final answer
    final_answer = aggregate_results(
        #  FIXME - Add proper Args / Ensure this works
        # web_search_results_dict: Dict,
        # report_language: str,
        # aggregation_api_endpoint: str
        # FIXME - Proper datatypes/expectations
        web_search_results_dict,
        report_language=search_params.get('output_lang', 'en'),
        api_endpoint=search_params.get('final_answer_llm')
    )

    # Return the final data
    # FIXME - Return full query details for debugging and analysis
    return web_search_results_dict


######################### Question Analysis #########################
#
#
def analyze_question(question: str, api_endpoint) -> Dict:
    logging.debug(f"Analyzing question: {question} with API endpoint: {api_endpoint}")
    """
    Analyzes the input question and generates sub-questions

    Returns:
        Dict containing:
        - main_goal: str
        - sub_questions: List[str]
        - search_queries: List[str]
        - analysis_prompt: str
    """
    original_query = question
    sub_question_generation_prompt = f"""
            You are an AI assistant that helps generate search queries. Given an original query, suggest alternative search queries that could help find relevant information. Your goal is to generate queries that are diverse, specific, and highly relevant to the original query, ensuring comprehensive coverage of the topic.

            Important instructions:
            1. Generate between 2 and 6 queries unless a fixed count is specified. Generate more queries for complex or multifaceted topics and fewer for simple or straightforward ones.
            2. Ensure the queries are diverse, covering different aspects or perspectives of the original query, while remaining highly relevant to its core intent.
            3. Prefer specific queries over general ones, as they are more likely to yield targeted and useful results.
            4. If the query involves comparing two topics, generate separate queries for each topic.
            5. If previous queries and an answer are provided, generate new queries that address the shortcomings of the previous answer and avoid repeating the previous queries.
            6. If the original query is broad or ambiguous, generate queries that explore specific subtopics or clarify the intent.
            7. If the query is too specific or unclear, generate queries that explore related or broader topics to ensure useful results.
            8. Return the queries as a JSON array in the format ["query_1", "query_2", ...].

            Examples:
            1. For the query "What are the benefits of exercise?", generate queries like:
               ["health benefits of physical activity", "mental health benefits of exercise", "long-term effects of regular exercise", "how exercise improves cardiovascular health", "role of exercise in weight management"]

            2. For the query "Compare Python and JavaScript", generate queries like:
               ["key features of Python programming language", "advantages of JavaScript for web development", "use cases for Python vs JavaScript", "performance comparison of Python and JavaScript", "ease of learning Python vs JavaScript"]

            3. For the query "How does climate change affect biodiversity?", generate queries like:
               ["impact of climate change on species extinction", "effects of global warming on ecosystems", "role of climate change in habitat loss", "how rising temperatures affect marine biodiversity", "climate change and its impact on migratory patterns"]

            4. For the query "Best practices for remote work", generate queries like:
               ["tips for staying productive while working from home", "how to maintain work-life balance in remote work", "tools for effective remote team collaboration", "managing communication in remote teams", "ergonomic setup for home offices"]

            5. For the query "What is quantum computing?", generate queries like:
               ["basic principles of quantum computing", "applications of quantum computing in real-world problems", "difference between classical and quantum computing", "key challenges in developing quantum computers", "future prospects of quantum computing"]

            Original query: {original_query}
            """

    input_data = "Follow the above instructions."

    sub_questions: List[str] = []
    try:
        for attempt in range(3):
            logging.info(f"Generating sub-questions (attempt {attempt + 1})")
            response = chat_api_call(api_endpoint, None, input_data, sub_question_generation_prompt, temp=0.7)
            if response:
                try:
                    # Try to parse as JSON first
                    parsed_response = json.loads(response)
                    sub_questions = parsed_response.get("sub_questions", [])
                    if sub_questions:
                        logging.info("Successfully generated sub-questions from JSON")
                        break
                except json.JSONDecodeError:
                    # If JSON parsing fails, attempt a regex-based fallback
                    logging.warning("Failed to parse as JSON. Attempting regex extraction.")
                    matches = re.findall(r'"([^"]*)"', response)
                    sub_questions = matches if matches else []
                    if sub_questions:
                        logging.info("Successfully extracted sub-questions using regex")
                        break

        # If still no sub-questions, log an error or handle appropriately
        if not sub_questions:
            logging.error("Failed to extract sub-questions from API response after all attempts.")

    except Exception as e:
        logging.error(f"Error generating sub-questions: {str(e)}")

    # Construct and return the result dictionary
    logging.info("Sub-questions generated successfully")
    return {
        "main_goal": original_query,
        "sub_questions": sub_questions,
        "search_queries": sub_questions,
        "analysis_prompt": sub_question_generation_prompt
    }


######################### Relevance Analysis #########################
#
def search_result_relevance(
    search_results: List[Dict],
    original_question: str,
    sub_questions: List[str],
    api_endpoint: str
) -> Dict[str, Dict]:
    """
    Evaluate whether each search result is relevant to the original question and sub-questions.

    Args:
        search_results (List[Dict]): List of search results to evaluate.
        original_question (str): The original question posed by the user.
        sub_questions (List[str]): List of sub-questions generated from the original question.
        api_endpoint (str): The LLM or API endpoint to use for relevance analysis.

    Returns:
        Dict[str, Dict]: A dictionary of relevant results, keyed by a unique ID or index.
    """
    relevant_results = {}

    for idx, result in enumerate(search_results):
        content = result.get("content", "")
        if not content:
            logging.error("No Content found in search results array!")

        eval_prompt = f"""
                Given the following search results for the user's question: "{original_question}" and the generated sub-questions: {sub_questions}, evaluate the relevance of the search result to the user's question.
                Explain your reasoning for selection.

                Search Results:
                {content}

                Instructions:
                1. You MUST only answer TRUE or False while providing your reasoning for your answer.
                2. A result is relevant if the result most likely contains comprehensive and relevant information to answer the user's question.
                3. Provide a brief reason for selection.

                You MUST respond using EXACTLY this format and nothing else:

                Selected Answer: [True or False]
                Reasoning: [Your reasoning for the selections]
                """
        input_data = "Evaluate the relevance of the search result."

        try:
            # Perform API call to evaluate relevance
            relevancy_result = chat_api_call(
                api_endpoint=api_endpoint,
                api_key=None,
                input_data=input_data,
                prompt=eval_prompt,
                temp=0.7
            )

            if relevancy_result:
                # Extract the selected answer and reasoning via regex
                selected_answer_match = re.search(
                    r"Selected Answer:\s*(True|False)",
                    relevancy_result,
                    re.IGNORECASE
                )
                reasoning_match = re.search(
                    r"Reasoning:\s*(.+)",
                    relevancy_result,
                    re.IGNORECASE
                )

                if selected_answer_match and reasoning_match:
                    is_relevant = selected_answer_match.group(1).strip().lower() == "true"
                    reasoning = reasoning_match.group(1).strip()

                    if is_relevant:
                        # Use the 'id' from the result if available, otherwise use idx
                        result_id = result.get("id", str(idx))
                        relevant_results[result_id] = {
                            "content": content,
                            "reasoning": reasoning
                        }
                        logging.info(f"Relevant result found: ID={result_id} Reasoning={reasoning}")
                    else:
                        logging.info(f"Irrelevant result: {reasoning}")

                else:
                    logging.warning("Failed to parse the API response for relevance analysis.")
        except Exception as e:
            logging.error(f"Error during relevance evaluation for result idx={idx}: {e}")

    return relevant_results


######################### Result Aggregation & Combination #########################
#
def aggregate_results(
    relevant_results: Dict[str, Dict],
    question: str,
    sub_questions: List[str],
    api_endpoint: str
) -> Dict:
    """
    Combines and summarizes relevant results into a final answer.

    Args:
        relevant_results (Dict[str, Dict]): Dictionary of relevant articles/content.
        question (str): Original question.
        sub_questions (List[str]): List of sub-questions.
        api_endpoint (str): LLM or API endpoint for summarization.

    Returns:
        Dict containing:
        - summary (str): Final summarized answer.
        - evidence (List[Dict]): List of relevant content items included in the summary.
        - confidence (float): A rough confidence score (placeholder).
    """
    if not relevant_results:
        return {
            "summary": "No relevant results found. Unable to provide an answer.",
            "evidence": [],
            "confidence": 0.0
        }

    # Concatenate relevant contents for summarization
    concatenated_texts = "\n\n".join(
        f"ID: {rid}\nContent: {res['content']}\nReasoning: {res['reasoning']}"
        for rid, res in relevant_results.items()
    )

    # Example summarization prompt
    summarize_prompt = f"""
Please provide a concise summary that answers the question: "{question}"

Relevant sub-questions: {sub_questions}

Here are the relevant excerpts from search results:
{concatenated_texts}

Instructions:
1. Provide a concise summary that incorporates the key points.
2. Avoid quoting large passages verbatim; aim to synthesize the main ideas.
3. Make sure your answer is coherent, logically consistent, and addresses the question directly.
"""

    input_data = "Summarize the relevant results."

    try:
        summary_response = chat_api_call(
            api_endpoint=api_endpoint,
            api_key=None,
            input_data=input_data,
            prompt=summarize_prompt,
            temp=0.7
        )
        if summary_response:
            # You could do further parsing or confidence estimation here
            return {
                "summary": summary_response,
                "evidence": list(relevant_results.values()),
                "confidence": 0.9  # Hardcoded or computed as needed
            }
    except Exception as e:
        logging.error(f"Error aggregating results: {e}")

    return {
        "summary": "Could not summarize the results due to an error.",
        "evidence": list(relevant_results.values()),
        "confidence": 0.0
    }


#
# End of Orchestration functions
#######################################################################################################################


#######################################################################################################################
#
# Search Engine Functions

# FIXME
def perform_websearch(search_engine, search_query, content_country, search_lang, output_lang, result_count, date_range=None,
                      safesearch=None, site_blacklist=None, exactTerms=None, excludeTerms=None, filter=None, geolocation=None, search_result_language=None, sort_results_by=None):
    if search_engine.lower() == "baidu":
        web_search_results = search_web_baidu(search_query, None, None)
        processed_results = process_web_search_results(web_search_results, "baidu")
        return processed_results

    elif search_engine.lower() == "bing":
        web_search_results = search_web_bing(search_query, search_lang, content_country, date_range, result_count)
        processed_results = process_web_search_results(web_search_results, "bing")
        return processed_results

    elif search_engine.lower() == "brave":
            web_search_results = search_web_brave(search_query, content_country, search_lang, output_lang, result_count, safesearch,
                                    site_blacklist, date_range)
            processed_results = process_web_search_results(web_search_results, "brave")
            return processed_results

    elif search_engine.lower() == "duckduckgo":
        web_search_results = search_web_duckduckgo(search_query, content_country, date_range, result_count)
        processed_results = process_web_search_results(web_search_results, "ddg")
        return processed_results

    elif search_engine.lower() == "google":
        web_search_results = search_web_google(search_query, result_count, content_country, date_range, exactTerms,
                                 excludeTerms, filter, geolocation, output_lang,
                      search_result_language, safesearch, site_blacklist, sort_results_by)
        processed_results = process_web_search_results(web_search_results, "google")
        return processed_results

    elif search_engine.lower() == "kagi":
        web_search_results = search_web_kagi(search_query, content_country)
        processed_results = process_web_search_results(web_search_results, "kagi")
        return processed_results

    elif search_engine.lower() == "serper":
        web_search_results = search_web_serper()
        processed_results = process_web_search_results(web_search_results, "serper")
        return processed_results

    elif search_engine.lower() == "tavily":
        web_search_results = search_web_tavily(search_query, result_count, site_blacklist)
        processed_results = process_web_search_results(web_search_results, "tavily")
        return processed_results

    elif search_engine.lower() == "searx":
        web_search_results = search_web_searx(search_query, language='auto', time_range='', safesearch=0, pageno=1, categories='general')
        processed_results = process_web_search_results(web_search_results, "bing")
        return processed_results

    elif search_engine.lower() == "yandex":
        web_search_results = search_web_yandex()
        processed_results = process_web_search_results(web_search_results, "bing")
        return processed_results

    else:
        return f"Error: Invalid Search Engine Name {search_engine}"


#
######################### Search Result Parsing ##################################################################
#

def process_web_search_results(search_results: Dict, search_engine: str) -> Dict:
    """
    Processes search results from a search engine and formats them into a standardized dictionary structure.

    Args:
        search_results (Dict): The raw search results from the search engine.
        search_engine (str): The name of the search engine (e.g., "Google", "Bing").

    Returns:
        Dict: A dictionary containing the processed search results in the specified structure.

    web_search_results_dict = {
        "search_engine": search_engine,
        "search_query": search_results.get("search_query", ""),
        "content_country": search_results.get("content_country", ""),
        "search_lang": search_results.get("search_lang", ""),
        "output_lang": search_results.get("output_lang", ""),
        "result_count": search_results.get("result_count", 0),
        "date_range": search_results.get("date_range", None),
        "safesearch": search_results.get("safesearch", None),
        "site_blacklist": search_results.get("site_blacklist", None),
        "exactTerms": search_results.get("exactTerms", None),
        "excludeTerms": search_results.get("excludeTerms", None),
        "filter": search_results.get("filter", None),
        "geolocation": search_results.get("geolocation", None),
        "search_result_language": search_results.get("search_result_language", None),
        "sort_results_by": search_results.get("sort_results_by", None),
        "results": [
            {
                "title": str,
                "url": str,
                "content": str,
                "metadata": {
                    "date_published": Optional[str],
                    "author": Optional[str],
                    "source": Optional[str],
                    "language": Optional[str],
                    "relevance_score": Optional[float],
                    "snippet": Optional[str]
                }
            },
        "total_results_found": search_results.get("total_results_found", 0),
        "search_time": search_results.get("search_time", 0.0),
        "error": search_results.get("error", None),
        "processing_error": None
    }
    """
    # Initialize the output dictionary with default values
    web_search_results_dict = {
        "search_engine": search_engine,
        "search_query": search_results.get("search_query", ""),
        "content_country": search_results.get("content_country", ""),
        "search_lang": search_results.get("search_lang", ""),
        "output_lang": search_results.get("output_lang", ""),
        "result_count": search_results.get("result_count", 0),
        "date_range": search_results.get("date_range", None),
        "safesearch": search_results.get("safesearch", None),
        "site_blacklist": search_results.get("site_blacklist", None),
        "exactTerms": search_results.get("exactTerms", None),
        "excludeTerms": search_results.get("excludeTerms", None),
        "filter": search_results.get("filter", None),
        "geolocation": search_results.get("geolocation", None),
        "search_result_language": search_results.get("search_result_language", None),
        "sort_results_by": search_results.get("sort_results_by", None),
        "results": [
            {
                "title": str,
                "url": str,
                "content": str,
                "metadata": {
                    "date_published": Optional[str],
                    "author": Optional[str],
                    "source": Optional[str],
                    "language": Optional[str],
                    "relevance_score": Optional[float],
                    "snippet": Optional[str]
                }
            },
        ],
        "total_results_found": search_results.get("total_results_found", 0),
        "search_time": search_results.get("search_time", 0.0),
        "error": search_results.get("error", None),
        "processing_error": None
    }
    try:
        if search_engine.lower() == "baidu":
            pass
        elif search_engine.lower() == "bing":
            parsed_results = parse_bing_results(search_results, web_search_results_dict)

        elif search_engine.lower() == "brave":
            parse_brave_results(search_results, web_search_results_dict)
            pass

        elif search_engine.lower() == "duckduckgo":
            parsed_results = parse_duckduckgo_results(search_results, web_search_results_dict)

        elif search_engine.lower() == "google":
            parsed_results = parse_google_results(search_results, web_search_results_dict)

        elif search_engine.lower() == "kagi":
            parsed_results = parse_kagi_results(search_results, web_search_results_dict)

        elif search_engine.lower() == "serper":
            parsed_results = parse_serper_results(search_results, web_search_results_dict)

        elif search_engine.lower() == "tavily":
            parsed_results = parse_tavily_results(search_results, web_search_results_dict)

        elif search_engine.lower() == "searx":
            parsed_results = parse_searx_results(search_results, web_search_results_dict)

        elif search_engine.lower() == "yandex":
            parsed_results = parse_yandex_results(search_results, web_search_results_dict)

        else:
            web_search_results_dict["processing_error"] = f"Error: Invalid Search Engine Name {search_engine}"
            raise ValueError(f"Error: Invalid Search Engine Name {search_engine}")
    except Exception as e:
        web_search_results_dict["processing_error"] = f"Error processing search results: {str(e)}"
        raise

    # Process individual search results
    for result in search_results.get("results", []):
        processed_result = {
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "content": result.get("content", ""),
            "metadata": {
                "date_published": result.get("metadata", {}).get("date_published", None),
                "author": result.get("metadata", {}).get("author", None),
                "source": result.get("metadata", {}).get("source", None),
                "language": result.get("metadata", {}).get("language", None),
                "relevance_score": result.get("metadata", {}).get("relevance_score", None),
                "snippet": result.get("metadata", {}).get("snippet", None)
            }
        }
        web_search_results_dict["results"].append(processed_result)

    return web_search_results_dict


def parse_html_search_results_generic(soup):
    results = []
    for result in soup.find_all('div', class_='result'):
        title = result.find('h3').text if result.find('h3') else ''
        url = result.find('a', class_='url')['href'] if result.find('a', class_='url') else ''
        content = result.find('p', class_='content').text if result.find('p', class_='content') else ''
        published_date = result.find('span', class_='published_date').text if result.find('span',
                                                                                          class_='published_date') else ''

        results.append({
            'title': title,
            'url': url,
            'content': content,
            'publishedDate': published_date
        })
    return results


######################### Baidu Search #########################
#
# https://cloud.baidu.com/doc/APIGUIDE/s/Xk1myz05f
# https://oxylabs.io/blog/how-to-scrape-baidu-search-results
def search_web_baidu(arg1, arg2, arg3):
    pass


def test_baidu_search(arg1, arg2, arg3):
    result = search_web_baidu(arg1, arg2, arg3)
    return result

def search_parse_baidu_results():
    pass


######################### Bing Search #########################
#
# https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/overview0
# https://learn.microsoft.com/en-us/bing/search-apis/bing-news-search/overview
# https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/quickstarts/rest/python
# https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/reference/query-parameters
# Country/Language code: https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/reference/market-codes#country-codes
# https://github.com/Azure-Samples/cognitive-services-REST-api-samples/tree/master/python/Search
def search_web_bing(search_query, bing_lang, bing_country, result_count=None, bing_api_key=None,
                    date_range=None):
    # Load Search API URL from config file
    search_url = loaded_config_data['search_engines']['bing_search_api_url']

    if not bing_api_key:
        # load key from config file
        bing_api_key = loaded_config_data['search_engines']['bing_search_api_key']
        if not bing_api_key:
            raise ValueError("Please Configure a valid Bing Search API key")

    if not result_count:
        # Perform check in config file for default search result count
        answer_count = loaded_config_data['search_engines']['search_result_max']
    else:
        answer_count = result_count

    # date_range = "day", "week", "month", or `YYYY-MM-DD..YYYY-MM-DD`
    if not date_range:
         date_range = None

    # Language settings
    if not bing_lang:
        # do config check for default search language
        setlang = bing_lang

    # Returns content for this Country market code
    if not bing_country:
        # do config check for default search country
        bing_country = loaded_config_data['search_engines']['bing_country_code']
    else:
        setcountry = bing_country
    # Construct a request
    mkt = 'en-US'
    params = {'q': search_query, 'mkt': mkt}
#    params = {"q": search_query, "mkt": bing_country, "textDecorations": True, "textFormat": "HTML", "count": answer_count,
#             "freshness": date_range, "promote": "webpages", "safeSearch": "Moderate"}
    headers = {'Ocp-Apim-Subscription-Key': bing_api_key}

    # Call the API
    try:
        response = requests.get(search_url, headers=headers, params=params)
        response.raise_for_status()

        logging.debug("Headers:  ")
        logging.debug(response.headers)

        logging.debug("JSON Response: ")
        logging.debug(response.json())
        bing_search_results = response.json()
        return bing_search_results
    except Exception as ex:
        raise ex


def test_search_web_bing():
    search_query = "How can I get started learning machine learning?"
    bing_lang = "en"
    bing_country =  "US"
    result_count = 10
    bing_api_key = None
    date_range = None
    result = search_web_bing(search_query, bing_lang, bing_country, result_count, bing_api_key, date_range)
    print("Bing Search Results:")
    print(result)


# FIXME - untested
def parse_bing_results(raw_results: Dict, output_dict: Dict) -> None:
    """
    Parse Bing search results and update the output dictionary

    Args:
        raw_results (Dict): Raw Bing API response
        output_dict (Dict): Dictionary to store processed results
    """
    try:
        # Extract web pages results
        if "webPages" in raw_results:
            web_pages = raw_results["webPages"]
            output_dict["total_results_found"] = web_pages.get("totalEstimatedMatches", 0)

            for result in web_pages.get("value", []):
                processed_result = {
                    "title": result.get("name", ""),
                    "url": result.get("url", ""),
                    "content": result.get("snippet", ""),
                    "metadata": {
                        "date_published": None,  # Bing doesn't typically provide this
                        "author": None,  # Bing doesn't typically provide this
                        "source": result.get("displayUrl", None),
                        "language": None,  # Could be extracted from result.get("language") if available
                        "relevance_score": None,  # Could be calculated from result.get("rank") if available
                        "snippet": result.get("snippet", None)
                    }
                }
                output_dict["results"].append(processed_result)

        # Optionally process other result types
        if "news" in raw_results:
            for news_item in raw_results["news"].get("value", []):
                processed_result = {
                    "title": news_item.get("name", ""),
                    "url": news_item.get("url", ""),
                    "content": news_item.get("description", ""),
                    "metadata": {
                        "date_published": news_item.get("datePublished", None),
                        "author": news_item.get("provider", [{}])[0].get("name", None),
                        "source": news_item.get("provider", [{}])[0].get("name", None),
                        "language": None,
                        "relevance_score": None,
                        "snippet": news_item.get("description", None)
                    }
                }
                output_dict["results"].append(processed_result)

        # Add spell suggestions if available
        if "spellSuggestion" in raw_results:
            output_dict["spell_suggestions"] = raw_results["spellSuggestion"]

        # Add related searches if available
        if "relatedSearches" in raw_results:
            output_dict["related_searches"] = [
                item.get("text", "")
                for item in raw_results["relatedSearches"].get("value", [])
            ]

    except Exception as e:
        output_dict["processing_error"] = f"Error processing Bing results: {str(e)}"


def test_parse_bing_results():
    pass


######################### Brave Search #########################
#
# https://brave.com/search/api/
# https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-brave-search/README.md
def search_web_brave(search_term, country, search_lang, ui_lang, result_count, safesearch="moderate",
                     brave_api_key=None, result_filter=None, search_type="ai", date_range=None):
    search_url = "https://api.search.brave.com/res/v1/web/search"
    if not brave_api_key:
        # load key from config file
        brave_api_key = loaded_config_data['search_engines']['brave_search_api_key']
        if not brave_api_key:
            raise ValueError("Please provide a valid Brave Search API subscription key")
    if not country:
        brave_country = loaded_config_data['search_engines']['search_engine_country_code_brave']
    else:
        country = "US"
    if not search_lang:
        search_lang = "en"
    if not ui_lang:
        ui_lang = "en"
    if not result_count:
        result_count = 10
    # if not date_range:
    #     date_range = "month"
    if not result_filter:
        result_filter = "webpages"
    if search_type == "ai":
        # FIXME - Option for switching between AI/Regular search
        pass


    headers = {"Accept": "application/json", "Accept-Encoding": "gzip", "X-Subscription-Token": brave_api_key}

    # https://api.search.brave.com/app/documentation/web-search/query#WebSearchAPIQueryParameters
    params = {"q": search_term, "textDecorations": True, "textFormat": "HTML", "count": result_count,
              "freshness": date_range, "promote": "webpages", "safeSearch": "Moderate"}

    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    # Response: https://api.search.brave.com/app/documentation/web-search/responses#WebSearchApiResponse
    brave_search_results = response.json()
    return brave_search_results


def test_search_brave():
    search_term = "How can I bake a cherry cake"
    country = "US"
    search_lang = "en"
    ui_lang = "en"
    result_count = 10
    safesearch = "moderate"
    date_range = None
    result_filter = None
    result = search_web_brave(search_term, country, search_lang, ui_lang, result_count, safesearch, date_range,
                          result_filter)
    print(result)
    return result


# FIXME - untested
def parse_brave_results(raw_results: Dict, output_dict: Dict) -> None:
    """
    Parse Brave search results and update the output dictionary

    Args:
        raw_results (Dict): Raw Brave API response
        output_dict (Dict): Dictionary to store processed results
    """
    try:
        # Extract query information
        if "query" in raw_results:
            query_info = raw_results["query"]
            output_dict.update({
                "search_query": query_info.get("original", ""),
                "content_country": query_info.get("country", ""),
                "city": query_info.get("city", ""),
                "state": query_info.get("state", ""),
                "more_results_available": query_info.get("more_results_available", False)
            })

        # Process web results
        if "web" in raw_results and "results" in raw_results["web"]:
            for result in raw_results["web"]["results"]:
                processed_result = {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("description", ""),
                    "metadata": {
                        "date_published": result.get("page_age", None),
                        "author": None,
                        "source": result.get("profile", {}).get("name", None),
                        "language": result.get("language", None),
                        "relevance_score": None,
                        "snippet": result.get("description", None),
                        "family_friendly": result.get("family_friendly", None),
                        "type": result.get("type", None),
                        "subtype": result.get("subtype", None),
                        "thumbnail": result.get("thumbnail", {}).get("src", None)
                    }
                }
                output_dict["results"].append(processed_result)

        # Update total results count
        if "mixed" in raw_results:
            output_dict["total_results_found"] = len(raw_results["mixed"].get("main", []))

        # Set family friendly status
        if "mixed" in raw_results:
            output_dict["family_friendly"] = raw_results.get("family_friendly", True)

    except Exception as e:
        output_dict["processing_error"] = f"Error processing Brave results: {str(e)}"
        raise

def test_parse_brave_results():
    pass


######################### DuckDuckGo Search #########################
#
# https://github.com/deedy5/duckduckgo_search
# Copied request format/structure from https://github.com/deedy5/duckduckgo_search/blob/main/duckduckgo_search/duckduckgo_search.py
def create_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def search_web_duckduckgo(
    keywords: str,
    region: str = "wt-wt",
    timelimit: str | None = None,
    max_results: int | None = None,
) -> list[dict[str, str]]:
    assert keywords, "keywords is mandatory"

    payload = {
        "q": keywords,
        "s": "0",
        "o": "json",
        "api": "d.js",
        "vqd": "",
        "kl": region,
        "bing_market": region,
    }

    def _normalize_url(url: str) -> str:
        """Unquote URL and replace spaces with '+'."""
        return unquote(url).replace(" ", "+") if url else ""

    def _normalize(raw_html: str) -> str:
        """Strip HTML tags from the raw_html string."""
        REGEX_STRIP_TAGS = re.compile("<.*?>")
        return unescape(REGEX_STRIP_TAGS.sub("", raw_html)) if raw_html else ""

    if timelimit:
        payload["df"] = timelimit

    cache = set()
    results: list[dict[str, str]] = []

    for _ in range(5):
        response = requests.post("https://html.duckduckgo.com/html", data=payload)
        resp_content = response.content
        if b"No  results." in resp_content:
            return results

        tree = document_fromstring(resp_content)
        elements = tree.xpath("//div[h2]")
        if not isinstance(elements, list):
            return results

        for e in elements:
            if isinstance(e, _Element):
                hrefxpath = e.xpath("./a/@href")
                href = str(hrefxpath[0]) if hrefxpath and isinstance(hrefxpath, list) else None
                if (
                    href
                    and href not in cache
                    and not href.startswith(
                        ("http://www.google.com/search?q=", "https://duckduckgo.com/y.js?ad_domain")
                    )
                ):
                    cache.add(href)
                    titlexpath = e.xpath("./h2/a/text()")
                    title = str(titlexpath[0]) if titlexpath and isinstance(titlexpath, list) else ""
                    bodyxpath = e.xpath("./a//text()")
                    body = "".join(str(x) for x in bodyxpath) if bodyxpath and isinstance(bodyxpath, list) else ""
                    results.append(
                        {
                            "title": _normalize(title),
                            "href": _normalize_url(href),
                            "body": _normalize(body),
                        }
                    )
                    if max_results and len(results) >= max_results:
                        return results

        npx = tree.xpath('.//div[@class="nav-link"]')
        if not npx or not max_results:
            return results
        next_page = npx[-1] if isinstance(npx, list) else None
        if isinstance(next_page, _Element):
            names = next_page.xpath('.//input[@type="hidden"]/@name')
            values = next_page.xpath('.//input[@type="hidden"]/@value')
            if isinstance(names, list) and isinstance(values, list):
                payload = {str(n): str(v) for n, v in zip(names, values)}

    return results

def test_search_duckduckgo():
    try:
        results = search_web_duckduckgo(
            keywords="How can I bake a cherry cake?",
            region="us-en",
            timelimit="w",
            max_results=10
        )
        print(f"Number of results: {len(results)}")
        for result in results:
            print(f"Title: {result['title']}")
            print(f"URL: {result['href']}")
            print(f"Snippet: {result['body']}")
            print("---")

    except ValueError as e:
        print(f"Invalid input: {str(e)}")
    except requests.RequestException as e:
        print(f"Request error: {str(e)}")


def parse_duckduckgo_results(raw_results: Dict, output_dict: Dict) -> None:
    """
    Parse DuckDuckGo search results and update the output dictionary

    Args:
        raw_results (Dict): Raw DuckDuckGo response
        output_dict (Dict): Dictionary to store processed results
    """
    try:
        # DuckDuckGo results appear to be in a simple list format
        # Each result is separated by "---"
        results = raw_results.get("results", [])

        for result in results:
            # Extract information using the consistent format in results
            title = ""
            url = ""
            snippet = ""

            # Parse the result text
            lines = result.split('\n')
            for line in lines:
                if line.startswith("Title: "):
                    title = line.replace("Title: ", "").strip()
                elif line.startswith("URL: "):
                    url = line.replace("URL: ", "").strip()
                elif line.startswith("Snippet: "):
                    snippet = line.replace("Snippet: ", "").strip()

            processed_result = {
                "title": title,
                "url": url,
                "content": snippet,
                "metadata": {
                    "date_published": None,  # DuckDuckGo doesn't typically provide this
                    "author": None,  # DuckDuckGo doesn't typically provide this
                    "source": extract_domain(url) if url else None,
                    "language": None,  # DuckDuckGo doesn't typically provide this
                    "relevance_score": None,  # DuckDuckGo doesn't typically provide this
                    "snippet": snippet
                }
            }

            output_dict["results"].append(processed_result)

        # Update total results count
        output_dict["total_results_found"] = len(output_dict["results"])

    except Exception as e:
        output_dict["processing_error"] = f"Error processing DuckDuckGo results: {str(e)}"


def extract_domain(url: str) -> str:
    """
    Extract domain name from URL

    Args:
        url (str): Full URL

    Returns:
        str: Domain name
    """
    try:
        from urllib.parse import urlparse
        parsed_uri = urlparse(url)
        domain = parsed_uri.netloc
        return domain.replace('www.', '')
    except:
        return url

def test_parse_duckduckgo_results():
    pass



######################### Google Search #########################
#
# https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list
def search_web_google(
    search_query: str,
    google_search_api_key: Optional[str] = None,
    google_search_engine_id: Optional[str] = None,
    result_count: Optional[int] = None,
    c2coff: Optional[str] = None,
    results_origin_country: Optional[str] = None,
    date_range: Optional[str] = None,
    exactTerms: Optional[str] = None,
    excludeTerms: Optional[str] = None,
    filter: Optional[str] = None,
    geolocation: Optional[str] = None,
    ui_language: Optional[str] = None,
    search_result_language: Optional[str] = None,
    safesearch: Optional[str] = None,
    site_blacklist: Optional[str] = None,
    sort_results_by: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform a Google web search with the given parameters.

    :param search_query: The search query string
    :param google_search_api_key: Google Search API key
    :param google_search_engine_id: Google Search Engine ID
    :param result_count: Number of results to return
    :param c2coff: Enable/disable traditional Chinese search
    :param results_origin_country: Limit results to a specific country
    :param date_range: Limit results to a specific date range
    :param exactTerms: Exact terms that must appear in results
    :param excludeTerms: Terms that must not appear in results
    :param filter: Control duplicate content filter
    :param geolocation: Geolocation of the user
    :param ui_language: Language of the user interface
    :param search_result_language: Language of search results
    :param safesearch: Safe search setting
    :param site_blacklist: Sites to exclude from search
    :param sort_results_by: Sorting criteria for results
    :return: JSON response from Google Search API
    """
    try:
        # Load Search API URL from config file
        search_url = loaded_config_data['search_engines']['google_search_api_url']
        logging.info(f"Using search URL: {search_url}")

        # Initialize params dictionary
        params: Dict[str, Any] = {"q": search_query}

        # Handle c2coff
        if c2coff is None:
            c2coff = loaded_config_data['search_engines']['google_simp_trad_chinese']
        if c2coff is not None:
            params["c2coff"] = c2coff

        # Handle results_origin_country
        if results_origin_country is None:
            limit_country_search = loaded_config_data['search_engines']['limit_google_search_to_country']
            if limit_country_search:
                results_origin_country = loaded_config_data['search_engines']['google_search_country']
        if results_origin_country:
            params["cr"] = results_origin_country

        # Handle google_search_engine_id
        if google_search_engine_id is None:
            google_search_engine_id = loaded_config_data['search_engines']['google_search_engine_id']
        if not google_search_engine_id:
            raise ValueError("Please set a valid Google Search Engine ID in the config file")
        params["cx"] = google_search_engine_id

        # Handle google_search_api_key
        if google_search_api_key is None:
            google_search_api_key = loaded_config_data['search_engines']['google_search_api_key']
        if not google_search_api_key:
            raise ValueError("Please provide a valid Google Search API subscription key")
        params["key"] = google_search_api_key

        # Handle other parameters
        if result_count:
            params["num"] = result_count
        if date_range:
            params["dateRestrict"] = date_range
        if exactTerms:
            params["exactTerms"] = exactTerms
        if excludeTerms:
            params["excludeTerms"] = excludeTerms
        if filter:
            params["filter"] = filter
        if geolocation:
            params["gl"] = geolocation
        if ui_language:
            params["hl"] = ui_language
        if search_result_language:
            params["lr"] = search_result_language
        if safesearch is None:
            safesearch = loaded_config_data['search_engines']['google_safe_search']
        if safesearch:
            params["safe"] = safesearch
        if site_blacklist:
            params["siteSearch"] = site_blacklist
            params["siteSearchFilter"] = "e"  # Exclude these sites
        if sort_results_by:
            params["sort"] = sort_results_by

        logging.info(f"Prepared parameters for Google Search: {params}")

        # Make the API call
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        google_search_results = response.json()

        logging.info(f"Successfully retrieved search results. Items found: {len(google_search_results.get('items', []))}")

        return google_search_results

    except ValueError as ve:
        logging.error(f"Configuration error: {str(ve)}")
        raise

    except RequestException as re:
        logging.error(f"Error during API request: {str(re)}")
        raise

    except Exception as e:
        logging.error(f"Unexpected error occurred: {str(e)}")
        raise


def test_search_google():
    search_query = "How can I bake a cherry cake?"
    google_search_api_key = loaded_config_data['search_engines']['google_search_api_key']
    google_search_engine_id = loaded_config_data['search_engines']['google_search_engine_id']
    result_count = 10
    c2coff = "1"
    results_origin_country = "US"
    date_range = None
    exactTerms = None
    excludeTerms = None
    filter = None
    geolocation = "us"
    ui_language = "en"
    search_result_language = "lang_en"
    safesearch = "off"
    site_blacklist = None
    sort_results_by = None
    result = search_web_google(search_query, google_search_api_key, google_search_engine_id, result_count, c2coff,
                      results_origin_country, date_range, exactTerms, excludeTerms, filter, geolocation,ui_language,
                      search_result_language, safesearch, site_blacklist, sort_results_by)
    print(result)


# FIXME - untested
def parse_google_results(raw_results: Dict, output_dict: Dict) -> None:
    """
    Parse Google Custom Search API results and update the output dictionary

    Args:
        raw_results (Dict): Raw Google API response
        output_dict (Dict): Dictionary to store processed results
    """
    try:
        # Extract search information
        if "searchInformation" in raw_results:
            search_info = raw_results["searchInformation"]
            output_dict["total_results_found"] = int(search_info.get("totalResults", "0"))
            output_dict["search_time"] = search_info.get("searchTime", 0.0)

        # Extract spelling suggestions
        if "spelling" in raw_results:
            output_dict["spell_suggestions"] = raw_results["spelling"].get("correctedQuery")

        # Extract search parameters from queries
        if "queries" in raw_results and "request" in raw_results["queries"]:
            request = raw_results["queries"]["request"][0]
            output_dict.update({
                "search_query": request.get("searchTerms", ""),
                "search_lang": request.get("language", ""),
                "result_count": request.get("count", 0),
                "safesearch": request.get("safe", None),
                "exactTerms": request.get("exactTerms", None),
                "excludeTerms": request.get("excludeTerms", None),
                "filter": request.get("filter", None),
                "geolocation": request.get("gl", None),
                "search_result_language": request.get("hl", None),
                "sort_results_by": request.get("sort", None)
            })

        # Process search results
        if "items" in raw_results:
            for item in raw_results["items"]:
                processed_result = {
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "content": item.get("snippet", ""),
                    "metadata": {
                        "date_published": item.get("pagemap", {}).get("metatags", [{}])[0].get(
                            "article:published_time"),
                        "author": item.get("pagemap", {}).get("metatags", [{}])[0].get("article:author"),
                        "source": item.get("displayLink", None),
                        "language": item.get("language", None),
                        "relevance_score": None,  # Google doesn't provide this directly
                        "snippet": item.get("snippet", None),
                        "file_format": item.get("fileFormat", None),
                        "mime_type": item.get("mime", None),
                        "cache_url": item.get("cacheId", None)
                    }
                }

                # Extract additional metadata if available
                if "pagemap" in item:
                    pagemap = item["pagemap"]
                    if "metatags" in pagemap and pagemap["metatags"]:
                        metatags = pagemap["metatags"][0]
                        processed_result["metadata"].update({
                            "description": metatags.get("og:description",
                                                        metatags.get("description")),
                            "keywords": metatags.get("keywords"),
                            "site_name": metatags.get("og:site_name")
                        })

                output_dict["results"].append(processed_result)

        # Add pagination information
        output_dict["pagination"] = {
            "has_next": "nextPage" in raw_results.get("queries", {}),
            "has_previous": "previousPage" in raw_results.get("queries", {}),
            "current_page": raw_results.get("queries", {}).get("request", [{}])[0].get("startIndex", 1)
        }

    except Exception as e:
        output_dict["processing_error"] = f"Error processing Google results: {str(e)}"

def test_parse_google_results():
    pass



######################### Kagi Search #########################
#
# https://help.kagi.com/kagi/api/search.html
def search_web_kagi(query: str, limit: int = 10) -> Dict:
    search_url = "https://kagi.com/api/v0/search"

    # load key from config file
    kagi_api_key = loaded_config_data['search_engines']['kagi_search_api_key']
    if not kagi_api_key:
        raise ValueError("Please provide a valid Kagi Search API subscription key")

    """
    Queries the Kagi Search API with the given query and limit.
    """
    if kagi_api_key is None:
        raise ValueError("API key is required.")

    headers = {"Authorization": f"Bot {kagi_api_key}"}
    endpoint = f"{search_url}/search"
    params = {"q": query, "limit": limit}

    response = requests.get(endpoint, headers=headers, params=params)
    response.raise_for_status()
    print(response.json())
    return response.json()


def test_search_kagi():
    search_term = "How can I bake a cherry cake"
    result_count = 10
    result = search_web_kagi(search_term, result_count)
    print(result)


def parse_kagi_results(raw_results: Dict, output_dict: Dict) -> None:
    """
    Parse Kagi search results and update the output dictionary

    Args:
        raw_results (Dict): Raw Kagi API response
        output_dict (Dict): Dictionary to store processed results
    """
    try:
        # Extract metadata
        if "meta" in raw_results:
            meta = raw_results["meta"]
            output_dict["search_time"] = meta.get("ms", 0) / 1000.0  # Convert to seconds
            output_dict["api_balance"] = meta.get("api_balance")
            output_dict["search_id"] = meta.get("id")
            output_dict["node"] = meta.get("node")

        # Process search results
        if "data" in raw_results:
            for item in raw_results["data"]:
                # Skip related searches (type 1)
                if item.get("t") == 1:
                    output_dict["related_searches"] = item.get("list", [])
                    continue

                # Process regular search results (type 0)
                if item.get("t") == 0:
                    processed_result = {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "content": item.get("snippet", ""),
                        "metadata": {
                            "date_published": item.get("published"),
                            "author": None,  # Kagi doesn't typically provide this
                            "source": None,  # Could be extracted from URL if needed
                            "language": None,  # Kagi doesn't typically provide this
                            "relevance_score": None,
                            "snippet": item.get("snippet"),
                            "thumbnail": item.get("thumbnail", {}).get("url") if "thumbnail" in item else None
                        }
                    }
                    output_dict["results"].append(processed_result)

            # Update total results count
            output_dict["total_results_found"] = len([
                item for item in raw_results["data"]
                if item.get("t") == 0
            ])

    except Exception as e:
        output_dict["processing_error"] = f"Error processing Kagi results: {str(e)}"


def test_parse_kagi_results():
    pass



######################### SearX Search #########################
#
# https://searx.space
# https://searx.github.io/searx/dev/search_api.html
def search_web_searx(search_query, language='auto', time_range='', safesearch=0, pageno=1, categories='general'):
    # Check if API URL is configured
    searx_url = loaded_config_data['search_engines']['searx_search_api_url']
    if not searx_url:
        return "SearX Search is disabled and no content was found. This functionality is disabled because the user has not set it up yet."

    # Validate and construct URL
    try:
        parsed_url = urlparse(searx_url)
        params = {
            'q': search_query,
            'language': language,
            'time_range': time_range,
            'safesearch': safesearch,
            'pageno': pageno,
            'categories': categories
        }
        search_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}?{urlencode(params)}"
    except Exception as e:
        return f"Search is disabled and no content was found. Invalid URL configuration: {str(e)}"

    # Perform the search request
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0'
        }

        response = requests.get(search_url, headers=headers)
        response.raise_for_status()

        # Check if the response is JSON
        content_type = response.headers.get('Content-Type', '')
        if 'application/json' in content_type:
            search_data = response.json()
        else:
            # If not JSON, assume it's HTML and parse it
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            search_data = parse_html_search_results_generic(soup)

        # Process results
        data = []
        for result in search_data:
            data.append({
                'title': result.get('title'),
                'link': result.get('url'),
                'snippet': result.get('content'),
                'publishedDate': result.get('publishedDate')
            })

        if not data:
            return "No information was found online for the search query."

        return json.dumps(data)

    except requests.exceptions.RequestException as e:
        return f"There was an error searching for content. {str(e)}"


def test_search_searx():
    result = search_web_searx("How can I bake a cherry cake?")
    print(result)
    pass

def parse_searx_results(searx_search_results, web_search_results_dict):
    pass

def test_parse_searx_results():
    pass




######################### Serper.dev Search #########################
#
# https://github.com/YassKhazzan/openperplex_backend_os/blob/main/sources_searcher.py
def search_web_serper():
    pass


def test_search_serper():
    pass

def parse_serper_results(serper_search_results, web_search_results_dict):
    pass




######################### Tavily Search #########################
#
# https://github.com/YassKhazzan/openperplex_backend_os/blob/main/sources_searcher.py
def search_web_tavily(search_query, result_count=10, site_whitelist=None, site_blacklist=None):
    # Check if API URL is configured
    tavily_api_url = "https://api.tavily.com/search"

    tavily_api_key = loaded_config_data['search_engines']['tavily_search_api_key']

    # Prepare the request payload
    payload = {
        "api_key": tavily_api_key,
        "query": search_query,
        "max_results": result_count
    }

    # Add optional parameters if provided
    if site_whitelist:
        payload["include_domains"] = site_whitelist
    if site_blacklist:
        payload["exclude_domains"] = site_blacklist

    # Perform the search request
    try:
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0'
        }

        response = requests.post(tavily_api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return f"There was an error searching for content. {str(e)}"


def test_search_tavily():
    result = search_web_tavily("How can I bake a cherry cake?")
    print(result)


def parse_tavily_results(tavily_search_results, web_search_results_dict):
    pass


def test_parse_tavily_results():
    pass




######################### Yandex Search #########################
#
# https://yandex.cloud/en/docs/search-api/operations/web-search
# https://yandex.cloud/en/docs/search-api/quickstart/
# https://yandex.cloud/en/docs/search-api/concepts/response
# https://github.com/yandex-cloud/cloudapi/blob/master/yandex/cloud/searchapi/v2/search_query.proto
def search_web_yandex():
    pass


def test_search_yandex():
    pass

def parse_yandex_results(yandex_search_results, web_search_results_dict):
    pass


#
# End of WebSearch_APIs.py
#######################################################################################################################
