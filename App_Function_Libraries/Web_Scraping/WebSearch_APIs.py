# WebSearch_APIs.py
# Description: This file contains the functions that are used for performing queries against various Search Engine APIs
#
# Imports
import json
import os
from random import choice
from time import sleep
from typing import Optional, Dict, List
from urllib.parse import urlparse, urlencode

import requests

from App_Function_Libraries.Utils.Utils import loaded_config_data


#
# 3rd-Party Imports
#
# Local Imports
#
#######################################################################################################################
#
# Functions:
# 1. perform_websearch
# 1. search_baidu
# 2. test_baidu_search
# 3. search_bing
# 4. test_search_bing
# 5. search_brave
# 6. test_search_brave
# 7. search_duckduckgo
# 8. test_duckduckgo_search
# 9. search_google
# 10. test_search_google
# 11. search_kagi
# 12. test_search_kagi
# 13. search_searx
# 14. test_search_searx
# 15. search_yandex
# 16. test_search_yandex
#
#######################################################################################################################
#

def perform_websearch(search_engine, search_query, country, search_lang, output_lang, result_count, date_range,
                      safesearch, site_blacklist, api_key):
    if search_engine.lower() == "baidu":
        return search_web_baidu()
    elif search_engine.lower() == "bing":
        return search_web_bing(search_query, search_lang, country, date_range, result_count, api_key)
    elif search_engine.lower() == "brave":
            return search_web_brave(search_query, country, search_lang, output_lang, result_count, safesearch, api_key,
                                site_blacklist, date_range)
    elif search_engine.lower() == "duckduckgo":
        return search_web_ddg()
    elif search_engine.lower() == "google":
        return search_web_google(search_query, api_key, google_search_engine_id, result_count, c2coff,
                      results_origin_country, date_range, exactTerms, excludeTerms, filter, geolocation, output_lang,
                      search_result_language, safesearch, site_blacklist, sort_results_by)
    elif search_engine.lower() == "kagi":
        return search_web_kagi(search_query, country, search_lang, output_lang, result_count, safesearch, date_range,
                               site_blacklist, api_key)
    elif search_engine.lower() == "serper":
        return search_web_serper()
    elif search_engine.lower() == "tavily":
        return search_web_tavily()
    elif search_engine.lower() == "searx":
        return search_web_searx()
    elif search_engine.lower() == "yandex":
        return search_web_yandex()
    else:
        return f"Error: Invalid Search Engine Name {search_engine}"


# https://cloud.baidu.com/doc/APIGUIDE/s/Xk1myz05f
def search_web_baidu():
    pass


def test_baidu_search(arg1, arg2, arg3):
    result = search_web_baidu(arg1, arg2, arg3)
    return result


# https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/quickstarts/rest/python
# https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/reference/query-parameters
# Country/Language code: https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/reference/market-codes#country-codes
def search_web_bing(search_query, bing_lang, bing_country, result_count=None, bing_api_key=None,
                    date_range=None):
    if not bing_api_key:
        # load key from config file
        bing_api_key = loaded_config_data['search-engines']['bing_search_api_key']
        if not bing_api_key:
            raise ValueError("Please provide a valid Bing Search API subscription key")
    search_url = "https://api.bing.microsoft.com/v7.0/search"
    if not result_count:
        # Perform check in config file for default search result count
        answer_count = loaded_config_data['search-engines']['search_result_max']
    else:
        answer_count = result_count

    # date_range = "day", "week", "month", or `YYYY-MM-DD..YYYY-MM-DD`
    if not date_range:
         date_range = None

    # Language settings
    if not bing_lang:
        # do config check for default search language
        setlang = bing_lang

    # Country settings
    if not bing_country:
        # do config check for default search country
        bing_country = loaded_config_data['search-engines']['bing_country_code']
    else:
        setcountry = bing_country

    headers = {"Ocp-Apim-Subscription-Key": bing_api_key}
    params = {"q": search_query, "mkt": bing_country, "textDecorations": True, "textFormat": "HTML", "count": answer_count,
              "freshness": date_range, "promote": "webpages", "safeSearch": "Moderate"}
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    bing_search_results = response.json()
    return bing_search_results


# https://brave.com/search/api/
# https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-brave-search/README.md
def search_web_brave(search_term, country, search_lang, ui_lang, result_count, safesearch="moderate",
                     brave_api_key=None, result_filter=None, date_range=None, ):
    search_url = "https://api.search.brave.com/res/v1/web/search"
    if not brave_api_key:
        # load key from config file
        brave_api_key = loaded_config_data['search-engines']['brave_search_api_key']
        if not brave_api_key:
            raise ValueError("Please provide a valid Brave Search API subscription key")
    if not country:
        brave_country = loaded_config_data['search-engines']['search_engine_country_code_brave']
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

    headers = {"Accept": "application/json", "Accept-Encoding": "gzip", "X-Subscription-Token": brave_api_key}

    # https://api.search.brave.com/app/documentation/web-search/query#WebSearchAPIQueryParameters
    params = {"q": search_term, "textDecorations": True, "textFormat": "HTML", "count": result_count,
              "freshness": date_range, "promote": "webpages", "safeSearch": "Moderate"}

    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    # Response: https://api.search.brave.com/app/documentation/web-search/responses#WebSearchApiResponse
    brave_search_results = response.json()
    return brave_search_results


def test_search_brave(search_term, country, search_lang, ui_lang, result_count, safesearch="moderate", date_range=None,
                      result_filter=None, brave_api_key=None):
    result = search_web_brave(search_term, country, search_lang, ui_lang, result_count, safesearch, date_range,
                          result_filter, brave_api_key)
    return result


# https://github.com/deedy5/duckduckgo_search/blob/main/duckduckgo_search/duckduckgo_search.py
# FIXME - 1shot gen with sonnet 3.5, untested.
def search_web_ddg(
        keywords: str,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        max_results: Optional[int] = None
) -> List[Dict[str, str]]:
    """
    Perform a DuckDuckGo search and return results.

    Args:
        keywords: Search query
        region: Region code (e.g., "wt-wt", "us-en")
        safesearch: "on", "moderate", or "off"
        max_results: Maximum number of results to return

    Returns:
        List of dictionaries containing search results with 'title', 'href', and 'body' keys
    """

    # User agent list to randomize requests
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
    ]

    # Helper function to get vqd parameter
    def get_vqd(keywords: str) -> str:
        headers = {'User-Agent': choice(user_agents)}
        resp = requests.get(
            "https://duckduckgo.com/",
            params={'q': keywords},
            headers=headers
        )

        # Extract vqd from response
        try:
            vqd = resp.text.split('vqd="')[1].split('"')[0]
            return vqd
        except IndexError:
            raise Exception("Could not extract vqd parameter")

    results = []
    vqd = get_vqd(keywords)

    # Search parameters
    params = {
        'q': keywords,
        'kl': region,
        'l': region,
        'p': '1' if safesearch == "on" else "",
        's': '0',
        'df': '',
        'vqd': vqd,
        'ex': '-1' if safesearch == "moderate" else '-2' if safesearch == "off" else ""
    }

    headers = {
        'User-Agent': choice(user_agents),
        'Referer': 'https://duckduckgo.com/'
    }

    # Keep track of seen URLs to avoid duplicates
    seen_urls = set()

    try:
        while True:
            response = requests.get(
                'https://links.duckduckgo.com/d.js',
                params=params,
                headers=headers
            )

            if response.status_code != 200:
                break

            # Parse JSON response
            try:
                data = json.loads(response.text.replace('DDG.pageLayout.load(', '').rstrip(');'))
            except json.JSONDecodeError:
                break

            # Extract results
            for result in data['results']:
                url = result.get('u')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    results.append({
                        'title': result.get('t', ''),
                        'href': url,
                        'body': result.get('a', '')
                    })

                    if max_results and len(results) >= max_results:
                        return results

            # Check for next page
            if 'next' not in data:
                break

            params['s'] = data['next']
            sleep(0.5)  # Be nice to the server

    except Exception as e:
        print(f"Error during search: {e}")

    return results
    # def example_usage():
    #     """Example usage of the DuckDuckGo search function"""
    #     try:
    #         # Basic search
    #         results = search_web_ddg("Python programming")
    #         print(f"Found {len(results)} results for 'Python programming'")
    #
    #         # Print first 3 results
    #         for i, result in enumerate(results[:3], 1):
    #             print(f"\nResult {i}:")
    #             print(f"Title: {result['title']}")
    #             print(f"URL: {result['href']}")
    #             print(f"Description: {result['body'][:150]}...")
    #
    #         # Search with different parameters
    #         limited_results = duckduckgo_search(
    #             keywords="artificial intelligence news",
    #             region="us-en",
    #             safesearch="on",
    #             max_results=5
    #         )
    #         print(f"\nFound {len(limited_results)} limited results")
    #
    #     except DuckDuckGoSearchException as e:
    #         print(f"Search failed: {e}")

def test_duckduckgo_search():
    pass


# https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list
def search_web_google(search_query, google_search_api_key, google_search_engine_id, result_count, c2coff,
                      results_origin_country, date_range, exactTerms, excludeTerms, filter, geolocation,ui_language,
                      search_result_language, safesearch, site_blacklist, sort_results_by):
    search_url = "https://www.googleapis.com/customsearch/v1?"
    # FIXME - build the query string dynamically
    if not c2coff:
        # check config file for default setting
        enable_traditional_chinese = loaded_config_data['search-engines']['google_simp_trad_chinese']
    else:
        enable_traditional_chinese = "0"

    if not results_origin_country:
        # check config file for default setting
        limit_country_search = loaded_config_data['search-engines']['limit_google_search_to_country']
    if limit_country_search:
        results_origin_country = loaded_config_data['search-engines']['google_search_country']

    if not google_search_engine_id:
        # load key from config file
        google_search_engine_id = loaded_config_data['search-engines']['google_search_engine_id']
        if not google_search_engine_id:
            raise ValueError("Please set a valid Google Search Engine ID in the config file")

    if not google_search_api_key:
        # load key from config file
        google_search_api_key = loaded_config_data['search-engines']['google_search_api_key']
        if not google_search_api_key:
            raise ValueError("Please provide a valid Bing Search API subscription key")

    # Logic for handling date range
    if not date_range:
        date_range = None

    # Logic for handling exact terms - FIXMe
    # Identifies a phrase that all documents in the search results must contain.
    if not exactTerms:
        exactTerms = None

    # Logic for handling exclude terms - FIXME
    # Identifies a word or phrase that should not appear in any documents in the search results.
    if not excludeTerms:
        excludeTerms = None

    # Logic for handling filter
    # Controls turning on or off the duplicate content filter.
    # Default is 1 (On).
    if not filter:
        filter = "1"

    # Logic for handling geolocation
    # Country codes: https://developers.google.com/custom-search/docs/json_api_reference#countryCodes
    # Country location of user
    if not geolocation:
        geolocation = None

    # Logic for handling UI language
    # https://developers.google.com/custom-search/docs/json_api_reference#wsInterfaceLanguages
    # https://developers.google.com/custom-search/docs/json_api_reference#interfaceLanguages
    if not ui_language:
        ui_language = None

    # Logic for handling search result language
    if not search_result_language:
        search_result_language = None

    if not safesearch:
        safe_search = loaded_config_data['search-engines']['google_safe_search']
    else:
        safe_search = safesearch

    if not site_blacklist:
        site_blacklist = None

    if not sort_results_by:
        sort_results_by = None

    params = {"c2coff": enable_traditional_chinese, "key": google_search_api_key, "cx": google_search_engine_id, "q": search_query}
    response = requests.get(search_url, params=params)
    response.raise_for_status()
    google_search_results = response.json()
    return google_search_results


def test_search_google():
    pass


# https://help.kagi.com/kagi/api/search.html
def search_web_kagi(search_term, country, search_lang, ui_lang, result_count, safesearch="moderate", date_range=None,
                    result_filter=None, kagi_api_key=None):
    search_url = "https://api.search.brave.com/res/v1/web/search"

    if not kagi_api_key:
        # load key from config file
        if not kagi_api_key:
            raise ValueError("Please provide a valid Kagi Search API subscription key")
    if not country:
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

    headers = {"Authorization: Bot " + kagi_api_key}

    # https://api.search.brave.com/app/documentation/web-search/query#WebSearchAPIQueryParameters
    params = {"q": search_term, "textDecorations": True, "textFormat": "HTML", "count": result_count,
              "freshness": date_range, "promote": "webpages", "safeSearch": "Moderate"}

    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    # Response: https://api.search.brave.com/app/documentation/web-search/responses#WebSearchApiResponse
    brave_search_results = response.json()
    return brave_search_results
    # curl - v \
    # - H
    # "Authorization: Bot $TOKEN" \
    #         https: // kagi.com / api / v0 / search\?q = steve + jobs
    pass


def test_search_kagi():
    pass


# https://searx.space
# https://searx.github.io/searx/dev/search_api.html
def search_web_searx(search_query):
    # Check if API URL is configured
    searx_url = loaded_config_data['search-engines']['searx_search_api_url']
    if not searx_url:
        return "Search is disabled and no content was found. This functionality is disabled because the user has not set it up yet."

    # Validate and construct URL
    try:
        parsed_url = urlparse(searx_url)
        params = {
            'q': search_query,
            'format': 'json'
        }
        search_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}?{urlencode(params)}"
    except Exception as e:
        return f"Search is disabled and no content was found. Invalid URL configuration: {str(e)}"

    # Perform the search request
    try:
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'anything-llm'
        }

        response = requests.get(search_url, headers=headers)
        response.raise_for_status()
        search_data = response.json()

        # Process results
        data = []
        for result in search_data.get('results', []):
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
    search_web_searx("How can I bake a cherry cake?")
    pass


def search_web_serper():
    pass


def test_search_serper():
    pass


# https://docs.tavily.com/docs/rest-api/api-reference
def search_web_tavily():
    tavily_url = "https://api.tavily.com/search"
    pass


def test_search_tavily():
    pass


# https://yandex.cloud/en/docs/search-api/operations/web-search
# https://yandex.cloud/en/docs/search-api/quickstart/
# https://yandex.cloud/en/docs/search-api/concepts/response
# https://github.com/yandex-cloud/cloudapi/blob/master/yandex/cloud/searchapi/v2/search_query.proto
def search_web_yandex():
    pass


def test_search_yandex():
    pass

#
# End of WebSearch_APIs.py
#######################################################################################################################
