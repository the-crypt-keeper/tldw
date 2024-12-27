# WebSearch_APIs.py
# Description: This file contains the functions that are used for performing queries against various Search Engine APIs
#
# Imports
import json
import logging
import re
from html import unescape
from typing import Optional, Dict, Any
from urllib.parse import urlparse, urlencode, unquote
#
# 3rd-Party Imports
import requests
from lxml.etree import _Element
from lxml.html import document_fromstring
from requests import RequestException
from requests.adapters import HTTPAdapter
from urllib3 import Retry

#
# Local Imports
from App_Function_Libraries.Utils.Utils import loaded_config_data
#
#######################################################################################################################
#
# Functions:
# 1. perform_websearch
# 2. search_web_baidu
#
#######################################################################################################################
#



def perform_websearch(search_engine, search_query, content_country, search_lang, output_lang, result_count, date_range=None,
                      safesearch=None, site_blacklist=None, exactTerms=None, excludeTerms=None, filter=None, geolocation=None, search_result_language=None, sort_results_by=None):
    if search_engine.lower() == "baidu":
        return search_web_baidu()
    elif search_engine.lower() == "bing":
        return search_web_bing(search_query, search_lang, content_country, date_range, result_count)
    elif search_engine.lower() == "brave":
            return search_web_brave(search_query, content_country, search_lang, output_lang, result_count, safesearch,
                                    site_blacklist, date_range)
    elif search_engine.lower() == "duckduckgo":
        return search_web_duckduckgo(search_query, content_country, date_range, result_count)
    elif search_engine.lower() == "google":
        return search_web_google(search_query, result_count, content_country, date_range, exactTerms,
                                 excludeTerms, filter, geolocation, output_lang,
                      search_result_language, safesearch, site_blacklist, sort_results_by)
    elif search_engine.lower() == "kagi":
        return search_web_kagi(search_query, content_country, search_lang, output_lang, result_count, safesearch, date_range,
                               site_blacklist)
    elif search_engine.lower() == "serper":
        return search_web_serper()
    elif search_engine.lower() == "tavily":
        return search_web_tavily()
    elif search_engine.lower() == "searx":
        return search_web_searx(search_query, language='auto', time_range='', safesearch=0, pageno=1, categories='general')
    elif search_engine.lower() == "yandex":
        return search_web_yandex()
    else:
        return f"Error: Invalid Search Engine Name {search_engine}"


######################### Search Results Parsing #########################
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
def search_web_baidu():
    pass


def test_baidu_search(arg1, arg2, arg3):
    result = search_web_baidu(arg1, arg2, arg3)
    return result


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

        print("Headers:  ")
        print(response.headers)

        print("JSON Response: ")
        print(response.json())
        bing_search_results = response.json()
        return bing_search_results
    except Exception as ex:
        raise ex


def test_search_web_bing():
    search_query = "How can I bake a cherry cake"
    bing_lang = "en"
    bing_country =  "US"
    result_count = 10
    bing_api_key = None
    date_range = None
    result = search_web_bing(search_query, bing_lang, bing_country, result_count, bing_api_key, date_range)
    print(result)


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


######################### Kagi Search #########################
#
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
    kagi_search_results = response.json()
    return kagi_search_results


def test_search_kagi():
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
            'User-Agent': 'anything-llm'
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


######################### Serper.dev Search #########################
#
# https://github.com/YassKhazzan/openperplex_backend_os/blob/main/sources_searcher.py
def search_web_serper():
    pass


def test_search_serper():
    pass


######################### Tavily Search #########################
#
# https://github.com/YassKhazzan/openperplex_backend_os/blob/main/sources_searcher.py
def search_web_tavily():
    pass


def test_search_tavily():
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

#
# End of WebSearch_APIs.py
#######################################################################################################################
