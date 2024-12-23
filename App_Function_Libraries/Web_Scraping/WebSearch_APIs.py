# WebSearch_APIs.py
# Description: This file contains the functions that are used for performing queries against various Search Engine APIs
#
# Imports
import requests
#
# 3rd-Party Imports
#
# Local Imports
#
#######################################################################################################################
#
# Functions:

# https://cloud.baidu.com/doc/APIGUIDE/s/Xk1myz05f
def search_baidu():
    pass

# https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/quickstarts/rest/python
# https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/reference/query-parameters
# Country/Language code: https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/reference/market-codes#country-codes
def search_bing(search_term, bing_lang="en", bing_country="en", date_range=None, result_count=None, bing_api_key=None):
    if not bing_api_key:
        # load key from config file
        if not bing_api_key:
            raise ValueError("Please provide a valid Bing Search API subscription key")
    search_url = "https://api.bing.microsoft.com/v7.0/search"
    if not result_count:
        # Perform check in config file for default search result count
        answer_count = 10
    else:
        answer_count = result_count

    # date_range = "day", "week", "month", or `YYYY-MM-DD..YYYY-MM-DD`
    # if not date_range:
    #     date_range = "month"

    # Language settings
    if not bing_lang:
        # do config check for default search language
        setlang = bing_lang

    # Country settings
    if not bing_country:
        # do config check for default search country
        setcountry = bing_country

    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    params = {"q": search_term, "textDecorations": True, "textFormat": "HTML", "count": answer_count,
              "freshness": date_range, "promote": "webpages", "safeSearch": "Moderate"}
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    bing_search_results = response.json()
    return bing_search_results
# Llamaindex https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-bing-search/llama_index/tools/bing_search/base.py
#     def _bing_request(self, endpoint: str, query: str, keys: List[str]):
#         response = requests.get(
#             ENDPOINT_BASE_URL + endpoint,
#             headers={"Ocp-Apim-Subscription-Key": self.api_key},
#             params={"q": query, "mkt": self.lang, "count": self.results},
#         )
#         response_json = response.json()
#         return [[result[key] for key in keys] for result in response_json["value"]]


# Display results bing search
#     from IPython.display import HTML
#
#     rows = "\n".join(["""<tr>
#                            <td><a href=\"{0}\">{1}</a></td>
#                            <td>{2}</td>
#                          </tr>""".format(v["url"], v["name"], v["snippet"])
#                       for v in search_results["webPages"]["value"]])
#     HTML("<table>{0}</table>".format(rows))


# https://brave.com/search/api/
# https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-brave-search/README.md
def search_brave(search_term, country, search_lang, ui_lang, result_count, safesearch="moderate", date_range=None,
                 result_filter=None, brave_api_key=None):
    search_url = "https://api.search.brave.com/res/v1/web/search"

    if not brave_api_key:
        # load key from config file
        if not brave_api_key:
            raise ValueError("Please provide a valid Brave Search API subscription key")
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

    headers = {"Accept": "application/json", "Accept-Encoding": "gzip", "X-Subscription-Token": brave_api_key}

    # https://api.search.brave.com/app/documentation/web-search/query#WebSearchAPIQueryParameters
    params = {"q": search_term, "textDecorations": True, "textFormat": "HTML", "count": result_count,
              "freshness": date_range, "promote": "webpages", "safeSearch": "Moderate"}

    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    # Response: https://api.search.brave.com/app/documentation/web-search/responses#WebSearchApiResponse
    brave_search_results = response.json()
    return brave_search_results

# https://github.com/deedy5/duckduckgo_search/blob/main/duckduckgo_search/duckduckgo_search.py
# https://github.com/deedy5/duckduckgo_search/blob/main/duckduckgo_search/duckduckgo_search.py#L204
def search_duckduckgo():
    pass


def search_google():
    pass
# Llamaindex
https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/tools/tool_spec/base.py
class GoogleSearchToolSpec(BaseToolSpec):
    """Google Search tool spec."""

    spec_functions = ["google_search"]

    def __init__(self, key: str, engine: str, num: Optional[int] = None) -> None:
        """Initialize with parameters."""
        self.key = key
        self.engine = engine
        self.num = num

    def google_search(self, query: str):
        """
        Make a query to the Google search engine to receive a list of results.

        Args:
            query (str): The query to be passed to Google search.
            num (int, optional): The number of search results to return. Defaults to None.

        Raises:
            ValueError: If the 'num' is not an integer between 1 and 10.
        """
        url = QUERY_URL_TMPL.format(
            key=self.key, engine=self.engine, query=urllib.parse.quote_plus(query)
        )

        if self.num is not None:
            if not 1 <= self.num <= 10:
                raise ValueError("num should be an integer between 1 and 10, inclusive")
            url += f"&num={self.num}"

        response = requests.get(url)
        return [Document(text=response.text)]


def search_kagi():
    pass


def search_searx():
    pass


def search_yandex():
    pass

#
# End of WebSearch_APIs.py
#######################################################################################################################
