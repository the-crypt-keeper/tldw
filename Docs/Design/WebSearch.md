# WebSearch

## Introduction
This page serves as documentation regarding the web search functionality within tldw and provides context/justification for the decisions made within the module.
- **High-Level Workflow**
    1. User inputs a search query.
    2. User selects a search engine (Option for default search engine + default query options in config file).
    3. User may select `Advanced Search` for additional search parameters (Language, Date Range, etc).
    4. The user presses 'Search'.
    5. Search is performed -> Results obtained, 
    6. Each individual item is first analyzed based on snippet, if relevant, entire page is fetched and analyzed, this is then stored in the results dictionary, and the process is repeated until all results are analyzed/limit is hit.
    7. Once all results are collected, they are then operated on, being used to create whatever final product is desired by the user.
    8. The final product is then passed back to the UI for display to the user.


### Current Status
- Bing, Brave, DDG, Google work for simple searches. Advanced search options are not fully working yet.
    - Brave: https://api.search.brave.com/app/documentation/web-search/query#WebSearchAPIQueryParameters
    - Bing: https://docs.microsoft.com/en-us/rest/api/cognitiveservices-bingsearch/bing-web-api-v7-reference
    - Google: https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list
- Baidu, SearX, Serper, Tavily, Yandex are not implemented yet.
- Kagi & SearX are implemented but not working (Kagi because API access and SearX because I'm not sure why)
- Parsing works for Bing, Brave, DDG, Google.
- Currently no use of Structured outputs (to-do...)
- Full Pipeline works.
    1. User enters query + Search options
    2. Query is processed, sub-queries are generated(if specified)
    3. Each query is sent to the search engine API
    4. Search results are returned
    5. Each search result is analyzed, if relevant, the full page is fetched and stored in the results dict
    6. Results are then aggregated and presented to the user
- **To Do:**
    2. Implement the saving options (Web Search DB - allow for saving search results in DB, File, Clipboard, Notes DB - allow for saving search report + citations in a note)
    2. User can also select which results are relevant, and which are not, and remove irrelevant results from the `web_search_results_dict`
    3. Implement the remaining search engines (Baidu, SearX, Serper, Tavily, Yandex)
    4. Implement the advanced search options
    5. Implement the structured outputs
    6. Implement the various output options (Style/Format) / Allow the user to customize the aggregation prompt
    7. Provide the user with follow-up questions + potential other questions, if these are selected, the content is added to the 'ongoing' document


----------------
### Setting the Stage

- **Text Search Workflow**
    1. User inputs a search query.
    2. User selects a search engine + Query options (Option for default search engine + default query options in config file).
    3. The user presses 'Search'.
        - Gradio UI Function makes a call to `process_question` with the search query + search parameters as a dictionary
    4. `process_question()` checks the search params dict to see if sub-query creation is enabled, if so, it creates a list of sub-queries based on the search query with a call to `analyze_question()`.
    5. `analyze_question()` takes the search query and generates a list of sub-queries based on the search query, attempts this 3 times, making a call to the LLM API, and then returns the list of sub-queries if successful.
    6. once back in `process_question()`, all queries are combined into a single query list `#L113`.
    7. `process_question()` then iterates through the query list, making a call to `perform_websearch()` with each query in the list, and the matching search parameters.
    8. `perform_websearch()` makes a call to the selected search engine's API with the query and search parameters, and returns the results.
       - This function is a `sink' for all search engine API calls, and is responsible for handling calling the appropriate search engine API call
    9. `process_web_search_results()` then takes the results from the search engine, and processes them, converting them into a dictionary of results in the `web_search_results_dict` dictionary.
       - FIXME - this is where I lose track of what's happening, need to re-read the code
       - This function returns a filled `web_search_results_dict` dictionary
    10. `process_question()` then takes the `web_search_results_dict` and processes it, checking to make sure it is valid and contains results.
        - FIXME - Verify this is correct
    11. FIXME - Make it optional to display the results to the user, and allow them to select which results are relevant before continuing processing
    12. `process_question()` then iterates through each search result, checking if it is relevant, and if so, adds it to the `relevant_results_dict`
          - FIXME - The results should be added back to the `web_search_results_dict` if they are relevant.
    13. `process_question()` then calls into `aggregate_results()` function with the `web_search_results_dict`
    14. `aggregate_results()` then takes the `web_search_results_dict` and processes it, combining all the results into a single document
        - FIXME - This is not implemented yet and also want various options available for this.
    15. `process_question()` then returns the `web_search_results_dict` to the calling function.
    15. The calling function then takes the `web_search_results_dict` and processes it, extracting the final results/aggregated report and presenting it to the user
    16. The user then has the option to save the results to the DB, or ask follow-up questions, etc.
    17. The user can also select which results are relevant, and which are not, and remove irrelevant results from the `web_search_results_dict`


- **Function Execution Steps:**
    - `def perform_websearch(search_engine, search_query, content_country, search_lang, output_lang, result_count, date_range=None, safesearch=None, site_blacklist=None, exactTerms=None, excludeTerms=None, filter=None, geolocation=None, search_result_language=None, sort_results_by=None)`
        - `search_engine` - The search engine to use for the search
        - `search_query` - The query to search for
        - `content_country` - The country of the content to search for
        - `search_lang` - The language to use for the search
        - `output_lang` - The language to use for the output
        - `result_count` - The number of results to return
        - `date_range` - The date range to search within
        - `safesearch` - Whether to enable safe search
        - `site_blacklist` - A list of sites to exclude from the search results
        - `exactTerms` - Terms that must be in the search results
        - `excludeTerms` - Terms that must not be in the search results
        - `filter` - A filter to apply to the search results
        - `geolocation` - The geolocation to use for the search
        - `search_result_language` - The language to use for the search results
        - `sort_results_by` - How to sort the search results
    - **Returns:** A list of search results as a dictionary. - FIXME: Define the structure of the dictionary
              - Each result should contain the title, URL, content, and metadata of the search result.



----------------
### Web Search API
- TBD
```def perform_websearch(search_engine, 
                            search_query, 
                            content_country, 
                            search_lang, 
                            output_lang, 
                            result_count, 
                            date_range=None,
                            safesearch=None, 
                            site_blacklist=None, 
                            exactTerms=None, 
                            excludeTerms=None, 
                            filter=None, 
                            geolocation=None, 
                            search_result_language=None, 
                            sort_results_by=None
```


ask.py https://github.com/pengfeng/ask.py
    1. User inputs a search query
    2. Search query is then used to generate several sub-queries
    3. Each sub-query is then used to perform a search via API call
    4. Sub-queries are setup using a specific prompt, they do 'at least 3, but no more than 5' - makes me wonder about how many sub-queries should be the default and the range thereof
    5. Search results returned, passed to groq


OpenPerplex
    1. Search Query
    2. Perform search via API
    3. Collect results, grab story titles + snippets
    4. Extract content from each URL
    5. Concat if exceeds X length, re-rank, then combine into a single document
    6. Present final doc to user



llm-websearch
    https://github.com/Jay4242/llm-websearch
    1. Performs query -> Fetches results 
    2. Iterates over each page description and decides whether to 'investigate' the page's content
    3. If so, it fetches the page content and creates a summary of it
    4. Repeat, until all URLs are processed.
    5. Create summary of all summaries/page contents and present final doc to user



mindsearch
    Uses Knowledge Graph for results
    Interesting, seems to combat repetition in search results
        top_p=0.8,
        top_k=1,
        temperature=0,
        max_new_tokens=8192,
        repetition_penalty=1.02,



Perplexity
    1. Search Query
    2. Perform search via API
    3. Collect results, grab story titles + snippets
    4. Extract content from each URL
    5. Concat if exceeds X length, re-rank, then combine into a single document
    6. Present final doc to user
    7. Display top 3 sources at top, with option to show/hide remaining sources - can click on sources to go that URL
    8. Option for Follow up questions + Potential other questions, if these are selected, the content is added to the 'ongoing' document
    9. Ongoing document is saved to the DB, and can be accessed later
    - Option for ranking/share/rewrite/copy to clipboard
    - Pro allows for seemingly 'planning'/sub-question breakdown of the user's original query, maybe just asking 'what would be questions likely asked?'?
    - Performs a planning step and then sub-query breakdown searches into 3(?) sub-queries
    - Non-pro simply performs the search, combines the results, and presents them to the user while showing the sources
    - Spaces: Seems to be a way of sharing collections of research document chains, with the ability to add comments, etc.



### Implementaiton
- Configuration Options:
    - Language
        - Language used for writing query
        - language used for performing query (query submitted to search engine)
        - language used for summary/analysis of search results
    - Query count
    - Whether to create Sub-Queries
        - Sub-Query count
    - Search Engine Selection
        - Search Engine API Key
    - Search Engine Customization Options
        - Safe Search
        - Language
        - Date Range
    - Search Result Options
        - Total number of queries to perform
        - Total number of results/pages to review per query
        - Result Sorting - Auto rerank according to ?
        - Result Filtering - Filter according to a blacklist of URLs? Maybe also content?
    - Search Result Display Options
        - Display Format (List, Briefing, Full)
        - Display Metadata (URL, Date, etc)
    - Search Result Default Saving Options
        - Save to DB
        - Save to File
        - Save to Clipboard
        - Save to Notes DB - Create a new note with the search results + metadata & query
    - Output Options
        - Default Output Format (Markdown, HTML, PDF) - ehhhh
        - Default Output Location (Display/ephemeral, DB)
        - Output style (Briefing, Full, List)
        - Output Metadata (URL, Date, etc)
        - Word count limit for output (per search result & total)
        - Output dialect (US, UK, a NY Radio DJ, etc)


Results dictionary:
```
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
```

----------------
### Search Engines

#### Baidu Search
- [Baidu Search API](https://www.baidu.com/)
- Baidu doens't have an official english API, so we'll have to scrape the results or use Serper


#### Bing Search
- [Bing Search API](https://www.microsoft.com/en-us/bing/apis/bing-web-search-api)
- Getting Started with Bing Search API
    - Sign up for a Bing Search API key via Azure (1000 Free Searches a month) - https://www.microsoft.com/en-us/bing/apis/pricing
    - Use the Bing Search API to perform searches - Add the generated subscription key to your config.txt file.
          - If for some reason you're doing modifications to the code(Fuck MS), be aware: https://github.com/Azure-Samples/cognitive-services-REST-api-samples/issues/139
    - Perform searches using Bing!


#### Brave Search
Two APIs, 1 for 'AI' the other for 'regular' search
    - [Brave Search API](https://brave.com/search/api/)


#### DuckDuckGo Search
Uses query to direct DDG search, then scrape the results.
Structure/approach taken from https://github.com/deedy5/duckduckgo_search


#### Google Search
- [Google Search API](https://developers.google.com/custom-search/v1/overview)
- Have to create a custom search engine first, get the ID and then the API key
- Setup:
  - Setup a `Programmable Search Engine`
  - Get the `API Key`
  - 100 Search queries per day for free
- Documentation for making requests: https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list


#### Kagi Search
- [Kagi Search API](https://help.kagi.com/kagi/api/search.html)
- Really straightforward, just a simple search API


#### SearX Search
- [Searx Search Documentation](https://searx.github.io/searx/)
- `SearXNG is a free internet metasearch engine which aggregates results from more than 70 search services. Users are neither tracked nor profiled.`
- Can host your own instance or use someone else's.


#### Serper Search


#### Tavily Search


#### Yandex Search
- https://yandex.cloud/en/docs/search-api/quickstart/
- [Yandex Search API](https://yandex.com/dev/search/)

#### Arxiv Search


#### PubMedCentral Search
- https://www.ncbi.nlm.nih.gov/home/develop/api/
