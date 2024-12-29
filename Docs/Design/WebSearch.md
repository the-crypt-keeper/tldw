# WebSearch

## Introduction
This page serves as documentation regarding the web search functionality within tldw and provides context/justification for the decisions made within the module.

Search is performed -> Results obtained, each individual item is first analyzed based on snippet, if relevant, entire page is fetched and analyzed, this is then stored in the results dictionary, and the process is repeated until all results are analyzed/limit is hit.
Once all results are collected, they are then operated on, being used to create whatever final product is desired by the user.

Pipeline:
1. User posts question
   - Gradio/UI/API
2. Question is analyzed
    - Question is analyzed to identify most likely purpose/goal of question, and Sub-questions are generated to support this
    - User has option of seeing/modifying prompt used for Analysis/sub-question creation
3. Search(es) is/are performed - User gated
    - Search is performed using the user's question and sub-questions
4. Results are collected, stored, and analyzed
    - Results are collected, stored in a 'search_results' dict, and analyzed for relevance, based on initial snippet(? or full page?)
    - User has the option of seeing all results, removing irrelevant results, and selecting which results are 'relevant'
    - User also has the option to select which 'relevant' results are used to answer the question
5. Irrelevant results are removed from the dictionary
    - Results determined to be relevant are then stored in a 'relevant_results' dictionary, and the process is repeated until all results are analyzed/limit is hit.
6. Once all results are collected, they are then used to answer the user's question/sub-questions
    - The remaining relevant results are then used to answer the user's question/sub-questions
    - Each result is abstractly summarized, and then combined into a single document and supplied to the LLM for final analysis
7. The final answer/'briefing' is then presented to the user
8. User has the option to save the results to the DB 
9. User has the option to ask follow-up questions / see potential other questions


- **Text Search Workflow**
    1. User inputs a search query
    2. User selects a search engine (Option for default search engine in config file)
    3. The user presses 'Search'
    4. The search query is passed to the selected search engine
    5. The appropriate search engine is used to perform a search via API call
    6. The search results are returned from the search engine's API
    7. Search engine results are then _MODIFIED_ (if necessary/enabled) to fit the user's preferences
       - This could include re-ranking, summarization/analysis, or other modifications
    8. The (modified) search results are displayed to the user
    9. Results are then displayed to the user, 
       - either as titles of pages with dropdown for all info,
       - or as a list of links with a briefing/summary of each link
       - or as a single briefing/summary of all results
    10. User may then select to save this resulting text to the DB as a plaintext entry, with metadata containing the search query, search engine, and any other relevant information
    11. Search results are then saved to the DB as a plaintext entry, with metadata containing the search query, search engine, and any other relevant information
    12. This is then searchable via the Media DB


----------------
### Setting the Stage
- **Purpose:**
    - The goal of this module is to provide a simple, easy-to-use interface for searching the web and retrieving results.
    - All the web searches are simple HTTP requests to an API or to the direct endpoint and then scraping the results.
    - Results are then reviewed for relevancy, if relevant, the full page is fetched and analyzed.
    - The results are then stored in a dictionary, and the process is repeated until all results are analyzed/limit is hit.
    - Once all results are collected, they are then operated on, being used to create whatever final product is desired by the user.
    - The goal is to provide a simple, easy-to-use interface for searching the web and retrieving results.
    - Other modules are responsible for anything else, this module just performs the search, and delivers the results.
- **Main Function:** (Start Here)
    - `def process_question(question: str, search_params: Dict) -> Dict:`
        - `question` - The question to search for
        - `search_params` - A dictionary containing the search parameters


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



https://github.com/scrapinghub/article-extraction-benchmark

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
{
    "search_engine": str,
    "search_query": str,
    "content_country": str,
    "search_lang": str,
    "output_lang": str,
    "result_count": int,
    "date_range": Optional[str],
    "safesearch": Optional[bool],
    "site_blacklist": Optional[List[str]],
    "exactTerms": Optional[List[str]],
    "excludeTerms": Optional[List[str]],
    "filter": Optional[str],
    "geolocation": Optional[str],
    "search_result_language": Optional[str],
    "sort_results_by": Optional[str],
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
        # ... more results ...
    ],
    "total_results_found": int,
    "search_time": float,
    "error": Optional[str]
}
```

----------------
### Search Engines

#### Baidu Search
- [Baidu Search API](https://www.baidu.com/)


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

