# WebSearch

## Introduction
This page serves as documentation regarding the web search functionality within tldw and provides context/justification for the decisions made within the module.


### Search Engines
- **Google Search**
    - [Google Search API](https://developers.google.com/custom-search/v1/overview)
    - Setup:
      - Setup a `Programmable Search Engine`
      - Get the `API Key`
      - 100 Search queries per day for free
- **Bing Search**
    - [Bing Search API](https://www.microsoft.com/en-us/bing/apis/bing-web-search-api)
    - 
- **Yandex Search**
    - [Yandex Search API](https://yandex.com/dev/search/)
    - 
- **Baidu Search**
    - [Baidu Search API](https://www.baidu.com/)
    - 
- **Searx Search**
    - [Searx Search API](https://searx.github.io/searx/)
    - 
- **Kagi**
    - [Kagi Search API](https://help.kagi.com/kagi/api/search.html)



### Implementaiton
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


### Link Dump:
https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/document_loaders/brave_search.py
    Could instantiate a browser, perform a search with X engine, and then parse the results.

