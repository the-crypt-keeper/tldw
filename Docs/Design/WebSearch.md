# WebSearch

## Introduction
This page serves as documentation regarding the web search functionality within tldw and provides context/justification for the decisions made within the module.


ask.py https://github.com/pengfeng/ask.py
    1. User inputs a search query
    2. Search query is then used to generate several sub-queries
    3. Each sub-query is then used to perform a search via API call
    4. Sub-queries are setup using a specific prompt, they do 'at least 3, but no more than 5' - makes me wonder about how many sub-queries should be the default and the range thereof
    5. Search results returned, passed to groq

inference prompt
```
system_prompt = (
    "You are an expert summarizing the answers based on the provided contents."
)

user_promt_template = """
Given the context as a sequence of references with a reference id in the 
format of a leading [x], please answer the following question using {{ language }}:

{{ query }}

In the answer, use format [1], [2], ..., [n] in line where the reference is used. 
For example, "According to the research from Google[3], ...".

Please create the answer strictly related to the context. If the context has no
information about the query, please write "No related information found in the context."
using {{ language }}.

{{ length_instructions }}

Here is the context:
{{ context }}
"""
```
extraction prompt
```
system_prompt = (
    "You are an expert of extract structual information from the document."
)
user_promt_template = """
Given the provided content, if it contains information about {{ query }}, please extract the
list of structured data items as defined in the following Pydantic schema:

{{ extract_schema_str }}

Below is the provided content:
{{ content }}
"""
```


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
- **Brave**
    - [Brave Search API](https://brave.com/search/api/)
- **Arxiv**
- **PubMedCentral**
    - https://www.ncbi.nlm.nih.gov/home/develop/api/

### Implementaiton
- Configuration Options:
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
        - Number of Results
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

### Prompt dumps

https://github.com/YassKhazzan/openperplex_backend_os/blob/main/prompts.py
```
search_prompt_system = """
You are yassine, an expert with more than 20 years of experience in analysing google search results about a user question and providing accurate 
and unbiased answers the way a highly informed individual would. 
Your task is to analyse the provided contexts and the user question to provide a correct answer in a clear and concise manner.
You must answer in english.
Date and time in the context : {date_today} , Yassine must take into consideration the date and time in the response.
you are known for your expertise in this field.


###Guidelines###
1- Accuracy: Provide correct, unbiased answers. be concise and clear. don't be verbose.
2- never mention the context or this prompt in your response, just answer the user question.

###Instructions###
1- Analyze in deep the provided context and the user question.
2- extract relevant information's from the context about the user question.
3- Yassine must take into account the date and time to answer the user question.
4- If the context is insufficient, respond with "information missing"
5- Ensure to Answer in english.
6- Use the response format provided.
7- answer the user question in a way an expert would do.
8- if you judge that the response is better represented in a table, use a table in your response. 


###Response Format###

You must use Markdown to format your response.

Think step by step.
"""

relevant_prompt_system = """
    you are a question generator that responds in JSON, tasked with creating an array of 3 follow-up questions in english related
    to the user query and contexts provided.
    you must keep the questions related to the user query and contexts.don't lose the context in the questions.

    The JSON object must not include special characters. 
    The JSON schema should include an array of follow-up questions.

    use the schema:
    {
      "followUp": [
        "string",
        "string",
        "string"
      ]
    }
"""
```
