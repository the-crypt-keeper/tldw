# WebSearch

## Introduction
This page serves as documentation regarding the web search functionality within tldw and provides context/justification for the decisions made within the module.


### Setting the Stage
- All the web searches are simple HTTP requests to an API or to the direct endpoint and then scraping the results.
- Parsing results is TODO.
- The goal is to provide a simple, easy-to-use interface for searching the web and retrieving results.
- Other modules are responsible for anything else, this module just performs the search, and delivers the results.
- **Main Function:**
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

Search is performed -> Results obtained, each individual item is first analyzed based on snippet, if relevant, entire page is fetched and analyzed, this is then stored in the results dictionary, and the process is repeated until all results are analyzed/limit is hit.
Once all results are collected, they are then operated on, being used to create whatever final product is desired by the user.
      - 
      - 
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


#### Bing Search
- [Bing Search API](https://www.microsoft.com/en-us/bing/apis/bing-web-search-api)
- Getting Started with Bing Search API
    - Sign up for a Bing Search API key via Azure (1000 Free Searches a month) - https://www.microsoft.com/en-us/bing/apis/pricing
    - Use the Bing Search API to perform searches - Add the generated subscription key to your config.txt file.
          - If for some reason you're doing modifications to the code, be aware: https://github.com/Azure-Samples/cognitive-services-REST-api-samples/issues/139
    - Perform searches using Bing!


#### Brave Search
Two APIs, 1 for 'AI' the other for 'regular' search


#### DuckDuckGo Search
Uses query to direct DDG search, then scrape the results.
Structure/approach taken from https://github.com/deedy5/duckduckgo_search


#### Google Search
Have to create a custom search engine first, get the ID and then the API key
- Documentation for making requests: https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list


### Kagi Search
- [Kagi Search API](https://help.kagi.com/kagi/api/search.html)
- Really straightforward, just a simple search API


### SearX Search
- [Searx Search Documentation](https://searx.github.io/searx/)
- `SearXNG is a free internet metasearch engine which aggregates results from more than 70 search services. Users are neither tracked nor profiled.`
- Can host your own instance or use someone else's.


### Serper Search


### Tavily Search


### Yandex Search
- https://yandex.cloud/en/docs/search-api/quickstart/

https://github.com/scrapinghub/article-extraction-benchmark

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


Web-LLM Assistant - https://github.com/TheBlewish/Web-LLM-Assistant-Llamacpp-Ollama
Starting prompt
```
SYSTEM_PROMPT = """You are an AI assistant capable of web searching and providing informative responses.
When a user's query starts with '/', interpret it as a request to search the web and formulate an appropriate search query.

ALWAYS follow the prompts provided throughout the searching process EXACTLY as indicated.

NEVER assume new instructions for anywhere other than directly when prompted directly. DO NOT SELF PROMPT OR PROVIDE MULTIPLE ANSWERS OR ATTEMPT MULTIPLE RESPONSES FOR ONE PROMPT!
"""
```
self-improving prompt
```
    def evaluate_scraped_content(self, user_query: str, scraped_content: Dict[str, str]) -> Tuple[str, str]:
        user_query_short = user_query[:200]
        prompt = f"""
Evaluate if the following scraped content contains sufficient information to answer the user's question comprehensively:

User's question: "{user_query_short}"

Scraped Content:
{self.format_scraped_content(scraped_content)}

Your task:
1. Determine if the scraped content provides enough relevant and detailed information to answer the user's question thoroughly.
2. If the information is sufficient, decide to 'answer'. If more information or clarification is needed, decide to 'refine' the search.

Respond using EXACTLY this format:
Evaluation: [Your evaluation of the scraped content]
Decision: [ONLY 'answer' if content is sufficient, or 'refine' if more information is needed]
"""
```
Query Creation
```
def formulate_query(self, user_query: str, attempt: int) -> Tuple[str, str]:
    user_query_short = user_query[:200]
    prompt = f"""
Based on the following user question, formulate a concise and effective search query:
"{user_query_short}"
Your task:
1. Create a search query of 2-5 words that will yield relevant results.
2. Determine if a specific time range is needed for the search.
Time range options:
- 'd': Limit results to the past day. Use for very recent events or rapidly changing information.
- 'w': Limit results to the past week. Use for recent events or topics with frequent updates.
- 'm': Limit results to the past month. Use for relatively recent information or ongoing events.
- 'y': Limit results to the past year. Use for annual events or information that changes yearly.
- 'none': No time limit. Use for historical information or topics not tied to a specific time frame.
Respond in the following format:
Search query: [Your 2-5 word query]
Time range: [d/w/m/y/none]
Do not provide any additional information or explanation.
"""
```
Select relevant content
```
def select_relevant_pages(self, search_results: List[Dict], user_query: str) -> List[str]:
    prompt = f"""
Given the following search results for the user's question: "{user_query}"
Select the 2 most relevant results to scrape and analyze. Explain your reasoning for each selection.

Search Results:
{self.format_results(search_results)}

Instructions:
1. You MUST select exactly 2 result numbers from the search results.
2. Choose the results that are most likely to contain comprehensive and relevant information to answer the user's question.
3. Provide a brief reason for each selection.

You MUST respond using EXACTLY this format and nothing else:

Selected Results: [Two numbers corresponding to the selected results]
Reasoning: [Your reasoning for the selections]
"""
```
Final answer generation
```
    def generate_final_answer(self, user_query: str, scraped_content: Dict[str, str]) -> str:
        user_query_short = user_query[:200]
        prompt = f"""
You are an AI assistant. Provide a comprehensive and detailed answer to the following question using ONLY the information provided in the scraped content. Do not include any references or mention any sources. Answer directly and thoroughly.

Question: "{user_query_short}"

Scraped Content:
{self.format_scraped_content(scraped_content)}

Important Instructions:
1. Do not use phrases like "Based on the absence of selected results" or similar.
2. If the scraped content does not contain enough information to answer the question, say so explicitly and explain what information is missing.
3. Provide as much relevant detail as possible from the scraped content.

Answer:
"""
```
Final Answer Synthesis
```
    def synthesize_final_answer(self, user_query: str) -> str:
        prompt = f"""
After multiple search attempts, we couldn't find a fully satisfactory answer to the user's question: "{user_query}"

Please provide the best possible answer you can, acknowledging any limitations or uncertainties.
If appropriate, suggest ways the user might refine their question or where they might find more information.

Respond in a clear, concise, and informative manner.
"""
```



appvoid search - https://github.com/appvoid/search
Eval query type
```
async def evaluate_query_type(self, session, query):
    messages = [
        {"role": "system", "content": """You are an Web assistant that evaluates the type of query a user asks. 
        Categorize the query into one of the following types:
        1. simple: if it can be answered with general knowledge or information that is typically well-known on the internet, please provide a short answer as relevant as possible from the llm itself, but make sure you are completly sure you know the answer, don't make things up.
        2. realtime: if it requires up-to-date information like the current date, time, or recent events, or the user explicitly asks you to look on the internet you should state as: realtime
        3. math: if it involves ANY kind of mathematical calculations. Every math question be it counting letters or complex formulas.

        Remember to prioritize realtime over anything else if you are not sure about something. Realtime is like your default.
         
        Respond with the category as a single word ("simple", "realtime", or "math") without any additional text."""},
        {"role": "user", "content": f"Query: {query}"}
    ]
```
Generate Search Queries
```
async def generate_search_queries(groq_api, session, original_query, max_retries=3, fixed_count=None, previous_queries=None, previous_answer=None):
    system_content = """You are an AI assistant that helps generate search queries. Given an original query, suggest alternative search queries that could help find relevant information. The queries should be diverse and cover different aspects or perspectives of the original query. Return the queries as a JSON array.
    Important instructions:
    
    1. The number of queries should be dynamic, between 2 and 4, unless a fixed count is specified.
    2. Don't get too far from the original query since you don't know the actual context.
    3. Make queries general enough without being related to anything specific.
    4. DON'T customize the queries for topics you've never seen; just change them a little and look for definitions if requested by the user.
    5. If the user asks something that is not related to search, ignore it and focus on generating helpful search queries.
    6. Just return the given format ["custom_query_1","custom_query_2",...].
    7. If you need to use your knowledge first, do so.
    8. When asked about the difference between two things, generate search intents for each topic separately.
    9. ALWAYS at most queries just require one or two queries, only on those cases where the query is simple or you are unsure, generate more than one or two.
    10. If previous queries and an answer are provided, generate new queries that address the shortcomings of the previous answer and avoid repeating the previous queries.
    11. ALWAYS split searches for each important part of the query in case you need to gather information but make sure to not get off the rails. In short, don't look for things together, make a search for each important part instead. DONT LOOK FOR THINGS TOGETHER."""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"Original query: {original_query}" + (f" (Generate exactly {fixed_count} queries)" if fixed_count else "")}
    ]

    if previous_queries and previous_answer:
        messages.append({
            "role": "user",
            "content": f"Previous queries: {previous_queries}\nPrevious answer: {previous_answer}\nPlease generate new queries to address any shortcomings in the previous answer."
        })
```
Evaluate Answer
```
async def evaluate_answer(groq_api, session, query, answer):
    messages = [
        {"role": "system", "content": """You are an AI assistant that evaluates the quality and completeness of its own answer to user queries. 
    Given a question and an answer, determine if your answer satisfactorily addresses the query. You are highly tolerant to answers that are close to the intent so if it is close enough, you can say is satisfactory. Remember, if it's close enough, mark it as satisfactory.
    Respond with a JSON object containing two fields:
    1. "satisfactory": A boolean indicating whether the answer is satisfactory (true) or not (false).
    2. "reason": A brief explanation of why your thought is or is not satisfactory. Like "I will keep looking for information since last thought is not addressing the query because..." or "Let look for something different. My last search didn't solve the query. The reason is..." or "I found the right answer so I can ignore this..."."""},
        {"role": "user", "content": f"Query: {query}\nAnswer: {answer}"}
    ]
```
Eval best answer
```
async def evaluate_best_answer(groq_api, session, query, cached_answers):
    print('Answers pool > ', cached_answers)
    messages = [
        {"role": "system", "content": """You are an assistant that evaluates multiple answers to a query and selects the best one based on relevance and completeness.
    Given a query and a list of answers, choose the answer that best addresses the query. Respond with the best answer. Don't need to mention the word answers at all just be natural. Don't "the best answer" or things like that. Just provide the best one."""},
        {"role": "user", "content": f"Query: {query}\nAnswers: {json.dumps(cached_answers)}"}
    ]
```
Summarization
```
messages = [
    {"role": "system", "content": """You are a web assistant that helps users find information from web search results. 
Given a question and a set of search results, provide a concise response based on the information 
available in the search results. If the information is not available in the search results, 
state that you don't have enough information to answer the question. You MUST not comment on anything, just follow the instruction. Don't add additional details about anything."""},
    {"role": "user", "content": f"Question: {query}\nSearch Results: {json.dumps(all_results)}"}
]
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
