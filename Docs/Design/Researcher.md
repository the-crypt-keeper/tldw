# Automated Researcher

## Introduction
- This page is to document efforts towards creating a 'research' agent workflow for use within the project. The goal is to create a system that can automatically generate research reports, summaries, and other research-related tasks.

### Researcher Goals
1. f
2. f
3. f
4. f
5. f
6. f
7. f
8. 


### Ideas
Gated, checkpoints with 'retry, skip, continue' options
s
Follow gptresearchers method at first, planner LLM -> query LLM -> analyzer LLM -> summarizer LLM


### Researcher Workflow


### Researcher Components
1. **Query/Search Engine**
    - f
    - f
2. **Planner**
    - f
    - f
3. **Analyzer**
    - f
    - f
4. **(Optional: Summarizer)**
    - f
    - f
5. **Report Generator**
    - f
    - f
6. **Knowledge Base Management**
    - f
    - f


### Researcher Config Definitions
- `default_search_engine`: The default search engine to use for queries

- Researcher config section
```
[researcher]
# Researcher settings
default_search_engine = google
# Options are: google, bing, yandex, baidu, searx, kagi, serper, tavily
default_search_type = web
# Options are: web, local, both
default_search_language = en
# Options are: FIXME
default_search_report_language = en
# Options are: FIXME
default_search_sort = relevance
# Options are: relevance, date
default_search_safe_search = moderate
# Options are: off, moderate, strict
default_search_planner = openai-o1-full
# Options are: FIXME
default_search_planner_max_tokens = 8192
default_search_analyzer = openai-o1-full
# Options are: FIXME
default_search_analyzer_max_tokens = 8192
default_search_summarization = openai-o1-full
# Options are: FIXME
default_search_summarization_max_tokens = 8192
search_max_results = 100
search_report_format = markdown
# Options are: markdown, html, pdf
search_max_iterations = 5
search_max_subtopics = 4
search_custom_user_agent = "CUSTOM_USER_AGENT_HERE"
search_blacklist_URLs = "URL1,URL2,URL3"
```




### Link Dump:
Articles
   https://docs.gptr.dev/blog/gptr-hybrid
   https://docs.gptr.dev/docs/gpt-researcher/context/local-docs
   https://docs.gptr.dev/docs/gpt-researcher/context/tailored-research#
   https://docs.gptr.dev/docs/gpt-researcher/gptr/pip-package


https://github.com/assafelovic/gpt-researcher
https://arxiv.org/abs/2411.15114
https://journaliststudio.google.com/pinpoint/about/
https://github.com/assafelovic/gpt-researcher/tree/master/gpt_researcher
https://blog.google/products/gemini/google-gemini-deep-research/
https://github.com/neuml/annotateai
https://docs.gptr.dev/docs/gpt-researcher/multi_agents/langgraph
https://pub.towardsai.net/learn-anything-with-ai-and-the-feynman-technique-00a33f6a02bc
https://help.openalex.org/hc/en-us/articles/24396686889751-About-us
https://www.ginkgonotes.com/
https://github.com/assafelovic/gpt-researcher/tree/master/multi_agents
https://www.reddit.com/r/Anki/comments/17u01ge/spaced_repetition_algorithm_a_threeday_journey/
https://github.com/open-spaced-repetition/fsrs4anki/wiki/Spaced-Repetition-Algorithm:-A-Three%E2%80%90Day-Journey-from-Novice-to-Expert#day-3-the-latest-progress
https://www.scrapingdog.com/blog/scrape-google-news/
https://github.com/mistralai/cookbook/blob/main/third_party/LlamaIndex/llamaindex_arxiv_agentic_rag.ipynb
https://github.com/ai-christianson/RA.Aid
https://github.com/cbuccella/perplexity_research_prompt/blob/main/general_research_prompt.md
https://github.com/0xeb/TheBigPromptLibrary/blob/main/SystemPrompts/Perplexity.ai/20241024-Perplexity-Desktop-App.md
https://github.com/rashadphz/farfalle
https://github.com/cbuccella/perplexity_research_prompt/blob/main/general_research_prompt.md
https://www.emergentmind.com/
https://github.com/neuml/paperai
https://github.com/neuml/paperetl
https://github.com/ai-christianson/RA.Aid
https://github.com/Future-House/paper-qa
https://openreview.net/
https://www.researchrabbit.ai/
https://github.com/faraz18001/Sales-Llama
https://github.com/memgraph/memgraph
https://github.com/rashadphz/farfalle/tree/main/src/backend
https://github.com/TheBlewish/Automated-AI-Web-Researcher-Ollama/blob/main/Self_Improving_Search.py
https://storm.genie.stanford.edu/
https://github.com/TheBlewish/Automated-AI-Web-Researcher-Ollama







### Researcher Prompts




https://github.com/SakanaAI/AI-Scientist


one2manny
 — 
Today at 12:43 AM
A great way to make studying more efficient and convenient is to take a digital PDF textbook, split it into separate files for each chapter, and organize them individually. I then create a dedicated notebook for each chapter, treating it as a focused single source. From there, I convert each chapter into an audio format, like a podcast. This approach makes it easy to study while commuting, relaxing in bed with your eyes closed, or at any time when reading isn’t practical.

I also recommend creating a study guide for each chapter, fully breaking down key concepts and definitions. For more complex topics, the “explain like I’m 5” method works wonders—it simplifies challenging ideas into digestible explanations.

To take this further, incorporate a Personal Knowledge Management (PKM) system into your routine. Apps like Obsidian are perfect for this, with their flexible folder structures and Markdown formatting. I optimize my AI outputs for Markdown so I can copy, paste, and organize them into clean, structured notes. This ensures your materials are not only well-organized but also easy to access and build on later. A solid PKM system is invaluable for managing knowledge and staying on top of your studies!


----------------
### Search Prompts

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