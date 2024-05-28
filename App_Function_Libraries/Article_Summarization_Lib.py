# Article_Summarization_Lib.py
#########################################
# Article Summarization Library
# This library is used to handle summarization of articles.

#
####

####################
# Function List
#
# 1.
#
####################



# Import necessary libraries
import os
import logging
import bs4
import huggingface_hub
import tokenizers
import torchvision
import transformers
import summarize


#######################################################################################################################
# Function Definitions
#

def ingest_article_to_db(url, title, author, content, keywords, summary, ingestion_date, custom_prompt):
    try:
        # Check if content is not empty or whitespace
        if not content.strip():
            raise ValueError("Content is empty.")

        db = Database()
        create_tables()
        keyword_list = keywords.split(',') if keywords else ["default"]
        keyword_str = ', '.join(keyword_list)

        # Set default values for missing fields
        url = url or 'Unknown'
        title = title or 'Unknown'
        author = author or 'Unknown'
        keywords = keywords or 'default'
        summary = summary or 'No summary available'
        ingestion_date = ingestion_date or datetime.now().strftime('%Y-%m-%d')

        # Log the values of all fields before calling add_media_with_keywords
        logging.debug(f"URL: {url}")
        logging.debug(f"Title: {title}")
        logging.debug(f"Author: {author}")
        logging.debug(f"Content: {content[:50]}... (length: {len(content)})")  # Log first 50 characters of content
        logging.debug(f"Keywords: {keywords}")
        logging.debug(f"Summary: {summary}")
        logging.debug(f"Ingestion Date: {ingestion_date}")
        logging.debug(f"Custom Prompt: {custom_prompt}")

        # Check if any required field is empty and log the specific missing field
        if not url:
            logging.error("URL is missing.")
            raise ValueError("URL is missing.")
        if not title:
            logging.error("Title is missing.")
            raise ValueError("Title is missing.")
        if not content:
            logging.error("Content is missing.")
            raise ValueError("Content is missing.")
        if not keywords:
            logging.error("Keywords are missing.")
            raise ValueError("Keywords are missing.")
        if not summary:
            logging.error("Summary is missing.")
            raise ValueError("Summary is missing.")
        if not ingestion_date:
            logging.error("Ingestion date is missing.")
            raise ValueError("Ingestion date is missing.")
        if not custom_prompt:
            logging.error("Custom prompt is missing.")
            raise ValueError("Custom prompt is missing.")

        # Add media with keywords to the database
        result = add_media_with_keywords(
            url=url,
            title=title,
            media_type='article',
            content=content,
            keywords=keyword_str or "article_default",
            prompt=custom_prompt or None,
            summary=summary or "No summary generated",
            transcription_model=None,  # or some default value if applicable
            author=author or 'Unknown',
            ingestion_date=ingestion_date
        )
        return result
    except Exception as e:
        logging.error(f"Failed to ingest article to the database: {e}")
        return str(e)


def scrape_and_summarize(url, custom_prompt_arg, api_name, api_key, keywords, custom_article_title):
    # Step 1: Scrape the article
    article_data = scrape_article(url)
    print(f"Scraped Article Data: {article_data}")  # Debugging statement
    if not article_data:
        return "Failed to scrape the article."

    # Use the custom title if provided, otherwise use the scraped title
    title = custom_article_title.strip() if custom_article_title else article_data.get('title', 'Untitled')
    author = article_data.get('author', 'Unknown')
    content = article_data.get('content', '')
    ingestion_date = datetime.now().strftime('%Y-%m-%d')

    print(f"Title: {title}, Author: {author}, Content Length: {len(content)}")  # Debugging statement

    # Custom prompt for the article
    article_custom_prompt = custom_prompt_arg or "Summarize this article."

    # Step 2: Summarize the article
    summary = None
    if api_name:
        logging.debug(f"Article_Summarizer: Summarization being performed by {api_name}")

        # Sanitize filename for saving the JSON file
        sanitized_title = sanitize_filename(title)
        json_file_path = os.path.join("Results", f"{sanitized_title}_segments.json")

        with open(json_file_path, 'w') as json_file:
            json.dump([{'text': content}], json_file, indent=2)

        try:
            if api_name.lower() == 'openai':
                openai_api_key = api_key if api_key else config.get('API', 'openai_api_key', fallback=None)
                logging.debug(f"Article_Summarizer: trying to summarize with openAI")
                summary = summarize_with_openai(openai_api_key, json_file_path, article_custom_prompt)
            elif api_name.lower() == "anthropic":
                anthropic_api_key = api_key if api_key else config.get('API', 'anthropic_api_key', fallback=None)
                logging.debug(f"Article_Summarizer: Trying to summarize with anthropic")
                summary = summarize_with_claude(anthropic_api_key, json_file_path, anthropic_model,
                                                custom_prompt_arg=article_custom_prompt)
            elif api_name.lower() == "cohere":
                cohere_api_key = api_key if api_key else config.get('API', 'cohere_api_key', fallback=None)
                logging.debug(f"Article_Summarizer: Trying to summarize with cohere")
                summary = summarize_with_cohere(cohere_api_key, json_file_path, cohere_model,
                                                custom_prompt_arg=article_custom_prompt)
            elif api_name.lower() == "groq":
                groq_api_key = api_key if api_key else config.get('API', 'groq_api_key', fallback=None)
                logging.debug(f"Article_Summarizer: Trying to summarize with Groq")
                summary = summarize_with_groq(groq_api_key, json_file_path, groq_model,
                                              custom_prompt_arg=article_custom_prompt)
            elif api_name.lower() == "llama":
                llama_token = api_key if api_key else config.get('API', 'llama_api_key', fallback=None)
                llama_ip = llama_api_IP
                logging.debug(f"Article_Summarizer: Trying to summarize with Llama.cpp")
                summary = summarize_with_llama(llama_ip, json_file_path, llama_token, article_custom_prompt)
            elif api_name.lower() == "kobold":
                kobold_token = api_key if api_key else config.get('API', 'kobold_api_key', fallback=None)
                kobold_ip = kobold_api_IP
                logging.debug(f"Article_Summarizer: Trying to summarize with kobold.cpp")
                summary = summarize_with_kobold(kobold_ip, json_file_path, kobold_token, article_custom_prompt)
            elif api_name.lower() == "ooba":
                ooba_token = api_key if api_key else config.get('API', 'ooba_api_key', fallback=None)
                ooba_ip = ooba_api_IP
                logging.debug(f"Article_Summarizer: Trying to summarize with oobabooga")
                summary = summarize_with_oobabooga(ooba_ip, json_file_path, ooba_token, article_custom_prompt)
            elif api_name.lower() == "tabbyapi":
                tabbyapi_key = api_key if api_key else config.get('API', 'tabby_api_key', fallback=None)
                tabbyapi_ip = tabby_api_IP
                logging.debug(f"Article_Summarizer: Trying to summarize with tabbyapi")
                tabby_model = llm_model
                summary = summarize_with_tabbyapi(tabbyapi_key, tabbyapi_ip, json_file_path, tabby_model,
                                                  article_custom_prompt)
            elif api_name.lower() == "vllm":
                logging.debug(f"Article_Summarizer: Trying to summarize with VLLM")
                summary = summarize_with_vllm(vllm_api_url, vllm_api_key, llm_model, json_file_path,
                                              article_custom_prompt)
            elif api_name.lower() == "huggingface":
                huggingface_api_key = api_key if api_key else config.get('API', 'huggingface_api_key', fallback=None)
                logging.debug(f"Article_Summarizer: Trying to summarize with huggingface")
                summary = summarize_with_huggingface(huggingface_api_key, json_file_path, article_custom_prompt)
            elif api_name.lower() == "openrouter":
                openrouter_api_key = api_key if api_key else config.get('API', 'openrouter_api_key', fallback=None)
                logging.debug(f"Article_Summarizer: Trying to summarize with openrouter")
                summary = summarize_with_openrouter(openrouter_api_key, json_file_path, article_custom_prompt)
        except requests.exceptions.ConnectionError as e:
            logging.error(f"Connection error while trying to summarize with {api_name}: {str(e)}")

        if summary:
            logging.info(f"Article_Summarizer: Summary generated using {api_name} API")
            save_summary_to_file(summary, json_file_path)
        else:
            summary = "Summary not available"
            logging.warning(f"Failed to generate summary using {api_name} API")

    else:
        summary = "Article Summarization: No API provided for summarization."

    print(f"Summary: {summary}")  # Debugging statement

    # Step 3: Ingest the article into the database
    ingestion_result = ingest_article_to_db(url, title, author, content, keywords, summary, ingestion_date,
                                            article_custom_prompt)

    return f"Title: {title}\nAuthor: {author}\nSummary: {summary}\nIngestion Result: {ingestion_result}"


def ingest_unstructured_text(text, custom_prompt, api_name, api_key, keywords, custom_article_title):
    title = custom_article_title.strip() if custom_article_title else "Unstructured Text"
    author = "Unknown"
    ingestion_date = datetime.now().strftime('%Y-%m-%d')

    # Summarize the unstructured text
    if api_name:
        json_file_path = f"Results/{title.replace(' ', '_')}_segments.json"
        with open(json_file_path, 'w') as json_file:
            json.dump([{'text': text}], json_file, indent=2)

        if api_name.lower() == 'openai':
            summary = summarize_with_openai(api_key, json_file_path, custom_prompt)
        # Add other APIs as needed
        else:
            summary = "Unsupported API."
    else:
        summary = "No API provided for summarization."

    # Ingest the unstructured text into the database
    ingestion_result = ingest_article_to_db('Unstructured Text', title, author, text, keywords, summary, ingestion_date,
                                            custom_prompt)
    return f"Title: {title}\nSummary: {summary}\nIngestion Result: {ingestion_result}"



#
#
#######################################################################################################################