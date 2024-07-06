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
import datetime
from datetime import datetime
import gradio as gr
import json
import os
import logging
import requests
# 3rd-Party Imports
import bs4
import huggingface_hub
import tokenizers
import torchvision
import transformers
from tqdm import tqdm
# Local Imports
from Article_Extractor_Lib import scrape_article
from Local_LLM_Inference_Engine_Lib import summarize_with_local_llm
from Local_Summarization_Lib import summarize_with_llama, summarize_with_oobabooga, summarize_with_tabbyapi, summarize_with_vllm, summarize_with_kobold, save_summary_to_file
from Summarization_General_Lib import summarize_with_openai, summarize_with_anthropic, summarize_with_cohere, summarize_with_groq, summarize_with_openrouter, summarize_with_deepseek, summarize_with_huggingface
from SQLite_DB import Database, create_tables, add_media_with_keywords
from Video_DL_Ingestion_Lib import sanitize_filename

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
        ingestion_date = ingestion_date or datetime.datetime.now().strftime('%Y-%m-%d')

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


def scrape_and_summarize_multiple(urls, custom_prompt_arg, api_name, api_key, keywords, custom_article_titles):
    urls = [url.strip() for url in urls.split('\n') if url.strip()]
    custom_titles = custom_article_titles.split('\n') if custom_article_titles else []

    results = []
    errors = []

    # Create a progress bar
    progress = gr.Progress()

    for i, url in tqdm(enumerate(urls), total=len(urls), desc="Processing URLs"):
        custom_title = custom_titles[i] if i < len(custom_titles) else None
        try:
            result = scrape_and_summarize(url, custom_prompt_arg, api_name, api_key, keywords, custom_title)
            results.append(f"Results for URL {i + 1}:\n{result}")
        except Exception as e:
            error_message = f"Error processing URL {i + 1} ({url}): {str(e)}"
            errors.append(error_message)
            results.append(f"Failed to process URL {i + 1}: {url}")

        # Update progress
        progress((i + 1) / len(urls), desc=f"Processed {i + 1}/{len(urls)} URLs")

    # Combine results and errors
    combined_output = "\n".join(results)
    if errors:
        combined_output += "\n\nErrors encountered:\n" + "\n".join(errors)

    return combined_output


def scrape_and_summarize(url, custom_prompt_arg, api_name, api_key, keywords, custom_article_title):
    try:
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
                    # def summarize_with_openai(api_key, input_data, custom_prompt_arg)
                    summary = summarize_with_openai(api_key, json_file_path, article_custom_prompt)

                elif api_name.lower() == "anthropic":
                    # def summarize_with_anthropic(api_key, input_data, model, custom_prompt_arg, max_retries=3, retry_delay=5):
                    summary = summarize_with_anthropic(api_key, json_file_path, article_custom_prompt)
                elif api_name.lower() == "cohere":
                    # def summarize_with_cohere(api_key, input_data, model, custom_prompt_arg)
                    summary = summarize_with_cohere(api_key, json_file_path, article_custom_prompt)

                elif api_name.lower() == "groq":
                    logging.debug(f"MAIN: Trying to summarize with groq")
                    # def summarize_with_groq(api_key, input_data, model, custom_prompt_arg):
                    summary = summarize_with_groq(api_key, json_file_path, article_custom_prompt)

                elif api_name.lower() == "openrouter":
                    logging.debug(f"MAIN: Trying to summarize with OpenRouter")
                    # def summarize_with_openrouter(api_key, input_data, custom_prompt_arg):
                    summary = summarize_with_openrouter(api_key, json_file_path, article_custom_prompt)

                elif api_name.lower() == "deepseek":
                    logging.debug(f"MAIN: Trying to summarize with DeepSeek")
                    # def summarize_with_deepseek(api_key, input_data, custom_prompt_arg):
                    summary = summarize_with_deepseek(api_key, json_file_path, article_custom_prompt)

                elif api_name.lower() == "llama.cpp":
                    logging.debug(f"MAIN: Trying to summarize with Llama.cpp")
                    # def summarize_with_llama(api_url, file_path, token, custom_prompt)
                    summary = summarize_with_llama(json_file_path, article_custom_prompt)

                elif api_name.lower() == "kobold":
                    logging.debug(f"MAIN: Trying to summarize with Kobold.cpp")
                    # def summarize_with_kobold(input_data, kobold_api_token, custom_prompt_input, api_url):
                    summary = summarize_with_kobold(json_file_path, api_key, article_custom_prompt)

                elif api_name.lower() == "ooba":
                    # def summarize_with_oobabooga(input_data, api_key, custom_prompt, api_url):
                    summary = summarize_with_oobabooga(json_file_path, api_key, article_custom_prompt)

                elif api_name.lower() == "tabbyapi":
                    # def summarize_with_tabbyapi(input_data, tabby_model, custom_prompt_input, api_key=None, api_IP):
                    summary = summarize_with_tabbyapi(json_file_path, article_custom_prompt)

                elif api_name.lower() == "vllm":
                    logging.debug(f"MAIN: Trying to summarize with VLLM")
                    # def summarize_with_vllm(api_key, input_data, custom_prompt_input):
                    summary = summarize_with_vllm(json_file_path, article_custom_prompt)

                elif api_name.lower() == "local-llm":
                    logging.debug(f"MAIN: Trying to summarize with Local LLM")
                    summary = summarize_with_local_llm(json_file_path, article_custom_prompt)

                elif api_name.lower() == "huggingface":
                    logging.debug(f"MAIN: Trying to summarize with huggingface")
                    # def summarize_with_huggingface(api_key, input_data, custom_prompt_arg):
                    summarize_with_huggingface(api_key, json_file_path, article_custom_prompt)
                # Add additional API handlers here...
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

        return f"Title: {title}\nAuthor: {author}\nIngestion Result: {ingestion_result}\n\nSummary: {summary}\n\nArticle Contents: {content}"
    except Exception as e:
        logging.error(f"Error processing URL {url}: {str(e)}")
        return f"Failed to process URL {url}: {str(e)}"


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