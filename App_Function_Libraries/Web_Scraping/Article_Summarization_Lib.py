# Article_Summarization_Lib.py
#########################################
# Article Summarization Library
# This library is used to handle summarization of articles.
import asyncio
# FIXME - this library should be refactored into `Article_Extractor_Lib` and then renamed to `Web_Scraping_Lib`

#
####
#
####################
# Function List
#
# 1.
#
####################
#
# Import necessary libraries
import datetime
from datetime import datetime
import gradio as gr
import json
import os
import logging
import requests
# 3rd-Party Imports
#
# Local Imports
from App_Function_Libraries.Utils.Utils import sanitize_filename, load_comprehensive_config
from App_Function_Libraries.Web_Scraping.Article_Extractor_Lib import scrape_article
from App_Function_Libraries.Summarization.Local_Summarization_Lib import summarize_with_llama, summarize_with_oobabooga, \
    summarize_with_tabbyapi, \
    summarize_with_vllm, summarize_with_kobold, save_summary_to_file, summarize_with_local_llm, summarize_with_ollama, \
    summarize_with_custom_openai
from App_Function_Libraries.Summarization.Summarization_General_Lib import summarize_with_openai, summarize_with_anthropic, summarize_with_cohere, \
    summarize_with_groq, summarize_with_openrouter, summarize_with_deepseek, summarize_with_huggingface, \
    summarize_with_mistral
from App_Function_Libraries.DB.DB_Manager import ingest_article_to_db
#
#######################################################################################################################
# Function Definitions
#

async def scrape_and_summarize_multiple(urls, custom_prompt_arg, api_name, api_key, keywords, custom_article_titles, system_message=None):
    urls = [url.strip() for url in urls.split('\n') if url.strip()]
    custom_titles = custom_article_titles.split('\n') if custom_article_titles else []

    results = []
    errors = []

    # Create a progress bar
    progress = gr.Progress()

    # FIXME - add progress tracking to the gradio UI
    for i, url in enumerate(urls):
        custom_title = custom_titles[i] if i < len(custom_titles) else None
        try:
            article = await scrape_article(url)
            if article and article['extraction_successful']:
                if custom_title:
                    article['title'] = custom_title
                results.append(article)
        except Exception as e:
            error_message = f"Error processing URL {i + 1} ({url}): {str(e)}"
            errors.append(error_message)

        # Update progress
        progress((i + 1) / len(urls), desc=f"Processed {i + 1}/{len(urls)} URLs")

    if errors:
        logging.error("\n".join(errors))

    return results



def scrape_and_summarize(url, custom_prompt_arg, api_name, api_key, keywords, custom_article_title, system_message=None):
    try:
        # Step 1: Scrape the article
        article_data = asyncio.run(scrape_article(url))
        print(f"Scraped Article Data: {article_data}")  # Debugging statement
        if not article_data:
            return "Failed to scrape the article."

        # Use the custom title if provided, otherwise use the scraped title
        title = custom_article_title.strip() if custom_article_title else article_data.get('title', 'Untitled')
        author = article_data.get('author', 'Unknown')
        content = article_data.get('content', '')
        ingestion_date = datetime.now().strftime('%Y-%m-%d')

        print(f"Title: {title}, Author: {author}, Content Length: {len(content)}")  # Debugging statement

        # Custom system prompt for the article
        system_message = system_message or "Act as a professional summarizer and summarize this article."
        # Custom prompt for the article
        article_custom_prompt = custom_prompt_arg or "Act as a professional summarizer and summarize this article."

        # Step 2: Summarize the article
        summary = None
        if api_name:
            logging.debug(f"Article_Summarizer: Summarization being performed by {api_name}")

            # Sanitize filename for saving the JSON file
            sanitized_title = sanitize_filename(title)
            json_file_path = os.path.join("Results", f"{sanitized_title}_segments.json")

            with open(json_file_path, 'w') as json_file:
                json.dump([{'text': content}], json_file, indent=2)
            config = load_comprehensive_config()
            try:
                if api_name.lower() == 'openai':
                    # def summarize_with_openai(api_key, input_data, custom_prompt_arg)
                    summary = summarize_with_openai(api_key, json_file_path, article_custom_prompt, system_message)

                elif api_name.lower() == "anthropic":
                    # def summarize_with_anthropic(api_key, input_data, model, custom_prompt_arg, max_retries=3, retry_delay=5):
                    summary = summarize_with_anthropic(api_key, json_file_path, article_custom_prompt, system_message)
                elif api_name.lower() == "cohere":
                    # def summarize_with_cohere(api_key, input_data, model, custom_prompt_arg)
                    summary = summarize_with_cohere(api_key, json_file_path, article_custom_prompt, system_message)

                elif api_name.lower() == "groq":
                    logging.debug(f"MAIN: Trying to summarize with groq")
                    # def summarize_with_groq(api_key, input_data, model, custom_prompt_arg):
                    summary = summarize_with_groq(api_key, json_file_path, article_custom_prompt, system_message)

                elif api_name.lower() == "openrouter":
                    logging.debug(f"MAIN: Trying to summarize with OpenRouter")
                    # def summarize_with_openrouter(api_key, input_data, custom_prompt_arg):
                    summary = summarize_with_openrouter(api_key, json_file_path, article_custom_prompt, system_message)

                elif api_name.lower() == "deepseek":
                    logging.debug(f"MAIN: Trying to summarize with DeepSeek")
                    # def summarize_with_deepseek(api_key, input_data, custom_prompt_arg):
                    summary = summarize_with_deepseek(api_key, json_file_path, article_custom_prompt, system_message)

                elif api_name.lower() == "mistral":
                    summary = summarize_with_mistral(api_key, json_file_path, article_custom_prompt, system_message)

                elif api_name.lower() == "llama.cpp":
                    logging.debug(f"MAIN: Trying to summarize with Llama.cpp")
                    # def summarize_with_llama(api_url, file_path, token, custom_prompt)
                    summary = summarize_with_llama(json_file_path, article_custom_prompt, config['Local-API']['llama_api_key'], None, system_message)
                elif api_name.lower() == "kobold":
                    logging.debug(f"MAIN: Trying to summarize with Kobold.cpp")
                    # def summarize_with_kobold(input_data, kobold_api_token, custom_prompt_input, api_url):
                    summary = summarize_with_kobold(json_file_path, api_key, article_custom_prompt, system_message)

                elif api_name.lower() == "ooba":
                    # def summarize_with_oobabooga(input_data, api_key, custom_prompt, api_url):
                    summary = summarize_with_oobabooga(json_file_path, api_key, article_custom_prompt, system_message)

                elif api_name.lower() == "tabbyapi":
                    # def summarize_with_tabbyapi(input_data, tabby_model, custom_prompt_input, api_key=None, api_IP):
                    summary = summarize_with_tabbyapi(json_file_path, article_custom_prompt, system_message)

                elif api_name.lower() == "vllm":
                    logging.debug(f"MAIN: Trying to summarize with VLLM")
                    # def summarize_with_vllm(api_key, input_data, custom_prompt_input):
                    summary = summarize_with_vllm(json_file_path, article_custom_prompt, None, None, system_message)
                elif api_name.lower() == "local-llm":
                    logging.debug(f"MAIN: Trying to summarize with Local LLM")
                    summary = summarize_with_local_llm(json_file_path, article_custom_prompt, system_message)

                elif api_name.lower() == "ollama":
                    logging.debug(f"MAIN: Trying to summarize with OLLAMA")
                    # def summarize_with_ollama(input_data, api_key, custom_prompt, api_url):
                    summary = summarize_with_ollama(json_file_path, article_custom_prompt, api_key, None, system_message, None)

                elif api_name == "custom_openai_api":
                    logging.debug(f"MAIN: Trying to summarize with Custom_OpenAI API")
                    summary = summarize_with_custom_openai(json_file_path, article_custom_prompt, api_key, temp=None, system_message=None)


                elif api_name.lower() == "huggingface":
                    logging.debug(f"MAIN: Trying to summarize with huggingface")
                    # def summarize_with_huggingface(api_key, input_data, custom_prompt_arg):
                    summarize_with_huggingface(api_key, json_file_path, article_custom_prompt, system_message)
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


def scrape_and_no_summarize_then_ingest(url, keywords, custom_article_title):
    try:
        # Step 1: Scrape the article
        article_data = asyncio.run(scrape_article(url))
        print(f"Scraped Article Data: {article_data}")  # Debugging statement
        if not article_data:
            return "Failed to scrape the article."

        # Use the custom title if provided, otherwise use the scraped title
        title = custom_article_title.strip() if custom_article_title else article_data.get('title', 'Untitled')
        author = article_data.get('author', 'Unknown')
        content = article_data.get('content', '')
        ingestion_date = datetime.now().strftime('%Y-%m-%d')

        print(f"Title: {title}, Author: {author}, Content Length: {len(content)}")  # Debugging statement

        # Step 2: Ingest the article into the database
        ingestion_result = ingest_article_to_db(url, title, author, content, keywords, ingestion_date, None, None)

        return f"Title: {title}\nAuthor: {author}\nIngestion Result: {ingestion_result}\n\nArticle Contents: {content}"
    except Exception as e:
        logging.error(f"Error processing URL {url}: {str(e)}")
        return f"Failed to process URL {url}: {str(e)}"


def ingest_unstructured_text(text, custom_prompt, api_name, api_key, keywords, custom_article_title, system_message=None):
    title = custom_article_title.strip() if custom_article_title else "Unstructured Text"
    author = "Unknown"
    ingestion_date = datetime.now().strftime('%Y-%m-%d')

    # Summarize the unstructured text
    if api_name:
        json_file_path = f"Results/{title.replace(' ', '_')}_segments.json"
        with open(json_file_path, 'w') as json_file:
            json.dump([{'text': text}], json_file, indent=2)

        if api_name.lower() == 'openai':
            summary = summarize_with_openai(api_key, json_file_path, custom_prompt, system_message)
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