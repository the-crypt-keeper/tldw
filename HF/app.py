#!/usr/bin/env python3
# Std Lib Imports
import argparse
import asyncio
import atexit
import configparser
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
import unicodedata
from multiprocessing import process
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
import webbrowser
import zipfile

# 3rd-Party Module Imports
from bs4 import BeautifulSoup
import gradio as gr
import nltk
from playwright.async_api import async_playwright
import requests
from requests.exceptions import RequestException
import trafilatura
import yt_dlp

# OpenAI Tokenizer support
from openai import OpenAI
from tqdm import tqdm
import tiktoken

# Other Tokenizers
from transformers import GPT2Tokenizer

#######################
# Logging Setup
#

log_level = "DEBUG"
logging.basicConfig(level=getattr(logging, log_level), format='%(asctime)s - %(levelname)s - %(message)s')
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

#############
# Global variables setup

custom_prompt = None

#
#
#######################

#######################
# Function Sections
#


abc_xyz = """
    Database Setup
    Config Loading
    System Checks
    DataBase Functions
    Processing Paths and local file handling
    Video Download/Handling
    Audio Transcription
    Diarization
    Chunking-related Techniques & Functions
    Tokenization-related Techniques & Functions
    Summarizers
    Gradio UI
    Main
"""

#
#
#######################


#######################
#
#       TL/DW: Too Long Didn't Watch
#
#  Project originally created by https://github.com/the-crypt-keeper
#  Modifications made by https://github.com/rmusser01
#  All credit to the original authors, I've just glued shit together.
#
#
# Usage:
#
#   Download Audio only from URL -> Transcribe audio:
#       python summarize.py https://www.youtube.com/watch?v=4nd1CDZP21s`
#
#   Download Audio+Video from URL -> Transcribe audio from Video:**
#       python summarize.py -v https://www.youtube.com/watch?v=4nd1CDZP21s`
#
#   Download Audio only from URL -> Transcribe audio -> Summarize using (`anthropic`/`cohere`/`openai`/`llama` (llama.cpp)/`ooba` (oobabooga/text-gen-webui)/`kobold` (kobold.cpp)/`tabby` (Tabbyapi)) API:**
#       python summarize.py -v https://www.youtube.com/watch?v=4nd1CDZP21s -api <your choice of API>` - Make sure to put your API key into `config.txt` under the appropriate API variable
#
#   Download Audio+Video from a list of videos in a text file (can be file paths or URLs) and have them all summarized:**
#       python summarize.py ./local/file_on_your/system --api_name <API_name>`
#
#   Run it as a WebApp**
#       python summarize.py -gui` - This requires you to either stuff your API keys into the `config.txt` file, or pass them into the app every time you want to use it.
#           Can be helpful for setting up a shared instance, but not wanting people to perform inference on your server.
#
#######################


#######################
# Random issues I've encountered and how I solved them:
#   1. Something about cuda nn library missing, even though cuda is installed...
#       https://github.com/tensorflow/tensorflow/issues/54784 - Basically, installing zlib made it go away. idk.
#       Or https://github.com/SYSTRAN/faster-whisper/issues/85
#
#   2. ERROR: Could not install packages due to an OSError: [WinError 2] The system cannot find the file specified: 'C:\\Python312\\Scripts\\dateparser-download.exe' -> 'C:\\Python312\\Scripts\\dateparser-download.exe.deleteme'
#       Resolved through adding --user to the pip install command
#
#   3. ?
#
#######################


#######################
# DB Setup

# Handled by SQLite_DB.py

#######################

######################
# Global Variables
global local_llm_model, \
    userOS, \
    processing_choice, \
    segments, \
    detail_level_number, \
    summary, \
    audio_file, \
    detail_level

process = None


#######################
# Config loading
#

# Read configuration from file
config = configparser.ConfigParser()
config.read('config.txt')

# API Keys
anthropic_api_key = config.get('API', 'anthropic_api_key', fallback=None)
logging.debug(f"Loaded Anthropic API Key: {anthropic_api_key}")

cohere_api_key = config.get('API', 'cohere_api_key', fallback=None)
logging.debug(f"Loaded cohere API Key: {cohere_api_key}")

groq_api_key = config.get('API', 'groq_api_key', fallback=None)
logging.debug(f"Loaded groq API Key: {groq_api_key}")

openai_api_key = config.get('API', 'openai_api_key', fallback=None)
logging.debug(f"Loaded openAI Face API Key: {openai_api_key}")

huggingface_api_key = config.get('API', 'huggingface_api_key', fallback=None)
logging.debug(f"Loaded HuggingFace Face API Key: {huggingface_api_key}")

openrouter_api_key = config.get('Local-API', 'openrouter', fallback=None)
logging.debug(f"Loaded OpenRouter API Key: {openrouter_api_key}")

# Models
anthropic_model = config.get('API', 'anthropic_model', fallback='claude-3-sonnet-20240229')
cohere_model = config.get('API', 'cohere_model', fallback='command-r-plus')
groq_model = config.get('API', 'groq_model', fallback='llama3-70b-8192')
openai_model = config.get('API', 'openai_model', fallback='gpt-4-turbo')
huggingface_model = config.get('API', 'huggingface_model', fallback='CohereForAI/c4ai-command-r-plus')
openrouter_model = config.get('API', 'openrouter_model', fallback='microsoft/wizardlm-2-8x22b')

# Local-Models
kobold_api_IP = config.get('Local-API', 'kobold_api_IP', fallback='http://127.0.0.1:5000/api/v1/generate')
kobold_api_key = config.get('Local-API', 'kobold_api_key', fallback='')

llama_api_IP = config.get('Local-API', 'llama_api_IP', fallback='http://127.0.0.1:8080/v1/chat/completions')
llama_api_key = config.get('Local-API', 'llama_api_key', fallback='')

ooba_api_IP = config.get('Local-API', 'ooba_api_IP', fallback='http://127.0.0.1:5000/v1/chat/completions')
ooba_api_key = config.get('Local-API', 'ooba_api_key', fallback='')

tabby_api_IP = config.get('Local-API', 'tabby_api_IP', fallback='http://127.0.0.1:5000/api/v1/generate')
tabby_api_key = config.get('Local-API', 'tabby_api_key', fallback=None)

vllm_api_url = config.get('Local-API', 'vllm_api_IP', fallback='http://127.0.0.1:500/api/v1/chat/completions')
vllm_api_key = config.get('Local-API', 'vllm_api_key', fallback=None)

# Retrieve output paths from the configuration file
output_path = config.get('Paths', 'output_path', fallback='results')

# Retrieve processing choice from the configuration file
processing_choice = config.get('Processing', 'processing_choice', fallback='cpu')

# Log file
# logging.basicConfig(filename='debug-runtime.log', encoding='utf-8', level=logging.DEBUG)

#
#
#######################


#######################
# System Startup Notice
#

# Dirty hack - sue me. - FIXME - fix this...
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

whisper_models = ["small", "medium", "small.en", "medium.en"]
source_languages = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "ko": "Korean",
    "fr": "French"
}
source_language_list = [key[0] for key in source_languages.items()]


def print_hello():
    print(r"""_____  _          ________  _    _                                 
|_   _|| |        / /|  _  \| |  | | _                              
  | |  | |       / / | | | || |  | |(_)                             
  | |  | |      / /  | | | || |/\| |                                
  | |  | |____ / /   | |/ / \  /\  / _                              
  \_/  \_____//_/    |___/   \/  \/ (_)                             


 _                   _                                              
| |                 | |                                             
| |_   ___    ___   | |  ___   _ __    __ _                         
| __| / _ \  / _ \  | | / _ \ | '_ \  / _` |                        
| |_ | (_) || (_) | | || (_) || | | || (_| | _                      
 \__| \___/  \___/  |_| \___/ |_| |_| \__, |( )                     
                                       __/ ||/                      
                                      |___/                         
     _  _      _         _  _                      _          _     
    | |(_)    | |       ( )| |                    | |        | |    
  __| | _   __| | _ __  |/ | |_  __      __  __ _ | |_   ___ | |__  
 / _` || | / _` || '_ \    | __| \ \ /\ / / / _` || __| / __|| '_ \ 
| (_| || || (_| || | | |   | |_   \ V  V / | (_| || |_ | (__ | | | |
 \__,_||_| \__,_||_| |_|    \__|   \_/\_/   \__,_| \__| \___||_| |_|
""")
    time.sleep(1)
    return


#
#
#######################################################################################################################


########################################################################################################################
# DB Setup
#
# 1. platform_check()
# 2. cuda_check()
# 3. decide_cpugpu()
# 4. check_ffmpeg()
# 5. download_ffmpeg()
#
#######################


#######################
# DB Functions
#
#     create_tables()
#     add_keyword()
#     delete_keyword()
#     add_keyword()
#     add_media_with_keywords()
#     search_db()
#     format_results()
#     search_and_display()
#     export_to_csv()
#     is_valid_url()
#     is_valid_date()
#
########################################################################################################################


########################################################################################################################
# Processing Paths and local file handling
#
# Function List
# 1. read_paths_from_file(file_path)
# 2. process_path(path)
# 3. process_local_file(file_path)
# 4. read_paths_from_file(file_path: str) -> List[str]
#
#
########################################################################################################################


#######################################################################################################################
# Online Article Extraction / Handling
#
# Article_Extractor_Lib.py
#########################################
# Article Extraction Library
# This library is used to handle scraping and extraction of articles from web pages.
# Currently, uses a combination of beatifulsoup4 and trafilatura to extract article text.
# Firecrawl would be a better option for this, but it is not yet implemented.
####

####################
# Function List
#
# 1. get_page_title(url)
# 2. get_article_text(url)
# 3. get_article_title(article_url_arg)
#
####################



# Import necessary libraries
import os
import logging
import huggingface_hub
import tokenizers
import torchvision
import transformers
# 3rd-Party Imports
import asyncio
import playwright
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import requests
import trafilatura
# Import Local
import summarize
def get_page_title(url: str) -> str:
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        title_tag = soup.find('title')
        return title_tag.string.strip() if title_tag else "Untitled"
    except requests.RequestException as e:
        logging.error(f"Error fetching page title: {e}")
        return "Untitled"


def get_artice_title(article_url_arg: str) -> str:
    # Use beautifulsoup to get the page title - Really should be using ytdlp for this....
    article_title = get_page_title(article_url_arg)


def scrape_article(url):
    async def fetch_html(url: str) -> str:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3")
            page = await context.new_page()
            await page.goto(url)
            await page.wait_for_load_state("networkidle")  # Wait for the network to be idle
            content = await page.content()
            await browser.close()
            return content

    def extract_article_data(html: str) -> dict:
        downloaded = trafilatura.extract(html, include_comments=False, include_tables=False, include_images=False)
        if downloaded:
            metadata = trafilatura.extract_metadata(html)
            if metadata:
                return {
                    'title': metadata.title if metadata.title else 'N/A',
                    'author': metadata.author if metadata.author else 'N/A',
                    'content': downloaded,
                    'date': metadata.date if metadata.date else 'N/A',
                }
            else:
                print("Metadata extraction failed.")
                return None
        else:
            print("Content extraction failed.")
            return None

    def convert_html_to_markdown(html: str) -> str:
        soup = BeautifulSoup(html, 'html.parser')
        # Convert each paragraph to markdown
        for para in soup.find_all('p'):
            para.append('\n')  # Add a newline at the end of each paragraph for markdown separation

        # Use .get_text() with separator to keep paragraph separation
        text = soup.get_text(separator='\n\n')

        return text

    async def fetch_and_extract_article(url: str):
        html = await fetch_html(url)
        print("HTML Content:", html[:500])  # Print first 500 characters of the HTML for inspection
        article_data = extract_article_data(html)
        if article_data:
            article_data['content'] = convert_html_to_markdown(article_data['content'])
            return article_data
        else:
            return None

    # Using asyncio.run to handle event loop creation and execution
    article_data = asyncio.run(fetch_and_extract_article(url))
    return article_data

#
#
#######################################################################################################################
#
#
# Article_Summarization_Lib.py


# Import necessary libraries
import datetime
from datetime import datetime
import json
import os
import logging
# 3rd-Party Imports
import bs4
import huggingface_hub
import tokenizers
import torchvision
import transformers
# Local Imports
import summarize
import summarize




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
                tabby_model = summarize.llm_model
                summary = summarize_with_tabbyapi(tabbyapi_key, tabbyapi_ip, json_file_path, tabby_model,
                                                  article_custom_prompt)
            elif api_name.lower() == "vllm":
                logging.debug(f"Article_Summarizer: Trying to summarize with VLLM")
                summary = summarize_with_vllm(vllm_api_url, vllm_api_key, summarize.llm_model, json_file_path,
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
#
#
#######################################################################################################################


#######################################################################################################################
# Video Download/Handling
# Video-DL-Ingestion-Lib
#
# Function List
# 1. get_video_info(url)
# 2. create_download_directory(title)
# 3. sanitize_filename(title)
# 4. normalize_title(title)
# 5. get_youtube(video_url)
# 6. get_playlist_videos(playlist_url)
# 7. download_video(video_url, download_path, info_dict, download_video_flag)
# 8. save_to_file(video_urls, filename)
# 9. save_summary_to_file(summary, file_path)
# 10. process_url(url, num_speakers, whisper_model, custom_prompt, offset, api_name, api_key, vad_filter, download_video, download_audio, rolling_summarization, detail_level, question_box, keywords, ) # FIXME - UPDATE
#
#
#######################################################################################################################


#######################################################################################################################
# Audio Transcription
#
# Function List
# 1. convert_to_wav(video_file_path, offset=0, overwrite=False)
# 2. speech_to_text(audio_file_path, selected_source_lang='en', whisper_model='small.en', vad_filter=False)


# Audio_Transcription_Lib.py
#########################################
# Transcription Library
# This library is used to perform transcription of audio files.
# Currently, uses faster_whisper for transcription.
#
####
import configparser
####################
# Function List
#
# 1. convert_to_wav(video_file_path, offset=0, overwrite=False)
# 2. speech_to_text(audio_file_path, selected_source_lang='en', whisper_model='small.en', vad_filter=False)
#
####################


# Import necessary libraries to run solo for testing
import json
import logging
import os
import sys
import subprocess
import time
# Import Local

#######################################################################################################################
# Function Definitions
#

# Convert video .m4a into .wav using ffmpeg
#   ffmpeg -i "example.mp4" -ar 16000 -ac 1 -c:a pcm_s16le "output.wav"
#       https://www.gyan.dev/ffmpeg/builds/
#


# os.system(r'.\Bin\ffmpeg.exe -ss 00:00:00 -i "{video_file_path}" -ar 16000 -ac 1 -c:a pcm_s16le "{out_path}"')
def convert_to_wav(video_file_path, offset=0, overwrite=False):
    out_path = os.path.splitext(video_file_path)[0] + ".wav"

    if os.path.exists(out_path) and not overwrite:
        print(f"File '{out_path}' already exists. Skipping conversion.")
        logging.info(f"Skipping conversion as file already exists: {out_path}")
        return out_path
    print("Starting conversion process of .m4a to .WAV")
    out_path = os.path.splitext(video_file_path)[0] + ".wav"

    try:
        if os.name == "nt":
            logging.debug("ffmpeg being ran on windows")

            if sys.platform.startswith('win'):
                ffmpeg_cmd = ".\\Bin\\ffmpeg.exe"
                logging.debug(f"ffmpeg_cmd: {ffmpeg_cmd}")
            else:
                ffmpeg_cmd = 'ffmpeg'  # Assume 'ffmpeg' is in PATH for non-Windows systems

            command = [
                ffmpeg_cmd,  # Assuming the working directory is correctly set where .\Bin exists
                "-ss", "00:00:00",  # Start at the beginning of the video
                "-i", video_file_path,
                "-ar", "16000",  # Audio sample rate
                "-ac", "1",  # Number of audio channels
                "-c:a", "pcm_s16le",  # Audio codec
                out_path
            ]
            try:
                # Redirect stdin from null device to prevent ffmpeg from waiting for input
                with open(os.devnull, 'rb') as null_file:
                    result = subprocess.run(command, stdin=null_file, text=True, capture_output=True)
                if result.returncode == 0:
                    logging.info("FFmpeg executed successfully")
                    logging.debug("FFmpeg output: %s", result.stdout)
                else:
                    logging.error("Error in running FFmpeg")
                    logging.error("FFmpeg stderr: %s", result.stderr)
                    raise RuntimeError(f"FFmpeg error: {result.stderr}")
            except Exception as e:
                logging.error("Error occurred - ffmpeg doesn't like windows")
                raise RuntimeError("ffmpeg failed")
        elif os.name == "posix":
            os.system(f'ffmpeg -ss 00:00:00 -i "{video_file_path}" -ar 16000 -ac 1 -c:a pcm_s16le "{out_path}"')
        else:
            raise RuntimeError("Unsupported operating system")
        logging.info("Conversion to WAV completed: %s", out_path)
    except subprocess.CalledProcessError as e:
        logging.error("Error executing FFmpeg command: %s", str(e))
        raise RuntimeError("Error converting video file to WAV")
    except Exception as e:
        logging.error("speech-to-text: Error transcribing audio: %s", str(e))
        return {"error": str(e)}
    return out_path


# Transcribe .wav into .segments.json
def speech_to_text(audio_file_path, selected_source_lang='en', whisper_model='small.en', vad_filter=False):
    logging.info('speech-to-text: Loading faster_whisper model: %s', whisper_model)
    from faster_whisper import WhisperModel
    # Retrieve processing choice from the configuration file
    config = configparser.ConfigParser()
    config.read('config.txt')
    processing_choice = config.get('Processing', 'processing_choice', fallback='cpu')
    model = WhisperModel(whisper_model, device=f"{processing_choice}")
    time_start = time.time()
    if audio_file_path is None:
        raise ValueError("speech-to-text: No audio file provided")
    logging.info("speech-to-text: Audio file path: %s", audio_file_path)

    try:
        _, file_ending = os.path.splitext(audio_file_path)
        out_file = audio_file_path.replace(file_ending, ".segments.json")
        prettified_out_file = audio_file_path.replace(file_ending, ".segments_pretty.json")
        if os.path.exists(out_file):
            logging.info("speech-to-text: Segments file already exists: %s", out_file)
            with open(out_file) as f:
                global segments
                segments = json.load(f)
            return segments

        logging.info('speech-to-text: Starting transcription...')
        options = dict(language=selected_source_lang, beam_size=5, best_of=5, vad_filter=vad_filter)
        transcribe_options = dict(task="transcribe", **options)
        segments_raw, info = model.transcribe(audio_file_path, **transcribe_options)

        segments = []
        for segment_chunk in segments_raw:
            chunk = {
                "Time_Start": segment_chunk.start,
                "Time_End": segment_chunk.end,
                "Text": segment_chunk.text
            }
            logging.debug("Segment: %s", chunk)
            segments.append(chunk)
        logging.info("speech-to-text: Transcription completed with faster_whisper")

        # Save prettified JSON
        with open(prettified_out_file, 'w') as f:
            json.dump(segments, f, indent=2)

        # Save non-prettified JSON
        with open(out_file, 'w') as f:
            json.dump(segments, f)

    except Exception as e:
        logging.error("speech-to-text: Error transcribing audio: %s", str(e))
        raise RuntimeError("speech-to-text: Error transcribing audio")
    return segments



#
#
#######################################################################################################################
# Chunk Lib
#
#

# from transformers import GPT2Tokenizer
# import nltk
# import re


#
# # FIXME - Make sure it only downloads if it already exists, and does a check first.
# # Ensure NLTK data is downloaded
# def ntlk_prep():
#     nltk.download('punkt')
#
# # Load GPT2 tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#
#
# def load_document(file_path):
#     with open(file_path, 'r') as file:
#         text = file.read()
#     return re.sub('\s+', ' ', text).strip()
#
#
# # Chunk based on maximum number of words, using ' ' (space) as a delimiter
# def chunk_text_by_words(text, max_words=300):
#     words = text.split()
#     chunks = [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
#     return chunks
#
#
# # Chunk based on sentences, not exceeding a max amount, using nltk
# def chunk_text_by_sentences(text, max_sentences=10):
#     sentences = nltk.tokenize.sent_tokenize(text)
#     chunks = [' '.join(sentences[i:i + max_sentences]) for i in range(0, len(sentences), max_sentences)]
#     return chunks
#
#
# # Chunk text by paragraph, marking paragraphs by (delimiter) '\n\n'
# def chunk_text_by_paragraphs(text, max_paragraphs=5):
#     paragraphs = text.split('\n\n')
#     chunks = ['\n\n'.join(paragraphs[i:i + max_paragraphs]) for i in range(0, len(paragraphs), max_paragraphs)]
#     return chunks
#
#
# # Naive chunking based on token count
# def chunk_text_by_tokens(text, max_tokens=1000):
#     tokens = tokenizer.encode(text)
#     chunks = [tokenizer.decode(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]
#     return chunks
#
#
# # Hybrid approach, chunk each sentence while ensuring total token size does not exceed a maximum number
# def chunk_text_hybrid(text, max_tokens=1000):
#     sentences = nltk.tokenize.sent_tokenize(text)
#     chunks = []
#     current_chunk = []
#     current_length = 0
#
#     for sentence in sentences:
#         tokens = tokenizer.encode(sentence)
#         if current_length + len(tokens) <= max_tokens:
#             current_chunk.append(sentence)
#             current_length += len(tokens)
#         else:
#             chunks.append(' '.join(current_chunk))
#             current_chunk = [sentence]
#             current_length = len(tokens)
#
#     if current_chunk:
#         chunks.append(' '.join(current_chunk))
#
#     return chunks


# Sample text for testing
sample_text = """
Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence 
concerned with the interactions between computers and human language, in particular how to program computers 
to process and analyze large amounts of natural language data. The result is a computer capable of "understanding" 
the contents of documents, including the contextual nuances of the language within them. The technology can then 
accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves.

Challenges in natural language processing frequently involve speech recognition, natural language understanding, 
and natural language generation.

Natural language processing has its roots in the 1950s. Already in 1950, Alan Turing published an article titled 
"Computing Machinery and Intelligence" which proposed what is now called the Turing test as a criterion of intelligence.
"""

# Example usage of different chunking methods
# print("Chunking by words:")
# print(chunk_text_by_words(sample_text, max_words=50))
#
# print("\nChunking by sentences:")
# print(chunk_text_by_sentences(sample_text, max_sentences=2))
#
# print("\nChunking by paragraphs:")
# print(chunk_text_by_paragraphs(sample_text, max_paragraphs=1))
#
# print("\nChunking by tokens:")
# print(chunk_text_by_tokens(sample_text, max_tokens=50))
#
# print("\nHybrid chunking:")
# print(chunk_text_hybrid(sample_text, max_tokens=50))



#
#
#######################################################################################################################


#######################################################################################################################
# Diarization
#
# Function List 1. speaker_diarize(video_file_path, segments, embedding_model = "pyannote/embedding",
#                                   embedding_size=512, num_speakers=0)

# Local_File_Processing_Lib.py
#########################################
# Local File Processing and File Path Handling Library
# This library is used to handle processing local filepaths and URLs.
# It checks for the OS, the availability of the GPU, and the availability of the ffmpeg executable.
# If the GPU is available, it asks the user if they would like to use it for processing.
# If ffmpeg is not found, it asks the user if they would like to download it.
# The script will exit if the user chooses not to download ffmpeg.
####

####################
# Function List
#
# 1. read_paths_from_file(file_path)
# 2. process_path(path)
# 3. process_local_file(file_path)
# 4. read_paths_from_file(file_path: str) -> List[str]
#
####################

# Import necessary libraries
import os
import logging


# Local_LLM_Inference_Engine_Lib.py
#########################################
# Local LLM Inference Engine Library
# This library is used to handle downloading, configuring, and launching the Local LLM Inference Engine
#   via (llama.cpp via llamafile)
#
#
####
import atexit
import hashlib
####################
# Function List
#
# 1. download_latest_llamafile(repo, asset_name_prefix, output_filename)
# 2. download_file(url, dest_path, expected_checksum=None, max_retries=3, delay=5)
# 3. verify_checksum(file_path, expected_checksum)
# 4. cleanup_process()
# 5. signal_handler(sig, frame)
# 6. local_llm_function()
# 7. launch_in_new_terminal_windows(executable, args)
# 8. launch_in_new_terminal_linux(executable, args)
# 9. launch_in_new_terminal_mac(executable, args)
#
####################

# Import necessary libraries
import json
import logging
from multiprocessing import Process as MpProcess
import requests
import sys
import os
# Import 3rd-pary Libraries
import gradio as gr
from tqdm import tqdm



# Local_Summarization_Lib.py
#########################################
# Local Summarization Library
# This library is used to perform summarization with a 'local' inference engine.
#
####

####################
# Function List
#
# 1. summarize_with_local_llm(file_path, custom_prompt_arg)
# 2. summarize_with_llama(api_url, file_path, token, custom_prompt)
# 3. summarize_with_kobold(api_url, file_path, kobold_api_token, custom_prompt)
# 4. summarize_with_oobabooga(api_url, file_path, ooba_api_token, custom_prompt)
# 5. summarize_with_vllm(vllm_api_url, vllm_api_key_function_arg, llm_model, text, vllm_custom_prompt_function_arg)
# 6. summarize_with_tabbyapi(tabby_api_key, tabby_api_IP, text, tabby_model, custom_prompt)
# 7. save_summary_to_file(summary, file_path)
#
#
####################


# Import necessary libraries
import os
import logging
from typing import Callable



# Old_Chunking_Lib.py
#########################################
# Old Chunking Library
# This library is used to handle chunking of text for summarization.
#
####



####################
# Function List
#
# 1. chunk_transcript(transcript: str, chunk_duration: int, words_per_second) -> List[str]
# 2. summarize_chunks(api_name: str, api_key: str, transcript: List[dict], chunk_duration: int, words_per_second: int) -> str
# 3. get_chat_completion(messages, model='gpt-4-turbo')
# 4. chunk_on_delimiter(input_string: str, max_tokens: int, delimiter: str) -> List[str]
# 5. combine_chunks_with_no_minimum(chunks: List[str], max_tokens: int, chunk_delimiter="\n\n", header: Optional[str] = None, add_ellipsis_for_overflow=False) -> Tuple[List[str], List[int]]
# 6. rolling_summarize(text: str, detail: float = 0, model: str = 'gpt-4-turbo', additional_instructions: Optional[str] = None, minimum_chunk_size: Optional[int] = 500, chunk_delimiter: str = ".", summarize_recursively=False, verbose=False)
# 7. chunk_transcript(transcript: str, chunk_duration: int, words_per_second) -> List[str]
# 8. summarize_chunks(api_name: str, api_key: str, transcript: List[dict], chunk_duration: int, words_per_second: int) -> str
#
####################

# Import necessary libraries
import os
from typing import Optional

# Import 3rd party
import openai
from openai import OpenAI



import csv
import logging
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime
from typing import List, Tuple

import gradio as gr
import pandas as pd

# Import Local





# Summarization_General_Lib.py
#########################################
# General Summarization Library
# This library is used to perform summarization.
#
####
import configparser
####################
# Function List
#
# 1. extract_text_from_segments(segments: List[Dict]) -> str
# 2. summarize_with_openai(api_key, file_path, custom_prompt_arg)
# 3. summarize_with_claude(api_key, file_path, model, custom_prompt_arg, max_retries=3, retry_delay=5)
# 4. summarize_with_cohere(api_key, file_path, model, custom_prompt_arg)
# 5. summarize_with_groq(api_key, file_path, model, custom_prompt_arg)
#
#
####################


# Import necessary libraries
import os
import logging
import time
import requests
from typing import List, Dict
import json
import configparser
from requests import RequestException




# System_Checks_Lib.py
#########################################
# System Checks Library
# This library is used to check the system for the necessary dependencies to run the script.
# It checks for the OS, the availability of the GPU, and the availability of the ffmpeg executable.
# If the GPU is available, it asks the user if they would like to use it for processing.
# If ffmpeg is not found, it asks the user if they would like to download it.
# The script will exit if the user chooses not to download ffmpeg.
####

####################
# Function List
#
# 1. platform_check()
# 2. cuda_check()
# 3. decide_cpugpu()
# 4. check_ffmpeg()
# 5. download_ffmpeg()
#
####################




# Import necessary libraries
import os
import platform
import subprocess
import shutil
import zipfile
import logging






# Video_DL_Ingestion_Lib.py
#########################################
# Video Downloader and Ingestion Library
# This library is used to handle downloading videos from YouTube and other platforms.
# It also handles the ingestion of the videos into the database.
# It uses yt-dlp to extract video information and download the videos.
####

####################
# Function List
#
# 1. get_video_info(url)
# 2. create_download_directory(title)
# 3. sanitize_filename(title)
# 4. normalize_title(title)
# 5. get_youtube(video_url)
# 6. get_playlist_videos(playlist_url)
# 7. download_video(video_url, download_path, info_dict, download_video_flag)
# 8. save_to_file(video_urls, filename)
# 9. save_summary_to_file(summary, file_path)
# 10. process_url(url, num_speakers, whisper_model, custom_prompt, offset, api_name, api_key, vad_filter, download_video, download_audio, rolling_summarization, detail_level, question_box, keywords, chunk_summarization, chunk_duration_input, words_per_second_input)
#
#
####################


# Import necessary libraries to run solo for testing
from datetime import datetime
import json
import logging
import os
import re
import subprocess
import sys
import unicodedata
# 3rd-Party Imports
import yt_dlp

server_mode = False
share_public = False


#######################################################################################################################
# Function Definitions
#

def get_video_info(url: str) -> dict:
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info_dict = ydl.extract_info(url, download=False)
            return info_dict
        except Exception as e:
            logging.error(f"Error extracting video info: {e}")
            return None


def create_download_directory(title):
    base_dir = "Results"
    # Remove characters that are illegal in Windows filenames and normalize
    safe_title = normalize_title(title)
    logging.debug(f"{title} successfully normalized")
    session_path = os.path.join(base_dir, safe_title)
    if not os.path.exists(session_path):
        os.makedirs(session_path, exist_ok=True)
        logging.debug(f"Created directory for downloaded video: {session_path}")
    else:
        logging.debug(f"Directory already exists for downloaded video: {session_path}")
    return session_path


def sanitize_filename(title, max_length=255):
    # Remove invalid path characters
    title = re.sub(r'[\\/*?:"<>|]', "", title)
    # Truncate long titles to avoid filesystem errors
    return title[:max_length].rstrip()


def normalize_title(title):
    # Normalize the string to 'NFKD' form and encode to 'ascii' ignoring non-ascii characters
    title = unicodedata.normalize('NFKD', title).encode('ascii', 'ignore').decode('ascii')
    title = title.replace('/', '_').replace('\\', '_').replace(':', '_').replace('"', '').replace('*', '').replace('?',
                                                                                                                   '').replace(
        '<', '').replace('>', '').replace('|', '')
    return title


def get_youtube(video_url):
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]',
        'noplaylist': False,
        'quiet': True,
        'extract_flat': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        logging.debug("About to extract youtube info")
        info_dict = ydl.extract_info(video_url, download=False)
        logging.debug("Youtube info successfully extracted")
    return info_dict


def get_playlist_videos(playlist_url):
    ydl_opts = {
        'extract_flat': True,
        'skip_download': True,
        'quiet': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)

        if 'entries' in info:
            video_urls = [entry['url'] for entry in info['entries']]
            playlist_title = info['title']
            return video_urls, playlist_title
        else:
            print("No videos found in the playlist.")
            return [], None


def download_video(video_url, download_path, info_dict, download_video_flag):
    global video_file_path, ffmpeg_path
    global audio_file_path

    # Normalize Video Title name
    logging.debug("About to normalize downloaded video title")
    normalized_video_title = normalize_title(info_dict['title'])
    video_file_path = os.path.join(download_path, f"{normalized_video_title}.{info_dict['ext']}")

    # Check for existence of video file
    if os.path.exists(video_file_path):
        logging.info(f"Video file already exists: {video_file_path}")
        return video_file_path

    # Setup path handling for ffmpeg on different OSs
    if sys.platform.startswith('win'):
        ffmpeg_path = os.path.join(os.getcwd(), 'Bin', 'ffmpeg.exe')
    elif sys.platform.startswith('linux'):
        ffmpeg_path = 'ffmpeg'
    elif sys.platform.startswith('darwin'):
        ffmpeg_path = 'ffmpeg'

    if download_video_flag:
        video_file_path = os.path.join(download_path, f"{normalized_video_title}.mp4")

        # Set options for video and audio
        ydl_opts_video = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]',
            'outtmpl': video_file_path,
            'ffmpeg_location': ffmpeg_path
        }

        with yt_dlp.YoutubeDL(ydl_opts_video) as ydl:
            logging.debug("yt_dlp: About to download video with youtube-dl")
            ydl.download([video_url])
            logging.debug("yt_dlp: Video successfully downloaded with youtube-dl")
        return video_file_path

    else:
        return None


def save_to_file(video_urls, filename):
    with open(filename, 'w') as file:
        file.write('\n'.join(video_urls))
    print(f"Video URLs saved to {filename}")

#
#
#######################################################################################################################



#

def openai_tokenize(text: str) -> List[str]:
    encoding = tiktoken.encoding_for_model('gpt-4-turbo')
    return encoding.encode(text)

def platform_check():
    global userOS
    if platform.system() == "Linux":
        print("Linux OS detected \n Running Linux appropriate commands")
        userOS = "Linux"
    elif platform.system() == "Windows":
        print("Windows OS detected \n Running Windows appropriate commands")
        userOS = "Windows"
    else:
        print("Other OS detected \n Maybe try running things manually?")
        exit()


# Check for NVIDIA GPU and CUDA availability
def cuda_check():
    global processing_choice
    try:
        # Run nvidia-smi to capture its output
        nvidia_smi_output = subprocess.check_output("nvidia-smi", shell=True).decode()

        # Look for CUDA version in the output
        if "CUDA Version" in nvidia_smi_output:
            cuda_version = next(
                (line.split(":")[-1].strip() for line in nvidia_smi_output.splitlines() if "CUDA Version" in line),
                "Not found")
            print(f"NVIDIA GPU with CUDA Version {cuda_version} is available.")
            processing_choice = "cuda"
        else:
            print("CUDA is not installed or configured correctly.")
            processing_choice = "cpu"

    except subprocess.CalledProcessError as e:
        print(f"Failed to run 'nvidia-smi': {str(e)}")
        processing_choice = "cpu"
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        processing_choice = "cpu"

    # Optionally, check for the CUDA_VISIBLE_DEVICES env variable as an additional check
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print("CUDA_VISIBLE_DEVICES is set:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("CUDA_VISIBLE_DEVICES not set.")


# Ask user if they would like to use either their GPU or their CPU for transcription
def decide_cpugpu():
    global processing_choice
    processing_input = input("Would you like to use your GPU or CPU for transcription? (1/cuda)GPU/(2/cpu)CPU): ")
    if processing_choice == "cuda" and (processing_input.lower() == "cuda" or processing_input == "1"):
        print("You've chosen to use the GPU.")
        logging.debug("GPU is being used for processing")
        processing_choice = "cuda"
    elif processing_input.lower() == "cpu" or processing_input == "2":
        print("You've chosen to use the CPU.")
        logging.debug("CPU is being used for processing")
        processing_choice = "cpu"
    else:
        print("Invalid choice. Please select either GPU or CPU.")


# check for existence of ffmpeg
def check_ffmpeg():
    if shutil.which("ffmpeg") or (os.path.exists("Bin") and os.path.isfile(".\\Bin\\ffmpeg.exe")):
        logging.debug("ffmpeg found installed on the local system, in the local PATH, or in the './Bin' folder")
        pass
    else:
        logging.debug("ffmpeg not installed on the local system/in local PATH")
        print(
            "ffmpeg is not installed.\n\n You can either install it manually, or through your package manager of "
            "choice.\n Windows users, builds are here: https://www.gyan.dev/ffmpeg/builds/")
        if userOS == "Windows":
            download_ffmpeg()
        elif userOS == "Linux":
            print(
                "You should install ffmpeg using your platform's appropriate package manager, 'apt install ffmpeg',"
                "'dnf install ffmpeg' or 'pacman', etc.")
        else:
            logging.debug("running an unsupported OS")
            print("You're running an unspported/Un-tested OS")
            exit_script = input("Let's exit the script, unless you're feeling lucky? (y/n)")
            if exit_script == "y" or "yes" or "1":
                exit()


# Download ffmpeg
def download_ffmpeg():
    user_choice = input("Do you want to download ffmpeg? (y)Yes/(n)No: ")
    if user_choice.lower() in ['yes', 'y', '1']:
        print("Downloading ffmpeg")
        url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
        response = requests.get(url)

        if response.status_code == 200:
            print("Saving ffmpeg zip file")
            logging.debug("Saving ffmpeg zip file")
            zip_path = "ffmpeg-release-essentials.zip"
            with open(zip_path, 'wb') as file:
                file.write(response.content)

            logging.debug("Extracting the 'ffmpeg.exe' file from the zip")
            print("Extracting ffmpeg.exe from zip file to '/Bin' folder")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Find the ffmpeg.exe file within the zip
                ffmpeg_path = None
                for file_info in zip_ref.infolist():
                    if file_info.filename.endswith("ffmpeg.exe"):
                        ffmpeg_path = file_info.filename
                        break

                if ffmpeg_path is None:
                    logging.error("ffmpeg.exe not found in the zip file.")
                    print("ffmpeg.exe not found in the zip file.")
                    return

                logging.debug("checking if the './Bin' folder exists, creating if not")
                bin_folder = "Bin"
                if not os.path.exists(bin_folder):
                    logging.debug("Creating a folder for './Bin', it didn't previously exist")
                    os.makedirs(bin_folder)

                logging.debug("Extracting 'ffmpeg.exe' to the './Bin' folder")
                zip_ref.extract(ffmpeg_path, path=bin_folder)

                logging.debug("Moving 'ffmpeg.exe' to the './Bin' folder")
                src_path = os.path.join(bin_folder, ffmpeg_path)
                dst_path = os.path.join(bin_folder, "ffmpeg.exe")
                shutil.move(src_path, dst_path)

            logging.debug("Removing ffmpeg zip file")
            print("Deleting zip file (we've already extracted ffmpeg.exe, no worries)")
            os.remove(zip_path)

            logging.debug("ffmpeg.exe has been downloaded and extracted to the './Bin' folder.")
            print("ffmpeg.exe has been successfully downloaded and extracted to the './Bin' folder.")
        else:
            logging.error("Failed to download the zip file.")
            print("Failed to download the zip file.")
    else:
        logging.debug("User chose to not download ffmpeg")
        print("ffmpeg will not be downloaded.")

#
#
#######################################################################################################################



# Read configuration from file
config = configparser.ConfigParser()
config.read('../config.txt')

# API Keys
anthropic_api_key = config.get('API', 'anthropic_api_key', fallback=None)
logging.debug(f"Loaded Anthropic API Key: {anthropic_api_key}")

cohere_api_key = config.get('API', 'cohere_api_key', fallback=None)
logging.debug(f"Loaded cohere API Key: {cohere_api_key}")

groq_api_key = config.get('API', 'groq_api_key', fallback=None)
logging.debug(f"Loaded groq API Key: {groq_api_key}")

openai_api_key = config.get('API', 'openai_api_key', fallback=None)
logging.debug(f"Loaded openAI Face API Key: {openai_api_key}")

huggingface_api_key = config.get('API', 'huggingface_api_key', fallback=None)
logging.debug(f"Loaded HuggingFace Face API Key: {huggingface_api_key}")

openrouter_api_token = config.get('API', 'openrouter_api_token', fallback=None)
logging.debug(f"Loaded OpenRouter API Key: {openrouter_api_token}")

# Models
anthropic_model = config.get('API', 'anthropic_model', fallback='claude-3-sonnet-20240229')
cohere_model = config.get('API', 'cohere_model', fallback='command-r-plus')
groq_model = config.get('API', 'groq_model', fallback='llama3-70b-8192')
openai_model = config.get('API', 'openai_model', fallback='gpt-4-turbo')
huggingface_model = config.get('API', 'huggingface_model', fallback='CohereForAI/c4ai-command-r-plus')
openrouter_model = config.get('API', 'openrouter_model', fallback='mistralai/mistral-7b-instruct:free')


#######################################################################################################################
# Function Definitions
#

# FIXME
# def extract_text_from_segments(segments: List[Dict]) -> str:
#     """Extract text from segments."""
#     return " ".join([segment['text'] for segment in segments])


def extract_text_from_segments(segments):
    logging.debug(f"Segments received: {segments}")
    logging.debug(f"Type of segments: {type(segments)}")

    text = ""
    for segment in segments:
        logging.debug(f"Current segment: {segment}")
        logging.debug(f"Type of segment: {type(segment)}")
        text += segment['Text'] + " "
    return text.strip()


def summarize_with_openai(api_key, json_file_path, custom_prompt_arg):
    try:
        logging.debug("openai: Loading json data for summarization")
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        logging.debug(f"openai: Loaded data: {data}")
        logging.debug(f"openai: Type of data: {type(data)}")

        if isinstance(data, dict) and 'summary' in data:
            # If the loaded data is a dictionary and already contains a summary, return it
            logging.debug("openai: Summary already exists in the loaded data")
            return data['summary']

        # If the loaded data is a list of segment dictionaries, proceed with summarization
        segments = data

        open_ai_model = openai_model or 'gpt-4-turbo'

        logging.debug("openai: Extracting text from the segments")
        text = extract_text_from_segments(segments)

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        logging.debug(f"openai: API Key is: {api_key}")
        logging.debug("openai: Preparing data + prompt for submittal")
        openai_prompt = f"{text} \n\n\n\n{custom_prompt_arg}"
        data = {
            "model": open_ai_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional summarizer."
                },
                {
                    "role": "user",
                    "content": openai_prompt
                }
            ],
            "max_tokens": 8192,  # Adjust tokens as needed
            "temperature": 0.1
        }
        logging.debug("openai: Posting request")
        response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)

        if response.status_code == 200:
            response_data = response.json()
            if 'choices' in response_data and len(response_data['choices']) > 0:
                summary = response_data['choices'][0]['message']['content'].strip()
                logging.debug("openai: Summarization successful")
                print("openai: Summarization successful.")
                return summary
            else:
                logging.warning("openai: Summary not found in the response data")
                return "openai: Summary not available"
        else:
            logging.debug("openai: Summarization failed")
            print("openai: Failed to process summary:", response.text)
            return "openai: Failed to process summary"
    except Exception as e:
        logging.debug("openai: Error in processing: %s", str(e))
        print("openai: Error occurred while processing summary with openai:", str(e))
        return "openai: Error occurred while processing summary"


def summarize_with_claude(api_key, file_path, model, custom_prompt_arg, max_retries=3, retry_delay=5):
    try:
        logging.debug("anthropic: Loading JSON data")
        with open(file_path, 'r') as file:
            segments = json.load(file)

        logging.debug("anthropic: Extracting text from the segments file")
        text = extract_text_from_segments(segments)

        headers = {
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01',
            'Content-Type': 'application/json'
        }

        anthropic_prompt = custom_prompt_arg  # Sanitize the custom prompt
        logging.debug(f"anthropic: Prompt is {anthropic_prompt}")
        user_message = {
            "role": "user",
            "content": f"{text} \n\n\n\n{anthropic_prompt}"
        }

        data = {
            "model": model,
            "max_tokens": 4096,  # max _possible_ tokens to return
            "messages": [user_message],
            "stop_sequences": ["\n\nHuman:"],
            "temperature": 0.1,
            "top_k": 0,
            "top_p": 1.0,
            "metadata": {
                "user_id": "example_user_id",
            },
            "stream": False,
            "system": "You are a professional summarizer."
        }

        for attempt in range(max_retries):
            try:
                logging.debug("anthropic: Posting request to API")
                response = requests.post('https://api.anthropic.com/v1/messages', headers=headers, json=data)

                # Check if the status code indicates success
                if response.status_code == 200:
                    logging.debug("anthropic: Post submittal successful")
                    response_data = response.json()
                    try:
                        summary = response_data['content'][0]['text'].strip()
                        logging.debug("anthropic: Summarization successful")
                        print("Summary processed successfully.")
                        return summary
                    except (IndexError, KeyError) as e:
                        logging.debug("anthropic: Unexpected data in response")
                        print("Unexpected response format from Claude API:", response.text)
                        return None
                elif response.status_code == 500:  # Handle internal server error specifically
                    logging.debug("anthropic: Internal server error")
                    print("Internal server error from API. Retrying may be necessary.")
                    time.sleep(retry_delay)
                else:
                    logging.debug(
                        f"anthropic: Failed to summarize, status code {response.status_code}: {response.text}")
                    print(f"Failed to process summary, status code {response.status_code}: {response.text}")
                    return None

            except RequestException as e:
                logging.error(f"anthropic: Network error during attempt {attempt + 1}/{max_retries}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    return f"anthropic: Network error: {str(e)}"

    except FileNotFoundError as e:
        logging.error(f"anthropic: File not found: {file_path}")
        return f"anthropic: File not found: {file_path}"
    except json.JSONDecodeError as e:
        logging.error(f"anthropic: Invalid JSON format in file: {file_path}")
        return f"anthropic: Invalid JSON format in file: {file_path}"
    except Exception as e:
        logging.error(f"anthropic: Error in processing: {str(e)}")
        return f"anthropic: Error occurred while processing summary with Anthropic: {str(e)}"


# Summarize with Cohere
def summarize_with_cohere(api_key, file_path, model, custom_prompt_arg):
    try:
        logging.debug("cohere: Loading JSON data")
        with open(file_path, 'r') as file:
            segments = json.load(file)

        logging.debug(f"cohere: Extracting text from segments file")
        text = extract_text_from_segments(segments)

        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }

        cohere_prompt = f"{text} \n\n\n\n{custom_prompt_arg}"
        logging.debug("cohere: Prompt being sent is {cohere_prompt}")

        data = {
            "chat_history": [
                {"role": "USER", "message": cohere_prompt}
            ],
            "message": "Please provide a summary.",
            "model": model,
            "connectors": [{"id": "web-search"}]
        }

        logging.debug("cohere: Submitting request to API endpoint")
        print("cohere: Submitting request to API endpoint")
        response = requests.post('https://api.cohere.ai/v1/chat', headers=headers, json=data)
        response_data = response.json()
        logging.debug("API Response Data: %s", response_data)

        if response.status_code == 200:
            if 'text' in response_data:
                summary = response_data['text'].strip()
                logging.debug("cohere: Summarization successful")
                print("Summary processed successfully.")
                return summary
            else:
                logging.error("Expected data not found in API response.")
                return "Expected data not found in API response."
        else:
            logging.error(f"cohere: API request failed with status code {response.status_code}: {response.text}")
            print(f"Failed to process summary, status code {response.status_code}: {response.text}")
            return f"cohere: API request failed: {response.text}"

    except Exception as e:
        logging.error("cohere: Error in processing: %s", str(e))
        return f"cohere: Error occurred while processing summary with Cohere: {str(e)}"


# https://console.groq.com/docs/quickstart
def summarize_with_groq(api_key, file_path, model, custom_prompt_arg):
    try:
        logging.debug("groq: Loading JSON data")
        with open(file_path, 'r') as file:
            segments = json.load(file)

        logging.debug(f"groq: Extracting text from segments file")
        text = extract_text_from_segments(segments)

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        groq_prompt = f"{text} \n\n\n\n{custom_prompt_arg}"
        logging.debug("groq: Prompt being sent is {groq_prompt}")

        data = {
            "messages": [
                {
                    "role": "user",
                    "content": groq_prompt
                }
            ],
            "model": model
        }

        logging.debug("groq: Submitting request to API endpoint")
        print("groq: Submitting request to API endpoint")
        response = requests.post('https://api.groq.com/openai/v1/chat/completions', headers=headers, json=data)

        response_data = response.json()
        logging.debug("API Response Data: %s", response_data)

        if response.status_code == 200:
            if 'choices' in response_data and len(response_data['choices']) > 0:
                summary = response_data['choices'][0]['message']['content'].strip()
                logging.debug("groq: Summarization successful")
                print("Summarization successful.")
                return summary
            else:
                logging.error("Expected data not found in API response.")
                return "Expected data not found in API response."
        else:
            logging.error(f"groq: API request failed with status code {response.status_code}: {response.text}")
            return f"groq: API request failed: {response.text}"

    except Exception as e:
        logging.error("groq: Error in processing: %s", str(e))
        return f"groq: Error occurred while processing summary with groq: {str(e)}"


def summarize_with_openrouter(api_key, json_file_path, custom_prompt_arg):
    import requests
    import json
    global openrouter_model

    config = configparser.ConfigParser()
    file_path = 'config.txt'

    # Check if the file exists in the specified path
    if os.path.exists(file_path):
        config.read(file_path)
    elif os.path.exists('config.txt'):  # Check in the current directory
        config.read('../config.txt')
    else:
        print("config.txt not found in the specified path or current directory.")

    openrouter_api_token = config.get('API', 'openrouter_api_token', fallback=None)
    if openrouter_model is None:
        openrouter_model = "mistralai/mistral-7b-instruct:free"

    openrouter_prompt = f"{json_file_path} \n\n\n\n{custom_prompt_arg}"

    try:
        logging.debug("openrouter: Submitting request to API endpoint")
        print("openrouter: Submitting request to API endpoint")
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_api_token}",
            },
            data=json.dumps({
                "model": f"{openrouter_model}",
                "messages": [
                    {"role": "user", "content": openrouter_prompt}
                ]
            })
        )

        response_data = response.json()
        logging.debug("API Response Data: %s", response_data)

        if response.status_code == 200:
            if 'choices' in response_data and len(response_data['choices']) > 0:
                summary = response_data['choices'][0]['message']['content'].strip()
                logging.debug("openrouter: Summarization successful")
                print("openrouter: Summarization successful.")
                return summary
            else:
                logging.error("openrouter: Expected data not found in API response.")
                return "openrouter: Expected data not found in API response."
        else:
            logging.error(f"openrouter:  API request failed with status code {response.status_code}: {response.text}")
            return f"openrouter: API request failed: {response.text}"
    except Exception as e:
        logging.error("openrouter: Error in processing: %s", str(e))
        return f"openrouter: Error occurred while processing summary with openrouter: {str(e)}"

def summarize_with_huggingface(api_key, file_path, custom_prompt_arg):
    logging.debug(f"huggingface: Summarization process starting...")
    try:
        logging.debug("huggingface: Loading json data for summarization")
        with open(file_path, 'r') as file:
            segments = json.load(file)

        logging.debug("huggingface: Extracting text from the segments")
        logging.debug(f"huggingface: Segments: {segments}")
        text = ' '.join([segment['text'] for segment in segments])

        print(f"huggingface: lets make sure the HF api key exists...\n\t {api_key}")
        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        model = "microsoft/Phi-3-mini-128k-instruct"
        API_URL = f"https://api-inference.huggingface.co/models/{model}"

        huggingface_prompt = f"{text}\n\n\n\n{custom_prompt_arg}"
        logging.debug("huggingface: Prompt being sent is {huggingface_prompt}")
        data = {
            "inputs": text,
            "parameters": {"max_length": 512, "min_length": 100}  # You can adjust max_length and min_length as needed
        }

        print(f"huggingface: lets make sure the HF api key is the same..\n\t {huggingface_api_key}")

        logging.debug("huggingface: Submitting request...")

        response = requests.post(API_URL, headers=headers, json=data)

        if response.status_code == 200:
            summary = response.json()[0]['summary_text']
            logging.debug("huggingface: Summarization successful")
            print("Summarization successful.")
            return summary
        else:
            logging.error(f"huggingface: Summarization failed with status code {response.status_code}: {response.text}")
            return f"Failed to process summary, status code {response.status_code}: {response.text}"
    except Exception as e:
        logging.error("huggingface: Error in processing: %s", str(e))
        print(f"Error occurred while processing summary with huggingface: {str(e)}")
        return None

    # FIXME
    # This is here for gradio authentication
    # Its just not setup.
    # def same_auth(username, password):
    #    return username == password


#
#
#######################################################################################################################








# Set up logging
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Custom exceptions
class DatabaseError(Exception):
    pass


class InputError(Exception):
    pass


# Database connection function with connection pooling
class Database:
    def __init__(self, db_name=None):
        self.db_name = db_name or os.getenv('DB_NAME', 'media_summary.db')
        self.pool = []
        self.pool_size = 10

    @contextmanager
    def get_connection(self):
        retry_count = 5
        retry_delay = 1
        conn = None
        while retry_count > 0:
            try:
                conn = self.pool.pop() if self.pool else sqlite3.connect(self.db_name, check_same_thread=False)
                yield conn
                self.pool.append(conn)
                return
            except sqlite3.OperationalError as e:
                if 'database is locked' in str(e):
                    logging.warning(f"Database is locked, retrying in {retry_delay} seconds...")
                    retry_count -= 1
                    time.sleep(retry_delay)
                else:
                    raise DatabaseError(f"Database error: {e}")
            except Exception as e:
                raise DatabaseError(f"Unexpected error: {e}")
            finally:
                # Ensure the connection is returned to the pool even on failure
                if conn:
                    self.pool.append(conn)
        raise DatabaseError("Database is locked and retries have been exhausted")

    def execute_query(self, query: str, params: Tuple = ()) -> None:
        with self.get_connection() as conn:
            try:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
            except sqlite3.Error as e:
                raise DatabaseError(f"Database error: {e}, Query: {query}")

db = Database()


# Function to create tables with the new media schema
def create_tables() -> None:
    table_queries = [
        '''
        CREATE TABLE IF NOT EXISTS Media (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,
            title TEXT NOT NULL,
            type TEXT NOT NULL,
            content TEXT,
            author TEXT,
            ingestion_date TEXT,
            prompt TEXT,
            summary TEXT,
            transcription_model TEXT
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS Keywords (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            keyword TEXT NOT NULL UNIQUE
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS MediaKeywords (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            media_id INTEGER NOT NULL,
            keyword_id INTEGER NOT NULL,
            FOREIGN KEY (media_id) REFERENCES Media(id),
            FOREIGN KEY (keyword_id) REFERENCES Keywords(id)
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS MediaVersion (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            media_id INTEGER NOT NULL,
            version INTEGER NOT NULL,
            prompt TEXT,
            summary TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (media_id) REFERENCES Media(id)
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS MediaModifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            media_id INTEGER NOT NULL,
            prompt TEXT,
            summary TEXT,
            modification_date TEXT,
            FOREIGN KEY (media_id) REFERENCES Media(id)
        )
        ''',
        '''
        CREATE VIRTUAL TABLE IF NOT EXISTS media_fts USING fts5(title, content);
        ''',
        '''
        CREATE VIRTUAL TABLE IF NOT EXISTS keyword_fts USING fts5(keyword);
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_media_title ON Media(title);
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_media_type ON Media(type);
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_media_author ON Media(author);
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_media_ingestion_date ON Media(ingestion_date);
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_keywords_keyword ON Keywords(keyword);
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_mediakeywords_media_id ON MediaKeywords(media_id);
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_mediakeywords_keyword_id ON MediaKeywords(keyword_id);
        ''',
        '''
        CREATE INDEX IF NOT EXISTS idx_media_version_media_id ON MediaVersion(media_id);
        '''
    ]
    for query in table_queries:
        db.execute_query(query)

create_tables()


#######################################################################################################################
# Keyword-related Functions
#

# Function to add a keyword
def add_keyword(keyword: str) -> int:
    keyword = keyword.strip().lower()
    with db.get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute('INSERT OR IGNORE INTO Keywords (keyword) VALUES (?)', (keyword,))
            cursor.execute('SELECT id FROM Keywords WHERE keyword = ?', (keyword,))
            keyword_id = cursor.fetchone()[0]
            cursor.execute('INSERT OR IGNORE INTO keyword_fts (rowid, keyword) VALUES (?, ?)', (keyword_id, keyword))
            logging.info(f"Keyword '{keyword}' added to keyword_fts with ID: {keyword_id}")
            conn.commit()
            return keyword_id
        except sqlite3.IntegrityError as e:
            logging.error(f"Integrity error adding keyword: {e}")
            raise DatabaseError(f"Integrity error adding keyword: {e}")
        except sqlite3.Error as e:
            logging.error(f"Error adding keyword: {e}")
            raise DatabaseError(f"Error adding keyword: {e}")


# Function to delete a keyword
def delete_keyword(keyword: str) -> str:
    keyword = keyword.strip().lower()
    with db.get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute('SELECT id FROM Keywords WHERE keyword = ?', (keyword,))
            keyword_id = cursor.fetchone()
            if keyword_id:
                cursor.execute('DELETE FROM Keywords WHERE keyword = ?', (keyword,))
                cursor.execute('DELETE FROM keyword_fts WHERE rowid = ?', (keyword_id[0],))
                conn.commit()
                return f"Keyword '{keyword}' deleted successfully."
            else:
                return f"Keyword '{keyword}' not found."
        except sqlite3.Error as e:
            raise DatabaseError(f"Error deleting keyword: {e}")



# Function to add media with keywords
def add_media_with_keywords(url, title, media_type, content, keywords, prompt, summary, transcription_model, author, ingestion_date):
    # Set default values for missing fields
    url = url or 'Unknown'
    title = title or 'Untitled'
    media_type = media_type or 'Unknown'
    content = content or 'No content available'
    keywords = keywords or 'default'
    prompt = prompt or 'No prompt available'
    summary = summary or 'No summary available'
    transcription_model = transcription_model or 'Unknown'
    author = author or 'Unknown'
    ingestion_date = ingestion_date or datetime.now().strftime('%Y-%m-%d')

    # Ensure URL is valid
    if not is_valid_url(url):
        url = 'localhost'

    if media_type not in ['document', 'video', 'article']:
        raise InputError("Invalid media type. Allowed types: document, video, article.")

    if ingestion_date and not is_valid_date(ingestion_date):
        raise InputError("Invalid ingestion date format. Use YYYY-MM-DD.")

    if not ingestion_date:
        ingestion_date = datetime.now().strftime('%Y-%m-%d')

    # Split keywords correctly by comma
    keyword_list = [keyword.strip().lower() for keyword in keywords.split(',')]

    logging.info(f"URL: {url}")
    logging.info(f"Title: {title}")
    logging.info(f"Media Type: {media_type}")
    logging.info(f"Keywords: {keywords}")
    logging.info(f"Content: {content}")
    logging.info(f"Prompt: {prompt}")
    logging.info(f"Summary: {summary}")
    logging.info(f"Author: {author}")
    logging.info(f"Ingestion Date: {ingestion_date}")
    logging.info(f"Transcription Model: {transcription_model}")

    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()

            # Initialize keyword_list
            keyword_list = [keyword.strip().lower() for keyword in keywords.split(',')]

            # Check if media already exists
            cursor.execute('SELECT id FROM Media WHERE url = ?', (url,))
            existing_media = cursor.fetchone()

            if existing_media:
                media_id = existing_media[0]
                logger.info(f"Existing media found with ID: {media_id}")

                # Insert new prompt and summary into MediaModifications
                cursor.execute('''
                INSERT INTO MediaModifications (media_id, prompt, summary, modification_date)
                VALUES (?, ?, ?, ?)
                ''', (media_id, prompt, summary, ingestion_date))
                logger.info("New summary and prompt added to MediaModifications")
            else:
                logger.info("New media entry being created")

                # Insert new media item
                cursor.execute('''
                INSERT INTO Media (url, title, type, content, author, ingestion_date, transcription_model)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (url, title, media_type, content, author, ingestion_date, transcription_model))
                media_id = cursor.lastrowid

                # Insert keywords and associate with media item
                for keyword in keyword_list:
                    keyword = keyword.strip().lower()
                    cursor.execute('INSERT OR IGNORE INTO Keywords (keyword) VALUES (?)', (keyword,))
                    cursor.execute('SELECT id FROM Keywords WHERE keyword = ?', (keyword,))
                    keyword_id = cursor.fetchone()[0]
                    cursor.execute('INSERT OR IGNORE INTO MediaKeywords (media_id, keyword_id) VALUES (?, ?)', (media_id, keyword_id))
                cursor.execute('INSERT INTO media_fts (rowid, title, content) VALUES (?, ?, ?)', (media_id, title, content))

                # Also insert the initial prompt and summary into MediaModifications
                cursor.execute('''
                INSERT INTO MediaModifications (media_id, prompt, summary, modification_date)
                VALUES (?, ?, ?, ?)
                ''', (media_id, prompt, summary, ingestion_date))

            conn.commit()

            # Insert initial version of the prompt and summary
            add_media_version(media_id, prompt, summary)

            return f"Media '{title}' added successfully with keywords: {', '.join(keyword_list)}"
    except sqlite3.IntegrityError as e:
        logger.error(f"Integrity Error: {e}")
        raise DatabaseError(f"Integrity error adding media with keywords: {e}")
    except sqlite3.Error as e:
        logger.error(f"SQL Error: {e}")
        raise DatabaseError(f"Error adding media with keywords: {e}")
    except Exception as e:
        logger.error(f"Unexpected Error: {e}")
        raise DatabaseError(f"Unexpected error: {e}")


def fetch_all_keywords() -> List[str]:
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT keyword FROM Keywords')
            keywords = [row[0] for row in cursor.fetchall()]
            return keywords
    except sqlite3.Error as e:
        raise DatabaseError(f"Error fetching keywords: {e}")

def keywords_browser_interface():
    keywords = fetch_all_keywords()
    return gr.Markdown("\n".join(f"- {keyword}" for keyword in keywords))

def display_keywords():
    try:
        keywords = fetch_all_keywords()
        return "\n".join(keywords) if keywords else "No keywords found."
    except DatabaseError as e:
        return str(e)


def export_keywords_to_csv():
    try:
        keywords = fetch_all_keywords()
        if not keywords:
            return None, "No keywords found in the database."

        filename = "keywords.csv"
        with open(filename, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Keyword"])
            for keyword in keywords:
                writer.writerow([keyword])

        return filename, f"Keywords exported to {filename}"
    except Exception as e:
        logger.error(f"Error exporting keywords to CSV: {e}")
        return None, f"Error exporting keywords: {e}"


#
#
#######################################################################################################################




# Function to add a version of a prompt and summary
def add_media_version(media_id: int, prompt: str, summary: str) -> None:
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()

            # Get the current version number
            cursor.execute('SELECT MAX(version) FROM MediaVersion WHERE media_id = ?', (media_id,))
            current_version = cursor.fetchone()[0] or 0

            # Insert the new version
            cursor.execute('''
            INSERT INTO MediaVersion (media_id, version, prompt, summary, created_at)
            VALUES (?, ?, ?, ?, ?)
            ''', (media_id, current_version + 1, prompt, summary, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            conn.commit()
    except sqlite3.Error as e:
        raise DatabaseError(f"Error adding media version: {e}")


# Function to search the database with advanced options, including keyword search and full-text search
def search_db(search_query: str, search_fields: List[str], keywords: str, page: int = 1, results_per_page: int = 10):
    if page < 1:
        raise ValueError("Page number must be 1 or greater.")

    # Prepare keywords by splitting and trimming
    keywords = [keyword.strip().lower() for keyword in keywords.split(',') if keyword.strip()]

    with db.get_connection() as conn:
        cursor = conn.cursor()
        offset = (page - 1) * results_per_page

        # Prepare the search conditions for general fields
        search_conditions = []
        params = []

        for field in search_fields:
            if search_query:  # Ensure there's a search query before adding this condition
                search_conditions.append(f"Media.{field} LIKE ?")
                params.append(f'%{search_query}%')

        # Prepare the conditions for keywords filtering
        keyword_conditions = []
        for keyword in keywords:
            keyword_conditions.append(
                f"EXISTS (SELECT 1 FROM MediaKeywords mk JOIN Keywords k ON mk.keyword_id = k.id WHERE mk.media_id = Media.id AND k.keyword LIKE ?)")
            params.append(f'%{keyword}%')

        # Combine all conditions
        where_clause = " AND ".join(
            search_conditions + keyword_conditions) if search_conditions or keyword_conditions else "1=1"

        # Complete the query
        query = f'''
        SELECT DISTINCT Media.url, Media.title, Media.type, Media.content, Media.author, Media.ingestion_date, Media.prompt, Media.summary
        FROM Media
        WHERE {where_clause}
        LIMIT ? OFFSET ?
        '''
        params.extend([results_per_page, offset])

        cursor.execute(query, params)
        results = cursor.fetchall()

        return results


# Gradio function to handle user input and display results with pagination, with better feedback
def search_and_display(search_query, search_fields, keywords, page):
    results = search_db(search_query, search_fields, keywords, page)

    if isinstance(results, pd.DataFrame):
        # Convert DataFrame to a list of tuples or lists
        processed_results = results.values.tolist()  # This converts DataFrame rows to lists
    elif isinstance(results, list):
        # Ensure that each element in the list is itself a list or tuple (not a dictionary)
        processed_results = [list(item.values()) if isinstance(item, dict) else item for item in results]
    else:
        raise TypeError("Unsupported data type for results")

    return processed_results


def display_details(index, results):
    if index is None or results is None:
        return "Please select a result to view details."

    try:
        # Ensure the index is an integer and access the row properly
        index = int(index)
        if isinstance(results, pd.DataFrame):
            if index >= len(results):
                return "Index out of range. Please select a valid index."
            selected_row = results.iloc[index]
        else:
            # If results is not a DataFrame, but a list (assuming list of dicts)
            selected_row = results[index]
    except ValueError:
        return "Index must be an integer."
    except IndexError:
        return "Index out of range. Please select a valid index."

    # Build HTML output safely
    details_html = f"""
    <h3>{selected_row.get('Title', 'No Title')}</h3>
    <p><strong>URL:</strong> {selected_row.get('URL', 'No URL')}</p>
    <p><strong>Type:</strong> {selected_row.get('Type', 'No Type')}</p>
    <p><strong>Author:</strong> {selected_row.get('Author', 'No Author')}</p>
    <p><strong>Ingestion Date:</strong> {selected_row.get('Ingestion Date', 'No Date')}</p>
    <p><strong>Prompt:</strong> {selected_row.get('Prompt', 'No Prompt')}</p>
    <p><strong>Summary:</strong> {selected_row.get('Summary', 'No Summary')}</p>
    <p><strong>Content:</strong> {selected_row.get('Content', 'No Content')}</p>
    """
    return details_html


def get_details(index, dataframe):
    if index is None or dataframe is None or index >= len(dataframe):
        return "Please select a result to view details."
    row = dataframe.iloc[index]
    details = f"""
    <h3>{row['Title']}</h3>
    <p><strong>URL:</strong> {row['URL']}</p>
    <p><strong>Type:</strong> {row['Type']}</p>
    <p><strong>Author:</strong> {row['Author']}</p>
    <p><strong>Ingestion Date:</strong> {row['Ingestion Date']}</p>
    <p><strong>Prompt:</strong> {row['Prompt']}</p>
    <p><strong>Summary:</strong> {row['Summary']}</p>
    <p><strong>Content:</strong></p>
    <pre>{row['Content']}</pre>
    """
    return details


def format_results(results):
    if not results:
        return pd.DataFrame(columns=['URL', 'Title', 'Type', 'Content', 'Author', 'Ingestion Date', 'Prompt', 'Summary'])

    df = pd.DataFrame(results, columns=['URL', 'Title', 'Type', 'Content', 'Author', 'Ingestion Date', 'Prompt', 'Summary'])
    logging.debug(f"Formatted DataFrame: {df}")

    return df

# Function to export search results to CSV with pagination
def export_to_csv(search_query: str, search_fields: List[str], keyword: str, page: int = 1, results_per_file: int = 1000):
    try:
        results = search_db(search_query, search_fields, keyword, page, results_per_file)
        df = format_results(results)
        filename = f'search_results_page_{page}.csv'
        df.to_csv(filename, index=False)
        return f"Results exported to {filename}"
    except (DatabaseError, InputError) as e:
        return str(e)


# Helper function to validate URL format
def is_valid_url(url: str) -> bool:
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # ...or ipv4
        r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # ...or ipv6
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, url) is not None


# Helper function to validate date format
def is_valid_date(date_string: str) -> bool:
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False

#
#
#######################################################################################################################




#######################################################################################################################
# Functions to manage prompts DB
#

def create_prompts_db():
    conn = sqlite3.connect('prompts.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Prompts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            details TEXT,
            system TEXT,
            user TEXT
        )
    ''')
    conn.commit()
    conn.close()

create_prompts_db()


def add_prompt(name, details, system, user=None):
    try:
        conn = sqlite3.connect('prompts.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO Prompts (name, details, system, user)
            VALUES (?, ?, ?, ?)
        ''', (name, details, system, user))
        conn.commit()
        conn.close()
        return "Prompt added successfully."
    except sqlite3.IntegrityError:
        return "Prompt with this name already exists."
    except sqlite3.Error as e:
        return f"Database error: {e}"

def fetch_prompt_details(name):
    conn = sqlite3.connect('prompts.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT details, system, user
        FROM Prompts
        WHERE name = ?
    ''', (name,))
    result = cursor.fetchone()
    conn.close()
    return result

def list_prompts():
    conn = sqlite3.connect('prompts.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT name
        FROM Prompts
    ''')
    results = cursor.fetchall()
    conn.close()
    return [row[0] for row in results]

def insert_prompt_to_db(title, description, system_prompt, user_prompt):
    result = add_prompt(title, description, system_prompt, user_prompt)
    return result

#
#
#######################################################################################################################






#######################################################################################################################
# Function Definitions
#

######### Words-per-second Chunking #########
def chunk_transcript(transcript: str, chunk_duration: int, words_per_second) -> List[str]:
    words = transcript.split()
    words_per_chunk = chunk_duration * words_per_second
    chunks = [' '.join(words[i:i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)]
    return chunks


# def summarize_chunks(api_name: str, api_key: str, transcript: List[dict], chunk_duration: int,
#                      words_per_second: int) -> str:
#     if api_name not in summarizers:  # See 'summarizers' dict in the main script
#         return f"Unsupported API: {api_name}"
#
#     summarizer = summarizers[api_name]
#     text = extract_text_from_segments(transcript)
#     chunks = chunk_transcript(text, chunk_duration, words_per_second)
#
#     summaries = []
#     for chunk in chunks:
#         if api_name == 'openai':
#             # Ensure the correct model and prompt are passed
#             summaries.append(summarizer(api_key, chunk, custom_prompt))
#         else:
#             summaries.append(summarizer(api_key, chunk))
#
#     return "\n\n".join(summaries)


################## ####################


######### Token-size Chunking ######### FIXME - OpenAI only currently
# This is dirty and shameful and terrible. It should be replaced with a proper implementation.
# anyways lets get to it....
openai_api_key = "Fake_key" # FIXME
client = OpenAI(api_key=openai_api_key)


def get_chat_completion(messages, model='gpt-4-turbo'):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content


# This function chunks a text into smaller pieces based on a maximum token count and a delimiter
def chunk_on_delimiter(input_string: str,
                       max_tokens: int,
                       delimiter: str) -> List[str]:
    chunks = input_string.split(delimiter)
    combined_chunks, _, dropped_chunk_count = combine_chunks_with_no_minimum(
        chunks, max_tokens, chunk_delimiter=delimiter, add_ellipsis_for_overflow=True)
    if dropped_chunk_count > 0:
        print(f"Warning: {dropped_chunk_count} chunks were dropped due to exceeding the token limit.")
    combined_chunks = [f"{chunk}{delimiter}" for chunk in combined_chunks]
    return combined_chunks


# This function combines text chunks into larger blocks without exceeding a specified token count.
#   It returns the combined chunks, their original indices, and the number of dropped chunks due to overflow.
def combine_chunks_with_no_minimum(
        chunks: List[str],
        max_tokens: int,
        chunk_delimiter="\n\n",
        header: Optional[str] = None,
        add_ellipsis_for_overflow=False,
) -> Tuple[List[str], List[int]]:
    dropped_chunk_count = 0
    output = []  # list to hold the final combined chunks
    output_indices = []  # list to hold the indices of the final combined chunks
    candidate = (
        [] if header is None else [header]
    )  # list to hold the current combined chunk candidate
    candidate_indices = []
    for chunk_i, chunk in enumerate(chunks):
        chunk_with_header = [chunk] if header is None else [header, chunk]
        # FIXME MAKE NOT OPENAI SPECIFIC
        if len(openai_tokenize(chunk_delimiter.join(chunk_with_header))) > max_tokens:
            print(f"warning: chunk overflow")
            if (
                    add_ellipsis_for_overflow
                    # FIXME MAKE NOT OPENAI SPECIFIC
                    and len(openai_tokenize(chunk_delimiter.join(candidate + ["..."]))) <= max_tokens
            ):
                candidate.append("...")
                dropped_chunk_count += 1
            continue  # this case would break downstream assumptions
        # estimate token count with the current chunk added
        # FIXME MAKE NOT OPENAI SPECIFIC
        extended_candidate_token_count = len(openai_tokenize(chunk_delimiter.join(candidate + [chunk])))
        # If the token count exceeds max_tokens, add the current candidate to output and start a new candidate
        if extended_candidate_token_count > max_tokens:
            output.append(chunk_delimiter.join(candidate))
            output_indices.append(candidate_indices)
            candidate = chunk_with_header  # re-initialize candidate
            candidate_indices = [chunk_i]
        # otherwise keep extending the candidate
        else:
            candidate.append(chunk)
            candidate_indices.append(chunk_i)
    # add the remaining candidate to output if it's not empty
    if (header is not None and len(candidate) > 1) or (header is None and len(candidate) > 0):
        output.append(chunk_delimiter.join(candidate))
        output_indices.append(candidate_indices)
    return output, output_indices, dropped_chunk_count


def rolling_summarize(text: str,
                      detail: float = 0,
                      model: str = 'gpt-4-turbo',
                      additional_instructions: Optional[str] = None,
                      minimum_chunk_size: Optional[int] = 500,
                      chunk_delimiter: str = ".",
                      summarize_recursively=False,
                      verbose=False):
    """
    Summarizes a given text by splitting it into chunks, each of which is summarized individually.
    The level of detail in the summary can be adjusted, and the process can optionally be made recursive.

    Parameters: - text (str): The text to be summarized. - detail (float, optional): A value between 0 and 1
    indicating the desired level of detail in the summary. 0 leads to a higher level summary, and 1 results in a more
    detailed summary. Defaults to 0. - model (str, optional): The model to use for generating summaries. Defaults to
    'gpt-3.5-turbo'. - additional_instructions (Optional[str], optional): Additional instructions to provide to the
    model for customizing summaries. - minimum_chunk_size (Optional[int], optional): The minimum size for text
    chunks. Defaults to 500. - chunk_delimiter (str, optional): The delimiter used to split the text into chunks.
    Defaults to ".". - summarize_recursively (bool, optional): If True, summaries are generated recursively,
    using previous summaries for context. - verbose (bool, optional): If True, prints detailed information about the
    chunking process.

    Returns:
    - str: The final compiled summary of the text.

    The function first determines the number of chunks by interpolating between a minimum and a maximum chunk count
    based on the `detail` parameter. It then splits the text into chunks and summarizes each chunk. If
    `summarize_recursively` is True, each summary is based on the previous summaries, adding more context to the
    summarization process. The function returns a compiled summary of all chunks.
    """

    # check detail is set correctly
    assert 0 <= detail <= 1

    # interpolate the number of chunks based to get specified level of detail
    max_chunks = len(chunk_on_delimiter(text, minimum_chunk_size, chunk_delimiter))
    min_chunks = 1
    num_chunks = int(min_chunks + detail * (max_chunks - min_chunks))

    # adjust chunk_size based on interpolated number of chunks
    # FIXME MAKE NOT OPENAI SPECIFIC
    document_length = len(openai_tokenize(text))
    chunk_size = max(minimum_chunk_size, document_length // num_chunks)
    text_chunks = chunk_on_delimiter(text, chunk_size, chunk_delimiter)
    if verbose:
        print(f"Splitting the text into {len(text_chunks)} chunks to be summarized.")
        # FIXME MAKE NOT OPENAI SPECIFIC
        print(f"Chunk lengths are {[len(openai_tokenize(x)) for x in text_chunks]}")

    # set system message
    system_message_content = "Rewrite this text in summarized form."
    if additional_instructions is not None:
        system_message_content += f"\n\n{additional_instructions}"

    accumulated_summaries = []
    for chunk in tqdm(text_chunks):
        if summarize_recursively and accumulated_summaries:
            # Creating a structured prompt for recursive summarization
            accumulated_summaries_string = '\n\n'.join(accumulated_summaries)
            user_message_content = f"Previous summaries:\n\n{accumulated_summaries_string}\n\nText to summarize next:\n\n{chunk}"
        else:
            # Directly passing the chunk for summarization without recursive context
            user_message_content = chunk

        # Constructing messages based on whether recursive summarization is applied
        messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": user_message_content}
        ]

        # Assuming this function gets the completion and works as expected
        response = get_chat_completion(messages, model=model)
        accumulated_summaries.append(response)

    # Compile final summary from partial summaries
    global final_summary
    final_summary = '\n\n'.join(accumulated_summaries)

    return final_summary


#######################################


######### Words-per-second Chunking #########
# FIXME - WHole section needs to be re-written
def chunk_transcript(transcript: str, chunk_duration: int, words_per_second) -> List[str]:
    words = transcript.split()
    words_per_chunk = chunk_duration * words_per_second
    chunks = [' '.join(words[i:i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)]
    return chunks


# def summarize_chunks(api_name: str, api_key: str, transcript: List[dict], chunk_duration: int,
#                      words_per_second: int) -> str:
#     if api_name not in summarizers:  # See 'summarizers' dict in the main script
#         return f"Unsupported API: {api_name}"
#
#     if not transcript:
#         logging.error("Empty or None transcript provided to summarize_chunks")
#         return "Error: Empty or None transcript provided"
#
#     text = extract_text_from_segments(transcript)
#     chunks = chunk_transcript(text, chunk_duration, words_per_second)
#
#     custom_prompt = args.custom_prompt
#
#     summaries = []
#     for chunk in chunks:
#         if api_name == 'openai':
#             # Ensure the correct model and prompt are passed
#             summaries.append(summarize_with_openai(api_key, chunk, custom_prompt))
#         elif api_name == 'anthropic':
#             summaries.append(summarize_with_cohere(api_key, chunk, anthropic_model, custom_prompt))
#         elif api_name == 'cohere':
#             summaries.append(summarize_with_claude(api_key, chunk, cohere_model, custom_prompt))
#         elif api_name == 'groq':
#             summaries.append(summarize_with_groq(api_key, chunk, groq_model, custom_prompt))
#         elif api_name == 'llama':
#             summaries.append(summarize_with_llama(llama_api_IP, chunk, api_key, custom_prompt))
#         elif api_name == 'kobold':
#             summaries.append(summarize_with_kobold(kobold_api_IP, chunk, api_key, custom_prompt))
#         elif api_name == 'ooba':
#             summaries.append(summarize_with_oobabooga(ooba_api_IP, chunk, api_key, custom_prompt))
#         elif api_name == 'tabbyapi':
#             summaries.append(summarize_with_vllm(api_key, tabby_api_IP, chunk, summarize.llm_model, custom_prompt))
#         elif api_name == 'local-llm':
#             summaries.append(summarize_with_local_llm(chunk, custom_prompt))
#         else:
#             return f"Unsupported API: {api_name}"
#
#     return "\n\n".join(summaries)

# FIXME - WHole section needs to be re-written
def summarize_with_detail_openai(text, detail, verbose=False):
    summary_with_detail_variable = rolling_summarize(text, detail=detail, verbose=True)
    print(len(openai_tokenize(summary_with_detail_variable)))
    return summary_with_detail_variable


def summarize_with_detail_recursive_openai(text, detail, verbose=False):
    summary_with_recursive_summarization = rolling_summarize(text, detail=detail, summarize_recursively=True)
    print(summary_with_recursive_summarization)

#
#
#################################################################################



# Read configuration from file
config = configparser.ConfigParser()
config.read('../config.txt')

# Local-Models
kobold_api_IP = config.get('Local-API', 'kobold_api_IP', fallback='http://127.0.0.1:5000/api/v1/generate')
kobold_api_key = config.get('Local-API', 'kobold_api_key', fallback='')

llama_api_IP = config.get('Local-API', 'llama_api_IP', fallback='http://127.0.0.1:8080/v1/chat/completions')
llama_api_key = config.get('Local-API', 'llama_api_key', fallback='')

ooba_api_IP = config.get('Local-API', 'ooba_api_IP', fallback='http://127.0.0.1:5000/v1/chat/completions')
ooba_api_key = config.get('Local-API', 'ooba_api_key', fallback='')

tabby_api_IP = config.get('Local-API', 'tabby_api_IP', fallback='http://127.0.0.1:5000/api/v1/generate')
tabby_api_key = config.get('Local-API', 'tabby_api_key', fallback=None)

vllm_api_url = config.get('Local-API', 'vllm_api_IP', fallback='http://127.0.0.1:500/api/v1/chat/completions')
vllm_api_key = config.get('Local-API', 'vllm_api_key', fallback=None)

#######################################################################################################################
# Function Definitions
#

def summarize_with_local_llm(file_path, custom_prompt_arg):
    try:
        logging.debug("Local LLM: Loading json data for summarization")
        with open(file_path, 'r') as file:
            segments = json.load(file)

        logging.debug("Local LLM: Extracting text from the segments")
        text = extract_text_from_segments(segments)

        headers = {
            'Content-Type': 'application/json'
        }

        logging.debug("Local LLM: Preparing data + prompt for submittal")
        local_llm_prompt = f"{text} \n\n\n\n{custom_prompt_arg}"
        data = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional summarizer."
                },
                {
                    "role": "user",
                    "content": local_llm_prompt
                }
            ],
            "max_tokens": 28000,  # Adjust tokens as needed
        }
        logging.debug("Local LLM: Posting request")
        response = requests.post('http://127.0.0.1:8080/v1/chat/completions', headers=headers, json=data)

        if response.status_code == 200:
            response_data = response.json()
            if 'choices' in response_data and len(response_data['choices']) > 0:
                summary = response_data['choices'][0]['message']['content'].strip()
                logging.debug("Local LLM: Summarization successful")
                print("Local LLM: Summarization successful.")
                return summary
            else:
                logging.warning("Local LLM: Summary not found in the response data")
                return "Local LLM: Summary not available"
        else:
            logging.debug("Local LLM: Summarization failed")
            print("Local LLM: Failed to process summary:", response.text)
            return "Local LLM: Failed to process summary"
    except Exception as e:
        logging.debug("Local LLM: Error in processing: %s", str(e))
        print("Error occurred while processing summary with Local LLM:", str(e))
        return "Local LLM: Error occurred while processing summary"

def summarize_with_llama(api_url, file_path, token, custom_prompt):
    try:
        logging.debug("llama: Loading JSON data")
        with open(file_path, 'r') as file:
            segments = json.load(file)

        logging.debug(f"llama: Extracting text from segments file")
        text = extract_text_from_segments(segments)  # Define this function to extract text properly

        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
        }
        if len(token) > 5:
            headers['Authorization'] = f'Bearer {token}'

        llama_prompt = f"{text} \n\n\n\n{custom_prompt}"
        logging.debug("llama: Prompt being sent is {llama_prompt}")

        data = {
            "prompt": llama_prompt
        }

        logging.debug("llama: Submitting request to API endpoint")
        print("llama: Submitting request to API endpoint")
        response = requests.post(api_url, headers=headers, json=data)
        response_data = response.json()
        logging.debug("API Response Data: %s", response_data)

        if response.status_code == 200:
            # if 'X' in response_data:
            logging.debug(response_data)
            summary = response_data['content'].strip()
            logging.debug("llama: Summarization successful")
            print("Summarization successful.")
            return summary
        else:
            logging.error(f"llama: API request failed with status code {response.status_code}: {response.text}")
            return f"llama: API request failed: {response.text}"

    except Exception as e:
        logging.error("llama: Error in processing: %s", str(e))
        return f"llama: Error occurred while processing summary with llama: {str(e)}"


# https://lite.koboldai.net/koboldcpp_api#/api%2Fv1/post_api_v1_generate
def summarize_with_kobold(api_url, file_path, kobold_api_token, custom_prompt):
    try:
        logging.debug("kobold: Loading JSON data")
        with open(file_path, 'r') as file:
            segments = json.load(file)

        logging.debug(f"kobold: Extracting text from segments file")
        text = extract_text_from_segments(segments)

        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
        }

        kobold_prompt = f"{text} \n\n\n\n{custom_prompt}"
        logging.debug("kobold: Prompt being sent is {kobold_prompt}")

        # FIXME
        # Values literally c/p from the api docs....
        data = {
            "max_context_length": 8096,
            "max_length": 4096,
            "prompt": f"{text}\n\n\n\n{custom_prompt}"
        }

        logging.debug("kobold: Submitting request to API endpoint")
        print("kobold: Submitting request to API endpoint")
        response = requests.post(api_url, headers=headers, json=data)
        response_data = response.json()
        logging.debug("kobold: API Response Data: %s", response_data)

        if response.status_code == 200:
            if 'results' in response_data and len(response_data['results']) > 0:
                summary = response_data['results'][0]['text'].strip()
                logging.debug("kobold: Summarization successful")
                print("Summarization successful.")
                save_summary_to_file(summary, file_path)  # Save the summary to a file
                return summary
            else:
                logging.error("Expected data not found in API response.")
                return "Expected data not found in API response."
        else:
            logging.error(f"kobold: API request failed with status code {response.status_code}: {response.text}")
            return f"kobold: API request failed: {response.text}"

    except Exception as e:
        logging.error("kobold: Error in processing: %s", str(e))
        return f"kobold: Error occurred while processing summary with kobold: {str(e)}"


# https://github.com/oobabooga/text-generation-webui/wiki/12-%E2%80%90-OpenAI-API
def summarize_with_oobabooga(api_url, file_path, ooba_api_token, custom_prompt):
    try:
        logging.debug("ooba: Loading JSON data")
        with open(file_path, 'r') as file:
            segments = json.load(file)

        logging.debug(f"ooba: Extracting text from segments file\n\n\n")
        text = extract_text_from_segments(segments)
        logging.debug(f"ooba: Finished extracting text from segments file")

        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
        }

        # prompt_text = "I like to eat cake and bake cakes. I am a baker. I work in a French bakery baking cakes. It
        # is a fun job. I have been baking cakes for ten years. I also bake lots of other baked goods, but cakes are
        # my favorite." prompt_text += f"\n\n{text}"  # Uncomment this line if you want to include the text variable
        ooba_prompt = f"{text}" + f"\n\n\n\n{custom_prompt}"
        logging.debug("ooba: Prompt being sent is {ooba_prompt}")

        data = {
            "mode": "chat",
            "character": "Example",
            "messages": [{"role": "user", "content": ooba_prompt}]
        }

        logging.debug("ooba: Submitting request to API endpoint")
        print("ooba: Submitting request to API endpoint")
        response = requests.post(api_url, headers=headers, json=data, verify=False)
        logging.debug("ooba: API Response Data: %s", response)

        if response.status_code == 200:
            response_data = response.json()
            summary = response.json()['choices'][0]['message']['content']
            logging.debug("ooba: Summarization successful")
            print("Summarization successful.")
            return summary
        else:
            logging.error(f"oobabooga: API request failed with status code {response.status_code}: {response.text}")
            return f"ooba: API request failed with status code {response.status_code}: {response.text}"

    except Exception as e:
        logging.error("ooba: Error in processing: %s", str(e))
        return f"ooba: Error occurred while processing summary with oobabooga: {str(e)}"


# FIXME - https://docs.vllm.ai/en/latest/getting_started/quickstart.html .... Great docs.
def summarize_with_vllm(vllm_api_url, vllm_api_key_function_arg, llm_model, text, vllm_custom_prompt_function_arg):
    vllm_client = OpenAI(
        base_url=vllm_api_url,
        api_key=vllm_api_key_function_arg
    )

    custom_prompt = vllm_custom_prompt_function_arg

    completion = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": "You are a professional summarizer."},
            {"role": "user", "content": f"{text} \n\n\n\n{custom_prompt}"}
        ]
    )
    vllm_summary = completion.choices[0].message.content
    return vllm_summary


# FIXME - Install is more trouble than care to deal with right now.
def summarize_with_tabbyapi(tabby_api_key, tabby_api_IP, text, tabby_model, custom_prompt):
    model = tabby_model
    headers = {
        'Authorization': f'Bearer {tabby_api_key}',
        'Content-Type': 'application/json'
    }
    data = {
        'text': text,
        'model': 'tabby'  # Specify the model if needed
    }
    try:
        response = requests.post('https://api.tabbyapi.com/summarize', headers=headers, json=data)
        response.raise_for_status()
        summary = response.json().get('summary', '')
        return summary
    except requests.exceptions.RequestException as e:
        logger.error(f"Error summarizing with TabbyAPI: {e}")
        return "Error summarizing with TabbyAPI."


def save_summary_to_file(summary, file_path):
    logging.debug("Now saving summary to file...")
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    summary_file_path = os.path.join(os.path.dirname(file_path), base_name + '_summary.txt')
    os.makedirs(os.path.dirname(summary_file_path), exist_ok=True)
    logging.debug("Opening summary file for writing, *segments.json with *_summary.txt")
    with open(summary_file_path, 'w') as file:
        file.write(summary)
    logging.info(f"Summary saved to file: {summary_file_path}")

# From Video_DL_Ingestion_Lib.py
# def save_summary_to_file(summary: str, file_path: str):
#     """Save summary to a JSON file."""
#     summary_data = {'summary': summary, 'generated_at': datetime.now().isoformat()}
#     with open(file_path, 'w') as file:
#         json.dump(summary_data, file, indent=4)


#
#
#######################################################################################################################






#######################################################################################################################
# Function Definitions
#

# Download latest llamafile from Github
    # Example usage
    #repo = "Mozilla-Ocho/llamafile"
    #asset_name_prefix = "llamafile-"
    #output_filename = "llamafile"
    #download_latest_llamafile(repo, asset_name_prefix, output_filename)
def download_latest_llamafile(repo, asset_name_prefix, output_filename):
    # Check if the file already exists
    print("Checking for and downloading Llamafile it it doesn't already exist...")
    if os.path.exists(output_filename):
        print("Llamafile already exists. Skipping download.")
        logging.debug(f"{output_filename} already exists. Skipping download.")
        llamafile_exists = True
    else:
        llamafile_exists = False

    if llamafile_exists == True:
        pass
    else:
        # Get the latest release information
        latest_release_url = f"https://api.github.com/repos/{repo}/releases/latest"
        response = requests.get(latest_release_url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch latest release info: {response.status_code}")

        latest_release_data = response.json()
        tag_name = latest_release_data['tag_name']

        # Get the release details using the tag name
        release_details_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag_name}"
        response = requests.get(release_details_url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch release details for tag {tag_name}: {response.status_code}")

        release_data = response.json()
        assets = release_data.get('assets', [])

        # Find the asset with the specified prefix
        asset_url = None
        for asset in assets:
            if re.match(f"{asset_name_prefix}.*", asset['name']):
                asset_url = asset['browser_download_url']
                break

        if not asset_url:
            raise Exception(f"No asset found with prefix {asset_name_prefix}")

        # Download the asset
        response = requests.get(asset_url)
        if response.status_code != 200:
            raise Exception(f"Failed to download asset: {response.status_code}")

        print("Llamafile downloaded successfully.")
        logging.debug("Main: Llamafile downloaded successfully.")

        # Save the file
        with open(output_filename, 'wb') as file:
            file.write(response.content)

        logging.debug(f"Downloaded {output_filename} from {asset_url}")
        print(f"Downloaded {output_filename} from {asset_url}")

    # Check to see if the LLM already exists, and if not, download the LLM
    print("Checking for and downloading LLM from Huggingface if needed...")
    logging.debug("Main: Checking and downloading LLM from Huggingface if needed...")
    mistral_7b_instruct_v0_2_q8_0_llamafile = "mistral-7b-instruct-v0.2.Q8_0.llamafile"
    Samantha_Mistral_Instruct_7B_Bulleted_Notes_Q8 = "samantha-mistral-instruct-7b-bulleted-notes.Q8_0.gguf"
    Phi_3_mini_128k_instruct_Q8_0_gguf = "Phi-3-mini-128k-instruct-Q8_0.gguf"
    if os.path.exists(mistral_7b_instruct_v0_2_q8_0_llamafile):
        llamafile_llm_url = "https://huggingface.co/Mozilla/Mistral-7B-Instruct-v0.2-llamafile/resolve/main/mistral-7b-instruct-v0.2.Q8_0.llamafile?download=true"
    elif os.path.exists(Samantha_Mistral_Instruct_7B_Bulleted_Notes_Q8):
        llamafile_llm_url = "https://huggingface.co/Mozilla/Mistral-7B-Instruct-v0.2-llamafile/resolve/main/mistral-7b-instruct-v0.2.Q8_0.llamafile?download=true"
        print("Model is already downloaded. Skipping download.")
        pass
    else:
        logging.debug("Main: Checking and downloading LLM from Huggingface if needed...")
        print("Downloading LLM from Huggingface...")
        time.sleep(1)
        print("Gonna be a bit...")
        time.sleep(1)
        print("Like seriously, an 8GB file...")
        time.sleep(2)
        dl_check = input("Final chance to back out, hit 'N'/'n' to cancel, or 'Y'/'y' to continue: ")
        if dl_check == "N" or dl_check == "n":
            exit()
        else:
            print("Downloading LLM from Huggingface...")
            # Establish hash values for LLM models
            mistral_7b_instruct_v0_2_q8_gguf_sha256 = "f326f5f4f137f3ad30f8c9cc21d4d39e54476583e8306ee2931d5a022cb85b06"
            samantha_mistral_instruct_7b_bulleted_notes_q8_0_gguf_sha256 = "6334c1ab56c565afd86535271fab52b03e67a5e31376946bce7bf5c144e847e4"
            mistral_7b_instruct_v0_2_q8_0_llamafile_sha256 = "1ee6114517d2f770425c880e5abc443da36b193c82abec8e2885dd7ce3b9bfa6"
            global llm_choice

            # FIXME - llm_choice
            llm_choice = 2
            llm_choice = input("Which LLM model would you like to download? 1. Mistral-7B-Instruct-v0.2-GGUF or 2. Samantha-Mistral-Instruct-7B-Bulleted-Notes) (plain or 'custom') or MS Flavor: Phi-3-mini-128k-instruct-Q8_0.gguf  \n\n\tPress '1' or '2' or '3' to specify: ")
            while llm_choice != "1" and llm_choice != "2" and llm_choice != "3":
                print("Invalid choice. Please try again.")
            if llm_choice == "1":
                llm_download_model = "Mistral-7B-Instruct-v0.2-Q8.llamafile"
                mistral_7b_instruct_v0_2_q8_0_llamafile_sha256 = "1ee6114517d2f770425c880e5abc443da36b193c82abec8e2885dd7ce3b9bfa6"
                llm_download_model_hash = mistral_7b_instruct_v0_2_q8_0_llamafile_sha256
                llamafile_llm_url = "https://huggingface.co/Mozilla/Mistral-7B-Instruct-v0.2-llamafile/resolve/main/mistral-7b-instruct-v0.2.Q8_0.llamafile?download=true"
                llamafile_llm_output_filename = "mistral-7b-instruct-v0.2.Q8_0.llamafile"
                download_file(llamafile_llm_url, llamafile_llm_output_filename, llm_download_model_hash)
            elif llm_choice == "2":
                llm_download_model = "Samantha-Mistral-Instruct-7B-Bulleted-Notes-Q8.gguf"
                samantha_mistral_instruct_7b_bulleted_notes_q8_0_gguf_sha256 = "6334c1ab56c565afd86535271fab52b03e67a5e31376946bce7bf5c144e847e4"
                llm_download_model_hash = samantha_mistral_instruct_7b_bulleted_notes_q8_0_gguf_sha256
                llamafile_llm_output_filename = "samantha-mistral-instruct-7b-bulleted-notes.Q8_0.gguf"
                llamafile_llm_url = "https://huggingface.co/cognitivetech/samantha-mistral-instruct-7b-bulleted-notes-GGUF/resolve/main/samantha-mistral-instruct-7b-bulleted-notes.Q8_0.gguf?download=true"
                download_file(llamafile_llm_url, llamafile_llm_output_filename, llm_download_model_hash)
            elif llm_choice == "3":
                llm_download_model = "Phi-3-mini-128k-instruct-Q8_0.gguf"
                Phi_3_mini_128k_instruct_Q8_0_gguf_sha256 = "6817b66d1c3c59ab06822e9732f0e594eea44e64cae2110906eac9d17f75d193"
                llm_download_model_hash = Phi_3_mini_128k_instruct_Q8_0_gguf_sha256
                llamafile_llm_output_filename = "Phi-3-mini-128k-instruct-Q8_0.gguf"
                llamafile_llm_url = "https://huggingface.co/gaianet/Phi-3-mini-128k-instruct-GGUF/resolve/main/Phi-3-mini-128k-instruct-Q8_0.gguf?download=true"
                download_file(llamafile_llm_url, llamafile_llm_output_filename, llm_download_model_hash)
            elif llm_choice == "4": # FIXME - and meta_Llama_3_8B_Instruct_Q8_0_llamafile_exists == False:
                meta_Llama_3_8B_Instruct_Q8_0_llamafile_sha256 = "406868a97f02f57183716c7e4441d427f223fdbc7fa42964ef10c4d60dd8ed37"
                llm_download_model_hash = meta_Llama_3_8B_Instruct_Q8_0_llamafile_sha256
                llamafile_llm_output_filename = " Meta-Llama-3-8B-Instruct.Q8_0.llamafile"
                llamafile_llm_url = "https://huggingface.co/Mozilla/Meta-Llama-3-8B-Instruct-llamafile/resolve/main/Meta-Llama-3-8B-Instruct.Q8_0.llamafile?download=true"
            else:
                print("Invalid choice. Please try again.")
    return output_filename


def download_file(url, dest_path, expected_checksum=None, max_retries=3, delay=5):
    temp_path = dest_path + '.tmp'

    for attempt in range(max_retries):
        try:
            # Check if a partial download exists and get its size
            resume_header = {}
            if os.path.exists(temp_path):
                resume_header = {'Range': f'bytes={os.path.getsize(temp_path)}-'}

            response = requests.get(url, stream=True, headers=resume_header)
            response.raise_for_status()

            # Get the total file size from headers
            total_size = int(response.headers.get('content-length', 0))
            initial_pos = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0

            mode = 'ab' if 'Range' in response.headers else 'wb'
            with open(temp_path, mode) as temp_file, tqdm(
                total=total_size, unit='B', unit_scale=True, desc=dest_path, initial=initial_pos, ascii=True
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive new chunks
                        temp_file.write(chunk)
                        pbar.update(len(chunk))

            # Verify the checksum if provided
            if expected_checksum:
                if not verify_checksum(temp_path, expected_checksum):
                    os.remove(temp_path)
                    raise ValueError("Downloaded file's checksum does not match the expected checksum")

            # Move the file to the final destination
            os.rename(temp_path, dest_path)
            print("Download complete and verified!")
            return dest_path

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Download failed.")
                raise

# FIXME / IMPLEMENT FULLY
# File download verification
#mistral_7b_llamafile_instruct_v02_q8_url = "https://huggingface.co/Mozilla/Mistral-7B-Instruct-v0.2-llamafile/resolve/main/mistral-7b-instruct-v0.2.Q8_0.llamafile?download=true"
#global mistral_7b_instruct_v0_2_q8_0_llamafile_sha256
#mistral_7b_instruct_v0_2_q8_0_llamafile_sha256 = "1ee6114517d2f770425c880e5abc443da36b193c82abec8e2885dd7ce3b9bfa6"

#mistral_7b_v02_instruct_model_q8_gguf_url = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q8_0.gguf?download=true"
#global mistral_7b_instruct_v0_2_q8_gguf_sha256
#mistral_7b_instruct_v0_2_q8_gguf_sha256 = "f326f5f4f137f3ad30f8c9cc21d4d39e54476583e8306ee2931d5a022cb85b06"

#samantha_instruct_model_q8_gguf_url = "https://huggingface.co/cognitivetech/samantha-mistral-instruct-7b_bulleted-notes_GGUF/resolve/main/samantha-mistral-instruct-7b-bulleted-notes.Q8_0.gguf?download=true"
#global samantha_mistral_instruct_7b_bulleted_notes_q8_0_gguf_sha256
#samantha_mistral_instruct_7b_bulleted_notes_q8_0_gguf_sha256 = "6334c1ab56c565afd86535271fab52b03e67a5e31376946bce7bf5c144e847e4"



def verify_checksum(file_path, expected_checksum):
    sha256_hash = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for byte_block in iter(lambda: f.read(4096), b''):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest() == expected_checksum

process = None
# Function to close out llamafile process on script exit.
def cleanup_process():
    global process
    if process is not None:
        process.kill()
        logging.debug("Main: Terminated the external process")


def signal_handler(sig, frame):
    logging.info('Signal handler called with signal: %s', sig)
    cleanup_process()
    sys.exit(0)


# FIXME - Add callout to gradio UI
def local_llm_function():
    global process
    repo = "Mozilla-Ocho/llamafile"
    asset_name_prefix = "llamafile-"
    useros = os.name
    if useros == "nt":
        output_filename = "llamafile.exe"
    else:
        output_filename = "llamafile"
    print(
        "WARNING - Checking for existence of llamafile and HuggingFace model, downloading if needed...This could be a while")
    print("WARNING - and I mean a while. We're talking an 8 Gigabyte model here...")
    print("WARNING - Hope you're comfy. Or it's already downloaded.")
    time.sleep(6)
    logging.debug("Main: Checking and downloading Llamafile from Github if needed...")
    llamafile_path = download_latest_llamafile(repo, asset_name_prefix, output_filename)
    logging.debug("Main: Llamafile downloaded successfully.")

    # FIXME - llm_choice
    global llm_choice
    llm_choice = 1
    # Launch the llamafile in an external process with the specified argument
    if llm_choice == 1:
        arguments = ["--ctx-size", "8192 ", " -m", "mistral-7b-instruct-v0.2.Q8_0.llamafile"]
    elif llm_choice == 2:
        arguments = ["--ctx-size", "8192 ", " -m", "samantha-mistral-instruct-7b-bulleted-notes.Q8_0.gguf"]
    elif llm_choice == 3:
        arguments = ["--ctx-size", "8192 ", " -m", "Phi-3-mini-128k-instruct-Q8_0.gguf"]
    elif llm_choice == 4:
        arguments = ["--ctx-size", "8192 ", " -m", "llama-3"] # FIXME

    try:
        logging.info("Main: Launching the LLM (llamafile) in an external terminal window...")
        if useros == "nt":
            launch_in_new_terminal_windows(llamafile_path, arguments)
        elif useros == "posix":
            launch_in_new_terminal_linux(llamafile_path, arguments)
        else:
            launch_in_new_terminal_mac(llamafile_path, arguments)
        # FIXME - pid doesn't exist in this context
        #logging.info(f"Main: Launched the {llamafile_path} with PID {process.pid}")
        atexit.register(cleanup_process, process)
    except Exception as e:
        logging.error(f"Failed to launch the process: {e}")
        print(f"Failed to launch the process: {e}")


def local_llm_gui_function(prompt, temperature, top_k, top_p, min_p, stream, stop, typical_p, repeat_penalty, repeat_last_n,
                       penalize_nl, presence_penalty, frequency_penalty, penalty_prompt, ignore_eos, system_prompt):
    repo = "Mozilla-Ocho/llamafile"
    asset_name_prefix = "llamafile-"
    useros = os.name
    if useros == "nt":
        output_filename = "llamafile.exe"
    else:
        output_filename = "llamafile"
    print(
        "WARNING - Checking for existence of llamafile and HuggingFace model, downloading if needed...This could be a while")
    print("WARNING - and I mean a while. We're talking an 8 Gigabyte model here...")
    print("WARNING - Hope you're comfy. Or it's already downloaded.")
    time.sleep(6)
    logging.debug("Main: Checking and downloading Llamafile from Github if needed...")
    llamafile_path = download_latest_llamafile(repo, asset_name_prefix, output_filename)
    logging.debug("Main: Llamafile downloaded successfully.")

    # FIXME - llm_choice
    global llm_choice
    llm_choice = 1
    # Launch the llamafile in an external process with the specified argument
    if llm_choice == 1:
        arguments = ["--ctx-size", "8192 ", " -m", "mistral-7b-instruct-v0.2.Q8_0.llamafile"]
    elif llm_choice == 2:
        arguments = ["--ctx-size", "8192 ", " -m", "samantha-mistral-instruct-7b-bulleted-notes.Q8_0.gguf"]
    elif llm_choice == 3:
        arguments = ["--ctx-size", "8192 ", " -m", "Phi-3-mini-128k-instruct-Q8_0.gguf"]
    elif llm_choice == 4:
        arguments = ["--ctx-size", "8192 ", " -m", "llama-3"] # FIXME

    try:
        logging.info("Main: Launching the LLM (llamafile) in an external terminal window...")
        if useros == "nt":
            launch_in_new_terminal_windows(llamafile_path, arguments)
        elif useros == "posix":
            launch_in_new_terminal_linux(llamafile_path, arguments)
        else:
            launch_in_new_terminal_mac(llamafile_path, arguments)
        # FIXME - pid doesn't exist in this context
        #logging.info(f"Main: Launched the {llamafile_path} with PID {process.pid}")
        atexit.register(cleanup_process, process)
    except Exception as e:
        logging.error(f"Failed to launch the process: {e}")
        print(f"Failed to launch the process: {e}")




# Launch the executable in a new terminal window # FIXME - really should figure out a cleaner way of doing this...
def launch_in_new_terminal_windows(executable, args):
    command = f'start cmd /k "{executable} {" ".join(args)}"'
    subprocess.Popen(command, shell=True)


# FIXME
def launch_in_new_terminal_linux(executable, args):
    command = f'gnome-terminal -- {executable} {" ".join(args)}'
    subprocess.Popen(command, shell=True)


# FIXME
def launch_in_new_terminal_mac(executable, args):
    command = f'open -a Terminal.app {executable} {" ".join(args)}'
    subprocess.Popen(command, shell=True)


#######################################################################################################################
# Function Definitions
#

def read_paths_from_file(file_path):
    """ Reads a file containing URLs or local file paths and returns them as a list. """
    paths = []  # Initialize paths as an empty list
    with open(file_path, 'r') as file:
        paths = [line.strip() for line in file]
    return paths


def process_path(path):
    """ Decides whether the path is a URL or a local file and processes accordingly. """
    if path.startswith('http'):
        logging.debug("file is a URL")
        # For YouTube URLs, modify to download and extract info
        return get_youtube(path)
    elif os.path.exists(path):
        logging.debug("File is a path")
        # For local files, define a function to handle them
        return process_local_file(path)
    else:
        logging.error(f"Path does not exist: {path}")
        return None


# FIXME
def process_local_file(file_path):
    logging.info(f"Processing local file: {file_path}")
    title = normalize_title(os.path.splitext(os.path.basename(file_path))[0])
    info_dict = {'title': title}
    logging.debug(f"Creating {title} directory...")
    download_path = create_download_directory(title)
    logging.debug(f"Converting '{title}' to an audio file (wav).")
    audio_file = convert_to_wav(file_path)  # Assumes input files are videos needing audio extraction
    logging.debug(f"'{title}' successfully converted to an audio file (wav).")
    return download_path, info_dict, audio_file


def read_paths_from_file(file_path: str) -> List[str]:
    """Read paths from a text file."""
    with open(file_path, 'r') as file:
        paths = file.readlines()
    return [path.strip() for path in paths]


#
#
#######################################################################################################################

#
#
#######################################################################################################################


#######################################################################################################################
# Chunking-related Techniques & Functions
#
#
# FIXME
#
#
#######################################################################################################################


#######################################################################################################################
# Tokenization-related Functions
#
#

# FIXME

#
#
#######################################################################################################################


#######################################################################################################################
# Website-related Techniques & Functions
#
#

#
#
#######################################################################################################################


#######################################################################################################################
# Summarizers
#
# Function List
# 1. extract_text_from_segments(segments: List[Dict]) -> str
# 2. summarize_with_openai(api_key, file_path, custom_prompt_arg)
# 3. summarize_with_claude(api_key, file_path, model, custom_prompt_arg, max_retries=3, retry_delay=5)
# 4. summarize_with_cohere(api_key, file_path, model, custom_prompt_arg)
# 5. summarize_with_groq(api_key, file_path, model, custom_prompt_arg)
#
#################################
# Local Summarization
#
# Function List
#
# 1. summarize_with_local_llm(file_path, custom_prompt_arg)
# 2. summarize_with_llama(api_url, file_path, token, custom_prompt)
# 3. summarize_with_kobold(api_url, file_path, kobold_api_token, custom_prompt)
# 4. summarize_with_oobabooga(api_url, file_path, ooba_api_token, custom_prompt)
# 5. summarize_with_vllm(vllm_api_url, vllm_api_key_function_arg, llm_model, text, vllm_custom_prompt_function_arg)
# 6. summarize_with_tabbyapi(tabby_api_key, tabby_api_IP, text, tabby_model, custom_prompt)
# 7. save_summary_to_file(summary, file_path)
#
#######################################################################################################################


#######################################################################################################################
# Summarization with Detail
#

# FIXME - see 'Old_Chunking_Lib.py'

#
#
#######################################################################################################################


#######################################################################################################################
# Gradio UI
#
#######################################################################################################################
# Function Definitions
#

# Only to be used when configured with Gradio for HF Space


def format_transcription(transcription_result_arg):
    if transcription_result_arg:
        json_data = transcription_result_arg['transcription']
        return json.dumps(json_data, indent=2)
    else:
        return ""


def format_file_path(file_path, fallback_path=None):
    if file_path and os.path.exists(file_path):
        logging.debug(f"File exists: {file_path}")
        return file_path
    elif fallback_path and os.path.exists(fallback_path):
        logging.debug(f"File does not exist: {file_path}. Returning fallback path: {fallback_path}")
        return fallback_path
    else:
        logging.debug(f"File does not exist: {file_path}. No fallback path available.")
        return None


def search_media(query, fields, keyword, page):
    try:
        results = search_and_display(query, fields, keyword, page)
        return results
    except Exception as e:
        logger.error(f"Error searching media: {e}")
        return str(e)


# FIXME - code for the 're-prompt' functionality
#- Change to use 'check_api()' function - also, create 'check_api()' function
# def ask_question(transcription, question, api_name, api_key):
#     if not question.strip():
#         return "Please enter a question."
#
#         prompt = f"""Transcription:\n{transcription}
#
#         Given the above transcription, please answer the following:\n\n{question}"""
#
#         # FIXME - Refactor main API checks so they're their own function - api_check()
#         # Call api_check() function here
#
#         if api_name.lower() == "openai":
#             openai_api_key = api_key if api_key else config.get('API', 'openai_api_key', fallback=None)
#             headers = {
#                 'Authorization': f'Bearer {openai_api_key}',
#                 'Content-Type': 'application/json'
#             }
#             if openai_model:
#                 pass
#             else:
#                 openai_model = 'gpt-4-turbo'
#             data = {
#                 "model": openai_model,
#                 "messages": [
#                     {
#                         "role": "system",
#                         "content": "You are a helpful assistant that answers questions based on the given "
#                                    "transcription and summary."
#                     },
#                     {
#                         "role": "user",
#                         "content": prompt
#                     }
#                 ],
#                 "max_tokens": 150000,
#                 "temperature": 0.1
#             }
#             response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
#
#         if response.status_code == 200:
#             answer = response.json()['choices'][0]['message']['content'].strip()
#             return answer
#         else:
#             return "Failed to process the question."
#     else:
#         return "Question answering is currently only supported with the OpenAI API."


# For the above 'ask_question()' function, the following APIs are supported:
# summarizers: Dict[str, Callable[[str, str], str]] = {
#     'tabbyapi': summarize_with_tabbyapi,
#     'openai': summarize_with_openai,
#     'anthropic': summarize_with_claude,
#     'cohere': summarize_with_cohere,
#     'groq': summarize_with_groq,
#     'llama': summarize_with_llama,
#     'kobold': summarize_with_kobold,
#     'oobabooga': summarize_with_oobabooga,
#     'local-llm': summarize_with_local_llm,
#     'huggingface': summarize_with_huggingface,
#     'openrouter': summarize_with_openrouter
#     # Add more APIs here as needed
# }

#########################################################################


# FIXME - Move to 'Web_UI_Lib.py'
# Gradio Search Function-related stuff
def display_details(media_id):
    if media_id:
        details = display_item_details(media_id)
        details_html = ""
        for detail in details:
            details_html += f"<h4>Prompt:</h4><p>{detail[0]}</p>"
            details_html += f"<h4>Summary:</h4><p>{detail[1]}</p>"
            details_html += f"<h4>Transcription:</h4><pre>{detail[2]}</pre><hr>"
        return details_html
    return "No details available."

def fetch_items_by_title_or_url(search_query: str, search_type: str):
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            if search_type == 'Title':
                cursor.execute("SELECT id, title, url FROM Media WHERE title LIKE ?", (f'%{search_query}%',))
            elif search_type == 'URL':
                cursor.execute("SELECT id, title, url FROM Media WHERE url LIKE ?", (f'%{search_query}%',))
            results = cursor.fetchall()
            return results
    except sqlite3.Error as e:
        raise DatabaseError(f"Error fetching items by {search_type}: {e}")


def fetch_items_by_keyword(search_query: str):
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT m.id, m.title, m.url
                FROM Media m
                JOIN MediaKeywords mk ON m.id = mk.media_id
                JOIN Keywords k ON mk.keyword_id = k.id
                WHERE k.keyword LIKE ?
            """, (f'%{search_query}%',))
            results = cursor.fetchall()
            return results
    except sqlite3.Error as e:
        raise DatabaseError(f"Error fetching items by keyword: {e}")

def fetch_items_by_content(search_query: str):
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, title, url FROM Media WHERE content LIKE ?", (f'%{search_query}%',))
            results = cursor.fetchall()
            return results
    except sqlite3.Error as e:
        raise DatabaseError(f"Error fetching items by content: {e}")


def fetch_item_details(media_id: int):
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT prompt, summary FROM MediaModifications WHERE media_id = ?", (media_id,))
            prompt_summary_results = cursor.fetchall()

            cursor.execute("SELECT content FROM Media WHERE id = ?", (media_id,))
            content_result = cursor.fetchone()
            content = content_result[0] if content_result else ""

            return prompt_summary_results, content
    except sqlite3.Error as e:
        raise DatabaseError(f"Error fetching item details: {e}")

def browse_items(search_query, search_type):
    if search_type == 'Keyword':
        results = fetch_items_by_keyword(search_query)
    elif search_type == 'Content':
        results = fetch_items_by_content(search_query)
    else:
        results = fetch_items_by_title_or_url(search_query, search_type)
    return results

def display_item_details(media_id):
    prompt_summary_results, content = fetch_item_details(media_id)
    content_section = f"<h4>Transcription:</h4><pre>{content}</pre><hr>"
    prompt_summary_section = ""
    for prompt, summary in prompt_summary_results:
        prompt_summary_section += f"<h4>Prompt:</h4><p>{prompt}</p>"
        prompt_summary_section += f"<h4>Summary:</h4><p>{summary}</p><hr>"
    return prompt_summary_section, content_section

def update_dropdown(search_query, search_type):
    results = browse_items(search_query, search_type)
    item_options = [f"{item[1]} ({item[2]})" for item in results]
    item_mapping = {f"{item[1]} ({item[2]})": item[0] for item in results}  # Map item display to media ID
    return gr.Dropdown.update(choices=item_options), item_mapping

def get_media_id(selected_item, item_mapping):
    return item_mapping.get(selected_item)

def update_detailed_view(selected_item, item_mapping):
    media_id = get_media_id(selected_item, item_mapping)
    if media_id:
        prompt_summary_html, content_html = display_item_details(media_id)
        return gr.update(value=prompt_summary_html), gr.update(value=content_html)
    return gr.update(value="No details available"), gr.update(value="No details available")

def update_prompt_dropdown():
    prompt_names = list_prompts()
    return gr.update(choices=prompt_names)

def display_prompt_details(selected_prompt):
    if selected_prompt:
        details = fetch_prompt_details(selected_prompt)
        if details:
            details_str = f"<h4>Details:</h4><p>{details[0]}</p>"
            system_str = f"<h4>System:</h4><p>{details[1]}</p>"
            user_str = f"<h4>User:</h4><p>{details[2]}</p>" if details[2] else ""
            return details_str + system_str + user_str
    return "No details available."

def insert_prompt_to_db(title, description, system_prompt, user_prompt):
    try:
        conn = sqlite3.connect('prompts.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO Prompts (name, details, system, user) VALUES (?, ?, ?, ?)",
            (title, description, system_prompt, user_prompt)
        )
        conn.commit()
        conn.close()
        return "Prompt added successfully!"
    except sqlite3.Error as e:
        return f"Error adding prompt: {e}"

def display_search_results(query):
    if not query.strip():
        return "Please enter a search query."

    results = search_prompts(query)

    # Debugging: Print the results to the console to see what is being returned
    print(f"Processed search results for query '{query}': {results}")

    if results:
        result_md = "## Search Results:\n"
        for result in results:
            # Debugging: Print each result to see its format
            print(f"Result item: {result}")

            if len(result) == 2:
                name, details = result
                result_md += f"**Title:** {name}\n\n**Description:** {details}\n\n---\n"
            else:
                result_md += "Error: Unexpected result format.\n\n---\n"
        return result_md
    return "No results found."




#
# End of Gradio Search Function-related stuff
############################################################


# def gradio UI
def launch_ui(demo_mode=False):
    whisper_models = ["small.en", "medium.en", "large"]
    # Set theme value with https://www.gradio.app/guides/theming-guide - 'theme='
    my_theme = gr.Theme.from_hub("gradio/seafoam")
    with gr.Blocks(theme=my_theme) as iface:
        # Tab 1: Audio Transcription + Summarization
        with gr.Tab("Audio Transcription + Summarization"):

            with gr.Row():
                # Light/Dark mode toggle switch
                theme_toggle = gr.Radio(choices=["Light", "Dark"], value="Light",
                                        label="Light/Dark Mode Toggle (Toggle to change UI color scheme)")

                # UI Mode toggle switch
                ui_frontpage_mode_toggle = gr.Radio(choices=["Simple List", "Advanced List"], value="Simple List",
                                                    label="UI Mode Options Toggle(Toggle to show a few/all options)")

                # Add the new toggle switch
                chunk_summarization_toggle = gr.Radio(choices=["Non-Chunked", "Chunked-Summarization"],
                                                      value="Non-Chunked",
                                                      label="Summarization Mode")

            # URL input is always visible
            url_input = gr.Textbox(label="URL (Mandatory) --> Playlist URLs will be stripped and only the linked video"
                          " will be downloaded)", placeholder="Enter the video URL here")
#            url_input = gr.Textbox(label="URL (Mandatory) --> Playlist URLs will be stripped and only the linked video"
#                                         " will be downloaded)", placeholder="Enter the video URL here")

            # Inputs to be shown or hidden
            num_speakers_input = gr.Number(value=2, label="Number of Speakers(Optional - Currently has no effect)",
                                           visible=False)
            whisper_model_input = gr.Dropdown(choices=whisper_models, value="small.en",
                                              label="Whisper Model(This is the ML model used for transcription.)",
                                              visible=False)
            custom_prompt_input = gr.Textbox(
                label="Custom Prompt (Customize your summarization, or ask a question about the video and have it "
                      "answered)\n Does not work against the summary currently.",
                placeholder="Above is the transcript of a video. Please read "
                            "through the transcript carefully. Identify the main topics that are discussed over the "
                            "course of the transcript. Then, summarize the key points about each main topic in a "
                            "concise bullet point. The bullet points should cover the key information conveyed about "
                            "each topic in the video, but should be much shorter than the full transcript. Please "
                            "output your bullet point summary inside <bulletpoints> tags.",
                lines=3, visible=True)
            offset_input = gr.Number(value=0, label="Offset (Seconds into the video to start transcribing at)",
                                     visible=False)
            api_name_input = gr.Dropdown(
                choices=[None, "Local-LLM", "OpenAI", "Anthropic", "Cohere", "Groq", "OpenRouter", "Llama.cpp",
                         "Kobold", "Ooba", "HuggingFace"],
                value=None,
                label="API Name (Mandatory) --> Unless you just want a Transcription", visible=True)
            api_key_input = gr.Textbox(
                label="API Key (Mandatory) --> Unless you're running a local model/server OR have no API selected",
                placeholder="Enter your API key here; Ignore if using Local API or Built-in API('Local-LLM')",
                visible=True)
            vad_filter_input = gr.Checkbox(label="VAD Filter (WIP)", value=False,
                                           visible=False)
            rolling_summarization_input = gr.Checkbox(label="Enable Rolling Summarization", value=False,
                                                      visible=False)
            download_video_input = gr.components.Checkbox(label="Download Video(Select to allow for file download of "
                                                                "selected video)", value=False, visible=False)
            download_audio_input = gr.components.Checkbox(label="Download Audio(Select to allow for file download of "
                                                                "selected Video's Audio)", value=False, visible=False)
            detail_level_input = gr.Slider(minimum=0.01, maximum=1.0, value=0.01, step=0.01, interactive=True,
                                           label="Summary Detail Level (Slide me) (Only OpenAI currently supported)",
                                           visible=False)
            keywords_input = gr.Textbox(label="Keywords", placeholder="Enter keywords here (comma-separated Example: "
                                                                      "tag_one,tag_two,tag_three)",
                                        value="default,no_keyword_set",
                                        visible=True)
            question_box_input = gr.Textbox(label="Question",
                                            placeholder="Enter a question to ask about the transcription",
                                            visible=False)
            # Add the additional input components
            chunk_text_by_words_checkbox = gr.Checkbox(label="Chunk Text by Words", value=False, visible=False)
            max_words_input = gr.Number(label="Max Words", value=0, precision=0, visible=False)

            chunk_text_by_sentences_checkbox = gr.Checkbox(label="Chunk Text by Sentences", value=False,
                                                           visible=False)
            max_sentences_input = gr.Number(label="Max Sentences", value=0, precision=0, visible=False)

            chunk_text_by_paragraphs_checkbox = gr.Checkbox(label="Chunk Text by Paragraphs", value=False,
                                                            visible=False)
            max_paragraphs_input = gr.Number(label="Max Paragraphs", value=0, precision=0, visible=False)

            chunk_text_by_tokens_checkbox = gr.Checkbox(label="Chunk Text by Tokens", value=False, visible=False)
            max_tokens_input = gr.Number(label="Max Tokens", value=0, precision=0, visible=False)

            inputs = [
                num_speakers_input, whisper_model_input, custom_prompt_input, offset_input, api_name_input,
                api_key_input, vad_filter_input, download_video_input, download_audio_input,
                rolling_summarization_input, detail_level_input, question_box_input, keywords_input,
                chunk_text_by_words_checkbox, max_words_input, chunk_text_by_sentences_checkbox,
                max_sentences_input, chunk_text_by_paragraphs_checkbox, max_paragraphs_input,
                chunk_text_by_tokens_checkbox, max_tokens_input
            ]

            all_inputs = [url_input] + inputs

            outputs = [
                gr.Textbox(label="Transcription (Resulting Transcription from your input URL)"),
                gr.Textbox(label="Summary or Status Message (Current status of Summary or Summary itself)"),
                gr.File(label="Download Transcription as JSON (Download the Transcription as a file)"),
                gr.File(label="Download Summary as Text (Download the Summary as a file)"),
                gr.File(label="Download Video (Download the Video as a file)", visible=True),
                gr.File(label="Download Audio (Download the Audio as a file)", visible=False),
            ]

            # Function to toggle visibility of advanced inputs
            def toggle_frontpage_ui(mode):
                visible_simple = mode == "Simple List"
                visible_advanced = mode == "Advanced List"

                return [
                    gr.update(visible=True),  # URL input should always be visible
                    gr.update(visible=visible_advanced),  # num_speakers_input
                    gr.update(visible=visible_advanced),  # whisper_model_input
                    gr.update(visible=True),  # custom_prompt_input
                    gr.update(visible=visible_advanced),  # offset_input
                    gr.update(visible=True),  # api_name_input
                    gr.update(visible=True),  # api_key_input
                    gr.update(visible=visible_advanced),  # vad_filter_input
                    gr.update(visible=visible_advanced),  # download_video_input
                    gr.update(visible=visible_advanced),  # download_audio_input
                    gr.update(visible=visible_advanced),  # rolling_summarization_input
                    gr.update(visible_advanced),  # detail_level_input
                    gr.update(visible_advanced),  # question_box_input
                    gr.update(visible=True),  # keywords_input
                    gr.update(visible_advanced),  # chunk_text_by_words_checkbox
                    gr.update(visible_advanced),  # max_words_input
                    gr.update(visible_advanced),  # chunk_text_by_sentences_checkbox
                    gr.update(visible_advanced),  # max_sentences_input
                    gr.update(visible_advanced),  # chunk_text_by_paragraphs_checkbox
                    gr.update(visible_advanced),  # max_paragraphs_input
                    gr.update(visible_advanced),  # chunk_text_by_tokens_checkbox
                    gr.update(visible_advanced),  # max_tokens_input
                ]

            def toggle_chunk_summarization(mode):
                visible = (mode == "Chunked-Summarization")
                return [
                    gr.update(visible=visible),  # chunk_text_by_words_checkbox
                    gr.update(visible=visible),  # max_words_input
                    gr.update(visible=visible),  # chunk_text_by_sentences_checkbox
                    gr.update(visible=visible),  # max_sentences_input
                    gr.update(visible=visible),  # chunk_text_by_paragraphs_checkbox
                    gr.update(visible=visible),  # max_paragraphs_input
                    gr.update(visible=visible),  # chunk_text_by_tokens_checkbox
                    gr.update(visible=visible)  # max_tokens_input
                ]

            chunk_summarization_toggle.change(fn=toggle_chunk_summarization, inputs=chunk_summarization_toggle,
                                              outputs=[
                                                  chunk_text_by_words_checkbox, max_words_input,
                                                  chunk_text_by_sentences_checkbox, max_sentences_input,
                                                  chunk_text_by_paragraphs_checkbox, max_paragraphs_input,
                                                  chunk_text_by_tokens_checkbox, max_tokens_input
                                              ])

            def start_llamafile(prompt, temperature, top_k, top_p, min_p, stream, stop, typical_p, repeat_penalty,
                                repeat_last_n,
                                penalize_nl, presence_penalty, frequency_penalty, penalty_prompt, ignore_eos,
                                system_prompt):
                # Code to start llamafile with the provided configuration
                local_llm_gui_function(prompt, temperature, top_k, top_p, min_p, stream, stop, typical_p,
                                       repeat_penalty,
                                       repeat_last_n,
                                       penalize_nl, presence_penalty, frequency_penalty, penalty_prompt, ignore_eos,
                                       system_prompt)
                # FIXME
                return "Llamafile started"

            def stop_llamafile():
                # Code to stop llamafile
                # ...
                return "Llamafile stopped"

            def toggle_light(mode):
                if mode == "Dark":
                    return """
                    <style>
                        body {
                            background-color: #1c1c1c;
                            color: #ffffff;
                        }
                        .gradio-container {
                            background-color: #1c1c1c;
                            color: #ffffff;
                        }
                        .gradio-button {
                            background-color: #4c4c4c;
                            color: #ffffff;
                        }
                        .gradio-input {
                            background-color: #4c4c4c;
                            color: #ffffff;
                        }
                        .gradio-dropdown {
                            background-color: #4c4c4c;
                            color: #ffffff;
                        }
                        .gradio-slider {
                            background-color: #4c4c4c;
                        }
                        .gradio-checkbox {
                            background-color: #4c4c4c;
                        }
                        .gradio-radio {
                            background-color: #4c4c4c;
                        }
                        .gradio-textbox {
                            background-color: #4c4c4c;
                            color: #ffffff;
                        }
                        .gradio-label {
                            color: #ffffff;
                        }
                    </style>
                    """
                else:
                    return """
                    <style>
                        body {
                            background-color: #ffffff;
                            color: #000000;
                        }
                        .gradio-container {
                            background-color: #ffffff;
                            color: #000000;
                        }
                        .gradio-button {
                            background-color: #f0f0f0;
                            color: #000000;
                        }
                        .gradio-input {
                            background-color: #f0f0f0;
                            color: #000000;
                        }
                        .gradio-dropdown {
                            background-color: #f0f0f0;
                            color: #000000;
                        }
                        .gradio-slider {
                            background-color: #f0f0f0;
                        }
                        .gradio-checkbox {
                            background-color: #f0f0f0;
                        }
                        .gradio-radio {
                            background-color: #f0f0f0;
                        }
                        .gradio-textbox {
                            background-color: #f0f0f0;
                            color: #000000;
                        }
                        .gradio-label {
                            color: #000000;
                        }
                    </style>
                    """

            # Set the event listener for the Light/Dark mode toggle switch
            theme_toggle.change(fn=toggle_light, inputs=theme_toggle, outputs=gr.HTML())

            ui_frontpage_mode_toggle.change(fn=toggle_frontpage_ui, inputs=ui_frontpage_mode_toggle, outputs=inputs)

            # Combine URL input and inputs lists
            all_inputs = [url_input] + inputs

            # lets try embedding the theme here - FIXME?
            gr.Interface(
                fn=process_url,
                inputs=all_inputs,
                outputs=outputs,
                title="Video Transcription and Summarization",
                description="Submit a video URL for transcription and summarization. Ensure you input all necessary "
                            "information including API keys.",
                theme='freddyaboulton/dracula_revamped',
                allow_flagging="never"
            )

        # Tab 2: Scrape & Summarize Articles/Websites
        with gr.Tab("Scrape & Summarize Articles/Websites"):
            url_input = gr.Textbox(label="Article URL", placeholder="Enter the article URL here")
            custom_article_title_input = gr.Textbox(label="Custom Article Title (Optional)",
                                                    placeholder="Enter a custom title for the article")
            custom_prompt_input = gr.Textbox(
                label="Custom Prompt (Optional)",
                placeholder="Provide a custom prompt for summarization",
                lines=3
            )
            api_name_input = gr.Dropdown(
                choices=[None, "huggingface", "openrouter", "openai", "anthropic", "cohere", "groq", "llama", "kobold",
                         "ooba"],
                value=None,
                label="API Name (Mandatory for Summarization)"
            )
            api_key_input = gr.Textbox(label="API Key (Mandatory if API Name is specified)",
                                       placeholder="Enter your API key here; Ignore if using Local API or Built-in API")
            keywords_input = gr.Textbox(label="Keywords", placeholder="Enter keywords here (comma-separated)",
                                        value="default,no_keyword_set", visible=True)

            scrape_button = gr.Button("Scrape and Summarize")
            result_output = gr.Textbox(label="Result")

            scrape_button.click(scrape_and_summarize, inputs=[url_input, custom_prompt_input, api_name_input,
                                                              api_key_input, keywords_input,
                                                              custom_article_title_input], outputs=result_output)

            gr.Markdown("### Or Paste Unstructured Text Below (Will use settings from above)")
            text_input = gr.Textbox(label="Unstructured Text", placeholder="Paste unstructured text here", lines=10)
            text_ingest_button = gr.Button("Ingest Unstructured Text")
            text_ingest_result = gr.Textbox(label="Result")

            text_ingest_button.click(ingest_unstructured_text,
                                     inputs=[text_input, custom_prompt_input, api_name_input, api_key_input,
                                             keywords_input, custom_article_title_input], outputs=text_ingest_result)

        with gr.Tab("Ingest & Summarize Documents"):
            gr.Markdown("Plan to put ingestion form for documents here")
            gr.Markdown("Will ingest documents and store into SQLite DB")
            gr.Markdown("RAG here we come....:/")

        # Function to update the visibility of the UI elements for Llamafile Settings
        def toggle_advanced_llamafile_mode(is_advanced):
            if is_advanced:
                return [gr.update(visible=True)] * 14
            else:
                return [gr.update(visible=False)] * 11 + [gr.update(visible=True)] * 3

    with gr.Blocks() as search_interface:
        with gr.Tab("Search & Detailed Entry View"):
            search_query_input = gr.Textbox(label="Search Query", placeholder="Enter your search query here...")
            search_type_input = gr.Radio(choices=["Title", "URL", "Keyword", "Content"], value="Title",
                                         label="Search By")

            search_button = gr.Button("Search")
            items_output = gr.Dropdown(label="Select Item", choices=[])
            item_mapping = gr.State({})

            search_button.click(fn=update_dropdown, inputs=[search_query_input, search_type_input],
                                outputs=[items_output, item_mapping])

            prompt_summary_output = gr.HTML(label="Prompt & Summary", visible=True)
            content_output = gr.HTML(label="Content", visible=True)
            items_output.change(fn=update_detailed_view, inputs=[items_output, item_mapping],
                                outputs=[prompt_summary_output, content_output])

        with gr.Tab("View Prompts"):
            with gr.Column():
                prompt_dropdown = gr.Dropdown(label="Select Prompt", choices=[])
                prompt_details_output = gr.HTML()

                prompt_dropdown.change(
                    fn=display_prompt_details,
                    inputs=prompt_dropdown,
                    outputs=prompt_details_output
                )

                prompt_list_button = gr.Button("List Prompts")
                prompt_list_button.click(
                    fn=update_prompt_dropdown,
                    outputs=prompt_dropdown
                )
        # FIXME
        with gr.Tab("Search Prompts"):
            with gr.Column():
                search_query_input = gr.Textbox(label="Search Query (It's broken)", placeholder="Enter your search query...")
                search_results_output = gr.Markdown()

                search_button = gr.Button("Search Prompts")
                search_button.click(
                    fn=display_search_results,
                    inputs=[search_query_input],
                    outputs=[search_results_output]
                )

                search_query_input.change(
                    fn=display_search_results,
                    inputs=[search_query_input],
                    outputs=[search_results_output]
                )

        with gr.Tab("Add Prompts"):
            gr.Markdown("### Add Prompt")
            title_input = gr.Textbox(label="Title", placeholder="Enter the prompt title")
            description_input = gr.Textbox(label="Description", placeholder="Enter the prompt description", lines=3)
            system_prompt_input = gr.Textbox(label="System Prompt", placeholder="Enter the system prompt", lines=3)
            user_prompt_input = gr.Textbox(label="User Prompt", placeholder="Enter the user prompt", lines=3)
            add_prompt_button = gr.Button("Add Prompt")
            add_prompt_output = gr.HTML()

            add_prompt_button.click(
                fn=add_prompt,
                inputs=[title_input, description_input, system_prompt_input, user_prompt_input],
                outputs=add_prompt_output
            )

    with gr.Blocks() as llamafile_interface:
        with gr.Tab("Llamafile Settings"):
            gr.Markdown("Settings for Llamafile")

            # Toggle switch for Advanced/Simple mode
            advanced_mode_toggle = gr.Checkbox(
                label="Advanced Mode - Click->Click again to only show 'simple' settings. Is a known bug...",
                value=False)

            # Start/Stop buttons
            start_button = gr.Button("Start Llamafile")
            stop_button = gr.Button("Stop Llamafile")

            # Configuration inputs
            prompt_input = gr.Textbox(label="Prompt", value="")
            temperature_input = gr.Number(label="Temperature", value=0.8)
            top_k_input = gr.Number(label="Top K", value=40)
            top_p_input = gr.Number(label="Top P", value=0.95)
            min_p_input = gr.Number(label="Min P", value=0.05)
            stream_input = gr.Checkbox(label="Stream", value=False)
            stop_input = gr.Textbox(label="Stop", value="[]")
            typical_p_input = gr.Number(label="Typical P", value=1.0)
            repeat_penalty_input = gr.Number(label="Repeat Penalty", value=1.1)
            repeat_last_n_input = gr.Number(label="Repeat Last N", value=64)
            penalize_nl_input = gr.Checkbox(label="Penalize New Lines", value=False)
            presence_penalty_input = gr.Number(label="Presence Penalty", value=0.0)
            frequency_penalty_input = gr.Number(label="Frequency Penalty", value=0.0)
            penalty_prompt_input = gr.Textbox(label="Penalty Prompt", value="")
            ignore_eos_input = gr.Checkbox(label="Ignore EOS", value=False)
            system_prompt_input = gr.Textbox(label="System Prompt", value="")

            # Output display
            output_display = gr.Textbox(label="Llamafile Output")

            # Function calls local_llm_gui_function() with the provided arguments
            # local_llm_gui_function() is found in 'Local_LLM_Inference_Engine_Lib.py' file
            start_button.click(start_llamafile,
                               inputs=[prompt_input, temperature_input, top_k_input, top_p_input, min_p_input,
                                       stream_input, stop_input, typical_p_input, repeat_penalty_input,
                                       repeat_last_n_input, penalize_nl_input, presence_penalty_input,
                                       frequency_penalty_input, penalty_prompt_input, ignore_eos_input,
                                       system_prompt_input], outputs=output_display)

            # This function is not implemented yet...
            # FIXME - Implement this function
            stop_button.click(stop_llamafile, outputs=output_display)

        # Toggle event for Advanced/Simple mode
        advanced_mode_toggle.change(toggle_advanced_llamafile_mode,
                                    inputs=[advanced_mode_toggle],
                                    outputs=[top_k_input, top_p_input, min_p_input, stream_input, stop_input,
                                             typical_p_input, repeat_penalty_input, repeat_last_n_input,
                                             penalize_nl_input, presence_penalty_input, frequency_penalty_input,
                                             penalty_prompt_input, ignore_eos_input])

        with gr.Tab("Llamafile Chat Interface"):
            gr.Markdown("Page to interact with Llamafile Server (iframe to Llamafile server port)")
            # Define the HTML content with the iframe
            html_content = """
            <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Llama.cpp Server Chat Interface - Loaded from  http://127.0.0.1:8080</title>
                    <style>
                        body, html {
                        height: 100%;
                        margin: 0;
                        padding: 0;
                    }
                    iframe {
                        border: none;
                        width: 85%;
                        height: 85vh; /* Full viewport height */
                    }
                </style>
            </head>
            <body>
                <iframe src="http://127.0.0.1:8080" title="Llama.cpp Server Chat Interface - Loaded from  http://127.0.0.1:8080"></iframe>
            </body>
            </html>
            """
            gr.HTML(html_content)




    export_keywords_interface = gr.Interface(
        fn=export_keywords_to_csv,
        inputs=[],
        outputs=[gr.File(label="Download Exported Keywords"), gr.Textbox(label="Status")],
        title="Export Keywords",
        description="Export all keywords in the database to a CSV file."
    )

    # Gradio interface for importing data
    def import_data(file):
        # Placeholder for actual import functionality
        return "Data imported successfully"

    import_interface = gr.Interface(
        fn=import_data,
        inputs=gr.File(label="Upload file for import"),
        outputs="text",
        title="Import Data",
        description="Import data into the database from a CSV file."
    )

    import_export_tab = gr.TabbedInterface(
        [gr.TabbedInterface(
            [gr.Interface(
                fn=export_to_csv,
                inputs=[
                    gr.Textbox(label="Search Query", placeholder="Enter your search query here..."),
                    gr.CheckboxGroup(label="Search Fields", choices=["Title", "Content"], value=["Title"]),
                    gr.Textbox(label="Keyword (Match ALL, can use multiple keywords, separated by ',' (comma) )",
                               placeholder="Enter keywords here..."),
                    gr.Number(label="Page", value=1, precision=0),
                    gr.Number(label="Results per File", value=1000, precision=0)
                ],
                outputs="text",
                title="Export Search Results to CSV",
                description="Export the search results to a CSV file."
            ),
                export_keywords_interface],
            ["Export Search Results", "Export Keywords"]
        ),
            import_interface],
        ["Export", "Import"]
    )

    keyword_add_interface = gr.Interface(
        fn=add_keyword,
        inputs=gr.Textbox(label="Add Keywords (comma-separated)", placeholder="Enter keywords here..."),
        outputs="text",
        title="Add Keywords",
        description="Add one, or multiple keywords to the database.",
        allow_flagging="never"
    )

    keyword_delete_interface = gr.Interface(
        fn=delete_keyword,
        inputs=gr.Textbox(label="Delete Keyword", placeholder="Enter keyword to delete here..."),
        outputs="text",
        title="Delete Keyword",
        description="Delete a keyword from the database.",
        allow_flagging="never"
    )

    browse_keywords_interface = gr.Interface(
        fn=keywords_browser_interface,
        inputs=[],
        outputs="markdown",
        title="Browse Keywords",
        description="View all keywords currently stored in the database."
    )

    keyword_tab = gr.TabbedInterface(
        [browse_keywords_interface, keyword_add_interface, keyword_delete_interface],
        ["Browse Keywords", "Add Keywords", "Delete Keywords"]
    )

    def ensure_dir_exists(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def gradio_download_youtube_video(url):
        """Download video using yt-dlp with specified options."""
        # Determine ffmpeg path based on the operating system.
        ffmpeg_path = './Bin/ffmpeg.exe' if os.name == 'nt' else 'ffmpeg'

        # Extract information about the video
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            sanitized_title = sanitize_filename(info_dict['title'])
            original_ext = info_dict['ext']

        # Setup the final directory and filename
        download_dir = Path(f"results/{sanitized_title}")
        download_dir.mkdir(parents=True, exist_ok=True)
        output_file_path = download_dir / f"{sanitized_title}.{original_ext}"

        # Initialize yt-dlp with generic options and the output template
        ydl_opts = {
            'format': 'bestvideo+bestaudio/best',
            'ffmpeg_location': ffmpeg_path,
            'outtmpl': str(output_file_path),
            'noplaylist': True, 'quiet': True
        }

        # Execute yt-dlp to download the video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Final check to ensure file exists
        if not output_file_path.exists():
            raise FileNotFoundError(f"Expected file was not found: {output_file_path}")

        return str(output_file_path)

    download_videos_interface = gr.Interface(
        fn=gradio_download_youtube_video,
        inputs=gr.Textbox(label="YouTube URL", placeholder="Enter YouTube video URL here"),
        outputs=gr.File(label="Download Video"),
        title="YouTube Video Downloader (Simple youtube video downloader tool, if you want a real one, check this project: https://github.com/StefanLobbenmeier/youtube-dl-gui or https://github.com/yt-dlg/yt-dlg )",
        description="Enter a YouTube URL to download the video.",
        allow_flagging="never"
    )

    # Combine interfaces into a tabbed interface
    tabbed_interface = gr.TabbedInterface([iface, search_interface, llamafile_interface, keyword_tab, import_export_tab, download_videos_interface],
                                          ["Transcription / Summarization / Ingestion", "Search / Detailed View",
                                           "Llamafile Interface", "Keywords", "Export/Import",  "Download Video/Audio Files"])
    # Launch the interface
    server_port_variable = 7860
    global server_mode, share_public
    if server_mode is True and share_public is False:
        tabbed_interface.launch(share=True, server_port=server_port_variable, server_name="http://0.0.0.0")
    elif share_public == True:
        tabbed_interface.launch(share=True, )
    else:
        tabbed_interface.launch(share=False, )


def clean_youtube_url(url):
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    if 'list' in query_params:
        query_params.pop('list')
    cleaned_query = urlencode(query_params, doseq=True)
    cleaned_url = urlunparse(parsed_url._replace(query=cleaned_query))
    return cleaned_url


def process_url(
        url,
        num_speakers,
        whisper_model,
        custom_prompt,
        offset,
        api_name,
        api_key,
        vad_filter,
        download_video,
        download_audio,
        rolling_summarization,
        detail_level,
        question_box,
        keywords,
        chunk_text_by_words,
        max_words,
        chunk_text_by_sentences,
        max_sentences,
        chunk_text_by_paragraphs,
        max_paragraphs,
        chunk_text_by_tokens,
        max_tokens
):
    # Handle the chunk summarization options
    set_chunk_txt_by_words = chunk_text_by_words
    set_max_txt_chunk_words = max_words
    set_chunk_txt_by_sentences = chunk_text_by_sentences
    set_max_txt_chunk_sentences = max_sentences
    set_chunk_txt_by_paragraphs = chunk_text_by_paragraphs
    set_max_txt_chunk_paragraphs = max_paragraphs
    set_chunk_txt_by_tokens = chunk_text_by_tokens
    set_max_txt_chunk_tokens = max_tokens

    # Validate input
    if not url:
        return "No URL provided.", "No URL provided.", None, None, None, None, None, None

    if not is_valid_url(url):
        return "Invalid URL format.", "Invalid URL format.", None, None, None, None, None, None

    # Clean the URL to remove playlist parameters if any
    url = clean_youtube_url(url)

    print("API Name received:", api_name)  # Debugging line

    logging.info(f"Processing URL: {url}")
    video_file_path = None
    global info_dict
    try:
        # Instantiate the database, db as a instance of the Database class
        db = Database()
        media_url = url

        info_dict = get_youtube(url)  # Extract video information using yt_dlp
        media_title = info_dict['title'] if 'title' in info_dict else 'Untitled'

        results = main(url, api_name=api_name, api_key=api_key,
                       num_speakers=num_speakers,
                       whisper_model=whisper_model,
                       offset=offset,
                       vad_filter=vad_filter,
                       download_video_flag=download_video,
                       custom_prompt=custom_prompt,
                       overwrite=args.overwrite,
                       rolling_summarization=rolling_summarization,
                       detail=detail_level,
                       keywords=keywords,
                       )

        if not results:
            return "No URL provided.", "No URL provided.", None, None, None, None, None, None

        transcription_result = results[0]
        transcription_text = json.dumps(transcription_result['transcription'], indent=2)
        summary_text = transcription_result.get('summary', 'Summary not available')

        # Prepare file paths for transcription and summary
        # Sanitize filenames
        audio_file_sanitized = sanitize_filename(transcription_result['audio_file'])
        json_pretty_file_path = os.path.join('Results', audio_file_sanitized.replace('.wav', '.segments_pretty.json'))
        json_file_path = os.path.join('Results', audio_file_sanitized.replace('.wav', '.segments.json'))
        summary_file_path = os.path.join('Results', audio_file_sanitized.replace('.wav', '_summary.txt'))

        logging.debug(f"Transcription result: {transcription_result}")
        logging.debug(f"Audio file path: {transcription_result['audio_file']}")

        # Write the transcription to the JSON File
        try:
            with open(json_file_path, 'w') as json_file:
                json.dump(transcription_result['transcription'], json_file, indent=2)
        except IOError as e:
            logging.error(f"Error writing transcription to JSON file: {e}")

        # Write the summary to the summary file
        with open(summary_file_path, 'w') as summary_file:
            summary_file.write(summary_text)

        try:
            if download_video:
                video_file_path = transcription_result.get('video_path', None)
                if video_file_path and os.path.exists(video_file_path):
                    logging.debug(f"Confirmed existence of video file at {video_file_path}")
                else:
                    logging.error(f"Video file not found at expected path: {video_file_path}")
                    video_file_path = None
            else:
                video_file_path = None

            if isinstance(transcription_result['transcription'], list):
                text = ' '.join([segment['Text'] for segment in transcription_result['transcription']])
            else:
                text = ''

        except Exception as e:
            logging.error(f"Error processing video file: {e}")

        # Check if files exist before returning paths
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"File not found: {json_file_path}")
        if not os.path.exists(summary_file_path):
            raise FileNotFoundError(f"File not found: {summary_file_path}")

        formatted_transcription = format_transcription(transcription_result)

        try:
            # Ensure these variables are correctly populated
            custom_prompt = args.custom_prompt if args.custom_prompt else ("\n\nabove is the transcript of a video "
                                                                           "Please read through the transcript carefully. Identify the main topics that are discussed over the "
                                                                           "course of the transcript. Then, summarize the key points about each main topic in a concise bullet "
                                                                           "point. The bullet points should cover the key information conveyed about each topic in the video, "
                                                                           "but should be much shorter than the full transcript. Please output your bullet point summary inside "
                                                                           "<bulletpoints> tags.")

            db = Database()
            create_tables()
            media_url = url
            # FIXME  - IDK?
            video_info = get_video_info(media_url)
            media_title = get_page_title(media_url)
            media_type = "video"
            media_content = transcription_text
            keyword_list = keywords.split(',') if keywords else ["default"]
            media_keywords = ', '.join(keyword_list)
            media_author = "auto_generated"
            media_ingestion_date = datetime.now().strftime('%Y-%m-%d')
            transcription_model = whisper_model  # Add the transcription model used

            # Log the values before calling the function
            logging.info(f"Media URL: {media_url}")
            logging.info(f"Media Title: {media_title}")
            logging.debug(f"Media Type: {media_type}")
            logging.debug(f"Media Content: {media_content}")
            logging.debug(f"Media Keywords: {media_keywords}")
            logging.debug(f"Media Author: {media_author}")
            logging.debug(f"Ingestion Date: {media_ingestion_date}")
            logging.debug(f"Custom Prompt: {custom_prompt}")
            logging.debug(f"Summary Text: {summary_text}")
            logging.debug(f"Transcription Model: {transcription_model}")

            # Check if any required field is empty
            if not media_url or not media_title or not media_type or not media_content or not media_keywords or not custom_prompt or not summary_text:
                raise InputError("Please provide all required fields.")

            add_media_with_keywords(
                url=media_url,
                title=media_title,
                media_type=media_type,
                content=media_content,
                keywords=media_keywords,
                prompt=custom_prompt,
                summary=summary_text,
                transcription_model=transcription_model,  # Pass the transcription model
                author=media_author,
                ingestion_date=media_ingestion_date
            )
        except Exception as e:
            logging.error(f"Failed to add media to the database: {e}")

        if summary_file_path and os.path.exists(summary_file_path):
            return transcription_text, summary_text, json_file_path, summary_file_path, video_file_path, None
        else:
            return transcription_text, summary_text, json_file_path, None, video_file_path, None
    except KeyError as e:
        logging.error(f"Error processing {url}: {str(e)}")
        return str(e), 'Error processing the request.', None, None, None, None
    except Exception as e:
        logging.error(f"Error processing URL: {e}")
        return str(e), 'Error processing the request.', None, None, None, None


# FIXME - Prompt sample box

# Sample data
prompts_category_1 = [
    "What are the key points discussed in the video?",
    "Summarize the main arguments made by the speaker.",
    "Describe the conclusions of the study presented."
]

prompts_category_2 = [
    "How does the proposed solution address the problem?",
    "What are the implications of the findings?",
    "Can you explain the theory behind the observed phenomenon?"
]

all_prompts = prompts_category_1 + prompts_category_2


# Search function
def search_prompts(query):
    filtered_prompts = [prompt for prompt in all_prompts if query.lower() in prompt.lower()]
    return "\n".join(filtered_prompts)


# Handle prompt selection
def handle_prompt_selection(prompt):
    return f"You selected: {prompt}"


#
#
#######################################################################################################################


#######################################################################################################################
# Local LLM Setup / Running
#
# Function List
# 1. download_latest_llamafile(repo, asset_name_prefix, output_filename)
# 2. download_file(url, dest_path, expected_checksum=None, max_retries=3, delay=5)
# 3. verify_checksum(file_path, expected_checksum)
# 4. cleanup_process()
# 5. signal_handler(sig, frame)
# 6. local_llm_function()
# 7. launch_in_new_terminal_windows(executable, args)
# 8. launch_in_new_terminal_linux(executable, args)
# 9. launch_in_new_terminal_mac(executable, args)
#
#
#######################################################################################################################


#######################################################################################################################
# Main()
#

def main(input_path, api_name=None, api_key=None,
         num_speakers=2,
         whisper_model="small.en",
         offset=0,
         vad_filter=False,
         download_video_flag=False,
         custom_prompt=None,
         overwrite=False,
         rolling_summarization=False,
         detail=0.01,
         keywords=None,
         llm_model=None,
         time_based=False,
         set_chunk_txt_by_words=False,
         set_max_txt_chunk_words=0,
         set_chunk_txt_by_sentences=False,
         set_max_txt_chunk_sentences=0,
         set_chunk_txt_by_paragraphs=False,
         set_max_txt_chunk_paragraphs=0,
         set_chunk_txt_by_tokens=False,
         set_max_txt_chunk_tokens=0,
         ):
    global detail_level_number, summary, audio_file, transcription_result, info_dict

    detail_level = detail

    print(f"Keywords: {keywords}")

    if input_path is None and args.user_interface:
        return []
    start_time = time.monotonic()
    paths = []  # Initialize paths as an empty list
    if os.path.isfile(input_path) and input_path.endswith('.txt'):
        logging.debug("MAIN: User passed in a text file, processing text file...")
        paths = read_paths_from_file(input_path)
    elif os.path.exists(input_path):
        logging.debug("MAIN: Local file path detected")
        paths = [input_path]
    elif (info_dict := get_youtube(input_path)) and 'entries' in info_dict:
        logging.debug("MAIN: YouTube playlist detected")
        print(
            "\n\nSorry, but playlists aren't currently supported. You can run the following command to generate a "
            "text file that you can then pass into this script though! (It may not work... playlist support seems "
            "spotty)" + """\n\n\tpython Get_Playlist_URLs.py <Youtube Playlist URL>\n\n\tThen,\n\n\tpython 
            diarizer.py <playlist text file name>\n\n""")
        return
    else:
        paths = [input_path]
    results = []

    for path in paths:
        try:
            if path.startswith('http'):
                logging.debug("MAIN: URL Detected")
                info_dict = get_youtube(path)
                json_file_path = None
                if info_dict:
                    logging.debug(f"MAIN: info_dict content: {info_dict}")
                    logging.debug("MAIN: Creating path for video file...")
                    download_path = create_download_directory(info_dict['title'])
                    logging.debug("MAIN: Path created successfully\n MAIN: Now Downloading video from yt_dlp...")
                    try:
                        video_path = download_video(path, download_path, info_dict, download_video_flag)
                        if video_path is None:
                            logging.error("MAIN: video_path is None after download_video")
                            continue
                    except RuntimeError as e:
                        logging.error(f"Error downloading video: {str(e)}")
                        # FIXME - figure something out for handling this situation....
                        continue
                    logging.debug("MAIN: Video downloaded successfully")
                    logging.debug("MAIN: Converting video file to WAV...")
                    audio_file = convert_to_wav(video_path, offset)
                    logging.debug("MAIN: Audio file converted successfully")
            else:
                if os.path.exists(path):
                    logging.debug("MAIN: Local file path detected")
                    download_path, info_dict, audio_file = process_local_file(path)
                else:
                    logging.error(f"File does not exist: {path}")
                    continue

            if info_dict:
                logging.debug("MAIN: Creating transcription file from WAV")
                segments = speech_to_text(audio_file, whisper_model=whisper_model, vad_filter=vad_filter)

                transcription_result = {
                    'video_path': path,
                    'audio_file': audio_file,
                    'transcription': segments
                }

                if isinstance(segments, dict) and "error" in segments:
                    logging.error(f"Error transcribing audio: {segments['error']}")
                    transcription_result['error'] = segments['error']

                results.append(transcription_result)
                logging.info(f"MAIN: Transcription complete: {audio_file}")

                # Check if segments is a dictionary before proceeding with summarization
                if isinstance(segments, dict):
                    logging.warning("Skipping summarization due to transcription error")
                    continue

                # FIXME
                # Perform rolling summarization based on API Name, detail level, and if an API key exists
                # Will remove the API key once rolling is added for llama.cpp

                # FIXME - Add input for model name for tabby and vllm

                if rolling_summarization:
                    logging.info("MAIN: Rolling Summarization")
                    api_key = openai_api_key
                    global client
                    client = OpenAI(api_key)
                    # Extract the text from the segments
                    text = extract_text_from_segments(segments)

                    # Set the json_file_path
                    json_file_path = audio_file.replace('.wav', '.segments.json')

                    # Perform rolling summarization
                    summary = summarize_with_detail_openai(text, detail=detail_level, verbose=False)

                    # Handle the summarized output
                    if summary:
                        transcription_result['summary'] = summary
                        logging.info("MAIN: Rolling Summarization successful.")
                        save_summary_to_file(summary, json_file_path)
                    else:
                        logging.warning("MAIN: Rolling Summarization failed.")
                # Perform summarization based on the specified API
                elif api_name:
                    logging.debug(f"MAIN: Summarization being performed by {api_name}")
                    json_file_path = audio_file.replace('.wav', '.segments.json')
                    if api_name.lower() == 'openai':
                        openai_api_key = api_key if api_key else config.get('API', 'openai_api_key',
                                                                            fallback=None)
                        try:
                            logging.debug(f"MAIN: trying to summarize with openAI")
                            summary = summarize_with_openai(openai_api_key, json_file_path, custom_prompt)
                            if summary != "openai: Error occurred while processing summary":
                                transcription_result['summary'] = summary
                                logging.info(f"Summary generated using {api_name} API")
                                save_summary_to_file(summary, json_file_path)
                                # Add media to the database
                                add_media_with_keywords(
                                    url=path,
                                    title=info_dict.get('title', 'Untitled'),
                                    media_type='video',
                                    content=' '.join([segment['text'] for segment in segments]),
                                    keywords=','.join(keywords),
                                    prompt=custom_prompt or 'No prompt provided',
                                    summary=summary or 'No summary provided',
                                    transcription_model=whisper_model,
                                    author=info_dict.get('uploader', 'Unknown'),
                                    ingestion_date=datetime.now().strftime('%Y-%m-%d')
                                )
                            else:
                                logging.warning(f"Failed to generate summary using {api_name} API")
                        except requests.exceptions.ConnectionError:
                            requests.status_code = "Connection: "
                    elif api_name.lower() == "anthropic":
                        anthropic_api_key = api_key if api_key else config.get('API', 'anthropic_api_key',
                                                                               fallback=None)
                        try:
                            logging.debug(f"MAIN: Trying to summarize with anthropic")
                            summary = summarize_with_claude(anthropic_api_key, json_file_path, anthropic_model,
                                                            custom_prompt)
                        except requests.exceptions.ConnectionError:
                            requests.status_code = "Connection: "
                    elif api_name.lower() == "cohere":
                        cohere_api_key = os.getenv('COHERE_TOKEN').replace('"', '') if api_key is None else api_key
                        try:
                            logging.debug(f"MAIN: Trying to summarize with cohere")
                            summary = summarize_with_cohere(cohere_api_key, json_file_path, cohere_model, custom_prompt)
                        except requests.exceptions.ConnectionError:
                            requests.status_code = "Connection: "
                    elif api_name.lower() == "groq":
                        groq_api_key = api_key if api_key else config.get('API', 'groq_api_key', fallback=None)
                        try:
                            logging.debug(f"MAIN: Trying to summarize with Groq")
                            summary = summarize_with_groq(groq_api_key, json_file_path, groq_model, custom_prompt)
                        except requests.exceptions.ConnectionError:
                            requests.status_code = "Connection: "
                    elif api_name.lower() == "openrouter":
                        openrouter_api_key = api_key if api_key else config.get('API', 'openrouter_api_key',
                                                                                fallback=None)
                        try:
                            logging.debug(f"MAIN: Trying to summarize with OpenRouter")
                            summary = summarize_with_openrouter(openrouter_api_key, json_file_path, custom_prompt)
                        except requests.exceptions.ConnectionError:
                            requests.status_code = "Connection: "
                    elif api_name.lower() == "llama":
                        llama_token = api_key if api_key else config.get('API', 'llama_api_key', fallback=None)
                        llama_ip = llama_api_IP
                        try:
                            logging.debug(f"MAIN: Trying to summarize with Llama.cpp")
                            summary = summarize_with_llama(llama_ip, json_file_path, llama_token, custom_prompt)
                        except requests.exceptions.ConnectionError:
                            requests.status_code = "Connection: "
                    elif api_name.lower() == "kobold":
                        kobold_token = api_key if api_key else config.get('API', 'kobold_api_key', fallback=None)
                        kobold_ip = kobold_api_IP
                        try:
                            logging.debug(f"MAIN: Trying to summarize with kobold.cpp")
                            summary = summarize_with_kobold(kobold_ip, json_file_path, kobold_token, custom_prompt)
                        except requests.exceptions.ConnectionError:
                            requests.status_code = "Connection: "
                    elif api_name.lower() == "ooba":
                        ooba_token = api_key if api_key else config.get('API', 'ooba_api_key', fallback=None)
                        ooba_ip = ooba_api_IP
                        try:
                            logging.debug(f"MAIN: Trying to summarize with oobabooga")
                            summary = summarize_with_oobabooga(ooba_ip, json_file_path, ooba_token, custom_prompt)
                        except requests.exceptions.ConnectionError:
                            requests.status_code = "Connection: "
                    elif api_name.lower() == "tabbyapi":
                        tabbyapi_key = api_key if api_key else config.get('API', 'tabby_api_key', fallback=None)
                        tabbyapi_ip = tabby_api_IP
                        try:
                            logging.debug(f"MAIN: Trying to summarize with tabbyapi")
                            tabby_model = llm_model
                            summary = summarize_with_tabbyapi(tabby_api_key, tabby_api_IP, json_file_path, tabby_model,
                                                              custom_prompt)
                        except requests.exceptions.ConnectionError:
                            requests.status_code = "Connection: "
                    elif api_name.lower() == "vllm":
                        logging.debug(f"MAIN: Trying to summarize with VLLM")
                        summary = summarize_with_vllm(vllm_api_url, vllm_api_key, llm_model, json_file_path,
                                                      custom_prompt)
                    elif api_name.lower() == "local-llm":
                        logging.debug(f"MAIN: Trying to summarize with the local LLM, Mistral Instruct v0.2")
                        local_llm_url = "http://127.0.0.1:8080"
                        summary = summarize_with_local_llm(json_file_path, custom_prompt)
                    elif api_name.lower() == "huggingface":
                        huggingface_api_key = api_key if api_key else config.get('API', 'huggingface_api_key',
                                                                                 fallback=None)
                        try:
                            logging.debug(f"MAIN: Trying to summarize with huggingface")
                            summarize_with_huggingface(huggingface_api_key, json_file_path, custom_prompt)
                        except requests.exceptions.ConnectionError:
                            requests.status_code = "Connection: "

                    else:
                        logging.warning(f"Unsupported API: {api_name}")
                        summary = None

                    if summary:
                        transcription_result['summary'] = summary
                        logging.info(f"Summary generated using {api_name} API")
                        save_summary_to_file(summary, json_file_path)
                    # FIXME
                    # elif final_summary:
                    #     logging.info(f"Rolling summary generated using {api_name} API")
                    #     logging.info(f"Final Rolling summary is {final_summary}\n\n")
                    #     save_summary_to_file(final_summary, json_file_path)
                    else:
                        logging.warning(f"Failed to generate summary using {api_name} API")
                else:
                    logging.info("MAIN: #2 - No API specified. Summarization will not be performed")

                # Add media to the database
                add_media_with_keywords(
                    url=path,
                    title=info_dict.get('title', 'Untitled'),
                    media_type='video',
                    content=' '.join([segment['text'] for segment in segments]),
                    keywords=','.join(keywords),
                    prompt=custom_prompt or 'No prompt provided',
                    summary=summary or 'No summary provided',
                    transcription_model=whisper_model,
                    author=info_dict.get('uploader', 'Unknown'),
                    ingestion_date=datetime.now().strftime('%Y-%m-%d')
                )

        except Exception as e:
            logging.error(f"Error processing {path}: {str(e)}")
            logging.error(str(e))
            continue
        # end_time = time.monotonic()
        # print("Total program execution time: " + timedelta(seconds=end_time - start_time))

    return results


def signal_handler(sig, frame):
    logging.info('Signal handler called with signal: %s', sig)
    cleanup_process()
    sys.exit(0)


############################## MAIN ##############################
#
#

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    # Establish logging baseline
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print_hello()
    parser = argparse.ArgumentParser(
        description='Transcribe and summarize videos.',
        epilog='''
Sample commands:
    1. Simple Sample command structure:
        summarize.py <path_to_video> -api openai -k tag_one tag_two tag_three

    2. Rolling Summary Sample command structure:
        summarize.py <path_to_video> -api openai -prompt "custom_prompt_goes_here-is-appended-after-transcription" -roll -detail 0.01 -k tag_one tag_two tag_three

    3. FULL Sample command structure:
        summarize.py <path_to_video> -api openai -ns 2 -wm small.en -off 0 -vad -log INFO -prompt "custom_prompt" -overwrite -roll -detail 0.01 -k tag_one tag_two tag_three

    4. Sample command structure for UI:
        summarize.py -gui -log DEBUG
        ''',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('input_path', type=str, help='Path or URL of the video', nargs='?')
    parser.add_argument('-v', '--video', action='store_true', help='Download the video instead of just the audio')
    parser.add_argument('-api', '--api_name', type=str, help='API name for summarization (optional)')
    parser.add_argument('-key', '--api_key', type=str, help='API key for summarization (optional)')
    parser.add_argument('-ns', '--num_speakers', type=int, default=2, help='Number of speakers (default: 2)')
    parser.add_argument('-wm', '--whisper_model', type=str, default='small.en',
                        help='Whisper model (default: small.en)')
    parser.add_argument('-off', '--offset', type=int, default=0, help='Offset in seconds (default: 0)')
    parser.add_argument('-vad', '--vad_filter', action='store_true', help='Enable VAD filter')
    parser.add_argument('-log', '--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Log level (default: INFO)')
    parser.add_argument('-gui', '--user_interface', action='store_true', help="Launch the Gradio user interface")
    parser.add_argument('-demo', '--demo_mode', action='store_true', help='Enable demo mode')
    parser.add_argument('-prompt', '--custom_prompt', type=str,
                        help='Pass in a custom prompt to be used in place of the existing one.\n (Probably should just '
                             'modify the script itself...)')
    parser.add_argument('-overwrite', '--overwrite', action='store_true', help='Overwrite existing files')
    parser.add_argument('-roll', '--rolling_summarization', action='store_true', help='Enable rolling summarization')
    parser.add_argument('-detail', '--detail_level', type=float, help='Mandatory if rolling summarization is enabled, '
                                                                      'defines the chunk  size.\n Default is 0.01(lots '
                                                                      'of chunks) -> 1.00 (few chunks)\n Currently '
                                                                      'only OpenAI works. ',
                        default=0.01, )
    parser.add_argument('-model', '--llm_model', type=str, default='',
                        help='Model to use for LLM summarization (only used for vLLM/TabbyAPI)')
    parser.add_argument('-k', '--keywords', nargs='+', default=['cli_ingest_no_tag'],
                        help='Keywords for tagging the media, can use multiple separated by spaces (default: cli_ingest_no_tag)')
    parser.add_argument('--log_file', type=str, help='Where to save logfile (non-default)')
    parser.add_argument('--local_llm', action='store_true',
                        help="Use a local LLM from the script(Downloads llamafile from github and 'mistral-7b-instruct-v0.2.Q8' - 8GB model from Huggingface)")
    parser.add_argument('--server_mode', action='store_true',
                        help='Run in server mode (This exposes the GUI/Server to the network)')
    parser.add_argument('--share_public', type=int, default=7860,
                        help="This will use Gradio's built-in ngrok tunneling to share the server publicly on the internet. Specify the port to use (default: 7860)")
    parser.add_argument('--port', type=int, default=7860, help='Port to run the server on')
    #parser.add_argument('--offload', type=int, default=20, help='Numbers of layers to offload to GPU for Llamafile usage')
    # parser.add_argument('-o', '--output_path', type=str, help='Path to save the output file')

    args = parser.parse_args()

    # Set Chunking values/variables
    set_chunk_txt_by_words = False
    set_max_txt_chunk_words = 0
    set_chunk_txt_by_sentences = False
    set_max_txt_chunk_sentences = 0
    set_chunk_txt_by_paragraphs = False
    set_max_txt_chunk_paragraphs = 0
    set_chunk_txt_by_tokens = False
    set_max_txt_chunk_tokens = 0

    if args.share_public:
        share_public = args.share_public
    else:
        share_public = None
    if args.server_mode:
        server_mode = args.server_mode
    else:
        server_mode = None
    if args.server_mode is True:
        server_mode = True
    if args.port:
        server_port = args.port
    else:
        server_port = None

    ########## Logging setup
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, args.log_level))

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, args.log_level))
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    if args.log_file:
        # Create file handler
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setLevel(getattr(logging, args.log_level))
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Log file created at: {args.log_file}")

    ########## Custom Prompt setup
    custom_prompt = args.custom_prompt

    if not args.custom_prompt:
        logging.debug("No custom prompt defined, will use default")
        args.custom_prompt = (
            "\n\nabove is the transcript of a video. "
            "Please read through the transcript carefully. Identify the main topics that are "
            "discussed over the course of the transcript. Then, summarize the key points about each "
            "main topic in a concise bullet point. The bullet points should cover the key "
            "information conveyed about each topic in the video, but should be much shorter than "
            "the full transcript. Please output your bullet point summary inside <bulletpoints> "
            "tags."
        )
        print("No custom prompt defined, will use default")

        custom_prompt = args.custom_prompt
    else:
        logging.debug(f"Custom prompt defined, will use \n\nf{custom_prompt} \n\nas the prompt")
        print(f"Custom Prompt has been defined. Custom prompt: \n\n {args.custom_prompt}")

    # Check if the user wants to use the local LLM from the script
    local_llm = args.local_llm
    logging.info(f'Local LLM flag: {local_llm}')

    if args.user_interface:
        if local_llm:
            local_llm_function()
            time.sleep(2)
            webbrowser.open_new_tab('http://127.0.0.1:7860')
        launch_ui(demo_mode=False)
    else:
        if not args.input_path:
            parser.print_help()
            sys.exit(1)

        logging.info('Starting the transcription and summarization process.')
        logging.info(f'Input path: {args.input_path}')
        logging.info(f'API Name: {args.api_name}')
        logging.info(f'Number of speakers: {args.num_speakers}')
        logging.info(f'Whisper model: {args.whisper_model}')
        logging.info(f'Offset: {args.offset}')
        logging.info(f'VAD filter: {args.vad_filter}')
        logging.info(f'Log Level: {args.log_level}')
        logging.info(f'Demo Mode: {args.demo_mode}')
        logging.info(f'Custom Prompt: {args.custom_prompt}')
        logging.info(f'Overwrite: {args.overwrite}')
        logging.info(f'Rolling Summarization: {args.rolling_summarization}')
        logging.info(f'User Interface: {args.user_interface}')
        logging.info(f'Video Download: {args.video}')
        # logging.info(f'Save File location: {args.output_path}')
        # logging.info(f'Log File location: {args.log_file}')

        # Get all API keys from the config
        api_keys = {key: value for key, value in config.items('API') if key.endswith('_api_key')}

        api_name = args.api_name

        # Rolling Summarization will only be performed if an API is specified and the API key is available
        # and the rolling summarization flag is set
        #
        summary = None  # Initialize to ensure it's always defined
        if args.detail_level == None:
            args.detail_level = 0.01
        if args.api_name and args.rolling_summarization and any(
                key.startswith(args.api_name) and value is not None for key, value in api_keys.items()):
            logging.info(f'MAIN: API used: {args.api_name}')
            logging.info('MAIN: Rolling Summarization will be performed.')

        elif args.api_name:
            logging.info(f'MAIN: API used: {args.api_name}')
            logging.info('MAIN: Summarization (not rolling) will be performed.')

        else:
            logging.info('No API specified. Summarization will not be performed.')

        logging.debug("Platform check being performed...")
        platform_check()
        logging.debug("CUDA check being performed...")
        cuda_check()
        logging.debug("ffmpeg check being performed...")
        check_ffmpeg()
        #download_ffmpeg()

        llm_model = args.llm_model or None

        try:
            results = main(args.input_path, api_name=args.api_name,
                           api_key=args.api_key,
                           num_speakers=args.num_speakers,
                           whisper_model=args.whisper_model,
                           offset=args.offset,
                           vad_filter=args.vad_filter,
                           download_video_flag=args.video,
                           custom_prompt=args.custom_prompt,
                           overwrite=args.overwrite,
                           rolling_summarization=args.rolling_summarization,
                           detail=args.detail_level,
                           keywords=args.keywords,
                           llm_model=args.llm_model,
                           time_based=args.time_based,
                           set_chunk_txt_by_words=set_chunk_txt_by_words,
                           set_max_txt_chunk_words=set_max_txt_chunk_words,
                           set_chunk_txt_by_sentences=set_chunk_txt_by_sentences,
                           set_max_txt_chunk_sentences=set_max_txt_chunk_sentences,
                           set_chunk_txt_by_paragraphs=set_chunk_txt_by_paragraphs,
                           set_max_txt_chunk_paragraphs=set_max_txt_chunk_paragraphs,
                           set_chunk_txt_by_tokens=set_chunk_txt_by_tokens,
                           set_max_txt_chunk_tokens=set_max_txt_chunk_tokens,
                           )

            logging.info('Transcription process completed.')
            atexit.register(cleanup_process)
        except Exception as e:
            logging.error('An error occurred during the transcription process.')
            logging.error(str(e))
            sys.exit(1)

        finally:
            cleanup_process()
