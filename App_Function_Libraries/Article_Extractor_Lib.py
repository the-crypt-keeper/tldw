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
from Article_Summarization_Lib import *
#from Article_Extractor_Lib import *
from Audio_Transcription_Lib import *
from Chunk_Lib import *
from Diarization_Lib import *
from Video_DL_Ingestion_Lib import *
from Local_File_Processing_Lib import *
from Local_LLM_Inference_Engine_Lib import *
from Local_Summarization_Lib import *
from Old_Chunking_Lib import *
from SQLite_DB import *
from Summarization_General_Lib import *
from System_Checks_Lib import *
from Tokenization_Methods_Lib import *
from Video_DL_Ingestion_Lib import *
from Web_UI_Lib import *



#######################################################################################################################
# Function Definitions
#

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