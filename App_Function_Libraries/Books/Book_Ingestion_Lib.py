# Book_Ingestion_Lib.py
#########################################
# Library to hold functions for ingesting book files.#
#
####################
# Function List
#
# 1. ingest_text_file(file_path, title=None, author=None, keywords=None):
# 2.
#
#
####################
#
# Import necessary libraries
import os
import re
import tempfile
import zipfile
from datetime import datetime
import logging

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub

from App_Function_Libraries.Chunk_Lib import chunk_ebook_by_chapters
#
# Import Local
from App_Function_Libraries.DB.DB_Manager import add_media_with_keywords, add_media_to_database
from App_Function_Libraries.Summarization.Summarization_General_Lib import perform_summarization


#
#######################################################################################################################
# Function Definitions
#

def import_epub(file_path, title=None, author=None, keywords=None, custom_prompt=None, system_prompt=None, summary=None,
               auto_summarize=False, api_name=None, api_key=None, chunk_options=None, custom_chapter_pattern=None):
    """
    Imports an EPUB file, extracts its content, chunks it, optionally summarizes it, and adds it to the database.

    Parameters:
        - file_path (str): Path to the EPUB file.
        - title (str, optional): Title of the book.
        - author (str, optional): Author of the book.
        - keywords (str, optional): Comma-separated keywords for the book.
        - custom_prompt (str, optional): Custom user prompt for summarization.
        - summary (str, optional): Predefined summary of the book.
        - auto_summarize (bool, optional): Whether to auto-summarize the chunks.
        - api_name (str, optional): API name for summarization.
        - api_key (str, optional): API key for summarization.
        - chunk_options (dict, optional): Options for chunking.
        - custom_chapter_pattern (str, optional): Custom regex pattern for chapter detection.

    Returns:
        - str: Status message indicating success or failure.
    """
    try:
        logging.info(f"Importing EPUB file from {file_path}")

        # Convert EPUB to Markdown
        markdown_content = epub_to_markdown(file_path)
        logging.debug("Converted EPUB to Markdown.")

        # Extract metadata if not provided
        if not title or not author:
            extracted_title, extracted_author = extract_epub_metadata(markdown_content)
            title = title or extracted_title or os.path.splitext(os.path.basename(file_path))[0]
            author = author or extracted_author or "Unknown"
            logging.debug(f"Extracted metadata - Title: {title}, Author: {author}")

        # Process keywords
        keyword_list = [kw.strip() for kw in keywords.split(',')] if keywords else []
        logging.debug(f"Keywords: {keyword_list}")

        # Set default chunk options if not provided
        if chunk_options is None:
            chunk_options = {
                'method': 'chapter',
                'max_size': 500,
                'overlap': 200,
                'custom_chapter_pattern': custom_chapter_pattern
            }
        else:
            # Ensure 'method' is set to 'chapter' when using chapter chunking
            chunk_options.setdefault('method', 'chapter')
            chunk_options.setdefault('custom_chapter_pattern', custom_chapter_pattern)

        # Chunk the content by chapters
        chunks = chunk_ebook_by_chapters(markdown_content, chunk_options)
        logging.info(f"Total chunks created: {len(chunks)}")
        if chunks:
            logging.debug(f"Structure of first chunk: {chunks[0].keys()}")


        # Handle summarization if enabled
        if auto_summarize and api_name and api_key:
            logging.info("Auto-summarization is enabled.")
            summarized_chunks = []
            for chunk in chunks:
                chunk_text = chunk.get('text', '')
                if chunk_text:
                    summary_text = perform_summarization(api_name, chunk_text, custom_prompt, api_key, recursive_summarization=False, temp=None, system_message=system_prompt)
                    chunk['metadata']['summary'] = summary_text
                    summarized_chunks.append(chunk)
            chunks = summarized_chunks
            logging.info("Summarization of chunks completed.")
        else:
            # If not summarizing, set a default summary or use provided summary
            if summary:
                logging.debug("Using provided summary.")
            else:
                summary = "No summary provided."

        # Create info_dict
        info_dict = {
            'title': title,
            'uploader': author,
            'ingestion_date': datetime.now().strftime('%Y-%m-%d')
        }

        # Prepare segments for database
        segments = [{'Text': chunk.get('text', chunk.get('content', ''))} for chunk in chunks]
        logging.debug(f"Prepared segments for database. Number of segments: {len(segments)}")

        # Add to database
        result = add_media_to_database(
            url=file_path,
            info_dict=info_dict,
            segments=segments,
            summary=summary,
            keywords=keyword_list,
            custom_prompt_input=custom_prompt,
            whisper_model="Imported",
            media_type="ebook",
            overwrite=False
        )

        logging.info(f"Ebook '{title}' by {author} imported successfully. Database result: {result}")
        return f"Ebook '{title}' by {author} imported successfully. Database result: {result}"

    except Exception as e:
        logging.exception(f"Error importing ebook: {str(e)}")
        return f"Error importing ebook: {str(e)}"

# FIXME
def process_zip_file(zip_file, title, author, keywords, custom_prompt, system_prompt, summary, auto_summarize, api_name, api_key, chunk_options):
    """
    Processes a ZIP file containing multiple EPUB files and imports each one.

    Parameters:
        - zip_file (file-like object): The ZIP file to process.
        - title (str): Title prefix for the books.
        - author (str): Author name for the books.
        - keywords (str): Comma-separated keywords.
        - custom_prompt (str): Custom user prompt for summarization.
        - summary (str): Predefined summary (not used in this context).
        - auto_summarize (bool): Whether to auto-summarize the chunks.
        - api_name (str): API name for summarization.
        - api_key (str): API key for summarization.
        - chunk_options (dict): Options for chunking.

    Returns:
        - str: Combined status messages for all EPUB files in the ZIP.
    """
    results = []
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = zip_file.name if hasattr(zip_file, 'name') else zip_file.path
            logging.info(f"Extracting ZIP file {zip_path} to temporary directory {temp_dir}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            for filename in os.listdir(temp_dir):
                if filename.lower().endswith('.epub'):
                    file_path = os.path.join(temp_dir, filename)
                    logging.info(f"Processing EPUB file {filename} from ZIP.")
                    result = import_epub(
                        file_path=file_path,
                        title=title,
                        author=author,
                        keywords=keywords,
                        custom_prompt=custom_prompt,
                        summary=summary,
                        auto_summarize=auto_summarize,
                        api_name=api_name,
                        api_key=api_key,
                        chunk_options=chunk_options,
                        custom_chapter_pattern=chunk_options.get('custom_chapter_pattern') if chunk_options else None
                    )
                    results.append(f"File: {filename} - {result}")
            logging.info("Completed processing all EPUB files in the ZIP.")
    except Exception as e:
        logging.exception(f"Error processing ZIP file: {str(e)}")
        return f"Error processing ZIP file: {str(e)}"

    return "\n".join(results)

def read_epub(file_path):
    """
    Reads and extracts text from an EPUB file.

    Parameters:
        - file_path (str): Path to the EPUB file.

    Returns:
        - str: Extracted text content from the EPUB.
    """
    try:
        logging.info(f"Reading EPUB file from {file_path}")
        book = epub.read_epub(file_path)
        chapters = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                chapters.append(item.get_content())

        text = ""
        for html_content in chapters:
            soup = BeautifulSoup(html_content, 'html.parser')
            text += soup.get_text(separator='\n\n') + "\n\n"
        logging.debug("EPUB content extraction completed.")
        return text
    except Exception as e:
        logging.exception(f"Error reading EPUB file: {str(e)}")
        raise


# Ingest a text file into the database with Title/Author/Keywords
def extract_epub_metadata(content):
    title_match = re.search(r'Title:\s*(.*?)\n', content)
    author_match = re.search(r'Author:\s*(.*?)\n', content)

    title = title_match.group(1) if title_match else None
    author = author_match.group(1) if author_match else None

    return title, author


def ingest_text_file(file_path, title=None, author=None, keywords=None):
    """
    Ingests a plain text file into the database with optional metadata.

    Parameters:
        - file_path (str): Path to the text file.
        - title (str, optional): Title of the document.
        - author (str, optional): Author of the document.
        - keywords (str, optional): Comma-separated keywords.

    Returns:
        - str: Status message indicating success or failure.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Check if it's a converted epub and extract metadata if so
        if 'epub_converted' in (keywords or '').lower():
            extracted_title, extracted_author = extract_epub_metadata(content)
            title = title or extracted_title
            author = author or extracted_author
            logging.debug(f"Extracted metadata for converted EPUB - Title: {title}, Author: {author}")

        # If title is still not provided, use the filename without extension
        if not title:
            title = os.path.splitext(os.path.basename(file_path))[0]

        # If author is still not provided, set it to 'Unknown'
        if not author:
            author = 'Unknown'

        # If keywords are not provided, use a default keyword
        if not keywords:
            keywords = 'text_file,epub_converted'
        else:
            keywords = f'text_file,epub_converted,{keywords}'

        # Add the text file to the database
        add_media_with_keywords(
            url=file_path,
            title=title,
            media_type='document',
            content=content,
            keywords=keywords,
            prompt='No prompt for text files',
            summary='No summary for text files',
            transcription_model='None',
            author=author,
            ingestion_date=datetime.now().strftime('%Y-%m-%d')
        )

        logging.info(f"Text file '{title}' by {author} ingested successfully.")
        return f"Text file '{title}' by {author} ingested successfully."
    except Exception as e:
        logging.error(f"Error ingesting text file: {str(e)}")
        return f"Error ingesting text file: {str(e)}"


def ingest_folder(folder_path, keywords=None):
    """
    Ingests all text files within a specified folder.

    Parameters:
        - folder_path (str): Path to the folder containing text files.
        - keywords (str, optional): Comma-separated keywords to add to each file.

    Returns:
        - str: Combined status messages for all ingested text files.
    """
    results = []
    try:
        logging.info(f"Ingesting all text files from folder {folder_path}")
        for filename in os.listdir(folder_path):
            if filename.lower().endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                result = ingest_text_file(file_path, keywords=keywords)
                results.append(result)
        logging.info("Completed ingestion of all text files in the folder.")
    except Exception as e:
        logging.exception(f"Error ingesting folder: {str(e)}")
        return f"Error ingesting folder: {str(e)}"

    return "\n".join(results)


def epub_to_markdown(epub_path):
    """
    Converts an EPUB file to Markdown format, including the table of contents and chapter contents.

    Parameters:
        - epub_path (str): Path to the EPUB file.

    Returns:
        - str: Markdown-formatted content of the EPUB.
    """
    try:
        logging.info(f"Converting EPUB to Markdown from {epub_path}")
        book = epub.read_epub(epub_path)
        markdown_content = "# Table of Contents\n\n"
        chapters = []

        # Extract and format the table of contents
        toc = book.toc
        for item in toc:
            if isinstance(item, tuple):
                section, children = item
                level = 1
                markdown_content += format_toc_item(section, level)
                for child in children:
                    markdown_content += format_toc_item(child, level + 1)
            else:
                markdown_content += format_toc_item(item, 1)

        markdown_content += "\n---\n\n"

        # Process each chapter
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                chapter_content = item.get_content().decode('utf-8')
                soup = BeautifulSoup(chapter_content, 'html.parser')

                # Extract chapter title
                title = soup.find(['h1', 'h2', 'h3'])
                if title:
                    chapter_title = title.get_text()
                    markdown_content += f"# {chapter_title}\n\n"

                # Process chapter content
                for elem in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol']):
                    if elem.name.startswith('h'):
                        level = int(elem.name[1])
                        markdown_content += f"{'#' * level} {elem.get_text()}\n\n"
                    elif elem.name == 'p':
                        markdown_content += f"{elem.get_text()}\n\n"
                    elif elem.name in ['ul', 'ol']:
                        for li in elem.find_all('li'):
                            prefix = '-' if elem.name == 'ul' else '1.'
                            markdown_content += f"{prefix} {li.get_text()}\n"
                        markdown_content += "\n"

                markdown_content += "---\n\n"

        logging.debug("EPUB to Markdown conversion completed.")
        return markdown_content

    except Exception as e:
        logging.exception(f"Error converting EPUB to Markdown: {str(e)}")
        raise


def format_toc_item(item, level):
    """
    Formats a table of contents item into Markdown list format.

    Parameters:
        - item (epub.Link or epub.Section): TOC item.
        - level (int): Heading level for indentation.

    Returns:
        - str: Markdown-formatted TOC item.
    """
    try:
        if isinstance(item, epub.Link):
            title = item.title
        elif isinstance(item, epub.Section):
            title = item.title
        else:
            title = str(item)

        return f"{'  ' * (level - 1)}- [{title}](#{slugify(title)})\n"
    except Exception as e:
        logging.exception(f"Error formatting TOC item: {str(e)}")
        return ""


def slugify(text):
    """
    Converts a string into a slug suitable for Markdown links.

    Parameters:
        - text (str): The text to slugify.

    Returns:
        - str: Slugified text.
    """
    return re.sub(r'[\W_]+', '-', text.lower()).strip('-')

#
# End of Function Definitions
#######################################################################################################################
