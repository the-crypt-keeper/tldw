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


# Import necessary libraries
import os
import re
from datetime import datetime
import logging


# Import Local
from SQLite_DB import *

#######################################################################################################################
# Function Definitions
#

# Ingest a text file into the database with Title/Author/Keywords

def extract_epub_metadata(content):
    title_match = re.search(r'Title:\s*(.*?)\n', content)
    author_match = re.search(r'Author:\s*(.*?)\n', content)

    title = title_match.group(1) if title_match else None
    author = author_match.group(1) if author_match else None

    return title, author


def ingest_text_file(file_path, title=None, author=None, keywords=None):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Check if it's a converted epub and extract metadata if so
        if 'epub_converted' in (keywords or ''):
            extracted_title, extracted_author = extract_epub_metadata(content)
            title = title or extracted_title
            author = author or extracted_author

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

        return f"Text file '{title}' by {author} ingested successfully."
    except Exception as e:
        logging.error(f"Error ingesting text file: {str(e)}")
        return f"Error ingesting text file: {str(e)}"


def ingest_folder(folder_path, keywords=None):
    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            result = ingest_text_file(file_path, keywords=keywords)
            results.append(result)



