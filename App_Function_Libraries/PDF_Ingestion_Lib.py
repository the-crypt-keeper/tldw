# PDF_Ingestion_Lib.py
#########################################
# Library to hold functions for ingesting PDF files.#
#
####################
# Function List
#
# 1. convert_pdf_to_markdown(pdf_path)
# 2. ingest_pdf_file(file_path, title=None, author=None, keywords=None):
# 3.
#
#
####################


# Import necessary libraries
from datetime import datetime
import logging
import subprocess
import os



# Import Local
from App_Function_Libraries.SQLite_DB import add_media_with_keywords

#######################################################################################################################
# Function Definitions
#

# Ingest a text file into the database with Title/Author/Keywords


# Constants
MAX_FILE_SIZE_MB = 50
CONVERSION_TIMEOUT_SECONDS = 300


def convert_pdf_to_markdown(pdf_path):
    """
    Convert a PDF file to Markdown by calling a script in another virtual environment.
    """
    file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"File size ({file_size_mb:.2f} MB) exceeds the maximum allowed size of {MAX_FILE_SIZE_MB} MB")

    # Path to the Python interpreter in the other virtual environment
    other_venv_python = "Helper_Scripts/marker_venv/bin/python"

    # Path to the conversion script
    converter_script = "Helper_Scripts/PDF_Converter.py"

    try:
        result = subprocess.run(
            [other_venv_python, converter_script, pdf_path],
            capture_output=True,
            text=True,
            timeout=CONVERSION_TIMEOUT_SECONDS
        )
        if result.returncode != 0:
            raise Exception(f"Conversion failed: {result.stderr}")
        return result.stdout
    except subprocess.TimeoutExpired:
        raise Exception(f"PDF conversion timed out after {CONVERSION_TIMEOUT_SECONDS} seconds")


def ingest_pdf_file(file_path, title=None, author=None, keywords=None):
    try:
        # Convert PDF to Markdown
        markdown_content = convert_pdf_to_markdown(file_path)

        # If title is not provided, use the filename without extension
        if not title:
            title = os.path.splitext(os.path.basename(file_path))[0]

        # If author is not provided, set it to 'Unknown'
        if not author:
            author = 'Unknown'

        # If keywords are not provided, use a default keyword
        if not keywords:
            keywords = 'pdf_file,markdown_converted'
        else:
            keywords = f'pdf_file,markdown_converted,{keywords}'

        # Add the markdown content to the database
        add_media_with_keywords(
            url=file_path,
            title=title,
            media_type='document',
            content=markdown_content,
            keywords=keywords,
            prompt='No prompt for PDF files',
            summary='No summary for PDF files',
            transcription_model='None',
            author=author,
            ingestion_date=datetime.now().strftime('%Y-%m-%d')
        )

        return f"PDF file '{title}' converted to Markdown and ingested successfully."
    except ValueError as e:
        logging.error(f"File size error: {str(e)}")
        return f"Error: {str(e)}"
    except Exception as e:
        logging.error(f"Error ingesting PDF file: {str(e)}")
        return f"Error ingesting PDF file: {str(e)}"