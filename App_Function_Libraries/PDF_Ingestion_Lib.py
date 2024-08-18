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
import re

# Import necessary libraries


# Import Local

#######################################################################################################################
# Function Definitions
#

# Ingest a text file into the database with Title/Author/Keywords


# Constants
MAX_FILE_SIZE_MB = 50
CONVERSION_TIMEOUT_SECONDS = 300

# Marker PDF solution
# def convert_pdf_to_markdown(pdf_path):
#     """
#     Convert a PDF file to Markdown by calling a script in another virtual environment.
#     """
#
#     logging.debug(f"Marker: Converting PDF file to Markdown: {pdf_path}")
#     # Check if the file size exceeds the maximum allowed size
#     file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
#     if file_size_mb > MAX_FILE_SIZE_MB:
#         raise ValueError(f"File size ({file_size_mb:.2f} MB) exceeds the maximum allowed size of {MAX_FILE_SIZE_MB} MB")
#
#     logging.debug("Marker: Converting PDF file to Markdown using Marker virtual environment")
#     # Path to the Python interpreter in the other virtual environment
#     other_venv_python = "Helper_Scripts/marker_venv/bin/python"
#
#     # Path to the conversion script
#     converter_script = "Helper_Scripts/PDF_Converter.py"
#
#     logging.debug("Marker: Attempting to convert PDF file to Markdown...")
#     try:
#         result = subprocess.run(
#             [other_venv_python, converter_script, pdf_path],
#             capture_output=True,
#             text=True,
#             timeout=CONVERSION_TIMEOUT_SECONDS
#         )
#         if result.returncode != 0:
#             raise Exception(f"Conversion failed: {result.stderr}")
#         return result.stdout
#     except subprocess.TimeoutExpired:
#         raise Exception(f"PDF conversion timed out after {CONVERSION_TIMEOUT_SECONDS} seconds")
#
#
# def process_and_ingest_pdf(file, title, author, keywords):
#     if file is None:
#         return "Please select a PDF file to upload."
#
#     try:
#         # Create a temporary directory
#         with tempfile.TemporaryDirectory() as temp_dir:
#             # Create a path for the temporary PDF file
#             temp_path = os.path.join(temp_dir, "temp.pdf")
#
#             # Copy the contents of the uploaded file to the temporary file
#             shutil.copy(file.name, temp_path)
#
#             # Call the ingest_pdf_file function with the temporary file path
#             result = ingest_pdf_file(temp_path, title, author, keywords)
#
#         return result
#     except Exception as e:
#         return f"Error processing PDF: {str(e)}"
#
#
# def ingest_pdf_file(file_path, title=None, author=None, keywords=None):
#     try:
#         # Convert PDF to Markdown
#         markdown_content = convert_pdf_to_markdown(file_path)
#
#         # If title is not provided, use the filename without extension
#         if not title:
#             title = os.path.splitext(os.path.basename(file_path))[0]
#
#         # If author is not provided, set it to 'Unknown'
#         if not author:
#             author = 'Unknown'
#
#         # If keywords are not provided, use a default keyword
#         if not keywords:
#             keywords = 'pdf_file,markdown_converted'
#         else:
#             keywords = f'pdf_file,markdown_converted,{keywords}'
#
#         # Add the markdown content to the database
#         add_media_with_keywords(
#             url=file_path,
#             title=title,
#             media_type='document',
#             content=markdown_content,
#             keywords=keywords,
#             prompt='No prompt for PDF files',
#             summary='No summary for PDF files',
#             transcription_model='None',
#             author=author,
#             ingestion_date=datetime.now().strftime('%Y-%m-%d')
#         )
#
#         return f"PDF file '{title}' converted to Markdown and ingested successfully.", file_path
#     except ValueError as e:
#         logging.error(f"File size error: {str(e)}")
#         return f"Error: {str(e)}", file_path
#     except Exception as e:
#         logging.error(f"Error ingesting PDF file: {str(e)}")
#         return f"Error ingesting PDF file: {str(e)}", file_path
#
#
# def process_and_cleanup_pdf(file, title, author, keywords):
#     # FIXME - Update to validate file upload/filetype is pdf....
#     if file is None:
#         return "No file uploaded. Please upload a PDF file."
#
#     temp_dir = tempfile.mkdtemp()
#     temp_file_path = os.path.join(temp_dir, "temp.pdf")
#
#     try:
#         # Copy the uploaded file to a temporary location
#         shutil.copy2(file.name, temp_file_path)
#
#         # Process the file
#         result, _ = ingest_pdf_file(temp_file_path, title, author, keywords)
#
#         return result
#     except Exception as e:
#         logging.error(f"Error in processing and cleanup: {str(e)}")
#         return f"Error: {str(e)}"
#     finally:
#         # Clean up the temporary directory and its contents
#         try:
#             shutil.rmtree(temp_dir)
#             logging.info(f"Removed temporary directory: {temp_dir}")
#         except Exception as cleanup_error:
#             logging.error(f"Error during cleanup: {str(cleanup_error)}")
#             result += f"\nWarning: Could not remove temporary files: {str(cleanup_error)}"


import logging
#
#
#######################################################################################################################
#
# Non-Marker implementation
import os
import shutil
import tempfile
from datetime import datetime

import pymupdf

from App_Function_Libraries.SQLite_DB import add_media_with_keywords


def extract_text_and_format_from_pdf(pdf_path):
    """
    Extract text from a PDF file and convert it to Markdown, preserving formatting.
    """
    try:
        markdown_text = ""
        with pymupdf.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, 1):
                markdown_text += f"## Page {page_num}\n\n"
                blocks = page.get_text("dict")["blocks"]
                current_paragraph = ""
                for block in blocks:
                    if block["type"] == 0:  # Text block
                        for line in block["lines"]:
                            line_text = ""
                            for span in line["spans"]:
                                text = span["text"]
                                font_size = span["size"]
                                font_flags = span["flags"]

                                # Apply formatting based on font size and flags
                                if font_size > 20:
                                    text = f"# {text}"
                                elif font_size > 16:
                                    text = f"## {text}"
                                elif font_size > 14:
                                    text = f"### {text}"

                                if font_flags & 2 ** 0:  # Bold
                                    text = f"**{text}**"
                                if font_flags & 2 ** 1:  # Italic
                                    text = f"*{text}*"

                                line_text += text + " "

                            # Remove hyphens at the end of lines
                            line_text = line_text.rstrip()
                            if line_text.endswith('-'):
                                line_text = line_text[:-1]
                            else:
                                line_text += " "

                            current_paragraph += line_text

                        # End of block, add paragraph
                        if current_paragraph:
                            # Remove extra spaces
                            current_paragraph = re.sub(r'\s+', ' ', current_paragraph).strip()
                            markdown_text += current_paragraph + "\n\n"
                            current_paragraph = ""
                    elif block["type"] == 1:  # Image block
                        markdown_text += "[Image]\n\n"
                markdown_text += "\n---\n\n"  # Page separator

        # Clean up hyphenated words
        markdown_text = re.sub(r'(\w+)-\s*\n(\w+)', r'\1\2', markdown_text)

        return markdown_text
    except Exception as e:
        logging.error(f"Error extracting text and formatting from PDF: {str(e)}")
        raise


def extract_metadata_from_pdf(pdf_path):
    """
    Extract metadata from a PDF file using PyMuPDF.
    """
    try:
        with pymupdf.open(pdf_path) as doc:
            metadata = doc.metadata
        return metadata
    except Exception as e:
        logging.error(f"Error extracting metadata from PDF: {str(e)}")
        return {}


def process_and_ingest_pdf(file, title, author, keywords):
    if file is None:
        return "Please select a PDF file to upload."

    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a path for the temporary PDF file
            temp_path = os.path.join(temp_dir, "temp.pdf")

            # Copy the contents of the uploaded file to the temporary file
            shutil.copy(file.name, temp_path)

            # Extract text and convert to Markdown
            markdown_text = extract_text_and_format_from_pdf(temp_path)

            # Extract metadata from PDF
            metadata = extract_metadata_from_pdf(temp_path)

            # Use metadata for title and author if not provided
            if not title:
                title = metadata.get('title', os.path.splitext(os.path.basename(file.name))[0])
            if not author:
                author = metadata.get('author', 'Unknown')

            # If keywords are not provided, use a default keyword
            if not keywords:
                keywords = 'pdf_file,markdown_converted'
            else:
                keywords = f'pdf_file,markdown_converted,{keywords}'

            # Add metadata-based keywords
            if 'subject' in metadata:
                keywords += f",{metadata['subject']}"

            # Add the PDF content to the database
            add_media_with_keywords(
                url=file.name,
                title=title,
                media_type='document',
                content=markdown_text,
                keywords=keywords,
                prompt='No prompt for PDF files',
                summary='No summary for PDF files',
                transcription_model='None',
                author=author,
                ingestion_date=datetime.now().strftime('%Y-%m-%d')
            )

        return f"PDF file '{title}' by {author} ingested successfully and converted to Markdown."
    except Exception as e:
        logging.error(f"Error ingesting PDF file: {str(e)}")
        return f"Error ingesting PDF file: {str(e)}"


def process_and_cleanup_pdf(file, title, author, keywords):
    if file is None:
        return "No file uploaded. Please upload a PDF file."

    try:
        result = process_and_ingest_pdf(file, title, author, keywords)
        return result
    except Exception as e:
        logging.error(f"Error in processing and cleanup: {str(e)}")
        return f"Error: {str(e)}"

#
# End of PDF_Ingestion_Lib.py
#######################################################################################################################