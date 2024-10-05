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
import re
import os
import shutil
import tempfile
from datetime import datetime
import pymupdf
import logging
#
# Import Local
from App_Function_Libraries.DB.DB_Manager import add_media_with_keywords
# Constants
MAX_FILE_SIZE_MB = 50
CONVERSION_TIMEOUT_SECONDS = 300
#
#######################################################################################################################
# Function Definitions
#

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
