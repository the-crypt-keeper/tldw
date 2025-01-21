# Plaintext_Files.py
# Description: This file contains functions for reading and writing plaintext files.
#
# Import necessary libraries
import json
import logging
import os
import tempfile
import zipfile
from datetime import datetime

#
# External Imports
from docx2txt import docx2txt
from pypandoc import convert_file
#
# Local Imports
from App_Function_Libraries.Metrics.metrics_logger import log_counter, log_histogram
#
#######################################################################################################################
#
# Function Definitions

def process_plain_text_zip_file(zip_file, author, keywords, system_prompt, user_prompt, auto_summarize, api_name, api_key):
    """Process multiple text files from a zip archive."""
    results = []
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_file.name, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            for filename in os.listdir(temp_dir):
                if filename.lower().endswith(('.md', '.txt', '.rtf', '.docx')):
                    file_path = os.path.join(temp_dir, filename)
                    result = import_plain_text_file(
                        file_path=file_path,
                        author=author,
                        keywords=keywords,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        auto_summarize=auto_summarize,
                        api_name=api_name,
                        api_key=api_key
                    )
                    results.append(f"📄 {filename}: {result}")

        return "\n\n".join(results)
    except Exception as e:
        logging.exception(f"Error processing zip file: {str(e)}")
        return f"Error processing zip file: {str(e)}"


def import_file_handler(files, author, keywords, system_prompt, user_prompt, auto_summarize, api_name, api_key):
    """
    Handle file imports with proper temp file handling.
    This function is wired to the Gradio import button; it must return a single string
    if we only have one output in the UI.
    """
    try:
        if not files:
            return "No files uploaded."

        results = []

        with tempfile.TemporaryDirectory() as temp_dir:
            for file_obj in files:
                filename = os.path.basename(file_obj.name)
                try:
                    # Make a temporary copy
                    temp_path = os.path.join(temp_dir, filename)
                    with open(temp_path, 'wb') as f:
                        f.write(open(file_obj.name, 'rb').read())

                    # Check if it's a zip
                    if temp_path.lower().endswith('.zip'):
                        with tempfile.TemporaryDirectory() as zip_temp_dir:
                            with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                                zip_ref.extractall(zip_temp_dir)

                            file_count = 0
                            for root, _, extracted_files in os.walk(zip_temp_dir):
                                for extracted_filename in extracted_files:
                                    if extracted_filename.lower().endswith(('.md', '.txt', '.rtf', '.docx')):
                                        extracted_path = os.path.join(root, extracted_filename)
                                        # Import each file inside the zip
                                        result = import_plain_text_file(
                                            extracted_path,
                                            author,
                                            keywords,
                                            system_prompt,
                                            user_prompt,
                                            auto_summarize,
                                            api_name,
                                            api_key
                                        )
                                        results.append(f"📄 {extracted_filename}: {result}")
                                        file_count += 1

                            results.append(f"📦 {filename} => Processed {file_count} file(s).")

                    else:
                        # Single file import
                        result = import_plain_text_file(
                            temp_path,
                            author,
                            keywords,
                            system_prompt,
                            user_prompt,
                            auto_summarize,
                            api_name,
                            api_key
                        )
                        results.append(f"📄 {filename}: {result}")

                except Exception as e:
                    logging.exception(f"❌ Error with file: {filename}")
                    results.append(f"❌ {filename} => {str(e)}")

        return "\n".join(results)

    except Exception as e:
        logging.exception(f"❌ Error during import: {str(e)}")
        return f"❌ Error during import: {str(e)}"


#
# End of Plaintext_Files.py
#######################################################################################################################

