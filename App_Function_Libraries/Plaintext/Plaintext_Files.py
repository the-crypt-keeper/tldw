# Plaintext_Files.py
# Description: This file contains functions for reading and writing plaintext files.
#
# Import necessary libraries
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
from App_Function_Libraries.Gradio_UI.Import_Functionality import import_data
from App_Function_Libraries.Metrics.metrics_logger import log_counter, log_histogram


#
#######################################################################################################################
#
# Function Definitions

def import_plain_text_file(file_path, author, keywords, system_prompt, user_prompt, auto_summarize, api_name, api_key):
    """
    Import a single plain text (or .docx/.rtf) file and pass it to import_data().
    Returns a single string result message.
    """
    try:
        log_counter("file_processing_attempt", labels={"file_path": file_path})

        # Extract title from filename
        title = os.path.splitext(os.path.basename(file_path))[0]

        # Determine the file type and convert if necessary
        file_extension = os.path.splitext(file_path)[1].lower()

        # Read the content, converting if necessary
        try:
            if file_extension == '.rtf':
                # Convert RTF to Markdown (via pypandoc), then read
                with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as temp_file:
                    convert_file(file_path, 'md', outputfile=temp_file.name)
                    file_path_md = temp_file.name
                with open(file_path_md, 'r', encoding='utf-8') as f:
                    content = f.read()
                log_counter("rtf_conversion_success", labels={"file_path": file_path})

            elif file_extension == '.docx':
                content = docx2txt.process(file_path)
                log_counter("docx_conversion_success", labels={"file_path": file_path})

            else:
                # Plain text / .md
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

        except Exception as e:
            logging.error(f"Error reading file content: {str(e)}")
            return f"Error reading file content: {str(e)}"

        # Pass to import_data.
        # Make sure system_prompt is passed if your import_data function expects it.
        result = import_data(
            file=content,
            title=title,
            author=author,
            keywords=keywords,
            custom_prompt=system_prompt + "\n\nUser Prompt:\n" + user_prompt,
            summary=None,
            auto_summarize=auto_summarize,
            api_name=api_name,
            api_key=api_key
        )

        log_counter("file_processing_success", labels={"file_path": file_path})
        return result

    except Exception as e:
        logging.exception(f"Error processing file {file_path}")
        log_counter("file_processing_error", labels={"file_path": file_path, "error": str(e)})
        return f"Error processing file {os.path.basename(file_path)}: {str(e)}"


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

