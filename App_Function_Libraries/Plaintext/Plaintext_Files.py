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
    """Import a single plain text file."""
    try:
        log_counter("file_processing_attempt", labels={"file_path": file_path})

        # Extract title from filename
        title = os.path.splitext(os.path.basename(file_path))[0]

        # Determine the file type and convert if necessary
        file_extension = os.path.splitext(file_path)[1].lower()

        # Get the content based on file type
        try:
            if file_extension == '.rtf':
                with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as temp_file:
                    convert_file(file_path, 'md', outputfile=temp_file.name)
                    file_path = temp_file.name
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                log_counter("rtf_conversion_success", labels={"file_path": file_path})
            elif file_extension == '.docx':
                content = docx2txt.process(file_path)
                log_counter("docx_conversion_success", labels={"file_path": file_path})
            else:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
        except Exception as e:
            logging.error(f"Error reading file content: {str(e)}")
            return f"Error reading file content: {str(e)}"

        # Import the content
        result = import_data(
            content,  # Pass the content directly
            title,
            author,
            keywords,
            user_prompt,  # This is the custom_prompt parameter
            None,        # No summary - let auto_summarize handle it
            auto_summarize,
            api_name,
            api_key
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
    """Handle the import of one or more files, including zip files."""
    try:
        if not files:
            log_counter("plaintext_import_error", labels={"error": "No files uploaded"})
            return "No files uploaded."

        # Convert single file to list for consistent processing
        if not isinstance(files, list):
            files = [files]

        results = []
        for file in files:
            log_counter("plaintext_import_attempt", labels={"file_name": file.name})

            start_time = datetime.now()

            if not os.path.exists(file.name):
                log_counter("plaintext_import_error", labels={"error": "File not found", "file_name": file.name})
                results.append(f"❌ File not found: {file.name}")
                continue

            if file.name.lower().endswith(('.md', '.txt', '.rtf', '.docx')):
                result = import_plain_text_file(
                    file_path=file.name,
                    author=author,
                    keywords=keywords,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    auto_summarize=auto_summarize,
                    api_name=api_name,
                    api_key=api_key
                )
                log_counter("plaintext_import_success", labels={"file_name": file.name})
                results.append(f"📄 {file.name}: {result}")

            elif file.name.lower().endswith('.zip'):
                result = process_plain_text_zip_file(
                    zip_file=file,
                    author=author,
                    keywords=keywords,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    auto_summarize=auto_summarize,
                    api_name=api_name,
                    api_key=api_key
                )
                log_counter("zip_import_success", labels={"file_name": file.name})
                results.append(f"📦 {file.name}:\n{result}")

            else:
                log_counter("unsupported_file_type", labels={"file_type": file.name.split('.')[-1]})
                results.append(f"❌ Unsupported file type: {file.name}")
                continue

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            log_histogram("plaintext_import_duration", processing_time, labels={"file_name": file.name})

        return "\n\n".join(results)

    except Exception as e:
        logging.exception("Error in import_file_handler")
        log_counter("plaintext_import_error", labels={"error": str(e)})
        return f"❌ Error during import: {str(e)}"

#
# End of Plaintext_Files.py
#######################################################################################################################

