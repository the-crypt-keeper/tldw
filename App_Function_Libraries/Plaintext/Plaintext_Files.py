# Plaintext_Files.py
# Description: This file contains functions for reading and writing plaintext files.
#
# Import necessary libraries
import os
import re
from datetime import datetime
import logging
import tempfile
import zipfile

from docx2txt import docx2txt
from pypandoc import convert_file

from App_Function_Libraries.Gradio_UI.Import_Functionality import import_data


#
# Non-Local Imports
#
# Local Imports
#
#######################################################################################################################
#
# Function Definitions

def import_plain_text_file(file_path, title, author, keywords, system_prompt, user_prompt, auto_summarize, api_name,
                           api_key):
    try:
        # Determine the file type and convert if necessary
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.rtf':
            with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as temp_file:
                convert_file(file_path, 'md', outputfile=temp_file.name)
                file_path = temp_file.name
        elif file_extension == '.docx':
            content = docx2txt.process(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

        # Process the content
        return import_data(content, title, author, keywords, system_prompt,
                           user_prompt, auto_summarize, api_name, api_key)
    except Exception as e:
        return f"Error processing file: {str(e)}"

def process_plain_text_zip_file(zip_file, title, author, keywords, system_prompt, user_prompt, auto_summarize, api_name, api_key):
    results = []
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_file.name, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        for filename in os.listdir(temp_dir):
            if filename.lower().endswith(('.md', '.txt', '.rtf', '.docx')):
                file_path = os.path.join(temp_dir, filename)
                result = import_plain_text_file(file_path, title, author, keywords, system_prompt,
                                                user_prompt, auto_summarize, api_name, api_key)
                results.append(f"File: {filename} - {result}")

    return "\n".join(results)


def import_file_handler(file, title, author, keywords, system_prompt, user_prompt, auto_summarize, api_name, api_key):
    if file.name.lower().endswith(('.md', '.txt', '.rtf', '.docx')):
        return import_plain_text_file(file.name, title, author, keywords, system_prompt, user_prompt, auto_summarize,
                                      api_name, api_key)
    elif file.name.lower().endswith('.zip'):
        return process_plain_text_zip_file(file, title, author, keywords, system_prompt, user_prompt, auto_summarize,
                                           api_name, api_key)
    else:
        return "Unsupported file type. Please upload a .md, .txt, .rtf, .docx file or a .zip file containing these file types."


