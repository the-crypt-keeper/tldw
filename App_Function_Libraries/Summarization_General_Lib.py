# Summarization_General_Lib.py
#########################################
# General Summarization Library
# This library is used to perform summarization.
#
####
import configparser
####################
# Function List
#
# 1. extract_text_from_segments(segments: List[Dict]) -> str
# 2. summarize_with_openai(api_key, file_path, custom_prompt_arg)
# 3. summarize_with_claude(api_key, file_path, model, custom_prompt_arg, max_retries=3, retry_delay=5)
# 4. summarize_with_cohere(api_key, file_path, model, custom_prompt_arg)
# 5. summarize_with_groq(api_key, file_path, model, custom_prompt_arg)
#
#
####################


# Import necessary libraries
import os
import logging
import time
import requests
from typing import List, Dict
import json
import configparser
from requests import RequestException

# Import Local
import summarize
from Article_Summarization_Lib import *
from Article_Extractor_Lib import *
from Audio_Transcription_Lib import *
from Chunk_Lib import *
from Diarization_Lib import *
from Local_File_Processing_Lib import *
from Local_LLM_Inference_Engine_Lib import *
from Local_Summarization_Lib import *
from Old_Chunking_Lib import *
from SQLite_DB import *
#from Summarization_General_Lib import *
from System_Checks_Lib import *
from Tokenization_Methods_Lib import *
from Video_DL_Ingestion_Lib import *
#from Web_UI_Lib import *


# Read configuration from file
config = configparser.ConfigParser()
config.read('../config.txt')

# API Keys
anthropic_api_key = config.get('API', 'anthropic_api_key', fallback=None)
logging.debug(f"Loaded Anthropic API Key: {anthropic_api_key}")

cohere_api_key = config.get('API', 'cohere_api_key', fallback=None)
logging.debug(f"Loaded cohere API Key: {cohere_api_key}")

groq_api_key = config.get('API', 'groq_api_key', fallback=None)
logging.debug(f"Loaded groq API Key: {groq_api_key}")

openai_api_key = config.get('API', 'openai_api_key', fallback=None)
logging.debug(f"Loaded openAI Face API Key: {openai_api_key}")

huggingface_api_key = config.get('API', 'huggingface_api_key', fallback=None)
logging.debug(f"Loaded HuggingFace Face API Key: {huggingface_api_key}")

openrouter_api_token = config.get('API', 'openrouter_api_token', fallback=None)
logging.debug(f"Loaded OpenRouter API Key: {openrouter_api_token}")

# Models
anthropic_model = config.get('API', 'anthropic_model', fallback='claude-3-sonnet-20240229')
cohere_model = config.get('API', 'cohere_model', fallback='command-r-plus')
groq_model = config.get('API', 'groq_model', fallback='llama3-70b-8192')
openai_model = config.get('API', 'openai_model', fallback='gpt-4-turbo')
huggingface_model = config.get('API', 'huggingface_model', fallback='CohereForAI/c4ai-command-r-plus')
openrouter_model = config.get('API', 'openrouter_model', fallback='mistralai/mistral-7b-instruct:free')


#######################################################################################################################
# Function Definitions
#

# FIXME
# def extract_text_from_segments(segments: List[Dict]) -> str:
#     """Extract text from segments."""
#     return " ".join([segment['text'] for segment in segments])


def extract_text_from_segments(segments):
    logging.debug(f"Segments received: {segments}")
    logging.debug(f"Type of segments: {type(segments)}")

    text = ""

    if isinstance(segments, dict):
        if 'segments' in segments:
            segment_list = segments['segments']
            if isinstance(segment_list, list):
                for segment in segment_list:
                    logging.debug(f"Current segment: {segment}")
                    logging.debug(f"Type of segment: {type(segment)}")
                    if 'Text' in segment:
                        text += segment['Text'] + " "
                    else:
                        logging.warning(f"Skipping segment due to missing 'Text' key: {segment}")
            else:
                logging.warning(f"Unexpected type of 'segments' value: {type(segment_list)}")
        else:
            logging.warning("'segments' key not found in the dictionary")
    else:
        logging.warning(f"Unexpected type of 'segments': {type(segments)}")

    return text.strip()


def summarize_with_openai(api_key, json_file_path, custom_prompt_arg):
    try:
        logging.debug("openai: Loading json data for summarization")
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        logging.debug(f"openai: Loaded data: {data}")
        logging.debug(f"openai: Type of data: {type(data)}")

        if isinstance(data, dict) and 'summary' in data:
            # If the loaded data is a dictionary and already contains a summary, return it
            logging.debug("openai: Summary already exists in the loaded data")
            return data['summary']

        # If the loaded data is a list of segment dictionaries, proceed with summarization
        segments = data

        open_ai_model = openai_model or 'gpt-4-turbo'

        logging.debug("openai: Extracting text from the segments")
        text = extract_text_from_segments(segments)

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        logging.debug(f"openai: API Key is: {api_key}")
        logging.debug("openai: Preparing data + prompt for submittal")
        openai_prompt = f"{text} \n\n\n\n{custom_prompt_arg}"
        data = {
            "model": open_ai_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional summarizer."
                },
                {
                    "role": "user",
                    "content": openai_prompt
                }
            ],
            "max_tokens": 4096,  # Adjust tokens as needed
            "temperature": 0.1
        }
        logging.debug("openai: Posting request")
        response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)

        if response.status_code == 200:
            response_data = response.json()
            if 'choices' in response_data and len(response_data['choices']) > 0:
                summary = response_data['choices'][0]['message']['content'].strip()
                logging.debug("openai: Summarization successful")
                print("openai: Summarization successful.")
                return summary
            else:
                logging.warning("openai: Summary not found in the response data")
                return "openai: Summary not available"
        else:
            logging.debug("openai: Summarization failed")
            print("openai: Failed to process summary:", response.text)
            return "openai: Failed to process summary"
    except Exception as e:
        logging.debug("openai: Error in processing: %s", str(e))
        print("openai: Error occurred while processing summary with openai:", str(e))
        return "openai: Error occurred while processing summary"


def summarize_with_claude(api_key, file_path, model, custom_prompt_arg, max_retries=3, retry_delay=5):
    try:
        logging.debug("anthropic: Loading JSON data")
        with open(file_path, 'r') as file:
            segments = json.load(file)

        logging.debug("anthropic: Extracting text from the segments file")
        text = extract_text_from_segments(segments)

        headers = {
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01',
            'Content-Type': 'application/json'
        }

        anthropic_prompt = custom_prompt_arg  # Sanitize the custom prompt
        logging.debug(f"anthropic: Prompt is {anthropic_prompt}")
        user_message = {
            "role": "user",
            "content": f"{text} \n\n\n\n{anthropic_prompt}"
        }

        data = {
            "model": model,
            "max_tokens": 4096,  # max _possible_ tokens to return
            "messages": [user_message],
            "stop_sequences": ["\n\nHuman:"],
            "temperature": 0.1,
            "top_k": 0,
            "top_p": 1.0,
            "metadata": {
                "user_id": "example_user_id",
            },
            "stream": False,
            "system": "You are a professional summarizer."
        }

        for attempt in range(max_retries):
            try:
                logging.debug("anthropic: Posting request to API")
                response = requests.post('https://api.anthropic.com/v1/messages', headers=headers, json=data)

                # Check if the status code indicates success
                if response.status_code == 200:
                    logging.debug("anthropic: Post submittal successful")
                    response_data = response.json()
                    try:
                        summary = response_data['content'][0]['text'].strip()
                        logging.debug("anthropic: Summarization successful")
                        print("Summary processed successfully.")
                        return summary
                    except (IndexError, KeyError) as e:
                        logging.debug("anthropic: Unexpected data in response")
                        print("Unexpected response format from Claude API:", response.text)
                        return None
                elif response.status_code == 500:  # Handle internal server error specifically
                    logging.debug("anthropic: Internal server error")
                    print("Internal server error from API. Retrying may be necessary.")
                    time.sleep(retry_delay)
                else:
                    logging.debug(
                        f"anthropic: Failed to summarize, status code {response.status_code}: {response.text}")
                    print(f"Failed to process summary, status code {response.status_code}: {response.text}")
                    return None

            except RequestException as e:
                logging.error(f"anthropic: Network error during attempt {attempt + 1}/{max_retries}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    return f"anthropic: Network error: {str(e)}"

    except FileNotFoundError as e:
        logging.error(f"anthropic: File not found: {file_path}")
        return f"anthropic: File not found: {file_path}"
    except json.JSONDecodeError as e:
        logging.error(f"anthropic: Invalid JSON format in file: {file_path}")
        return f"anthropic: Invalid JSON format in file: {file_path}"
    except Exception as e:
        logging.error(f"anthropic: Error in processing: {str(e)}")
        return f"anthropic: Error occurred while processing summary with Anthropic: {str(e)}"


# Summarize with Cohere
def summarize_with_cohere(api_key, file_path, model, custom_prompt_arg):
    try:
        logging.debug("cohere: Loading JSON data")
        with open(file_path, 'r') as file:
            segments = json.load(file)

        logging.debug(f"cohere: Extracting text from segments file")
        text = extract_text_from_segments(segments)

        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }

        cohere_prompt = f"{text} \n\n\n\n{custom_prompt_arg}"
        logging.debug("cohere: Prompt being sent is {cohere_prompt}")

        data = {
            "chat_history": [
                {"role": "USER", "message": cohere_prompt}
            ],
            "message": "Please provide a summary.",
            "model": model,
            "connectors": [{"id": "web-search"}]
        }

        logging.debug("cohere: Submitting request to API endpoint")
        print("cohere: Submitting request to API endpoint")
        response = requests.post('https://api.cohere.ai/v1/chat', headers=headers, json=data)
        response_data = response.json()
        logging.debug("API Response Data: %s", response_data)

        if response.status_code == 200:
            if 'text' in response_data:
                summary = response_data['text'].strip()
                logging.debug("cohere: Summarization successful")
                print("Summary processed successfully.")
                return summary
            else:
                logging.error("Expected data not found in API response.")
                return "Expected data not found in API response."
        else:
            logging.error(f"cohere: API request failed with status code {response.status_code}: {response.text}")
            print(f"Failed to process summary, status code {response.status_code}: {response.text}")
            return f"cohere: API request failed: {response.text}"

    except Exception as e:
        logging.error("cohere: Error in processing: %s", str(e))
        return f"cohere: Error occurred while processing summary with Cohere: {str(e)}"


# https://console.groq.com/docs/quickstart
def summarize_with_groq(api_key, file_path, model, custom_prompt_arg):
    try:
        logging.debug("groq: Loading JSON data")
        with open(file_path, 'r') as file:
            segments = json.load(file)

        logging.debug(f"groq: Extracting text from segments file")
        text = extract_text_from_segments(segments)

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        groq_prompt = f"{text} \n\n\n\n{custom_prompt_arg}"
        logging.debug("groq: Prompt being sent is {groq_prompt}")

        data = {
            "messages": [
                {
                    "role": "user",
                    "content": groq_prompt
                }
            ],
            "model": model
        }

        logging.debug("groq: Submitting request to API endpoint")
        print("groq: Submitting request to API endpoint")
        response = requests.post('https://api.groq.com/openai/v1/chat/completions', headers=headers, json=data)

        response_data = response.json()
        logging.debug("API Response Data: %s", response_data)

        if response.status_code == 200:
            if 'choices' in response_data and len(response_data['choices']) > 0:
                summary = response_data['choices'][0]['message']['content'].strip()
                logging.debug("groq: Summarization successful")
                print("Summarization successful.")
                return summary
            else:
                logging.error("Expected data not found in API response.")
                return "Expected data not found in API response."
        else:
            logging.error(f"groq: API request failed with status code {response.status_code}: {response.text}")
            return f"groq: API request failed: {response.text}"

    except Exception as e:
        logging.error("groq: Error in processing: %s", str(e))
        return f"groq: Error occurred while processing summary with groq: {str(e)}"


def summarize_with_openrouter(api_key, json_file_path, custom_prompt_arg):
    import requests
    import json
    global openrouter_model

    config = configparser.ConfigParser()
    file_path = 'config.txt'

    # Check if the file exists in the specified path
    if os.path.exists(file_path):
        config.read(file_path)
    elif os.path.exists('config.txt'):  # Check in the current directory
        config.read('../config.txt')
    else:
        print("config.txt not found in the specified path or current directory.")

    openrouter_api_token = config.get('API', 'openrouter_api_token', fallback=None)
    if openrouter_model is None:
        openrouter_model = "mistralai/mistral-7b-instruct:free"

    openrouter_prompt = f"{json_file_path} \n\n\n\n{custom_prompt_arg}"

    try:
        logging.debug("openrouter: Submitting request to API endpoint")
        print("openrouter: Submitting request to API endpoint")
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_api_token}",
            },
            data=json.dumps({
                "model": f"{openrouter_model}",
                "messages": [
                    {"role": "user", "content": openrouter_prompt}
                ]
            })
        )

        response_data = response.json()
        logging.debug("API Response Data: %s", response_data)

        if response.status_code == 200:
            if 'choices' in response_data and len(response_data['choices']) > 0:
                summary = response_data['choices'][0]['message']['content'].strip()
                logging.debug("openrouter: Summarization successful")
                print("openrouter: Summarization successful.")
                return summary
            else:
                logging.error("openrouter: Expected data not found in API response.")
                return "openrouter: Expected data not found in API response."
        else:
            logging.error(f"openrouter:  API request failed with status code {response.status_code}: {response.text}")
            return f"openrouter: API request failed: {response.text}"
    except Exception as e:
        logging.error("openrouter: Error in processing: %s", str(e))
        return f"openrouter: Error occurred while processing summary with openrouter: {str(e)}"

def summarize_with_huggingface(api_key, file_path, custom_prompt_arg):
    logging.debug(f"huggingface: Summarization process starting...")
    try:
        logging.debug("huggingface: Loading json data for summarization")
        with open(file_path, 'r') as file:
            segments = json.load(file)

        logging.debug("huggingface: Extracting text from the segments")
        logging.debug(f"huggingface: Segments: {segments}")
        text = ' '.join([segment['text'] for segment in segments])

        print(f"huggingface: lets make sure the HF api key exists...\n\t {api_key}")
        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        model = "microsoft/Phi-3-mini-128k-instruct"
        API_URL = f"https://api-inference.huggingface.co/models/{model}"

        huggingface_prompt = f"{text}\n\n\n\n{custom_prompt_arg}"
        logging.debug("huggingface: Prompt being sent is {huggingface_prompt}")
        data = {
            "inputs": text,
            "parameters": {"max_length": 512, "min_length": 100}  # You can adjust max_length and min_length as needed
        }

        print(f"huggingface: lets make sure the HF api key is the same..\n\t {huggingface_api_key}")

        logging.debug("huggingface: Submitting request...")

        response = requests.post(API_URL, headers=headers, json=data)

        if response.status_code == 200:
            summary = response.json()[0]['summary_text']
            logging.debug("huggingface: Summarization successful")
            print("Summarization successful.")
            return summary
        else:
            logging.error(f"huggingface: Summarization failed with status code {response.status_code}: {response.text}")
            return f"Failed to process summary, status code {response.status_code}: {response.text}"
    except Exception as e:
        logging.error("huggingface: Error in processing: %s", str(e))
        print(f"Error occurred while processing summary with huggingface: {str(e)}")
        return None

    # FIXME
    # This is here for gradio authentication
    # Its just not setup.
    # def same_auth(username, password):
    #    return username == password


#
#
#######################################################################################################################