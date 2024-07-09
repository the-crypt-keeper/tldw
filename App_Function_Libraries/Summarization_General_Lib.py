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
# 3. summarize_with_anthropic(api_key, file_path, model, custom_prompt_arg, max_retries=3, retry_delay=5)
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
import json
from requests import RequestException

from App_Function_Libraries.Audio_Transcription_Lib import convert_to_wav, speech_to_text
from App_Function_Libraries.Diarization_Lib import combine_transcription_and_diarization
from App_Function_Libraries.Local_Summarization_Lib import summarize_with_llama, summarize_with_kobold, \
    summarize_with_oobabooga, summarize_with_tabbyapi, summarize_with_vllm, summarize_with_local_llm
from App_Function_Libraries.SQLite_DB import is_valid_url, add_media_to_database
# Import Local
from App_Function_Libraries.Utils import load_and_log_configs, load_comprehensive_config, sanitize_filename, \
    clean_youtube_url, extract_video_info, create_download_directory
from App_Function_Libraries.Video_DL_Ingestion_Lib import download_video

#
#######################################################################################################################
# Function Definitions
#
config = load_comprehensive_config()
openai_api_key = config.get('API', 'openai_api_key', fallback=None)

def extract_text_from_segments(segments):
    logging.debug(f"Segments received: {segments}")
    logging.debug(f"Type of segments: {type(segments)}")

    text = ""

    if isinstance(segments, list):
        for segment in segments:
            logging.debug(f"Current segment: {segment}")
            logging.debug(f"Type of segment: {type(segment)}")
            if 'Text' in segment:
                text += segment['Text'] + " "
            else:
                logging.warning(f"Skipping segment due to missing 'Text' key: {segment}")
    else:
        logging.warning(f"Unexpected type of 'segments': {type(segments)}")

    return text.strip()


def summarize_with_openai(api_key, input_data, custom_prompt_arg):
    loaded_config_data = load_and_log_configs()
    try:
        # API key validation
        if api_key is None or api_key.strip() == "":
            logging.info("OpenAI: API key not provided as parameter")
            logging.info("OpenAI: Attempting to use API key from config file")
            api_key = loaded_config_data['api_keys']['openai']

        if api_key is None or api_key.strip() == "":
            logging.error("OpenAI: API key not found or is empty")
            return "OpenAI: API Key Not Provided/Found in Config file or is empty"

        logging.debug(f"OpenAI: Using API Key: {api_key[:5]}...{api_key[-5:]}")

        # Input data handling
        if isinstance(input_data, str) and os.path.isfile(input_data):
            logging.debug("OpenAI: Loading json data for summarization")
            with open(input_data, 'r') as file:
                data = json.load(file)
        else:
            logging.debug("OpenAI: Using provided string data for summarization")
            data = input_data

        logging.debug(f"OpenAI: Loaded data: {data}")
        logging.debug(f"OpenAI: Type of data: {type(data)}")

        if isinstance(data, dict) and 'summary' in data:
            # If the loaded data is a dictionary and already contains a summary, return it
            logging.debug("OpenAI: Summary already exists in the loaded data")
            return data['summary']

        # Text extraction
        if isinstance(data, list):
            segments = data
            text = extract_text_from_segments(segments)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError("OpenAI: Invalid input data format")

        openai_model = loaded_config_data['models']['openai'] or "gpt-4o"

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        logging.debug(
            f"OpenAI API Key: {openai_api_key[:5]}...{openai_api_key[-5:] if openai_api_key else None}")
        logging.debug("openai: Preparing data + prompt for submittal")
        openai_prompt = f"{text} \n\n\n\n{custom_prompt_arg}"
        data = {
            "model": openai_model,
            "messages": [
                {"role": "system", "content": "You are a professional summarizer."},
                {"role": "user", "content": openai_prompt}
            ],
            "max_tokens": 4096,
            "temperature": 0.1
        }

        logging.debug("openai: Posting request")
        response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)

        if response.status_code == 200:
            response_data = response.json()
            if 'choices' in response_data and len(response_data['choices']) > 0:
                summary = response_data['choices'][0]['message']['content'].strip()
                logging.debug("openai: Summarization successful")
                return summary
            else:
                logging.warning("openai: Summary not found in the response data")
                return "openai: Summary not available"
        else:
            logging.error(f"openai: Summarization failed with status code {response.status_code}")
            logging.error(f"openai: Error response: {response.text}")
            return f"openai: Failed to process summary. Status code: {response.status_code}"
    except Exception as e:
        logging.error(f"openai: Error in processing: {str(e)}", exc_info=True)
        return f"openai: Error occurred while processing summary: {str(e)}"


def summarize_with_anthropic(api_key, input_data, custom_prompt_arg, max_retries=3, retry_delay=5):
    try:
        loaded_config_data = load_and_log_configs()
        global anthropic_api_key
        # API key validation
        if api_key is None:
            logging.info("Anthropic: API key not provided as parameter")
            logging.info("Anthropic: Attempting to use API key from config file")
            anthropic_api_key = loaded_config_data['api_keys']['anthropic']

        if api_key is None or api_key.strip() == "":
            logging.error("Anthropic: API key not found or is empty")
            return "Anthropic: API Key Not Provided/Found in Config file or is empty"

        logging.debug(f"Anthropic: Using API Key: {api_key[:5]}...{api_key[-5:]}")

        if isinstance(input_data, str) and os.path.isfile(input_data):
            logging.debug("AnthropicAI: Loading json data for summarization")
            with open(input_data, 'r') as file:
                data = json.load(file)
        else:
            logging.debug("AnthropicAI: Using provided string data for summarization")
            data = input_data

        logging.debug(f"AnthropicAI: Loaded data: {data}")
        logging.debug(f"AnthropicAI: Type of data: {type(data)}")

        if isinstance(data, dict) and 'summary' in data:
            # If the loaded data is a dictionary and already contains a summary, return it
            logging.debug("Anthropic: Summary already exists in the loaded data")
            return data['summary']

        # If the loaded data is a list of segment dictionaries or a string, proceed with summarization
        if isinstance(data, list):
            segments = data
            text = extract_text_from_segments(segments)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError("Anthropic: Invalid input data format")

        anthropic_model = loaded_config_data['models']['anthropic']

        headers = {
            'x-api-key': anthropic_api_key,
            'anthropic-version': '2023-06-01',
            'Content-Type': 'application/json'
        }

        anthropic_prompt = custom_prompt_arg
        logging.debug(f"Anthropic: Prompt is {anthropic_prompt}")
        user_message = {
            "role": "user",
            "content": f"{text} \n\n\n\n{anthropic_prompt}"
        }

        model = loaded_config_data['models']['anthropic']

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
                        print("Unexpected response format from Anthropic API:", response.text)
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
        logging.error(f"anthropic: File not found: {input_data}")
        return f"anthropic: File not found: {input_data}"
    except json.JSONDecodeError as e:
        logging.error(f"anthropic: Invalid JSON format in file: {input_data}")
        return f"anthropic: Invalid JSON format in file: {input_data}"
    except Exception as e:
        logging.error(f"anthropic: Error in processing: {str(e)}")
        return f"anthropic: Error occurred while processing summary with Anthropic: {str(e)}"


# Summarize with Cohere
def summarize_with_cohere(api_key, input_data, custom_prompt_arg):
    loaded_config_data = load_and_log_configs()
    try:
        # API key validation
        if api_key is None:
            logging.info("cohere: API key not provided as parameter")
            logging.info("cohere: Attempting to use API key from config file")
            cohere_api_key = loaded_config_data['api_keys']['cohere']

        if api_key is None or api_key.strip() == "":
            logging.error("cohere: API key not found or is empty")
            return "cohere: API Key Not Provided/Found in Config file or is empty"

        logging.debug(f"cohere: Using API Key: {api_key[:5]}...{api_key[-5:]}")

        if isinstance(input_data, str) and os.path.isfile(input_data):
            logging.debug("Cohere: Loading json data for summarization")
            with open(input_data, 'r') as file:
                data = json.load(file)
        else:
            logging.debug("Cohere: Using provided string data for summarization")
            data = input_data

        logging.debug(f"Cohere: Loaded data: {data}")
        logging.debug(f"Cohere: Type of data: {type(data)}")

        if isinstance(data, dict) and 'summary' in data:
            # If the loaded data is a dictionary and already contains a summary, return it
            logging.debug("Cohere: Summary already exists in the loaded data")
            return data['summary']

        # If the loaded data is a list of segment dictionaries or a string, proceed with summarization
        if isinstance(data, list):
            segments = data
            text = extract_text_from_segments(segments)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError("Invalid input data format")

        cohere_model = loaded_config_data['models']['cohere']

        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
            'Authorization': f'Bearer {cohere_api_key}'
        }

        cohere_prompt = f"{text} \n\n\n\n{custom_prompt_arg}"
        logging.debug("cohere: Prompt being sent is {cohere_prompt}")

        model = loaded_config_data['models']['anthropic']

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
def summarize_with_groq(api_key, input_data, custom_prompt_arg):
    loaded_config_data = load_and_log_configs()
    try:
        # API key validation
        if api_key is None:
            logging.info("groq: API key not provided as parameter")
            logging.info("groq: Attempting to use API key from config file")
            groq_api_key = loaded_config_data['api_keys']['groq']

        if api_key is None or api_key.strip() == "":
            logging.error("groq: API key not found or is empty")
            return "groq: API Key Not Provided/Found in Config file or is empty"

        logging.debug(f"groq: Using API Key: {api_key[:5]}...{api_key[-5:]}")

        # Transcript data handling & Validation
        if isinstance(input_data, str) and os.path.isfile(input_data):
            logging.debug("Groq: Loading json data for summarization")
            with open(input_data, 'r') as file:
                data = json.load(file)
        else:
            logging.debug("Groq: Using provided string data for summarization")
            data = input_data

        logging.debug(f"Groq: Loaded data: {data}")
        logging.debug(f"Groq: Type of data: {type(data)}")

        if isinstance(data, dict) and 'summary' in data:
            # If the loaded data is a dictionary and already contains a summary, return it
            logging.debug("Groq: Summary already exists in the loaded data")
            return data['summary']

        # If the loaded data is a list of segment dictionaries or a string, proceed with summarization
        if isinstance(data, list):
            segments = data
            text = extract_text_from_segments(segments)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError("Groq: Invalid input data format")

        # Set the model to be used
        groq_model = loaded_config_data['models']['groq']

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
            "model": groq_model
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


def summarize_with_openrouter(api_key, input_data, custom_prompt_arg):
    loaded_config_data = load_and_log_configs()
    import requests
    import json
    global openrouter_model, openrouter_api_key
    # API key validation
    if api_key is None:
        logging.info("openrouter: API key not provided as parameter")
        logging.info("openrouter: Attempting to use API key from config file")
        openrouter_api_key = loaded_config_data['api_keys']['openrouter']

    if api_key is None or api_key.strip() == "":
        logging.error("openrouter: API key not found or is empty")
        return "openrouter: API Key Not Provided/Found in Config file or is empty"

    logging.debug(f"openai: Using API Key: {api_key[:5]}...{api_key[-5:]}")

    if isinstance(input_data, str) and os.path.isfile(input_data):
        logging.debug("openrouter: Loading json data for summarization")
        with open(input_data, 'r') as file:
            data = json.load(file)
    else:
        logging.debug("openrouter: Using provided string data for summarization")
        data = input_data

    logging.debug(f"openrouter: Loaded data: {data}")
    logging.debug(f"openrouter: Type of data: {type(data)}")

    if isinstance(data, dict) and 'summary' in data:
        # If the loaded data is a dictionary and already contains a summary, return it
        logging.debug("openrouter: Summary already exists in the loaded data")
        return data['summary']

    # If the loaded data is a list of segment dictionaries or a string, proceed with summarization
    if isinstance(data, list):
        segments = data
        text = extract_text_from_segments(segments)
    elif isinstance(data, str):
        text = data
    else:
        raise ValueError("Invalid input data format")

    config = configparser.ConfigParser()
    file_path = 'config.txt'

    # Check if the file exists in the specified path
    if os.path.exists(file_path):
        config.read(file_path)
    elif os.path.exists('config.txt'):  # Check in the current directory
        config.read('../config.txt')
    else:
        print("config.txt not found in the specified path or current directory.")

    openrouter_prompt = f"{input_data} \n\n\n\n{custom_prompt_arg}"

    try:
        logging.debug("openrouter: Submitting request to API endpoint")
        print("openrouter: Submitting request to API endpoint")
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_api_key}",
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

def summarize_with_huggingface(api_key, input_data, custom_prompt_arg):
    loaded_config_data = load_and_log_configs()
    global huggingface_api_key
    logging.debug(f"huggingface: Summarization process starting...")
    try:
        # API key validation
        if api_key is None:
            logging.info("HuggingFace: API key not provided as parameter")
            logging.info("HuggingFace: Attempting to use API key from config file")
            huggingface_api_key = loaded_config_data['api_keys']['openai']

        if api_key is None or api_key.strip() == "":
            logging.error("HuggingFace: API key not found or is empty")
            return "HuggingFace: API Key Not Provided/Found in Config file or is empty"

        logging.debug(f"HuggingFace: Using API Key: {api_key[:5]}...{api_key[-5:]}")

        if isinstance(input_data, str) and os.path.isfile(input_data):
            logging.debug("HuggingFace: Loading json data for summarization")
            with open(input_data, 'r') as file:
                data = json.load(file)
        else:
            logging.debug("HuggingFace: Using provided string data for summarization")
            data = input_data

        logging.debug(f"HuggingFace: Loaded data: {data}")
        logging.debug(f"HuggingFace: Type of data: {type(data)}")

        if isinstance(data, dict) and 'summary' in data:
            # If the loaded data is a dictionary and already contains a summary, return it
            logging.debug("HuggingFace: Summary already exists in the loaded data")
            return data['summary']

        # If the loaded data is a list of segment dictionaries or a string, proceed with summarization
        if isinstance(data, list):
            segments = data
            text = extract_text_from_segments(segments)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError("HuggingFace: Invalid input data format")

        print(f"HuggingFace: lets make sure the HF api key exists...\n\t {api_key}")
        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        huggingface_model = loaded_config_data['models']['huggingface']
        API_URL = f"https://api-inference.huggingface.co/models/{huggingface_model}"

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


def summarize_with_deepseek(api_key, input_data, custom_prompt_arg):
    loaded_config_data = load_and_log_configs()
    try:
        # API key validation
        if api_key is None or api_key.strip() == "":
            logging.info("DeepSeek: API key not provided as parameter")
            logging.info("DeepSeek: Attempting to use API key from config file")
            api_key = loaded_config_data['api_keys']['deepseek']

        if api_key is None or api_key.strip() == "":
            logging.error("DeepSeek: API key not found or is empty")
            return "DeepSeek: API Key Not Provided/Found in Config file or is empty"

        logging.debug(f"DeepSeek: Using API Key: {api_key[:5]}...{api_key[-5:]}")

        # Input data handling
        if isinstance(input_data, str) and os.path.isfile(input_data):
            logging.debug("DeepSeek: Loading json data for summarization")
            with open(input_data, 'r') as file:
                data = json.load(file)
        else:
            logging.debug("DeepSeek: Using provided string data for summarization")
            data = input_data

        logging.debug(f"DeepSeek: Loaded data: {data}")
        logging.debug(f"DeepSeek: Type of data: {type(data)}")

        if isinstance(data, dict) and 'summary' in data:
            # If the loaded data is a dictionary and already contains a summary, return it
            logging.debug("DeepSeek: Summary already exists in the loaded data")
            return data['summary']

        # Text extraction
        if isinstance(data, list):
            segments = data
            text = extract_text_from_segments(segments)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError("DeepSeek: Invalid input data format")

        deepseek_model = loaded_config_data['models']['deepseek'] or "deepseek-chat"

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        logging.debug(
            f"Deepseek API Key: {api_key[:5]}...{api_key[-5:] if api_key else None}")
        logging.debug("openai: Preparing data + prompt for submittal")
        deepseek_prompt = f"{text} \n\n\n\n{custom_prompt_arg}"
        data = {
            "model": deepseek_model,
            "messages": [
                {"role": "system", "content": "You are a professional summarizer."},
                {"role": "user", "content": deepseek_prompt}
            ],
            "stream": False,
            "temperature": 0.8
        }

        logging.debug("DeepSeek: Posting request")
        response = requests.post('https://api.deepseek.com/chat/completions', headers=headers, json=data)

        if response.status_code == 200:
            response_data = response.json()
            if 'choices' in response_data and len(response_data['choices']) > 0:
                summary = response_data['choices'][0]['message']['content'].strip()
                logging.debug("DeepSeek: Summarization successful")
                return summary
            else:
                logging.warning("DeepSeek: Summary not found in the response data")
                return "DeepSeek: Summary not available"
        else:
            logging.error(f"DeepSeek: Summarization failed with status code {response.status_code}")
            logging.error(f"DeepSeek: Error response: {response.text}")
            return f"DeepSeek: Failed to process summary. Status code: {response.status_code}"
    except Exception as e:
        logging.error(f"DeepSeek: Error in processing: {str(e)}", exc_info=True)
        return f"DeepSeek: Error occurred while processing summary: {str(e)}"


#
#
#######################################################################################################################
#
#
# Gradio File Processing


# Handle multiple videos as input
def process_video_urls(url_list, num_speakers, whisper_model, custom_prompt_input, offset, api_name, api_key, vad_filter,
                       download_video_flag, download_audio, rolling_summarization, detail_level, question_box,
                       keywords, chunk_text_by_words, max_words, chunk_text_by_sentences, max_sentences,
                       chunk_text_by_paragraphs, max_paragraphs, chunk_text_by_tokens, max_tokens):
    global current_progress
    progress = []  # This must always be a list
    status = []  # This must always be a list

    def update_progress(index, url, message):
        progress.append(f"Processing {index + 1}/{len(url_list)}: {url}")  # Append to list
        status.append(message)  # Append to list
        return "\n".join(progress), "\n".join(status)  # Return strings for display


    for index, url in enumerate(url_list):
        try:
            transcription, summary, json_file_path, summary_file_path, _, _ = process_url(
                url=url,
                num_speakers=num_speakers,
                whisper_model=whisper_model,
                custom_prompt_input=custom_prompt_input,
                offset=offset,
                api_name=api_name,
                api_key=api_key,
                vad_filter=vad_filter,
                download_video_flag=download_video_flag,
                download_audio=download_audio,
                rolling_summarization=rolling_summarization,
                detail_level=detail_level,
                question_box=question_box,
                keywords=keywords,
                chunk_text_by_words=chunk_text_by_words,
                max_words=max_words,
                chunk_text_by_sentences=max_sentences,
                max_sentences=max_sentences,
                chunk_text_by_paragraphs=chunk_text_by_paragraphs,
                max_paragraphs=max_paragraphs,
                chunk_text_by_tokens=chunk_text_by_tokens,
                max_tokens=max_tokens
            )
            # Update progress and transcription properly
            current_progress, current_status = update_progress(index, url, "Video processed and ingested into the database.")
        except Exception as e:
            current_progress, current_status = update_progress(index, url, f"Error: {str(e)}")

    success_message = "All videos have been transcribed, summarized, and ingested into the database successfully."
    return current_progress, success_message, None, None, None, None


# stuff
def perform_transcription(video_path, offset, whisper_model, vad_filter, diarize=False):
    global segments_json_path
    audio_file_path = convert_to_wav(video_path, offset)
    segments_json_path = audio_file_path.replace('.wav', '.segments.json')

    if diarize:
        diarized_json_path = audio_file_path.replace('.wav', '.diarized.json')

        # Check if diarized JSON already exists
        if os.path.exists(diarized_json_path):
            logging.info(f"Diarized file already exists: {diarized_json_path}")
            try:
                with open(diarized_json_path, 'r') as file:
                    diarized_segments = json.load(file)
                if not diarized_segments:
                    logging.warning(f"Diarized JSON file is empty, re-generating: {diarized_json_path}")
                    raise ValueError("Empty diarized JSON file")
                logging.debug(f"Loaded diarized segments from {diarized_json_path}")
                return audio_file_path, diarized_segments
            except (json.JSONDecodeError, ValueError) as e:
                logging.error(f"Failed to read or parse the diarized JSON file: {e}")
                os.remove(diarized_json_path)

        # If diarized file doesn't exist or was corrupted, generate new diarized transcription
        logging.info(f"Generating diarized transcription for {audio_file_path}")
        diarized_segments = combine_transcription_and_diarization(audio_file_path)

        # Save diarized segments
        with open(diarized_json_path, 'w') as file:
            json.dump(diarized_segments, file, indent=2)

        return audio_file_path, diarized_segments

    # Non-diarized transcription (existing functionality)
    if os.path.exists(segments_json_path):
        logging.info(f"Segments file already exists: {segments_json_path}")
        try:
            with open(segments_json_path, 'r') as file:
                segments = json.load(file)
            if not segments:
                logging.warning(f"Segments JSON file is empty, re-generating: {segments_json_path}")
                raise ValueError("Empty segments JSON file")
            logging.debug(f"Loaded segments from {segments_json_path}")
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Failed to read or parse the segments JSON file: {e}")
            os.remove(segments_json_path)
            logging.info(f"Re-generating transcription for {audio_file_path}")
            audio_file, segments = re_generate_transcription(audio_file_path, whisper_model, vad_filter)
            if segments is None:
                return None, None
    else:
        audio_file, segments = re_generate_transcription(audio_file_path, whisper_model, vad_filter)

    return audio_file_path, segments


def re_generate_transcription(audio_file_path, whisper_model, vad_filter):
    try:
        segments = speech_to_text(audio_file_path, whisper_model=whisper_model, vad_filter=vad_filter)
        # Save segments to JSON
        with open(segments_json_path, 'w') as file:
            json.dump(segments, file, indent=2)
        logging.debug(f"Transcription segments saved to {segments_json_path}")
        return audio_file_path, segments
    except Exception as e:
        logging.error(f"Error in re-generating transcription: {str(e)}")
        return None, None


def save_transcription_and_summary(transcription_text, summary_text, download_path):
    video_title = sanitize_filename(info_dict.get('title', 'Untitled'))

    json_file_path = os.path.join(download_path, f"{video_title}.segments.json")
    summary_file_path = os.path.join(download_path, f"{video_title}_summary.txt")

    with open(json_file_path, 'w') as json_file:
        json.dump(transcription_text['transcription'], json_file, indent=2)

    if summary_text is not None:
        with open(summary_file_path, 'w') as file:
            file.write(summary_text)
    else:
        logging.warning("Summary text is None. Skipping summary file creation.")
        summary_file_path = None

    return json_file_path, summary_file_path





def perform_summarization(api_name, json_file_path, custom_prompt_input, api_key):
    # Load Config
    loaded_config_data = load_and_log_configs()

    if custom_prompt_input is None:
        # FIXME - Setup proper default prompt & extract said prompt from config file or prompts.db file.
        #custom_prompt_input = config.get('Prompts', 'video_summarize_prompt', fallback="Above is the transcript of a video. Please read through the transcript carefully. Identify the main topics that are discussed over the course of the transcript. Then, summarize the key points about each main topic in bullet points. The bullet points should cover the key information conveyed about each topic in the video, but should be much shorter than the full transcript. Please output your bullet point summary inside <bulletpoints> tags. Do not repeat yourself while writing the summary.")
        custom_prompt_input = """
        You are a bulleted notes specialist. ```When creating comprehensive bulleted notes, you should follow these guidelines: Use multiple headings based on the referenced topics, not categories like quotes or terms. Headings should be surrounded by bold formatting and not be listed as bullet points themselves. Leave no space between headings and their corresponding list items underneath. Important terms within the content should be emphasized by setting them in bold font. Any text that ends with a colon should also be bolded. Before submitting your response, review the instructions, and make any corrections necessary to adhered to the specified format. Do not reference these instructions within the notes.``` \nBased on the content between backticks create comprehensive bulleted notes.
**Bulleted Note Creation Guidelines**

**Headings**:
- Based on referenced topics, not categories like quotes or terms
- Surrounded by **bold** formatting 
- Not listed as bullet points
- No space between headings and list items underneath

**Emphasis**:
- **Important terms** set in bold font
- **Text ending in a colon**: also bolded

**Review**:
- Ensure adherence to specified format
- Do not reference these instructions in your response.</s>[INST] {{ .Prompt }} [/INST]"""
    summary = None
    try:
        if not json_file_path or not os.path.exists(json_file_path):
            logging.error(f"JSON file does not exist: {json_file_path}")
            return None

        with open(json_file_path, 'r') as file:
            data = json.load(file)

        segments = data
        if not isinstance(segments, list):
            logging.error(f"Segments is not a list: {type(segments)}")
            return None

        text = extract_text_from_segments(segments)

        if api_name.lower() == 'openai':
            #def summarize_with_openai(api_key, input_data, custom_prompt_arg)
            summary = summarize_with_openai(api_key, text, custom_prompt_input)

        elif api_name.lower() == "anthropic":
            # def summarize_with_anthropic(api_key, input_data, model, custom_prompt_arg, max_retries=3, retry_delay=5):
            summary = summarize_with_anthropic(api_key, text, custom_prompt_input)
        elif api_name.lower() == "cohere":
            # def summarize_with_cohere(api_key, input_data, model, custom_prompt_arg)
            summary = summarize_with_cohere(api_key, text, custom_prompt_input)

        elif api_name.lower() == "groq":
            logging.debug(f"MAIN: Trying to summarize with groq")
            # def summarize_with_groq(api_key, input_data, model, custom_prompt_arg):
            summary = summarize_with_groq(api_key, text, custom_prompt_input)

        elif api_name.lower() == "openrouter":
            logging.debug(f"MAIN: Trying to summarize with OpenRouter")
            # def summarize_with_openrouter(api_key, input_data, custom_prompt_arg):
            summary = summarize_with_openrouter(api_key, text, custom_prompt_input)

        elif api_name.lower() == "deepseek":
            logging.debug(f"MAIN: Trying to summarize with DeepSeek")
            # def summarize_with_deepseek(api_key, input_data, custom_prompt_arg):
            summary = summarize_with_deepseek(api_key, text, custom_prompt_input)

        elif api_name.lower() == "llama.cpp":
            logging.debug(f"MAIN: Trying to summarize with Llama.cpp")
            # def summarize_with_llama(api_url, file_path, token, custom_prompt)
            summary = summarize_with_llama(text, custom_prompt_input)

        elif api_name.lower() == "kobold":
            logging.debug(f"MAIN: Trying to summarize with Kobold.cpp")
            # def summarize_with_kobold(input_data, kobold_api_token, custom_prompt_input, api_url):
            summary = summarize_with_kobold(text, api_key, custom_prompt_input)

        elif api_name.lower() == "ooba":
            # def summarize_with_oobabooga(input_data, api_key, custom_prompt, api_url):
            summary = summarize_with_oobabooga(text, api_key, custom_prompt_input)

        elif api_name.lower() == "tabbyapi":
            # def summarize_with_tabbyapi(input_data, tabby_model, custom_prompt_input, api_key=None, api_IP):
            summary = summarize_with_tabbyapi(text, custom_prompt_input)

        elif api_name.lower() == "vllm":
            logging.debug(f"MAIN: Trying to summarize with VLLM")
            # def summarize_with_vllm(api_key, input_data, custom_prompt_input):
            summary = summarize_with_vllm(text, custom_prompt_input)

        elif api_name.lower() == "local-llm":
            logging.debug(f"MAIN: Trying to summarize with Local LLM")
            summary = summarize_with_local_llm(text, custom_prompt_input)

        elif api_name.lower() == "huggingface":
            logging.debug(f"MAIN: Trying to summarize with huggingface")
            # def summarize_with_huggingface(api_key, input_data, custom_prompt_arg):
            summarize_with_huggingface(api_key, text, custom_prompt_input)
        # Add additional API handlers here...

        else:
            logging.warning(f"Unsupported API: {api_name}")

        if summary is None:
            logging.debug("Summarization did not return valid text.")

        if summary:
            logging.info(f"Summary generated using {api_name} API")
            # Save the summary file in the same directory as the JSON file
            summary_file_path = json_file_path.replace('.json', '_summary.txt')
            with open(summary_file_path, 'w') as file:
                file.write(summary)
        else:
            logging.warning(f"Failed to generate summary using {api_name} API")
        return summary

    except requests.exceptions.ConnectionError:
            logging.error("Connection error while summarizing")
    except Exception as e:
        logging.error(f"Error summarizing with {api_name}: {str(e)}")

    return summary



def process_url(
        url,
        num_speakers,
        whisper_model,
        custom_prompt_input,
        offset,
        api_name,
        api_key,
        vad_filter,
        download_video_flag,
        download_audio,
        rolling_summarization,
        detail_level,
        # It's for the asking a question about a returned prompt - needs to be removed #FIXME
        question_box,
        keywords,
        chunk_text_by_words,
        max_words,
        chunk_text_by_sentences,
        max_sentences,
        chunk_text_by_paragraphs,
        max_paragraphs,
        chunk_text_by_tokens,
        max_tokens,
        local_file_path=None,
        diarize=False
):
    # Handle the chunk summarization options
    set_chunk_txt_by_words = chunk_text_by_words
    set_max_txt_chunk_words = max_words
    set_chunk_txt_by_sentences = chunk_text_by_sentences
    set_max_txt_chunk_sentences = max_sentences
    set_chunk_txt_by_paragraphs = chunk_text_by_paragraphs
    set_max_txt_chunk_paragraphs = max_paragraphs
    set_chunk_txt_by_tokens = chunk_text_by_tokens
    set_max_txt_chunk_tokens = max_tokens

    progress = []
    success_message = "All videos processed successfully. Transcriptions and summaries have been ingested into the database."


    # Validate input
    if not url and not local_file_path:
        return "Process_URL: No URL provided.", "No URL provided.", None, None, None, None, None, None

    # FIXME - Chatgpt again?
    if isinstance(url, str):
        urls = url.strip().split('\n')
        if len(urls) > 1:
            return process_video_urls(urls, num_speakers, whisper_model, custom_prompt_input, offset, api_name, api_key, vad_filter,
                                      download_video_flag, download_audio, rolling_summarization, detail_level, question_box,
                                      keywords, chunk_text_by_words, max_words, chunk_text_by_sentences, max_sentences,
                                      chunk_text_by_paragraphs, max_paragraphs, chunk_text_by_tokens, max_tokens)
        else:
            urls = [url]

    if url and not is_valid_url(url):
        return "Process_URL: Invalid URL format.", "Invalid URL format.", None, None, None, None, None, None

    if url:
        # Clean the URL to remove playlist parameters if any
        url = clean_youtube_url(url)
        logging.info(f"Process_URL: Processing URL: {url}")

    if api_name:
        print("Process_URL: API Name received:", api_name)  # Debugging line

    video_file_path = None
    global info_dict

    # FIXME - need to handle local audio file processing
    # If Local audio file is provided
    if local_file_path:
        try:
            pass
            # # insert code to process local audio file
            # # Need to be able to add a title/author/etc for ingestion into the database
            # # Also want to be able to optionally _just_ ingest it, and not ingest.
            # # FIXME
            # #download_path = create_download_directory(title)
            # #audio_path = download_video(url, download_path, info_dict, download_video_flag)
            #
            # audio_file_path = local_file_path
            # global segments
            # audio_file_path, segments = perform_transcription(audio_file_path, offset, whisper_model, vad_filter)
            #
            # if audio_file_path is None or segments is None:
            #     logging.error("Process_URL: Transcription failed or segments not available.")
            #     return "Process_URL: Transcription failed.", "Transcription failed.", None, None, None, None
            #
            # logging.debug(f"Process_URL: Transcription audio_file: {audio_file_path}")
            # logging.debug(f"Process_URL: Transcription segments: {segments}")
            #
            # transcription_text = {'audio_file': audio_file_path, 'transcription': segments}
            # logging.debug(f"Process_URL: Transcription text: {transcription_text}")
            #
            # if rolling_summarization:
            #     text = extract_text_from_segments(segments)
            #     summary_text = rolling_summarize_function(
            #         transcription_text,
            #         detail=detail_level,
            #         api_name=api_name,
            #         api_key=api_key,
            #         custom_prompt=custom_prompt,
            #         chunk_by_words=chunk_text_by_words,
            #         max_words=max_words,
            #         chunk_by_sentences=chunk_text_by_sentences,
            #         max_sentences=max_sentences,
            #         chunk_by_paragraphs=chunk_text_by_paragraphs,
            #         max_paragraphs=max_paragraphs,
            #         chunk_by_tokens=chunk_text_by_tokens,
            #         max_tokens=max_tokens
            #     )
            # if api_name:
            #     summary_text = perform_summarization(api_name, segments_json_path, custom_prompt, api_key, config)
            #     if summary_text is None:
            #         logging.error("Summary text is None. Check summarization function.")
            #         summary_file_path = None  # Set summary_file_path to None if summary is not generated
            # else:
            #     summary_text = 'Summary not available'
            #     summary_file_path = None  # Set summary_file_path to None if summary is not generated
            #
            # json_file_path, summary_file_path = save_transcription_and_summary(transcription_text, summary_text, download_path)
            #
            # add_media_to_database(url, info_dict, segments, summary_text, keywords, custom_prompt, whisper_model)
            #
            # return transcription_text, summary_text, json_file_path, summary_file_path, None, None

        except Exception as e:
            logging.error(f": {e}")
            return str(e), 'process_url: Error processing the request.', None, None, None, None


    # If URL/Local video file is provided
    try:
        info_dict, title = extract_video_info(url)
        download_path = create_download_directory(title)
        video_path = download_video(url, download_path, info_dict, download_video_flag)
        global segments
        audio_file_path, segments = perform_transcription(video_path, offset, whisper_model, vad_filter)

        if diarize:
            transcription_text = combine_transcription_and_diarization(audio_file_path)
        else:
            audio_file, segments = perform_transcription(video_path, offset, whisper_model, vad_filter)
            transcription_text = {'audio_file': audio_file, 'transcription': segments}


        if audio_file_path is None or segments is None:
            logging.error("Process_URL: Transcription failed or segments not available.")
            return "Process_URL: Transcription failed.", "Transcription failed.", None, None, None, None

        logging.debug(f"Process_URL: Transcription audio_file: {audio_file_path}")
        logging.debug(f"Process_URL: Transcription segments: {segments}")

        transcription_text = {'audio_file': audio_file_path, 'transcription': segments}
        logging.debug(f"Process_URL: Transcription text: {transcription_text}")

        # FIXME - rolling summarization
        # if rolling_summarization:
        #     text = extract_text_from_segments(segments)
        #     summary_text = rolling_summarize_function(
        #         transcription_text,
        #         detail=detail_level,
        #         api_name=api_name,
        #         api_key=api_key,
        #         custom_prompt_input=custom_prompt_input,
        #         chunk_by_words=chunk_text_by_words,
        #         max_words=max_words,
        #         chunk_by_sentences=chunk_text_by_sentences,
        #         max_sentences=max_sentences,
        #         chunk_by_paragraphs=chunk_text_by_paragraphs,
        #         max_paragraphs=max_paragraphs,
        #         chunk_by_tokens=chunk_text_by_tokens,
        #         max_tokens=max_tokens
        #     )
        if api_name:
            summary_text = perform_summarization(api_name, segments_json_path, custom_prompt_input, api_key)
            if summary_text is None:
                logging.error("Summary text is None. Check summarization function.")
                summary_file_path = None  # Set summary_file_path to None if summary is not generated
        else:
            summary_text = 'Summary not available'
            summary_file_path = None  # Set summary_file_path to None if summary is not generated

        json_file_path, summary_file_path = save_transcription_and_summary(transcription_text, summary_text, download_path)

        add_media_to_database(url, info_dict, segments, summary_text, keywords, custom_prompt_input, whisper_model)

        return transcription_text, summary_text, json_file_path, summary_file_path, None, None

    except Exception as e:
        logging.error(f": {e}")
        return str(e), 'process_url: Error processing the request.', None, None, None, None

























