# Summarization_General_Lib.py
#########################################
# General Summarization Library
# This library is used to perform summarization.
#
####

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
from typing import List, Dict
import json

#######################################################################################################################
# Function Definitions
#

# FIXME
def extract_text_from_segments(segments: List[Dict]) -> str:
    """Extract text from segments."""
    return " ".join([segment['text'] for segment in segments])



# Fixme , function is replicated....
def extract_text_from_segments(segments):
    logging.debug(f"Main: extracting text from {segments}")
    text = ' '.join([segment['text'] for segment in segments])
    logging.debug(f"Main: Successfully extracted text from {segments}")
    return text


def summarize_with_openai(api_key, file_path, custom_prompt_arg):
    try:
        logging.debug("openai: Loading json data for summarization")
        with open(file_path, 'r') as file:
            segments = json.load(file)

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
            "max_tokens": 8192,  # Adjust tokens as needed
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


def summarize_with_openrouter(api_key, file_path, custom_prompt_arg)
    import requests
    import json
    OPENROUTER_API_KEY = api_key
    global openrouter_model
    if openrouter_model is None:
        openrouter_model = "microsoft/wizardlm-2-8x22b"

    response = requests.post('https://api.cohere.ai/v1/chat', headers=headers, json=data)

    logging.debug("openrouter: Submitting request to API endpoint")
    print("openrouter: Submitting request to API endpoint")
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        },
        data=json.dumps({
            "model": f"{openrouter_model}",
            "messages": [
                {"role": "user", "content": f"{custom_prompt_arg}"}
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


#
#
#######################################################################################################################