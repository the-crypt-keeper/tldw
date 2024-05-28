# Local_Summarization_Lib.py
#########################################
# Local Summarization Library
# This library is used to perform summarization with a 'local' inference engine.
#
####

####################
# Function List
#
# 1. summarize_with_local_llm(file_path, custom_prompt_arg)
# 2. summarize_with_llama(api_url, file_path, token, custom_prompt)
# 3. summarize_with_kobold(api_url, file_path, kobold_api_token, custom_prompt)
# 4. summarize_with_oobabooga(api_url, file_path, ooba_api_token, custom_prompt)
# 5. summarize_with_vllm(vllm_api_url, vllm_api_key_function_arg, llm_model, text, vllm_custom_prompt_function_arg)
# 6. summarize_with_tabbyapi(tabby_api_key, tabby_api_IP, text, tabby_model, custom_prompt)
# 7. save_summary_to_file(summary, file_path)
#
#
####################


# Import necessary libraries
import os
import logging
from typing import Callable

# Import Local
import summarize
from Article_Summarization_Lib import *
from Article_Extractor_Lib import *
from Audio_Transcription_Lib import *
from Chunk_Lib import *
from Diarization_Lib import *
from Local_File_Processing_Lib import *
from Local_LLM_Inference_Engine_Lib import *
#from Local_Summarization_Lib import *
from Old_Chunking_Lib import *
from SQLite_DB import *
from Summarization_General_Lib import *
from System_Checks_Lib import *
from Tokenization_Methods_Lib import *
from Video_DL_Ingestion_Lib import *
from Web_UI_Lib import *

# Read configuration from file
config = configparser.ConfigParser()
config.read('config.txt')

# Local-Models
kobold_api_IP = config.get('Local-API', 'kobold_api_IP', fallback='http://127.0.0.1:5000/api/v1/generate')
kobold_api_key = config.get('Local-API', 'kobold_api_key', fallback='')

llama_api_IP = config.get('Local-API', 'llama_api_IP', fallback='http://127.0.0.1:8080/v1/chat/completions')
llama_api_key = config.get('Local-API', 'llama_api_key', fallback='')

ooba_api_IP = config.get('Local-API', 'ooba_api_IP', fallback='http://127.0.0.1:5000/v1/chat/completions')
ooba_api_key = config.get('Local-API', 'ooba_api_key', fallback='')

tabby_api_IP = config.get('Local-API', 'tabby_api_IP', fallback='http://127.0.0.1:5000/api/v1/generate')
tabby_api_key = config.get('Local-API', 'tabby_api_key', fallback=None)

vllm_api_url = config.get('Local-API', 'vllm_api_IP', fallback='http://127.0.0.1:500/api/v1/chat/completions')
vllm_api_key = config.get('Local-API', 'vllm_api_key', fallback=None)

#######################################################################################################################
# Function Definitions
#

def summarize_with_local_llm(file_path, custom_prompt_arg):
    try:
        logging.debug("Local LLM: Loading json data for summarization")
        with open(file_path, 'r') as file:
            segments = json.load(file)

        logging.debug("Local LLM: Extracting text from the segments")
        text = extract_text_from_segments(segments)

        headers = {
            'Content-Type': 'application/json'
        }

        logging.debug("Local LLM: Preparing data + prompt for submittal")
        local_llm_prompt = f"{text} \n\n\n\n{custom_prompt_arg}"
        data = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional summarizer."
                },
                {
                    "role": "user",
                    "content": local_llm_prompt
                }
            ],
            "max_tokens": 28000,  # Adjust tokens as needed
        }
        logging.debug("Local LLM: Posting request")
        response = requests.post('http://127.0.0.1:8080/v1/chat/completions', headers=headers, json=data)

        if response.status_code == 200:
            response_data = response.json()
            if 'choices' in response_data and len(response_data['choices']) > 0:
                summary = response_data['choices'][0]['message']['content'].strip()
                logging.debug("Local LLM: Summarization successful")
                print("Local LLM: Summarization successful.")
                return summary
            else:
                logging.warning("Local LLM: Summary not found in the response data")
                return "Local LLM: Summary not available"
        else:
            logging.debug("Local LLM: Summarization failed")
            print("Local LLM: Failed to process summary:", response.text)
            return "Local LLM: Failed to process summary"
    except Exception as e:
        logging.debug("Local LLM: Error in processing: %s", str(e))
        print("Error occurred while processing summary with Local LLM:", str(e))
        return "Local LLM: Error occurred while processing summary"

def summarize_with_llama(api_url, file_path, token, custom_prompt):
    try:
        logging.debug("llama: Loading JSON data")
        with open(file_path, 'r') as file:
            segments = json.load(file)

        logging.debug(f"llama: Extracting text from segments file")
        text = extract_text_from_segments(segments)  # Define this function to extract text properly

        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
        }
        if len(token) > 5:
            headers['Authorization'] = f'Bearer {token}'

        llama_prompt = f"{text} \n\n\n\n{custom_prompt}"
        logging.debug("llama: Prompt being sent is {llama_prompt}")

        data = {
            "prompt": llama_prompt
        }

        logging.debug("llama: Submitting request to API endpoint")
        print("llama: Submitting request to API endpoint")
        response = requests.post(api_url, headers=headers, json=data)
        response_data = response.json()
        logging.debug("API Response Data: %s", response_data)

        if response.status_code == 200:
            # if 'X' in response_data:
            logging.debug(response_data)
            summary = response_data['content'].strip()
            logging.debug("llama: Summarization successful")
            print("Summarization successful.")
            return summary
        else:
            logging.error(f"llama: API request failed with status code {response.status_code}: {response.text}")
            return f"llama: API request failed: {response.text}"

    except Exception as e:
        logging.error("llama: Error in processing: %s", str(e))
        return f"llama: Error occurred while processing summary with llama: {str(e)}"


# https://lite.koboldai.net/koboldcpp_api#/api%2Fv1/post_api_v1_generate
def summarize_with_kobold(api_url, file_path, kobold_api_token, custom_prompt):
    try:
        logging.debug("kobold: Loading JSON data")
        with open(file_path, 'r') as file:
            segments = json.load(file)

        logging.debug(f"kobold: Extracting text from segments file")
        text = extract_text_from_segments(segments)

        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
        }

        kobold_prompt = f"{text} \n\n\n\n{custom_prompt}"
        logging.debug("kobold: Prompt being sent is {kobold_prompt}")

        # FIXME
        # Values literally c/p from the api docs....
        data = {
            "max_context_length": 8096,
            "max_length": 4096,
            "prompt": kobold_prompt,
        }

        logging.debug("kobold: Submitting request to API endpoint")
        print("kobold: Submitting request to API endpoint")
        response = requests.post(api_url, headers=headers, json=data)
        response_data = response.json()
        logging.debug("kobold: API Response Data: %s", response_data)

        if response.status_code == 200:
            if 'results' in response_data and len(response_data['results']) > 0:
                summary = response_data['results'][0]['text'].strip()
                logging.debug("kobold: Summarization successful")
                print("Summarization successful.")
                return summary
            else:
                logging.error("Expected data not found in API response.")
                return "Expected data not found in API response."
        else:
            logging.error(f"kobold: API request failed with status code {response.status_code}: {response.text}")
            return f"kobold: API request failed: {response.text}"

    except Exception as e:
        logging.error("kobold: Error in processing: %s", str(e))
        return f"kobold: Error occurred while processing summary with kobold: {str(e)}"


# https://github.com/oobabooga/text-generation-webui/wiki/12-%E2%80%90-OpenAI-API
def summarize_with_oobabooga(api_url, file_path, ooba_api_token, custom_prompt):
    try:
        logging.debug("ooba: Loading JSON data")
        with open(file_path, 'r') as file:
            segments = json.load(file)

        logging.debug(f"ooba: Extracting text from segments file\n\n\n")
        text = extract_text_from_segments(segments)
        logging.debug(f"ooba: Finished extracting text from segments file")

        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
        }

        # prompt_text = "I like to eat cake and bake cakes. I am a baker. I work in a French bakery baking cakes. It
        # is a fun job. I have been baking cakes for ten years. I also bake lots of other baked goods, but cakes are
        # my favorite." prompt_text += f"\n\n{text}"  # Uncomment this line if you want to include the text variable
        ooba_prompt = "{text}\n\n\n\n{custom_prompt}"
        logging.debug("ooba: Prompt being sent is {ooba_prompt}")

        data = {
            "mode": "chat",
            "character": "Example",
            "messages": [{"role": "user", "content": ooba_prompt}]
        }

        logging.debug("ooba: Submitting request to API endpoint")
        print("ooba: Submitting request to API endpoint")
        response = requests.post(api_url, headers=headers, json=data, verify=False)
        logging.debug("ooba: API Response Data: %s", response)

        if response.status_code == 200:
            response_data = response.json()
            summary = response.json()['choices'][0]['message']['content']
            logging.debug("ooba: Summarization successful")
            print("Summarization successful.")
            return summary
        else:
            logging.error(f"oobabooga: API request failed with status code {response.status_code}: {response.text}")
            return f"ooba: API request failed with status code {response.status_code}: {response.text}"

    except Exception as e:
        logging.error("ooba: Error in processing: %s", str(e))
        return f"ooba: Error occurred while processing summary with oobabooga: {str(e)}"


# FIXME - https://docs.vllm.ai/en/latest/getting_started/quickstart.html .... Great docs.
def summarize_with_vllm(vllm_api_url, vllm_api_key_function_arg, llm_model, text, vllm_custom_prompt_function_arg):
    vllm_client = OpenAI(
        base_url=vllm_api_url,
        api_key=vllm_api_key_function_arg
    )

    custom_prompt = vllm_custom_prompt_function_arg

    completion = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": "You are a professional summarizer."},
            {"role": "user", "content": f"{text} \n\n\n\n{custom_prompt}"}
        ]
    )
    vllm_summary = completion.choices[0].message.content
    return vllm_summary


# FIXME - Install is more trouble than care to deal with right now.
def summarize_with_tabbyapi(tabby_api_key, tabby_api_IP, text, tabby_model, custom_prompt):
    model = tabby_model
    headers = {
        'Authorization': f'Bearer {tabby_api_key}',
        'Content-Type': 'application/json'
    }
    data = {
        'text': text,
        'model': 'tabby'  # Specify the model if needed
    }
    try:
        response = requests.post('https://api.tabbyapi.com/summarize', headers=headers, json=data)
        response.raise_for_status()
        summary = response.json().get('summary', '')
        return summary
    except requests.exceptions.RequestException as e:
        logger.error(f"Error summarizing with TabbyAPI: {e}")
        return "Error summarizing with TabbyAPI."


def save_summary_to_file(summary, file_path):
    logging.debug("Now saving summary to file...")
    summary_file_path = file_path.replace('.segments.json', '_summary.txt')
    logging.debug("Opening summary file for writing, *segments.json with *_summary.txt")
    with open(summary_file_path, 'w') as file:
        file.write(summary)
    logging.info(f"Summary saved to file: {summary_file_path}")


summarizers: Dict[str, Callable[[str, str], str]] = {
    'tabbyapi': summarize_with_tabbyapi,
    'openai': summarize_with_openai,
    'anthropic': summarize_with_claude,
    'cohere': summarize_with_cohere,
    'groq': summarize_with_groq,
    'llama': summarize_with_llama,
    'kobold': summarize_with_kobold,
    'oobabooga': summarize_with_oobabooga
    # Add more APIs here as needed
}



