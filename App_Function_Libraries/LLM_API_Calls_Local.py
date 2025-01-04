# Local_Summarization_Lib.py
#########################################
# Local Summarization Library
# This library is used to perform summarization with a 'local' inference engine.
#
####
import logging
from typing import Union

####################
# Function List
# FIXME - UPDATE
# 1. chat_with_local_llm(text, custom_prompt_arg)
# 2. chat_with_llama(api_url, text, token, custom_prompt)
# 3. chat_with_kobold(api_url, text, kobold_api_token, custom_prompt)
# 4. chat_with_oobabooga(api_url, text, ooba_api_token, custom_prompt)
# 5. chat_with_vllm(vllm_api_url, vllm_api_key_function_arg, llm_model, text, vllm_custom_prompt_function_arg)
# 6. chat_with_tabbyapi(tabby_api_key, tabby_api_IP, text, tabby_model, custom_prompt)
# 7. save_summary_to_file(summary, file_path)
#
#
####################
# Import necessary libraries
# Import Local
from App_Function_Libraries.Utils.Utils import *
#
#######################################################################################################################
# Function Definitions
#

def chat_with_local_llm(input_data, custom_prompt_arg, temp, system_message=None, streaming=False):
    try:
        if isinstance(input_data, str) and os.path.isfile(input_data):
            logging.debug("Local LLM: Loading json data for summarization")
            with open(input_data, 'r') as file:
                data = json.load(file)
        else:
            logging.debug("openai: Using provided string data for summarization")
            data = input_data

        logging.debug(f"Local LLM: Loaded data: {data}")
        logging.debug(f"Local LLM: Type of data: {type(data)}")

        if isinstance(data, dict) and 'summary' in data:
            # If the loaded data is a dictionary and already contains a summary, return it
            logging.debug("Local LLM: Summary already exists in the loaded data")
            return data['summary']

        # If the loaded data is a list of segment dictionaries or a string, proceed with summarization
        if isinstance(data, list):
            segments = data
            text = extract_text_from_segments(segments)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError("Invalid input data format")

        if system_message is None:
            system_message = "You are a helpful AI assistant."

        headers = {
            'Content-Type': 'application/json'
        }

        logging.debug("Local LLM: Preparing data + prompt for submittal")
        local_llm_prompt = f"{text} \n\n\n\n{custom_prompt_arg}"
        data = {
            "messages": [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": local_llm_prompt
                }
            ],
            "max_tokens": 8192,
            "temperature": temp,
            "stream": streaming
        }

        logging.debug("Local LLM: Posting request")
        response = requests.post('http://127.0.0.1:8080/v1/chat/completions', headers=headers, json=data)

        if response.status_code == 200:
            if streaming:
                logging.debug("Local LLM: Processing streaming response")

                def stream_generator():
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8').strip()
                            if decoded_line.startswith('data:'):
                                data_str = decoded_line[len('data:'):].strip()
                                if data_str == '[DONE]':
                                    break
                                try:
                                    data_json = json.loads(data_str)
                                    if 'choices' in data_json and len(data_json['choices']) > 0:
                                        delta = data_json['choices'][0].get('delta', {})
                                        if 'content' in delta:
                                            content = delta['content']
                                            yield content
                                except json.JSONDecodeError:
                                    logging.error(f"Local LLM: Error decoding JSON from line: {decoded_line}")
                                    continue
                return stream_generator()
            else:
                logging.debug("Local LLM: Processing non-streaming response")
                response_data = response.json()
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    summary = response_data['choices'][0]['message']['content'].strip()
                    logging.debug("Local LLM: Summarization successful")
                    logging.info("Local LLM: Summarization successful.")
                    return summary
                else:
                    logging.warning("Local LLM: Summary not found in the response data")
                    return "Local LLM: Summary not available"
        else:
            logging.error(f"Local LLM: Request failed with status code {response.status_code}")
            print("Local LLM: Failed to process Chat response:", response.text)
            return f"Local LLM: Failed to process Chat response, status code {response.status_code}"
    except Exception as e:
        logging.debug("Local LLM: Error in processing: %s", str(e))
        print("Error occurred while processing Chat request with Local LLM:", str(e))
        return f"Local LLM: Error occurred while processing Chat response: {str(e)}"


# FIXME
def chat_with_llama(input_data, custom_prompt, temp, api_url="http://127.0.0.1:8080/completion", api_key=None, system_prompt=None, streaming=False):
    loaded_config_data = load_and_log_configs()
    try:
        # API key validation
        if api_key is None:
            logging.info("llama.cpp: API key not provided as parameter")
            logging.info("llama.cpp: Attempting to use API key from config file")
            api_key = loaded_config_data['llama_api']['api_key']

        if api_key is None or api_key.strip() == "":
            logging.info("llama.cpp: API key not found or is empty")

        logging.debug(f"llama.cpp: Using API Key: {api_key[:5]}...{api_key[-5:]}")

        if api_url is None:
            logging.info("llama.cpp: API URL not provided as parameter")
            logging.info("llama.cpp: Attempting to use API URL from config file")
            api_url = loaded_config_data['llama_api']['api_ip']

        if api_url is None or api_url.strip() == "":
            logging.info("llama.cpp: API URL not found or is empty")
            return "llama.cpp: API URL not found or is empty"

        if isinstance(streaming, str):
            streaming = streaming.lower() == "true"
        elif isinstance(streaming, int):
            streaming = bool(streaming)  # Convert integers (1/0) to boolean
        elif streaming is None:
            streaming = loaded_config_data.get('llama_api', {}).get('streaming', False)
            logging.debug("Llama.cpp: Streaming mode enabled")
        else:
            logging.debug("Llama.cpp: Streaming mode disabled")
        if not isinstance(streaming, bool):
            raise ValueError(f"Invalid type for 'streaming': Expected a boolean, got {type(streaming).__name__}")

        # Prepare headers
        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
        }
        if len(api_key) > 5:
            headers['Authorization'] = f'Bearer {api_key}'

        if system_prompt is None:
            system_prompt = "You are a helpful AI assistant that provides accurate and concise information."

        logging.debug("Llama.cpp: System prompt being used is: %s", system_prompt)
        logging.debug("Llama.cpp: User prompt being used is: %s", custom_prompt)


        llama_prompt = f"{custom_prompt} \n\n\n\n{input_data}"
        logging.debug(f"llama: Prompt being sent is {llama_prompt}")

        data = {
            "prompt": f"{llama_prompt}",
            "system_prompt": f"{system_prompt}",
            'temperature': temp,
            #'top_k': '40',
            #'top_p': '0.95',
            #'min_p': '0.05',
            #'n_predict': '-1',
            #'n_keep': '0',
            'stream': 'True',
            #'stop': '["\n"]',
            #'tfs_z': '1.0',
            #'repeat_penalty': '1.1',
            #'repeat_last_n': '64',
            #'presence_penalty': '0.0',
            #'frequency_penalty': '0.0',
            #'mirostat': '0',
            #'grammar': '0',
            #'json_schema': '0',
            #'ignore_eos': 'false',
            #'logit_bias': [],
            #'n_probs': '0',
            #'min_keep': '0',
            #'samplers': '["top_k", "tfs_z", "typical_p", "top_p", "min_p", "temperature"]',

        }

        logging.debug("llama: Submitting request to API endpoint")
        print("llama: Submitting request to API endpoint")
        response = requests.post(api_url, headers=headers, json=data, stream=streaming)

        if response.status_code == 200:
            if streaming:
                logging.debug("Llama: Processing streaming response")

                def stream_generator():
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8').strip()
                            if decoded_line.startswith('data:'):
                                data_str = decoded_line[len('data:'):].strip()
                                if data_str == '[DONE]':
                                    break
                                try:
                                    data_json = json.loads(data_str)
                                    if 'choices' in data_json and len(data_json['choices']) > 0:
                                        delta = data_json['choices'][0].get('delta', {})
                                        if 'content' in delta:
                                            content = delta['content']
                                            yield content
                                except json.JSONDecodeError:
                                    logging.error(f"Llama: Error decoding JSON from line: {decoded_line}")
                                    continue
                return stream_generator()
            else:
                response_data = response.json()
                logging.debug("API Response Data: %s", response_data)

                if response.status_code == 200:
                    # if 'X' in response_data:
                    logging.debug(response_data)
                    summary = response_data['content'].strip()
                    logging.debug("Llama: Chat request successful")
                    print("Llama: Chat request successful.")
                    return summary
                else:
                    logging.error(f"Llama: API Chat request failed with status code {response.status_code}: {response.text}")
                    return f"Llama: API Chat request failed: {response.text}"

    except Exception as e:
        logging.error("Llama: Error in processing: %s", str(e))
        return f"Llama: Error occurred while processing summary with llama: {str(e)}"


# System prompts not supported through API requests.
# https://lite.koboldai.net/koboldcpp_api#/api%2Fv1/post_api_v1_generate
def chat_with_kobold(input_data, api_key, custom_prompt_input, kobold_api_ip="http://127.0.0.1:5001/api/v1/generate", temp=None, system_message=None, streaming=False):
    logging.debug("Kobold: Summarization process starting...")
    try:
        logging.debug("Kobold: Loading and validating configurations")
        loaded_config_data = load_and_log_configs()
        if loaded_config_data is None:
            logging.error("Failed to load configuration data")
            kobold_api_key = None
        else:
            # Prioritize the API key passed as a parameter
            if api_key and api_key.strip():
                kobold_api_key = api_key
                logging.info("Kobold: Using API key provided as parameter")
            else:
                # If no parameter is provided, use the key from the config
                kobold_api_key = loaded_config_data['kobold_api'].get('api_key')
                if kobold_api_key:
                    logging.info("Kobold: Using API key from config file")
                else:
                    logging.warning("Kobold: No API key found in config file")

        logging.debug(f"Kobold: Using API Key: {kobold_api_key[:5]}...{kobold_api_key[-5:]}")

        if isinstance(streaming, str):
            streaming = streaming.lower() == "true"
        elif isinstance(streaming, int):
            streaming = bool(streaming)  # Convert integers (1/0) to boolean
        elif streaming is None:
            streaming = loaded_config_data.get('kobold_api', {}).get('streaming', False)
            logging.debug("Kobold.cpp: Streaming mode enabled")
        else:
            logging.debug("Kobold.cpp: Streaming mode disabled")
        if not isinstance(streaming, bool):
            raise ValueError(f"Invalid type for 'streaming': Expected a boolean, got {type(streaming).__name__}")

        if isinstance(input_data, str) and os.path.isfile(input_data):
            logging.debug("Kobold.cpp: Loading json data for summarization")
            with open(input_data, 'r') as file:
                data = json.load(file)
        else:
            logging.debug("Kobold.cpp: Using provided string data for summarization")
            data = input_data

        logging.debug(f"Kobold.cpp: Loaded data: {data}")
        logging.debug(f"Kobold.cpp: Type of data: {type(data)}")

        if isinstance(data, dict) and 'summary' in data:
            # If the loaded data is a dictionary and already contains a summary, return it
            logging.debug("Kobold.cpp: Summary already exists in the loaded data")
            return data['summary']

        # If the loaded data is a list of segment dictionaries or a string, proceed with summarization
        if isinstance(data, list):
            segments = data
            text = extract_text_from_segments(segments)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError("Kobold.cpp: Invalid input data format")

        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
        }

        kobold_prompt = f"{custom_prompt_input}\n\n{text}"
        logging.debug("kobold: Prompt being sent is {kobold_prompt}")

        # FIXME
        # Values literally c/p from the api docs....
        data = {
            "prompt": kobold_prompt,
            "temperature": 0.7,
            #"top_p": 0.9,
            #"top_k": 100
            #"rep_penalty": 1.0,
            "stream": streaming
        }

        logging.debug("kobold: Submitting request to API endpoint")
        print("kobold: Submitting request to API endpoint")
        kobold_api_ip = loaded_config_data['kobold_api']['api_ip']

        if streaming:
            logging.debug("Kobold Summarization: Streaming mode enabled")
            try:
                # Send the request with streaming enabled
                # Get the Streaming API IP from the config
                kobold_openai_api_IP = loaded_config_data['kobold_api']['api_streaming_ip']
                response = requests.post(
                    kobold_openai_api_IP, headers=headers, json=data, stream=True
                )
                logging.debug(
                    "Kobold Summarization: API Response Status Code: %d",
                    response.status_code,
                )

                if response.status_code == 200:
                    # Process the streamed response
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')
                            logging.debug(
                                "Kobold: Received streamed data: %s", decoded_line
                            )
                            # OpenAI API streams data prefixed with 'data: '
                            if decoded_line.startswith('data: '):
                                content = decoded_line[len('data: '):].strip()
                                if content == '[DONE]':
                                    break
                                try:
                                    data_chunk = json.loads(content)
                                    if 'choices' in data_chunk and len(data_chunk['choices']) > 0:
                                        delta = data_chunk['choices'][0].get('delta', {})
                                        text = delta.get('content', '')
                                        if text:
                                            yield text
                                    else:
                                        logging.error(
                                            "Kobold: Expected data not found in streamed response."
                                        )
                                except json.JSONDecodeError as e:
                                    logging.error(
                                        "Kobold: Error decoding streamed JSON: %s", str(e)
                                    )
                            else:
                                logging.debug("Kobold: Ignoring line: %s", decoded_line)
                else:
                    logging.error(
                        f"Kobold: API request failed with status code {response.status_code}: {response.text}"
                    )
                    yield f"Kobold: API request failed: {response.text}"
            except Exception as e:
                logging.error("Kobold: Error in processing: %s", str(e))
                yield f"Kobold: Error occurred while processing summary with Kobold: {str(e)}"
        else:
            try:
                response = requests.post(
                    kobold_api_ip, headers=headers, json=data
                )
                logging.debug(
                    "Kobold Summarization: API Response Status Code: %d",
                    response.status_code,
                )

                if response.status_code == 200:
                    try:
                        response_data = response.json()
                        logging.debug("Kobold: API Response Data: %s", response_data)

                        if (
                            response_data
                            and 'results' in response_data
                            and len(response_data['results']) > 0
                        ):
                            summary = response_data['results'][0]['text'].strip()
                            logging.debug("Kobold: Chat request successful")
                            return summary
                        else:
                            logging.error("Expected data not found in API response.")
                            return "Expected data not found in API response."
                    except ValueError as e:
                        logging.error(
                            "Kobold: Error parsing JSON response: %s", str(e)
                        )
                        return f"Error parsing JSON response: {str(e)}"
                else:
                    logging.error(
                        f"Kobold: API request failed with status code {response.status_code}: {response.text}"
                    )
                    return f"Kobold: API request failed: {response.text}"
            except Exception as e:
                logging.error("Kobold: Error in processing: %s", str(e))
                return f"Kobold: Error occurred while processing summary with Kobold: {str(e)}"
    except Exception as e:
        logging.error("kobold: Error in processing: %s", str(e))
        return f"kobold: Error occurred while processing chat response with kobold: {str(e)}"


# System prompt doesn't work. FIXME
# https://github.com/oobabooga/text-generation-webui/wiki/12-%E2%80%90-OpenAI-API
def chat_with_oobabooga(input_data, api_key, custom_prompt, api_url="http://127.0.0.1:5000/v1/chat/completions", system_prompt=None, streaming=False):
    loaded_config_data = load_and_log_configs()
    try:
        # API key validation
        if api_key is None:
            logging.info("ooba: API key not provided as parameter")
            logging.info("ooba: Attempting to use API key from config file")
            api_key = loaded_config_data['ooba_api']['api_key']

        if api_key is None or api_key.strip() == "":
            logging.info("ooba: API key not found or is empty")

        if system_prompt is None:
            system_prompt = "You are a helpful AI assistant that provides accurate and concise information."

        if isinstance(streaming, str):
            streaming = streaming.lower() == "true"
        elif isinstance(streaming, int):
            streaming = bool(streaming)  # Convert integers (1/0) to boolean
        elif streaming is None:
            streaming = loaded_config_data.get('ooba_api', {}).get('streaming', False)
            logging.debug("Oobabooga: Streaming mode enabled")
        else:
            logging.debug("Oobabooga: Streaming mode disabled")
        if not isinstance(streaming, bool):
            raise ValueError(f"Invalid type for 'streaming': Expected a boolean, got {type(streaming).__name__}")

        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
        }

        # prompt_text = "I like to eat cake and bake cakes. I am a baker. I work in a French bakery baking cakes. It
        # is a fun job. I have been baking cakes for ten years. I also bake lots of other baked goods, but cakes are
        # my favorite." prompt_text += f"\n\n{text}"  # Uncomment this line if you want to include the text variable
        ooba_prompt = f"{input_data}" + f"\n\n\n\n{custom_prompt}"
        logging.debug("ooba: Prompt being sent is {ooba_prompt}")

        data = {
            "mode": "chat",
            "character": "Example",
            "messages": [{"role": "user", "content": ooba_prompt}],
            "stream": streaming
        }
        if streaming:
            logging.debug("Ooba Summarization: Streaming mode enabled")
            try:
                # Send the request with streaming enabled
                response = requests.post(
                    api_url, headers=headers, json=data, stream=True
                )
                logging.debug(
                    "Ooba Summarization: API Response Status Code: %d",
                    response.status_code,
                )

                if response.status_code == 200:
                    # Process the streamed response
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')
                            logging.debug(
                                "Ooba: Received streamed data: %s", decoded_line
                            )
                            # OpenAI API streams data prefixed with 'data: '
                            if decoded_line.startswith('data: '):
                                content = decoded_line[len('data: '):].strip()
                                if content == '[DONE]':
                                    break
                                try:
                                    data_chunk = json.loads(content)
                                    if 'choices' in data_chunk and len(data_chunk['choices']) > 0:
                                        delta = data_chunk['choices'][0].get('delta', {})
                                        text = delta.get('content', '')
                                        if text:
                                            yield text
                                    else:
                                        logging.error(
                                            "Ooba: Expected data not found in streamed response."
                                        )
                                except json.JSONDecodeError as e:
                                    logging.error(
                                        "Ooba: Error decoding streamed JSON: %s", str(e)
                                    )
                            else:
                                logging.debug("Ooba: Ignoring line: %s", decoded_line)
                else:
                    logging.error(
                        f"Ooba: API request failed with status code {response.status_code}: {response.text}"
                    )
                    yield f"Ooba: API request failed: {response.text}"
            except Exception as e:
                logging.error("Ooba: Error in processing: %s", str(e))
                yield f"Ooba: Error occurred while processing summary with Ooba: {str(e)}"
        else:
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


def chat_with_tabbyapi(
    input_data,
    custom_prompt_input,
    system_message=None,
    api_key=None,
    temp=None,
    api_IP="http://127.0.0.1:5000/v1/chat/completions",
    streaming=False
):
    logging.debug("TabbyAPI: Chat request process starting...")
    try:
        logging.debug("TabbyAPI: Loading and validating configurations")
        loaded_config_data = load_and_log_configs()
        if loaded_config_data is None:
            logging.error("Failed to load configuration data")
            tabby_api_key = None
        else:
            # Prioritize the API key passed as a parameter
            if api_key and api_key.strip():
                tabby_api_key = api_key
                logging.info("TabbyAPI: Using API key provided as parameter")
            else:
                # If no parameter is provided, use the key from the config
                tabby_api_key = loaded_config_data['tabby_api'].get('api_key')
                if tabby_api_key:
                    logging.info("TabbyAPI: Using API key from config file")
                else:
                    logging.warning("TabbyAPI: No API key found in config file")

        if isinstance(streaming, str):
            streaming = streaming.lower() == "true"
        elif isinstance(streaming, int):
            streaming = bool(streaming)  # Convert integers (1/0) to boolean
        elif streaming is None:
            streaming = loaded_config_data.get('tabby_api', {}).get('streaming', False)
            logging.debug("TabbyAPI: Streaming mode enabled")
        else:
            logging.debug("TabbyAPI: Streaming mode disabled")
        if not isinstance(streaming, bool):
            raise ValueError(f"Invalid type for 'streaming': Expected a boolean, got {type(streaming).__name__}")

        # Set API IP and model from config.txt
        tabby_api_ip = loaded_config_data['tabby_api']['api_ip']
        tabby_model = loaded_config_data['tabby_api']['model']
        if temp is None:
            temp = 0.7

        logging.debug(f"TabbyAPI: Using API Key: {tabby_api_key[:5]}...{tabby_api_key[-5:] if tabby_api_key else 'None'}")

        if system_message is None:
            system_message = "You are a helpful AI assistant."

        if custom_prompt_input is None:
            custom_prompt_input = f"{input_data}"
        else:
            custom_prompt_input = f"{custom_prompt_input}\n\n{input_data}"

        headers = {
            'Content-Type': 'application/json'
        }
        if tabby_api_key:
            headers['Authorization'] = f'Bearer {tabby_api_key}'

        data2 = {
            'model': tabby_model,
            'messages': [
                {'role': 'system',
                 'content': system_message
                 },
                {'role': 'user',
                 'content': custom_prompt_input
                 }
            ],
            'temperature': temp,
            'max_tokens': 4096,
            "min_tokens": 0,
            #'top_p': 1.0,
            #'top_k': 0,
            #'frequency_penalty': 0,
            #'presence_penalty': 0.0,
            #"repetition_penalty": 1.0,
            "stream": streaming
        }

        if streaming:
            logging.debug("TabbyAPI: Streaming mode enabled for chat request")
            try:
                response = requests.post(tabby_api_ip, headers=headers, json=data2, stream=True)
                response.raise_for_status()
                # Process the streamed response
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8').strip()
                        if decoded_line.startswith('data: '):
                            data_line = decoded_line[len('data: '):]
                            if data_line == '[DONE]':
                                break
                            try:
                                data_json = json.loads(data_line)
                                if 'choices' in data_json and len(data_json['choices']) > 0:
                                    delta = data_json['choices'][0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        yield content
                            except json.JSONDecodeError as e:
                                logging.error(f"TabbyAPI: Failed to parse JSON streamed data: {str(e)}")
                        else:
                            logging.debug(f"TabbyAPI: Received non-data line: {decoded_line}")
            except requests.exceptions.RequestException as e:
                logging.error(f"TabbyAPI: Error making chat request with TabbyAPI: {e}")
                yield f"TabbyAPI: Error making chat request with TabbyAPI: {str(e)}"
            except Exception as e:
                logging.error(f"Unexpected error in making chat request with summarize_with_tabbyapi: {e}")
                yield f"TabbyAPI: Unexpected error in making chat request: {str(e)}"
        else:
            try:
                response = requests.post(tabby_api_ip, headers=headers, json=data2)
                response.raise_for_status()
                response_json = response.json()

                # Validate the response structure
                if all(key in response_json for key in ['id', 'choices', 'created', 'model', 'object', 'usage']):
                    logging.info("TabbyAPI: Received a valid 200 response")
                    chat_response = response_json['choices'][0].get('message', {}).get('content', '')
                    return chat_response
                else:
                    logging.error("TabbyAPI: Received a 200 response, but the structure is invalid")
                    return "Error: Received an invalid response structure from TabbyAPI."

            except requests.exceptions.RequestException as e:
                logging.error(f"Error making chat request with TabbyAPI: {e}")
                return f"Error making chat request with TabbyAPI: {str(e)}"
            except json.JSONDecodeError:
                logging.error("TabbyAPI: Received an invalid JSON response")
                return "TabbyAPI: Error: Received an invalid JSON response from TabbyAPI."
            except Exception as e:
                logging.error(f"Unexpected error in chat_with_tabbyapi: {e}")
                return f"TabbyAPI: Unexpected error in chat request process: {str(e)}"

    except Exception as e:
        logging.error(f"TabbyAPI: Unexpected error in chat_with_tabbyapi: {e}")
        if streaming:
            yield f"TabbyAPI: Unexpected error in chat request process: {str(e)}"
        else:
            return f"TabbyAPI: Unexpected error in chat request process: {str(e)}"


# FIXME aphrodite engine - code was literally tab complete in one go from copilot... :/
def chat_with_aphrodite(input_data, custom_prompt_input, api_key=None, api_IP="http://127.0.0.1:8080/completion", streaming=False):
    loaded_config_data = load_and_log_configs()
    model = loaded_config_data['aphrodite_api']['model']
    # API key validation
    if api_key is None:
        logging.info("aphrodite: API key not provided as parameter")
        logging.info("aphrodite: Attempting to use API key from config file")
        api_key = loaded_config_data['aphrodite_api']['api_key']

    if api_key is None or api_key.strip() == "":
        logging.info("aphrodite: API key not found or is empty")

        if isinstance(streaming, str):
            streaming = streaming.lower() == "true"
        elif isinstance(streaming, int):
            streaming = bool(streaming)  # Convert integers (1/0) to boolean
        elif streaming is None:
            streaming = loaded_config_data.get('aphrodite_api', {}).get('streaming', False)
            logging.debug("Aphrodite: Streaming mode enabled")
        else:
            logging.debug("Aphrodite: Streaming mode disabled")
        if not isinstance(streaming, bool):
            raise ValueError(f"Invalid type for 'streaming': Expected a boolean, got {type(streaming).__name__}")

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    data2 = {
        'text': input_data,
    }
    try:
        response = requests.post(api_IP, headers=headers, json=data2)
        response.raise_for_status()
        summary = response.json().get('summary', '')
        return summary
    except requests.exceptions.RequestException as e:
        logging.error(f"Error summarizing with Aphrodite: {e}")
        return "Error summarizing with Aphrodite."


def chat_with_ollama(
    input_data,
    custom_prompt,
    api_url="http://127.0.0.1:11434/v1/chat/completions",
    api_key=None,
    temp=None,
    system_message=None,
    model=None,
    max_retries=5,
    retry_delay=20,
    streaming=False
):
    try:
        logging.debug("Ollama: Loading and validating configurations")
        loaded_config_data = load_and_log_configs()
        if loaded_config_data is None:
            logging.error("Failed to load configuration data")
            ollama_api_key = None
        else:
            # Prioritize the API key passed as a parameter
            if api_key and api_key.strip():
                ollama_api_key = api_key
                logging.info("Ollama: Using API key provided as parameter")
            else:
                # If no parameter is provided, use the key from the config
                ollama_api_key = loaded_config_data['ollama_api'].get('api_key')
                if ollama_api_key:
                    logging.info("Ollama: Using API key from config file")
                else:
                    logging.warning("Ollama: No API key found in config file")

        # Set model from parameter or config
        if model is None:
            model = loaded_config_data['ollama_api'].get('model')
            if model is None:
                logging.error("Ollama: Model not found in config file")
                return "Ollama: Model not found in config file"

        # Set api_url from parameter or config
        if api_url is None:
            api_url = loaded_config_data['ollama_api'].get('api_ip')
            if api_url is None:
                logging.error("Ollama: API URL not found in config file")
                return "Ollama: API URL not found in config file"

        if isinstance(streaming, str):
            streaming = streaming.lower() == "true"
        elif isinstance(streaming, int):
            streaming = bool(streaming)  # Convert integers (1/0) to boolean
        elif streaming is None:
            streaming = loaded_config_data.get('ollama_api', {}).get('streaming', False)
            logging.debug("Ollama: Streaming mode enabled")
        else:
            logging.debug("Ollama: Streaming mode disabled")
        if not isinstance(streaming, bool):
            raise ValueError(f"Invalid type for 'streaming': Expected a boolean, got {type(streaming).__name__}")

        # Load transcript
        logging.debug("Ollama: Loading JSON data")
        if isinstance(input_data, str) and os.path.isfile(input_data):
            logging.debug("Ollama: Loading json data for summarization")
            with open(input_data, 'r') as file:
                data = json.load(file)
        else:
            logging.debug("Ollama: Using provided string data for summarization")
            data = input_data

        logging.debug(f"Ollama: Loaded data: {data}")
        logging.debug(f"Ollama: Type of data: {type(data)}")

        if isinstance(data, dict) and 'summary' in data:
            # If the loaded data is a dictionary and already contains a summary, return it
            logging.debug("Ollama: Summary already exists in the loaded data")
            return data['summary']

        # If the loaded data is a list of segment dictionaries or a string, proceed with summarization
        if isinstance(data, list):
            segments = data
            text = extract_text_from_segments(segments)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError("Ollama: Invalid input data format")

        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
        }
        if ollama_api_key and len(ollama_api_key) > 5:
            headers['Authorization'] = f'Bearer {ollama_api_key}'

        ollama_prompt = f"{custom_prompt}\n\n{text}"
        if system_message is None:
            system_message = "You are a helpful AI assistant."
        logging.debug(f"Ollama: Prompt being sent is: {ollama_prompt}")

        data_payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": ollama_prompt
                }
            ],
            "temperature": temp,
            "stream": streaming
        }

        if streaming:
            # Add streaming support
            data_payload['stream'] = True

            for attempt in range(1, max_retries + 1):
                logging.debug("Ollama: Submitting streaming request to API endpoint")
                print("Ollama: Submitting streaming request to API endpoint")
                try:
                    response = requests.post(api_url, headers=headers, json=data_payload, stream=True)
                    response.raise_for_status()  # Raises HTTPError for bad responses

                    # Process the streamed response
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')
                            logging.debug(f"Ollama: Received line: {decoded_line}")
                            try:
                                json_data = json.loads(decoded_line)
                                if 'response' in json_data:
                                    text_chunk = json_data['response']
                                    yield text_chunk
                                if json_data.get('done', False):
                                    logging.debug("Ollama: Streaming complete.")
                                    break
                            except json.JSONDecodeError:
                                logging.error("Ollama: Failed to decode JSON from streamed line.")
                    return  # Exit after streaming is complete
                except requests.exceptions.Timeout:
                    logging.error("Ollama: Request timed out.")
                    yield "Ollama: Request timed out."
                except requests.exceptions.HTTPError as http_err:
                    logging.error(f"Ollama: HTTP error occurred: {http_err}")
                    yield f"Ollama: HTTP error occurred: {http_err}"
                except requests.exceptions.RequestException as req_err:
                    logging.error(f"Ollama: Request exception: {req_err}")
                    yield f"Ollama: Request exception: {req_err}"
                except Exception as e:
                    logging.error(f"Ollama: An unexpected error occurred: {str(e)}")
                    yield f"Ollama: An unexpected error occurred: {str(e)}"
                break  # Break out of retry loop after successful streaming
        else:
            for attempt in range(1, max_retries + 1):
                logging.debug("Ollama: Submitting request to API endpoint")
                print("Ollama: Submitting request to API endpoint")
                try:
                    config = load_and_log_configs()
                    local_api_timeout = config['local_api_timeout']
                    response = requests.post(api_url, headers=headers, json=data_payload, timeout=local_api_timeout)
                    response.raise_for_status()  # Raises HTTPError for bad responses
                    response_data = response.json()
                except requests.exceptions.Timeout:
                    logging.error("Ollama: Request timed out.")
                    return "Ollama: Request timed out."
                except requests.exceptions.HTTPError as http_err:
                    logging.error(f"Ollama: HTTP error occurred: {http_err}")
                    return f"Ollama: HTTP error occurred: {http_err}"
                except requests.exceptions.RequestException as req_err:
                    logging.error(f"Ollama: Request exception: {req_err}")
                    return f"Ollama: Request exception: {req_err}"
                except json.JSONDecodeError:
                    logging.error("Ollama: Failed to decode JSON response")
                    return "Ollama: Failed to decode JSON response."
                except Exception as e:
                    logging.error(f"Ollama: An unexpected error occurred: {str(e)}")
                    return f"Ollama: An unexpected error occurred: {str(e)}"

                logging.debug(f"API Response Data: {response_data}")

                if response.status_code == 200:
                    # Inspect available keys
                    available_keys = list(response_data.keys())
                    logging.debug(f"Ollama: Available keys in response: {available_keys}")

                    # Attempt to retrieve 'response'
                    summary = None
                    if 'response' in response_data and response_data['response']:
                        summary = response_data['response'].strip()
                    elif 'choices' in response_data and len(response_data['choices']) > 0:
                        choice = response_data['choices'][0]
                        if 'message' in choice and 'content' in choice['message']:
                            summary = choice['message']['content'].strip()

                    if summary:
                        logging.debug("Ollama: Chat request successful")
                        print("\n\nChat request successful.")
                        return summary
                    elif response_data.get('done_reason') == 'load':
                        logging.warning(f"Ollama: Model is loading. Attempt {attempt} of {max_retries}. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        logging.error("Ollama: API response does not contain 'response' or 'choices'.")
                        return "Ollama: API response does not contain 'response' or 'choices'."
                else:
                    logging.error(f"Ollama: API request failed with status code {response.status_code}: {response.text}")
                    return f"Ollama: API request failed: {response.text}"

            logging.error("Ollama: Maximum retry attempts reached. Model is still loading.")
            return "Ollama: Maximum retry attempts reached. Model is still loading."

    except Exception as e:
        logging.error("\n\nOllama: Error in processing: %s", str(e))
        return f"Ollama: Error occurred while processing summary with Ollama: {str(e)}"


def chat_with_vllm(
        input_data: Union[str, dict, list],
        custom_prompt_input: str,
        api_key: str = None,
        vllm_api_url: str = "http://127.0.0.1:8000/v1/chat/completions",
        model: str = None,
        system_prompt: str = None,
        temp: float = 0.7,
        streaming=False
) -> str:
    logging.debug("vLLM: Chat request being made...")
    try:
        logging.debug("vLLM: Loading and validating configurations")
        loaded_config_data = load_and_log_configs()
        if loaded_config_data is None:
            logging.error("Failed to load configuration data")
            vllm_api_key = None
        else:
            # Prioritize the API key passed as a parameter
            if api_key and api_key.strip():
                vllm_api_key = api_key
                logging.info("vLLM: Using API key provided as parameter")
            else:
                # If no parameter is provided, use the key from the config
                vllm_api_key = loaded_config_data['vllm_api'].get('api_key')
                if vllm_api_key:
                    logging.info("vLLM: Using API key from config file")
                else:
                    logging.warning("vLLM: No API key found in config file")
            if 'api_ip' in loaded_config_data['vllm_api']:
                vllm_api_url = loaded_config_data['vllm_api']['api_ip']
                logging.info(f"vLLM: Using API URL from config file: {vllm_api_url}")
            else:
                logging.error("vLLM: API URL not found in config file")

        if isinstance(streaming, str):
            streaming = streaming.lower() == "true"
        elif isinstance(streaming, int):
            streaming = bool(streaming)  # Convert integers (1/0) to boolean
        elif streaming is None:
            streaming = loaded_config_data.get('vllm_api', {}).get('streaming', False)
            logging.debug("vllm: Streaming mode enabled")
        else:
            logging.debug("vllm: Streaming mode disabled")
        if not isinstance(streaming, bool):
            raise ValueError(f"Invalid type for 'streaming': Expected a boolean, got {type(streaming).__name__}")

        logging.debug(f"vLLM: Using API Key: {vllm_api_key[:5]}...{vllm_api_key[-5:] if vllm_api_key else 'None'}")
        # Process input data

        if system_prompt is None:
            system_prompt = "You are a helpful AI assistant."

        model = model or loaded_config_data['vllm_api']['model']
        if system_prompt is None:
            system_prompt = "You are a helpful AI assistant."

        # Prepare the API request
        headers = {
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{custom_prompt_input}\n\n{input_data}"},
            ],
            "temperature": temp,
            "stream": streaming
        }

        logging.debug(f"vLLM: Sending request to {vllm_api_url}")

        if streaming:
            # Send the request with streaming enabled
            response = requests.post(vllm_api_url, headers=headers, json=payload, stream=True)
            response.raise_for_status()
            # Process the streamed response
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8').strip()
                    if decoded_line.startswith('data: '):
                        data_line = decoded_line[len('data: '):]
                        if data_line == '[DONE]':
                            break
                        try:
                            data_json = json.loads(data_line)
                            if 'choices' in data_json and len(data_json['choices']) > 0:
                                delta = data_json['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    yield content
                        except json.JSONDecodeError as e:
                            logging.error(f"vLLM: Failed to parse JSON streamed data: {str(e)}")
                    else:
                        logging.debug(f"vLLM: Received non-data line: {decoded_line}")
        else:
            # Make the API call
            logging.debug(f"vLLM: Sending request to {vllm_api_url}")
            response = requests.post(vllm_api_url, headers=headers, json=payload)
            # Check for successful response
            response.raise_for_status()
            # Extract and return the summary
            response_data = response.json()
            if 'choices' in response_data and len(response_data['choices']) > 0:
                summary = response_data['choices'][0]['message']['content']
                logging.debug("vLLM: Summarization successful")
                logging.debug(f"vLLM: Summary (first 500 chars): {summary[:500]}...")
                return summary
            else:
                raise ValueError("Unexpected response format from vLLM API")

    except requests.RequestException as e:
        logging.error(f"vLLM: API request failed: {str(e)}")
        if streaming:
            yield f"Error: vLLM API request failed - {str(e)}"
        else:
            return f"Error: vLLM API request failed - {str(e)}"
    except json.JSONDecodeError as e:
        logging.error(f"vLLM: Failed to parse API response: {str(e)}")
        if streaming:
            yield f"Error: Failed to parse vLLM API response - {str(e)}"
        else:
            return f"Error: Failed to parse vLLM API response - {str(e)}"
    except Exception as e:
        logging.error(f"vLLM: Unexpected error during summarization: {str(e)}")
        if streaming:
            yield f"Error: Unexpected error during vLLM summarization - {str(e)}"
        else:
            return f"Error: Unexpected error during vLLM summarization - {str(e)}"


def chat_with_custom_openai(api_key, input_data, custom_prompt_arg, temp=None, system_message=None, streaming=False):
    loaded_config_data = load_and_log_configs()
    custom_openai_api_key = api_key
    try:
        # API key validation
        if not custom_openai_api_key:
            logging.info("Custom OpenAI API: API key not provided as parameter")
            logging.info("Custom OpenAI API: Attempting to use API key from config file")
            custom_openai_api_key = loaded_config_data['custom_openai_api']['api_key']

        if not custom_openai_api_key:
            logging.error("Custom OpenAI API: API key not found or is empty")
            return "Custom OpenAI API: API Key Not Provided/Found in Config file or is empty"

        logging.debug(f"Custom OpenAI API: Using API Key: {custom_openai_api_key[:5]}...{custom_openai_api_key[-5:]}")

        if isinstance(streaming, str):
            streaming = streaming.lower() == "true"
        elif isinstance(streaming, int):
            streaming = bool(streaming)  # Convert integers (1/0) to boolean
        elif streaming is None:
            streaming = loaded_config_data.get('custom_openai_api', {}).get('streaming', False)
            logging.debug("Custom OpenAI API: Streaming mode enabled")
        else:
            logging.debug("Custom OpenAI API: Streaming mode disabled")
        if not isinstance(streaming, bool):
            raise ValueError(f"Invalid type for 'streaming': Expected a boolean, got {type(streaming).__name__}")

        # Input data handling
        logging.debug(f"Custom OpenAI API: Raw input data type: {type(input_data)}")
        logging.debug(f"Custom OpenAI API: Raw input data (first 500 chars): {str(input_data)[:500]}...")

        if isinstance(input_data, str):
            if input_data.strip().startswith('{'):
                # It's likely a JSON string
                logging.debug("Custom OpenAI API: Parsing provided JSON string data for summarization")
                try:
                    data = json.loads(input_data)
                except json.JSONDecodeError as e:
                    logging.error(f"Custom OpenAI API: Error parsing JSON string: {str(e)}")
                    return f"Custom OpenAI API: Error parsing JSON input: {str(e)}"
            elif os.path.isfile(input_data):
                logging.debug("Custom OpenAI API: Loading JSON data from file for summarization")
                with open(input_data, 'r') as file:
                    data = json.load(file)
            else:
                logging.debug("Custom OpenAI API: Using provided string data for summarization")
                data = input_data
        else:
            data = input_data

        logging.debug(f"Custom OpenAI API: Processed data type: {type(data)}")
        logging.debug(f"Custom OpenAI API: Processed data (first 500 chars): {str(data)[:500]}...")

        # Text extraction
        if isinstance(data, dict):
            if 'summary' in data:
                logging.debug("Custom OpenAI API: Summary already exists in the loaded data")
                return data['summary']
            elif 'segments' in data:
                text = extract_text_from_segments(data['segments'])
            else:
                text = json.dumps(data)  # Convert dict to string if no specific format
        elif isinstance(data, list):
            text = extract_text_from_segments(data)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError(f"Custom OpenAI API: Invalid input data format: {type(data)}")

        logging.debug(f"Custom OpenAI API: Extracted text (first 500 chars): {text[:500]}...")
        logging.debug(f"v: Custom prompt: {custom_prompt_arg}")

        openai_model = loaded_config_data['custom_openai_api']['model']
        logging.debug(f"Custom OpenAI API: Using model: {openai_model}")

        headers = {
            'Authorization': f'Bearer {custom_openai_api_key}',
            'Content-Type': 'application/json'
        }

        logging.debug(
            f"OpenAI API Key: {custom_openai_api_key[:5]}...{custom_openai_api_key[-5:] if custom_openai_api_key else None}")
        logging.debug("Custom OpenAI API: Preparing data + prompt for submittal")
        openai_prompt = f"{text} \n\n\n\n{custom_prompt_arg}"
        if temp is None:
            temp = 0.7
        if system_message is None:
            system_message = "You are a helpful AI assistant who does whatever the user requests."
        temp = float(temp)
        data = {
            "model": openai_model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": openai_prompt}
            ],
            "max_tokens": 4096,
            "temperature": temp,
            "stream": streaming
        }

        custom_openai_url = loaded_config_data['custom_openai_api']['api_ip']

        if streaming:
            response = requests.post(
                custom_openai_url,
                headers=headers,
                json=data,
                stream=True
            )
            response.raise_for_status()

            def stream_generator():
                collected_messages = ""
                for line in response.iter_lines():
                    line = line.decode("utf-8").strip()

                    if line == "":
                        continue

                    if line.startswith("data: "):
                        data_str = line[len("data: "):]
                        if data_str == "[DONE]":
                            break
                        try:
                            data_json = json.loads(data_str)
                            chunk = data_json["choices"][0]["delta"].get("content", "")
                            collected_messages += chunk
                            yield chunk
                        except json.JSONDecodeError:
                            logging.error(f"OpenAI: Error decoding JSON from line: {line}")
                            continue
                yield collected_messages
            return stream_generator()
        else:
            logging.debug("Custom OpenAI API: Posting request")
            response = requests.post(custom_openai_url, headers=headers, json=data)
            logging.debug(f"Custom OpenAI API full API response data: {response}")
            if response.status_code == 200:
                response_data = response.json()
                logging.debug(response_data)
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    chat_response = response_data['choices'][0]['message']['content'].strip()
                    logging.debug("Custom OpenAI API: Chat Sent successfully")
                    logging.debug(f"Custom OpenAI API: Chat response: {chat_response}")
                    return chat_response
                else:
                    logging.warning("Custom OpenAI API: Chat response not found in the response data")
                    return "Custom OpenAI API: Chat not available"
            else:
                logging.error(f"Custom OpenAI API: Chat request failed with status code {response.status_code}")
                logging.error(f"Custom OpenAI API: Error response: {response.text}")
                return f"OpenAI: Failed to process chat response. Status code: {response.status_code}"
    except json.JSONDecodeError as e:
        logging.error(f"Custom OpenAI API: Error decoding JSON: {str(e)}", exc_info=True)
        return f"Custom OpenAI API: Error decoding JSON input: {str(e)}"
    except requests.RequestException as e:
        logging.error(f"Custom OpenAI API: Error making API request: {str(e)}", exc_info=True)
        return f"Custom OpenAI API: Error making API request: {str(e)}"
    except Exception as e:
        logging.error(f"Custom OpenAI API: Unexpected error: {str(e)}", exc_info=True)
        return f"Custom OpenAI API: Unexpected error occurred: {str(e)}"


def save_summary_to_file(summary, file_path):
    logging.debug("Now saving summary to file...")
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    summary_file_path = os.path.join(os.path.dirname(file_path), base_name + '_summary.txt')
    os.makedirs(os.path.dirname(summary_file_path), exist_ok=True)
    logging.debug("Opening summary file for writing, *segments.json with *_summary.txt")
    with open(summary_file_path, 'w') as file:
        file.write(summary)
    logging.info(f"Summary saved to file: {summary_file_path}")

#
#
#######################################################################################################################



