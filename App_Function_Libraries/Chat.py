# Chat.py
# Chat functions for interacting with the LLMs as chatbots
import base64
# Imports
import json
import logging
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
#
# External Imports
#
# Local Imports
from App_Function_Libraries.DB.DB_Manager import get_conversation_name, save_chat_history_to_database
from App_Function_Libraries.LLM_API_Calls import chat_with_openai, chat_with_anthropic, chat_with_cohere, \
    chat_with_groq, chat_with_openrouter, chat_with_deepseek, chat_with_mistral, chat_with_huggingface  #, chat_with_vllm
from App_Function_Libraries.LLM_API_Calls_Local import chat_with_aphrodite, chat_with_local_llm, chat_with_ollama, \
    chat_with_kobold, chat_with_llama, chat_with_oobabooga, chat_with_tabbyapi, chat_with_vllm, chat_with_custom_openai
from App_Function_Libraries.DB.SQLite_DB import load_media_content
from App_Function_Libraries.Utils.Utils import generate_unique_filename
#
####################################################################################################
#
# Functions:

def chat_api_call(api_endpoint, api_key, input_data, prompt, temp, system_message=None):
    if not api_key:
        api_key = None
    try:
        logging.info(f"Debug - Chat API Call - API Endpoint: {api_endpoint}")
        logging.info(f"Debug - Chat API Call - API Key: {api_key}")
        logging.info(f"Debug - Chat chat_api_call - API Endpoint: {api_endpoint}")
        if api_endpoint.lower() == 'openai':
            response = chat_with_openai(api_key, input_data, prompt, temp, system_message)
        elif api_endpoint.lower() == "anthropic":
            response = chat_with_anthropic(api_key, input_data, prompt, temp, system_message)
        elif api_endpoint.lower() == "cohere":
            response = chat_with_cohere(api_key, input_data, prompt, temp, system_message)
        elif api_endpoint.lower() == "groq":
            response = chat_with_groq(api_key, input_data, prompt, temp, system_message)
        elif api_endpoint.lower() == "openrouter":
            response = chat_with_openrouter(api_key, input_data, prompt, temp, system_message)
        elif api_endpoint.lower() == "deepseek":
            response = chat_with_deepseek(api_key, input_data, prompt, temp, system_message)
        elif api_endpoint.lower() == "mistral":
            response = chat_with_mistral(api_key, input_data, prompt, temp, system_message)
        elif api_endpoint.lower() == "llama.cpp":
            response = chat_with_llama(input_data, prompt, temp, system_message)
        elif api_endpoint.lower() == "kobold":
            response = chat_with_kobold(input_data, api_key, prompt, temp, system_message)
        elif api_endpoint.lower() == "ooba":
            response = chat_with_oobabooga(input_data, api_key, prompt, temp, system_message)
        elif api_endpoint.lower() == "tabbyapi":
            response = chat_with_tabbyapi(input_data, prompt, temp, system_message)
        elif api_endpoint.lower() == "vllm":
            response = chat_with_vllm(input_data, prompt, system_message)
        elif api_endpoint.lower() == "local-llm":
            response = chat_with_local_llm(input_data, prompt, temp, system_message)
        elif api_endpoint.lower() == "huggingface":
            response = chat_with_huggingface(api_key, input_data, prompt, temp)  # , system_message)
        elif api_endpoint.lower() == "ollama":
            response = chat_with_ollama(input_data, prompt, temp, system_message)
        elif api_endpoint.lower() == "aphrodite":
            response = chat_with_aphrodite(input_data, prompt, temp, system_message)
        elif api_endpoint.lower() == "custom-openai-api":
            response = chat_with_custom_openai(api_key, input_data, prompt, temp, system_message)
        else:
            raise ValueError(f"Unsupported API endpoint: {api_endpoint}")

        return response

    except Exception as e:
        logging.error(f"Error in chat function: {str(e)}")
        return f"An error occurred: {str(e)}"

def chat(message, history, media_content, selected_parts, api_endpoint, api_key, prompt, temperature,
         system_message=None):
    try:
        logging.info(f"Debug - Chat Function - Message: {message}")
        logging.info(f"Debug - Chat Function - Media Content: {media_content}")
        logging.info(f"Debug - Chat Function - Selected Parts: {selected_parts}")
        logging.info(f"Debug - Chat Function - API Endpoint: {api_endpoint}")
        # logging.info(f"Debug - Chat Function - Prompt: {prompt}")

        # Ensure selected_parts is a list
        if not isinstance(selected_parts, (list, tuple)):
            selected_parts = [selected_parts] if selected_parts else []

        # logging.debug(f"Debug - Chat Function - Selected Parts (after check): {selected_parts}")

        # Combine the selected parts of the media content
        combined_content = "\n\n".join(
            [f"{part.capitalize()}: {media_content.get(part, '')}" for part in selected_parts if part in media_content])
        # Print first 500 chars
        # logging.debug(f"Debug - Chat Function - Combined Content: {combined_content[:500]}...")

        # Prepare the input for the API
        if not history:
            input_data = f"{combined_content}\n\nUser: {message}\n"
        else:
            input_data = f"User: {message}\n"
        # Print first 500 chars
        # logging.info(f"Debug - Chat Function - Input Data: {input_data[:500]}...")

        if system_message:
            print(f"System message: {system_message}")
            logging.debug(f"Debug - Chat Function - System Message: {system_message}")
        temperature = float(temperature) if temperature else 0.7
        temp = temperature

        logging.debug("Debug - Chat Function - Temperature: {temperature}")
        logging.debug(f"Debug - Chat Function - API Key: {api_key[:10]}")
        logging.debug(f"Debug - Chat Function - Prompt: {prompt}")

        # Use the existing API request code based on the selected endpoint
        response = chat_api_call(api_endpoint, api_key, input_data, prompt, temp, system_message)

        return response
    except Exception as e:
        logging.error(f"Error in chat function: {str(e)}")
        return f"An error occurred: {str(e)}"



def save_chat_history_to_db_wrapper(chatbot, conversation_id, media_content):
    logging.info(f"Attempting to save chat history. Media content type: {type(media_content)}")
    try:
        # Extract the media_id and media_name from the media_content
        media_id = None
        media_name = None
        if isinstance(media_content, dict):
            logging.debug(f"Media content keys: {media_content.keys()}")
            if 'content' in media_content:
                try:
                    content = media_content['content']
                    if isinstance(content, str):
                        content_json = json.loads(content)
                    elif isinstance(content, dict):
                        content_json = content
                    else:
                        raise ValueError(f"Unexpected content type: {type(content)}")

                    # Use the webpage_url as the media_id
                    media_id = content_json.get('webpage_url')
                    # Use the title as the media_name
                    media_name = content_json.get('title')

                    logging.info(f"Extracted media_id: {media_id}, media_name: {media_name}")
                except json.JSONDecodeError:
                    logging.error("Failed to decode JSON from media_content['content']")
                except Exception as e:
                    logging.error(f"Error processing media_content: {str(e)}")
            else:
                logging.warning("'content' key not found in media_content")
        else:
            logging.warning(f"media_content is not a dictionary. Type: {type(media_content)}")

        if media_id is None:
            # If we couldn't find a media_id, we'll use a placeholder
            media_id = "unknown_media"
            logging.warning(f"Unable to extract media_id from media_content. Using placeholder: {media_id}")

        if media_name is None:
            media_name = "Unnamed Media"
            logging.warning(f"Unable to extract media_name from media_content. Using placeholder: {media_name}")

        # Generate a unique conversation name using media_id and current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        conversation_name = f"Chat_{media_id}_{timestamp}"

        new_conversation_id = save_chat_history_to_database(chatbot, conversation_id, media_id, media_name,
                                                            conversation_name)
        return new_conversation_id, f"Chat history saved successfully as {conversation_name}!"
    except Exception as e:
        error_message = f"Failed to save chat history: {str(e)}"
        logging.error(error_message, exc_info=True)
        return conversation_id, error_message


def save_chat_history(history, conversation_id, media_content):
    try:
        content, conversation_name = generate_chat_history_content(history, conversation_id, media_content)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_conversation_name = re.sub(r'[^a-zA-Z0-9_-]', '_', conversation_name)
        base_filename = f"{safe_conversation_name}_{timestamp}.json"

        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Generate a unique filename
        unique_filename = generate_unique_filename(os.path.dirname(temp_file_path), base_filename)
        final_path = os.path.join(os.path.dirname(temp_file_path), unique_filename)

        # Rename the temporary file to the unique filename
        os.rename(temp_file_path, final_path)

        return final_path
    except Exception as e:
        logging.error(f"Error saving chat history: {str(e)}")
        return None


def generate_chat_history_content(history, conversation_id, media_content):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    conversation_name = get_conversation_name(conversation_id)

    if not conversation_name:
        media_name = extract_media_name(media_content)
        if media_name:
            conversation_name = f"{media_name}-chat"
        else:
            conversation_name = f"chat-{timestamp}"  # Fallback name

    chat_data = {
        "conversation_id": conversation_id,
        "conversation_name": conversation_name,
        "timestamp": timestamp,
        "history": [
            {
                "role": "user" if i % 2 == 0 else "bot",
                "content": msg[0] if isinstance(msg, tuple) else msg
            }
            for i, msg in enumerate(history)
        ]
    }

    return json.dumps(chat_data, indent=2), conversation_name


def extract_media_name(media_content):
    if isinstance(media_content, dict):
        content = media_content.get('content', {})
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                logging.warning("Failed to parse media_content JSON string")
                return None

        # Try to extract title from the content
        if isinstance(content, dict):
            return content.get('title') or content.get('name')

    logging.warning(f"Unexpected media_content format: {type(media_content)}")
    return None


def update_chat_content(selected_item, use_content, use_summary, use_prompt, item_mapping):
    logging.debug(f"Debug - Update Chat Content - Selected Item: {selected_item}\n")
    logging.debug(f"Debug - Update Chat Content - Use Content: {use_content}\n\n\n\n")
    logging.debug(f"Debug - Update Chat Content - Use Summary: {use_summary}\n\n")
    logging.debug(f"Debug - Update Chat Content - Use Prompt: {use_prompt}\n\n")
    logging.debug(f"Debug - Update Chat Content - Item Mapping: {item_mapping}\n\n")

    if selected_item and selected_item in item_mapping:
        media_id = item_mapping[selected_item]
        content = load_media_content(media_id)
        selected_parts = []
        if use_content and "content" in content:
            selected_parts.append("content")
        if use_summary and "summary" in content:
            selected_parts.append("summary")
        if use_prompt and "prompt" in content:
            selected_parts.append("prompt")

        # Modified debug print
        if isinstance(content, dict):
            print(f"Debug - Update Chat Content - Content keys: {list(content.keys())}")
            for key, value in content.items():
                print(f"Debug - Update Chat Content - {key} (first 500 char): {str(value)[:500]}\n\n\n\n")
        else:
            print(f"Debug - Update Chat Content - Content(first 500 char): {str(content)[:500]}\n\n\n\n")

        print(f"Debug - Update Chat Content - Selected Parts: {selected_parts}")
        return content, selected_parts
    else:
        print(f"Debug - Update Chat Content - No item selected or item not in mapping")
        return {}, []

#
# End of Chat functions
##########################################################################################################################


##########################################################################################################################
#
# Character Card Functions

CHARACTERS_FILE = Path('.', 'Helper_Scripts', 'Character_Cards', 'Characters.json')


def save_character(character_data):
    characters_file = os.path.join(os.path.dirname(__file__), '..', 'Helper_Scripts', 'Character_Cards', 'Characters.json')
    characters_dir = os.path.dirname(characters_file)

    try:
        if os.path.exists(characters_file):
            with open(characters_file, 'r') as f:
                characters = json.load(f)
        else:
            characters = {}

        char_name = character_data['name']

        # Save the image separately if it exists
        if 'image' in character_data:
            img_data = base64.b64decode(character_data['image'])
            img_filename = f"{char_name.replace(' ', '_')}.png"
            img_path = os.path.join(characters_dir, img_filename)
            with open(img_path, 'wb') as f:
                f.write(img_data)
            character_data['image_path'] = os.path.abspath(img_path)
            del character_data['image']  # Remove the base64 image data from the JSON

        characters[char_name] = character_data

        with open(characters_file, 'w') as f:
            json.dump(characters, f, indent=2)

        logging.info(f"Character '{char_name}' saved successfully.")
    except Exception as e:
        logging.error(f"Error saving character: {str(e)}")



def load_characters():
    characters_file = os.path.join(os.path.dirname(__file__), '..', 'Helper_Scripts', 'Character_Cards', 'Characters.json')
    if os.path.exists(characters_file):
        with open(characters_file, 'r') as f:
            characters = json.load(f)
        logging.debug(f"Loaded {len(characters)} characters from {characters_file}")
        return characters
    logging.warning(f"Characters file not found: {characters_file}")
    return {}


def get_character_names():
    characters = load_characters()
    return list(characters.keys())


#
# End of Chat.py
##########################################################################################################################