# Chat_Functions.py
# Chat functions for interacting with the LLMs as chatbots
import base64
# Imports
import json
import logging
import os
import random
import re
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
#
# External Imports
#
# Local Imports
from App_Function_Libraries.DB.DB_Manager import start_new_conversation, delete_messages_in_conversation, save_message
from App_Function_Libraries.DB.RAG_QA_Chat_DB import get_db_connection, get_conversation_name
from App_Function_Libraries.LLM_API_Calls import chat_with_openai, chat_with_anthropic, chat_with_cohere, \
    chat_with_groq, chat_with_openrouter, chat_with_deepseek, chat_with_mistral, chat_with_huggingface, chat_with_google
from App_Function_Libraries.LLM_API_Calls_Local import chat_with_aphrodite, chat_with_local_llm, chat_with_ollama, \
    chat_with_kobold, chat_with_llama, chat_with_oobabooga, chat_with_tabbyapi, chat_with_vllm, chat_with_custom_openai
from App_Function_Libraries.DB.SQLite_DB import load_media_content
from App_Function_Libraries.Utils.Utils import generate_unique_filename, load_and_log_configs
from App_Function_Libraries.Metrics.metrics_logger import log_counter, log_histogram
#
####################################################################################################
#
# Functions:

def approximate_token_count(history):
    try:
        total_text = ''
        for user_msg, bot_msg in history:
            if user_msg:
                total_text += user_msg + ' '
            if bot_msg:
                total_text += bot_msg + ' '
        total_tokens = len(total_text.split())
        return total_tokens
    except Exception as e:
        logging.error(f"Error calculating token count: {str(e)}")
        return 0


# FIXME - add model parameter
def chat_api_call(api_endpoint, api_key=None, input_data=None, prompt=None, temp=None, system_message=None, streaming=None, minp=None, maxp=None, model=None, topk=None, topp=None):
    logging.info(f"Debug - Chat API Call - API Endpoint: {api_endpoint}")
    log_counter("chat_api_call_attempt", labels={"api_endpoint": api_endpoint})
    start_time = time.time()
    try:
        logging.info(f"Debug - Chat API Call - API Endpoint: {api_endpoint}")
        logging.info(f"Debug - Chat API Call - API Key: {api_key[:4]}...{api_key[-4:]}")
        logging.info(f"Debug - Chat chat_api_call - API Endpoint: {api_endpoint}")
        if api_endpoint.lower() == 'openai':
            response = chat_with_openai(api_key, input_data, prompt, temp, system_message, streaming, maxp, model)

        elif api_endpoint.lower() == 'anthropic':
            # Retrieve the model from config
            loaded_config_data = load_and_log_configs()
            if not model:
                model = loaded_config_data['anthropic_api']['model']
            response = chat_with_anthropic(
                api_key=api_key,
                input_data=input_data,
                model=model,
                custom_prompt_arg=prompt,
                max_retries=3,
                retry_delay=5,
                system_prompt=system_message,
                streaming=streaming,
            )

        elif api_endpoint.lower() == "cohere":
            response = chat_with_cohere(
                api_key,
                input_data,
                model=model,
                custom_prompt_arg=prompt,
                system_prompt=system_message,
                temp=temp,
                streaming=streaming,
                topp=topp,
                topk=topk
            )

        elif api_endpoint.lower() == "groq":
            response = chat_with_groq(
                api_key=api_key,
                input_data=input_data,
                custom_prompt_arg=prompt,
                temp=temp,
                system_message=system_message,
                streaming=streaming,
                maxp=maxp
            )

        elif api_endpoint.lower() == "openrouter":
            response = chat_with_openrouter(api_key,
                input_data,
                prompt,
                temp,
                system_message,
                streaming,
                topp,
                topk,
                minp
            )

        elif api_endpoint.lower() == "deepseek":
            response = chat_with_deepseek(api_key, input_data, prompt, temp, system_message, streaming, topp)

        elif api_endpoint.lower() == "mistral":
            response = chat_with_mistral(api_key,
                 input_data,
                 prompt,
                 temp,
                 system_message,
                 streaming,
                 topp,
                 model
            )

        elif api_endpoint.lower() == "google":
            response = chat_with_google(api_key,
                input_data,
                prompt,
                temp,
                system_message,
                streaming,
                topp,
                topk
            )

        elif api_endpoint.lower() == "huggingface":
            response = chat_with_huggingface(api_key,
                 input_data,
                 prompt,
                 system_message,
                 temp,
                 streaming
            )

        elif api_endpoint.lower() == "llama.cpp":
            response = chat_with_llama(input_data,
               prompt,
               temp,
               None,
               api_key,
               system_message,
               streaming,
               topk,
               topp,
               minp
            )
        elif api_endpoint.lower() == "kobold":
            response = chat_with_kobold(input_data,
                api_key,
                prompt,
                temp,
                system_message,
                streaming,
                topk,
                topp
            )

        elif api_endpoint.lower() == "ooba":
            response = chat_with_oobabooga(input_data, api_key, prompt, temp, system_message)

        elif api_endpoint.lower() == "tabbyapi":
            response = chat_with_tabbyapi(input_data, prompt, temp, system_message)

        elif api_endpoint.lower() == "vllm":
            response = chat_with_vllm(input_data, prompt, system_message)

        elif api_endpoint.lower() == "local-llm":
            response = chat_with_local_llm(input_data, prompt, temp, system_message)

        elif api_endpoint.lower() == "ollama":
            response = chat_with_ollama(input_data, prompt, None, api_key, temp, system_message)

        elif api_endpoint.lower() == "aphrodite":
            response = chat_with_aphrodite(input_data, prompt, temp, system_message)

        elif api_endpoint.lower() == "custom-openai-api":
            response = chat_with_custom_openai(api_key, input_data, prompt, temp, system_message)

        else:
            raise ValueError(f"Unsupported API endpoint: {api_endpoint}")

        call_duration = time.time() - start_time
        log_histogram("chat_api_call_duration", call_duration, labels={"api_endpoint": api_endpoint})
        log_counter("chat_api_call_success", labels={"api_endpoint": api_endpoint})
        logging.debug(f"Debug - Chat API Call - Response: {response}")
        return response

    except Exception as e:
        log_counter("chat_api_call_error", labels={"api_endpoint": api_endpoint, "error": str(e)})
        logging.error(f"Error in chat function: {str(e)}")
        return f"An error occurred: {str(e)}"


def chat(message, history, media_content, selected_parts, api_endpoint, api_key, custom_prompt, temperature,
         system_message=None, streaming=False, minp=None, maxp=None, model=None, topp=None, topk=None):
    log_counter("chat_attempt", labels={"api_endpoint": api_endpoint})
    start_time = time.time()
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
        input_data = f"{combined_content}\n\n" if combined_content else ""
        for old_message, old_response in history:
            input_data += f"{old_message}\nAssistant: {old_response}\n\n"
        input_data += f"{message}\n"

        if system_message:
            print(f"System message: {system_message}")
            logging.debug(f"Debug - Chat Function - System Message: {system_message}")
        temperature = float(temperature) if temperature else 0.7
        temp = temperature

        logging.debug(f"Debug - Chat Function - Temperature: {temperature}")
        logging.debug(f"Debug - Chat Function - API Key: {api_key[:10]}")
        logging.debug(f"Debug - Chat Function - Prompt: {custom_prompt}")

        # Use the existing API request code based on the selected endpoint
        response = chat_api_call(api_endpoint, api_key, input_data, custom_prompt, temp, system_message, streaming, minp, maxp, model, topp, topk)

        if streaming:
            logging.debug(f"Debug - Chat Function - Response: {response}")
            return response
        else:
            chat_duration = time.time() - start_time
            log_histogram("chat_duration", chat_duration, labels={"api_endpoint": api_endpoint})
            log_counter("chat_success", labels={"api_endpoint": api_endpoint})
            logging.debug(f"Debug - Chat Function - Response: {response}")
            return response
    except Exception as e:
        log_counter("chat_error", labels={"api_endpoint": api_endpoint, "error": str(e)})
        logging.error(f"Error in chat function: {str(e)}")
        return f"An error occurred: {str(e)}"


def save_chat_history_to_db_wrapper(chatbot, conversation_id, media_content, media_name=None):
    log_counter("save_chat_history_to_db_attempt")
    start_time = time.time()
    logging.info(f"Attempting to save chat history. Media content type: {type(media_content)}")

    try:
        # First check if we can access the database
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
        except sqlite3.DatabaseError as db_error:
            logging.error(f"Database is corrupted or inaccessible: {str(db_error)}")
            return conversation_id, "Database error: The database file appears to be corrupted. Please contact support."

        # Now attempt the save
        if not conversation_id:
            # Only for new conversations, not updates
            media_id = None
            if isinstance(media_content, dict) and 'content' in media_content:
                try:
                    content = media_content['content']
                    content_json = content if isinstance(content, dict) else json.loads(content)
                    media_id = content_json.get('webpage_url')
                    media_name = media_name or content_json.get('title', 'Unnamed Media')
                except (json.JSONDecodeError, AttributeError) as e:
                    logging.error(f"Error processing media content: {str(e)}")
                    media_id = "unknown_media"
                    media_name = media_name or "Unnamed Media"
            else:
                media_id = "unknown_media"
                media_name = media_name or "Unnamed Media"

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            conversation_title = f"{media_name}_{timestamp}"
            conversation_id = start_new_conversation(title=conversation_title, media_id=media_id)
            logging.info(f"Created new conversation with ID: {conversation_id}")

        # For both new and existing conversations
        try:
            delete_messages_in_conversation(conversation_id)
            for user_msg, assistant_msg in chatbot:
                if user_msg:
                    save_message(conversation_id, "user", user_msg)
                if assistant_msg:
                    save_message(conversation_id, "assistant", assistant_msg)
        except sqlite3.DatabaseError as db_error:
            logging.error(f"Database error during message save: {str(db_error)}")
            return conversation_id, "Database error: Unable to save messages. Please try again or contact support."

        save_duration = time.time() - start_time
        log_histogram("save_chat_history_to_db_duration", save_duration)
        log_counter("save_chat_history_to_db_success")

        return conversation_id, "Chat history saved successfully!"

    except Exception as e:
        log_counter("save_chat_history_to_db_error", labels={"error": str(e)})
        error_message = f"Failed to save chat history: {str(e)}"
        logging.error(error_message, exc_info=True)
        return conversation_id, error_message


def save_chat_history(history, conversation_id, media_content):
    log_counter("save_chat_history_attempt")
    start_time = time.time()
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

        save_duration = time.time() - start_time
        log_histogram("save_chat_history_duration", save_duration)
        log_counter("save_chat_history_success")
        return final_path
    except Exception as e:
        log_counter("save_chat_history_error", labels={"error": str(e)})
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
    log_counter("update_chat_content_attempt")
    start_time = time.time()
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
        update_duration = time.time() - start_time
        log_histogram("update_chat_content_duration", update_duration)
        log_counter("update_chat_content_success")
        return content, selected_parts
    else:
        log_counter("update_chat_content_error", labels={"error": str("No item selected or item not in mapping")})
        print(f"Debug - Update Chat Content - No item selected or item not in mapping")
        return {}, []

#
# End of Chat functions
#######################################################################################################################


#######################################################################################################################
#
# Chat Dictionary Functions

def parse_user_dict_markdown_file(file_path):
    """
    Parse a Markdown file to extract key-value pairs, including multi-line values.
    """
    replacement_dict = {}
    current_key = None
    current_value = []

    with open(file_path, 'r') as file:
        for line in file:
            # Match lines with "key: value" or "key: |" format
            key_value_match = re.match(r'^\s*(\w+)\s*:\s*(.*)$', line)
            if key_value_match:
                key, value = key_value_match.groups()

                # If the value is "|", prepare for multi-line
                if value.strip() == '|':
                    current_key = key
                    current_value = []
                else:
                    # Single-line key-value pair
                    replacement_dict[key] = value.strip()
            elif current_key:
                # Append multi-line values
                stripped_line = line.strip()
                if stripped_line:  # Skip empty lines
                    current_value.append(stripped_line)
            else:
                continue

            # If we encounter an empty line or EOF, store the multi-line value
            if current_key and (line.strip() == '' or line == ''):
                replacement_dict[current_key] = '\n'.join(current_value)
                current_key, current_value = None, []

    # Handle any remaining multi-line value at EOF
    if current_key:
        replacement_dict[current_key] = '\n'.join(current_value)

    return replacement_dict


# World Info Entry Class with Enhanced Capabilities
class WorldInfoEntry:
    def __init__(self, key, content, probability=100, group=None, timed_effects=None, max_replacements=1):
        self.key = self.compile_key(key)
        self.content = content
        self.probability = probability
        self.group = group
        self.timed_effects = timed_effects or {"sticky": 0, "cooldown": 0, "delay": 0}
        self.last_triggered = None  # Track when it was last triggered (for timed effects)
        self.max_replacements = max_replacements  # New: Limit replacements

    @staticmethod
    def compile_key(key):
        # Compile regex if wrapped with "/" delimiters
        if key.startswith("/") and key.endswith("/"):
            return re.compile(key[1:-1], re.IGNORECASE)
        return key

    def matches(self, text):
        # Match either regex or plain text
        if isinstance(self.key, re.Pattern):
            return self.key.search(text) is not None
        return self.key in text


# Strategy for inclusion
def apply_strategy(entries, strategy="sorted_evenly"):
    if strategy == "sorted_evenly":
        return sorted(entries, key=lambda e: e.key)
    elif strategy == "character_lore_first":
        return sorted(entries, key=lambda e: (e.group != "character", e.key))
    elif strategy == "global_lore_first":
        return sorted(entries, key=lambda e: (e.group != "global", e.key))


# Probability modification of inclusion
def filter_by_probability(entries):
    return [entry for entry in entries if random.randint(1, 100) <= entry.probability]


# Group Scoring - Situation where multiple entries are triggered in different groups in a single message
def group_scoring(entries):
    grouped_entries = {}
    for entry in entries:
        if entry.group:
            grouped_entries.setdefault(entry.group, []).append(entry)

    selected_entries = []
    for group, group_entries in grouped_entries.items():
        selected_entries.append(max(group_entries, key=lambda e: len(re.findall(e.key, e.content)) if e.key else 0))

    return selected_entries

# Timed Effects
def apply_timed_effects(entry, current_time):
    if entry.timed_effects["delay"] > 0:
        if entry.last_triggered is None or current_time - entry.last_triggered < timedelta(seconds=entry.timed_effects["delay"]):
            return False
    if entry.timed_effects["cooldown"] > 0:
        if entry.last_triggered and current_time - entry.last_triggered < timedelta(seconds=entry.timed_effects["cooldown"]):
            return False
    entry.last_triggered = current_time
    return True

# Context/Token Budget Mgmt
def calculate_token_usage(entries):
    return sum(len(entry.content.split()) for entry in entries)

def enforce_token_budget(entries, max_tokens):
    total_tokens = 0
    valid_entries = []
    for entry in entries:
        tokens = len(entry.content.split())
        if total_tokens + tokens <= max_tokens:
            valid_entries.append(entry)
            total_tokens += tokens
    return valid_entries

# Match whole words
def match_whole_words(entries, text):
    matched_entries = []
    for entry in entries:
        if re.search(rf'\b{entry.key}\b', text):
            matched_entries.append(entry)
    return matched_entries

# Token Budget Mgmt
def alert_token_budget_exceeded(entries, max_tokens):
    token_usage = calculate_token_usage(entries)
    if token_usage > max_tokens:
        print(f"Alert: Token budget exceeded! Used: {token_usage}, Allowed: {max_tokens}")

# Single Replacement Function
def apply_replacement_once(text, entry):
    """
    Replaces the 'entry.key' in 'text' exactly once (if found).
    Returns the new text and the number of replacements actually performed.
    """
    if isinstance(entry.key, re.Pattern):
        replaced_text, replaced_count = re.subn(entry.key, entry.content, text, count=1)
    else:
        replaced_count = text.count(entry.key)
        if replaced_count > 0:
            replaced_text = text.replace(entry.key, entry.content, 1)
        else:
            replaced_text = text
    return replaced_text, replaced_count

# Chat Dictionary Pipeline
def process_user_input(user_input, entries, max_tokens, strategy="sorted_evenly"):
    current_time = datetime.now()

    # 1. Match entries using regex or plain text
    matched_entries = [entry for entry in entries if entry.matches(user_input)]

    # 2. Apply group scoring
    matched_entries = group_scoring(matched_entries)

    # 3. Apply probability filter
    matched_entries = filter_by_probability(matched_entries)

    # Apply timed effects
    matched_entries = [entry for entry in matched_entries if apply_timed_effects(entry, current_time)]

    # Enforce token budget
    matched_entries = enforce_token_budget(matched_entries, max_tokens)

    # Alert if token budget exceeded
    alert_token_budget_exceeded(matched_entries, max_tokens)

    # Apply replacement strategy
    matched_entries = apply_strategy(matched_entries, strategy)

    # Generate output with single replacement per match
    for entry in matched_entries:
        if entry.max_replacements > 0:
            user_input, replaced_count = apply_replacement_once(user_input, entry)
            if replaced_count > 0:
                entry.max_replacements -= 1

    return user_input

# Sample Usage:
# entries = [
#     WorldInfoEntry(key="hello", content="Hi there!", probability=90, group="greeting"),
#     WorldInfoEntry(key=re.compile(r"\bweather\b", re.IGNORECASE), content="It's sunny today.", probability=50, group="weather"),
# ]
#
# user_input = "Hello, can you tell me about the weather?"
# max_tokens = 10
#
# response = process_user_input(user_input, entries, max_tokens, strategy="character_lore_first")
# print(response)

#
# End of Chat Dictionary functions
#######################################################################################################################


#######################################################################################################################
#
# Character Card Functions

CHARACTERS_FILE = Path('.', 'Helper_Scripts', 'Character_Cards', 'Characters.json')

def save_character(character_data):
    log_counter("save_character_attempt")
    start_time = time.time()
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

        save_duration = time.time() - start_time
        log_histogram("save_character_duration", save_duration)
        log_counter("save_character_success")
        logging.info(f"Character '{char_name}' saved successfully.")
    except Exception as e:
        log_counter("save_character_error", labels={"error": str(e)})
        logging.error(f"Error saving character: {str(e)}")


def load_characters():
    log_counter("load_characters_attempt")
    start_time = time.time()
    try:
        characters_file = os.path.join(os.path.dirname(__file__), '..', 'Helper_Scripts', 'Character_Cards', 'Characters.json')
        if os.path.exists(characters_file):
            with open(characters_file, 'r') as f:
                characters = json.load(f)
            logging.debug(f"Loaded {len(characters)} characters from {characters_file}")
            load_duration = time.time() - start_time
            log_histogram("load_characters_duration", load_duration)
            log_counter("load_characters_success", labels={"character_count": len(characters)})
            return characters
        else:
            logging.warning(f"Characters file not found: {characters_file}")
            return {}
    except Exception as e:
        log_counter("load_characters_error", labels={"error": str(e)})
        return {}


def get_character_names():
    log_counter("get_character_names_attempt")
    start_time = time.time()
    try:
        characters = load_characters()
        names = list(characters.keys())
        get_names_duration = time.time() - start_time
        log_histogram("get_character_names_duration", get_names_duration)
        log_counter("get_character_names_success", labels={"name_count": len(names)})
        return names
    except Exception as e:
        log_counter("get_character_names_error", labels={"error": str(e)})
        logging.error(f"Error getting character names: {str(e)}")
        return []

#
# End of Chat.py
##########################################################################################################################
