# Character_Interaction_Library_3.py
# Description: Library for character card import functions
#
# Imports
import re
import uuid
from datetime import datetime
import json
import logging
import io
import base64
from typing import Dict, Any, Optional, List, Tuple, Union, cast
#
# External Imports
from PIL import Image
import gradio as gr
#
# Local Imports
from App_Function_Libraries.Chat import chat
from App_Function_Libraries.DB.Character_Chat_DB import (
    add_character_card,
    get_character_cards,
    get_character_card_by_id,
    add_character_chat,
    get_character_chats,
    get_character_chat_by_id,
    update_character_chat,
    delete_character_chat,
    delete_character_card,
    update_character_card, search_character_chats,
)
from App_Function_Libraries.Utils.Utils import sanitize_user_input
#
############################################################################################################
#
# Functions:


#################################################################################
#
# Placeholder functions:

def replace_placeholders(text: str, char_name: str, user_name: str) -> str:
    """
    Replace placeholders in the given text with appropriate values.

    Args:
        text (str): The text containing placeholders.
        char_name (str): The name of the character.
        user_name (str): The name of the user.

    Returns:
        str: The text with placeholders replaced.
    """
    replacements = {
        '{{char}}': char_name,
        '{{user}}': user_name,
        '{{random_user}}': user_name  # Assuming random_user is the same as user for simplicity
    }

    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)

    return text

def replace_user_placeholder(history, user_name):
    """
    Replaces all instances of '{{user}}' in the chat history with the actual user name.

    Args:
        history (list): The current chat history as a list of tuples (user_message, bot_message).
        user_name (str): The name entered by the user.

    Returns:
        list: Updated chat history with placeholders replaced.
    """
    if not user_name:
        user_name = "User"  # Default name if none provided

    updated_history = []
    for user_msg, bot_msg in history:
        # Replace in user message
        if user_msg:
            user_msg = user_msg.replace("{{user}}", user_name)
        # Replace in bot message
        if bot_msg:
            bot_msg = bot_msg.replace("{{user}}", user_name)
        updated_history.append((user_msg, bot_msg))
    return updated_history

#
# End of Placeholder functions
#################################################################################


#################################################################################
#
# Character card import functions:

def import_character_card(file):
    if file is None:
        return None, gr.update(), "No file provided for character card import"

    try:
        if file.name.lower().endswith(('.png', '.webp')):
            json_data = extract_json_from_image(file)
            if not json_data:
                return None, gr.update(), "No character card data found in the image. This might not be a valid character card image."
        elif file.name.lower().endswith('.json'):
            with open(file.name, 'r', encoding='utf-8') as f:
                json_data = f.read()
        else:
            return None, gr.update(), "Unsupported file type. Please upload a PNG/WebP image or a JSON file."

        card_data = import_character_card_json(json_data)
        if not card_data:
            return None, gr.update(), "Failed to parse character card data. The file might not contain valid character information."

        # Save image data for PNG/WebP files
        if file.name.lower().endswith(('.png', '.webp')):
            with Image.open(file) as img:
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                card_data['image'] = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

        # Save character card to database
        character_id = add_character_card(card_data)
        if character_id:
            characters = get_character_cards()
            character_names = [char['name'] for char in characters]
            return card_data, gr.update(
                choices=character_names), f"Character card '{card_data['name']}' imported successfully."
        else:
            return None, gr.update(), f"Failed to save character card '{card_data.get('name', 'Unknown')}'. It may already exist."
    except Exception as e:
        logging.error(f"Error importing character card: {e}")
        return None, gr.update(), f"Error importing character card: {e}"


def import_character_card_json(json_content: str) -> Optional[Dict[str, Any]]:
    try:
        json_content = json_content.strip()
        card_data = json.loads(json_content)

        if 'spec' in card_data and card_data['spec'] == 'chara_card_v2':
            logging.info("Detected V2 character card")
            return parse_v2_card(card_data)
        else:
            logging.info("Assuming V1 character card")
            return parse_v1_card(card_data)
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error parsing JSON: {e}")
    return None

def extract_json_from_image(image_file):
    logging.debug(f"Attempting to extract JSON from image: {image_file.name}")
    try:
        with Image.open(image_file) as img:
            logging.debug("Image opened successfully")
            metadata = img.info
            if 'chara' in metadata:
                logging.debug("Found 'chara' in image metadata")
                chara_content = metadata['chara']
                logging.debug(f"Content of 'chara' metadata (first 100 chars): {chara_content[:100]}...")
                try:
                    decoded_content = base64.b64decode(chara_content).decode('utf-8')
                    logging.debug(f"Decoded content (first 100 chars): {decoded_content[:100]}...")
                    return decoded_content
                except Exception as e:
                    logging.error(f"Error decoding base64 content: {e}")

            logging.warning("'chara' not found in metadata, attempting to find JSON data in image bytes")
            # Alternative method to extract embedded JSON from image bytes if metadata is not available
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            img_str = img_bytes.decode('latin1')  # Use 'latin1' to preserve byte values

            # Search for JSON-like structures in the image bytes
            json_start = img_str.find('{')
            json_end = img_str.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                possible_json = img_str[json_start:json_end+1]
                try:
                    json.loads(possible_json)
                    logging.debug("Found JSON data in image bytes")
                    return possible_json
                except json.JSONDecodeError:
                    logging.debug("No valid JSON found in image bytes")

            logging.warning("No JSON data found in the image")
    except Exception as e:
        logging.error(f"Error extracting JSON from image: {e}")
    return None


def process_chat_history(chat_history: List[Tuple[str, str]], char_name: str, user_name: str) -> List[Tuple[str, str]]:
    """
    Process the chat history to replace placeholders in both user and character messages.

    Args:
        chat_history (List[Tuple[str, str]]): The chat history.
        char_name (str): The name of the character.
        user_name (str): The name of the user.

    Returns:
        List[Tuple[str, str]]: The processed chat history.
    """
    processed_history = []
    for user_msg, char_msg in chat_history:
        if user_msg:
            user_msg = replace_placeholders(user_msg, char_name, user_name)
        if char_msg:
            char_msg = replace_placeholders(char_msg, char_name, user_name)
        processed_history.append((user_msg, char_msg))
    return processed_history

def parse_v2_card(card_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        # Validate spec_version
        if card_data.get('spec_version') != '2.0':
            logging.warning(f"Unsupported V2 spec version: {card_data.get('spec_version')}")
            return None

        data = card_data['data']

        # Ensure all required fields are present
        required_fields = ['name', 'description', 'personality', 'scenario', 'first_mes', 'mes_example']
        for field in required_fields:
            if field not in data:
                logging.error(f"Missing required field in V2 card: {field}")
                return None

        # Handle new V2 fields
        parsed_data = {
            'name': data['name'],
            'description': data['description'],
            'personality': data['personality'],
            'scenario': data['scenario'],
            'first_mes': data['first_mes'],
            'mes_example': data['mes_example'],
            'creator_notes': data.get('creator_notes', ''),
            'system_prompt': data.get('system_prompt', ''),
            'post_history_instructions': data.get('post_history_instructions', ''),
            'alternate_greetings': data.get('alternate_greetings', []),
            'tags': data.get('tags', []),
            'creator': data.get('creator', ''),
            'character_version': data.get('character_version', ''),
            'extensions': data.get('extensions', {})
        }

        # Handle character_book if present
        if 'character_book' in data:
            parsed_data['character_book'] = parse_character_book(data['character_book'])

        return parsed_data
    except KeyError as e:
        logging.error(f"Missing key in V2 card structure: {e}")
    except Exception as e:
        logging.error(f"Error parsing V2 card: {e}")
    return None

def parse_v1_card(card_data: Dict[str, Any]) -> Dict[str, Any]:
    # Ensure all required V1 fields are present
    required_fields = ['name', 'description', 'personality', 'scenario', 'first_mes', 'mes_example']
    for field in required_fields:
        if field not in card_data:
            logging.error(f"Missing required field in V1 card: {field}")
            raise ValueError(f"Missing required field in V1 card: {field}")

    # Convert V1 to V2 format
    v2_data: Dict[str, Union[str, List[str], Dict[str, Any]]] = {
        'name': card_data['name'],
        'description': card_data['description'],
        'personality': card_data['personality'],
        'scenario': card_data['scenario'],
        'first_mes': card_data['first_mes'],
        'mes_example': card_data['mes_example'],
        'creator_notes': cast(str, card_data.get('creator_notes', '')),
        'system_prompt': cast(str, card_data.get('system_prompt', '')),
        'post_history_instructions': cast(str, card_data.get('post_history_instructions', '')),
        'alternate_greetings': cast(List[str], card_data.get('alternate_greetings', [])),
        'tags': cast(List[str], card_data.get('tags', [])),
        'creator': cast(str, card_data.get('creator', '')),
        'character_version': cast(str, card_data.get('character_version', '')),
        'extensions': {}
    }

    # Move any non-standard V1 fields to extensions
    for key, value in card_data.items():
        if key not in v2_data:
            v2_data['extensions'][key] = value

    return v2_data

def extract_character_id(choice: str) -> int:
    """Extract the character ID from the dropdown selection string."""
    return int(choice.split('(ID: ')[1].rstrip(')'))

def load_character_wrapper(character_id: int, user_name: str) -> Tuple[Dict[str, Any], List[Tuple[Optional[str], str]], Optional[Image.Image]]:
    """Wrapper function to load character and image using the extracted ID."""
    char_data, chat_history, img = load_character_and_image(character_id, user_name)
    return char_data, chat_history, img

def parse_character_book(book_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse the character book data from a V2 character card.

    Args:
        book_data (Dict[str, Any]): The raw character book data from the character card.

    Returns:
        Dict[str, Any]: The parsed and structured character book data.
    """
    parsed_book = {
        'name': book_data.get('name', ''),
        'description': book_data.get('description', ''),
        'scan_depth': book_data.get('scan_depth'),
        'token_budget': book_data.get('token_budget'),
        'recursive_scanning': book_data.get('recursive_scanning', False),
        'extensions': book_data.get('extensions', {}),
        'entries': []
    }

    for entry in book_data.get('entries', []):
        parsed_entry = {
            'keys': entry['keys'],
            'content': entry['content'],
            'extensions': entry.get('extensions', {}),
            'enabled': entry['enabled'],
            'insertion_order': entry['insertion_order'],
            'case_sensitive': entry.get('case_sensitive', False),
            'name': entry.get('name', ''),
            'priority': entry.get('priority'),
            'id': entry.get('id'),
            'comment': entry.get('comment', ''),
            'selective': entry.get('selective', False),
            'secondary_keys': entry.get('secondary_keys', []),
            'constant': entry.get('constant', False),
            'position': entry.get('position')
        }
        parsed_book['entries'].append(parsed_entry)

    return parsed_book

def load_character_and_image(character_id: int, user_name: str) -> Tuple[Optional[Dict[str, Any]], List[Tuple[Optional[str], str]], Optional[Image.Image]]:
    """
    Load a character and its associated image based on the character ID.

    Args:
        character_id (int): The ID of the character to load.
        user_name (str): The name of the user, used for placeholder replacement.

    Returns:
        Tuple[Optional[Dict[str, Any]], List[Tuple[Optional[str], str]], Optional[Image.Image]]:
        A tuple containing the character data, chat history, and character image (if available).
    """
    try:
        char_data = get_character_card_by_id(character_id)
        if not char_data:
            logging.warning(f"No character data found for ID: {character_id}")
            return None, [], None

        # Replace placeholders in character data
        for field in ['first_mes', 'mes_example', 'scenario', 'description', 'personality']:
            if field in char_data:
                char_data[field] = replace_placeholders(char_data[field], char_data['name'], user_name)

        # Replace placeholders in first_mes
        first_mes = char_data.get('first_mes', "Hello! I'm ready to chat.")
        first_mes = replace_placeholders(first_mes, char_data['name'], user_name)

        chat_history = [(None, first_mes)] if first_mes else []

        img = None
        if char_data.get('image'):
            try:
                image_data = base64.b64decode(char_data['image'])
                img = Image.open(io.BytesIO(image_data)).convert("RGBA")
            except Exception as e:
                logging.error(f"Error processing image for character '{char_data['name']}': {e}")

        return char_data, chat_history, img

    except Exception as e:
        logging.error(f"Error in load_character_and_image: {e}")
        return None, [], None

def load_chat_and_character(chat_id: int, user_name: str) -> Tuple[Optional[Dict[str, Any]], List[Tuple[str, str]], Optional[Image.Image]]:
    """
    Load a chat and its associated character, including the character image and process templates.

    Args:
        chat_id (int): The ID of the chat to load.
        user_name (str): The name of the user.

    Returns:
        Tuple[Optional[Dict[str, Any]], List[Tuple[str, str]], Optional[Image.Image]]:
        A tuple containing the character data, processed chat history, and character image (if available).
    """
    try:
        # Load the chat
        chat = get_character_chat_by_id(chat_id)
        if not chat:
            logging.warning(f"No chat found with ID: {chat_id}")
            return None, [], None

        # Load the associated character
        character_id = chat['character_id']
        char_data = get_character_card_by_id(character_id)
        if not char_data:
            logging.warning(f"No character found for chat ID: {chat_id}")
            return None, chat['chat_history'], None

        # Process the chat history
        processed_history = process_chat_history(chat['chat_history'], char_data['name'], user_name)

        # Load the character image
        img = None
        if char_data.get('image'):
            try:
                image_data = base64.b64decode(char_data['image'])
                img = Image.open(io.BytesIO(image_data)).convert("RGBA")
            except Exception as e:
                logging.error(f"Error processing image for character '{char_data['name']}': {e}")

        # Process character data templates
        for field in ['first_mes', 'mes_example', 'scenario', 'description', 'personality']:
            if field in char_data:
                char_data[field] = replace_placeholders(char_data[field], char_data['name'], user_name)

        return char_data, processed_history, img

    except Exception as e:
        logging.error(f"Error in load_chat_and_character: {e}")
        return None, [], None


def load_chat_history(file):
    try:
        content = file.read().decode('utf-8')
        chat_data = json.loads(content)

        # Extract history and character name from the loaded data
        history = chat_data.get('history') or chat_data.get('messages')
        character_name = chat_data.get('character') or chat_data.get('character_name')

        if not history or not character_name:
            logging.error("Chat history or character name missing in the imported file.")
            return None, None

        return history, character_name
    except Exception as e:
        logging.error(f"Error loading chat history: {e}")
        return None, None

####################################################
#
# Gradio tabs

def create_character_card_interaction_tab():
    with gr.TabItem("Chat with a Character Card"):
        gr.Markdown("# Chat with a Character Card")
        with gr.Row():
            with gr.Column(scale=1):
                character_image = gr.Image(label="Character Image", type="pil")
                character_card_upload = gr.File(
                    label="Upload Character Card (PNG, WEBP, JSON)",
                    file_types=[".png", ".webp", ".json"]
                )
                import_card_button = gr.Button("Import Character Card")
                load_characters_button = gr.Button("Load Existing Characters")
                character_dropdown = gr.Dropdown(label="Select Character", choices=[])
                user_name_input = gr.Textbox(label="Your Name", placeholder="Enter your name here")
                api_name_input = gr.Dropdown(
                    choices=[
                        "Local-LLM", "OpenAI", "Anthropic", "Cohere", "Groq", "DeepSeek", "Mistral",
                        "OpenRouter", "Llama.cpp", "Kobold", "Ooba", "Tabbyapi", "VLLM", "ollama", "HuggingFace",
                        "Custom-OpenAI-API"
                    ],
                    value="HuggingFace",
                    label="API for Interaction (Mandatory)"
                )
                api_key_input = gr.Textbox(
                    label="API Key (if not set in Config_Files/config.txt)",
                    placeholder="Enter your API key here", type="password"
                )
                temperature_slider = gr.Slider(
                    minimum=0.0, maximum=2.0, value=0.7, step=0.05, label="Temperature"
                )
                import_chat_button = gr.Button("Import Chat History")
                chat_file_upload = gr.File(label="Upload Chat History JSON", visible=True)

                # Chat History Import and Search
                gr.Markdown("## Search and Load Existing Chats")
                chat_search_query = gr.Textbox(
                    label="Search Chats",
                    placeholder="Enter chat name or keywords to search"
                )
                chat_search_button = gr.Button("Search Chats")
                chat_search_dropdown = gr.Dropdown(label="Search Results", choices=[], visible=False)
                load_chat_button = gr.Button("Load Selected Chat", visible=False)

                # Checkbox to Decide Whether to Save Chats by Default
                auto_save_checkbox = gr.Checkbox(label="Save chats automatically", value=True)
                chat_media_name = gr.Textbox(label="Custom Chat Name (optional)", visible=True)
                save_chat_history_to_db = gr.Button("Save Chat History to Database")
                save_status = gr.Textbox(label="Save Status", interactive=False)

            with gr.Column(scale=2):
                chat_history = gr.Chatbot(label="Conversation", height=800)
                user_input = gr.Textbox(label="Your message")
                send_message_button = gr.Button("Send Message")
                regenerate_button = gr.Button("Regenerate Last Message")
                clear_chat_button = gr.Button("Clear Chat")
                save_snapshot_button = gr.Button("Save Chat Snapshot")
                update_chat_dropdown = gr.Dropdown(label="Select Chat to Update", choices=[], visible=False)
                load_selected_chat_button = gr.Button("Load Selected Chat", visible=False)
                update_chat_button = gr.Button("Update Selected Chat", visible=False)

        # States
        character_data = gr.State(None)
        user_name = gr.State("")
        selected_chat_id = gr.State(None)  # To track the selected chat for updates

        # Callback Functions

        def search_existing_chats(query):
            results, message = search_character_chats(query)
            if results:
                # Format search results for dropdown
                formatted_results = [
                    f"{chat['conversation_name']} (ID: {chat['id']})" for chat in results
                ]
            else:
                formatted_results = []
            return formatted_results, message

        def load_selected_chat_from_search(selected_chat, user_name):
            if not selected_chat:
                return None, [], None, "No chat selected."

            try:
                chat_id_match = re.search(r'\(ID:\s*(\d+)\)', selected_chat)
                if not chat_id_match:
                    return None, [], None, "Invalid chat selection format."

                chat_id = int(chat_id_match.group(1))

                # Use the new function to load chat and character data
                char_data, chat_history, img = load_chat_and_character(chat_id, user_name)

                if not char_data:
                    return None, [], None, "Failed to load character data for the selected chat."

                return char_data, chat_history, img, f"Chat '{selected_chat}' loaded successfully."
            except Exception as e:
                logging.error(f"Error loading selected chat: {e}")
                return None, [], None, f"Error loading chat: {e}"


        def import_chat_history(file, current_history, char_data, user_name_val):
            """
            Imports chat history from a file, replacing '{{user}}' with the actual user name.

            Args:
                file (file): The uploaded chat history file.
                current_history (list): The current chat history.
                char_data (dict): The current character data.
                user_name_val (str): The user's name.

            Returns:
                tuple: Updated chat history, updated character data, and a status message.
            """
            loaded_history, char_name = load_chat_history(file)
            if loaded_history is None:
                return current_history, char_data, "Failed to load chat history."

            # Replace '{{user}}' in the loaded chat history
            loaded_history = replace_user_placeholder(loaded_history, user_name_val)

            # Check if the loaded chat is for the current character
            if char_data and char_data.get('name') != char_name:
                return current_history, char_data, (
                    f"Warning: Loaded chat is for character '{char_name}', "
                    f"but current character is '{char_data.get('name')}'. Chat not imported."
                )

            # If no character is selected, try to load the character from the chat
            if not char_data:
                characters = get_character_cards()
                character = next((char for char in characters if char['name'] == char_name), None)
                if character:
                    char_data = character
                    # Replace '{{user}}' in the first_message if necessary
                    if character.get('first_message'):
                        character['first_message'] = character['first_message'].replace("{{user}}",
                                                                                        user_name_val if user_name_val else "User")
                else:
                    return current_history, char_data, (
                        f"Warning: Character '{char_name}' not found. Please select the character manually."
                    )

            return loaded_history, char_data, f"Chat history for '{char_name}' imported successfully."

        def load_character(name):
            characters = get_character_cards()
            character = next((char for char in characters if char['name'] == name), None)
            if character:
                first_message = character.get('first_message', "Hello! I'm ready to chat.")
                return character, [(None, first_message)] if first_message else [], None
            return None, [], None

        def load_character_image(name):
            character = next((char for char in get_character_cards() if char['name'] == name), None)
            if character and 'image' in character and character['image']:
                try:
                    # Decode the base64 image
                    image_data = base64.b64decode(character['image'])
                    # Load as PIL Image
                    img = Image.open(io.BytesIO(image_data)).convert("RGBA")
                    return img
                except Exception as e:
                    logging.error(f"Error loading image for character '{name}': {e}")
                    return None
            return None

        def character_chat_wrapper(
                message, history, char_data, api_endpoint, api_key,
                temperature, user_name_val, auto_save
        ):
            if not char_data:
                return history, "Please select a character first."

            user_name_val = user_name_val or "User"
            char_name = char_data.get('name', 'AI Assistant')

            # Prepare the character's background information
            char_background = f"""
            Name: {char_name}
            Description: {char_data.get('description', 'N/A')}
            Personality: {char_data.get('personality', 'N/A')}
            Scenario: {char_data.get('scenario', 'N/A')}
            """

            # Prepare the system prompt
            system_message = f"""You are roleplaying as {char_name}. {char_data.get('system_prompt', '')}"""

            # Prepare chat context
            media_content = {
                'id': char_name,
                'title': char_name,
                'content': char_background,
                'description': char_data.get('description', ''),
                'personality': char_data.get('personality', ''),
                'scenario': char_data.get('scenario', '')
            }
            selected_parts = ['description', 'personality', 'scenario']

            prompt = char_data.get('post_history_instructions', '')

            # Sanitize and format user message
            user_message = sanitize_user_input(message)
            user_message = replace_placeholders(user_message, char_name, user_name_val)
            full_message = f"{user_name_val}: {user_message}"

            # Generate bot response
            bot_message = chat(
                full_message,
                history,
                media_content,
                selected_parts,
                api_endpoint,
                api_key,
                prompt,
                temperature,
                system_message
            )

            # Replace placeholders in bot message
            bot_message = replace_placeholders(bot_message, char_name, user_name_val)

            # Update history
            history.append((user_message, bot_message))

            # Auto-save if enabled
            save_status = ""
            if auto_save:
                character_id = char_data.get('id')
                if character_id:
                    conversation_name = f"Auto-saved chat {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    add_character_chat(character_id, conversation_name, history)
                    save_status = "Chat auto-saved."
                else:
                    save_status = "Character ID not found; chat not saved."

            return history, save_status

        def save_chat_history_to_db_wrapper(
            chat_history, conversation_id, media_content,
            chat_media_name, char_data, auto_save
        ):
            if not char_data or not chat_history:
                return "No character or chat history available.", ""

            character_id = char_data.get('id')
            if not character_id:
                return "Character ID not found.", ""

            conversation_name = chat_media_name or f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            chat_id = add_character_chat(character_id, conversation_name, chat_history)
            if chat_id:
                return f"Chat saved successfully with ID {chat_id}.", ""
            else:
                return "Failed to save chat.", ""

        def update_character_info(name):
            return load_character_and_image(name, user_name.value)

        def on_character_select(name, user_name_val):
            logging.debug(f"Character selected: {name}")
            char_data, chat_history, img = load_character_and_image(name, user_name_val)
            return char_data, chat_history, img

        def clear_chat_history(char_data, user_name_val):
            """
            Clears the chat history and initializes it with the character's first message,
            replacing the '{{user}}' placeholder with the actual user name.

            Args:
                char_data (dict): The current character data.
                user_name_val (str): The user's name.

            Returns:
                tuple: Updated chat history and the unchanged char_data.
            """
            if char_data and 'first_message' in char_data and char_data['first_message']:
                # Replace '{{user}}' in the first_message
                first_message = char_data['first_message'].replace("{{user}}",
                                                                   user_name_val if user_name_val else "User")
                # Initialize chat history with the updated first_message
                return [(None, first_message)], char_data
            else:
                # If no first_message is defined, simply clear the chat
                return [], char_data

        def regenerate_last_message(
                history, char_data, api_endpoint, api_key,
                temperature, user_name_val, auto_save
        ):
            """
            Regenerates the last bot message by removing it and resending the corresponding user message.

            Args:
                history (list): The current chat history as a list of tuples (user_message, bot_message).
                char_data (dict): The current character data.
                api_endpoint (str): The API endpoint to use for the LLM.
                api_key (str): The API key for authentication.
                temperature (float): The temperature setting for the LLM.
                user_name_val (str): The user's name.
                auto_save (bool): Flag indicating whether to auto-save the chat.

            Returns:
                tuple: Updated chat history and a save status message.
            """
            if not history:
                return history, "No messages to regenerate."

            last_entry = history[-1]
            last_user_message, last_bot_message = last_entry

            # Check if the last bot message exists
            if last_bot_message is None:
                return history, "The last message is not from the bot."

            # Remove the last bot message
            new_history = history[:-1]

            # Resend the last user message to generate a new bot response
            if not last_user_message:
                return new_history, "No user message to regenerate the bot response."

            # Prepare the character's background information
            char_name = char_data.get('name', 'AI Assistant')
            char_background = f"""
            Name: {char_name}
            Description: {char_data.get('description', 'N/A')}
            Personality: {char_data.get('personality', 'N/A')}
            Scenario: {char_data.get('scenario', 'N/A')}
            """

            # Prepare the system prompt for character impersonation
            system_message = f"""You are roleplaying as {char_name}, the character described below. Respond to the user's messages in character, maintaining the personality and background provided. Do not break character or refer to yourself as an AI. Always refer to yourself as "{char_name}" and refer to the user as "{user_name_val}".

            {char_background}

            Additional instructions: {char_data.get('post_history_instructions', '')}
            """

            # Prepare media_content and selected_parts
            media_content = {
                'id': char_name,
                'title': char_name,
                'content': char_background,
                'description': char_data.get('description', ''),
                'personality': char_data.get('personality', ''),
                'scenario': char_data.get('scenario', '')
            }
            selected_parts = ['description', 'personality', 'scenario']

            prompt = char_data.get('post_history_instructions', '')

            # Prepare the input for the chat function
            full_message = f"{user_name_val}: {last_user_message}" if last_user_message else f"{user_name_val}: "

            # Call the chat function to get a new bot message
            bot_message = chat(
                full_message,
                new_history,
                media_content,
                selected_parts,
                api_endpoint,
                api_key,
                prompt,
                temperature,
                system_message
            )

            # Append the new bot message to the history
            new_history.append((last_user_message, bot_message))

            # Auto-save if enabled
            if auto_save:
                character_id = char_data.get('id')
                if character_id:
                    conversation_name = f"Auto-saved chat {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    add_character_chat(character_id, conversation_name, new_history)
                    save_status = "Chat auto-saved."
                else:
                    save_status = "Character ID not found; chat not saved."
            else:
                save_status = ""

            return new_history, save_status

        def toggle_chat_file_upload():
            return gr.update(visible=True)

        def save_untracked_chat_action(history, char_data):
            if not char_data or not history:
                return "No chat to save or character not selected."

            character_id = char_data.get('id')
            if not character_id:
                return "Character ID not found."

            conversation_name = f"Snapshot {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            chat_id = add_character_chat(character_id, conversation_name, history, is_snapshot=True)
            if chat_id:
                return f"Chat snapshot saved successfully with ID {chat_id}."
            else:
                return "Failed to save chat snapshot."

        def select_chat_for_update():
            # Fetch all chats for the selected character
            if character_data.value:
                character_id = character_data.value.get('id')
                if character_id:
                    chats = get_character_chats(character_id)
                    chat_choices = [
                        f"{chat['conversation_name']} (ID: {chat['id']})" for chat in chats
                    ]
                    return gr.update(choices=chat_choices), None
            return gr.update(choices=[]), "No character selected."

        def load_selected_chat(chat_selection):
            if not chat_selection:
                return [], "No chat selected."

            try:
                chat_id = int(chat_selection.split('(ID: ')[1].rstrip(')'))
                chat = get_character_chat_by_id(chat_id)
                if chat:
                    history = chat['chat_history']
                    selected_chat_id.value = chat_id  # Update the selected_chat_id state
                    return history, f"Loaded chat '{chat['conversation_name']}' successfully."
                else:
                    return [], "Chat not found."
            except Exception as e:
                logging.error(f"Error loading selected chat: {e}")
                return [], f"Error loading chat: {e}"

        def update_chat(chat_id, updated_history):
            success = update_character_chat(chat_id, updated_history)
            if success:
                return "Chat updated successfully."
            else:
                return "Failed to update chat."

        # Define States for conversation_id and media_content, which are required for saving chat history
        conversation_id = gr.State(str(uuid.uuid4()))
        media_content = gr.State({})

        # Button Callbacks

        import_card_button.click(
            fn=import_character_card,
            inputs=[character_card_upload],
            outputs=[character_data, character_dropdown, save_status]
        )

        load_characters_button.click(
            fn=lambda: gr.update(choices=[f"{char['name']} (ID: {char['id']})" for char in get_character_cards()]),
            outputs=character_dropdown
        )

        # FIXME user_name_val = validate_user_name(user_name_val)
        clear_chat_button.click(
            fn=clear_chat_history,
            inputs=[character_data, user_name_input],
            outputs=[chat_history, character_data]
        )

        character_dropdown.change(
            fn=extract_character_id,
            inputs=[character_dropdown],
            outputs=character_data
        ).then(
            fn=load_character_wrapper,
            inputs=[character_data, user_name_input],
            outputs=[character_data, chat_history, character_image]
        )

        send_message_button.click(
            fn=character_chat_wrapper,
            inputs=[
                user_input,
                chat_history,
                character_data,
                api_name_input,
                api_key_input,
                temperature_slider,
                user_name_input,
                auto_save_checkbox
            ],
            outputs=[chat_history, save_status]
        ).then(lambda: "", outputs=user_input)

        regenerate_button.click(
            fn=regenerate_last_message,
            inputs=[
                chat_history,
                character_data,
                api_name_input,
                api_key_input,
                temperature_slider,
                user_name_input,
                auto_save_checkbox
            ],
            outputs=[chat_history, save_status]
        )

        import_chat_button.click(
            fn=lambda: gr.update(visible=True),
            outputs=chat_file_upload
        )

        chat_file_upload.change(
            fn=import_chat_history,
            inputs=[chat_file_upload, chat_history, character_data],
            outputs=[chat_history, character_data, save_status]
        )

        save_chat_history_to_db.click(
            fn=save_chat_history_to_db_wrapper,
            inputs=[
                chat_history,
                conversation_id,
                media_content,
                chat_media_name,
                character_data,
                auto_save_checkbox  # Pass the auto_save state
            ],
            outputs=[conversation_id, save_status]
        )

        # Populate the update_chat_dropdown based on selected character
        character_dropdown.change(
            fn=select_chat_for_update,
            inputs=[],
            outputs=[update_chat_dropdown, save_status]
        )

        load_selected_chat_button.click(
            fn=load_selected_chat,
            inputs=[update_chat_dropdown],
            outputs=[chat_history, save_status]
        )

        save_snapshot_button.click(
            fn=save_untracked_chat_action,
            inputs=[chat_history, character_data],
            outputs=save_status
        )

        update_chat_button.click(
            fn=update_chat,
            inputs=[selected_chat_id, chat_history],
            outputs=save_status
        )

        # Search Chats
        chat_search_button.click(
            fn=search_existing_chats,
            inputs=[chat_search_query],
            outputs=[chat_search_dropdown, save_status]
        ).then(
            fn=lambda choices, msg: gr.update(choices=choices, visible=True) if choices else gr.update(visible=False),
            inputs=[chat_search_dropdown, save_status],
            outputs=[chat_search_dropdown]
        )

        # Load Selected Chat from Search
        load_chat_button.click(
            fn=load_selected_chat_from_search,
            inputs=[chat_search_dropdown, user_name_input],
            outputs=[character_data, chat_history, character_image, save_status]
        )

        # Show Load Chat Button when a chat is selected
        chat_search_dropdown.change(
            fn=lambda selected: gr.update(visible=True) if selected else gr.update(visible=False),
            inputs=[chat_search_dropdown],
            outputs=[load_chat_button]
        )


        return character_data, chat_history, user_input, user_name, character_image


def create_character_chat_mgmt_tab():
    with gr.TabItem("Character and Chat Management"):
        gr.Markdown("# Character and Chat Management")

        with gr.Row():
            # Left Column: Character Import and Chat Management
            with gr.Column(scale=1):
                gr.Markdown("## Import Characters")
                character_files = gr.File(
                    label="Upload Character Files (PNG, WEBP, JSON)",
                    file_types=[".png", ".webp", ".json"],
                    file_count="multiple"
                )
                import_characters_button = gr.Button("Import Characters")
                import_status = gr.Markdown("")

            # Right Column: Character Selection and Image Display
            with gr.Column(scale=2):
                gr.Markdown("## Select Character")
                characters = get_character_cards()
                character_choices = [f"{char['name']} (ID: {char['id']})" for char in characters]
                select_character = gr.Dropdown(label="Select Character", choices=character_choices, interactive=True)
                character_image = gr.Image(label="Character Image", type="pil", interactive=False)

                gr.Markdown("## Search Conversations")
                search_query = gr.Textbox(label="Search Conversations", placeholder="Enter search keywords")
                search_button = gr.Button("Search")
                search_results = gr.Dropdown(label="Search Results", choices=[], visible=False)
                search_status = gr.Markdown("", visible=True)

        with gr.Row():
            gr.Markdown("## Chat Management")
            select_chat = gr.Dropdown(label="Select Chat", choices=[], visible=False, interactive=True)
            load_chat_button = gr.Button("Load Selected Chat", visible=False)
            conversation_list = gr.Dropdown(label="Select Conversation or Character", choices=[])
            conversation_mapping = gr.State({})

        with gr.Tabs():
            with gr.TabItem("Edit"):
                chat_content = gr.TextArea(label="Chat/Character Content (JSON)", lines=20, max_lines=50)
                save_button = gr.Button("Save Changes")
                delete_button = gr.Button("Delete Conversation/Character", variant="stop")

            with gr.TabItem("Preview"):
                chat_preview = gr.HTML(label="Chat/Character Preview")
        result_message = gr.Markdown("")

        # Callback Functions

        def load_character_image(character_selection):
            if not character_selection:
                return None

            try:
                character_id = int(character_selection.split('(ID: ')[1].rstrip(')'))
                character = get_character_card_by_id(character_id)
                if character and 'image' in character:
                    image_data = base64.b64decode(character['image'])
                    img = Image.open(io.BytesIO(image_data))
                    return img
            except Exception as e:
                logging.error(f"Error loading character image: {e}")

            return None

        def search_conversations_or_characters(query, selected_character):
            if not query.strip():
                return gr.update(choices=[], visible=False), "Please enter a search query."

            try:
                # Extract character ID from the selected character
                character_id = None
                if selected_character:
                    character_id = int(selected_character.split('(ID: ')[1].rstrip(')'))

                # Search Chats using FTS5, filtered by character_id if provided
                chat_results, chat_message = search_character_chats(query, character_id)

                # Format chat results
                formatted_chat_results = [
                    f"Chat: {chat['conversation_name']} (ID: {chat['id']})" for chat in chat_results
                ]

                # If no character is selected, also search for characters
                if not character_id:
                    characters = get_character_cards()
                    filtered_characters = [
                        char for char in characters
                        if query.lower() in char['name'].lower()
                    ]
                    formatted_character_results = [
                        f"Character: {char['name']} (ID: {char['id']})" for char in filtered_characters
                    ]
                else:
                    formatted_character_results = []

                # Combine results
                all_choices = formatted_chat_results + formatted_character_results

                if all_choices:
                    return gr.update(choices=all_choices, visible=True), chat_message
                else:
                    return gr.update(choices=[], visible=False), f"No results found for '{query}'."

            except Exception as e:
                logging.error(f"Error during search: {e}")
                return gr.update(choices=[], visible=False), f"Error occurred during search: {e}"

        def load_conversation_or_character(selected, conversation_mapping):
            if not selected or selected not in conversation_mapping:
                return "", "<p>No selection made.</p>"

            selected_id = conversation_mapping[selected]
            if selected.startswith("Chat:"):
                chat = get_character_chat_by_id(selected_id)
                if chat:
                    json_content = json.dumps({
                        "conversation_id": chat['id'],
                        "conversation_name": chat['conversation_name'],
                        "messages": chat['chat_history']
                    }, indent=2)

                    html_preview = create_chat_preview_html(chat['chat_history'])
                    return json_content, html_preview
            elif selected.startswith("Character:"):
                character = get_character_card_by_id(selected_id)
                if character:
                    json_content = json.dumps({
                        "id": character['id'],
                        "name": character['name'],
                        "description": character['description'],
                        "personality": character['personality'],
                        "scenario": character['scenario'],
                        "post_history_instructions": character['post_history_instructions'],
                        "first_mes": character['first_mes'],
                        "mes_example": character['mes_example'],
                        "creator_notes": character.get('creator_notes', ''),
                        "system_prompt": character.get('system_prompt', ''),
                        "tags": character.get('tags', []),
                        "creator": character.get('creator', ''),
                        "character_version": character.get('character_version', ''),
                        "extensions": character.get('extensions', {})
                    }, indent=2)

                    html_preview = create_character_preview_html(character)
                    return json_content, html_preview

            return "", "<p>Unable to load the selected item.</p>"

        def validate_content(selected, content):
            try:
                data = json.loads(content)
                if selected.startswith("Chat:"):
                    assert "conversation_id" in data and "messages" in data
                elif selected.startswith("Character:"):
                    assert "id" in data and "name" in data
                return True, data
            except Exception as e:
                return False, f"Invalid JSON: {e}"

        def save_conversation_or_character(selected, conversation_mapping, content):
            if not selected or selected not in conversation_mapping:
                return "Please select an item to save.", "<p>No changes made.</p>"

            is_valid, result = validate_content(selected, content)
            if not is_valid:
                return f"Error: {result}", "<p>No changes made due to validation error.</p>"

            selected_id = conversation_mapping[selected]

            if selected.startswith("Chat:"):
                success = update_character_chat(selected_id, result['messages'])
                return ("Chat updated successfully." if success else "Failed to update chat."), ("<p>Chat updated.</p>" if success else "<p>Failed to update chat.</p>")
            elif selected.startswith("Character:"):
                success = update_character_card(selected_id, result)
                return ("Character updated successfully." if success else "Failed to update character."), ("<p>Character updated.</p>" if success else "<p>Failed to update character.</p>")

            return "Unknown item type.", "<p>No changes made.</p>"

        def delete_conversation_or_character(selected, conversation_mapping):
            if not selected or selected not in conversation_mapping:
                return "Please select an item to delete.", "<p>No changes made.</p>", gr.update(choices=[])

            selected_id = conversation_mapping[selected]

            if selected.startswith("Chat:"):
                success = delete_character_chat(selected_id)
            elif selected.startswith("Character:"):
                success = delete_character_card(selected_id)
            else:
                return "Unknown item type.", "<p>No changes made.</p>", gr.update()

            if success:
                updated_choices = [choice for choice in conversation_mapping.keys() if choice != selected]
                conversation_mapping.value.pop(selected, None)
                return f"{selected.split(':')[0]} deleted successfully.", f"<p>{selected.split(':')[0]} deleted.</p>", gr.update(choices=updated_choices)
            else:
                return f"Failed to delete {selected.split(':')[0].lower()}.", f"<p>Failed to delete {selected.split(':')[0].lower()}.</p>", gr.update()

        def populate_chats(character_selection):
            if not character_selection:
                return gr.update(choices=[], visible=False), "Please select a character first."

            try:
                character_id = int(character_selection.split('(ID: ')[1].rstrip(')'))
                chats = get_character_chats(character_id=character_id)

                if not chats:
                    return gr.update(choices=[], visible=False), f"No chats found for the selected character."

                formatted_chats = [f"{chat['conversation_name']} (ID: {chat['id']})" for chat in chats]
                return gr.update(choices=formatted_chats, visible=True), f"Found {len(formatted_chats)} chat(s)."
            except Exception as e:
                logging.error(f"Error populating chats: {e}")
                return gr.update(choices=[], visible=False), f"Error occurred: {e}"

        def load_chat_from_character(selected_chat):
            if not selected_chat:
                return "", "<p>No chat selected.</p>"

            try:
                chat_id = int(selected_chat.split('(ID: ')[1].rstrip(')'))
                chat = get_character_chat_by_id(chat_id)
                if not chat:
                    return "", "<p>Selected chat not found.</p>"

                json_content = json.dumps({
                    "conversation_id": chat['id'],
                    "conversation_name": chat['conversation_name'],
                    "messages": chat['chat_history']
                }, indent=2)

                html_preview = create_chat_preview_html(chat['chat_history'])
                return json_content, html_preview
            except Exception as e:
                logging.error(f"Error loading chat: {e}")
                return "", f"<p>Error loading chat: {e}</p>"

        def create_chat_preview_html(chat_history):
            html_preview = "<div style='max-height: 500px; overflow-y: auto;'>"
            for user_msg, bot_msg in chat_history:
                user_style = "background-color: #e6f3ff; padding: 10px; border-radius: 5px; margin-bottom: 5px;"
                bot_style = "background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 10px;"
                html_preview += f"<div style='{user_style}'><strong>User:</strong> {user_msg}</div>"
                html_preview += f"<div style='{bot_style}'><strong>Bot:</strong> {bot_msg}</div>"
            html_preview += "</div>"
            return html_preview

        def create_character_preview_html(character):
            return f"""
            <div>
                <h2>{character['name']}</h2>
                <p><strong>Description:</strong> {character['description']}</p>
                <p><strong>Personality:</strong> {character['personality']}</p>
                <p><strong>Scenario:</strong> {character['scenario']}</p>
                <p><strong>First Message:</strong> {character['first_mes']}</p>
                <p><strong>Example Message:</strong> {character['mes_example']}</p>
                <p><strong>Post History Instructions:</strong> {character['post_history_instructions']}</p>
                <p><strong>System Prompt:</strong> {character.get('system_prompt', 'N/A')}</p>
                <p><strong>Tags:</strong> {', '.join(character.get('tags', []))}</p>
                <p><strong>Creator:</strong> {character.get('creator', 'N/A')}</p>
                <p><strong>Version:</strong> {character.get('character_version', 'N/A')}</p>
            </div>
            """
        def import_multiple_characters(files):
            if not files:
                return "No files provided for character import."

            results = []
            for file in files:
                result, _, message = import_character_card(file)
                if result:
                    results.append(f"Imported: {result['name']}")
                else:
                    results.append(f"Failed: {file.name} - {message}")

            # Refresh character choices
            characters = get_character_cards()
            character_choices = [f"{char['name']} (ID: {char['id']})" for char in characters]
            select_character.choices = character_choices

            return "Import results:\n" + "\n".join(results)

        # Register new callback for character import
        import_characters_button.click(
            fn=import_multiple_characters,
            inputs=[character_files],
            outputs=[import_status]
        ).then(
            fn=lambda: gr.update(choices=[f"{char['name']} (ID: {char['id']})" for char in get_character_cards()]),
            outputs=select_character
        )

        # Register Callback Functions with Gradio Components
        search_button.click(
            fn=search_conversations_or_characters,
            inputs=[search_query, select_character],
            outputs=[search_results, search_status]
        )

        search_results.change(
            fn=load_conversation_or_character,
            inputs=[search_results, conversation_mapping],
            outputs=[chat_content, chat_preview]
        )

        save_button.click(
            fn=save_conversation_or_character,
            inputs=[conversation_list, conversation_mapping, chat_content],
            outputs=[result_message, chat_preview]
        )

        delete_button.click(
            fn=delete_conversation_or_character,
            inputs=[conversation_list, conversation_mapping],
            outputs=[result_message, chat_preview, conversation_list]
        )

        select_character.change(
            fn=load_character_image,
            inputs=[select_character],
            outputs=[character_image]
        ).then(
            fn=populate_chats,
            inputs=[select_character],
            outputs=[select_chat, search_status]
        )

        select_chat.change(
            fn=load_chat_from_character,
            inputs=[select_chat],
            outputs=[chat_content, chat_preview]
        )

        load_chat_button.click(
            fn=load_chat_from_character,
            inputs=[select_chat],
            outputs=[chat_content, chat_preview]
        )

        return (
            character_files, import_characters_button, import_status,
            search_query, search_button, search_results, search_status,
            select_character, select_chat, load_chat_button,
            conversation_list, conversation_mapping,
            chat_content, save_button, delete_button,
            chat_preview, result_message, character_image
        )

# def create_character_chat_mgmt_tab():
#     with gr.TabItem("Chat Management"):
#         gr.Markdown("# Chat Management")
#
#         with gr.Row():
#             # Search Section
#             with gr.Column(scale=1):
#                 gr.Markdown("## Search Conversations or Characters")
#                 search_query = gr.Textbox(label="Search Conversations or Characters", placeholder="Enter search keywords")
#                 search_button = gr.Button("Search")
#                 search_results = gr.Dropdown(label="Search Results", choices=[], visible=False)
#                 search_status = gr.Markdown("", visible=True)
#
#             # Select Character and Chat Section
#             with gr.Column(scale=1):
#                 gr.Markdown("## Select Character and Associated Chats")
#                 characters = get_character_cards()
#                 character_choices = [f"{char['name']} (ID: {char['id']})" for char in characters]
#                 select_character = gr.Dropdown(label="Select Character", choices=character_choices, interactive=True)
#                 select_chat = gr.Dropdown(label="Select Chat", choices=[], visible=False, interactive=True)
#                 load_chat_button = gr.Button("Load Selected Chat", visible=False)
#
#         with gr.Row():
#             conversation_list = gr.Dropdown(label="Select Conversation or Character", choices=[])
#             conversation_mapping = gr.State({})
#
#         with gr.Tabs():
#             with gr.TabItem("Edit"):
#                 chat_content = gr.TextArea(label="Chat/Character Content (JSON)", lines=20, max_lines=50)
#                 save_button = gr.Button("Save Changes")
#                 delete_button = gr.Button("Delete Conversation/Character", variant="stop")
#
#             with gr.TabItem("Preview"):
#                 chat_preview = gr.HTML(label="Chat/Character Preview")
#         result_message = gr.Markdown("")
#
#         # Callback Functions
#
#         def search_conversations_or_characters(query):
#             if not query.strip():
#                 return gr.update(choices=[], visible=False), "Please enter a search query."
#
#             try:
#                 # Search Chats using FTS5
#                 chat_results, chat_message = search_character_chats(query)
#
#                 # Format chat results
#                 formatted_chat_results = [
#                     f"Chat: {chat['conversation_name']} (ID: {chat['id']})" for chat in chat_results
#                 ]
#
#                 # Search Characters using substring match
#                 characters = get_character_cards()
#                 filtered_characters = [
#                     char for char in characters
#                     if query.lower() in char['name'].lower()
#                 ]
#                 formatted_character_results = [
#                     f"Character: {char['name']} (ID: {char['id']})" for char in filtered_characters
#                 ]
#
#                 # Combine results
#                 all_choices = formatted_chat_results + formatted_character_results
#                 mapping = {choice: conv['id'] for choice, conv in zip(formatted_chat_results, chat_results)}
#                 mapping.update({choice: char['id'] for choice, char in zip(formatted_character_results, filtered_characters)})
#
#                 if all_choices:
#                     return gr.update(choices=all_choices, visible=True), f"Found {len(all_choices)} result(s) matching '{query}'."
#                 else:
#                     return gr.update(choices=[], visible=False), f"No results found for '{query}'."
#
#             except Exception as e:
#                 logging.error(f"Error during search: {e}")
#                 return gr.update(choices=[], visible=False), f"Error occurred during search: {e}"
#
#         def load_conversation_or_character(selected, conversation_mapping):
#             if not selected or selected not in conversation_mapping:
#                 return "", "<p>No selection made.</p>"
#
#             selected_id = conversation_mapping[selected]
#             if selected.startswith("Chat:"):
#                 chat = get_character_chat_by_id(selected_id)
#                 if chat:
#                     json_content = json.dumps({
#                         "conversation_id": chat['id'],
#                         "conversation_name": chat['conversation_name'],
#                         "messages": chat['chat_history']
#                     }, indent=2)
#
#                     html_preview = create_chat_preview_html(chat['chat_history'])
#                     return json_content, html_preview
#             elif selected.startswith("Character:"):
#                 character = get_character_card_by_id(selected_id)
#                 if character:
#                     json_content = json.dumps({
#                         "id": character['id'],
#                         "name": character['name'],
#                         "description": character['description'],
#                         "personality": character['personality'],
#                         "scenario": character['scenario'],
#                         "post_history_instructions": character['post_history_instructions'],
#                         "first_mes": character['first_mes'],
#                         "mes_example": character['mes_example'],
#                         "creator_notes": character.get('creator_notes', ''),
#                         "system_prompt": character.get('system_prompt', ''),
#                         "tags": character.get('tags', []),
#                         "creator": character.get('creator', ''),
#                         "character_version": character.get('character_version', ''),
#                         "extensions": character.get('extensions', {})
#                     }, indent=2)
#
#                     html_preview = create_character_preview_html(character)
#                     return json_content, html_preview
#
#             return "", "<p>Unable to load the selected item.</p>"
#
#         def validate_content(selected, content):
#             try:
#                 data = json.loads(content)
#                 if selected.startswith("Chat:"):
#                     assert "conversation_id" in data and "messages" in data
#                 elif selected.startswith("Character:"):
#                     assert "id" in data and "name" in data
#                 return True, data
#             except Exception as e:
#                 return False, f"Invalid JSON: {e}"
#
#         def save_conversation_or_character(selected, conversation_mapping, content):
#             if not selected or selected not in conversation_mapping:
#                 return "Please select an item to save.", "<p>No changes made.</p>"
#
#             is_valid, result = validate_content(selected, content)
#             if not is_valid:
#                 return f"Error: {result}", "<p>No changes made due to validation error.</p>"
#
#             selected_id = conversation_mapping[selected]
#
#             if selected.startswith("Chat:"):
#                 success = update_character_chat(selected_id, result['messages'])
#                 return ("Chat updated successfully." if success else "Failed to update chat."), ("<p>Chat updated.</p>" if success else "<p>Failed to update chat.</p>")
#             elif selected.startswith("Character:"):
#                 success = update_character_card(selected_id, result)
#                 return ("Character updated successfully." if success else "Failed to update character."), ("<p>Character updated.</p>" if success else "<p>Failed to update character.</p>")
#
#             return "Unknown item type.", "<p>No changes made.</p>"
#
#         def delete_conversation_or_character(selected, conversation_mapping):
#             if not selected or selected not in conversation_mapping:
#                 return "Please select an item to delete.", "<p>No changes made.</p>", gr.update(choices=[])
#
#             selected_id = conversation_mapping[selected]
#
#             if selected.startswith("Chat:"):
#                 success = delete_character_chat(selected_id)
#             elif selected.startswith("Character:"):
#                 success = delete_character_card(selected_id)
#             else:
#                 return "Unknown item type.", "<p>No changes made.</p>", gr.update()
#
#             if success:
#                 updated_choices = [choice for choice in conversation_mapping.keys() if choice != selected]
#                 conversation_mapping.value.pop(selected, None)
#                 return f"{selected.split(':')[0]} deleted successfully.", f"<p>{selected.split(':')[0]} deleted.</p>", gr.update(choices=updated_choices)
#             else:
#                 return f"Failed to delete {selected.split(':')[0].lower()}.", f"<p>Failed to delete {selected.split(':')[0].lower()}.</p>", gr.update()
#
#         def populate_chats(character_selection):
#             if not character_selection:
#                 return gr.update(choices=[], visible=False), "Please select a character first."
#
#             try:
#                 character_id = int(character_selection.split('(ID: ')[1].rstrip(')'))
#                 chats = get_character_chats(character_id=character_id)
#
#                 if not chats:
#                     return gr.update(choices=[], visible=False), f"No chats found for the selected character."
#
#                 formatted_chats = [f"{chat['conversation_name']} (ID: {chat['id']})" for chat in chats]
#                 return gr.update(choices=formatted_chats, visible=True), f"Found {len(formatted_chats)} chat(s)."
#             except Exception as e:
#                 logging.error(f"Error populating chats: {e}")
#                 return gr.update(choices=[], visible=False), f"Error occurred: {e}"
#
#         def load_chat_from_character(selected_chat):
#             if not selected_chat:
#                 return "", "<p>No chat selected.</p>"
#
#             try:
#                 chat_id = int(selected_chat.split('(ID: ')[1].rstrip(')'))
#                 chat = get_character_chat_by_id(chat_id)
#                 if not chat:
#                     return "", "<p>Selected chat not found.</p>"
#
#                 json_content = json.dumps({
#                     "conversation_id": chat['id'],
#                     "conversation_name": chat['conversation_name'],
#                     "messages": chat['chat_history']
#                 }, indent=2)
#
#                 html_preview = create_chat_preview_html(chat['chat_history'])
#                 return json_content, html_preview
#             except Exception as e:
#                 logging.error(f"Error loading chat: {e}")
#                 return "", f"<p>Error loading chat: {e}</p>"
#
#         def create_chat_preview_html(chat_history):
#             html_preview = "<div style='max-height: 500px; overflow-y: auto;'>"
#             for user_msg, bot_msg in chat_history:
#                 user_style = "background-color: #e6f3ff; padding: 10px; border-radius: 5px; margin-bottom: 5px;"
#                 bot_style = "background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 10px;"
#                 html_preview += f"<div style='{user_style}'><strong>User:</strong> {user_msg}</div>"
#                 html_preview += f"<div style='{bot_style}'><strong>Bot:</strong> {bot_msg}</div>"
#             html_preview += "</div>"
#             return html_preview
#
#         def create_character_preview_html(character):
#             return f"""
#             <div>
#                 <h2>{character['name']}</h2>
#                 <p><strong>Description:</strong> {character['description']}</p>
#                 <p><strong>Personality:</strong> {character['personality']}</p>
#                 <p><strong>Scenario:</strong> {character['scenario']}</p>
#                 <p><strong>First Message:</strong> {character['first_mes']}</p>
#                 <p><strong>Example Message:</strong> {character['mes_example']}</p>
#                 <p><strong>Post History Instructions:</strong> {character['post_history_instructions']}</p>
#                 <p><strong>System Prompt:</strong> {character.get('system_prompt', 'N/A')}</p>
#                 <p><strong>Tags:</strong> {', '.join(character.get('tags', []))}</p>
#                 <p><strong>Creator:</strong> {character.get('creator', 'N/A')}</p>
#                 <p><strong>Version:</strong> {character.get('character_version', 'N/A')}</p>
#             </div>
#             """
#
#         # Register Callback Functions with Gradio Components
#         search_button.click(
#             fn=search_conversations_or_characters,
#             inputs=[search_query],
#             outputs=[search_results, search_status]
#         )
#
#         search_results.change(
#             fn=load_conversation_or_character,
#             inputs=[search_results, conversation_mapping],
#             outputs=[chat_content, chat_preview]
#         )
#
#         save_button.click(
#             fn=save_conversation_or_character,
#             inputs=[conversation_list, conversation_mapping, chat_content],
#             outputs=[result_message, chat_preview]
#         )
#
#         delete_button.click(
#             fn=delete_conversation_or_character,
#             inputs=[conversation_list, conversation_mapping],
#             outputs=[result_message, chat_preview, conversation_list]
#         )
#
#         select_character.change(
#             fn=populate_chats,
#             inputs=[select_character],
#             outputs=[select_chat, search_status]
#         )
#
#         select_chat.change(
#             fn=load_chat_from_character,
#             inputs=[select_chat],
#             outputs=[chat_content, chat_preview]
#         )
#
#         load_chat_button.click(
#             fn=load_chat_from_character,
#             inputs=[select_chat],
#             outputs=[chat_content, chat_preview]
#         )
#
#         return (
#             search_query, search_button, search_results, search_status,
#             select_character, select_chat, load_chat_button,
#             conversation_list, conversation_mapping,
#             chat_content, save_button, delete_button,
#             chat_preview, result_message
#         )

#
# End of Character_Chat_tab.py
#######################################################################################################################