# Character_Interaction_2.py
# Description: Functions for character interaction in Gradio UI
# Imports
import base64
import io
import re
import uuid
from datetime import datetime
import logging
import json
import os
from typing import List

import gradio as gr
from PIL import Image
import sqlite3

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
from App_Function_Libraries.Gradio_UI.Writing_tab import generate_writing_feedback
from App_Function_Libraries.RAG.RAG_Libary_2 import enhanced_rag_pipeline


#
#######################################################################################################################
#
# Functions:

####################################################
#
# RAG

def handle_message(user_message, history, api_choice, keywords):
    """
    Handles user messages by integrating the RAG pipeline.
    """
    try:
        # Retrieve context using RAG
        rag_response = enhanced_rag_pipeline(query=user_message, api_choice=api_choice, keywords=keywords)
        answer = rag_response.get("answer", "I'm sorry, I couldn't process that.")
        context = rag_response.get("context", "")

        # Update chat history
        history = history + [(user_message, answer)]

        # Optionally, log or display the context
        logging.debug(f"Context used for response: {context}")

        return history, ""

    except Exception as e:
        logging.error(f"Error handling message: {e}")
        return history + [(user_message, "An error occurred while processing your request.")], f"Error: {e}"

#
#
#######################################################


####################################################
#
# Utility Functions

def validate_user_name(name):
    """
    Validates the user's name input.

    Args:
        name (str): The user's name.

    Returns:
        str: Validated user name or a default value.
    """
    if not name.strip():
        return "User"  # Default name if input is empty
    return name.strip()

def sanitize_user_input(message):
    """
    Removes or escapes '{{' and '}}' to prevent placeholder injection.

    Args:
        message (str): The user's message.

    Returns:
        str: Sanitized message.
    """
    # Replace '{{' and '}}' with their escaped versions
    message = re.sub(r'\{\{', '{ {', message)
    message = re.sub(r'\}\}', '} }', message)
    return message

def replace_placeholders(text, replacements):
    """
    Replaces placeholders in the text with corresponding values.

    Args:
        text (str): The text containing placeholders.
        replacements (dict): A dictionary of placeholder-value pairs.

    Returns:
        str: The text with placeholders replaced.
    """
    for placeholder, value in replacements.items():
        if placeholder in text:
            logging.debug(f"Replacing {placeholder} with {value}")
            text = text.replace(placeholder, value)
        else:
            logging.debug(f"No occurrence of {placeholder} found in text.")
    return text

# placeholders = {
#     '{{user}}': user_name_val,
#     '{{date}}': datetime.now().strftime('%Y-%m-%d')
# }
# post_history_instructions = replace_placeholders(char_data.get('post_history_instructions', ''), placeholders)

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

def import_character_card(file):
    if file is None:
        logging.warning("No file provided for character card import")
        return None, gr.update(), "No file provided for character card import"

    try:
        # Determine if the file is an image or a JSON file based on the file extension
        if file.name.lower().endswith(('.png', '.webp')):
            logging.info(f"Attempting to import character card from image: {file.name}")
            json_data = extract_json_from_image(file)
            if json_data:
                logging.info("JSON data extracted from image, attempting to parse")
                # Parse the JSON data
                card_data = import_character_card_json(json_data)
                if card_data:
                    # Save the image data
                    with Image.open(file) as img:
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='PNG')
                        card_data['image'] = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                else:
                    logging.warning("Failed to parse character card JSON.")
                    return None, gr.update(), "Failed to parse character card JSON."
            else:
                logging.warning("No JSON data found in the image")
                return None, gr.update(), "No JSON data found in the image."
        elif file.name.lower().endswith('.json'):
            logging.info(f"Attempting to import character card from JSON file: {file.name}")
            # Open and read the JSON file
            with open(file.name, 'r', encoding='utf-8') as f:
                content = f.read()
            # Parse the JSON content
            card_data = import_character_card_json(content)
            if card_data:
                # Ensure all necessary fields are present
                if 'image' not in card_data or not card_data['image']:
                    card_data['image'] = ''
                upload_status_message = "JSON file uploaded successfully."
            else:
                logging.warning("Failed to parse character card JSON.")
                return None, gr.update(), "Failed to parse character card JSON."
        else:
            logging.warning("Unsupported file type for character card import.")
            return None, gr.update(), "Unsupported file type. Please upload a PNG/WebP image or a JSON file."

        # Ensure all necessary fields are present
        if 'post_history_instructions' not in card_data:
            card_data['post_history_instructions'] = ''
        if 'first_message' not in card_data:
            card_data['first_message'] = card_data.get('first_mes', "Hello! I'm ready to chat.")

        # Save character card using the database function
        character_id = add_character_card(card_data)
        if character_id:
            logging.info(f"Character card '{card_data['name']}' saved with ID {character_id}")
            # Update the card_data with the assigned ID
            card_data['id'] = character_id
            # Update the character dropdown choices
            characters = get_character_cards()
            character_names = [char['name'] for char in characters]
            if file.name.lower().endswith('.json'):
                upload_status_message = "Character card JSON imported successfully."
            else:
                upload_status_message = f"Character card '{card_data['name']}' imported successfully from image."
            return card_data, gr.update(choices=character_names), f"Character card '{card_data['name']}' imported successfully.", upload_status_message
        else:
            logging.error("Failed to save character card to database.")
            return None, gr.update(), f"Failed to save character card '{card_data.get('name', 'Unknown')}'. It may already exist.", "Failed to save character card."
    except Exception as e:
        logging.error(f"Error importing character card: {e}")
        return None, gr.update(), f"Error importing character card: {e}", f"Error importing character card: {e}"


def import_character_card_json(json_content):
    try:
        # Remove any leading/trailing whitespace
        json_content = json_content.strip()

        # Log the first 100 characters of the content
        logging.debug(f"JSON content (first 100 chars): {json_content[:100]}...")

        card_data = json.loads(json_content)
        logging.debug(f"Parsed JSON data keys: {list(card_data.keys())}")

        if 'spec' in card_data and card_data['spec'] == 'chara_card_v2':
            logging.info("Detected V2 character card")
            return card_data['data']
        else:
            logging.info("Assuming V1 character card")
            return card_data
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {e}")
        logging.error(f"Problematic JSON content: {json_content[:500]}...")
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

def save_untracked_chat(history, char_data):
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

def regenerate_last_message(
    history, char_data, api_endpoint, api_key,
    temperature, user_name_val, auto_save
):
    if not history:
        return history, ""

    last_user_message = history[-1][0]
    new_history = history[:-1]

    # Re-generate the last message
    bot_message = generate_writing_feedback(
        last_user_message, char_data['name'], "Overall", api_endpoint, api_key
    )
    new_history.append((last_user_message, bot_message))

    # Update history
    history = new_history.copy()

    # Auto-save if enabled
    if auto_save:
        character_id = char_data.get('id')
        if character_id:
            conversation_name = f"Auto-saved chat {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            add_character_chat(character_id, conversation_name, history)
            save_status = "Chat auto-saved."
        else:
            save_status = "Character ID not found; chat not saved."
    else:
        save_status = ""

    return history, save_status

def update_chat(chat_id, updated_history):
    success = update_character_chat(chat_id, updated_history)
    if success:
        return "Chat updated successfully."
    else:
        return "Failed to update chat."


#
# End of Utility Functions
####################################################


####################################################
#
# Gradio tabs

def create_character_card_interaction_tab_two():
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

        def load_selected_chat_from_search(selected_chat):
            if not selected_chat:
                return None, [], None, "No chat selected."

            try:
                # Extract chat ID from the selected option
                chat_id_match = re.search(r'\(ID:\s*(\d+)\)', selected_chat)
                if not chat_id_match:
                    return None, [], None, "Invalid chat selection format."

                chat_id = int(chat_id_match.group(1))

                # Fetch chat details
                chat = get_character_chat_by_id(chat_id)
                if not chat:
                    return None, [], None, "Selected chat not found."

                # Fetch associated character
                character_id = chat['character_id']
                character = get_character_card_by_id(character_id)
                if not character:
                    return None, [], None, "Associated character for the chat not found."

                # Load character data and image
                char_data, _, img = load_character_and_image(character['name'], user_name.value)

                # Overwrite chat_history with the loaded chat
                chat_history_updated = chat['chat_history']

                # Update selected_chat_id state
                selected_chat_id.value = chat_id

                return char_data, chat_history_updated, img, f"Chat '{chat['conversation_name']}' loaded successfully."
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

        def load_character_and_image(name, user_name_val):
            """
            Loads character data and image, replacing '{{user}}' in the first_message.

            Args:
                name (str): The name of the character to load.
                user_name_val (str): The user's name.

            Returns:
                tuple: Character data, updated chat history, and PIL Image object.
            """
            char_data, chat_history, _ = load_character(name)
            if char_data and 'image' in char_data and char_data['image']:
                try:
                    # Decode the base64 image data
                    image_data = base64.b64decode(char_data['image'])
                    # Load the image as a PIL Image and convert to RGBA
                    img = Image.open(io.BytesIO(image_data)).convert("RGBA")
                    return char_data, chat_history, img
                except Exception as e:
                    logging.error(f"Error processing image for character '{name}': {e}")
                    return char_data, chat_history, None
            else:
                # Replace '{{user}}' in chat_history if necessary
                updated_chat_history = replace_user_placeholder(chat_history, user_name_val)
                return char_data, updated_chat_history, None

        def character_chat_wrapper(
                message, history, char_data, api_endpoint, api_key,
                temperature, user_name_val, auto_save
        ):
            logging.debug("Entered character_chat_wrapper")
            if char_data is None:
                return history, "Please select a character first."

            if not user_name_val:
                user_name_val = "User"

            char_name = char_data.get('name', 'AI Assistant')

            # Prepare the character's background information
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

            # Replace '{{user}}' in the user message before sending
            user_message = sanitize_user_input(message).replace("{{user}}", user_name_val if user_name_val else "User")
            full_message = f"{user_name_val}: {user_message}" if user_message else f"{user_name_val}: "

            # Call the chat function
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

            # Replace '{{user}}' in the bot message before appending
            bot_message = bot_message.replace("{{user}}", user_name_val if user_name_val else "User")

            # Update history
            history.append((user_message, bot_message))

            # Auto-save if enabled
            if auto_save:
                character_id = char_data.get('id')
                if character_id:
                    conversation_name = f"Auto-saved chat {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    add_character_chat(character_id, conversation_name, history)
                    save_status = "Chat auto-saved."
                else:
                    save_status = "Character ID not found; chat not saved."
            else:
                save_status = ""

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
            fn=lambda: gr.update(choices=[char['name'] for char in get_character_cards()]),
            outputs=character_dropdown
        )

        # FIXME user_name_val = validate_user_name(user_name_val)
        clear_chat_button.click(
            fn=clear_chat_history,
            inputs=[character_data, user_name_input],
            outputs=[chat_history, character_data]
        )

        character_dropdown.change(
            fn=on_character_select,
            inputs=[character_dropdown],
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
                auto_save_checkbox  # Pass the auto_save state
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
            inputs=[chat_search_dropdown],
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
    with gr.TabItem("Chat Management"):
        gr.Markdown("# Chat Management")

        with gr.Row():
            # Search Section
            with gr.Column(scale=1):
                gr.Markdown("## Search Conversations or Characters")
                search_query = gr.Textbox(label="Search Conversations or Characters", placeholder="Enter search keywords")
                search_button = gr.Button("Search")
                search_results = gr.Dropdown(label="Search Results", choices=[], visible=False)
                search_status = gr.Markdown("", visible=True)

            # Select Character and Chat Section
            with gr.Column(scale=1):
                gr.Markdown("## Select Character and Associated Chats")
                # Fetch characters from the database
                characters = get_character_cards()
                character_names = [char['name'] for char in characters]
                select_character = gr.Dropdown(label="Select Character", choices=character_names, interactive=True)
                select_chat = gr.Dropdown(label="Select Chat", choices=[], visible=False, interactive=True)
                load_chat_button = gr.Button("Load Selected Chat", visible=False)

        with gr.Row():
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

        # 1. Search Functionality with FTS5 Integration
        def search_conversations_or_characters(query):
            """
            Search for conversations using FTS5 and characters using substring match.
            Returns combined results.
            """
            if not query.strip():
                return gr.update(choices=[], visible=False), "Please enter a search query."

            try:
                # Search Chats using FTS5
                chat_results, chat_message = search_character_chats(query)

                # Format chat results
                formatted_chat_results = [
                    f"Chat: {chat['conversation_name']} (ID: {chat['id']})" for chat in chat_results
                ]

                # Search Characters using substring match
                characters = get_character_cards()
                filtered_characters = [
                    char for char in characters
                    if query.lower() in char['name'].lower()
                ]
                formatted_character_results = [
                    f"Character: {char['name']} (ID: {char['id']})" for char in filtered_characters
                ]

                # Combine results
                all_choices = formatted_chat_results + formatted_character_results
                mapping = {choice: conv['id'] for choice, conv in zip(formatted_chat_results, chat_results)}
                mapping.update({choice: char['id'] for choice, char in zip(formatted_character_results, filtered_characters)})

                if all_choices:
                    return gr.update(choices=all_choices, visible=True), f"Found {len(all_choices)} result(s) matching '{query}'."
                else:
                    return gr.update(choices=[], visible=False), f"No results found for '{query}'."

            except Exception as e:
                logging.error(f"Error during search: {e}")
                return gr.update(choices=[], visible=False), f"Error occurred during search: {e}"

        # 2. Load Conversation or Character from Search Results
        def load_conversation_or_character(selected, conversation_mapping):
            if not selected or selected not in conversation_mapping:
                return "", "<p>No selection made.</p>"

            selected_id = conversation_mapping[selected]
            if selected.startswith("Chat:"):
                # Load Chat
                chat = get_character_chat_by_id(selected_id)
                if chat:
                    json_content = json.dumps({
                        "conversation_id": chat['id'],
                        "conversation_name": chat['conversation_name'],
                        "messages": chat['chat_history']
                    }, indent=2)

                    # Create HTML preview
                    html_preview = "<div style='max-height: 500px; overflow-y: auto;'>"
                    for idx, (user_msg, bot_msg) in enumerate(chat['chat_history']):
                        user_style = "background-color: #e6f3ff; padding: 10px; border-radius: 5px;"
                        bot_style = "background-color: #f0f0f0; padding: 10px; border-radius: 5px;"

                        html_preview += f"<div style='margin-bottom: 10px;'>"
                        html_preview += f"<div style='{user_style}'><strong>User:</strong> {user_msg}</div>"
                        html_preview += f"<div style='{bot_style}'><strong>Bot:</strong> {bot_msg}</div>"
                        html_preview += "</div>"
                    html_preview += "</div>"

                    return json_content, html_preview
            elif selected.startswith("Character:"):
                # Load Character
                character = get_character_card_by_id(selected_id)
                if character:
                    json_content = json.dumps({
                        "id": character['id'],
                        "name": character['name'],
                        "description": character['description'],
                        "personality": character['personality'],
                        "scenario": character['scenario'],
                        "post_history_instructions": character['post_history_instructions'],
                        "first_message": character['first_message'],
                        "image": character['image']  # Include image data if necessary
                    }, indent=2)

                    # Create HTML preview
                    html_preview = f"""
                    <div>
                        <h2>{character['name']}</h2>
                        <p><strong>Description:</strong> {character['description']}</p>
                        <p><strong>Personality:</strong> {character['personality']}</p>
                        <p><strong>Scenario:</strong> {character['scenario']}</p>
                        <p><strong>First Message:</strong> {character['first_message']}</p>
                    </div>
                    """

                    return json_content, html_preview

            return "", "<p>Unable to load the selected item.</p>"

        # 3. Validate JSON Content
        def validate_content(selected, content):
            if selected.startswith("Chat:"):
                # Validate Chat JSON
                try:
                    data = json.loads(content)
                    assert "conversation_id" in data and "messages" in data
                    return True, data
                except Exception as e:
                    return False, f"Invalid Chat JSON: {e}"
            elif selected.startswith("Character:"):
                # Validate Character JSON
                try:
                    data = json.loads(content)
                    assert "id" in data and "name" in data
                    return True, data
                except Exception as e:
                    return False, f"Invalid Character JSON: {e}"
            return False, "Unknown selection type."

        # 4. Save Changes to Chat or Character
        def save_conversation_or_character(selected, conversation_mapping, content):
            if not selected or selected not in conversation_mapping:
                return "Please select an item to save.", "<p>No changes made.</p>"

            is_valid, result = validate_content(selected, content)
            if not is_valid:
                return f"Error: {result}", "<p>No changes made due to validation error.</p>"

            selected_id = conversation_mapping[selected]

            if selected.startswith("Chat:"):
                # Update Chat
                chat_data = result
                success = update_character_chat(selected_id, chat_data['messages'])
                if success:
                    return "Chat updated successfully.", "<p>Chat updated.</p>"
                else:
                    return "Failed to update chat.", "<p>Failed to update chat.</p>"
            elif selected.startswith("Character:"):
                # Update Character
                character_data = result
                # Ensure the JSON aligns with the expected fields
                card_data = {
                    'name': character_data.get('name'),
                    'description': character_data.get('description'),
                    'personality': character_data.get('personality'),
                    'scenario': character_data.get('scenario'),
                    'post_history_instructions': character_data.get('post_history_instructions'),
                    'first_message': character_data.get('first_message'),
                    'image': character_data.get('image', '')  # Handle image data appropriately
                }
                success = update_character_card(selected_id, card_data)
                if success:
                    return "Character updated successfully.", "<p>Character updated.</p>"
                else:
                    return "Failed to update character.", "<p>Failed to update character.</p>"

            return "Unknown item type.", "<p>No changes made.</p>"

        # 5. Delete Conversation or Character
        def delete_conversation_or_character(selected, conversation_mapping):
            if not selected or selected not in conversation_mapping:
                return "Please select an item to delete.", "<p>No changes made.</p>", gr.update(choices=[])

            selected_id = conversation_mapping[selected]

            if selected.startswith("Chat:"):
                success = delete_character_chat(selected_id)
                if success:
                    # Remove the deleted item from the conversation list
                    updated_choices = [choice for choice in conversation_mapping.keys() if choice != selected]
                    # Remove the item from the mapping
                    conversation_mapping.value.pop(selected, None)
                    return "Chat deleted successfully.", "<p>Chat deleted.</p>", gr.update(choices=updated_choices)
                else:
                    return "Failed to delete chat.", "<p>Failed to delete chat.</p>", gr.update()
            elif selected.startswith("Character:"):
                success = delete_character_card(selected_id)
                if success:
                    # Remove the deleted item from the conversation list
                    updated_choices = [choice for choice in conversation_mapping.keys() if choice != selected]
                    # Remove the item from the mapping
                    conversation_mapping.value.pop(selected, None)
                    return "Character deleted successfully.", "<p>Character deleted.</p>", gr.update(choices=updated_choices)
                else:
                    return "Failed to delete character.", "<p>Failed to delete character.</p>", gr.update()

            return "Unknown item type.", "<p>No changes made.</p>", gr.update()

        # 6. Populate Select Chat Dropdown Based on Selected Character
        def populate_chats(character_name):
            """
            Retrieves all chats associated with the selected character.
            """
            if not character_name:
                return gr.update(choices=[], visible=False), "Please select a character first."

            try:
                # Fetch character by name to get character_id
                characters = get_character_cards()
                character = next((char for char in characters if char['name'] == character_name), None)
                if not character:
                    return gr.update(choices=[], visible=False), f"Character '{character_name}' not found."

                character_id = character['id']
                chats = get_character_chats(character_id=character_id)

                if not chats:
                    return gr.update(choices=[], visible=False), f"No chats found for character '{character_name}'."

                # Format chat choices
                formatted_chats = [f"{chat['conversation_name']} (ID: {chat['id']})" for chat in chats]

                return gr.update(choices=formatted_chats, visible=True), f"Found {len(formatted_chats)} chat(s) for '{character_name}'."
            except Exception as e:
                logging.error(f"Error populating chats: {e}")
                return gr.update(choices=[], visible=False), f"Error occurred: {e}"

        # 7. Load Selected Chat from Character's Chats
        def load_chat_from_character(selected_chat):
            if not selected_chat:
                return "", "<p>No chat selected.</p>"

            try:
                # Extract chat ID from the selected option
                chat_id_match = re.search(r'\(ID:\s*(\d+)\)', selected_chat)
                if not chat_id_match:
                    return "", "<p>Invalid chat selection format.</p>"

                chat_id = int(chat_id_match.group(1))

                # Fetch chat details
                chat = get_character_chat_by_id(chat_id)
                if not chat:
                    return "", "<p>Selected chat not found.</p>"

                # Load chat content
                json_content = json.dumps({
                    "conversation_id": chat['id'],
                    "conversation_name": chat['conversation_name'],
                    "messages": chat['chat_history']
                }, indent=2)

                # Create HTML preview
                html_preview = "<div style='max-height: 500px; overflow-y: auto;'>"
                for idx, (user_msg, bot_msg) in enumerate(chat['chat_history']):
                    user_style = "background-color: #e6f3ff; padding: 10px; border-radius: 5px;"
                    bot_style = "background-color: #f0f0f0; padding: 10px; border-radius: 5px;"

                    html_preview += f"<div style='margin-bottom: 10px;'>"
                    html_preview += f"<div style='{user_style}'><strong>User:</strong> {user_msg}</div>"
                    html_preview += f"<div style='{bot_style}'><strong>Bot:</strong> {bot_msg}</div>"
                    html_preview += "</div>"
                html_preview += "</div>"

                return json_content, html_preview
            except Exception as e:
                logging.error(f"Error loading chat: {e}")
                return "", f"<p>Error loading chat: {e}</p>"

        # Initialize Select Character Dropdown on Page Load
        # Instead of using gr.on_load, we populate the dropdown when the tab is created
        # Already done by fetching characters above

        # Register Callback Functions with Gradio Components

        # 1. Search Button Click
        search_button.click(
            fn=search_conversations_or_characters,
            inputs=[search_query],
            outputs=[search_results, search_status]
        )

        # 2. Search Results Dropdown Change
        search_results.change(
            fn=load_conversation_or_character,
            inputs=[search_results, conversation_mapping],
            outputs=[chat_content, chat_preview]
        )

        # 3. Save Button Click
        save_button.click(
            fn=save_conversation_or_character,
            inputs=[conversation_list, conversation_mapping, chat_content],
            outputs=[result_message, chat_preview]
        )

        # 4. Delete Button Click
        delete_button.click(
            fn=delete_conversation_or_character,
            inputs=[conversation_list, conversation_mapping],
            outputs=[result_message, chat_preview, conversation_list]
        )

        # 5. Select Character Dropdown Change
        select_character.change(
            fn=populate_chats,
            inputs=[select_character],
            outputs=[select_chat, search_status]
        )

        # 6. Select Chat Dropdown Change
        select_chat.change(
            fn=load_chat_from_character,
            inputs=[select_chat],
            outputs=[chat_content, chat_preview]
        )

        # 7. Load Chat Button Click (Optional: Can be used if you prefer loading via button)
        load_chat_button.click(
            fn=load_chat_from_character,
            inputs=[select_chat],
            outputs=[chat_content, chat_preview]
        )

        return (
            search_query, search_button, search_results, search_status,
            select_character, select_chat, load_chat_button,
            conversation_list, conversation_mapping,
            chat_content, save_button, delete_button,
            chat_preview, result_message
        )

#
# End of Character_Interaction_2.py
#######################################################################################################################
