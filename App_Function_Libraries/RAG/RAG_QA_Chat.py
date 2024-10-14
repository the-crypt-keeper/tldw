# Podcast_tab.py
# Description: Gradio UI for ingesting podcasts into the database
#
# Imports
#
#
# External Imports
import json
import logging
import tempfile
from typing import List, Tuple, IO, Union
#
# Local Imports
from App_Function_Libraries.DB.DB_Manager import db, search_db, DatabaseError, get_media_content
from App_Function_Libraries.RAG.RAG_Library_2 import generate_answer
#
########################################################################################################################
#
# Functions:

def rag_qa_chat(message: str, history: List[Tuple[str, str]], context: Union[str, IO[str]], api_choice: str) -> Tuple[List[Tuple[str, str]], str]:
    try:
        # Prepare the context based on the selected source
        if hasattr(context, 'read'):
            # Handle uploaded file
            context_text = context.read()
            if isinstance(context_text, bytes):
                context_text = context_text.decode('utf-8')
        elif isinstance(context, str) and context.startswith("media_id:"):
            # Handle existing file or search result
            media_id = int(context.split(":")[1])
            context_text = get_media_content(media_id)  # Implement this function to fetch content from the database
        else:
            context_text = str(context)

        # Prepare the full context including chat history
        full_context = "\n".join([f"Human: {h[0]}\nAI: {h[1]}" for h in history])
        full_context += f"\n\nContext: {context_text}\n\nHuman: {message}\nAI:"

        # Generate response using the selected API
        response = generate_answer(api_choice, full_context, message)

        # Update history
        history.append((message, response))

        return history, ""
    except DatabaseError as e:
        logging.error(f"Database error in rag_qa_chat: {str(e)}")
        return history, f"An error occurred while accessing the database: {str(e)}"
    except Exception as e:
        logging.error(f"Unexpected error in rag_qa_chat: {str(e)}")
        return history, f"An unexpected error occurred: {str(e)}"



def save_chat_history(history: List[Tuple[str, str]]) -> str:
    # Save chat history to a file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
        json.dump(history, temp_file)
        return temp_file.name


def load_chat_history(file: IO[str]) -> List[Tuple[str, str]]:
    # Load chat history from a file
    return json.load(file)


def search_database(query: str) -> List[Tuple[int, str]]:
    # Implement database search functionality
    results = search_db(query, ["title", "content"], "", page=1, results_per_page=10)
    return [(result['id'], result['title']) for result in results]


def get_existing_files() -> List[Tuple[int, str]]:
    # Fetch list of existing files from the database
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, title FROM Media ORDER BY title")
        return cursor.fetchall()


#
# End of RAG_QA_Chat.py
########################################################################################################################
