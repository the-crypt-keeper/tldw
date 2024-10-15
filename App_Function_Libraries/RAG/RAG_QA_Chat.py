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
import time
from typing import List, Tuple, IO, Union
#
# Local Imports
from App_Function_Libraries.DB.DB_Manager import db, search_db, DatabaseError, get_media_content
from App_Function_Libraries.RAG.RAG_Library_2 import generate_answer
from App_Function_Libraries.Metrics.metrics_logger import log_counter, log_histogram
#
########################################################################################################################
#
# Functions:

def rag_qa_chat(message: str, history: List[Tuple[str, str]], context: Union[str, IO[str]], api_choice: str) -> Tuple[List[Tuple[str, str]], str]:
    log_counter("rag_qa_chat_attempt", labels={"api_choice": api_choice})
    start_time = time.time()
    try:
        # Prepare the context based on the selected source
        if hasattr(context, 'read'):
            # Handle uploaded file
            context_text = context.read()
            if isinstance(context_text, bytes):
                context_text = context_text.decode('utf-8')
            log_counter("rag_qa_chat_uploaded_file")
        elif isinstance(context, str) and context.startswith("media_id:"):
            # Handle existing file or search result
            media_id = int(context.split(":")[1])
            context_text = get_media_content(media_id)
            log_counter("rag_qa_chat_existing_media", labels={"media_id": media_id})
        else:
            context_text = str(context)
            log_counter("rag_qa_chat_string_context")

        # Prepare the full context including chat history
        full_context = "\n".join([f"Human: {h[0]}\nAI: {h[1]}" for h in history])
        full_context += f"\n\nContext: {context_text}\n\nHuman: {message}\nAI:"

        # Generate response using the selected API
        response = generate_answer(api_choice, full_context, message)

        # Update history
        history.append((message, response))

        chat_duration = time.time() - start_time
        log_histogram("rag_qa_chat_duration", chat_duration, labels={"api_choice": api_choice})
        log_counter("rag_qa_chat_success", labels={"api_choice": api_choice})

        return history, ""
    except DatabaseError as e:
        log_counter("rag_qa_chat_database_error", labels={"error": str(e)})
        logging.error(f"Database error in rag_qa_chat: {str(e)}")
        return history, f"An error occurred while accessing the database: {str(e)}"
    except Exception as e:
        log_counter("rag_qa_chat_unexpected_error", labels={"error": str(e)})
        logging.error(f"Unexpected error in rag_qa_chat: {str(e)}")
        return history, f"An unexpected error occurred: {str(e)}"



def save_chat_history(history: List[Tuple[str, str]]) -> str:
    # Save chat history to a file
    log_counter("save_chat_history_attempt")
    start_time = time.time()
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
            json.dump(history, temp_file)
            save_duration = time.time() - start_time
            log_histogram("save_chat_history_duration", save_duration)
            log_counter("save_chat_history_success")
            return temp_file.name
    except Exception as e:
        log_counter("save_chat_history_error", labels={"error": str(e)})
        logging.error(f"Error saving chat history: {str(e)}")
        raise


def load_chat_history(file: IO[str]) -> List[Tuple[str, str]]:
    log_counter("load_chat_history_attempt")
    start_time = time.time()
    try:
        # Load chat history from a file
        history = json.load(file)
        load_duration = time.time() - start_time
        log_histogram("load_chat_history_duration", load_duration)
        log_counter("load_chat_history_success")
        return history
    except Exception as e:
        log_counter("load_chat_history_error", labels={"error": str(e)})
        logging.error(f"Error loading chat history: {str(e)}")
        raise

def search_database(query: str) -> List[Tuple[int, str]]:
    try:
        log_counter("search_database_attempt")
        start_time = time.time()
        # Implement database search functionality
        results = search_db(query, ["title", "content"], "", page=1, results_per_page=10)
        search_duration = time.time() - start_time
        log_histogram("search_database_duration", search_duration)
        log_counter("search_database_success", labels={"result_count": len(results)})
        return [(result['id'], result['title']) for result in results]
    except Exception as e:
        log_counter("search_database_error", labels={"error": str(e)})
        logging.error(f"Error searching database: {str(e)}")
        raise


def get_existing_files() -> List[Tuple[int, str]]:
    log_counter("get_existing_files_attempt")
    start_time = time.time()
    try:
        # Fetch list of existing files from the database
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, title FROM Media ORDER BY title")
            results = cursor.fetchall()
        fetch_duration = time.time() - start_time
        log_histogram("get_existing_files_duration", fetch_duration)
        log_counter("get_existing_files_success", labels={"file_count": len(results)})
        return results
    except Exception as e:
        log_counter("get_existing_files_error", labels={"error": str(e)})
        logging.error(f"Error fetching existing files: {str(e)}")
        raise

#
# End of RAG_QA_Chat.py
########################################################################################################################
