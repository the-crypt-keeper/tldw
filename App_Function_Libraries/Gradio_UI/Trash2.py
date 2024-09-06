# Trash.py
# Gradio UI for deleting items from the database
import html
import sqlite3

# Imports

# External Imports
import gradio as gr
#
# Local Imports
from App_Function_Libraries.DB.DB_Manager import delete_prompt, empty_trash, get_trashed_items, user_delete_item


def delete_item(media_id, force):
    return user_delete_item(media_id, force)

def list_trash():
    items = get_trashed_items()
    return "\n".join(
        [f"ID: {item['id']}, Title: {item['title']}, Trashed on: {item['trash_date']}" for item in items])

def empty_trash_ui(days):
    deleted, remaining = empty_trash(days)
    return f"Deleted {deleted} items. {remaining} items remain in trash."

def create_view_trash_tab():
    with gr.TabItem("View Trash"):
        view_button = gr.Button("View Trash")
        trash_list = gr.Textbox(label="Trashed Items")
        view_button.click(list_trash, inputs=[], outputs=trash_list)




def search_prompts_for_deletion(query):
    try:
        with sqlite3.connect('prompts.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, name, details
                FROM Prompts
                WHERE name LIKE ? OR details LIKE ?
                LIMIT 10
            ''', (f'%{query}%', f'%{query}%'))
            results = cursor.fetchall()

            if not results:
                return "No matching prompts found."

            output = "<h3>Matching Prompts:</h3>"
            for row in results:
                output += f"<p><strong>ID:</strong> {row[0]} | <strong>Name:</strong> {html.escape(row[1])} | <strong>Details:</strong> {html.escape(row[2][:100])}...</p>"
            return output
    except sqlite3.Error as e:
        return f"An error occurred while searching prompts: {e}"


def search_media_for_deletion(query):
    try:
        with sqlite3.connect('media.db') as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, title, description
                FROM media
                WHERE title LIKE ? OR description LIKE ?
                LIMIT 10
            ''', (f'%{query}%', f'%{query}%'))
            results = cursor.fetchall()

            if not results:
                return "No matching media found."

            output = "<h3>Matching Media:</h3>"
            for row in results:
                output += f"<p><strong>ID:</strong> {row[0]} | <strong>Title:</strong> {html.escape(row[1])} | <strong>Description:</strong> {html.escape(row[2][:100])}...</p>"
            return output
    except sqlite3.Error as e:
        return f"An error occurred while searching media: {e}"

def create_delete_trash_tab():
    with gr.TabItem("Delete DB Item"):
        gr.Markdown("# Search and Delete Items from Databases")

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Search and Delete Prompts")
                prompt_search_input = gr.Textbox(label="Search Prompts")
                prompt_search_button = gr.Button("Search Prompts")
                prompt_search_results = gr.HTML()
                prompt_id_input = gr.Number(label="Prompt ID")
                prompt_delete_button = gr.Button("Delete Prompt")
                prompt_delete_output = gr.Textbox(label="Delete Result")

            with gr.Column():
                gr.Markdown("## Search and Delete Media")
                media_search_input = gr.Textbox(label="Search Media")
                media_search_button = gr.Button("Search Media")
                media_search_results = gr.HTML()
                media_id_input = gr.Number(label="Media ID")
                media_force_checkbox = gr.Checkbox(label="Force Delete")
                media_delete_button = gr.Button("Delete Media")
                media_delete_output = gr.Textbox(label="Delete Result")

        prompt_search_button.click(
            search_prompts_for_deletion,
            inputs=[prompt_search_input],
            outputs=prompt_search_results
        )

        prompt_delete_button.click(
            delete_prompt,
            inputs=[prompt_id_input],
            outputs=prompt_delete_output
        )

        media_search_button.click(
            search_media_for_deletion,
            inputs=[media_search_input],
            outputs=media_search_results
        )

        media_delete_button.click(
            delete_item,
            inputs=[media_id_input, media_force_checkbox],
            outputs=media_delete_output
        )

def create_empty_trash_tab():
    with gr.TabItem("Empty Trash"):
        days_input = gr.Slider(minimum=15, maximum=90, step=5, label="Delete items older than (days)")
        empty_button = gr.Button("Empty Trash")
        empty_output = gr.Textbox(label="Result")
        empty_button.click(empty_trash_ui, inputs=[days_input], outputs=empty_output)