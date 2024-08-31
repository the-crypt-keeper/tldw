# Export_Functionality.py
# Functionality for exporting items as markdown files
import os
import json
import math
import logging
import shutil
import tempfile
from typing import List, Dict, Optional, Tuple
import gradio as gr
from App_Function_Libraries.DB_Manager import DatabaseError, create_automated_backup, db_path, backup_dir
from App_Function_Libraries.Gradio_UI.Gradio_Shared import fetch_item_details, fetch_items_by_keyword, browse_items

logger = logging.getLogger(__name__)

def export_item_as_markdown(media_id: int) -> Tuple[Optional[str], str]:
    try:
        content, prompt, summary = fetch_item_details(media_id)
        title = f"Item {media_id}"  # You might want to fetch the actual title
        markdown_content = f"# {title}\n\n## Prompt\n{prompt}\n\n## Summary\n{summary}\n\n## Content\n{content}"

        filename = f"export_item_{media_id}.md"
        with open(filename, "w", encoding='utf-8') as f:
            f.write(markdown_content)

        logger.info(f"Successfully exported item {media_id} to {filename}")
        return filename, f"Successfully exported item {media_id} to {filename}"
    except Exception as e:
        error_message = f"Error exporting item {media_id}: {str(e)}"
        logger.error(error_message)
        return None, error_message


def export_items_by_keyword(keyword: str) -> str:
    try:
        items = fetch_items_by_keyword(keyword)
        if not items:
            logger.warning(f"No items found for keyword: {keyword}")
            return None

        # Create a temporary directory to store individual markdown files
        with tempfile.TemporaryDirectory() as temp_dir:
            folder_name = f"export_keyword_{keyword}"
            export_folder = os.path.join(temp_dir, folder_name)
            os.makedirs(export_folder)

            for item in items:
                content, prompt, summary = fetch_item_details(item['id'])
                markdown_content = f"# {item['title']}\n\n## Prompt\n{prompt}\n\n## Summary\n{summary}\n\n## Content\n{content}"

                # Create individual markdown file for each item
                file_name = f"{item['id']}_{item['title'][:50]}.md"  # Limit filename length
                file_path = os.path.join(export_folder, file_name)
                with open(file_path, "w", encoding='utf-8') as f:
                    f.write(markdown_content)

            # Create a zip file containing all markdown files
            zip_filename = f"{folder_name}.zip"
            shutil.make_archive(os.path.join(temp_dir, folder_name), 'zip', export_folder)

            # Move the zip file to a location accessible by Gradio
            final_zip_path = os.path.join(os.getcwd(), zip_filename)
            shutil.move(os.path.join(temp_dir, zip_filename), final_zip_path)

        logger.info(f"Successfully exported {len(items)} items for keyword '{keyword}' to {zip_filename}")
        return final_zip_path
    except Exception as e:
        logger.error(f"Error exporting items for keyword '{keyword}': {str(e)}")
        return None


def export_selected_items(selected_items: List[Dict]) -> Tuple[Optional[str], str]:
    try:
        logger.debug(f"Received selected_items: {selected_items}")
        if not selected_items:
            logger.warning("No items selected for export")
            return None, "No items selected for export"

        markdown_content = "# Selected Items\n\n"
        for item in selected_items:
            logger.debug(f"Processing item: {item}")
            try:
                # Check if 'value' is a string (JSON) or already a dictionary
                if isinstance(item, str):
                    item_data = json.loads(item)
                elif isinstance(item, dict) and 'value' in item:
                    item_data = item['value'] if isinstance(item['value'], dict) else json.loads(item['value'])
                else:
                    item_data = item

                logger.debug(f"Item data after processing: {item_data}")

                if 'id' not in item_data:
                    logger.error(f"'id' not found in item data: {item_data}")
                    continue

                content, prompt, summary = fetch_item_details(item_data['id'])
                markdown_content += f"## {item_data.get('title', 'Item {}'.format(item_data['id']))}\n\n### Prompt\n{prompt}\n\n### Summary\n{summary}\n\n### Content\n{content}\n\n---\n\n"
            except Exception as e:
                logger.error(f"Error processing item {item}: {str(e)}")
                markdown_content += f"## Error\n\nUnable to process this item.\n\n---\n\n"

        filename = "export_selected_items.md"
        with open(filename, "w", encoding='utf-8') as f:
            f.write(markdown_content)

        logger.info(f"Successfully exported {len(selected_items)} selected items to {filename}")
        return filename, f"Successfully exported {len(selected_items)} items to {filename}"
    except Exception as e:
        error_message = f"Error exporting selected items: {str(e)}"
        logger.error(error_message)
        return None, error_message


def display_search_results_export_tab(search_query: str, search_type: str, page: int = 1, items_per_page: int = 10):
    logger.info(f"Searching with query: '{search_query}', type: '{search_type}', page: {page}")
    try:
        results = browse_items(search_query, search_type)
        logger.info(f"browse_items returned {len(results)} results")

        if not results:
            return [], f"No results found for query: '{search_query}'", 1, 1

        total_pages = math.ceil(len(results) / items_per_page)
        start_index = (page - 1) * items_per_page
        end_index = start_index + items_per_page
        paginated_results = results[start_index:end_index]

        checkbox_data = [
            {
                "name": f"Name: {item[1]}\nURL: {item[2]}",
                "value": {"id": item[0], "title": item[1], "url": item[2]}
            }
            for item in paginated_results
        ]

        logger.info(f"Returning {len(checkbox_data)} items for checkbox (page {page} of {total_pages})")
        return checkbox_data, f"Found {len(results)} results (showing page {page} of {total_pages})", page, total_pages

    except DatabaseError as e:
        error_message = f"Error in display_search_results_export_tab: {str(e)}"
        logger.error(error_message)
        return [], error_message, 1, 1
    except Exception as e:
        error_message = f"Unexpected error in display_search_results_export_tab: {str(e)}"
        logger.error(error_message)
        return [], error_message, 1, 1


def create_export_tab():
    with gr.Tab("Search and Export"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("# Search and Export Items")
                gr.Markdown("Search for items and export them as markdown files")
                gr.Markdown("You can also export items by keyword")
                search_query = gr.Textbox(label="Search Query")
                search_type = gr.Radio(["Title", "URL", "Keyword", "Content"], label="Search By")
                search_button = gr.Button("Search")

            with gr.Column():
                prev_button = gr.Button("Previous Page")
                next_button = gr.Button("Next Page")

        current_page = gr.State(1)
        total_pages = gr.State(1)

        search_results = gr.CheckboxGroup(label="Search Results", choices=[])
        export_selected_button = gr.Button("Export Selected Items")

        keyword_input = gr.Textbox(label="Enter keyword for export")
        export_by_keyword_button = gr.Button("Export items by keyword")

        export_output = gr.File(label="Download Exported File")
        error_output = gr.Textbox(label="Status/Error Messages", interactive=False)

    def search_and_update(query, search_type, page):
        results, message, current, total = display_search_results_export_tab(query, search_type, page)
        logger.debug(f"search_and_update results: {results}")
        return results, message, current, total, gr.update(choices=results)

    search_button.click(
        fn=search_and_update,
        inputs=[search_query, search_type, current_page],
        outputs=[search_results, error_output, current_page, total_pages, search_results],
        show_progress="full"
    )


    def update_page(current, total, direction):
        new_page = max(1, min(total, current + direction))
        return new_page

    prev_button.click(
        fn=update_page,
        inputs=[current_page, total_pages, gr.State(-1)],
        outputs=[current_page]
    ).then(
        fn=search_and_update,
        inputs=[search_query, search_type, current_page],
        outputs=[search_results, error_output, current_page, total_pages],
        show_progress=True
    )

    next_button.click(
        fn=update_page,
        inputs=[current_page, total_pages, gr.State(1)],
        outputs=[current_page]
    ).then(
        fn=search_and_update,
        inputs=[search_query, search_type, current_page],
        outputs=[search_results, error_output, current_page, total_pages],
        show_progress=True
    )

    def handle_export_selected(selected_items):
        logger.debug(f"Exporting selected items: {selected_items}")
        return export_selected_items(selected_items)

    export_selected_button.click(
        fn=handle_export_selected,
        inputs=[search_results],
        outputs=[export_output, error_output],
        show_progress="full"
    )

    export_by_keyword_button.click(
        fn=export_items_by_keyword,
        inputs=[keyword_input],
        outputs=[export_output, error_output],
        show_progress="full"
    )

    def handle_item_selection(selected_items):
        logger.debug(f"Selected items: {selected_items}")
        if not selected_items:
            return None, "No item selected"

        try:
            # Assuming selected_items is a list of dictionaries
            selected_item = selected_items[0]
            logger.debug(f"First selected item: {selected_item}")

            # Check if 'value' is a string (JSON) or already a dictionary
            if isinstance(selected_item['value'], str):
                item_data = json.loads(selected_item['value'])
            else:
                item_data = selected_item['value']

            logger.debug(f"Item data: {item_data}")

            item_id = item_data['id']
            return export_item_as_markdown(item_id)
        except Exception as e:
            error_message = f"Error processing selected item: {str(e)}"
            logger.error(error_message)
            return None, error_message

    search_results.select(
        fn=handle_item_selection,
        inputs=[search_results],
        outputs=[export_output, error_output],
        show_progress="full"
    )



def create_backup():
    backup_file = create_automated_backup(db_path, backup_dir)
    return f"Backup created: {backup_file}"

def list_backups():
    backups = [f for f in os.listdir(backup_dir) if f.endswith('.db')]
    return "\n".join(backups)

def restore_backup(backup_name):
    backup_path = os.path.join(backup_dir, backup_name)
    if os.path.exists(backup_path):
        shutil.copy2(backup_path, db_path)
        return f"Database restored from {backup_name}"
    else:
        return "Backup file not found"


def create_backup_tab():
    with gr.Tab("Create Backup"):
        gr.Markdown("# Create a backup of the database")
        with gr.Row():
            with gr.Column():
                create_button = gr.Button("Create Backup")
                create_output = gr.Textbox(label="Result")
            with gr.Column():
                create_button.click(create_backup, inputs=[], outputs=create_output)

def create_view_backups_tab():
    with gr.TabItem("View Backups"):
        gr.Markdown("# Browse available backups")
        with gr.Row():
            with gr.Column():
                view_button = gr.Button("View Backups")
            with gr.Column():
                backup_list = gr.Textbox(label="Available Backups")
                view_button.click(list_backups, inputs=[], outputs=backup_list)


def create_restore_backup_tab():
    with gr.TabItem("Restore Backup"):
        gr.Markdown("# Restore a backup of the database")
        with gr.Column():
            backup_input = gr.Textbox(label="Backup Filename")
            restore_button = gr.Button("Restore")
        with gr.Column():
            restore_output = gr.Textbox(label="Result")
            restore_button.click(restore_backup, inputs=[backup_input], outputs=restore_output)
