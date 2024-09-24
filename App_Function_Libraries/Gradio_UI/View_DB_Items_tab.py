# View_DB_Items_tab.py
# Description: This file contains the code for the search tab in the Gradio UI
#
# Imports
import html
import sqlite3
#
# External Imports
import gradio as gr
#
# Local Imports
from App_Function_Libraries.DB.DB_Manager import view_database, get_all_document_versions, \
    fetch_item_details_single, fetch_paginated_data
from App_Function_Libraries.DB.SQLite_DB import get_document_version
from App_Function_Libraries.Utils.Utils import get_database_path, format_text_with_line_breaks
#
#
####################################################################################################
#
# Functions

def create_prompt_view_tab():
    with gr.TabItem("View Prompt Database"):
        gr.Markdown("# View Prompt Database Entries")
        with gr.Row():
            with gr.Column():
                entries_per_page = gr.Dropdown(choices=[10, 20, 50, 100], label="Entries per Page", value=10)
                page_number = gr.Number(value=1, label="Page Number", precision=0)
                view_button = gr.Button("View Page")
                next_page_button = gr.Button("Next Page")
                previous_page_button = gr.Button("Previous Page")
                pagination_info = gr.Textbox(label="Pagination Info", interactive=False)
            with gr.Column():
                results_display = gr.HTML()

        # FIXME - SQL functions to be moved to DB_Manager
        
        def view_database(page, entries_per_page):
            offset = (page - 1) * entries_per_page
            try:
                with sqlite3.connect(get_database_path('prompts.db')) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT p.name, p.details, p.system, p.user, GROUP_CONCAT(k.keyword, ', ') as keywords
                        FROM Prompts p
                        LEFT JOIN PromptKeywords pk ON p.id = pk.prompt_id
                        LEFT JOIN Keywords k ON pk.keyword_id = k.id
                        GROUP BY p.id
                        ORDER BY p.name
                        LIMIT ? OFFSET ?
                    ''', (entries_per_page, offset))
                    prompts = cursor.fetchall()

                    cursor.execute('SELECT COUNT(*) FROM Prompts')
                    total_prompts = cursor.fetchone()[0]

                results = ""
                for prompt in prompts:
                    # Escape HTML special characters and replace newlines with <br> tags
                    title = html.escape(prompt[0]).replace('\n', '<br>')
                    details = html.escape(prompt[1] or '').replace('\n', '<br>')
                    system_prompt = html.escape(prompt[2] or '')
                    user_prompt = html.escape(prompt[3] or '')
                    keywords = html.escape(prompt[4] or '').replace('\n', '<br>')

                    results += f"""
                    <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 20px;">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                            <div><strong>Title:</strong> {title}</div>
                            <div><strong>Details:</strong> {details}</div>
                        </div>
                        <div style="margin-top: 10px;">
                            <strong>User Prompt:</strong>
                            <pre style="white-space: pre-wrap; word-wrap: break-word;">{user_prompt}</pre>
                        </div>
                        <div style="margin-top: 10px;">
                            <strong>System Prompt:</strong>
                            <pre style="white-space: pre-wrap; word-wrap: break-word;">{system_prompt}</pre>
                        </div>
                        <div style="margin-top: 10px;">
                            <strong>Keywords:</strong> {keywords}
                        </div>
                    </div>
                    """

                total_pages = (total_prompts + entries_per_page - 1) // entries_per_page
                pagination = f"Page {page} of {total_pages} (Total prompts: {total_prompts})"

                return results, pagination, total_pages
            except sqlite3.Error as e:
                return f"<p>Error fetching prompts: {e}</p>", "Error", 0

        def update_page(page, entries_per_page):
            results, pagination, total_pages = view_database(page, entries_per_page)
            next_disabled = page >= total_pages
            prev_disabled = page <= 1
            return results, pagination, page, gr.update(interactive=not next_disabled), gr.update(
                interactive=not prev_disabled)

        def go_to_next_page(current_page, entries_per_page):
            next_page = current_page + 1
            return update_page(next_page, entries_per_page)

        def go_to_previous_page(current_page, entries_per_page):
            previous_page = max(1, current_page - 1)
            return update_page(previous_page, entries_per_page)

        view_button.click(
            fn=update_page,
            inputs=[page_number, entries_per_page],
            outputs=[results_display, pagination_info, page_number, next_page_button, previous_page_button]
        )

        next_page_button.click(
            fn=go_to_next_page,
            inputs=[page_number, entries_per_page],
            outputs=[results_display, pagination_info, page_number, next_page_button, previous_page_button]
        )

        previous_page_button.click(
            fn=go_to_previous_page,
            inputs=[page_number, entries_per_page],
            outputs=[results_display, pagination_info, page_number, next_page_button, previous_page_button]
        )


def create_view_all_with_versions_tab():
    with gr.TabItem("View All Items"):
        gr.Markdown("# View All Database Entries with Version Selection")
        with gr.Row():
            with gr.Column(scale=1):
                entries_per_page = gr.Dropdown(choices=[10, 20, 50, 100], label="Entries per Page", value=10)
                page_number = gr.Number(value=1, label="Page Number", precision=0)
                view_button = gr.Button("View Page")
                next_page_button = gr.Button("Next Page")
                previous_page_button = gr.Button("Previous Page")
            with gr.Column(scale=2):
                items_output = gr.Dropdown(label="Select Item to View Details", choices=[])
                version_dropdown = gr.Dropdown(label="Select Version", choices=[], visible=False)
        with gr.Row():
            with gr.Column(scale=1):
                pagination_info = gr.Textbox(label="Pagination Info", interactive=False)
            with gr.Column(scale=2):
                details_display = gr.HTML(label="Item Details")

        item_mapping = gr.State({})

        def update_page(page, entries_per_page):
            results, total_entries = fetch_paginated_data(page, entries_per_page)
            total_pages = (total_entries + entries_per_page - 1) // entries_per_page
            pagination = f"Page {page} of {total_pages} (Total items: {total_entries})"

            choices = [f"{item[1]} (ID: {item[0]})" for item in results]
            new_item_mapping = {f"{item[1]} (ID: {item[0]})": item[0] for item in results}

            next_disabled = page >= total_pages
            prev_disabled = page <= 1

            return (gr.update(choices=choices, value=None),
                    pagination,
                    page,
                    gr.update(interactive=not next_disabled),
                    gr.update(interactive=not prev_disabled),
                    gr.update(visible=False, choices=[]),
                    "",
                    new_item_mapping)

        def display_item_details(selected_item, item_mapping):
            if selected_item and item_mapping:
                media_id = item_mapping[selected_item]
                prompt, summary, content = fetch_item_details_single(media_id)
                versions = get_all_document_versions(media_id)
                version_choices = [f"Version {v['version_number']} ({v['created_at']})" for v in versions]

                formatted_prompt = format_text_with_line_breaks(prompt)
                formatted_summary = format_text_with_line_breaks(summary)
                formatted_content = format_text_with_line_breaks(content[:500])

                details_html = f"""
                <h3>{selected_item}</h3>
                <strong>Prompt:</strong><br>{formatted_prompt}<br><br>
                <strong>Summary:</strong><br>{formatted_summary}<br><br>
                <strong>Content (first 500 characters):</strong><br>{formatted_content}...
                """

                return (
                gr.update(visible=True, choices=version_choices, value=version_choices[0] if version_choices else None),
                details_html)
            return gr.update(visible=False, choices=[]), ""

        def update_version_content(selected_item, item_mapping, selected_version):
            if selected_item and item_mapping and selected_version:
                media_id = item_mapping[selected_item]
                version_number = int(selected_version.split()[1].split('(')[0])
                version_data = get_document_version(media_id, version_number)

                if 'error' not in version_data:
                    formatted_content = format_text_with_line_breaks(version_data['content'])
                    details_html = f"""
                    <h3>{selected_item}</h3>
                    <strong>Version:</strong> {version_number}<br>
                    <strong>Created at:</strong> {version_data['created_at']}<br><br>
                    <strong>Content:</strong><br>{formatted_content}
                    """
                    return details_html
            return ""

        view_button.click(
            fn=update_page,
            inputs=[page_number, entries_per_page],
            outputs=[items_output, pagination_info, page_number, next_page_button, previous_page_button,
                     version_dropdown, details_display, item_mapping]
        )

        next_page_button.click(
            fn=lambda page, entries: update_page(page + 1, entries),
            inputs=[page_number, entries_per_page],
            outputs=[items_output, pagination_info, page_number, next_page_button, previous_page_button,
                     version_dropdown, details_display, item_mapping]
        )

        previous_page_button.click(
            fn=lambda page, entries: update_page(max(1, page - 1), entries),
            inputs=[page_number, entries_per_page],
            outputs=[items_output, pagination_info, page_number, next_page_button, previous_page_button,
                     version_dropdown, details_display, item_mapping]
        )

        items_output.change(
            fn=display_item_details,
            inputs=[items_output, item_mapping],
            outputs=[version_dropdown, details_display]
        )

        version_dropdown.change(
            fn=update_version_content,
            inputs=[items_output, item_mapping, version_dropdown],
            outputs=[details_display]
        )


def create_viewing_tab():
    with gr.TabItem("View Database Entries"):
        gr.Markdown("# View Database Entries")
        with gr.Row():
            with gr.Column():
                entries_per_page = gr.Dropdown(choices=[10, 20, 50, 100], label="Entries per Page", value=10)
                page_number = gr.Number(value=1, label="Page Number", precision=0)
                view_button = gr.Button("View Page")
                next_page_button = gr.Button("Next Page")
                previous_page_button = gr.Button("Previous Page")
                pagination_info = gr.Textbox(label="Pagination Info", interactive=False)
            with gr.Column():
                results_display = gr.HTML()


        def update_page(page, entries_per_page):
            results, pagination, total_pages = view_database(page, entries_per_page)
            next_disabled = page >= total_pages
            prev_disabled = page <= 1
            return results, pagination, page, gr.update(interactive=not next_disabled), gr.update(interactive=not prev_disabled)

        def go_to_next_page(current_page, entries_per_page):
            next_page = current_page + 1
            return update_page(next_page, entries_per_page)

        def go_to_previous_page(current_page, entries_per_page):
            previous_page = max(1, current_page - 1)
            return update_page(previous_page, entries_per_page)

        view_button.click(
            fn=update_page,
            inputs=[page_number, entries_per_page],
            outputs=[results_display, pagination_info, page_number, next_page_button, previous_page_button]
        )

        next_page_button.click(
            fn=go_to_next_page,
            inputs=[page_number, entries_per_page],
            outputs=[results_display, pagination_info, page_number, next_page_button, previous_page_button]
        )

        previous_page_button.click(
            fn=go_to_previous_page,
            inputs=[page_number, entries_per_page],
            outputs=[results_display, pagination_info, page_number, next_page_button, previous_page_button]
        )

#
####################################################################################################