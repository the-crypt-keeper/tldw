# Search_Tab.py
# Description: This file contains the code for the search tab in the Gradio UI
#
# Imports
import html
import logging
import sqlite3
#
# External Imports
import gradio as gr
#
# Local Imports
from App_Function_Libraries.DB.DB_Manager import view_database, search_and_display_items, get_all_document_versions, \
    fetch_item_details_single
from App_Function_Libraries.DB.SQLite_DB import search_prompts, get_document_version
from App_Function_Libraries.Gradio_UI.Gradio_Shared import update_dropdown, update_detailed_view
from App_Function_Libraries.Utils.Utils import get_database_path
#
###################################################################################################
#
# Functions:

logger = logging.getLogger()


def update_detailed_view_with_versions(selected_item, item_mapping):
    if selected_item and item_mapping and selected_item in item_mapping:
        media_id = item_mapping[selected_item]
        prompt, summary, content = fetch_item_details_single(media_id)

        # Fetch all versions for the media item
        versions = get_all_document_versions(media_id)
        version_choices = [f"Version {v['version_number']} ({v['created_at']})" for v in versions]

        prompt_summary_html = f"<strong>Prompt:</strong><br>{prompt}<br><br><strong>Summary:</strong><br>{summary}"

        return prompt_summary_html, content, gr.update(choices=version_choices, visible=True)
    return "", "", gr.update(choices=[], visible=False)


def update_content_for_version(selected_item, item_mapping, selected_version):
    if selected_item and item_mapping and selected_item in item_mapping:
        media_id = item_mapping[selected_item]
        version_number = int(selected_version.split()[1].split('(')[0])

        version_data = get_document_version(media_id, version_number)
        if 'error' not in version_data:
            content = version_data['content']
            prompt_summary_html = f"<strong>Version:</strong> {version_number}<br><strong>Created at:</strong> {version_data['created_at']}"
            return prompt_summary_html, content
    return "", ""


def create_search_tab():
    with gr.TabItem("Search / Detailed View"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("# Search across all ingested items in the Database")
                gr.Markdown(" by Title / URL / Keyword / or Content via SQLite Full-Text-Search")
                search_query_input = gr.Textbox(label="Search Query", placeholder="Enter your search query here...")
                search_type_input = gr.Radio(choices=["Title", "URL", "Keyword", "Content"], value="Title", label="Search By")
                search_button = gr.Button("Search")
                items_output = gr.Dropdown(label="Select Item", choices=[])
                item_mapping = gr.State({})
                version_dropdown = gr.Dropdown(label="Select Version", choices=[], visible=False)
                prompt_summary_output = gr.HTML(label="Prompt & Summary", visible=True)

                search_button.click(
                    fn=update_dropdown,
                    inputs=[search_query_input, search_type_input],
                    outputs=[items_output, item_mapping]
                )
            with gr.Column():
                content_output = gr.Markdown(label="Content", visible=True)
                items_output.change(
                    fn=update_detailed_view_with_versions,
                    inputs=[items_output, item_mapping],
                    outputs=[prompt_summary_output, content_output, version_dropdown]
                )
                version_dropdown.change(
                    fn=update_content_for_version,
                    inputs=[items_output, item_mapping, version_dropdown],
                    outputs=[prompt_summary_output, content_output]
                )


def display_search_results(query):
    if not query.strip():
        return "Please enter a search query."

    results = search_prompts(query)

    # Debugging: Print the results to the console to see what is being returned
    print(f"Processed search results for query '{query}': {results}")

    if results:
        result_md = "## Search Results:\n"
        for result in results:
            # Debugging: Print each result to see its format
            print(f"Result item: {result}")

            if len(result) == 2:
                name, details = result
                result_md += f"**Title:** {name}\n\n**Description:** {details}\n\n---\n"

            elif len(result) == 4:
                name, details, system, user = result
                result_md += f"**Title:** {name}\n\n"
                result_md += f"**Description:** {details}\n\n"
                result_md += f"**System Prompt:** {system}\n\n"
                result_md += f"**User Prompt:** {user}\n\n"
                result_md += "---\n"
            else:
                result_md += "Error: Unexpected result format.\n\n---\n"
        return result_md
    return "No results found."


def create_viewing_tab():
    with gr.TabItem("View Database"):
        gr.Markdown("# View Database Entries")
        with gr.Row():
            with gr.Column():
                entries_per_page = gr.Dropdown(choices=[10, 20, 50, 100], label="Entries per Page", value=10)
                page_number = gr.Number(value=1, label="Page Number", precision=0)
                view_button = gr.Button("View Page")
                next_page_button = gr.Button("Next Page")
                previous_page_button = gr.Button("Previous Page")
            with gr.Column():
                results_display = gr.HTML()
                pagination_info = gr.Textbox(label="Pagination Info", interactive=False)

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


def create_search_summaries_tab():
    with gr.TabItem("Search/View Title+Summary "):
        gr.Markdown("# Search across all ingested items in the Database and review their summaries")
        gr.Markdown("Search by Title / URL / Keyword / or Content via SQLite Full-Text-Search")
        with gr.Row():
            with gr.Column():
                search_query_input = gr.Textbox(label="Search Query", placeholder="Enter your search query here...")
                search_type_input = gr.Radio(choices=["Title", "URL", "Keyword", "Content"], value="Title",
                                             label="Search By")
                entries_per_page = gr.Dropdown(choices=[10, 20, 50, 100], label="Entries per Page", value=10)
                page_number = gr.Number(value=1, label="Page Number", precision=0)
                char_count_input = gr.Number(value=5000, label="Amount of characters to display from the main content",
                                             precision=0)
            with gr.Column():
                search_button = gr.Button("Search")
                next_page_button = gr.Button("Next Page")
                previous_page_button = gr.Button("Previous Page")
                pagination_info = gr.Textbox(label="Pagination Info", interactive=False)
        search_results_output = gr.HTML()


        def update_search_page(query, search_type, page, entries_per_page, char_count):
            # Ensure char_count is a positive integer
            char_count = max(1, int(char_count)) if char_count else 5000
            results, pagination, total_pages = search_and_display_items(query, search_type, page, entries_per_page, char_count)
            next_disabled = page >= total_pages
            prev_disabled = page <= 1
            return results, pagination, page, gr.update(interactive=not next_disabled), gr.update(
                interactive=not prev_disabled)

        def go_to_next_search_page(query, search_type, current_page, entries_per_page, char_count):
            next_page = current_page + 1
            return update_search_page(query, search_type, next_page, entries_per_page, char_count)

        def go_to_previous_search_page(query, search_type, current_page, entries_per_page, char_count):
            previous_page = max(1, current_page - 1)
            return update_search_page(query, search_type, previous_page, entries_per_page, char_count)

        search_button.click(
            fn=update_search_page,
            inputs=[search_query_input, search_type_input, page_number, entries_per_page, char_count_input],
            outputs=[search_results_output, pagination_info, page_number, next_page_button, previous_page_button]
        )

        next_page_button.click(
            fn=go_to_next_search_page,
            inputs=[search_query_input, search_type_input, page_number, entries_per_page, char_count_input],
            outputs=[search_results_output, pagination_info, page_number, next_page_button, previous_page_button]
        )

        previous_page_button.click(
            fn=go_to_previous_search_page,
            inputs=[search_query_input, search_type_input, page_number, entries_per_page, char_count_input],
            outputs=[search_results_output, pagination_info, page_number, next_page_button, previous_page_button]
        )



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
            with gr.Column():
                pagination_info = gr.Textbox(label="Pagination Info", interactive=False)
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



def create_prompt_search_tab():
    with gr.TabItem("Search Prompts"):
        gr.Markdown("# Search and View Prompt Details")
        gr.Markdown("Currently has all of the https://github.com/danielmiessler/fabric prompts already available")
        with gr.Row():
            with gr.Column():
                search_query_input = gr.Textbox(label="Search Prompts", placeholder="Enter your search query...")
                entries_per_page = gr.Dropdown(choices=[10, 20, 50, 100], label="Entries per Page", value=10)
                page_number = gr.Number(value=1, label="Page Number", precision=0)
            with gr.Column():
                search_button = gr.Button("Search Prompts")
                next_page_button = gr.Button("Next Page")
                previous_page_button = gr.Button("Previous Page")
                pagination_info = gr.Textbox(label="Pagination Info", interactive=False)
        search_results_output = gr.HTML()

        # This is dirty and shouldn't be in the UI code, but it's a quick way to get the search working.
        # FIXME - SQL functions to be moved to DB_Manager
        def search_and_display_prompts(query, page, entries_per_page):
            offset = (page - 1) * entries_per_page
            try:
                # FIXME - SQL functions to be moved to DB_Manager
                with sqlite3.connect(get_database_path('prompts.db')) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT p.name, p.details, p.system, p.user, GROUP_CONCAT(k.keyword, ', ') as keywords
                        FROM Prompts p
                        LEFT JOIN PromptKeywords pk ON p.id = pk.prompt_id
                        LEFT JOIN Keywords k ON pk.keyword_id = k.id
                        WHERE p.name LIKE ? OR p.details LIKE ? OR p.system LIKE ? OR p.user LIKE ? OR k.keyword LIKE ?
                        GROUP BY p.id
                        ORDER BY p.name
                        LIMIT ? OFFSET ?
                    ''', (f'%{query}%', f'%{query}%', f'%{query}%', f'%{query}%', f'%{query}%', entries_per_page, offset))
                    prompts = cursor.fetchall()

                    cursor.execute('''
                        SELECT COUNT(DISTINCT p.id)
                        FROM Prompts p
                        LEFT JOIN PromptKeywords pk ON p.id = pk.prompt_id
                        LEFT JOIN Keywords k ON pk.keyword_id = k.id
                        WHERE p.name LIKE ? OR p.details LIKE ? OR p.system LIKE ? OR p.user LIKE ? OR k.keyword LIKE ?
                    ''', (f'%{query}%', f'%{query}%', f'%{query}%', f'%{query}%', f'%{query}%'))
                    total_prompts = cursor.fetchone()[0]

                results = ""
                for prompt in prompts:
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
                return f"<p>Error searching prompts: {e}</p>", "Error", 0

        def update_search_page(query, page, entries_per_page):
            results, pagination, total_pages = search_and_display_prompts(query, page, entries_per_page)
            next_disabled = page >= total_pages
            prev_disabled = page <= 1
            return results, pagination, page, gr.update(interactive=not next_disabled), gr.update(interactive=not prev_disabled)

        def go_to_next_search_page(query, current_page, entries_per_page):
            next_page = current_page + 1
            return update_search_page(query, next_page, entries_per_page)

        def go_to_previous_search_page(query, current_page, entries_per_page):
            previous_page = max(1, current_page - 1)
            return update_search_page(query, previous_page, entries_per_page)

        search_button.click(
            fn=update_search_page,
            inputs=[search_query_input, page_number, entries_per_page],
            outputs=[search_results_output, pagination_info, page_number, next_page_button, previous_page_button]
        )

        next_page_button.click(
            fn=go_to_next_search_page,
            inputs=[search_query_input, page_number, entries_per_page],
            outputs=[search_results_output, pagination_info, page_number, next_page_button, previous_page_button]
        )

        previous_page_button.click(
            fn=go_to_previous_search_page,
            inputs=[search_query_input, page_number, entries_per_page],
            outputs=[search_results_output, pagination_info, page_number, next_page_button, previous_page_button]
        )
