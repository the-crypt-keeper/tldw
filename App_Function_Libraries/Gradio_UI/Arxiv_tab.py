# Arxiv_tab.py
# Description: This file contains the Gradio UI for searching, browsing, and ingesting arXiv papers.
#
# Imports
from datetime import datetime

import gradio as gr
import requests
from bs4 import BeautifulSoup

from App_Function_Libraries.DB.DB_Manager import add_media_with_keywords
#
# Local Imports
from App_Function_Libraries.Third_Party.Arxiv import search_arxiv, process_and_ingest_arxiv_paper
#####################################################################################################
#
# Functions:

# Number of results per page
PAGE_SIZE = 10

def create_arxiv_tab():
    with gr.TabItem("Arxiv Search & Ingest"):
        gr.Markdown("# arXiv Search, Browse, Download, and Ingest")

        # Search Inputs
        with gr.Row():
            search_query = gr.Textbox(label="Search Query", placeholder="e.g., machine learning")
            author_filter = gr.Textbox(label="Author", placeholder="e.g., John Doe")
            year_filter = gr.Number(label="Year", precision=0)
            search_button = gr.Button("Search")

        # Pagination Controls
        with gr.Row():
            prev_button = gr.Button("Previous")
            next_button = gr.Button("Next")
            page_info = gr.Textbox(label="Page", value="1", interactive=False)

        # Results Dataframe (removed max_rows)
        results_df = gr.Dataframe(
            headers=["Select", "Title", "Authors", "Published"],
            datatype=["bool", "str", "str", "str"],
            col_count=(4, "fixed"),
            interactive=True
        )

        # Paper Details View
        paper_view = gr.Markdown(label="Paper Details")

        # Ingestion Section
        with gr.Row():
            ingest_button = gr.Button("Ingest Selected Paper")
            arxiv_keywords = gr.Textbox(label="Additional Keywords (comma-separated)", placeholder="e.g., AI, Deep Learning")

        ingest_result = gr.Textbox(label="Ingestion Result", interactive=False)

        # Define States for Pagination and Selection
        state = gr.State(value={"start": 0, "current_page": 1, "last_query": None})
        selected_paper_id = gr.State(value=None)

        def build_query_url(query, author, year, start):
            base_url = "http://export.arxiv.org/api/query?"
            search_params = []

            # Build search query
            if query:
                search_params.append(f"all:{query}")
            if author:
                search_params.append(f'au:"{author}"')
            if year:
                search_params.append(f'submittedDate:[{year}01010000 TO {year}12312359]')

            search_query = "+AND+".join(search_params) if search_params else "all:*"

            url = f"{base_url}search_query={search_query}&start={start}&max_results={PAGE_SIZE}"
            return url

        def parse_arxiv_feed(xml_content):
            soup = BeautifulSoup(xml_content, 'xml')
            entries = []
            for entry in soup.find_all('entry'):
                title = entry.title.text.strip()
                paper_id = entry.id.text.strip().split('/abs/')[-1]
                authors = ', '.join(author.find('name').text.strip() for author in entry.find_all('author'))
                published = entry.published.text.strip().split('T')[0]
                abstract = entry.summary.text.strip()
                entries.append({
                    'id': paper_id,
                    'title': title,
                    'authors': authors,
                    'published': published,
                    'abstract': abstract
                })
            return entries

        def search_arxiv(query, author, year):
            start = 0
            url = build_query_url(query, author, year, start)
            try:
                response = requests.get(url)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                return gr.update(value=[["", f"**Error:** {str(e)}"]]), "1", state.value

            entries = parse_arxiv_feed(response.text)
            state.value = {"start": start, "current_page": 1, "last_query": (query, author, year)}
            if not entries:
                return gr.update(value=[["", "No results found."]]), "1", state.value

            # Prepare data for Dataframe
            data = []
            for entry in entries:
                data.append([
                    False,  # For selection checkbox
                    entry['title'],
                    entry['authors'],
                    entry['published']
                ])

            return gr.update(value=data), "1", state.value

        def handle_pagination(direction):
            current_state = state.value
            query, author, year = current_state["last_query"]
            new_page = current_state["current_page"] + direction
            if new_page < 1:
                new_page = 1
            start = (new_page - 1) * PAGE_SIZE
            url = build_query_url(query, author, year, start)
            try:
                response = requests.get(url)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                return gr.update(), gr.update()

            entries = parse_arxiv_feed(response.text)
            if entries:
                current_state["start"] = start
                current_state["current_page"] = new_page
                state.value = current_state

                # Prepare data for Dataframe
                data = []
                for entry in entries:
                    data.append([
                        False,  # For selection checkbox
                        entry['title'],
                        entry['authors'],
                        entry['published']
                    ])

                return gr.update(value=data), str(new_page)
            else:
                # If no entries, do not change the page
                return gr.update(), gr.update()

        def select_paper(data):
            selected_rows = [row for row in data if row[0]]  # Check if 'Select' is True
            if not selected_rows:
                return "Please select a paper to view.", None
            selected_paper = selected_rows[0]
            # Find the paper ID based on the title (assuming titles are unique in the page)
            for entry in state.value.get('entries', []):
                if entry['title'] == selected_paper[1]:
                    paper_id = entry['id']
                    break
            else:
                return "Paper not found.", None
            xml_content = fetch_arxiv_xml(paper_id)
            markdown, _, _, _ = convert_xml_to_markdown(xml_content)
            selected_paper_id.value = paper_id
            return markdown, selected_paper_id.value

        def process_and_ingest_arxiv_paper(paper_id, additional_keywords):
            try:
                if not paper_id:
                    return "Please select a paper to ingest."
                xml_content = fetch_arxiv_xml(paper_id)
                markdown, title, authors, categories = convert_xml_to_markdown(xml_content)

                keywords = f"arxiv,{','.join(categories)}"
                if additional_keywords:
                    keywords += f",{additional_keywords}"

                add_media_with_keywords(
                    url=f"https://arxiv.org/abs/{paper_id}",
                    title=title,
                    media_type='document',
                    content=markdown,
                    keywords=keywords,
                    prompt='No prompt for arXiv papers',
                    summary='arXiv paper ingested from XML',
                    transcription_model='None',
                    author=', '.join(authors),
                    ingestion_date=datetime.now().strftime('%Y-%m-%d')
                )

                return f"arXiv paper '{title}' ingested successfully."
            except Exception as e:
                return f"Error processing arXiv paper: {str(e)}"

        def fetch_arxiv_xml(paper_id):
            base_url = "http://export.arxiv.org/api/query?id_list="
            response = requests.get(base_url + paper_id)
            response.raise_for_status()
            return response.text

        def convert_xml_to_markdown(xml_content):
            soup = BeautifulSoup(xml_content, 'xml')

            title = soup.find('title').text.strip()
            authors = [author.text.strip() for author in soup.find_all('name')]
            abstract = soup.find('summary').text.strip()
            published = soup.find('published').text.strip()
            updated = soup.find('updated').text.strip()

            categories = [category['term'] for category in soup.find_all('category')]

            markdown = f"# {title}\n\n"
            markdown += f"**Authors:** {', '.join(authors)}\n\n"
            markdown += f"**Published Date:** {published}\n\n"
            markdown += f"**Abstract:**\n\n{abstract}\n\n"
            markdown += f"**Categories:** {', '.join(categories)}\n\n"

            return markdown, title, authors, categories

        def add_media_with_keywords(url, title, media_type, content, keywords, prompt, summary, transcription_model, author, ingestion_date):
            """
            Placeholder function for ingesting media with keywords.
            Implement this function based on your ingestion system.
            """
            # Example implementation (to be replaced with actual ingestion logic)
            print(f"Ingesting '{title}' with keywords: {keywords}")
            # Add your ingestion code here

        # Event Handlers

        # Connect Search Button
        search_button.click(
            fn=search_arxiv,
            inputs=[search_query, author_filter, year_filter],
            outputs=[results_df, page_info, state],
            queue=True
        )

        # Connect Next Button
        next_button.click(
            fn=lambda: handle_pagination(1),
            inputs=None,
            outputs=[results_df, page_info],
            queue=True
        )

        # Connect Previous Button
        prev_button.click(
            fn=lambda: handle_pagination(-1),
            inputs=None,
            outputs=[results_df, page_info],
            queue=True
        )

        # When the user selects a paper in the Dataframe
        results_df.select(
            fn=select_paper,
            inputs=results_df,
            outputs=[paper_view, selected_paper_id],
            queue=True
        )

        # Connect Ingest Button
        ingest_button.click(
            fn=process_and_ingest_arxiv_paper,
            inputs=[selected_paper_id, arxiv_keywords],
            outputs=ingest_result,
            queue=True
        )

#
# End of File
#####################################################################################################
