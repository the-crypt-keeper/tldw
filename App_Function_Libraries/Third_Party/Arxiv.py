# Arxiv.py
# Description: This file contains the functions for searching and ingesting arXiv papers.
import arxiv
import requests
from bs4 import BeautifulSoup
from datetime import datetime
#
# Local Imports
from App_Function_Libraries.DB.DB_Manager import add_media_with_keywords
#
#####################################################################################################
#
# Functions:

def search_arxiv(query):
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=10,
        sort_by=arxiv.SortCriterion.Relevance
    )

    results = []
    for result in client.results(search):
        results.append([
            result.title,
            result.entry_id.split('/')[-1],  # Extract the ID from the entry_id
            ', '.join(author.name for author in result.authors),
            result.summary
        ])

    return results


def fetch_arxiv_xml(paper_id):
    base_url = "http://export.arxiv.org/api/query?id_list="
    response = requests.get(base_url + paper_id)
    return response.text


def convert_xml_to_markdown(xml_content):
    soup = BeautifulSoup(xml_content, 'xml')

    title = soup.find('title').text.strip()
    authors = [author.text.strip() for author in soup.find_all('name')]
    abstract = soup.find('summary').text.strip()
    categories = [category['term'] for category in soup.find_all('category')]

    markdown = f"# {title}\n\n"
    markdown += f"**Authors:** {', '.join(authors)}\n\n"
    markdown += f"**Categories:** {', '.join(categories)}\n\n"
    markdown += f"## Abstract\n\n{abstract}\n\n"

    return markdown, title, authors, categories


def process_and_ingest_arxiv_paper(paper_id, additional_keywords):
    try:
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

#
# End of Arxiv.py
####################################################################################################
