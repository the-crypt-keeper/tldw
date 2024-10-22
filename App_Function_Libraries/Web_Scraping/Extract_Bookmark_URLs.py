# Extract_Bookmark_URLs.py
# Description: Extract URLs from Chrome/Edge and Firefox bookmarks files, as well as CSV files.
#
# Imports
import json
import sqlite3
from pathlib import Path
from typing import Set, Union
from urllib.parse import urlparse
import logging
#
# External Libraries
from bs4 import BeautifulSoup
#
####################################################################################################
#
# Logging
logger = logging.getLogger(__name__)
#
####################################################################################################
#
# Functions

def is_valid_url(url: str) -> bool:
    """Validate URL using urlparse"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def clean_url(url: str) -> str:
    """Clean and normalize URL"""
    url = url.strip()
    if not url.startswith(('http://', 'https://', 'ftp://')):
        url = 'https://' + url
    return url


def extract_from_chrome_like(file_path: Path) -> Set[str]:
    """Extract URLs from Chrome/Edge JSON bookmarks file"""
    urls = set()

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    def process_node(node):
        if isinstance(node, dict):
            if node.get('type') == 'url':
                url = node.get('url', '').strip()
                if url and is_valid_url(url):
                    urls.add(clean_url(url))
            for child in node.get('children', []):
                process_node(child)

    if 'roots' in data:
        for root in data['roots'].values():
            process_node(root)

    return urls


def extract_from_firefox_sqlite(file_path: Path) -> Set[str]:
    """Extract URLs from Firefox places.sqlite database"""
    urls = set()

    conn = sqlite3.connect(f'file:{file_path}?mode=ro', uri=True)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT url FROM moz_places 
            JOIN moz_bookmarks ON moz_bookmarks.fk = moz_places.id 
            WHERE url IS NOT NULL
        """)

        for (url,) in cursor.fetchall():
            if url and is_valid_url(url):
                urls.add(clean_url(url))
    finally:
        conn.close()

    return urls


def extract_from_html_bookmarks(file_path: Path) -> Set[str]:
    """Extract URLs from HTML bookmarks export file"""
    urls = set()

    encodings = ['utf-8', 'utf-8-sig', 'latin1']
    content = None

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
                break
        except UnicodeDecodeError:
            continue

    if not content:
        raise ValueError("Could not decode file with supported encodings")

    soup = BeautifulSoup(content, 'html.parser')
    for link in soup.find_all('a'):
        url = link.get('href', '').strip()
        if url and not url.startswith('#') and is_valid_url(url):
            urls.add(clean_url(url))

    return urls


def extract_urls(file_path: Union[str, Path]) -> Set[str]:
    """
    Extract URLs from browser bookmarks file.
    Supports Chrome/Edge JSON, Firefox SQLite, and HTML exports.

    Args:
        file_path: Path to bookmarks file

    Returns:
        Set of unique URLs

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()

    try:
        if suffix == '.json':
            return extract_from_chrome_like(file_path)
        elif suffix == '.sqlite':
            return extract_from_firefox_sqlite(file_path)
        elif suffix in ['.htm', '.html']:
            return extract_from_html_bookmarks(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    except Exception as e:
        raise ValueError(f"Failed to process {file_path}: {str(e)}")


def save_urls(urls: Set[str], output_file: Union[str, Path]) -> None:
    """Save URLs to a text file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for url in sorted(urls):
            f.write(f"{url}\n")


if __name__ == "__main__":
    # Simple command line interface
    logging.basicConfig(level=logging.INFO)

    file_path = input("Enter the path to your bookmarks file: ")
    try:
        urls = extract_urls(file_path)
        output_file = Path(file_path).with_suffix('.urls.txt')
        save_urls(urls, output_file)
        print(f"Found {len(urls)} URLs")
        print(f"Saved to {output_file}")
    except Exception as e:
        print(f"Error: {e}")

#
# End of Extract_Bookmark_URLs.py
####################################################################################################
