import sys
import yt_dlp
from urllib.parse import urlparse, parse_qs

def get_playlist_videos(playlist_url):
    ydl_opts = {
        'extract_flat': True,
        'skip_download': True,
        'quiet': False
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(playlist_url, download=False)

        if 'entries' in info:
            video_urls = [entry['url'] for entry in info['entries']]
            playlist_title = info['title']
            return video_urls, playlist_title
        else:
            print("No videos found in the playlist.")
            return [], None
    except Exception as e:
        print(f"An error occurred: {e}")
        return [], None

def save_to_file(video_urls, filename):
    with open(filename, 'w') as file:
        file.write('\n'.join(video_urls))
    print(f"Video URLs saved to {filename}")

def parse_playlist_url(url):
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)

    if 'list' in query_params:
        playlist_id = query_params['list'][0]
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
        playlist_url = f"{base_url}?list={playlist_id}"
        return playlist_url
    else:
        return url

def process_input(input):
    if input.lower().endswith('.txt'):
        with open(input, 'r') as file:
            urls = file.readlines()
            urls = [url.strip() for url in urls]
        return urls
    else:
        return [input]

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please provide the playlist URL or a path to a text file containing playlist URLs.")
        print("Example usage:\n\t python Get_Playlist_URLs.py 'https://www.youtube.com/playlist?list=PLH15HpR5qRsWalnnt-9eYELxbEcYBPB6I'")
        print("Or:\n\t python Get_Playlist_URLs.py 'playlists.txt'")
        sys.exit(1)

    input_source = sys.argv[1]
    inputs = process_input(input_source)

    for playlist_input in inputs:
        parsed_playlist_url = parse_playlist_url(playlist_input)
        video_urls, playlist_title = get_playlist_videos(parsed_playlist_url)

        if video_urls:
            filename = f"{playlist_title}.txt"
            save_to_file(video_urls, filename)