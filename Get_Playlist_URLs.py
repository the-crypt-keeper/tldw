import sys
import yt_dlp

def get_playlist_videos(playlist_url):
    ydl_opts = {
        'extract_flat': True,
        'skip_download': True,
        'quiet': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)

        if 'entries' in info:
            video_urls = [entry['url'] for entry in info['entries']]
            playlist_title = info['title']
            return video_urls, playlist_title
        else:
            print("No videos found in the playlist.")
            return [], None

def save_to_file(video_urls, filename):
    with open(filename, 'w') as file:
        file.write('\n'.join(video_urls))
    print(f"Video URLs saved to {filename}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please provide the playlist URL as a command-line argument.")
        sys.exit(1)

    playlist_url = sys.argv[1]
    video_urls, playlist_title = get_playlist_videos(playlist_url)

    if video_urls:
        filename = f"{playlist_title}.txt"
        save_to_file(video_urls, filename)