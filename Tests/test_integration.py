# tests/test_integration.py
import pytest
from App_Function_Libraries.DB.DB_Manager import add_media_to_database, get_media_content, search_media_database
from App_Function_Libraries.Video_DL_Ingestion_Lib import extract_video_info, download_video
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_video_info():
    return {
        'title': 'Test Video',
        'uploader': 'Test Channel',
        'upload_date': '20230101',
        'duration': 300,
    }, 'Test Video'

@pytest.fixture
def mock_video_download():
    return '/path/to/downloaded/video.mp4'

@patch('App_Function_Libraries.Video_DL_Ingestion_Lib.extract_video_info')
@patch('App_Function_Libraries.Video_DL_Ingestion_Lib.download_video')
def test_media_ingestion_and_retrieval(mock_download, mock_extract, empty_db, mock_video_info, mock_video_download):
    mock_extract.return_value = mock_video_info
    mock_download.return_value = mock_video_download

    # Simulate video download and info extraction
    url = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
    info_dict, title = extract_video_info(url)
    video_path = download_video(url, './downloads', info_dict, True, "small")

    # Add media to database
    media_id = add_media_to_database(url, info_dict, [], "Test summary", ["test", "video"], "Test prompt", "whisper-1")

    # Verify media was added correctly
    content = get_media_content(media_id)
    assert content is not None
    assert 'Test Video' in content

    # Test search functionality
    search_results = search_media_database('Test Video')
    assert len(search_results) == 1
    assert search_results[0][1] == 'Test Video'