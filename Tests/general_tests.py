import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock, PropertyMock, call, ANY
import faster_whisper

sys.path.append("../")

# Import the necessary functions and classes from your script
import summarize
from summarize import (
    read_paths_from_file,
    process_path,
    get_youtube,
    download_video,
    convert_to_wav,
    speech_to_text,
    summarize_with_openai,
    summarize_with_claude,
    summarize_with_cohere,
    summarize_with_groq,
    summarize_with_llama,
    summarize_with_kobold,
    summarize_with_oobabooga,
    main
)


class TestTranscriptionScript(unittest.TestCase):
    def setUp(self):
        # Set up any necessary resources before each test
        pass

    def tearDown(self):
        # Clean up any resources after each test
        pass

    def test_read_paths_from_file(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("http://example.com/video1.mp4\n")
            temp_file.write("http://example.com/video2.mp4\n")
            temp_file.write("http://example.com/video3.mp4\n")
            temp_file_path = temp_file.name

        paths = read_paths_from_file(temp_file_path)
        expected_paths = [
            "http://example.com/video1.mp4",
            "http://example.com/video2.mp4",
            "http://example.com/video3.mp4"
        ]
        self.assertListEqual(paths, expected_paths)

        os.unlink(temp_file_path)

    def test_process_path(self):
        with patch('summarize.get_youtube', return_value={'title': 'Sample Video'}):
            result = process_path("http://example.com/video.mp4")
            self.assertIsNotNone(result)

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file_path = temp_file.name
        result = process_path(temp_file_path)
        self.assertIsNotNone(result)
        os.unlink(temp_file_path)

        result = process_path("non_existent_path")
        self.assertIsNone(result)

    def test_get_youtube(self):
        with patch('yt_dlp.YoutubeDL.extract_info', return_value={'title': 'Sample YouTube Video'}):
            info_dict = get_youtube("http://example.com/youtube_video.mp4")
            self.assertIsNotNone(info_dict)
            self.assertEqual(info_dict['title'], 'Sample YouTube Video')

    def test_download_video(self):
        with patch('yt_dlp.YoutubeDL.download') as mock_download:
            video_path = download_video("http://example.com/video.mp4", "download_path", {'title': 'Sample Video'}, False)
            self.assertIsNotNone(video_path)
            mock_download.assert_called_once()

    def test_convert_to_wav(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file_path = temp_file.name
        with patch('subprocess.run', return_value=MagicMock(returncode=0)):
            wav_path = convert_to_wav(temp_file_path)
            self.assertIsNotNone(wav_path)
            self.assertTrue(wav_path.endswith(".wav"))
        os.unlink(temp_file_path)

    def test_speech_to_text(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file_path = temp_file.name
        with patch('faster_whisper.WhisperModel.transcribe', return_value=([], {})):
            segments = speech_to_text(temp_file_path)
            self.assertIsInstance(segments, list)
        os.unlink(temp_file_path)

    def test_summarize_with_openai(self):
        with patch('requests.post') as mock_post:
            mock_post.return_value = MagicMock(status_code=200, json=lambda: {'choices': [{'message': {'content': 'Sample summary'}}]})
            summary = summarize_with_openai("api_key", "file_path", "model")
            self.assertEqual(summary, 'Sample summary')
            mock_post.assert_called_once_with(
                'https://api.openai.com/v1/chat/completions',
                headers=ANY,
                json=ANY
            )

    # Add similar tests for other summarization functions
    # ...

    def test_main(self):
        with patch('summarize.process_path', return_value=("download_path", {'title': 'Sample Video'}, "audio_file")):
            with patch('summarize.speech_to_text', return_value=[]):
                with patch('summarize.summarize_with_openai', return_value='Sample summary'):
                    results = main("https://www.youtube.com/watch?v=YRfN-UGoKJY", api_name="openai", api_key="api_key")
                    self.assertIsInstance(results, list)
                    self.assertEqual(len(results), 1)
                    self.assertIn('summary', results[0])


    # Add more integration tests for different scenarios
    # ...

    def test_error_handling(self):
        with self.assertRaises(ValueError):
            speech_to_text(None)

        with self.assertRaises(RuntimeError):
            with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'ffmpeg')):
                convert_to_wav("invalid_path")


    def test_warnings(self):
        with self.assertWarns(UserWarning):
            # Code that triggers a warning
            pass

    # Add more tests for different aspects of the script
    # ...

if __name__ == '__main__':
    unittest.main()