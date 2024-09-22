import pytest
from App_Function_Libraries.Utils.Utils import is_valid_url, sanitize_filename, normalize_title

@pytest.mark.parametrize("url,expected", [
    ("https://www.example.com", True),
    ("http://subdomain.example.com", True),
    ("ftp://ftp.example.com", True),
    ("not_a_url", False),
    ("http://.com", False),
])
def test_is_valid_url(url, expected):
    assert is_valid_url(url) == expected

@pytest.mark.parametrize("filename,expected", [
    ("normal_file.txt", "normal_file.txt"),
    ("file with spaces.txt", "file_with_spaces.txt"),
    ("file/with/slashes.txt", "file_with_slashes.txt"),
    ("file:with:colons.txt", "file_with_colons.txt"),
])
def test_sanitize_filename(filename, expected):
    assert sanitize_filename(filename) == expected

@pytest.mark.parametrize("title,expected", [
    ("Normal Title", "Normal Title"),
    ("Title with Ümlauts", "Title with Umlauts"),
    ("Title with / Slashes", "Title with _ Slashes"),
])
def test_normalize_title(title, expected):
    assert normalize_title(title) == expected
