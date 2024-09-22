# tests/test_logging.py
import pytest
import logging
from App_Function_Libraries.DB.DB_Manager import add_media_to_database
#
####################################################################################################
# Test Status:
# FIXME

def test_logging(caplog):
    caplog.set_level(logging.INFO)

    info_dict = {'title': 'Logging Test Video', 'uploader': 'Test Uploader'}
    add_media_to_database('https://example.com/logging_test', info_dict, [], 'Test summary', ['logging'], 'Test prompt',
                          'whisper-1')

    assert 'Media \'Logging Test Video\' added/updated successfully' in caplog.text