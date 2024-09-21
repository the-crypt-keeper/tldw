# tests/test_performance.py
import pytest
import time
from App_Function_Libraries.DB.DB_Manager import search_media_database, add_media_to_database


@pytest.mark.parametrize("num_records", [10, 100, 1000])
def test_search_performance(empty_db, num_records):
    # Populate the database with test records
    for i in range(num_records):
        info_dict = {'title': f'Performance Test Video {i}', 'uploader': 'Test Uploader'}
        add_media_to_database(f'https://example.com/perf_test_{i}', info_dict, [], 'Test summary', ['performance'],
                              'Test prompt', 'whisper-1')

    start_time = time.time()
    results = search_media_database('Performance Test')
    end_time = time.time()

    assert len(results) == num_records
    assert end_time - start_time < 1.0  # Assuming search should complete within 1 second