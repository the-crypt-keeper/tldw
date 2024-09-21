# tests/test_document_versioning.py
import pytest
from App_Function_Libraries.DB.DB_Manager import create_document_version, get_document_version


@pytest.fixture
def sample_document(empty_db, sample_media):
    return create_document_version(sample_media, "Initial content")


def test_create_document_version(empty_db, sample_media):
    version = create_document_version(sample_media, "Test content")
    assert version == 1


def test_get_document_version(empty_db, sample_document):
    version_data = get_document_version(sample_document)
    assert version_data['version_number'] == 1
    assert version_data['content'] == "Initial content"


def test_create_multiple_versions(empty_db, sample_media):
    v1 = create_document_version(sample_media, "Version 1 content")
    v2 = create_document_version(sample_media, "Version 2 content")
    assert v1 == 1
    assert v2 == 2

    v2_data = get_document_version(sample_media, 2)
    assert v2_data['content'] == "Version 2 content"