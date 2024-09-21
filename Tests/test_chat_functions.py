# tests/test_chat_functions.py
import pytest
from App_Function_Libraries.DB.DB_Manager import create_chat_conversation, add_chat_message, get_chat_messages, update_chat_message, delete_chat_message

@pytest.fixture
def sample_conversation(empty_db):
    return create_chat_conversation(None, "Test Conversation")

def test_create_chat_conversation(empty_db):
    conversation_id = create_chat_conversation(None, "Test Conversation")
    assert conversation_id is not None

def test_add_chat_message(empty_db, sample_conversation):
    message_id = add_chat_message(sample_conversation, "user", "Hello, world!")
    assert message_id is not None

def test_get_chat_messages(empty_db, sample_conversation):
    add_chat_message(sample_conversation, "user", "Hello, world!")
    add_chat_message(sample_conversation, "ai", "Hi there!")
    messages = get_chat_messages(sample_conversation)
    assert len(messages) == 2
    assert messages[0]['message'] == "Hello, world!"
    assert messages[1]['message'] == "Hi there!"

def test_update_chat_message(empty_db, sample_conversation):
    message_id = add_chat_message(sample_conversation, "user", "Hello, world!")
    update_chat_message(message_id, "Updated message")
    messages = get_chat_messages(sample_conversation)
    assert messages[0]['message'] == "Updated message"

def test_delete_chat_message(empty_db, sample_conversation):
    message_id = add_chat_message(sample_conversation, "user", "Hello, world!")
    delete_chat_message(message_id)
    messages = get_chat_messages(sample_conversation)
    assert len(messages) == 0