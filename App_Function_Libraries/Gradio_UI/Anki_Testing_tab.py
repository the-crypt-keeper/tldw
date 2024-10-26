import gradio as gr
import json
import base64
from datetime import datetime, timedelta
from pathlib import Path
import os
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import re
from contextlib import contextmanager
import logging
import sqlite3
import uuid

# Import existing functions from your Anki library
from App_Function_Libraries.Third_Party.Anki import (
    validate_flashcards,
    sanitize_html,
    process_apkg_file,
    handle_file_upload as original_handle_file_upload,
    enhanced_file_upload as original_enhanced_file_upload,
    update_card_choices,
    load_card_for_editing,
    handle_validation,
)

# Configure logging
logging.basicConfig(level=logging.INFO)


class ReviewResult(Enum):
    AGAIN = 1
    HARD = 2
    GOOD = 3
    EASY = 4


@dataclass
class CardProgress:
    card_id: str
    last_reviewed: datetime
    next_review: datetime
    ease_factor: float
    interval: int
    review_count: int


class DatabaseManager:
    def __init__(self, db_path: str = "anki_progress.db"):
        self.db_path = db_path
        self.init_db()

    @contextmanager
    def get_connection(self) -> sqlite3.Connection:
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Enable row factory for named columns
        try:
            yield conn
        finally:
            conn.close()

    def init_db(self) -> None:
        """Initialize the database with enhanced schema."""
        with self.get_connection() as conn:
            try:
                # Decks table with metadata
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS decks (
                        deck_id TEXT PRIMARY KEY,
                        deck_name TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_reviewed TIMESTAMP,
                        card_count INTEGER,
                        description TEXT,
                        settings TEXT  -- JSON field for deck-specific settings
                    )
                """
                )

                # Cards table with enhanced media support
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cards (
                        card_id TEXT PRIMARY KEY,
                        deck_id TEXT,
                        front_content TEXT NOT NULL,
                        back_content TEXT NOT NULL,
                        card_type TEXT NOT NULL,
                        tags TEXT,  -- JSON array of tags
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (deck_id) REFERENCES decks(deck_id) ON DELETE CASCADE
                    )
                """
                )

                # Media assets table
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS media_assets (
                        asset_id TEXT PRIMARY KEY,
                        card_id TEXT,
                        content BLOB NOT NULL,
                        media_type TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (card_id) REFERENCES cards(card_id) ON DELETE CASCADE
                    )
                """
                )

                # Card progress tracking
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS card_progress (
                        card_id TEXT PRIMARY KEY,
                        deck_id TEXT,
                        last_reviewed TIMESTAMP,
                        next_review TIMESTAMP,
                        ease_factor REAL DEFAULT 2.5,
                        interval INTEGER DEFAULT 0,
                        review_count INTEGER DEFAULT 0,
                        review_history TEXT,  -- JSON array of review results
                        FOREIGN KEY (card_id) REFERENCES cards(card_id) ON DELETE CASCADE,
                        FOREIGN KEY (deck_id) REFERENCES decks(deck_id) ON DELETE CASCADE
                    )
                """
                )

                # Create indices for better performance
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_cards_deck ON cards(deck_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_media_card ON media_assets(card_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_progress_deck ON card_progress(deck_id)"
                )

                # Create full-text search virtual table
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS cards_fts USING fts5(
                        front_content, back_content, tags, notes,
                        content='cards',
                        content_rowid='card_id'
                    )
                """
                )

                # Create triggers to update FTS table
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS cards_ai AFTER INSERT ON cards BEGIN
                        INSERT INTO cards_fts(rowid, front_content, back_content, tags, notes)
                        VALUES (new.card_id, new.front_content, new.back_content, new.tags, new.notes);
                    END;
                """
                )

                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS cards_au AFTER UPDATE ON cards BEGIN
                        INSERT INTO cards_fts(cards_fts, rowid, front_content, back_content, tags, notes)
                        VALUES('delete', old.card_id, old.front_content, old.back_content, old.tags, old.notes);
                        INSERT INTO cards_fts(rowid, front_content, back_content, tags, notes)
                        VALUES (new.card_id, new.front_content, new.back_content, new.tags, new.notes);
                    END;
                """
                )

                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS cards_ad AFTER DELETE ON cards BEGIN
                        INSERT INTO cards_fts(cards_fts, rowid, front_content, back_content, tags, notes)
                        VALUES('delete', old.card_id, old.front_content, old.back_content, old.tags, old.notes);
                    END;
                """
                )

                conn.commit()
            except Exception as e:
                logging.exception("Error initializing database")
                raise


class DeckManager:
    def __init__(self, db_path: str = "anki_progress.db"):
        self.db = DatabaseManager(db_path)

    def save_deck_from_json(self, deck_name: str, content: str, description: str = "") -> str:
        try:
            # Validate flashcards
            is_valid, validation_message = validate_flashcards(content)
            if not is_valid:
                raise ValueError(validation_message)

            deck_data = json.loads(content)
            deck_id = str(uuid.uuid5(uuid.NAMESPACE_OID, deck_name))

            with self.db.get_connection() as conn:
                # Save deck metadata
                conn.execute(
                    """
                    INSERT OR REPLACE INTO decks 
                    (deck_id, deck_name, card_count, description)
                    VALUES (?, ?, ?, ?)
                """,
                    (deck_id, deck_name, len(deck_data["cards"]), description),
                )

                # Process each card
                for card in deck_data["cards"]:
                    # Sanitize content
                    front_content = sanitize_html(card.get("front", ""))
                    back_content = sanitize_html(card.get("back", ""))

                    # Ensure all fields are strings
                    card_id = str(card.get('id', ''))
                    card_type = str(card.get('type', ''))
                    tags_list = card.get('tags', [])
                    if not isinstance(tags_list, list):
                        tags_list = []
                    else:
                        tags_list = [str(tag) for tag in tags_list]
                    tags = json.dumps(tags_list)
                    notes = str(card.get('note', ''))

                    # Debug statements
                    logging.debug(f"Processing card with ID: {card_id}")
                    logging.debug(
                        f"Types: card_id={type(card_id)}, deck_id={type(deck_id)}, front_content={type(front_content)}, back_content={type(back_content)}, card_type={type(card_type)}, tags={type(tags)}, notes={type(notes)}")
                    logging.debug(
                        f"Values: card_id={card_id}, deck_id={deck_id}, front_content={front_content}, back_content={back_content}, card_type={card_type}, tags={tags}, notes={notes}")

                    # Save card
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO cards 
                        (card_id, deck_id, front_content, back_content, card_type, tags, notes)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            card_id,
                            deck_id,
                            front_content,
                            back_content,
                            card_type,
                            tags,
                            notes,
                        ),
                    )

                    # Initialize progress tracking
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO card_progress 
                        (card_id, deck_id, review_history)
                        VALUES (?, ?, '[]')
                    """,
                        (card_id, deck_id),
                    )

                conn.commit()
                logging.info(
                    f"Deck '{deck_name}' saved successfully with ID '{deck_id}'"
                )
            return deck_id
        except Exception as e:
            logging.exception("Error saving deck")
            raise Exception("An error occurred while saving the deck.")

    def get_deck_choices(self) -> List[str]:
        """
        Get a list of deck choices for dropdown menus.

        Returns:
            List[str]: List of deck names with IDs.
        """
        try:
            with self.db.get_connection() as conn:
                decks = conn.execute(
                    """
                    SELECT deck_id, deck_name FROM decks
                """
                ).fetchall()
                return [
                    f"{deck['deck_name']} ({deck['deck_id']})" for deck in decks
                ]
        except Exception as e:
            logging.exception("Error retrieving deck choices")
            return []


class AnkiReviewInterface:
    def __init__(self):
        self.deck_manager = DeckManager()
        self.current_card = None
        self.review_queue = []
        self.current_deck_id = None

    def create_interface(self):
        with gr.Blocks() as interface:
            # Previous state variables
            deck_select = gr.Dropdown(
                choices=self.deck_manager.get_deck_choices(), label="Select Deck"
            )
            start_review_button = gr.Button("Start Review")

            with gr.Tab("Review"):
                with gr.Column():
                    card_front = gr.HTML(label="Question")
                    show_answer_button = gr.Button("Show Answer")
                    card_back = gr.HTML(label="Answer", visible=False)
                    with gr.Row(visible=False) as review_buttons_row:
                        again_button = gr.Button("Again")
                        hard_button = gr.Button("Hard")
                        good_button = gr.Button("Good")
                        easy_button = gr.Button("Easy")

            # Import and create deck from JSON or APKG
            with gr.Tab("Import Deck"):
                gr.Markdown("## Import or Create Flashcards")

                input_type = gr.Radio(
                    choices=["JSON", "APKG"],
                    label="Input Type",
                    value="JSON"
                )

                with gr.Group() as json_input_group:
                    flashcard_input = gr.TextArea(
                        label="Enter Flashcards (JSON format)",
                        placeholder='''{
    "cards": [
        {
            "id": "CARD_001",
            "type": "basic",
            "front": "What is the capital of France?",
            "back": "Paris",
            "tags": ["geography", "europe"],
            "note": "Remember: City of Light"
        }
    ]
}''',
                        lines=10
                    )

                    import_json = gr.File(
                        label="Or Import JSON File",
                        file_types=[".json"]
                    )

                with gr.Group(visible=False) as apkg_input_group:
                    import_apkg = gr.File(
                        label="Import APKG File",
                        file_types=None  # Accept all file types
                    )
                    deck_info = gr.JSON(
                        label="Deck Information",
                        visible=False
                    )

                deck_name_input = gr.Textbox(
                    label="Deck Name",
                    placeholder="Enter a name for the deck"
                )

                description_input = gr.Textbox(
                    label="Deck Description",
                    placeholder="Optional description"
                )

                validate_button = gr.Button("Validate Flashcards")
                validation_status = gr.Markdown("")

                save_deck_button = gr.Button("Save Deck")

            # Event handlers
            input_type.change(
                fn=lambda t: (
                    gr.update(visible=t == "JSON"),
                    gr.update(visible=t == "APKG"),
                    gr.update(visible=t == "APKG")
                ),
                inputs=[input_type],
                outputs=[json_input_group, apkg_input_group, deck_info]
            )

            # File upload handlers
            import_json.upload(
                fn=self.handle_file_upload,
                inputs=[import_json, input_type],
                outputs=[
                    flashcard_input,
                    deck_info,
                    validation_status
                ]
            )

            import_apkg.upload(
                fn=self.handle_file_upload,
                inputs=[import_apkg, input_type],
                outputs=[
                    flashcard_input,
                    deck_info,
                    validation_status
                ]
            )

            # Validation handler
            validate_button.click(
                fn=lambda content, input_format: (
                    handle_validation(content, input_format)
                ),
                inputs=[flashcard_input, input_type],
                outputs=[validation_status]
            )

            # Save deck handler
            save_deck_button.click(
                fn=self.save_deck,
                inputs=[deck_name_input, flashcard_input, description_input],
                outputs=[deck_select, validation_status]
            )

            # Start review handlers
            start_review_button.click(
                self.start_review,
                inputs=[deck_select],
                outputs=[card_front, card_back, review_buttons_row, show_answer_button],
            )

            show_answer_button.click(
                self.show_answer,
                outputs=[card_back, review_buttons_row, show_answer_button],
            )

            again_button.click(
                self.record_review,
                inputs=[gr.State(ReviewResult.AGAIN)],
                outputs=[card_front, card_back, review_buttons_row, show_answer_button],
            )

            hard_button.click(
                self.record_review,
                inputs=[gr.State(ReviewResult.HARD)],
                outputs=[card_front, card_back, review_buttons_row, show_answer_button],
            )

            good_button.click(
                self.record_review,
                inputs=[gr.State(ReviewResult.GOOD)],
                outputs=[card_front, card_back, review_buttons_row, show_answer_button],
            )

            easy_button.click(
                self.record_review,
                inputs=[gr.State(ReviewResult.EASY)],
                outputs=[card_front, card_back, review_buttons_row, show_answer_button],
            )

        return interface

    def handle_file_upload(self, file: Any, input_type: str) -> Tuple[Optional[str], Optional[Dict], str]:
        """Handle file uploads and return appropriate outputs."""
        if not file:
            return None, None, "No file uploaded."

        # Extract the file name and extension
        if hasattr(file, 'name'):
            filename = file.name
        else:
            filename = str(file)

        if input_type == "APKG":
            if not filename.lower().endswith('.apkg'):
                return None, None, "Invalid file type. Please upload a .apkg file."

            content, deck_info, message, _ = original_enhanced_file_upload(file, input_type)
            if deck_info:
                deck_info = deck_info  # Process deck_info if needed
            else:
                deck_info = None
            return content, deck_info, message
        else:
            content, _, message, _ = original_handle_file_upload(file, input_type)
            return content, None, message

    def save_deck(self, deck_name: str, content: str, description: str):
        """
        Save the deck and update deck choices.

        Args:
            deck_name (str): The name of the deck.
            content (str): The JSON content of the deck.
            description (str): The deck description.

        Returns:
            Tuple[ComponentUpdate, str]: Updated deck choices and status message.
        """
        try:
            deck_id = self.deck_manager.save_deck_from_json(deck_name, content, description)
            deck_choices = self.deck_manager.get_deck_choices()
            # Update the deck_select Dropdown choices
            return gr.update(choices=deck_choices), f"Deck '{deck_name}' saved successfully!"
        except Exception as e:
            logging.exception("Error saving deck")
            return gr.update(), "An error occurred while saving the deck."

    def start_review(self, deck_choice: str):
        """
        Start the review session for the selected deck.

        Args:
            deck_choice (str): The selected deck from the dropdown.

        Returns:
            Tuple[HTML, HTML, Component, Component]: Updated components.
        """
        try:
            if not deck_choice:
                return (
                    gr.update(value="Please select a deck."),
                    "",
                    gr.update(visible=False),
                    gr.update(visible=False),
                )

            # Extract deck_id from deck_choice
            deck_id = deck_choice.split("(")[-1].strip(")")

            with self.deck_manager.db.get_connection() as conn:
                # Get cards due for review
                today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cards = conn.execute(
                    """
                    SELECT c.front_content, c.back_content, c.card_id
                    FROM cards c
                    JOIN card_progress p ON c.card_id = p.card_id
                    WHERE p.deck_id = ?
                    AND (p.next_review IS NULL OR p.next_review <= ?)
                    ORDER BY p.next_review ASC
                """,
                    (deck_id, today),
                ).fetchall()

            if not cards:
                return (
                    gr.update(value="No cards due for review."),
                    "",
                    gr.update(visible=False),
                    gr.update(visible=False),
                )

            self.review_queue = list(cards)
            self.current_deck_id = deck_id

            # Load the first card
            self.current_card = self.review_queue.pop(0)
            front_content = self.current_card["front_content"]

            return front_content, "", gr.update(visible=False), gr.update(visible=True)
        except Exception as e:
            logging.exception("Error starting review")
            return (
                "An error occurred while starting the review.",
                "",
                gr.update(visible=False),
                gr.update(visible=False),
            )

    def show_answer(self):
        """
        Show the answer for the current card.

        Returns:
            Tuple[HTML, Component, Component]: Updated components.
        """
        try:
            if not self.current_card:
                return "", gr.update(visible=False), gr.update(visible=False)

            back_content = self.current_card["back_content"]
            return back_content, gr.update(visible=True), gr.update(visible=False)
        except Exception as e:
            logging.exception("Error showing answer")
            return (
                "An error occurred while showing the answer.",
                gr.update(visible=False),
                gr.update(visible=False),
            )

    def record_review(self, review_result: ReviewResult):
        """
        Record the review result and load the next card.

        Args:
            review_result (ReviewResult): The result of the review.

        Returns:
            Tuple[HTML, HTML, Component, Component]: Updated components.
        """
        try:
            if not self.current_card:
                return "", "", gr.update(visible=False), gr.update(visible=False)

            card_id = self.current_card["card_id"]
            self.update_card_progress(card_id, review_result)

            if self.review_queue:
                self.current_card = self.review_queue.pop(0)
                front_content = self.current_card["front_content"]
                return front_content, "", gr.update(visible=False), gr.update(visible=True)
            else:
                # No more cards
                self.current_card = None
                return (
                    "Review session completed.",
                    "",
                    gr.update(visible=False),
                    gr.update(visible=False),
                )
        except Exception as e:
            logging.exception("Error recording review")
            return (
                "An error occurred while recording the review.",
                "",
                gr.update(visible=False),
                gr.update(visible=False),
            )

    def update_card_progress(self, card_id: str, review_result: ReviewResult):
        """
        Update the card's progress based on the review result.

        Args:
            card_id (str): The ID of the card.
            review_result (ReviewResult): The result of the review.
        """
        try:
            with self.deck_manager.db.get_connection() as conn:
                progress = conn.execute(
                    """
                    SELECT * FROM card_progress WHERE card_id = ?
                """,
                    (card_id,),
                ).fetchone()

                if not progress:
                    logging.error(f"Progress not found for card_id {card_id}")
                    return

                # Update progress based on SM-2 algorithm
                ease_factor = progress["ease_factor"]
                interval = progress["interval"]
                review_count = progress["review_count"]

                if review_result == ReviewResult.AGAIN:
                    interval = 1
                    ease_factor = max(1.3, ease_factor - 0.2)
                elif review_result == ReviewResult.HARD:
                    interval = max(1, int(interval * 1.2))
                    ease_factor = max(1.3, ease_factor - 0.15)
                elif review_result == ReviewResult.GOOD:
                    interval = max(1, int(interval * ease_factor))
                elif review_result == ReviewResult.EASY:
                    interval = max(1, int(interval * ease_factor * 1.3))
                    ease_factor = min(2.5, ease_factor + 0.15)

                next_review_date = datetime.now() + timedelta(days=interval)
                review_history = (
                    json.loads(progress["review_history"])
                    if progress["review_history"]
                    else []
                )
                review_history.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "result": review_result.name,
                    }
                )

                conn.execute(
                    """
                    UPDATE card_progress
                    SET last_reviewed = ?, next_review = ?, ease_factor = ?, interval = ?, review_count = ?, review_history = ?
                    WHERE card_id = ?
                """,
                    (
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        next_review_date.strftime("%Y-%m-%d %H:%M:%S"),
                        ease_factor,
                        interval,
                        review_count + 1,
                        json.dumps(review_history),
                        card_id,
                    ),
                )
                conn.commit()
        except Exception as e:
            logging.exception("Error updating card progress")


def main():
    interface = AnkiReviewInterface()
    review_interface = interface.create_interface()
    review_interface.launch()


if __name__ == "__main__":
    main()
