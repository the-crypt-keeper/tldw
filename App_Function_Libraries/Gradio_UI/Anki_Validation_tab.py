# Anki_Validation_tab.py
# Description: Gradio functions for the Anki Validation tab
#
# Imports
import json
from typing import Optional, Tuple, List
#
# External Imports
import gradio as gr
#from outlines import models, prompts
#
# Local Imports
from App_Function_Libraries.Third_Party.Anki import sanitize_html, generate_card_choices, \
    export_cards, load_card_for_editing, handle_file_upload, \
    validate_for_ui, update_card_with_validation, update_card_choices, enhanced_file_upload, \
    handle_validation
#
############################################################################################################
#
# Functions:

def create_anki_validation_tab():
    with gr.TabItem("Anki Flashcard Validation", visible=True):
        gr.Markdown("# Anki Flashcard Validation and Editor")

        # State variables for internal tracking
        current_card_data = gr.State({})
        preview_update_flag = gr.State(False)

        with gr.Row():
            # Left Column: Input and Validation
            with gr.Column(scale=1):
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
                        file_types=[".apkg"]
                    )
                    deck_info = gr.JSON(
                        label="Deck Information",
                        visible=False
                    )

                validate_button = gr.Button("Validate Flashcards")

            # Right Column: Validation Results and Editor
            with gr.Column(scale=1):
                gr.Markdown("## Validation Results")
                validation_status = gr.Markdown("")

                with gr.Accordion("Validation Rules", open=False):
                    gr.Markdown("""
                    ### Required Fields:
                    - Unique ID
                    - Card Type (basic, cloze, reverse)
                    - Front content
                    - Back content
                    - At least one tag

                    ### Content Rules:
                    - No empty fields
                    - Front side should be a clear question/prompt
                    - Back side should contain complete answer
                    - Cloze deletions must have valid syntax
                    - No duplicate IDs

                    ### Image Rules:
                    - Valid image tags
                    - Supported formats (JPG, PNG, GIF)
                    - Base64 encoded or valid URL

                    ### APKG-specific Rules:
                    - Valid SQLite database structure
                    - Media files properly referenced
                    - Note types match Anki standards
                    - Card templates are well-formed
                    """)

        with gr.Row():
            # Card Editor
            gr.Markdown("## Card Editor")
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("Edit Individual Cards", open=True):
                    card_selector = gr.Dropdown(
                        label="Select Card to Edit",
                        choices=[],
                        interactive=True
                    )

                    card_type = gr.Radio(
                        choices=["basic", "cloze", "reverse"],
                        label="Card Type",
                        value="basic"
                    )

                    # Front content with preview
                    with gr.Group():
                        gr.Markdown("### Front Content")
                        front_content = gr.TextArea(
                            label="Content (HTML supported)",
                            lines=3
                        )
                        front_preview = gr.HTML(
                            label="Preview"
                        )

                    # Back content with preview
                    with gr.Group():
                        gr.Markdown("### Back Content")
                        back_content = gr.TextArea(
                            label="Content (HTML supported)",
                            lines=3
                        )
                        back_preview = gr.HTML(
                            label="Preview"
                        )

                    tags_input = gr.TextArea(
                        label="Tags (comma-separated)",
                        lines=1
                    )

                    notes_input = gr.TextArea(
                        label="Additional Notes",
                        lines=2
                    )

                    with gr.Row():
                        update_card_button = gr.Button("Update Card")
                        delete_card_button = gr.Button("Delete Card", variant="stop")

        with gr.Row():
            with gr.Column(scale=1):
                # Export Options
                gr.Markdown("## Export Options")
                export_format = gr.Radio(
                    choices=["Anki CSV", "JSON", "Plain Text"],
                    label="Export Format",
                    value="Anki CSV"
                )
                export_button = gr.Button("Export Valid Cards")
                export_file = gr.File(label="Download Validated Cards")
                export_status = gr.Markdown("")
            with gr.Column(scale=1):
                gr.Markdown("## Export Instructions")
                gr.Markdown("""
                ### Anki CSV Format:
                - Front, Back, Tags, Type, Note
                - Use for importing into Anki
                - Images preserved as HTML

                ### JSON Format:
                - JSON array of cards
                - Images as base64 or URLs
                - Use for custom processing

                ### Plain Text Format:
                - Question and Answer pairs
                - Images represented as [IMG] placeholder
                - Use for manual review
                """)

        def update_preview(content):
            """Update preview with sanitized content."""
            if not content:
                return ""
            return sanitize_html(content)

        # Event handlers
        def validation_chain(content: str) -> Tuple[str, List[str]]:
            """Combined validation and card choice update."""
            validation_message = validate_for_ui(content)
            card_choices = update_card_choices(content)
            return validation_message, card_choices

        def delete_card(card_selection, current_content):
            """Delete selected card and return updated content."""
            if not card_selection or not current_content:
                return current_content, "No card selected", []

            try:
                data = json.loads(current_content)
                selected_id = card_selection.split(" - ")[0]

                data['cards'] = [card for card in data['cards'] if card['id'] != selected_id]
                new_content = json.dumps(data, indent=2)

                return (
                    new_content,
                    "Card deleted successfully!",
                    generate_card_choices(new_content)
                )

            except Exception as e:
                return current_content, f"Error deleting card: {str(e)}", []

        def process_validation_result(is_valid, message):
            """Process validation result into a formatted markdown string."""
            if is_valid:
                return f"✅ {message}"
            else:
                return f"❌ {message}"

        # Register event handlers
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
            fn=handle_file_upload,
            inputs=[import_json, input_type],
            outputs=[
                flashcard_input,
                deck_info,
                validation_status,
                card_selector
            ]
        )

        import_apkg.upload(
            fn=enhanced_file_upload,
            inputs=[import_apkg, input_type],
            outputs=[
                flashcard_input,
                deck_info,
                validation_status,
                card_selector
            ]
        )

        # Validation handler
        validate_button.click(
            fn=lambda content, input_format: (
                handle_validation(content, input_format),
                generate_card_choices(content) if content else []
            ),
            inputs=[flashcard_input, input_type],
            outputs=[validation_status, card_selector]
        )

        # Card editing handlers
        # Card selector change event
        card_selector.change(
            fn=load_card_for_editing,
            inputs=[card_selector, flashcard_input],
            outputs=[
                card_type,
                front_content,
                back_content,
                tags_input,
                notes_input,
                front_preview,
                back_preview
            ]
        )

        # Live preview updates
        front_content.change(
            fn=update_preview,
            inputs=[front_content],
            outputs=[front_preview]
        )

        back_content.change(
            fn=update_preview,
            inputs=[back_content],
            outputs=[back_preview]
        )

        # Card update handler
        update_card_button.click(
            fn=update_card_with_validation,
            inputs=[
                flashcard_input,
                card_selector,
                card_type,
                front_content,
                back_content,
                tags_input,
                notes_input
            ],
            outputs=[
                flashcard_input,
                validation_status,
                card_selector
            ]
        )

        # Delete card handler
        delete_card_button.click(
            fn=delete_card,
            inputs=[card_selector, flashcard_input],
            outputs=[flashcard_input, validation_status, card_selector]
        )

        # Export handler
        export_button.click(
            fn=export_cards,
            inputs=[flashcard_input, export_format],
            outputs=[export_status, export_file]
        )

        return (
            flashcard_input,
            import_json,
            import_apkg,
            validate_button,
            validation_status,
            card_selector,
            card_type,
            front_content,
            back_content,
            front_preview,
            back_preview,
            tags_input,
            notes_input,
            update_card_button,
            delete_card_button,
            export_format,
            export_button,
            export_file,
            export_status,
            deck_info
        )


def create_anki_generator_tab():
    import genanki
    import json
    from typing import List, Dict, Any
    import tempfile
    import os

    with gr.TabItem("Anki Deck Generator", visible=True):
        gr.Markdown("# Create Anki Deck")

        with gr.Row():
            # Left Column: Deck Settings
            with gr.Column(scale=1):
                gr.Markdown("## Deck Settings")
                deck_name = gr.Textbox(
                    label="Deck Name",
                    placeholder="My Study Deck",
                    value="My Study Deck"
                )

                deck_description = gr.Textbox(
                    label="Deck Description",
                    placeholder="Description of your deck",
                    lines=2
                )

                note_type = gr.Radio(
                    choices=["Basic", "Basic (and reversed)", "Cloze"],
                    label="Note Type",
                    value="Basic"
                )

                # Card Fields based on note type
                with gr.Group() as basic_fields:
                    front_template = gr.Textbox(
                        label="Front Template (HTML)",
                        value="{{Front}}",
                        lines=3
                    )
                    back_template = gr.Textbox(
                        label="Back Template (HTML)",
                        value="{{FrontSide}}<hr id='answer'>{{Back}}",
                        lines=3
                    )

                with gr.Group() as cloze_fields:
                    cloze_template = gr.Textbox(
                        label="Cloze Template (HTML)",
                        value="{{cloze:Text}}",
                        lines=3,
                        visible=False
                    )

                css_styling = gr.Textbox(
                    label="Card Styling (CSS)",
                    value=".card {\n font-family: arial;\n font-size: 20px;\n text-align: center;\n color: black;\n background-color: white;\n}\n\n.cloze {\n font-weight: bold;\n color: blue;\n}",
                    lines=5
                )

            # Right Column: Card Creation
            with gr.Column(scale=1):
                gr.Markdown("## Add Cards")

                with gr.Group() as basic_input:
                    front_content = gr.TextArea(
                        label="Front Content",
                        placeholder="Question or prompt",
                        lines=3
                    )
                    back_content = gr.TextArea(
                        label="Back Content",
                        placeholder="Answer",
                        lines=3
                    )

                with gr.Group() as cloze_input:
                    cloze_content = gr.TextArea(
                        label="Cloze Content",
                        placeholder="Text with {{c1::cloze}} deletions",
                        lines=3,
                        visible=False
                    )

                tags_input = gr.TextArea(
                    label="Tags (comma-separated)",
                    placeholder="tag1, tag2, tag3",
                    lines=1
                )

                add_card_btn = gr.Button("Add Card")

                cards_list = gr.JSON(
                    label="Cards in Deck",
                    value={"cards": []}
                )

                clear_cards_btn = gr.Button("Clear All Cards", variant="stop")

        with gr.Row():
            generate_deck_btn = gr.Button("Generate Deck", variant="primary")
            download_deck = gr.File(label="Download Deck")
            generation_status = gr.Markdown("")

        def update_note_type_fields(note_type: str):
            if note_type == "Cloze":
                return {
                    basic_input: gr.update(visible=False),
                    cloze_input: gr.update(visible=True),
                    basic_fields: gr.update(visible=False),
                    cloze_fields: gr.update(visible=True)
                }
            else:
                return {
                    basic_input: gr.update(visible=True),
                    cloze_input: gr.update(visible=False),
                    basic_fields: gr.update(visible=True),
                    cloze_fields: gr.update(visible=False)
                }

        def add_card(note_type: str, front: str, back: str, cloze: str, tags: str, current_cards: Dict[str, List]):
            if not current_cards:
                current_cards = {"cards": []}

            cards_data = current_cards["cards"]

            # Process tags
            card_tags = [tag.strip() for tag in tags.split(',') if tag.strip()]

            new_card = {
                "id": f"CARD_{len(cards_data) + 1}",
                "tags": card_tags
            }

            if note_type == "Cloze":
                if not cloze or "{{c" not in cloze:
                    return current_cards, "❌ Invalid cloze format. Use {{c1::text}} syntax."
                new_card.update({
                    "type": "cloze",
                    "content": cloze
                })
            else:
                if not front or not back:
                    return current_cards, "❌ Both front and back content are required."
                new_card.update({
                    "type": "basic",
                    "front": front,
                    "back": back,
                    "is_reverse": note_type == "Basic (and reversed)"
                })

            cards_data.append(new_card)
            return {"cards": cards_data}, "✅ Card added successfully!"

        def clear_cards() -> Tuple[Dict[str, List], str]:
            return {"cards": []}, "✅ All cards cleared!"

        def generate_anki_deck(
            deck_name: str,
            deck_description: str,
            note_type: str,
            front_template: str,
            back_template: str,
            cloze_template: str,
            css: str,
            cards_data: Dict[str, List]
        ) -> Tuple[Optional[str], str]:
            try:
                if not cards_data or not cards_data.get("cards"):
                    return None, "❌ No cards to generate deck from!"

                # Create model based on note type
                if note_type == "Cloze":
                    model = genanki.Model(
                        1483883320,  # Random model ID
                        'Cloze Model',
                        fields=[
                            {'name': 'Text'},
                            {'name': 'Back Extra'}
                        ],
                        templates=[{
                            'name': 'Cloze Card',
                            'qfmt': cloze_template,
                            'afmt': cloze_template + '<br><hr id="extra">{{Back Extra}}'
                        }],
                        css=css,
                        # FIXME CLOZE DOESNT EXIST
                        model_type=1
                    )
                else:
                    templates = [{
                        'name': 'Card 1',
                        'qfmt': front_template,
                        'afmt': back_template
                    }]

                    if note_type == "Basic (and reversed)":
                        templates.append({
                            'name': 'Card 2',
                            'qfmt': '{{Back}}',
                            'afmt': '{{FrontSide}}<hr id="answer">{{Front}}'
                        })

                    model = genanki.Model(
                        1607392319,  # Random model ID
                        'Basic Model',
                        fields=[
                            {'name': 'Front'},
                            {'name': 'Back'}
                        ],
                        templates=templates,
                        css=css
                    )

                # Create deck
                deck = genanki.Deck(
                    2059400110,  # Random deck ID
                    deck_name,
                    description=deck_description
                )

                # Add cards to deck
                for card in cards_data["cards"]:
                    if card["type"] == "cloze":
                        note = genanki.Note(
                            model=model,
                            fields=[card["content"], ""],
                            tags=card["tags"]
                        )
                    else:
                        note = genanki.Note(
                            model=model,
                            fields=[card["front"], card["back"]],
                            tags=card["tags"]
                        )
                    deck.add_note(note)

                # Save deck to temporary file
                temp_dir = tempfile.mkdtemp()
                deck_path = os.path.join(temp_dir, f"{deck_name}.apkg")
                genanki.Package(deck).write_to_file(deck_path)

                return deck_path, "✅ Deck generated successfully!"

            except Exception as e:
                return None, f"❌ Error generating deck: {str(e)}"

        # Register event handlers
        note_type.change(
            fn=update_note_type_fields,
            inputs=[note_type],
            outputs=[basic_input, cloze_input, basic_fields, cloze_fields]
        )

        add_card_btn.click(
            fn=add_card,
            inputs=[
                note_type,
                front_content,
                back_content,
                cloze_content,
                tags_input,
                cards_list
            ],
            outputs=[cards_list, generation_status]
        )

        clear_cards_btn.click(
            fn=clear_cards,
            inputs=[],
            outputs=[cards_list, generation_status]
        )

        generate_deck_btn.click(
            fn=generate_anki_deck,
            inputs=[
                deck_name,
                deck_description,
                note_type,
                front_template,
                back_template,
                cloze_template,
                css_styling,
                cards_list
            ],
            outputs=[download_deck, generation_status]
        )


        return (
            deck_name,
            deck_description,
            note_type,
            front_template,
            back_template,
            cloze_template,
            css_styling,
            front_content,
            back_content,
            cloze_content,
            tags_input,
            cards_list,
            add_card_btn,
            clear_cards_btn,
            generate_deck_btn,
            download_deck,
            generation_status
        )

#
# End of Anki_Validation_tab.py
############################################################################################################
