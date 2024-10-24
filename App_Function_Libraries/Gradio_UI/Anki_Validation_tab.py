# Anki_Validation_tab.py
# Description: Gradio functions for the Anki Validation tab
#
# Imports
import json
#
# External Imports
import gradio as gr
#
# Local Imports
#
############################################################################################################
#
# Functions:

def create_anki_validation_tab():
    with gr.TabItem("Anki Flashcard Validation", visible=True):
        gr.Markdown("# Anki Flashcard Validation and Editor")

        with gr.Row():
            # Left Column: Input and Validation
            with gr.Column(scale=1):
                gr.Markdown("## Import or Create Flashcards")
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

                import_file = gr.File(
                    label="Or Import JSON File",
                    file_types=[".json"]
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
                    """)

        with gr.Row():
            # Card Editor
            gr.Markdown("## Card Editor")
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

                front_content = gr.TextArea(
                    label="Front Content",
                    lines=3
                )

                back_content = gr.TextArea(
                    label="Back Content",
                    lines=3
                )

                tags_input = gr.TextArea(
                    label="Tags (comma-separated)",
                    lines=1
                )

                notes_input = gr.TextArea(
                    label="Additional Notes",
                    lines=2
                )

                update_card_button = gr.Button("Update Card")
                delete_card_button = gr.Button("Delete Card", variant="stop")

        with gr.Row():
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

        # Helper Functions
        def validate_flashcards(content):
            try:
                data = json.loads(content)
                validation_results = []
                is_valid = True

                if not isinstance(data, dict) or 'cards' not in data:
                    return False, "Invalid JSON format. Must contain 'cards' array."

                seen_ids = set()
                for idx, card in enumerate(data['cards']):
                    card_issues = []

                    # Check required fields
                    if 'id' not in card:
                        card_issues.append("Missing ID")
                    elif card['id'] in seen_ids:
                        card_issues.append("Duplicate ID")
                    else:
                        seen_ids.add(card['id'])

                    if 'type' not in card or card['type'] not in ['basic', 'cloze', 'reverse']:
                        card_issues.append("Invalid card type")

                    if 'front' not in card or not card['front'].strip():
                        card_issues.append("Missing front content")

                    if 'back' not in card or not card['back'].strip():
                        card_issues.append("Missing back content")

                    if 'tags' not in card or not card['tags']:
                        card_issues.append("Missing tags")

                    # Content-specific validation
                    if card.get('type') == 'cloze':
                        if '{{c1::' not in card['front']:
                            card_issues.append("Invalid cloze format")

                    if card_issues:
                        is_valid = False
                        validation_results.append(f"Card {card['id']}: {', '.join(card_issues)}")

                return is_valid, "\n".join(validation_results) if validation_results else "All cards are valid!"

            except json.JSONDecodeError:
                return False, "Invalid JSON format"
            except Exception as e:
                return False, f"Validation error: {str(e)}"

        def load_card_for_editing(card_selection, current_content):
            if not card_selection or not current_content:
                return {}, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

            try:
                data = json.loads(current_content)
                selected_id = card_selection.split(" - ")[0]

                for card in data['cards']:
                    if card['id'] == selected_id:
                        return (
                            card,
                            card['type'],
                            card['front'],
                            card['back'],
                            ", ".join(card['tags']),
                            card.get('note', '')
                        )

                return {}, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

            except Exception as e:
                return {}, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

        def update_card(current_content, card_selection, card_type, front, back, tags, notes):
            try:
                data = json.loads(current_content)
                selected_id = card_selection.split(" - ")[0]

                for card in data['cards']:
                    if card['id'] == selected_id:
                        card['type'] = card_type
                        card['front'] = front
                        card['back'] = back
                        card['tags'] = [tag.strip() for tag in tags.split(',')]
                        card['note'] = notes

                return json.dumps(data, indent=2), "Card updated successfully!"

            except Exception as e:
                return current_content, f"Error updating card: {str(e)}"

        def export_cards(content, format_type):
            try:
                is_valid, validation_message = validate_flashcards(content)
                if not is_valid:
                    return "Please fix validation issues before exporting.", None

                data = json.loads(content)

                if format_type == "Anki CSV":
                    output = "Front,Back,Tags,Type,Note\n"
                    for card in data['cards']:
                        output += f'"{card["front"]}","{card["back"]}","{" ".join(card["tags"])}","{card["type"]}","{card.get("note", "")}"\n'
                    return "Cards exported successfully!", ("anki_cards.csv", output, "text/csv")

                elif format_type == "JSON":
                    return "Cards exported successfully!", ("anki_cards.json", content, "application/json")

                else:  # Plain Text
                    output = ""
                    for card in data['cards']:
                        output += f"Q: {card['front']}\nA: {card['back']}\n\n"
                    return "Cards exported successfully!", ("anki_cards.txt", output, "text/plain")

            except Exception as e:
                return f"Export error: {str(e)}", None

        # Register callbacks
        validate_button.click(
            fn=validate_flashcards,
            inputs=[flashcard_input],
            outputs=[validation_status]
        )

        card_selector.change(
            fn=load_card_for_editing,
            inputs=[card_selector, flashcard_input],
            outputs=[
                gr.State(),  # For storing current card data
                card_type,
                front_content,
                back_content,
                tags_input,
                notes_input
            ]
        )

        update_card_button.click(
            fn=update_card,
            inputs=[
                flashcard_input,
                card_selector,
                card_type,
                front_content,
                back_content,
                tags_input,
                notes_input
            ],
            outputs=[flashcard_input, validation_status]
        )

        export_button.click(
            fn=export_cards,
            inputs=[flashcard_input, export_format],
            outputs=[export_status, export_file]
        )

        return (
            flashcard_input,
            import_file,
            validate_button,
            validation_status,
            card_selector,
            card_type,
            front_content,
            back_content,
            tags_input,
            notes_input,
            update_card_button,
            delete_card_button,
            export_format,
            export_button,
            export_file,
            export_status
        )

#
# End of Anki_Validation_tab.py
############################################################################################################
