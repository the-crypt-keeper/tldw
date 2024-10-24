# Anki.py
# Description: Functions for Anki card generation
#
# Imports
#
# External Imports
from outlines import models, prompts
# Local Imports
#
############################################################################################################
#
# Functions:

def create_anki_schema():
    """Define schema for card generation using Outlines"""
    return {
        "type": "object",
        "properties": {
            "cards": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "type": {"type": "string", "enum": ["basic", "cloze", "reverse"]},
                        "front": {"type": "string"},
                        "back": {"type": "string"},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "note": {"type": "string"}
                    },
                    "required": ["id", "type", "front", "back", "tags"]
                }
            }
        }
    }


def generate_cards_with_outlines(text, num_cards, config):
    """Generate cards using Outlines for structured output"""
    schema = create_anki_schema()

    # Create prompt template
    template = prompts.TextTemplate("""
    Generate {num_cards} Anki flashcards about: {text}

    Requirements:
    - Difficulty: {difficulty}
    - Subject: {subject}
    - Card Types: {card_types}

    Each card must have:
    1. Unique ID
    2. Type (basic/cloze/reverse)
    3. Front content
    4. Back content
    5. Relevant tags
    6. Optional note/hint
    """)

    # Configure model
    model = models.Model(api_endpoint)

    # Generate with schema validation
    response = model.generate(
        template,
        schema=schema,
        num_cards=num_cards,
        text=text,
        **config
    )

    return response

#
# End of Anki.py
############################################################################################################
