# Chat_related_functions.py
# Contains functions related to chat
# WIP.
#
# Importing required libraries
import json
import os
from pathlib import Path
import json
#
########################################################################################################################
# Set globals
CHARACTERS_FILE = Path('.', 'Helper_Scripts', 'Character_Cards', 'Characters.json')

def save_character(character_data):
    if CHARACTERS_FILE.exists():
        with CHARACTERS_FILE.open('r') as f:
            characters = json.load(f)
    else:
        characters = {}

    characters[character_data['name']] = character_data

    with CHARACTERS_FILE.open('w') as f:
        json.dump(characters, f, indent=2)


def load_characters():
    if os.path.exists(CHARACTERS_FILE):
        with open(CHARACTERS_FILE, 'r') as f:
            return json.load(f)
    return {}


def get_character_names():
    characters = load_characters()
    return list(characters.keys())




