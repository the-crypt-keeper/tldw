#!/usr/bin/env python
#
# Usage:
#           python ingest_prompts.py /path/to/your/parent/folder
#
import os
import sqlite3
import argparse

DATABASE_PATH = '../prompts.db'


def ingest_prompts_from_folder(parent_folder):
    # Function to read the content of a file
    def read_file_content(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()

    # Connect to the prompts database
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # Iterate over each folder in the parent folder
    for folder_name in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder_name)
        if os.path.isdir(folder_path):
            # Set the default values
            name = folder_name
            details = 'ingested-via-script'
            system = ''

            user = None

            # Check for the system.md file and read its content
            system_file_path = os.path.join(folder_path, 'system.md')
            if os.path.isfile(system_file_path):
                system = read_file_content(system_file_path)

            # Check for the user.md file and read its content, if it exists
            user_file_path = os.path.join(folder_path, 'user.md')
            if os.path.isfile(user_file_path):
                user = read_file_content(user_file_path)

            # Insert the data into the Prompts table
            try:
                cursor.execute('''
                    INSERT INTO Prompts (name, details, system, user)
                    VALUES (?, ?, ?, ?)
                ''', (name, details, system, user))
                print(f"Successfully ingested prompt: {name}")
            except sqlite3.IntegrityError:
                print(f"Prompt with name '{name}' already exists.")
            except sqlite3.Error as e:
                print(f"Database error: {e}")

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Ingest prompts from a specified folder into the Prompts database.')
    parser.add_argument('folder', type=str, help='The path to the parent folder containing prompt folders.')

    # Parse the arguments
    args = parser.parse_args()

    # Run the ingestion script
    ingest_prompts_from_folder(args.folder)