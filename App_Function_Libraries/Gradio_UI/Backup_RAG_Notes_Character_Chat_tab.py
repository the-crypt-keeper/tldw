# Backup_Functionality.py
# Functionality for managing database backups
#
# Imports:
import os
import shutil
import gradio as gr
from typing import Dict, Optional
#
# Local Imports:
from App_Function_Libraries.DB.DB_Manager import create_automated_backup
from App_Function_Libraries.DB.DB_Backups import create_backup, create_incremental_backup, list_backups, restore_backup


#
# End of Imports
#######################################################################################################################
#
# Functions:

def create_backup_tab(db_path: str, backup_dir: str, db_name: str):
    """Create the backup creation tab for a database."""
    gr.Markdown("## Create Database Backup")
    gr.Markdown(f"This will create a backup in the directory: `{backup_dir}`")
    with gr.Row():
        with gr.Column():
            automated_backup_btn = gr.Button("Create Simple Backup")
            full_backup_btn = gr.Button("Create Full Backup")
            incr_backup_btn = gr.Button("Create Incremental Backup")
        with gr.Column():
            backup_output = gr.Textbox(label="Result")

    def create_db_backup():
        backup_file = create_automated_backup(db_path, backup_dir)
        return f"Backup created: {backup_file}"

    automated_backup_btn.click(
        fn=create_db_backup,
        inputs=[],
        outputs=[backup_output]
    )
    full_backup_btn.click(
        fn=lambda: create_backup(db_path, backup_dir, db_name),
        inputs=[],
        outputs=[backup_output]
    )
    incr_backup_btn.click(
        fn=lambda: create_incremental_backup(db_path, backup_dir, db_name),
        inputs=[],
        outputs=[backup_output]
    )


def create_view_backups_tab(backup_dir: str):
    """Create the backup viewing tab for a database."""
    gr.Markdown("## Available Backups")
    with gr.Row():
        with gr.Column():
            view_btn = gr.Button("Refresh Backup List")
        with gr.Column():
            backup_list = gr.Textbox(label="Available Backups")

    def list_all_backups():
        backups = [f for f in os.listdir(backup_dir) if f.endswith('.db')]
        return "\n".join(backups)

    view_btn.click(
        fn=list_all_backups,
        inputs=[],
        outputs=[backup_list]
    )


def create_restore_backup_tab(db_path: str, backup_dir: str, db_name: str):
    """Create the backup restoration tab for a database."""
    gr.Markdown("## Restore Database")
    gr.Markdown("⚠️ **Warning**: Restoring a backup will overwrite the current database.")
    with gr.Row():
        with gr.Column():
            backup_input = gr.Textbox(label="Backup Filename")
            restore_btn = gr.Button("Restore", variant="primary")
        with gr.Column():
            restore_output = gr.Textbox(label="Result")

    def unified_restore(backup_name: str) -> str:
        backup_path = os.path.join(backup_dir, backup_name)
        if not os.path.exists(backup_path):
            return "Backup file not found"

        if backup_name.startswith(f"{db_name}_"):
            # Use managed backup restoration
            return restore_backup(db_path, backup_dir, db_name, backup_name)
        else:
            # Use automated backup restoration
            shutil.copy2(backup_path, db_path)
            return f"Database restored from {backup_name}"

    restore_btn.click(
        fn=unified_restore,
        inputs=[backup_input],
        outputs=[restore_output]
    )


def create_media_db_tabs(db_config: Dict[str, str]):
    """Create all tabs for the Media database."""
    create_backup_tab(
        db_path=db_config['db_path'],
        backup_dir=db_config['backup_dir'],
        db_name='media'
    )
    create_view_backups_tab(db_config['backup_dir'])
    create_restore_backup_tab(
        db_path=db_config['db_path'],
        backup_dir=db_config['backup_dir'],
        db_name='media'
    )


def create_rag_chat_tabs(db_config: Dict[str, str]):
    """Create all tabs for the RAG Chat database."""
    create_backup_tab(
        db_path=db_config['db_path'],
        backup_dir=db_config['backup_dir'],
        db_name='rag_chat'
    )
    create_view_backups_tab(db_config['backup_dir'])
    create_restore_backup_tab(
        db_path=db_config['db_path'],
        backup_dir=db_config['backup_dir'],
        db_name='rag_chat'
    )


def create_character_chat_tabs(db_config: Dict[str, str]):
    """Create all tabs for the Character Chat database."""
    create_backup_tab(
        db_path=db_config['db_path'],
        backup_dir=db_config['backup_dir'],
        db_name='character_chat'
    )
    create_view_backups_tab(db_config['backup_dir'])
    create_restore_backup_tab(
        db_path=db_config['db_path'],
        backup_dir=db_config['backup_dir'],
        db_name='character_chat'
    )


def create_database_management_interface(
        media_db_config: Dict[str, str],
        rag_db_config: Dict[str, str],
        char_db_config: Dict[str, str]
):
    """Create the main database management interface with tabs for each database."""
    with gr.TabItem("Media Database", id="media_db_group", visible=True):
        create_media_db_tabs(media_db_config)

    with gr.TabItem("RAG Chat Database", id="rag_chat_group", visible=True):
        create_rag_chat_tabs(rag_db_config)

    with gr.TabItem("Character Chat Database", id="character_chat_group", visible=True):
        create_character_chat_tabs(char_db_config)

#
# End of Functions
#######################################################################################################################