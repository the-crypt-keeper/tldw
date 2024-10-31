# Keywords.py
# Purpose: This file contains the functions to create the Keywords tab in the Gradio UI.
#
# The Keywords tab allows the user to add, delete, view, and export keywords from the database.
#
# Imports:
#
# External Imports
import gradio as gr

from App_Function_Libraries.DB.Character_Chat_DB import view_char_keywords, add_char_keywords
#
# Internal Imports
from App_Function_Libraries.DB.DB_Manager import add_keyword, delete_keyword, keywords_browser_interface, export_keywords_to_csv
from App_Function_Libraries.DB.RAG_QA_Chat_DB import view_rag_keywords, get_all_collections, \
    get_keywords_for_collection, create_keyword_collection, add_keyword_to_collection


#
######################################################################################################################
#
# Functions:

def create_export_keywords_tab():
    with gr.TabItem("Export MediaDB Keywords", visible=True):
        with gr.Row():
            with gr.Column():
                export_keywords_button = gr.Button("Export Keywords")
            with gr.Column():
                export_keywords_output = gr.File(label="Download Exported Keywords")
                export_keywords_status = gr.Textbox(label="Export Status")

            export_keywords_button.click(
                fn=export_keywords_to_csv,
                outputs=[export_keywords_status, export_keywords_output]
            )

def create_view_keywords_tab():
    with gr.TabItem("View MediaDB Keywords", visible=True):
        gr.Markdown("# Browse MediaDB Keywords")
        with gr.Column():
            browse_output = gr.Markdown()
            browse_button = gr.Button("View Existing Keywords")
            browse_button.click(fn=keywords_browser_interface, outputs=browse_output)


def create_add_keyword_tab():
    with gr.TabItem("Add MediaDB Keywords", visible=True):
        with gr.Row():
            with gr.Column():
                gr.Markdown("# Add Keywords to the Database")
                add_input = gr.Textbox(label="Add Keywords (comma-separated)", placeholder="Enter keywords here...")
                add_button = gr.Button("Add Keywords")
            with gr.Row():
                add_output = gr.Textbox(label="Result")
                add_button.click(fn=add_keyword, inputs=add_input, outputs=add_output)


def create_delete_keyword_tab():
    with gr.Tab("Delete MediaDB Keywords", visible=True):
        with gr.Row():
            with gr.Column():
                gr.Markdown("# Delete Keywords from the Database")
                delete_input = gr.Textbox(label="Delete Keyword", placeholder="Enter keyword to delete here...")
                delete_button = gr.Button("Delete Keyword")
            with gr.Row():
                delete_output = gr.Textbox(label="Result")
                delete_button.click(fn=delete_keyword, inputs=delete_input, outputs=delete_output)

#
# End of Media DB Keyword tabs
##########################################################


############################################################
#
# Character DB Keyword functions

def create_character_keywords_tab():
    """Creates the Character Keywords management tab"""
    with gr.Tab("Character Keywords"):
        gr.Markdown("# Character Keywords Management")

        with gr.Tabs():
            # View Character Keywords Tab
            with gr.TabItem("View Keywords"):
                with gr.Column():
                    refresh_char_keywords = gr.Button("Refresh Character Keywords")
                    char_keywords_output = gr.Markdown()
                    view_char_keywords()
                    refresh_char_keywords.click(
                        fn=view_char_keywords,
                        outputs=char_keywords_output
                    )

            # Add Character Keywords Tab
            with gr.TabItem("Add Keywords"):
                with gr.Column():
                    char_name = gr.Textbox(label="Character Name")
                    new_keywords = gr.Textbox(label="New Keywords (comma-separated)")
                    add_char_keyword_btn = gr.Button("Add Keywords")
                    add_char_result = gr.Markdown()

                    add_char_keyword_btn.click(
                        fn=add_char_keywords,
                        inputs=[char_name, new_keywords],
                        outputs=add_char_result
                    )

#
# End of Character Keywords tab
##########################################################

############################################################
#
# RAG QA Keywords functions

def create_rag_qa_keywords_tab():
    """Creates the RAG QA Keywords management tab"""
    with gr.Tab("RAG QA Keywords"):
        gr.Markdown("# RAG QA Keywords Management")

        with gr.Tabs():
            # View RAG QA Keywords Tab
            with gr.TabItem("View Keywords"):
                with gr.Column():
                    refresh_rag_keywords = gr.Button("Refresh RAG QA Keywords")
                    rag_keywords_output = gr.Markdown()

                    view_rag_keywords()

                    refresh_rag_keywords.click(
                        fn=view_rag_keywords,
                        outputs=rag_keywords_output
                    )

            # Add RAG QA Keywords Tab
            with gr.TabItem("Add Keywords"):
                with gr.Column():
                    new_rag_keywords = gr.Textbox(label="New Keywords (comma-separated)")
                    add_rag_keyword_btn = gr.Button("Add Keywords")
                    add_rag_result = gr.Markdown()

                    add_rag_keyword_btn.click(
                        fn=add_keyword,
                        inputs=new_rag_keywords,
                        outputs=add_rag_result
                    )

#
# End of RAG QA Keywords tab
##########################################################


############################################################
#
# Meta-Keywords functions

def create_meta_keywords_tab():
    """Creates the Meta-Keywords management tab"""
    with gr.Tab("Meta-Keywords"):
        gr.Markdown("# Meta-Keywords Management")

        with gr.Tabs():
            # View Meta-Keywords Tab
            with gr.TabItem("View Collections"):
                with gr.Column():
                    refresh_collections = gr.Button("Refresh Collections")
                    collections_output = gr.Markdown()

                    def view_collections():
                        try:
                            collections, _, _ = get_all_collections()
                            if collections:
                                result = "### Keyword Collections:\n"
                                for collection in collections:
                                    keywords = get_keywords_for_collection(collection)
                                    result += f"\n**{collection}**:\n"
                                    result += "\n".join([f"- {k}" for k in keywords])
                                    result += "\n"
                                return result
                            return "No collections found."
                        except Exception as e:
                            return f"Error retrieving collections: {str(e)}"

                    refresh_collections.click(
                        fn=view_collections,
                        outputs=collections_output
                    )

            # Create Collection Tab
            with gr.TabItem("Create Collection"):
                with gr.Column():
                    collection_name = gr.Textbox(label="Collection Name")
                    create_collection_btn = gr.Button("Create Collection")
                    create_result = gr.Markdown()

                    def create_collection(name: str):
                        try:
                            create_keyword_collection(name)
                            return f"Successfully created collection: {name}"
                        except Exception as e:
                            return f"Error creating collection: {str(e)}"

                    create_collection_btn.click(
                        fn=create_collection,
                        inputs=collection_name,
                        outputs=create_result
                    )

            # Add Keywords to Collection Tab
            with gr.TabItem("Add to Collection"):
                with gr.Column():
                    collection_select = gr.Textbox(label="Collection Name")
                    keywords_to_add = gr.Textbox(label="Keywords to Add (comma-separated)")
                    add_to_collection_btn = gr.Button("Add Keywords to Collection")
                    add_to_collection_result = gr.Markdown()

                    def add_keywords_to_collection(collection: str, keywords: str):
                        try:
                            keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
                            for keyword in keywords_list:
                                add_keyword_to_collection(collection, keyword)
                            return f"Successfully added {len(keywords_list)} keywords to collection {collection}"
                        except Exception as e:
                            return f"Error adding keywords to collection: {str(e)}"

                    add_to_collection_btn.click(
                        fn=add_keywords_to_collection,
                        inputs=[collection_select, keywords_to_add],
                        outputs=add_to_collection_result
                    )

#
# End of Meta-Keywords tab
##########################################################

#
# End of Keywords.py
######################################################################################################################
