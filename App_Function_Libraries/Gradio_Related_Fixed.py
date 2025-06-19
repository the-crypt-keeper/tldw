# Gradio_Related_Fixed.py
# Enhanced version with better error handling and fixes

import os
import sys
import webbrowser
import traceback
import gradio as gr
from pathlib import Path

# Import with error handling
try:
    from App_Function_Libraries.DB.DB_Manager import get_db_config
    from App_Function_Libraries.DB.RAG_QA_Chat_DB import create_tables
    from App_Function_Libraries.Utils.Utils import load_and_log_configs, logging
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all dependencies are installed and paths are correct.")
    sys.exit(1)

# Ensure directories exist
def ensure_directories():
    """Create necessary directories if they don't exist"""
    dirs = [
        'Databases',
        'Logs',
        'Config_Files'
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(exist_ok=True)

def safe_import_tabs():
    """Import all tab creation functions with error handling"""
    tab_imports = {}
    
    # List of all tab modules to import
    tab_modules = [
        ('Anki_tab', ['create_anki_validation_tab', 'create_anki_generator_tab']),
        ('Arxiv_tab', ['create_arxiv_tab']),
        ('Audio_ingestion_tab', ['create_audio_processing_tab']),
        ('Video_transcription_tab', ['create_video_transcription_tab']),
        # Add more as needed
    ]
    
    for module_name, functions in tab_modules:
        try:
            module = __import__(f'App_Function_Libraries.Gradio_UI.{module_name}', fromlist=functions)
            for func_name in functions:
                tab_imports[func_name] = getattr(module, func_name)
        except Exception as e:
            logging.error(f"Failed to import {module_name}: {e}")
            # Create a dummy function that shows an error tab
            for func_name in functions:
                tab_imports[func_name] = lambda: gr.Markdown(f"Error loading {func_name}: {str(e)}")
    
    return tab_imports

def launch_ui_safe(share_public=None, server_mode=False, demo_mode=False):
    """Enhanced launch_ui with better error handling"""
    
    # Ensure directories exist
    ensure_directories()
    
    # Don't open browser in demo mode
    if not demo_mode:
        try:
            webbrowser.open_new_tab('http://127.0.0.1:7860/?__theme=dark')
        except Exception as e:
            logging.warning(f"Could not open browser: {e}")
    
    share = share_public
    
    # CSS styling
    css = """
    .result-box {
        margin-bottom: 20px;
        border: 1px solid #ddd;
        padding: 10px;
    }
    .result-box.error {
        border-color: #ff0000;
        background-color: #ffeeee;
    }
    .transcription, .summary {
        max-height: 800px;
        overflow-y: auto;
        border: 1px solid #eee;
        padding: 10px;
        margin-top: 10px;
    }
    #scrollable-textbox textarea {
        max-height: 500px !important; 
        overflow-y: auto !important;
    }
    """
    
    try:
        # Load configuration with error handling
        config = load_and_log_configs()
        if not config:
            logging.error("Failed to load configuration")
            config = {'db_config': {'sqlite_path': './Databases/media_db.db', 'type': 'sqlite'}}
        
        # Get database paths
        db_config = config.get('db_config', {})
        media_db_path = db_config.get('sqlite_path', './Databases/media_db.db')
        
        # Ensure database directory exists
        db_dir = os.path.dirname(media_db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            logging.info(f"Created database directory: {db_dir}")
        
        character_chat_db_path = os.path.join(db_dir, "chatDB.db")
        rag_chat_db_path = os.path.join(db_dir, "rag_qa.db")
        
        # Initialize databases with error handling
        try:
            create_tables()
            logging.info("Database tables created successfully")
        except Exception as e:
            logging.error(f"Error creating database tables: {e}")
        
        # Import all tab functions
        tabs = safe_import_tabs()
        
        # Create Gradio interface
        with gr.Blocks(theme='default', css=css) as iface:
            # Add dark mode script
            gr.HTML("""
                <script>
                document.addEventListener('DOMContentLoaded', (event) => {
                    document.body.classList.add('dark');
                    document.querySelector('gradio-app').style.backgroundColor = 'var(--color-background-primary)';
                });
                </script>
            """)
            
            # Get database type
            db_type = db_config.get('type', 'sqlite')
            
            # Header
            gr.Markdown("# tl/dw: Your LLM-powered Research Multi-tool")
            gr.Markdown(f"(Using {db_type.capitalize()} Database)")
            
            # Create minimal interface for testing
            with gr.Tabs():
                with gr.TabItem("Status", id="status"):
                    gr.Markdown("## System Status")
                    gr.Markdown(f"✅ Application loaded successfully")
                    gr.Markdown(f"📁 Database path: {media_db_path}")
                    gr.Markdown(f"🗄️ Database type: {db_type}")
                    
                with gr.TabItem("Test", id="test"):
                    gr.Markdown("## Test Tab")
                    test_input = gr.Textbox(label="Test Input")
                    test_output = gr.Textbox(label="Test Output")
                    test_button = gr.Button("Test")
                    
                    def test_function(text):
                        return f"Echo: {text}"
                    
                    test_button.click(test_function, inputs=test_input, outputs=test_output)
        
        # Launch settings
        server_port = int(os.getenv('GRADIO_SERVER_PORT', 7860))
        
        # Disable analytics
        os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
        
        # Launch the interface
        launch_kwargs = {
            'share': share,
            'server_port': server_port,
            'show_error': True
        }
        
        if server_mode:
            launch_kwargs['server_name'] = "0.0.0.0"
        
        try:
            iface.launch(**launch_kwargs)
        except Exception as e:
            logging.error(f"Error launching Gradio interface: {e}")
            # Try alternative port
            logging.info("Trying alternative port 7861...")
            launch_kwargs['server_port'] = 7861
            iface.launch(**launch_kwargs)
            
    except Exception as e:
        logging.error(f"Critical error in launch_ui: {e}")
        logging.error(traceback.format_exc())
        
        # Create minimal error interface
        with gr.Blocks() as error_iface:
            gr.Markdown("# Error Loading Application")
            gr.Markdown(f"An error occurred: {str(e)}")
            gr.Markdown("Please check the logs for more information.")
        
        error_iface.launch(share=False, server_port=7860)