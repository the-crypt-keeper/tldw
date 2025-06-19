#!/usr/bin/env python3
"""
Fixed version of the tldw application with enhanced error handling
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add the App_Function_Libraries to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'App_Function_Libraries')))

# Disable Gradio analytics
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

def ensure_dependencies():
    """Check and install missing dependencies"""
    missing_packages = []
    
    # Critical packages to check
    packages = [
        'gradio',
        'torch',
        'nltk',
        'transformers',
        'sqlite3'
    ]
    
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logging.error(f"Missing packages: {', '.join(missing_packages)}")
        logging.info("Please run: pip install -r requirements.txt")
        return False
    
    # Download NLTK data if needed
    try:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logging.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
    except Exception as e:
        logging.warning(f"NLTK setup warning: {e}")
    
    return True

def ensure_directories():
    """Create necessary directories"""
    dirs = [
        'Databases',
        'Logs',
        'Config_Files',
        'Docs/Screenshots'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logging.debug(f"Ensured directory exists: {dir_path}")

def create_default_config():
    """Create default configuration if missing"""
    config_path = Path('Config_Files/config.txt')
    
    if not config_path.exists():
        default_config = """# Configuration file for tldw
# Add your API keys and settings here

[API_KEYS]
openai_api_key = 
anthropic_api_key = 
groq_api_key = 
cohere_api_key = 

[Database]
type = sqlite
sqlite_path = ./Databases/media_db.db

[Transcription]
whisper_model = small
vad_filter = false

[Local_LLM]
enable_local_llm = false
local_llm_path = ./Models/

[Server]
host = 127.0.0.1
port = 7860
share_public = false

[Logging]
log_level = INFO
log_file = ./Logs/tldw.log
"""
        config_path.parent.mkdir(exist_ok=True)
        config_path.write_text(default_config)
        logging.info(f"Created default config at {config_path}")

def main():
    """Main entry point with enhanced error handling"""
    parser = argparse.ArgumentParser(
        description='tl/dw - Too Long Didn\'t Watch: Transcribe and summarize media',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('-gui', '--user_interface', action='store_true', 
                       help='Launch the Gradio user interface')
    parser.add_argument('--port', type=int, default=7860, 
                       help='Port to run the server on (default: 7860)')
    parser.add_argument('--share', action='store_true',
                       help='Share the interface publicly via Gradio')
    parser.add_argument('--server-mode', action='store_true',
                       help='Run in server mode (expose to network)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Set logging level')
    parser.add_argument('--check-health', action='store_true',
                       help='Run health check and exit')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Run health check if requested
    if args.check_health:
        from check_app_health import main as health_check
        health_check()
        return
    
    try:
        # Ensure all prerequisites are met
        logging.info("Initializing tldw application...")
        
        ensure_directories()
        create_default_config()
        
        if not ensure_dependencies():
            logging.error("Missing dependencies. Please install requirements.")
            sys.exit(1)
        
        if args.user_interface:
            logging.info("Launching Gradio interface...")
            
            # Try to import the fixed version first
            try:
                from App_Function_Libraries.Gradio_Related_Fixed import launch_ui_safe
                launch_ui_safe(
                    share_public=args.share,
                    server_mode=args.server_mode
                )
            except ImportError:
                # Fall back to original if fixed version doesn't exist
                logging.warning("Using original Gradio launcher...")
                from App_Function_Libraries.Gradio_Related import launch_ui
                launch_ui(
                    share_public=args.share,
                    server_mode=args.server_mode
                )
        else:
            parser.print_help()
            logging.info("\nTo launch the GUI, run: python app_fixed.py -gui")
            
    except KeyboardInterrupt:
        logging.info("\nShutting down gracefully...")
    except Exception as e:
        logging.error(f"Critical error: {e}", exc_info=True)
        logging.info("\nFor troubleshooting, run: python app_fixed.py --check-health")
        sys.exit(1)

if __name__ == "__main__":
    main()