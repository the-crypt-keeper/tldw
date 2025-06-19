#!/usr/bin/env python3
"""
Health check script for tldw Gradio application
Checks for common issues and provides fixes
"""

import os
import sys
import importlib
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ required.")
        return False
    print(f"✅ Python {version.major}.{version.minor} detected.")
    return True

def check_required_directories():
    """Check if required directories exist"""
    print("\nChecking required directories...")
    required_dirs = [
        "App_Function_Libraries",
        "Databases",
        "Config_Files",
        "Docs",
        "Logs"
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing_dirs.append(dir_name)
            print(f"❌ Missing directory: {dir_name}")
    
    if missing_dirs:
        print("\nCreating missing directories...")
        for dir_name in missing_dirs:
            os.makedirs(dir_name, exist_ok=True)
            print(f"✅ Created: {dir_name}")
    else:
        print("✅ All required directories exist.")
    
    return True

def check_config_files():
    """Check if configuration files exist"""
    print("\nChecking configuration files...")
    config_files = {
        "Config_Files/config.txt": """# Configuration file for tldw
# Add your API keys and settings here

[API_KEYS]
openai_api_key = 
anthropic_api_key = 
groq_api_key = 

[PATHS]
media_db = ./Databases/media_db.db
""",
        "App_Function_Libraries/config.yaml": """pipeline:
  name: pyannote.audio.pipelines.SpeakerDiarization
  params:
    clustering: AgglomerativeClustering
    embedding: pyannote_model_wespeaker-voxceleb-resnet34-LM.bin
    embedding_batch_size: 1
    embedding_exclude_overlap: true
    segmentation: pyannote_model_segmentation-3.0.bin
    segmentation_batch_size: 32

params:
  clustering:
    method: centroid
    min_cluster_size: 12
    threshold: 0.7045654963945799
  segmentation:
    min_duration_off: 0.0
"""
    }
    
    for config_file, default_content in config_files.items():
        if not os.path.exists(config_file):
            print(f"❌ Missing config file: {config_file}")
            # Create with default content
            os.makedirs(os.path.dirname(config_file), exist_ok=True)
            with open(config_file, 'w') as f:
                f.write(default_content)
            print(f"✅ Created default config: {config_file}")
        else:
            print(f"✅ Config file exists: {config_file}")
    
    return True

def check_imports():
    """Check if all required modules can be imported"""
    print("\nChecking module imports...")
    
    # Critical imports
    critical_modules = [
        'gradio',
        'torch',
        'whisper',
        'nltk',
        'transformers',
        'chromadb',
        'sqlite3',
        'fastapi'
    ]
    
    missing_modules = []
    for module in critical_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {str(e)}")
            missing_modules.append(module)
    
    # Check local imports
    print("\nChecking local module imports...")
    sys.path.append(os.path.abspath('App_Function_Libraries'))
    
    local_modules = [
        'App_Function_Libraries.Gradio_Related',
        'App_Function_Libraries.DB.DB_Manager',
        'App_Function_Libraries.DB.SQLite_DB',
        'App_Function_Libraries.Utils.Utils'
    ]
    
    for module in local_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {str(e)}")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n⚠️  Missing {len(missing_modules)} modules. Run: pip install -r requirements.txt")
        return False
    
    return True

def check_gradio_theme():
    """Check and fix Gradio theme issues"""
    print("\nChecking Gradio configuration...")
    
    # Check if we need to update the theme
    gradio_file = "App_Function_Libraries/Gradio_Related.py"
    if os.path.exists(gradio_file):
        with open(gradio_file, 'r') as f:
            content = f.read()
        
        if 'bethecloud/storj_theme' in content:
            print("⚠️  Deprecated theme detected. Updating to default theme...")
            # Update to a safe default theme
            updated_content = content.replace(
                "theme='bethecloud/storj_theme'", 
                "theme='default'"
            )
            with open(gradio_file, 'w') as f:
                f.write(updated_content)
            print("✅ Updated Gradio theme to default")
    
    return True

def check_database():
    """Check database setup"""
    print("\nChecking database setup...")
    
    db_dir = "Databases"
    if not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
        print(f"✅ Created database directory: {db_dir}")
    
    # Check for SQLite database
    db_files = ['media_db.db', 'chatDB.db', 'rag_qa.db']
    for db_file in db_files:
        db_path = os.path.join(db_dir, db_file)
        if os.path.exists(db_path):
            print(f"✅ Database exists: {db_file}")
        else:
            print(f"⚠️  Database will be created on first run: {db_file}")
    
    return True

def check_ffmpeg():
    """Check if ffmpeg is installed"""
    print("\nChecking ffmpeg installation...")
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ ffmpeg is installed")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ ffmpeg not found. Please install ffmpeg:")
    print("  - macOS: brew install ffmpeg")
    print("  - Ubuntu/Debian: sudo apt-get install ffmpeg")
    print("  - Windows: Download from https://ffmpeg.org/download.html")
    return False

def main():
    """Run all health checks"""
    print("=" * 60)
    print("tldw Application Health Check")
    print("=" * 60)
    
    checks = [
        check_python_version(),
        check_required_directories(),
        check_config_files(),
        check_imports(),
        check_gradio_theme(),
        check_database(),
        check_ffmpeg()
    ]
    
    passed = sum(checks)
    total = len(checks)
    
    print("\n" + "=" * 60)
    print(f"Health Check Summary: {passed}/{total} checks passed")
    
    if passed == total:
        print("✅ All checks passed! The application should run properly.")
        print("\nTo start the application, run:")
        print("  python summarize.py -gui")
    else:
        print("⚠️  Some checks failed. Please address the issues above.")
        print("\nCommon fixes:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Install ffmpeg (see instructions above)")
        print("  3. Check Python version (3.8+ required)")
    
    print("=" * 60)

if __name__ == "__main__":
    main()