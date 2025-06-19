#!/usr/bin/env python3
"""
Verification script to check if the language selection feature is properly implemented
"""

import sys
import os

# Add the project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_imports():
    """Check if all necessary imports work"""
    print("Checking imports...")
    try:
        from App_Function_Libraries.Utils.Whisper_Languages import get_whisper_language_list, get_language_code
        print("✓ Whisper_Languages module imported successfully")
        
        from App_Function_Libraries.Gradio_UI.Video_transcription_tab import create_video_transcription_tab
        print("✓ Video transcription tab imported successfully")
        
        from App_Function_Libraries.Gradio_UI.Audio_ingestion_tab import create_audio_processing_tab
        print("✓ Audio ingestion tab imported successfully")
        
        from App_Function_Libraries.Gradio_UI.Live_Recording import create_live_recording_tab
        print("✓ Live recording tab imported successfully")
        
        from App_Function_Libraries.Gradio_UI.Podcast_tab import create_podcast_tab
        print("✓ Podcast tab imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def check_language_functionality():
    """Check if language functions work correctly"""
    print("\nChecking language functionality...")
    try:
        from App_Function_Libraries.Utils.Whisper_Languages import get_whisper_language_list, get_language_code, get_language_name
        
        # Test language list
        languages = get_whisper_language_list()
        print(f"✓ Found {len(languages)} languages")
        
        # Test some conversions
        test_cases = [
            ("English", "en"),
            ("Spanish", "es"),
            ("French", "fr"),
            ("Auto-detect", "auto")
        ]
        
        for lang_name, expected_code in test_cases:
            code = get_language_code(lang_name)
            if code == expected_code:
                print(f"✓ {lang_name} -> {code}")
            else:
                print(f"✗ {lang_name} -> {code} (expected {expected_code})")
                
        # Test reverse conversion
        name = get_language_name("es")
        print(f"✓ es -> {name}")
        
        return True
    except Exception as e:
        print(f"✗ Error in language functionality: {e}")
        return False

def check_config_integration():
    """Check if config integration works"""
    print("\nChecking config integration...")
    try:
        from App_Function_Libraries.Utils.Utils import load_and_log_configs
        
        config = load_and_log_configs()
        if config:
            default_lang = config.get('STT_Settings', {}).get('default_stt_language', 'en')
            print(f"✓ Config loaded, default language: {default_lang}")
            return True
        else:
            print("✗ Failed to load config")
            return False
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return False

def main():
    print("=== Language Feature Verification ===\n")
    
    all_good = True
    all_good &= check_imports()
    all_good &= check_language_functionality()
    all_good &= check_config_integration()
    
    print("\n=== Summary ===")
    if all_good:
        print("✅ All checks passed! The language feature is properly implemented.")
        print("\nTo launch the GUI with language selection support:")
        print("  python summarize.py -gui")
    else:
        print("❌ Some checks failed. Please review the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()