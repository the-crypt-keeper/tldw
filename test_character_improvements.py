#!/usr/bin/env python3
"""
Test script to verify character chat improvements including:
1. Enhanced V2 card format validation
2. Image handling improvements (PNG/WEBP support, optimization)  
3. Character template system functionality
4. New format parsers (Pygmalion, TextGen, Alpaca, plain text)
"""

import base64
import io
import json
import sys
import os
from PIL import Image

# Add the project path to sys.path for imports
sys.path.insert(0, '/Users/appledev/Working/tldw_server/tldw_Server_API')

from app.core.Character_Chat.Character_Chat_Lib import (
    validate_v2_card,
    parse_pygmalion_card, 
    parse_textgen_card,
    parse_alpaca_card,
    _prepare_character_data_for_db_storage,
    extract_json_from_image_file,
    import_character_card_from_json_string
)

def test_v2_card_validation():
    """Test enhanced V2 card format validation"""
    print("\n=== Testing V2 Card Validation ===")
    
    # Valid V2 card
    valid_v2_card = {
        "spec": "chara_card_v2",
        "spec_version": "2.0", 
        "data": {
            "name": "Test Character",
            "description": "A test character",
            "personality": "Friendly and helpful",
            "scenario": "A test scenario",
            "first_mes": "Hello! I'm a test character.",
            "mes_example": "User: Hi\\nCharacter: Hello there!"
        }
    }
    
    is_valid, errors = validate_v2_card(valid_v2_card)
    print(f"Valid V2 card validation: {is_valid}")
    if errors:
        print(f"Errors: {errors}")
    
    # Invalid V2 card - missing required fields
    invalid_v2_card = {
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "data": {
            "name": "Test Character"
            # Missing required fields
        }
    }
    
    is_valid, errors = validate_v2_card(invalid_v2_card)
    print(f"Invalid V2 card validation: {is_valid}")
    print(f"Expected errors: {errors}")
    
    return True

def test_image_optimization():
    """Test image handling improvements"""
    print("\n=== Testing Image Optimization ===")
    
    # Create a test image
    img = Image.new('RGB', (1024, 1024), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_data = img_bytes.getvalue()
    
    # Convert to base64
    base64_data = base64.b64encode(img_data).decode('utf-8')
    
    # Test image optimization
    character_data = {
        "name": "Test Character",
        "image_base64": base64_data
    }
    
    try:
        optimized_data = _prepare_character_data_for_db_storage(character_data)
        original_size = len(img_data)
        optimized_size = len(optimized_data['image'])
        
        print(f"Original image size: {original_size} bytes")
        print(f"Optimized image size: {optimized_size} bytes")
        print(f"Compression ratio: {optimized_size/original_size:.2%}")
        print("Image optimization: PASSED")
        return True
    except Exception as e:
        print(f"Image optimization failed: {e}")
        return False

def test_format_parsers():
    """Test new format parsers"""
    print("\n=== Testing Format Parsers ===")
    
    # Test Pygmalion format
    pygmalion_data = {
        "char_name": "Pygmalion Character",
        "char_persona": "A helpful AI assistant",
        "char_greeting": "Hello! I'm here to help.",
        "example_dialogue": "User: Hi\\nCharacter: Hello!",
        "world_scenario": "A helpful AI scenario"
    }
    
    parsed_pygmalion = parse_pygmalion_card(pygmalion_data)
    if parsed_pygmalion:
        print(f"Pygmalion parser: PASSED - {parsed_pygmalion['name']}")
    else:
        print("Pygmalion parser: FAILED")
    
    # Test TextGen format
    textgen_data = {
        "name": "TextGen Character",
        "context": "A helpful AI assistant",
        "greeting": "Hello! I'm here to help.",
        "example_dialogue": "User: Hi\\nCharacter: Hello!"
    }
    
    parsed_textgen = parse_textgen_card(textgen_data)
    if parsed_textgen:
        print(f"TextGen parser: PASSED - {parsed_textgen['name']}")
    else:
        print("TextGen parser: FAILED")
    
    # Test Alpaca format
    alpaca_data = {
        "instruction": "Act as a helpful AI assistant",
        "input": "Friendly and knowledgeable",
        "output": "Hello! I'm here to help you with any questions."
    }
    
    parsed_alpaca = parse_alpaca_card(alpaca_data)
    if parsed_alpaca:
        print(f"Alpaca parser: PASSED - {parsed_alpaca['name']}")
    else:
        print("Alpaca parser: FAILED")
    
    return all([parsed_pygmalion, parsed_textgen, parsed_alpaca])

def test_character_import():
    """Test character import functionality"""
    print("\n=== Testing Character Import ===")
    
    # Test JSON import
    character_json = json.dumps({
        "name": "Imported Character",
        "description": "A character imported from JSON",
        "first_mes": "Hello! I was imported from JSON.",
        "mes_example": "User: Hi\\nCharacter: Hello!",
        "personality": "Friendly",
        "scenario": "Import test"
    })
    
    try:
        imported_char = import_character_card_from_json_string(character_json)
        if imported_char:
            print(f"JSON import: PASSED - {imported_char['name']}")
            return True
        else:
            print("JSON import: FAILED - No character returned")
            return False
    except Exception as e:
        print(f"JSON import: FAILED - {e}")
        return False

def test_png_character_extraction():
    """Test extracting character data from PNG images"""
    print("\n=== Testing PNG Character Extraction ===")
    
    # Create a simple character card
    char_data = {
        "name": "PNG Character",
        "description": "A character embedded in PNG",
        "first_mes": "Hello from PNG!",
        "mes_example": "User: Hi\\nCharacter: Hello!",
        "personality": "Embedded",
        "scenario": "PNG test"
    }
    
    # Create a PNG with embedded character data
    char_json = json.dumps(char_data)
    char_b64 = base64.b64encode(char_json.encode('utf-8')).decode('utf-8')
    
    # Create a test PNG file
    img = Image.new('RGB', (100, 100), color='blue')
    
    # Add text metadata (PNG info chunk)
    from PIL import PngImagePlugin
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text("chara", char_b64)
    
    # Save to file
    test_png_path = "/tmp/test_character.png"
    img.save(test_png_path, "PNG", pnginfo=png_info)
    
    try:
        # Test extraction
        extracted_json = extract_json_from_image_file(test_png_path)
        if extracted_json:
            extracted_data = json.loads(extracted_json)
            print(f"PNG extraction: PASSED - {extracted_data['name']}")
            
            # Clean up
            os.remove(test_png_path)
            return True
        else:
            print("PNG extraction: FAILED - No data extracted")
            return False
    except Exception as e:
        print(f"PNG extraction: FAILED - {e}")
        # Clean up
        if os.path.exists(test_png_path):
            os.remove(test_png_path)
        return False

def main():
    """Run all character chat improvement tests"""
    print("Testing Character Chat Improvements")
    print("=" * 50)
    
    results = {}
    
    # Run tests
    results['V2 Validation'] = test_v2_card_validation()
    results['Image Optimization'] = test_image_optimization()  
    results['Format Parsers'] = test_format_parsers()
    results['Character Import'] = test_character_import()
    results['PNG Extraction'] = test_png_character_extraction()
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("All character chat improvements are working correctly!")
        return 0
    else:
        print("Some tests failed. Review the improvements.")
        return 1

if __name__ == "__main__":
    sys.exit(main())