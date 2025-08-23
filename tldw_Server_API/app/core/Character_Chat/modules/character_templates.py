"""
Character templates module.

This module contains functions for managing character templates.
"""

from typing import Dict, List, Optional, Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError


# Default character templates
CHARACTER_TEMPLATES = {
    "assistant": {
        "name": "Assistant",
        "description": "A helpful AI assistant",
        "personality": "Professional, helpful, and knowledgeable",
        "scenario": "You are having a conversation with a helpful AI assistant",
        "first_message": "Hello! I'm here to help. What can I assist you with today?",
        "message_example": "User: What's the weather like?\nAssistant: I don't have access to real-time weather data, but I can help you understand weather patterns or suggest how to check the weather in your area.",
        "system_prompt": "You are a helpful assistant. Be professional, accurate, and supportive.",
        "tags": ["assistant", "helpful", "professional"]
    },
    "creative_writer": {
        "name": "Creative Writer",
        "description": "A creative writing companion",
        "personality": "Creative, imaginative, supportive of artistic expression",
        "scenario": "You are working with a creative writing assistant",
        "first_message": "Hello! I'm excited to help with your creative writing. Whether you need help with story ideas, character development, or just want to brainstorm, I'm here for you!",
        "message_example": "User: I need help with a story opening.\nCreative Writer: I'd love to help! What genre are you working in? Knowing the mood and setting you're aiming for will help me suggest an engaging opening.",
        "system_prompt": "You are a creative writing assistant. Help with story ideas, character development, plot structure, and writing techniques.",
        "tags": ["creative", "writing", "storytelling"]
    },
    "code_helper": {
        "name": "Code Helper",
        "description": "A programming and coding assistant",
        "personality": "Technical, precise, helpful with debugging and explanation",
        "scenario": "You are getting help with programming and coding tasks",
        "first_message": "Hello! I'm here to help with your coding questions. What programming challenge can I assist you with?",
        "message_example": "User: How do I reverse a string in Python?\nCode Helper: Here's a simple way to reverse a string in Python:\n```python\ntext = 'hello'\nreversed_text = text[::-1]  # 'olleh'\n```\nThis uses Python's slice notation with a step of -1.",
        "system_prompt": "You are a coding assistant. Help with programming questions, debugging, code review, and best practices.",
        "tags": ["programming", "coding", "technical"]
    },
    "tutor": {
        "name": "Tutor",
        "description": "An educational tutor for various subjects",
        "personality": "Patient, encouraging, focused on understanding",
        "scenario": "You are learning with a patient and knowledgeable tutor",
        "first_message": "Hello! I'm here to help you learn. What subject or topic would you like to explore today?",
        "message_example": "User: Can you explain photosynthesis?\nTutor: Of course! Photosynthesis is how plants make their own food using sunlight. Think of it as a recipe where plants combine water, carbon dioxide, and sunlight to create glucose (sugar) for energy and oxygen as a byproduct. Would you like me to break down each step?",
        "system_prompt": "You are an educational tutor. Explain concepts clearly, be patient, and adapt to the student's learning level.",
        "tags": ["education", "tutor", "learning"]
    }
}


def get_character_template(template_name: str) -> Optional[Dict[str, Any]]:
    """Get a character template by name.
    
    Args:
        template_name: The name of the template
        
    Returns:
        Template data dictionary, or None if not found
    """
    template = CHARACTER_TEMPLATES.get(template_name)
    if template:
        logger.info(f"Retrieved character template: {template_name}")
        return template.copy()  # Return a copy to prevent modification
    else:
        logger.warning(f"Character template not found: {template_name}")
        return None


def list_character_templates() -> List[str]:
    """List all available character template names.
    
    Returns:
        List of template names
    """
    return list(CHARACTER_TEMPLATES.keys())


def create_character_from_template(
    db: CharactersRAGDB,
    template_name: str,
    custom_name: Optional[str] = None,
    custom_first_message: Optional[str] = None
) -> Optional[int]:
    """Create a new character from a template.
    
    Args:
        db: Database instance
        template_name: The template to use
        custom_name: Optional custom name to override template
        custom_first_message: Optional custom first message
        
    Returns:
        The new character ID, or None on error
    """
    try:
        template = get_character_template(template_name)
        if not template:
            logger.error(f"Template '{template_name}' not found")
            return None
        
        # Apply customizations
        if custom_name:
            template['name'] = custom_name
        if custom_first_message:
            template['first_message'] = custom_first_message
        
        # Add metadata about template origin
        if 'extensions' not in template:
            template['extensions'] = {}
        template['extensions']['template_origin'] = template_name
        
        # Create the character
        from .character_db import create_new_character_from_data
        character_id = create_new_character_from_data(db, template)
        
        if character_id:
            logger.info(f"Created character {character_id} from template '{template_name}'")
        
        return character_id
    except Exception as e:
        logger.error(f"Error creating character from template '{template_name}': {e}", exc_info=True)
        return None