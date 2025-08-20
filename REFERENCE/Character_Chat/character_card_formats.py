# character_card_formats.py
# Description: Character card format detection and parsing for various AI chat platforms
#
"""
Character Card Format Support
----------------------------

This module provides parsers for various character card formats used by different
AI chat platforms, converting them all to the standard V2 format for internal use.

Supported formats:
- V1 (TavernAI) - Original flat JSON format
- V2 (Standard) - Current hierarchical format with spec_version
- Agnai/AgnAIstic - JSON with avatar, scenario, persona
- CharacterAI - Export format with participant__name
- KoboldAI - Extended V1 with memory and author's note
- TextGeneration WebUI - YAML-based format
- Chub.ai - Extended V2 with tags and ratings
"""

import json
import re
import yaml
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from loguru import logger


class CharacterCardFormatDetector:
    """Detects and parses various character card formats."""
    
    def __init__(self):
        """Initialize the format detector with known format patterns."""
        self.format_patterns = {
            'v2': self._is_v2_format,
            'v1': self._is_v1_format,
            'agnai': self._is_agnai_format,
            'characterai': self._is_characterai_format,
            'koboldai': self._is_koboldai_format,
            'textgen': self._is_textgen_format,
            'chub': self._is_chub_format,
        }
        
        self.parsers = {
            'v2': self._parse_v2_format,
            'v1': self._parse_v1_format,
            'agnai': self._parse_agnai_format,
            'characterai': self._parse_characterai_format,
            'koboldai': self._parse_koboldai_format,
            'textgen': self._parse_textgen_format,
            'chub': self._parse_chub_format,
        }
    
    def detect_and_parse(self, data: Any) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Detect the format and parse the character card.
        
        Args:
            data: Character card data (dict, string, or bytes)
            
        Returns:
            Tuple of (parsed_card, format_name) or (None, 'unknown')
        """
        # Handle string data (could be JSON or YAML)
        if isinstance(data, (str, bytes)):
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            
            # Try to parse as YAML first (which also parses JSON)
            try:
                parsed_data = yaml.safe_load(data)
                if parsed_data:
                    data = parsed_data
            except:
                # Try JSON if YAML fails
                try:
                    data = json.loads(data)
                except:
                    logger.error("Failed to parse character card data as JSON or YAML")
                    return None, 'unknown'
        
        if not isinstance(data, dict):
            logger.error(f"Character card data must be a dictionary, got {type(data)}")
            return None, 'unknown'
        
        # Detect format
        for format_name, detector in self.format_patterns.items():
            if detector(data):
                logger.info(f"Detected character card format: {format_name}")
                try:
                    parsed = self.parsers[format_name](data)
                    return parsed, format_name
                except Exception as e:
                    logger.error(f"Failed to parse {format_name} format: {e}")
                    continue
        
        # Try generic parsing as last resort
        logger.warning("Unknown character card format, attempting generic parsing")
        try:
            parsed = self._parse_generic_format(data)
            return parsed, 'generic'
        except Exception as e:
            logger.error(f"Generic parsing failed: {e}")
            return None, 'unknown'
    
    # Format detection methods
    def _is_v2_format(self, data: Dict[str, Any]) -> bool:
        """Check if data is V2 character card format."""
        return (
            data.get('spec') == 'chara_card_v2' or
            data.get('spec_version') == '2.0' or
            ('data' in data and isinstance(data['data'], dict))
        )
    
    def _is_v1_format(self, data: Dict[str, Any]) -> bool:
        """Check if data is V1 character card format."""
        # V1 has these fields at root level
        v1_fields = {'name', 'description', 'personality', 'first_mes', 'mes_example'}
        return (
            'name' in data and 
            'description' in data and
            'spec' not in data and
            'spec_version' not in data and
            len(v1_fields.intersection(data.keys())) >= 3
        )
    
    def _is_agnai_format(self, data: Dict[str, Any]) -> bool:
        """Check if data is Agnai/AgnAIstic format."""
        agnai_fields = {'avatar', 'persona', 'scenario', 'greeting'}
        return (
            len(agnai_fields.intersection(data.keys())) >= 2 or
            data.get('kind') == 'character' or
            '_id' in data and 'userId' in data
        )
    
    def _is_characterai_format(self, data: Dict[str, Any]) -> bool:
        """Check if data is CharacterAI export format."""
        return (
            'participant__name' in data or
            'external_id' in data or
            ('info' in data and 'character' in data['info']) or
            'greeting' in data and 'categories' in data
        )
    
    def _is_koboldai_format(self, data: Dict[str, Any]) -> bool:
        """Check if data is KoboldAI format."""
        return (
            'memory' in data or
            'authors_note' in data or
            ('world_info' in data and isinstance(data['world_info'], dict))
        )
    
    def _is_textgen_format(self, data: Dict[str, Any]) -> bool:
        """Check if data is TextGeneration WebUI format."""
        return (
            'char_name' in data or
            'char_persona' in data or
            'char_greeting' in data or
            'context' in data and 'greeting' in data
        )
    
    def _is_chub_format(self, data: Dict[str, Any]) -> bool:
        """Check if data is Chub.ai format."""
        return (
            data.get('spec') == 'chara_card_v2' and
            ('chub' in data or 'tags' in data or 'rating' in data)
        )
    
    # Parsing methods
    def _parse_v2_format(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse V2 format (already standard, just validate)."""
        char_data = data.get('data', data)
        
        return {
            'spec': 'chara_card_v2',
            'spec_version': '2.0',
            'data': {
                'name': char_data.get('name', 'Unknown'),
                'description': char_data.get('description', ''),
                'personality': char_data.get('personality', ''),
                'scenario': char_data.get('scenario', ''),
                'first_mes': char_data.get('first_mes', ''),
                'mes_example': char_data.get('mes_example', ''),
                'creator_notes': char_data.get('creator_notes', ''),
                'system_prompt': char_data.get('system_prompt', ''),
                'post_history_instructions': char_data.get('post_history_instructions', ''),
                'alternate_greetings': char_data.get('alternate_greetings', []),
                'character_book': char_data.get('character_book'),
                'tags': char_data.get('tags', []),
                'creator': char_data.get('creator', ''),
                'character_version': char_data.get('character_version', ''),
                'extensions': char_data.get('extensions', {})
            }
        }
    
    def _parse_v1_format(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse V1 format to V2."""
        return {
            'spec': 'chara_card_v2',
            'spec_version': '2.0',
            'data': {
                'name': data.get('name', 'Unknown'),
                'description': data.get('description', ''),
                'personality': data.get('personality', ''),
                'scenario': data.get('scenario', ''),
                'first_mes': data.get('first_mes', ''),
                'mes_example': data.get('mes_example', ''),
                'creator_notes': data.get('creator_notes', ''),
                'system_prompt': data.get('system_prompt', ''),
                'post_history_instructions': data.get('post_history_instructions', ''),
                'alternate_greetings': data.get('alternate_greetings', []),
                'character_book': data.get('character_book'),
                'tags': data.get('tags', []),
                'creator': data.get('creator', ''),
                'character_version': data.get('character_version', ''),
                'extensions': {}
            }
        }
    
    def _parse_agnai_format(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Agnai format to V2."""
        # Extract personality from persona
        persona = data.get('persona', data.get('personality', ''))
        personality = persona
        
        # Agnai often uses attributes/traits
        if 'attributes' in data:
            attrs = data['attributes']
            if isinstance(attrs, dict):
                personality = '\n'.join([f"{k}: {v}" for k, v in attrs.items()])
        
        return {
            'spec': 'chara_card_v2',
            'spec_version': '2.0',
            'data': {
                'name': data.get('name', 'Unknown'),
                'description': data.get('description', data.get('scenario', '')),
                'personality': personality,
                'scenario': data.get('scenario', ''),
                'first_mes': data.get('greeting', data.get('first_mes', '')),
                'mes_example': data.get('sampleChat', data.get('mes_example', '')),
                'creator_notes': data.get('notes', ''),
                'system_prompt': data.get('systemPrompt', ''),
                'post_history_instructions': data.get('postHistoryInstructions', ''),
                'alternate_greetings': data.get('alternateGreetings', []),
                'character_book': None,
                'tags': data.get('tags', []),
                'creator': data.get('creator', data.get('userId', '')),
                'character_version': data.get('version', ''),
                'extensions': {
                    'agnai_avatar': data.get('avatar'),
                    'agnai_voice': data.get('voice'),
                    'agnai_culture': data.get('culture')
                }
            }
        }
    
    def _parse_characterai_format(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse CharacterAI format to V2."""
        # Handle different CAI export formats
        name = (
            data.get('participant__name') or
            data.get('name') or
            data.get('info', {}).get('character', {}).get('name') or
            'Unknown'
        )
        
        description = (
            data.get('description') or
            data.get('info', {}).get('character', {}).get('description') or
            ''
        )
        
        greeting = (
            data.get('greeting') or
            data.get('info', {}).get('character', {}).get('greeting') or
            ''
        )
        
        return {
            'spec': 'chara_card_v2',
            'spec_version': '2.0',
            'data': {
                'name': name,
                'description': description,
                'personality': data.get('personality', data.get('definition', '')),
                'scenario': '',
                'first_mes': greeting,
                'mes_example': data.get('example_conversations', ''),
                'creator_notes': '',
                'system_prompt': '',
                'post_history_instructions': '',
                'alternate_greetings': [],
                'character_book': None,
                'tags': data.get('categories', []),
                'creator': data.get('user__username', ''),
                'character_version': '',
                'extensions': {
                    'cai_external_id': data.get('external_id'),
                    'cai_visibility': data.get('visibility'),
                    'cai_interactions': data.get('participant__num_interactions', 0)
                }
            }
        }
    
    def _parse_koboldai_format(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse KoboldAI format to V2."""
        # KoboldAI is similar to V1 but with additional fields
        v2_card = self._parse_v1_format(data)
        
        # Add KoboldAI specific fields
        v2_card['data']['extensions']['kobold_memory'] = data.get('memory', '')
        v2_card['data']['extensions']['kobold_authors_note'] = data.get('authors_note', '')
        v2_card['data']['extensions']['kobold_world_info'] = data.get('world_info', {})
        
        return v2_card
    
    def _parse_textgen_format(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse TextGeneration WebUI format to V2."""
        return {
            'spec': 'chara_card_v2',
            'spec_version': '2.0',
            'data': {
                'name': data.get('char_name', data.get('name', 'Unknown')),
                'description': data.get('char_persona', data.get('description', '')),
                'personality': data.get('char_persona', data.get('personality', '')),
                'scenario': data.get('world_scenario', data.get('scenario', '')),
                'first_mes': data.get('char_greeting', data.get('greeting', '')),
                'mes_example': data.get('example_dialogue', ''),
                'creator_notes': '',
                'system_prompt': data.get('context', ''),
                'post_history_instructions': '',
                'alternate_greetings': [],
                'character_book': None,
                'tags': [],
                'creator': '',
                'character_version': '',
                'extensions': {
                    'textgen_context': data.get('context'),
                    'textgen_instruct': data.get('instruction_template')
                }
            }
        }
    
    def _parse_chub_format(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Chub.ai format to V2."""
        # Chub is already V2, but may have additional fields
        v2_card = self._parse_v2_format(data)
        
        # Add Chub-specific fields
        if 'chub' in data:
            v2_card['data']['extensions']['chub'] = data['chub']
        if 'rating' in data:
            v2_card['data']['extensions']['rating'] = data['rating']
        if 'version_history' in data:
            v2_card['data']['extensions']['version_history'] = data['version_history']
        
        return v2_card
    
    def _parse_generic_format(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generic parser for unknown formats."""
        # Try to map common field names
        name_fields = ['name', 'char_name', 'character_name', 'title']
        desc_fields = ['description', 'desc', 'summary', 'about']
        personality_fields = ['personality', 'persona', 'traits', 'attributes']
        greeting_fields = ['greeting', 'first_mes', 'first_message', 'intro']
        example_fields = ['mes_example', 'example_dialogue', 'examples', 'sample_messages']
        
        def find_field(field_list):
            for field in field_list:
                if field in data:
                    return data[field]
            return ''
        
        return {
            'spec': 'chara_card_v2',
            'spec_version': '2.0',
            'data': {
                'name': find_field(name_fields) or 'Unknown',
                'description': find_field(desc_fields),
                'personality': find_field(personality_fields),
                'scenario': data.get('scenario', ''),
                'first_mes': find_field(greeting_fields),
                'mes_example': find_field(example_fields),
                'creator_notes': '',
                'system_prompt': '',
                'post_history_instructions': '',
                'alternate_greetings': [],
                'character_book': None,
                'tags': data.get('tags', []),
                'creator': data.get('creator', ''),
                'character_version': '',
                'extensions': {
                    'original_format': 'unknown',
                    'original_fields': list(data.keys())
                }
            }
        }


# Convenience function
def detect_and_parse_character_card(data: Any) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Detect and parse a character card from various formats.
    
    Args:
        data: Character card data in any supported format
        
    Returns:
        Tuple of (parsed_v2_card, format_name) or (None, 'unknown')
    """
    detector = CharacterCardFormatDetector()
    return detector.detect_and_parse(data)