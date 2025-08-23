# strategies/json_xml.py
"""
JSON and XML chunking strategies.
Handles structured data formats while preserving structure and relationships.
"""

import json
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError
from typing import List, Dict, Any, Optional, Tuple, Generator
from loguru import logger

from ..base import BaseChunkingStrategy, ChunkResult, ChunkMetadata
from ..exceptions import InvalidInputError, ChunkingError


class JSONChunkingStrategy(BaseChunkingStrategy):
    """
    Chunks JSON data while preserving structure.
    Supports both arrays and objects with configurable chunking keys.
    """
    
    MAX_JSON_SIZE = 50_000_000  # 50MB limit for security
    
    def __init__(self, language: str = 'en'):
        """
        Initialize JSON chunking strategy.
        
        Args:
            language: Language code (not used for JSON but kept for consistency)
        """
        super().__init__(language)
        logger.debug("JSONChunkingStrategy initialized")
    
    def chunk(self,
              text: str,
              max_size: int,
              overlap: int = 0,
              **options) -> List[str]:
        """
        Chunk JSON data.
        
        Args:
            text: JSON string to chunk
            max_size: Maximum items/keys per chunk
            overlap: Number of items/keys to overlap
            **options: Additional options:
                - chunkable_key: Key to chunk in dict (default: 'data')
                - preserve_metadata: Keep non-chunked keys in each chunk
                - output_format: 'json' or 'text'
                
        Returns:
            List of JSON strings or text representations
        """
        if not self.validate_parameters(text, max_size, overlap):
            return []
        
        # Security check
        if len(text) > self.MAX_JSON_SIZE:
            raise InvalidInputError(
                f"JSON text size ({len(text)} bytes) exceeds maximum "
                f"allowed size ({self.MAX_JSON_SIZE} bytes)"
            )
        
        # Parse JSON
        try:
            json_data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON data: {e}")
            raise InvalidInputError(f"Invalid JSON data: {e}") from e
        
        # Get options
        output_format = options.get('output_format', 'json')
        
        # Chunk based on JSON type
        if isinstance(json_data, list):
            chunks = self._chunk_json_list(json_data, max_size, overlap, **options)
        elif isinstance(json_data, dict):
            chunks = self._chunk_json_dict(json_data, max_size, overlap, **options)
        else:
            raise InvalidInputError(
                "JSON must be a top-level array or object. "
                f"Got: {type(json_data).__name__}"
            )
        
        # Convert to requested format
        if output_format == 'json':
            return [json.dumps(chunk, indent=2) for chunk in chunks]
        else:
            return [self._json_to_text(chunk) for chunk in chunks]
    
    def _chunk_json_list(self, 
                        json_list: List[Any],
                        max_size: int,
                        overlap: int,
                        **options) -> List[List[Any]]:
        """
        Chunk a JSON array.
        
        Args:
            json_list: List to chunk
            max_size: Maximum items per chunk
            overlap: Number of items to overlap
            **options: Additional options
            
        Returns:
            List of chunked arrays
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        
        if overlap >= max_size:
            logger.warning(f"Overlap {overlap} >= max_size {max_size}. Setting to 0.")
            overlap = 0
        
        chunks = []
        step = max(1, max_size - overlap)
        
        for i in range(0, len(json_list), step):
            chunk = json_list[i:i + max_size]
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_json_dict(self,
                        json_dict: Dict[str, Any],
                        max_size: int,
                        overlap: int,
                        **options) -> List[Dict[str, Any]]:
        """
        Chunk a JSON object.
        
        Args:
            json_dict: Dictionary to chunk
            max_size: Maximum keys per chunk
            overlap: Number of keys to overlap
            **options: Additional options including 'chunkable_key'
            
        Returns:
            List of chunked dictionaries
        """
        chunkable_key = options.get('chunkable_key', 'data')
        preserve_metadata = options.get('preserve_metadata', True)
        
        # Check if chunkable key exists and is dict or list
        if chunkable_key not in json_dict:
            # If no chunkable key, chunk the entire dict by keys
            return self._chunk_dict_by_keys(json_dict, max_size, overlap)
        
        chunkable_data = json_dict[chunkable_key]
        
        if isinstance(chunkable_data, list):
            # Chunk the list within the dict
            chunked_lists = self._chunk_json_list(chunkable_data, max_size, overlap)
            
            chunks = []
            for chunk_list in chunked_lists:
                if preserve_metadata:
                    # Keep other keys
                    new_chunk = json_dict.copy()
                    new_chunk[chunkable_key] = chunk_list
                else:
                    new_chunk = {chunkable_key: chunk_list}
                chunks.append(new_chunk)
            
            return chunks
            
        elif isinstance(chunkable_data, dict):
            # Chunk the nested dict
            chunked_dicts = self._chunk_dict_by_keys(chunkable_data, max_size, overlap)
            
            chunks = []
            for chunk_dict in chunked_dicts:
                if preserve_metadata:
                    # Keep other keys
                    new_chunk = json_dict.copy()
                    new_chunk[chunkable_key] = chunk_dict
                else:
                    new_chunk = {chunkable_key: chunk_dict}
                chunks.append(new_chunk)
            
            return chunks
        else:
            # Can't chunk this key's value
            logger.warning(f"Chunkable key '{chunkable_key}' is not a list or dict")
            return [json_dict]
    
    def _chunk_dict_by_keys(self,
                           data_dict: Dict[str, Any],
                           max_size: int,
                           overlap: int) -> List[Dict[str, Any]]:
        """
        Chunk a dictionary by its keys.
        
        Args:
            data_dict: Dictionary to chunk
            max_size: Maximum keys per chunk
            overlap: Number of keys to overlap
            
        Returns:
            List of chunked dictionaries
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        
        if overlap >= max_size:
            logger.warning(f"Overlap {overlap} >= max_size {max_size}. Setting to 0.")
            overlap = 0
        
        all_keys = list(data_dict.keys())
        chunks = []
        step = max(1, max_size - overlap)
        
        for i in range(0, len(all_keys), step):
            chunk_keys = all_keys[i:i + max_size]
            chunk_dict = {k: data_dict[k] for k in chunk_keys}
            chunks.append(chunk_dict)
        
        return chunks
    
    def _json_to_text(self, json_obj: Any) -> str:
        """
        Convert JSON object to human-readable text.
        
        Args:
            json_obj: JSON object to convert
            
        Returns:
            Text representation
        """
        if isinstance(json_obj, dict):
            lines = []
            for key, value in json_obj.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{key}: {json.dumps(value, indent=2)}")
                else:
                    lines.append(f"{key}: {value}")
            return '\n'.join(lines)
        elif isinstance(json_obj, list):
            return '\n'.join([str(item) for item in json_obj])
        else:
            return str(json_obj)


class XMLChunkingStrategy(BaseChunkingStrategy):
    """
    Chunks XML data while preserving structure and hierarchy.
    """
    
    MAX_XML_SIZE = 50_000_000  # 50MB limit for security
    
    def __init__(self, language: str = 'en'):
        """
        Initialize XML chunking strategy.
        
        Args:
            language: Language code for text content
        """
        super().__init__(language)
        logger.debug("XMLChunkingStrategy initialized")
    
    def chunk(self,
              text: str,
              max_size: int,
              overlap: int = 0,
              **options) -> List[str]:
        """
        Chunk XML data.
        
        Args:
            text: XML string to chunk
            max_size: Maximum words in content per chunk
            overlap: Number of XML elements to overlap
            **options: Additional options:
                - preserve_structure: Keep parent elements
                - output_format: 'xml' or 'text'
                - include_paths: Include element paths in output
                
        Returns:
            List of XML strings or text representations
        """
        if not self.validate_parameters(text, max_size, overlap):
            return []
        
        # Security check
        if len(text) > self.MAX_XML_SIZE:
            raise InvalidInputError(
                f"XML text size ({len(text)} bytes) exceeds maximum "
                f"allowed size ({self.MAX_XML_SIZE} bytes)"
            )
        
        # Parse XML with security hardening against XXE attacks
        try:
            # Try to use defusedxml if available (most secure)
            try:
                import defusedxml.ElementTree as DefusedET
                import defusedxml.common
                
                # Additional pre-check for DOCTYPE SYSTEM which might not always be caught
                if 'DOCTYPE' in text and 'SYSTEM' in text:
                    logger.error("XML contains DOCTYPE SYSTEM declaration")
                    raise InvalidInputError("XML contains DOCTYPE SYSTEM declaration which is not allowed for security reasons")
                
                try:
                    root = DefusedET.fromstring(text)
                    logger.debug("Using defusedxml for secure XML parsing")
                except (defusedxml.common.EntitiesForbidden, 
                        defusedxml.common.ExternalReferenceForbidden,
                        defusedxml.common.DTDForbidden,
                        defusedxml.common.NotSupportedError) as e:
                    logger.error(f"Blocked potential XXE attack: {e}")
                    raise InvalidInputError(f"XML contains forbidden constructs (potential XXE attack): {e}")
            except ImportError:
                # Fall back to standard library with security measures
                # Create parser with entity resolution disabled
                parser = ET.XMLParser()
                parser.entity = {}  # Clear entity definitions
                parser.default = lambda x: None  # Ignore undefined entities
                
                # Pre-check for dangerous XML constructs
                if '<!DOCTYPE' in text or '<!ENTITY' in text or 'SYSTEM' in text:
                    raise InvalidInputError(
                        "XML contains potentially dangerous DTD/Entity declarations. "
                        "These are not allowed for security reasons."
                    )
                
                # Additional check for external references
                if any(x in text.lower() for x in ['file://', 'http://', 'https://', 'ftp://']):
                    raise InvalidInputError(
                        "XML contains external references which are not allowed for security reasons."
                    )
                
                root = ET.fromstring(text, parser=parser)
                logger.debug("Using standard xml.etree with security checks")
        except ParseError as e:
            logger.error(f"Invalid XML data: {e}")
            raise InvalidInputError(f"Invalid XML data: {e}") from e
        
        # Get options
        output_format = options.get('output_format', 'text')
        include_paths = options.get('include_paths', True)
        
        # Extract XML structure
        elements = self._extract_xml_elements(root)
        
        if not elements:
            return []
        
        # Chunk elements
        chunks = self._chunk_xml_elements(elements, max_size, overlap, **options)
        
        # Convert to requested format
        result = []
        for chunk in chunks:
            if output_format == 'xml':
                # Reconstruct XML from elements
                result.append(self._elements_to_xml(chunk, root.tag, root.attrib))
            else:
                # Convert to text
                result.append(self._elements_to_text(chunk, include_paths))
        
        return result
    
    def _extract_xml_elements(self, 
                             element: ET.Element,
                             path: str = "") -> List[Tuple[str, str, ET.Element]]:
        """
        Recursively extract XML elements with paths.
        
        Args:
            element: XML element to process
            path: Current path in XML tree
            
        Returns:
            List of (path, text_content, element) tuples
        """
        results = []
        
        # Build current path
        current_path = f"{path}/{element.tag}" if path else element.tag
        
        # Get element text content
        text_content = ""
        if element.text:
            text_content = element.text.strip()
        
        # Add current element if it has content
        if text_content:
            results.append((current_path, text_content, element))
        
        # Process children
        for child in element:
            results.extend(self._extract_xml_elements(child, current_path))
            # Also get tail text
            if child.tail:
                tail_text = child.tail.strip()
                if tail_text:
                    results.append((f"{current_path}[tail]", tail_text, element))
        
        return results
    
    def _chunk_xml_elements(self,
                           elements: List[Tuple[str, str, ET.Element]],
                           max_size: int,
                           overlap: int,
                           **options) -> List[List[Tuple[str, str, ET.Element]]]:
        """
        Chunk XML elements by content size.
        
        Args:
            elements: List of (path, content, element) tuples
            max_size: Maximum words per chunk
            overlap: Number of elements to overlap
            **options: Additional options
            
        Returns:
            List of chunked element lists
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        
        if overlap >= len(elements) and len(elements) > 0:
            logger.warning(f"Overlap {overlap} >= total elements. Setting to 0.")
            overlap = 0
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for i, (path, content, elem) in enumerate(elements):
            word_count = len(content.split())
            
            # Check if adding this element exceeds max_size
            if current_word_count + word_count > max_size and current_chunk:
                # Save current chunk
                chunks.append(current_chunk)
                
                # Handle overlap
                if overlap > 0 and len(current_chunk) > overlap:
                    # Keep last 'overlap' elements
                    current_chunk = current_chunk[-overlap:]
                    current_word_count = sum(len(c.split()) for _, c, _ in current_chunk)
                else:
                    current_chunk = []
                    current_word_count = 0
            
            # Add element to current chunk
            current_chunk.append((path, content, elem))
            current_word_count += word_count
        
        # Add remaining elements
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _elements_to_text(self,
                         elements: List[Tuple[str, str, ET.Element]],
                         include_paths: bool) -> str:
        """
        Convert XML elements to text representation.
        
        Args:
            elements: List of (path, content, element) tuples
            include_paths: Whether to include element paths
            
        Returns:
            Text representation
        """
        lines = []
        
        for path, content, _ in elements:
            if include_paths:
                lines.append(f"[{path}]: {content}")
            else:
                lines.append(content)
        
        return '\n'.join(lines)
    
    def _elements_to_xml(self,
                        elements: List[Tuple[str, str, ET.Element]],
                        root_tag: str,
                        root_attrib: Dict[str, str]) -> str:
        """
        Reconstruct XML from elements.
        
        Args:
            elements: List of (path, content, element) tuples
            root_tag: Root element tag
            root_attrib: Root element attributes
            
        Returns:
            XML string
        """
        # Create new root
        new_root = ET.Element(root_tag, root_attrib)
        
        # Track added elements to avoid duplicates
        added_elements = set()
        
        for path, content, elem in elements:
            if elem not in added_elements:
                # Simplified: just add as child of root
                # In production, would reconstruct full hierarchy
                new_elem = ET.SubElement(new_root, elem.tag, elem.attrib)
                new_elem.text = content
                added_elements.add(elem)
        
        return ET.tostring(new_root, encoding='unicode')