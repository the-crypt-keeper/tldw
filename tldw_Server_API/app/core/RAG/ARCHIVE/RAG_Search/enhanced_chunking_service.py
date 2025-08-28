"""
Enhanced chunking service with advanced features from external RAG pipeline.

This module provides sophisticated chunking capabilities including:
- Accurate character-level position tracking
- Hierarchical document structure preservation
- Parent-child chunk relationships
- Advanced text cleaning and processing
- Structure-aware chunking
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import re
import logging
from dataclasses import dataclass, field
from enum import Enum
import hashlib

from .chunking_service import ChunkingService, ChunkingError
from .table_serializer import TableProcessor, serialize_table

logger = logging.getLogger(__name__)


class ChunkType(Enum):
    """Types of chunks based on document structure."""
    PARAGRAPH = "paragraph"
    SECTION = "section"
    SUBSECTION = "subsection"
    LIST = "list"
    TABLE = "table"
    CODE_BLOCK = "code_block"
    QUOTE = "quote"
    FOOTNOTE = "footnote"
    HEADER = "header"
    TEXT = "text"  # Generic text


@dataclass
class StructuredChunk:
    """Enhanced chunk with structural information."""
    text: str
    start_char: int
    end_char: int
    chunk_index: int
    chunk_type: ChunkType
    level: int = 0  # Hierarchical level (0 = root)
    parent_index: Optional[int] = None
    children_indices: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def word_count(self) -> int:
        """Calculate word count."""
        return len(self.text.split())
    
    @property
    def char_count(self) -> int:
        """Calculate character count."""
        return len(self.text)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'text': self.text,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'chunk_index': self.chunk_index,
            'chunk_type': self.chunk_type.value,
            'level': self.level,
            'parent_index': self.parent_index,
            'children_indices': self.children_indices,
            'word_count': self.word_count,
            'char_count': self.char_count,
            'metadata': self.metadata
        }


class DocumentStructureParser:
    """Parse document structure to identify hierarchical elements."""
    
    # Patterns for different document structures
    PATTERNS = {
        'header_h1': re.compile(r'^#\s+(.+)$', re.MULTILINE),
        'header_h2': re.compile(r'^##\s+(.+)$', re.MULTILINE),
        'header_h3': re.compile(r'^###\s+(.+)$', re.MULTILINE),
        'header_h4': re.compile(r'^####\s+(.+)$', re.MULTILINE),
        'numbered_section': re.compile(r'^(\d+(?:\.\d+)*)\s+(.+)$', re.MULTILINE),
        'bullet_list': re.compile(r'^[\*\-\+]\s+(.+)$', re.MULTILINE),
        'numbered_list': re.compile(r'^\d+\.\s+(.+)$', re.MULTILINE),
        'code_block': re.compile(r'```[\s\S]*?```', re.MULTILINE),
        'quote_block': re.compile(r'^>\s+(.+)$', re.MULTILINE),
        'table_delimiter': re.compile(r'^\|.*\|$', re.MULTILINE),
        'footnote': re.compile(r'^\[\^(\d+)\]:\s+(.+)$', re.MULTILINE),
        'paragraph': re.compile(r'((?:(?!^#|\n\n|\*\s|\-\s|\+\s|\d+\.\s|>|```|\|).)+)', re.MULTILINE | re.DOTALL)
    }
    
    # PDF artifact patterns to clean
    ARTIFACT_PATTERNS = {
        'command_mapping': {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'period': '.', 'comma': ',', 'colon': ':', 'hyphen': '-',
            'percent': '%', 'dollar': '$', 'space': ' ', 'plus': '+',
            'minus': '-', 'slash': '/', 'asterisk': '*', 'lparen': '(',
            'rparen': ')', 'parenright': ')', 'parenleft': '('
        },
        'glyph_pattern': re.compile(r'glyph<[^>]*>'),
        'cap_pattern': re.compile(r'/([A-Z])\.cap'),
        'command_pattern': None  # Will be built from command_mapping
    }
    
    def __init__(self):
        """Initialize the parser."""
        # Build command pattern from mapping
        commands = "|".join(self.ARTIFACT_PATTERNS['command_mapping'].keys())
        self.ARTIFACT_PATTERNS['command_pattern'] = re.compile(
            rf"/({commands})(\.pl\.tnum|\.tnum\.pl|\.pl|\.tnum|\.case|\.sups)?"
        )
    
    def clean_text(self, text: str) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Clean text from PDF artifacts and formatting issues.
        
        Returns:
            Tuple of (cleaned_text, corrections_made)
        """
        corrections = []
        
        # Apply command replacements
        def replace_command(match):
            base_command = match.group(1)
            replacement = self.ARTIFACT_PATTERNS['command_mapping'].get(base_command, match.group(0))
            if replacement != match.group(0):
                corrections.append((match.group(0), replacement))
            return replacement
        
        text = self.ARTIFACT_PATTERNS['command_pattern'].sub(replace_command, text)
        
        # Remove glyph artifacts
        def replace_glyph(match):
            corrections.append((match.group(0), ''))
            return ''
        
        text = self.ARTIFACT_PATTERNS['glyph_pattern'].sub(replace_glyph, text)
        
        # Fix cap patterns
        def replace_cap(match):
            corrections.append((match.group(0), match.group(1)))
            return match.group(1)
        
        text = self.ARTIFACT_PATTERNS['cap_pattern'].sub(replace_cap, text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip(), corrections
    
    def parse_structure(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse document structure into hierarchical elements.
        
        Returns:
            List of structural elements with positions and hierarchy
        """
        elements = []
        
        # First, find all structural elements with their positions
        for pattern_name, pattern in self.PATTERNS.items():
            for match in pattern.finditer(text):
                element = {
                    'type': pattern_name,
                    'start': match.start(),
                    'end': match.end(),
                    'text': match.group(0),
                    'groups': match.groups() if match.groups() else []
                }
                
                # Determine hierarchical level
                if pattern_name.startswith('header_h'):
                    element['level'] = int(pattern_name[-1])
                elif pattern_name == 'numbered_section':
                    # Level based on section depth (1.2.3 = level 3)
                    element['level'] = len(match.group(1).split('.'))
                else:
                    element['level'] = 0
                
                elements.append(element)
        
        # Sort by position
        elements.sort(key=lambda x: x['start'])
        
        # Remove overlapping elements (keep the most specific)
        filtered_elements = []
        for i, elem in enumerate(elements):
            # Check if this element is contained within another
            is_contained = False
            for j, other in enumerate(elements):
                if i != j and other['start'] <= elem['start'] and other['end'] >= elem['end']:
                    # This element is contained in another
                    if elem['type'] != 'paragraph' or other['type'] == 'paragraph':
                        is_contained = True
                        break
            
            if not is_contained:
                filtered_elements.append(elem)
        
        return filtered_elements


class EnhancedChunkingService(ChunkingService):
    """
    Enhanced chunking service with advanced features.
    
    Features:
    - Accurate character-level position tracking
    - Hierarchical document structure preservation
    - Parent-child chunk relationships
    - PDF artifact cleaning
    - Structure-aware chunking
    - Table serialization support
    """
    
    def __init__(self):
        """Initialize enhanced chunking service."""
        super().__init__()
        self.structure_parser = DocumentStructureParser()
        self.table_processor = TableProcessor()
        logger.info("Initialized EnhancedChunkingService with structure parsing and table serialization")
    
    def chunk_text_with_structure(self,
                                 content: str,
                                 chunk_size: int = 400,
                                 chunk_overlap: int = 100,
                                 method: str = "hierarchical",
                                 preserve_structure: bool = True,
                                 clean_artifacts: bool = True,
                                 serialize_tables: bool = True) -> List[StructuredChunk]:
        """
        Enhanced chunking with structure preservation.
        
        Args:
            content: Text to chunk
            chunk_size: Target size of chunks
            chunk_overlap: Overlap between chunks
            method: Chunking method ("hierarchical", "semantic", "structural")
            preserve_structure: Whether to preserve document structure
            clean_artifacts: Whether to clean PDF artifacts
            serialize_tables: Whether to serialize tables for better understanding
            
        Returns:
            List of StructuredChunk objects
        """
        # Clean text if requested
        if clean_artifacts:
            content, corrections = self.structure_parser.clean_text(content)
            if corrections:
                logger.info(f"Cleaned {len(corrections)} text artifacts")
        
        # Process tables if requested
        table_metadata = []
        if serialize_tables:
            processed_text, table_metadata = self.table_processor.process_document_tables(content, serialize_method="hybrid")
            content = processed_text
            if table_metadata:
                logger.info(f"Serialized {len(table_metadata)} tables in document")
        
        # Parse document structure
        if preserve_structure and method in ["hierarchical", "structural"]:
            elements = self.structure_parser.parse_structure(content)
            logger.debug(f"Found {len(elements)} structural elements")
            
            if method == "hierarchical":
                return self._hierarchical_chunking(content, elements, chunk_size, chunk_overlap)
            elif method == "structural":
                return self._structural_chunking(content, elements, chunk_size, chunk_overlap)
        
        # Fall back to enhanced base chunking
        return self._enhanced_base_chunking(content, chunk_size, chunk_overlap, method)
    
    def _hierarchical_chunking(self,
                              content: str,
                              elements: List[Dict[str, Any]],
                              chunk_size: int,
                              chunk_overlap: int) -> List[StructuredChunk]:
        """
        Chunk text while preserving hierarchical structure.
        
        Creates parent-child relationships between chunks based on
        document structure (headers, sections, subsections).
        """
        chunks = []
        chunk_index = 0
        
        # Build hierarchy tree
        hierarchy_stack = []  # Stack of (level, chunk_index)
        
        # Add text before first element if exists
        if elements and elements[0]['start'] > 0:
            pre_text = content[:elements[0]['start']].strip()
            if pre_text:
                chunk = StructuredChunk(
                    text=pre_text,
                    start_char=0,
                    end_char=elements[0]['start'],
                    chunk_index=chunk_index,
                    chunk_type=ChunkType.TEXT,
                    level=0
                )
                chunks.append(chunk)
                chunk_index += 1
        
        # Process each structural element
        for i, elem in enumerate(elements):
            elem_type = elem['type']
            elem_level = elem['level']
            elem_start = elem['start']
            elem_end = elem['end']
            
            # Determine chunk type
            if elem_type.startswith('header'):
                chunk_type = ChunkType.HEADER
            elif elem_type.endswith('_list'):
                chunk_type = ChunkType.LIST
            elif elem_type == 'table_delimiter':
                chunk_type = ChunkType.TABLE
            elif elem_type == 'code_block':
                chunk_type = ChunkType.CODE_BLOCK
            elif elem_type == 'quote_block':
                chunk_type = ChunkType.QUOTE
            elif elem_type == 'footnote':
                chunk_type = ChunkType.FOOTNOTE
            else:
                chunk_type = ChunkType.TEXT
            
            # Update hierarchy stack
            while hierarchy_stack and hierarchy_stack[-1][0] >= elem_level:
                hierarchy_stack.pop()
            
            parent_index = hierarchy_stack[-1][1] if hierarchy_stack else None
            
            # Get content until next element or end
            if i + 1 < len(elements):
                content_end = elements[i + 1]['start']
            else:
                content_end = len(content)
            
            elem_content = content[elem_start:content_end].strip()
            
            # Check if we need to sub-chunk this element
            if len(elem_content.split()) > chunk_size:
                # Sub-chunk large elements
                sub_chunks = self._sub_chunk_element(
                    elem_content, elem_start, chunk_size, chunk_overlap,
                    chunk_type, elem_level, parent_index, chunk_index
                )
                
                for sub_chunk in sub_chunks:
                    chunks.append(sub_chunk)
                    
                    # Update parent's children
                    if parent_index is not None:
                        chunks[parent_index].children_indices.append(sub_chunk.chunk_index)
                    
                    chunk_index += 1
                
                # Add this element to hierarchy with last sub-chunk index
                if sub_chunks:
                    hierarchy_stack.append((elem_level, sub_chunks[-1].chunk_index))
            else:
                # Create single chunk for this element
                chunk = StructuredChunk(
                    text=elem_content,
                    start_char=elem_start,
                    end_char=content_end,
                    chunk_index=chunk_index,
                    chunk_type=chunk_type,
                    level=elem_level,
                    parent_index=parent_index,
                    metadata={'element_type': elem_type}
                )
                
                chunks.append(chunk)
                
                # Update parent's children
                if parent_index is not None:
                    chunks[parent_index].children_indices.append(chunk_index)
                
                # Add to hierarchy stack
                hierarchy_stack.append((elem_level, chunk_index))
                chunk_index += 1
        
        logger.info(f"Created {len(chunks)} hierarchical chunks")
        return chunks
    
    def _structural_chunking(self,
                           content: str,
                           elements: List[Dict[str, Any]],
                           chunk_size: int,
                           chunk_overlap: int) -> List[StructuredChunk]:
        """
        Chunk text based on structural boundaries.
        
        Respects document structure by preferring to break at
        structural boundaries (paragraphs, sections, etc.).
        """
        chunks = []
        chunk_index = 0
        current_chunk_parts = []
        current_chunk_start = 0
        current_word_count = 0
        
        # Process gaps between elements
        prev_end = 0
        
        for elem in elements:
            # Add text before this element
            if elem['start'] > prev_end:
                gap_text = content[prev_end:elem['start']].strip()
                if gap_text:
                    gap_words = gap_text.split()
                    
                    # Check if adding this would exceed chunk size
                    if current_word_count + len(gap_words) > chunk_size and current_chunk_parts:
                        # Finalize current chunk
                        chunk_text = ' '.join(current_chunk_parts)
                        chunk = StructuredChunk(
                            text=chunk_text,
                            start_char=current_chunk_start,
                            end_char=prev_end,
                            chunk_index=chunk_index,
                            chunk_type=ChunkType.TEXT,
                            level=0
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                        
                        # Start new chunk with overlap
                        if chunk_overlap > 0 and len(current_chunk_parts) > 0:
                            overlap_words = ' '.join(current_chunk_parts).split()[-chunk_overlap:]
                            current_chunk_parts = [' '.join(overlap_words)]
                            current_word_count = len(overlap_words)
                        else:
                            current_chunk_parts = []
                            current_word_count = 0
                            current_chunk_start = prev_end
                    
                    current_chunk_parts.append(gap_text)
                    current_word_count += len(gap_words)
            
            # Process the element itself
            elem_text = content[elem['start']:elem['end']].strip()
            elem_words = elem_text.split()
            
            # Determine if this element should start a new chunk
            should_break = False
            if elem['type'].startswith('header') or elem['type'] == 'numbered_section':
                should_break = True
            elif current_word_count + len(elem_words) > chunk_size * 1.5:  # Allow some flexibility
                should_break = True
            
            if should_break and current_chunk_parts:
                # Finalize current chunk
                chunk_text = ' '.join(current_chunk_parts)
                chunk = StructuredChunk(
                    text=chunk_text,
                    start_char=current_chunk_start,
                    end_char=elem['start'],
                    chunk_index=chunk_index,
                    chunk_type=ChunkType.TEXT,
                    level=0
                )
                chunks.append(chunk)
                chunk_index += 1
                
                current_chunk_parts = []
                current_word_count = 0
                current_chunk_start = elem['start']
            
            # Add element to current chunk
            current_chunk_parts.append(elem_text)
            current_word_count += len(elem_words)
            prev_end = elem['end']
        
        # Handle remaining text
        if prev_end < len(content):
            remaining_text = content[prev_end:].strip()
            if remaining_text:
                current_chunk_parts.append(remaining_text)
        
        # Finalize last chunk
        if current_chunk_parts:
            chunk_text = ' '.join(current_chunk_parts)
            chunk = StructuredChunk(
                text=chunk_text,
                start_char=current_chunk_start,
                end_char=len(content),
                chunk_index=chunk_index,
                chunk_type=ChunkType.TEXT,
                level=0
            )
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} structural chunks")
        return chunks
    
    def _enhanced_base_chunking(self,
                               content: str,
                               chunk_size: int,
                               chunk_overlap: int,
                               method: str) -> List[StructuredChunk]:
        """
        Enhanced version of base chunking with accurate position tracking.
        """
        # Use parent class chunking
        base_chunks = super().chunk_text(content, chunk_size, chunk_overlap, method)
        
        # Convert to StructuredChunk objects
        structured_chunks = []
        for i, chunk_data in enumerate(base_chunks):
            chunk = StructuredChunk(
                text=chunk_data['text'],
                start_char=chunk_data.get('start_char', 0),
                end_char=chunk_data.get('end_char', len(chunk_data['text'])),
                chunk_index=i,
                chunk_type=ChunkType.TEXT,
                level=0,
                metadata={
                    'method': method,
                    'word_count': chunk_data.get('word_count', len(chunk_data['text'].split()))
                }
            )
            structured_chunks.append(chunk)
        
        return structured_chunks
    
    def _sub_chunk_element(self,
                          text: str,
                          start_offset: int,
                          chunk_size: int,
                          chunk_overlap: int,
                          chunk_type: ChunkType,
                          level: int,
                          parent_index: Optional[int],
                          base_index: int) -> List[StructuredChunk]:
        """
        Sub-chunk a large element while preserving its metadata.
        """
        words = text.split()
        sub_chunks = []
        sub_index = 0
        
        step = max(1, chunk_size - chunk_overlap)
        
        for i in range(0, len(words), step):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            # Calculate character positions within the element
            if i == 0:
                rel_start = 0
            else:
                # Find the character position of the ith word
                prefix = ' '.join(words[:i])
                rel_start = len(prefix) + 1  # +1 for the space
            
            if i + len(chunk_words) >= len(words):
                rel_end = len(text)
            else:
                prefix = ' '.join(words[:i + len(chunk_words)])
                rel_end = len(prefix)
            
            sub_chunk = StructuredChunk(
                text=chunk_text,
                start_char=start_offset + rel_start,
                end_char=start_offset + rel_end,
                chunk_index=base_index + sub_index,
                chunk_type=chunk_type,
                level=level,
                parent_index=parent_index,
                metadata={
                    'is_sub_chunk': True,
                    'sub_chunk_index': sub_index,
                    'total_sub_chunks': None  # Will be set after loop
                }
            )
            sub_chunks.append(sub_chunk)
            sub_index += 1
        
        # Update total sub-chunks metadata
        for chunk in sub_chunks:
            chunk.metadata['total_sub_chunks'] = len(sub_chunks)
        
        return sub_chunks
    
    def chunk_with_parent_retrieval(self,
                                   content: str,
                                   chunk_size: int = 400,
                                   chunk_overlap: int = 100,
                                   parent_size_multiplier: int = 3) -> Dict[str, Any]:
        """
        Chunk text with parent document retrieval support.
        
        Creates smaller chunks for retrieval but stores references to
        larger parent chunks for context expansion during retrieval.
        
        Args:
            content: Text to chunk
            chunk_size: Size of retrieval chunks
            chunk_overlap: Overlap between chunks
            parent_size_multiplier: Parent chunks are this many times larger
            
        Returns:
            Dictionary with 'chunks' and 'parent_chunks'
        """
        # Create fine-grained chunks for retrieval
        retrieval_chunks = self.chunk_text_with_structure(
            content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            method="structural"
        )
        
        # Create larger parent chunks
        parent_chunk_size = chunk_size * parent_size_multiplier
        parent_chunks = self.chunk_text_with_structure(
            content,
            chunk_size=parent_chunk_size,
            chunk_overlap=chunk_overlap * parent_size_multiplier,
            method="structural"
        )
        
        # Map retrieval chunks to parent chunks
        for retrieval_chunk in retrieval_chunks:
            # Find parent chunk that contains this retrieval chunk
            for parent_idx, parent_chunk in enumerate(parent_chunks):
                if (parent_chunk.start_char <= retrieval_chunk.start_char and
                    parent_chunk.end_char >= retrieval_chunk.end_char):
                    retrieval_chunk.metadata['parent_chunk_index'] = parent_idx
                    break
        
        return {
            'chunks': [chunk.to_dict() for chunk in retrieval_chunks],
            'parent_chunks': [chunk.to_dict() for chunk in parent_chunks],
            'metadata': {
                'chunk_size': chunk_size,
                'parent_chunk_size': parent_chunk_size,
                'total_chunks': len(retrieval_chunks),
                'total_parent_chunks': len(parent_chunks)
            }
        }
    
    def serialize_table(self, table_text: str, format: str = "markdown") -> Dict[str, Any]:
        """
        Serialize table content for better semantic understanding.
        
        Args:
            table_text: Table content (markdown, CSV, etc.)
            format: Input format of the table
            
        Returns:
            Dictionary with original and serialized representations
        """
        # This is a simplified implementation
        # In production, you'd want more sophisticated table parsing
        
        serialized_blocks = []
        
        if format == "markdown":
            lines = table_text.strip().split('\n')
            if len(lines) >= 3:  # Header, separator, at least one row
                # Parse header
                headers = [h.strip() for h in lines[0].split('|') if h.strip()]
                
                # Parse rows
                for line in lines[2:]:  # Skip separator line
                    cells = [c.strip() for c in line.split('|') if c.strip()]
                    if len(cells) == len(headers):
                        # Create information block for this row
                        row_info = []
                        for header, cell in zip(headers, cells):
                            row_info.append(f"{header}: {cell}")
                        serialized_blocks.append({
                            "information_block": "; ".join(row_info)
                        })
        
        return {
            "markdown": table_text,
            "serialized": {
                "information_blocks": serialized_blocks
            }
        }


# Convenience function to maintain API compatibility
def create_enhanced_chunking_service() -> EnhancedChunkingService:
    """Create an instance of the enhanced chunking service."""
    return EnhancedChunkingService()