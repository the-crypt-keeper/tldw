"""
Enhanced chunking service with structure preservation and smart boundaries.

This module provides advanced chunking capabilities including:
- Structure-aware chunking (headers, lists, tables)
- PDF artifact cleaning
- Smart boundary detection
- Code block and table preservation
- Advanced table serialization for improved semantic understanding
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

from loguru import logger

# Import table serialization functionality
from .table_serialization import TableProcessor, TableFormat


@dataclass
class Chunk:
    """Represents a document chunk with metadata."""
    id: str
    content: str
    start_char: int
    end_char: int
    chunk_index: int
    metadata: Dict[str, Any]
    parent_id: Optional[str] = None
    chunk_type: str = "text"  # "text", "code", "table", "header", etc.


class ChunkingStrategy(ABC):
    """Base class for chunking strategies."""
    
    @abstractmethod
    def chunk(self, text: str, chunk_size: int, overlap: int) -> List[Chunk]:
        """
        Chunk text into smaller pieces.
        
        Args:
            text: Text to chunk
            chunk_size: Target size for chunks
            overlap: Overlap between chunks
            
        Returns:
            List of chunks
        """
        pass


class EnhancedChunkingService:
    """
    Enhanced chunking service with multiple strategies.
    
    Features:
    - Structure preservation (markdown, code, tables)
    - Smart boundary detection
    - PDF artifact cleaning
    - Parent-child chunk relationships
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize enhanced chunking service.
        
        Args:
            config: Chunking configuration
        """
        self.config = config or {}
        
        # Configuration
        self.preserve_structure = self.config.get('preserve_structure', True)
        self.clean_pdf = self.config.get('clean_pdf_artifacts', True)
        self.preserve_tables = self.config.get('preserve_tables', True)
        self.preserve_code_blocks = self.config.get('preserve_code_blocks', True)
        self.use_sentence_boundaries = self.config.get('use_sentence_boundaries', True)
        self.min_chunk_size = self.config.get('min_chunk_size', 100)
        self.max_chunk_size = self.config.get('max_chunk_size', 1000)
        
        # Table serialization configuration
        self.table_serialize_method = self.config.get('table_serialize_method', 'hybrid')
        self.table_processor = TableProcessor(self.table_serialize_method)
        
        logger.info(f"Initialized EnhancedChunkingService with table serialization: {self.table_serialize_method}")
    
    def chunk_text(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 128,
        doc_id: Optional[str] = None
    ) -> List[Chunk]:
        """
        Chunk text with enhanced strategies.
        
        Args:
            text: Text to chunk
            chunk_size: Target chunk size
            overlap: Overlap between chunks
            doc_id: Document ID for parent tracking
            
        Returns:
            List of chunks
        """
        # Clean PDF artifacts if needed
        if self.clean_pdf:
            text = self._clean_pdf_artifacts(text)
        
        # Extract special blocks first
        text, code_blocks = self._extract_code_blocks(text)
        text, tables = self._extract_tables(text)
        
        # Perform structure-aware chunking
        if self.preserve_structure:
            chunks = self._structure_aware_chunking(text, chunk_size, overlap)
        else:
            chunks = self._simple_chunking(text, chunk_size, overlap)
        
        # Add special blocks as separate chunks
        chunks.extend(self._create_code_chunks(code_blocks, len(chunks)))
        chunks.extend(self._create_table_chunks(tables, len(chunks)))
        
        # Set parent relationships if doc_id provided
        if doc_id:
            for chunk in chunks:
                chunk.parent_id = doc_id
        
        logger.debug(f"Created {len(chunks)} chunks from {len(text)} characters")
        
        return chunks
    
    def _clean_pdf_artifacts(self, text: str) -> str:
        """
        Clean common PDF extraction artifacts.
        
        Args:
            text: Text potentially containing PDF artifacts
            
        Returns:
            Cleaned text
        """
        # Remove page numbers (various formats)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'Page \d+ of \d+', '', text)
        
        # Fix hyphenation at line breaks
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Fix common OCR errors
        text = text.replace('ﬁ', 'fi')
        text = text.replace('ﬂ', 'fl')
        text = text.replace('ﬀ', 'ff')
        
        # Remove headers/footers (heuristic)
        lines = text.split('\n')
        cleaned_lines = []
        for i, line in enumerate(lines):
            # Skip likely headers/footers (short, repeated lines)
            if len(line) < 50 and i > 0 and i < len(lines) - 1:
                if line in lines[:i] or line in lines[i+1:]:
                    continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _extract_code_blocks(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extract code blocks from text.
        
        Args:
            text: Text containing code blocks
            
        Returns:
            Tuple of (text with placeholders, list of code blocks)
        """
        code_blocks = []
        
        # Match markdown code blocks
        pattern = r'```[\w]*\n(.*?)\n```'
        matches = list(re.finditer(pattern, text, re.DOTALL))
        
        for i, match in enumerate(reversed(matches)):
            code_id = f"__CODE_BLOCK_{len(matches) - i - 1}__"
            code_blocks.append({
                'id': code_id,
                'content': match.group(0),
                'start': match.start(),
                'end': match.end()
            })
            text = text[:match.start()] + code_id + text[match.end():]
        
        return text, code_blocks
    
    def _extract_tables(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extract and serialize tables from text.
        
        Args:
            text: Text containing tables
            
        Returns:
            Tuple of (text with placeholders and serialization, list of tables)
        """
        tables = []
        
        # Use advanced table detection
        table_pattern = re.compile(
            r'((?:^\|[^\n]+\|$\n)+)', 
            re.MULTILINE
        )
        
        processed_text = text
        offset = 0
        
        for match in table_pattern.finditer(text):
            try:
                table_text = match.group(0)
                
                # Process table with advanced serialization
                serialized_data = self.table_processor.process_table(table_text, TableFormat.MARKDOWN)
                
                # Create placeholder and serialized content
                table_id = f"__TABLE_{len(tables)}__"
                
                # Build replacement text with both placeholder and serialization
                replacement_parts = [table_id]
                
                # Add serialized search text for better retrieval
                if 'search_text' in serialized_data:
                    replacement_parts.append(f"\n[Table Content: {serialized_data['search_text'][:200]}...]")
                
                replacement_text = '\n'.join(replacement_parts)
                
                # Calculate positions
                start = match.start() + offset
                end = match.end() + offset
                
                # Store table information
                tables.append({
                    'id': table_id,
                    'content': table_text,
                    'serialized': serialized_data,
                    'start_pos': start,
                    'end_pos': end
                })
                
                # Replace in text
                processed_text = processed_text[:start] + replacement_text + processed_text[end:]
                offset += len(replacement_text) - len(table_text)
                
                logger.debug(f"Extracted and serialized table with {serialized_data.get('metadata', {}).get('num_rows', 0)} rows")
                
            except Exception as e:
                logger.warning(f"Failed to process table: {e}")
                # Fallback to simple extraction
                table_id = f"__TABLE_{len(tables)}__"
                tables.append({
                    'id': table_id,
                    'content': match.group(0),
                    'serialized': None,
                    'start_pos': match.start(),
                    'end_pos': match.end()
                })
                processed_text = processed_text[:match.start() + offset] + table_id + processed_text[match.end() + offset:]
                offset += len(table_id) - len(match.group(0))
        
        return processed_text, tables
    
    def _structure_aware_chunking(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[Chunk]:
        """
        Perform structure-aware chunking.
        
        Args:
            text: Text to chunk
            chunk_size: Target chunk size
            overlap: Overlap between chunks
            
        Returns:
            List of chunks
        """
        chunks = []
        
        # Split by headers first (markdown style)
        sections = self._split_by_headers(text)
        
        chunk_index = 0
        current_position = 0
        
        for section in sections:
            section_text = section['text']
            section_type = section['type']
            
            if len(section_text) <= chunk_size:
                # Section fits in one chunk
                chunks.append(Chunk(
                    id=f"chunk_{chunk_index}",
                    content=section_text,
                    start_char=current_position,
                    end_char=current_position + len(section_text),
                    chunk_index=chunk_index,
                    metadata={'section_type': section_type},
                    chunk_type=section_type
                ))
                chunk_index += 1
            else:
                # Need to split section further
                if self.use_sentence_boundaries:
                    sub_chunks = self._chunk_by_sentences(
                        section_text, chunk_size, overlap, chunk_index
                    )
                else:
                    sub_chunks = self._chunk_by_words(
                        section_text, chunk_size, overlap, chunk_index
                    )
                
                for sub_chunk in sub_chunks:
                    sub_chunk.start_char += current_position
                    sub_chunk.end_char += current_position
                    sub_chunk.metadata['section_type'] = section_type
                    chunks.append(sub_chunk)
                    chunk_index += 1
            
            current_position += len(section_text)
        
        return chunks
    
    def _split_by_headers(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text by markdown headers.
        
        Args:
            text: Text with headers
            
        Returns:
            List of sections with type information
        """
        sections = []
        
        # Match markdown headers
        header_pattern = r'^(#{1,6})\s+(.+)$'
        
        lines = text.split('\n')
        current_section = []
        current_type = 'text'
        
        for line in lines:
            header_match = re.match(header_pattern, line)
            if header_match:
                # Save previous section
                if current_section:
                    sections.append({
                        'text': '\n'.join(current_section),
                        'type': current_type
                    })
                
                # Start new section
                current_section = [line]
                header_level = len(header_match.group(1))
                current_type = f'header_{header_level}'
            else:
                current_section.append(line)
        
        # Add last section
        if current_section:
            sections.append({
                'text': '\n'.join(current_section),
                'type': current_type
            })
        
        return sections if sections else [{'text': text, 'type': 'text'}]
    
    def _chunk_by_sentences(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
        start_index: int
    ) -> List[Chunk]:
        """
        Chunk text by sentence boundaries.
        
        Args:
            text: Text to chunk
            chunk_size: Target chunk size
            overlap: Overlap between chunks
            start_index: Starting chunk index
            
        Returns:
            List of chunks
        """
        chunks = []
        
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = []
        current_size = 0
        chunk_index = start_index
        current_position = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(Chunk(
                    id=f"chunk_{chunk_index}",
                    content=chunk_text,
                    start_char=current_position,
                    end_char=current_position + len(chunk_text),
                    chunk_index=chunk_index,
                    metadata={'chunking_method': 'sentence'},
                    chunk_type='text'
                ))
                
                # Handle overlap
                if overlap > 0:
                    # Keep last few sentences for overlap
                    overlap_sentences = []
                    overlap_size = 0
                    for sent in reversed(current_chunk):
                        overlap_size += len(sent)
                        overlap_sentences.insert(0, sent)
                        if overlap_size >= overlap:
                            break
                    current_chunk = overlap_sentences
                    current_size = overlap_size
                else:
                    current_chunk = []
                    current_size = 0
                
                chunk_index += 1
                current_position += len(chunk_text) - overlap
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add remaining sentences
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(Chunk(
                id=f"chunk_{chunk_index}",
                content=chunk_text,
                start_char=current_position,
                end_char=current_position + len(chunk_text),
                chunk_index=chunk_index,
                metadata={'chunking_method': 'sentence'},
                chunk_type='text'
            ))
        
        return chunks
    
    def _chunk_by_words(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
        start_index: int
    ) -> List[Chunk]:
        """
        Simple word-based chunking.
        
        Args:
            text: Text to chunk
            chunk_size: Target chunk size
            overlap: Overlap between chunks
            start_index: Starting chunk index
            
        Returns:
            List of chunks
        """
        chunks = []
        words = text.split()
        
        chunk_index = start_index
        i = 0
        
        while i < len(words):
            # Calculate chunk end
            chunk_words = []
            current_size = 0
            
            while i < len(words) and current_size < chunk_size:
                word = words[i]
                chunk_words.append(word)
                current_size += len(word) + 1  # +1 for space
                i += 1
            
            chunk_text = ' '.join(chunk_words)
            chunks.append(Chunk(
                id=f"chunk_{chunk_index}",
                content=chunk_text,
                start_char=0,  # Would need proper tracking
                end_char=len(chunk_text),
                chunk_index=chunk_index,
                metadata={'chunking_method': 'word'},
                chunk_type='text'
            ))
            
            chunk_index += 1
            
            # Handle overlap
            if overlap > 0 and i < len(words):
                # Move back for overlap
                overlap_words = overlap // 10  # Approximate
                i = max(i - overlap_words, 0)
        
        return chunks
    
    def _simple_chunking(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[Chunk]:
        """
        Simple character-based chunking.
        
        Args:
            text: Text to chunk
            chunk_size: Target chunk size
            overlap: Overlap between chunks
            
        Returns:
            List of chunks
        """
        chunks = []
        chunk_index = 0
        i = 0
        
        while i < len(text):
            end = min(i + chunk_size, len(text))
            chunk_text = text[i:end]
            
            chunks.append(Chunk(
                id=f"chunk_{chunk_index}",
                content=chunk_text,
                start_char=i,
                end_char=end,
                chunk_index=chunk_index,
                metadata={'chunking_method': 'simple'},
                chunk_type='text'
            ))
            
            chunk_index += 1
            i += chunk_size - overlap
        
        return chunks
    
    def _create_code_chunks(
        self,
        code_blocks: List[Dict[str, Any]],
        start_index: int
    ) -> List[Chunk]:
        """Create chunks for code blocks."""
        chunks = []
        
        for i, block in enumerate(code_blocks):
            chunks.append(Chunk(
                id=f"chunk_{start_index + i}",
                content=block['content'],
                start_char=block['start'],
                end_char=block['end'],
                chunk_index=start_index + i,
                metadata={'original_id': block['id']},
                chunk_type='code'
            ))
        
        return chunks
    
    def _create_table_chunks(
        self,
        tables: List[Dict[str, Any]],
        start_index: int
    ) -> List[Chunk]:
        """Create chunks for tables with advanced serialization."""
        chunks = []
        
        for i, table in enumerate(tables):
            # Create content that includes both original and serialized versions
            content_parts = [table['content']]
            
            # Add serialized representations if available
            if table.get('serialized'):
                serialized = table['serialized']
                
                # Add entity blocks if present
                if 'entity_blocks' in serialized:
                    content_parts.append("\n[Table as structured data:]")
                    for block in serialized['entity_blocks'][:5]:  # Limit to first 5 rows for chunk
                        content_parts.append(f"- {block['information_block']}")
                
                # Add sentences if present
                if 'sentences' in serialized:
                    content_parts.append("\n[Table description:]")
                    content_parts.extend(serialized['sentences'][:3])  # First 3 sentences
            
            full_content = '\n'.join(content_parts)
            
            # Create metadata including serialization info
            metadata = {
                'original_id': table['id'],
                'table_format': 'markdown',
                'has_serialization': table.get('serialized') is not None
            }
            
            if table.get('serialized') and 'metadata' in table['serialized']:
                metadata.update({
                    'num_rows': table['serialized']['metadata'].get('num_rows', 0),
                    'num_columns': table['serialized']['metadata'].get('num_columns', 0),
                    'headers': table['serialized']['metadata'].get('headers', [])
                })
            
            chunks.append(Chunk(
                id=f"chunk_{start_index + i}",
                content=full_content,
                start_char=table.get('start_pos', 0),
                end_char=table.get('end_pos', len(table['content'])),
                chunk_index=start_index + i,
                metadata=metadata,
                chunk_type='table'
            ))
            
            logger.debug(f"Created table chunk with {len(full_content)} characters")
        
        return chunks