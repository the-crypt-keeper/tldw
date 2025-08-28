# strategies/structure_aware.py
"""
Structure-aware chunking strategy.
Preserves document structure including tables, headers, code blocks, and lists.
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from loguru import logger

from ..base import BaseChunkingStrategy, ChunkResult, ChunkMetadata


class StructureType(Enum):
    """Types of document structures."""
    PARAGRAPH = "paragraph"
    HEADER = "header"
    TABLE = "table"
    CODE_BLOCK = "code_block"
    LIST = "list"
    QUOTE = "quote"
    METADATA = "metadata"


class TableFormat(Enum):
    """Supported table formats."""
    MARKDOWN = "markdown"
    CSV = "csv"
    TSV = "tsv"
    HTML = "html"
    JSON = "json"
    PIPE_DELIMITED = "pipe"


@dataclass
class DocumentElement:
    """Represents a structural element in a document."""
    type: StructureType
    content: str
    level: Optional[int] = None  # For headers (h1=1, h2=2, etc.)
    language: Optional[str] = None  # For code blocks
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Table:
    """Represents a parsed table."""
    headers: List[str]
    rows: List[List[str]]
    format: TableFormat
    metadata: Dict[str, Any] = None
    
    @property
    def num_columns(self) -> int:
        """Get number of columns."""
        return len(self.headers) if self.headers else (len(self.rows[0]) if self.rows else 0)
    
    @property
    def num_rows(self) -> int:
        """Get number of data rows (excluding header)."""
        return len(self.rows)
    
    def to_markdown(self) -> str:
        """Convert table to markdown format."""
        if not self.headers and not self.rows:
            return ""
        
        # Use headers or generate default ones
        headers = self.headers if self.headers else [f"Col{i+1}" for i in range(self.num_columns)]
        
        # Build markdown table
        lines = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---" for _ in headers]) + "|")
        
        for row in self.rows:
            # Ensure row has correct number of columns
            padded_row = row + [""] * (len(headers) - len(row))
            lines.append("| " + " | ".join(padded_row[:len(headers)]) + " |")
        
        return "\n".join(lines)
    
    def to_text(self, style: str = "entity") -> str:
        """
        Convert table to text representation.
        
        Args:
            style: Serialization style ('entity', 'narrative', 'compact')
        """
        if style == "entity":
            return self._to_entity_text()
        elif style == "narrative":
            return self._to_narrative_text()
        else:
            return self._to_compact_text()
    
    def _to_entity_text(self) -> str:
        """Convert to entity-based text representation."""
        lines = []
        headers = self.headers if self.headers else [f"Col{i+1}" for i in range(self.num_columns)]
        
        for i, row in enumerate(self.rows):
            entity_parts = []
            for j, header in enumerate(headers):
                if j < len(row) and row[j]:
                    entity_parts.append(f"{header}: {row[j]}")
            
            if entity_parts:
                lines.append(f"Row {i+1}: " + "; ".join(entity_parts))
        
        return "\n".join(lines)
    
    def _to_narrative_text(self) -> str:
        """Convert to narrative text representation."""
        if not self.rows:
            return "Empty table"
        
        headers = self.headers if self.headers else [f"Col{i+1}" for i in range(self.num_columns)]
        
        narrative = f"A table with {len(headers)} columns ({', '.join(headers)}) "
        narrative += f"containing {len(self.rows)} rows of data. "
        
        # Describe first few rows
        for i, row in enumerate(self.rows[:3]):
            narrative += f"Row {i+1} contains: "
            parts = []
            for j, header in enumerate(headers):
                if j < len(row) and row[j]:
                    parts.append(f"{header} is '{row[j]}'")
            narrative += ", ".join(parts) + ". "
        
        if len(self.rows) > 3:
            narrative += f"... and {len(self.rows) - 3} more rows."
        
        return narrative
    
    def _to_compact_text(self) -> str:
        """Convert to compact text representation."""
        lines = []
        headers = self.headers if self.headers else []
        
        if headers:
            lines.append(" | ".join(headers))
        
        for row in self.rows:
            lines.append(" | ".join(row))
        
        return "\n".join(lines)


class StructureAwareChunkingStrategy(BaseChunkingStrategy):
    """
    Chunks text while preserving document structure.
    Handles tables, headers, code blocks, lists, and other structural elements.
    """
    
    def __init__(self, language: str = 'en'):
        """
        Initialize structure-aware chunking strategy.
        
        Args:
            language: Language code for text processing
        """
        super().__init__(language)
        
        # Regex patterns for structure detection
        self.patterns = {
            'markdown_header': re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE),
            'code_block': re.compile(r'```(\w*)\n(.*?)\n```', re.DOTALL),
            'table_markdown': re.compile(r'^\|.*\|.*$', re.MULTILINE),
            'list_item': re.compile(r'^[\s]*[-*+]\s+(.+)$', re.MULTILINE),
            'numbered_list': re.compile(r'^[\s]*\d+\.\s+(.+)$', re.MULTILINE),
            'quote': re.compile(r'^>\s+(.+)$', re.MULTILINE),
        }
        
        logger.debug(f"StructureAwareChunkingStrategy initialized for language: {language}")
    
    def chunk(self,
              text: str,
              max_size: int,
              overlap: int = 0,
              **options) -> List[str]:
        """
        Chunk text while preserving structure.
        
        Args:
            text: Text to chunk
            max_size: Maximum size per chunk (in elements, not characters)
            overlap: Number of elements to overlap between chunks
            **options: Additional options:
                - preserve_tables: Keep tables intact
                - preserve_code_blocks: Keep code blocks intact
                - preserve_headers: Include header hierarchy
                - table_serialization: How to serialize tables ('markdown', 'entity', 'narrative')
                
        Returns:
            List of text chunks preserving structure
        """
        if not self.validate_parameters(text, max_size, overlap):
            return []
        
        # Parse document structure
        elements = self._parse_document_structure(text, **options)
        
        if not elements:
            return []
        
        logger.debug(f"Parsed {len(elements)} structural elements")
        
        # Group elements into chunks
        chunks = self._group_elements_into_chunks(elements, max_size, overlap, **options)
        
        # Convert chunks to text
        text_chunks = []
        for chunk_elements in chunks:
            chunk_text = self._elements_to_text(chunk_elements, **options)
            if chunk_text.strip():
                text_chunks.append(chunk_text)
        
        logger.info(f"Created {len(text_chunks)} structure-aware chunks")
        return text_chunks
    
    def _parse_document_structure(self, text: str, **options) -> List[DocumentElement]:
        """
        Parse document into structural elements.
        
        Args:
            text: Document text
            **options: Parsing options
            
        Returns:
            List of document elements
        """
        elements = []
        processed_ranges = []
        
        # Extract code blocks first (they can contain other patterns)
        if options.get('preserve_code_blocks', True):
            for match in self.patterns['code_block'].finditer(text):
                language = match.group(1) or 'text'
                code_content = match.group(2)
                elements.append(DocumentElement(
                    type=StructureType.CODE_BLOCK,
                    content=code_content,
                    language=language,
                    metadata={'start': match.start(), 'end': match.end()}
                ))
                processed_ranges.append((match.start(), match.end()))
        
        # Extract tables
        if options.get('preserve_tables', True):
            tables = self._extract_tables(text, processed_ranges)
            elements.extend(tables)
            for elem in tables:
                if 'start' in elem.metadata and 'end' in elem.metadata:
                    processed_ranges.append((elem.metadata['start'], elem.metadata['end']))
        
        # Extract headers
        if options.get('preserve_headers', True):
            for match in self.patterns['markdown_header'].finditer(text):
                if not self._in_processed_range(match.start(), processed_ranges):
                    level = len(match.group(1))
                    header_text = match.group(2)
                    elements.append(DocumentElement(
                        type=StructureType.HEADER,
                        content=header_text,
                        level=level,
                        metadata={'start': match.start(), 'end': match.end()}
                    ))
                    processed_ranges.append((match.start(), match.end()))
        
        # Extract lists
        if options.get('preserve_lists', True):
            # Bullet lists
            for match in self.patterns['list_item'].finditer(text):
                if not self._in_processed_range(match.start(), processed_ranges):
                    elements.append(DocumentElement(
                        type=StructureType.LIST,
                        content=match.group(0),
                        metadata={'list_type': 'bullet', 'start': match.start(), 'end': match.end()}
                    ))
                    processed_ranges.append((match.start(), match.end()))
            
            # Numbered lists
            for match in self.patterns['numbered_list'].finditer(text):
                if not self._in_processed_range(match.start(), processed_ranges):
                    elements.append(DocumentElement(
                        type=StructureType.LIST,
                        content=match.group(0),
                        metadata={'list_type': 'numbered', 'start': match.start(), 'end': match.end()}
                    ))
                    processed_ranges.append((match.start(), match.end()))
        
        # Extract remaining paragraphs
        processed_ranges.sort()
        last_end = 0
        
        for start, end in processed_ranges:
            if start > last_end:
                paragraph_text = text[last_end:start].strip()
                if paragraph_text:
                    elements.append(DocumentElement(
                        type=StructureType.PARAGRAPH,
                        content=paragraph_text
                    ))
            last_end = max(last_end, end)
        
        # Add final paragraph if any
        if last_end < len(text):
            paragraph_text = text[last_end:].strip()
            if paragraph_text:
                elements.append(DocumentElement(
                    type=StructureType.PARAGRAPH,
                    content=paragraph_text
                ))
        
        # Sort elements by position
        elements.sort(key=lambda e: e.metadata.get('start', float('inf')))
        
        return elements
    
    def _extract_tables(self, text: str, processed_ranges: List[Tuple[int, int]]) -> List[DocumentElement]:
        """
        Extract tables from text.
        
        Args:
            text: Document text
            processed_ranges: Already processed text ranges
            
        Returns:
            List of table elements
        """
        table_elements = []
        
        # Look for markdown tables
        lines = text.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Check if this looks like a markdown table
            if '|' in line and not self._in_processed_range(i, processed_ranges):
                # Try to parse as table
                table_lines = [line]
                j = i + 1
                
                # Look for separator line
                if j < len(lines) and re.match(r'^[\s\|:\-]+$', lines[j]):
                    table_lines.append(lines[j])
                    j += 1
                    
                    # Collect data rows
                    while j < len(lines) and '|' in lines[j]:
                        table_lines.append(lines[j])
                        j += 1
                    
                    # Parse the table
                    table = self._parse_markdown_table('\n'.join(table_lines))
                    if table:
                        table_text = table.to_markdown()
                        table_elements.append(DocumentElement(
                            type=StructureType.TABLE,
                            content=table_text,
                            metadata={
                                'table': table,
                                'format': 'markdown',
                                'start': i,
                                'end': j
                            }
                        ))
                        i = j
                        continue
            
            i += 1
        
        return table_elements
    
    def _parse_markdown_table(self, text: str) -> Optional[Table]:
        """Parse a markdown table."""
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        
        if len(lines) < 2:
            return None
        
        # Parse header
        header_line = lines[0]
        headers = [cell.strip() for cell in re.split(r'\s*\|\s*', header_line) if cell.strip()]
        
        # Skip separator line
        separator_idx = 1
        for i, line in enumerate(lines[1:], 1):
            if re.match(r'^[\s\|:\-]+$', line):
                separator_idx = i
                break
        
        # Parse rows
        rows = []
        for line in lines[separator_idx + 1:]:
            if '|' in line:
                row = [cell.strip() for cell in re.split(r'\s*\|\s*', line) if cell.strip()]
                if row:
                    rows.append(row)
        
        if headers or rows:
            return Table(headers=headers, rows=rows, format=TableFormat.MARKDOWN)
        
        return None
    
    def _in_processed_range(self, position: int, ranges: List[Tuple[int, int]]) -> bool:
        """Check if position is in any processed range."""
        for start, end in ranges:
            if start <= position < end:
                return True
        return False
    
    def _group_elements_into_chunks(self, 
                                   elements: List[DocumentElement],
                                   max_size: int,
                                   overlap: int,
                                   **options) -> List[List[DocumentElement]]:
        """
        Group elements into chunks respecting structure.
        
        Args:
            elements: List of document elements
            max_size: Maximum elements per chunk
            overlap: Number of elements to overlap
            **options: Chunking options
            
        Returns:
            List of element groups (chunks)
        """
        chunks = []
        current_chunk = []
        current_size = 0
        
        for element in elements:
            # Check if element should be kept intact
            keep_intact = False
            
            if element.type == StructureType.TABLE and options.get('preserve_tables', True):
                keep_intact = True
            elif element.type == StructureType.CODE_BLOCK and options.get('preserve_code_blocks', True):
                keep_intact = True
            
            # Estimate element size (simple count for now)
            element_size = 1
            
            # If adding element exceeds max_size and chunk is not empty
            if current_size + element_size > max_size and current_chunk:
                chunks.append(current_chunk)
                
                # Handle overlap
                if overlap > 0:
                    # Keep last 'overlap' elements for next chunk
                    current_chunk = current_chunk[-overlap:]
                    current_size = len(current_chunk)
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(element)
            current_size += element_size
            
            # If element must be kept intact and exceeds max_size alone
            if keep_intact and element_size > max_size and len(current_chunk) == 1:
                chunks.append(current_chunk)
                current_chunk = []
                current_size = 0
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _elements_to_text(self, elements: List[DocumentElement], **options) -> str:
        """
        Convert elements back to text.
        
        Args:
            elements: List of document elements
            **options: Serialization options
            
        Returns:
            Text representation
        """
        lines = []
        
        for element in elements:
            if element.type == StructureType.HEADER:
                # Recreate markdown header
                level = element.level or 1
                lines.append(f"{'#' * level} {element.content}")
                lines.append("")  # Empty line after header
                
            elif element.type == StructureType.CODE_BLOCK:
                # Recreate code block
                language = element.language or ''
                lines.append(f"```{language}")
                lines.append(element.content)
                lines.append("```")
                lines.append("")
                
            elif element.type == StructureType.TABLE:
                # Serialize table based on option
                if 'table' in element.metadata:
                    table = element.metadata['table']
                    style = options.get('table_serialization', 'markdown')
                    
                    if style == 'markdown':
                        lines.append(table.to_markdown())
                    elif style == 'entity':
                        lines.append(table.to_text('entity'))
                    elif style == 'narrative':
                        lines.append(table.to_text('narrative'))
                    else:
                        lines.append(element.content)
                else:
                    lines.append(element.content)
                lines.append("")
                
            elif element.type == StructureType.LIST:
                lines.append(element.content)
                
            elif element.type == StructureType.PARAGRAPH:
                lines.append(element.content)
                lines.append("")  # Empty line between paragraphs
                
            else:
                lines.append(element.content)
        
        return '\n'.join(lines).strip()