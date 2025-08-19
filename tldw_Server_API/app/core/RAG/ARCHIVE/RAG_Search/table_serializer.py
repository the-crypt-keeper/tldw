"""
Advanced table serialization for improved semantic understanding in RAG.

This module provides sophisticated table parsing and serialization capabilities
to convert tabular data into more semantically meaningful text representations.
"""

import re
import csv
import json
from typing import List, Dict, Any, Tuple, Optional, Union, Literal
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TableFormat(Enum):
    """Supported table formats."""
    MARKDOWN = "markdown"
    CSV = "csv"
    TSV = "tsv"
    HTML = "html"
    JSON = "json"
    PIPE_DELIMITED = "pipe"


@dataclass
class TableCell:
    """Represents a single table cell."""
    value: str
    row_idx: int
    col_idx: int
    is_header: bool = False
    colspan: int = 1
    rowspan: int = 1
    
    @property
    def is_empty(self) -> bool:
        """Check if cell is empty."""
        return not self.value.strip()


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
    
    def to_dataframe_dict(self) -> Dict[str, List[Any]]:
        """Convert to dictionary format suitable for DataFrame creation."""
        if not self.headers:
            # Generate default headers
            self.headers = [f"Column_{i+1}" for i in range(self.num_columns)]
        
        result = {header: [] for header in self.headers}
        
        for row in self.rows:
            for i, (header, value) in enumerate(zip(self.headers, row)):
                result[header].append(value if i < len(row) else "")
        
        return result


class TableParser:
    """Parse tables from various formats."""
    
    @staticmethod
    def detect_format(text: str) -> Optional[TableFormat]:
        """
        Detect table format from text.
        
        Args:
            text: Input text containing table
            
        Returns:
            Detected format or None
        """
        lines = text.strip().split('\n')
        if not lines:
            return None
        
        # Check for Markdown table
        if any('|' in line for line in lines[:3]):
            # Check for separator line (e.g., |---|---|)
            for line in lines[:5]:
                if re.match(r'^\s*\|?\s*:?-+:?\s*\|', line):
                    return TableFormat.MARKDOWN
        
        # Check for HTML table
        if '<table' in text.lower() and '</table>' in text.lower():
            return TableFormat.HTML
        
        # Check for JSON array
        try:
            data = json.loads(text)
            if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                return TableFormat.JSON
        except:
            pass
        
        # Check for CSV/TSV
        first_line = lines[0]
        if '\t' in first_line:
            return TableFormat.TSV
        elif ',' in first_line:
            # Simple CSV detection
            return TableFormat.CSV
        
        return None
    
    @staticmethod
    def parse_markdown_table(text: str) -> Table:
        """Parse a Markdown-formatted table."""
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        
        if len(lines) < 2:
            raise ValueError("Invalid Markdown table: too few lines")
        
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
            if line:
                cells = [cell.strip() for cell in re.split(r'\s*\|\s*', line) if cell.strip()]
                if cells:
                    # Ensure row has same number of columns as header
                    while len(cells) < len(headers):
                        cells.append("")
                    rows.append(cells[:len(headers)])
        
        return Table(headers=headers, rows=rows, format=TableFormat.MARKDOWN)
    
    @staticmethod
    def parse_csv_table(text: str, delimiter: str = ',') -> Table:
        """Parse a CSV/TSV formatted table."""
        import io
        
        reader = csv.reader(io.StringIO(text.strip()), delimiter=delimiter)
        rows_data = list(reader)
        
        if not rows_data:
            raise ValueError("Empty CSV data")
        
        # First row is assumed to be headers
        headers = rows_data[0] if rows_data else []
        rows = rows_data[1:] if len(rows_data) > 1 else []
        
        # Normalize row lengths
        max_cols = max(len(headers), max(len(row) for row in rows) if rows else 0)
        headers.extend([""] * (max_cols - len(headers)))
        
        normalized_rows = []
        for row in rows:
            normalized_row = row + [""] * (max_cols - len(row))
            normalized_rows.append(normalized_row[:max_cols])
        
        format_type = TableFormat.TSV if delimiter == '\t' else TableFormat.CSV
        return Table(headers=headers, rows=normalized_rows, format=format_type)
    
    @staticmethod
    def parse_json_table(text: str) -> Table:
        """Parse a JSON array of objects as a table."""
        data = json.loads(text)
        
        if not isinstance(data, list):
            raise ValueError("JSON must be an array of objects")
        
        if not data:
            return Table(headers=[], rows=[], format=TableFormat.JSON)
        
        # Extract headers from all objects (union of all keys)
        all_keys = set()
        for item in data:
            if isinstance(item, dict):
                all_keys.update(item.keys())
        
        headers = sorted(list(all_keys))
        
        # Extract rows
        rows = []
        for item in data:
            if isinstance(item, dict):
                row = [str(item.get(key, "")) for key in headers]
                rows.append(row)
        
        return Table(headers=headers, rows=rows, format=TableFormat.JSON)
    
    @classmethod
    def parse(cls, text: str, format: Optional[TableFormat] = None) -> Table:
        """
        Parse table from text.
        
        Args:
            text: Table text
            format: Format hint (auto-detect if None)
            
        Returns:
            Parsed Table object
        """
        if format is None:
            format = cls.detect_format(text)
            if format is None:
                raise ValueError("Could not detect table format")
        
        if format == TableFormat.MARKDOWN:
            return cls.parse_markdown_table(text)
        elif format == TableFormat.CSV:
            return cls.parse_csv_table(text, delimiter=',')
        elif format == TableFormat.TSV:
            return cls.parse_csv_table(text, delimiter='\t')
        elif format == TableFormat.JSON:
            return cls.parse_json_table(text)
        else:
            raise ValueError(f"Unsupported format: {format}")


class TableSerializer:
    """Serialize tables into semantic text representations."""
    
    @staticmethod
    def serialize_to_entities(table: Table, 
                            include_row_numbers: bool = True,
                            value_separator: str = ": ",
                            field_separator: str = "; ") -> List[Dict[str, str]]:
        """
        Serialize table rows as entity descriptions.
        
        Each row becomes a text block describing an entity with its attributes.
        
        Args:
            table: Table to serialize
            include_row_numbers: Whether to include row numbers
            value_separator: Separator between header and value
            field_separator: Separator between fields
            
        Returns:
            List of serialized entity blocks
        """
        serialized_blocks = []
        
        for i, row in enumerate(table.rows):
            entity_parts = []
            
            if include_row_numbers:
                entity_parts.append(f"Row {i + 1}")
            
            for header, value in zip(table.headers, row):
                if value and value.strip():  # Skip empty values
                    entity_parts.append(f"{header}{value_separator}{value}")
            
            if entity_parts:
                serialized_blocks.append({
                    "information_block": field_separator.join(entity_parts),
                    "row_index": i,
                    "metadata": {
                        "type": "table_row",
                        "row_number": i + 1,
                        "num_fields": len([v for v in row if v.strip()])
                    }
                })
        
        return serialized_blocks
    
    @staticmethod
    def serialize_to_sentences(table: Table,
                             template: Optional[str] = None,
                             include_summary: bool = True) -> List[str]:
        """
        Serialize table rows as natural language sentences.
        
        Args:
            table: Table to serialize
            template: Template for sentence generation (uses headers as placeholders)
            include_summary: Whether to include a summary sentence
            
        Returns:
            List of sentences describing the table
        """
        sentences = []
        
        # Add summary if requested
        if include_summary:
            summary = f"This table contains {table.num_rows} rows and {table.num_columns} columns"
            if table.headers:
                summary += f" with headers: {', '.join(table.headers)}"
            sentences.append(summary + ".")
        
        # Generate sentences for each row
        for i, row in enumerate(table.rows):
            if template:
                # Use custom template
                sentence = template
                for header, value in zip(table.headers, row):
                    placeholder = f"{{{header}}}"
                    sentence = sentence.replace(placeholder, value)
                sentences.append(sentence)
            else:
                # Generate default sentence
                row_desc = []
                for header, value in zip(table.headers, row):
                    if value and value.strip():
                        row_desc.append(f"{header} is {value}")
                
                if row_desc:
                    sentence = f"In row {i + 1}, " + ", ".join(row_desc) + "."
                    sentences.append(sentence)
        
        return sentences
    
    @staticmethod
    def serialize_to_key_value_pairs(table: Table,
                                   group_by_column: Optional[str] = None) -> Dict[str, List[Dict[str, str]]]:
        """
        Serialize table as key-value pairs, optionally grouped by a column.
        
        Args:
            table: Table to serialize
            group_by_column: Column to group by (if None, no grouping)
            
        Returns:
            Dictionary of key-value representations
        """
        if group_by_column and group_by_column not in table.headers:
            raise ValueError(f"Group by column '{group_by_column}' not found in headers")
        
        result = {}
        
        if group_by_column:
            # Group by specified column
            group_idx = table.headers.index(group_by_column)
            
            for row in table.rows:
                group_key = row[group_idx] if group_idx < len(row) else "Unknown"
                
                if group_key not in result:
                    result[group_key] = []
                
                row_data = {}
                for j, (header, value) in enumerate(zip(table.headers, row)):
                    if j != group_idx and value.strip():  # Skip group column and empty values
                        row_data[header] = value
                
                if row_data:
                    result[group_key].append(row_data)
        else:
            # No grouping, all rows in one list
            all_rows = []
            for row in table.rows:
                row_data = {}
                for header, value in zip(table.headers, row):
                    if value.strip():
                        row_data[header] = value
                if row_data:
                    all_rows.append(row_data)
            result["all_rows"] = all_rows
        
        return result
    
    @staticmethod
    def serialize_for_rag(table: Table,
                         method: Literal["entities", "sentences", "hybrid"] = "hybrid",
                         include_original: bool = True) -> Dict[str, Any]:
        """
        Serialize table for RAG indexing with multiple representations.
        
        Args:
            table: Table to serialize
            method: Serialization method
            include_original: Whether to include original markdown
            
        Returns:
            Dictionary with multiple representations
        """
        result = {
            "metadata": {
                "num_rows": table.num_rows,
                "num_columns": table.num_columns,
                "headers": table.headers,
                "format": table.format.value
            }
        }
        
        # Include original markdown representation
        if include_original:
            result["markdown"] = TableSerializer.table_to_markdown(table)
        
        # Add serialized representations
        if method in ["entities", "hybrid"]:
            result["entity_blocks"] = TableSerializer.serialize_to_entities(table)
        
        if method in ["sentences", "hybrid"]:
            result["sentences"] = TableSerializer.serialize_to_sentences(table)
        
        # Add searchable text combining all representations
        search_text_parts = []
        
        if "entity_blocks" in result:
            for block in result["entity_blocks"]:
                search_text_parts.append(block["information_block"])
        
        if "sentences" in result:
            search_text_parts.extend(result["sentences"])
        
        result["search_text"] = " ".join(search_text_parts)
        
        return result
    
    @staticmethod
    def table_to_markdown(table: Table) -> str:
        """Convert table back to markdown format."""
        if not table.headers and not table.rows:
            return ""
        
        lines = []
        
        # Add headers
        if table.headers:
            lines.append("| " + " | ".join(table.headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(table.headers)) + " |")
        
        # Add rows
        for row in table.rows:
            lines.append("| " + " | ".join(row) + " |")
        
        return "\n".join(lines)


class TableProcessor:
    """High-level table processing for documents."""
    
    def __init__(self):
        """Initialize table processor."""
        self.parser = TableParser()
        self.serializer = TableSerializer()
    
    def process_document_tables(self, 
                              text: str,
                              serialize_method: Literal["entities", "sentences", "hybrid"] = "hybrid") -> Tuple[str, List[Dict[str, Any]]]:
        """
        Find and process all tables in a document.
        
        Args:
            text: Document text
            serialize_method: How to serialize tables
            
        Returns:
            Tuple of (text_with_serialized_tables, table_metadata)
        """
        # Find markdown tables (most common in our use case)
        table_pattern = re.compile(
            r'((?:^\|[^\n]+\|$\n)+)', 
            re.MULTILINE
        )
        
        processed_text = text
        table_metadata = []
        offset = 0
        
        for match in table_pattern.finditer(text):
            try:
                table_text = match.group(0)
                table = self.parser.parse(table_text, TableFormat.MARKDOWN)
                
                # Serialize table
                serialized = self.serializer.serialize_for_rag(
                    table, 
                    method=serialize_method,
                    include_original=True
                )
                
                # Create replacement text
                replacement_parts = [table_text]  # Keep original
                
                if serialize_method in ["entities", "hybrid"]:
                    replacement_parts.append("\nTable contents as structured data:")
                    for block in serialized.get("entity_blocks", []):
                        replacement_parts.append(f"- {block['information_block']}")
                
                if serialize_method in ["sentences", "hybrid"]:
                    replacement_parts.append("\nTable description:")
                    replacement_parts.extend(serialized.get("sentences", []))
                
                replacement_text = "\n".join(replacement_parts)
                
                # Calculate positions
                start = match.start() + offset
                end = match.end() + offset
                new_end = start + len(replacement_text)
                
                # Replace in text
                processed_text = processed_text[:start] + replacement_text + processed_text[end:]
                
                # Update offset
                offset += len(replacement_text) - len(table_text)
                
                # Store metadata
                table_metadata.append({
                    "start_pos": start,
                    "end_pos": new_end,
                    "original_length": len(table_text),
                    "serialized_length": len(replacement_text),
                    "table_info": serialized["metadata"],
                    "serialization_method": serialize_method
                })
                
                logger.debug(f"Processed table with {table.num_rows} rows and {table.num_columns} columns")
                
            except Exception as e:
                logger.warning(f"Failed to process table: {e}")
                continue
        
        return processed_text, table_metadata


# Convenience functions
def serialize_table(table_text: str, 
                   format: Optional[TableFormat] = None,
                   method: Literal["entities", "sentences", "hybrid"] = "hybrid") -> Dict[str, Any]:
    """
    Serialize a table for improved semantic understanding.
    
    Args:
        table_text: Raw table text
        format: Table format (auto-detect if None)
        method: Serialization method
        
    Returns:
        Serialized table data
    """
    parser = TableParser()
    serializer = TableSerializer()
    
    table = parser.parse(table_text, format)
    return serializer.serialize_for_rag(table, method)


def process_document_with_tables(document_text: str,
                               serialize_method: Literal["entities", "sentences", "hybrid"] = "hybrid") -> Dict[str, Any]:
    """
    Process a document and serialize any tables found.
    
    Args:
        document_text: Full document text
        serialize_method: How to serialize tables
        
    Returns:
        Dictionary with processed text and metadata
    """
    processor = TableProcessor()
    processed_text, table_metadata = processor.process_document_tables(document_text, serialize_method)
    
    return {
        "processed_text": processed_text,
        "table_count": len(table_metadata),
        "table_metadata": table_metadata,
        "processing_method": serialize_method
    }