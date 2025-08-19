"""
Tests for advanced table serialization functionality.
"""

import pytest
from typing import Dict, Any, List

from tldw_Server_API.app.core.RAG.rag_service.table_serialization import (
    TableParser, TableSerializer, TableProcessor, Table, TableFormat,
    serialize_table, process_document_with_tables
)


class TestTableParser:
    """Test table parsing functionality."""
    
    def test_detect_markdown_format(self):
        """Test markdown table format detection."""
        markdown_table = """
| Name | Age | City |
|------|-----|------|
| John | 30  | NYC  |
| Jane | 25  | LA   |
"""
        format = TableParser.detect_format(markdown_table)
        assert format == TableFormat.MARKDOWN
    
    def test_detect_csv_format(self):
        """Test CSV format detection."""
        csv_data = "Name,Age,City\nJohn,30,NYC\nJane,25,LA"
        format = TableParser.detect_format(csv_data)
        assert format == TableFormat.CSV
    
    def test_detect_tsv_format(self):
        """Test TSV format detection."""
        tsv_data = "Name\tAge\tCity\nJohn\t30\tNYC\nJane\t25\tLA"
        format = TableParser.detect_format(tsv_data)
        assert format == TableFormat.TSV
    
    def test_parse_markdown_table(self):
        """Test parsing a markdown table."""
        markdown_table = """
| Name | Age | City |
|------|-----|------|
| John | 30  | NYC  |
| Jane | 25  | LA   |
"""
        table = TableParser.parse_markdown_table(markdown_table)
        
        assert table.headers == ["Name", "Age", "City"]
        assert table.num_rows == 2
        assert table.num_columns == 3
        assert table.rows[0] == ["John", "30", "NYC"]
        assert table.rows[1] == ["Jane", "25", "LA"]
    
    def test_parse_csv_table(self):
        """Test parsing a CSV table."""
        csv_data = "Name,Age,City\nJohn,30,NYC\nJane,25,LA"
        table = TableParser.parse_csv_table(csv_data)
        
        assert table.headers == ["Name", "Age", "City"]
        assert table.num_rows == 2
        assert table.rows[0] == ["John", "30", "NYC"]
    
    def test_parse_with_empty_cells(self):
        """Test parsing table with empty cells."""
        markdown_table = """
| Name | Age | City |
|------|-----|------|
| John |     | NYC  |
| Jane | 25  |      |
"""
        table = TableParser.parse_markdown_table(markdown_table)
        
        assert table.rows[0] == ["John", "", "NYC"]
        assert table.rows[1] == ["Jane", "25", ""]


class TestTableSerializer:
    """Test table serialization functionality."""
    
    @pytest.fixture
    def sample_table(self):
        """Create a sample table for testing."""
        return Table(
            headers=["Product", "Price", "Stock"],
            rows=[
                ["Laptop", "$999", "15"],
                ["Mouse", "$25", "50"],
                ["Keyboard", "$75", "30"]
            ],
            format=TableFormat.MARKDOWN
        )
    
    def test_serialize_to_entities(self, sample_table):
        """Test entity serialization."""
        entities = TableSerializer.serialize_to_entities(sample_table)
        
        assert len(entities) == 3
        assert "Product: Laptop" in entities[0]["information_block"]
        assert "Price: $999" in entities[0]["information_block"]
        assert entities[0]["metadata"]["row_number"] == 1
    
    def test_serialize_to_sentences(self, sample_table):
        """Test sentence serialization."""
        sentences = TableSerializer.serialize_to_sentences(sample_table)
        
        assert len(sentences) > 0
        assert "3 rows and 3 columns" in sentences[0]
        assert "Product is Laptop" in sentences[1]
    
    def test_serialize_to_key_value_pairs(self, sample_table):
        """Test key-value serialization."""
        kv_pairs = TableSerializer.serialize_to_key_value_pairs(sample_table)
        
        assert "all_rows" in kv_pairs
        assert len(kv_pairs["all_rows"]) == 3
        assert kv_pairs["all_rows"][0]["Product"] == "Laptop"
    
    def test_serialize_for_rag(self, sample_table):
        """Test RAG serialization."""
        result = TableSerializer.serialize_for_rag(sample_table, method="hybrid")
        
        assert "metadata" in result
        assert "entity_blocks" in result
        assert "sentences" in result
        assert "search_text" in result
        assert result["metadata"]["num_rows"] == 3
    
    def test_table_to_markdown(self, sample_table):
        """Test converting table back to markdown."""
        markdown = TableSerializer.table_to_markdown(sample_table)
        
        assert "| Product | Price | Stock |" in markdown
        assert "| Laptop | $999 | 15 |" in markdown


class TestTableProcessor:
    """Test high-level table processing."""
    
    def test_process_single_table(self):
        """Test processing a single table."""
        processor = TableProcessor()
        table_text = """
| Feature | Status | Priority |
|---------|--------|----------|
| Search  | Done   | High     |
| Filter  | WIP    | Medium   |
"""
        result = processor.process_table(table_text)
        
        assert "metadata" in result
        assert "search_text" in result
        assert result["metadata"]["num_rows"] == 2
    
    def test_process_document_with_tables(self):
        """Test processing a document containing tables."""
        processor = TableProcessor()
        document = """
This is a document with a table.

| Name | Role | Department |
|------|------|------------|
| Alice | Manager | Sales |
| Bob | Engineer | Tech |

Some text after the table.

| ID | Status |
|----|--------|
| 1  | Active |
| 2  | Pending |

End of document.
"""
        processed_text, metadata = processor.process_document_tables(document)
        
        assert len(metadata) == 2
        assert "[Table serialized for search:]" in processed_text
        assert "Table contents as structured data:" in processed_text
        assert metadata[0]["table_info"]["num_rows"] == 2
        assert metadata[1]["table_info"]["num_rows"] == 2
    
    def test_process_with_different_methods(self):
        """Test different serialization methods."""
        table_text = """
| A | B |
|---|---|
| 1 | 2 |
"""
        
        # Test entities method
        result_entities = serialize_table(table_text, method="entities")
        assert "entity_blocks" in result_entities
        assert "sentences" not in result_entities
        
        # Test sentences method
        result_sentences = serialize_table(table_text, method="sentences")
        assert "sentences" in result_sentences
        assert "entity_blocks" not in result_sentences
        
        # Test hybrid method
        result_hybrid = serialize_table(table_text, method="hybrid")
        assert "entity_blocks" in result_hybrid
        assert "sentences" in result_hybrid


class TestIntegrationWithChunking:
    """Test integration with enhanced chunking service."""
    
    def test_chunking_with_tables(self):
        """Test that enhanced chunking properly handles tables."""
        from tldw_Server_API.app.core.RAG.rag_service.enhanced_chunking import EnhancedChunkingService
        
        service = EnhancedChunkingService({
            'table_serialize_method': 'hybrid'
        })
        
        text = """
# Document Title

Some introductory text.

| Metric | Value | Unit |
|--------|-------|------|
| Speed  | 100   | mph  |
| Weight | 2000  | kg   |

Conclusion text.
"""
        
        chunks = service.chunk_text(text, chunk_size=512)
        
        # Find the table chunk
        table_chunks = [c for c in chunks if c.chunk_type == 'table']
        assert len(table_chunks) > 0
        
        table_chunk = table_chunks[0]
        assert "Metric | Value | Unit" in table_chunk.content
        assert table_chunk.metadata.get('has_serialization') is True
        assert table_chunk.metadata.get('num_rows') == 2
    
    def test_complex_document_processing(self):
        """Test processing a complex document with multiple tables."""
        document = """
# Sales Report

## Q1 Results

| Month | Revenue | Growth |
|-------|---------|--------|
| Jan   | $100K   | 10%    |
| Feb   | $120K   | 20%    |
| Mar   | $150K   | 25%    |

The Q1 results show strong growth.

## Product Performance

| Product | Units | Revenue |
|---------|-------|---------|
| A       | 500   | $50K    |
| B       | 300   | $45K    |
| C       | 700   | $55K    |

Product C is our best performer.
"""
        
        result = process_document_with_tables(document, serialize_method="hybrid")
        
        assert result["table_count"] == 2
        assert "Table contents as structured data:" in result["processed_text"]
        assert len(result["table_metadata"]) == 2


class TestErrorHandling:
    """Test error handling in table processing."""
    
    def test_invalid_table_format(self):
        """Test handling of invalid table format."""
        processor = TableProcessor()
        invalid_table = "This is not a table"
        
        result = processor.process_table(invalid_table)
        assert "error" in result or "search_text" in result
    
    def test_malformed_markdown_table(self):
        """Test handling of malformed markdown table."""
        malformed = """
| Col1 | Col2
|------|
| Val1 |
"""
        processor = TableProcessor()
        result = processor.process_table(malformed)
        
        # Should handle gracefully
        assert result is not None
    
    def test_empty_table(self):
        """Test handling of empty table."""
        empty_table = """
| | |
|-|-|
"""
        table = TableParser.parse_markdown_table(empty_table)
        assert table.num_columns == 2
        assert table.num_rows == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])