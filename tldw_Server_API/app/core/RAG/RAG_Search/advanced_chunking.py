# advanced_chunking.py - Advanced Chunking Strategies
"""
Advanced chunking module with semantic-aware and structure-preserving strategies.

Features:
- Semantic coherence chunking
- Sliding window with smart boundaries
- Hierarchical chunking with parent-child relationships
- Table and code block preservation
- Multi-modal chunking (text + metadata)
- Adaptive chunk sizing based on content
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import logging
from abc import ABC, abstractmethod
import numpy as np

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Available chunking strategies"""
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    SLIDING_WINDOW = "sliding_window"
    RECURSIVE = "recursive"
    ADAPTIVE = "adaptive"
    HYBRID = "hybrid"


@dataclass
class ChunkMetadata:
    """Metadata for a chunk"""
    chunk_id: str
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    chunk_type: str = "text"
    start_char: int = 0
    end_char: int = 0
    start_line: int = 0
    end_line: int = 0
    word_count: int = 0
    char_count: int = 0
    density_score: float = 0.0  # Information density
    coherence_score: float = 0.0  # Semantic coherence
    structural_level: int = 0  # 0=root, 1=section, 2=subsection, etc.
    tags: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A text chunk with metadata"""
    text: str
    metadata: ChunkMetadata
    embedding: Optional[np.ndarray] = None
    
    @property
    def id(self) -> str:
        return self.metadata.chunk_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "text": self.text,
            "metadata": {
                "chunk_id": self.metadata.chunk_id,
                "parent_id": self.metadata.parent_id,
                "children_ids": self.metadata.children_ids,
                "chunk_type": self.metadata.chunk_type,
                "start_char": self.metadata.start_char,
                "end_char": self.metadata.end_char,
                "word_count": self.metadata.word_count,
                "char_count": self.metadata.char_count,
                "density_score": self.metadata.density_score,
                "coherence_score": self.metadata.coherence_score,
                "structural_level": self.metadata.structural_level,
                "tags": self.metadata.tags,
                "properties": self.metadata.properties
            }
        }


class BaseChunker(ABC):
    """Base class for chunking strategies"""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2048
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    @abstractmethod
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Chunk text into smaller pieces"""
        pass
    
    def _generate_chunk_id(self, text: str, position: int) -> str:
        """Generate unique chunk ID"""
        content = f"{text[:50]}_{position}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _calculate_density(self, text: str) -> float:
        """Calculate information density of text"""
        if not text:
            return 0.0
        
        # Simple density: ratio of unique words to total words
        words = text.lower().split()
        if not words:
            return 0.0
        
        unique_words = set(words)
        density = len(unique_words) / len(words)
        
        # Adjust for very short texts
        if len(words) < 10:
            density *= 0.8
        
        return min(1.0, density)
    
    def _detect_boundaries(self, text: str) -> List[int]:
        """Detect natural boundaries in text"""
        boundaries = [0]
        
        # Paragraph boundaries
        for match in re.finditer(r'\n\n+', text):
            boundaries.append(match.start())
        
        # Sentence boundaries
        for match in re.finditer(r'[.!?]\s+', text):
            boundaries.append(match.end())
        
        # Section headers
        for match in re.finditer(r'\n#+\s+', text):
            boundaries.append(match.start())
        
        boundaries.append(len(text))
        return sorted(set(boundaries))


class SemanticChunker(BaseChunker):
    """Chunks text based on semantic coherence"""
    
    def __init__(self, *args, embeddings_model=None, coherence_threshold=0.7, **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddings_model = embeddings_model
        self.coherence_threshold = coherence_threshold
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Chunk based on semantic boundaries"""
        # First, get sentence boundaries
        sentences = self._split_sentences(text)
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_start_char = 0
        
        for i, sentence in enumerate(sentences):
            sentence_size = len(sentence)
            
            # Check if adding this sentence exceeds chunk size
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Check semantic coherence if we have embeddings
                if self.embeddings_model:
                    coherence = self._calculate_coherence(
                        ' '.join(current_chunk), 
                        sentence
                    )
                    
                    # If coherence is high, include the sentence anyway (up to max size)
                    if coherence > self.coherence_threshold and \
                       current_size + sentence_size <= self.max_chunk_size:
                        current_chunk.append(sentence)
                        current_size += sentence_size
                        continue
                
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunk_end_char = chunk_start_char + len(chunk_text)
                
                chunk_metadata = ChunkMetadata(
                    chunk_id=self._generate_chunk_id(chunk_text, chunk_start_char),
                    chunk_type="semantic",
                    start_char=chunk_start_char,
                    end_char=chunk_end_char,
                    word_count=len(chunk_text.split()),
                    char_count=len(chunk_text),
                    density_score=self._calculate_density(chunk_text),
                    coherence_score=1.0  # High coherence by design
                )
                
                chunks.append(Chunk(text=chunk_text, metadata=chunk_metadata))
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences + [sentence]
                current_size = sum(len(s) for s in current_chunk)
                chunk_start_char = chunk_end_char - sum(len(s) for s in overlap_sentences)
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_metadata = ChunkMetadata(
                chunk_id=self._generate_chunk_id(chunk_text, chunk_start_char),
                chunk_type="semantic",
                start_char=chunk_start_char,
                end_char=chunk_start_char + len(chunk_text),
                word_count=len(chunk_text.split()),
                char_count=len(chunk_text),
                density_score=self._calculate_density(chunk_text),
                coherence_score=1.0
            )
            chunks.append(Chunk(text=chunk_text, metadata=chunk_metadata))
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting - in production use spaCy or NLTK
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_coherence(self, chunk_text: str, new_sentence: str) -> float:
        """Calculate semantic coherence between chunk and new sentence"""
        if not self.embeddings_model:
            # Fallback to simple word overlap
            chunk_words = set(chunk_text.lower().split())
            sentence_words = set(new_sentence.lower().split())
            
            if not sentence_words:
                return 0.0
            
            overlap = len(chunk_words.intersection(sentence_words))
            return overlap / len(sentence_words)
        
        # TODO: Use actual embeddings for coherence
        return 0.5
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences for overlap"""
        if not sentences:
            return []
        
        # Calculate how many sentences to include for overlap
        overlap_size = 0
        overlap_sentences = []
        
        for sentence in reversed(sentences):
            overlap_size += len(sentence)
            overlap_sentences.insert(0, sentence)
            
            if overlap_size >= self.chunk_overlap:
                break
        
        return overlap_sentences


class StructuralChunker(BaseChunker):
    """Chunks text based on document structure"""
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Chunk based on structural elements"""
        # Parse document structure
        elements = self._parse_structure(text)
        
        chunks = []
        for element in elements:
            # Check if element needs to be split
            if len(element['text']) > self.max_chunk_size:
                # Split large elements
                sub_chunks = self._split_element(element)
                chunks.extend(sub_chunks)
            else:
                # Create chunk from element
                chunk_metadata = ChunkMetadata(
                    chunk_id=self._generate_chunk_id(element['text'], element['start']),
                    chunk_type=element['type'],
                    start_char=element['start'],
                    end_char=element['end'],
                    word_count=len(element['text'].split()),
                    char_count=len(element['text']),
                    density_score=self._calculate_density(element['text']),
                    structural_level=element['level'],
                    tags=element.get('tags', [])
                )
                
                chunks.append(Chunk(text=element['text'], metadata=chunk_metadata))
        
        # Establish parent-child relationships
        self._establish_hierarchy(chunks)
        
        return chunks
    
    def _parse_structure(self, text: str) -> List[Dict[str, Any]]:
        """Parse document structure into elements"""
        elements = []
        
        # Define patterns for different elements
        patterns = {
            'header': (r'^(#{1,6})\s+(.+)$', 'header'),
            'list': (r'^[\*\-\+]\s+(.+)$', 'list_item'),
            'numbered': (r'^\d+\.\s+(.+)$', 'numbered_item'),
            'code': (r'```[\s\S]*?```', 'code_block'),
            'table': (r'(\|.+\|[\s\S]*?\n)(?=\n|$)', 'table'),
            'quote': (r'^>\s+(.+)$', 'quote'),
            'paragraph': (r'((?:(?!^#|\n\n).)+)', 'paragraph')
        }
        
        # Track position
        current_pos = 0
        
        # Extract special blocks first (code, tables)
        special_blocks = []
        
        # Extract code blocks
        for match in re.finditer(patterns['code'][0], text, re.MULTILINE):
            special_blocks.append({
                'type': 'code_block',
                'text': match.group(0),
                'start': match.start(),
                'end': match.end(),
                'level': 0
            })
        
        # Extract tables
        for match in re.finditer(patterns['table'][0], text, re.MULTILINE):
            special_blocks.append({
                'type': 'table',
                'text': match.group(0),
                'start': match.start(),
                'end': match.end(),
                'level': 0
            })
        
        # Sort by position
        special_blocks.sort(key=lambda x: x['start'])
        
        # Now parse the rest
        lines = text.split('\n')
        line_start = 0
        
        for line in lines:
            line_end = line_start + len(line)
            
            # Skip if we're inside a special block
            in_special = any(
                block['start'] <= line_start < block['end'] 
                for block in special_blocks
            )
            
            if not in_special and line.strip():
                # Check headers
                header_match = re.match(patterns['header'][0], line)
                if header_match:
                    level = len(header_match.group(1))
                    elements.append({
                        'type': 'header',
                        'text': line,
                        'start': line_start,
                        'end': line_end,
                        'level': level,
                        'tags': [f'h{level}']
                    })
                # Check lists
                elif re.match(patterns['list'][0], line):
                    elements.append({
                        'type': 'list_item',
                        'text': line,
                        'start': line_start,
                        'end': line_end,
                        'level': 2
                    })
                # Regular paragraph
                else:
                    # Merge with previous paragraph if close
                    if (elements and 
                        elements[-1]['type'] == 'paragraph' and 
                        line_start - elements[-1]['end'] <= 1):
                        elements[-1]['text'] += '\n' + line
                        elements[-1]['end'] = line_end
                    else:
                        elements.append({
                            'type': 'paragraph',
                            'text': line,
                            'start': line_start,
                            'end': line_end,
                            'level': 3
                        })
            
            line_start = line_end + 1  # +1 for newline
        
        # Add special blocks in order
        all_elements = elements + special_blocks
        all_elements.sort(key=lambda x: x['start'])
        
        return all_elements
    
    def _split_element(self, element: Dict[str, Any]) -> List[Chunk]:
        """Split large structural element into smaller chunks"""
        chunks = []
        text = element['text']
        
        # Use sliding window for large elements
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk_text = text[i:i + self.chunk_size]
            
            if len(chunk_text) < self.min_chunk_size:
                continue
            
            chunk_metadata = ChunkMetadata(
                chunk_id=self._generate_chunk_id(chunk_text, element['start'] + i),
                chunk_type=element['type'],
                start_char=element['start'] + i,
                end_char=element['start'] + i + len(chunk_text),
                word_count=len(chunk_text.split()),
                char_count=len(chunk_text),
                density_score=self._calculate_density(chunk_text),
                structural_level=element['level'],
                tags=element.get('tags', [])
            )
            
            chunks.append(Chunk(text=chunk_text, metadata=chunk_metadata))
        
        return chunks
    
    def _establish_hierarchy(self, chunks: List[Chunk]):
        """Establish parent-child relationships between chunks"""
        # Sort by position and level
        sorted_chunks = sorted(
            chunks, 
            key=lambda c: (c.metadata.start_char, c.metadata.structural_level)
        )
        
        # Stack to track current parents at each level
        parent_stack = []
        
        for chunk in sorted_chunks:
            level = chunk.metadata.structural_level
            
            # Pop stack until we find appropriate parent level
            while parent_stack and parent_stack[-1][1] >= level:
                parent_stack.pop()
            
            # Set parent if exists
            if parent_stack:
                parent_chunk, _ = parent_stack[-1]
                chunk.metadata.parent_id = parent_chunk.metadata.chunk_id
                parent_chunk.metadata.children_ids.append(chunk.metadata.chunk_id)
            
            # Add to stack if it can be a parent
            if level < 3:  # Headers and major sections
                parent_stack.append((chunk, level))


class AdaptiveChunker(BaseChunker):
    """Adapts chunk size based on content characteristics"""
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Chunk with adaptive sizing"""
        # Analyze text characteristics
        characteristics = self._analyze_text(text)
        
        # Determine optimal chunk size
        optimal_size = self._determine_optimal_size(characteristics)
        
        # Use appropriate chunker based on content
        if characteristics['has_structure']:
            chunker = StructuralChunker(
                chunk_size=optimal_size,
                chunk_overlap=self.chunk_overlap
            )
        elif characteristics['high_density']:
            chunker = SemanticChunker(
                chunk_size=optimal_size,
                chunk_overlap=self.chunk_overlap
            )
        else:
            # Fall back to sliding window
            chunker = SlidingWindowChunker(
                chunk_size=optimal_size,
                chunk_overlap=self.chunk_overlap
            )
        
        chunks = chunker.chunk(text, metadata)
        
        # Add adaptive metadata
        for chunk in chunks:
            chunk.metadata.properties['adaptive_size'] = optimal_size
            chunk.metadata.properties['content_type'] = characteristics['primary_type']
        
        return chunks
    
    def _analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text characteristics"""
        characteristics = {
            'length': len(text),
            'avg_sentence_length': 0,
            'has_structure': False,
            'has_code': False,
            'has_tables': False,
            'high_density': False,
            'primary_type': 'general'
        }
        
        # Calculate average sentence length
        sentences = re.split(r'[.!?]+', text)
        if sentences:
            characteristics['avg_sentence_length'] = sum(
                len(s) for s in sentences
            ) / len(sentences)
        
        # Check for structure
        if re.search(r'^#{1,6}\s+', text, re.MULTILINE):
            characteristics['has_structure'] = True
            characteristics['primary_type'] = 'structured'
        
        # Check for code
        if re.search(r'```|^\s{4,}', text, re.MULTILINE):
            characteristics['has_code'] = True
            characteristics['primary_type'] = 'technical'
        
        # Check for tables
        if re.search(r'\|.*\|', text):
            characteristics['has_tables'] = True
        
        # Check density
        density = self._calculate_density(text)
        if density > 0.7:
            characteristics['high_density'] = True
            characteristics['primary_type'] = 'dense'
        
        return characteristics
    
    def _determine_optimal_size(self, characteristics: Dict[str, Any]) -> int:
        """Determine optimal chunk size based on characteristics"""
        base_size = self.chunk_size
        
        # Adjust based on content type
        if characteristics['primary_type'] == 'technical':
            # Larger chunks for code
            base_size = int(base_size * 1.5)
        elif characteristics['primary_type'] == 'dense':
            # Smaller chunks for dense content
            base_size = int(base_size * 0.8)
        elif characteristics['has_structure']:
            # Standard size for structured content
            pass
        
        # Adjust based on sentence length
        avg_sentence = characteristics['avg_sentence_length']
        if avg_sentence > 100:
            # Long sentences, reduce chunk size
            base_size = int(base_size * 0.9)
        elif avg_sentence < 50:
            # Short sentences, increase chunk size
            base_size = int(base_size * 1.1)
        
        # Ensure within bounds
        return max(
            self.min_chunk_size,
            min(self.max_chunk_size, base_size)
        )


class SlidingWindowChunker(BaseChunker):
    """Simple sliding window chunker with smart boundaries"""
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Chunk using sliding window"""
        chunks = []
        boundaries = self._detect_boundaries(text)
        
        i = 0
        while i < len(text):
            # Find chunk end
            chunk_end = min(i + self.chunk_size, len(text))
            
            # Adjust to nearest boundary
            nearest_boundary = min(
                boundaries,
                key=lambda b: abs(b - chunk_end) if b > i else float('inf')
            )
            
            if abs(nearest_boundary - chunk_end) < 50:  # Within 50 chars
                chunk_end = nearest_boundary
            
            # Extract chunk
            chunk_text = text[i:chunk_end]
            
            if len(chunk_text) >= self.min_chunk_size:
                chunk_metadata = ChunkMetadata(
                    chunk_id=self._generate_chunk_id(chunk_text, i),
                    chunk_type="sliding_window",
                    start_char=i,
                    end_char=chunk_end,
                    word_count=len(chunk_text.split()),
                    char_count=len(chunk_text),
                    density_score=self._calculate_density(chunk_text)
                )
                
                chunks.append(Chunk(text=chunk_text, metadata=chunk_metadata))
            
            # Move window
            i += self.chunk_size - self.chunk_overlap
        
        return chunks


class HybridChunker(BaseChunker):
    """Combines multiple chunking strategies"""
    
    def __init__(self, *args, strategies: Optional[List[ChunkingStrategy]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategies = strategies or [
            ChunkingStrategy.STRUCTURAL,
            ChunkingStrategy.SEMANTIC
        ]
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Chunk using multiple strategies and merge results"""
        all_chunks = []
        
        # Apply each strategy
        for strategy in self.strategies:
            if strategy == ChunkingStrategy.STRUCTURAL:
                chunker = StructuralChunker(
                    self.chunk_size, 
                    self.chunk_overlap
                )
            elif strategy == ChunkingStrategy.SEMANTIC:
                chunker = SemanticChunker(
                    self.chunk_size,
                    self.chunk_overlap
                )
            elif strategy == ChunkingStrategy.ADAPTIVE:
                chunker = AdaptiveChunker(
                    self.chunk_size,
                    self.chunk_overlap
                )
            else:
                continue
            
            chunks = chunker.chunk(text, metadata)
            all_chunks.extend(chunks)
        
        # Merge overlapping chunks
        merged_chunks = self._merge_chunks(all_chunks)
        
        return merged_chunks
    
    def _merge_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Merge overlapping chunks intelligently"""
        if not chunks:
            return []
        
        # Sort by start position
        sorted_chunks = sorted(chunks, key=lambda c: c.metadata.start_char)
        
        merged = []
        current = sorted_chunks[0]
        
        for chunk in sorted_chunks[1:]:
            # Check for overlap
            if chunk.metadata.start_char < current.metadata.end_char:
                # Merge if significant overlap
                overlap_ratio = (
                    current.metadata.end_char - chunk.metadata.start_char
                ) / chunk.metadata.char_count
                
                if overlap_ratio > 0.5:
                    # Merge chunks
                    current = self._merge_two_chunks(current, chunk)
                else:
                    merged.append(current)
                    current = chunk
            else:
                merged.append(current)
                current = chunk
        
        merged.append(current)
        return merged
    
    def _merge_two_chunks(self, chunk1: Chunk, chunk2: Chunk) -> Chunk:
        """Merge two chunks"""
        # Combine text (remove overlap)
        overlap_start = chunk2.metadata.start_char
        overlap_end = chunk1.metadata.end_char
        
        if overlap_start < overlap_end:
            # Remove overlapping part from chunk2
            overlap_chars = overlap_end - overlap_start
            merged_text = chunk1.text + chunk2.text[overlap_chars:]
        else:
            merged_text = chunk1.text + chunk2.text
        
        # Merge metadata
        merged_metadata = ChunkMetadata(
            chunk_id=self._generate_chunk_id(
                merged_text, 
                chunk1.metadata.start_char
            ),
            chunk_type="hybrid",
            start_char=chunk1.metadata.start_char,
            end_char=chunk2.metadata.end_char,
            word_count=len(merged_text.split()),
            char_count=len(merged_text),
            density_score=(
                chunk1.metadata.density_score + 
                chunk2.metadata.density_score
            ) / 2,
            coherence_score=(
                chunk1.metadata.coherence_score + 
                chunk2.metadata.coherence_score
            ) / 2,
            structural_level=min(
                chunk1.metadata.structural_level,
                chunk2.metadata.structural_level
            ),
            tags=list(set(chunk1.metadata.tags + chunk2.metadata.tags))
        )
        
        return Chunk(text=merged_text, metadata=merged_metadata)


def create_chunker(
    strategy: Union[str, ChunkingStrategy],
    **kwargs
) -> BaseChunker:
    """
    Factory function to create chunker.
    
    Args:
        strategy: Chunking strategy to use
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured chunker instance
    """
    if isinstance(strategy, str):
        strategy = ChunkingStrategy(strategy)
    
    if strategy == ChunkingStrategy.SEMANTIC:
        return SemanticChunker(**kwargs)
    elif strategy == ChunkingStrategy.STRUCTURAL:
        return StructuralChunker(**kwargs)
    elif strategy == ChunkingStrategy.ADAPTIVE:
        return AdaptiveChunker(**kwargs)
    elif strategy == ChunkingStrategy.SLIDING_WINDOW:
        return SlidingWindowChunker(**kwargs)
    elif strategy == ChunkingStrategy.HYBRID:
        return HybridChunker(**kwargs)
    else:
        # Default to sliding window
        return SlidingWindowChunker(**kwargs)


# Example usage
if __name__ == "__main__":
    # Test text with structure
    test_text = """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

## Types of Machine Learning

There are three main types of machine learning:

1. Supervised Learning
   - Uses labeled training data
   - Examples: classification, regression
   
2. Unsupervised Learning
   - Works with unlabeled data
   - Examples: clustering, dimensionality reduction
   
3. Reinforcement Learning
   - Learns through interaction with environment
   - Examples: game playing, robotics

### Code Example

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
X, y = load_data()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

## Conclusion

Machine learning continues to evolve rapidly, with new techniques and applications emerging regularly.
"""
    
    # Test different chunking strategies
    strategies = [
        ChunkingStrategy.STRUCTURAL,
        ChunkingStrategy.SEMANTIC,
        ChunkingStrategy.ADAPTIVE,
        ChunkingStrategy.HYBRID
    ]
    
    for strategy in strategies:
        print(f"\n{'='*50}")
        print(f"Testing {strategy.value} chunker")
        print('='*50)
        
        chunker = create_chunker(strategy, chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk(test_text)
        
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i+1}:")
            print(f"  Type: {chunk.metadata.chunk_type}")
            print(f"  Level: {chunk.metadata.structural_level}")
            print(f"  Size: {chunk.metadata.char_count} chars")
            print(f"  Text: {chunk.text[:100]}...")
            if chunk.metadata.parent_id:
                print(f"  Parent: {chunk.metadata.parent_id}")
            if chunk.metadata.children_ids:
                print(f"  Children: {chunk.metadata.children_ids}")