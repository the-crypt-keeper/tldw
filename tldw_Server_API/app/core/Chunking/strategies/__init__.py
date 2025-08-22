"""
Chunking strategies for different text processing methods.
Each strategy implements the ChunkingStrategy protocol from base.py.
"""

from typing import Dict, Type, Any
from ..base import ChunkingStrategy

# Import strategies as they are implemented
from .words import WordChunkingStrategy
from .sentences import SentenceChunkingStrategy
from .tokens import TokenChunkingStrategy
from .structure_aware import StructureAwareChunkingStrategy
from .semantic import SemanticChunkingStrategy
from .json_xml import JSONChunkingStrategy, XMLChunkingStrategy

# Strategy registry will be populated as strategies are implemented
STRATEGY_REGISTRY: Dict[str, Type[ChunkingStrategy]] = {
    'words': WordChunkingStrategy,
    'sentences': SentenceChunkingStrategy,
    'tokens': TokenChunkingStrategy,
    'structure_aware': StructureAwareChunkingStrategy,
    'semantic': SemanticChunkingStrategy,
    'json': JSONChunkingStrategy,
    'xml': XMLChunkingStrategy,
}

def get_strategy(name: str) -> Type[ChunkingStrategy]:
    """
    Get a chunking strategy by name.
    
    Args:
        name: Strategy name
        
    Returns:
        Strategy class
        
    Raises:
        ValueError: If strategy not found
    """
    if name not in STRATEGY_REGISTRY:
        available = ', '.join(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown strategy: {name}. Available: {available}")
    return STRATEGY_REGISTRY[name]

__all__ = ['STRATEGY_REGISTRY', 'get_strategy']