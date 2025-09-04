# templates.py
"""
Template-based chunking system for flexible processing pipelines.
Provides a way to define reusable chunking strategies with preprocessing
and postprocessing stages.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from loguru import logger
import re

from .base import ChunkerConfig
from .chunker import Chunker
from .exceptions import TemplateError


@dataclass
class TemplateStage:
    """Represents a stage in the chunking pipeline."""
    name: str  # 'preprocess', 'chunk', 'postprocess'
    operations: List[Dict[str, Any]] = field(default_factory=list)
    enabled: bool = True
    

@dataclass 
class ChunkingTemplate:
    """Defines a complete chunking strategy template."""
    name: str
    description: str = ""
    base_method: str = "words"  # Default chunking method
    stages: List[TemplateStage] = field(default_factory=list)
    default_options: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    

class TemplateProcessor:
    """Processes text through template-defined chunking pipelines."""
    
    def __init__(self):
        """Initialize the template processor."""
        self._operations: Dict[str, Callable] = {}
        self._register_builtin_operations()
        self._chunker = None
        
        logger.debug("TemplateProcessor initialized")
    
    def _get_chunker(self) -> Chunker:
        """Get or create chunker instance."""
        if self._chunker is None:
            self._chunker = Chunker()
        return self._chunker
    
    def _register_builtin_operations(self):
        """Register built-in preprocessing and postprocessing operations."""
        # Preprocessing operations
        self.register_operation("normalize_whitespace", self._normalize_whitespace)
        self.register_operation("remove_headers", self._remove_headers)
        self.register_operation("extract_sections", self._extract_sections)
        self.register_operation("clean_markdown", self._clean_markdown)
        self.register_operation("detect_language", self._detect_language)
        
        # Postprocessing operations
        self.register_operation("add_overlap", self._add_overlap)
        self.register_operation("filter_empty", self._filter_empty)
        self.register_operation("merge_small", self._merge_small)
        self.register_operation("add_metadata", self._add_metadata)
        self.register_operation("format_chunks", self._format_chunks)
    
    def register_operation(self, name: str, func: Callable):
        """
        Register a custom operation.
        
        Args:
            name: Operation name
            func: Function to execute (signature: func(data, options) -> data)
        """
        self._operations[name] = func
        logger.debug(f"Registered operation: {name}")
    
    def process_template(self,
                        text: str,
                        template: ChunkingTemplate,
                        **options) -> List[str]:
        """
        Process text through a template pipeline.
        
        Args:
            text: Text to process
            template: Template defining the pipeline
            **options: Additional options overriding template defaults
            
        Returns:
            List of processed chunks
        """
        # Merge options with template defaults
        final_options = {**template.default_options, **options}
        
        # Initialize data that flows through pipeline
        data = {"text": text, "chunks": [], "metadata": {}}
        
        # Process each stage
        for stage in template.stages:
            if not stage.enabled:
                continue
            
            logger.debug(f"Processing stage: {stage.name}")
            
            if stage.name == "preprocess":
                data = self._run_preprocess_stage(data, stage, final_options)
            elif stage.name == "chunk":
                data = self._run_chunk_stage(data, stage, final_options, template.base_method)
            elif stage.name == "postprocess":
                data = self._run_postprocess_stage(data, stage, final_options)
            else:
                logger.warning(f"Unknown stage: {stage.name}")
        
        return data.get("chunks", [])
    
    def _run_preprocess_stage(self,
                             data: Dict[str, Any],
                             stage: TemplateStage,
                             options: Dict[str, Any]) -> Dict[str, Any]:
        """Run preprocessing operations on text."""
        text = data["text"]
        
        for operation in stage.operations:
            op_name = operation.get("type")
            op_options = {**options, **operation.get("params", {})}
            
            if op_name in self._operations:
                logger.debug(f"Running preprocessing: {op_name}")
                result = self._operations[op_name](text, op_options)
                
                # Handle different return types
                if isinstance(result, str):
                    text = result
                elif isinstance(result, dict):
                    text = result.get("text", text)
                    data["metadata"].update(result.get("metadata", {}))
            else:
                logger.warning(f"Unknown operation: {op_name}")
        
        data["text"] = text
        return data
    
    def _run_chunk_stage(self,
                        data: Dict[str, Any],
                        stage: TemplateStage,
                        options: Dict[str, Any],
                        base_method: str) -> Dict[str, Any]:
        """Run chunking operation."""
        text = data["text"]
        
        # Get chunking parameters
        chunk_ops = stage.operations[0] if stage.operations else {}
        method = chunk_ops.get("method", base_method)
        max_size = chunk_ops.get("max_size", options.get("max_size", 400))
        overlap = chunk_ops.get("overlap", options.get("overlap", 50))
        
        # Perform chunking
        chunker = self._get_chunker()
        chunks = chunker.chunk_text(
            text=text,
            method=method,
            max_size=max_size,
            overlap=overlap,
            **chunk_ops.get("params", {})
        )
        
        data["chunks"] = chunks
        return data
    
    def _run_postprocess_stage(self,
                              data: Dict[str, Any],
                              stage: TemplateStage,
                              options: Dict[str, Any]) -> Dict[str, Any]:
        """Run postprocessing operations on chunks."""
        chunks = data.get("chunks", [])
        
        for operation in stage.operations:
            op_name = operation.get("type")
            op_options = {**options, **operation.get("params", {})}
            
            if op_name in self._operations:
                logger.debug(f"Running postprocessing: {op_name}")
                chunks = self._operations[op_name](chunks, op_options)
            else:
                logger.warning(f"Unknown operation: {op_name}")
        
        data["chunks"] = chunks
        return data
    
    # Built-in preprocessing operations
    def _normalize_whitespace(self, text: str, options: Dict[str, Any]) -> str:
        """Normalize whitespace in text."""
        # Replace multiple spaces with single space
        text = re.sub(r'[ \t]+', ' ', text)
        # Normalize line breaks
        text = re.sub(r'\r\n', '\n', text)
        # Remove excessive line breaks
        max_breaks = options.get("max_line_breaks", 2)
        text = re.sub(r'\n{' + str(max_breaks + 1) + ',}', '\n' * max_breaks, text)
        return text.strip()
    
    def _remove_headers(self, text: str, options: Dict[str, Any]) -> str:
        """Remove headers/footers from text."""
        patterns = options.get("patterns", [])
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE)
        return text
    
    def _extract_sections(self, text: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Extract sections from text and store as metadata."""
        section_pattern = options.get("pattern", r'^#+\s+(.+)$')
        sections = []
        
        for match in re.finditer(section_pattern, text, re.MULTILINE):
            sections.append({
                "title": match.group(1),
                "position": match.start()
            })
        
        return {
            "text": text,
            "metadata": {"sections": sections}
        }
    
    def _clean_markdown(self, text: str, options: Dict[str, Any]) -> str:
        """Clean markdown formatting from text."""
        if options.get("remove_links", False):
            # Remove markdown links but keep text
            text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        if options.get("remove_images", False):
            # Remove markdown images
            text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', text)
        
        if options.get("remove_formatting", False):
            # Remove bold/italic
            text = re.sub(r'\*{1,2}([^\*]+)\*{1,2}', r'\1', text)
            text = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', text)
        
        return text
    
    def _detect_language(self, text: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Detect text language and store as metadata."""
        # Simplified language detection based on character patterns
        language = "en"  # Default
        
        # Check for common non-English patterns
        if re.search(r'[\u4e00-\u9fff]', text):
            language = "zh"  # Chinese
        elif re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
            language = "ja"  # Japanese
        elif re.search(r'[\uac00-\ud7af]', text):
            language = "ko"  # Korean
        elif re.search(r'[\u0600-\u06ff]', text):
            language = "ar"  # Arabic
        
        return {
            "text": text,
            "metadata": {"detected_language": language}
        }
    
    # Built-in postprocessing operations
    def _add_overlap(self, chunks: List[str], options: Dict[str, Any]) -> List[str]:
        """Add overlap context between chunks."""
        overlap_size = options.get("size", 50)  # characters
        overlap_marker = options.get("marker", "")
        
        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            enhanced_chunk = chunk
            
            # Add end of previous chunk
            if i > 0 and overlap_size > 0:
                prev_end = chunks[i-1][-overlap_size:]
                if overlap_marker:
                    enhanced_chunk = f"{overlap_marker}\n{prev_end}\n{overlap_marker}\n{enhanced_chunk}"
                else:
                    enhanced_chunk = prev_end + enhanced_chunk
            
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks
    
    def _filter_empty(self, chunks: List[str], options: Dict[str, Any]) -> List[str]:
        """Filter out empty or too-small chunks."""
        min_length = options.get("min_length", 10)
        return [chunk for chunk in chunks if len(chunk.strip()) >= min_length]
    
    def _merge_small(self, chunks: List[str], options: Dict[str, Any]) -> List[str]:
        """Merge small chunks together."""
        min_size = options.get("min_size", 100)
        separator = options.get("separator", "\n\n")
        
        merged_chunks = []
        current_chunk = ""
        
        for chunk in chunks:
            if not current_chunk:
                current_chunk = chunk
            elif len(current_chunk) < min_size:
                # Keep merging if current chunk is still too small
                current_chunk = current_chunk + separator + chunk
            else:
                # Current chunk is large enough, save it and start new one
                merged_chunks.append(current_chunk)
                current_chunk = chunk
        
        if current_chunk:
            merged_chunks.append(current_chunk)
        
        return merged_chunks
    
    def _add_metadata(self, chunks: List[str], options: Dict[str, Any]) -> List[str]:
        """Add metadata to chunks."""
        prefix = options.get("prefix", "")
        suffix = options.get("suffix", "")
        
        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            enhanced = chunk
            
            if prefix:
                enhanced = prefix.format(index=i, total=len(chunks)) + enhanced
            if suffix:
                enhanced = enhanced + suffix.format(index=i, total=len(chunks))
            
            enhanced_chunks.append(enhanced)
        
        return enhanced_chunks
    
    def _format_chunks(self, chunks: List[str], options: Dict[str, Any]) -> List[str]:
        """Format chunks according to template."""
        template_str = options.get("template", "{chunk}")
        
        formatted_chunks = []
        for i, chunk in enumerate(chunks):
            formatted = template_str.format(
                chunk=chunk,
                index=i,
                total=len(chunks)
            )
            formatted_chunks.append(formatted)
        
        return formatted_chunks


class TemplateManager:
    """Manages loading and saving of chunking templates."""
    
    def __init__(self, templates_dir: Optional[Path] = None):
        """
        Initialize template manager.
        
        Args:
            templates_dir: Directory containing template files
        """
        self.templates_dir = templates_dir or Path(__file__).parent / "template_library"
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        self._cache: Dict[str, ChunkingTemplate] = {}
        self._processor = TemplateProcessor()
        
        # Load built-in templates
        self._load_builtin_templates()
        
        logger.info(f"TemplateManager initialized with dir: {self.templates_dir}")
    
    def _load_builtin_templates(self):
        """Load built-in template definitions."""
        # Academic paper template
        self.register_template(ChunkingTemplate(
            name="academic_paper",
            description="Template for processing academic papers",
            base_method="sentences",
            stages=[
                TemplateStage("preprocess", [
                    {"type": "normalize_whitespace", "params": {"max_line_breaks": 2}},
                    {"type": "extract_sections", "params": {"pattern": r"^#+\s+(.+)$"}},
                ]),
                TemplateStage("chunk", [
                    {"method": "sentences", "max_size": 5, "overlap": 1}
                ]),
                TemplateStage("postprocess", [
                    {"type": "filter_empty", "params": {"min_length": 20}},
                    {"type": "merge_small", "params": {"min_size": 200}},
                ])
            ],
            default_options={"max_size": 5, "overlap": 1}
        ))
        
        # Code documentation template
        self.register_template(ChunkingTemplate(
            name="code_documentation",
            description="Template for processing code documentation",
            base_method="structure_aware",
            stages=[
                TemplateStage("preprocess", [
                    {"type": "clean_markdown", "params": {"remove_images": True}},
                ]),
                TemplateStage("chunk", [
                    {"method": "structure_aware", "max_size": 500, "overlap": 50,
                     "params": {"preserve_code_blocks": True, "preserve_headers": True}}
                ]),
                TemplateStage("postprocess", [
                    {"type": "filter_empty", "params": {"min_length": 50}},
                ])
            ]
        ))
        
        # Chat conversation template
        self.register_template(ChunkingTemplate(
            name="chat_conversation",
            description="Template for processing chat conversations",
            base_method="sentences",
            stages=[
                TemplateStage("preprocess", [
                    {"type": "normalize_whitespace", "params": {"max_line_breaks": 1}},
                ]),
                TemplateStage("chunk", [
                    {"method": "sentences", "max_size": 10, "overlap": 2}
                ]),
                TemplateStage("postprocess", [
                    {"type": "add_overlap", "params": {"size": 100, "marker": "---"}},
                ])
            ]
        ))
    
    def register_template(self, template: ChunkingTemplate):
        """Register a template in the cache."""
        self._cache[template.name] = template
        logger.debug(f"Registered template: {template.name}")
    
    def get_template(self, name: str) -> Optional[ChunkingTemplate]:
        """Get a template by name."""
        # Check cache first
        if name in self._cache:
            return self._cache[name]
        
        # Try to load from file
        template_file = self.templates_dir / f"{name}.json"
        if template_file.exists():
            try:
                return self.load_template(template_file)
            except Exception as e:
                logger.error(f"Failed to load template {name}: {e}")
        
        return None
    
    def load_template(self, path: Path) -> ChunkingTemplate:
        """Load a template from file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to template object
        stages = []
        for stage_data in data.get("stages", []):
            stages.append(TemplateStage(
                name=stage_data["name"],
                operations=stage_data.get("operations", []),
                enabled=stage_data.get("enabled", True)
            ))
        
        template = ChunkingTemplate(
            name=data["name"],
            description=data.get("description", ""),
            base_method=data.get("base_method", "words"),
            stages=stages,
            default_options=data.get("default_options", {}),
            metadata=data.get("metadata", {})
        )
        
        # Cache it
        self._cache[template.name] = template
        return template
    
    def save_template(self, template: ChunkingTemplate, path: Optional[Path] = None):
        """Save a template to file."""
        if path is None:
            path = self.templates_dir / f"{template.name}.json"
        
        # Convert to JSON-serializable format
        data = {
            "name": template.name,
            "description": template.description,
            "base_method": template.base_method,
            "stages": [
                {
                    "name": stage.name,
                    "operations": stage.operations,
                    "enabled": stage.enabled
                }
                for stage in template.stages
            ],
            "default_options": template.default_options,
            "metadata": template.metadata
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved template {template.name} to {path}")
    
    def list_templates(self) -> List[str]:
        """List available template names."""
        templates = set(self._cache.keys())
        
        # Add templates from files
        if self.templates_dir.exists():
            for path in self.templates_dir.glob("*.json"):
                templates.add(path.stem)
        
        return sorted(list(templates))
    
    def process(self, text: str, template_name: str, **options) -> List[str]:
        """
        Process text using a named template.
        
        Args:
            text: Text to process
            template_name: Name of template to use
            **options: Additional options
            
        Returns:
            List of processed chunks
            
        Raises:
            TemplateError: If template not found
        """
        template = self.get_template(template_name)
        if not template:
            raise TemplateError(f"Template not found: {template_name}")
        
        return self._processor.process_template(text, template, **options)