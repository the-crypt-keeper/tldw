# chunking_templates.py
"""
Modular chunking template system for flexible text chunking strategies.
This module provides a template-based approach to define and execute
complex chunking pipelines with preprocessing and postprocessing stages.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Protocol, Callable
from pydantic import BaseModel, Field, ValidationError
from loguru import logger
from jinja2.sandbox import SandboxedEnvironment


class ChunkingOperation(BaseModel):
    """Represents a single operation in a chunking pipeline stage."""
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)
    condition: Optional[str] = None  # Jinja2 expression for conditional execution


class ChunkingStage(BaseModel):
    """Represents a stage in the chunking pipeline."""
    stage: str  # 'preprocess', 'chunk', 'postprocess'
    operations: Optional[List[ChunkingOperation]] = None
    method: Optional[str] = None  # For 'chunk' stage
    options: Dict[str, Any] = Field(default_factory=dict)


class ChunkingTemplate(BaseModel):
    """Defines a complete chunking strategy template."""
    name: str
    description: Optional[str] = None
    base_method: str = "words"  # Default chunking method
    pipeline: List[ChunkingStage]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    parent_template: Optional[str] = None  # For template inheritance


class OperationProtocol(Protocol):
    """Protocol for chunking operations."""
    def __call__(self, text: str, chunks: List[str], options: Dict[str, Any]) -> Any:
        """Execute the operation."""
        ...


class ChunkingTemplateManager:
    """Manages loading, caching, and validation of chunking templates."""
    
    def __init__(self, 
                 templates_dir: Optional[Path] = None,
                 user_templates_dir: Optional[Path] = None):
        """
        Initialize the template manager.
        
        Args:
            templates_dir: Directory containing built-in templates
            user_templates_dir: Directory for user-defined templates
        """
        self.templates_dir = templates_dir or (Path(__file__).parent / "templates")
        self.user_templates_dir = user_templates_dir or self._get_user_templates_dir()
        
        # Template cache
        self._templates: Dict[str, ChunkingTemplate] = {}
        
        # Jinja2 sandbox for safe expression evaluation
        self._sandbox = SandboxedEnvironment(
            autoescape=True,
            enable_async=False,
        )
        
        # Operation registry
        self._operations: Dict[str, OperationProtocol] = {}
        self._register_builtin_operations()
        
        logger.info(f"ChunkingTemplateManager initialized with templates_dir: {self.templates_dir}")
    
    def _get_user_templates_dir(self) -> Path:
        """Get the user templates directory, creating it if necessary."""
        from ..config import get_cli_data_dir
        user_dir = get_cli_data_dir() / "chunking_templates"
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir
    
    def _register_builtin_operations(self):
        """Register built-in chunking operations."""
        # Text preprocessing operations
        self.register_operation("normalize_whitespace", self._op_normalize_whitespace)
        self.register_operation("extract_metadata", self._op_extract_metadata)
        self.register_operation("section_detection", self._op_section_detection)
        self.register_operation("code_block_detection", self._op_code_block_detection)
        
        # Postprocessing operations
        self.register_operation("add_context", self._op_add_context)
        self.register_operation("add_overlap", self._op_add_overlap)
        self.register_operation("filter_empty", self._op_filter_empty)
        self.register_operation("merge_small", self._op_merge_small)
    
    def register_operation(self, name: str, operation: OperationProtocol):
        """Register a custom operation."""
        self._operations[name] = operation
        logger.debug(f"Registered operation: {name}")
    
    def load_template(self, template_name: str) -> Optional[ChunkingTemplate]:
        """Load a template by name, checking cache first."""
        if template_name in self._templates:
            return self._templates[template_name]
        
        # Check user templates first, then built-in
        for templates_dir in [self.user_templates_dir, self.templates_dir]:
            template_path = templates_dir / f"{template_name}.json"
            if template_path.exists():
                try:
                    template = self._load_template_from_file(template_path)
                    if template:
                        # Handle template inheritance
                        if template.parent_template:
                            parent = self.load_template(template.parent_template)
                            if parent:
                                template = self._merge_templates(parent, template)
                        
                        self._templates[template_name] = template
                        logger.info(f"Loaded template: {template_name}")
                        return template
                except Exception as e:
                    logger.error(f"Error loading template {template_name}: {e}")
        
        logger.warning(f"Template not found: {template_name}")
        return None
    
    def _load_template_from_file(self, path: Path) -> Optional[ChunkingTemplate]:
        """Load a template from a JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return ChunkingTemplate(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Invalid template file {path}: {e}")
            return None
    
    def _merge_templates(self, parent: ChunkingTemplate, child: ChunkingTemplate) -> ChunkingTemplate:
        """Merge parent and child templates for inheritance."""
        # Start with parent's data
        merged_data = parent.dict()
        
        # Override with child's data
        child_data = child.dict(exclude_unset=True)
        
        # Merge pipeline stages intelligently
        if 'pipeline' in child_data:
            parent_stages = {stage['stage']: stage for stage in merged_data['pipeline']}
            for stage in child_data['pipeline']:
                stage_name = stage['stage']
                if stage_name in parent_stages:
                    # Merge stage options
                    parent_stages[stage_name]['options'].update(stage.get('options', {}))
                    # Override operations if provided
                    if 'operations' in stage:
                        parent_stages[stage_name]['operations'] = stage['operations']
                    if 'method' in stage:
                        parent_stages[stage_name]['method'] = stage['method']
                else:
                    parent_stages[stage_name] = stage
            merged_data['pipeline'] = list(parent_stages.values())
        
        # Update other fields
        for key in ['name', 'description', 'base_method', 'metadata']:
            if key in child_data:
                merged_data[key] = child_data[key]
        
        return ChunkingTemplate(**merged_data)
    
    def get_available_templates(self) -> List[str]:
        """Get list of available template names."""
        templates = set()
        
        # Scan both directories
        for templates_dir in [self.templates_dir, self.user_templates_dir]:
            if templates_dir.exists():
                for path in templates_dir.glob("*.json"):
                    templates.add(path.stem)
        
        return sorted(list(templates))
    
    def save_template(self, template: ChunkingTemplate, user_template: bool = True):
        """Save a template to disk."""
        save_dir = self.user_templates_dir if user_template else self.templates_dir
        save_path = save_dir / f"{template.name}.json"
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(template.dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Saved template: {template.name} to {save_path}")
        except Exception as e:
            logger.error(f"Error saving template {template.name}: {e}")
            raise
    
    def evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Safely evaluate a condition expression."""
        if not condition:
            return True
        
        try:
            tmpl = self._sandbox.from_string(f"{{{{ {condition} }}}}")
            result = tmpl.render(**context)
            return result.lower() in ('true', '1', 'yes')
        except Exception as e:
            logger.warning(f"Condition evaluation failed: {condition}, error: {e}")
            return False
    
    # Built-in operations
    def _op_normalize_whitespace(self, text: str, chunks: List[str], options: Dict[str, Any]) -> str:
        """Normalize whitespace in text."""
        import re
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        return text.strip()
    
    def _op_extract_metadata(self, text: str, chunks: List[str], options: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from text based on patterns."""
        import re
        metadata = {}
        patterns = options.get('patterns', [])
        
        for pattern_name in patterns:
            if pattern_name == 'abstract':
                match = re.search(r'(?i)abstract[:\s]*([^\n]+(?:\n(?![\n\w]*:)[^\n]+)*)', text)
                if match:
                    metadata['abstract'] = match.group(1).strip()
            elif pattern_name == 'keywords':
                match = re.search(r'(?i)keywords[:\s]*([^\n]+)', text)
                if match:
                    metadata['keywords'] = [k.strip() for k in match.group(1).split(',')]
            # Add more patterns as needed
        
        return metadata
    
    def _op_section_detection(self, text: str, chunks: List[str], options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect sections in text based on headers."""
        import re
        headers = options.get('headers', [])
        sections = []
        
        # Create regex pattern from headers
        if headers:
            pattern = '|'.join(re.escape(h) for h in headers)
            header_regex = re.compile(rf'^({pattern})\s*$', re.MULTILINE | re.IGNORECASE)
            
            matches = list(header_regex.finditer(text))
            for i, match in enumerate(matches):
                start = match.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                sections.append({
                    'header': match.group(1),
                    'start': start,
                    'end': end,
                    'content': text[start:end].strip()
                })
        
        return sections
    
    def _op_code_block_detection(self, text: str, chunks: List[str], options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect code blocks in text."""
        import re
        code_blocks = []
        
        # Detect fenced code blocks
        pattern = re.compile(r'```(\w*)\n(.*?)\n```', re.DOTALL)
        for match in pattern.finditer(text):
            code_blocks.append({
                'language': match.group(1) or 'text',
                'code': match.group(2),
                'start': match.start(),
                'end': match.end()
            })
        
        return code_blocks
    
    def _op_add_context(self, text: str, chunks: List[str], options: Dict[str, Any]) -> List[str]:
        """Add surrounding context to chunks."""
        context_size = options.get('context_size', 1)
        enhanced_chunks = []
        
        for i, chunk in enumerate(chunks):
            context_before = []
            context_after = []
            
            # Add context from previous chunks
            for j in range(max(0, i - context_size), i):
                context_before.append(f"[CONTEXT] {chunks[j][:100]}...")
            
            # Add context from following chunks
            for j in range(i + 1, min(len(chunks), i + context_size + 1)):
                context_after.append(f"[CONTEXT] {chunks[j][:100]}...")
            
            enhanced_chunk = '\n'.join(context_before + [chunk] + context_after)
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks
    
    def _op_add_overlap(self, text: str, chunks: List[str], options: Dict[str, Any]) -> List[str]:
        """Add overlap between consecutive chunks."""
        overlap_size = options.get('overlap_size', 50)  # characters
        enhanced_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i > 0:
                # Add end of previous chunk
                prev_end = chunks[i-1][-overlap_size:] if len(chunks[i-1]) > overlap_size else chunks[i-1]
                chunk = prev_end + "\n[OVERLAP]\n" + chunk
            
            if i < len(chunks) - 1:
                # Add beginning of next chunk
                next_start = chunks[i+1][:overlap_size] if len(chunks[i+1]) > overlap_size else chunks[i+1]
                chunk = chunk + "\n[OVERLAP]\n" + next_start
            
            enhanced_chunks.append(chunk)
        
        return enhanced_chunks
    
    def _op_filter_empty(self, text: str, chunks: List[str], options: Dict[str, Any]) -> List[str]:
        """Filter out empty or near-empty chunks."""
        min_length = options.get('min_length', 10)
        return [chunk for chunk in chunks if len(chunk.strip()) >= min_length]
    
    def _op_merge_small(self, text: str, chunks: List[str], options: Dict[str, Any]) -> List[str]:
        """Merge small chunks together."""
        min_size = options.get('min_size', 100)
        merged_chunks = []
        current_chunk = ""
        
        for chunk in chunks:
            if len(current_chunk) + len(chunk) < min_size:
                current_chunk += "\n" + chunk if current_chunk else chunk
            else:
                if current_chunk:
                    merged_chunks.append(current_chunk)
                current_chunk = chunk
        
        if current_chunk:
            merged_chunks.append(current_chunk)
        
        return merged_chunks


class ChunkingPipeline:
    """Executes chunking templates with multi-stage processing."""
    
    def __init__(self, template_manager: ChunkingTemplateManager):
        """Initialize the pipeline with a template manager."""
        self.template_manager = template_manager
        self.context: Dict[str, Any] = {}
    
    def execute(self, 
                text: str, 
                template: ChunkingTemplate,
                chunker_instance: Any,
                **kwargs) -> List[Dict[str, Any]]:
        """
        Execute a chunking template pipeline.
        
        Args:
            text: Text to chunk
            template: Chunking template to execute
            chunker_instance: Instance of the Chunker class
            **kwargs: Additional options passed to chunking
            
        Returns:
            List of chunks with metadata
        """
        logger.info(f"Executing chunking pipeline: {template.name}")
        
        # Initialize context
        self.context = {
            'text': text,
            'text_length': len(text),
            'template': template.name,
            'metadata': {},
            **kwargs
        }
        
        # Process pipeline stages
        processed_text = text
        chunks = []
        
        for stage in template.pipeline:
            logger.debug(f"Processing stage: {stage.stage}")
            
            if stage.stage == 'preprocess':
                processed_text = self._execute_preprocess(processed_text, stage)
            
            elif stage.stage == 'chunk':
                chunks = self._execute_chunk(processed_text, stage, chunker_instance)
            
            elif stage.stage == 'postprocess':
                chunks = self._execute_postprocess(processed_text, chunks, stage)
        
        # If no chunk stage was executed, use default
        if not chunks:
            method = template.base_method
            options = template.metadata.get('default_options', {})
            
            # Save current chunker options
            original_options = chunker_instance.options.copy()
            
            try:
                # Create a new options dict with original options updated by default options
                temp_options = original_options.copy()
                temp_options.update(options)
                
                # Ensure overlap is valid for the given max_size
                if 'max_size' in temp_options and 'overlap' in temp_options:
                    max_size = temp_options['max_size']
                    overlap = temp_options['overlap']
                    if overlap >= max_size:
                        # Adjust overlap to be less than max_size
                        temp_options['overlap'] = max(0, max_size - 1)
                        logger.debug(f"Adjusted overlap from {overlap} to {temp_options['overlap']} to be less than max_size {max_size}")
                
                # Update chunker options temporarily
                chunker_instance.options = temp_options
                chunks = chunker_instance.chunk_text(processed_text, method=method, use_template=False)
            finally:
                # Restore original options
                chunker_instance.options = original_options
        
        # Convert to expected format
        return self._format_chunks(chunks, template)
    
    def _execute_preprocess(self, text: str, stage: ChunkingStage) -> str:
        """Execute preprocessing operations."""
        if not stage.operations:
            return text
        
        for operation in stage.operations:
            if self.template_manager.evaluate_condition(operation.condition, self.context):
                op_func = self.template_manager._operations.get(operation.type)
                if op_func:
                    result = op_func(text, [], operation.params)
                    if isinstance(result, str):
                        text = result
                    elif isinstance(result, dict):
                        self.context['metadata'].update(result)
                    elif isinstance(result, list):
                        self.context[f'{operation.type}_results'] = result
                    logger.debug(f"Executed preprocess operation: {operation.type}")
                else:
                    logger.warning(f"Unknown operation: {operation.type}")
        
        return text
    
    def _execute_chunk(self, text: str, stage: ChunkingStage, chunker_instance: Any) -> List[Any]:
        """Execute the main chunking stage."""
        method = stage.method or 'words'
        options = stage.options.copy()
        
        # Map template options to chunker options
        if 'overlap' not in options and 'overlap_size' in options:
            options['overlap'] = options.pop('overlap_size')
        
        # Save current chunker options
        original_options = chunker_instance.options.copy()
        
        try:
            # Create a new options dict with original options updated by stage options
            temp_options = original_options.copy()
            temp_options.update(options)
            
            # Ensure overlap is valid for the given max_size
            if 'max_size' in temp_options and 'overlap' in temp_options:
                max_size = temp_options['max_size']
                overlap = temp_options['overlap']
                if overlap >= max_size:
                    # Adjust overlap to be less than max_size
                    temp_options['overlap'] = max(0, max_size - 1)
                    logger.debug(f"Adjusted overlap from {overlap} to {temp_options['overlap']} to be less than max_size {max_size}")
            
            # Update chunker options temporarily
            chunker_instance.options = temp_options
            
            # Set use_template=False to avoid infinite recursion
            chunks = chunker_instance.chunk_text(text, method=method, use_template=False)
        finally:
            # Restore original options
            chunker_instance.options = original_options
        
        # Convert to list of strings if needed (for dict-based chunks)
        if chunks and isinstance(chunks[0], dict):
            # Keep as is for now, will be handled in _format_chunks
            pass
        else:
            # Ensure chunks is a list
            if not isinstance(chunks, list):
                chunks = [chunks]
        
        # Handle operations if any
        if stage.operations:
            for operation in stage.operations:
                if self.template_manager.evaluate_condition(operation.condition, self.context):
                    op_func = self.template_manager._operations.get(operation.type)
                    if op_func:
                        result = op_func(text, chunks, operation.params)
                        if isinstance(result, list):
                            chunks = result
        
        return chunks
    
    def _execute_postprocess(self, text: str, chunks: List[str], stage: ChunkingStage) -> List[str]:
        """Execute postprocessing operations."""
        if not stage.operations:
            return chunks
        
        for operation in stage.operations:
            if self.template_manager.evaluate_condition(operation.condition, self.context):
                op_func = self.template_manager._operations.get(operation.type)
                if op_func:
                    result = op_func(text, chunks, operation.params)
                    if isinstance(result, list):
                        chunks = result
                    logger.debug(f"Executed postprocess operation: {operation.type}")
                else:
                    logger.warning(f"Unknown operation: {operation.type}")
        
        return chunks
    
    def _format_chunks(self, chunks: List[Any], template: ChunkingTemplate) -> List[Dict[str, Any]]:
        """Format chunks into the expected output format."""
        formatted_chunks = []
        
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, dict):
                # Already formatted
                formatted_chunks.append(chunk)
            else:
                # Convert to standard format
                formatted_chunks.append({
                    'text': str(chunk),
                    'metadata': {
                        'chunk_index': i + 1,
                        'total_chunks': len(chunks),
                        'template': template.name,
                        'template_metadata': self.context.get('metadata', {}),
                        **template.metadata
                    }
                })
        
        return formatted_chunks