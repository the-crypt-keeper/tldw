# rolling_summarize.py
"""
Rolling summarization chunking strategy.

This strategy creates chunks by progressively summarizing content using an LLM,
building a rolling context that maintains continuity across chunk boundaries.
"""

from typing import List, Dict, Any, Optional, Callable
from loguru import logger
from ..base import BaseChunkingStrategy


class RollingSummarizeStrategy(BaseChunkingStrategy):
    """
    Implements rolling summarization chunking.
    
    This strategy:
    1. Splits text into initial segments
    2. Summarizes each segment with rolling context
    3. Maintains continuity between chunks through overlapping summaries
    """
    
    def __init__(self, 
                 language: str = 'en',
                 llm_call_func: Optional[Callable] = None,
                 llm_config: Optional[Dict[str, Any]] = None):
        """
        Initialize rolling summarize strategy.
        
        Args:
            language: Language code for text processing
            llm_call_func: Function to call LLM for summarization
            llm_config: Configuration for LLM calls
        """
        super().__init__(language)
        self.llm_call_func = llm_call_func
        self.llm_config = llm_config or {}
        
    def chunk(self, 
              text: str, 
              max_size: int,
              overlap: int = 0,
              **options) -> List[str]:
        """
        Chunk text using rolling summarization.
        
        Args:
            text: Text to chunk
            max_size: Maximum size of each chunk (in sentences for initial split)
            overlap: Number of sentences to overlap (used for context)
            **options: Additional options:
                - summarization_detail: Float 0.0-1.0, how detailed summaries should be
                - preserve_structure: Whether to preserve document structure
                - context_window: Number of previous summaries to include as context
                
        Returns:
            List of summarized chunks
        """
        if not self.validate_parameters(text, max_size, overlap):
            return []
            
        # Get options
        summarization_detail = options.get('summarization_detail', 0.5)
        preserve_structure = options.get('preserve_structure', True)
        context_window = options.get('context_window', 2)
        
        # Split text into sentences for initial segmentation
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
            
        # Group sentences into segments
        segments = []
        current_segment = []
        
        for sentence in sentences:
            current_segment.append(sentence)
            if len(current_segment) >= max_size:
                segments.append(' '.join(current_segment))
                # Keep overlap sentences for next segment
                if overlap > 0:
                    current_segment = current_segment[-overlap:]
                else:
                    current_segment = []
        
        # Add remaining sentences as final segment
        if current_segment:
            segments.append(' '.join(current_segment))
        
        # If no LLM function provided, return segments as-is
        if not self.llm_call_func:
            logger.warning("No LLM function provided for rolling_summarize, returning raw segments")
            return segments
            
        # Process segments with rolling summarization
        summarized_chunks = []
        rolling_context = []
        
        for i, segment in enumerate(segments):
            # Build context from previous summaries
            context = ""
            if rolling_context:
                # Use last N summaries as context
                context_items = rolling_context[-context_window:]
                context = "Previous context:\n" + "\n".join(context_items) + "\n\n"
            
            # Create prompt for summarization
            prompt = self._create_summarization_prompt(
                segment, 
                context,
                summarization_detail,
                preserve_structure,
                i == 0  # First segment
            )
            
            try:
                # Call LLM for summarization
                summary = self._call_llm(prompt)
                
                if summary:
                    summarized_chunks.append(summary)
                    # Add to rolling context (keep it concise)
                    context_summary = self._create_context_summary(summary)
                    rolling_context.append(context_summary)
                else:
                    # Fallback to original segment if summarization fails
                    logger.warning(f"Summarization failed for segment {i}, using original text")
                    summarized_chunks.append(segment)
                    rolling_context.append(segment[:200] + "...")  # Truncate for context
                    
            except Exception as e:
                logger.error(f"Error during summarization of segment {i}: {e}")
                # Fallback to original segment
                summarized_chunks.append(segment)
                rolling_context.append(segment[:200] + "...")
        
        return summarized_chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - can be enhanced with pysbd
        import re
        
        # Handle common abbreviations
        text = re.sub(r'\b(Mr|Mrs|Dr|Ms|Prof|Sr|Jr)\.\s*', r'\1<DOT> ', text)
        text = re.sub(r'\b(Inc|Ltd|Corp|Co)\.\s*', r'\1<DOT> ', text)
        text = re.sub(r'\b(i\.e|e\.g|etc|vs|viz)\.\s*', r'\1<DOT> ', text)
        
        # Split on sentence endings
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore dots
        sentences = [s.replace('<DOT>', '.') for s in sentences]
        
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _create_summarization_prompt(self, 
                                    segment: str, 
                                    context: str,
                                    detail_level: float,
                                    preserve_structure: bool,
                                    is_first: bool) -> str:
        """Create prompt for LLM summarization."""
        
        # Determine target length based on detail level
        if detail_level < 0.3:
            length_instruction = "very brief summary (1-2 sentences)"
        elif detail_level < 0.6:
            length_instruction = "concise summary (3-4 sentences)"
        elif detail_level < 0.8:
            length_instruction = "detailed summary (5-7 sentences)"
        else:
            length_instruction = "comprehensive summary (8-10 sentences)"
        
        # Build prompt
        if is_first:
            prompt = f"""Please provide a {length_instruction} of the following text.
Focus on the main points and key information."""
        else:
            prompt = f"""Continue summarizing the document. Provide a {length_instruction} of the following text.
Maintain continuity with the previous context."""
        
        if context:
            prompt += f"\n\n{context}"
            
        if preserve_structure:
            prompt += "\nPreserve any important structural elements (headings, lists, etc.) in the summary."
            
        prompt += f"\n\nText to summarize:\n{segment}\n\nSummary:"
        
        return prompt
    
    def _create_context_summary(self, summary: str) -> str:
        """Create a brief context summary for rolling context."""
        # Take first 150 characters or first sentence for context
        if len(summary) <= 150:
            return summary
        
        # Try to break at sentence boundary
        first_period = summary.find('. ', 0, 150)
        if first_period > 0:
            return summary[:first_period + 1]
        
        return summary[:150] + "..."
    
    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM for summarization."""
        if not self.llm_call_func:
            return None
            
        try:
            # Prepare config for LLM call
            config = self.llm_config.copy()
            
            # Use the provided LLM function
            # The analyze function signature: analyze(api_name, input_data, custom_prompt_arg, api_key, system_message, temp, ...)
            result = self.llm_call_func(
                config.get('api_name', 'openai'),  # api_name
                prompt,  # input_data
                None,  # custom_prompt_arg (use None since prompt already contains instructions)
                config.get('api_key'),  # api_key
                config.get('system_message', "You are a helpful assistant that creates concise, accurate summaries."),  # system_message
                config.get('temp', 0.3),  # temp
                False,  # streaming
                False,  # recursive_summarization
                False,  # chunked_summarization
                None  # chunk_options
            )
            
            if result and isinstance(result, tuple) and len(result) > 0:
                return result[0]  # Extract summary from result tuple
            elif isinstance(result, str):
                return result
            else:
                logger.warning(f"Unexpected LLM response format: {type(result)}")
                return None
                
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return None