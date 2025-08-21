# Chunk_Lib.py
#########################################
# Chunking Library
# This library is used to perform chunking of input files.
# Currently, uses naive approaches. Nothing fancy.
#
####
import hashlib
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Generator
import xml.etree.ElementTree as ET
#
# Import 3rd party
from loguru import logger

# Import optional dependencies
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    # Define placeholder classes for when langdetect is not available
    class LangDetectException(Exception):
        pass
    def detect(text):
        raise ImportError("langdetect is not installed. Install with: pip install langdetect")

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    # Define placeholder for when nltk is not available
    def sent_tokenize(text):
        # Simple fallback - split on common sentence endings
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

#
# Import Local
from tldw_chatbook.config import load_settings, get_cli_setting
from .language_chunkers import LanguageChunkerFactory
from .token_chunker import create_token_chunker
from .chunking_templates import ChunkingTemplateManager, ChunkingPipeline, ChunkingTemplate
from ..Metrics.metrics_logger import log_counter, log_histogram
#
#######################################################################################################################
# Custom Exceptions
class ChunkingError(Exception):
    """Base exception for chunking errors."""
    pass

class InvalidChunkingMethodError(ChunkingError):
    """Raised when an invalid chunking method is specified."""
    pass

class InvalidInputError(ChunkingError):
    """Raised for invalid input data, e.g., bad JSON."""
    pass

class LanguageDetectionError(ChunkingError):
    """Raised when language detection fails critically."""
    pass

class MemoryLimitError(ChunkingError):
    """Raised when input exceeds memory limits."""
    pass

#######################################################################################################################
# Constants and Limits
#

# Maximum limits for chunk sizes to prevent memory issues
MAX_CHUNK_SIZE_WORDS = 10000  # Maximum words per chunk
MAX_CHUNK_SIZE_SENTENCES = 1000  # Maximum sentences per chunk
MAX_CHUNK_SIZE_PARAGRAPHS = 100  # Maximum paragraphs per chunk
MAX_CHUNK_SIZE_TOKENS = 10000  # Maximum tokens per chunk
MAX_DOCUMENT_SIZE_MB = 100  # Maximum document size in MB
MAX_DOCUMENT_SIZE_BYTES = MAX_DOCUMENT_SIZE_MB * 1024 * 1024  # In bytes

#######################################################################################################################
# Config Settings & NLTK
#

def ensure_nltk_data():
    if not NLTK_AVAILABLE:
        logger.debug("NLTK not available, skipping punkt tokenizer check")
        return
        
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logger.info("NLTK 'punkt' tokenizer not found. Downloading...")
        try:
            nltk.download('punkt')
            logger.info("'punkt' downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download 'punkt': {e}")
            # Depending on how critical this is, you might raise an error or just warn
            # For now, we'll let it proceed, and sent_tokenize will fail later if needed.
ensure_nltk_data()


### REWRITTEN SECTION: New Configuration Loading ###
# This section replaces the old `load_and_log_configs` function and `_global_config` variable.
# We now build the default options dictionary directly using the central `get_cli_setting` function.

def _get_bool_setting(section: str, key: str, default: bool) -> bool:
    """Helper to reliably parse boolean values from config which might be strings."""
    val = get_cli_setting(section, key, default)
    if isinstance(val, bool):
        return val
    # Handles string representations like 'true', '1', 'yes'
    return str(val).lower() in ('true', '1', 't', 'y', 'yes')

# Load configuration from the central config system.
# This dictionary holds the default values for chunking, loaded from the config file.
# The Chunker class will use these as its base, allowing for per-instance overrides.
_default_chunk_options_from_config = {
    'method': get_cli_setting('chunking_config', 'chunking_method', 'words'),
    'max_size': int(get_cli_setting('chunking_config', 'chunk_max_size', 400)),
    'overlap': int(get_cli_setting('chunking_config', 'chunk_overlap', 200)),
    'adaptive': _get_bool_setting('chunking_config', 'adaptive_chunking', False),
    'multi_level': _get_bool_setting('chunking_config', 'multi_level', False),
    'language': get_cli_setting('chunking_config', 'chunk_language', None), # Can be None, will be auto-detected
    'custom_chapter_pattern': get_cli_setting('chunking_config', 'custom_chapter_pattern', None),
    'semantic_similarity_threshold': float(get_cli_setting('chunking_config', 'semantic_similarity_threshold', 0.5)),
    'semantic_overlap_sentences': int(get_cli_setting('chunking_config', 'semantic_overlap_sentences', 3)),
    'base_adaptive_chunk_size': int(get_cli_setting('chunking_config', 'base_adaptive_chunk_size', 1000)),
    'min_adaptive_chunk_size': int(get_cli_setting('chunking_config', 'min_adaptive_chunk_size', 500)),
    'max_adaptive_chunk_size': int(get_cli_setting('chunking_config', 'max_adaptive_chunk_size', 2000)),
    'tokenizer_name_or_path': get_cli_setting('chunking_config', 'tokenizer_name_or_path', "gpt2"),
    'summarization_detail': float(get_cli_setting('chunking_config', 'summarization_detail', 0.5)),
    'summarize_min_chunk_tokens': int(get_cli_setting('chunking_config', 'summarize_min_chunk_tokens', 500)),
    'summarize_chunk_delimiter': get_cli_setting('chunking_config', 'summarize_chunk_delimiter', "."),
    'summarize_recursively': _get_bool_setting('chunking_config', 'summarize_recursively', False),
    'summarize_verbose': _get_bool_setting('chunking_config', 'summarize_verbose', False),
    'summarize_system_prompt': get_cli_setting('chunking_config', 'summarize_system_prompt', "Rewrite this text in summarized form."),
    'summarize_additional_instructions': get_cli_setting('chunking_config', 'summarize_additional_instructions', None),
    'summarize_temperature': float(get_cli_setting('chunking_config', 'summarize_temperature', 0.1)),
    'summarization_llm_provider': get_cli_setting('chunking_config', 'summarization_llm_provider', 'openai'),
    'summarization_llm_model': get_cli_setting('chunking_config', 'summarization_llm_model', 'gpt-4o'),
    'template': get_cli_setting('chunking_config', 'template', None)  # Template name if using template-based chunking
}
logger.info("Chunking library defaults loaded from config.")
logger.debug(f"Default chunking options: {_default_chunk_options_from_config}")


# Expose the library's default options for other modules (e.g., API endpoints) to use
DEFAULT_CHUNK_OPTIONS = _default_chunk_options_from_config.copy()

#
# End of settings
#######################################################################################################################
#
# Functions:

class Chunker:
    def __init__(self,
                 options: Optional[Dict[str, Any]] = None,
                 tokenizer_name_or_path: str = "gpt2",
                 template: Optional[str] = None,
                 template_manager: Optional[ChunkingTemplateManager] = None,
                 # Specific methods needing LLMs will take them as args or use a callback.
                 ):
        """
        Initializes the Chunker.

        Args:
            options (Optional[Dict[str, Any]]): Custom chunking options to override defaults.
            tokenizer_name_or_path (str): Name or path of the Hugging Face tokenizer to use.
                                           Defaults to "gpt2".
            template (Optional[str]): Name of chunking template to use.
            template_manager (Optional[ChunkingTemplateManager]): Template manager instance.
        """
        # Initialize template manager
        self.template_manager = template_manager or ChunkingTemplateManager()
        self.pipeline = ChunkingPipeline(self.template_manager)
        
        # Load template if specified
        self.template: Optional[ChunkingTemplate] = None
        if template:
            self.template = self.template_manager.load_template(template)
            if self.template:
                logger.info(f"Loaded chunking template: {template}")
                # Extract options from template
                template_options = {}
                for stage in self.template.pipeline:
                    if stage.stage == 'chunk':
                        template_options.update(stage.options)
                        if stage.method:
                            template_options['method'] = stage.method
                        break
                # Template options have lower priority than explicit options
                if options:
                    template_options.update(options)
                options = template_options
            else:
                logger.warning(f"Template '{template}' not found, using default options")
        
        # Initialize options: start with defaults, then update with provided options
        self.options = DEFAULT_CHUNK_OPTIONS.copy()
        if options:
            # Ensure type consistency for options that need it
            for key in ['max_size', 'overlap',
                        'semantic_overlap_sentences', 'base_adaptive_chunk_size',
                        'min_adaptive_chunk_size', 'max_adaptive_chunk_size']:
                if key in options and options[key] is not None:
                    try:
                        options[key] = int(options[key])
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid type for option '{key}': {options[key]}. Using default or ignoring.")
                        options[key] = self.options.get(key) # Revert to default from self.options

            for key in ['semantic_similarity_threshold']:
                 if key in options and options[key] is not None:
                    try:
                        options[key] = float(options[key])
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid type for option '{key}': {options[key]}. Using default or ignoring.")
                        options[key] = self.options.get(key) # Revert to default

            self.options.update(options)

        logger.debug(f"Chunker initialized with options: {self.options}")

        self._token_chunker = None
        self._tokenizer_path_to_load: str = self.options.get('tokenizer_name_or_path', tokenizer_name_or_path)

    @property
    def token_chunker(self):
        """Get the token-based chunker, creating it if needed."""
        if self._token_chunker is None:
            self._token_chunker = create_token_chunker(self._tokenizer_path_to_load)
        return self._token_chunker
    
    @property 
    def tokenizer(self):
        """Get the underlying tokenizer for backward compatibility."""
        return self.token_chunker.tokenizer

    def _get_option(self, key: str, default_override: Optional[Any] = None) -> Any:
        """Helper to get an option, allowing for a dynamic default."""
        # Try to get from self.options first
        value = self.options.get(key)
        if value is not None:
            return value
        # If not found in self.options or is None, use default_override
        return default_override

    def detect_language(self, text: str) -> str:
        """
        Detects the language of the given text.

        Args:
            text (str): The text to detect language from.

        Returns:
            str: The detected language code (e.g., 'en', 'zh-cn').
                 Defaults to 'en' if detection fails.
        """
        start_time = time.time()
        log_counter("chunking_language_detection_attempt")
        
        if not text or not text.strip():
            logger.warning("Attempted to detect language from empty or whitespace-only text. Defaulting to 'en'.")
            log_counter("chunking_language_detection_empty_text")
            return self._get_option('language', 'en') # Use option if available, else 'en'
        
        # Check if langdetect is available
        if not LANGDETECT_AVAILABLE:
            logger.debug("langdetect not available, defaulting to configured language or 'en'")
            return self._get_option('language', 'en')
            
        try:
            lang = detect(text)
            logger.debug(f"Detected language: {lang}")
            
            # Log success metrics
            duration = time.time() - start_time
            log_histogram("chunking_language_detection_duration", duration, labels={"status": "success"})
            log_counter("chunking_language_detection_success", labels={"language": lang})
            
            return lang
        except LangDetectException as e:
            logger.warning(f"Language detection failed: {e}. Defaulting to 'en'.")
            
            # Log detection failure
            duration = time.time() - start_time
            log_histogram("chunking_language_detection_duration", duration, labels={"status": "error"})
            log_counter("chunking_language_detection_error", labels={"error_type": "lang_detect_exception"})
            
            return self._get_option('language', 'en')
        except Exception as e_gen:
            logger.error(f"Unexpected error during language detection: {e_gen}. Defaulting to 'en'.")
            
            # Log unexpected error
            duration = time.time() - start_time
            log_histogram("chunking_language_detection_duration", duration, labels={"status": "error"})
            log_counter("chunking_language_detection_error", labels={"error_type": type(e_gen).__name__})
            
            return self._get_option('language', 'en')


    def _ensure_language(self, text: str, language_option: Optional[str] = None) -> str:
        """
        Ensures a language is determined, using option, detection, or default.
        """
        # Priority: 1. Explicit language_option, 2. self.options['language'], 3. detect_language
        if language_option:
            return language_option
        instance_lang_opt = self._get_option('language') # Get from self.options
        if instance_lang_opt:
            return instance_lang_opt
        return self.detect_language(text)

    def _post_process_chunks(self, chunks: List[str]) -> List[str]:
        """
        Strips whitespace from each chunk and removes empty chunks.
        """
        return [chunk.strip() for chunk in chunks if chunk and chunk.strip()]

    def chunk_text(self,
                   text: str,
                   method: Optional[str] = None,
                   llm_call_function: Optional[Callable[[Dict[str, Any]], Union[str, Generator[str, None, None]]]] = None,
                   llm_api_config: Optional[Dict[str, Any]] = None,
                   use_template: Optional[bool] = None,
                   ) -> List[Union[str, Dict[str, Any]]]:
        """
        Main method to chunk text based on the specified method in options or argument.

        Args:
            text (str): The text to chunk.
            method (Optional[str]): Override the chunking method defined in options.
            use_template (Optional[bool]): Force use/bypass of template if loaded.

        Returns:
            List[Union[str, Dict[str, Any]]]: A list of chunks.
                                              Strings for most methods, Dicts for JSON-based chunking.

        Raises:
            InvalidChunkingMethodError: If the method is not supported.
            ChunkingError: For errors during the chunking process.
            MemoryLimitError: If the input text exceeds memory limits.
        """
        # Check document size before processing
        text_size_bytes = len(text.encode('utf-8'))
        if text_size_bytes > MAX_DOCUMENT_SIZE_BYTES:
            text_size_mb = text_size_bytes / (1024 * 1024)
            raise MemoryLimitError(
                f"Document size {text_size_mb:.2f} MB exceeds maximum allowed size of {MAX_DOCUMENT_SIZE_MB} MB"
            )
        # Check if we should use template-based chunking
        if use_template is None:
            use_template = self.template is not None
        
        if use_template and self.template:
            logger.info(f"Using template-based chunking with template: {self.template.name}")
            # Execute template pipeline
            template_results = self.pipeline.execute(
                text=text,
                template=self.template,
                chunker_instance=self,
                llm_call_function=llm_call_function,
                llm_api_config=llm_api_config
            )
            # Convert template results to expected format
            chunks = []
            for result in template_results:
                if isinstance(result, dict) and 'text' in result:
                    chunks.append(result['text'])
                else:
                    chunks.append(result)
            return chunks
        chunk_method = method if method else self._get_option('method', 'words')
        max_size = self._get_option('max_size') # Already int from __init__
        overlap = self._get_option('overlap')   # Already int from __init__
        language = self._ensure_language(text, self._get_option('language')) # Ensure language is determined

        logger.debug(f"Chunking text with method='{chunk_method}', max_size={max_size}, overlap={overlap}, language='{language}'")

        # Adaptive chunking can modify max_size before the main method is called
        if self._get_option('adaptive', False) and chunk_method not in ['semantic', 'json', 'xml', 'ebook_chapters', 'rolling_summarize']:
            # Note: Adaptive sizing might not make sense for all methods.
            # Here, we apply it to general text methods.
            base_adaptive_size = self._get_option('base_adaptive_chunk_size')
            min_adaptive_size = self._get_option('min_adaptive_chunk_size')
            max_adaptive_size = self._get_option('max_adaptive_chunk_size')
            # Try to use NLTK-based adaptive sizing if available
            try:
                max_size = self._adaptive_chunk_size_nltk(text, base_adaptive_size, min_adaptive_size, max_adaptive_size, language)
            except Exception as e:
                logger.warning(f"NLTK-based adaptive sizing failed: {e}. Using non-NLTK adaptive sizing.")
                max_size = self._adaptive_chunk_size_non_punkt(text, base_adaptive_size, min_adaptive_size, max_adaptive_size)
            logger.info(f"Adaptive chunking adjusted max_size to: {max_size}")


        # Multi-level chunking is a wrapper around other methods
        if self._get_option('multi_level', False) and chunk_method in ['words', 'sentences']:
             logger.info(f"Applying multi-level chunking with base method: {chunk_method}")
             return self._multi_level_chunking(text, chunk_method, max_size, overlap, language)


        if chunk_method == 'words':
            return self._chunk_text_by_words(text, max_words=max_size, overlap=overlap, language=language)
        elif chunk_method == 'sentences':
            return self._chunk_text_by_sentences(text, max_sentences=max_size, overlap=overlap, language=language)
        elif chunk_method == 'paragraphs':
            return self._chunk_text_by_paragraphs(text, max_paragraphs=max_size, overlap=overlap)
        elif chunk_method == 'tokens':
            return self._chunk_text_by_tokens(text, max_tokens=max_size, overlap=overlap)
        elif chunk_method == 'semantic':
            # semantic_chunking needs to be a method of the class too
            return self._semantic_chunking(text, max_chunk_size=max_size, unit='words') # unit can be an option
        elif chunk_method == 'json':
            # chunk_text_by_json and its helpers need to be methods
            return self._chunk_text_by_json(text, max_size=max_size, overlap=overlap)
        elif chunk_method == 'ebook_chapters':
            # Needs to be a method
            return self._chunk_ebook_by_chapters(text,
                                                 max_size=max_size, # max_size here might mean something different, e.g. sub-chunking chapters
                                                 overlap=overlap,
                                                 custom_pattern=self._get_option('custom_chapter_pattern'),
                                                 language=language)
        elif chunk_method == 'xml':
            # Needs to be a method
            return self._chunk_xml(text, max_size=max_size, overlap=overlap, language=language)
        elif chunk_method == 'rolling_summarize':
            if not llm_call_function:
                raise ChunkingError("Missing 'llm_call_function' for 'rolling_summarize' method.")

            summary = self._rolling_summarize(
                text_to_summarize=text,
                llm_summarize_step_func=llm_call_function,  # Pass the generic call function
                llm_api_config=llm_api_config or {},  # Pass relevant API name, model, key for the call_func
                # Other summarization-specific options from self.options
                detail=self._get_option('summarization_detail', 0.5),
                min_chunk_tokens=self._get_option('summarize_min_chunk_tokens', 500),
                chunk_delimiter=self._get_option('summarize_chunk_delimiter', "."),
                recursive_summarization=self._get_option('summarize_recursively', False),
                verbose=self._get_option('summarize_verbose', False),
                system_prompt_content=self._get_option('summarize_system_prompt',
                                                       "Rewrite this text in summarized form."),
                additional_instructions=self._get_option('summarize_additional_instructions', None)
            )
            return [summary]  # Wrap in list
        else:
            logger.warning(f"Unknown chunking method '{chunk_method}'. Returning full text as a single chunk.")
            raise InvalidChunkingMethodError(f"Unsupported chunking method: '{chunk_method}'")


    def _chunk_text_by_words(self, text: str, max_words: int, overlap: int, language: str) -> List[str]:
        start_time = time.time()
        logger.info(f"Chunking by words: max_words={max_words}, overlap={overlap}, language='{language}'")
        
        # Use language-specific chunker
        language_chunker = LanguageChunkerFactory.get_chunker(language)
        words = language_chunker.tokenize_words(text)

        logger.debug(f"Total words: {len(words)}")
        if max_words <= 0 :
            raise ValueError(f"max_words must be positive, got {max_words}")
        if max_words > MAX_CHUNK_SIZE_WORDS:
            raise ValueError(f"max_words {max_words} exceeds maximum allowed {MAX_CHUNK_SIZE_WORDS}")
        if overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {overlap}")
        if overlap >= max_words :
            raise ValueError(f"Overlap {overlap} must be less than max_words {max_words}")


        chunks = []
        # Ensure step is at least 1 to prevent infinite loops if max_words equals overlap (though handled above)
        step = max_words - overlap
        if step <= 0: step = max_words # Should not happen if overlap < max_words

        for i in range(0, len(words), step):
            chunk_words = words[i : i + max_words]
            chunks.append(' '.join(chunk_words))
            logger.debug(f"Created word chunk {len(chunks)} with {len(chunk_words)} words")

        # Log metrics
        duration = time.time() - start_time
        log_histogram("chunking_method_duration", duration, labels={"method": "words", "language": language})
        log_histogram("chunking_words_per_chunk", max_words)
        log_histogram("chunking_total_words", len(words))
        log_counter("chunking_method_words_success", labels={"chunks_created": str(len(chunks))})
        
        processed_chunks = self._post_process_chunks(chunks)
        logger.info(f"Word chunking complete: created {len(processed_chunks)} chunks from {len(words)} words")
        return processed_chunks


    def _chunk_text_by_sentences(self, text: str, max_sentences: int, overlap: int, language: str) -> List[str]:
        start_time = time.time()
        logger.info(f"Chunking by sentences: max_sentences={max_sentences}, overlap={overlap}, lang='{language}'")
        log_counter("chunking_method_sentences_attempt", labels={"language": language})
        
        # Use language-specific chunker
        language_chunker = LanguageChunkerFactory.get_chunker(language)
        sentences = language_chunker.tokenize_sentences(text)

        if max_sentences <= 0:
            raise ValueError(f"max_sentences must be positive, got {max_sentences}")
        if max_sentences > MAX_CHUNK_SIZE_SENTENCES:
            raise ValueError(f"max_sentences {max_sentences} exceeds maximum allowed {MAX_CHUNK_SIZE_SENTENCES}")
        if overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {overlap}")
        if overlap >= max_sentences:
            raise ValueError(f"Overlap {overlap} must be less than max_sentences {max_sentences}")

        chunks = []
        step = max_sentences - overlap
        if step <= 0: step = max_sentences

        for i in range(0, len(sentences), step):
            chunk_sentences = sentences[i : i + max_sentences]
            chunks.append(' '.join(chunk_sentences))
            logger.debug(f"Created sentence chunk {len(chunks)} with {len(chunk_sentences)} sentences")

        processed_chunks = self._post_process_chunks(chunks)
        logger.info(f"Sentence chunking complete: created {len(processed_chunks)} chunks from {len(sentences)} sentences")
        return processed_chunks


    def _chunk_text_by_paragraphs(self, text: str, max_paragraphs: int, overlap: int) -> List[str]:
        logger.info(f"Chunking by paragraphs: max_paragraphs={max_paragraphs}, overlap={overlap}")
        # Split by one or more empty lines (common paragraph delimiter)
        paragraphs = re.split(r'\n\s*\n+', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()] # Remove empty paragraphs

        if not paragraphs:
            return []
        if max_paragraphs <= 0:
            raise ValueError(f"max_paragraphs must be positive, got {max_paragraphs}")
        if max_paragraphs > MAX_CHUNK_SIZE_PARAGRAPHS:
            raise ValueError(f"max_paragraphs {max_paragraphs} exceeds maximum allowed {MAX_CHUNK_SIZE_PARAGRAPHS}")
        if overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {overlap}")
        if overlap >= max_paragraphs:
            raise ValueError(f"Overlap {overlap} must be less than max_paragraphs {max_paragraphs}")

        chunks = []
        step = max_paragraphs - overlap
        if step <= 0: step = max_paragraphs

        for i in range(0, len(paragraphs), step):
            chunk_paragraphs = paragraphs[i : i + max_paragraphs]
            chunks.append('\n\n'.join(chunk_paragraphs)) # Join with double newline to preserve paragraph structure
            logger.debug(f"Created paragraph chunk {len(chunks)} with {len(chunk_paragraphs)} paragraphs")

        processed_chunks = self._post_process_chunks(chunks) # post_process_chunks strips leading/trailing, which is fine
        logger.info(f"Paragraph chunking complete: created {len(processed_chunks)} chunks from {len(paragraphs)} paragraphs")
        return processed_chunks


    def _chunk_text_by_tokens(self, text: str, max_tokens: int, overlap: int) -> List[str]:
        """Chunk text by token count using the modular token chunker."""
        logger.info(f"Chunking by tokens: max_tokens={max_tokens}, overlap_tokens={overlap}")
        
        chunks = self.token_chunker.chunk_by_tokens(text, max_tokens, overlap)
        processed_chunks = self._post_process_chunks(chunks)
        logger.info(f"Token chunking complete: created {len(processed_chunks)} chunks")
        return processed_chunks


    # --- Adaptive Chunking Methods ---
    def _adaptive_chunk_size_nltk(self, text: str, base_size: int, min_size: int, max_size: int, language: str) -> int:
        """Adjusts chunk size based on NLTK sentence tokenization."""
        logger.debug(f"Calculating adaptive chunk size (NLTK) for lang '{language}'. Base: {base_size}, Min: {min_size}, Max: {max_size}")
        try:
            nltk_lang_map = {'en': 'english', 'es': 'spanish', 'fr': 'french', 'de': 'german'}
            nltk_language = nltk_lang_map.get(language.lower(), language.lower())
            sentences = sent_tokenize(text, language=nltk_language)
        except LookupError:
            logger.warning(f"NLTK Punkt for '{language}' not found for adaptive sizing. Using non-NLTK fallback.")
            return self._adaptive_chunk_size_non_punkt(text, base_size, min_size, max_size)
        except Exception as e:
            logger.warning(f"Error tokenizing sentences for adaptive sizing with NLTK: {e}. Using non-NLTK fallback.")
            return self._adaptive_chunk_size_non_punkt(text, base_size, min_size, max_size)


        if not sentences:
            return base_size

        avg_sentence_length_words = sum(len(s.split()) for s in sentences) / len(sentences)
        logger.debug(f"Avg sentence length (words): {avg_sentence_length_words}")

        size_factor = 1.0
        if avg_sentence_length_words < 10: # Short sentences
            size_factor = 1.2
        elif avg_sentence_length_words > 25: # Long sentences
            size_factor = 0.8

        adaptive_size = int(base_size * size_factor)
        final_size = max(min_size, min(adaptive_size, max_size))
        logger.debug(f"Adaptive size calculated (NLTK): {final_size} (factor: {size_factor})")
        return final_size

    def _adaptive_chunk_size_non_punkt(self, text: str, base_size: int, min_size: int, max_size: int) -> int:
        """Adjusts chunk size based on average word length if NLTK is not available."""
        logger.debug(f"Calculating adaptive chunk size (non-NLTK). Base: {base_size}, Min: {min_size}, Max: {max_size}")
        words = text.split()
        if not words:
            return base_size

        # Using character length of words as a proxy for complexity
        avg_word_char_length = sum(len(word) for word in words) / len(words) if words else 0
        logger.debug(f"Avg word char length: {avg_word_char_length}")

        size_factor = 1.0
        if avg_word_char_length > 7:  # Longer average words -> potentially more complex
            size_factor = 0.85
        elif avg_word_char_length < 4: # Shorter average words -> potentially simpler
            size_factor = 1.15

        adaptive_size = int(base_size * size_factor)
        final_size = max(min_size, min(adaptive_size, max_size))
        logger.debug(f"Adaptive size calculated (non-NLTK): {final_size} (factor: {size_factor})")
        return final_size

    # Multi-level chunking - can be a wrapper method
    def _multi_level_chunking(self, text: str, base_method_name: str, max_size: int, overlap: int, language: str) -> List[str]:
        logger.debug(f"Multi-level chunking: base_method='{base_method_name}', max_size={max_size}, overlap={overlap}, lang='{language}'")

        # First level: chunk by paragraphs (configurable, but paragraphs is common)
        # The max_paragraphs for this initial split could be larger or an independent option.
        # Using a doubled max_size as a heuristic for paragraph character length.
        initial_paragraph_chunks = self._chunk_text_by_paragraphs(text, max_paragraphs=5, overlap=0) # Small overlap for first pass

        final_chunks = []
        for para_chunk in initial_paragraph_chunks:
            if base_method_name == 'words':
                final_chunks.extend(self._chunk_text_by_words(para_chunk, max_words=max_size, overlap=overlap, language=language))
            elif base_method_name == 'sentences':
                final_chunks.extend(self._chunk_text_by_sentences(para_chunk, max_sentences=max_size, overlap=overlap, language=language))
            else:
                # Should not happen if multi_level is only enabled for words/sentences
                final_chunks.append(para_chunk)
        return final_chunks # Already post-processed by the inner calls

    # --- Stubs for other methods to be moved ---
    def _semantic_chunking(self, text: str, max_chunk_size: int, unit: str) -> List[str]:
        logger.info("Semantic chunking called...")
        # Lazy import for sklearn if not already at top
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            raise ChunkingError("Scikit-learn not installed. Cannot use 'semantic' chunking. Install with 'pip install scikit-learn'")

        language = self._ensure_language(text, self._get_option('language'))
        nltk_lang_map = {'en': 'english', 'es': 'spanish', 'fr': 'french', 'de': 'german'}
        nltk_language = nltk_lang_map.get(language.lower(), language.lower())

        try:
            sentences = sent_tokenize(text, language=nltk_language)
        except LookupError:
            logger.warning(f"NLTK Punkt for '{language}' (for semantic chunking) not found. Defaulting to 'english'.")
            sentences = sent_tokenize(text, language='english')
        except Exception as e:
            logger.error(f"Error sentence tokenizing for semantic chunking: {e}. Using newline split.")
            sentences = text.splitlines()

        if not sentences:
            return []

        try:
            vectorizer = TfidfVectorizer()
            # Filter out empty strings from sentences before fitting, if any from regex or bad tokenization
            valid_sentences = [s for s in sentences if s.strip()]
            if not valid_sentences: return [] # No valid sentences to process
            sentence_vectors = vectorizer.fit_transform(valid_sentences)
        except ValueError as ve: # TFidfVectorizer can raise ValueError if vocabulary is empty (e.g. all stop words)
            logger.warning(f"TF-IDF Vectorizer error during semantic chunking (perhaps all stop words or very short text): {ve}. Falling back to simple chunking.")
            # Fall back to sentence-based chunking
            return self._chunk_text_by_sentences(text, max_sentences=max_chunk_size // 10, overlap=0, language=language)


        chunks = []
        current_chunk_sentences = []
        current_size_units = 0
        # Semantic options
        similarity_threshold = self._get_option('semantic_similarity_threshold', 0.3) # Default if not in options
        overlap_sentences_count = self._get_option('semantic_overlap_sentences', 1) # Default if not in options

        # Helper to count units (words, tokens, characters)
        def _count_units(txt: str, unit_type: str) -> int:
            if unit_type == 'words':
                return len(txt.split())
            elif unit_type == 'tokens':
                try:
                    return self.token_chunker.count_tokens(txt)
                except Exception as e:
                    logger.warning(f"Token counting failed: {e}. Falling back to word count.")
                    return len(txt.split())
            elif unit_type == 'characters':
                return len(txt)
            else:
                logger.warning(f"Unknown unit type '{unit_type}'. Defaulting to word count.")
                return len(txt.split())


        for i, sentence_text in enumerate(valid_sentences):
            sentence_unit_count = _count_units(sentence_text, unit)

            # Break condition 1: Max chunk size exceeded
            if current_size_units + sentence_unit_count > max_chunk_size and current_chunk_sentences:
                chunks.append(' '.join(current_chunk_sentences))
                # Apply overlap: take last N sentences for the new chunk
                current_chunk_sentences = current_chunk_sentences[-overlap_sentences_count:] if overlap_sentences_count > 0 and len(current_chunk_sentences) > overlap_sentences_count else []
                current_size_units = _count_units(' '.join(current_chunk_sentences), unit)

            current_chunk_sentences.append(sentence_text)
            current_size_units += sentence_unit_count

            # Break condition 2: Semantic similarity drop (only if we have a next sentence)
            if i + 1 < len(valid_sentences):
                # Ensure vectors are 2D for cosine_similarity
                current_sentence_vector = sentence_vectors[i:i+1]
                next_sentence_vector = sentence_vectors[i+1:i+2]

                similarity = 0.0
                try:
                    similarity = cosine_similarity(current_sentence_vector, next_sentence_vector)[0, 0]
                except IndexError: # Can happen if vectors are not as expected
                    logger.warning(f"Could not compute similarity for sentence index {i}. Assuming low similarity.")

                # Break if similarity drops AND current chunk has substantial size (e.g., half of max_chunk_size)
                if similarity < similarity_threshold and current_size_units >= (max_chunk_size // 2) and current_chunk_sentences:
                    chunks.append(' '.join(current_chunk_sentences))
                    current_chunk_sentences = current_chunk_sentences[-overlap_sentences_count:] if overlap_sentences_count > 0 and len(current_chunk_sentences) > overlap_sentences_count else []
                    current_size_units = _count_units(' '.join(current_chunk_sentences), unit)

        # Add any remaining sentences in current_chunk
        if current_chunk_sentences:
            chunks.append(' '.join(current_chunk_sentences))

        return self._post_process_chunks(chunks)


    def _chunk_text_by_json(self, text: str, max_size: int, overlap: int) -> List[Dict[str, Any]]:
        logger.debug("chunk_text_by_json (method) started...")
        try:
            json_data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON data provided to _chunk_text_by_json: {e}")
            raise InvalidInputError(f"Invalid JSON data: {e}") from e

        if isinstance(json_data, list):
            return self._chunk_json_list(json_data, max_size, overlap)
        elif isinstance(json_data, dict):
            # `chunk_json_dict` assumes a specific structure ('data', 'metadata').
            # This might need to be more generic or have options to specify keys.
            return self._chunk_json_dict(json_data, max_size, overlap)
        else:
            msg = "Unsupported JSON structure. _chunk_text_by_json only supports top-level JSON objects or arrays."
            logger.error(msg)
            raise InvalidInputError(msg)

    def _chunk_json_list(self, json_list: List[Any], max_size: int, overlap: int) -> List[Dict[str, Any]]:
        logger.debug(f"Chunking JSON list: max_items_per_chunk={max_size}, overlap_items={overlap}")
        if max_size <= 0:
            raise ValueError("max_size for JSON list chunking must be positive.")
        if overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {overlap}")
        if overlap >= max_size:
            raise ValueError(f"JSON list overlap {overlap} must be less than max_size {max_size}")

        chunks_output = []
        total_items = len(json_list)
        step = max_size - overlap
        if step <= 0: step = max_size

        for i in range(0, total_items, step):
            chunk_data = json_list[i : i + max_size]
            # Metadata specific to this chunking method
            metadata = {
                'original_list_size': total_items,
                'item_start_index': i,
                'item_end_index': i + len(chunk_data) -1,
                # 'chunk_index', 'total_chunks' etc. will be added by improved_chunking_process wrapper
            }
            chunks_output.append({
                'json': chunk_data, # The actual JSON content of the chunk
                'metadata': metadata  # Metadata specific to this JSON chunk
            })
        return chunks_output


    def _chunk_json_dict(self, json_dict: Dict[str, Any], max_size: int, overlap: int) -> List[Dict[str, Any]]:
        # This method is quite specific to a schema with a 'data' key.
        # Consider making 'chunkable_key' an option.
        logger.debug(f"Chunking JSON dict: max_keys_in_data_per_chunk={max_size}, overlap_keys={overlap}")
        chunkable_key = self._get_option('json_chunkable_data_key', 'data') # e.g. {'json_chunkable_data_key': 'entries'}

        if chunkable_key not in json_dict or not isinstance(json_dict[chunkable_key], dict):
            msg = f"Chunkable key '{chunkable_key}' not found in JSON dictionary or is not a dictionary itself."
            logger.error(msg)
            raise InvalidInputError(msg)

        if max_size <= 0:
            raise ValueError("max_size for JSON dict chunking must be positive.")
        if overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {overlap}")
        if overlap >= max_size:
            raise ValueError(f"JSON dict overlap {overlap} must be less than max_size {max_size}")

        data_to_chunk = json_dict[chunkable_key]
        all_keys = list(data_to_chunk.keys())
        total_keys = len(all_keys)

        chunks_output = []
        step = max_size - overlap
        if step <= 0: step = max_size

        # Preserve other parts of the original dictionary
        preserved_data_shell = {k: v for k, v in json_dict.items() if k != chunkable_key}

        for i in range(0, total_keys, step):
            current_chunk_keys = all_keys[i : i + max_size]
            # Note: Overlap for dict keys might be complex if order isn't guaranteed or meaningful.
            # This simple slicing assumes order is somewhat stable or user understands the implication.

            chunk_data_content = {key: data_to_chunk[key] for key in current_chunk_keys}

            # Create the new chunked JSON structure
            new_json_chunk = preserved_data_shell.copy()
            new_json_chunk[chunkable_key] = chunk_data_content

            metadata = {
                'original_dict_total_keys_in_data': total_keys,
                'key_start_index_in_data': i, # Based on original list of keys
                'keys_in_this_chunk_data': len(current_chunk_keys),
            }
            chunks_output.append({
                'json': new_json_chunk,
                'metadata': metadata
            })
        return chunks_output

        # In tldw_Server_API/app/core/Utils/Chunk_Lib.py

    def _chunk_ebook_by_chapters(self, text: str, max_size: int, overlap: int, custom_pattern: Optional[str],
                                 language: str) -> List[Dict[str, Any]]:
        logger.debug(f"Chunking Ebook by Chapters. Custom pattern: {custom_pattern}, Lang: {language}")

        # ... The rest of this method's implementation is unchanged ...
        chapter_patterns = [
            custom_pattern,
            r'^\s*chapter\s+\d+([:.\-\s].*)?$',
            r'^\s*chapter\s+[ivxlcdm]+([:.\-\s].*)?$',
            r'^\s*(Part|Book|Volume)\s+[A-Za-z0-9]+([:.\-\s].*)?$',
            r'^\s*\d+\s*([:.\-\s][^\r\n]{1,150}|[^\r\n]{1,150})?$',
            r'^\s*#{1,4}\s+[^\r\n]+',
            r'^\s*(PREFACE|INTRODUCTION|CONTENTS|APPENDIX|EPILOGUE|PROLOGUE|ACKNOWLEDGMENTS?|SECTION\s*\d*|UNIT\s*\d*)\s*$',
        ]
        active_patterns = [p for p in chapter_patterns if p is not None]
        if not active_patterns:
            logger.warning("No chapter patterns available for ebook chunking.")
            if text.strip():
                return [{'text': text,
                         'metadata': {'chunk_type': 'single_document_no_chapters', 'chapter_title': 'Full Document'}}]
            return []

        lines = text.splitlines()
        chapter_splits: List[Dict[str, Any]] = []
        chapter_number = 0
        first_heading_index = -1
        first_heading_title_text = "Preface or Introduction"
        current_scan_patterns = list(active_patterns)
        for line_idx, line_content in enumerate(lines):
            for pattern_str in list(current_scan_patterns):
                try:
                    if re.match(pattern_str, line_content, re.IGNORECASE):
                        first_heading_index = line_idx
                        first_heading_title_text = line_content.strip()
                        break
                except re.error as re_e:
                    logger.warning(
                        f"Regex error in chapter pattern '{pattern_str}' during initial scan: {re_e}. Disabling this pattern.")
                    if pattern_str in current_scan_patterns: current_scan_patterns.remove(pattern_str)
                    if pattern_str in active_patterns: active_patterns.remove(pattern_str)
            if first_heading_index != -1:
                break
        if first_heading_index > 0:
            preface_content_lines = lines[:first_heading_index]
            preface_text = "\n".join(preface_content_lines).strip()
            if preface_text:
                chapter_number += 1
                chapter_splits.append({
                    'text': preface_text,
                    'metadata': {
                        'chunk_type': 'preface',
                        'chapter_number': chapter_number,
                        'chapter_title': "Preface/Introduction",
                        'detected_chapter_pattern': 'preface_heuristic',
                    }
                })
        elif first_heading_index == -1:
            if text.strip():
                logger.warning(
                    "No chapter headings found using patterns. Returning document as a single chapter chunk.")
                return [{'text': text,
                         'metadata': {'chunk_type': 'single_document_no_chapters', 'chapter_title': 'Full Document'}}]
            return []
        start_line_of_current_chapter_content = first_heading_index
        current_chapter_title = first_heading_title_text
        current_chapter_pattern_str = "unknown"
        if first_heading_index != -1 and first_heading_index < len(lines):
            for p_str in active_patterns:
                if re.match(p_str, lines[first_heading_index], re.IGNORECASE):
                    current_chapter_pattern_str = p_str
                    break
        for line_idx in range(first_heading_index + 1, len(lines)):
            line_content = lines[line_idx]
            is_new_chapter_heading = False
            new_heading_pattern_str = None
            for pattern_str in active_patterns:
                try:
                    if re.match(pattern_str, line_content, re.IGNORECASE):
                        is_new_chapter_heading = True
                        new_heading_pattern_str = pattern_str
                        break
                except re.error as re_e_inner:
                    logger.warning(f"Regex error (inner loop) for pattern '{pattern_str}': {re_e_inner}.")
            if is_new_chapter_heading:
                chapter_content_lines = lines[start_line_of_current_chapter_content: line_idx]
                chapter_text = "\n".join(chapter_content_lines).strip()
                if chapter_text:
                    chapter_number += 1
                    chunk_type = 'chapter'
                    if chapter_number == 1 and first_heading_index == 0 and \
                            (current_chapter_title.lower() == "preface" or
                             current_chapter_title.lower() == "introduction" or
                             current_chapter_title == "Preface or Introduction"):
                        chunk_type = 'preface'
                    final_metadata_title = current_chapter_title
                    if chunk_type == 'preface':
                        final_metadata_title = "Preface/Introduction"
                    chapter_splits.append({
                        'text': chapter_text,
                        'metadata': {
                            'chunk_type': chunk_type,
                            'chapter_number': chapter_number,
                            'chapter_title': final_metadata_title,
                            'detected_chapter_pattern': current_chapter_pattern_str,
                        }
                    })
                start_line_of_current_chapter_content = line_idx
                current_chapter_title = line_content.strip()
                current_chapter_pattern_str = new_heading_pattern_str or "unknown"
        if start_line_of_current_chapter_content < len(lines):
            last_chapter_content_lines = lines[start_line_of_current_chapter_content:]
            last_chapter_text = "\n".join(last_chapter_content_lines).strip()
            if last_chapter_text:
                chapter_number += 1
                chunk_type = 'chapter'
                is_first_processed_block = not chapter_splits
                condition_is_first_block_and_preface_like_title = \
                    (is_first_processed_block or (chapter_number == 1 and first_heading_index == 0)) and \
                    (current_chapter_title.lower() == "preface" or \
                     current_chapter_title.lower() == "introduction" or \
                     current_chapter_title == "Preface or Introduction")
                if condition_is_first_block_and_preface_like_title:
                    chunk_type = 'preface'
                final_metadata_title = current_chapter_title
                if chunk_type == 'preface':
                    final_metadata_title = "Preface/Introduction"
                chapter_splits.append({
                    'text': last_chapter_text,
                    'metadata': {
                        'chunk_type': chunk_type,
                        'chapter_number': chapter_number,
                        'chapter_title': final_metadata_title,
                        'detected_chapter_pattern': current_chapter_pattern_str,
                    }
                })
        final_chapter_chunks: List[Dict[str, Any]] = []
        for i, chap_data in enumerate(chapter_splits):
            chap_data['metadata']['chunk_index_in_book'] = i + 1
            chap_data['metadata']['total_chapters_detected'] = len(chapter_splits)
            
            # Check if we need to sub-chunk this chapter
            tokenizer_available = False
            chapter_token_count = 0
            try:
                chapter_token_count = self.token_chunker.count_tokens(chap_data['text'])
                tokenizer_available = True
            except Exception as e_tok_check:
                logger.warning(f"Could not count tokens for chapter sub-chunking: {e_tok_check}")

            if max_size > 0 and tokenizer_available and chapter_token_count > max_size:
                logger.info(
                    f"Chapter '{chap_data['metadata']['chapter_title']}' (length {chapter_token_count} tokens) exceeds max_size {max_size}. Sub-chunking.")
                sub_chunks = self._chunk_text_by_tokens(chap_data['text'], max_tokens=max_size,
                                                        overlap=overlap if overlap < max_size else max_size // 5)
                for sub_idx, sub_chunk_text in enumerate(sub_chunks):
                    sub_chunk_metadata = chap_data['metadata'].copy()
                    sub_chunk_metadata['sub_chunk_index_in_chapter'] = sub_idx + 1
                    sub_chunk_metadata['total_sub_chunks_in_chapter'] = len(sub_chunks)
                    sub_chunk_metadata['chunk_type'] = 'chapter_sub_chunk'
                    final_chapter_chunks.append({'text': sub_chunk_text, 'metadata': sub_chunk_metadata})
            else:
                final_chapter_chunks.append(chap_data)
        return final_chapter_chunks


    def _extract_xml_structure_recursive(self, element, path="") -> List[Tuple[str, str]]:
        """ Helper for _chunk_xml: Recursively extract XML structure and content. """
        results = []
        current_path = f"{path}/{element.tag}" if path else element.tag

        if element.text and element.text.strip():
            results.append((current_path, element.text.strip()))
        if element.attrib:
            for key, value in element.attrib.items():
                results.append((f"{current_path}/@{key}", value))
        for child in element:
            results.extend(self._extract_xml_structure_recursive(child, current_path))
        # Include tail text if present (text after a child element, within its parent)
        if element.tail and element.tail.strip():
            results.append((f"{path}/{element.tag}/#tail", element.tail.strip())) # Path indicates it's tail of current_path
        return results

    def _chunk_xml(self, xml_text: str, max_size: int, overlap: int, language: str) -> List[Dict[str, Any]]:
        # max_size is in words for the content of combined XML elements
        # overlap is in number of XML elements (path-content pairs)
        logger.debug(f"Chunking XML: max_words_per_chunk_content={max_size}, overlap_elements={overlap}, Lang: {language}")
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            raise InvalidInputError(f"Invalid XML content: {e}") from e

        xml_elements_with_paths = self._extract_xml_structure_recursive(root)
        if not xml_elements_with_paths:
            return []

        if max_size <=0:
            raise ValueError("max_size for XML chunking must be positive.")
        if overlap >= len(xml_elements_with_paths) and len(xml_elements_with_paths) > 0 : # Check if overlap makes sense
            logger.warning(f"XML overlap elements {overlap} >= total elements {len(xml_elements_with_paths)}. Setting overlap to 0.")
            overlap = 0
        elif overlap < 0:
            overlap = 0


        chunks_output = []
        current_chunk_elements = [] # List of (path, content) tuples
        current_word_count = 0

        # Step is by element, but decision to cut is by word count of accumulated content
        for i in range(len(xml_elements_with_paths)):
            path, content = xml_elements_with_paths[i]
            content_word_count = len(content.split())

            if current_word_count + content_word_count > max_size and current_chunk_elements:
                # Finalize current chunk
                chunk_text_parts = [f"{p}: {c}" for p, c in current_chunk_elements]
                chunk_metadata = {
                    'xml_paths': [p for p, _ in current_chunk_elements],
                    'root_tag': root.tag,
                    'original_xml_attributes': dict(root.attrib),
                    'num_xml_elements_in_chunk': len(current_chunk_elements)
                }
                chunks_output.append({'text': '\n'.join(chunk_text_parts), 'metadata': chunk_metadata})

                # Start new chunk with overlap
                if overlap > 0 and len(current_chunk_elements) > overlap:
                    current_chunk_elements = current_chunk_elements[-overlap:]
                else: # Not enough elements for full overlap, or overlap is 0
                    current_chunk_elements = []
                current_word_count = sum(len(c.split()) for _, c in current_chunk_elements)

            current_chunk_elements.append((path, content))
            current_word_count += content_word_count

        # Add the last remaining chunk
        if current_chunk_elements:
            chunk_text_parts = [f"{p}: {c}" for p, c in current_chunk_elements]
            chunk_metadata = {
                'xml_paths': [p for p, _ in current_chunk_elements],
                'root_tag': root.tag,
                'original_xml_attributes': dict(root.attrib),
                'num_xml_elements_in_chunk': len(current_chunk_elements)
            }
            chunks_output.append({'text': '\n'.join(chunk_text_parts), 'metadata': chunk_metadata})

        return chunks_output


    def _rolling_summarize(self,
                           text_to_summarize: str,
                           llm_summarize_step_func: Callable, # The function that will call the actual LLM API
                           llm_api_config: Dict[str, Any],    # Contains {'api_name', 'model', 'api_key', 'temp', etc.}
                           detail: float,
                           min_chunk_tokens: int,
                           chunk_delimiter: str,
                           recursive_summarization: bool,
                           verbose: bool,
                           system_prompt_content: str,
                           additional_instructions: Optional[str]
                           ) -> str:
        logger.info(f"Rolling summarization called. Detail: {detail}")
        text_token_length = self.token_chunker.count_tokens(text_to_summarize)
        max_summarization_chunks = max(1, text_token_length // min_chunk_tokens)
        min_summarization_chunks = 1
        num_summarization_chunks = int(min_summarization_chunks + detail * (max_summarization_chunks - min_summarization_chunks))
        num_summarization_chunks = max(1, num_summarization_chunks)
        llm_input_chunk_size_tokens = max(min_chunk_tokens, text_token_length // num_summarization_chunks)

        text_chunks_for_llm, _, dropped_count = self._chunk_on_delimiter_for_llm(
            text_to_summarize,
            llm_input_chunk_size_tokens,
            delimiter=chunk_delimiter
        )
        if dropped_count > 0 and verbose:
            logger.warning(f"{dropped_count} parts were dropped during text splitting for summarization.")
        if verbose:
            logger.info(f"Splitting text for summarization into {len(text_chunks_for_llm)} parts.")

        final_system_prompt = system_prompt_content
        if additional_instructions:
            final_system_prompt += f"\n\n{additional_instructions}"

        try:
            from tqdm import tqdm # Import here
        except ImportError:
            logger.warning("tqdm library not found. Progress bar for summarization parts will be disabled. Install with 'pip install tqdm'.")
            # Define a dummy tqdm if not found, so the loop doesn't break
            def tqdm(iterable, *args, **kwargs):
                return iterable

        accumulated_summaries = []
        for i, chunk_for_llm in enumerate(tqdm(text_chunks_for_llm, desc="Summarizing parts", disable=not verbose)):
            user_message_content = chunk_for_llm
            if recursive_summarization and accumulated_summaries:
                user_message_content = f"Previous summary context:\n{accumulated_summaries[-1]}\n\nNew content to summarize and integrate:\n{chunk_for_llm}"

            # Prepare payload for the llm_summarize_step_func
            # This payload structure should match what your `Summarization_General_Lib.analyze` or `_dispatch_to_api` expects
            payload_for_llm_call = {
                "api_name": llm_api_config.get("api_name", "openai"), # Default or from config
                "input_data": user_message_content, # This is the text for the LLM
                "custom_prompt_arg": "", # Rolling summarize manages the full prompt content
                "api_key": llm_api_config.get("api_key"),
                "system_message": final_system_prompt,
                "temp": llm_api_config.get("temperature", self._get_option('summarize_temperature', 0.1)),
                "streaming": False, # Internal steps of rolling summary should not stream to Chunker
                "model": llm_api_config.get("model"),
                "max_tokens": llm_api_config.get("max_tokens")
            }
            try:
                # `llm_summarize_step_func` should be blocking and return a string
                summary_content = llm_summarize_step_func(payload_for_llm_call)

                if isinstance(summary_content, str) and summary_content.startswith("Error:"):
                    logger.error(f"LLM call for summarization part {i+1} failed: {summary_content}")
                    accumulated_summaries.append(f"[Summarization failed for this part: {chunk_for_llm[:100]}...]")
                elif isinstance(summary_content, str):
                    accumulated_summaries.append(summary_content)
                else: # Should not happen if llm_summarize_step_func is well-behaved
                    logger.error(f"LLM call for summarization part {i+1} returned non-string: {type(summary_content)}")
                    accumulated_summaries.append(f"[Summarization error for this part (unexpected type): {chunk_for_llm[:100]}...]")

            except Exception as e_llm:
                logger.error(f"Exception calling llm_summarize_step_func for part {i+1}: {e_llm}", exc_info=True)
                accumulated_summaries.append(f"[Exception during summarization for this part: {chunk_for_llm[:100]}...]")

        final_summary = '\n\n---\n\n'.join(accumulated_summaries) # Join with a clear separator
        return final_summary.strip()

    # Helper for rolling_summarize (was combine_chunks_with_no_minimum)
    def _combine_chunks_for_llm(self,
                                 chunks: List[str],
                                 max_tokens: int, # Max tokens for the combined output for the LLM
                                 chunk_delimiter: str = "\n\n",
                                 header: Optional[str] = None,
                                 add_ellipsis_for_overflow: bool = True,
                                 ) -> Tuple[List[str], List[List[int]], int]:
        dropped_chunk_count = 0
        output_combined_texts = []
        output_original_indices = [] # To track which original chunks went into which combined text

        current_candidate_text_parts = [header] if header else []
        current_candidate_indices = []

        for chunk_idx, chunk_content in enumerate(chunks):
            # Tentatively add the new chunk (with header if it's the first part of a new candidate)
            parts_to_test = current_candidate_text_parts + ([chunk_content] if current_candidate_text_parts or not header else [header, chunk_content])

            # If current_candidate_text_parts is empty and header exists, it means we are starting a new combination.
            # The header should only be added once at the beginning of such a combination.
            if not current_candidate_text_parts and header:
                # This logic seems slightly off if header is meant per combined chunk rather than per original chunk.
                # Assuming header is for the combined block.
                # Let's simplify: if current_candidate_text_parts is empty, it might start with header.
                # The passed `chunks` list are the primary content.
                pass # Header is already in current_candidate_text_parts if it's the very start.

            test_text = chunk_delimiter.join(parts_to_test)
            token_count = self.token_chunker.count_tokens(test_text)

            if token_count > max_tokens:
                # Current candidate (before adding new chunk) was likely the max fit
                if current_candidate_text_parts and (not header or len(current_candidate_text_parts) > 1 or current_candidate_text_parts[0] != header): # Check if it's more than just a header
                    if add_ellipsis_for_overflow:
                        # Check if adding ellipsis to the *previous* candidate (current_candidate_text_parts) is valid
                        # This part is tricky. Ellipsis usually indicates the *new* chunk couldn't fit.
                        # The original code added ellipsis if the *new* chunk made it overflow.
                        # Let's stick to: if adding the current 'chunk_content' overflows, then the 'current_candidate_text_parts' is finalized.
                        # If 'add_ellipsis_for_overflow' is true, it means the *dropped* chunk_content is represented by ellipsis.
                        # This seems more aligned with the original 'dropped_chunk_count'.
                        pass # Ellipsis logic might be better applied if a single chunk is too big.

                    output_combined_texts.append(chunk_delimiter.join(current_candidate_text_parts))
                    output_original_indices.append(current_candidate_indices)

                    # Start new candidate with the current chunk_content that caused overflow
                    current_candidate_text_parts = [header, chunk_content] if header else [chunk_content]
                    current_candidate_indices = [chunk_idx]

                    # If this new chunk *itself* is too large (even with header)
                    current_candidate_only_text = chunk_delimiter.join(current_candidate_text_parts)
                    if self.token_chunker.count_tokens(current_candidate_only_text) > max_tokens:
                        logger.warning(f"Single chunk (index {chunk_idx}, content: '{chunk_content[:50]}...') itself exceeds max_tokens ({max_tokens}) even after starting new. It will be dropped.")
                        dropped_chunk_count +=1
                        current_candidate_text_parts = [header] if header else [] # Reset for next
                        current_candidate_indices = []
                        # continue # Skip to next chunk_content
                else: # current_candidate_text_parts was empty or just header, and new chunk overflows
                    logger.warning(f"Single chunk (index {chunk_idx}, content: '{chunk_content[:50]}...') itself exceeds max_tokens ({max_tokens}). It will be dropped.")
                    dropped_chunk_count +=1
                    # current_candidate_text_parts remains [header] or []
                    # current_candidate_indices remains []
            else:
                # It fits, so add current chunk_content to candidate
                # If current_candidate_text_parts is empty and there's a header, it's already there.
                # If it's not empty, just append.
                # If it's empty and no header, it becomes the first part.
                if not current_candidate_text_parts: # Starting fresh
                    if header: current_candidate_text_parts = [header, chunk_content]
                    else: current_candidate_text_parts = [chunk_content]
                else: # Appending to existing candidate
                    current_candidate_text_parts.append(chunk_content)
                current_candidate_indices.append(chunk_idx)

        # Add the last candidate if it has content (more than just a header)
        if current_candidate_text_parts and (not header or len(current_candidate_text_parts) > 1 or (header and current_candidate_text_parts[0] != header) or not header):
            output_combined_texts.append(chunk_delimiter.join(current_candidate_text_parts))
            output_original_indices.append(current_candidate_indices)

        return output_combined_texts, output_original_indices, dropped_chunk_count

    # Helper for rolling_summarize (was chunk_on_delimiter)
    def _chunk_on_delimiter_for_llm(self, input_string: str,
                                     max_tokens_for_llm_input: int, # Max tokens for each final combined chunk for LLM
                                     delimiter: str) -> Tuple[List[str], List[List[int]], int]:
        # This function first splits the input_string by delimiter,
        # then combines these smaller parts into larger blocks suitable for an LLM,
        # ensuring no block exceeds max_tokens_for_llm_input.

        initial_parts = input_string.split(delimiter)

        # We need to re-add the delimiter for context, but only *between* parts, not at the very end of the LLM input block.
        # The _combine_chunks_for_llm will handle joining with its own delimiter.
        # So, we will pass parts and let _combine_chunks_for_llm join them.
        # If the original delimiter is important *within* the LLM's view of a combined chunk, it should be part of 'initial_parts'.

        # Let's adjust how parts are formed: append delimiter to all but the last part from the split.
        reconstructed_parts = []
        for i, part_text in enumerate(initial_parts):
            if i < len(initial_parts) - 1: # Not the last part
                reconstructed_parts.append(part_text + delimiter)
            else: # Last part, don't append delimiter
                if part_text: # Add if not empty
                     reconstructed_parts.append(part_text)

        # Filter out any empty strings that might result if there were multiple delimiters together
        reconstructed_parts = [p for p in reconstructed_parts if p]

        if not reconstructed_parts:
            return [], [], 0

        # Now, combine these 'reconstructed_parts' into blocks that are under 'max_tokens_for_llm_input'.
        # The delimiter for _combine_chunks_for_llm should be something that makes sense for joining these parts,
        # often an empty string if the delimiter is already part of reconstructed_parts, or a space.
        # Let's use "" as the delimiter for _combine_chunks_for_llm, as reconstructed_parts already contain their trailing delimiters.

        combined_texts_for_llm, original_indices, dropped_count = self._combine_chunks_for_llm(
            chunks=reconstructed_parts, # These are the parts including their original delimiters
            max_tokens=max_tokens_for_llm_input,
            chunk_delimiter="", # Join these parts directly
            add_ellipsis_for_overflow=True # Or make this an option
        )

        # The original `chunk_on_delimiter` added the delimiter to the end of each *combined_chunk*.
        # This might not be what we want for direct LLM input if the LLM is processing it as a whole.
        # If the goal is that each item in `combined_texts_for_llm` *looks like* it was split by `delimiter`
        # only where appropriate, the current `reconstructed_parts` logic is better.
        # The example `combined_chunks = [f"{chunk}{delimiter}" for chunk in combined_chunks]` is removed.

        return combined_texts_for_llm, original_indices, dropped_count


# ... (end of Chunker class) ...

# The global `improved_chunking_process` function (defined in Part 1)
# already instantiates and uses `Chunker`.

# Remove old global functions that are now methods in Chunker:
# detect_language (done)
# chunk_text (done, is main dispatcher)
# chunk_text_by_words, _sentences, _paragraphs, _tokens (done)
# post_process_chunks (done)
# adaptive_chunk_size, adaptive_chunking (partially done, integrated into chunk_text and as _adaptive_... methods)
# multi_level_chunking (done)
# semantic_chunking (done)
# chunk_text_by_json, chunk_json_list, chunk_json_dict (done)
# chunk_ebook_by_chapters (done)
# chunk_xml, extract_xml_structure (done)
# rolling_summarize, combine_chunks_with_no_minimum, chunk_on_delimiter (done, integrated as _rolling_summarize and helpers)

# The following functions might still be useful as standalone utilities or need review if they should be class methods:
# - load_document (utility, can stay global or become static method if Chunker needs to load files)
# - determine_chunk_position (utility for metadata, improved_chunking_process handles relative_position differently)
# - get_chunk_metadata (largely superseded or its logic integrated elsewhere)
# - chunk_for_embedding (uses improved_chunking_process, so it's fine. Might add a Chunker arg)
# - process_document_with_metadata (uses improved_chunking_process)

# Let's refine `chunk_for_embedding` and `process_document_with_metadata`
# to potentially accept a Chunker instance or Chunker options.

def chunk_for_embedding(text: str,
                        file_name: str,
                        custom_chunk_options: Optional[Dict[str, Any]] = None,
                        tokenizer_name_or_path: str = "gpt2",
                        llm_call_function: Optional[Callable] = None, # Added
                        llm_api_config: Optional[Dict[str, Any]] = None # Added
                        ) -> List[Dict[str, Any]]:
    """
    Prepares chunks specifically for embedding, adding headers with context.
    Uses improved_chunking_process internally.
    """
    # `improved_chunking_process` will create a Chunker instance with these options
    logger.info(f"Chunking for embedding. File: {file_name}. Custom options: {custom_chunk_options}")
    chunks_from_improved_process = improved_chunking_process(
        text,
        chunk_options_dict=custom_chunk_options,
        tokenizer_name_or_path=tokenizer_name_or_path,
        llm_call_function_for_chunker=llm_call_function, # Pass through
        llm_api_config_for_chunker=llm_api_config      # Pass through
    )
    # The options used are now part of the Chunker instance within improved_chunking_process

    chunked_text_with_headers_list = []
    total_chunks_count = len(chunks_from_improved_process) # Get from the result

    for i, chunk_data in enumerate(chunks_from_improved_process):
        # chunk_data is {'text': ..., 'metadata': ...}
        chunk_text_content = chunk_data['text']
        chunk_metadata = chunk_data['metadata'] # This metadata is already rich

        # Determine position string (optional, could make this a helper)
        relative_pos = chunk_metadata.get('relative_position', 0.0)
        position_description = "middle"
        if relative_pos < 0.33: position_description = "beginning"
        elif relative_pos > 0.66: position_description = "end"

        # Construct header for embedding context
        # The metadata already contains chunk_index and total_chunks from improved_chunking_process
        chunk_header = f"""[DOCUMENT: {file_name}]
[CHUNK: {chunk_metadata.get('chunk_index', i+1)} OF {chunk_metadata.get('total_chunks', total_chunks_count)}]
[POSITION: This chunk is from the {position_description} of the document.]
---BEGIN CHUNK CONTENT---
"""
        # Add more from metadata if useful, e.g., chunk_metadata.get('initial_document_json_metadata')
        # or specific things from ebook/xml if the method was that.

        full_chunk_text_for_embedding = chunk_header + chunk_text_content + "\n---END CHUNK CONTENT---"

        # Create a new structure for the embedding-specific output
        embedding_chunk_data = {
            'text_for_embedding': full_chunk_text_for_embedding,
            'original_chunk_text': chunk_text_content,
            'source_document_name': file_name,
            'chunk_metadata': chunk_metadata # Carry over all the detailed metadata
        }
        chunked_text_with_headers_list.append(embedding_chunk_data)

    return chunked_text_with_headers_list


def process_document_with_metadata(text: str,
                                   chunk_options_dict: Dict[str, Any],
                                   document_metadata: Dict[str, Any],
                                   tokenizer_name_or_path: str = "gpt2",
                                   llm_call_function: Optional[Callable] = None, # Added
                                   llm_api_config: Optional[Dict[str, Any]] = None # Added
                                   ) -> Dict[str, Any]:
    """
    Processes a document, chunks it, and associates document-level metadata with the chunked output.
    """
    logger.info(f"Processing document with metadata. Options: {chunk_options_dict}")
    chunks_result = improved_chunking_process(
        text,
        chunk_options_dict=chunk_options_dict,
        tokenizer_name_or_path=tokenizer_name_or_path,
        llm_call_function_for_chunker=llm_call_function, # Pass through
        llm_api_config_for_chunker=llm_api_config      # Pass through
    )
    for chunk_item in chunks_result:
        if 'document_level_metadata' not in chunk_item['metadata']:
            chunk_item['metadata']['document_level_metadata'] = {}
        chunk_item['metadata']['document_level_metadata'].update(document_metadata)
    return {
        'original_document_metadata': document_metadata,
        'chunks': chunks_result
    }


# Keep improved_chunking_process and process_document_with_metadata outside the class for now,
# or make them static methods / helper functions that instantiate and use the Chunker.
# For now, let's adapt improved_chunking_process to use the Chunker class.

def improved_chunking_process(text: str,
                              chunk_options_dict: Optional[Dict[str, Any]] = None,
                              tokenizer_name_or_path: str = "gpt2",
                              template: Optional[str] = None,
                              template_manager: Optional[ChunkingTemplateManager] = None,
                              # Parameters for LLM calls if needed by a chunking method
                              llm_call_function_for_chunker: Optional[Callable] = None,
                              llm_api_config_for_chunker: Optional[Dict[str, Any]] = None
                              ) -> List[Dict[str, Any]]:
    logger.info("Improved chunking process started...")
    logger.debug(f"Received chunk_options_dict: {chunk_options_dict}")
    logger.debug(f"Text length: {len(text)} characters, tokenizer: {tokenizer_name_or_path}")
    if template:
        logger.debug(f"Using template: {template}")

    chunker_instance = Chunker(options=chunk_options_dict,
                               tokenizer_name_or_path=tokenizer_name_or_path,
                               template=template,
                               template_manager=template_manager,
                                )

    # Get effective options from the chunker instance (these are now resolved)
    effective_options = chunker_instance.options.copy() # Use a copy
    # (JSON and header extraction logic using `text` and storing in `json_content_metadata`, `header_text_content`, update `processed_text`)
    # This was the previous flow:
    json_content_metadata = {}
    processed_text = text
    try:
        if processed_text.strip().startswith("{"):
            json_end_match = re.search(r"\}\s*\n", processed_text)
            if json_end_match:
                json_end_index = json_end_match.end()
                potential_json_str = processed_text[:json_end_index].strip()
                try:
                    json_content_metadata = json.loads(potential_json_str)
                    processed_text = processed_text[json_end_index:].strip()
                    logger.debug(f"Extracted JSON metadata: {json_content_metadata}")
                except json.JSONDecodeError:
                    logger.debug("Potential JSON at start, but failed to parse. Treating as normal text.")
                    json_content_metadata = {}
            else:
                logger.debug("Text starts with '{' but no clear '}\\n' end for JSON metadata.")
        else:
            logger.debug("No JSON metadata found at the beginning of the text.")
    except Exception as e_json:
        logger.warning(f"Error during JSON metadata extraction: {e_json}. Proceeding without it.")
        json_content_metadata = {}

    header_re = re.compile(r"""^ (This[ ]text[ ]was[ ]transcribed[ ]using (?:[^\n]*\n)*?\n) """, re.MULTILINE | re.VERBOSE)
    header_text_content = ""
    header_match = header_re.match(processed_text)
    if header_match:
        header_text_content = header_match.group(1)
        processed_text = processed_text[len(header_text_content):].strip()
        logger.debug(f"Extracted header text: {header_text_content}")

    if not effective_options.get('language'):
        detected_lang = chunker_instance.detect_language(processed_text)
        effective_options['language'] = str(detected_lang)
        logger.debug(f"Language for overall process set to: {effective_options['language']}")

    try:
        raw_chunks = chunker_instance.chunk_text(
            processed_text,
            method=effective_options['method'],
            llm_call_function=llm_call_function_for_chunker, # Pass it down
            llm_api_config=llm_api_config_for_chunker      # Pass it down
        )
        logger.debug(f"Created {len(raw_chunks)} raw_chunks using method {effective_options['method']}")
    except ChunkingError as ce:
        logger.error(f"ChunkingError in chunking process: {ce}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chunking process: {e}", exc_info=True)
        raise ChunkingError(f"Unexpected error in chunking process: {e}") from e

    chunks_with_metadata_list = []
    total_chunks_count = len(raw_chunks)
    logger.info(f"Processing {total_chunks_count} chunks for metadata enrichment")
    try:
        for i, chunk_item in enumerate(raw_chunks):
            logger.debug(f"Processing chunk {i+1}/{total_chunks_count}")
            actual_text_content: str
            is_json_chunk = False
            chunk_specific_metadata = {} # Initialize

            if isinstance(chunk_item, dict) and 'json' in chunk_item and 'metadata' in chunk_item : # From JSON methods
                actual_text_content = json.dumps(chunk_item['json'], ensure_ascii=False)
                chunk_specific_metadata = chunk_item['metadata']
                is_json_chunk = True
            elif isinstance(chunk_item, dict) and 'text' in chunk_item and 'metadata' in chunk_item: # From Ebook/XML methods
                actual_text_content = chunk_item['text']
                chunk_specific_metadata = chunk_item['metadata']
            elif isinstance(chunk_item, str):
                actual_text_content = chunk_item
            else:
                logger.warning(f"Unexpected chunk item type: {type(chunk_item)}. Skipping.")
                continue

            current_chunk_metadata = {
                'chunk_index': i + 1,
                'total_chunks': total_chunks_count,
                'chunk_method': effective_options['method'],
                'max_size_setting': effective_options['max_size'],
                'overlap_setting': effective_options['overlap'],
                'language': effective_options.get('language', 'unknown'),
                'relative_position': float((i + 1) / total_chunks_count) if total_chunks_count > 0 else 0.0,
                'adaptive_chunking_used': effective_options.get('adaptive', False),
            }
            current_chunk_metadata.update(chunk_specific_metadata) # Merge method-specific metadata

            if json_content_metadata:
                current_chunk_metadata['initial_document_json_metadata'] = json_content_metadata
            if header_text_content:
                current_chunk_metadata['initial_document_header_text'] = header_text_content
            current_chunk_metadata['chunk_content_hash'] = hashlib.md5(actual_text_content.encode('utf-8')).hexdigest()

            chunks_with_metadata_list.append({
                'text': actual_text_content,
                'metadata': current_chunk_metadata
            })

        logger.debug(f"Successfully created metadata for all {len(chunks_with_metadata_list)} chunks")
        logger.info(f"Improved chunking process completed: {len(chunks_with_metadata_list)} chunks created using method '{effective_options['method']}', language: {effective_options.get('language', 'unknown')}")
        return chunks_with_metadata_list
    except Exception as e:
        logger.error(f"Error creating chunk metadata: {e}", exc_info=True)
        raise ChunkingError(f"Error creating chunk metadata: {e}") from e


# Example of how other functions might change or be integrated:
# load_document can remain a utility function if needed outside, or become a static method.
def load_document(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return re.sub(r'\s+', ' ', text).strip()
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading document {file_path}: {e}")
        raise
