# multilingual.py
"""
Enhanced multi-lingual support for the chunking system.
Provides language detection, specialized tokenizers, and language-specific rules.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import re
from loguru import logger


class LanguageFamily(Enum):
    """Language families for grouping similar languages."""
    GERMANIC = "germanic"  # English, German, Dutch, Swedish, etc.
    ROMANCE = "romance"    # Spanish, French, Italian, Portuguese, etc.
    SLAVIC = "slavic"      # Russian, Polish, Czech, etc.
    SINO_TIBETAN = "sino_tibetan"  # Chinese, Burmese, etc.
    JAPONIC = "japonic"    # Japanese
    KOREANIC = "koreanic"  # Korean
    SEMITIC = "semitic"    # Arabic, Hebrew
    INDIC = "indic"        # Hindi, Bengali, etc.
    OTHER = "other"


@dataclass
class LanguageConfig:
    """Configuration for a specific language."""
    code: str                     # ISO 639-1 code
    name: str                     # Full name
    family: LanguageFamily        # Language family
    direction: str = "ltr"        # Text direction: ltr or rtl
    sentence_delimiters: List[str] = None  # Sentence ending markers
    word_tokenizer: Optional[str] = None   # Specialized tokenizer
    requires_spacing: bool = True  # Whether words are space-separated
    
    def __post_init__(self):
        if self.sentence_delimiters is None:
            self.sentence_delimiters = ['.', '!', '?']


class LanguageDetector:
    """Detect language of text using character patterns and heuristics."""
    
    def __init__(self):
        """Initialize language detector with pattern rules."""
        self.patterns = {
            'en': (r'[a-zA-Z]', 0.8),  # English
            'zh': (r'[\u4e00-\u9fff]', 0.3),  # Chinese
            'ja': (r'[\u3040-\u309f\u30a0-\u30ff]', 0.2),  # Japanese
            'ko': (r'[\uac00-\ud7af]', 0.3),  # Korean
            'ar': (r'[\u0600-\u06ff]', 0.3),  # Arabic
            'he': (r'[\u0590-\u05ff]', 0.3),  # Hebrew
            'ru': (r'[\u0400-\u04ff]', 0.3),  # Russian/Cyrillic
            'hi': (r'[\u0900-\u097f]', 0.3),  # Hindi/Devanagari
            'th': (r'[\u0e00-\u0e7f]', 0.3),  # Thai
            'el': (r'[\u0370-\u03ff]', 0.3),  # Greek
        }
        
        # Common words for language detection
        self.common_words = {
            'en': ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that'],
            'es': ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'por'],
            'fr': ['le', 'de', 'et', 'la', 'les', 'un', 'une', 'que'],
            'de': ['der', 'die', 'und', 'in', 'das', 'ist', 'ein', 'zu'],
            'it': ['il', 'di', 'e', 'la', 'che', 'un', 'per', 'in'],
            'pt': ['o', 'de', 'e', 'a', 'que', 'em', 'um', 'para'],
            'ru': ['и', 'в', 'на', 'что', 'с', 'не', 'это', 'как'],
            'zh': ['的', '是', '在', '和', '了', '有', '我', '你'],
            'ja': ['の', 'は', 'を', 'が', 'に', 'で', 'と', 'も'],
        }
    
    def detect(self, text: str) -> Tuple[str, float]:
        """
        Detect language of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (language_code, confidence)
        """
        if not text:
            return 'en', 0.0
        
        # Check character-based patterns
        scores = {}
        text_length = len(text)
        
        for lang, (pattern, threshold) in self.patterns.items():
            matches = len(re.findall(pattern, text))
            ratio = matches / text_length if text_length > 0 else 0
            
            if ratio >= threshold:
                scores[lang] = ratio
        
        # If strong character pattern match, return it
        if scores:
            best_lang = max(scores.items(), key=lambda x: x[1])
            if best_lang[1] >= 0.5:
                return best_lang[0], best_lang[1]
        
        # Check common words
        text_lower = text.lower()
        words = text_lower.split()
        
        for lang, common in self.common_words.items():
            matches = sum(1 for word in words if word in common)
            if matches >= 3:
                confidence = min(matches / 10.0, 1.0)
                return lang, confidence
        
        # Default to English with low confidence
        return 'en', 0.1


class MultilingualTokenizer:
    """
    Multi-lingual tokenizer with language-specific rules.
    """
    
    def __init__(self):
        """Initialize tokenizer with language configurations."""
        self.configs = self._init_language_configs()
        self.detector = LanguageDetector()
        self._tokenizers = {}
        
        logger.debug("MultilingualTokenizer initialized")
    
    def _init_language_configs(self) -> Dict[str, LanguageConfig]:
        """Initialize language configurations."""
        configs = {
            # Germanic languages
            'en': LanguageConfig('en', 'English', LanguageFamily.GERMANIC),
            'de': LanguageConfig('de', 'German', LanguageFamily.GERMANIC,
                               sentence_delimiters=['.', '!', '?', '。']),
            'nl': LanguageConfig('nl', 'Dutch', LanguageFamily.GERMANIC),
            'sv': LanguageConfig('sv', 'Swedish', LanguageFamily.GERMANIC),
            
            # Romance languages
            'es': LanguageConfig('es', 'Spanish', LanguageFamily.ROMANCE,
                               sentence_delimiters=['.', '!', '?', '¡', '¿']),
            'fr': LanguageConfig('fr', 'French', LanguageFamily.ROMANCE,
                               sentence_delimiters=['.', '!', '?', '»', '«']),
            'it': LanguageConfig('it', 'Italian', LanguageFamily.ROMANCE),
            'pt': LanguageConfig('pt', 'Portuguese', LanguageFamily.ROMANCE),
            
            # Slavic languages
            'ru': LanguageConfig('ru', 'Russian', LanguageFamily.SLAVIC,
                               sentence_delimiters=['.', '!', '?', '。', '！', '？']),
            'pl': LanguageConfig('pl', 'Polish', LanguageFamily.SLAVIC),
            
            # Asian languages
            'zh': LanguageConfig('zh', 'Chinese', LanguageFamily.SINO_TIBETAN,
                               sentence_delimiters=['。', '！', '？', '；', '.', '!', '?'],
                               word_tokenizer='jieba',
                               requires_spacing=False),
            'ja': LanguageConfig('ja', 'Japanese', LanguageFamily.JAPONIC,
                               sentence_delimiters=['。', '！', '？', '.', '!', '?'],
                               word_tokenizer='fugashi',
                               requires_spacing=False),
            'ko': LanguageConfig('ko', 'Korean', LanguageFamily.KOREANIC,
                               sentence_delimiters=['。', '！', '？', '.', '!', '?'],
                               word_tokenizer='konlpy',
                               requires_spacing=True),
            
            # Semitic languages
            'ar': LanguageConfig('ar', 'Arabic', LanguageFamily.SEMITIC,
                               direction='rtl',
                               sentence_delimiters=['.', '!', '?', '؟', '۔']),
            'he': LanguageConfig('he', 'Hebrew', LanguageFamily.SEMITIC,
                               direction='rtl',
                               sentence_delimiters=['.', '!', '?']),
            
            # Indic languages
            'hi': LanguageConfig('hi', 'Hindi', LanguageFamily.INDIC,
                               sentence_delimiters=['।', '.', '!', '?']),
            
            # Other
            'th': LanguageConfig('th', 'Thai', LanguageFamily.OTHER,
                               word_tokenizer='pythainlp',
                               requires_spacing=False),
        }
        
        return configs
    
    def get_config(self, language: str) -> LanguageConfig:
        """
        Get configuration for a language.
        
        Args:
            language: Language code
            
        Returns:
            Language configuration
        """
        return self.configs.get(language, self.configs['en'])
    
    def tokenize_words(self, text: str, language: Optional[str] = None) -> List[str]:
        """
        Tokenize text into words using language-specific rules.
        
        Args:
            text: Text to tokenize
            language: Language code (auto-detected if not provided)
            
        Returns:
            List of word tokens
        """
        # Auto-detect language if not provided
        if not language:
            language, _ = self.detector.detect(text)
        
        config = self.get_config(language)
        
        # Use specialized tokenizer if available
        if config.word_tokenizer:
            return self._tokenize_with_library(text, config.word_tokenizer)
        
        # Use simple space-based tokenization for space-separated languages
        if config.requires_spacing:
            return text.split()
        
        # For non-spaced languages without specialized tokenizer,
        # fall back to character-based tokenization
        return list(text)
    
    def _tokenize_with_library(self, text: str, tokenizer_name: str) -> List[str]:
        """
        Use specialized tokenization library.
        
        Args:
            text: Text to tokenize
            tokenizer_name: Name of tokenizer library
            
        Returns:
            List of tokens
        """
        try:
            if tokenizer_name == 'jieba':
                import jieba
                return list(jieba.cut(text))
            
            elif tokenizer_name == 'fugashi':
                try:
                    import fugashi
                    tagger = fugashi.Tagger()
                    return [word.surface for word in tagger(text)]
                except ImportError:
                    logger.warning("fugashi not available, using character tokenization")
                    return list(text)
            
            elif tokenizer_name == 'konlpy':
                try:
                    from konlpy.tag import Okt
                    okt = Okt()
                    return okt.morphs(text)
                except ImportError:
                    logger.warning("konlpy not available, using space tokenization")
                    return text.split()
            
            elif tokenizer_name == 'pythainlp':
                try:
                    from pythainlp import word_tokenize
                    return word_tokenize(text)
                except ImportError:
                    logger.warning("pythainlp not available, using character tokenization")
                    return list(text)
            
            else:
                logger.warning(f"Unknown tokenizer: {tokenizer_name}")
                return text.split()
                
        except Exception as e:
            logger.error(f"Tokenization error with {tokenizer_name}: {e}")
            return text.split() if ' ' in text else list(text)
    
    def tokenize_sentences(self, text: str, language: Optional[str] = None) -> List[str]:
        """
        Tokenize text into sentences using language-specific rules.
        
        Args:
            text: Text to tokenize
            language: Language code (auto-detected if not provided)
            
        Returns:
            List of sentence tokens
        """
        # Auto-detect language if not provided
        if not language:
            language, _ = self.detector.detect(text)
        
        config = self.get_config(language)
        
        # Build regex pattern from delimiters
        delimiters = config.sentence_delimiters
        pattern = '|'.join(re.escape(d) for d in delimiters)
        pattern = f'(?<=[{pattern}])\\s+'
        
        # Split on sentence delimiters
        sentences = re.split(pattern, text)
        
        # Filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences


class LanguageAdapter:
    """
    Adapter for integrating multi-lingual support into chunking strategies.
    """
    
    def __init__(self):
        """Initialize language adapter."""
        self.tokenizer = MultilingualTokenizer()
        self.detector = LanguageDetector()
        
        # Cache for detected languages
        self._language_cache = {}
        
        logger.debug("LanguageAdapter initialized")
    
    def adapt_strategy(self, strategy: Any, text: str) -> str:
        """
        Adapt a chunking strategy for the detected language.
        
        Args:
            strategy: Chunking strategy instance
            text: Text to process
            
        Returns:
            Detected language code
        """
        # Detect language
        language, confidence = self.detector.detect(text)
        
        # Update strategy language if different
        if hasattr(strategy, 'language') and strategy.language != language:
            logger.info(f"Adapting strategy from {strategy.language} to {language} (confidence: {confidence:.2f})")
            strategy.language = language
        
        return language
    
    def get_sentence_splitter(self, language: str):
        """
        Get sentence splitter function for a language.
        
        Args:
            language: Language code
            
        Returns:
            Sentence splitter function
        """
        def splitter(text: str) -> List[str]:
            return self.tokenizer.tokenize_sentences(text, language)
        
        return splitter
    
    def get_word_tokenizer(self, language: str):
        """
        Get word tokenizer function for a language.
        
        Args:
            language: Language code
            
        Returns:
            Word tokenizer function
        """
        def tokenizer(text: str) -> List[str]:
            return self.tokenizer.tokenize_words(text, language)
        
        return tokenizer
    
    def preprocess_text(self, text: str, language: Optional[str] = None) -> str:
        """
        Preprocess text based on language-specific rules.
        
        Args:
            text: Text to preprocess
            language: Language code (auto-detected if not provided)
            
        Returns:
            Preprocessed text
        """
        if not language:
            language, _ = self.detector.detect(text)
        
        config = self.tokenizer.get_config(language)
        
        # Handle RTL languages
        if config.direction == 'rtl':
            # Add RTL mark for proper display
            text = '\u202B' + text + '\u202C'
        
        # Normalize whitespace for languages that use it
        if config.requires_spacing:
            text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def get_optimal_chunk_size(self, language: str, base_size: int) -> int:
        """
        Get optimal chunk size for a language.
        
        Args:
            language: Language code
            base_size: Base chunk size in words
            
        Returns:
            Adjusted chunk size
        """
        config = self.tokenizer.get_config(language)
        
        # Adjust chunk size based on language characteristics
        adjustments = {
            LanguageFamily.SINO_TIBETAN: 2.0,  # Chinese characters are more dense
            LanguageFamily.JAPONIC: 1.5,       # Japanese is somewhat dense
            LanguageFamily.KOREANIC: 1.2,      # Korean is moderately dense
            LanguageFamily.SEMITIC: 0.8,       # Arabic/Hebrew may need smaller chunks
        }
        
        multiplier = adjustments.get(config.family, 1.0)
        return int(base_size * multiplier)