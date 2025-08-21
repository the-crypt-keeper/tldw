# language_chunkers.py
"""
Language-specific chunking implementations.
This module provides language-specific word and sentence tokenization
with graceful fallbacks when dependencies are not available.
"""

import re
from typing import List, Optional, Protocol
from loguru import logger
from ..Utils.optional_deps import get_safe_import, check_dependency


class LanguageChunker(Protocol):
    """Protocol for language-specific chunkers."""
    
    def tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into words."""
        ...
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """Tokenize text into sentences."""
        ...


class ChineseChunker:
    """Chinese language chunker using jieba."""
    
    def __init__(self):
        self.jieba = get_safe_import('jieba')
        self.available = self.jieba is not None
        if not self.available:
            logger.debug("jieba not available, Chinese chunking will fall back to space splitting")
    
    def tokenize_words(self, text: str) -> List[str]:
        """Tokenize Chinese text into words using jieba."""
        if not self.available:
            logger.debug("jieba not available, falling back to character splitting for Chinese")
            # For Chinese, character-level splitting might be more appropriate than space splitting
            return list(text.replace(' ', '').replace('\n', '').replace('\t', ''))
        
        try:
            words = list(self.jieba.cut(text))
            logger.debug(f"jieba tokenized {len(text)} chars into {len(words)} words")
            return words
        except Exception as e:
            logger.warning(f"jieba tokenization failed: {e}, falling back to character splitting")
            return list(text.replace(' ', '').replace('\n', '').replace('\t', ''))
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """Tokenize Chinese text into sentences using punctuation."""
        # Chinese sentence delimiters
        sentences = [s.strip() for s in re.split(r'([。！？；])', text) if s.strip()]
        
        # Join sentence with its delimiter if present
        processed_sentences = []
        temp_sentence = ""
        for part in sentences:
            if part in ['。', '！', '？', '；']:
                if temp_sentence:
                    processed_sentences.append(temp_sentence + part)
                    temp_sentence = ""
            else:
                if temp_sentence:
                    processed_sentences.append(temp_sentence)
                temp_sentence = part
        
        if temp_sentence:
            processed_sentences.append(temp_sentence)
        
        return [s for s in processed_sentences if s]


class JapaneseChunker:
    """Japanese language chunker using fugashi."""
    
    def __init__(self):
        self.fugashi = get_safe_import('fugashi')
        self.tagger = None
        self.available = False
        
        if self.fugashi:
            try:
                self.tagger = self.fugashi.Tagger('-Owakati')
                self.available = True
                logger.debug("fugashi tagger initialized successfully")
            except Exception as e:
                logger.warning(f"fugashi tagger initialization failed: {e}")
                self.available = False
        else:
            logger.debug("fugashi not available, Japanese chunking will fall back to space splitting")
    
    def tokenize_words(self, text: str) -> List[str]:
        """Tokenize Japanese text into words using fugashi."""
        if not self.available:
            logger.debug("fugashi not available, falling back to character splitting for Japanese")
            # For Japanese, character-level might be more appropriate
            return list(text.replace(' ', '').replace('\n', '').replace('\t', ''))
        
        try:
            words = self.tagger.parse(text).split()
            logger.debug(f"fugashi tokenized {len(text)} chars into {len(words)} words")
            return words
        except Exception as e:
            logger.warning(f"fugashi tokenization failed: {e}, falling back to character splitting")
            return list(text.replace(' ', '').replace('\n', '').replace('\t', ''))
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """Tokenize Japanese text into sentences using punctuation."""
        # Japanese sentence delimiters
        sentences = [s.strip() for s in re.split(r'([。！？])', text) if s.strip()]
        
        # Join sentence with its delimiter if present
        processed_sentences = []
        temp_sentence = ""
        for part in sentences:
            if part in ['。', '！', '？']:
                if temp_sentence:
                    processed_sentences.append(temp_sentence + part)
                    temp_sentence = ""
            else:
                if temp_sentence:
                    processed_sentences.append(temp_sentence)
                temp_sentence = part
        
        if temp_sentence:
            processed_sentences.append(temp_sentence)
        
        return [s for s in processed_sentences if s]


class DefaultChunker:
    """Default chunker for languages without specific support."""
    
    def __init__(self, language: str = 'en'):
        self.language = language
        self.nltk = get_safe_import('nltk')
        
    def tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into words using simple space splitting."""
        return text.split()
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """Tokenize text into sentences using NLTK or basic fallback."""
        if self.nltk:
            try:
                from nltk.tokenize import sent_tokenize
                
                # Map language codes to NLTK language names
                nltk_lang_map = {
                    'en': 'english', 'es': 'spanish', 'fr': 'french', 
                    'de': 'german', 'pt': 'portuguese', 'it': 'italian'
                }
                nltk_language = nltk_lang_map.get(self.language.lower(), 'english')
                
                sentences = sent_tokenize(text, language=nltk_language)
                logger.debug(f"NLTK tokenized text into {len(sentences)} sentences")
                return sentences
                
            except LookupError:
                logger.warning(f"NLTK punkt tokenizer not found for language '{self.language}', using basic fallback")
            except Exception as e:
                logger.warning(f"NLTK sentence tokenization failed: {e}, using basic fallback")
        
        # Basic fallback: split on sentence-ending punctuation
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]


class LanguageChunkerFactory:
    """Factory for creating language-specific chunkers."""
    
    @staticmethod
    def get_chunker(language: str) -> LanguageChunker:
        """Get appropriate chunker for the given language."""
        language = language.lower()
        
        if language.startswith('zh'):
            return ChineseChunker()
        elif language == 'ja':
            return JapaneseChunker()
        else:
            return DefaultChunker(language)
    
    @staticmethod
    def get_available_languages() -> List[str]:
        """Get list of languages with enhanced support available."""
        available = ['default']
        
        if check_dependency('jieba'):
            available.append('zh (Chinese)')
        if check_dependency('fugashi'):
            available.append('ja (Japanese)')
            
        return available