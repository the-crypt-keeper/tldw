# Audio_Buffered_Transcription.py
#########################################
# Advanced Buffered/Chunked Transcription for Long Audio
# Based on NVIDIA's buffered inference implementation for RNNT models
# Supports multiple merge algorithms for optimal transcription quality
#
####################
# Function List
#
# 1. BufferedTranscriber - Base class for buffered transcription
# 2. MiddleMergeTranscriber - Middle token merge algorithm
# 3. LCSMergeTranscriber - Longest Common Subsequence merge
# 4. TDTMergeTranscriber - TDT-specific merge algorithm
# 5. transcribe_long_audio() - Main entry point for long audio transcription
#
####################

import math
import logging
from typing import Optional, List, Tuple, Union, Callable, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class MergeAlgorithm(Enum):
    """Supported merge algorithms for chunked transcription."""
    MIDDLE = "middle"      # Middle token merge (default for RNNT)
    LCS = "lcs"           # Longest Common Subsequence
    TDT = "tdt"           # TDT-specific algorithm
    OVERLAP = "overlap"    # Simple overlap removal
    SIMPLE = "simple"      # Simple concatenation


@dataclass
class BufferedTranscriptionConfig:
    """Configuration for buffered transcription."""
    chunk_duration: float = 2.0        # Chunk length in seconds
    total_buffer: float = 4.0          # Total buffer (chunk + padding) in seconds
    batch_size: int = 1                # Batch size for processing
    merge_algo: MergeAlgorithm = MergeAlgorithm.MIDDLE
    max_steps_per_timestep: int = 5    # For RNNT decoding
    stateful_decoding: bool = False    # Maintain state between chunks
    model_stride: float = 0.08         # Model's time stride in seconds (e.g., 80ms)
    preserve_timestamps: bool = False   # Preserve timing information
    device: str = 'cpu'                # Device for processing


class BufferedTranscriber:
    """
    Base class for buffered transcription of long audio files.
    
    Implements chunking and buffering strategy similar to NVIDIA's approach.
    """
    
    def __init__(self, config: BufferedTranscriptionConfig):
        """Initialize buffered transcriber."""
        self.config = config
        self.transcription_history = []
        self.timestamp_history = []
        
        # Calculate chunk parameters
        self.chunk_samples_at_16k = int(config.chunk_duration * 16000)
        self.buffer_samples_at_16k = int(config.total_buffer * 16000)
        
        # Calculate tokens per chunk based on model stride
        self.tokens_per_chunk = math.ceil(config.chunk_duration / config.model_stride)
        
        # Calculate delays for merging
        left_padding = (config.total_buffer - config.chunk_duration) / 2
        self.mid_delay = math.ceil((config.chunk_duration + left_padding) / config.model_stride)
    
    def process_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        transcribe_fn: Callable[[np.ndarray], str],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> str:
        """
        Process long audio with chunking and merging.
        
        Args:
            audio_data: Complete audio data
            sample_rate: Sample rate of audio
            transcribe_fn: Function to transcribe a single chunk
            progress_callback: Optional progress callback
        
        Returns:
            Complete transcription
        """
        # Resample if needed
        if sample_rate != 16000:
            audio_data = self._resample(audio_data, sample_rate, 16000)
            sample_rate = 16000
        
        # Calculate chunks
        chunks = self._create_chunks(audio_data)
        num_chunks = len(chunks)
        
        logger.info(f"Processing {num_chunks} chunks with {self.config.merge_algo.value} algorithm")
        
        # Process each chunk
        chunk_results = []
        for i, chunk in enumerate(chunks):
            # Transcribe chunk
            result = transcribe_fn(chunk['audio'])
            
            # Store result with metadata
            chunk_results.append({
                'text': result,
                'start': chunk['start'],
                'end': chunk['end'],
                'overlap_start': chunk.get('overlap_start', 0),
                'overlap_end': chunk.get('overlap_end', 0)
            })
            
            # Progress callback
            if progress_callback:
                progress_callback(i + 1, num_chunks)
        
        # Merge results
        merged_text = self._merge_results(chunk_results)
        
        return merged_text
    
    def _create_chunks(self, audio_data: np.ndarray) -> List[dict]:
        """
        Create overlapping chunks from audio data.
        
        Returns list of chunk dictionaries with audio and metadata.
        """
        chunks = []
        total_samples = len(audio_data)
        
        # Calculate stride (non-overlapping part)
        padding_samples = self.buffer_samples_at_16k - self.chunk_samples_at_16k
        stride_samples = self.chunk_samples_at_16k - padding_samples // 2
        
        # Create chunks
        position = 0
        while position < total_samples:
            # Calculate chunk boundaries
            chunk_start = max(0, position - padding_samples // 2)
            chunk_end = min(total_samples, position + self.chunk_samples_at_16k + padding_samples // 2)
            
            # Extract chunk
            chunk_audio = audio_data[chunk_start:chunk_end]
            
            # Pad if needed
            if len(chunk_audio) < self.buffer_samples_at_16k:
                chunk_audio = np.pad(
                    chunk_audio,
                    (0, self.buffer_samples_at_16k - len(chunk_audio)),
                    mode='constant'
                )
            
            chunks.append({
                'audio': chunk_audio,
                'start': chunk_start / 16000,  # Convert to seconds
                'end': chunk_end / 16000,
                'overlap_start': (position - chunk_start) / 16000 if position > chunk_start else 0,
                'overlap_end': (chunk_end - (position + self.chunk_samples_at_16k)) / 16000 
                             if chunk_end > position + self.chunk_samples_at_16k else 0
            })
            
            # Move to next position
            position += stride_samples
        
        return chunks
    
    def _merge_results(self, chunk_results: List[dict]) -> str:
        """
        Merge chunk results based on configured algorithm.
        
        Override this in subclasses for specific merge strategies.
        """
        if self.config.merge_algo == MergeAlgorithm.SIMPLE:
            # Simple concatenation
            return ' '.join(r['text'] for r in chunk_results if r['text'])
        else:
            # Default to middle merge
            return self._middle_merge(chunk_results)
    
    def _middle_merge(self, chunk_results: List[dict]) -> str:
        """
        Implement middle token merge algorithm.
        
        Removes overlapping portions from chunk boundaries.
        """
        if not chunk_results:
            return ""
        
        merged_texts = []
        
        for i, result in enumerate(chunk_results):
            text = result['text']
            if not text:
                continue
            
            if i > 0 and result['overlap_start'] > 0:
                # Remove overlapping start portion
                words = text.split()
                overlap_ratio = result['overlap_start'] / self.config.chunk_duration
                words_to_skip = int(len(words) * overlap_ratio * 0.5)  # Take middle
                text = ' '.join(words[words_to_skip:]) if words_to_skip < len(words) else text
            
            if i < len(chunk_results) - 1 and result['overlap_end'] > 0:
                # Remove overlapping end portion
                words = text.split()
                overlap_ratio = result['overlap_end'] / self.config.chunk_duration
                words_to_keep = len(words) - int(len(words) * overlap_ratio * 0.5)
                text = ' '.join(words[:words_to_keep]) if words_to_keep > 0 else text
            
            merged_texts.append(text)
        
        return ' '.join(merged_texts)
    
    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            logger.warning("librosa not available, returning original audio")
            return audio


class LCSMergeTranscriber(BufferedTranscriber):
    """
    Transcriber using Longest Common Subsequence merge algorithm.
    
    Better handling of overlapping regions by finding common sequences.
    """
    
    def _merge_results(self, chunk_results: List[dict]) -> str:
        """Merge using LCS algorithm."""
        if not chunk_results:
            return ""
        
        if len(chunk_results) == 1:
            return chunk_results[0]['text']
        
        # Start with first chunk
        merged = chunk_results[0]['text']
        
        for i in range(1, len(chunk_results)):
            current = chunk_results[i]['text']
            if not current:
                continue
            
            # Find LCS between end of merged and start of current
            merged = self._merge_with_lcs(merged, current)
        
        return merged
    
    def _merge_with_lcs(self, text1: str, text2: str) -> str:
        """
        Merge two texts using LCS to find overlap.
        """
        words1 = text1.split()
        words2 = text2.split()
        
        if not words1:
            return text2
        if not words2:
            return text1
        
        # Look for overlap in last part of text1 and first part of text2
        max_overlap = min(len(words1), len(words2), 10)  # Limit search
        
        best_overlap = 0
        for overlap_size in range(max_overlap, 0, -1):
            if words1[-overlap_size:] == words2[:overlap_size]:
                best_overlap = overlap_size
                break
        
        if best_overlap > 0:
            # Merge without duplicating overlap
            return ' '.join(words1 + words2[best_overlap:])
        else:
            # No overlap found, concatenate
            return text1 + ' ' + text2
    
    def _lcs_length(self, X: List[str], Y: List[str]) -> int:
        """
        Calculate length of longest common subsequence.
        """
        m, n = len(X), len(Y)
        L = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif X[i-1] == Y[j-1]:
                    L[i][j] = L[i-1][j-1] + 1
                else:
                    L[i][j] = max(L[i-1][j], L[i][j-1])
        
        return L[m][n]


def transcribe_long_audio(
    audio_path: Union[str, Path, np.ndarray],
    model_name: str = 'parakeet',
    variant: str = 'mlx',
    chunk_duration: float = 30.0,
    total_buffer: float = 40.0,
    merge_algo: str = 'middle',
    device: str = 'cpu',
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> str:
    """
    Transcribe long audio files using buffered/chunked processing.
    
    Args:
        audio_path: Path to audio file or numpy array
        model_name: Model to use ('parakeet', 'canary')
        variant: Model variant ('standard', 'onnx', 'mlx')
        chunk_duration: Duration of each chunk in seconds
        total_buffer: Total buffer size (chunk + padding)
        merge_algo: Merge algorithm ('middle', 'lcs', 'overlap', 'simple')
        device: Device for processing
        progress_callback: Optional callback for progress
    
    Returns:
        Complete transcription
    """
    import soundfile as sf
    
    # Load audio
    if isinstance(audio_path, (str, Path)):
        audio_data, sample_rate = sf.read(str(audio_path))
    else:
        audio_data = audio_path
        sample_rate = 16000  # Assume 16kHz
    
    # Convert to mono if needed
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Create config
    config = BufferedTranscriptionConfig(
        chunk_duration=chunk_duration,
        total_buffer=total_buffer,
        merge_algo=MergeAlgorithm(merge_algo),
        device=device
    )
    
    # Select transcriber based on merge algorithm
    if config.merge_algo == MergeAlgorithm.LCS:
        transcriber = LCSMergeTranscriber(config)
    else:
        transcriber = BufferedTranscriber(config)
    
    # Create transcribe function
    def transcribe_chunk(chunk_audio: np.ndarray) -> str:
        # Import appropriate transcription function
        if variant == 'mlx':
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX import (
                transcribe_with_parakeet_mlx
            )
            return transcribe_with_parakeet_mlx(chunk_audio, sample_rate=16000)
        elif variant == 'onnx':
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX import (
                transcribe_with_parakeet_onnx
            )
            return transcribe_with_parakeet_onnx(chunk_audio, sample_rate=16000, device=device)
        else:
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
                transcribe_with_nemo
            )
            return transcribe_with_nemo(
                chunk_audio,
                sample_rate=16000,
                model=model_name,
                variant=variant
            )
    
    # Process audio
    result = transcriber.process_audio(
        audio_data,
        sample_rate,
        transcribe_chunk,
        progress_callback
    )
    
    return result


#######################################################################################################################
# End of Audio_Buffered_Transcription.py
#######################################################################################################################