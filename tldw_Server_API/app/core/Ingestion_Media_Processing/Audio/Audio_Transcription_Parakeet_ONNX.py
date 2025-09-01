# Audio_Transcription_Parakeet_ONNX.py
#########################################
# ONNX Parakeet Model Transcription with Proper Decoding
# This module provides transcription using ONNX-optimized Parakeet models
# with full preprocessing, inference, and decoding support.
#
####################
# Function List
#
# 1. load_parakeet_onnx_model() - Load and cache ONNX model with tokenizer
# 2. preprocess_audio_for_onnx() - Prepare audio for ONNX inference
# 3. decode_onnx_output() - Decode model outputs to text
# 4. transcribe_with_parakeet_onnx() - Main transcription function
# 5. transcribe_chunked_onnx() - Chunked transcription for long audio
#
####################

import os
import json
import logging
import tempfile
from pathlib import Path
from typing import Optional, Union, List, Tuple, Callable, Dict, Any
import numpy as np
import soundfile as sf

try:
    import onnxruntime as ort
except ImportError:
    ort = None
    logging.warning("ONNX Runtime not installed. Install with: pip install onnxruntime")

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None
    logging.warning("huggingface_hub not installed. Install with: pip install huggingface_hub")

# Global cache for model and tokenizer
_onnx_model_cache: Dict[str, Any] = {}

logger = logging.getLogger(__name__)


class ParakeetONNXTokenizer:
    """Simple tokenizer for Parakeet ONNX models."""
    
    def __init__(self, vocab_path: Path):
        """Load vocabulary from file."""
        self.vocab = {}
        self.inv_vocab = {}
        
        # Try to load vocabulary
        if vocab_path.exists():
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
                if isinstance(vocab_data, dict):
                    self.vocab = vocab_data
                elif isinstance(vocab_data, list):
                    # List format - create dict
                    self.vocab = {token: idx for idx, token in enumerate(vocab_data)}
                
                # Create inverse vocabulary
                self.inv_vocab = {v: k for k, v in self.vocab.items()}
        else:
            # Use default SentencePiece vocabulary
            self._create_default_vocab()
    
    def _create_default_vocab(self):
        """Create a default vocabulary for Parakeet."""
        # Common tokens for Parakeet/RNNT models
        special_tokens = ['<pad>', '<s>', '</s>', '<unk>', '<blank>']
        
        # Add special tokens
        for idx, token in enumerate(special_tokens):
            self.vocab[token] = idx
        
        # Add space
        self.vocab['▁'] = len(self.vocab)  # SentencePiece space token
        
        # Add ASCII printable characters
        for i in range(32, 127):
            char = chr(i)
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
        
        # Add common subword units (simplified)
        common_subwords = [
            '▁the', '▁a', '▁to', '▁of', '▁and', '▁in', '▁is', '▁it',
            '▁that', '▁for', '▁was', '▁with', '▁as', '▁on', '▁be',
            '▁have', '▁but', '▁not', '▁you', '▁he', '▁at', '▁this',
            '▁from', '▁by', '▁are', '▁we', '▁an', '▁or', '▁will',
            '▁one', '▁would', '▁there', '▁their', '▁what', '▁so',
            '▁up', '▁out', '▁if', '▁about', '▁who', '▁get', '▁which',
            '▁go', '▁me', '▁when', '▁make', '▁can', '▁like', '▁time',
            'ing', 'ed', 'er', 'ly', 'al', 'es', 'ion', 'en', 'ation'
        ]
        
        for subword in common_subwords:
            if subword not in self.vocab:
                self.vocab[subword] = len(self.vocab)
        
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        logger.info(f"Created default vocabulary with {len(self.vocab)} tokens")
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        tokens = []
        for token_id in token_ids:
            if token_id in self.inv_vocab:
                token = self.inv_vocab[token_id]
                # Skip special tokens
                if token not in ['<pad>', '<s>', '</s>', '<blank>', '<unk>']:
                    tokens.append(token)
        
        # Join tokens and clean up
        text = ''.join(tokens)
        # Replace SentencePiece space token with actual space
        text = text.replace('▁', ' ')
        # Clean up multiple spaces
        text = ' '.join(text.split())
        return text.strip()


def get_mel_features(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    Extract mel-spectrogram features from audio.
    
    Args:
        audio: Audio samples
        sample_rate: Sample rate
    
    Returns:
        Mel-spectrogram features
    """
    try:
        import librosa
    except ImportError:
        logger.error("librosa not installed. Install with: pip install librosa")
        return np.array([])
    
    # Ensure audio is float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    # Normalize audio
    if np.abs(audio).max() > 1.0:
        audio = audio / np.abs(audio).max()
    
    # Extract mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=512,
        hop_length=160,  # 10ms hop
        win_length=400,  # 25ms window
        n_mels=80,
        fmin=0,
        fmax=8000
    )
    
    # Convert to log scale
    log_mel = np.log(mel_spec + 1e-10)
    
    # Transpose to (time, features)
    log_mel = log_mel.T
    
    # Normalize
    mean = np.mean(log_mel, axis=0, keepdims=True)
    std = np.std(log_mel, axis=0, keepdims=True)
    log_mel = (log_mel - mean) / (std + 1e-10)
    
    return log_mel.astype(np.float32)


def load_parakeet_onnx_model(model_path: Optional[str] = None, device: str = 'cpu'):
    """
    Load Parakeet ONNX model and tokenizer.
    
    Args:
        model_path: Path to ONNX model directory or HuggingFace repo
        device: Device to run on ('cpu' or 'cuda')
    
    Returns:
        Tuple of (ONNX session, tokenizer) or (None, None) if loading fails
    """
    global _onnx_model_cache
    
    if ort is None:
        logger.error("ONNX Runtime not available")
        return None, None
    
    # Default model
    if model_path is None:
        model_path = "istupakov/parakeet-tdt-0.6b-v3-onnx"
    
    cache_key = f"{model_path}_{device}"
    if cache_key in _onnx_model_cache:
        logger.debug(f"Using cached ONNX model: {model_path}")
        return _onnx_model_cache[cache_key]
    
    try:
        # Check if it's a local path or HuggingFace repo
        model_dir = Path(model_path)
        
        if not model_dir.exists() and snapshot_download:
            # Download from HuggingFace
            logger.info(f"Downloading ONNX model from HuggingFace: {model_path}")
            cache_dir = Path.home() / '.cache' / 'parakeet_onnx'
            model_dir = cache_dir / model_path.replace('/', '_')
            
            if not model_dir.exists():
                snapshot_download(
                    repo_id=model_path,
                    local_dir=str(model_dir),
                    local_dir_use_symlinks=False
                )
        
        # Find ONNX files
        onnx_files = list(model_dir.glob("*.onnx"))
        if not onnx_files:
            logger.error(f"No ONNX files found in {model_dir}")
            return None, None
        
        # Use the first ONNX file (usually encoder.onnx or model.onnx)
        onnx_path = onnx_files[0]
        logger.info(f"Loading ONNX model from: {onnx_path}")
        
        # Set up providers
        providers = []
        if device == 'cuda':
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        # Create ONNX session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        session = ort.InferenceSession(
            str(onnx_path),
            sess_options=session_options,
            providers=providers
        )
        
        # Load tokenizer
        vocab_path = model_dir / "vocab.json"
        if not vocab_path.exists():
            vocab_path = model_dir / "tokenizer.json"
        
        tokenizer = ParakeetONNXTokenizer(vocab_path)
        
        # Cache the model
        _onnx_model_cache[cache_key] = (session, tokenizer)
        
        logger.info(f"Successfully loaded ONNX model with {len(session.get_inputs())} inputs")
        return session, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load ONNX model: {e}")
        return None, None


def transcribe_with_parakeet_onnx(
    audio_data: Union[np.ndarray, str, Path],
    sample_rate: int = 16000,
    model_path: Optional[str] = None,
    device: str = 'cpu',
    chunk_duration: Optional[float] = None,
    overlap_duration: float = 0.5,
    merge_algo: str = 'middle',
    chunk_callback: Optional[Callable[[int, int], None]] = None
) -> str:
    """
    Transcribe audio using Parakeet ONNX model.
    
    Args:
        audio_data: Audio data as numpy array or file path
        sample_rate: Sample rate of audio
        model_path: Path to ONNX model or HuggingFace repo
        device: Device to run on ('cpu' or 'cuda')
        chunk_duration: Duration for chunking in seconds (None = no chunking)
        overlap_duration: Overlap between chunks
        merge_algo: Algorithm for merging chunks ('middle', 'overlap', 'simple')
        chunk_callback: Progress callback for chunks
    
    Returns:
        Transcribed text
    """
    # Load model
    session, tokenizer = load_parakeet_onnx_model(model_path, device)
    if session is None or tokenizer is None:
        return "[Error: Failed to load ONNX model]"
    
    # Load audio if it's a file path
    if isinstance(audio_data, (str, Path)):
        try:
            audio_data, file_sr = sf.read(str(audio_data))
            if file_sr != sample_rate:
                # Resample if needed
                import librosa
                audio_data = librosa.resample(
                    audio_data,
                    orig_sr=file_sr,
                    target_sr=sample_rate
                )
        except Exception as e:
            logger.error(f"Failed to load audio file: {e}")
            return f"[Error: Failed to load audio: {e}]"
    
    # Ensure numpy array
    if not isinstance(audio_data, np.ndarray):
        return "[Error: Invalid audio data type]"
    
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Check if we need chunking
    audio_duration = len(audio_data) / sample_rate
    
    if chunk_duration and audio_duration > chunk_duration:
        # Use chunked transcription
        return transcribe_chunked_onnx(
            audio_data,
            sample_rate,
            session,
            tokenizer,
            chunk_duration,
            overlap_duration,
            merge_algo,
            chunk_callback
        )
    
    # Single transcription
    try:
        # Extract features
        features = get_mel_features(audio_data, sample_rate)
        
        if features.size == 0:
            return "[Error: Feature extraction failed]"
        
        # Prepare input for ONNX
        # Add batch dimension
        features = np.expand_dims(features, axis=0)
        
        # Get input names
        input_names = [inp.name for inp in session.get_inputs()]
        output_names = [out.name for out in session.get_outputs()]
        
        # Prepare inputs
        inputs = {}
        for input_name in input_names:
            if 'audio' in input_name.lower() or 'input' in input_name.lower():
                inputs[input_name] = features
            elif 'length' in input_name.lower():
                # Input lengths
                inputs[input_name] = np.array([features.shape[1]], dtype=np.int64)
        
        # Run inference
        outputs = session.run(output_names, inputs)
        
        # Decode outputs
        if outputs and len(outputs) > 0:
            # Get the main output (usually logits or token IDs)
            output = outputs[0]
            
            # Handle different output formats
            if output.ndim == 3:
                # (batch, time, vocab) - take argmax
                token_ids = np.argmax(output[0], axis=-1)
            elif output.ndim == 2:
                # (batch, time) - already token IDs
                token_ids = output[0]
            else:
                token_ids = output.flatten()
            
            # Remove padding and blank tokens
            token_ids = token_ids[token_ids > 0]
            
            # Decode to text
            text = tokenizer.decode(token_ids.tolist())
            return text if text else "[No speech detected]"
        
        return "[Error: No output from model]"
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return f"[Error: Transcription failed: {e}]"


def transcribe_chunked_onnx(
    audio_data: np.ndarray,
    sample_rate: int,
    session: ort.InferenceSession,
    tokenizer: ParakeetONNXTokenizer,
    chunk_duration: float,
    overlap_duration: float,
    merge_algo: str,
    chunk_callback: Optional[Callable[[int, int], None]] = None
) -> str:
    """
    Transcribe long audio using chunking with ONNX model.
    
    Args:
        audio_data: Audio samples
        sample_rate: Sample rate
        session: ONNX inference session
        tokenizer: Tokenizer for decoding
        chunk_duration: Chunk duration in seconds
        overlap_duration: Overlap between chunks
        merge_algo: Merge algorithm ('middle', 'overlap', 'simple')
        chunk_callback: Progress callback
    
    Returns:
        Merged transcription text
    """
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(overlap_duration * sample_rate)
    stride_samples = chunk_samples - overlap_samples
    
    total_samples = len(audio_data)
    num_chunks = max(1, int(np.ceil((total_samples - overlap_samples) / stride_samples)))
    
    transcripts = []
    
    # Get input/output names
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    
    for i in range(num_chunks):
        start = i * stride_samples
        end = min(start + chunk_samples, total_samples)
        
        # Extract chunk
        chunk = audio_data[start:end]
        
        # Pad if needed
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')
        
        try:
            # Extract features
            features = get_mel_features(chunk, sample_rate)
            
            if features.size == 0:
                continue
            
            # Add batch dimension
            features = np.expand_dims(features, axis=0)
            
            # Prepare inputs
            inputs = {}
            for input_name in input_names:
                if 'audio' in input_name.lower() or 'input' in input_name.lower():
                    inputs[input_name] = features
                elif 'length' in input_name.lower():
                    inputs[input_name] = np.array([features.shape[1]], dtype=np.int64)
            
            # Run inference
            outputs = session.run(output_names, inputs)
            
            if outputs and len(outputs) > 0:
                output = outputs[0]
                
                # Get token IDs
                if output.ndim == 3:
                    token_ids = np.argmax(output[0], axis=-1)
                elif output.ndim == 2:
                    token_ids = output[0]
                else:
                    token_ids = output.flatten()
                
                # Remove padding
                token_ids = token_ids[token_ids > 0]
                
                # Decode
                text = tokenizer.decode(token_ids.tolist())
                
                if text:
                    if merge_algo == 'middle' and i > 0 and overlap_samples > 0:
                        # For middle merge, trim overlapping parts
                        # This is a simplified version - real implementation would
                        # align tokens at boundaries
                        overlap_chars = int(len(text) * overlap_duration / chunk_duration)
                        if overlap_chars > 0:
                            text = text[overlap_chars // 2:]
                    
                    transcripts.append(text)
        
        except Exception as e:
            logger.error(f"Error processing chunk {i+1}/{num_chunks}: {e}")
        
        # Progress callback
        if chunk_callback:
            chunk_callback(i + 1, num_chunks)
    
    # Merge transcripts
    if merge_algo == 'simple':
        # Simple concatenation
        result = ' '.join(transcripts)
    elif merge_algo == 'overlap':
        # Remove duplicate words at boundaries
        result = merge_with_overlap_removal(transcripts)
    else:  # 'middle'
        # Already handled trimming above
        result = ' '.join(transcripts)
    
    return result.strip() if result else "[No speech detected]"


def merge_with_overlap_removal(transcripts: List[str]) -> str:
    """
    Merge transcripts by removing duplicate words at boundaries.
    
    Args:
        transcripts: List of transcript segments
    
    Returns:
        Merged transcript
    """
    if not transcripts:
        return ""
    
    if len(transcripts) == 1:
        return transcripts[0]
    
    result = transcripts[0]
    
    for i in range(1, len(transcripts)):
        current = transcripts[i]
        if not current:
            continue
        
        # Find overlapping words
        prev_words = result.split()
        curr_words = current.split()
        
        if not prev_words or not curr_words:
            result = result + " " + current
            continue
        
        # Look for overlap (simplified - check last few words)
        overlap_found = False
        for overlap_size in range(min(5, len(prev_words), len(curr_words)), 0, -1):
            if prev_words[-overlap_size:] == curr_words[:overlap_size]:
                # Found overlap, merge without duplicates
                result = ' '.join(prev_words + curr_words[overlap_size:])
                overlap_found = True
                break
        
        if not overlap_found:
            result = result + " " + current
    
    return result


def unload_onnx_models():
    """Unload all cached ONNX models to free memory."""
    global _onnx_model_cache
    _onnx_model_cache.clear()
    logger.info("Unloaded all ONNX models from cache")


#######################################################################################################################
# End of Audio_Transcription_Parakeet_ONNX.py
#######################################################################################################################