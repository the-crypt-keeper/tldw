# Audio_Transcription_Nemo.py
#########################################
# Nemo Transcription Module
# This module provides transcription using NVIDIA Nemo models:
# - Canary-1b: Multilingual model supporting English, Spanish, German, and French
# - Parakeet TDT: Efficient model with support for standard, ONNX, and MLX variants
#
####################
# Function List
#
# 1. load_canary_model() - Load and cache Canary-1b model
# 2. load_parakeet_model(variant='standard') - Load Parakeet model (standard/ONNX/MLX)
# 3. transcribe_with_canary(audio_data, sample_rate) - Transcribe using Canary
# 4. transcribe_with_parakeet(audio_data, sample_rate, variant='standard') - Transcribe using Parakeet
# 5. transcribe_with_nemo(audio_data, sample_rate, model='parakeet', variant='standard') - Unified entry point
#
####################

import os
import sys
import logging
import tempfile
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any, Callable
import numpy as np
import torch

# Import local config
from tldw_Server_API.app.core.config import load_and_log_configs, loaded_config_data

# Global model cache
_model_cache: Dict[str, Any] = {}

#######################################################################################################################
# Model Loading Functions
#

def _get_model_cache_key(model_name: str, variant: str = 'standard') -> str:
    """Generate a cache key for model storage."""
    return f"{model_name}_{variant}"


def _get_cache_dir() -> Path:
    """Get the cache directory for Nemo models."""
    config = loaded_config_data or load_and_log_configs()
    if config and 'STT-Settings' in config:
        cache_dir = config['STT-Settings'].get('nemo_cache_dir', './models/nemo')
    else:
        cache_dir = './models/nemo'
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def load_canary_model():
    """
    Load and cache the Canary-1b model.
    
    Returns:
        The loaded Canary model instance, or None if loading fails.
    """
    cache_key = _get_model_cache_key('canary', 'standard')
    
    if cache_key in _model_cache:
        logging.debug(f"Using cached Canary model")
        return _model_cache[cache_key]
    
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError as e:
        logging.error("Nemo toolkit not installed. Install with: pip install nemo_toolkit[asr]")
        return None
    
    try:
        logging.info("Loading Canary-1b model from NVIDIA...")
        
        # Set cache directory for Nemo
        cache_dir = _get_cache_dir()
        os.environ['NEMO_CACHE_DIR'] = str(cache_dir)
        
        # Load the model
        model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained("nvidia/canary-1b-v2")
        
        # Configure device
        config = loaded_config_data or load_and_log_configs()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if config and 'STT-Settings' in config:
            device = config['STT-Settings'].get('nemo_device', device)
        
        if device == 'cuda' and torch.cuda.is_available():
            model = model.cuda()
        else:
            model = model.cpu()
        
        model.eval()
        
        _model_cache[cache_key] = model
        logging.info(f"Successfully loaded Canary-1b model on {device}")
        return model
        
    except Exception as e:
        logging.error(f"Failed to load Canary model: {e}")
        return None


def load_parakeet_model(variant: str = 'standard'):
    """
    Load and cache the Parakeet TDT model.
    
    Args:
        variant: Model variant to load ('standard', 'onnx', 'mlx')
    
    Returns:
        The loaded Parakeet model instance, or None if loading fails.
    """
    cache_key = _get_model_cache_key('parakeet', variant)
    
    if cache_key in _model_cache:
        logging.debug(f"Using cached Parakeet model (variant: {variant})")
        return _model_cache[cache_key]
    
    config = loaded_config_data or load_and_log_configs()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if config and 'STT-Settings' in config:
        device = config['STT-Settings'].get('nemo_device', device)
    
    try:
        if variant == 'onnx':
            return _load_parakeet_onnx(device)
        elif variant == 'mlx':
            return _load_parakeet_mlx()
        else:  # standard
            return _load_parakeet_standard(device)
    except Exception as e:
        logging.error(f"Failed to load Parakeet model (variant: {variant}): {e}")
        return None


def _load_parakeet_standard(device: str):
    """Load standard Nemo Parakeet model."""
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        logging.error("Nemo toolkit not installed. Install with: pip install nemo_toolkit[asr]")
        return None
    
    logging.info("Loading Parakeet TDT model from NVIDIA...")
    
    # Set cache directory
    cache_dir = _get_cache_dir()
    os.environ['NEMO_CACHE_DIR'] = str(cache_dir)
    
    # Load the model
    model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
    
    # Configure for efficient inference
    model.change_decoding_strategy(None)  # Use greedy decoding for speed
    
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()
    
    model.eval()
    
    cache_key = _get_model_cache_key('parakeet', 'standard')
    _model_cache[cache_key] = model
    logging.info(f"Successfully loaded Parakeet TDT model on {device}")
    return model


def _load_parakeet_onnx(device: str):
    """Load ONNX variant of Parakeet model using proper implementation."""
    try:
        # Import the proper ONNX implementation
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX import (
            load_parakeet_onnx_model
        )
        
        logging.info("Loading Parakeet TDT ONNX model...")
        
        # Load model with proper tokenizer
        session, tokenizer = load_parakeet_onnx_model(device=device)
        
        if session is None or tokenizer is None:
            logging.error("Failed to load ONNX model or tokenizer")
            return None
        
        # Wrap in a class for consistent interface
        class ONNXParakeetModel:
            def __init__(self, session, tokenizer):
                self.session = session
                self.tokenizer = tokenizer
            
            def transcribe(self, audio_path, chunk_duration=None, overlap_duration=15.0, chunk_callback=None):
                # Use the proper ONNX transcription
                from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX import (
                    transcribe_with_parakeet_onnx
                )
                
                result = transcribe_with_parakeet_onnx(
                    audio_path,
                    device=device,
                    chunk_duration=chunk_duration,
                    overlap_duration=overlap_duration,
                    chunk_callback=chunk_callback
                )
                
                # Return as list for compatibility
                return [result] if result else ["[No transcription produced]"]
        
        model = ONNXParakeetModel(session, tokenizer)
        cache_key = _get_model_cache_key('parakeet', 'onnx')
        _model_cache[cache_key] = model
        logging.info(f"Successfully loaded Parakeet ONNX model with tokenizer")
        return model
        
    except ImportError as e:
        logging.error(f"Failed to import ONNX implementation: {e}")
        logging.error("Ensure Audio_Transcription_Parakeet_ONNX.py is available")
        return None
    except Exception as e:
        logging.error(f"Failed to load ONNX model: {e}")
        return None


def _load_parakeet_mlx():
    """Load MLX variant of Parakeet model for Apple Silicon."""
    # Check if we're on macOS with Apple Silicon
    if sys.platform != 'darwin':
        logging.warning("MLX variant is only supported on macOS. Falling back to standard variant.")
        return _load_parakeet_standard('cpu')
    
    try:
        # Import the specialized MLX implementation
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX import (
            load_parakeet_mlx_model,
            check_mlx_available
        )
        
        if not check_mlx_available():
            logging.warning("MLX not available. Falling back to standard variant.")
            return _load_parakeet_standard('cpu')
        
        logging.info("Loading Parakeet MLX model using specialized implementation...")
        model = load_parakeet_mlx_model()
        
        if model is not None:
            cache_key = _get_model_cache_key('parakeet', 'mlx')
            _model_cache[cache_key] = model
            logging.info("Successfully loaded Parakeet MLX model")
            return model
        else:
            logging.warning("Failed to load MLX model. Falling back to standard variant.")
            return _load_parakeet_standard('cpu')
            
    except ImportError as e:
        logging.warning(f"MLX implementation not available: {e}. Falling back to standard variant.")
        return _load_parakeet_standard('cpu')
    except Exception as e:
        logging.error(f"Error loading Parakeet MLX model: {e}")
        return _load_parakeet_standard('cpu')


#######################################################################################################################
# Transcription Functions
#

def transcribe_with_canary(
    audio_data: Union[np.ndarray, str],
    sample_rate: int = 16000,
    language: Optional[str] = None
) -> str:
    """
    Transcribe audio using the Canary-1b model.
    
    Args:
        audio_data: Either a numpy array of audio samples or path to audio file
        sample_rate: Sample rate of the audio
        language: Target language (en, es, de, fr). If None, auto-detect.
    
    Returns:
        Transcribed text string
    """
    model = load_canary_model()
    if model is None:
        return "[Error: Canary model could not be loaded]"
    
    # Save audio to temporary file if needed
    if isinstance(audio_data, np.ndarray):
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_data, sample_rate)
            audio_path = tmp_file.name
            cleanup_temp = True
    else:
        audio_path = audio_data
        cleanup_temp = False
    
    try:
        # Prepare the transcription prompt
        # Canary uses special tokens for language specification
        if language:
            lang_map = {'en': 'en', 'es': 'es', 'de': 'de', 'fr': 'fr'}
            target_lang = lang_map.get(language, 'en')
        else:
            target_lang = 'en'  # Default to English
        
        # Transcribe with Canary
        # The model expects specific prompt format
        manifest = {
            "audio_filepath": audio_path,
            "duration": 1.0,  # Will be calculated by model
            "taskname": "asr",
            "source_lang": "en",  # Source language for transcription
            "target_lang": target_lang,
            "pnc": "yes",  # Punctuation and capitalization
            "answer": "na"
        }
        
        # Perform transcription
        transcriptions = model.transcribe(
            [audio_path],
            batch_size=1
        )
        
        if transcriptions and len(transcriptions) > 0:
            result = transcriptions[0]
        else:
            result = "[No transcription produced]"
        
        return result
        
    except Exception as e:
        logging.error(f"Error during Canary transcription: {e}")
        return f"[Transcription error: {str(e)}]"
    finally:
        if cleanup_temp and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass


def transcribe_with_parakeet(
    audio_data: Union[np.ndarray, str],
    sample_rate: int = 16000,
    variant: str = 'standard',
    chunk_duration: Optional[float] = None,
    overlap_duration: float = 15.0,
    chunk_callback: Optional[Callable[[int, int], None]] = None
) -> str:
    """
    Transcribe audio using the Parakeet TDT model.
    
    Args:
        audio_data: Either a numpy array of audio samples or path to audio file
        sample_rate: Sample rate of the audio
        variant: Model variant to use ('standard', 'onnx', 'mlx')
        chunk_duration: Duration in seconds for chunking long audio (None = no chunking)
        overlap_duration: Overlap between chunks in seconds (default 15.0)
        chunk_callback: Callback function for chunk progress (current, total)
    
    Returns:
        Transcribed text string
    """
    # Get variant from config if not specified
    if variant == 'auto':
        config = loaded_config_data or load_and_log_configs()
        if config and 'STT-Settings' in config:
            variant = config['STT-Settings'].get('nemo_model_variant', 'standard')
    
    model = load_parakeet_model(variant)
    if model is None:
        return f"[Error: Parakeet model ({variant}) could not be loaded]"
    
    # Save audio to temporary file if needed
    if isinstance(audio_data, np.ndarray):
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_data, sample_rate)
            audio_path = tmp_file.name
            cleanup_temp = True
    else:
        audio_path = audio_data
        cleanup_temp = False
    
    try:
        # Perform transcription based on variant
        if variant == 'mlx':
            # Use specialized MLX transcription
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX import (
                transcribe_with_parakeet_mlx as mlx_transcribe
            )
            result = mlx_transcribe(
                audio_path, 
                sample_rate=sample_rate,
                chunk_duration=chunk_duration,
                overlap_duration=overlap_duration,
                chunk_callback=chunk_callback
            )
            transcriptions = [result] if result else ["[No transcription produced]"]
        elif variant == 'onnx' and hasattr(model, 'transcribe'):
            # ONNX model transcription with chunking support
            transcriptions = model.transcribe(
                audio_path,
                chunk_duration=chunk_duration,
                overlap_duration=overlap_duration,
                chunk_callback=chunk_callback
            )
        else:
            # Standard Nemo model transcription
            transcriptions = model.transcribe([audio_path], batch_size=1)
        
        if transcriptions and len(transcriptions) > 0:
            result = transcriptions[0]
        else:
            result = "[No transcription produced]"
        
        return result
        
    except Exception as e:
        logging.error(f"Error during Parakeet transcription: {e}")
        return f"[Transcription error: {str(e)}]"
    finally:
        if cleanup_temp and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass


def transcribe_with_nemo(
    audio_data: Union[np.ndarray, str],
    sample_rate: int = 16000,
    model: str = 'parakeet',
    variant: str = 'standard',
    language: Optional[str] = None,
    chunk_duration: Optional[float] = None,
    overlap_duration: float = 15.0,
    chunk_callback: Optional[Callable[[int, int], None]] = None
) -> str:
    """
    Unified entry point for Nemo model transcription.
    
    Args:
        audio_data: Either a numpy array of audio samples or path to audio file
        sample_rate: Sample rate of the audio
        model: Which model to use ('parakeet' or 'canary')
        variant: Model variant for Parakeet ('standard', 'onnx', 'mlx')
        language: Target language for Canary (en, es, de, fr)
        chunk_duration: Duration in seconds for chunking long audio (None = no chunking)
        overlap_duration: Overlap between chunks in seconds (default 15.0)
        chunk_callback: Callback function for chunk progress (current, total)
    
    Returns:
        Transcribed text string
    """
    if model.lower() == 'canary':
        # Note: Canary doesn't support chunking in current implementation
        return transcribe_with_canary(audio_data, sample_rate, language)
    elif model.lower() == 'parakeet':
        return transcribe_with_parakeet(
            audio_data, sample_rate, variant,
            chunk_duration=chunk_duration,
            overlap_duration=overlap_duration,
            chunk_callback=chunk_callback
        )
    else:
        return f"[Error: Unknown Nemo model: {model}]"


def unload_nemo_models():
    """Unload all cached Nemo models to free memory."""
    global _model_cache
    
    for key, model in _model_cache.items():
        try:
            # Try to free GPU memory if applicable
            if hasattr(model, 'cpu'):
                model.cpu()
            del model
        except:
            pass
    
    _model_cache.clear()
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logging.info("Unloaded all Nemo models from memory")


#######################################################################################################################
# End of Audio_Transcription_Nemo.py
#######################################################################################################################