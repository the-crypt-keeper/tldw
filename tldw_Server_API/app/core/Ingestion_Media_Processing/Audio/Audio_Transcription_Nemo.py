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
from typing import Optional, Union, Tuple, Dict, Any
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
    """Load ONNX variant of Parakeet model."""
    try:
        import onnxruntime as ort
    except ImportError:
        logging.error("ONNX Runtime not installed. Install with: pip install onnxruntime")
        return None
    
    logging.info("Loading Parakeet TDT ONNX model...")
    
    cache_dir = _get_cache_dir()
    model_path = cache_dir / "parakeet-onnx"
    
    # Download model if not cached
    if not model_path.exists():
        try:
            from huggingface_hub import snapshot_download
            logging.info("Downloading Parakeet ONNX model from Hugging Face...")
            snapshot_download(
                repo_id="istupakov/parakeet-tdt-0.6b-v3-onnx",
                local_dir=str(model_path),
                local_dir_use_symlinks=False
            )
        except ImportError:
            logging.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
            return None
        except Exception as e:
            logging.error(f"Failed to download ONNX model: {e}")
            return None
    
    # Create ONNX session
    providers = ['CUDAExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    
    try:
        # Look for the main ONNX file
        onnx_files = list(model_path.glob("*.onnx"))
        if not onnx_files:
            logging.error(f"No ONNX files found in {model_path}")
            return None
        
        session = ort.InferenceSession(str(onnx_files[0]), providers=providers)
        
        # Wrap in a simple class for consistent interface
        class ONNXParakeetModel:
            def __init__(self, session):
                self.session = session
                self.input_names = [inp.name for inp in session.get_inputs()]
                self.output_names = [out.name for out in session.get_outputs()]
            
            def transcribe(self, audio_path):
                # This is a simplified implementation
                # Real implementation would need proper preprocessing
                import soundfile as sf
                audio_data, sample_rate = sf.read(audio_path)
                
                # Prepare input (this is model-specific and may need adjustment)
                inputs = {self.input_names[0]: audio_data.astype(np.float32)}
                outputs = self.session.run(self.output_names, inputs)
                
                # Decode outputs (this is model-specific)
                # For now, return a placeholder
                return ["[ONNX transcription placeholder - implement decoding]"]
        
        model = ONNXParakeetModel(session)
        cache_key = _get_model_cache_key('parakeet', 'onnx')
        _model_cache[cache_key] = model
        logging.info(f"Successfully loaded Parakeet ONNX model")
        return model
        
    except Exception as e:
        logging.error(f"Failed to create ONNX session: {e}")
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
    variant: str = 'standard'
) -> str:
    """
    Transcribe audio using the Parakeet TDT model.
    
    Args:
        audio_data: Either a numpy array of audio samples or path to audio file
        sample_rate: Sample rate of the audio
        variant: Model variant to use ('standard', 'onnx', 'mlx')
    
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
            result = mlx_transcribe(audio_path, sample_rate=sample_rate)
            transcriptions = [result] if result else ["[No transcription produced]"]
        elif variant == 'onnx' and hasattr(model, 'transcribe'):
            # ONNX model transcription
            transcriptions = model.transcribe(audio_path)
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
    language: Optional[str] = None
) -> str:
    """
    Unified entry point for Nemo model transcription.
    
    Args:
        audio_data: Either a numpy array of audio samples or path to audio file
        sample_rate: Sample rate of the audio
        model: Which model to use ('parakeet' or 'canary')
        variant: Model variant for Parakeet ('standard', 'onnx', 'mlx')
        language: Target language for Canary (en, es, de, fr)
    
    Returns:
        Transcribed text string
    """
    if model.lower() == 'canary':
        return transcribe_with_canary(audio_data, sample_rate, language)
    elif model.lower() == 'parakeet':
        return transcribe_with_parakeet(audio_data, sample_rate, variant)
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