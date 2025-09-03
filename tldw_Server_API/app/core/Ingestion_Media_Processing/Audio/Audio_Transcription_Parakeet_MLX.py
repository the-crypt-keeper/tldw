# Audio_Transcription_Parakeet_MLX.py
#########################################
# Parakeet MLX Implementation for Apple Silicon
# Based on: https://github.com/senstella/parakeet-mlx
#
# This module provides optimized Parakeet transcription for Apple Silicon Macs
# using the MLX framework for efficient on-device inference.
#
####################
# Function List
#
# 1. load_parakeet_mlx_model() - Load the MLX Parakeet model
# 2. transcribe_with_parakeet_mlx() - Transcribe using MLX model
# 3. install_parakeet_mlx() - Install the parakeet-mlx package if needed
#
####################

import os
import sys
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Callable
import numpy as np

# Check if we're on macOS
IS_MACOS = sys.platform == 'darwin'

# Global model cache
_mlx_model_cache: Optional[Any] = None

#######################################################################################################################
# Installation and Setup
#

def check_mlx_available() -> bool:
    """Check if MLX is available and we're on macOS."""
    if not IS_MACOS:
        logging.debug("MLX is only available on macOS")
        return False
    
    try:
        import mlx
        import mlx.core as mx
        return True
    except ImportError:
        logging.debug("MLX not installed")
        return False


def check_parakeet_mlx_installed() -> bool:
    """Check if parakeet-mlx is installed."""
    try:
        import parakeet_mlx
        return True
    except ImportError:
        return False


def install_parakeet_mlx() -> bool:
    """
    Install parakeet-mlx package from GitHub.
    
    Returns:
        True if installation successful, False otherwise
    """
    if not IS_MACOS:
        logging.error("parakeet-mlx is only supported on macOS")
        return False
    
    if check_parakeet_mlx_installed():
        logging.info("parakeet-mlx is already installed")
        return True
    
    try:
        logging.info("Installing parakeet-mlx from GitHub...")
        
        # Install using pip
        cmd = [
            sys.executable, "-m", "pip", "install", 
            "git+https://github.com/senstella/parakeet-mlx.git"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logging.info("Successfully installed parakeet-mlx")
            return True
        else:
            logging.error(f"Failed to install parakeet-mlx: {result.stderr}")
            return False
            
    except Exception as e:
        logging.error(f"Error installing parakeet-mlx: {e}")
        return False


#######################################################################################################################
# Model Loading and Management
#

def load_parakeet_mlx_model(force_reload: bool = False):
    """
    Load the Parakeet MLX model.
    
    Args:
        force_reload: Force reload even if model is cached
    
    Returns:
        Loaded model instance or None if loading fails
    """
    global _mlx_model_cache
    
    if not IS_MACOS:
        logging.error("Parakeet MLX is only supported on macOS with Apple Silicon")
        return None
    
    if _mlx_model_cache and not force_reload:
        logging.debug("Using cached Parakeet MLX model")
        return _mlx_model_cache
    
    # Check MLX availability
    if not check_mlx_available():
        logging.error("MLX is not available. Install with: pip install mlx")
        return None
    
    # Check/install parakeet-mlx
    if not check_parakeet_mlx_installed():
        logging.info("parakeet-mlx not found, attempting to install...")
        if not install_parakeet_mlx():
            logging.error("Failed to install parakeet-mlx")
            return None
    
    try:
        import parakeet_mlx
        import mlx.core as mx
        
        logging.info("Loading Parakeet MLX model...")
        
        # Initialize the model
        # The parakeet-mlx library handles model downloading and caching
        # Try to load from Hugging Face model ID
        # Default model from the parakeet-mlx CLI
        model_id = "mlx-community/parakeet-tdt-0.6b-v2"
        
        try:
            # Try to load the model from Hugging Face
            logging.info(f"Loading model from: {model_id}")
            model = parakeet_mlx.from_pretrained(model_id, dtype=mx.bfloat16)
        except FileNotFoundError:
            # Model might need to be downloaded first
            logging.info("Model not found locally, downloading from Hugging Face...")
            try:
                # The model will be downloaded automatically
                model = parakeet_mlx.from_pretrained(model_id, dtype=mx.bfloat16)
            except Exception as e2:
                logging.error(f"Failed to download/load model: {e2}")
                return None
        except Exception as e:
            logging.error(f"Failed to load model {model_id}: {e}")
            return None
        
        _mlx_model_cache = model
        logging.info("Successfully loaded Parakeet MLX model")
        
        return model
        
    except ImportError as e:
        logging.error(f"Failed to import parakeet: {e}")
        logging.info("Try installing manually: pip install git+https://github.com/senstella/parakeet-mlx.git")
        return None
    except Exception as e:
        logging.error(f"Failed to load Parakeet MLX model: {e}")
        return None


#######################################################################################################################
# Transcription Functions
#

def transcribe_with_parakeet_mlx(
    audio_data: Union[np.ndarray, str, Path],
    sample_rate: int = 16000,
    language: Optional[str] = None,
    batch_size: int = 1,
    verbose: bool = False,
    chunk_duration: Optional[float] = None,
    overlap_duration: float = 15.0,
    chunk_callback: Optional[Callable[[int, int], None]] = None
) -> str:
    """
    Transcribe audio using Parakeet MLX model.
    
    Args:
        audio_data: Audio data as numpy array, file path, or Path object
        sample_rate: Sample rate of audio (will be resampled if needed)
        language: Language hint (not used by Parakeet, included for compatibility)
        batch_size: Batch size for processing
        verbose: Enable verbose output
        chunk_duration: Duration in seconds for chunking long audio (None = no chunking)
        overlap_duration: Overlap between chunks in seconds (default 15.0)
        chunk_callback: Callback function for chunk progress (current, total)
    
    Returns:
        Transcribed text string
    """
    if not IS_MACOS:
        return "[Error: Parakeet MLX is only supported on macOS with Apple Silicon]"
    
    # Load model
    model = load_parakeet_mlx_model()
    if model is None:
        return "[Error: Failed to load Parakeet MLX model]"
    
    try:
        import parakeet_mlx as parakeet
        import soundfile as sf
        
        # Handle different input types
        audio_file_path = None
        
        if isinstance(audio_data, (str, Path)):
            # Already a file path
            audio_path = Path(audio_data)
            if not audio_path.exists():
                return f"[Error: Audio file not found: {audio_path}]"
            
            # Check if we need to resample
            audio_np, file_sr = sf.read(str(audio_path))
            
            # Convert to mono if stereo
            if len(audio_np.shape) > 1:
                audio_np = np.mean(audio_np, axis=1)
            
            # Only create new file if resampling is needed
            if file_sr != 16000:
                import librosa
                audio_np = librosa.resample(
                    audio_np, 
                    orig_sr=file_sr, 
                    target_sr=16000
                )
                audio_data = audio_np  # Will be saved to temp file later
            else:
                # Can use the original file directly
                audio_file_path = str(audio_path)
                audio_data = None  # Clear audio_data to avoid processing it later
        
        elif isinstance(audio_data, np.ndarray):
            # Ensure float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Ensure mono
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resample if needed
            if sample_rate != 16000:
                import librosa
                audio_data = librosa.resample(
                    audio_data, 
                    orig_sr=sample_rate, 
                    target_sr=16000
                )
        else:
            return "[Error: Invalid audio data type]"
        
        # Normalize audio to [-1, 1] range (only if we have numpy array)
        if isinstance(audio_data, np.ndarray) and np.abs(audio_data).max() > 1.0:
            audio_data = audio_data / np.abs(audio_data).max()
        
        # Transcribe using parakeet-mlx
        if verbose:
            if isinstance(audio_data, np.ndarray):
                logging.info(f"Transcribing audio of length {len(audio_data)/16000:.2f} seconds")
            else:
                logging.info(f"Transcribing audio file")
        
        # The parakeet-mlx model's transcribe method expects a file path
        # Use existing file path if available, otherwise create temp file
        if audio_file_path:
            # We already have a file path that doesn't need resampling
            # Build kwargs for transcribe method
            transcribe_kwargs = {}
            if chunk_duration is not None:
                transcribe_kwargs['chunk_duration'] = chunk_duration
                transcribe_kwargs['overlap_duration'] = overlap_duration
            if chunk_callback is not None:
                transcribe_kwargs['chunk_callback'] = chunk_callback
            
            result = model.transcribe(audio_file_path, **transcribe_kwargs)
        elif isinstance(audio_data, np.ndarray):
            # Need to save numpy array to temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, audio_data, 16000, format='WAV')
                temp_audio_path = tmp_file.name
            
            # Build kwargs for transcribe method
            transcribe_kwargs = {}
            if chunk_duration is not None:
                transcribe_kwargs['chunk_duration'] = chunk_duration
                transcribe_kwargs['overlap_duration'] = overlap_duration
            if chunk_callback is not None:
                transcribe_kwargs['chunk_callback'] = chunk_callback
            
            # Use the transcribe method with file path and chunking parameters
            result = model.transcribe(temp_audio_path, **transcribe_kwargs)
            
            # Clean up temp file
            try:
                os.remove(temp_audio_path)
            except:
                pass
        else:
            # Shouldn't happen, but handle gracefully
            return "[Error: Invalid audio data format]"
        
        # The transcribe method returns an AlignedResult object
        # Extract the text from it
        if hasattr(result, 'text'):
            transcription = result.text
        else:
            transcription = result
        
        if isinstance(transcription, dict):
            # Handle structured output
            text = transcription.get('text', '')
        else:
            # Direct text output
            text = str(transcription)
        
        if verbose:
            logging.info(f"Transcription complete: {text[:100]}...")
        
        return text
        
    except ImportError as e:
        logging.error(f"Missing required library: {e}")
        return f"[Error: Missing required library: {e}]"
    except Exception as e:
        import traceback
        logging.error(f"Error during Parakeet MLX transcription: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return f"[Error: Transcription failed: {str(e)}]"


def transcribe_streaming_mlx(
    audio_stream,
    chunk_size: int = 16000,  # 1 second chunks at 16kHz
    overlap: float = 0.1,      # 10% overlap
    verbose: bool = False
) -> List[str]:
    """
    Transcribe audio stream using Parakeet MLX.
    
    Args:
        audio_stream: Iterator or generator yielding audio chunks
        chunk_size: Size of chunks to process
        overlap: Overlap ratio between chunks
        verbose: Enable verbose output
    
    Yields:
        Transcribed text for each chunk
    """
    if not IS_MACOS:
        yield "[Error: Parakeet MLX is only supported on macOS]"
        return
    
    model = load_parakeet_mlx_model()
    if model is None:
        yield "[Error: Failed to load model]"
        return
    
    buffer = np.array([], dtype=np.float32)
    overlap_size = int(chunk_size * overlap)
    
    try:
        for audio_chunk in audio_stream:
            # Add chunk to buffer
            buffer = np.concatenate([buffer, audio_chunk])
            
            # Process when we have enough data
            while len(buffer) >= chunk_size:
                # Extract chunk
                chunk = buffer[:chunk_size]
                
                # Transcribe chunk
                text = transcribe_with_parakeet_mlx(
                    chunk, 
                    sample_rate=16000,
                    verbose=verbose
                )
                
                if text and not text.startswith("[Error"):
                    yield text
                
                # Move buffer forward with overlap
                buffer = buffer[chunk_size - overlap_size:]
        
        # Process remaining buffer
        if len(buffer) > 0:
            text = transcribe_with_parakeet_mlx(
                buffer,
                sample_rate=16000,
                verbose=verbose
            )
            if text and not text.startswith("[Error"):
                yield text
                
    except Exception as e:
        logging.error(f"Error in streaming transcription: {e}")
        yield f"[Error: {str(e)}]"


def unload_parakeet_mlx_model():
    """Unload the cached Parakeet MLX model to free memory."""
    global _mlx_model_cache
    
    if _mlx_model_cache is not None:
        try:
            # MLX models can be deleted directly
            del _mlx_model_cache
            _mlx_model_cache = None
            
            # MLX specific cleanup
            try:
                import mlx.core as mx
                mx.metal.clear_cache()
            except:
                pass
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logging.info("Unloaded Parakeet MLX model from memory")
        except Exception as e:
            logging.error(f"Error unloading model: {e}")


#######################################################################################################################
# Utility Functions
#

def get_mlx_device_info() -> Dict[str, Any]:
    """Get information about MLX device and capabilities."""
    info = {
        'available': False,
        'platform': sys.platform,
        'is_apple_silicon': False,
        'metal_available': False
    }
    
    if not IS_MACOS:
        return info
    
    try:
        import mlx.core as mx
        
        info['available'] = True
        info['is_apple_silicon'] = True
        info['metal_available'] = mx.metal.is_available()
        
        # Get device info
        if info['metal_available']:
            info['device'] = 'Metal GPU'
            # Additional Metal info could be added here
        
    except ImportError:
        pass
    except Exception as e:
        logging.debug(f"Error getting MLX device info: {e}")
    
    return info


def benchmark_parakeet_mlx(audio_duration: float = 10.0) -> Dict[str, float]:
    """
    Benchmark Parakeet MLX performance.
    
    Args:
        audio_duration: Duration of test audio in seconds
    
    Returns:
        Dictionary with benchmark results
    """
    results = {
        'status': 'failed',
        'audio_duration': audio_duration,
        'transcription_time': 0,
        'real_time_factor': 0,
        'model_load_time': 0
    }
    
    if not IS_MACOS:
        results['error'] = 'Not on macOS'
        return results
    
    try:
        import time
        
        # Time model loading
        start = time.time()
        model = load_parakeet_mlx_model(force_reload=True)
        results['model_load_time'] = time.time() - start
        
        if model is None:
            results['error'] = 'Failed to load model'
            return results
        
        # Create test audio (silence)
        sample_rate = 16000
        test_audio = np.zeros(int(audio_duration * sample_rate), dtype=np.float32)
        
        # Time transcription
        start = time.time()
        text = transcribe_with_parakeet_mlx(test_audio, sample_rate=sample_rate)
        results['transcription_time'] = time.time() - start
        
        # Calculate real-time factor
        results['real_time_factor'] = audio_duration / results['transcription_time']
        results['status'] = 'success'
        results['transcription_length'] = len(text) if text and not text.startswith("[Error") else 0
        
    except Exception as e:
        results['error'] = str(e)
    
    return results


#######################################################################################################################
# Integration with main Nemo module
#

def integrate_with_nemo_module():
    """
    Update the main Audio_Transcription_Nemo.py to use this MLX implementation.
    This function patches the _load_parakeet_mlx function.
    """
    try:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Transcription_Nemo
        
        # Replace the placeholder MLX loader with our implementation
        def _load_parakeet_mlx_patched():
            """Load MLX variant of Parakeet model using specialized implementation."""
            return load_parakeet_mlx_model()
        
        # Patch the function
        Audio_Transcription_Nemo._load_parakeet_mlx = _load_parakeet_mlx_patched
        
        logging.info("Successfully integrated Parakeet MLX with main Nemo module")
        return True
        
    except Exception as e:
        logging.error(f"Failed to integrate with Nemo module: {e}")
        return False


#######################################################################################################################
# Main entry point for testing
#

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Parakeet MLX Module Information:")
    print("-" * 40)
    
    # Check system
    info = get_mlx_device_info()
    print(f"Platform: {info['platform']}")
    print(f"MLX Available: {info['available']}")
    print(f"Apple Silicon: {info['is_apple_silicon']}")
    print(f"Metal Available: {info.get('metal_available', False)}")
    
    if not IS_MACOS:
        print("\nThis module requires macOS with Apple Silicon")
        sys.exit(1)
    
    # Try to load model
    print("\nLoading Parakeet MLX model...")
    model = load_parakeet_mlx_model()
    
    if model:
        print("✓ Model loaded successfully")
        
        # Run benchmark
        print("\nRunning benchmark...")
        results = benchmark_parakeet_mlx(audio_duration=5.0)
        
        print(f"Status: {results['status']}")
        if results['status'] == 'success':
            print(f"Model Load Time: {results['model_load_time']:.2f}s")
            print(f"Transcription Time: {results['transcription_time']:.2f}s")
            print(f"Real-time Factor: {results['real_time_factor']:.2f}x")
        else:
            print(f"Error: {results.get('error', 'Unknown')}")
    else:
        print("✗ Failed to load model")
        print("Try: pip install git+https://github.com/senstella/parakeet-mlx.git")


#######################################################################################################################
# End of Audio_Transcription_Parakeet_MLX.py
#######################################################################################################################