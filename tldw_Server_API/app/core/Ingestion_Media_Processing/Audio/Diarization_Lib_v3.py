# diarization_service.py
"""
Speaker diarization service for tldw_chatbook.
Implements speaker identification using vector embeddings approach.
"""
#
# Imports
import os
import sys
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Callable, Tuple, TYPE_CHECKING, TypedDict
import json
#
# 3rd-Party Libraries
from loguru import logger
from contextlib import contextmanager
from enum import Enum
#
# Local Imports
#
######################################################################################################################
# Type checking imports (not loaded at runtime)
if TYPE_CHECKING:
    import numpy as np
    import torch


# Check availability of required modules without importing
def _check_torch_available():
    try:
        import torch
        return True
    except ImportError:
        return False


def _check_speechbrain_available():
    try:
        import speechbrain
        return True
    except ImportError:
        return False


def _check_sklearn_available():
    try:
        import sklearn
        return True
    except ImportError:
        return False


def _check_torchaudio_available():
    try:
        import torchaudio
        return True
    except ImportError:
        return False


# Module availability flags
TORCH_AVAILABLE = _check_torch_available()
SPEECHBRAIN_AVAILABLE = _check_speechbrain_available()
SKLEARN_AVAILABLE = _check_sklearn_available()
TORCHAUDIO_AVAILABLE = _check_torchaudio_available()

# Lazy-loaded modules (will be imported only when needed)
_torch = None
_numpy = None
_silero_vad_model = None
_silero_vad_utils = None
_speechbrain_encoder = None
_sklearn_modules = None
_torchaudio = None


# Enums and Constants
class ClusteringMethod(Enum):
    """Clustering methods for speaker identification."""
    SPECTRAL = 'spectral'
    AGGLOMERATIVE = 'agglomerative'


class EmbeddingDevice(Enum):
    """Device options for embedding model."""
    AUTO = 'auto'
    CPU = 'cpu'
    CUDA = 'cuda'


class SegmentDict(TypedDict, total=False):
    """Type definition for segment dictionaries."""
    start: float
    end: float
    waveform: Any  # torch.Tensor
    speaker_id: Optional[int]
    speaker_label: Optional[str]
    is_padded: bool
    original_duration: float
    speech_region: Dict[str, Any]
    # Memory-efficient fields
    start_sample: Optional[int]
    end_sample: Optional[int]
    waveform_ref: Optional[Any]  # Reference to original waveform instead of copy


class DiarizationResult(TypedDict):
    """Type definition for diarization results."""
    segments: List[Dict[str, Any]]
    speakers: List[Dict[str, Any]]
    duration: float
    num_speakers: int
    processing_time: float


# Constants
DEFAULT_VAD_THRESHOLD = 0.5
DEFAULT_SEGMENT_DURATION = 2.0
DEFAULT_SEGMENT_OVERLAP = 0.5
DEFAULT_MIN_SPEAKERS = 1
DEFAULT_MAX_SPEAKERS = 10
DEFAULT_SIMILARITY_THRESHOLD = 0.85
DEFAULT_EMBEDDING_BATCH_SIZE = 32
DEFAULT_EMBEDDING_MODEL = 'speechbrain/spkrec-ecapa-voxceleb'
SPEAKER_LABEL_PREFIX = 'SPEAKER_'
# Memory-efficient mode constants
DEFAULT_MEMORY_EFFICIENT = False
DEFAULT_MAX_MEMORY_MB = 2048  # 2GB default memory limit


def _sanitize_path_component(name: str) -> str:
    """Sanitize a string to be safe for use as a directory/file name.

    Args:
        name: The string to sanitize

    Returns:
        A sanitized string safe for use in file paths
    """
    # Replace path separators and other unsafe characters
    safe_name = name.replace('/', '_').replace('\\', '_').replace(':', '_')
    safe_name = safe_name.replace('..', '_')  # Prevent directory traversal

    # Keep only alphanumeric, underscore, hyphen, and dot
    safe_name = ''.join(c if c.isalnum() or c in ('_', '-', '.') else '_' for c in safe_name)

    # Remove leading/trailing dots and underscores
    safe_name = safe_name.strip('._')

    # Ensure it's not empty
    if not safe_name:
        safe_name = 'model'

    return safe_name


def _lazy_import_torch():
    """Lazy import torch."""
    global _torch
    if _torch is None and TORCH_AVAILABLE:
        try:
            import torch
            _torch = torch
        except ImportError as e:
            logger.warning(f"Failed to import torch: {e}")
            _torch = None
    return _torch


def _lazy_import_numpy():
    """Lazy import numpy."""
    global _numpy
    if _numpy is None:
        try:
            import numpy
            _numpy = numpy
        except ImportError:
            logger.warning("NumPy not available")
            return None
    return _numpy


def _lazy_import_silero_vad():
    """Lazy import Silero VAD model.

    This function loads the Silero VAD model from torch hub.
    The model returns a tuple of (model, utils) where utils is a tuple of functions.

    Returns:
        tuple: (model, utils) or (None, None) if loading fails

    Note:
        The utils tuple order is critical and can break between versions:
        - utils[0]: get_speech_timestamps - Main VAD function
        - utils[1]: save_audio - Save audio to file
        - utils[2]: read_audio - Read audio from file
        - utils[3]: VADIterator - Class for streaming VAD
        - utils[4]: collect_chunks - Collect speech chunks
    """
    global _silero_vad_model, _silero_vad_utils

    # Check if already loaded
    if _silero_vad_model is not None:
        return _silero_vad_model, _silero_vad_utils

    # Check torch availability
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available, cannot load Silero VAD")
        return None, None

    torch = _lazy_import_torch()
    if not torch:
        logger.warning("Failed to import torch for Silero VAD")
        return None, None

    try:
        logger.info("Loading Silero VAD model from torch hub...")

        # Set torch hub directory if not set
        default_hub_dir = str(Path.home() / '.cache' / 'torch' / 'hub')
        hub_dir = Path(os.environ.get('TORCH_HUB', default_hub_dir))
        hub_dir.mkdir(parents=True, exist_ok=True)

        # Load model with explicit parameters
        result = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,  # Use cached version if available
            trust_repo=True,  # Required for loading
            verbose=False  # Reduce output noise
        )

        # Validate the result format
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            logger.error(
                f"Unexpected Silero VAD return format. Expected (model, utils) tuple, "
                f"got {type(result).__name__} with length {len(result) if hasattr(result, '__len__') else 'unknown'}"
            )
            return None, None

        model, utils = result

        # Validate model
        if model is None:
            logger.error("Silero VAD model is None")
            return None, None

        # Validate utils format
        if not isinstance(utils, (tuple, list)) or len(utils) < 5:
            logger.error(
                f"Unexpected Silero VAD utils format. Expected tuple/list with 5+ items, "
                f"got {type(utils).__name__} with {len(utils) if hasattr(utils, '__len__') else 'unknown'} items"
            )
            return None, None

        # Store globally for future use
        _silero_vad_model = model
        _silero_vad_utils = utils

        logger.info("Silero VAD loaded successfully")
        logger.debug(f"Silero VAD utils count: {len(utils)}")

        return model, utils

    except Exception as e:
        logger.error(f"Failed to load Silero VAD: {type(e).__name__}: {e}")
        logger.debug("Full error:", exc_info=True)

        # Reset globals on failure
        _silero_vad_model = None
        _silero_vad_utils = None

        return None, None


def _lazy_import_speechbrain():
    """Lazy import SpeechBrain encoder."""
    global _speechbrain_encoder
    if _speechbrain_encoder is None and SPEECHBRAIN_AVAILABLE:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            _speechbrain_encoder = EncoderClassifier
        except ImportError:
            try:
                # Fallback for older versions
                from speechbrain.pretrained import EncoderClassifier
                _speechbrain_encoder = EncoderClassifier
            except ImportError as e:
                logger.warning(f"Failed to import SpeechBrain EncoderClassifier: {e}")
                _speechbrain_encoder = None
    return _speechbrain_encoder


def _lazy_import_sklearn():
    """Lazy import sklearn modules."""
    global _sklearn_modules
    if _sklearn_modules is None and SKLEARN_AVAILABLE:
        try:
            from sklearn.cluster import SpectralClustering, AgglomerativeClustering
            from sklearn.preprocessing import normalize
            from sklearn.metrics import silhouette_score
            from sklearn.metrics.pairwise import cosine_similarity
            _sklearn_modules = {
                'SpectralClustering': SpectralClustering,
                'AgglomerativeClustering': AgglomerativeClustering,
                'normalize': normalize,
                'silhouette_score': silhouette_score,
                'cosine_similarity': cosine_similarity
            }
        except ImportError as e:
            logger.warning(f"Failed to import sklearn modules: {e}")
            _sklearn_modules = None
    return _sklearn_modules


def _lazy_import_torchaudio():
    """Lazy import torchaudio."""
    global _torchaudio
    if _torchaudio is None and TORCHAUDIO_AVAILABLE:
        try:
            import torchaudio
            _torchaudio = torchaudio
        except ImportError as e:
            logger.warning(f"Failed to import torchaudio: {e}")
            _torchaudio = None
    return _torchaudio


class DiarizationError(Exception):
    """Base exception for diarization errors."""
    pass


class DiarizationService:
    """
    Speaker diarization service using vector embeddings approach.

    Pipeline:
    1. Voice Activity Detection (VAD) to find speech segments
    2. Split speech into fixed-length overlapping segments
    3. Extract speaker embeddings for each segment
    4. Cluster embeddings to identify speakers
    5. Merge consecutive segments from same speaker

    Attributes:
        is_available (bool): Whether all required dependencies are available.
                           Can be accessed directly or via is_diarization_available().
        config (dict): Configuration parameters for diarization.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 config_loader: Optional[Callable[[], Dict[str, Any]]] = None):
        """Initialize the diarization service.

        Args:
            config: Optional configuration override
            config_loader: Optional configuration loader function
        """
        logger.info("Initializing DiarizationService...")

        # Use provided config loader or default
        if config_loader is None:
            config_loader = self._default_config_loader

        # Load configuration
        self.config = config_loader()

        # Override with provided config
        if config:
            self.config.update(config)

        logger.debug(f"Diarization service configuration: {self.config}")

        # Validate configuration
        self._validate_config(self.config)

        # Model storage (lazy loaded)
        self._vad_model = None
        self._vad_utils = None
        self._embedding_model = None
        self._model_lock = threading.RLock()

        # Check availability (without importing)
        # Public attribute - can be accessed directly by callers
        self.is_available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if all required dependencies are available."""
        required = [
            (TORCH_AVAILABLE, "PyTorch"),
            (SPEECHBRAIN_AVAILABLE, "SpeechBrain"),
            (SKLEARN_AVAILABLE, "scikit-learn"),
        ]

        missing = [name for available, name in required if not available]
        if missing:
            logger.warning(f"Diarization unavailable. Missing: {', '.join(missing)}")
            return False

        return True

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            # VAD settings
            'vad_threshold': DEFAULT_VAD_THRESHOLD,
            'vad_min_speech_duration': 0.25,
            'vad_min_silence_duration': 0.25,

            # Segmentation settings
            'segment_duration': DEFAULT_SEGMENT_DURATION,
            'segment_overlap': DEFAULT_SEGMENT_OVERLAP,
            'min_segment_duration': 1.0,
            'max_segment_duration': 3.0,

            # Embedding model
            'embedding_model': DEFAULT_EMBEDDING_MODEL,
            'embedding_device': EmbeddingDevice.AUTO.value,

            # Clustering settings
            'clustering_method': ClusteringMethod.SPECTRAL.value,
            'similarity_threshold': DEFAULT_SIMILARITY_THRESHOLD,
            'min_speakers': DEFAULT_MIN_SPEAKERS,
            'max_speakers': DEFAULT_MAX_SPEAKERS,

            # Post-processing
            'merge_threshold': 0.5,
            'min_speaker_duration': 3.0,

            # Batch processing
            'embedding_batch_size': DEFAULT_EMBEDDING_BATCH_SIZE,

            # Memory-efficient mode
            'memory_efficient': DEFAULT_MEMORY_EFFICIENT,
            'max_memory_mb': DEFAULT_MAX_MEMORY_MB,

            # Overlapping speech detection
            'detect_overlapping_speech': False,
            'overlap_confidence_threshold': 0.7,
        }

    def _default_config_loader(self) -> Dict[str, Any]:
        """Default configuration loader using get_cli_setting."""
        default_config = self._get_default_config()

        # Override with settings from config file
        config = {}
        for key, default_value in default_config.items():
            config[key] = get_cli_setting(f'diarization.{key}', default_value) or default_value

        return config

    def _validate_config(self, config: Dict) -> None:
        """Validate configuration parameters.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValueError: If configuration is invalid
        """
        # VAD settings validation
        if config['vad_threshold'] < 0 or config['vad_threshold'] > 1:
            raise ValueError("vad_threshold must be between 0 and 1")

        if config['vad_min_speech_duration'] < 0:
            raise ValueError("vad_min_speech_duration must be non-negative")

        if config['vad_min_silence_duration'] < 0:
            raise ValueError("vad_min_silence_duration must be non-negative")

        # Segmentation settings validation
        if config['segment_overlap'] >= config['segment_duration']:
            raise ValueError("segment_overlap must be less than segment_duration")

        if config['segment_overlap'] < 0:
            raise ValueError("segment_overlap must be non-negative")

        if config['min_segment_duration'] > config['max_segment_duration']:
            raise ValueError("min_segment_duration must be <= max_segment_duration")

        if config['segment_duration'] > config['max_segment_duration']:
            raise ValueError("segment_duration must be <= max_segment_duration")

        # Clustering settings validation
        if config['min_speakers'] < 1:
            raise ValueError("min_speakers must be at least 1")

        if config['max_speakers'] < config['min_speakers']:
            raise ValueError("max_speakers must be >= min_speakers")

        if config['similarity_threshold'] < 0 or config['similarity_threshold'] > 1:
            raise ValueError("similarity_threshold must be between 0 and 1")

        # Post-processing validation
        if config['merge_threshold'] < 0:
            raise ValueError("merge_threshold must be non-negative")

        if config['min_speaker_duration'] < 0:
            raise ValueError("min_speaker_duration must be non-negative")

        # Batch processing validation
        if config['embedding_batch_size'] < 1:
            raise ValueError("embedding_batch_size must be at least 1")

        # Embedding device validation
        valid_devices = [e.value for e in EmbeddingDevice]
        if config['embedding_device'] not in valid_devices:
            raise ValueError(f"embedding_device must be one of {valid_devices}")

        # Clustering method validation
        valid_methods = [m.value for m in ClusteringMethod]
        if config['clustering_method'] not in valid_methods:
            raise ValueError(f"clustering_method must be one of {valid_methods}")

    def _get_device(self) -> str:
        """Determine the device to use for inference."""
        if self.config['embedding_device'] == EmbeddingDevice.AUTO.value:
            torch = _lazy_import_torch()
            if torch:
                try:
                    if hasattr(torch, 'cuda') and torch.cuda.is_available():
                        return EmbeddingDevice.CUDA.value
                except (AttributeError, RuntimeError) as e:
                    logger.debug(f"Error checking CUDA availability: {e}")
            return EmbeddingDevice.CPU.value
        return self.config['embedding_device']

    def _load_embedding_model(self):
        """Load the speaker embedding model (lazy loading)."""
        with self._model_lock:
            if self._embedding_model is None:
                logger.info(f"Loading embedding model: {self.config['embedding_model']}")
                try:
                    EncoderClassifier = _lazy_import_speechbrain()
                    if not EncoderClassifier:
                        raise DiarizationError("SpeechBrain EncoderClassifier not available")

                    device = self._get_device()
                    # Sanitize model name for safe directory creation
                    model_name = self.config['embedding_model']
                    safe_model_name = _sanitize_path_component(model_name)

                    # Use pathlib for path construction
                    model_dir = Path('pretrained_models') / safe_model_name

                    self._embedding_model = EncoderClassifier.from_hparams(
                        source=self.config['embedding_model'],
                        savedir=str(model_dir),
                        run_opts={"device": device}
                    )
                    logger.info(f"Embedding model loaded successfully on {device}")
                except Exception as e:
                    logger.error(f"Failed to load embedding model: {e}")
                    raise DiarizationError(f"Failed to load embedding model: {e}") from e

    def _load_vad_model(self):
        """Load the VAD model (lazy loading).

        This method loads the Silero VAD model and its utility functions.
        The VAD utilities are particularly brittle as they return as a tuple
        in a specific order that can change between versions.
        """
        with self._model_lock:  # Add thread safety
            if self._vad_model is None:
                try:
                    model, utils = _lazy_import_silero_vad()
                    if not model or not utils:
                        raise DiarizationError("Silero VAD model or utilities not available")

                    # Basic validation - detailed validation already done in _lazy_import_silero_vad
                    if not utils:
                        raise DiarizationError("Silero VAD utilities not available")

                    # Store model
                    self._vad_model = model

                    # Map utilities with extensive validation
                    # NOTE: This mapping is fragile and depends on Silero's return order
                    try:
                        self._vad_utils = {
                            'get_speech_timestamps': utils[0],  # Main VAD function
                            'save_audio': utils[1],  # Audio saving utility
                            'read_audio': utils[2],  # Audio loading utility
                            'VADIterator': utils[3],  # Streaming VAD class
                            'collect_chunks': utils[4]  # Chunk collection utility
                        }

                        # Validate that each utility is callable (except VADIterator which is a class)
                        for name, func in self._vad_utils.items():
                            if name != 'VADIterator' and not callable(func):
                                raise DiarizationError(f"VAD utility '{name}' is not callable")

                        logger.debug("VAD utilities loaded and validated successfully")

                    except (IndexError, TypeError) as e:
                        raise DiarizationError(
                            f"Failed to map Silero VAD utilities. The utility order may have changed. Error: {e}"
                        ) from e

                except Exception as e:
                    logger.error(f"Failed to load VAD model: {e}")
                    self._vad_model = None
                    self._vad_utils = None
                    raise DiarizationError(f"Failed to load Silero VAD model: {e}") from e

    def _get_vad_utility(self, name: str) -> Callable:
        """Safely get a VAD utility function with validation.

        Args:
            name: Name of the utility ('get_speech_timestamps', 'read_audio', etc.)

        Returns:
            The utility function

        Raises:
            DiarizationError: If utility is not available or not callable
        """
        if not self._vad_utils:
            raise DiarizationError("VAD utilities not loaded")

        if name not in self._vad_utils:
            raise DiarizationError(
                f"VAD utility '{name}' not found. Available: {list(self._vad_utils.keys())}"
            )

        utility = self._vad_utils[name]

        # Special case for VADIterator which is a class, not a function
        if name == 'VADIterator':
            return utility

        if not callable(utility):
            raise DiarizationError(
                f"VAD utility '{name}' is not callable. Got type: {type(utility).__name__}"
            )

        return utility

    def diarize(
            self,
            audio_path: str,
            transcription_segments: Optional[List[Dict]] = None,
            num_speakers: Optional[int] = None,
            progress_callback: Optional[Callable[[float, str, Optional[Dict]], None]] = None
    ) -> Dict[str, Any]:
        """
        Perform speaker diarization on audio file.

        Args:
            audio_path: Path to audio file. For best performance, provide a
                        16kHz mono WAV file, though the service will attempt to
                        convert other common audio formats
            transcription_segments: Optional transcription segments to align with
            num_speakers: Optional number of speakers (if known)
            progress_callback: Optional callback for progress updates

        Returns:
            Dict with diarization results including segments with speaker IDs
        """
        if not self.is_available:
            raise DiarizationError("Diarization service is not available due to missing dependencies")

        start_time = time.time()
        logger.info(f"Starting diarization for: {audio_path}")

        try:
            # Load audio
            if progress_callback:
                progress_callback(0, "Loading audio file...", None)

            waveform = self._load_audio(audio_path)
            sample_rate = 16000  # Assuming 16kHz as standard

            # Step 1: Voice Activity Detection
            if progress_callback:
                progress_callback(10, "Detecting speech segments...", None)

            # Use streaming VAD if memory-efficient mode is enabled
            streaming_vad = self.config.get('memory_efficient', False)
            speech_timestamps = self._detect_speech(waveform, sample_rate, streaming=streaming_vad)
            logger.info(f"Found {len(speech_timestamps)} speech segments")

            if not speech_timestamps:
                logger.warning("No speech detected in audio")
                return {
                    'segments': [],
                    'speakers': [],
                    'duration': len(waveform) / sample_rate,
                    'num_speakers': 0
                }

            # Step 2: Create overlapping segments
            if progress_callback:
                progress_callback(20, "Creating analysis segments...", None)

            segments = self._create_segments(waveform, speech_timestamps, sample_rate)
            logger.info(f"Created {len(segments)} analysis segments")

            # Step 3: Extract embeddings
            if progress_callback:
                progress_callback(30, "Extracting speaker embeddings...", None)

            embeddings = self._extract_embeddings(segments, progress_callback)
            logger.info(f"Extracted {len(embeddings)} embeddings")

            # Step 4: Cluster speakers
            if progress_callback:
                progress_callback(70, "Clustering speakers...", None)

            speaker_labels = self._cluster_speakers(
                embeddings,
                num_speakers=num_speakers
            )

            # Count unique speakers
            unique_speakers = len(set(speaker_labels))
            logger.info(f"Identified {unique_speakers} speakers")

            # Step 5: Assign speakers to segments
            for segment, label in zip(segments, speaker_labels):
                segment['speaker_id'] = int(label)
                segment['speaker_label'] = f"{SPEAKER_LABEL_PREFIX}{label}"

            # Step 5b: Detect overlapping speech (if configured)
            if self.config.get('detect_overlapping_speech', False):
                if progress_callback:
                    progress_callback(75, "Detecting overlapping speech...", None)

                segments = self._detect_overlapping_speech(segments, embeddings, speaker_labels)

            # Step 6: Merge consecutive segments
            if progress_callback:
                progress_callback(85, "Merging segments...", None)

            merged_segments = self._merge_segments(segments)

            # Step 7: Align with transcription if provided
            if transcription_segments:
                if progress_callback:
                    progress_callback(90, "Aligning with transcription...", None)

                aligned_segments = self._align_with_transcription(
                    merged_segments,
                    transcription_segments
                )
            else:
                aligned_segments = merged_segments

            # Calculate speaker statistics
            speaker_stats = self._calculate_speaker_stats(aligned_segments)

            duration = time.time() - start_time
            logger.info(f"Diarization completed in {duration:.2f} seconds")

            if progress_callback:
                progress_callback(100, "Diarization complete", {
                    'num_speakers': unique_speakers,
                    'duration': duration
                })

            result: DiarizationResult = {
                'segments': aligned_segments,
                'speakers': speaker_stats,
                'duration': len(waveform) / sample_rate,
                'num_speakers': unique_speakers,
                'processing_time': duration
            }
            return result

        except Exception as e:
            logger.error(f"Diarization failed: {e}", exc_info=True)
            raise DiarizationError(f"Diarization failed: {str(e)}") from e

    def _load_audio(self, audio_path: str):
        """Load audio file and convert to correct format."""
        torchaudio = _lazy_import_torchaudio()
        torch = _lazy_import_torch()

        if torchaudio and torch:
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    waveform = resampler(waveform)
                return waveform.squeeze()
            except Exception as e:
                logger.warning(f"Failed to load audio with torchaudio: {e}")
                # Fall through to Silero VAD fallback
        else:
            # Fallback to read_audio from Silero VAD utilities
            logger.info("Falling back to Silero VAD read_audio function")

            # Ensure VAD utilities are loaded
            if not self._vad_utils:
                try:
                    self._load_vad_model()
                except Exception as e:
                    logger.error(f"Failed to load VAD model for audio reading: {e}")
                    raise DiarizationError(f"Cannot load audio: VAD model load failed: {e}") from e

            # Validate read_audio function exists and is callable
            if not self._vad_utils or 'read_audio' not in self._vad_utils:
                raise DiarizationError(
                    "VAD utilities missing 'read_audio' function. "
                    "Neither torchaudio nor Silero VAD audio loading available."
                )

            # Get read_audio function using safe getter
            read_audio = self._get_vad_utility('read_audio')

            try:
                # Call read_audio with proper parameters
                # NOTE: Silero's read_audio expects 'sampling_rate' not 'sample_rate'
                waveform = read_audio(audio_path, sampling_rate=16000)

                # Validate the loaded waveform
                if waveform is None:
                    raise DiarizationError("read_audio returned None")

                return waveform

            except Exception as e:
                logger.error(f"Failed to load audio with Silero read_audio: {e}")
                raise DiarizationError(
                    f"Failed to load audio file '{audio_path}': {str(e)}"
                ) from e

    def _detect_speech(self, waveform, sample_rate: int, streaming: bool = False) -> List[Dict]:
        """Detect speech segments using VAD.

        Args:
            waveform: Audio waveform tensor
            sample_rate: Sample rate of the audio
            streaming: If True, use streaming VAD for lower memory usage

        Returns:
            List of speech segments with start/end times

        Raises:
            DiarizationError: If VAD fails or utilities are not properly loaded
        """
        # Ensure VAD model is loaded
        if not self._vad_model:
            self._load_vad_model()

        # Validate VAD utilities are loaded
        if not self._vad_utils or 'get_speech_timestamps' not in self._vad_utils:
            raise DiarizationError(
                "VAD utilities not properly loaded. Missing 'get_speech_timestamps' function."
            )

        if streaming and 'VADIterator' in self._vad_utils:
            # Use streaming VAD for lower memory usage
            try:
                VADIterator = self._get_vad_utility('VADIterator')
                vad_iterator = VADIterator(
                    model=self._vad_model,
                    threshold=self.config['vad_threshold'],
                    sampling_rate=sample_rate,
                    min_silence_duration_ms=int(self.config['vad_min_silence_duration'] * 1000),
                    speech_pad_ms=int(self.config.get('vad_speech_pad_ms', 30))
                )

                # Process in chunks for streaming
                chunk_size = int(sample_rate * 10)  # 10 second chunks
                speech_timestamps = []

                for i in range(0, len(waveform), chunk_size):
                    chunk = waveform[i:i + chunk_size]
                    speech_dict = vad_iterator(chunk, return_seconds=False)

                    if speech_dict:
                        # Adjust timestamps for chunk offset
                        for ts in speech_dict.get('speech_timestamps', []):
                            ts['start'] = ts['start'] + i
                            ts['end'] = ts['end'] + i
                            speech_timestamps.append(ts)

                # Reset iterator
                vad_iterator.reset_states()

            except Exception as e:
                logger.warning(f"Streaming VAD failed, falling back to standard VAD: {e}")
                streaming = False

        if not streaming:
            # Standard (non-streaming) VAD
            # Get the speech detection function using safe getter
            get_speech_timestamps = self._get_vad_utility('get_speech_timestamps')

            try:
                # Call the VAD function with proper parameters
                # NOTE: Parameter names and order are critical for Silero VAD
                speech_timestamps = get_speech_timestamps(
                    waveform,
                    self._vad_model,
                    sampling_rate=sample_rate,  # Must be 'sampling_rate', not 'sample_rate'
                    threshold=self.config['vad_threshold'],
                    min_speech_duration_ms=int(self.config['vad_min_speech_duration'] * 1000),
                    min_silence_duration_ms=int(self.config['vad_min_silence_duration'] * 1000)
                )

                # Validate the output format
                if not isinstance(speech_timestamps, list):
                    raise DiarizationError(
                        f"Expected list of timestamps, got {type(speech_timestamps).__name__}"
                    )

                # Validate each timestamp has required fields
                for i, ts in enumerate(speech_timestamps):
                    if not isinstance(ts, dict):
                        raise DiarizationError(
                            f"Timestamp {i} is not a dict: {type(ts).__name__}"
                        )
                    if 'start' not in ts or 'end' not in ts:
                        raise DiarizationError(
                            f"Timestamp {i} missing 'start' or 'end' field: {ts.keys()}"
                        )

            except Exception as e:
                logger.error(f"VAD detection failed: {e}")
                raise DiarizationError(f"Speech detection failed: {str(e)}") from e

        # Convert to seconds
        for ts in speech_timestamps:
            ts['start'] = ts['start'] / sample_rate
            ts['end'] = ts['end'] / sample_rate

        return speech_timestamps

    def _create_segments(
            self,
            waveform: "torch.Tensor",
            speech_timestamps: List[Dict],
            sample_rate: int
    ) -> List[SegmentDict]:
        """Create fixed-length overlapping segments from speech regions.

        In memory-efficient mode, stores segment indices instead of waveform copies.
        """
        torch = _lazy_import_torch()
        if not torch:
            raise DiarizationError("PyTorch not available for segment creation")

        segments = []
        segment_samples = int(self.config['segment_duration'] * sample_rate)
        min_segment_samples = int(self.config.get('min_segment_duration', 1.0) * sample_rate)
        overlap_samples = int(self.config['segment_overlap'] * sample_rate)
        step_samples = segment_samples - overlap_samples

        # Check if memory-efficient mode is enabled
        memory_efficient = self.config.get('memory_efficient', False)

        for speech in speech_timestamps:
            start_sample = int(speech['start'] * sample_rate)
            end_sample = int(speech['end'] * sample_rate)
            speech_duration = end_sample - start_sample

            if speech_duration < min_segment_samples:
                # Handle short segments by padding
                if memory_efficient:
                    # Store indices instead of waveform copy
                    segments.append({
                        'start': start_sample / sample_rate,
                        'end': end_sample / sample_rate,
                        'start_sample': start_sample,
                        'end_sample': end_sample,
                        'waveform_ref': waveform,  # Reference to original
                        'is_padded': True,
                        'padding_needed': min_segment_samples - speech_duration,
                        'original_duration': speech_duration / sample_rate,
                        'speech_region': speech
                    })
                else:
                    # Original behavior - copy waveform
                    segment_waveform = waveform[start_sample:end_sample]
                    # Pad to minimum length with silence
                    padding_needed = min_segment_samples - speech_duration
                    try:
                        padded_waveform = torch.nn.functional.pad(segment_waveform, (0, padding_needed))
                    except Exception as e:
                        logger.warning(f"Failed to pad short segment: {e}")
                        continue  # Skip this segment if padding fails

                    segments.append({
                        'start': start_sample / sample_rate,
                        'end': end_sample / sample_rate,
                        'waveform': padded_waveform,
                        'is_padded': True,
                        'original_duration': speech_duration / sample_rate,
                        'speech_region': speech  # Keep reference to original speech region
                    })
            else:
                # Create overlapping segments within this speech region
                for i in range(start_sample, end_sample - segment_samples + 1, step_samples):
                    if memory_efficient:
                        # Store indices instead of waveform copy
                        segments.append({
                            'start': i / sample_rate,
                            'end': (i + segment_samples) / sample_rate,
                            'start_sample': i,
                            'end_sample': i + segment_samples,
                            'waveform_ref': waveform,  # Reference to original
                            'is_padded': False,
                            'speech_region': speech
                        })
                    else:
                        # Original behavior - copy waveform
                        segment_waveform = waveform[i:i + segment_samples]

                        segments.append({
                            'start': i / sample_rate,
                            'end': (i + segment_samples) / sample_rate,
                            'waveform': segment_waveform,
                            'is_padded': False,
                            'speech_region': speech  # Keep reference to original speech region
                        })

                # Handle the last segment if it's shorter than segment_duration but longer than min_segment_duration
                last_segment_start = start_sample + (
                            (end_sample - start_sample - segment_samples) // step_samples) * step_samples + step_samples
                if last_segment_start < end_sample:
                    remaining_samples = end_sample - last_segment_start
                    if remaining_samples >= min_segment_samples:
                        if memory_efficient:
                            # Store indices for last segment
                            segments.append({
                                'start': last_segment_start / sample_rate,
                                'end': end_sample / sample_rate,
                                'start_sample': last_segment_start,
                                'end_sample': end_sample,
                                'waveform_ref': waveform,  # Reference to original
                                'is_padded': True,
                                'padding_needed': segment_samples - remaining_samples,
                                'original_duration': remaining_samples / sample_rate,
                                'speech_region': speech
                            })
                        else:
                            # Original behavior - copy waveform
                            segment_waveform = waveform[last_segment_start:end_sample]
                            # Pad to segment_duration
                            padding_needed = segment_samples - remaining_samples
                            try:
                                padded_waveform = torch.nn.functional.pad(segment_waveform, (0, padding_needed))
                                segments.append({
                                    'start': last_segment_start / sample_rate,
                                    'end': end_sample / sample_rate,
                                    'waveform': padded_waveform,
                                    'is_padded': True,
                                    'original_duration': remaining_samples / sample_rate,
                                    'speech_region': speech
                                })
                            except Exception as e:
                                logger.warning(f"Failed to pad last segment: {e}")

        return segments

    def _extract_embeddings(
            self,
            segments: List[SegmentDict],
            progress_callback: Optional[Callable[[float, str, Optional[Dict]], None]] = None
    ) -> "np.ndarray":
        """Extract speaker embeddings for each segment using batch processing.

        Supports memory-efficient mode by loading waveforms on-demand.
        """
        # Load embedding model if not already loaded
        self._load_embedding_model()

        embeddings = []
        total_segments = len(segments)
        batch_size = self.config.get('embedding_batch_size', 32)
        memory_efficient = self.config.get('memory_efficient', False)

        torch = _lazy_import_torch()
        if not torch:
            raise DiarizationError("PyTorch not available for embedding extraction")

        # Process segments in batches
        for batch_idx in range(0, len(segments), batch_size):
            batch_segments = segments[batch_idx:batch_idx + batch_size]

            # Stack waveforms for batch processing
            try:
                if memory_efficient:
                    # Load waveforms on-demand for memory-efficient mode
                    batch_waveforms = []
                    for seg in batch_segments:
                        if 'waveform_ref' in seg and 'start_sample' in seg:
                            # Extract waveform from reference
                            start = seg['start_sample']
                            end = seg['end_sample']
                            waveform = seg['waveform_ref'][start:end]

                            # Apply padding if needed
                            if seg.get('is_padded', False) and 'padding_needed' in seg:
                                waveform = torch.nn.functional.pad(waveform, (0, seg['padding_needed']))

                            batch_waveforms.append(waveform.unsqueeze(0))
                        else:
                            # Fallback to stored waveform
                            batch_waveforms.append(seg['waveform'].unsqueeze(0))

                    waveforms = torch.stack(batch_waveforms)
                else:
                    # Original behavior - use stored waveforms
                    waveforms = torch.stack([seg['waveform'].unsqueeze(0) for seg in batch_segments])
            except Exception as e:
                logger.error(f"Failed to stack waveforms for batch {batch_idx}: {e}")
                raise DiarizationError(f"Failed to prepare batch: {e}") from e

            # Extract embeddings for the batch
            try:
                if hasattr(torch, 'no_grad'):
                    try:
                        with torch.no_grad():
                            batch_embeddings = self._embedding_model.encode_batch(waveforms)
                    except (AttributeError, RuntimeError) as e:
                        logger.debug(f"Failed to use torch.no_grad(): {e}")
                        # Fallback without no_grad context
                        batch_embeddings = self._embedding_model.encode_batch(waveforms)
                else:
                    # No no_grad available, run without context manager
                    batch_embeddings = self._embedding_model.encode_batch(waveforms)

                # Convert to numpy
                batch_embeddings = batch_embeddings.cpu().numpy()

                # Add each embedding from the batch
                for embedding in batch_embeddings:
                    embeddings.append(embedding.squeeze())

            except Exception as e:
                logger.error(f"Failed to extract embeddings for batch starting at {batch_idx}: {e}")
                raise DiarizationError(f"Batch embedding extraction failed: {e}") from e

            # Progress update
            if progress_callback:
                processed = min(batch_idx + len(batch_segments), total_segments)
                progress = 30 + (40 * processed / total_segments)  # 30-70% range
                progress_callback(
                    progress,
                    f"Processing batch {batch_idx // batch_size + 1}/{(total_segments + batch_size - 1) // batch_size}",
                    {'current': processed, 'total': total_segments}
                )

        np = _lazy_import_numpy()
        if not np:
            raise DiarizationError("NumPy not available for creating embedding array")

        try:
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Failed to create numpy array from embeddings: {e}")
            raise DiarizationError(f"Failed to create embedding array: {e}") from e

    def _cluster_speakers(
            self,
            embeddings: "np.ndarray",
            num_speakers: Optional[int] = None
    ) -> "np.ndarray":
        """Cluster embeddings to identify speakers."""
        np = _lazy_import_numpy()
        if not np:
            raise DiarizationError("NumPy not available for clustering")

        # Handle single speaker case
        if num_speakers == 1:
            return np.zeros(len(embeddings), dtype=int)

        sklearn_modules = _lazy_import_sklearn()
        if not sklearn_modules:
            raise DiarizationError("scikit-learn modules not available for clustering")

        # Normalize embeddings
        normalize = sklearn_modules['normalize']
        embeddings = normalize(embeddings, axis=1, norm='l2')

        # Add single-speaker detection before clustering
        if num_speakers is None:
            if self._is_single_speaker(embeddings):
                return np.zeros(len(embeddings), dtype=int)
            num_speakers = self._estimate_num_speakers(embeddings)
            logger.info(f"Estimated {num_speakers} speakers")

        # Ensure num_speakers is within bounds
        num_speakers = max(self.config['min_speakers'],
                           min(num_speakers, self.config['max_speakers']))

        if self.config['clustering_method'] == ClusteringMethod.SPECTRAL.value:
            SpectralClustering = sklearn_modules['SpectralClustering']
            clustering = SpectralClustering(
                n_clusters=num_speakers,
                affinity='cosine',
                assign_labels='kmeans',
                random_state=42
            )
        else:  # agglomerative
            AgglomerativeClustering = sklearn_modules['AgglomerativeClustering']
            clustering = AgglomerativeClustering(
                n_clusters=num_speakers,
                affinity='cosine',
                linkage='average'
            )

        labels = clustering.fit_predict(embeddings)
        return labels

    def _estimate_num_speakers(self, embeddings: "np.ndarray") -> int:
        """Estimate the number of speakers using silhouette analysis."""
        sklearn_modules = _lazy_import_sklearn()
        if not sklearn_modules:
            # Default to 2 speakers if sklearn not available
            return 2

        max_score = -1
        best_n = 2

        SpectralClustering = sklearn_modules['SpectralClustering']
        silhouette_score = sklearn_modules['silhouette_score']

        # Try different numbers of speakers
        for n in range(2, min(len(embeddings), self.config['max_speakers'] + 1)):
            try:
                clustering = SpectralClustering(
                    n_clusters=n,
                    affinity='cosine',
                    assign_labels='kmeans',
                    random_state=42
                )
                labels = clustering.fit_predict(embeddings)

                # Calculate silhouette score
                score = silhouette_score(embeddings, labels, metric='cosine')

                if score > max_score:
                    max_score = score
                    best_n = n

            except Exception as e:
                logger.warning(f"Failed to test {n} speakers: {e}")

        return best_n

    def _is_single_speaker(self, embeddings: "np.ndarray", threshold: Optional[float] = None) -> bool:
        """Check if all embeddings likely belong to a single speaker.

        Args:
            embeddings: Normalized speaker embeddings
            threshold: Similarity threshold (default from config)

        Returns:
            True if likely single speaker, False otherwise
        """
        if threshold is None:
            threshold = self.config.get('similarity_threshold', 0.85)

        np = _lazy_import_numpy()
        if not np:
            # Can't check without numpy, assume multiple speakers
            return False

        sklearn_modules = _lazy_import_sklearn()
        if not sklearn_modules or 'normalize' not in sklearn_modules:
            # Can't normalize without sklearn, assume multiple speakers
            return False

        try:
            # Ensure embeddings are normalized
            normalize = sklearn_modules['normalize']
            normalized = normalize(embeddings, axis=1, norm='l2')

            # Compute pairwise cosine similarities
            similarities = normalized @ normalized.T

            # Calculate average similarity (excluding diagonal)
            n = len(embeddings)
            if n <= 1:
                return True  # Single embedding is single speaker

            # Sum all similarities minus diagonal, divide by number of pairs
            avg_similarity = (similarities.sum() - n) / (n * (n - 1))

            logger.debug(f"Average cosine similarity: {avg_similarity:.3f}, threshold: {threshold}")

            # If average similarity is very high, likely single speaker
            return avg_similarity > threshold

        except Exception as e:
            logger.warning(f"Failed to check single speaker: {e}")
            # On error, assume multiple speakers for safety
            return False

    def _merge_segments(self, segments: List[Dict]) -> List[Dict]:
        """Merge consecutive segments from the same speaker."""
        if not segments:
            return []

        # Sort by start time
        segments = sorted(segments, key=lambda x: x['start'])

        merged = []
        current = segments[0].copy()

        for segment in segments[1:]:
            # Check if same speaker and close enough in time
            same_speaker = segment['speaker_id'] == current['speaker_id']
            close_enough = segment['start'] - current['end'] <= self.config['merge_threshold']

            if same_speaker and close_enough:
                # Extend current segment
                current['end'] = segment['end']
            else:
                # Save current and start new
                merged.append(current)
                current = segment.copy()

        # Don't forget the last segment
        merged.append(current)

        return merged

    def _align_with_transcription(
            self,
            diarization_segments: List[Dict],
            transcription_segments: List[Dict]
    ) -> List[Dict]:
        """Align diarization results with transcription segments."""
        aligned = []

        for trans_seg in transcription_segments:
            # Find overlapping diarization segments
            overlaps = []

            for diar_seg in diarization_segments:
                # Check for overlap
                overlap_start = max(trans_seg['start'], diar_seg['start'])
                overlap_end = min(trans_seg['end'], diar_seg['end'])

                if overlap_end > overlap_start:
                    overlap_duration = overlap_end - overlap_start
                    overlaps.append((diar_seg['speaker_id'], overlap_duration))

            # Assign speaker based on maximum overlap
            if overlaps:
                # Sort by overlap duration
                overlaps.sort(key=lambda x: x[1], reverse=True)
                speaker_id = overlaps[0][0]

                # Create aligned segment
                aligned_seg = trans_seg.copy()
                aligned_seg['speaker_id'] = speaker_id
                aligned_seg['speaker_label'] = f"{SPEAKER_LABEL_PREFIX}{speaker_id}"
                aligned.append(aligned_seg)
            else:
                # No overlap found, keep original
                aligned.append(trans_seg)

        return aligned

    def _calculate_speaker_stats(self, segments: List[Dict]) -> List[Dict]:
        """Calculate statistics for each speaker."""
        speaker_times = {}

        for segment in segments:
            speaker_id = segment.get('speaker_id', -1)
            duration = segment['end'] - segment['start']

            if speaker_id not in speaker_times:
                speaker_times[speaker_id] = {
                    'total_time': 0,
                    'segment_count': 0,
                    'first_appearance': segment['start'],
                    'last_appearance': segment['end']
                }

            stats = speaker_times[speaker_id]
            stats['total_time'] += duration
            stats['segment_count'] += 1
            stats['last_appearance'] = segment['end']

        # Convert to list format
        speakers = []
        for speaker_id, stats in speaker_times.items():
            speakers.append({
                'speaker_id': speaker_id,
                'speaker_label': f"{SPEAKER_LABEL_PREFIX}{speaker_id}",
                'total_time': stats['total_time'],
                'segment_count': stats['segment_count'],
                'first_appearance': stats['first_appearance'],
                'last_appearance': stats['last_appearance']
            })

        # Sort by total time (most talkative first)
        speakers.sort(key=lambda x: x['total_time'], reverse=True)

        return speakers

    def _detect_overlapping_speech(
            self,
            segments: List[Dict],
            embeddings: "np.ndarray",
            primary_labels: "np.ndarray"
    ) -> List[Dict]:
        """Detect potential overlapping speech in segments.

        Args:
            segments: List of segment dictionaries
            embeddings: Speaker embeddings for each segment
            primary_labels: Primary speaker labels from clustering

        Returns:
            Updated segments with overlap information
        """
        # Get threshold from config
        confidence_threshold = self.config.get('overlap_confidence_threshold', 0.7)

        # Import required modules
        np = _lazy_import_numpy()
        if not np:
            logger.warning("NumPy not available for overlap detection")
            return segments

        sklearn_modules = _lazy_import_sklearn()
        if not sklearn_modules:
            logger.warning("scikit-learn not available for overlap detection")
            return segments

        try:
            # Get cosine_similarity function
            cosine_similarity = sklearn_modules['cosine_similarity']

            # Get cluster centers (mean of embeddings per cluster)
            unique_labels = sorted(set(primary_labels))
            cluster_centers = []
            for label in unique_labels:
                mask = primary_labels == label
                cluster_center = embeddings[mask].mean(axis=0)
                cluster_centers.append(cluster_center)

            cluster_centers = np.array(cluster_centers)

            # Calculate similarity to all clusters for each segment
            similarities = cosine_similarity(embeddings, cluster_centers)

            # Detect overlapping speech
            for i, (segment, sim_scores) in enumerate(zip(segments, similarities)):
                primary_speaker = primary_labels[i]
                primary_confidence = sim_scores[primary_speaker]

                # Check if confidence is low (potential overlap)
                if primary_confidence < confidence_threshold:
                    # Find secondary speaker(s)
                    secondary_speakers = []
                    for speaker_id, confidence in enumerate(sim_scores):
                        if speaker_id != primary_speaker and confidence > 0.3:
                            secondary_speakers.append({
                                'speaker_id': speaker_id,
                                'confidence': float(confidence)
                            })

                    if secondary_speakers:
                        segment['is_overlapping'] = True
                        segment['primary_confidence'] = float(primary_confidence)
                        segment['secondary_speakers'] = sorted(
                            secondary_speakers,
                            key=lambda x: x['confidence'],
                            reverse=True
                        )[:2]  # Keep top 2 secondary speakers
                        logger.debug(
                            f"Potential overlap detected at {segment['start']:.2f}s: "
                            f"Primary speaker {primary_speaker} ({primary_confidence:.2f}), "
                            f"Secondary: {secondary_speakers[0]}"
                        )
                else:
                    segment['is_overlapping'] = False
                    segment['primary_confidence'] = float(primary_confidence)

            # Log overlap statistics
            overlapping_count = sum(1 for s in segments if s.get('is_overlapping', False))
            if overlapping_count > 0:
                logger.info(
                    f"Detected {overlapping_count} segments ({overlapping_count / len(segments) * 100:.1f}%) "
                    f"with potential overlapping speech"
                )

        except Exception as e:
            logger.warning(f"Failed to detect overlapping speech: {e}")
            # Don't fail the whole process, just skip overlap detection

        return segments

    def is_diarization_available(self) -> bool:
        """Check if diarization is available.

        Returns:
            bool: True if all required dependencies are available

        Note:
            You can also directly access the `is_available` attribute
            for the same information.
        """
        return self.is_available

    def get_requirements(self) -> Dict[str, bool]:
        """Get the status of required dependencies."""
        return {
            'torch': TORCH_AVAILABLE,
            'speechbrain': SPEECHBRAIN_AVAILABLE,
            'sklearn': SKLEARN_AVAILABLE,
            'torchaudio': TORCHAUDIO_AVAILABLE
        }

#
# End of Diarization_Lib_v3.py
######################################################################################################################
