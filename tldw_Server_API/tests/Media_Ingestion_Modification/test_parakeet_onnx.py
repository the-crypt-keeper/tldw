"""
Unit and integration tests for Parakeet ONNX transcription implementation.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import soundfile as sf
import json

pytestmark = pytest.mark.unit


class TestParakeetONNX:
    """Test suite for Parakeet ONNX transcription."""
    
    @pytest.fixture
    def sample_audio_data(self):
        """Generate audio data for testing."""
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = 0.5 * np.sin(440 * 2 * np.pi * t)
        return audio.astype(np.float32), sample_rate
    
    @pytest.fixture
    def mock_onnx_session(self):
        """Create mock ONNX inference session."""
        session = MagicMock()
        
        # Mock inputs
        input1 = MagicMock()
        input1.name = "encoder_outputs"
        input1.shape = [1, None, 512]
        
        session.get_inputs.return_value = [input1]
        
        # Mock outputs
        output1 = MagicMock()
        output1.name = "logits"
        
        session.get_outputs.return_value = [output1]
        
        # Mock run method
        def mock_run(output_names, input_dict):
            batch_size = input_dict["encoder_outputs"].shape[0]
            seq_len = 50  # Mock sequence length
            vocab_size = 128256
            logits = np.random.randn(batch_size, seq_len, vocab_size).astype(np.float32)
            return [logits]
        
        session.run = mock_run
        
        return session
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.vocab = {f"token_{i}": i for i in range(128256)}
        tokenizer.vocab["<pad>"] = 0
        tokenizer.vocab["<s>"] = 1
        tokenizer.vocab["</s>"] = 2
        tokenizer.vocab["<unk>"] = 3
        tokenizer.vocab[" "] = 32
        
        # Add some real words
        words = ["Hello", "world", "this", "is", "a", "test", "transcription"]
        for i, word in enumerate(words):
            tokenizer.vocab[word] = 100 + i
        
        # Reverse vocab for decoding
        tokenizer.id_to_token = {v: k for k, v in tokenizer.vocab.items()}
        
        def decode(token_ids):
            tokens = [tokenizer.id_to_token.get(tid, "<unk>") for tid in token_ids]
            text = " ".join(t for t in tokens if t not in ["<pad>", "<s>", "</s>", "<unk>"])
            return text
        
        tokenizer.decode = decode
        
        return tokenizer
    
    def test_import_module(self):
        """Test that the ONNX module can be imported."""
        try:
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX import (
                transcribe_with_parakeet_onnx,
                load_parakeet_onnx_model,
                ParakeetONNXTokenizer
            )
            assert transcribe_with_parakeet_onnx is not None
            assert load_parakeet_onnx_model is not None
            assert ParakeetONNXTokenizer is not None
        except ImportError as e:
            pytest.skip(f"ONNX module not available: {e}")
    
    @patch('onnxruntime.InferenceSession')
    @patch('huggingface_hub.snapshot_download')
    def test_model_loading(self, mock_download, mock_ort_session, mock_onnx_session):
        """Test ONNX model loading."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX import (
            load_parakeet_onnx_model
        )
        
        # Setup mocks
        mock_download.return_value = "/path/to/model"
        mock_ort_session.return_value = mock_onnx_session
        
        # Create mock vocab file
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_path = Path(tmpdir) / "vocab.txt"
            with open(vocab_path, 'w') as f:
                for i in range(100):
                    f.write(f"token_{i} {i}\n")
            
            with patch('pathlib.Path.exists', return_value=True):
                with patch('builtins.open', create=True) as mock_open:
                    mock_open.return_value.__enter__.return_value.readlines.return_value = [
                        f"token_{i} {i}\n" for i in range(100)
                    ]
                    
                    session, tokenizer = load_parakeet_onnx_model()
                    
                    assert session is not None
                    assert tokenizer is not None
                    mock_ort_session.assert_called_once()
    
    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX.load_parakeet_onnx_model')
    def test_transcribe_simple(self, mock_load_model, sample_audio_data, mock_onnx_session, mock_tokenizer):
        """Test simple transcription without chunking."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX import (
            transcribe_with_parakeet_onnx
        )
        
        audio_data, sample_rate = sample_audio_data
        
        # Setup mocks
        mock_load_model.return_value = (mock_onnx_session, mock_tokenizer)
        
        # Mock preprocessing
        with patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX._preprocess_audio') as mock_preprocess:
            mock_features = np.random.randn(1, 100, 512).astype(np.float32)
            mock_preprocess.return_value = mock_features
            
            # Transcribe
            result = transcribe_with_parakeet_onnx(audio_data, sample_rate)
            
            assert result is not None
            assert isinstance(result, str)
            mock_preprocess.assert_called_once()
            mock_onnx_session.run.assert_called()
    
    def test_preprocessing(self):
        """Test audio preprocessing functions."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX import (
            _preprocess_audio
        )
        
        # Generate test audio
        sample_rate = 16000
        duration = 1.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)
        
        # Preprocess
        features = _preprocess_audio(audio, sample_rate)
        
        assert features is not None
        assert features.ndim == 3  # [batch, time, features]
        assert features.shape[0] == 1  # Batch size
        assert features.shape[2] == 80  # Mel features
    
    def test_tokenizer(self):
        """Test ParakeetONNXTokenizer functionality."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX import (
            ParakeetONNXTokenizer
        )
        
        # Create tokenizer with test vocab
        vocab = {
            "<pad>": 0,
            "<s>": 1,
            "</s>": 2,
            "<unk>": 3,
            "Hello": 100,
            "world": 101,
            " ": 32,
        }
        
        tokenizer = ParakeetONNXTokenizer(vocab)
        
        # Test decoding
        token_ids = [1, 100, 32, 101, 2]  # <s> Hello   world </s>
        text = tokenizer.decode(token_ids)
        
        assert "Hello" in text
        assert "world" in text
        assert "<s>" not in text  # Special tokens should be filtered
        assert "</s>" not in text
    
    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX.load_parakeet_onnx_model')
    def test_transcribe_with_chunking(self, mock_load_model, mock_onnx_session, mock_tokenizer):
        """Test transcription with chunking."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX import (
            transcribe_with_parakeet_onnx
        )
        
        # Generate long audio
        sample_rate = 16000
        duration = 120.0
        audio_data = np.random.randn(int(sample_rate * duration)).astype(np.float32)
        
        # Setup mocks
        mock_load_model.return_value = (mock_onnx_session, mock_tokenizer)
        
        # Track chunk callbacks
        chunk_callbacks = []
        def chunk_callback(current, total):
            chunk_callbacks.append((current, total))
        
        with patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX._preprocess_audio') as mock_preprocess:
            mock_features = np.random.randn(1, 100, 512).astype(np.float32)
            mock_preprocess.return_value = mock_features
            
            # Transcribe with chunking
            result = transcribe_with_parakeet_onnx(
                audio_data,
                sample_rate,
                chunk_duration=30.0,
                overlap_duration=5.0,
                chunk_callback=chunk_callback
            )
            
            assert result is not None
            assert len(chunk_callbacks) > 0  # Callbacks were triggered
            
            # Verify chunks were processed
            expected_chunks = int(np.ceil(duration / 25.0))  # 30s chunks with 5s overlap
            assert len(chunk_callbacks) >= expected_chunks - 1
    
    def test_merge_algorithms(self):
        """Test different merge algorithms for chunked transcription."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX import (
            _merge_chunks_middle,
            _merge_chunks_lcs
        )
        
        # Test middle merge
        chunks = [
            "Hello world this is",
            "this is a test",
            "a test transcription"
        ]
        
        merged_middle = _merge_chunks_middle(chunks)
        assert "Hello world" in merged_middle
        assert "transcription" in merged_middle
        
        # Test LCS merge
        merged_lcs = _merge_chunks_lcs(chunks)
        assert "Hello world" in merged_lcs
        assert "transcription" in merged_lcs
    
    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX.load_parakeet_onnx_model')
    def test_error_handling(self, mock_load_model, sample_audio_data):
        """Test error handling during transcription."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX import (
            transcribe_with_parakeet_onnx
        )
        
        audio_data, sample_rate = sample_audio_data
        
        # Test model loading failure
        mock_load_model.side_effect = Exception("Model loading failed")
        
        result = transcribe_with_parakeet_onnx(audio_data, sample_rate)
        
        assert "[Error:" in result
        assert "Model loading failed" in result
    
    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX.load_parakeet_onnx_model')
    def test_custom_model_path(self, mock_load_model, sample_audio_data, mock_onnx_session, mock_tokenizer):
        """Test using custom model path."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX import (
            transcribe_with_parakeet_onnx
        )
        
        audio_data, sample_rate = sample_audio_data
        custom_path = "/custom/model/path"
        
        mock_load_model.return_value = (mock_onnx_session, mock_tokenizer)
        
        with patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX._preprocess_audio') as mock_preprocess:
            mock_features = np.random.randn(1, 100, 512).astype(np.float32)
            mock_preprocess.return_value = mock_features
            
            result = transcribe_with_parakeet_onnx(
                audio_data,
                sample_rate,
                model_path=custom_path
            )
            
            assert result is not None
            mock_load_model.assert_called_with(custom_path, 'cpu')
    
    def test_device_selection(self):
        """Test device selection for ONNX runtime."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX import (
            _get_ort_providers
        )
        
        # Test CPU providers
        cpu_providers = _get_ort_providers('cpu')
        assert 'CPUExecutionProvider' in cpu_providers
        
        # Test CUDA providers
        cuda_providers = _get_ort_providers('cuda')
        assert 'CUDAExecutionProvider' in cuda_providers or 'CPUExecutionProvider' in cuda_providers


@pytest.mark.integration
class TestParakeetONNXIntegration:
    """Integration tests for Parakeet ONNX."""
    
    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX.transcribe_with_parakeet_onnx')
    def test_integration_with_nemo_module(self, mock_transcribe):
        """Test integration with Nemo module."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
            transcribe_with_parakeet
        )
        
        mock_transcribe.return_value = "ONNX transcription result"
        
        audio_data = np.array([0.1, 0.2, 0.3])
        
        # Need to patch the variant check
        with patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo.loaded_config_data') as mock_config:
            mock_config.return_value = {
                'STT-Settings': {
                    'nemo_model_variant': 'onnx'
                }
            }
            
            result = transcribe_with_parakeet(audio_data, 16000, variant='onnx')
            
            assert result == "ONNX transcription result"
            mock_transcribe.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])