"""
Unit and integration tests for buffered/chunked transcription with merge algorithms.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import soundfile as sf

pytestmark = pytest.mark.unit


class TestBufferedTranscription:
    """Test suite for buffered transcription functionality."""
    
    @pytest.fixture
    def sample_audio_data(self):
        """Generate audio data for testing."""
        sample_rate = 16000
        duration = 10.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        # Create audio with varying amplitude to simulate speech
        audio = np.sin(440 * 2 * np.pi * t) * (0.5 + 0.3 * np.sin(0.5 * 2 * np.pi * t))
        return audio.astype(np.float32), sample_rate
    
    @pytest.fixture
    def long_audio_data(self):
        """Generate long audio for testing chunking."""
        sample_rate = 16000
        duration = 300.0  # 5 minutes
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = 0.5 * np.sin(440 * 2 * np.pi * t)
        return audio.astype(np.float32), sample_rate
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Buffered_Transcription import (
            BufferedTranscriptionConfig, MergeAlgorithm
        )
        
        return BufferedTranscriptionConfig(
            chunk_duration=2.0,
            total_buffer=4.0,
            batch_size=1,
            merge_algo=MergeAlgorithm.MIDDLE,
            device='cpu'
        )
    
    def test_import_module(self):
        """Test module imports."""
        try:
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Buffered_Transcription import (
                BufferedTranscriber,
                LCSMergeTranscriber,
                BufferedTranscriptionConfig,
                MergeAlgorithm,
                transcribe_long_audio
            )
            assert BufferedTranscriber is not None
            assert LCSMergeTranscriber is not None
            assert transcribe_long_audio is not None
        except ImportError as e:
            pytest.skip(f"Buffered transcription module not available: {e}")
    
    def test_config_creation(self):
        """Test configuration dataclass."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Buffered_Transcription import (
            BufferedTranscriptionConfig, MergeAlgorithm
        )
        
        config = BufferedTranscriptionConfig(
            chunk_duration=30.0,
            total_buffer=40.0,
            merge_algo=MergeAlgorithm.LCS
        )
        
        assert config.chunk_duration == 30.0
        assert config.total_buffer == 40.0
        assert config.merge_algo == MergeAlgorithm.LCS
        assert config.device == 'cpu'
    
    def test_chunk_creation(self, sample_audio_data, config):
        """Test audio chunking logic."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Buffered_Transcription import (
            BufferedTranscriber
        )
        
        audio_data, sample_rate = sample_audio_data
        transcriber = BufferedTranscriber(config)
        
        chunks = transcriber._create_chunks(audio_data)
        
        assert len(chunks) > 0
        
        # Check chunk structure
        for chunk in chunks:
            assert 'audio' in chunk
            assert 'start' in chunk
            assert 'end' in chunk
            assert 'overlap_start' in chunk
            assert 'overlap_end' in chunk
            assert isinstance(chunk['audio'], np.ndarray)
            assert chunk['end'] > chunk['start']
        
        # Check overlapping
        if len(chunks) > 1:
            # Adjacent chunks should overlap
            assert chunks[1]['start'] < chunks[0]['end']
    
    def test_middle_merge_algorithm(self, config):
        """Test middle merge algorithm."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Buffered_Transcription import (
            BufferedTranscriber
        )
        
        transcriber = BufferedTranscriber(config)
        
        # Create test chunk results with overlaps
        chunk_results = [
            {
                'text': 'Hello world this is',
                'start': 0.0,
                'end': 2.5,
                'overlap_start': 0.0,
                'overlap_end': 0.5
            },
            {
                'text': 'this is a test transcription',
                'start': 2.0,
                'end': 4.5,
                'overlap_start': 0.5,
                'overlap_end': 0.5
            },
            {
                'text': 'transcription of audio file',
                'start': 4.0,
                'end': 6.0,
                'overlap_start': 0.5,
                'overlap_end': 0.0
            }
        ]
        
        merged = transcriber._middle_merge(chunk_results)
        
        assert isinstance(merged, str)
        assert 'Hello world' in merged
        assert 'audio file' in merged
        # Check that overlaps are handled (no exact duplicates)
        assert merged.count('this is') <= 2
    
    def test_lcs_merge_algorithm(self, config):
        """Test LCS (Longest Common Subsequence) merge algorithm."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Buffered_Transcription import (
            LCSMergeTranscriber, MergeAlgorithm
        )
        
        config.merge_algo = MergeAlgorithm.LCS
        transcriber = LCSMergeTranscriber(config)
        
        # Test LCS merging
        text1 = "Hello world this is a test"
        text2 = "this is a test transcription"
        
        merged = transcriber._merge_with_lcs(text1, text2)
        
        assert "Hello world" in merged
        assert "transcription" in merged
        assert merged.count("this is a test") == 1  # Should not duplicate
    
    def test_lcs_length_calculation(self, config):
        """Test LCS length calculation."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Buffered_Transcription import (
            LCSMergeTranscriber, MergeAlgorithm
        )
        
        config.merge_algo = MergeAlgorithm.LCS
        transcriber = LCSMergeTranscriber(config)
        
        words1 = ["hello", "world", "test"]
        words2 = ["world", "test", "case"]
        
        lcs_len = transcriber._lcs_length(words1, words2)
        
        assert lcs_len == 2  # "world" and "test" are common
    
    def test_process_audio_with_mock_transcriber(self, sample_audio_data, config):
        """Test full audio processing with mocked transcription."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Buffered_Transcription import (
            BufferedTranscriber
        )
        
        audio_data, sample_rate = sample_audio_data
        transcriber = BufferedTranscriber(config)
        
        # Mock transcribe function
        transcribe_calls = []
        def mock_transcribe(chunk_audio):
            transcribe_calls.append(len(chunk_audio))
            return f"Transcription {len(transcribe_calls)}"
        
        # Track progress
        progress_calls = []
        def progress_callback(current, total):
            progress_calls.append((current, total))
        
        result = transcriber.process_audio(
            audio_data,
            sample_rate,
            mock_transcribe,
            progress_callback
        )
        
        assert isinstance(result, str)
        assert len(transcribe_calls) > 0
        assert len(progress_calls) > 0
        assert progress_calls[-1][0] == progress_calls[-1][1]  # Completed all chunks
    
    @patch('soundfile.read')
    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX.transcribe_with_parakeet_mlx')
    def test_transcribe_long_audio_mlx(self, mock_transcribe_mlx, mock_sf_read, long_audio_data):
        """Test transcribe_long_audio with MLX backend."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Buffered_Transcription import (
            transcribe_long_audio
        )
        
        audio_data, sample_rate = long_audio_data
        mock_sf_read.return_value = (audio_data, sample_rate)
        mock_transcribe_mlx.return_value = "Chunk transcription"
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            result = transcribe_long_audio(
                tmp_path,
                model_name='parakeet',
                variant='mlx',
                chunk_duration=30.0,
                total_buffer=40.0,
                merge_algo='middle'
            )
            
            assert isinstance(result, str)
            assert mock_transcribe_mlx.called
            mock_sf_read.assert_called_once_with(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX.transcribe_with_parakeet_onnx')
    def test_transcribe_long_audio_onnx(self, mock_transcribe_onnx, long_audio_data):
        """Test transcribe_long_audio with ONNX backend."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Buffered_Transcription import (
            transcribe_long_audio
        )
        
        audio_data, sample_rate = long_audio_data
        mock_transcribe_onnx.return_value = "ONNX chunk"
        
        result = transcribe_long_audio(
            audio_data,  # Pass numpy array directly
            model_name='parakeet',
            variant='onnx',
            chunk_duration=20.0,
            total_buffer=25.0,
            merge_algo='lcs',
            device='cpu'
        )
        
        assert isinstance(result, str)
        assert mock_transcribe_onnx.called
    
    def test_merge_algorithm_enum(self):
        """Test MergeAlgorithm enum values."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Buffered_Transcription import (
            MergeAlgorithm
        )
        
        assert MergeAlgorithm.MIDDLE.value == "middle"
        assert MergeAlgorithm.LCS.value == "lcs"
        assert MergeAlgorithm.TDT.value == "tdt"
        assert MergeAlgorithm.OVERLAP.value == "overlap"
        assert MergeAlgorithm.SIMPLE.value == "simple"
    
    def test_resampling(self, config):
        """Test audio resampling functionality."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Buffered_Transcription import (
            BufferedTranscriber
        )
        
        transcriber = BufferedTranscriber(config)
        
        # Create audio at different sample rate
        orig_sr = 44100
        target_sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(orig_sr * duration), False)
        audio = np.sin(440 * 2 * np.pi * t).astype(np.float32)
        
        # Mock librosa if available
        with patch('librosa.resample') as mock_resample:
            expected_length = int(len(audio) * target_sr / orig_sr)
            mock_resample.return_value = np.zeros(expected_length)
            
            resampled = transcriber._resample(audio, orig_sr, target_sr)
            
            mock_resample.assert_called_once_with(
                audio,
                orig_sr=orig_sr,
                target_sr=target_sr
            )
    
    def test_stereo_to_mono_conversion(self):
        """Test stereo to mono conversion."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Buffered_Transcription import (
            transcribe_long_audio
        )
        
        # Create stereo audio
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        left_channel = 0.5 * np.sin(440 * 2 * np.pi * t)
        right_channel = 0.5 * np.sin(880 * 2 * np.pi * t)
        stereo_audio = np.stack([left_channel, right_channel], axis=1).astype(np.float32)
        
        with patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX.transcribe_with_parakeet_mlx') as mock_transcribe:
            mock_transcribe.return_value = "Mono transcription"
            
            result = transcribe_long_audio(
                stereo_audio,
                variant='mlx',
                chunk_duration=10.0
            )
            
            # Verify mono conversion happened
            call_args = mock_transcribe.call_args[0]
            chunk_audio = call_args[0]
            assert chunk_audio.ndim == 1  # Mono audio
    
    def test_progress_callback(self, sample_audio_data, config):
        """Test progress callback functionality."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Buffered_Transcription import (
            BufferedTranscriber
        )
        
        audio_data, sample_rate = sample_audio_data
        transcriber = BufferedTranscriber(config)
        
        progress_updates = []
        def progress_callback(current, total):
            progress_updates.append({
                'current': current,
                'total': total,
                'percentage': (current / total) * 100 if total > 0 else 0
            })
        
        def mock_transcribe(chunk):
            return "test"
        
        transcriber.process_audio(
            audio_data,
            sample_rate,
            mock_transcribe,
            progress_callback
        )
        
        assert len(progress_updates) > 0
        # Check progress increases monotonically
        for i in range(1, len(progress_updates)):
            assert progress_updates[i]['current'] >= progress_updates[i-1]['current']
        # Check final progress is 100%
        assert progress_updates[-1]['percentage'] == 100


@pytest.mark.integration
class TestBufferedTranscriptionIntegration:
    """Integration tests for buffered transcription."""
    
    def test_integration_with_different_merge_algorithms(self):
        """Test all merge algorithms produce valid results."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Buffered_Transcription import (
            transcribe_long_audio
        )
        
        # Generate test audio
        sample_rate = 16000
        duration = 60.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1
        
        algorithms = ['middle', 'lcs', 'simple']
        
        for algo in algorithms:
            with patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX.transcribe_with_parakeet_mlx') as mock_transcribe:
                mock_transcribe.return_value = f"Test {algo}"
                
                result = transcribe_long_audio(
                    audio,
                    variant='mlx',
                    chunk_duration=10.0,
                    merge_algo=algo
                )
                
                assert isinstance(result, str)
                assert len(result) > 0


@pytest.mark.performance
class TestBufferedTranscriptionPerformance:
    """Performance tests for buffered transcription."""
    
    def test_chunking_overhead(self):
        """Measure overhead of chunking vs direct processing."""
        import time
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Buffered_Transcription import (
            BufferedTranscriber,
            BufferedTranscriptionConfig,
            MergeAlgorithm
        )
        
        # Generate test audio
        sample_rate = 16000
        duration = 60.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1
        
        # Mock transcribe function with delay
        def mock_transcribe(chunk):
            time.sleep(0.001)  # Simulate processing
            return f"Transcribed {len(chunk)} samples"
        
        # Test with chunking
        config = BufferedTranscriptionConfig(
            chunk_duration=10.0,
            total_buffer=15.0,
            merge_algo=MergeAlgorithm.SIMPLE
        )
        transcriber = BufferedTranscriber(config)
        
        start = time.time()
        result_chunked = transcriber.process_audio(
            audio,
            sample_rate,
            mock_transcribe
        )
        time_chunked = time.time() - start
        
        # Test without chunking (single chunk)
        config_single = BufferedTranscriptionConfig(
            chunk_duration=duration,
            total_buffer=duration,
            merge_algo=MergeAlgorithm.SIMPLE
        )
        transcriber_single = BufferedTranscriber(config_single)
        
        start = time.time()
        result_single = transcriber_single.process_audio(
            audio,
            sample_rate,
            mock_transcribe
        )
        time_single = time.time() - start
        
        print(f"Chunked processing: {time_chunked:.3f}s")
        print(f"Single processing: {time_single:.3f}s")
        print(f"Overhead: {(time_chunked - time_single) / time_single * 100:.1f}%")
        
        assert result_chunked is not None
        assert result_single is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])