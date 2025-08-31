"""
Unit and integration tests for WebSocket-based streaming transcription.
"""

import pytest
import asyncio
import numpy as np
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import websockets
from typing import Dict, Any, Optional

pytestmark = pytest.mark.unit


class TestStreamingTranscription:
    """Test suite for streaming transcription functionality."""
    
    @pytest.fixture
    def audio_chunk(self):
        """Generate a single audio chunk for streaming."""
        sample_rate = 16000
        duration = 0.5  # 500ms chunk
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = (0.5 * np.sin(440 * 2 * np.pi * t)).astype(np.float32)
        return audio.tobytes(), sample_rate
    
    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket connection."""
        ws = AsyncMock()
        ws.send = AsyncMock()
        ws.recv = AsyncMock()
        ws.close = AsyncMock()
        ws.closed = False
        return ws
    
    def test_import_module(self):
        """Test that streaming module can be imported."""
        try:
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet import (
                ParakeetStreamingTranscriber,
                StreamingConfig,
                AudioBuffer
            )
            assert ParakeetStreamingTranscriber is not None
            assert StreamingConfig is not None
            assert AudioBuffer is not None
        except ImportError as e:
            pytest.skip(f"Streaming module not available: {e}")
    
    def test_streaming_config(self):
        """Test StreamingConfig creation."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet import (
            StreamingConfig
        )
        
        config = StreamingConfig(
            sample_rate=16000,
            chunk_duration=0.5,
            min_audio_length=1.0,
            silence_threshold=0.01,
            model_variant='mlx'
        )
        
        assert config.sample_rate == 16000
        assert config.chunk_duration == 0.5
        assert config.min_audio_length == 1.0
        assert config.model_variant == 'mlx'
    
    def test_audio_buffer(self):
        """Test AudioBuffer functionality."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet import (
            AudioBuffer
        )
        
        buffer = AudioBuffer(max_size=16000 * 10)  # 10 seconds at 16kHz
        
        # Add audio chunks
        chunk1 = np.random.randn(8000).astype(np.float32)
        chunk2 = np.random.randn(8000).astype(np.float32)
        
        buffer.add(chunk1)
        assert buffer.size() == 8000
        
        buffer.add(chunk2)
        assert buffer.size() == 16000
        
        # Get buffered audio
        audio = buffer.get()
        assert len(audio) == 16000
        assert np.array_equal(audio[:8000], chunk1)
        assert np.array_equal(audio[8000:], chunk2)
        
        # Clear buffer
        buffer.clear()
        assert buffer.size() == 0
    
    def test_audio_buffer_overflow(self):
        """Test AudioBuffer with overflow."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet import (
            AudioBuffer
        )
        
        max_size = 1000
        buffer = AudioBuffer(max_size=max_size)
        
        # Add more than max size
        large_chunk = np.random.randn(1500).astype(np.float32)
        buffer.add(large_chunk)
        
        # Should keep only last max_size samples
        assert buffer.size() == max_size
        audio = buffer.get()
        assert len(audio) == max_size
        assert np.array_equal(audio, large_chunk[-max_size:])
    
    @pytest.mark.asyncio
    async def test_streaming_transcriber_init(self):
        """Test ParakeetStreamingTranscriber initialization."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet import (
            ParakeetStreamingTranscriber,
            StreamingConfig
        )
        
        config = StreamingConfig()
        
        with patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet.load_parakeet_mlx_model') as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model
            
            transcriber = ParakeetStreamingTranscriber(config)
            await transcriber.initialize()
            
            assert transcriber.model == mock_model
            mock_load.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_audio_chunk(self):
        """Test processing single audio chunk."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet import (
            ParakeetStreamingTranscriber,
            StreamingConfig
        )
        
        config = StreamingConfig(min_audio_length=0.1)  # Low threshold for testing
        transcriber = ParakeetStreamingTranscriber(config)
        
        # Mock model
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "Test transcription"
        mock_model.transcribe.return_value = mock_result
        transcriber.model = mock_model
        
        # Create audio chunk
        audio_data = np.random.randn(16000).astype(np.float32).tobytes()
        
        result = await transcriber.process_audio_chunk(audio_data)
        
        assert result is not None
        assert result['type'] == 'transcription'
        assert result['text'] == "Test transcription"
        assert result['is_final'] == True
    
    @pytest.mark.asyncio
    async def test_voice_activity_detection(self):
        """Test voice activity detection."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet import (
            ParakeetStreamingTranscriber,
            StreamingConfig
        )
        
        config = StreamingConfig(silence_threshold=0.01)
        transcriber = ParakeetStreamingTranscriber(config)
        
        # Test with silence (low amplitude)
        silence = np.random.randn(1000).astype(np.float32) * 0.001
        is_speech_silence = transcriber._detect_voice_activity(silence)
        assert is_speech_silence == False
        
        # Test with speech (higher amplitude)
        speech = np.random.randn(1000).astype(np.float32) * 0.5
        is_speech_speech = transcriber._detect_voice_activity(speech)
        assert is_speech_speech == True
    
    @pytest.mark.asyncio
    async def test_websocket_handler(self, mock_websocket):
        """Test WebSocket connection handler."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet import (
            handle_websocket_connection
        )
        
        # Mock receive messages
        messages = [
            json.dumps({'type': 'start', 'config': {'sample_rate': 16000}}),
            json.dumps({'type': 'audio', 'data': 'base64audiodata'}),
            json.dumps({'type': 'stop'})
        ]
        mock_websocket.recv.side_effect = messages + [websockets.exceptions.ConnectionClosed(None, None)]
        
        with patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet.ParakeetStreamingTranscriber') as mock_transcriber_class:
            mock_transcriber = AsyncMock()
            mock_transcriber.initialize = AsyncMock()
            mock_transcriber.process_audio_chunk = AsyncMock(return_value={
                'type': 'transcription',
                'text': 'Test',
                'is_final': True
            })
            mock_transcriber.finalize = AsyncMock(return_value="Final transcription")
            mock_transcriber_class.return_value = mock_transcriber
            
            await handle_websocket_connection(mock_websocket, "/ws")
            
            # Verify WebSocket interactions
            assert mock_websocket.send.called
            assert mock_transcriber.initialize.called
            assert mock_transcriber.process_audio_chunk.called
    
    @pytest.mark.asyncio
    async def test_streaming_with_buffer_accumulation(self):
        """Test streaming with audio buffer accumulation."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet import (
            ParakeetStreamingTranscriber,
            StreamingConfig
        )
        
        config = StreamingConfig(
            min_audio_length=2.0,  # Require 2 seconds
            sample_rate=16000
        )
        transcriber = ParakeetStreamingTranscriber(config)
        
        # Mock model
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "Accumulated transcription"
        mock_model.transcribe.return_value = mock_result
        transcriber.model = mock_model
        
        # Send small chunks that need accumulation
        small_chunk = np.random.randn(8000).astype(np.float32).tobytes()  # 0.5 seconds
        
        # First chunk - should not transcribe yet
        result1 = await transcriber.process_audio_chunk(small_chunk)
        assert result1 is None or result1.get('type') == 'partial'
        
        # Second chunk - should not transcribe yet
        result2 = await transcriber.process_audio_chunk(small_chunk)
        assert result2 is None or result2.get('type') == 'partial'
        
        # Third chunk - should trigger transcription (1.5 seconds total)
        result3 = await transcriber.process_audio_chunk(small_chunk)
        assert result3 is None or result3.get('type') == 'partial'
        
        # Fourth chunk - should have enough audio (2 seconds)
        result4 = await transcriber.process_audio_chunk(small_chunk)
        assert result4 is not None
        if result4.get('type') == 'transcription':
            assert result4['text'] == "Accumulated transcription"
    
    @pytest.mark.asyncio
    async def test_finalize_transcription(self):
        """Test finalizing transcription on stream end."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet import (
            ParakeetStreamingTranscriber,
            StreamingConfig
        )
        
        config = StreamingConfig()
        transcriber = ParakeetStreamingTranscriber(config)
        
        # Add some audio to buffer
        audio_chunk = np.random.randn(16000).astype(np.float32)
        transcriber.audio_buffer.add(audio_chunk)
        
        # Mock model
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "Final transcription"
        mock_model.transcribe.return_value = mock_result
        transcriber.model = mock_model
        
        # Finalize
        final_text = await transcriber.finalize()
        
        assert final_text == "Final transcription"
        assert transcriber.audio_buffer.size() == 0  # Buffer should be cleared
    
    @pytest.mark.asyncio
    async def test_error_handling(self, mock_websocket):
        """Test error handling in streaming."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet import (
            ParakeetStreamingTranscriber,
            StreamingConfig
        )
        
        config = StreamingConfig()
        transcriber = ParakeetStreamingTranscriber(config)
        
        # Mock model that raises error
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = Exception("Model error")
        transcriber.model = mock_model
        
        # Add audio
        audio_data = np.random.randn(16000).astype(np.float32).tobytes()
        
        # Should handle error gracefully
        result = await transcriber.process_audio_chunk(audio_data)
        
        # Check error result
        if result:
            assert result.get('type') == 'error' or result.get('error') is not None


@pytest.mark.integration
class TestStreamingIntegration:
    """Integration tests for streaming transcription."""
    
    @pytest.mark.asyncio
    async def test_full_streaming_session(self):
        """Test complete streaming session."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet import (
            ParakeetStreamingTranscriber,
            StreamingConfig
        )
        
        config = StreamingConfig(
            sample_rate=16000,
            chunk_duration=0.5,
            min_audio_length=1.0
        )
        
        with patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet.load_parakeet_mlx_model') as mock_load:
            mock_model = MagicMock()
            mock_result = MagicMock()
            mock_result.text = "Streaming test"
            mock_model.transcribe.return_value = mock_result
            mock_load.return_value = mock_model
            
            transcriber = ParakeetStreamingTranscriber(config)
            await transcriber.initialize()
            
            # Simulate streaming session
            transcriptions = []
            
            # Send multiple chunks
            for i in range(5):
                audio_chunk = np.random.randn(8000).astype(np.float32).tobytes()
                result = await transcriber.process_audio_chunk(audio_chunk)
                if result and result.get('type') == 'transcription':
                    transcriptions.append(result['text'])
            
            # Finalize
            final = await transcriber.finalize()
            if final:
                transcriptions.append(final)
            
            assert len(transcriptions) > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_streams(self):
        """Test handling multiple concurrent streams."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet import (
            ParakeetStreamingTranscriber,
            StreamingConfig
        )
        
        config = StreamingConfig()
        
        async def simulate_stream(stream_id: int):
            with patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet.load_parakeet_mlx_model') as mock_load:
                mock_model = MagicMock()
                mock_result = MagicMock()
                mock_result.text = f"Stream {stream_id}"
                mock_model.transcribe.return_value = mock_result
                mock_load.return_value = mock_model
                
                transcriber = ParakeetStreamingTranscriber(config)
                await transcriber.initialize()
                
                # Process audio
                audio = np.random.randn(16000).astype(np.float32).tobytes()
                result = await transcriber.process_audio_chunk(audio)
                
                return result
        
        # Run multiple streams concurrently
        results = await asyncio.gather(
            simulate_stream(1),
            simulate_stream(2),
            simulate_stream(3)
        )
        
        assert len(results) == 3
        for i, result in enumerate(results):
            if result:
                assert f"Stream {i+1}" in result.get('text', '')


@pytest.mark.performance
class TestStreamingPerformance:
    """Performance tests for streaming transcription."""
    
    @pytest.mark.asyncio
    async def test_latency(self):
        """Test streaming latency."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet import (
            ParakeetStreamingTranscriber,
            StreamingConfig
        )
        import time
        
        config = StreamingConfig(min_audio_length=0.5)
        
        with patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet.load_parakeet_mlx_model') as mock_load:
            mock_model = MagicMock()
            
            def mock_transcribe(*args, **kwargs):
                # Simulate processing time
                time.sleep(0.01)
                result = MagicMock()
                result.text = "Low latency"
                return result
            
            mock_model.transcribe = mock_transcribe
            mock_load.return_value = mock_model
            
            transcriber = ParakeetStreamingTranscriber(config)
            await transcriber.initialize()
            
            # Measure latency
            audio_chunk = np.random.randn(8000).astype(np.float32).tobytes()
            
            start = time.time()
            result = await transcriber.process_audio_chunk(audio_chunk)
            latency = time.time() - start
            
            assert latency < 0.1  # Should be under 100ms
            print(f"Streaming latency: {latency*1000:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_throughput(self):
        """Test streaming throughput."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet import (
            ParakeetStreamingTranscriber,
            StreamingConfig
        )
        import time
        
        config = StreamingConfig()
        
        with patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet.load_parakeet_mlx_model') as mock_load:
            mock_model = MagicMock()
            mock_result = MagicMock()
            mock_result.text = "Throughput test"
            mock_model.transcribe.return_value = mock_result
            mock_load.return_value = mock_model
            
            transcriber = ParakeetStreamingTranscriber(config)
            await transcriber.initialize()
            
            # Process multiple chunks
            num_chunks = 100
            chunk_size = 8000  # 0.5 seconds at 16kHz
            total_audio_duration = (num_chunks * chunk_size) / 16000
            
            start = time.time()
            for _ in range(num_chunks):
                audio_chunk = np.random.randn(chunk_size).astype(np.float32).tobytes()
                await transcriber.process_audio_chunk(audio_chunk)
            elapsed = time.time() - start
            
            throughput = total_audio_duration / elapsed
            print(f"Streaming throughput: {throughput:.1f}x real-time")
            
            assert throughput > 1.0  # Should process faster than real-time


if __name__ == "__main__":
    pytest.main([__file__, "-v"])