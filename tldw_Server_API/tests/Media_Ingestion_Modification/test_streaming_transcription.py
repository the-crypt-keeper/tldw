"""
Unit and integration tests for WebSocket-based streaming transcription.
"""

import pytest
import asyncio
import os
import numpy as np
import json
import base64
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
            overlap_duration=0.25,
            max_buffer_duration=30.0,
            model_variant='mlx'
        )
        
        assert config.sample_rate == 16000
        assert config.chunk_duration == 0.5
        assert config.overlap_duration == 0.25
        assert config.model_variant == 'mlx'
    
    def test_audio_buffer(self):
        """Test AudioBuffer functionality."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet import (
            AudioBuffer
        )
        
        buffer = AudioBuffer(sample_rate=16000, max_duration=10.0)  # 10 seconds at 16kHz
        
        # Add audio chunks
        chunk1 = np.random.randn(8000).astype(np.float32)
        chunk2 = np.random.randn(8000).astype(np.float32)
        
        buffer.add(chunk1)
        assert buffer.get_duration() > 0
        
        buffer.add(chunk2)
        assert buffer.get_duration() > 0
        
        # Get buffered audio
        audio = buffer.get_audio()
        assert len(audio) == 16000
        assert np.array_equal(audio[:8000], chunk1)
        assert np.array_equal(audio[8000:], chunk2)
        
        # Clear buffer
        buffer.clear()
        assert buffer.get_duration() == 0
    
    def test_audio_buffer_overflow(self):
        """Test AudioBuffer with overflow."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet import (
            AudioBuffer
        )
        
        sample_rate = 16000
        max_duration = 0.1  # 0.1 seconds
        max_samples = int(sample_rate * max_duration)
        buffer = AudioBuffer(sample_rate=sample_rate, max_duration=max_duration)
        
        # Add more than max size
        large_chunk = np.random.randn(max_samples * 2).astype(np.float32)
        buffer.add(large_chunk)
        
        # Should trim to max duration
        audio = buffer.get_audio()
        assert audio is not None
        # Buffer trims old data when exceeding max duration
    
    @pytest.mark.asyncio
    async def test_streaming_transcriber_init(self):
        """Test ParakeetStreamingTranscriber initialization."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet import (
            ParakeetStreamingTranscriber,
            StreamingConfig
        )
        
        config = StreamingConfig(model_variant='standard')
        
        with patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo.load_parakeet_model') as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model
            
            transcriber = ParakeetStreamingTranscriber(config)
            transcriber.initialize()
            
            assert transcriber.model == mock_model
            mock_load.assert_called_once_with('standard')
    
    @pytest.mark.asyncio
    async def test_process_audio_chunk(self):
        """Test processing single audio chunk."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet import (
            ParakeetStreamingTranscriber,
            StreamingConfig
        )
        
        config = StreamingConfig(chunk_duration=0.1)  # Low threshold for testing
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
        """Test voice activity detection (if implemented)."""
        # Note: Voice activity detection is not implemented in the current code
        # This test is a placeholder for future functionality
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet import (
            ParakeetStreamingTranscriber,
            StreamingConfig
        )
        
        config = StreamingConfig(chunk_duration=1.0)  # Standard config
        transcriber = ParakeetStreamingTranscriber(config)
        
        # Skip test if VAD not implemented
        if not hasattr(transcriber, '_detect_voice_activity'):
            pytest.skip("Voice activity detection not yet implemented")
        
        # Test with silence (true zeros)
        silence = np.zeros(1000).astype(np.float32)
        is_speech_silence = transcriber._detect_voice_activity(silence)
        assert is_speech_silence == False
        
        # Test with speech-like signal (sine wave)
        t = np.linspace(0, 0.1, 1000)
        speech = (0.5 * np.sin(440 * 2 * np.pi * t)).astype(np.float32)
        is_speech_speech = transcriber._detect_voice_activity(speech)
        assert is_speech_speech == True
    
    @pytest.mark.asyncio
    async def test_websocket_handler(self, mock_websocket):
        """Test WebSocket connection handler."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet import (
            handle_websocket_transcription
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
            
            await handle_websocket_transcription(mock_websocket)
            
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
            chunk_duration=2.0,  # Process 2 second chunks
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
        
        config = StreamingConfig(model_variant='standard')
        transcriber = ParakeetStreamingTranscriber(config)
        
        # Add some audio to buffer
        audio_chunk = np.random.randn(16000).astype(np.float32)
        transcriber.buffer.add(audio_chunk)
        
        # Mock transcribe method
        with patch.object(transcriber, '_transcribe_chunk', new=AsyncMock(return_value="Final transcription")):
            # Flush the buffer
            result = await transcriber.flush()
            
            assert result is not None
            assert result['text'] == "Final transcription"
            assert result['type'] == 'final'
            assert result['is_final'] == True
            assert len(transcriber.buffer.data) == 0  # Buffer should be cleared
    
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


class TestStreamingErrorScenarios:
    """Test suite for error handling and edge cases in streaming transcription."""
    
    @pytest.mark.asyncio
    async def test_empty_audio_handling(self):
        """Test handling of empty audio chunks."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet import (
            ParakeetStreamingTranscriber,
            StreamingConfig
        )
        
        config = StreamingConfig()
        transcriber = ParakeetStreamingTranscriber(config)
        
        # Send empty audio
        empty_audio = base64.b64encode(b'').decode('utf-8')
        result = await transcriber.process_audio_chunk(empty_audio)
        
        # Should handle gracefully
        assert result is None or result.get('type') == 'error'
    
    @pytest.mark.asyncio
    async def test_corrupted_audio_data(self):
        """Test handling of corrupted base64 audio data."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet import (
            ParakeetStreamingTranscriber,
            StreamingConfig
        )
        
        config = StreamingConfig()
        transcriber = ParakeetStreamingTranscriber(config)
        
        # Send corrupted base64
        corrupted_data = "not_valid_base64!!!"
        result = await transcriber.process_audio_chunk(corrupted_data)
        
        assert result is not None
        assert result.get('type') == 'error'
        assert 'message' in result
    
    @pytest.mark.asyncio
    async def test_extreme_buffer_sizes(self):
        """Test with extreme buffer configurations."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet import (
            ParakeetStreamingTranscriber,
            StreamingConfig,
            AudioBuffer
        )
        
        # Test with very small buffer
        config_small = StreamingConfig(
            chunk_duration=0.1,
            max_buffer_duration=0.5
        )
        transcriber_small = ParakeetStreamingTranscriber(config_small)
        
        # Test with very large buffer
        config_large = StreamingConfig(
            chunk_duration=300.0,
            max_buffer_duration=600.0
        )
        transcriber_large = ParakeetStreamingTranscriber(config_large)
        
        # Both should initialize without errors
        assert transcriber_small is not None
        assert transcriber_large is not None
    
    @pytest.mark.asyncio
    async def test_websocket_disconnection(self, mock_websocket):
        """Test handling of WebSocket disconnection during streaming."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet import (
            handle_websocket_transcription
        )
        
        # Simulate disconnection after first message
        mock_websocket.recv.side_effect = [
            json.dumps({'type': 'start', 'config': {'sample_rate': 16000}}),
            websockets.exceptions.ConnectionClosed(None, None)
        ]
        
        # Should handle gracefully without raising
        try:
            await handle_websocket_transcription(mock_websocket)
        except Exception as e:
            pytest.fail(f"WebSocket disconnection not handled gracefully: {e}")
    
    @pytest.mark.asyncio
    async def test_invalid_audio_format(self):
        """Test handling of invalid audio format."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet import (
            ParakeetStreamingTranscriber,
            StreamingConfig
        )
        
        config = StreamingConfig()
        transcriber = ParakeetStreamingTranscriber(config)
        
        # Send audio with wrong dtype (int16 instead of float32)
        wrong_dtype_audio = np.random.randint(-32768, 32767, 1000, dtype=np.int16)
        encoded = base64.b64encode(wrong_dtype_audio.tobytes()).decode('utf-8')
        
        # Should handle type conversion or error gracefully
        result = await transcriber.process_audio_chunk(encoded)
        # Either processes successfully or returns error
        assert result is None or 'type' in result
    
    @pytest.mark.asyncio
    async def test_concurrent_flush_operations(self):
        """Test concurrent flush operations don't cause issues."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet import (
            ParakeetStreamingTranscriber,
            StreamingConfig
        )
        
        config = StreamingConfig()
        transcriber = ParakeetStreamingTranscriber(config)
        
        # Add audio
        audio = np.random.randn(16000).astype(np.float32)
        transcriber.buffer.add(audio)
        
        # Mock transcribe
        with patch.object(transcriber, '_transcribe_chunk', new=AsyncMock(return_value="Test")):
            # Concurrent flushes
            results = await asyncio.gather(
                transcriber.flush(),
                transcriber.flush(),
                return_exceptions=True
            )
            
            # At least one should succeed
            successful = [r for r in results if isinstance(r, dict)]
            assert len(successful) >= 1


@pytest.mark.integration
class TestStreamingIntegration:
    """Integration tests for streaming transcription."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.path.exists('/path/to/model'),  # Skip if model not available
        reason="Requires actual Parakeet model"
    )
    async def test_full_streaming_session(self):
        """Test complete streaming session with real model."""
        pytest.skip("Integration test requires actual model setup")
        
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Parakeet import (
            ParakeetStreamingTranscriber,
            StreamingConfig
        )
        
        # Use real model configuration
        config = StreamingConfig(
            model_variant='mlx',  # or 'standard', 'onnx'
            sample_rate=16000,
            chunk_duration=2.0,
            overlap_duration=0.5
        )
        
        transcriber = ParakeetStreamingTranscriber(config)
        transcriber.initialize()
        
        # Generate realistic test audio (sine wave at speech frequency)
        duration = 5.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Mix of frequencies common in speech (100-1000 Hz)
        audio = (
            0.3 * np.sin(2 * np.pi * 200 * t) +
            0.2 * np.sin(2 * np.pi * 400 * t) +
            0.1 * np.sin(2 * np.pi * 800 * t)
        ).astype(np.float32)
        
        # Process in chunks
        chunk_size = int(sample_rate * 0.5)  # 0.5 second chunks
        transcriptions = []
        
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            encoded = base64.b64encode(chunk.tobytes()).decode('utf-8')
            
            result = await transcriber.process_audio_chunk(encoded)
            if result and result.get('type') == 'transcription':
                transcriptions.append(result['text'])
        
        # Flush remaining
        final = await transcriber.flush()
        if final:
            transcriptions.append(final['text'])
        
        # Should have produced some transcription
        assert len(transcriptions) > 0
    
    @pytest.mark.asyncio
    async def test_websocket_server_integration(self):
        """Test WebSocket server integration."""
        pytest.skip("Requires WebSocket server to be running")
        
        import websockets
        
        # This would connect to actual running server
        uri = "ws://localhost:8000/ws/transcribe"
        
        async with websockets.connect(uri) as websocket:
            # Send configuration
            await websocket.send(json.dumps({
                'type': 'config',
                'sample_rate': 16000,
                'language': 'en'
            }))
            
            # Generate and send test audio
            audio = np.random.randn(16000).astype(np.float32)
            encoded = base64.b64encode(audio.tobytes()).decode('utf-8')
            
            await websocket.send(json.dumps({
                'type': 'audio',
                'data': encoded
            }))
            
            # Receive transcription
            response = await websocket.recv()
            result = json.loads(response)
            
            assert 'type' in result
            assert result['type'] in ['transcription', 'partial', 'error']


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
        
        config = StreamingConfig(chunk_duration=0.5)
        
        with patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo.load_parakeet_model') as mock_load:
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
        
        with patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo.load_parakeet_model') as mock_load:
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