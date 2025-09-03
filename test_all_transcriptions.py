#!/usr/bin/env python3
"""
Test all transcription implementations with the sample audio file.
"""

import sys
import time
import os
sys.path.insert(0, '.')
os.chdir('/Users/macbook-dev/Documents/GitHub/tldw_server/tldw_Server_API')

# Test audio file (extracted from MP4)
TEST_AUDIO = "/tmp/test_audio.wav"

def test_mlx_implementation():
    """Test MLX Parakeet implementation."""
    print("\n" + "="*60)
    print("Testing MLX Parakeet Implementation")
    print("="*60)
    
    try:
        from app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX import (
            transcribe_with_parakeet_mlx, check_mlx_available
        )
        
        if not check_mlx_available():
            print("❌ MLX not available on this system")
            return None
        
        print("✓ MLX is available")
        
        # Test without chunking
        print("\n1. Testing without chunking...")
        start_time = time.time()
        
        result = transcribe_with_parakeet_mlx(
            TEST_AUDIO,
            sample_rate=16000,
            verbose=True
        )
        
        elapsed = time.time() - start_time
        print(f"Time taken: {elapsed:.2f} seconds")
        print(f"Result preview: {result[:200] if result else 'No result'}...")
        
        # Test with chunking
        print("\n2. Testing with chunking (30-second chunks)...")
        chunks_processed = []
        
        def chunk_callback(current, total):
            chunks_processed.append(f"{current}/{total}")
            print(f"  Processing chunk {current}/{total}")
        
        start_time = time.time()
        
        result_chunked = transcribe_with_parakeet_mlx(
            TEST_AUDIO,
            sample_rate=16000,
            chunk_duration=30.0,
            overlap_duration=5.0,
            chunk_callback=chunk_callback
        )
        
        elapsed = time.time() - start_time
        print(f"Time taken: {elapsed:.2f} seconds")
        print(f"Chunks processed: {len(chunks_processed)}")
        print(f"Result preview: {result_chunked[:200] if result_chunked else 'No result'}...")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_onnx_implementation():
    """Test ONNX Parakeet implementation."""
    print("\n" + "="*60)
    print("Testing ONNX Parakeet Implementation")
    print("="*60)
    
    try:
        from app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX import (
            transcribe_with_parakeet_onnx, load_parakeet_onnx_model
        )
        
        print("Loading ONNX model...")
        session, tokenizer = load_parakeet_onnx_model(device='cpu')
        
        if session is None:
            print("❌ Failed to load ONNX model")
            return None
        
        print(f"✓ ONNX model loaded")
        print(f"  - Session inputs: {[inp.name for inp in session.get_inputs()]}")
        print(f"  - Vocabulary size: {len(tokenizer.vocab)}")
        
        # Test without chunking
        print("\n1. Testing without chunking...")
        start_time = time.time()
        
        result = transcribe_with_parakeet_onnx(
            TEST_AUDIO,
            sample_rate=16000,
            device='cpu'
        )
        
        elapsed = time.time() - start_time
        print(f"Time taken: {elapsed:.2f} seconds")
        print(f"Result preview: {result[:200] if result else 'No result'}...")
        
        # Test with chunking
        print("\n2. Testing with chunking (30-second chunks)...")
        chunks_processed = []
        
        def chunk_callback(current, total):
            chunks_processed.append(f"{current}/{total}")
            print(f"  Processing chunk {current}/{total}")
        
        start_time = time.time()
        
        result_chunked = transcribe_with_parakeet_onnx(
            TEST_AUDIO,
            sample_rate=16000,
            chunk_duration=30.0,
            overlap_duration=5.0,
            merge_algo='lcs',  # Try LCS merge
            chunk_callback=chunk_callback
        )
        
        elapsed = time.time() - start_time
        print(f"Time taken: {elapsed:.2f} seconds")
        print(f"Chunks processed: {len(chunks_processed)}")
        print(f"Result preview: {result_chunked[:200] if result_chunked else 'No result'}...")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_nemo_implementation():
    """Test standard Nemo implementation."""
    print("\n" + "="*60)
    print("Testing Standard Nemo Implementation")
    print("="*60)
    
    try:
        from app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
            transcribe_with_nemo, transcribe_with_parakeet
        )
        
        # Test Parakeet standard variant
        print("\n1. Testing Parakeet standard variant...")
        start_time = time.time()
        
        result = transcribe_with_parakeet(
            TEST_AUDIO,
            sample_rate=16000,
            variant='standard'
        )
        
        elapsed = time.time() - start_time
        print(f"Time taken: {elapsed:.2f} seconds")
        print(f"Result preview: {result[:200] if result else 'No result'}...")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_buffered_transcription():
    """Test advanced buffered transcription."""
    print("\n" + "="*60)
    print("Testing Advanced Buffered Transcription")
    print("="*60)
    
    try:
        from app.core.Ingestion_Media_Processing.Audio.Audio_Buffered_Transcription import (
            transcribe_long_audio
        )
        
        # Test with middle merge
        print("\n1. Testing with middle merge algorithm...")
        
        def progress_callback(current, total):
            print(f"  Processing chunk {current}/{total}")
        
        start_time = time.time()
        
        result_middle = transcribe_long_audio(
            TEST_AUDIO,
            model_name='parakeet',
            variant='mlx',
            chunk_duration=20.0,
            total_buffer=25.0,
            merge_algo='middle',
            progress_callback=progress_callback
        )
        
        elapsed = time.time() - start_time
        print(f"Time taken: {elapsed:.2f} seconds")
        print(f"Result preview: {result_middle[:200] if result_middle else 'No result'}...")
        
        # Test with LCS merge
        print("\n2. Testing with LCS merge algorithm...")
        start_time = time.time()
        
        result_lcs = transcribe_long_audio(
            TEST_AUDIO,
            model_name='parakeet',
            variant='mlx',
            chunk_duration=20.0,
            total_buffer=25.0,
            merge_algo='lcs',
            progress_callback=progress_callback
        )
        
        elapsed = time.time() - start_time
        print(f"Time taken: {elapsed:.2f} seconds")
        print(f"Result preview: {result_lcs[:200] if result_lcs else 'No result'}...")
        
        return result_middle
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all tests."""
    print("="*60)
    print("TRANSCRIPTION IMPLEMENTATION TEST SUITE")
    print("="*60)
    print(f"Test audio file: {TEST_AUDIO}")
    
    # Check if test audio exists
    import os
    if not os.path.exists(TEST_AUDIO):
        print(f"❌ Test audio file not found: {TEST_AUDIO}")
        print("Please extract audio from MP4 first:")
        print("ffmpeg -i sample.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 /tmp/test_audio.wav")
        return
    
    # Get file info
    import soundfile as sf
    info = sf.info(TEST_AUDIO)
    print(f"Audio duration: {info.duration:.2f} seconds")
    print(f"Sample rate: {info.samplerate} Hz")
    print(f"Channels: {info.channels}")
    
    results = {}
    
    # Test each implementation
    print("\nStarting tests...")
    
    # 1. Test MLX
    results['mlx'] = test_mlx_implementation()
    
    # 2. Test ONNX
    results['onnx'] = test_onnx_implementation()
    
    # 3. Test Standard Nemo
    # results['nemo'] = test_nemo_implementation()  # Skip if Nemo toolkit not installed
    
    # 4. Test Buffered Transcription
    results['buffered'] = test_buffered_transcription()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for impl, result in results.items():
        status = "✓" if result else "✗"
        length = len(result) if result else 0
        print(f"{status} {impl.upper()}: {'Success' if result else 'Failed'} (Result length: {length} chars)")
    
    # Compare results if multiple succeeded
    successful = [r for r in results.values() if r]
    if len(successful) > 1:
        print("\nResult comparison:")
        print(f"All results start with same text: {all(r[:50] == successful[0][:50] for r in successful)}")


if __name__ == "__main__":
    main()