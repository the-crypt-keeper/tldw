# Text-To-Speech / Speech-To-Text Documentation

## Overview
Use of functions for individual services.
Function for each service, streaming & non-streaming.
Non-streaming will return a file, streaming will return a stream.
Use of temporary files for storage.
Use of pydub for audio manipulation.
Use of pydub for audio merging.

Flow:
1. Clean/format input text
2. Split text into segments
3. Generate audio for each segment using designated provider function
4. Merge audio segments into single output file
5. Clean up temporary files

### Services
- Google Cloud Text-to-Speech
    - https://cloud.google.com/text-to-speech/docs/ssml
  
  
### Benchmarks
https://huggingface.co/blog/big-bench-audio-release
    https://huggingface.co/datasets/ArtificialAnalysis/big_bench_audio
https://artificialanalysis.ai/models/speech-to-speech





### Link Dump:
https://github.com/albirrkarim/react-speech-highlight-demo
https://funaudiollm.github.io/cosyvoice2/
https://funaudiollm.github.io/cosyvoice2/
https://github.com/InternLM/InternLM-XComposer/tree/main/InternLM-XComposer-2.5-OmniLive
https://github.com/huggingface/transformers.js-examples/tree/main/moonshine-web
https://huggingface.co/onnx-community/moonshine-base-ONNX
https://github.com/usefulsensors/moonshine
https://github.com/Azure-Samples/aisearch-openai-rag-audio
https://www.reddit.com/r/LocalLLaMA/comments/1f0awd6/best_local_open_source_texttospeech_and/
https://github.com/FanaHOVA/smol-podcaster
https://docs.inferless.com/cookbook/serverless-customer-service-bot
https://wave-pulse.io/
https://huggingface.co/spaces/saq1b/podcastgen/blob/main/app.py
https://huggingface.co/spaces/mozilla-ai/document-to-podcast/blob/main/app.py
https://huggingface.co/spaces/Nymbo/Voice-Clone-Multilingual/tree/main
https://github.com/aedocw/epub2tts
https://github.com/microsoft/SpeechT5
https://github.com/smellslikeml/dolla_llama


STT
    https://github.com/KoljaB/RealtimeSTT

TTS
    https://github.com/KoljaB/RealtimeTTS

101
    https://www.inferless.com/learn/comparing-different-text-to-speech---tts--models-for-different-use-cases
    https://clideo.com/resources/what-is-tts
    RVC 101
        https://gudgud96.github.io/2024/09/26/annotated-rvc/

Datasets(?)
    https://voice-models.com/

Bark
https://github.com/suno-ai/bark


ChatTTS
https://huggingface.co/2Noise/ChatTTS
https://chattts.com/#Demo



Coqui TTS
    https://github.com/idiap/coqui-ai-TTS
    https://huggingface.co/spaces/coqui/xtts/blob/main/app.py

Cartesia
    https://docs.cartesia.ai/get-started/make-an-api-request

F5 TTS
    https://github.com/SWivid/F5-TTS

lina TTS
https://github.com/theodorblackbird/lina-speech/blob/main/InferenceLina.ipynb
https://github.com/theodorblackbird/lina-speech

Podcastfy
    https://github.com/souzatharsis/podcastfy/blob/main/podcastfy/tts/base.py
    https://github.com/souzatharsis/podcastfy/blob/main/podcastfy/text_to_speech.py
    https://github.com/souzatharsis/podcastfy/blob/main/podcastfy/content_generator.py

GLM-4-Voice
    https://github.com/THUDM/GLM-4-Voice/blob/main/README_en.md
    https://github.com/THUDM/GLM-4-Voice/tree/main

MoonShine
    https://huggingface.co/onnx-community/moonshine-base-ONNX
    https://huggingface.co/spaces/webml-community/moonshine-web
    https://github.com/huggingface/transformers.js-examples/tree/main/moonshine-web

Gemini
    https://ai.google.dev/gemini-api/docs#rest
    https://ai.google.dev/gemini-api/docs/models/gemini-v2

ElevenLabs
    https://github.com/elevenlabs/elevenlabs-examples/blob/main/examples/text-to-speech/python/text_to_speech_file.py
    https://elevenlabs.io/docs/api-reference/text-to-speech
    https://elevenlabs.io/docs/developer-guides/how-to-use-tts-with-streaming

Models
      https://huggingface.co/NexaAIDev/Qwen2-Audio-7B-GGUF

Merging Audio
    https://github.com/jiaaro/pydub



MaskGCT
    https://maskgct.github.io/#emotion-samples
    https://github.com/open-mmlab/Amphion/blob/main/models/tts/maskgct/README.md
    https://github.com/open-mmlab/Amphion/blob/main/models/tts/maskgct/maskgct_demo.ipynb
    https://github.com/open-mmlab/Amphion/blob/main/models/tts/maskgct/maskgct_inference.py
    https://huggingface.co/amphion/MaskGCT

Parler
    https://github.com/huggingface/parler-tts


YourTTS
    https://github.com/Edresson/YourTTS


TTS Pipeline
    https://www.astramind.ai/post/auralis

https://github.com/cpumaxx/sovits-ff-plugin



Train using: https://github.com/Mangio621/Mangio-RVC-Fork/releases,
import the .pth into https://huggingface.co/wok000/vcclient000/tree/main to convert your voice in near real time with about a .25s delay

https://www.hackster.io/lhl/voicechat2-local-ai-voice-chat-4c48f2

https://github.com/abus-aikorea/voice-pro

https://github.com/myshell-ai/MeloTTS
https://github.com/idiap/coqui-ai-TTS
https://docs.inferless.com/cookbook/serverless-customer-service-bot


https://huggingface.co/spaces/lamm-mit/PDF2Audio

https://huggingface.co/spaces/bencser/episodegen

https://github.com/Picovoice/speech-to-text-benchmark

https://huggingface.co/papers/2410.02678

https://github.com/livekit/agents
https://github.com/pipecat-ai/pipecat/tree/a367a038f1a3967292b5de5b43b8600a82a73fb6?tab=readme-ov-file

https://github.com/lamm-mit/PDF2Audio
https://github.com/Purfview/whisper-standalone-win
https://github.com/ictnlp/LLaMA-Omni
https://levelup.gitconnected.com/build-a-real-time-ai-voice-and-video-chat-app-with-function-calling-by-gemini-2-0-49599a48fbe9?gi=c894f6c092be
https://github.com/agituts/gemini-2-podcast
https://github.com/SWivid/F5-TTS


https://github.com/matatonic/openedai-speech

https://github.com/RVC-Boss/GPT-SoVITS
https://www.bilibili.com/video/BV11iiNegEGP/
https://github.com/RVC-Boss/GPT-SoVITS/wiki/GPT%E2%80%90SoVITS%E2%80%90v2%E2%80%90features-(%E6%96%B0%E7%89%B9%E6%80%A7)
https://rentry.org/GPT-SoVITS-guide
https://rentry.org/GPT-SoVITS-guide
It's just the 3 buttons (speech-to-text, ssl, semantics) and then training. 

The default training settings on the gradio UI are fine but I save epoch 12-16-24 on SoVITS for testing as that's the sweet spot range.

Next thing that matters a lot is the ref audio you pick, and you can also drop your entire dataset into the "multiple references to average tone" box, which can improve the voice

Only thing I changed was remove the space at the beginning of each lines in your list file

(Look at batch size/ list file)

And make sure you get the latest version https://github.com/RVC-Boss/GPT-SoVITS/releases

https://github.com/souzatharsis/podcastfy

https://github.com/THUDM/GLM-4-Voice/tree/main

https://huggingface.co/cydxg/glm-4-voice-9b-int4/blob/main/README_en.md

https://github.com/meta-llama/llama-recipes/tree/main/recipes%2Fquickstart%2FNotebookLlama


https://sakshi113.github.io/mmau_homepage/

https://github.com/fishaudio/fish-speech/tree/main
https://github.com/fishaudio/fish-speech/blob/main/Start_Agent.md
https://huggingface.co/fishaudio/fish-agent-v0.1-3b/tree/main

https://github.com/pixelpump/Ai-Interview-Assistant-Python
https://github.com/coqui-ai/TTS
https://github.com/Standard-Intelligence/hertz-dev
https://github.com/2noise/ChatTTS

https://github.com/edwko/OuteTTS
https://huggingface.co/OuteAI/OuteTTS-0.2-500M-GGUF
https://huggingface.co/NexaAIDev/Qwen2-Audio-7B-GGUF

https://www.twilio.com/en-us/blog/twilio-openai-realtime-api-launch-integration
https://github.com/huggingface/speech-to-speech
https://github.com/harvestingmoon/S2S
https://github.com/collabora/WhisperLive
https://github.com/JarodMica/audiobook_maker
https://github.com/myshell-ai/OpenVoice
https://github.com/JarodMica/GPT-SoVITS-Package
https://github.com/shagunmistry/NotebookLM_Alternative/tree/main/ai_helper
https://docs.cartesia.ai/get-started/make-an-api-request
https://github.com/JarodMica/open-neruosama
https://github.com/flatmax/speech-to-text
https://arxiv.org/abs/2412.18566
https://github.com/Rolandjg/skool4free


SoundStorm
    https://deepmind.google/discover/blog/pushing-the-frontiers-of-audio-generation/
    https://github.com/lucidrains/soundstorm-pytorch


Google
https://github.com/google-gemini/cookbook/tree/main/gemini-2
https://discuss.ai.google.dev/t/how-does-one-get-access-to-the-api-for-tts-features-of-gemini-2-0/53925/15
https://illuminate.google.com/home?pli=1
```
import asyncio
import base64
import json
import numpy as np
import os
import websockets
import wave
import contextlib
import pygame
from IPython.display import display, Markdown

# ANSI color codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"

voices = {"Puck", "Charon", "Kore", "Fenrir", "Aoede"}

# --- Configuration ---
MODEL = 'models/gemini-2.0-flash-exp'
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")
HOST = 'generativelanguage.googleapis.com'
URI = f'wss://{HOST}/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={GOOGLE_API_KEY}'

# Audio parameters
WAVE_CHANNELS = 1  # Mono audio
WAVE_RATE = 24000
WAVE_SAMPLE_WIDTH = 2


@contextlib.contextmanager
def wave_file(filename, channels=WAVE_CHANNELS, rate=WAVE_RATE, sample_width=WAVE_SAMPLE_WIDTH):
    """Context manager for creating and managing wave files."""
    try:
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            yield wf
    except wave.Error as e:
        print(f"{RED}Error opening wave file '{filename}': {e}{RESET}")
        raise


async def audio_playback_task(file_name, stop_event):
    """Plays audio using pygame until stopped."""
    print(f"{BLUE}Starting playback: {file_name}{RESET}")
    try:
        pygame.mixer.music.load(file_name)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy() and not stop_event.is_set():
            await asyncio.sleep(0.1)
    except pygame.error as e:
        print(f"{RED}Pygame error during playback: {e}{RESET}")
    except Exception as e:
        print(f"{RED}Unexpected error during playback: {e}{RESET}")
    finally:
        print(f"{BLUE}Playback complete: {file_name}{RESET}")


async def generate_audio(ws, text_input: str, voice_name="Kore") -> None:
    """
    Sends text input to the Gemini API, receives an audio response, saves it to a file, and plays it back.
    Relies on the server to maintain the session history.
    """
    pygame.mixer.init()  # Initialize pygame mixer

    msg = {
        "client_content": {
            "turns": [{"role": "user", "parts": [{"text": text_input}]}],
            "turn_complete": True,
        }
    }
    await ws.send(json.dumps(msg))

    responses = []
    async for raw_response in ws:
        response = json.loads(raw_response.decode())
        server_content = response.get("serverContent")
        if server_content is None:
            break

        model_turn = server_content.get("modelTurn")
        if model_turn:
            parts = model_turn.get("parts")
            if parts:
                for part in parts:
                    if "inlineData" in part and "data" in part["inlineData"]:
                        pcm_data = base64.b64decode(part["inlineData"]["data"])
                        responses.append(np.frombuffer(pcm_data, dtype=np.int16))

        turn_complete = server_content.get("turnComplete")
        if turn_complete:
            break

    if responses:
        display(Markdown(f"{YELLOW}**Response >**{RESET}"))
        audio_array = np.concatenate(responses)
        file_name = 'output.wav'
        with wave_file(file_name) as wf:
            wf.writeframes(audio_array.tobytes())
        stop_event = asyncio.Event()
        try:
            await audio_playback_task(file_name, stop_event)
        except Exception as e:
            print(f"{RED}Error during audio playback: {e}{RESET}")
    else:
        print(f"{YELLOW}No audio returned{RESET}")
    pygame.mixer.quit()  # clean up pygame mixer


async def main():
    print(f"{GREEN}Available voices: {', '.join(voices)}{RESET}")
    default_voice = "Kore"
    print(f"{GREEN}Default voice is set to: {default_voice}, you can change it in the code{RESET}")

    config = {
        "response_modalities": ["AUDIO"],
        "speech_config": {
            "voice_config": {
                "prebuilt_voice_config": {
                    "voice_name": default_voice  # Set voice
                }
            }
        }
    }

    async with websockets.connect(URI) as ws:

        async def setup(ws) -> None:
            await ws.send(
                json.dumps(
                    {
                        "setup": {
                            "model": MODEL,
                            "generation_config": config,
                        }
                    }
                )
            )

            raw_response = await ws.recv(decode=False)
            setup_response = json.loads(raw_response.decode("ascii"))
            print(f"{GREEN}Connected: {setup_response}{RESET}")

        await setup(ws)
        while True:
            text_prompt = input(f"{YELLOW}Enter your text (or type 'exit' to quit): {RESET}")
            if text_prompt.lower() == "exit":
                break

            try:
                await generate_audio(ws, text_prompt, default_voice)
            except Exception as e:
                print(f"{RED}An error occurred: {e}{RESET}")


if __name__ == "__main__":
    asyncio.run(main())
```


### GPT-SoVITS

- [GPT-SoVITS](f)
- [GPT-SoVITS-guide rentry.org](https://rentry.org/GPT-SoVITS-guide)
- Setup Guide: https://ai-hub-docs.vercel.app/tts/gpt-sovits/


GPT-SoviTTS
    https://levelup.gitconnected.com/great-api-design-comprehensive-guide-from-basics-to-best-practices-9b4e0b613a44?source=home---------56-1--------------------0fc48da7_5ce6_48ca_92d2_260680a20318-------3
    https://rentry.org/GPT-SoVITS-guide
    https://github.com/RVC-Boss/GPT-SoVITS
    https://github.com/cpumaxx/sovits-ff-plugin
    https://github.com/HanxSmile/Simplify-GPT-SoVITS
    https://github.com/lrxwisdom001/GPT-SoVITS-Novels/tree/main/voice_synthesis
    openneurosama - https://github.com/JarodMica/open-neruosama/blob/master/main.py


https://tts.x86.st/
Finetuning is very quick (about 5 minutes). Captioning of audio was automated with faster-whisper (it is required that the audio is captioned).
With the default batch size of 12, training takes 9.5~ GB.

Inference
    https://github.com/RVC-Boss/GPT-SoVITS/blob/main/GPT_SoVITS/inference_cli.py
    No WebUI Inference on Colab: https://colab.research.google.com/drive/1gC1lRxuOh4qW8Yz5TA10BEUPR28nJ3VR
    Training on Colab: https://colab.research.google.com/drive/1NQGKXYxJcJyTPnHsSyusTdD0l4IdMS37#scrollTo=nhyKqVwcPnvz
    No WebUI Training on Colab: https://colab.research.google.com/drive/1LmeM8yUyT9MTYF8OXc-NiBonvdh6hII6

Datasets
    https://ai-hub-docs.vercel.app/rvc/resources/datasets/
    https://mvsep.com/en

API
    https://github.com/cpumaxx/sovits-ff-plugin

Comfyui integration
    https://github.com/heshengtao/comfyui_LLM_party



- **101**
    - F
- **Setup**
    - F
- **Training**
    - F
- **Inference**
    - F
- **Fine-Tuning**
    - F

### Dataset Creation/Curation



