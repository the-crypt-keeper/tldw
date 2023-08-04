# import whisper
from faster_whisper import WhisperModel
import datetime
import subprocess
import gradio as gr
from pathlib import Path
import pandas as pd
import re
import time
import os 
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from pytube import YouTube
import yt_dlp
import torch
import pyannote.audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment

from gpuinfo import GPUInfo

import wave
import contextlib
from transformers import pipeline
import psutil

whisper_models = ["tiny", "base", "small", "medium", "large-v1", "large-v2"]
source_languages = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "ko": "Korean",
    "fr": "French",
    "ja": "Japanese",
    "pt": "Portuguese",
    "tr": "Turkish",
    "pl": "Polish",
    "ca": "Catalan",
    "nl": "Dutch",
    "ar": "Arabic",
    "sv": "Swedish",
    "it": "Italian",
    "id": "Indonesian",
    "hi": "Hindi",
    "fi": "Finnish",
    "vi": "Vietnamese",
    "he": "Hebrew",
    "uk": "Ukrainian",
    "el": "Greek",
    "ms": "Malay",
    "cs": "Czech",
    "ro": "Romanian",
    "da": "Danish",
    "hu": "Hungarian",
    "ta": "Tamil",
    "no": "Norwegian",
    "th": "Thai",
    "ur": "Urdu",
    "hr": "Croatian",
    "bg": "Bulgarian",
    "lt": "Lithuanian",
    "la": "Latin",
    "mi": "Maori",
    "ml": "Malayalam",
    "cy": "Welsh",
    "sk": "Slovak",
    "te": "Telugu",
    "fa": "Persian",
    "lv": "Latvian",
    "bn": "Bengali",
    "sr": "Serbian",
    "az": "Azerbaijani",
    "sl": "Slovenian",
    "kn": "Kannada",
    "et": "Estonian",
    "mk": "Macedonian",
    "br": "Breton",
    "eu": "Basque",
    "is": "Icelandic",
    "hy": "Armenian",
    "ne": "Nepali",
    "mn": "Mongolian",
    "bs": "Bosnian",
    "kk": "Kazakh",
    "sq": "Albanian",
    "sw": "Swahili",
    "gl": "Galician",
    "mr": "Marathi",
    "pa": "Punjabi",
    "si": "Sinhala",
    "km": "Khmer",
    "sn": "Shona",
    "yo": "Yoruba",
    "so": "Somali",
    "af": "Afrikaans",
    "oc": "Occitan",
    "ka": "Georgian",
    "be": "Belarusian",
    "tg": "Tajik",
    "sd": "Sindhi",
    "gu": "Gujarati",
    "am": "Amharic",
    "yi": "Yiddish",
    "lo": "Lao",
    "uz": "Uzbek",
    "fo": "Faroese",
    "ht": "Haitian creole",
    "ps": "Pashto",
    "tk": "Turkmen",
    "nn": "Nynorsk",
    "mt": "Maltese",
    "sa": "Sanskrit",
    "lb": "Luxembourgish",
    "my": "Myanmar",
    "bo": "Tibetan",
    "tl": "Tagalog",
    "mg": "Malagasy",
    "as": "Assamese",
    "tt": "Tatar",
    "haw": "Hawaiian",
    "ln": "Lingala",
    "ha": "Hausa",
    "ba": "Bashkir",
    "jw": "Javanese",
    "su": "Sundanese",
}

source_language_list = [key[0] for key in source_languages.items()]

MODEL_NAME = "vumichien/whisper-medium-jp"
lang = "ja"

device = 0 if torch.cuda.is_available() else "cpu"
pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
)
os.makedirs('output', exist_ok=True)
pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language=lang, task="transcribe")

embedding_model = PretrainedSpeakerEmbedding( 
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def transcribe(microphone, file_upload):
    warn_output = ""
    if (microphone is not None) and (file_upload is not None):
        warn_output = (
            "WARNING: You've uploaded an audio file and used the microphone. "
            "The recorded file from the microphone will be used and the uploaded audio will be discarded.\n"
        )

    elif (microphone is None) and (file_upload is None):
        return "ERROR: You have to either use the microphone or upload an audio file"

    file = microphone if microphone is not None else file_upload

    text = pipe(file)["text"]

    return warn_output + text

def _return_yt_html_embed(yt_url):
    video_id = yt_url.split("?v=")[-1]
    HTML_str = (
        f'<center> <iframe width="500" height="320" src="https://www.youtube.com/embed/{video_id}"> </iframe>'
        " </center>"
    )
    return HTML_str

def yt_transcribe(yt_url):
    # yt = YouTube(yt_url)
    # html_embed_str = _return_yt_html_embed(yt_url)
    # stream = yt.streams.filter(only_audio=True)[0]
    # stream.download(filename="audio.mp3")

    ydl_opts = {
        'format': 'bestvideo*+bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl':'audio.%(ext)s',
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([yt_url])
        
    text = pipe("audio.mp3")["text"]
    return html_embed_str, text

def convert_time(secs):
    return datetime.timedelta(seconds=round(secs))

def get_youtube(video_url):
    # yt = YouTube(video_url)
    # abs_video_path = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download()
    return "elon.m4a"
    
    ydl_opts = {
      'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        abs_video_path = ydl.prepare_filename(info)
        ydl.process_info(info) 
        
    print("Success download video")
    print(abs_video_path)
    return abs_video_path

def speech_to_text(video_file_path, selected_source_lang, whisper_model, num_speakers):
    """
    # Transcribe youtube link using OpenAI Whisper
    1. Using Open AI's Whisper model to seperate audio into segments and generate transcripts.
    2. Generating speaker embeddings for each segments.
    3. Applying agglomerative clustering on the embeddings to identify the speaker for each segment.
    
    Speech Recognition is based on models from OpenAI Whisper https://github.com/openai/whisper
    Speaker diarization model and pipeline from by https://github.com/pyannote/pyannote-audio
    """
    
    # model = whisper.load_model(whisper_model)
    print('loading whisper model..')
    model = WhisperModel(whisper_model, device="cuda", compute_type="int8_float16")
    #model = WhisperModel(whisper_model, compute_type="int8")
    time_start = time.time()
    if(video_file_path == None):
        raise ValueError("Error no video input")
    print(video_file_path)

    try:
        # Read and convert youtube video
        _,file_ending = os.path.splitext(f'{video_file_path}')
        print(f'file enging is {file_ending}')
        audio_file = video_file_path.replace(file_ending, ".wav")
        print("starting conversion to wav")
        os.system(f'ffmpeg -i "{video_file_path}" -ar 16000 -ac 1 -c:a pcm_s16le "{audio_file}"')
        
        # Get duration
        with contextlib.closing(wave.open(audio_file,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
        print(f"conversion to wav ready, duration of audio file: {duration}")

        # Transcribe audio
        options = dict(language=selected_source_lang, beam_size=5, best_of=5)
        transcribe_options = dict(task="transcribe", **options)
        segments_raw, info = model.transcribe(audio_file, **transcribe_options)

        # Convert back to original openai format
        segments = []
        i = 0
        for segment_chunk in segments_raw:
            chunk = {}
            chunk["start"] = segment_chunk.start
            chunk["end"] = segment_chunk.end
            chunk["text"] = segment_chunk.text
            print(chunk)
            segments.append(chunk)
            i += 1
        print("transcribe audio done with fast whisper")
    except Exception as e:
        raise RuntimeError("Error converting video to audio")

    try:
        # Create embedding
        def segment_embedding(segment):
            audio = Audio()
            start = segment["start"]
            # Whisper overshoots the end timestamp in the last segment
            end = min(duration, segment["end"])
            clip = Segment(start, end)
            waveform, sample_rate = audio.crop(audio_file, clip)
            return embedding_model(waveform[None])

        embeddings = np.zeros(shape=(len(segments), 192))
        for i, segment in enumerate(segments):
            embeddings[i] = segment_embedding(segment)
        embeddings = np.nan_to_num(embeddings)
        print(f'Embedding shape: {embeddings.shape}')

        if num_speakers == 0:
        # Find the best number of speakers
            score_num_speakers = {}
    
            for num_speakers in range(2, 10+1):
                clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
                score = silhouette_score(embeddings, clustering.labels_, metric='euclidean')
                score_num_speakers[num_speakers] = score
            best_num_speaker = max(score_num_speakers, key=lambda x:score_num_speakers[x])
            print(f"The best number of speakers: {best_num_speaker} with {score_num_speakers[best_num_speaker]} score")
        else:
            best_num_speaker = num_speakers
            
        # Assign speaker label   
        clustering = AgglomerativeClustering(best_num_speaker).fit(embeddings)
        labels = clustering.labels_
        for i in range(len(segments)):
            segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

        # Make output
        objects = {
            'Start' : [],
            'End': [],
            'Speaker': [],
            'Text': []
        }
        text = ''
        for (i, segment) in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                objects['Start'].append(str(convert_time(segment["start"])))
                objects['Speaker'].append(segment["speaker"])
                if i != 0:
                    objects['End'].append(str(convert_time(segments[i - 1]["end"])))
                    objects['Text'].append(text)
                    text = ''
            text += segment["text"] + ' '
        objects['End'].append(str(convert_time(segments[i - 1]["end"])))
        objects['Text'].append(text)
        
        time_end = time.time()
        time_diff = time_end - time_start
        memory = psutil.virtual_memory()
        gpu_utilization, gpu_memory = GPUInfo.gpu_usage()
        gpu_utilization = gpu_utilization[0] if len(gpu_utilization) > 0 else 0
        gpu_memory = gpu_memory[0] if len(gpu_memory) > 0 else 0
        system_info = f"""
        *Memory: {memory.total / (1024 * 1024 * 1024):.2f}GB, used: {memory.percent}%, available: {memory.available / (1024 * 1024 * 1024):.2f}GB.* 
        *Processing time: {time_diff:.5} seconds.*
        *GPU Utilization: {gpu_utilization}%, GPU Memory: {gpu_memory}MiB.*
        """
        save_path = "output/transcript_result.csv"
        df_results = pd.DataFrame(objects)
        df_results.to_csv(save_path)
        return df_results, system_info, save_path
    
    except Exception as e:
        raise RuntimeError("Error Running inference with local model", e)


# ---- Gradio Layout -----
# Inspiration from https://huggingface.co/spaces/RASMUS/Whisper-youtube-crosslingual-subtitles
video_in = gr.Video(label="Video file", mirror_webcam=False)
youtube_url_in = gr.Textbox(label="Youtube url", lines=1, interactive=True)
df_init = pd.DataFrame(columns=['Start', 'End', 'Speaker', 'Text'])
memory = psutil.virtual_memory()
selected_source_lang = gr.Dropdown(choices=source_language_list, type="value", value="en", label="Spoken language in video", interactive=True)
selected_whisper_model = gr.Dropdown(choices=whisper_models, type="value", value="base", label="Selected Whisper model", interactive=True)
number_speakers = gr.Number(precision=0, value=0, label="Input number of speakers for better results. If value=0, model will automatic find the best number of speakers", interactive=True)
system_info = gr.Markdown(f"*Memory: {memory.total / (1024 * 1024 * 1024):.2f}GB, used: {memory.percent}%, available: {memory.available / (1024 * 1024 * 1024):.2f}GB*")
download_transcript = gr.File(label="Download transcript")
transcription_df = gr.DataFrame(value=df_init,label="Transcription dataframe", row_count=(0, "dynamic"), max_rows = 10, wrap=True, overflow_row_behaviour='paginate')
title = "Whisper speaker diarization"
demo = gr.Blocks(title=title)
demo.encrypt = False


with demo:
    with gr.Tab("Whisper speaker diarization"):
        gr.Markdown('''
            <div>
            <h1 style='text-align: center'>Whisper speaker diarization</h1>
            This space uses Whisper models from <a href='https://github.com/openai/whisper' target='_blank'><b>OpenAI</b></a> with <a href='https://github.com/guillaumekln/faster-whisper' target='_blank'><b>CTranslate2</b></a> which is a fast inference engine for Transformer models to recognize the speech (4 times faster than original openai model with same accuracy)
            and ECAPA-TDNN model from <a href='https://github.com/speechbrain/speechbrain' target='_blank'><b>SpeechBrain</b></a> to encode and clasify speakers
            </div>
        ''')

        with gr.Row():
            gr.Markdown('''
            ### Transcribe youtube link using OpenAI Whisper
            ##### 1. Using Open AI's Whisper model to seperate audio into segments and generate transcripts.
            ##### 2. Generating speaker embeddings for each segments.
            ##### 3. Applying agglomerative clustering on the embeddings to identify the speaker for each segment.
            ''')
            
        with gr.Row():         
            gr.Markdown('''
                ### You can test by following examples:
                ''')
        examples = gr.Examples(examples=
                [ "https://www.youtube.com/watch?v=j7BfEzAFuYc&t=32s", 
                  "https://www.youtube.com/watch?v=-UX0X45sYe4", 
                  "https://www.youtube.com/watch?v=7minSgqi-Gw"],
              label="Examples", inputs=[youtube_url_in])
              

        with gr.Row():
            with gr.Column():
                youtube_url_in.render()
                download_youtube_btn = gr.Button("Download Youtube video")
                download_youtube_btn.click(get_youtube, [youtube_url_in], [video_in])
                print(video_in)
                

        with gr.Row():
            with gr.Column():
                video_in.render()
                with gr.Column():
                    gr.Markdown('''
                    ##### Here you can start the transcription process.
                    ##### Please select the source language for transcription.
                    ##### You can select a range of assumed numbers of speakers.
                    ''')
                selected_source_lang.render()
                selected_whisper_model.render()
                number_speakers.render()
                transcribe_btn = gr.Button("Transcribe audio and diarization")
                transcribe_btn.click(speech_to_text, 
                                     [video_in, selected_source_lang, selected_whisper_model, number_speakers], 
                                     [transcription_df, system_info, download_transcript]
                                    )
                
        with gr.Row():
            gr.Markdown('''
            ##### Here you will get transcription  output
            ##### ''')
            

        with gr.Row():
            with gr.Column():
                download_transcript.render()
                transcription_df.render()
                system_info.render()
                gr.Markdown('''<center><img src='https://visitor-badge.glitch.me/badge?page_id=WhisperDiarizationSpeakers' alt='visitor badge'><a href="https://opensource.org/licenses/Apache-2.0"><img src='https://img.shields.io/badge/License-Apache_2.0-blue.svg' alt='License: Apache 2.0'></center>''')

                
    
    with gr.Tab("Whisper Transcribe Japanese Audio"):
        gr.Markdown(f'''
              <div>
              <h1 style='text-align: center'>Whisper Transcribe Japanese Audio</h1>
              </div>
              Transcribe long-form microphone or audio inputs with the click of a button! The fine-tuned
              checkpoint <a href='https://huggingface.co/{MODEL_NAME}' target='_blank'><b>{MODEL_NAME}</b></a> to transcribe audio files of arbitrary length.
          ''')
        microphone = gr.inputs.Audio(source="microphone", type="filepath", optional=True)
        upload = gr.inputs.Audio(source="upload", type="filepath", optional=True)
        transcribe_btn = gr.Button("Transcribe Audio")
        text_output = gr.Textbox()
        with gr.Row():         
            gr.Markdown('''
                ### You can test by following examples:
                ''')
        examples = gr.Examples(examples=
              [ "sample1.wav", 
                "sample2.wav", 
                ],
              label="Examples", inputs=[upload])
        transcribe_btn.click(transcribe, [microphone, upload], outputs=text_output)
    
    with gr.Tab("Whisper Transcribe Japanese YouTube"):
        gr.Markdown(f'''
              <div>
              <h1 style='text-align: center'>Whisper Transcribe Japanese YouTube</h1>
              </div>
                Transcribe long-form YouTube videos with the click of a button! The fine-tuned checkpoint:
                <a href='https://huggingface.co/{MODEL_NAME}' target='_blank'><b>{MODEL_NAME}</b></a> to transcribe audio files of arbitrary length.
            ''')
        youtube_link = gr.Textbox(label="Youtube url", lines=1, interactive=True)
        yt_transcribe_btn = gr.Button("Transcribe YouTube")
        text_output2 = gr.Textbox()
        html_output = gr.Markdown()
        yt_transcribe_btn.click(yt_transcribe, [youtube_link], outputs=[html_output, text_output2])

demo.launch(debug=True, server_port=8888, share=True)