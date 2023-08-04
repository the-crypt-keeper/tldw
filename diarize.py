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

whisper_models = ["small", "medium", "small.en","medium.en"]
source_languages = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "ko": "Korean",
    "fr": "French"
}

source_language_list = [key[0] for key in source_languages.items()]

embedding_model = PretrainedSpeakerEmbedding( 
    #"speechbrain/spkrec-ecapa-voxceleb",
    "pyannote/embedding",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def convert_time(secs):
    return datetime.timedelta(seconds=round(secs))

def get_youtube(video_url):
    # yt = YouTube(video_url)
    # abs_video_path = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download()
    return "lex.m4a"
    
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
    model = WhisperModel(whisper_model, device="cuda", compute_type="float16")
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

        embeddings = np.zeros(shape=(len(segments), 512))
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
selected_whisper_model = gr.Dropdown(choices=whisper_models, type="value", value="small.en", label="Selected Whisper model", interactive=True)
number_speakers = gr.Number(precision=0, value=0, label="Input number of speakers for better results. If value=0, model will automatic find the best number of speakers", interactive=True)
system_info = gr.Markdown(f"*Memory: {memory.total / (1024 * 1024 * 1024):.2f}GB, used: {memory.percent}%, available: {memory.available / (1024 * 1024 * 1024):.2f}GB*")
download_transcript = gr.File(label="Download transcript")
transcription_df = gr.DataFrame(value=df_init,label="Transcription dataframe", row_count=(0, "dynamic"), max_rows = 10, wrap=True, overflow_row_behaviour='paginate')
title = "Whisper speaker diarization"
demo = gr.Blocks(title=title)
demo.encrypt = False


with demo:
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

demo.launch(debug=True, server_port=8888, share=True)