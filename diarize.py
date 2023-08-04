#!/usr/bin/env python3
import datetime
import time
import os 
import json
import torch
import contextlib

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

# Download video .m4a and info.json
def get_youtube(video_url):
    import yt_dlp

    ydl_opts = { 'format': 'bestaudio[ext=m4a]' }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        abs_video_path = ydl.prepare_filename(info)
        with open(abs_video_path.replace('m4a','info.json'), 'w') as outfile:
            json.dump(info, outfile, indent=2)
        ydl.process_info(info) 
        
    print("Success download",video_url,"to", abs_video_path)
    return abs_video_path

# Convert video .m4a into .wav
def convert_to_wav(video_file_path):
    out_path = video_file_path.replace("m4a","wav")
    if os.path.exists(out_path):
        print("wav file already exists:", out_path)
        return out_path

    try:
        print("starting conversion to wav")
        os.system(f'ffmpeg -i "{video_file_path}" -ar 16000 -ac 1 -c:a pcm_s16le "{out_path}"')
        print("conversion to wav ready:", out_path)
    except Exception as e:
        raise RuntimeError("Error converting.")
    
    return out_path

# Transcribe .wav into .segments.json
def speech_to_text(video_file_path, selected_source_lang = 'en', whisper_model = 'small.en'):   
    print('loading faster_whisper model:', whisper_model)
    from faster_whisper import WhisperModel
    model = WhisperModel(whisper_model, device="cuda", compute_type="float16")
    time_start = time.time()
    if(video_file_path == None):
        raise ValueError("Error no video input")
    print(video_file_path)

    try:
        # Read and convert youtube video
        _,file_ending = os.path.splitext(f'{video_file_path}')
        audio_file = video_file_path.replace(file_ending, ".wav")
        out_file = video_file_path.replace(file_ending, ".segments.json")
        if os.path.exists(out_file):
            print("segments file already exists:", out_file)
            with open(out_file) as f:
                segments = json.load(f)
            return segments
        
        # Transcribe audio
        print('starting transcription...')
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

        with open(out_file,'w') as f:
            out_file.write(json.dumps(segments, indent=2))

    except Exception as e:
        raise RuntimeError("Error transcribing.")
    
    return segments

# embedding_model: "pyannote/embedding", embedding_size: 512
def speaker_diarize(video_file_path, segments, embedding_model = "speechbrain/spkrec-ecapa-voxceleb", embedding_size=192, num_speakers=0):
    """
    1. Generating speaker embeddings for each segments.
    2. Applying agglomerative clustering on the embeddings to identify the speaker for each segment.
    """
    try:
        # Load embedding model
        from pyannote.audio import Audio
        from pyannote.core import Segment

        from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
        embedding_model = PretrainedSpeakerEmbedding( embedding_model, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        import numpy as np
        import pandas as pd
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score

        _,file_ending = os.path.splitext(f'{video_file_path}')
        audio_file = video_file_path.replace(file_ending, ".wav")
        out_file = video_file_path.replace(file_ending, ".diarize.json")
        if os.path.exists(out_file):
            print("segments file already exists:", out_file)
            with open(out_file) as f:
                segments = json.load(f)
            return segments
        
        # Get duration
        import wave
        with contextlib.closing(wave.open(audio_file,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
        print(f"duration of audio file: {duration}")

        # Create embedding
        def segment_embedding(segment):
            audio = Audio()
            start = segment["start"]
            # Whisper overshoots the end timestamp in the last segment
            end = min(duration, segment["end"])
            clip = Segment(start, end)
            waveform, sample_rate = audio.crop(audio_file, clip)
            return embedding_model(waveform[None])

        embeddings = np.zeros(shape=(len(segments), embedding_size))
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

        with open(out_file,'w') as f:
            f.write(json.dumps(segments, indent=2))

        # Make CSV output
        def convert_time(secs):
            return datetime.timedelta(seconds=round(secs))
        
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
        
        save_path = video_file_path.replace(file_ending, ".csv")
        df_results = pd.DataFrame(objects)
        df_results.to_csv(save_path)
        return df_results, save_path
    
    except Exception as e:
        raise RuntimeError("Error Running inference with local model", e)

def main(youtube_url: str, num_speakers: int = 2, whisper_model: str = "small.en"):
    video_path = get_youtube(youtube_url)
    convert_to_wav(video_path)
    segments = speech_to_text(video_path, whisper_model=whisper_model)
    df_results, save_path = speaker_diarize(video_path, segments, num_speakers=num_speakers)
    print("diarize complete:", save_path)

if __name__ == "__main__":
    import fire
    fire.Fire(main)