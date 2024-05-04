#!/usr/bin/env python3
import datetime
import unicodedata
import time
import os 
import subprocess
import json
import logging
import torch
import contextlib
import platform # used for checking OS version
import shutil # used for checking existence of ffmpeg
import ffmpeg # Used for issuing commands to underlying ffmpeg executable, pip package ffmpeg is from 2018
import yt_dlp
# idk....

# To Dos
# Implement more logging and remove print statements

os.environ['KMP_DUPLICATE_LIB_OK']='True'

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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variable to store the path of the last downloaded video
last_video_path = None




print(r"""_____  _          ________  _    _                                 
|_   _|| |        / /|  _  \| |  | | _                              
  | |  | |       / / | | | || |  | |(_)                             
  | |  | |      / /  | | | || |/\| |                                
  | |  | |____ / /   | |/ / \  /\  / _                              
  \_/  \_____//_/    |___/   \/  \/ (_)                             
                                                                    
                                                                    
 _                   _                                              
| |                 | |                                             
| |_   ___    ___   | |  ___   _ __    __ _                         
| __| / _ \  / _ \  | | / _ \ | '_ \  / _` |                        
| |_ | (_) || (_) | | || (_) || | | || (_| | _                      
 \__| \___/  \___/  |_| \___/ |_| |_| \__, |( )                     
                                       __/ ||/                      
                                      |___/                         
     _  _      _         _  _                      _          _     
    | |(_)    | |       ( )| |                    | |        | |    
  __| | _   __| | _ __  |/ | |_  __      __  __ _ | |_   ___ | |__  
 / _` || | / _` || '_ \    | __| \ \ /\ / / / _` || __| / __|| '_ \ 
| (_| || || (_| || | | |   | |_   \ V  V / | (_| || |_ | (__ | | | |
 \__,_||_| \__,_||_| |_|    \__|   \_/\_/   \__,_| \__| \___||_| |_|
""")


# Perform Platform Check
userOS = ""
def platform_check():
    if platform.system() == "Linux":
        print("Linux OS detected \n Running Linux appropriate commands")
        userOS = "Linux"
    elif platform.system() == "Windows":
        print("Windows OS detected \n Running Windows appropriate commands")
        userOS = "Windows"
    else:
        print("Other OS detected \n Maybe try running things manually?")
        exit()
#print(userOS)




# Check for NVIDIA GPU and CUDA availability
def cuda_check():
    global processing_choice
    try:
        nvidia_smi = subprocess.check_output("nvidia-smi", shell=True).decode()
        if "NVIDIA-SMI" in nvidia_smi:
            print("NVIDIA GPU with CUDA is available.")
            processing_choice = "gpu"  # Set processing_choice to gpu if NVIDIA GPU with CUDA is available
        else:
            print("NVIDIA GPU with CUDA is not available.\nYou either have an AMD GPU, or you're stuck with CPU only.")
            processing_choice = "cpu"  # Set processing_choice to cpu if NVIDIA GPU with CUDA is not available
    except subprocess.CalledProcessError:
        print("NVIDIA GPU with CUDA is not available.\nYou either have an AMD GPU, or you're stuck with CPU only.")
        processing_choice = "cpu"  # Set processing_choice to cpu if nvidia-smi command fails



# Ask user if they would like to use either their GPU or their CPU for transcription
def decide_cpugpu():
    global processing_choice
    processing_input = input("Would you like to use your GPU or CPU for transcription? (1)GPU/(2)CPU): ")
    if processing_choice == "gpu" and (processing_input.lower() == "gpu" or processing_input == "1"):
        print("You've chosen to use the GPU.")
        processing_choice = "gpu"
    elif processing_input.lower() == "cpu" or processing_input == "2":
        print("You've chosen to use the CPU.")
        processing_choice = "cpu"
    else:
        print("Invalid choice. Please select either GPU or CPU.")



# check for existence of ffmpeg
def check_ffmpeg():
    if shutil.which("ffmpeg"):
        pass
    else:
        print("ffmpeg is not installed.\n You can either install it manually, or through your package manager of choice.\n Windows users, builds are here: https://www.gyan.dev/ffmpeg/builds/")
        print("Script will continue, but is likely to break")
#print(processing_choice)



# Ask the user for the URL of the video to be downloaded. Alternatively, ask the user for the location of a local txt file to be read in and parsed to a list to be processed individually
def get_video_url():
    user_choice = input("Enter '1' to provide a video URL or '2' to specify a local text file path\n\t(the text file may contain both URLs and local file paths: ")
    if user_choice == '1':
        video_url = input("Enter the URL of the video to be downloaded: ")
        return video_url
    elif user_choice == '2':
        file_path = input("Enter the path of the local text file to be read and processed: ")
        return file_path
    else:
        print("Invalid choice. Please enter either '1' or '2'.")
        return None

# Perform processing of list to create array of URLs/Files to be downloaded & converted.
# Parse list for lines starting with 'http' -> Sort into urls_array[]
# Parse list for file paths (?) -> Sort into urls_local[]
# Download + convert items in urls_array[] list
# Convert (if necessary) items in urls_array[] list



def create_download_directory(title):
    base_dir = "Results"
    # Remove characters that are illegal in Windows filenames and normalize
    safe_title = normalize_title(title)
    session_path = os.path.join(base_dir, safe_title)
    if not os.path.exists(session_path):
        os.makedirs(session_path, exist_ok=True)
        print(f"Created directory: {session_path}")
    else:
        print(f"Directory already exists: {session_path}")
    return session_path

def normalize_title(title):
    # Normalize the string to 'NFKD' form and encode to 'ascii' ignoring non-ascii characters
    title = unicodedata.normalize('NFKD', title).encode('ascii', 'ignore').decode('ascii')
    # Remove or replace illegal characters
    title = title.replace('/', '_').replace('\\', '_').replace(':', '_').replace('"', '').replace('*', '').replace('?', '').replace('<', '').replace('>', '').replace('|', '')
    return title

def get_youtube(video_url):
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]',
        'noplaylist': True,
        'quiet': True,
        'extract_flat': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=False)
    return info_dict

def download_video(video_url, download_path, info_dict):
    title = normalize_title(info_dict['title'])
    file_path = os.path.join(download_path, f"{title}.m4a")
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]',
        'outtmpl': file_path,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    return file_path



# Convert video .m4a into .wav using ffmpeg
# ffmpeg -i "example.mp4" -ar 16000 -ac 1 -c:a pcm_s16le "output.wav"
# https://www.gyan.dev/ffmpeg/builds/



#os.system(r'.\Bin\ffmpeg.exe -ss 00:00:00 -i "{video_file_path}" -ar 16000 -ac 1 -c:a pcm_s16le "{out_path}"')
def convert_to_wav(video_file_path, offset=0):
    print("Starting conversion process of .m4a to .WAV\n\t You may have to hit 'ENTER' after a minute or two...")
    # Change the extension of the output file to .wav
    out_path = video_file_path.rsplit('.', 1)[0] + ".wav"

    try:
        if os.name == "nt":  # Check if the operating system is Windows
            command = [
                r".\Bin\ffmpeg.exe",   # Assuming the working directory is correctly set where .\Bin exists
                "-ss", "00:00:00",     # Start at the beginning of the video
                "-i", video_file_path,
                "-ar", "16000",        # Audio sample rate
                "-ac", "1",            # Number of audio channels
                "-c:a", "pcm_s16le",   # Audio codec
                out_path
            ]
            result = subprocess.run(command, text=True, capture_output=True)
            if result.returncode == 0:
                print("FFmpeg executed successfully")
                print("Output:", result.stdout)
            else:
                print("Error in running FFmpeg")
                print("Error Output:", result.stderr)
        elif os.name == "posix":  # Check if the operating system is Linux or macOS
            os.system(f'ffmpeg -ss 00:00:00 -i "{video_file_path}" -ar 16000 -ac 1 -c:a pcm_s16le "{out_path}"')
        else:
            print("Other OS detected. Not sure how you got here...")
        print("Conversion to WAV completed:", out_path)
    except Exception as e:
        raise RuntimeError("Error converting video file to WAV. An issue occurred with ffmpeg.")
    return out_path






# Transcribe .wav into .segments.json
def speech_to_text(video_file_path, selected_source_lang='en', whisper_model='small.en', vad_filter=False):
    print('loading faster_whisper model:', whisper_model)
    from faster_whisper import WhisperModel
    # printf(processing_choice)
    # 1 == GPU / 2 == CPU
    model = WhisperModel(whisper_model, device=processing_choice)
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
        options = dict(language=selected_source_lang, beam_size=5, best_of=5, vad_filter=vad_filter)
        transcribe_options = dict(task="transcribe", **options)
        # TODO: https://github.com/SYSTRAN/faster-whisper#vad-filter
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
            f.write(json.dumps(segments, indent=2))

    except Exception as e:
        raise RuntimeError("Error transcribing.")
    
    return segments



## Using Whisper.cpp
# Get-Whisper-GGML.ps1
# https://github.com/ggerganov/whisper.cpp/releases/latest



# TODO: https://huggingface.co/pyannote/speaker-diarization-3.1
# embedding_model = "pyannote/embedding", embedding_size=512
# embedding_model = "speechbrain/spkrec-ecapa-voxceleb", embedding_size=192
def speaker_diarize(video_file_path, segments, embedding_model = "pyannote/embedding", embedding_size=512, num_speakers=0):
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
        import tqdm

        _,file_ending = os.path.splitext(f'{video_file_path}')
        audio_file = video_file_path.replace(file_ending, ".wav")
        out_file = video_file_path.replace(file_ending, ".diarize.json")
        
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
            end = segment["end"]

            # enforce a minimum segment length
            if end-start < 0.3:
                padding = 0.3-(end-start)
                start -= padding/2
                end += padding/2
                print('Padded segment because it was too short:',segment)

            # Whisper overshoots the end timestamp in the last segment
            end = min(duration, end)
            # clip audio and embed
            clip = Segment(start, end)
            waveform, sample_rate = audio.crop(audio_file, clip)
            return embedding_model(waveform[None])

        embeddings = np.zeros(shape=(len(segments), embedding_size))
        for i, segment in enumerate(tqdm.tqdm(segments)):
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



def main(youtube_url: str, num_speakers: int = 2, whisper_model: str = "small.en", offset: int = 0, vad_filter : bool = False):
#    if user_choice == '2':
#        video_path = get_youtube(list_of_videos)
#FIXME

#    video_info = get_youtube(youtube_url)
#    download_path = create_download_directory(video_info['title'])
#    video_path = download_video(youtube_url, download_path)
#
    info_dict = get_youtube(youtube_url)
    download_path = create_download_directory(info_dict['title'])
    video_path = download_video(youtube_url, download_path, info_dict)
#
    audio_file = convert_to_wav(video_path, offset)
    segments = speech_to_text(video_path, whisper_model=whisper_model, vad_filter=vad_filter)
#    df_results, save_path = speaker_diarize(video_path, segments, num_speakers=num_speakers)
#    print("diarize complete:", save_path)
    print("Transcription complete:", audio_file)
#FIXME


# Main Function - Execution starts here
if __name__ == "__main__":
    import fire
    platform_check()
    cuda_check()
    decide_cpugpu()
    check_ffmpeg()
    fire.Fire(main)