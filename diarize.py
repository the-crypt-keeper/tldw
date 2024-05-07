#!/usr/bin/env python3
import argparse, configparser, datetime, json, logging, os, platform, requests, shutil, subprocess, sys, time, unicodedata
import zipfile
from datetime import datetime
import contextlib
import ffmpeg # Used for issuing commands to underlying ffmpeg executable, pip package ffmpeg is from 2018
import torch
import yt_dlp


#######
# Function Sections
#
# System Checks
# Processing Paths and local file handling
# Video Download/Handling
# Audio Transcription
# Diarization
# Summarizers
# Main
#
#######

# To Do
# Offline diarization - https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/community/offline_usage_speaker_diarization.ipynb


####
#
#       TL/DW: Too Long Didn't Watch
#
#  Project originally created by https://github.com/the-crypt-keeper
#  Modifications made by https://github.com/rmusser01
#  All credit to the original authors, I've just glued shit together.
#
#
# Usage:
#          Transcribe a single URL: 
#                python diarize.py https://example.com/video.mp4
#
#          Transcribe a single URL and have the resulting transcription summarized: 
#                python diarize.py https://example.com/video.mp4
#
#          Transcribe a list of files:
#               python diarize.py ./path/to/your/text_file.txt
#
#          Transcribe a local file:
#               python diarize.py /path/to/your/localfile.mp4
#
#          Transcribe a local file and have it summarized:
#               python diarize.py ./input.mp4 --api_name openai --api_key <your_openai_api_key>
#
#          Transcribe a list of files and have them all summarized:
#               python diarize.py path_to_your_text_file.txt --api_name <openai> --api_key <your_openai_api_key>
#
###


#######################
# Config loading
#

# Read configuration from file
config = configparser.ConfigParser()
config.read('config.txt')

# API Keys
cohere_api_key = config.get('API', 'cohere_api_key', fallback=None)
anthropic_api_key = config.get('API', 'anthropic_api_key', fallback=None)
openai_api_key = config.get('API', 'openai_api_key', fallback=None)
llama_api_key = config.get('API', 'llama_api_key', fallback=None)

# Models
anthropic_model = config.get('API', 'anthropic_model', fallback='claude-3-sonnet-20240229')
cohere_model = config.get('API', 'cohere_model', fallback='command-r-plus')
openai_model = config.get('API', 'openai_model', fallback='gpt-4-turbo')

# Local-Models
llama_ip = config.get('API', 'llama_api_IP', fallback='127.0.0.1:8080/v1/chat/completions')

# Retrieve output paths from the configuration file
output_path = config.get('Paths', 'output_path', fallback='Results')

# Retrieve processing choice from the configuration file
processing_choice = config.get('Processing', 'processing_choice', fallback='cpu')

#
#
#######################

# Dirty hack - sue me.
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

####################################################################################################################################
# System Checks
# 
# 

# Perform Platform Check
userOS = ""
def platform_check():
    global userOS
    if platform.system() == "Linux":
        print("Linux OS detected \n Running Linux appropriate commands")
        userOS = "Linux"
    elif platform.system() == "Windows":
        print("Windows OS detected \n Running Windows appropriate commands")
        userOS = "Windows"
    else:
        print("Other OS detected \n Maybe try running things manually?")
        exit()



# Check for NVIDIA GPU and CUDA availability
def cuda_check():
    global processing_choice
    try:
        nvidia_smi = subprocess.check_output("nvidia-smi", shell=True).decode()
        if "NVIDIA-SMI" in nvidia_smi:
            print("NVIDIA GPU with CUDA is available.")
            processing_choice = "cuda"  # Set processing_choice to gpu if NVIDIA GPU with CUDA is available
        else:
            print("NVIDIA GPU with CUDA is not available.\nYou either have an AMD GPU, or you're stuck with CPU only.")
            processing_choice = "cpu"  # Set processing_choice to cpu if NVIDIA GPU with CUDA is not available
    except subprocess.CalledProcessError:
        print("NVIDIA GPU with CUDA is not available.\nYou either have an AMD GPU, or you're stuck with CPU only.")
        processing_choice = "cpu"  # Set processing_choice to cpu if nvidia-smi command fails



# Ask user if they would like to use either their GPU or their CPU for transcription
def decide_cpugpu():
    global processing_choice
    processing_input = input("Would you like to use your GPU or CPU for transcription? (1/cuda)GPU/(2/cpu)CPU): ")
    if processing_choice == "cuda" and (processing_input.lower() == "cuda" or processing_input == "1"):
        print("You've chosen to use the GPU.")
        logging.debug("GPU is being used for processing")
        processing_choice = "cuda"
    elif processing_input.lower() == "cpu" or processing_input == "2":
        print("You've chosen to use the CPU.")
        logging.debug("CPU is being used for processing")
        processing_choice = "cpu"
    else:
        print("Invalid choice. Please select either GPU or CPU.")



# check for existence of ffmpeg
def check_ffmpeg():
    if shutil.which("ffmpeg"):
        logging.debug("ffmpeg found installed on the local system, or at least in the local PATH")
        pass
    else:
        logging.debug("ffmpeg not installed on the local system/in local PATH")
        print("ffmpeg is not installed.\n\n You can either install it manually, or through your package manager of choice.\n Windows users, builds are here: https://www.gyan.dev/ffmpeg/builds/")
    if userOS == "Windows":
        download_ffmpeg()
    elif userOS == "Linux":
        print("You should install ffmpeg using your platform's appropriate package manager, 'apt install ffmpeg','dnf install ffmpeg' or 'pacman', etc.")
    else:
        logging.debug("running an unsupported OS")
        print("You're running an unspported/Un-tested OS")
        exit_script = input("Let's exit the script, unless you're feeling lucky? (y/n)")
        if exit_script == "y" or "yes" or "1":
            exit()



# Download ffmpeg
def download_ffmpeg():
    user_choice = input("Do you want to download ffmpeg? (y)Yes/(n)No: ")
    if user_choice.lower() == 'yes' or 'y' or '1':
        print("Downloading ffmpeg")
        url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
        response = requests.get(url)
        
        if response.status_code == 200:
            print("Saving ffmpeg zip file")
            logging.debug("Saving ffmpeg zip file")
            zip_path = "ffmpeg-release-essentials.zip"
            with open(zip_path, 'wb') as file:
                file.write(response.content)
            
            logging.debug("Extracting the 'ffmpeg.exe' file from the zip")
            print("Extracting ffmpeg.exe from zip file to '/Bin' folder")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                ffmpeg_path = "ffmpeg-7.0-essentials_build/bin/ffmpeg.exe"
                
                logging.debug("checking if the './Bin' folder exists, creating if not")
                bin_folder = "Bin"
                if not os.path.exists(bin_folder):
                    logging.debug("Creating a folder for './Bin', it didn't previously exist")
                    os.makedirs(bin_folder)
                
                logging.debug("Extracting 'ffmpeg.exe' to the './Bin' folder")
                zip_ref.extract(ffmpeg_path, path=bin_folder)
                
                logging.debug("Moving 'ffmpeg.exe' to the './Bin' folder")
                src_path = os.path.join(bin_folder, ffmpeg_path)
                dst_path = os.path.join(bin_folder, "ffmpeg.exe")
                shutil.move(src_path, dst_path)
            
            logging.debug("Removing ffmpeg zip file")
            print("Deleting zip file (we've already extracted ffmpeg.exe, no worries)")
            os.remove(zip_path)

            logging.debug("ffmpeg.exe has been downloaded and extracted to the './Bin' folder.")
            print("ffmpeg.exe has been successfully downloaded and extracted to the './Bin' folder.")
        else:
            logging.error("Failed to download the zip file.")
            print("Failed to download the zip file.")
    else:
        logging.debug("User chose to not download ffmpeg")
        print("ffmpeg will not be downloaded.")

# 
# 
####################################################################################################################################







####################################################################################################################################
# Processing Paths and local file handling
# 
#

def read_paths_from_file(file_path):
    """ Reads a file containing URLs or local file paths and returns them as a list. """
    paths = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not os.path.exists(os.path.join('Results', normalize_title(line.split('/')[-1].split('.')[0]) + '.json')):
                logging.debug("line successfully imported from file and added to list to be transcribed")
                paths.append(line)
    return paths



def process_path(path):
    """ Decides whether the path is a URL or a local file and processes accordingly. """
    if path.startswith('http'):
        logging.debug("file is a URL")
        return get_youtube(path)  # For YouTube URLs, modify to download and extract info
    elif os.path.exists(path):
        logging.debug("File is a path")
        return process_local_file(path)  # For local files, define a function to handle them
    else:
        logging.error(f"Path does not exist: {path}")
        return None



# FIXME
def process_local_file(file_path):
    logging.info(f"Processing local file: {file_path}")
    title = normalize_title(os.path.splitext(os.path.basename(file_path))[0])
    info_dict = {'title': title}
    logging.debug(f"Creating {title} directory...")
    download_path = create_download_directory(title)
    logging.debug(f"Converting '{title}' to an audio file (wav).")
    audio_file = convert_to_wav(file_path)  # Assumes input files are videos needing audio extraction
    logging.debug(f"'{title}' succesfully converted to an audio file (wav).")   
    return download_path, info_dict, audio_file
# 
#
####################################################################################################################################






####################################################################################################################################
# Video Download/Handling
#

def create_download_directory(title):
    base_dir = "Results"
    # Remove characters that are illegal in Windows filenames and normalize
    safe_title = normalize_title(title)
    logging.debug(f"{title} successfully normalized")
    session_path = os.path.join(base_dir, safe_title)
    if not os.path.exists(session_path):
        os.makedirs(session_path, exist_ok=True)
        logging.debug(f"Created directory for downloaded video: {session_path}")
    else:
        logging.debug(f"Directory already exists for downloaded video: {session_path}")
    return session_path



def normalize_title(title):
    # Normalize the string to 'NFKD' form and encode to 'ascii' ignoring non-ascii characters
    title = unicodedata.normalize('NFKD', title).encode('ascii', 'ignore').decode('ascii')
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
        logging.debug("About to extract youtube info")
        info_dict = ydl.extract_info(video_url, download=False)
        logging.debug("Youtube info successfully extracted")
    return info_dict



def download_video(video_url, download_path, info_dict):
    logging.debug("About to normalize downloaded video title")
    title = normalize_title(info_dict['title'])
    file_path = os.path.join(download_path, f"{title}.m4a")
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]',
        'outtmpl': file_path,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        logging.debug("About to download video with youtube-dl")
        ydl.download([video_url])
        logging.debug("Video successfully downloaded with youtube-dl")
    return file_path
#
#
####################################################################################################################################






####################################################################################################################################
# Audio Transcription
#
# Convert video .m4a into .wav using ffmpeg
#   ffmpeg -i "example.mp4" -ar 16000 -ac 1 -c:a pcm_s16le "output.wav"
#       https://www.gyan.dev/ffmpeg/builds/
#

#os.system(r'.\Bin\ffmpeg.exe -ss 00:00:00 -i "{video_file_path}" -ar 16000 -ac 1 -c:a pcm_s16le "{out_path}"')
def convert_to_wav(video_file_path, offset=0):
    print("Starting conversion process of .m4a to .WAV\n\t...You may need to hit enter(once or twice) after a minute or so...")
    out_path = os.path.splitext(video_file_path)[0] + ".wav"

    try:
        if os.name == "nt":
            logging.debug("Whisper being ran on windows")
            command = [
                r".\Bin\ffmpeg.exe",        # Assuming the working directory is correctly set where .\Bin exists
                "-ss", "00:00:00",          # Start at the beginning of the video
                "-i", video_file_path,
                "-ar", "16000",             # Audio sample rate
                "-ac", "1",                 # Number of audio channels
                "-c:a", "pcm_s16le",        # Audio codec
                out_path
            ]
            result = subprocess.run(command, text=True, capture_output=True)
            if result.returncode == 0:
                logging.info("FFmpeg executed successfully")
                logging.debug("Output: %s", result.stdout)
            else:
                logging.error("Error in running FFmpeg")
                logging.error("Error Output: %s", result.stderr)
        elif os.name == "posix":
            os.system(f'ffmpeg -ss 00:00:00 -i "{video_file_path}" -ar 16000 -ac 1 -c:a pcm_s16le "{out_path}"')
        else:
            raise RuntimeError("Unsupported operating system")
        logging.info("Conversion to WAV completed: %s", out_path)
    except subprocess.CalledProcessError as e:
        logging.error("Error executing FFmpeg command: %s", str(e))
        raise RuntimeError("Error converting video file to WAV")
    except Exception as e:
        logging.error("Unexpected error occurred: %s", str(e))
        raise RuntimeError("Error converting video file to WAV")
    return out_path



# Transcribe .wav into .segments.json
def speech_to_text(audio_file_path, selected_source_lang='en', whisper_model='small.en', vad_filter=False):
    logging.info('Loading faster_whisper model: %s', whisper_model)
    from faster_whisper import WhisperModel
    model = WhisperModel(whisper_model, device=f"{processing_choice}")
    time_start = time.time()
    if audio_file_path is None:
        raise ValueError("No audio file provided")
    logging.info("Audio file path: %s", audio_file_path)

    try:
        _, file_ending = os.path.splitext(audio_file_path)
        out_file = audio_file_path.replace(file_ending, ".segments.json")
        if os.path.exists(out_file):
            logging.info("Segments file already exists: %s", out_file)
            with open(out_file) as f:
                segments = json.load(f)
            return segments
        
        logging.info('Starting transcription...')
        options = dict(language=selected_source_lang, beam_size=5, best_of=5, vad_filter=vad_filter)
        transcribe_options = dict(task="transcribe", **options)
        segments_raw, info = model.transcribe(audio_file_path, **transcribe_options)

        segments = []
        for segment_chunk in segments_raw:
            chunk = {
                "start": segment_chunk.start,
                "end": segment_chunk.end,
                "text": segment_chunk.text
            }
            logging.debug("Segment: %s", chunk)
            segments.append(chunk)
        logging.info("Transcription completed with faster_whisper")
        with open(out_file, 'w') as f:
            json.dump(segments, f, indent=2)
    except Exception as e:
        logging.error("Error transcribing audio: %s", str(e))
        raise RuntimeError("Error transcribing audio")
    return segments
#
#
####################################################################################################################################






####################################################################################################################################
# Diarization
#
# TODO: https://huggingface.co/pyannote/speaker-diarization-3.1
# embedding_model = "pyannote/embedding", embedding_size=512
# embedding_model = "speechbrain/spkrec-ecapa-voxceleb", embedding_size=192
def speaker_diarize(video_file_path, segments, embedding_model = "pyannote/embedding", embedding_size=512, num_speakers=0):
    """
    1. Generating speaker embeddings for each segments.
    2. Applying agglomerative clustering on the embeddings to identify the speaker for each segment.
    """
    try:
        from pyannote.audio import Audio
        from pyannote.core import Segment
        from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
        import numpy as np
        import pandas as pd
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score
        import tqdm
        import wave

        embedding_model = PretrainedSpeakerEmbedding( embedding_model, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))


        _,file_ending = os.path.splitext(f'{video_file_path}')
        audio_file = video_file_path.replace(file_ending, ".wav")
        out_file = video_file_path.replace(file_ending, ".diarize.json")
        
        logging.debug("getting duration of audio file")
        with contextlib.closing(wave.open(audio_file,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
        logging.debug("duration of audio file obtained")
        print(f"duration of audio file: {duration}")

        def segment_embedding(segment):
            logging.debug("Creating embedding")
            audio = Audio()
            start = segment["start"]
            end = segment["end"]

            # Enforcing a minimum segment length
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
#
#
####################################################################################################################################






####################################################################################################################################
#Summarizers
#
#

# Summarize with OpenAI ChatGPT
def extract_text_from_segments(segments):
    logging.debug(f"openai: extracting text from {segment}")
    text = ' '.join([segment['text'] for segment in segments])
    return text

def summarize_with_openai(api_key, file_path, model):
    try:
        logging.debug("openai: Loading json data for summarization")
        with open(file_path, 'r') as file:
            segments = json.load(file)
        
        logging.debug("openai: Extracting text from the segments")
        text = extract_text_from_segments(segments)

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        logging.debug("openai: Preparing data + prompt for submittal")
        prompt_text = f"{text} \n\n\n\nPlease provide a detailed, bulleted list of the points made throughout the transcribed video and any supporting arguments made for said points"
        data = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional summarizer."
                },
                {
                    "role": "user",
                    "content": prompt_text
                }
            ],
            "max_tokens": 4096,  # Adjust tokens as needed
            "temperature": 0.7
        }
        logging.debug("openai: Posting request")
        response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
        
        if response.status_code == 200:
            summary = response.json()['choices'][0]['message']['content'].strip()
            logging.debug("openai: Summarization successful")
            print("Summarization successful.")
            return summary
        else:
            logging.debug("openai: Summarization failed")
            print("Failed to process summary:", response.text)
            return None
    except Exception as e:
        logging.debug("openai: Generalized error, see above")
        print("Error occurred while processing summary with OpenAI:", str(e))
        return None


def summarize_with_claude(api_key, file_path, model):
    try:
        logging.debug("anthropic: Loading JSON data")
        with open(file_path, 'r') as file:
            segments = json.load(file)
        
        logging.debug("anthropic: Extracting text from the segments")
        text = extract_text_from_segments(segments)

        headers = {
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01',
            'Content-Type': 'application/json'
        }
        
        logging.debug("anthropic: Prepping data + prompt for submittal")
        user_message = {
            "role": "user",
            "content": f"{text} \n\n\n\nPlease provide a detailed, bulleted list of the points made throughout the transcribed video and any supporting arguments made for said points"
        }

        data = {
            "model": model,
            "max_tokens": 4096,            # max _possible_ tokens to return
            "messages": [user_message],
            "stop_sequences": ["\n\nHuman:"],
            "temperature": 0.7,
            "top_k": 0,
            "top_p": 1.0,
            "metadata": {
                "user_id": "example_user_id",
            },
            "stream": False,
            "system": "You are a professional summarizer."
        }
        
        logging.debug("anthropic: Posting request to API")
        response = requests.post('https://api.anthropic.com/v1/messages', headers=headers, json=data)
        
        # Check if the status code indicates success
        if response.status_code == 200:
            logging.debug("anthropic: Post submittal successful")
            response_data = response.json()
            try:
                summary = response_data['content'][0]['text'].strip()
                logging.debug("anthropic: Summarization succesful")
                print("Summary processed successfully.")
                return summary
            except (IndexError, KeyError) as e:
                logging.debug("anthropic: Unexpected data in response")
                print("Unexpected response format from Claude API:", response.text)
                return None
        elif response.status_code == 500:  # Handle internal server error specifically
            logging.debug("anthropic: Internal server error")
            print("Internal server error from API. Retrying may be necessary.")
            return None
        else:
            logging.debug(f"anthropic: Failed to summarize, status code {response.status_code}: {response.text}")
            print(f"Failed to process summary, status code {response.status_code}: {response.text}")
            return None

    except Exception as e:
        logging.debug("anthropic: Generalized error, see above")
        print("Error occurred while processing summary with Claude:", str(e))
        return None



# Summarize with Cohere
def summarize_with_cohere(api_key, file_path, model):
    logging.debug("cohere: Loading JSON data")
    with open(file_path, 'r') as file:
        segments = json.load(file)
    
    logging.debug("cohere: Extracting text from segments")
    text = extract_text_from_segments(segments)

    headers = {
        'accept': 'application/json',
        'content-type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    logging.debug("cohere: Preparing data + prompt for submittal")
    #prompt_text = f"As a professional summarizer, create a concise and comprehensive summary of: {text}"
    prompt_text = f"{text} \n\n\n\nAs a professional summarizer, create a concise and comprehensive summary of the provided text, be it an article, post, conversation, or passage, while adhering to these guidelines: Craft a summary that is detailed, thorough, in-depth, and complex, while maintaining clarity and conciseness. Incorporate main ideas and essential information, eliminating extraneous language and focusing on critical aspects. Rely strictly on the provided text, without including external information. Format the summary in paragraph form for easy understanding. Conclude your notes with [End of Notes, Message #X] to indicate completion, where 'X' represents the total number of messages that I have sent. In other words, include a message counter where you start with #1 and add 1 to the message counter every time I send a message. By following this optimized prompt, you will generate an effective summary that encapsulates the essence of the given text in a clear, concise, and reader-friendly manner. Utilize markdown to cleanly format your output. Example: Bold key subject matter and potential areas that may need expanded information"
    data = {
        "chat_history": [
            {"role": "USER", "message": prompt_text}
        ],
        "message": "Please provide a summary.",
        "model": model,
        "connectors": [{"id": "web-search"}]
    }

    logging.debug("cohere: Submitting request to API endpoint")
    response = requests.post('https://api.cohere.ai/v1/chat', headers=headers, json=data)
    
    if response.status_code == 200:
        logging.debug("cohere: Request was successful!")
        summary = response.json()['response'].strip()
        print("Summary processed successfully.")
        return summary
    else:
        logging.debug("cohere: Unsuccessful request :(")
        print("Failed to process summary:", response.text)
        return None



def summarize_with_llama(api_url, file_path):
    try:
        logging.debug("llama: Loading JSON data")
        with open(file_path, 'r') as file:
            segments = json.load(file)
        
        logging.debug("llama: Extracting text from segments")
        text = extract_text_from_segments(segments)

        logging.debug("llama: Preparing data + prompt for submittal")
        data = {
            "prompt": f"{text} \n\n\n\nPlease provide a detailed, bulleted list of the points made throughout the transcribed video and any supporting arguments made for said points",
            "max_tokens": 4096,
            "stop": ["\n\nHuman:"],
            "temperature": 0.7,
            "top_k": 0,
            "top_p": 1.0,
            "repeat_penalty": 1.0,
            "repeat_last_n": 64,
            "seed": -1,
            "threads": 4,
            "n_predict": 4096
        }

        logging.debug("llama: POSTing data to API endpoint")
        response = requests.post(api_url, json=data)

        if response.status_code == 200:
            logging.debug("llama: POST Successful")
            summary = response.json()['content'].strip()
            print("Summary processed successfully.")
            return summary
        else:
            logging.debug("llama: Unsuccessful POST")
            print("Failed to process summary:", response.text)
            return None
    except Exception as e:
        logging.debug("llama: Generalized error, see above")
        print("Error occurred while processing summary with llama.cpp:", str(e))
        return None


def save_summary_to_file(summary, file_path):
    summary_file_path = file_path.replace('.segments.json', '_summary.txt')
    logging.debug("Opening summary file for writing, *segments.json with *_summary.txt")
    with open(summary_file_path, 'w') as file:
        file.write(summary)
    logging.info(f"Summary saved to file: {summary_file_path}")
#
#
####################################################################################################################################







####################################################################################################################################
# Main()
#

def main(input_path, api_name=None, api_key=None, num_speakers=2, whisper_model="small.en", offset=0, vad_filter=False):
    if os.path.isfile(input_path) and input_path.endswith('.txt'):
        logging.debug("MAIN: User passed in a text file, processing text file...")
        paths = read_paths_from_file(input_path)
    else:
        paths = [input_path]

    results = []
    for path in paths:
        try:
            if path.startswith('http'):
                logging.debug("MAIN: URL Detected")
                info_dict = get_youtube(path)
                if info_dict:
                    logging.debug("MAIN: Creating path for video file...")
                    download_path = create_download_directory(info_dict['title'])
                    logging.debug("MAIN: Path created successfully")
                    logging.debug("MAIN: Downloading video from yt_dlp...")
                    video_path = download_video(path, download_path, info_dict)
                    logging.debug("MAIN: Video downloaded successfully")
                    logging.debug("MAIN: Converting video file to WAV...")
                    audio_file = convert_to_wav(video_path, offset)
                    logging.debug("MAIN: Audio file converted succesfully")
            else:
                if os.path.exists(path):
                    logging.debug("MAIN: Local file path detected")
                    download_path, info_dict, audio_file = process_local_file(path)
                else:
                    logging.error(f"File does not exist: {path}")
                    continue

            if info_dict:
                logging.debug("MAIN: Creating transcription file from WAV")
                segments = speech_to_text(audio_file, whisper_model=whisper_model, vad_filter=vad_filter)
                transcription_result = {
                    'video_path': path,
                    'audio_file': audio_file,
                    'transcription': segments
                }
                results.append(transcription_result)
                logging.info(f"Transcription complete: {audio_file}")

                # Perform summarization based on the specified API
                if api_name:
                    logging.debug(f"MAIN: Summarization being performed by {api_name}")
                    json_file_path = audio_file.replace('.wav', '.segments.json')
                    if api_name.lower() == 'openai':
                        api_key = openai_api_key
                        summary = summarize_with_openai(api_key, json_file_path, openai_model)
                    elif api_name.lower() == 'anthropic':
                        api_key = anthropic_api_key
                        summary = summarize_with_claude(api_key, json_file_path, anthropic_model)
                    elif api_name.lower() == 'cohere':
                        api_key = cohere_api_key
                        summary = summarize_with_cohere(api_key, json_file_path, cohere_model)
                    elif api_name.lower() == 'llama':
                        summary = summarize_with_llmaa(llama_ip, json_file_path)
                    else:
                        logging.warning(f"Unsupported API: {api_name}")
                        summary = None

                    if summary:
                        transcription_result['summary'] = summary
                        logging.info(f"Summary generated using {api_name} API")
                        save_summary_to_file(summary, json_file_path)
                    else:
                        logging.warning(f"Failed to generate summary using {api_name} API")

        except Exception as e:
            logging.error(f"Error processing path: {path}")
            logging.error(str(e))

    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transcribe and summarize videos.')
    parser.add_argument('input_path', type=str, help='Path or URL of the video', nargs='?')
    parser.add_argument('--api_name', type=str, help='API name for summarization (optional)')
    parser.add_argument('--api_key', type=str, help='API key for summarization (optional)')
    parser.add_argument('--num_speakers', type=int, default=2, help='Number of speakers (default: 2)')
    parser.add_argument('--whisper_model', type=str, default='small.en', help='Whisper model (default: small.en)')
    parser.add_argument('--offset', type=int, default=0, help='Offset in seconds (default: 0)')
    parser.add_argument('--vad_filter', action='store_true', help='Enable VAD filter')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Log level (default: INFO)')
    args = parser.parse_args()

    if args.input_path is None:
        parser.print_help()
        sys.exit(1)

    logging.basicConfig(level=getattr(logging, args.log_level), format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info('Starting the transcription and summarization process.')
    logging.info(f'Input path: {args.input_path}')
    logging.info(f'API Name: {args.api_name}')
    logging.debug(f'API Key: {args.api_key}') # ehhhhh
    logging.info(f'Number of speakers: {args.num_speakers}')
    logging.info(f'Whisper model: {args.whisper_model}')
    logging.info(f'Offset: {args.offset}')
    logging.info(f'VAD filter: {args.vad_filter}')
    logging.info(f'Log Level: {args.log_level}') #lol

    if args.api_name and args.api_key:
        logging.info(f'API: {args.api_name}')
        logging.info('Summarization will be performed.')
    else:
        logging.info('No API specified. Summarization will not be performed.')

    logging.debug("Platform check being performed...")
    platform_check()
    logging.debug("CUDA check being performed...")
    cuda_check()
    logging.debug("ffmpeg check being performed...")
    check_ffmpeg()

    try:
        results = main(args.input_path, api_name=args.api_name, api_key=args.api_key, num_speakers=args.num_speakers, whisper_model=args.whisper_model, offset=args.offset, vad_filter=args.vad_filter)
        logging.info('Transcription process completed.')
    except Exception as e:
        logging.error('An error occurred during the transcription process.')
        logging.error(str(e))
        sys.exit(1)

