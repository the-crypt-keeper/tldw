# Diarization_Lib.py
#########################################
# Diarization Library
# This library is used to perform diarization of audio files.
# Currently, uses FIXME for transcription.
#
####

####################
# Function List
#
# 1. speaker_diarize(video_file_path, segments, embedding_model = "pyannote/embedding", embedding_size=512, num_speakers=0)
#
####################



# FIXME - Need to replace the following imports with the correct ones
# Need to replace sklearn with scikit-learn





# Import necessary libraries
import os
import logging
from pathlib import *
from pyannote.audio import Audio
from pyannote.core import Segment
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import tqdm
import wave
# Import Local
import summarize
from Article_Summarization_Lib import *
from Article_Extractor_Lib import *
from Audio_Transcription_Lib import *
from Chunk_Lib import *
#from Diarization_Lib import *
from Video_DL_Ingestion_Lib import *
from Local_File_Processing_Lib import *
from Local_LLM_Inference_Engine_Lib import *
from Local_Summarization_Lib import *
from Old_Chunking_Lib import *
from SQLite_DB import *
from Summarization_General_Lib import *
from System_Checks_Lib import *
from Tokenization_Methods_Lib import *
from Video_DL_Ingestion_Lib import *
from Web_UI_Lib import *



import os
import json
import logging
import configparser
import time
from pyannote.audio import Pipeline

#######################################################################################################################
# Function Definitions
#

def load_pipeline_from_pretrained(path_to_config: str | Path) -> Pipeline:
    path_to_config = Path(path_to_config)

    print(f"Loading pyannote pipeline from {path_to_config}...")
    # the paths in the config are relative to the current working directory
    # so we need to change the working directory to the model path
    # and then change it back

    cwd = Path.cwd().resolve()  # store current working directory

    # first .parent is the folder of the config, second .parent is the folder containing the 'models' folder
    cd_to = path_to_config.parent.parent.resolve()

    print(f"Changing working directory to {cd_to}")
    os.chdir(cd_to)

    pipeline = Pipeline.from_pretrained(path_to_config)

    print(f"Changing working directory back to {cwd}")
    os.chdir(cwd)

    return pipeline

PATH_TO_CONFIG = "../models/config.yaml"
pipeline = load_pipeline_from_pretrained(PATH_TO_CONFIG)


def audio_diarization(audio_file_path):
    logging.info('audio-diarization: Loading pyannote pipeline')
    # Retrieve processing choice from the configuration file
    config = configparser.ConfigParser()
    config.read('config.txt')
    processing_choice = config.get('Processing', 'processing_choice', fallback='cpu')

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", device=f"{processing_choice}")
    time_start = time.time()
    if audio_file_path is None:
        raise ValueError("audio-diarization: No audio file provided")
    logging.info("audio-diarization: Audio file path: %s", audio_file_path)

    try:
        _, file_ending = os.path.splitext(audio_file_path)
        out_file = audio_file_path.replace(file_ending, ".diarization.json")
        prettified_out_file = audio_file_path.replace(file_ending, ".diarization_pretty.json")
        if os.path.exists(out_file):
            logging.info("audio-diarization: Diarization file already exists: %s", out_file)
            with open(out_file) as f:
                global diarization_result
                diarization_result = json.load(f)
            return diarization_result

        logging.info('audio-diarization: Starting diarization...')
        diarization_result = pipeline(audio_file_path)

        segments = []
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            chunk = {
                "Time_Start": turn.start,
                "Time_End": turn.end,
                "Speaker": speaker
            }
            logging.debug("Segment: %s", chunk)
            segments.append(chunk)
        logging.info("audio-diarization: Diarization completed with pyannote")

        # Create a dictionary with the 'segments' key
        output_data = {'segments': segments}

        # Save prettified JSON
        logging.info("audio-diarization: Saving prettified JSON to %s", prettified_out_file)
        with open(prettified_out_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        # Save non-prettified JSON
        logging.info("audio-diarization: Saving JSON to %s", out_file)
        with open(out_file, 'w') as f:
            json.dump(output_data, f)

    except Exception as e:
        logging.error("audio-diarization: Error performing diarization: %s", str(e))
        raise RuntimeError("audio-diarization: Error performing diarization")
    return segments

load_pipeline_from_pretrained()
combine_transcription_and_diarization()

# Example use of speech_to_text function
audio_file_path = "example_audio.wav"
transcription_result = speech_to_text(audio_file_path)
print("Transcription Result:", transcription_result)


# Example use of audio_diarization function
audio_file_path = "example_audio.wav"
diarization_result = audio_diarization(audio_file_path)
print("Diarization Result:", diarization_result)


# Example use of combine_transcription_and_diarization function
audio_file_path = "example_audio.wav"
combined_result = combine_transcription_and_diarization(audio_file_path)
print("Combined Result:", combined_result)





def combine_transcription_and_diarization(audio_file_path):
    logging.info('combine-transcription-and-diarization: Starting transcription and diarization...')

    # Run the transcription function
    transcription_result = speech_to_text(audio_file_path)

    # Run the diarization function
    diarization_result = audio_diarization(audio_file_path)

    # Combine the results
    combined_result = []
    for transcription_segment in transcription_result:
        for diarization_segment in diarization_result:
            if transcription_segment['Time_Start'] >= diarization_segment['Time_Start'] and transcription_segment[
                'Time_End'] <= diarization_segment['Time_End']:
                combined_segment = {
                    "Time_Start": transcription_segment['Time_Start'],
                    "Time_End": transcription_segment['Time_End'],
                    "Speaker": diarization_segment['Speaker'],
                    "Text": transcription_segment['Text']
                }
                combined_result.append(combined_segment)
                break

    # Save the combined result to a JSON file
    _, file_ending = os.path.splitext(audio_file_path)
    out_file = audio_file_path.replace(file_ending, ".combined.json")
    prettified_out_file = audio_file_path.replace(file_ending, ".combined_pretty.json")

    logging.info("combine-transcription-and-diarization: Saving prettified JSON to %s", prettified_out_file)
    with open(prettified_out_file, 'w') as f:
        json.dump(combined_result, f, indent=2)

    logging.info("combine-transcription-and-diarization: Saving JSON to %s", out_file)
    with open(out_file, 'w') as f:
        json.dump(combined_result, f)

    return combined_result










#
# # OLD FUNCTION
# # TODO: https://huggingface.co/pyannote/speaker-diarization-3.1
# # FIXME
# embedding_model = "pyannote/embedding", embedding_size=512
# embedding_model = "speechbrain/spkrec-ecapa-voxceleb", embedding_size=192
# def speaker_diarize(video_file_path, segments, embedding_model = "pyannote/embedding", embedding_size=512, num_speakers=0):
#     """
#     1. Generating speaker embeddings for each segments.
#     2. Applying agglomerative clustering on the embeddings to identify the speaker for each segment.
#     """
#     try:
#         embedding_model = PretrainedSpeakerEmbedding( embedding_model, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#
#         _,file_ending = os.path.splitext(f'{video_file_path}')
#         audio_file = video_file_path.replace(file_ending, ".wav")
#         out_file = video_file_path.replace(file_ending, ".diarize.json")
#
#         logging.debug("getting duration of audio file")
#         with contextlib.closing(wave.open(audio_file,'r')) as f:
#             frames = f.getnframes()
#             rate = f.getframerate()
#             duration = frames / float(rate)
#         logging.debug("duration of audio file obtained")
#         print(f"duration of audio file: {duration}")
#
#         def segment_embedding(segment):
#             logging.debug("Creating embedding")
#             audio = Audio()
#             start = segment["start"]
#             end = segment["end"]
#
#             # Enforcing a minimum segment length
#             if end-start < 0.3:
#                 padding = 0.3-(end-start)
#                 start -= padding/2
#                 end += padding/2
#                 print('Padded segment because it was too short:',segment)
#
#             # Whisper overshoots the end timestamp in the last segment
#             end = min(duration, end)
#             # clip audio and embed
#             clip = Segment(start, end)
#             waveform, sample_rate = audio.crop(audio_file, clip)
#             return embedding_model(waveform[None])
#
#         embeddings = np.zeros(shape=(len(segments), embedding_size))
#         for i, segment in enumerate(tqdm.tqdm(segments)):
#             embeddings[i] = segment_embedding(segment)
#         embeddings = np.nan_to_num(embeddings)
#         print(f'Embedding shape: {embeddings.shape}')
#
#         if num_speakers == 0:
#         # Find the best number of speakers
#             score_num_speakers = {}
#
#             for num_speakers in range(2, 10+1):
#                 clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
#                 score = silhouette_score(embeddings, clustering.labels_, metric='euclidean')
#                 score_num_speakers[num_speakers] = score
#             best_num_speaker = max(score_num_speakers, key=lambda x:score_num_speakers[x])
#             print(f"The best number of speakers: {best_num_speaker} with {score_num_speakers[best_num_speaker]} score")
#         else:
#             best_num_speaker = num_speakers
#
#         # Assign speaker label
#         clustering = AgglomerativeClustering(best_num_speaker).fit(embeddings)
#         labels = clustering.labels_
#         for i in range(len(segments)):
#             segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)
#
#         with open(out_file,'w') as f:
#             f.write(json.dumps(segments, indent=2))
#
#         # Make CSV output
#         def convert_time(secs):
#             return datetime.timedelta(seconds=round(secs))
#
#         objects = {
#             'Start' : [],
#             'End': [],
#             'Speaker': [],
#             'Text': []
#         }
#         text = ''
#         for (i, segment) in enumerate(segments):
#             if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
#                 objects['Start'].append(str(convert_time(segment["start"])))
#                 objects['Speaker'].append(segment["speaker"])
#                 if i != 0:
#                     objects['End'].append(str(convert_time(segments[i - 1]["end"])))
#                     objects['Text'].append(text)
#                     text = ''
#             text += segment["text"] + ' '
#         objects['End'].append(str(convert_time(segments[i - 1]["end"])))
#         objects['Text'].append(text)
#
#         save_path = video_file_path.replace(file_ending, ".csv")
#         df_results = pd.DataFrame(objects)
#         df_results.to_csv(save_path)
#         return df_results, save_path
#
#     except Exception as e:
#         raise RuntimeError("Error Running inference with local model", e)

#
#
#######################################################################################################################