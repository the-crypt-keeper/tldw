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
# import os
# import logging
# from pyannote.audio import Audio
# from pyannote.core import Segment
# from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
# import numpy as np
# import pandas as pd
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.metrics import silhouette_score
# import tqdm
# import wave
# # Import Local
import summarize
# from Article_Summarization_Lib import *
# from Article_Extractor_Lib import *
# from Audio_Transcription_Lib import *
# from Chunk_Lib import *
# #from Diarization_Lib import *
# from Video_DL_Ingestion_Lib import *
# from Local_File_Processing_Lib import *
# from Local_LLM_Inference_Engine_Lib import *
# from Local_Summarization_Lib import *
# from Old_Chunking_Lib import *
# from SQLite_DB import *
# from Summarization_General_Lib import *
# from System_Checks_Lib import *
# from Tokenization_Methods_Lib import *
# from Video_DL_Ingestion_Lib import *
# from Web_UI_Lib import *

#######################################################################################################################
# Function Definitions
#

# TODO: https://huggingface.co/pyannote/speaker-diarization-3.1
# FIXME
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