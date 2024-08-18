# Diarization_Lib.py
#########################################
# Diarization Library
# This library is used to perform diarization of audio files.
# Currently, uses FIXME for transcription.
#
####################
####################
# Function List
#
# 1. speaker_diarize(video_file_path, segments, embedding_model = "pyannote/embedding", embedding_size=512, num_speakers=0)
#
####################
# Import necessary libraries
import configparser
import json
import logging
import os
from pathlib import Path
import time
# Import Local
from App_Function_Libraries.Audio_Transcription_Lib import speech_to_text
#
# Import 3rd Party
from pyannote.audio import Model
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
import torch
import yaml
#
#######################################################################################################################
# Function Definitions
#

def load_pipeline_from_pretrained(path_to_config: str | Path) -> SpeakerDiarization:
    path_to_config = Path(path_to_config).resolve()
    logging.debug(f"Loading pyannote pipeline from {path_to_config}...")

    if not path_to_config.exists():
        raise FileNotFoundError(f"Config file not found: {path_to_config}")

    # Load the YAML configuration
    with open(path_to_config, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Store current working directory
    cwd = Path.cwd().resolve()

    try:
        # Create a SpeakerDiarization pipeline
        pipeline = SpeakerDiarization()

        # Load models explicitly from local paths
        embedding_path = Path(config['pipeline']['params']['embedding']).resolve()
        segmentation_path = Path(config['pipeline']['params']['segmentation']).resolve()

        if not embedding_path.exists():
            raise FileNotFoundError(f"Embedding model file not found: {embedding_path}")
        if not segmentation_path.exists():
            raise FileNotFoundError(f"Segmentation model file not found: {segmentation_path}")

        # Load the models from local paths using pyannote's Model class
        pipeline.embedding = Model.from_pretrained(str(embedding_path), map_location=torch.device('cpu'))
        pipeline.segmentation = Model.from_pretrained(str(segmentation_path), map_location=torch.device('cpu'))

        # Set other parameters
        pipeline.clustering = config['pipeline']['params']['clustering']
        pipeline.embedding_batch_size = config['pipeline']['params']['embedding_batch_size']
        pipeline.embedding_exclude_overlap = config['pipeline']['params']['embedding_exclude_overlap']
        pipeline.segmentation_batch_size = config['pipeline']['params']['segmentation_batch_size']

        # Set additional parameters
        pipeline.instantiate(config['params'])

    finally:
        # Change back to the original working directory
        print(f"Changing working directory back to {cwd}")
        os.chdir(cwd)

    return pipeline

def audio_diarization(audio_file_path):
    logging.info('audio-diarization: Loading pyannote pipeline')
    config = configparser.ConfigParser()
    config.read('config.txt')
    processing_choice = config.get('Processing', 'processing_choice', fallback='cpu')

    base_dir = Path(__file__).parent.resolve()
    config_path = base_dir / 'models' / 'config.yaml'
    pipeline = load_pipeline_from_pretrained(config_path)

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

        output_data = {'segments': segments}

        logging.info("audio-diarization: Saving prettified JSON to %s", prettified_out_file)
        with open(prettified_out_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        logging.info("audio-diarization: Saving JSON to %s", out_file)
        with open(out_file, 'w') as f:
            json.dump(output_data, f)

    except Exception as e:
        logging.error("audio-diarization: Error performing diarization: %s", str(e))
        raise RuntimeError("audio-diarization: Error performing diarization")
    return segments

def combine_transcription_and_diarization(audio_file_path):
    logging.info('combine-transcription-and-diarization: Starting transcription and diarization...')

    transcription_result = speech_to_text(audio_file_path)

    diarization_result = audio_diarization(audio_file_path)

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
#
#######################################################################################################################