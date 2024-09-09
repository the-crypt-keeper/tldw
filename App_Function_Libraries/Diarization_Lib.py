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
import logging
from pathlib import Path
from typing import Dict, List, Any
#
# Import 3rd Party Libraries
from tqdm import tqdm
import soundfile as sf
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
import yaml
#
# Import Local Libraries
from App_Function_Libraries.Audio_Transcription_Lib import speech_to_text
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

    # Debug: print the entire config
    logging.debug(f"Loaded config: {config}")

    # Create the SpeakerDiarization pipeline
    try:
        pipeline = SpeakerDiarization(
            segmentation=config['pipeline']['params']['segmentation'],
            embedding=config['pipeline']['params']['embedding'],
            clustering=config['pipeline']['params']['clustering'],
        )
    except KeyError as e:
        logging.error(f"Error accessing config key: {e}")
        raise

    # Set other parameters
    try:
        pipeline_params = {
            "segmentation": {},
            "clustering": {},
        }

        if 'params' in config and 'segmentation' in config['params']:
            if 'min_duration_off' in config['params']['segmentation']:
                pipeline_params["segmentation"]["min_duration_off"] = config['params']['segmentation']['min_duration_off']

        if 'params' in config and 'clustering' in config['params']:
            if 'method' in config['params']['clustering']:
                pipeline_params["clustering"]["method"] = config['params']['clustering']['method']
            if 'min_cluster_size' in config['params']['clustering']:
                pipeline_params["clustering"]["min_cluster_size"] = config['params']['clustering']['min_cluster_size']
            if 'threshold' in config['params']['clustering']:
                pipeline_params["clustering"]["threshold"] = config['params']['clustering']['threshold']

        if 'pipeline' in config and 'params' in config['pipeline']:
            if 'embedding_batch_size' in config['pipeline']['params']:
                pipeline_params["embedding_batch_size"] = config['pipeline']['params']['embedding_batch_size']
            if 'embedding_exclude_overlap' in config['pipeline']['params']:
                pipeline_params["embedding_exclude_overlap"] = config['pipeline']['params']['embedding_exclude_overlap']
            if 'segmentation_batch_size' in config['pipeline']['params']:
                pipeline_params["segmentation_batch_size"] = config['pipeline']['params']['segmentation_batch_size']

        logging.debug(f"Pipeline params: {pipeline_params}")
        pipeline.instantiate(pipeline_params)
    except KeyError as e:
        logging.error(f"Error accessing config key: {e}")
        raise
    except Exception as e:
        logging.error(f"Error instantiating pipeline: {e}")
        raise

    return pipeline

# FIXME - optimize this function
def audio_diarization(audio_file_path: str) -> list:
    logging.info('audio-diarization: Loading pyannote pipeline')

    base_dir = Path(__file__).parent.resolve()
    config_path = base_dir / 'models' / 'pyannote_diarization_config.yaml'
    logging.info(f"audio-diarization: Loading pipeline from {config_path}")

    try:
        pipeline = load_pipeline_from_pretrained(config_path)
    except Exception as e:
        logging.error(f"Failed to load pipeline: {str(e)}")
        raise

    logging.info(f"audio-diarization: Audio file path: {audio_file_path}")

    try:
        logging.info('audio-diarization: Starting diarization...')

        # Get audio duration
        with sf.SoundFile(audio_file_path) as audio_file:
            duration = len(audio_file) / audio_file.samplerate

        # Initialize progress bar
        pbar = tqdm(total=100, desc="Diarization Progress", unit="%")

        # Custom callback function to update progress
        def progress_callback(current_time: float):
            progress = min(100, int((current_time / duration) * 100))
            pbar.n = progress
            pbar.refresh()

        # Perform diarization with progress callback
        diarization_result = pipeline(audio_file_path, callback=progress_callback)

        segments = []
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            segment = {
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            }
            logging.debug(f"Segment: {segment}")
            segments.append(segment)

        pbar.close()
        logging.info("audio-diarization: Diarization completed with pyannote")

        return segments

    except Exception as e:
        logging.error(f"audio-diarization: Error performing diarization: {str(e)}")
        raise RuntimeError("audio-diarization: Error performing diarization") from e


# Old
# def audio_diarization(audio_file_path):
#     logging.info('audio-diarization: Loading pyannote pipeline')
#
#     #config file loading
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     # Construct the path to the config file
#     config_path = os.path.join(current_dir, 'Config_Files', 'config.txt')
#     # Read the config file
#     config = configparser.ConfigParser()
#     config.read(config_path)
#     processing_choice = config.get('Processing', 'processing_choice', fallback='cpu')
#
#     base_dir = Path(__file__).parent.resolve()
#     config_path = base_dir / 'models' / 'config.yaml'
#     pipeline = load_pipeline_from_pretrained(config_path)
#
#     time_start = time.time()
#     if audio_file_path is None:
#         raise ValueError("audio-diarization: No audio file provided")
#     logging.info("audio-diarization: Audio file path: %s", audio_file_path)
#
#     try:
#         _, file_ending = os.path.splitext(audio_file_path)
#         out_file = audio_file_path.replace(file_ending, ".diarization.json")
#         prettified_out_file = audio_file_path.replace(file_ending, ".diarization_pretty.json")
#         if os.path.exists(out_file):
#             logging.info("audio-diarization: Diarization file already exists: %s", out_file)
#             with open(out_file) as f:
#                 global diarization_result
#                 diarization_result = json.load(f)
#             return diarization_result
#
#         logging.info('audio-diarization: Starting diarization...')
#         diarization_result = pipeline(audio_file_path)
#
#         segments = []
#         for turn, _, speaker in diarization_result.itertracks(yield_label=True):
#             chunk = {
#                 "Time_Start": turn.start,
#                 "Time_End": turn.end,
#                 "Speaker": speaker
#             }
#             logging.debug("Segment: %s", chunk)
#             segments.append(chunk)
#         logging.info("audio-diarization: Diarization completed with pyannote")
#
#         output_data = {'segments': segments}
#
#         logging.info("audio-diarization: Saving prettified JSON to %s", prettified_out_file)
#         with open(prettified_out_file, 'w') as f:
#             json.dump(output_data, f, indent=2)
#
#         logging.info("audio-diarization: Saving JSON to %s", out_file)
#         with open(out_file, 'w') as f:
#             json.dump(output_data, f)
#
#     except Exception as e:
#         logging.error("audio-diarization: Error performing diarization: %s", str(e))
#         raise RuntimeError("audio-diarization: Error performing diarization")
#     return segments

def combine_transcription_and_diarization(audio_file_path: str) -> List[Dict[str, Any]]:
    logging.info('combine-transcription-and-diarization: Starting transcription and diarization...')

    try:
        transcription_result = speech_to_text(audio_file_path)
        diarization_result = audio_diarization(audio_file_path)

        if not isinstance(transcription_result, list) or not isinstance(diarization_result, list):
            raise ValueError("Unexpected format for transcription or diarization results")

        combined_result = []
        for transcription_segment in transcription_result:
            if not isinstance(transcription_segment, dict):
                logging.warning(f"Unexpected transcription segment format: {transcription_segment}")
                continue

            for diarization_segment in diarization_result:
                if not isinstance(diarization_segment, dict):
                    logging.warning(f"Unexpected diarization segment format: {diarization_segment}")
                    continue

                try:
                    if (transcription_segment.get('Time_Start', 0) >= diarization_segment.get('start', 0) and
                        transcription_segment.get('Time_End', 0) <= diarization_segment.get('end', 0)):
                        combined_segment = {
                            "Time_Start": transcription_segment.get('Time_Start', 0),
                            "Time_End": transcription_segment.get('Time_End', 0),
                            "Speaker": diarization_segment.get('speaker', 'Unknown'),
                            "Text": transcription_segment.get('Text', '')
                        }
                        combined_result.append(combined_segment)
                        break
                except Exception as e:
                    logging.error(f"Error processing segment: {str(e)}")
                    continue

        return combined_result

    except Exception as e:
        logging.error(f"Error in combine_transcription_and_diarization: {str(e)}")
        return []
#
#
#######################################################################################################################
