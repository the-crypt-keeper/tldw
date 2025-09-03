# audio_schemas.py
# This module defines the Pydantic schemas for audio-related data models.
#
# Imports
#
# Third-party Libraries
#
# Local Imports
#
#######################################################################################################################
from typing import Literal, Optional

from pydantic import Field, BaseModel


class NormalizationOptions(BaseModel):
    """Options for the normalization system"""

    normalize: bool = Field(
        default=True,
        description="Normalizes input text to make it easier for the model to say",
    )
    unit_normalization: bool = Field(
        default=False, description="Transforms units like 10KB to 10 kilobytes"
    )
    url_normalization: bool = Field(
        default=True,
        description="Changes urls so they can be properly pronounced by kokoro",
    )
    email_normalization: bool = Field(
        default=True,
        description="Changes emails so they can be properly pronouced by kokoro",
    )
    optional_pluralization_normalization: bool = Field(
        default=True,
        description="Replaces (s) with s so some words get pronounced correctly",
    )
    phone_normalization: bool = Field(
        default=True,
        description="Changes phone numbers so they can be properly pronouced by kokoro",
    )


class OpenAISpeechRequest(BaseModel):
    """Request schema for OpenAI-compatible speech endpoint"""

    model: str = Field(
        default="kokoro",
        description="The model to use for generation. Supported models: tts-1, tts-1-hd, kokoro, higgs, chatterbox, vibevoice",
    )
    input: str = Field(..., description="The text to generate audio for")
    voice: str = Field(
        default="af_heart",
        description="The voice to use for generation. Can be a base voice or a combined voice name.",
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description="The format to return audio in. Supported formats: mp3, opus, flac, wav, pcm. PCM format returns raw 16-bit samples without headers. AAC is not currently supported.",
    )
    download_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]] = (
        Field(
            default=None,
            description="Optional different format for the final download. If not provided, uses response_format.",
        )
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="The speed of the generated audio. Select a value from 0.25 to 4.0.",
    )
    stream: bool = Field(
        default=True,  # Default to streaming for OpenAI compatibility
        description="If true (default), audio will be streamed as it's generated. Each chunk will be a complete sentence.",
    )
    return_download_link: bool = Field(
        default=False,
        description="If true, returns a download link in X-Download-Path header after streaming completes",
    )
    lang_code: Optional[str] = Field(
        default=None,
        description="Optional language code to use for text processing. If not provided, will use first letter of voice name.",
    )
    normalization_options: Optional[NormalizationOptions] = Field(
        default=NormalizationOptions(),
        description="Options for the normalization system",
    )
    voice_reference: Optional[str] = Field(
        default=None,
        description="Base64-encoded audio data for voice cloning/reference. Supported by Higgs (3-10s), Chatterbox (5-20s), and VibeVoice models.",
    )
    reference_duration_min: Optional[float] = Field(
        default=None,
        ge=3.0,
        le=60.0,
        description="Minimum duration in seconds for voice reference audio. If provided, will validate reference audio length.",
    )


class OpenAITranscriptionRequest(BaseModel):
    """Request schema for OpenAI-compatible transcription endpoint"""
    
    file: bytes = Field(..., description="The audio file to transcribe")
    model: str = Field(
        default="whisper-1",
        description="ID of the model to use. Options: whisper-1, parakeet, canary, qwen2audio"
    )
    language: Optional[str] = Field(
        default=None,
        description="The language of the input audio. Supplying the input language in ISO-639-1 format will improve accuracy and latency."
    )
    prompt: Optional[str] = Field(
        default=None,
        description="An optional text to guide the model's style or continue a previous audio segment. The prompt should match the audio language."
    )
    response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] = Field(
        default="json",
        description="The format of the transcript output"
    )
    temperature: Optional[float] = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="The sampling temperature, between 0 and 1. Higher values make the output more random."
    )
    timestamp_granularities: Optional[list[Literal["word", "segment"]]] = Field(
        default=["segment"],
        description="The timestamp granularities to populate for this transcription"
    )


class OpenAITranscriptionResponse(BaseModel):
    """Response schema for OpenAI-compatible transcription endpoint"""
    
    text: str = Field(..., description="The transcribed text")
    language: Optional[str] = Field(None, description="The language of the input audio")
    duration: Optional[float] = Field(None, description="The duration of the input audio in seconds")
    words: Optional[list] = Field(None, description="Word-level timestamps if requested")
    segments: Optional[list] = Field(None, description="Segment-level timestamps if requested")


class OpenAITranslationRequest(BaseModel):
    """Request schema for OpenAI-compatible translation endpoint"""
    
    file: bytes = Field(..., description="The audio file to translate")
    model: str = Field(
        default="whisper-1",
        description="ID of the model to use. Currently only whisper-1 is available"
    )
    prompt: Optional[str] = Field(
        default=None,
        description="An optional text to guide the model's style or continue a previous audio segment"
    )
    response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] = Field(
        default="json",
        description="The format of the transcript output"
    )
    temperature: Optional[float] = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="The sampling temperature"
    )

#
# End of audio_schemas.py
#######################################################################################################################
