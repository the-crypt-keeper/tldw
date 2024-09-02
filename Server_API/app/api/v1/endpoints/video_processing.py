from fastapi import APIRouter, BackgroundTasks, HTTPException
from typing import List, Optional
from Server_API.app.services.video_processing_service import process_video_task

router = APIRouter()

# @router.post("/process-video", summary="Process a video", description="Download, transcribe, and summarize a video from the given URL.")
# async def process_video(
#     url: str = Query(..., description="URL of the video to process"),
#     whisper_model: str = Query(..., description="Whisper model to use for transcription"),
#     custom_prompt: Optional[str] = Query(None, description="Custom prompt for summarization"),
#     api_name: str = Query(..., description="Name of the API to use for summarization"),
#     api_key: str = Query(..., description="API key for the summarization service"),
#     keywords: List[str] = Query(default=[], description="Keywords to associate with the video"),
#     diarize: bool = Query(False, description="Whether to perform speaker diarization"),
#     start_time: Optional[str] = Query(None, description="Start time for processing (format: HH:MM:SS)"),
#     end_time: Optional[str] = Query(None, description="End time for processing (format: HH:MM:SS)"),
#     include_timestamps: bool = Query(True, description="Whether to include timestamps in the transcription"),
#     keep_original_video: bool = Query(False, description="Whether to keep the original video file after processing"),
#     background_tasks: BackgroundTasks = BackgroundTasks()
# ):
#     task_id = f"task_{url.replace('://', '_').replace('/', '_')}"
#     background_tasks.add_task(process_video_task, url, whisper_model, custom_prompt, api_name, api_key,
#                               keywords, diarize, start_time, end_time, include_timestamps, keep_original_video)
#     return {"task_id": task_id, "message": "Video processing started"}