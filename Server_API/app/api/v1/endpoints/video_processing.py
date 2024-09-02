from fastapi import APIRouter, BackgroundTasks, HTTPException
from typing import List, Optional
from app.services.video_processing_service import process_video_task

router = APIRouter()

@router.post("/process-video")
async def process_video(
    url: str,
    whisper_model: str,
    custom_prompt: Optional[str] = None,
    api_name: str,
    api_key: str,
    keywords: List[str] = [],
    diarize: bool = False,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    include_timestamps: bool = True,
    keep_original_video: bool = False,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    task_id = f"task_{url.replace('://', '_').replace('/', '_')}"
    background_tasks.add_task(process_video_task, url, whisper_model, custom_prompt, api_name, api_key,
                              keywords, diarize, start_time, end_time, include_timestamps, keep_original_video)
    return {"task_id": task_id, "message": "Video processing started"}