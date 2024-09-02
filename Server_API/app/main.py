from fastapi import FastAPI
from app.api.v1.endpoints import video_processing

app = FastAPI(title="TLDW API", version="1.0.0")

app.include_router(video_processing.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Welcome to the TLDW API"}
