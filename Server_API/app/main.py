from fastapi import FastAPI
from Server_API.app.api.v1.endpoints import video_processing
from Server_API.app.core.exceptions import setup_exception_handlers

app = FastAPI(title="TLDW API", version="1.0.0")
setup_exception_handlers(app)
app.include_router(video_processing.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Welcome to the TLDW API"}
