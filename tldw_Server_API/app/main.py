# main.py
# Description: This file contains the main FastAPI application, which serves as the primary API for the tldw application.
#
# Imports
import logging
#
# 3rd-party Libraries
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from loguru import logger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles
#
# Local Imports
#
# Audio Endpoint
from tldw_Server_API.app.api.v1.endpoints.audio import router as audio_router
#
# Chat Endpoint
from tldw_Server_API.app.api.v1.endpoints.chat import router as chat_router
#
# Character Endpoint
from tldw_Server_API.app.api.v1.endpoints.characters_endpoint import router as character_router
#
# Chunking Endpoint
from tldw_Server_API.app.api.v1.endpoints.chunking import chunking_router as chunking_router
#
# Embedding Endpoint (v4 with multi-provider support)
from tldw_Server_API.app.api.v1.endpoints.embeddings_v4 import router as embeddings_router
# Previous versions (To be removed)
# from tldw_Server_API.app.api.v1.endpoints.embeddings_v3 import router as embeddings_router_v3
# from tldw_Server_API.app.api.v1.endpoints.embeddings_v2 import router as embeddings_router_v2
# from tldw_Server_API.app.api.v1.endpoints.embeddings import router as embeddings_router_old
#
# Media Endpoint
from tldw_Server_API.app.api.v1.endpoints.media import router as media_router
#
# Notes Endpoint
from tldw_Server_API.app.api.v1.endpoints.notes import router as notes_router
#
# Prompt Management Endpoint
from tldw_Server_API.app.api.v1.endpoints.prompts import router as prompt_router
#
# RAG Endpoint (New simplified version)
from tldw_Server_API.app.api.v1.endpoints.rag_v2 import router as rag_router
# Legacy RAG Endpoint (To be removed)
# from tldw_Server_API.app.api.v1.endpoints.rag import router as retrieval_agent_router
#
# Research Endpoint
from tldw_Server_API.app.api.v1.endpoints.research import router as research_router
#
# Evaluation Endpoint (OLD - to be removed)
from tldw_Server_API.app.api.v1.endpoints.evals import router as evaluation_router
#
# OpenAI-compatible Evaluation Endpoint (NEW)
from tldw_Server_API.app.api.v1.endpoints.evals_openai import router as openai_evals_router
#
# Sync Endpoint
from tldw_Server_API.app.api.v1.endpoints.sync import router as sync_router
#
# Tools Endpoint
from tldw_Server_API.app.api.v1.endpoints.tools import router as tools_router
## Trash Endpoint
#from tldw_Server_API.app.api.v1.endpoints.trash import router as trash_router
#
# MCP Endpoint
from tldw_Server_API.app.api.v1.endpoints.mcp_endpoint import router as mcp_router
#
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
#
########################################################################################################################
#
# Functions:


# --- Loguru Configuration with Intercept Handler ---

# Define a handler class to intercept standard logging messages
class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

# Remove default handler
logger.remove()

# Add your desired Loguru sink (e.g., stderr)
log_level = "DEBUG"
logger.add(
    sys.stderr,
    level=log_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
)

# Configure standard logging to use the InterceptHandler
loggers_to_intercept = ["uvicorn", "uvicorn.error", "uvicorn.access"] # Add other library names if needed
for logger_name in loggers_to_intercept:
    mod_logger = logging.getLogger(logger_name)
    mod_logger.handlers = [InterceptHandler()]
    mod_logger.propagate = False # Prevent messages from reaching the root logger
    # Optionally set level if you only want certain levels from that lib
    # mod_logger.setLevel(logging.DEBUG)

logger.info("Loguru logger configured with SPECIFIC standard logging interception!")


BASE_DIR     = Path(__file__).resolve().parent
FAVICON_PATH = BASE_DIR / "static" / "favicon.ico"
app = FastAPI(
    title="tldw API",
    version="0.0.1",
    description="Version 0.0.1: Smooth Slide | FastAPI Backend for the tldw project"
)

############################# TEST DB Handling #####################################
# --- TEST DB Instance ---
test_db_instance_ref = None # Global or context variable to hold the test DB instance

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code can go here
    yield
    # Shutdown code
    global test_db_instance_ref
    if test_db_instance_ref and hasattr(test_db_instance_ref, 'close_all_connections'):
        logger.info("App Shutdown: Closing DB connections")
        test_db_instance_ref.close_all_connections()
    else:
        logger.info("App Shutdown: No test DB instance found to close")
#
############################# End of Test DB Handling###################


# --- FIX: Add CORS Middleware ---
# Import from config
from tldw_Server_API.app.core.config import ALLOWED_ORIGINS, API_V1_PREFIX

# Use configured origins
origins = ALLOWED_ORIGINS if ALLOWED_ORIGINS else ["*"]

# FIXME - CORS
# # -- If you have any global middleware, add it here --
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Must include OPTIONS, GET, POST, DELETE etc.
    allow_headers=["*"],
)

# Static files serving
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Favicon serving
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(FAVICON_PATH, media_type="image/x-icon")

@app.get("/")
async def root():
    return {"message": "Welcome to the tldw API; If you're seeing this, the server is running!"}

# Router for media endpoints/media file handling
app.include_router(media_router, prefix=f"{API_V1_PREFIX}/media", tags=["media"])

# Router for /audio/ endpoints
app.include_router(audio_router, prefix=f"{API_V1_PREFIX}/audio", tags=["audio"])


# Router for chat endpoints/chat temp-file handling
app.include_router(chat_router, prefix=f"{API_V1_PREFIX}/chat", tags=["chat"])


# Router for chat endpoints/chat temp-file handling
app.include_router(character_router, prefix=f"{API_V1_PREFIX}/characters", tags=["character, persona"])


# Router for Chunking Endpoint
app.include_router(chunking_router, prefix=f"{API_V1_PREFIX}/chunking", tags=["chunking"])


# Router for Embedding Endpoint (OpenAI-compatible path)
app.include_router(embeddings_router, prefix=f"{API_V1_PREFIX}", tags=["embeddings"])


# Router for Note Management endpoints
app.include_router(notes_router, prefix=f"{API_V1_PREFIX}/notes", tags=["notes"])


# Router for Prompt Management endpoints
app.include_router(prompt_router, prefix=f"{API_V1_PREFIX}/prompts", tags=["prompts"])


# Router for RAG endpoint (New simplified version)
app.include_router(rag_router, prefix=f"{API_V1_PREFIX}/rag", tags=["RAG"])


# Router for Research endpoint
app.include_router(research_router, prefix=f"{API_V1_PREFIX}/research", tags=["research"])


# Router for Evaluation endpoint (OLD - will be removed)
# app.include_router(evaluation_router, prefix=f"{API_V1_PREFIX}", tags=["evaluations"])

# Router for OpenAI-compatible Evaluation endpoint (NEW)
app.include_router(openai_evals_router, tags=["evaluations"])


# Router for Sync endpoint
app.include_router(sync_router, prefix=f"{API_V1_PREFIX}/sync", tags=["sync"])


# Router for Tools endpoint
app.include_router(tools_router, prefix=f"{API_V1_PREFIX}/tools", tags=["tools"])


# Router for MCP (Model Context Protocol) endpoint
app.include_router(mcp_router, prefix=f"{API_V1_PREFIX}", tags=["MCP"])


# Router for trash endpoints - deletion of media items / trash file handling (FIXME: Secure delete vs lag on delete?)
#app.include_router(trash_router, prefix=f"{API_V1_PREFIX}/trash", tags=["trash"])

# Router for authentication endpoint
#app.include_router(auth_router, prefix=f"{API_V1_PREFIX}/auth", tags=["auth"])
# The docs at http://localhost:8000/docs will show an “Authorize” button. You can log in by calling POST /api/v1/auth/login with a form that includes username and password. The docs interface is automatically aware because we used OAuth2PasswordBearer.

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

#
## End of main.py
########################################################################################################################
