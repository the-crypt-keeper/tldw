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
# Auth Endpoint (NEW)
from tldw_Server_API.app.api.v1.endpoints.auth import router as auth_router
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
# Metrics Endpoint
from tldw_Server_API.app.api.v1.endpoints.metrics import router as metrics_router
#
# Chunking Endpoints
from tldw_Server_API.app.api.v1.endpoints.chunking import chunking_router as chunking_router
from tldw_Server_API.app.api.v1.endpoints.chunking_templates import router as chunking_templates_router
#
# Embedding Endpoint (v5 enhanced version with circuit breaker)
from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import router as embeddings_router
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
# Prompt Studio Endpoints
from tldw_Server_API.app.api.v1.endpoints.prompt_studio_projects import router as prompt_studio_projects_router
from tldw_Server_API.app.api.v1.endpoints.prompt_studio_prompts import router as prompt_studio_prompts_router
from tldw_Server_API.app.api.v1.endpoints.prompt_studio_test_cases import router as prompt_studio_test_cases_router
from tldw_Server_API.app.api.v1.endpoints.prompt_studio_optimization import router as prompt_studio_optimization_router
from tldw_Server_API.app.api.v1.endpoints.prompt_studio_websocket import router as prompt_studio_websocket_router
from tldw_Server_API.app.api.v1.endpoints.prompt_studio_evaluations import router as prompt_studio_evaluations_router
#
# RAG Endpoints
from tldw_Server_API.app.api.v1.endpoints.rag_api import router as rag_api_router  # Production RAG API using functional pipeline
# Legacy RAG Endpoint (Deprecated)
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
# Benchmark Endpoint
from tldw_Server_API.app.api.v1.endpoints.benchmark_api import router as benchmark_router
#
# Sync Endpoint
from tldw_Server_API.app.api.v1.endpoints.sync import router as sync_router
#
# Tools Endpoint
from tldw_Server_API.app.api.v1.endpoints.tools import router as tools_router
#
# Users Endpoint (NEW)
from tldw_Server_API.app.api.v1.endpoints.users import router as users_router
## Trash Endpoint
#from tldw_Server_API.app.api.v1.endpoints.trash import router as trash_router
#
# MCP Unified Endpoint (Production-ready, secure implementation)
from tldw_Server_API.app.api.v1.endpoints.mcp_unified_endpoint import router as mcp_unified_router
# Note: Old MCP endpoints have been archived due to security vulnerabilities
#
# Chatbooks Endpoint
from tldw_Server_API.app.api.v1.endpoints.chatbooks import router as chatbooks_router
#
# Metrics and Telemetry
from tldw_Server_API.app.core.Metrics import (
    initialize_telemetry,
    shutdown_telemetry,
    get_metrics_registry,
    track_metrics,
    OTEL_AVAILABLE
)
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

############################# TEST DB Handling #####################################
# --- TEST DB Instance ---
test_db_instance_ref = None # Global or context variable to hold the test DB instance

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize telemetry and metrics
    logger.info("App Startup: Initializing telemetry and metrics...")
    try:
        telemetry_manager = initialize_telemetry()
        if OTEL_AVAILABLE:
            logger.info(f"App Startup: OpenTelemetry initialized for service: {telemetry_manager.config.service_name}")
        else:
            logger.warning("App Startup: OpenTelemetry not available, using fallback metrics")
    except Exception as e:
        logger.error(f"App Startup: Failed to initialize telemetry: {e}")
    
    # Startup: Initialize auth services
    logger.info("App Startup: Initializing authentication services...")
    try:
        # Initialize database pool for auth
        from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
        db_pool = await get_db_pool()
        logger.info("App Startup: Database pool initialized")
        
        # Initialize session manager
        from tldw_Server_API.app.core.AuthNZ.session_manager import get_session_manager
        session_manager = await get_session_manager()
        logger.info("App Startup: Session manager initialized")
    except Exception as e:
        logger.error(f"App Startup: Failed to initialize auth services: {e}")
        # Continue startup even if auth services fail (for backward compatibility)
    
    # Initialize MCP Unified Server (secure, production-ready)
    logger.info("App Startup: Initializing MCP Unified server...")
    try:
        from tldw_Server_API.app.core.MCP_unified import get_mcp_server
        mcp_server = get_mcp_server()
        await mcp_server.initialize()
        logger.info("App Startup: MCP Unified server initialized successfully")
    except Exception as e:
        logger.error(f"App Startup: Failed to initialize MCP Unified server: {e}")
        logger.warning("Ensure MCP_JWT_SECRET and MCP_API_KEY_SALT environment variables are set")
        # Continue startup even if MCP fails (for backward compatibility)
    
    # Initialize Chat Module Components
    logger.info("App Startup: Initializing Chat module components...")
    
    # Initialize Provider Manager
    try:
        from tldw_Server_API.app.core.Chat.provider_manager import initialize_provider_manager
        from tldw_Server_API.app.core.Chat.Chat_Functions import API_CALL_HANDLERS
        
        # Get list of configured providers
        providers = list(API_CALL_HANDLERS.keys())
        provider_manager = initialize_provider_manager(providers, primary_provider=providers[0] if providers else None)
        await provider_manager.start_health_checks()
        logger.info(f"App Startup: Provider manager initialized with {len(providers)} providers")
    except Exception as e:
        logger.error(f"App Startup: Failed to initialize provider manager: {e}")
    
    # Initialize Request Queue
    try:
        from tldw_Server_API.app.core.Chat.request_queue import initialize_request_queue
        from tldw_Server_API.app.core.config import load_comprehensive_config
        
        config = load_comprehensive_config()
        chat_config = {}
        if config and config.has_section('Chat-Module'):
            chat_config = dict(config.items('Chat-Module'))
        
        request_queue = initialize_request_queue(
            max_queue_size=int(chat_config.get('max_queue_size', 100)),
            max_concurrent=int(chat_config.get('max_concurrent_requests', 10)),
            global_rate_limit=int(chat_config.get('rate_limit_per_minute', 60)),
            per_client_rate_limit=int(chat_config.get('rate_limit_per_conversation_per_minute', 20))
        )
        await request_queue.start(num_workers=4)
        logger.info("App Startup: Request queue initialized with 4 workers")
    except Exception as e:
        logger.error(f"App Startup: Failed to initialize request queue: {e}")
    
    # Initialize Rate Limiter
    try:
        from tldw_Server_API.app.core.Chat.rate_limiter import initialize_rate_limiter, RateLimitConfig
        
        rate_config = RateLimitConfig(
            global_rpm=int(chat_config.get('rate_limit_per_minute', 60)),
            per_user_rpm=int(chat_config.get('rate_limit_per_user_per_minute', 20)),
            per_conversation_rpm=int(chat_config.get('rate_limit_per_conversation_per_minute', 10)),
            per_user_tokens_per_minute=int(chat_config.get('rate_limit_tokens_per_minute', 10000))
        )
        rate_limiter = initialize_rate_limiter(rate_config)
        logger.info("App Startup: Rate limiter initialized")
    except Exception as e:
        logger.error(f"App Startup: Failed to initialize rate limiter: {e}")
    
    # Initialize TTS Service
    logger.info("App Startup: Initializing TTS service...")
    try:
        from tldw_Server_API.app.core.TTS.tts_generation import get_tts_service
        from tldw_Server_API.app.core.config import load_comprehensive_config
        
        # Load TTS configuration
        tts_config = load_comprehensive_config()
        
        # Initialize the TTS service with configuration
        tts_service = await get_tts_service(app_config=tts_config)
        logger.info("App Startup: TTS service initialized successfully")
    except Exception as e:
        logger.error(f"App Startup: Failed to initialize TTS service: {e}")
        logger.warning("TTS functionality will be unavailable")
        # Continue startup even if TTS fails (for backward compatibility)
    
    # Initialize Chunking Templates
    logger.info("App Startup: Initializing chunking templates...")
    try:
        from tldw_Server_API.app.core.Chunking.template_initialization import ensure_templates_initialized
        
        if ensure_templates_initialized():
            logger.info("App Startup: Chunking templates initialized successfully")
        else:
            logger.warning("App Startup: Chunking templates initialization incomplete")
    except Exception as e:
        logger.error(f"App Startup: Failed to initialize chunking templates: {e}")
        # Continue startup even if template initialization fails
    
    # Initialize Unified Audit Service
    try:
        from tldw_Server_API.app.core.Audit.unified_audit_service import get_unified_audit_service
        
        audit_service = await get_unified_audit_service()
        logger.info("App Startup: Unified audit service initialized")
    except Exception as e:
        logger.error(f"App Startup: Failed to initialize audit service: {e}")
    
    yield
    
    # Shutdown: Clean up resources
    logger.info("App Shutdown: Cleaning up resources...")
    
    # Shutdown unified audit service
    try:
        from tldw_Server_API.app.core.Audit.unified_audit_service import shutdown_audit_service
        await shutdown_audit_service()
        logger.info("App Shutdown: Audit service shutdown complete")
    except Exception as e:
        logger.error(f"App Shutdown: Error shutting down audit service: {e}")
    
    # Close auth database pool
    try:
        if 'db_pool' in locals():
            await db_pool.close()
            logger.info("App Shutdown: Auth database pool closed")
    except Exception as e:
        logger.error(f"App Shutdown: Error closing auth database pool: {e}")
    
    # Shutdown session manager
    try:
        if 'session_manager' in locals():
            await session_manager.shutdown()
            logger.info("App Shutdown: Session manager shutdown")
    except Exception as e:
        logger.error(f"App Shutdown: Error shutting down session manager: {e}")
    
    # Shutdown MCP Unified server
    try:
        if 'mcp_server' in locals():
            await mcp_server.shutdown()
            logger.info("App Shutdown: MCP Unified server shutdown")
    except Exception as e:
        logger.error(f"App Shutdown: Error shutting down MCP Unified server: {e}")
    
    # Shutdown TTS Service
    try:
        from tldw_Server_API.app.core.TTS.tts_generation import close_tts_resources
        await close_tts_resources()
        logger.info("App Shutdown: TTS service shutdown complete")
    except Exception as e:
        logger.error(f"App Shutdown: Error shutting down TTS service: {e}")
    
    # Shutdown Chat Module Components
    logger.info("App Shutdown: Cleaning up Chat module components...")
    
    # Shutdown Provider Manager
    try:
        if 'provider_manager' in locals():
            await provider_manager.stop_health_checks()
            logger.info("App Shutdown: Provider manager stopped")
    except Exception as e:
        logger.error(f"App Shutdown: Error stopping provider manager: {e}")
    
    # Shutdown Request Queue
    try:
        if 'request_queue' in locals():
            await request_queue.stop()
            logger.info("App Shutdown: Request queue stopped")
    except Exception as e:
        logger.error(f"App Shutdown: Error stopping request queue: {e}")
    
    # Shutdown Audit Logger
    try:
        if 'audit_logger' in locals():
            await audit_logger.stop()
            logger.info("App Shutdown: Audit logger stopped")
    except Exception as e:
        logger.error(f"App Shutdown: Error stopping audit logger: {e}")
    
    # Cleanup CPU pools
    try:
        from tldw_Server_API.app.core.Utils.cpu_bound_handler import cleanup_pools
        cleanup_pools()
        logger.info("App Shutdown: CPU pools cleaned up")
    except Exception as e:
        logger.error(f"App Shutdown: Error cleaning up CPU pools: {e}")
    
    # Shutdown telemetry
    try:
        shutdown_telemetry()
        logger.info("App Shutdown: Telemetry shutdown")
    except Exception as e:
        logger.error(f"App Shutdown: Error shutting down telemetry: {e}")
    
    # Original test DB cleanup
    global test_db_instance_ref
    if test_db_instance_ref and hasattr(test_db_instance_ref, 'close_all_connections'):
        logger.info("App Shutdown: Closing test DB connections")
        test_db_instance_ref.close_all_connections()
    else:
        logger.info("App Shutdown: No test DB instance found to close")
#
############################# End of Test DB Handling###################

# Create FastAPI app with lifespan
app = FastAPI(
    title="tldw API",
    version="0.0.1",
    description="Version 0.0.1: Smooth Slide | FastAPI Backend for the tldw project",
    lifespan=lifespan
)

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

# Add CSRF Protection Middleware (NEW)
from tldw_Server_API.app.core.AuthNZ.csrf_protection import add_csrf_protection
add_csrf_protection(app)

# Static files serving
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Favicon serving
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(FAVICON_PATH, media_type="image/x-icon")

@app.get("/")
async def root():
    return {"message": "Welcome to the tldw API; If you're seeing this, the server is running!"}

# Metrics endpoint for Prometheus scraping
@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Prometheus metrics endpoint."""
    from fastapi.responses import PlainTextResponse
    
    registry = get_metrics_registry()
    metrics_text = registry.export_prometheus_format()
    
    return PlainTextResponse(metrics_text, media_type="text/plain; version=0.0.4")

# OpenTelemetry metrics endpoint (if using OTLP)
@app.get("/api/v1/metrics", tags=["monitoring"])
@track_metrics(labels={"endpoint": "metrics"})
async def api_metrics():
    """Get current metrics in JSON format."""
    registry = get_metrics_registry()
    return registry.get_all_metrics()

# Router for health monitoring endpoints (NEW)
from tldw_Server_API.app.api.v1.endpoints.health import router as health_router
app.include_router(health_router, prefix=f"{API_V1_PREFIX}", tags=["health"])

# Router for authentication endpoints (NEW)
app.include_router(auth_router, prefix=f"{API_V1_PREFIX}", tags=["authentication"])

# Router for user management endpoints (NEW)
app.include_router(users_router, prefix=f"{API_V1_PREFIX}", tags=["users"])

# Router for admin endpoints (NEW)
from tldw_Server_API.app.api.v1.endpoints.admin import router as admin_router
app.include_router(admin_router, prefix=f"{API_V1_PREFIX}", tags=["admin"])

# Router for media endpoints/media file handling
app.include_router(media_router, prefix=f"{API_V1_PREFIX}/media", tags=["media"])

# Router for /audio/ endpoints
app.include_router(audio_router, prefix=f"{API_V1_PREFIX}/audio", tags=["audio"])


# Router for chat endpoints/chat temp-file handling
app.include_router(chat_router, prefix=f"{API_V1_PREFIX}/chat", tags=["chat"])


# Router for chat endpoints/chat temp-file handling
app.include_router(character_router, prefix=f"{API_V1_PREFIX}/characters", tags=["character, persona"])


# Router for metrics endpoints
app.include_router(metrics_router, prefix=f"{API_V1_PREFIX}", tags=["metrics"])


# Router for Chunking Endpoint
app.include_router(chunking_router, prefix=f"{API_V1_PREFIX}/chunking", tags=["chunking"])

# Router for Chunking Templates
app.include_router(chunking_templates_router, prefix=f"{API_V1_PREFIX}", tags=["chunking templates"])


# Router for Embedding Endpoint (OpenAI-compatible path)
app.include_router(embeddings_router, prefix=f"{API_V1_PREFIX}", tags=["embeddings"])


# Router for Note Management endpoints
app.include_router(notes_router, prefix=f"{API_V1_PREFIX}/notes", tags=["notes"])


# Router for Prompt Management endpoints
app.include_router(prompt_router, prefix=f"{API_V1_PREFIX}/prompts", tags=["prompts"])


# Router for Prompt Studio endpoints
app.include_router(prompt_studio_projects_router, tags=["Prompt Studio"])
app.include_router(prompt_studio_prompts_router, tags=["Prompt Studio"])
app.include_router(prompt_studio_test_cases_router, tags=["Prompt Studio"])
app.include_router(prompt_studio_optimization_router, tags=["Prompt Studio"])
app.include_router(prompt_studio_evaluations_router, tags=["Prompt Studio"])
app.include_router(prompt_studio_websocket_router, tags=["Prompt Studio"])


# Router for RAG endpoints
# RAG API - Production API using functional pipeline
app.include_router(rag_api_router, tags=["RAG"])


# Router for Research endpoint
app.include_router(research_router, prefix=f"{API_V1_PREFIX}/research", tags=["research"])


# Router for Evaluation endpoint
app.include_router(evaluation_router, prefix=f"{API_V1_PREFIX}", tags=["evaluations"])

# Router for OpenAI-compatible Evaluation endpoint (NEW)
app.include_router(openai_evals_router, prefix=f"{API_V1_PREFIX}", tags=["evaluations"])

# Router for Benchmark endpoint (NEW)
app.include_router(benchmark_router, prefix=f"{API_V1_PREFIX}", tags=["benchmarks"])

# Router for Configuration Info endpoint (for documentation)
from tldw_Server_API.app.api.v1.endpoints.config_info import router as config_info_router
app.include_router(config_info_router, prefix=f"{API_V1_PREFIX}", tags=["config"])

# Router for Sync endpoint
app.include_router(sync_router, prefix=f"{API_V1_PREFIX}/sync", tags=["sync"])


# Router for Tools endpoint
app.include_router(tools_router, prefix=f"{API_V1_PREFIX}/tools", tags=["tools"])


# Router for MCP Unified (Secure, production-ready implementation)
app.include_router(mcp_unified_router, prefix=f"{API_V1_PREFIX}", tags=["MCP Unified"])
# Note: Old MCP routers have been archived due to security vulnerabilities

# Router for Chatbooks - import/export functionality
app.include_router(chatbooks_router, prefix=f"{API_V1_PREFIX}", tags=["chatbooks"])

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
## Entry point for running the server
########################################################################################################################
def run_server():
    """Run the FastAPI server using uvicorn."""
    import uvicorn
    uvicorn.run(
        "tldw_Server_API.app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    run_server()

#
## End of main.py
########################################################################################################################
