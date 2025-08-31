# v1-endpoint-deps.py
# Description: This file is to serve as a sink for dependencies across the v1 endpoints.
# Imports
#
# 3rd-party Libraries
from fastapi import Header, HTTPException
from fastapi.security import OAuth2PasswordBearer
from loguru import logger
from starlette import status

from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service
from tldw_Server_API.app.core.AuthNZ.exceptions import InvalidTokenError, TokenExpiredError

#
# Local Imports
#
#######################################################################################################################
#
# Static Variables
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)
#
# Functions:

async def verify_token(
    Token: str = Header(None),
    x_api_key: str = Header(None, alias="X-API-KEY")
):  # Check both Token and X-API-KEY headers
    # In single-user mode, check X-API-KEY; in multi-user mode, check Token
    if settings.get("SINGLE_USER_MODE"):
        # Single-user mode uses X-API-KEY header
        auth_token = x_api_key or Token  # Check X-API-KEY first, fallback to Token
        if not auth_token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing authentication token (X-API-KEY required).")
    else:
        # Multi-user mode uses Token header (JWT)
        auth_token = Token
        if not auth_token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing authentication token.")

    if settings.get("SINGLE_USER_MODE"):
        # Get the API key from AuthNZ settings which is the canonical source
        from tldw_Server_API.app.core.AuthNZ.settings import get_settings as get_auth_settings
        auth_settings = get_auth_settings()
        expected_token = auth_settings.SINGLE_USER_API_KEY
        if not expected_token:  # This means settings are not properly loaded or key is missing
            logger.critical("SINGLE_USER_API_KEY is not configured in settings for single-user mode.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Server authentication misconfigured (API key missing).")

        # Direct comparison, no "Bearer " prefix stripping
        if auth_token != expected_token:
            logger.warning(f"Invalid token received. Expected: '{expected_token[:5]}...', Got: '{auth_token[:10]}...'")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token.")
    else:
        # Multi-user mode: Validate JWT token
        jwt_service = get_jwt_service()
        try:
            # In multi-user mode, the token should be a JWT (not prefixed with "Bearer ")
            # since this is a header token, not from the Authorization header
            payload = jwt_service.decode_access_token(auth_token)
            
            # Check if token has valid user information
            user_id = payload.get("user_id") or payload.get("sub")
            if not user_id:
                logger.warning("JWT token missing user_id/sub claim")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token: missing user information"
                )
            
            logger.debug(f"JWT token validated for user_id: {user_id}")
            
        except (InvalidTokenError, TokenExpiredError) as e:
            logger.warning(f"JWT validation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid or expired token: {e}"
            )
        except Exception as e:
            logger.error(f"Unexpected error validating JWT: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error validating authentication token"
            )

    return True


#
# End of v1-endpoint-deps.py
#######################################################################################################################
