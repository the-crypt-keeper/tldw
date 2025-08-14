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

async def verify_token(Token: str = Header(None)):  # Token is the API key itself
    if not Token:  # FastAPI will pass None if header is missing
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing authentication token.")

    if settings.get("SINGLE_USER_MODE"):
        expected_token = settings.get("SINGLE_USER_API_KEY")
        if not expected_token:  # This means settings are not properly loaded or key is missing
            logger.critical("SINGLE_USER_API_KEY is not configured in settings for single-user mode.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Server authentication misconfigured (API key missing).")

        # Direct comparison, no "Bearer " prefix stripping
        if Token != expected_token:
            logger.warning(f"Invalid token received. Expected: '{expected_token[:5]}...', Got: '{Token[:10]}...'")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token.")
    else:
        # Multi-user mode: Validate JWT token
        jwt_service = get_jwt_service()
        try:
            # In multi-user mode, the token should be a JWT (not prefixed with "Bearer ")
            # since this is a header token, not from the Authorization header
            payload = jwt_service.decode_access_token(Token)
            
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
