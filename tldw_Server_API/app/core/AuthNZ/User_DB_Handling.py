# User_DB_Handling.py
# Description: Handles user authentication and identification based on application mode.
#
# Imports
from typing import Optional
#
# 3rd-Party Libraries
from fastapi import Depends, HTTPException, status, Header
from pydantic import BaseModel, ValidationError
#
# Local Imports
# New unified settings
from tldw_Server_API.app.core.AuthNZ.settings import get_settings, is_single_user_mode, is_multi_user_mode
# New JWT service
from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service
from tldw_Server_API.app.core.AuthNZ.exceptions import InvalidTokenError, TokenExpiredError
# Utils
from loguru import logger
# API Dependencies
from tldw_Server_API.app.api.v1.API_Deps.v1_endpoint_deps import oauth2_scheme

#######################################################################################################################

# --- User Model ---
# Standardized User object, used even for the dummy single user.
class User(BaseModel):
    id: int
    username: str
    email: Optional[str] = None
    is_active: bool = True

# --- Single User "Dummy" Object ---
# Created when in single-user mode using values from the settings
_single_user_instance = None

def get_single_user_instance() -> User:
    """Get or create the single user instance"""
    global _single_user_instance
    if _single_user_instance is None:
        settings = get_settings()
        _single_user_instance = User(
            id=settings.SINGLE_USER_FIXED_ID,
            username="single_user",
            is_active=True
        )
    return _single_user_instance

#######################################################################################################################

# --- Mode-Specific Verification Dependencies ---

async def verify_single_user_api_key(api_key: str = Header(..., alias="X-API-KEY")):
    """
    Dependency to verify the fixed API key in single-user mode.
    Uses the unified settings system.
    """
    # Check mode using the helper function
    if not is_single_user_mode():
         logger.error("verify_single_user_api_key called unexpectedly in multi-user mode.")
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Configuration error")

    # Compare with the API key from settings
    settings = get_settings()
    if api_key != settings.SINGLE_USER_API_KEY:
        logger.warning(f"Invalid API Key received in single-user mode: '{api_key[:5]}...'")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key"
        )
    logger.debug("Single-user API Key verified successfully.")
    # Return value doesn't strictly matter for a verification dependency
    return True


async def verify_jwt_and_fetch_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    Dependency to verify JWT and fetch user details in multi-user mode.
    Uses the new JWT service for token validation.
    """
    # Check mode using the helper function
    if is_single_user_mode():
         logger.error("verify_jwt_and_fetch_user called unexpectedly in single-user mode.")
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Configuration error")

    # Import Users_DB here to avoid import errors in single-user mode
    try:
        from tldw_Server_API.app.core.DB_Management.Users_DB import get_user_by_id, UserNotFoundError
    except ImportError:
        logger.error("Multi-user mode requires Users_DB module, but it's not available.")
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Multi-user mode requires Users_DB implementation."
        )

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # Use new JWT service to decode token
    jwt_service = get_jwt_service()
    try:
        payload = jwt_service.decode_access_token(token)
        user_id = payload.get("user_id") or payload.get("sub")  # Handle both formats
        if not user_id:
            logger.warning("Token payload missing user_id/sub claim")
            raise credentials_exception
        # Convert to int if it's a string
        if isinstance(user_id, str):
            user_id = int(user_id)
    except (InvalidTokenError, TokenExpiredError) as e:
        logger.warning(f"Token validation failed: {e}")
        raise credentials_exception
    except Exception as e:
        logger.error(f"Unexpected error decoding token: {e}")
        raise credentials_exception
    
    logger.debug(f"Token decoded successfully for user_id: {user_id}")

    # --- Fetch and Validate User Data ---
    user_data: Optional[dict] = None # Initialize to satisfy linters potentially
    try:
        user_data = await get_user_by_id(user_id) # Assume returns dict or None

        # --- Explicit Check for dictionary type ---
        if not isinstance(user_data, dict):
            # Log appropriately based on whether it was None or an unexpected type
            if user_data is None:
                 logger.warning(f"User with ID {user_id} from token not found in Users_DB.")
                 # Raise the standard credentials exception if user not found
                 raise credentials_exception
            else:
                 # This indicates an issue with the get_user_by_id implementation
                 logger.error(f"Data retrieved for user {user_id} is not a dictionary (type: {type(user_data)}).")
                 raise HTTPException(
                     status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                     detail="Internal error retrieving user data format."
                 )
        # --- If we reach here, user_data is guaranteed to be a dictionary ---

    except UserNotFoundError: # Catch specific exception if get_user_by_id raises it
        logger.warning(f"User with ID {user_id} from token not found in Users_DB (UserNotFoundError).")
        raise credentials_exception
    except Exception as e: # Catch other errors during DB fetch
        logger.error(f"Error fetching user {user_id} from Users_DB: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving user information."
        )

    # --- Create and validate the User Pydantic model ---
    try:
        # Now the IDE should be confident that user_data is a dictionary
        user = User(**user_data)
    except ValidationError as e: # Catch Pydantic validation errors specifically
        logger.error(f"Failed to validate user data for user {user_id} into User model: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing user data: Invalid format - {e}" # Include details
        )
    except Exception as e: # Catch other potential errors during model creation
        logger.error(f"Unexpected error creating User model for user {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error processing user data."
        )

    # --- Final User Status Check ---
    if not user.is_active:
        logger.warning(f"Authentication attempt by inactive user: {user.username} (ID: {user.id})")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")

    logger.info(f"Authenticated active user: {user.username} (ID: {user.id})")
    return user


# --- Combined Primary Authentication Dependency ---

async def get_request_user(
    api_key: Optional[str] = Header(None, alias="X-API-KEY"),
    token: Optional[str] = Depends(oauth2_scheme) # No need for use_cache=False if auto_error handles it
    ) -> User:
    """
    Determines the current user based on the application mode (single/multi)
    by checking the 'settings' dictionary.

    - In Single-User Mode: Verifies X-API-KEY from header against settings["SINGLE_USER_API_KEY"]
      and returns a fixed User object (_single_user_instance).
    - In Multi-User Mode: Verifies the Bearer token (passed via 'token' parameter)
      and returns the User object fetched from Users_DB.
    """
    #print(f"DEBUGPRINT: Inside get_request_user. api_key from header: '{api_key}', token from scheme: '{token}'") #DEBUGPRINT
    # Check mode from the settings
    settings = get_settings()
    logger.debug(f"Authentication mode: {'single_user' if is_single_user_mode() else 'multi_user'} (AUTH_MODE={settings.AUTH_MODE})")
    if is_single_user_mode():
        # Single-User Mode: X-API-KEY is primary.
        # The 'token' parameter from oauth2_scheme will likely be None here, which is fine.
        logger.debug("get_request_user: In SINGLE_USER_MODE.")
        if api_key is None:
            logger.warning("Single-User Mode: X-API-KEY header is missing or not resolved by FastAPI.")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="X-API-KEY header required for single-user mode"
            )
        if api_key != settings.SINGLE_USER_API_KEY:
            logger.warning(
                f"Single-User Mode: Invalid X-API-KEY. Got: '{api_key[:10]}...'")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid X-API-KEY"
            )
        logger.debug("Single-user API Key verified. Returning fixed user object.")
        return get_single_user_instance()  # Use the getter function
    else:
        # Multi-User Mode: Bearer token is primary.
        # 'api_key' might be present or None, but we ignore it in multi-user mode if 'token' is valid.
        logger.debug("get_request_user: In MULTI_USER_MODE.")
        if token is None:
            # This condition is now critical because auto_error=False on oauth2_scheme
            logger.warning("Multi-User Mode: Authorization Bearer token is missing (token is None).")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated (Bearer token required for multi-user mode)",
                headers={"WWW-Authenticate": "Bearer"},
            )
        # If token is present, proceed to verify it.
        # verify_jwt_and_fetch_user should handle the actual decoding and user lookup.
        logger.debug(f"Multi-User Mode: Attempting to verify token: '{token[:15]}...'")
        return await verify_jwt_and_fetch_user(token)



#
# End of User_DB_Handling.py
#######################################################################################################################
