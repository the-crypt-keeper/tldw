"""
Base provider class for all LLM API implementations.

This module defines the abstract base class that all LLM providers must implement,
ensuring consistency across different API integrations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Generator
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from loguru import logger

from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAPIError, ChatAuthenticationError, ChatConfigurationError,
    ChatBadRequestError, ChatRateLimitError, ChatProviderError
)
from ..security import get_key_manager, validate_api_request, ValidationError


class ProviderType(Enum):
    """Types of LLM providers."""
    COMMERCIAL = "commercial"  # OpenAI, Anthropic, etc.
    LOCAL = "local"  # Llama.cpp, Ollama, etc.
    CUSTOM = "custom"  # Custom implementations


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    name: str
    type: ProviderType
    api_base_url: str
    api_key_required: bool = True
    supports_streaming: bool = True
    supports_functions: bool = False
    supports_vision: bool = False
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 90.0
    max_tokens_limit: Optional[int] = None
    default_model: Optional[str] = None
    custom_headers: Optional[Dict[str, str]] = None


@dataclass
class APIResponse:
    """Standardized response from LLM providers."""
    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    created_at: datetime = None
    response_time: float = 0.0
    raw_response: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class BaseProvider(ABC):
    """
    Abstract base class for all LLM providers.
    
    This class provides common functionality and defines the interface
    that all provider implementations must follow.
    """
    
    def __init__(self, config: ProviderConfig):
        """
        Initialize the provider with configuration.
        
        Args:
            config: Provider configuration
        """
        self.config = config
        self.key_manager = get_key_manager()
        self._session = None
        self._metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0,
            'total_response_time': 0.0,
        }
    
    def get_session(self) -> requests.Session:
        """
        Get or create a requests session with retry configuration.
        
        Returns:
            Configured requests session
        """
        if self._session is None:
            self._session = requests.Session()
            
            # Configure retry strategy
            retry_strategy = Retry(
                total=self.config.max_retries,
                backoff_factor=self.config.retry_delay,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["POST", "GET"]
            )
            
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)
        
        return self._session
    
    def get_api_key(self, api_key: Optional[str] = None) -> Optional[str]:
        """
        Get API key for the provider.
        
        Args:
            api_key: Optional API key provided by user
            
        Returns:
            API key if available
            
        Raises:
            ChatAuthenticationError: If API key is required but not found
        """
        if not self.config.api_key_required:
            return api_key
        
        key = self.key_manager.get_api_key(self.config.name, api_key)
        
        if not key:
            raise ChatAuthenticationError(
                provider=self.config.name,
                message=f"{self.config.name} API key is required but not found"
            )
        
        return key
    
    def validate_request(self, **kwargs) -> Dict[str, Any]:
        """
        Validate and sanitize request parameters.
        
        Args:
            **kwargs: Request parameters
            
        Returns:
            Validated parameters
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            return validate_api_request(**kwargs)
        except ValidationError as e:
            raise ChatBadRequestError(
                provider=self.config.name,
                message=f"Invalid request: {str(e)}"
            )
    
    def build_headers(self, api_key: Optional[str] = None) -> Dict[str, str]:
        """
        Build request headers for the provider.
        
        Args:
            api_key: API key for authentication
            
        Returns:
            Dictionary of headers
        """
        headers = {
            'Content-Type': 'application/json',
        }
        
        # Add authentication if required
        if self.config.api_key_required and api_key:
            headers.update(self._get_auth_headers(api_key))
        
        # Add custom headers if configured
        if self.config.custom_headers:
            headers.update(self.config.custom_headers)
        
        return headers
    
    @abstractmethod
    def _get_auth_headers(self, api_key: str) -> Dict[str, str]:
        """
        Get authentication headers for the provider.
        
        Args:
            api_key: API key for authentication
            
        Returns:
            Dictionary of auth headers
        """
        pass
    
    @abstractmethod
    def _build_request_payload(self, **kwargs) -> Dict[str, Any]:
        """
        Build the request payload for the provider's API.
        
        Args:
            **kwargs: Request parameters
            
        Returns:
            Request payload dictionary
        """
        pass
    
    @abstractmethod
    def _parse_response(self, response: Dict[str, Any]) -> APIResponse:
        """
        Parse the provider's response into standardized format.
        
        Args:
            response: Raw response from provider
            
        Returns:
            Standardized APIResponse
        """
        pass
    
    @abstractmethod
    def _parse_streaming_response(self, line: str) -> Optional[str]:
        """
        Parse a streaming response line.
        
        Args:
            line: Single line from streaming response
            
        Returns:
            Parsed content or None if no content
        """
        pass
    
    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        streaming: bool = False,
        **kwargs
    ) -> Union[APIResponse, Generator[str, None, None]]:
        """
        Send a chat completion request to the provider.
        
        Args:
            messages: List of message objects
            model: Model to use
            api_key: Optional API key
            streaming: Whether to stream the response
            **kwargs: Additional provider-specific parameters
            
        Returns:
            APIResponse or generator for streaming
            
        Raises:
            ChatAPIError: If the request fails
        """
        start_time = time.time()
        self._metrics['total_requests'] += 1
        
        try:
            # Validate request
            validated_params = self.validate_request(
                messages=messages,
                model=model,
                **kwargs
            )
            
            # Get API key
            api_key = self.get_api_key(api_key)
            
            # Build request
            headers = self.build_headers(api_key)
            payload = self._build_request_payload(
                messages=validated_params['messages'],
                model=model or self.config.default_model,
                **validated_params
            )
            
            # Log request (without sensitive data)
            logger.debug(f"{self.config.name}: Sending request to {self.config.api_base_url}")
            logger.debug(f"{self.config.name}: Model: {model or self.config.default_model}")
            
            # Make request
            if streaming and self.config.supports_streaming:
                return self._handle_streaming_request(headers, payload)
            else:
                return self._handle_regular_request(headers, payload, start_time)
        
        except ChatAPIError:
            self._metrics['failed_requests'] += 1
            raise
        except Exception as e:
            self._metrics['failed_requests'] += 1
            logger.error(f"{self.config.name}: Unexpected error: {e}")
            raise ChatProviderError(
                provider=self.config.name,
                message=f"Unexpected error: {str(e)}"
            )
    
    def _handle_regular_request(
        self, 
        headers: Dict[str, str], 
        payload: Dict[str, Any],
        start_time: float
    ) -> APIResponse:
        """
        Handle a non-streaming request.
        
        Args:
            headers: Request headers
            payload: Request payload
            start_time: Request start time
            
        Returns:
            APIResponse
        """
        session = self.get_session()
        
        try:
            response = session.post(
                self.config.api_base_url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            # Parse response
            api_response = self._parse_response(response_data)
            api_response.response_time = time.time() - start_time
            
            # Update metrics
            self._metrics['successful_requests'] += 1
            self._metrics['total_response_time'] += api_response.response_time
            if api_response.usage:
                self._metrics['total_tokens'] += api_response.usage.get('total_tokens', 0)
            
            # Audit log
            self.key_manager.audit_log(
                provider=self.config.name,
                action='chat_completion',
                success=True,
                metadata={
                    'model': api_response.model,
                    'response_time': api_response.response_time,
                    'tokens': api_response.usage
                }
            )
            
            return api_response
        
        except requests.exceptions.HTTPError as e:
            self._handle_http_error(e)
        except requests.exceptions.RequestException as e:
            logger.error(f"{self.config.name}: Request failed: {e}")
            raise ChatProviderError(
                provider=self.config.name,
                message=f"Request failed: {str(e)}"
            )
    
    def _handle_streaming_request(
        self,
        headers: Dict[str, str],
        payload: Dict[str, Any]
    ) -> Generator[str, None, None]:
        """
        Handle a streaming request.
        
        Args:
            headers: Request headers
            payload: Request payload
            
        Yields:
            Streaming response chunks
        """
        session = self.get_session()
        payload['stream'] = True
        
        try:
            response = session.post(
                self.config.api_base_url,
                headers=headers,
                json=payload,
                stream=True,
                timeout=self.config.timeout
            )
            
            response.raise_for_status()
            
            for line in response.iter_lines(decode_unicode=True):
                if line and line.strip():
                    content = self._parse_streaming_response(line)
                    if content:
                        yield content
            
            # Send done signal
            yield "data: [DONE]\n\n"
            
            self._metrics['successful_requests'] += 1
            
        except requests.exceptions.HTTPError as e:
            self._handle_http_error(e)
        except Exception as e:
            logger.error(f"{self.config.name}: Streaming error: {e}")
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
            yield "data: [DONE]\n\n"
    
    def _handle_http_error(self, error: requests.exceptions.HTTPError):
        """
        Handle HTTP errors from the provider.
        
        Args:
            error: HTTP error exception
            
        Raises:
            Appropriate ChatAPIError subclass
        """
        if error.response is None:
            raise ChatProviderError(
                provider=self.config.name,
                message="No response from provider"
            )
        
        status_code = error.response.status_code
        
        # Try to parse error message from response
        try:
            error_data = error.response.json()
            error_message = (
                error_data.get('error', {}).get('message') or
                error_data.get('message') or
                str(error)
            )
        except:
            error_message = error.response.text or str(error)
        
        # Log the error
        logger.error(f"{self.config.name}: HTTP {status_code}: {error_message}")
        
        # Audit log
        self.key_manager.audit_log(
            provider=self.config.name,
            action='chat_completion',
            success=False,
            error=f"HTTP {status_code}: {error_message}"
        )
        
        # Map to appropriate exception
        if status_code == 401:
            raise ChatAuthenticationError(
                provider=self.config.name,
                message=error_message
            )
        elif status_code == 429:
            raise ChatRateLimitError(
                provider=self.config.name,
                message=error_message
            )
        elif status_code == 400:
            raise ChatBadRequestError(
                provider=self.config.name,
                message=error_message
            )
        else:
            raise ChatProviderError(
                provider=self.config.name,
                message=error_message,
                status_code=status_code
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get provider metrics.
        
        Returns:
            Dictionary of metrics
        """
        metrics = self._metrics.copy()
        
        # Calculate averages
        if metrics['successful_requests'] > 0:
            metrics['avg_response_time'] = (
                metrics['total_response_time'] / metrics['successful_requests']
            )
            metrics['avg_tokens'] = (
                metrics['total_tokens'] / metrics['successful_requests']
            )
        else:
            metrics['avg_response_time'] = 0.0
            metrics['avg_tokens'] = 0
        
        # Add provider info
        metrics['provider'] = self.config.name
        metrics['provider_type'] = self.config.type.value
        
        return metrics
    
    def reset_metrics(self):
        """Reset provider metrics."""
        self._metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0,
            'total_response_time': 0.0,
        }
        logger.info(f"{self.config.name}: Metrics reset")
    
    def __str__(self) -> str:
        """String representation of provider."""
        return f"{self.__class__.__name__}({self.config.name})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"{self.__class__.__name__}("
            f"name={self.config.name}, "
            f"type={self.config.type.value}, "
            f"url={self.config.api_base_url})"
        )