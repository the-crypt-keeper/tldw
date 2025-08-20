# tldw_chatbook/LLM_Calls/huggingface_api.py
"""
HuggingFace API client for browsing and downloading GGUF models.
"""

import os
import asyncio
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import httpx
from loguru import logger
import json


class HuggingFaceAPI:
    """Client for interacting with HuggingFace API."""
    
    BASE_URL = "https://huggingface.co"
    API_BASE = f"{BASE_URL}/api"
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize HuggingFace API client.
        
        Args:
            token: Optional HuggingFace API token for private repos
        """
        self.token = token or os.environ.get("HUGGINGFACE_API_KEY", "")
        self.headers = {}
        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}"
    
    async def search_models(
        self, 
        query: str = "",
        filter_tags: Optional[List[str]] = None,
        sort: str = "downloads",
        limit: int = 50,
        full_search: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search for models on HuggingFace.
        
        Args:
            query: Search query string
            filter_tags: List of tags to filter by (e.g., ["gguf", "llama"])
            sort: Sort by "downloads", "likes", "lastModified"
            limit: Maximum number of results
            full_search: If True, search in model card content too
            
        Returns:
            List of model information dictionaries
        """
        params = {
            "limit": limit,
            "sort": sort,
            "direction": -1,  # Descending order
            "full": full_search
        }
        
        # Add search query
        if query:
            params["search"] = query
        
        # Build filter string
        filters = []
        if filter_tags:
            for tag in filter_tags:
                filters.append(tag)
        
        # Always filter for GGUF models
        filters.append("gguf")
        
        if filters:
            params["filter"] = filters
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.API_BASE}/models",
                    params=params,
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                logger.error(f"Error searching models: {e}")
                return []
    
    async def get_model_info(self, repo_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific model.
        
        Args:
            repo_id: Repository ID (e.g., "TheBloke/Llama-2-7B-GGUF")
            
        Returns:
            Model information dictionary or None if error
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.API_BASE}/models/{repo_id}",
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                logger.error(f"Error getting model info for {repo_id}: {e}")
                return None
    
    async def list_model_files(self, repo_id: str, path: str = "") -> List[Dict[str, Any]]:
        """
        List files in a model repository.
        
        Args:
            repo_id: Repository ID
            path: Path within repository (default is root)
            
        Returns:
            List of file information dictionaries
        """
        async with httpx.AsyncClient() as client:
            try:
                # Use the tree endpoint to get all files
                response = await client.get(
                    f"{self.API_BASE}/models/{repo_id}/tree/main",
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                files = response.json()
                
                # Check if files is a list
                if not isinstance(files, list):
                    logger.warning(f"Unexpected response format for {repo_id}: {type(files)}")
                    return []
                
                # Filter for GGUF files
                gguf_files = []
                for file in files:
                    if isinstance(file, dict) and file.get("path", "").endswith(".gguf"):
                        # Add human-readable size
                        size_bytes = file.get("size", 0)
                        file["size_human"] = self._format_bytes(size_bytes)
                        gguf_files.append(file)
                
                logger.info(f"Found {len(gguf_files)} GGUF files for {repo_id}")
                return gguf_files
            except httpx.HTTPError as e:
                logger.error(f"Error listing files for {repo_id}: {e}")
                return []
    
    async def get_model_readme(self, repo_id: str) -> Optional[str]:
        """
        Get the README content for a model.
        
        Args:
            repo_id: Repository ID
            
        Returns:
            README content as string or None if not found
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.BASE_URL}/{repo_id}/raw/main/README.md",
                    headers=self.headers,
                    timeout=30.0
                )
                response.raise_for_status()
                return response.text
            except httpx.HTTPError:
                # Try without .md extension
                try:
                    response = await client.get(
                        f"{self.BASE_URL}/{repo_id}/raw/main/README",
                        headers=self.headers,
                        timeout=30.0
                    )
                    response.raise_for_status()
                    return response.text
                except httpx.HTTPError as e:
                    logger.debug(f"No README found for {repo_id}: {e}")
                    return None
    
    async def download_file(
        self,
        repo_id: str,
        filename: str,
        destination: Path,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> bool:
        """
        Download a file from a model repository.
        
        Args:
            repo_id: Repository ID
            filename: Name of file to download
            destination: Path where to save the file
            progress_callback: Optional callback for progress updates (downloaded_bytes, total_bytes)
            
        Returns:
            True if successful, False otherwise
        """
        url = f"{self.BASE_URL}/{repo_id}/resolve/main/{filename}"
        
        # Ensure destination directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        async with httpx.AsyncClient() as client:
            try:
                # First get file size with HEAD request
                head_response = await client.head(url, headers=self.headers, follow_redirects=True)
                total_size = int(head_response.headers.get("content-length", 0))
                
                # Download with streaming
                downloaded = 0
                with open(destination, "wb") as f:
                    async with client.stream("GET", url, headers=self.headers, follow_redirects=True) as response:
                        response.raise_for_status()
                        
                        async for chunk in response.aiter_bytes(chunk_size=1024*1024):  # 1MB chunks
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            if progress_callback:
                                progress_callback(downloaded, total_size)
                
                logger.info(f"Successfully downloaded {filename} to {destination}")
                return True
                
            except httpx.HTTPError as e:
                logger.error(f"Error downloading {filename}: {e}")
                # Clean up partial download
                if destination.exists():
                    destination.unlink()
                return False
            except Exception as e:
                logger.error(f"Unexpected error downloading {filename}: {e}")
                if destination.exists():
                    destination.unlink()
                return False
    
    @staticmethod
    def _format_bytes(bytes: int) -> str:
        """Format bytes to human readable string."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes < 1024.0:
                return f"{bytes:.1f} {unit}"
            bytes /= 1024.0
        return f"{bytes:.1f} PB"