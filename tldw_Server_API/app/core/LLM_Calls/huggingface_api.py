# huggingface_api.py
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
                
                # Filter by path if specified
                if path:
                    files = [f for f in files if f.get("path", "").startswith(path)]
                
                # Filter for GGUF files
                gguf_files = [f for f in files if f.get("path", "").endswith(".gguf")]
                
                return gguf_files
            except httpx.HTTPError as e:
                logger.error(f"Error listing files for {repo_id}: {e}")
                return []
    
    async def get_download_url(self, repo_id: str, filename: str, revision: str = "main") -> Optional[str]:
        """
        Get the download URL for a specific file.
        
        Args:
            repo_id: Repository ID
            filename: Name of the file to download
            revision: Git revision (branch, tag, or commit)
            
        Returns:
            Download URL or None if error
        """
        return f"{self.BASE_URL}/{repo_id}/resolve/{revision}/{filename}"
    
    async def download_file(
        self,
        repo_id: str,
        filename: str,
        destination: Path,
        revision: str = "main",
        progress_callback: Optional[Callable[[int, int], None]] = None,
        chunk_size: int = 8192
    ) -> bool:
        """
        Download a file from a HuggingFace repository.
        
        Args:
            repo_id: Repository ID
            filename: Name of the file to download
            destination: Destination path for the downloaded file
            revision: Git revision
            progress_callback: Optional callback for progress updates (downloaded_bytes, total_bytes)
            chunk_size: Download chunk size in bytes
            
        Returns:
            True if successful, False otherwise
        """
        url = await self.get_download_url(repo_id, filename, revision)
        if not url:
            return False
        
        # Ensure destination directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        async with httpx.AsyncClient() as client:
            try:
                # Get file size first
                head_response = await client.head(url, headers=self.headers, follow_redirects=True)
                total_size = int(head_response.headers.get("content-length", 0))
                
                # Check if file already exists and has the right size
                if destination.exists() and destination.stat().st_size == total_size:
                    logger.info(f"File {filename} already exists with correct size, skipping download")
                    return True
                
                # Download with progress
                downloaded = 0
                temp_file = destination.with_suffix(".tmp")
                
                async with client.stream("GET", url, headers=self.headers, follow_redirects=True) as response:
                    response.raise_for_status()
                    
                    with open(temp_file, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size):
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            if progress_callback:
                                progress_callback(downloaded, total_size)
                
                # Move temp file to final destination
                temp_file.rename(destination)
                logger.info(f"Successfully downloaded {filename} to {destination}")
                return True
                
            except httpx.HTTPError as e:
                logger.error(f"Error downloading {filename}: {e}")
                # Clean up temp file if exists
                if temp_file.exists():
                    temp_file.unlink()
                return False
            except Exception as e:
                logger.error(f"Unexpected error downloading {filename}: {e}")
                if temp_file.exists():
                    temp_file.unlink()
                return False
    
    async def get_model_readme(self, repo_id: str) -> Optional[str]:
        """
        Get the README content for a model.
        
        Args:
            repo_id: Repository ID
            
        Returns:
            README content as string or None if not found
        """
        url = f"{self.BASE_URL}/{repo_id}/raw/main/README.md"
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=self.headers, timeout=30.0)
                response.raise_for_status()
                return response.text
            except httpx.HTTPError:
                # Try without .md extension
                try:
                    url = f"{self.BASE_URL}/{repo_id}/raw/main/README"
                    response = await client.get(url, headers=self.headers, timeout=30.0)
                    response.raise_for_status()
                    return response.text
                except httpx.HTTPError as e:
                    logger.debug(f"No README found for {repo_id}: {e}")
                    return None
    
    async def get_model_config(self, repo_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the config.json for a model.
        
        Args:
            repo_id: Repository ID
            
        Returns:
            Config dictionary or None if not found
        """
        url = f"{self.BASE_URL}/{repo_id}/raw/main/config.json"
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=self.headers, timeout=30.0)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                logger.debug(f"No config.json found for {repo_id}: {e}")
                return None
    
    async def search_gguf_models(
        self,
        query: str = "",
        model_type: Optional[str] = None,
        size_range: Optional[tuple[int, int]] = None,
        quantization: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search specifically for GGUF models with additional filters.
        
        Args:
            query: Search query
            model_type: Filter by model type (e.g., "llama", "mistral", "mixtral")
            size_range: Tuple of (min_size_gb, max_size_gb)
            quantization: Quantization type (e.g., "Q4_K_M", "Q5_K_S")
            limit: Maximum results
            
        Returns:
            List of matching GGUF models
        """
        # Build filter tags
        filter_tags = ["gguf"]
        if model_type:
            filter_tags.append(model_type.lower())
        
        # Search models
        models = await self.search_models(
            query=query,
            filter_tags=filter_tags,
            sort="downloads",
            limit=limit * 2  # Get more to filter
        )
        
        # Further filter results
        filtered_models = []
        for model in models:
            # Get model files to check sizes and quantization
            if quantization or size_range:
                files = await self.list_model_files(model["modelId"])
                
                # Check quantization
                if quantization:
                    has_quant = any(quantization.lower() in f.get("path", "").lower() for f in files)
                    if not has_quant:
                        continue
                
                # Check size range
                if size_range and files:
                    # Get total size of GGUF files
                    total_size_bytes = sum(f.get("size", 0) for f in files)
                    total_size_gb = total_size_bytes / (1024 ** 3)
                    
                    if not (size_range[0] <= total_size_gb <= size_range[1]):
                        continue
            
            filtered_models.append(model)
            
            if len(filtered_models) >= limit:
                break
        
        return filtered_models


# Utility functions for common operations
async def find_best_gguf_model(
    model_name: str,
    max_size_gb: float = 10.0,
    preferred_quant: Optional[str] = "Q4_K_M"
) -> Optional[Dict[str, Any]]:
    """
    Find the best GGUF version of a model based on criteria.
    
    Args:
        model_name: Name of the model to search for
        max_size_gb: Maximum model size in GB
        preferred_quant: Preferred quantization type
        
    Returns:
        Best matching model info or None
    """
    api = HuggingFaceAPI()
    
    # Search for the model
    models = await api.search_gguf_models(
        query=model_name,
        size_range=(0, max_size_gb),
        quantization=preferred_quant,
        limit=5
    )
    
    if not models:
        # Try without quantization preference
        models = await api.search_gguf_models(
            query=model_name,
            size_range=(0, max_size_gb),
            limit=5
        )
    
    # Return the most downloaded one
    return models[0] if models else None


async def download_gguf_model(
    repo_id: str,
    model_file: str,
    destination_dir: Path,
    show_progress: bool = True
) -> bool:
    """
    Download a GGUF model file with progress indication.
    
    Args:
        repo_id: HuggingFace repository ID
        model_file: Name of the GGUF file
        destination_dir: Directory to save the model
        show_progress: Whether to show download progress
        
    Returns:
        True if successful
    """
    api = HuggingFaceAPI()
    
    destination = destination_dir / model_file
    
    def progress_callback(downloaded: int, total: int):
        if show_progress and total > 0:
            percent = (downloaded / total) * 100
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total / (1024 * 1024)
            print(f"\rDownloading: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="")
    
    success = await api.download_file(
        repo_id=repo_id,
        filename=model_file,
        destination=destination,
        progress_callback=progress_callback if show_progress else None
    )
    
    if show_progress:
        print()  # New line after progress
    
    return success