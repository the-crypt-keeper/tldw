# model_warmup.py
# Optional model warmup for faster first requests

import asyncio
import time
from typing import List, Dict, Any, Optional
from loguru import logger

from tldw_Server_API.app.core.Embeddings.simplified_config import get_config
from tldw_Server_API.app.core.Embeddings.metrics_integration import get_metrics


class ModelWarmup:
    """
    Handles optional model warmup to pre-load models for faster first requests.
    Disabled by default, can be enabled via configuration.
    """
    
    def __init__(self, config: Optional[Any] = None):
        """
        Initialize model warmup handler.
        
        Args:
            config: Optional configuration override
        """
        self.config = config or get_config()
        self.metrics = get_metrics()
        self.warmed_models: Dict[str, bool] = {}
        
    async def warmup_model(self, provider: str, model: str) -> bool:
        """
        Warmup a single model by loading it and generating a test embedding.
        
        Args:
            provider: Provider name
            model: Model name
            
        Returns:
            True if warmup successful, False otherwise
        """
        try:
            start_time = time.time()
            logger.info(f"Warming up model {model} from {provider}...")
            
            # Import here to avoid circular dependency
            from tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create import create_embedding
            
            # Generate a test embedding
            test_text = "Model warmup test"
            
            # Create config for this specific model
            test_config = {
                "embedding_config": {
                    "default_model_id": f"{provider}:{model}",
                    "models": {
                        f"{provider}:{model}": {
                            "provider": provider,
                            "model_name_or_path": model
                        }
                    }
                }
            }
            
            # Try to create an embedding
            _ = create_embedding(test_text, test_config, model_id_override=f"{provider}:{model}")
            
            elapsed = time.time() - start_time
            
            # Log success
            logger.info(f"Model {model} warmed up successfully in {elapsed:.2f}s")
            self.metrics.log_model_load(model, elapsed)
            
            # Mark as warmed
            model_key = f"{provider}:{model}"
            self.warmed_models[model_key] = True
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to warmup model {model} from {provider}: {e}")
            self.metrics.log_error(provider, "warmup_failed")
            return False
    
    async def warmup_provider_models(self, provider_name: str) -> int:
        """
        Warmup all models for a specific provider.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Number of models successfully warmed up
        """
        provider_config = self.config.get_provider(provider_name)
        
        if not provider_config or not provider_config.enabled:
            logger.warning(f"Provider {provider_name} not found or not enabled")
            return 0
        
        success_count = 0
        
        for model in provider_config.models:
            if await self.warmup_model(provider_name, model):
                success_count += 1
            
            # Small delay between models to avoid overwhelming the system
            await asyncio.sleep(0.5)
        
        return success_count
    
    async def warmup_priority_models(self) -> Dict[str, int]:
        """
        Warmup models based on configuration priority.
        Only warms up models explicitly listed in warmup_models config.
        
        Returns:
            Dictionary of provider -> number of models warmed
        """
        if not self.config.resources.enable_model_warmup:
            logger.info("Model warmup is disabled in configuration")
            return {}
        
        warmup_models = self.config.resources.warmup_models
        
        if not warmup_models:
            logger.info("No models configured for warmup")
            return {}
        
        logger.info(f"Starting warmup for {len(warmup_models)} models...")
        results = {}
        
        for model_spec in warmup_models:
            # Parse model specification (format: "provider:model" or just "model")
            if ":" in model_spec:
                provider, model = model_spec.split(":", 1)
            else:
                # Use default provider
                provider = self.config.default_provider
                model = model_spec
            
            # Warmup the model
            success = await self.warmup_model(provider, model)
            
            # Track results
            if provider not in results:
                results[provider] = 0
            if success:
                results[provider] += 1
        
        logger.info(f"Model warmup completed: {results}")
        return results
    
    async def warmup_all_enabled_providers(self) -> Dict[str, int]:
        """
        Warmup all models from all enabled providers.
        This is more aggressive and typically not recommended.
        
        Returns:
            Dictionary of provider -> number of models warmed
        """
        if not self.config.resources.enable_model_warmup:
            logger.info("Model warmup is disabled in configuration")
            return {}
        
        results = {}
        
        for provider in self.config.get_enabled_providers():
            count = await self.warmup_provider_models(provider.name)
            results[provider.name] = count
        
        return results
    
    def is_warmed(self, provider: str, model: str) -> bool:
        """
        Check if a model has been warmed up.
        
        Args:
            provider: Provider name
            model: Model name
            
        Returns:
            True if model has been warmed up
        """
        model_key = f"{provider}:{model}"
        return self.warmed_models.get(model_key, False)
    
    def get_warmup_status(self) -> Dict[str, Any]:
        """
        Get current warmup status.
        
        Returns:
            Dictionary with warmup statistics
        """
        total_configured = len(self.config.resources.warmup_models)
        total_warmed = len([v for v in self.warmed_models.values() if v])
        
        return {
            "enabled": self.config.resources.enable_model_warmup,
            "configured_models": total_configured,
            "warmed_models": total_warmed,
            "models": self.warmed_models
        }
    
    async def periodic_warmup(self, interval_seconds: int = 3600):
        """
        Periodically re-warm models to keep them in memory.
        
        Args:
            interval_seconds: Interval between warmup cycles
        """
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                
                if self.config.resources.enable_model_warmup:
                    logger.info("Starting periodic model warmup...")
                    await self.warmup_priority_models()
                    
            except Exception as e:
                logger.error(f"Error in periodic warmup: {e}")


# Global warmup handler
_warmup_handler: Optional[ModelWarmup] = None


def get_warmup_handler() -> ModelWarmup:
    """Get or create the global warmup handler."""
    global _warmup_handler
    if _warmup_handler is None:
        _warmup_handler = ModelWarmup()
    return _warmup_handler


async def warmup_on_startup():
    """
    Function to be called on application startup to warmup models.
    Only runs if warmup is enabled in configuration.
    """
    handler = get_warmup_handler()
    
    if handler.config.resources.enable_model_warmup:
        logger.info("Running model warmup on startup...")
        results = await handler.warmup_priority_models()
        
        if results:
            total = sum(results.values())
            logger.info(f"Startup warmup completed: {total} models loaded")
        else:
            logger.info("No models warmed during startup")
    else:
        logger.info("Model warmup disabled - skipping startup warmup")


# Example configuration for model warmup
WARMUP_CONFIG_EXAMPLE = """
resources:
  enable_model_warmup: false  # Disabled by default
  warmup_models:
    # List specific models to warmup
    # Format: "provider:model" or just "model" (uses default provider)
    - "openai:text-embedding-3-small"
    - "huggingface:sentence-transformers/all-MiniLM-L6-v2"
"""