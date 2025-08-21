# prompt_studio_optimization_requests.py
# Request models for optimization endpoints

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict

class CompareStrategiesRequest(BaseModel):
    """Request model for comparing optimization strategies."""
    model_config = ConfigDict(extra='forbid')
    
    prompt_id: int = Field(..., description="Prompt to optimize")
    test_case_ids: List[int] = Field(..., description="Test cases for evaluation")
    strategies: List[str] = Field(..., description="Strategies to compare")
    model_configuration: Dict[str, Any] = Field(..., description="Model configuration")