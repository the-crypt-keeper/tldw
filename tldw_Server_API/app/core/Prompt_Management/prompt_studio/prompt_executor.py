# prompt_executor.py
# Prompt execution engine for Prompt Studio

import json
import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from loguru import logger

from tldw_Server_API.app.core.LLM_Calls.LLM_API_Calls import (
    chat_with_openai, chat_with_anthropic, chat_with_cohere,
    chat_with_groq, chat_with_openrouter, chat_with_deepseek,
    chat_with_mistral, chat_with_google
)
from tldw_Server_API.app.core.LLM_Calls.LLM_API_Calls_Local import (
    chat_with_llama, chat_with_kobold, chat_with_oobabooga,
    chat_with_tabbyapi, chat_with_vllm, chat_with_aphrodite,
    chat_with_ollama, chat_with_custom_openai
)
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase

########################################################################################################################
# Prompt Executor

class PromptExecutor:
    """Executes prompts with various LLM providers."""
    
    # Map provider names to functions
    PROVIDER_FUNCTIONS = {
        "openai": chat_with_openai,
        "anthropic": chat_with_anthropic,
        "cohere": chat_with_cohere,
        "groq": chat_with_groq,
        "openrouter": chat_with_openrouter,
        "deepseek": chat_with_deepseek,
        "mistral": chat_with_mistral,
        "google": chat_with_google,
        # Local LLM providers
        "llama": chat_with_llama,
        "kobold": chat_with_kobold,
        "ooba": chat_with_oobabooga,
        "oobabooga": chat_with_oobabooga,
        "tabby": chat_with_tabbyapi,
        "tabbyapi": chat_with_tabbyapi,
        "vllm": chat_with_vllm,
        "aphrodite": chat_with_aphrodite,
        "ollama": chat_with_ollama,
        "custom": chat_with_custom_openai,
        "custom_openai": chat_with_custom_openai
    }
    
    def __init__(self, db: PromptStudioDatabase):
        """
        Initialize PromptExecutor.
        
        Args:
            db: Database instance
        """
        self.db = db
        self.client_id = db.client_id
    
    ####################################################################################################################
    # Prompt Execution
    
    def execute_prompt(self, prompt_id: int, test_inputs: Dict[str, Any],
                            model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a prompt with given inputs and model configuration.
        
        Args:
            prompt_id: Prompt ID
            test_inputs: Input values for the prompt
            model_config: Model configuration (provider, model, parameters)
            
        Returns:
            Execution result with output, metrics, and metadata
        """
        start_time = time.time()
        
        try:
            # Get prompt details
            prompt = self._get_prompt(prompt_id)
            if not prompt:
                raise ValueError(f"Prompt {prompt_id} not found")
            
            # Get signature if linked
            signature = None
            if prompt.get("signature_id"):
                signature = self._get_signature(prompt["signature_id"])
            
            # Build the final prompt
            final_prompt = self._build_prompt(prompt, signature, test_inputs)
            
            # Execute with LLM
            provider = model_config.get("provider", "openai")
            model = model_config.get("model", "gpt-3.5-turbo")
            
            result = self._call_llm(
                provider=provider,
                model=model,
                prompt=final_prompt,
                system_prompt=prompt.get("system_prompt"),
                parameters=model_config.get("parameters", {})
            )
            
            # Parse output based on signature
            parsed_output = self._parse_output(result["content"], signature)
            
            # Calculate metrics
            execution_time = (time.time() - start_time) * 1000  # ms
            
            return {
                "success": True,
                "prompt_id": prompt_id,
                "inputs": test_inputs,
                "raw_output": result["content"],
                "parsed_output": parsed_output,
                "model": model,
                "provider": provider,
                "execution_time_ms": execution_time,
                "tokens_used": result.get("tokens", 0),
                "cost_estimate": self._estimate_cost(
                    provider, model, result.get("tokens", 0)
                ),
                "metadata": {
                    "temperature": model_config.get("parameters", {}).get("temperature"),
                    "max_tokens": model_config.get("parameters", {}).get("max_tokens"),
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Prompt execution failed: {e}")
            execution_time = (time.time() - start_time) * 1000
            
            return {
                "success": False,
                "prompt_id": prompt_id,
                "inputs": test_inputs,
                "error": str(e),
                "model": model_config.get("model"),
                "provider": model_config.get("provider"),
                "execution_time_ms": execution_time,
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
    
    async def execute_batch(self, prompt_id: int, test_cases: List[Dict[str, Any]],
                           model_configs: List[Dict[str, Any]],
                           max_concurrent: int = 5) -> List[Dict[str, Any]]:
        """
        Execute a prompt with multiple test cases and model configurations.
        
        Args:
            prompt_id: Prompt ID
            test_cases: List of test cases with inputs
            model_configs: List of model configurations
            max_concurrent: Maximum concurrent executions
            
        Returns:
            List of execution results
        """
        results = []
        
        # Create all execution tasks
        tasks = []
        for test_case in test_cases:
            for model_config in model_configs:
                task = self.execute_prompt(
                    prompt_id=prompt_id,
                    test_inputs=test_case.get("inputs", {}),
                    model_config=model_config
                )
                tasks.append((test_case, model_config, task))
        
        # Execute in batches
        for i in range(0, len(tasks), max_concurrent):
            batch = tasks[i:i + max_concurrent]
            batch_results = await asyncio.gather(
                *[task for _, _, task in batch],
                return_exceptions=True
            )
            
            # Process results
            for (test_case, model_config, _), result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Batch execution error: {result}")
                    results.append({
                        "success": False,
                        "test_case_id": test_case.get("id"),
                        "error": str(result),
                        "model": model_config.get("model"),
                        "provider": model_config.get("provider")
                    })
                else:
                    result["test_case_id"] = test_case.get("id")
                    result["test_case_name"] = test_case.get("name")
                    results.append(result)
        
        return results
    
    ####################################################################################################################
    # LLM Integration
    
    async def _call_llm(self, provider: str, model: str, prompt: str,
                       system_prompt: Optional[str] = None,
                       parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Call the appropriate LLM provider.
        
        Args:
            provider: Provider name
            model: Model name
            prompt: User prompt
            system_prompt: System prompt
            parameters: Additional parameters
            
        Returns:
            LLM response
        """
        # Get provider function
        provider_func = self.PROVIDER_FUNCTIONS.get(provider.lower())
        if not provider_func:
            raise ValueError(f"Unknown provider: {provider}")
        
        # Prepare parameters
        params = parameters or {}
        temperature = params.get("temperature", 0.7)
        max_tokens = params.get("max_tokens", 1000)
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Call provider (most providers have similar signatures)
            if provider in ["openai", "anthropic", "groq", "mistral", "deepseek"]:
                response = await asyncio.to_thread(
                    provider_func,
                    messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False
                )
            elif provider == "ollama":
                response = await asyncio.to_thread(
                    provider_func,
                    messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                # Local models
                response = await asyncio.to_thread(
                    provider_func,
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            
            # Parse response
            if isinstance(response, tuple):
                content, tokens = response
                return {"content": content, "tokens": tokens}
            elif isinstance(response, str):
                return {"content": response, "tokens": len(response.split()) * 1.3}  # Estimate
            else:
                return {"content": str(response), "tokens": 0}
                
        except Exception as e:
            logger.error(f"LLM call failed for {provider}/{model}: {e}")
            raise
    
    ####################################################################################################################
    # Helper Methods
    
    def _get_prompt(self, prompt_id: int) -> Optional[Dict[str, Any]]:
        """Get prompt details from database."""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM prompt_studio_prompts
            WHERE id = ? AND deleted = 0
        """, (prompt_id,))
        
        row = cursor.fetchone()
        if row:
            return self.db._row_to_dict(cursor, row)
        return None
    
    def _get_signature(self, signature_id: int) -> Optional[Dict[str, Any]]:
        """Get signature details from database."""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM prompt_studio_signatures
            WHERE id = ? AND deleted = 0
        """, (signature_id,))
        
        row = cursor.fetchone()
        if row:
            sig = self.db._row_to_dict(cursor, row)
            # Parse schemas
            try:
                sig["input_schema"] = json.loads(sig["input_schema"]) if sig["input_schema"] else []
                sig["output_schema"] = json.loads(sig["output_schema"]) if sig["output_schema"] else []
            except:
                sig["input_schema"] = []
                sig["output_schema"] = []
            return sig
        return None
    
    def _build_prompt(self, prompt: Dict[str, Any], signature: Optional[Dict[str, Any]],
                     inputs: Dict[str, Any]) -> str:
        """
        Build the final prompt by substituting variables.
        
        Args:
            prompt: Prompt data
            signature: Optional signature data
            inputs: Input values
            
        Returns:
            Final prompt string
        """
        template = prompt.get("content", "")
        
        # Replace variables in template
        for key, value in inputs.items():
            # Handle different placeholder formats
            template = template.replace(f"{{{key}}}", str(value))
            template = template.replace(f"{{{{key}}}}", str(value))
            template = template.replace(f"${key}", str(value))
            template = template.replace(f"<{key}>", str(value))
        
        # Add signature instructions if present
        if signature:
            sig_instruction = signature.get("instruction", "")
            if sig_instruction:
                template = f"{sig_instruction}\n\n{template}"
            
            # Add output format instruction
            if signature.get("output_schema"):
                template += "\n\nPlease format your response as JSON with the following structure:\n"
                template += json.dumps(
                    {field["name"]: f"<{field.get('type', 'string')}>" 
                     for field in signature["output_schema"]
                     if isinstance(field, dict)},
                    indent=2
                )
        
        return template
    
    def _parse_output(self, output: str, signature: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse LLM output based on signature schema.
        
        Args:
            output: Raw LLM output
            signature: Optional signature with output schema
            
        Returns:
            Parsed output
        """
        if not signature or not signature.get("output_schema"):
            return {"raw": output}
        
        # Try to parse as JSON
        try:
            # Look for JSON in the output
            import re
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return parsed
        except:
            pass
        
        # Try to extract fields from text
        parsed = {}
        for field in signature.get("output_schema", []):
            if isinstance(field, dict):
                field_name = field.get("name")
                if field_name:
                    # Simple extraction (can be improved)
                    pattern = f"{field_name}[:\s]+(.*?)(?:\n|$)"
                    match = re.search(pattern, output, re.IGNORECASE)
                    if match:
                        parsed[field_name] = match.group(1).strip()
        
        if not parsed:
            parsed = {"raw": output}
        
        return parsed
    
    def _estimate_cost(self, provider: str, model: str, tokens: int) -> float:
        """
        Estimate cost based on provider and token usage.
        
        Args:
            provider: Provider name
            model: Model name
            tokens: Token count
            
        Returns:
            Estimated cost in USD
        """
        # Rough cost estimates per 1K tokens (input + output averaged)
        cost_per_1k = {
            "openai": {
                "gpt-4": 0.03,
                "gpt-4-turbo": 0.01,
                "gpt-3.5-turbo": 0.002,
                "gpt-3.5-turbo-16k": 0.003
            },
            "anthropic": {
                "claude-3-opus": 0.015,
                "claude-3-sonnet": 0.003,
                "claude-3-haiku": 0.00025,
                "claude-2.1": 0.008,
                "claude-2": 0.008
            },
            "groq": {
                "llama2-70b": 0.0007,
                "mixtral-8x7b": 0.0006,
                "llama3-70b": 0.0008,
                "llama3-8b": 0.0001
            },
            "mistral": {
                "mistral-tiny": 0.00025,
                "mistral-small": 0.0006,
                "mistral-medium": 0.0027,
                "mistral-large": 0.008
            },
            "deepseek": {
                "deepseek-coder": 0.0001,
                "deepseek-chat": 0.0002
            }
        }
        
        # Get cost rate
        provider_costs = cost_per_1k.get(provider.lower(), {})
        
        # Try exact model match first
        cost_rate = provider_costs.get(model.lower(), 0)
        
        # If not found, try partial match
        if cost_rate == 0:
            for model_key, rate in provider_costs.items():
                if model_key in model.lower() or model.lower() in model_key:
                    cost_rate = rate
                    break
        
        # Default to very small cost if unknown
        if cost_rate == 0:
            cost_rate = 0.0001
        
        # Calculate cost
        return (tokens / 1000.0) * cost_rate

########################################################################################################################
# Prompt Validator

class PromptValidator:
    """Validates prompts and signatures before execution."""
    
    @staticmethod
    def validate_prompt(prompt: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate a prompt.
        
        Args:
            prompt: Prompt data
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not prompt.get("content"):
            return False, "Prompt content is required"
        
        if len(prompt["content"]) > 50000:
            return False, "Prompt content exceeds maximum length"
        
        # Check for required variables
        import re
        variables = re.findall(r'\{(\w+)\}|\$(\w+)|<(\w+)>', prompt["content"])
        flat_vars = [v for group in variables for v in group if v]
        
        if len(set(flat_vars)) > 20:
            return False, "Too many variables (max 20)"
        
        return True, None
    
    @staticmethod
    def validate_signature(signature: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate a signature.
        
        Args:
            signature: Signature data
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate input schema
        if signature.get("input_schema"):
            try:
                input_schema = json.loads(signature["input_schema"]) if isinstance(signature["input_schema"], str) else signature["input_schema"]
                if not isinstance(input_schema, list):
                    return False, "Input schema must be a list"
                
                for field in input_schema:
                    if not isinstance(field, dict):
                        return False, "Each input field must be an object"
                    if not field.get("name"):
                        return False, "Each input field must have a name"
            except:
                return False, "Invalid input schema format"
        
        # Validate output schema
        if signature.get("output_schema"):
            try:
                output_schema = json.loads(signature["output_schema"]) if isinstance(signature["output_schema"], str) else signature["output_schema"]
                if not isinstance(output_schema, list):
                    return False, "Output schema must be a list"
                
                for field in output_schema:
                    if not isinstance(field, dict):
                        return False, "Each output field must be an object"
                    if not field.get("name"):
                        return False, "Each output field must have a name"
            except:
                return False, "Invalid output schema format"
        
        return True, None
    
    @staticmethod
    def validate_test_inputs(inputs: Dict[str, Any], signature: Optional[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
        """
        Validate test inputs against signature schema.
        
        Args:
            inputs: Test input values
            signature: Optional signature with schema
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not signature or not signature.get("input_schema"):
            return True, None
        
        try:
            input_schema = signature["input_schema"]
            if isinstance(input_schema, str):
                input_schema = json.loads(input_schema)
            
            # Check required fields
            for field in input_schema:
                if isinstance(field, dict):
                    field_name = field.get("name")
                    if field.get("required", True) and field_name not in inputs:
                        return False, f"Required input field missing: {field_name}"
                    
                    # Type validation (basic)
                    if field_name in inputs:
                        value = inputs[field_name]
                        field_type = field.get("type", "string")
                        
                        if field_type == "integer" and not isinstance(value, int):
                            return False, f"Field {field_name} must be an integer"
                        elif field_type == "boolean" and not isinstance(value, bool):
                            return False, f"Field {field_name} must be a boolean"
                        elif field_type == "array" and not isinstance(value, list):
                            return False, f"Field {field_name} must be an array"
                        elif field_type == "object" and not isinstance(value, dict):
                            return False, f"Field {field_name} must be an object"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"