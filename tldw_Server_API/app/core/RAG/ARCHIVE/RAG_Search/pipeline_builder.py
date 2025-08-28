"""
Pipeline builder that creates executable pipelines from TOML configuration.

This module is responsible for reading pipeline configurations and composing
the individual functions into complete, executable pipelines. It handles
step sequencing, error propagation, and effect collection.
"""

import asyncio
from typing import Dict, Any, List, Tuple, Callable, Awaitable, Optional, Union
from dataclasses import dataclass
import time
import uuid
from loguru import logger

from .pipeline_core import (
    PipelineContext, PipelineConfig, PipelineStep, StepType,
    SearchResult, PipelineError, PipelineErrorType,
    Result, Success, Failure, Effect, TypedEffect,
    PipelineTrace, PipelineMetrics, ValidationResult,
    Pipeline
)
from .pipeline_functions import get_function, PIPELINE_FUNCTIONS
from .pipeline_resources import PipelineResources


# ==============================================================================
# Pipeline Builder
# ==============================================================================

class PipelineBuilder:
    """Builds executable pipelines from configuration."""
    
    def __init__(self):
        self.validators = []
        self.middleware = []
    
    def build(self, config: PipelineConfig) -> Pipeline:
        """
        Build an executable pipeline from configuration.
        
        Args:
            config: Pipeline configuration object
            
        Returns:
            Async function that executes the pipeline
        """
        # Validate configuration
        validation = self.validate_config(config)
        validation.raise_if_invalid()
        
        # Create the pipeline function
        async def execute_pipeline(context: PipelineContext) -> Tuple[Result[List[SearchResult], PipelineError], List[Effect]]:
            """Execute the configured pipeline."""
            # Initialize trace
            trace = PipelineTrace(
                trace_id=context.get('trace_id', str(uuid.uuid4())),
                pipeline_id=config.id,
                start_time=time.time()
            )
            
            # Initialize metrics
            metrics = PipelineMetrics(
                total_duration_ms=0,
                step_durations_ms={},
                retrieval_count=0,
                processing_count=0,
                final_result_count=0
            )
            
            # Initialize context
            context['pipeline_config'] = config
            context['trace'] = trace
            context['metrics'] = metrics
            context['effects'] = []
            context['results'] = []
            
            # Log pipeline start
            context['effects'].append(TypedEffect.log(
                'info',
                f"Starting pipeline: {config.name}",
                pipeline_id=config.id,
                trace_id=trace.trace_id
            ))
            
            try:
                # Execute pipeline steps
                result = await self._execute_steps(config.steps, context)
                
                # Finalize trace
                trace.end_time = time.time()
                metrics.total_duration_ms = (trace.end_time - trace.start_time) * 1000
                
                if isinstance(result, Success):
                    metrics.final_result_count = len(result.value)
                    context['effects'].append(TypedEffect.metric(
                        'pipeline_execution_time',
                        metrics.total_duration_ms,
                        'histogram'
                    ))
                
                # Return results and collected effects
                return result, context['effects']
                
            except Exception as e:
                logger.error(f"Pipeline execution error: {e}", exc_info=True)
                error = PipelineError(
                    error_type=PipelineErrorType.UNKNOWN_ERROR,
                    message=f"Pipeline execution failed: {str(e)}",
                    step_name=config.name,
                    cause=e
                )
                return Failure(error), context['effects']
        
        # Apply middleware to the pipeline
        for mw in self.middleware:
            execute_pipeline = mw(execute_pipeline)
        
        return execute_pipeline
    
    async def _execute_steps(
        self,
        steps: List[PipelineStep],
        context: PipelineContext
    ) -> Result[List[SearchResult], PipelineError]:
        """Execute a list of pipeline steps."""
        context['step_index'] = 0
        context['total_steps'] = len(steps)
        
        for i, step in enumerate(steps):
            context['step_index'] = i
            step_start = time.time()
            
            # Log step execution
            context['effects'].append(TypedEffect.log(
                'debug',
                f"Executing step {i+1}/{len(steps)}: {step.step_type.value}",
                step_name=step.name or step.function_name
            ))
            
            try:
                # Execute step based on type
                if step.step_type == StepType.RETRIEVE:
                    result = await self._execute_retrieve_step(step, context)
                elif step.step_type == StepType.PROCESS:
                    result = await self._execute_process_step(step, context)
                elif step.step_type == StepType.FORMAT:
                    result = await self._execute_format_step(step, context)
                elif step.step_type == StepType.PARALLEL:
                    result = await self._execute_parallel_step(step, context)
                elif step.step_type == StepType.MERGE:
                    result = await self._execute_merge_step(step, context)
                elif step.step_type == StepType.CONDITIONAL:
                    result = await self._execute_conditional_step(step, context)
                else:
                    raise ValueError(f"Unknown step type: {step.step_type}")
                
                # Handle step result
                if isinstance(result, Failure):
                    # Step failed
                    context['metrics'].errors.append(result.error)
                    
                    # Check error handling policy
                    if context['pipeline_config'].on_error == 'fail':
                        return result
                    elif context['pipeline_config'].on_error == 'continue':
                        logger.warning(f"Step {i+1} failed, continuing: {result.error.message}")
                        continue
                    elif context['pipeline_config'].on_error == 'fallback':
                        # TODO: Implement fallback pipeline execution
                        return result
                
                # Update context based on step type
                if step.step_type in [StepType.RETRIEVE, StepType.PROCESS, StepType.PARALLEL, StepType.MERGE]:
                    if isinstance(result, Success):
                        context['results'] = result.value
                elif step.step_type == StepType.FORMAT:
                    if isinstance(result, Success):
                        context['formatted_output'] = result.value
                
                # Record step duration
                step_duration_ms = (time.time() - step_start) * 1000
                step_name = step.name or f"{step.step_type.value}_{i+1}"
                context['metrics'].step_durations_ms[step_name] = step_duration_ms
                
            except Exception as e:
                logger.error(f"Step execution error: {e}", exc_info=True)
                return Failure(PipelineError(
                    error_type=PipelineErrorType.UNKNOWN_ERROR,
                    message=f"Step {i+1} failed: {str(e)}",
                    step_name=step.name or step.function_name,
                    cause=e
                ))
        
        # Return final results
        return Success(context.get('results', []))
    
    async def _execute_retrieve_step(
        self,
        step: PipelineStep,
        context: PipelineContext
    ) -> Result[List[SearchResult], PipelineError]:
        """Execute a retrieval step."""
        if not step.function_name:
            return Failure(PipelineError(
                error_type=PipelineErrorType.VALIDATION_ERROR,
                message="Retrieve step must specify a function name",
                step_name=step.name
            ))
        
        func = get_function(step.function_name)
        
        # Merge step config with context params
        config = {**context.get('params', {}), **step.config}
        
        # Apply timeout if specified
        if step.timeout_seconds:
            try:
                result = await asyncio.wait_for(
                    func(context, config),
                    timeout=step.timeout_seconds
                )
            except asyncio.TimeoutError:
                return Failure(PipelineError(
                    error_type=PipelineErrorType.TIMEOUT_ERROR,
                    message=f"Retrieval step timed out after {step.timeout_seconds}s",
                    step_name=step.name or step.function_name
                ))
        else:
            result = await func(context, config)
        
        # Update metrics
        if isinstance(result, Success):
            context['metrics'].retrieval_count += len(result.value)
        
        return result
    
    async def _execute_process_step(
        self,
        step: PipelineStep,
        context: PipelineContext
    ) -> Result[List[SearchResult], PipelineError]:
        """Execute a processing step."""
        if not step.function_name:
            return Failure(PipelineError(
                error_type=PipelineErrorType.VALIDATION_ERROR,
                message="Process step must specify a function name",
                step_name=step.name
            ))
        
        func = get_function(step.function_name)
        
        # Get current results
        results = context.get('results', [])
        if not results:
            logger.warning(f"No results to process in step: {step.name or step.function_name}")
            return Success([])
        
        # Merge step config with context params
        config = {**context.get('params', {}), **step.config}
        
        # Most processing functions are synchronous
        if asyncio.iscoroutinefunction(func):
            result = await func(results, context, config)
        else:
            result = func(results, context, config)
        
        # Update metrics
        if isinstance(result, Success):
            context['metrics'].processing_count += 1
        
        return result
    
    async def _execute_format_step(
        self,
        step: PipelineStep,
        context: PipelineContext
    ) -> Result[str, PipelineError]:
        """Execute a formatting step."""
        if not step.function_name:
            return Failure(PipelineError(
                error_type=PipelineErrorType.VALIDATION_ERROR,
                message="Format step must specify a function name",
                step_name=step.name
            ))
        
        func = get_function(step.function_name)
        
        # Get current results
        results = context.get('results', [])
        
        # Merge step config with context params
        config = {**context.get('params', {}), **step.config}
        
        # Formatting functions are typically synchronous
        if asyncio.iscoroutinefunction(func):
            result = await func(results, context, config)
        else:
            result = func(results, context, config)
        
        return result
    
    async def _execute_parallel_step(
        self,
        step: PipelineStep,
        context: PipelineContext
    ) -> Result[List[SearchResult], PipelineError]:
        """Execute parallel retrieval functions."""
        if not step.parallel_functions:
            return Failure(PipelineError(
                error_type=PipelineErrorType.VALIDATION_ERROR,
                message="Parallel step must specify functions",
                step_name=step.name
            ))
        
        # Create tasks for each parallel function
        tasks = []
        function_names = []
        
        for func_step in step.parallel_functions:
            if func_step.function_name:
                func = get_function(func_step.function_name)
                config = {**context.get('params', {}), **func_step.config}
                task = func(context, config)
                tasks.append(task)
                function_names.append(func_step.function_name)
        
        if not tasks:
            return Success([])
        
        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        all_results = []
        errors = []
        parallel_results = []
        
        for i, (name, result) in enumerate(zip(function_names, results)):
            if isinstance(result, Exception):
                errors.append(f"{name} failed: {str(result)}")
            elif isinstance(result, Success):
                parallel_results.append(result.value)
                all_results.extend(result.value)
            elif isinstance(result, Failure):
                errors.append(f"{name} error: {result.error.message}")
        
        # Store individual results for merge step
        context['parallel_results'] = parallel_results
        
        if errors and not all_results:
            return Failure(PipelineError(
                error_type=PipelineErrorType.RETRIEVAL_ERROR,
                message=f"All parallel functions failed: {'; '.join(errors)}",
                step_name=step.name
            ))
        
        return Success(all_results)
    
    async def _execute_merge_step(
        self,
        step: PipelineStep,
        context: PipelineContext
    ) -> Result[List[SearchResult], PipelineError]:
        """Execute a merge step for parallel results."""
        func_name = step.function_name or 'weighted_merge'
        func = get_function(func_name)
        
        # Get parallel results
        parallel_results = context.get('parallel_results', [])
        if not parallel_results:
            logger.warning("No parallel results to merge")
            return Success(context.get('results', []))
        
        # Get weights from config
        weights = step.config.get('weights', [1.0] * len(parallel_results))
        
        # Execute merge
        if asyncio.iscoroutinefunction(func):
            merged = await func(parallel_results, weights)
        else:
            # weighted_merge is async, but if it wasn't:
            merged = await func(parallel_results, weights)
        
        return Success(merged)
    
    async def _execute_conditional_step(
        self,
        step: PipelineStep,
        context: PipelineContext
    ) -> Result[List[SearchResult], PipelineError]:
        """Execute a conditional step."""
        if not step.condition:
            return Failure(PipelineError(
                error_type=PipelineErrorType.VALIDATION_ERROR,
                message="Conditional step must specify a condition",
                step_name=step.name
            ))
        
        # Evaluate condition
        condition_result = self._evaluate_condition(step.condition, context)
        
        # Execute appropriate branch
        if condition_result:
            if step.if_true:
                return await self._execute_steps([step.if_true], context)
        else:
            if step.if_false:
                return await self._execute_steps([step.if_false], context)
        
        # No branch to execute
        return Success(context.get('results', []))
    
    def _evaluate_condition(self, condition: str, context: PipelineContext) -> bool:
        """Evaluate a simple condition."""
        # Simple condition evaluation
        # TODO: Implement more sophisticated condition evaluation
        
        if condition == "has_results":
            return len(context.get('results', [])) > 0
        elif condition == "no_results":
            return len(context.get('results', [])) == 0
        elif condition.startswith("min_results:"):
            min_count = int(condition.split(":")[1])
            return len(context.get('results', [])) >= min_count
        elif condition.startswith("source:"):
            source = condition.split(":")[1]
            return context.get('sources', {}).get(source, False)
        else:
            logger.warning(f"Unknown condition: {condition}")
            return False
    
    def validate_config(self, config: PipelineConfig) -> ValidationResult:
        """Validate pipeline configuration."""
        errors = []
        warnings = []
        
        # Basic validation
        if not config.id:
            errors.append("Pipeline must have an ID")
        
        if not config.name:
            errors.append("Pipeline must have a name")
        
        if not config.steps:
            errors.append("Pipeline must have at least one step")
        
        # Validate each step
        for i, step in enumerate(config.steps):
            step_errors = self._validate_step(step, i)
            errors.extend(step_errors)
        
        # Check for format step
        has_format = any(s.step_type == StepType.FORMAT for s in config.steps)
        if not has_format:
            warnings.append("Pipeline has no format step - results will not be formatted")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _validate_step(self, step: PipelineStep, index: int) -> List[str]:
        """Validate a single step."""
        errors = []
        
        if step.step_type in [StepType.RETRIEVE, StepType.PROCESS, StepType.FORMAT]:
            if not step.function_name:
                errors.append(f"Step {index+1} ({step.step_type.value}) must specify a function name")
            elif step.function_name not in PIPELINE_FUNCTIONS:
                errors.append(f"Step {index+1}: Unknown function '{step.function_name}'")
        
        elif step.step_type == StepType.PARALLEL:
            if not step.parallel_functions:
                errors.append(f"Step {index+1} (parallel) must specify functions")
            else:
                for j, func_step in enumerate(step.parallel_functions):
                    sub_errors = self._validate_step(func_step, j)
                    errors.extend([f"Step {index+1}.{j+1}: {e}" for e in sub_errors])
        
        elif step.step_type == StepType.CONDITIONAL:
            if not step.condition:
                errors.append(f"Step {index+1} (conditional) must specify a condition")
        
        return errors
    
    def add_validator(self, validator: Callable[[PipelineConfig], List[str]]):
        """Add a custom configuration validator."""
        self.validators.append(validator)
    
    def add_middleware(self, middleware: Callable[[Pipeline], Pipeline]):
        """Add middleware to wrap pipelines."""
        self.middleware.append(middleware)


# ==============================================================================
# Pipeline Factory Functions
# ==============================================================================

def build_pipeline_from_dict(config_dict: Dict[str, Any]) -> Pipeline:
    """Build a pipeline from a dictionary configuration."""
    # Convert dict to PipelineConfig
    steps = []
    for step_dict in config_dict.get('steps', []):
        step = PipelineStep(
            step_type=StepType(step_dict['type']),
            function_name=step_dict.get('function'),
            config=step_dict.get('config', {}),
            name=step_dict.get('name'),
            description=step_dict.get('description'),
            timeout_seconds=step_dict.get('timeout_seconds')
        )
        
        # Handle parallel functions
        if step.step_type == StepType.PARALLEL and 'functions' in step_dict:
            step.parallel_functions = []
            for func_dict in step_dict['functions']:
                func_step = PipelineStep(
                    step_type=StepType.RETRIEVE,
                    function_name=func_dict['function'],
                    config=func_dict.get('config', {})
                )
                step.parallel_functions.append(func_step)
        
        steps.append(step)
    
    config = PipelineConfig(
        id=config_dict['id'],
        name=config_dict['name'],
        description=config_dict.get('description', ''),
        steps=steps,
        version=config_dict.get('version', '1.0'),
        tags=config_dict.get('tags', []),
        enabled=config_dict.get('enabled', True),
        timeout_seconds=config_dict.get('timeout_seconds', 30.0),
        cache_results=config_dict.get('cache_results', True),
        cache_ttl_seconds=config_dict.get('cache_ttl_seconds', 3600.0),
        on_error=config_dict.get('on_error', 'fail'),
        fallback_pipeline=config_dict.get('fallback_pipeline')
    )
    
    builder = PipelineBuilder()
    return builder.build(config)


def build_pipeline_from_toml(toml_path: str) -> Pipeline:
    """Build a pipeline from a TOML file."""
    import tomllib
    
    with open(toml_path, 'rb') as f:
        config_dict = tomllib.load(f)
    
    return build_pipeline_from_dict(config_dict)


# ==============================================================================
# Pipeline Middleware
# ==============================================================================

def caching_middleware(cache_key_func: Optional[Callable] = None):
    """Middleware that adds caching to pipelines."""
    def middleware(pipeline: Pipeline) -> Pipeline:
        async def cached_pipeline(context: PipelineContext) -> Tuple[Result[List[SearchResult], PipelineError], List[Effect]]:
            # Check if caching is enabled
            config = context.get('pipeline_config')
            if not config or not config.cache_results:
                return await pipeline(context)
            
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(context)
            else:
                # Default cache key based on query and sources
                cache_key = f"{context['query']}:{sorted(context['sources'].items())}"
            
            # Check cache
            resources = context.get('resources')
            if resources and resources.cache:
                cached = resources.cache.get(cache_key)
                if cached:
                    context['effects'].append(TypedEffect.metric(
                        'cache_hit',
                        1,
                        'counter'
                    ))
                    return cached
            
            # Execute pipeline
            result = await pipeline(context)
            
            # Cache result
            if resources and resources.cache and isinstance(result[0], Success):
                ttl = config.cache_ttl_seconds
                resources.cache.set(cache_key, result, ttl)
                context['effects'].append(TypedEffect.metric(
                    'cache_miss',
                    1,
                    'counter'
                ))
            
            return result
        
        return cached_pipeline
    
    return middleware


def logging_middleware(log_level: str = 'info'):
    """Middleware that adds detailed logging to pipelines."""
    def middleware(pipeline: Pipeline) -> Pipeline:
        async def logged_pipeline(context: PipelineContext) -> Tuple[Result[List[SearchResult], PipelineError], List[Effect]]:
            start_time = time.time()
            pipeline_id = context.get('pipeline_config', {}).get('id', 'unknown')
            
            logger.log(log_level, f"Pipeline {pipeline_id} started", extra={
                'query': context.get('query'),
                'sources': context.get('sources'),
                'trace_id': context.get('trace_id')
            })
            
            try:
                result = await pipeline(context)
                
                duration_ms = (time.time() - start_time) * 1000
                logger.log(log_level, f"Pipeline {pipeline_id} completed in {duration_ms:.2f}ms", extra={
                    'duration_ms': duration_ms,
                    'success': isinstance(result[0], Success),
                    'result_count': len(result[0].value) if isinstance(result[0], Success) else 0
                })
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(f"Pipeline {pipeline_id} failed after {duration_ms:.2f}ms", extra={
                    'duration_ms': duration_ms,
                    'error': str(e)
                }, exc_info=True)
                raise
        
        return logged_pipeline
    
    return middleware


# ==============================================================================
# Global Pipeline Builder Instance
# ==============================================================================

_global_builder: Optional[PipelineBuilder] = None


def get_pipeline_builder() -> PipelineBuilder:
    """Get or create the global pipeline builder."""
    global _global_builder
    if _global_builder is None:
        _global_builder = PipelineBuilder()
        # Add default middleware
        _global_builder.add_middleware(logging_middleware())
        _global_builder.add_middleware(caching_middleware())
    return _global_builder