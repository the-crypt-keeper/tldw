"""
Enhanced CLI for the evaluation module with full API/model selection support.

This provides a comprehensive CLI interface for running evaluations
with support for all configured LLM APIs (commercial and self-hosted).
"""

import sys
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from collections import defaultdict

import click
from loguru import logger
from tabulate import tabulate

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from tldw_Server_API.app.core.Evaluations.evaluation_manager import EvaluationManager
from tldw_Server_API.app.core.Evaluations.benchmark_registry import get_registry
from tldw_Server_API.app.core.Evaluations.benchmark_loaders import load_benchmark_dataset
from tldw_Server_API.app.core.config import load_and_log_configs
from tldw_Server_API.app.core.Chat.Chat_Functions import chat_api_call


# Mapping of config keys to friendly API names
API_CONFIG_MAPPING = {
    'anthropic_api': 'anthropic',
    'cohere_api': 'cohere',
    'deepseek_api': 'deepseek',
    'google_api': 'google',
    'groq_api': 'groq',
    'huggingface_api': 'huggingface',
    'mistral_api': 'mistral',
    'openrouter_api': 'openrouter',
    'openai_api': 'openai',
    'llama_api': 'llama.cpp',
    'ooba_api': 'ooba',
    'kobold_api': 'kobold',
    'tabby_api': 'tabbyapi',
    'vllm_api': 'vllm',
    'ollama_api': 'ollama',
    'aphrodite_api': 'aphrodite',
    'custom_openai_api': 'custom-openai-api',
    'custom_openai2_api': 'custom-openai-api-2'
}

# API categories for better organization
API_CATEGORIES = {
    'Commercial': ['openai', 'anthropic', 'cohere', 'google', 'groq', 
                   'huggingface', 'mistral', 'openrouter', 'deepseek'],
    'Self-Hosted': ['llama.cpp', 'ooba', 'kobold', 'tabbyapi', 'vllm', 
                    'ollama', 'aphrodite'],
    'Custom': ['custom-openai-api', 'custom-openai-api-2']
}


def get_available_apis() -> Dict[str, Dict[str, Any]]:
    """
    Get all available APIs from configuration with their status.
    
    Returns:
        Dict mapping API names to their configuration and status
    """
    config = load_and_log_configs()
    if not config:
        return {}
    
    available_apis = {}
    
    for config_key, api_name in API_CONFIG_MAPPING.items():
        if config_key in config:
            api_config = config[config_key]
            
            # Check if API is configured (has API key or endpoint)
            is_configured = False
            config_status = []
            
            # Check for API key
            if 'api_key' in api_config:
                if api_config['api_key'] and api_config['api_key'] not in [None, '', 'None']:
                    is_configured = True
                    config_status.append('API key set')
                else:
                    config_status.append('API key missing')
            
            # Check for API endpoint (for self-hosted)
            if 'api_ip' in api_config or 'api_url' in api_config:
                endpoint = api_config.get('api_ip') or api_config.get('api_url')
                if endpoint and endpoint not in [None, '', 'None']:
                    is_configured = True
                    config_status.append(f'Endpoint: {endpoint}')
                else:
                    config_status.append('Endpoint missing')
            
            # Get model info
            model = api_config.get('model', 'Not specified')
            
            # Determine category
            category = 'Unknown'
            for cat, apis in API_CATEGORIES.items():
                if api_name in apis:
                    category = cat
                    break
            
            available_apis[api_name] = {
                'configured': is_configured,
                'model': model,
                'category': category,
                'status': ', '.join(config_status) if config_status else 'Not configured',
                'config': api_config
            }
    
    return available_apis


def validate_api_config(api_name: str) -> Tuple[bool, str]:
    """
    Validate if an API is properly configured for use.
    
    Args:
        api_name: Name of the API to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    apis = get_available_apis()
    
    if api_name not in apis:
        available = ', '.join(sorted(apis.keys()))
        return False, f"Unknown API '{api_name}'. Available: {available}"
    
    api_info = apis[api_name]
    
    if not api_info['configured']:
        return False, f"API '{api_name}' is not properly configured. {api_info['status']}"
    
    return True, ""


def get_api_model(api_name: str, model_override: Optional[str] = None) -> str:
    """
    Get the model to use for a specific API.
    
    Args:
        api_name: Name of the API
        model_override: Optional model override
        
    Returns:
        Model string to use
    """
    if model_override:
        return model_override
    
    apis = get_available_apis()
    if api_name in apis:
        return apis[api_name].get('model', 'default')
    
    return 'default'


def run_evaluation_with_llm(
    samples: List[Dict],
    api_name: str,
    model: str,
    eval_config: Dict[str, Any],
    progress_callback: Optional[callable] = None
) -> List[Dict[str, Any]]:
    """
    Run evaluation using actual LLM calls via chat API.
    
    Args:
        samples: Evaluation samples
        api_name: API to use
        model: Model to use
        eval_config: Evaluation configuration
        progress_callback: Optional callback for progress updates
        
    Returns:
        List of evaluation results
    """
    results = []
    
    # Get API configuration
    apis = get_available_apis()
    api_config = apis.get(api_name, {}).get('config', {})
    api_key = api_config.get('api_key', None)
    
    for i, sample in enumerate(samples):
        try:
            # Prepare the evaluation prompt
            eval_prompt = eval_config.get('prompt_template', 
                "Evaluate the following: {input}\n\nProvide a score from 0 to 1.")
            
            if 'input' in sample:
                input_text = json.dumps(sample['input']) if isinstance(sample['input'], dict) else str(sample['input'])
            else:
                input_text = json.dumps(sample)
            
            prompt = eval_prompt.format(input=input_text)
            
            # Prepare messages for chat API
            messages_payload = [
                {
                    "role": "system",
                    "content": "You are an evaluation system. Provide accurate scores. Return only a numeric score between 0 and 1."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Call the chat API
            # Note: chat_api_call returns a generator for streaming or a complete response
            response = chat_api_call(
                api_endpoint=api_name,
                messages_payload=messages_payload,
                api_key=api_key,
                model=model,
                temp=0.7,
                streaming=False,
                max_tokens=100  # We only need a short response with the score
            )
            
            # Handle response (could be a generator even with streaming=False for some providers)
            if hasattr(response, '__iter__') and not isinstance(response, str):
                # If it's a generator, consume it
                response_text = ''.join(str(chunk) for chunk in response)
            else:
                response_text = str(response)
            
            # Parse the response to extract score
            try:
                # Try to extract a float from the response
                import re
                score_match = re.search(r'([0-9]*\.?[0-9]+)', response_text)
                score = float(score_match.group(1)) if score_match else 0.5
                score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
            except:
                score = 0.5  # Default score if parsing fails
            
            result = {
                "sample_id": sample.get("id", i),
                "score": score,
                "response": response_text,
                "metadata": {
                    "api": api_name,
                    "model": model
                }
            }
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, len(samples))
            
        except Exception as e:
            logger.error(f"Error evaluating sample {i}: {e}")
            results.append({
                "sample_id": sample.get("id", i),
                "score": 0.0,
                "error": str(e),
                "metadata": {
                    "api": api_name,
                    "model": model
                }
            })
    
    return results


@click.group(context_settings={'help_option_names': ['-h', '--help']})
@click.option('--config', type=click.Path(exists=True), help='Config file path')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']), 
              default='INFO', help='Logging level')
@click.pass_context
def cli(ctx, config, log_level):
    """tldw Evaluations CLI - Run evaluation benchmarks with any configured LLM API."""
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    
    # Store context
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    ctx.obj['manager'] = EvaluationManager()
    ctx.obj['registry'] = get_registry()


@cli.command()
@click.option('--category', '-c', type=click.Choice(['all', 'commercial', 'self-hosted', 'custom']), 
              default='all', help='Filter by API category')
@click.option('--show-config', '-s', is_flag=True, help='Show detailed configuration')
def list_apis(category, show_config):
    """List all available LLM APIs and their configuration status."""
    apis = get_available_apis()
    
    if not apis:
        click.echo("No APIs found in configuration")
        return
    
    # Group APIs by category
    grouped = defaultdict(list)
    for api_name, info in apis.items():
        if category == 'all' or info['category'].lower() == category:
            grouped[info['category']].append((api_name, info))
    
    # Display APIs
    for cat in ['Commercial', 'Self-Hosted', 'Custom']:
        if cat in grouped:
            click.echo(f"\n{cat} APIs:")
            click.echo("-" * 60)
            
            table_data = []
            for api_name, info in sorted(grouped[cat]):
                status_symbol = "✓" if info['configured'] else "✗"
                row = [
                    f"{status_symbol} {api_name}",
                    info['model'],
                    info['status']
                ]
                table_data.append(row)
            
            headers = ["API", "Model", "Configuration"]
            click.echo(tabulate(table_data, headers=headers, tablefmt="simple"))
            
            if show_config:
                click.echo("\nDetailed Configuration:")
                for api_name, info in sorted(grouped[cat]):
                    if info['configured']:
                        click.echo(f"\n  {api_name}:")
                        for key, value in info['config'].items():
                            if key != 'api_key' or not value:  # Don't show actual API keys
                                display_value = value
                            else:
                                display_value = "***" + value[-4:] if len(value) > 4 else "***"
                            click.echo(f"    {key}: {display_value}")


@cli.command()
@click.option('--detailed', '-d', is_flag=True, help='Show detailed information')
def list_benchmarks(detailed):
    """List available benchmarks."""
    registry = get_registry()
    benchmarks = registry.list_benchmarks()
    
    if not benchmarks:
        click.echo("No benchmarks available")
        return
    
    click.echo(f"Available benchmarks ({len(benchmarks)}):")
    for name in sorted(benchmarks):
        if detailed:
            info = registry.get_benchmark_info(name)
            click.echo(f"\n  {name}:")
            click.echo(f"    Type: {info.get('evaluation_type', 'N/A')}")
            click.echo(f"    Description: {info.get('description', 'N/A')}")
            if info.get('metadata'):
                click.echo(f"    Metadata: {json.dumps(info['metadata'], indent=6)}")
        else:
            click.echo(f"  - {name}")


@cli.command()
@click.argument('benchmark')
@click.option('--limit', '-l', type=int, help='Limit number of samples')
@click.option('--api', '-a', help='API to use for evaluation (e.g., openai, anthropic, llama.cpp)')
@click.option('--model', '-m', help='Model to use (overrides config default)')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.option('--parallel', '-p', type=int, default=1, help='Parallel workers')
@click.option('--dry-run', is_flag=True, help='Validate configuration without running')
@click.pass_context
def run(ctx, benchmark, limit, api, model, output, parallel, dry_run):
    """Run a benchmark evaluation using specified API."""
    registry = ctx.obj['registry']
    manager = ctx.obj['manager']
    
    # Check if benchmark exists
    config = registry.get(benchmark)
    if not config:
        click.echo(f"Error: Benchmark '{benchmark}' not found", err=True)
        click.echo("Use 'list-benchmarks' to see available benchmarks")
        sys.exit(1)
    
    # Determine which API to use
    if not api:
        # Try to get default from config
        apis = get_available_apis()
        configured_apis = [name for name, info in apis.items() if info['configured']]
        
        if not configured_apis:
            click.echo("Error: No APIs are configured. Please configure at least one API.", err=True)
            click.echo("Available APIs that need configuration:")
            for name in sorted(apis.keys()):
                click.echo(f"  - {name}")
            sys.exit(1)
        
        # Use first configured API as default
        api = configured_apis[0]
        click.echo(f"No API specified, using default: {api}")
    
    # Validate API configuration
    is_valid, error_msg = validate_api_config(api)
    if not is_valid:
        click.echo(f"Error: {error_msg}", err=True)
        sys.exit(1)
    
    # Get model
    model_to_use = get_api_model(api, model)
    
    click.echo(f"Running benchmark: {benchmark}")
    click.echo(f"Evaluation type: {config.evaluation_type}")
    click.echo(f"Using API: {api}")
    click.echo(f"Using model: {model_to_use}")
    
    if dry_run:
        click.echo("\nDry run - configuration validated successfully")
        return
    
    # Load dataset
    try:
        dataset = load_benchmark_dataset(benchmark, limit=limit)
        click.echo(f"Loaded {len(dataset)} samples")
    except Exception as e:
        click.echo(f"Error loading dataset: {e}", err=True)
        sys.exit(1)
    
    # Create evaluator
    evaluator = registry.create_evaluator(benchmark)
    if not evaluator:
        click.echo(f"Error: Could not create evaluator for {config.evaluation_type}", err=True)
        sys.exit(1)
    
    results = []
    errors = 0
    
    # Progress callback
    def update_progress(current, total):
        click.echo(f"Progress: {current}/{total} samples evaluated", nl=False)
        click.echo('\r', nl=False)
    
    # Process samples using actual LLM
    # Note: run_evaluation_with_llm is now synchronous since chat_api_call handles async internally
    
    # Run evaluation
    click.echo("Starting evaluation...")
    try:
        results = run_evaluation_with_llm(
            dataset,
            api,
            model_to_use,
            {"prompt_template": config.get("prompt_template", "Evaluate: {input}")},
            update_progress
        )
    except Exception as e:
        click.echo(f"\nError during evaluation: {e}", err=True)
        sys.exit(1)
    
    # Calculate summary statistics
    scores = [r['score'] for r in results if 'score' in r and not r.get('error')]
    errors = len([r for r in results if r.get('error')])
    
    if scores:
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
    else:
        avg_score = min_score = max_score = 0.0
    
    click.echo(f"\n\nResults:")
    click.echo(f"  Total samples: {len(results)}")
    click.echo(f"  Successful: {len(scores)}")
    click.echo(f"  Errors: {errors}")
    if scores:
        click.echo(f"  Average score: {avg_score:.3f}")
        click.echo(f"  Min score: {min_score:.3f}")
        click.echo(f"  Max score: {max_score:.3f}")
    
    # Save results if requested
    if output:
        output_data = {
            "benchmark": benchmark,
            "api": api,
            "model": model_to_use,
            "results": results,
            "summary": {
                "total": len(results),
                "successful": len(scores),
                "errors": errors,
                "average_score": avg_score,
                "min_score": min_score,
                "max_score": max_score
            }
        }
        
        with open(output, 'w') as f:
            json.dump(output_data, f, indent=2)
        click.echo(f"\nResults saved to {output}")


@cli.command()
@click.argument('api_name')
@click.option('--test-prompt', '-t', default="Hello, please respond with 'OK'", 
              help='Test prompt to send')
@click.option('--model', '-m', help='Model to use (overrides config default)')
def test_api(api_name, test_prompt, model):
    """Test connectivity and configuration for a specific API."""
    # Validate API exists and is configured
    is_valid, error_msg = validate_api_config(api_name)
    if not is_valid:
        click.echo(f"Error: {error_msg}", err=True)
        sys.exit(1)
    
    # Get API configuration
    apis = get_available_apis()
    api_config = apis.get(api_name, {}).get('config', {})
    api_key = api_config.get('api_key', None)
    model_to_use = get_api_model(api_name, model)
    
    click.echo(f"Testing API: {api_name}")
    click.echo(f"Model: {model_to_use}")
    click.echo(f"Test prompt: {test_prompt}")
    click.echo("-" * 50)
    
    try:
        # Prepare messages for chat API
        messages_payload = [
            {
                "role": "user",
                "content": test_prompt
            }
        ]
        
        # Send test request via chat API
        response = chat_api_call(
            api_endpoint=api_name,
            messages_payload=messages_payload,
            api_key=api_key,
            model=model_to_use,
            streaming=False,
            max_tokens=100
        )
        
        # Handle response
        if hasattr(response, '__iter__') and not isinstance(response, str):
            response_text = ''.join(str(chunk) for chunk in response)
        else:
            response_text = str(response)
        
        click.echo(f"✓ API responded successfully")
        click.echo(f"Response: {response_text[:200]}..." if len(response_text) > 200 else f"Response: {response_text}")
        
    except Exception as e:
        click.echo(f"✗ API test failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('name')
@click.argument('config_file', type=click.Path(exists=True))
def register(name, config_file):
    """Register a custom benchmark from config file."""
    registry = get_registry()
    
    try:
        config = registry.load_config(config_file)
        if not config:
            click.echo(f"Error: Could not load config from {config_file}", err=True)
            sys.exit(1)
        
        # Override name if provided
        config.name = name
        
        registry.register(config)
        click.echo(f"Registered benchmark: {name}")
        
    except Exception as e:
        click.echo(f"Error registering benchmark: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('benchmark')
@click.option('--samples', '-s', type=int, default=3, help='Number of samples to show')
def validate(benchmark, samples):
    """Validate a benchmark by loading sample data."""
    registry = get_registry()
    
    config = registry.get(benchmark)
    if not config:
        click.echo(f"Error: Benchmark '{benchmark}' not found", err=True)
        sys.exit(1)
    
    click.echo(f"Validating benchmark: {benchmark}")
    click.echo(f"Type: {config.evaluation_type}")
    click.echo(f"Source: {config.dataset_source}")
    
    try:
        # Load sample data
        dataset = load_benchmark_dataset(benchmark, limit=samples)
        click.echo(f"\nLoaded {len(dataset)} samples successfully")
        
        # Show sample structure
        if dataset:
            click.echo("\nSample structure:")
            sample = dataset[0]
            for key, value in sample.items():
                if key != '_original':
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    elif isinstance(value, list):
                        value = f"[list with {len(value)} items]"
                    click.echo(f"  {key}: {value}")
        
        # Try to create evaluator
        evaluator = registry.create_evaluator(benchmark)
        if evaluator:
            click.echo("\nEvaluator created successfully")
            
            # Try formatting a sample
            if dataset:
                try:
                    formatted = evaluator.format_for_custom_metric(dataset[0])
                    click.echo("Sample formatted successfully for evaluation")
                except Exception as e:
                    click.echo(f"Warning: Could not format sample: {e}", err=True)
        else:
            click.echo("Warning: Evaluator not available for this type", err=True)
            
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def health():
    """Check evaluation system health including API configurations."""
    try:
        # Check evaluation manager
        manager = EvaluationManager()
        click.echo("✓ Evaluation manager initialized")
        
        # Check registry
        registry = get_registry()
        benchmarks = registry.list_benchmarks()
        click.echo(f"✓ Registry loaded with {len(benchmarks)} benchmarks")
        
        # Check database connection
        import sqlite3
        with sqlite3.connect(manager.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            click.echo(f"✓ Database connected ({table_count} tables)")
        
        # Check API configurations
        apis = get_available_apis()
        configured_count = sum(1 for api in apis.values() if api['configured'])
        click.echo(f"✓ APIs: {configured_count}/{len(apis)} configured")
        
        if configured_count == 0:
            click.echo("\n⚠ Warning: No APIs are configured")
            click.echo("  Configure at least one API in Config_Files/config.txt")
        else:
            click.echo("\nConfigured APIs:")
            for api_name, info in apis.items():
                if info['configured']:
                    click.echo(f"  - {api_name} ({info['category']})")
        
        click.echo("\nSystem health: OK")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def main():
    """Entry point for the CLI."""
    cli(obj={})


if __name__ == '__main__':
    main()