"""
Simple CLI for the evaluation module.

This provides a standalone CLI interface for running evaluations
without needing the full API server.
"""

import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from collections import defaultdict

import click
from loguru import logger
from tabulate import tabulate

from tldw_Server_API.app.core.Evaluations.evaluation_manager import EvaluationManager
from tldw_Server_API.app.core.Evaluations.benchmark_registry import get_registry
from tldw_Server_API.app.core.Evaluations.benchmark_loaders import load_benchmark_dataset
from tldw_Server_API.app.core.Evaluations.cli.api_utils import (
    get_available_apis,
    validate_api_config, 
    get_api_model,
    get_configured_apis,
    get_default_api,
    format_api_info
)
from tldw_Server_API.app.core.Chat.Chat_Functions import chat_api_call


@click.group(context_settings={'help_option_names': ['-h', '--help']})
@click.option('--config', type=click.Path(exists=True), help='Config file path')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']), 
              default='INFO', help='Logging level')
@click.pass_context
def cli(ctx, config, log_level):
    """tldw Evaluations CLI - Run evaluation benchmarks."""
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
                        click.echo(format_api_info(api_name, detailed=True))


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
@click.option('--system-prompt', '-s', help='Custom system prompt for evaluation')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.option('--parallel', '-p', type=int, default=1, help='Parallel workers')
@click.option('--dry-run', is_flag=True, help='Validate configuration without running')
@click.pass_context
def run(ctx, benchmark, limit, api, model, system_prompt, output, parallel, dry_run):
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
        api = get_default_api()
        if not api:
            click.echo("Error: No APIs are configured. Please configure at least one API.", err=True)
            click.echo("Available APIs that need configuration:")
            for name in sorted(get_available_apis().keys()):
                click.echo(f"  - {name}")
            sys.exit(1)
        click.echo(f"No API specified, using default: {api}")
    
    # Validate API configuration
    is_valid, error_msg = validate_api_config(api)
    if not is_valid:
        click.echo(f"Error: {error_msg}", err=True)
        sys.exit(1)
    
    # Get model
    model_to_use = get_api_model(api, model)
    
    # Default system prompt if not provided
    if not system_prompt:
        system_prompt = "You are an evaluation system. Provide accurate scores. Return only a numeric score between 0 and 1."
    
    click.echo(f"Running benchmark: {benchmark}")
    click.echo(f"Evaluation type: {config.evaluation_type}")
    click.echo(f"Using API: {api}")
    click.echo(f"Using model: {model_to_use}")
    click.echo(f"System prompt: {system_prompt[:100]}..." if len(system_prompt) > 100 else f"System prompt: {system_prompt}")
    
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
    
    # Get API configuration for actual evaluation
    apis = get_available_apis()
    api_config = apis.get(api, {}).get('config', {})
    api_key = api_config.get('api_key', None)
    
    results = []
    errors = 0
    
    # Process samples with actual LLM calls
    with click.progressbar(dataset, label='Evaluating') as samples:
        for i, sample in enumerate(samples):
            try:
                # Format for evaluation
                eval_data = evaluator.format_for_custom_metric(sample)
                
                # Prepare the evaluation prompt
                eval_prompt = config.get('prompt_template', 
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
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
                
                # Call the chat API
                response = chat_api_call(
                    api_endpoint=api,
                    messages_payload=messages_payload,
                    api_key=api_key,
                    model=model_to_use,
                    temp=0.7,
                    streaming=False,
                    max_tokens=100  # We only need a short response with the score
                )
                
                # Handle response (could be a generator even with streaming=False for some providers)
                if hasattr(response, '__iter__') and not isinstance(response, str):
                    response_text = ''.join(str(chunk) for chunk in response)
                else:
                    response_text = str(response)
                
                # Parse the response to extract score
                try:
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
                        "api": api,
                        "model": model_to_use,
                        **eval_data.get("metadata", {})
                    }
                }
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating sample {i}: {e}")
                results.append({
                    "sample_id": sample.get("id", i),
                    "score": 0.0,
                    "error": str(e),
                    "metadata": {
                        "api": api,
                        "model": model_to_use
                    }
                })
                errors += 1
    
    # Calculate summary statistics
    scores = [r['score'] for r in results if 'score' in r and not r.get('error')]
    errors_count = len([r for r in results if r.get('error')])
    
    if scores:
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
    else:
        avg_score = min_score = max_score = 0.0
    
    click.echo(f"\nResults:")
    click.echo(f"  Total samples: {len(results)}")
    click.echo(f"  Successful: {len(scores)}")
    click.echo(f"  Errors: {errors_count}")
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
            "system_prompt": system_prompt,
            "results": results,
            "summary": {
                "total": len(results),
                "successful": len(scores),
                "errors": errors_count,
                "average_score": avg_score,
                "min_score": min_score if scores else None,
                "max_score": max_score if scores else None
            }
        }
        
        with open(output, 'w') as f:
            json.dump(output_data, f, indent=2)
        click.echo(f"Results saved to {output}")


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
@click.argument('api_name')
@click.option('--test-prompt', '-t', default="Hello, please respond with 'OK'", 
              help='Test prompt to send')
@click.option('--model', '-m', help='Model to use (overrides config default)')
@click.option('--system-prompt', '-s', default="You are a helpful assistant.", 
              help='System prompt to use')
def test_api(api_name, test_prompt, model, system_prompt):
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
    click.echo(f"System prompt: {system_prompt}")
    click.echo("-" * 50)
    
    try:
        # Prepare messages for chat API
        messages_payload = [
            {
                "role": "system",
                "content": system_prompt
            },
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
            max_tokens=200
        )
        
        # Handle response
        if hasattr(response, '__iter__') and not isinstance(response, str):
            response_text = ''.join(str(chunk) for chunk in response)
        else:
            response_text = str(response)
        
        click.echo(f"✓ API responded successfully")
        click.echo(f"Response: {response_text[:500]}..." if len(response_text) > 500 else f"Response: {response_text}")
        
    except Exception as e:
        click.echo(f"✗ API test failed: {e}", err=True)
        sys.exit(1)


@cli.command()
def health():
    """Check evaluation system health including API configurations."""
    try:
        manager = EvaluationManager()
        click.echo("✓ Evaluation manager initialized")
        
        registry = get_registry()
        benchmarks = registry.list_benchmarks()
        click.echo(f"✓ Registry loaded with {len(benchmarks)} benchmarks")
        
        # Test database connection
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