"""
Simple CLI for the evaluation module.

This provides a standalone CLI interface for running evaluations
without needing the full API server.
"""

import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any

import click
from loguru import logger

from tldw_Server_API.app.core.Evaluations.evaluation_manager import EvaluationManager
from tldw_Server_API.app.core.Evaluations.benchmark_registry import get_registry
from tldw_Server_API.app.core.Evaluations.benchmark_loaders import load_benchmark_dataset


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
@click.option('--api', '-a', default='openai', help='API to use for evaluation')
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.option('--parallel', '-p', type=int, default=1, help='Parallel workers')
@click.pass_context
def run(ctx, benchmark, limit, api, output, parallel):
    """Run a benchmark evaluation."""
    registry = ctx.obj['registry']
    manager = ctx.obj['manager']
    
    # Check if benchmark exists
    config = registry.get(benchmark)
    if not config:
        click.echo(f"Error: Benchmark '{benchmark}' not found", err=True)
        sys.exit(1)
    
    click.echo(f"Running benchmark: {benchmark}")
    click.echo(f"Evaluation type: {config.evaluation_type}")
    
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
    
    # Process samples
    with click.progressbar(dataset, label='Evaluating') as samples:
        for sample in samples:
            try:
                # Format for evaluation
                eval_data = evaluator.format_for_custom_metric(sample)
                
                # Run evaluation (simplified - would use async in production)
                result = {
                    "sample_id": sample.get("id", len(results)),
                    "score": 0.5,  # Placeholder - would call actual evaluation
                    "metadata": eval_data.get("metadata", {})
                }
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating sample: {e}")
                errors += 1
    
    # Calculate summary statistics
    total = len(results)
    avg_score = sum(r['score'] for r in results) / total if total > 0 else 0
    
    click.echo(f"\nResults:")
    click.echo(f"  Total samples: {total}")
    click.echo(f"  Average score: {avg_score:.3f}")
    click.echo(f"  Errors: {errors}")
    
    # Save results if requested
    if output:
        output_data = {
            "benchmark": benchmark,
            "results": results,
            "summary": {
                "total": total,
                "average_score": avg_score,
                "errors": errors
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
def health():
    """Check evaluation system health."""
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
        
        click.echo("\nSystem health: OK")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def main():
    """Entry point for the CLI."""
    cli(obj={})


if __name__ == '__main__':
    main()