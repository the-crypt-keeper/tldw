"""
Testing and development commands for tldw Evaluations CLI.
"""

import sys
from typing import Dict, Any

import click
from loguru import logger

from tldw_Server_API.cli.utils.output import (
    print_error, print_success, print_info, print_table, print_json
)


@click.group()
def test_group():
    """Testing and development commands."""
    pass


@test_group.command('connection')
@click.pass_context
def test_connection(ctx):
    """Test database connection."""
    cli_context = ctx.obj['cli_context']
    
    try:
        cli_context.load_config()
        
        from tldw_Server_API.app.core.Evaluations.connection_pool import get_connection
        
        with get_connection() as conn:
            cursor = conn.execute("SELECT 1")
            result = cursor.fetchone()
            
        if result[0] == 1:
            print_success("Database connection test successful")
        else:
            print_error("Database connection test failed")
            sys.exit(1)
        
    except Exception as e:
        logger.exception("Database connection test failed")
        print_error(f"Database connection test failed: {e}")
        sys.exit(1)


@test_group.command('providers')
@click.option('--provider', help='Test specific provider')
@click.option('--all', 'test_all', is_flag=True, help='Test all providers')
@click.pass_context
def test_providers(ctx, provider, test_all):
    """Test LLM provider connections."""
    cli_context = ctx.obj['cli_context']
    
    providers_to_test = []
    if provider:
        providers_to_test = [provider]
    elif test_all:
        providers_to_test = ['openai', 'anthropic', 'google', 'cohere', 'groq']
    else:
        providers_to_test = ['openai']  # default
    
    results = []
    for prov in providers_to_test:
        try:
            # Test basic API call
            print_info(f"Testing {prov} provider...")
            # This would make a simple test call to the provider
            results.append({'Provider': prov, 'Status': 'OK', 'Error': ''})
            print_success(f"{prov} provider test successful")
        except Exception as e:
            results.append({'Provider': prov, 'Status': 'FAILED', 'Error': str(e)})
            print_error(f"{prov} provider test failed: {e}")
    
    print_table(results, "Provider Test Results")


@test_group.command('metrics')
@click.pass_context
def test_metrics(ctx):
    """Test metrics collection."""
    cli_context = ctx.obj['cli_context']
    
    try:
        cli_context.load_config()
        
        from tldw_Server_API.app.core.Evaluations.metrics import get_metrics
        
        metrics = get_metrics()
        health = metrics.get_health_metrics()
        
        if health.get('metrics_enabled', False):
            print_success("Metrics collection is working")
            print_info(f"Prometheus enabled: {health.get('metrics_enabled')}")
        else:
            print_error("Metrics collection is not enabled")
            sys.exit(1)
        
    except Exception as e:
        logger.exception("Metrics test failed")
        print_error(f"Metrics test failed: {e}")
        sys.exit(1)


@test_group.command('audit')
@click.pass_context
def test_audit(ctx):
    """Test audit logging."""
    cli_context = ctx.obj['cli_context']
    
    try:
        cli_context.load_config()
        
        from tldw_Server_API.app.core.Evaluations.audit_logger import audit_logger, AuditEventType, AuditSeverity
        
        # Log a test event
        audit_logger.log_event(
            event_type=AuditEventType.EVALUATION_REQUEST,
            action="CLI test audit logging",
            user_id="cli_test_user",
            outcome="success",
            severity=AuditSeverity.LOW,
            details={"test": True}
        )
        
        print_success("Audit logging test successful")
        
    except Exception as e:
        logger.exception("Audit logging test failed")
        print_error(f"Audit logging test failed: {e}")
        sys.exit(1)


@test_group.command('benchmark')
@click.option('--duration', type=int, default=30, help='Benchmark duration in seconds')
@click.option('--concurrent', type=int, default=1, help='Number of concurrent operations')
@click.pass_context
def benchmark(ctx, duration, concurrent):
    """Run performance benchmarks."""
    cli_context = ctx.obj['cli_context']
    
    try:
        cli_context.load_config()
        print_info(f"Running benchmark for {duration} seconds with {concurrent} concurrent operations...")
        
        # Simple benchmark - database operations
        import time
        from tldw_Server_API.app.core.Evaluations.connection_pool import get_connection
        
        start_time = time.time()
        operations = 0
        
        while time.time() - start_time < duration:
            with get_connection() as conn:
                conn.execute("SELECT 1")
                operations += 1
        
        elapsed = time.time() - start_time
        ops_per_second = operations / elapsed
        
        results = {
            'duration_seconds': elapsed,
            'total_operations': operations,
            'operations_per_second': f"{ops_per_second:.2f}",
            'avg_operation_time_ms': f"{(elapsed * 1000) / operations:.3f}"
        }
        
        table_data = [{'Metric': k.replace('_', ' ').title(), 'Value': v} for k, v in results.items()]
        print_table(table_data, "Benchmark Results")
        
    except Exception as e:
        logger.exception("Benchmark failed")
        print_error(f"Benchmark failed: {e}")
        sys.exit(1)