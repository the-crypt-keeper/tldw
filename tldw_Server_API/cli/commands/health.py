"""
Health and status commands for tldw Evaluations CLI.

Provides comprehensive health monitoring and status reporting
for all evaluation system components.
"""

import asyncio
import sys
from typing import Dict, Any

import click
from loguru import logger

from tldw_Server_API.cli.utils.output import (
    print_error, print_success, print_info, print_health_status,
    print_table, print_json, print_metrics_summary, format_timestamp,
    format_bytes, format_duration
)


@click.group()
def health_group():
    """Health monitoring and status commands."""
    pass


@health_group.command('check')
@click.option('--components', is_flag=True, help='Show detailed component health')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table', help='Output format')
@click.pass_context
def health_check(ctx, components, output_format):
    """
    Check overall system health.
    
    Performs comprehensive health checks on all evaluation system components
    including database connections, configuration validity, and service status.
    
    Examples:
        tldw-evals health check                    # Basic health check
        tldw-evals health check --components       # Detailed component status
        tldw-evals health check --format json     # JSON output
    """
    cli_context = ctx.obj['cli_context']
    
    try:
        cli_context.load_config()
        health_data = _perform_health_check(cli_context.config, detailed=components)
        
        if output_format == 'json':
            print_json(health_data, "System Health Status")
        else:
            print_health_status(health_data)
        
        # Exit with error code if unhealthy
        if health_data.get('status') == 'unhealthy':
            sys.exit(1)
            
    except Exception as e:
        logger.exception("Health check failed")
        print_error(f"Health check failed: {e}")
        sys.exit(1)


@health_group.command('status')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table', help='Output format')
@click.pass_context
def status(ctx, output_format):
    """
    Show current system status and basic metrics.
    
    Displays system uptime, recent activity, and key performance indicators.
    
    Examples:
        tldw-evals health status              # Show system status
        tldw-evals health status --format json    # JSON output
    """
    cli_context = ctx.obj['cli_context']
    
    try:
        cli_context.load_config()
        status_data = _get_system_status(cli_context.config)
        
        if output_format == 'json':
            print_json(status_data, "System Status")
        else:
            _display_status_table(status_data)
            
    except Exception as e:
        logger.exception("Status check failed")
        print_error(f"Status check failed: {e}")
        sys.exit(1)


@health_group.command('metrics')
@click.option('--component', help='Show metrics for specific component')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table', help='Output format')
@click.pass_context
def metrics(ctx, component, output_format):
    """
    Display system metrics and performance data.
    
    Shows detailed metrics including request counts, response times,
    error rates, and resource utilization.
    
    Examples:
        tldw-evals health metrics                    # All metrics
        tldw-evals health metrics --component database    # Database metrics only
    """
    cli_context = ctx.obj['cli_context']
    
    try:
        cli_context.load_config()
        metrics_data = _get_system_metrics(cli_context.config, component)
        
        if output_format == 'json':
            print_json(metrics_data, "System Metrics")
        else:
            print_metrics_summary(metrics_data)
            
    except Exception as e:
        logger.exception("Metrics collection failed")
        print_error(f"Metrics collection failed: {e}")
        sys.exit(1)


def _perform_health_check(config: Dict[str, Any], detailed: bool = False) -> Dict[str, Any]:
    """Perform comprehensive health check."""
    health_data = {
        'status': 'healthy',
        'timestamp': format_timestamp(None),
        'components': {}
    }
    
    # Database health check
    try:
        from tldw_Server_API.app.core.Evaluations.connection_pool import get_connection_health
        db_health = get_connection_health()
        health_data['components']['database'] = db_health
        
        if db_health['status'] in ['unhealthy', 'degraded']:
            health_data['status'] = 'degraded'
            
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        health_data['components']['database'] = {
            'status': 'error',
            'error': str(e)
        }
        health_data['status'] = 'unhealthy'
    
    # Configuration validation
    try:
        from tldw_Server_API.app.core.Evaluations.config_manager import validate_config
        config_errors = validate_config()
        
        if config_errors:
            health_data['components']['configuration'] = {
                'status': 'error',
                'errors': config_errors
            }
            health_data['status'] = 'degraded'
        else:
            health_data['components']['configuration'] = {
                'status': 'ok',
                'message': 'Configuration is valid'
            }
            
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        health_data['components']['configuration'] = {
            'status': 'error',
            'error': str(e)
        }
        health_data['status'] = 'degraded'
    
    # Rate limiting service health
    try:
        from tldw_Server_API.app.core.Evaluations.user_rate_limiter import user_rate_limiter
        # Simple health check - try to get user stats
        test_stats = user_rate_limiter.get_user_stats('health_check_user')
        health_data['components']['rate_limiting'] = {
            'status': 'ok',
            'message': 'Rate limiting service operational'
        }
        
    except Exception as e:
        logger.error(f"Rate limiting health check failed: {e}")
        health_data['components']['rate_limiting'] = {
            'status': 'error',
            'error': str(e)
        }
        health_data['status'] = 'degraded'
    
    # Webhook service health
    try:
        from tldw_Server_API.app.core.Evaluations.webhook_manager import webhook_manager
        webhook_health = {
            'status': 'ok',
            'message': 'Webhook service operational'
        }
        health_data['components']['webhooks'] = webhook_health
        
    except Exception as e:
        logger.error(f"Webhook service health check failed: {e}")
        health_data['components']['webhooks'] = {
            'status': 'error',
            'error': str(e)
        }
        health_data['status'] = 'degraded'
    
    # Metrics collection health
    try:
        from tldw_Server_API.app.core.Evaluations.metrics import get_metrics
        metrics = get_metrics()
        metrics_health = metrics.get_health_metrics()
        health_data['components']['metrics'] = {
            'status': 'ok',
            'enabled': metrics_health.get('metrics_enabled', False)
        }
        
    except Exception as e:
        logger.error(f"Metrics health check failed: {e}")
        health_data['components']['metrics'] = {
            'status': 'error',
            'error': str(e)
        }
        # Metrics failure is not critical
    
    # Audit logging health
    try:
        from tldw_Server_API.app.core.Evaluations.audit_logger import audit_logger
        # Test audit logging
        audit_logger.get_recent_events(limit=1)
        health_data['components']['audit_logging'] = {
            'status': 'ok',
            'message': 'Audit logging operational'
        }
        
    except Exception as e:
        logger.error(f"Audit logging health check failed: {e}")
        health_data['components']['audit_logging'] = {
            'status': 'error',
            'error': str(e)
        }
        # Audit logging failure is not critical for basic operations
    
    return health_data


def _get_system_status(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get current system status."""
    status_data = {
        'timestamp': format_timestamp(None),
        'configuration': {
            'config_loaded': bool(config),
            'database_path': config.get('database', {}).get('path', 'Unknown'),
            'log_level': config.get('monitoring', {}).get('logging', {}).get('level', 'INFO')
        }
    }
    
    # Database statistics
    try:
        from tldw_Server_API.app.core.Evaluations.connection_pool import get_connection_stats
        db_stats = get_connection_stats()
        status_data['database'] = {
            'total_connections': db_stats.total_connections,
            'active_connections': db_stats.active_connections,
            'idle_connections': db_stats.idle_connections,
            'checkout_count': db_stats.checkout_count,
            'avg_checkout_time': f"{db_stats.avg_checkout_time:.3f}s"
        }
        
    except Exception as e:
        logger.warning(f"Could not get database stats: {e}")
        status_data['database'] = {'error': str(e)}
    
    # Evaluation statistics
    try:
        from tldw_Server_API.app.core.Evaluations.evaluation_manager import EvaluationManager
        eval_manager = EvaluationManager()
        
        # Get recent evaluation count (this is a simple example)
        recent_evals = eval_manager.get_evaluation_history(limit=10)
        status_data['evaluations'] = {
            'recent_count': len(recent_evals),
            'last_evaluation': format_timestamp(recent_evals[0]['timestamp']) if recent_evals else 'None'
        }
        
    except Exception as e:
        logger.warning(f"Could not get evaluation stats: {e}")
        status_data['evaluations'] = {'error': str(e)}
    
    return status_data


def _get_system_metrics(config: Dict[str, Any], component: str = None) -> Dict[str, Any]:
    """Get system metrics."""
    metrics_data = {}
    
    try:
        from tldw_Server_API.app.core.Evaluations.metrics import get_metrics
        metrics = get_metrics()
        
        if metrics.enabled:
            # Get Prometheus metrics
            metrics_bytes = metrics.get_metrics()
            metrics_data['prometheus_enabled'] = True
            metrics_data['metrics_size'] = format_bytes(len(metrics_bytes))
        else:
            metrics_data['prometheus_enabled'] = False
            metrics_data['message'] = 'Prometheus client not available'
        
        # Add health metrics
        health_metrics = metrics.get_health_metrics()
        metrics_data.update(health_metrics)
        
    except Exception as e:
        logger.warning(f"Could not collect metrics: {e}")
        metrics_data['error'] = str(e)
    
    # Database metrics
    if not component or component == 'database':
        try:
            from tldw_Server_API.app.core.Evaluations.connection_pool import get_connection_stats
            db_stats = get_connection_stats()
            
            metrics_data['database'] = {
                'total_connections': db_stats.total_connections,
                'active_connections': db_stats.active_connections,
                'idle_connections': db_stats.idle_connections,
                'created_connections': db_stats.created_connections,
                'closed_connections': db_stats.closed_connections,
                'checkout_count': db_stats.checkout_count,
                'avg_checkout_time': format_duration(db_stats.avg_checkout_time),
                'max_checkout_time': format_duration(db_stats.max_checkout_time),
                'connection_errors': db_stats.connection_errors,
                'pool_exhausted_count': db_stats.pool_exhausted_count
            }
            
        except Exception as e:
            logger.warning(f"Could not get database metrics: {e}")
            if 'database' not in metrics_data:
                metrics_data['database'] = {}
            metrics_data['database']['error'] = str(e)
    
    return metrics_data


def _display_status_table(status_data: Dict[str, Any]):
    """Display status data in table format."""
    from rich.table import Table
    from tldw_Server_API.cli.utils.output import console
    
    # Configuration table
    if 'configuration' in status_data:
        config_table = Table(title="Configuration Status", show_header=True, header_style="bold cyan")
        config_table.add_column("Setting")
        config_table.add_column("Value")
        
        for key, value in status_data['configuration'].items():
            config_table.add_row(key.replace('_', ' ').title(), str(value))
        
        console.print(config_table)
    
    # Database table
    if 'database' in status_data and 'error' not in status_data['database']:
        db_table = Table(title="Database Status", show_header=True, header_style="bold green")
        db_table.add_column("Metric")
        db_table.add_column("Value")
        
        for key, value in status_data['database'].items():
            db_table.add_row(key.replace('_', ' ').title(), str(value))
        
        console.print(db_table)
    
    # Evaluations table
    if 'evaluations' in status_data and 'error' not in status_data['evaluations']:
        eval_table = Table(title="Evaluation Status", show_header=True, header_style="bold magenta")
        eval_table.add_column("Metric")
        eval_table.add_column("Value")
        
        for key, value in status_data['evaluations'].items():
            eval_table.add_row(key.replace('_', ' ').title(), str(value))
        
        console.print(eval_table)
    
    # Show timestamp
    print_info(f"Status checked at: {status_data['timestamp']}")