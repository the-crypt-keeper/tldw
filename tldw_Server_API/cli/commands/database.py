"""
Database management commands for tldw Evaluations CLI.
"""

import sys
from pathlib import Path
from datetime import datetime

import click
from loguru import logger

from tldw_Server_API.cli.utils.output import (
    print_error, print_success, print_info, print_table, print_json
)


@click.group()
def db_group():
    """Database management commands."""
    pass


@db_group.command('init')
@click.pass_context
def init_db(ctx):
    """Initialize database and run migrations."""
    cli_context = ctx.obj['cli_context']
    
    try:
        cli_context.load_config()
        db_path = cli_context.config['database']['path']
        
        # Ensure database directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Run migrations
        from tldw_Server_API.app.core.DB_Management.migrations_v6_audit_logging import run_migration
        run_migration(db_path)
        
        print_success(f"Database initialized at {db_path}")
        
    except Exception as e:
        logger.exception("Database initialization failed")
        print_error(f"Database initialization failed: {e}")
        sys.exit(1)


@db_group.command('status')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table')
@click.pass_context
def db_status(ctx, output_format):
    """Show database status and statistics."""
    cli_context = ctx.obj['cli_context']
    
    try:
        cli_context.load_config()
        
        from tldw_Server_API.app.core.Evaluations.connection_pool import get_connection_stats, get_connection_health
        
        stats = get_connection_stats()
        health = get_connection_health()
        
        status_data = {
            'database_path': cli_context.config['database']['path'],
            'connection_health': health['status'],
            'total_connections': stats.total_connections,
            'active_connections': stats.active_connections,
            'idle_connections': stats.idle_connections,
            'checkout_count': stats.checkout_count,
            'connection_errors': stats.connection_errors
        }
        
        if output_format == 'json':
            print_json(status_data, "Database Status")
        else:
            table_data = [{'Metric': k.replace('_', ' ').title(), 'Value': v} for k, v in status_data.items()]
            print_table(table_data, "Database Status")
        
    except Exception as e:
        logger.exception("Database status check failed")
        print_error(f"Database status check failed: {e}")
        sys.exit(1)


@db_group.command('backup')
@click.argument('backup_path', type=click.Path(path_type=Path))
@click.pass_context
def backup_db(ctx, backup_path):
    """Create database backup."""
    cli_context = ctx.obj['cli_context']
    
    try:
        cli_context.load_config()
        db_path = Path(cli_context.config['database']['path'])
        
        if not db_path.exists():
            print_error(f"Database file not found: {db_path}")
            sys.exit(1)
        
        # Ensure backup directory exists
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Simple file copy for SQLite
        import shutil
        shutil.copy2(db_path, backup_path)
        
        print_success(f"Database backed up to {backup_path}")
        
    except Exception as e:
        logger.exception("Database backup failed")
        print_error(f"Database backup failed: {e}")
        sys.exit(1)


@db_group.command('cleanup')
@click.option('--days', type=int, default=90, help='Days to retain records')
@click.option('--dry-run', is_flag=True, help='Show what would be deleted without deleting')
@click.pass_context
def cleanup_db(ctx, days, dry_run):
    """Clean up old evaluation records."""
    cli_context = ctx.obj['cli_context']
    
    try:
        cli_context.load_config()
        
        from tldw_Server_API.app.core.Evaluations.audit_logger import audit_logger
        
        if dry_run:
            print_info(f"DRY RUN: Would clean up records older than {days} days")
        else:
            cleaned_count = audit_logger.cleanup_old_events(days)
            print_success(f"Cleaned up {cleaned_count} old records")
        
    except Exception as e:
        logger.exception("Database cleanup failed")
        print_error(f"Database cleanup failed: {e}")
        sys.exit(1)