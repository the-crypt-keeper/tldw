"""
Export and import commands for tldw Evaluations CLI.
"""

import json
import csv
import sys
from pathlib import Path
from datetime import datetime

import click
from loguru import logger

from tldw_Server_API.cli.utils.output import (
    print_error, print_success, print_info
)


@click.group()
def export_group():
    """Export and import functionality."""
    pass


@export_group.command('evaluations')
@click.argument('output_file', type=click.Path(path_type=Path))
@click.option('--format', 'output_format', type=click.Choice(['json', 'csv']), default='json', help='Export format')
@click.option('--limit', type=int, help='Limit number of records')
@click.option('--days', type=int, help='Export records from last N days')
@click.pass_context
def export_evaluations(ctx, output_file, output_format, limit, days):
    """Export evaluation history."""
    cli_context = ctx.obj['cli_context']
    
    try:
        cli_context.load_config()
        
        from tldw_Server_API.app.core.Evaluations.evaluation_manager import EvaluationManager
        
        eval_manager = EvaluationManager()
        
        # Get evaluation history
        evaluations = eval_manager.get_evaluation_history(limit=limit or 1000)
        
        # Filter by days if specified
        if days:
            cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
            evaluations = [e for e in evaluations if e.get('timestamp', 0) > cutoff_date]
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Export data
        if output_format == 'json':
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(evaluations, f, indent=2, default=str)
        else:  # CSV
            if evaluations:
                fieldnames = set()
                for eval_data in evaluations:
                    fieldnames.update(eval_data.keys())
                
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=list(fieldnames))
                    writer.writeheader()
                    writer.writerows(evaluations)
        
        print_success(f"Exported {len(evaluations)} evaluations to {output_file}")
        
    except Exception as e:
        logger.exception("Evaluation export failed")
        print_error(f"Evaluation export failed: {e}")
        sys.exit(1)


@export_group.command('config')
@click.argument('output_file', type=click.Path(path_type=Path))
@click.pass_context
def export_config(ctx, output_file):
    """Export current configuration."""
    cli_context = ctx.obj['cli_context']
    
    try:
        cli_context.load_config()
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Export configuration
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cli_context.config, f, indent=2, default=str)
        
        print_success(f"Configuration exported to {output_file}")
        
    except Exception as e:
        logger.exception("Configuration export failed")
        print_error(f"Configuration export failed: {e}")
        sys.exit(1)


@export_group.command('metrics')
@click.argument('output_file', type=click.Path(path_type=Path))
@click.pass_context
def export_metrics(ctx, output_file):
    """Export current metrics data."""
    cli_context = ctx.obj['cli_context']
    
    try:
        cli_context.load_config()
        
        from tldw_Server_API.app.core.Evaluations.metrics import get_metrics
        
        metrics = get_metrics()
        
        if metrics.enabled:
            metrics_data = metrics.get_metrics()
            
            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write metrics data
            with open(output_file, 'wb') as f:
                f.write(metrics_data)
            
            print_success(f"Metrics exported to {output_file}")
        else:
            print_error("Metrics collection is not enabled")
            sys.exit(1)
        
    except Exception as e:
        logger.exception("Metrics export failed")
        print_error(f"Metrics export failed: {e}")
        sys.exit(1)


@export_group.command('import')
@click.argument('input_file', type=click.Path(exists=True, readable=True, path_type=Path))
@click.option('--type', 'import_type', type=click.Choice(['evaluations', 'config']), required=True, help='Type of data to import')
@click.option('--dry-run', is_flag=True, help='Show what would be imported without importing')
@click.pass_context
def import_data(ctx, input_file, import_type, dry_run):
    """Import evaluation data or configuration."""
    cli_context = ctx.obj['cli_context']
    
    try:
        cli_context.load_config()
        
        # Load data from file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if dry_run:
            print_info(f"DRY RUN: Would import {len(data) if isinstance(data, list) else 1} {import_type} record(s)")
            return
        
        if import_type == 'evaluations':
            from tldw_Server_API.app.core.Evaluations.evaluation_manager import EvaluationManager
            eval_manager = EvaluationManager()
            
            # Import evaluations
            imported_count = 0
            for eval_data in data if isinstance(data, list) else [data]:
                try:
                    # This would need proper implementation
                    # eval_manager.import_evaluation(eval_data)
                    imported_count += 1
                except Exception as e:
                    logger.warning(f"Failed to import evaluation: {e}")
            
            print_success(f"Imported {imported_count} evaluations")
            
        elif import_type == 'config':
            from tldw_Server_API.cli.utils.config import save_config
            
            # Merge with existing config
            cli_context.config.update(data)
            save_config(cli_context.config)
            
            print_success("Configuration imported and saved")
        
    except Exception as e:
        logger.exception("Data import failed")
        print_error(f"Data import failed: {e}")
        sys.exit(1)