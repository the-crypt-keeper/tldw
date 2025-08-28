"""
Configuration management commands for tldw Evaluations CLI.
"""

import sys
from typing import Dict, Any

import click
from loguru import logger

from tldw_Server_API.cli.utils.output import (
    print_error, print_success, print_info, print_config_section, print_json
)
from tldw_Server_API.cli.utils.config import save_config, get_config_value


@click.group()
def config_group():
    """Configuration management commands."""
    pass


@config_group.command('show')
@click.option('--section', help='Show specific configuration section')
@click.option('--format', 'output_format', type=click.Choice(['yaml', 'json']), default='yaml')
@click.pass_context
def show_config(ctx, section, output_format):
    """Display current configuration."""
    cli_context = ctx.obj['cli_context']
    
    try:
        cli_context.load_config()
        
        if output_format == 'json':
            if section:
                section_data = cli_context.config.get(section, {})
                print_json(section_data, f"Configuration Section: {section}")
            else:
                print_json(cli_context.config, "Full Configuration")
        else:
            print_config_section(cli_context.config, section)
        
    except Exception as e:
        logger.exception("Configuration display failed")
        print_error(f"Configuration display failed: {e}")
        sys.exit(1)


@config_group.command('validate')
@click.pass_context
def validate_config(ctx):
    """Validate configuration files."""
    cli_context = ctx.obj['cli_context']
    
    try:
        cli_context.load_config()
        
        from tldw_Server_API.app.core.Evaluations.config_manager import validate_config
        errors = validate_config()
        
        if errors:
            print_error("Configuration validation failed:")
            for error in errors:
                print_error(f"  - {error}")
            sys.exit(1)
        else:
            print_success("Configuration is valid")
        
    except Exception as e:
        logger.exception("Configuration validation failed")
        print_error(f"Configuration validation failed: {e}")
        sys.exit(1)


@config_group.command('set')
@click.argument('path')
@click.argument('value')
@click.option('--type', 'value_type', type=click.Choice(['str', 'int', 'float', 'bool']), default='str')
@click.pass_context
def set_config_value(ctx, path, value, value_type):
    """Set configuration value by path."""
    cli_context = ctx.obj['cli_context']
    
    try:
        cli_context.load_config()
        
        # Convert value to appropriate type
        if value_type == 'int':
            value = int(value)
        elif value_type == 'float':
            value = float(value)
        elif value_type == 'bool':
            value = value.lower() in ('true', '1', 'yes', 'on')
        
        # Set value in config
        _set_nested_value(cli_context.config, path, value)
        
        # Save config
        save_config(cli_context.config)
        
        print_success(f"Configuration updated: {path} = {value}")
        
    except Exception as e:
        logger.exception("Configuration update failed")
        print_error(f"Configuration update failed: {e}")
        sys.exit(1)


@config_group.command('reload')
@click.pass_context
def reload_config(ctx):
    """Reload configuration (if hot reload enabled)."""
    cli_context = ctx.obj['cli_context']
    
    try:
        from tldw_Server_API.app.core.Evaluations.config_manager import reload_config
        
        if reload_config():
            print_success("Configuration reloaded successfully")
        else:
            print_error("Configuration reload failed")
            sys.exit(1)
        
    except Exception as e:
        logger.exception("Configuration reload failed")
        print_error(f"Configuration reload failed: {e}")
        sys.exit(1)


def _set_nested_value(config: Dict[str, Any], path: str, value: Any):
    """Set nested configuration value by dot-separated path."""
    keys = path.split('.')
    current = config
    
    # Navigate to parent of target key
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set the value
    current[keys[-1]] = value