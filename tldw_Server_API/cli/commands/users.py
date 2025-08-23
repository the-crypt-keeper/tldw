"""
User and rate limit management commands for tldw Evaluations CLI.
"""

import sys
from typing import Dict, Any

import click
from loguru import logger

from tldw_Server_API.cli.utils.output import (
    print_error, print_success, print_info, print_table, print_json
)


@click.group()
def users_group():
    """User and rate limit management commands."""
    pass


@users_group.command('list')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table')
@click.pass_context
def list_users(ctx, output_format):
    """List users and their tiers."""
    cli_context = ctx.obj['cli_context']
    
    try:
        cli_context.load_config()
        
        from tldw_Server_API.app.core.Evaluations.user_rate_limiter import user_rate_limiter
        
        # Get user list (simplified - would need proper user management)
        users_data = []  # This would come from actual user database
        
        if output_format == 'json':
            print_json(users_data, "Users List")
        else:
            if users_data:
                print_table(users_data, "Users and Tiers")
            else:
                print_info("No users found")
        
    except Exception as e:
        logger.exception("User listing failed")
        print_error(f"User listing failed: {e}")
        sys.exit(1)


@users_group.command('limits')
@click.argument('user_id')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table')
@click.pass_context
def show_user_limits(ctx, user_id, output_format):
    """Show user's current limits and usage."""
    cli_context = ctx.obj['cli_context']
    
    try:
        cli_context.load_config()
        
        from tldw_Server_API.app.core.Evaluations.user_rate_limiter import user_rate_limiter
        
        stats = user_rate_limiter.get_user_stats(user_id)
        
        if output_format == 'json':
            print_json(stats, f"User Limits: {user_id}")
        else:
            if stats:
                table_data = [{'Metric': k.replace('_', ' ').title(), 'Value': v} for k, v in stats.items()]
                print_table(table_data, f"User Limits: {user_id}")
            else:
                print_info(f"No data found for user: {user_id}")
        
    except Exception as e:
        logger.exception("User limits retrieval failed")
        print_error(f"User limits retrieval failed: {e}")
        sys.exit(1)


@users_group.command('set-tier')
@click.argument('user_id')
@click.argument('tier', type=click.Choice(['free', 'basic', 'premium', 'enterprise', 'custom']))
@click.pass_context
def set_user_tier(ctx, user_id, tier):
    """Set user tier."""
    cli_context = ctx.obj['cli_context']
    
    try:
        cli_context.load_config()
        
        from tldw_Server_API.app.core.Evaluations.user_rate_limiter import user_rate_limiter, UserTier
        
        tier_enum = UserTier(tier)
        success = user_rate_limiter.set_user_tier(user_id, tier_enum)
        
        if success:
            print_success(f"User {user_id} tier set to {tier}")
        else:
            print_error(f"Failed to set tier for user {user_id}")
            sys.exit(1)
        
    except Exception as e:
        logger.exception("User tier setting failed")
        print_error(f"User tier setting failed: {e}")
        sys.exit(1)


@users_group.command('reset')
@click.argument('user_id')
@click.pass_context
def reset_user_limits(ctx, user_id):
    """Reset user's rate limits."""
    cli_context = ctx.obj['cli_context']
    
    try:
        cli_context.load_config()
        
        from tldw_Server_API.app.core.Evaluations.user_rate_limiter import user_rate_limiter
        
        success = user_rate_limiter.reset_user_limits(user_id)
        
        if success:
            print_success(f"Rate limits reset for user {user_id}")
        else:
            print_error(f"Failed to reset limits for user {user_id}")
            sys.exit(1)
        
    except Exception as e:
        logger.exception("User limits reset failed")
        print_error(f"User limits reset failed: {e}")
        sys.exit(1)