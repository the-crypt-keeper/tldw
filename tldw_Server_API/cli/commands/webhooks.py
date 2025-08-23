"""
Webhook management commands for tldw Evaluations CLI.
"""

import sys
from typing import List

import click
from loguru import logger

from tldw_Server_API.cli.utils.output import (
    print_error, print_success, print_info, print_table, print_json
)


@click.group()
def webhook_group():
    """Webhook management commands."""
    pass


@webhook_group.command('register')
@click.argument('url')
@click.option('--events', help='Comma-separated list of events to subscribe to')
@click.option('--user', default='cli_user', help='User ID for webhook registration')
@click.pass_context
def register_webhook(ctx, url, events, user):
    """Register a webhook for evaluation events."""
    cli_context = ctx.obj['cli_context']
    
    try:
        cli_context.load_config()
        
        from tldw_Server_API.app.core.Evaluations.webhook_manager import webhook_manager, WebhookEvent
        
        # Parse events
        if events:
            event_list = [WebhookEvent(e.strip()) for e in events.split(',')]
        else:
            # Default to all evaluation events
            event_list = [
                WebhookEvent.EVALUATION_COMPLETED,
                WebhookEvent.EVALUATION_FAILED,
                WebhookEvent.BATCH_COMPLETED
            ]
        
        result = webhook_manager.register_webhook(user, url, event_list)
        
        print_success(f"Webhook registered successfully")
        print_info(f"Webhook ID: {result.get('webhook_id')}")
        print_info(f"Events: {', '.join([e.value for e in event_list])}")
        
    except Exception as e:
        logger.exception("Webhook registration failed")
        print_error(f"Webhook registration failed: {e}")
        sys.exit(1)


@webhook_group.command('list')
@click.option('--user', help='Filter by user ID')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table')
@click.pass_context
def list_webhooks(ctx, user, output_format):
    """List registered webhooks."""
    cli_context = ctx.obj['cli_context']
    
    try:
        cli_context.load_config()
        
        from tldw_Server_API.app.core.Evaluations.webhook_manager import webhook_manager
        
        webhooks = webhook_manager.get_webhook_status(user or 'cli_user')
        
        if output_format == 'json':
            print_json(webhooks, "Registered Webhooks")
        else:
            if webhooks:
                # Flatten webhook data for table display
                table_data = []
                for webhook in webhooks:
                    table_data.append({
                        'ID': webhook['id'],
                        'URL': webhook['url'][:50] + '...' if len(webhook['url']) > 50 else webhook['url'],
                        'Active': webhook['active'],
                        'Events': ', '.join(webhook['events'][:2]) + ('...' if len(webhook['events']) > 2 else ''),
                        'Success Rate': f"{webhook['statistics']['success_rate']:.2%}",
                        'Total Deliveries': webhook['statistics']['total_deliveries']
                    })
                print_table(table_data, "Registered Webhooks")
            else:
                print_info("No webhooks registered")
        
    except Exception as e:
        logger.exception("Webhook listing failed")
        print_error(f"Webhook listing failed: {e}")
        sys.exit(1)


@webhook_group.command('test')
@click.argument('url')
@click.option('--user', default='cli_user', help='User ID for webhook test')
@click.pass_context
def test_webhook(ctx, url, user):
    """Test webhook delivery."""
    cli_context = ctx.obj['cli_context']
    
    try:
        cli_context.load_config()
        
        from tldw_Server_API.app.core.Evaluations.webhook_manager import webhook_manager
        
        result = webhook_manager.test_webhook(user, url)
        
        if result['success']:
            print_success(f"Webhook test successful")
            print_info(f"Status Code: {result['status_code']}")
            print_info(f"Response Time: {result['response_time_ms']}ms")
        else:
            print_error(f"Webhook test failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
        
    except Exception as e:
        logger.exception("Webhook test failed")
        print_error(f"Webhook test failed: {e}")
        sys.exit(1)


@webhook_group.command('unregister')
@click.argument('url')
@click.option('--user', default='cli_user', help='User ID for webhook unregistration')
@click.pass_context
def unregister_webhook(ctx, url, user):
    """Unregister a webhook."""
    cli_context = ctx.obj['cli_context']
    
    try:
        cli_context.load_config()
        
        from tldw_Server_API.app.core.Evaluations.webhook_manager import webhook_manager
        
        result = webhook_manager.unregister_webhook(user, url)
        
        if result['success']:
            print_success("Webhook unregistered successfully")
        else:
            print_error(f"Webhook unregistration failed: {result.get('error')}")
            sys.exit(1)
        
    except Exception as e:
        logger.exception("Webhook unregistration failed")
        print_error(f"Webhook unregistration failed: {e}")
        sys.exit(1)


@webhook_group.command('status')
@click.option('--url', help='Check status for specific URL')
@click.option('--user', default='cli_user', help='User ID for webhook status')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table')
@click.pass_context
def webhook_status(ctx, url, user, output_format):
    """Show webhook status and statistics."""
    cli_context = ctx.obj['cli_context']
    
    try:
        cli_context.load_config()
        
        from tldw_Server_API.app.core.Evaluations.webhook_manager import webhook_manager
        
        webhooks = webhook_manager.get_webhook_status(user, url)
        
        if output_format == 'json':
            print_json(webhooks, "Webhook Status")
        else:
            if webhooks:
                for webhook in webhooks:
                    print_info(f"\nWebhook ID: {webhook['id']}")
                    print_info(f"URL: {webhook['url']}")
                    print_info(f"Active: {webhook['active']}")
                    print_info(f"Events: {', '.join(webhook['events'])}")
                    
                    stats = webhook['statistics']
                    stats_data = [
                        {'Metric': 'Total Deliveries', 'Value': stats['total_deliveries']},
                        {'Metric': 'Successful', 'Value': stats['successful_deliveries']},
                        {'Metric': 'Failed', 'Value': stats['failed_deliveries']},
                        {'Metric': 'Success Rate', 'Value': f"{stats['success_rate']:.2%}"}
                    ]
                    print_table(stats_data, "Delivery Statistics")
            else:
                print_info("No webhook found")
        
    except Exception as e:
        logger.exception("Webhook status check failed")
        print_error(f"Webhook status check failed: {e}")
        sys.exit(1)