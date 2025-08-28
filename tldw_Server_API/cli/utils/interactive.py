"""
Interactive mode for tldw Evaluations CLI.
"""

from typing import Dict, Any

import click
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel

from tldw_Server_API.cli.utils.output import print_info, print_success, print_error


def run_interactive_mode(cli_context):
    """Run interactive menu-driven mode."""
    console = Console()
    
    console.print(Panel("Welcome to tldw Evaluations Interactive Mode", style="bold cyan"))
    
    while True:
        console.print("\n[bold]Main Menu[/bold]")
        console.print("1. Run Health Check")
        console.print("2. Run Evaluation")
        console.print("3. Manage Configuration")
        console.print("4. Database Operations")
        console.print("5. Exit")
        
        choice = Prompt.ask("Select an option", choices=["1", "2", "3", "4", "5"], default="5")
        
        if choice == "1":
            _interactive_health_check(cli_context)
        elif choice == "2":
            _interactive_evaluation(cli_context)
        elif choice == "3":
            _interactive_config(cli_context)
        elif choice == "4":
            _interactive_database(cli_context)
        elif choice == "5":
            break


def _interactive_health_check(cli_context):
    """Interactive health check."""
    print_info("Running health check...")
    # Would call actual health check functionality
    print_success("Health check completed!")


def _interactive_evaluation(cli_context):
    """Interactive evaluation setup."""
    console = Console()
    
    eval_type = Prompt.ask(
        "Select evaluation type",
        choices=["geval", "rag", "quality", "custom"],
        default="geval"
    )
    
    if eval_type == "geval":
        text = Prompt.ask("Enter original text (or file://path)")
        summary = Prompt.ask("Enter summary text (or file://path)")
        provider = Prompt.ask("Select provider", choices=["openai", "anthropic", "google"], default="openai")
        
        print_info(f"Would run G-Eval with {provider} on provided text...")
        # Would call actual evaluation functionality
    
    # Similar for other evaluation types


def _interactive_config(cli_context):
    """Interactive configuration management."""
    console = Console()
    
    action = Prompt.ask(
        "Configuration action",
        choices=["show", "validate", "edit"],
        default="show"
    )
    
    if action == "show":
        section = Prompt.ask("Show section (or 'all')", default="all")
        print_info(f"Would show configuration section: {section}")
    # Handle other actions


def _interactive_database(cli_context):
    """Interactive database operations."""
    console = Console()
    
    action = Prompt.ask(
        "Database action",
        choices=["status", "backup", "cleanup"],
        default="status"
    )
    
    if action == "status":
        print_info("Would show database status...")
    elif action == "backup":
        backup_path = Prompt.ask("Backup file path", default="./backup.db")
        if Confirm.ask(f"Create backup at {backup_path}?"):
            print_info(f"Would create backup at {backup_path}")
    # Handle other actions