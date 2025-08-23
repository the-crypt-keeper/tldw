"""
Evaluation CLI module - provides command-line interface for evaluation system.
"""

from .benchmark_cli import benchmark_group
from .evals_cli import main

__all__ = ['benchmark_group', 'main']