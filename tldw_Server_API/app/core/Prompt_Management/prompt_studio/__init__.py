# __init__.py
# Prompt Studio module initialization

"""
Prompt Studio - A structured prompt engineering platform for tldw_server

This module provides comprehensive prompt engineering capabilities including:
- Project and prompt management with versioning
- Test case creation and management
- Automated testing and evaluation
- Prompt optimization using various strategies
- Real-time job processing and monitoring
"""

# Core managers
from .test_case_manager import TestCaseManager
from .test_case_io import TestCaseIO
from .test_case_generator import TestCaseGenerator
from .job_manager import JobManager, JobType, JobStatus
from .job_processor import JobProcessor

# Prompt generation and improvement
from .prompt_generator import PromptGenerator
from .prompt_improver import PromptImprover
from .bootstrap_manager import BootstrapManager

# Testing and evaluation
from .test_runner import TestRunner
from .prompt_executor import PromptExecutor
from .evaluation_metrics import EvaluationMetrics, MetricType
from .evaluation_reports import EvaluationReportGenerator

# Optimization
from .optimization_engine import OptimizationEngine
from .optimization_strategies import HyperparameterOptimizer

# Event handling and monitoring
from .event_broadcaster import EventBroadcaster, EventType
from .monitoring import PromptStudioMetrics

# Security and permissions
from .auth_permissions import PermissionManager, Permission

__all__ = [
    # Core managers
    'TestCaseManager',
    'TestCaseIO',
    'TestCaseGenerator',
    'JobManager',
    'JobType',
    'JobStatus',
    'JobProcessor',
    
    # Prompt generation and improvement
    'PromptGenerator',
    'PromptImprover',
    'BootstrapManager',
    
    # Testing and evaluation
    'TestRunner',
    'PromptExecutor',
    'EvaluationMetrics',
    'MetricType',
    'EvaluationReportGenerator',
    
    # Optimization
    'OptimizationEngine',
    'HyperparameterOptimizer',
    
    # Event handling and monitoring
    'EventBroadcaster',
    'EventType',
    'PromptStudioMetrics',
    
    # Security and permissions
    'PermissionManager',
    'Permission',
]

__version__ = '0.1.0'
__author__ = 'tldw_server Development Team'