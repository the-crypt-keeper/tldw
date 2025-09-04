#!/usr/bin/env python3
"""
Comprehensive test runner for evaluations module.

This script runs all tests and generates a coverage report.
"""

import subprocess
import sys
from pathlib import Path


def run_tests_with_coverage():
    """Run all evaluation tests with coverage reporting."""
    
    # Get the test directory
    test_dir = Path(__file__).parent
    project_root = test_dir.parent.parent
    
    print("=" * 60)
    print("Running Comprehensive Evaluation Module Tests")
    print("=" * 60)
    
    # Test files to run
    test_files = [
        "test_rag_evaluator_embeddings.py",
        "test_evaluation_integration.py",
        "test_error_scenarios.py",
        "test_circuit_breaker.py",
        "test_evals_openai.py",  # Existing test file
    ]
    
    # Build pytest command
    cmd = [
        sys.executable, "-m", "pytest",
        "-v",  # Verbose
        "--tb=short",  # Short traceback
        "--cov=tldw_Server_API.app.core.Evaluations",  # Coverage for evaluations module
        "--cov=tldw_Server_API.app.api.v1.endpoints.evals_openai",  # Coverage for API
        "--cov-report=term-missing",  # Show missing lines
        "--cov-report=html",  # Generate HTML report
        "--cov-report=xml",  # Generate XML report for CI
        "--cov-fail-under=80",  # Fail if coverage < 80%
    ]
    
    # Add test files
    for test_file in test_files:
        test_path = test_dir / test_file
        if test_path.exists():
            cmd.append(str(test_path))
    
    print(f"Running command: {' '.join(cmd)}")
    print()
    
    # Run tests
    result = subprocess.run(cmd, cwd=project_root)
    
    print()
    print("=" * 60)
    
    if result.returncode == 0:
        print("✅ All tests passed with >80% coverage!")
        print()
        print("📊 Coverage report generated:")
        print(f"   - HTML: {project_root}/htmlcov/index.html")
        print(f"   - XML: {project_root}/coverage.xml")
    else:
        print("❌ Tests failed or coverage < 80%")
        print("   Please review the output above for details")
    
    print("=" * 60)
    
    return result.returncode


def run_specific_test_suite(suite_name):
    """Run a specific test suite."""
    
    suites = {
        "unit": [
            "test_rag_evaluator_embeddings.py::TestRAGEvaluatorEmbeddings",
            "test_circuit_breaker.py::TestCircuitBreaker",
        ],
        "integration": [
            "test_evaluation_integration.py::TestEvaluationIntegration",
            "test_circuit_breaker.py::TestCircuitBreakerIntegration",
        ],
        "error": [
            "test_error_scenarios.py::TestErrorScenarios",
            "test_error_scenarios.py::TestEdgeCases",
        ],
        "security": [
            "test_evaluation_integration.py::TestAuthentication",
            "test_evaluation_integration.py::TestRateLimiting",
        ]
    }
    
    if suite_name not in suites:
        print(f"Unknown suite: {suite_name}")
        print(f"Available suites: {', '.join(suites.keys())}")
        return 1
    
    test_dir = Path(__file__).parent
    project_root = test_dir.parent.parent
    
    print(f"Running {suite_name} test suite...")
    
    cmd = [
        sys.executable, "-m", "pytest",
        "-v",
        "--tb=short",
    ]
    
    for test_spec in suites[suite_name]:
        cmd.append(str(test_dir / test_spec))
    
    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run comprehensive tests for evaluations module"
    )
    parser.add_argument(
        "--suite",
        choices=["all", "unit", "integration", "error", "security"],
        default="all",
        help="Test suite to run"
    )
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Skip coverage reporting"
    )
    
    args = parser.parse_args()
    
    if args.suite == "all" and not args.no_coverage:
        return run_tests_with_coverage()
    elif args.suite == "all":
        # Run all tests without coverage
        test_dir = Path(__file__).parent
        cmd = [sys.executable, "-m", "pytest", "-v", str(test_dir)]
        result = subprocess.run(cmd)
        return result.returncode
    else:
        return run_specific_test_suite(args.suite)


if __name__ == "__main__":
    sys.exit(main())