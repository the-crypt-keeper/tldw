#!/usr/bin/env python3
"""Simple test runner for STT module tests."""

import sys
import unittest
from io import StringIO

# Add current directory to path
sys.path.insert(0, '.')

def run_test_module(module_name, test_name):
    """Run tests from a single module."""
    try:
        # Import the test module
        module = __import__(f'tests.Media_Ingestion_Modification.{module_name}', fromlist=['*'])
        
        # Create test suite by discovering all test classes
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # Find all test classes in the module
        import inspect
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and name.startswith('Test'):
                # Load tests from this test class
                tests = loader.loadTestsFromName(f'tests.Media_Ingestion_Modification.{module_name}.{name}')
                suite.addTests(tests)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Return summary
        return {
            'name': test_name,
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'passed': result.testsRun - len(result.failures) - len(result.errors),
            'success': result.wasSuccessful()
        }
    except Exception as e:
        return {
            'name': test_name,
            'error': str(e)
        }

def main():
    """Run all STT module tests."""
    test_modules = [
        ('test_parakeet_mlx', 'MLX Tests'),
        ('test_external_provider', 'External Provider Tests'),
        ('test_buffered_transcription', 'Buffered Transcription Tests'),
        ('test_streaming_transcription', 'Streaming Tests'),
        ('test_parakeet_onnx', 'ONNX Tests')
    ]
    
    print("=" * 80)
    print("STT MODULE TEST RESULTS - FINAL RUN")
    print("=" * 80)
    print()
    
    total_tests = 0
    total_passed = 0
    total_failed = 0
    total_errors = 0
    
    for module_name, test_name in test_modules:
        print(f"\n{'='*60}")
        print(f"Running {test_name}...")
        print('='*60)
        
        result = run_test_module(module_name, test_name)
        
        if 'error' in result:
            print(f"❌ Module load error: {result['error']}")
        else:
            print(f"\nResults for {test_name}:")
            print(f"  Tests run: {result['tests_run']}")
            print(f"  Passed: {result['passed']} ✅")
            print(f"  Failed: {result['failures']} ❌")
            print(f"  Errors: {result['errors']} ⚠️")
            
            if result['success']:
                print(f"  Status: SUCCESS ✅")
            else:
                print(f"  Status: FAILED ❌")
            
            total_tests += result['tests_run']
            total_passed += result['passed']
            total_failed += result['failures']
            total_errors += result['errors']
    
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total tests run: {total_tests}")
    print(f"Total passed: {total_passed} ({total_passed/total_tests*100:.1f}%)" if total_tests > 0 else "Total passed: 0")
    print(f"Total failed: {total_failed}")
    print(f"Total errors: {total_errors}")
    print(f"\nOverall success rate: {total_passed/total_tests*100:.1f}%" if total_tests > 0 else "Overall success rate: N/A")
    print("=" * 80)

if __name__ == '__main__':
    main()