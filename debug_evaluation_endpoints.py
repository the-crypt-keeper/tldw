#!/usr/bin/env python3
"""
Diagnostic script to debug FastAPI evaluation endpoint 500 errors.

This script performs progressive testing to isolate the root cause:
1. Test basic app initialization 
2. Test individual imports
3. Test endpoint registration
4. Test authentication flow
5. Test specific endpoint functionality
"""

import sys
import traceback
from pathlib import Path

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_fastapi_app():
    """Test if we can create a basic FastAPI app"""
    print("=== Testing Basic FastAPI App Creation ===")
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        
        app = FastAPI()
        
        @app.get("/test")
        def test():
            return {"status": "ok"}
        
        client = TestClient(app)
        response = client.get("/test")
        print(f"✓ Basic FastAPI app works: {response.status_code}")
        return True
    except Exception as e:
        print(f"✗ Basic FastAPI app failed: {e}")
        traceback.print_exc()
        return False

def test_evaluation_imports():
    """Test importing evaluation-related modules"""
    print("\n=== Testing Evaluation Module Imports ===")
    
    imports_to_test = [
        ("tldw_Server_API.app.api.v1.schemas.evaluation_schemas_unified", "evaluation schemas"),
        ("tldw_Server_API.app.core.Evaluations.unified_evaluation_service", "unified service"),
        ("tldw_Server_API.app.core.AuthNZ.settings", "auth settings"),
        ("tldw_Server_API.app.core.AuthNZ.jwt_service", "JWT service"),
        ("tldw_Server_API.app.core.AuthNZ.rate_limiter", "rate limiter"),
        ("tldw_Server_API.app.api.v1.API_Deps.auth_deps", "auth dependencies"),
        ("tldw_Server_API.app.core.Evaluations.webhook_manager", "webhook manager"),
        ("tldw_Server_API.app.core.Evaluations.user_rate_limiter", "user rate limiter"),
        ("tldw_Server_API.app.core.Evaluations.metrics_advanced", "advanced metrics"),
    ]
    
    results = {}
    for module_name, description in imports_to_test:
        try:
            __import__(module_name)
            print(f"✓ {description}: OK")
            results[module_name] = True
        except ImportError as e:
            print(f"✗ {description}: ImportError - {e}")
            results[module_name] = False
        except Exception as e:
            print(f"✗ {description}: Error - {e}")
            results[module_name] = False
    
    return results

def test_unified_evaluation_service():
    """Test unified evaluation service initialization"""
    print("\n=== Testing Unified Evaluation Service ===")
    
    try:
        from tldw_Server_API.app.core.Evaluations.unified_evaluation_service import get_unified_evaluation_service
        service = get_unified_evaluation_service()
        print(f"✓ Unified evaluation service created: {type(service)}")
        return True
    except Exception as e:
        print(f"✗ Unified evaluation service failed: {e}")
        traceback.print_exc()
        return False

def test_evaluation_endpoint_import():
    """Test importing the evaluation endpoint"""
    print("\n=== Testing Evaluation Endpoint Import ===")
    
    try:
        from tldw_Server_API.app.api.v1.endpoints.evaluations_unified import router
        print(f"✓ Evaluation endpoint imported: {router}")
        return True
    except Exception as e:
        print(f"✗ Evaluation endpoint import failed: {e}")
        traceback.print_exc()
        return False

def test_app_with_evaluation_router():
    """Test creating app with evaluation router"""
    print("\n=== Testing App with Evaluation Router ===")
    
    try:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from tldw_Server_API.app.api.v1.endpoints.evaluations_unified import router as eval_router
        
        app = FastAPI()
        app.include_router(eval_router, prefix="/api/v1")
        
        client = TestClient(app)
        print("✓ App created with evaluation router")
        return True, client
    except Exception as e:
        print(f"✗ App with evaluation router failed: {e}")
        traceback.print_exc()
        return False, None

def test_health_endpoint(client):
    """Test the health endpoint"""
    print("\n=== Testing Health Endpoint ===")
    
    if not client:
        print("✗ No client available")
        return False
    
    try:
        response = client.get("/api/v1/evaluations/health")
        print(f"✓ Health endpoint response: {response.status_code}")
        if response.status_code != 200:
            print(f"Response body: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"✗ Health endpoint failed: {e}")
        traceback.print_exc()
        return False

def test_auth_endpoint(client):
    """Test authenticated endpoint"""
    print("\n=== Testing Authenticated Endpoint ===")
    
    if not client:
        print("✗ No client available")
        return False
    
    # Set up test environment first
    import os
    os.environ["AUTH_MODE"] = "single_user"
    os.environ["SINGLE_USER_API_KEY"] = "test-api-key-for-authentication-testing"
    
    try:
        headers = {"Authorization": "Bearer test-api-key-for-authentication-testing"}
        response = client.get("/api/v1/evaluations", headers=headers)
        print(f"✓ Auth endpoint response: {response.status_code}")
        if response.status_code >= 500:
            print(f"Response body: {response.text}")
        return response.status_code < 500
    except Exception as e:
        print(f"✗ Auth endpoint failed: {e}")
        traceback.print_exc()
        return False

def test_main_app():
    """Test the main app initialization"""
    print("\n=== Testing Main App ===")
    
    try:
        from tldw_Server_API.app.main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        print("✓ Main app imported and client created")
        return True, client
    except Exception as e:
        print(f"✗ Main app failed: {e}")
        traceback.print_exc()
        return False, None

def main():
    """Run all diagnostic tests"""
    print("FastAPI Evaluation Endpoints Diagnostic Tool")
    print("=" * 50)
    
    # Test results
    results = {}
    
    # 1. Basic FastAPI test
    results['basic_fastapi'] = test_basic_fastapi_app()
    
    # 2. Import tests
    import_results = test_evaluation_imports()
    results.update(import_results)
    
    # 3. Service tests
    results['unified_service'] = test_unified_evaluation_service()
    results['endpoint_import'] = test_evaluation_endpoint_import()
    
    # 4. App with router test
    app_success, test_client = test_app_with_evaluation_router()
    results['app_with_router'] = app_success
    
    # 5. Endpoint tests
    if app_success:
        results['health_endpoint'] = test_health_endpoint(test_client)
        results['auth_endpoint'] = test_auth_endpoint(test_client)
    
    # 6. Main app test
    main_app_success, main_client = test_main_app()
    results['main_app'] = main_app_success
    
    if main_app_success:
        results['main_health'] = test_health_endpoint(main_client)
        results['main_auth'] = test_auth_endpoint(main_client)
    
    # Summary
    print("\n" + "=" * 50)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The issue may be elsewhere.")
    else:
        print(f"\n⚠️  {total - passed} tests failed. These are likely the root cause of 500 errors.")
    
    return results

if __name__ == "__main__":
    main()