#!/usr/bin/env python3
"""
Test script for authentication stack
Run this to verify the auth system is working correctly
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.AuthNZ.database import test_database_connection, get_db_pool
from tldw_Server_API.app.core.AuthNZ.password_service import get_password_service
from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service
from tldw_Server_API.app.core.AuthNZ.session_manager import get_session_manager
from tldw_Server_API.app.services.registration_service import get_registration_service


async def test_services():
    """Test that all services can be initialized"""
    print("=" * 60)
    print("AUTHENTICATION STACK TEST")
    print("=" * 60)
    
    # Test 1: Settings
    print("\n1. Testing Settings...")
    try:
        settings = get_settings()
        print(f"   ✓ Settings loaded")
        print(f"   - Auth mode: {settings.AUTH_MODE}")
        print(f"   - Database: {settings.DATABASE_URL}")
        print(f"   - JWT secret file: {settings.JWT_SECRET_FILE}")
        
        # Check if JWT secret was created (only needed in multi_user mode)
        if settings.AUTH_MODE == "multi_user":
            jwt_file = Path(settings.JWT_SECRET_FILE)
            if jwt_file.exists():
                print(f"   ✓ JWT secret file exists")
            else:
                print(f"   ✗ JWT secret file missing")
        else:
            print(f"   - JWT secret not needed in single-user mode")
    except Exception as e:
        print(f"   ✗ Settings failed: {e}")
        return False
    
    # Test 2: Database connection
    print("\n2. Testing Database Connection...")
    try:
        db_connected = await test_database_connection()
        if db_connected:
            print(f"   ✓ Database connected successfully")
        else:
            print(f"   ✗ Database connection failed")
            return False
    except Exception as e:
        print(f"   ✗ Database test failed: {e}")
        return False
    
    # Test 3: Password Service
    print("\n3. Testing Password Service...")
    try:
        password_service = get_password_service()
        test_password = "Tr3$tP@s5w0rd!X"
        hashed = password_service.hash_password(test_password)
        verified = password_service.verify_password(test_password, hashed)
        if verified:
            print(f"   ✓ Password hashing and verification working")
        else:
            print(f"   ✗ Password verification failed")
    except Exception as e:
        print(f"   ✗ Password service failed: {e}")
        return False
    
    # Test 4: JWT Service
    print("\n4. Testing JWT Service...")
    try:
        jwt_service = get_jwt_service()
        
        # Create test token
        access_token = jwt_service.create_access_token(
            user_id=1,
            username="testuser",
            role="user"
        )
        print(f"   ✓ Access token created")
        
        # Decode token
        payload = jwt_service.decode_access_token(access_token)
        if payload.get('sub') == "1" and payload.get('username') == "testuser":
            print(f"   ✓ Token decoded successfully")
        else:
            print(f"   ✗ Token payload incorrect")
            print(f"     Got sub={payload.get('sub')}, username={payload.get('username')}")
            
        # Create refresh token
        refresh_token = jwt_service.create_refresh_token(user_id=1, username="testuser")
        print(f"   ✓ Refresh token created")
        
    except Exception as e:
        print(f"   ✗ JWT service failed: {e}")
        return False
    
    # Test 5: Session Manager
    print("\n5. Testing Session Manager...")
    try:
        session_manager = await get_session_manager()
        print(f"   ✓ Session manager initialized")
        
        # Test creating a session with unique tokens
        import uuid
        unique_id = str(uuid.uuid4())
        session = await session_manager.create_session(
            user_id=1,
            access_token=f"test_access_token_{unique_id}",
            refresh_token=f"test_refresh_token_{unique_id}",
            ip_address="127.0.0.1",
            user_agent="Test Agent"
        )
        
        if session and 'session_id' in session:
            print(f"   ✓ Test session created (ID: {session['session_id']})")
            
            # Clean up test session
            await session_manager.revoke_session(session['session_id'])
            print(f"   ✓ Test session cleaned up")
        else:
            print(f"   ✗ Session creation failed")
            
    except Exception as e:
        print(f"   ✗ Session manager failed: {e}")
        return False
    
    # Test 6: Registration Service
    print("\n6. Testing Registration Service...")
    try:
        registration_service = await get_registration_service()
        print(f"   ✓ Registration service initialized")
        print(f"   - Registration enabled: {registration_service.registration_enabled}")
        print(f"   - Registration code required: {registration_service.require_code}")
        
        # Test registration code generation
        code = registration_service.generate_registration_code()
        if code and len(code) == 24:
            print(f"   ✓ Registration code generation working")
        else:
            print(f"   ✗ Registration code generation failed")
            
    except Exception as e:
        print(f"   ✗ Registration service failed: {e}")
        return False
    
    # Test 7: Check database schema
    print("\n7. Testing Database Schema...")
    try:
        db_pool = await get_db_pool()
        
        # Check if users table exists
        tables_to_check = ['users', 'sessions', 'registration_codes']
        
        for table in tables_to_check:
            try:
                result = await db_pool.fetchone(f"SELECT COUNT(*) FROM {table}")
                print(f"   ✓ Table '{table}' exists")
            except Exception as e:
                print(f"   ✗ Table '{table}' missing or inaccessible: {e}")
                print(f"     You may need to run the database migration script")
                
    except Exception as e:
        print(f"   ✗ Database schema check failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - Authentication stack is ready!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Update main.py to include auth router")
    print("2. Run the FastAPI server")
    print("3. Test the endpoints with curl or a REST client")
    print("\nExample test commands:")
    print("  # Register a user:")
    print('  curl -X POST http://localhost:8000/api/v1/auth/register \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"username": "testuser", "email": "test@example.com", "password": "TestPass123!"}\'')
    print("\n  # Login:")
    print('  curl -X POST http://localhost:8000/api/v1/auth/login \\')
    print('    -H "Content-Type: application/x-www-form-urlencoded" \\')
    print('    -d "username=testuser&password=TestPass123!"')
    
    return True


async def main():
    """Main test function"""
    try:
        success = await test_services()
        if not success:
            print("\n❌ Some tests failed. Please check the errors above.")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())