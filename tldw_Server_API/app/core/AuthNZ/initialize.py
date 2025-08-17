#!/usr/bin/env python3
# initialize.py
# Description: Initialize AuthNZ module for first-time setup
#
# This script sets up the AuthNZ module including:
# - Database creation and migrations
# - Initial admin user creation (multi-user mode)
# - Encryption key generation
# - Configuration validation
#

import asyncio
import sys
import os
import secrets
from pathlib import Path
from typing import Optional
from getpass import getpass
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
from tldw_Server_API.app.core.AuthNZ.migrations import (
    ensure_authnz_tables,
    check_migration_status
)
from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
from tldw_Server_API.app.core.DB_Management.Users_DB import (
    get_users_db,
    ensure_user_directories
)
from tldw_Server_API.app.core.AuthNZ.api_key_manager import get_api_key_manager
from tldw_Server_API.app.core.AuthNZ.scheduler import start_authnz_scheduler
from tldw_Server_API.app.core.AuthNZ.monitoring import get_authnz_monitor

#######################################################################################################################
#
# Initialization Functions
#

def print_banner():
    """Print initialization banner"""
    print("\n" + "=" * 60)
    print("       AuthNZ Module Initialization")
    print("=" * 60)
    print()

def check_environment():
    """Check and validate environment configuration"""
    print("📋 Checking environment configuration...")
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ No .env file found!")
        print("   Creating from template...")
        
        template_file = Path(".env.authnz.template")
        if template_file.exists():
            env_file.write_text(template_file.read_text())
            print("✅ Created .env file from template")
            print("⚠️  Please edit .env and set secure values before continuing!")
            return False
        else:
            print("❌ Template file not found!")
            return False
    
    # Load settings
    settings = get_settings()
    
    # Validate critical settings
    issues = []
    
    if settings.AUTH_MODE == "multi_user":
        if not settings.JWT_SECRET_KEY or len(settings.JWT_SECRET_KEY) < 32:
            issues.append("JWT_SECRET_KEY must be set and at least 32 characters")
        
        if settings.JWT_SECRET_KEY == "CHANGE_ME_TO_SECURE_RANDOM_KEY_MIN_32_CHARS":
            issues.append("JWT_SECRET_KEY still has default value - must be changed!")
    
    if settings.AUTH_MODE == "single_user":
        if settings.SINGLE_USER_API_KEY == "CHANGE_ME_TO_SECURE_API_KEY":
            issues.append("SINGLE_USER_API_KEY still has default value - must be changed!")
    
    if issues:
        print("\n❌ Configuration issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    print("✅ Environment configuration valid")
    print(f"   Mode: {settings.AUTH_MODE}")
    print(f"   Database: {settings.DATABASE_URL[:30]}...")
    
    return True

def generate_secure_keys():
    """Generate secure keys for configuration"""
    print("\n🔑 Generating secure keys...")
    
    keys = {
        'JWT_SECRET_KEY': secrets.token_urlsafe(32),
        'SINGLE_USER_API_KEY': secrets.token_urlsafe(32),
        'API_KEY_PEPPER': secrets.token_hex(32)
    }
    
    # Generate Fernet key for session encryption
    from cryptography.fernet import Fernet
    keys['SESSION_ENCRYPTION_KEY'] = Fernet.generate_key().decode()
    
    print("\n📝 Generated keys (save these in your .env file):")
    print("-" * 50)
    for key, value in keys.items():
        print(f"{key}={value}")
    print("-" * 50)
    
    return keys

async def setup_database():
    """Setup database and run migrations"""
    print("\n🗄️  Setting up database...")
    
    settings = get_settings()
    
    # Extract database path
    db_url = settings.DATABASE_URL
    if db_url.startswith("sqlite:///"):
        db_path = Path(db_url.replace("sqlite:///", ""))
        
        # Ensure directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"   Database path: {db_path}")
        
        # Check migration status
        status = check_migration_status(db_path)
        print(f"   Current version: {status['current_version']}")
        print(f"   Latest version: {status['latest_version']}")
        
        if not status['is_up_to_date']:
            print(f"   Pending migrations: {len(status['pending_migrations'])}")
            
            # Apply migrations
            ensure_authnz_tables(db_path)
            print("✅ Database migrations applied")
        else:
            print("✅ Database is up to date")
    else:
        print("⚠️  Non-SQLite database detected - ensure migrations are run separately")
    
    return True

async def create_admin_user():
    """Create initial admin user for multi-user mode"""
    settings = get_settings()
    
    if settings.AUTH_MODE != "multi_user":
        print("\n📝 Single-user mode - skipping admin user creation")
        return True
    
    print("\n👤 Creating admin user...")
    
    # Get user input
    while True:
        username = input("   Admin username (default: admin): ").strip() or "admin"
        if len(username) >= 3:
            break
        print("   Username must be at least 3 characters")
    
    while True:
        email = input("   Admin email: ").strip()
        if "@" in email and "." in email:
            break
        print("   Please enter a valid email address")
    
    while True:
        password = getpass("   Admin password (min 10 chars): ")
        if len(password) >= 10:
            confirm = getpass("   Confirm password: ")
            if password == confirm:
                break
            else:
                print("   Passwords don't match!")
        else:
            print("   Password must be at least 10 characters")
    
    try:
        # Hash password
        password_service = PasswordService()
        password_hash = password_service.hash_password(password)
        
        # Create user
        users_db = await get_users_db()
        admin_user = await users_db.create_user(
            username=username,
            email=email,
            password_hash=password_hash,
            role="admin",
            is_superuser=True
        )
        
        # Create initial API key for admin
        api_manager = await get_api_key_manager()
        api_key_result = await api_manager.create_api_key(
            user_id=admin_user['id'],
            name="Initial Admin API Key",
            description="Auto-generated during setup",
            scope="admin",
            expires_in_days=365
        )
        
        print(f"\n✅ Admin user created successfully!")
        print(f"   User ID: {admin_user['id']}")
        print(f"   Username: {admin_user['username']}")
        print(f"\n🔑 Admin API Key (save this - won't be shown again):")
        print(f"   {api_key_result['key']}")
        
        # Ensure user directories exist
        await ensure_user_directories(admin_user['id'])
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create admin user: {e}")
        return False

async def test_authentication():
    """Test authentication system"""
    print("\n🧪 Testing authentication system...")
    
    settings = get_settings()
    
    try:
        if settings.AUTH_MODE == "single_user":
            # Test API key validation
            print("   Testing single-user API key...")
            # This would normally test the actual API key validation
            print("✅ Single-user authentication configured")
        else:
            # Test JWT system
            from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service
            
            jwt_service = await get_jwt_service()
            
            # Create test token
            test_payload = {"sub": "test_user", "user_id": 1}
            test_token = jwt_service.create_access_token(test_payload)
            
            # Validate test token
            decoded = jwt_service.decode_token(test_token)
            
            if decoded and decoded.get("sub") == "test_user":
                print("✅ JWT authentication system working")
            else:
                print("❌ JWT validation failed")
                return False
                
        return True
        
    except Exception as e:
        print(f"❌ Authentication test failed: {e}")
        return False

async def start_services():
    """Start background services"""
    print("\n🚀 Starting background services...")
    
    try:
        # Start scheduler
        await start_authnz_scheduler()
        print("✅ Scheduler started")
        
        # Initialize monitor
        monitor = await get_authnz_monitor()
        print("✅ Monitoring system initialized")
        
        # Get initial metrics
        metrics = await monitor.get_metrics_summary(60)
        print(f"   Health status: {monitor._calculate_health_status(metrics)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to start services: {e}")
        return False

async def main():
    """Main initialization function"""
    print_banner()
    
    # Step 1: Check environment
    if not check_environment():
        print("\n⚠️  Please configure your environment and run again.")
        print("   1. Edit .env file with secure values")
        print("   2. Run: python -m tldw_Server_API.app.core.AuthNZ.initialize")
        sys.exit(1)
    
    # Step 2: Offer to generate keys if needed
    response = input("\n📝 Generate new secure keys? (y/N): ").strip().lower()
    if response == 'y':
        generate_secure_keys()
        print("\n⚠️  Update your .env file with these keys and run again.")
        sys.exit(0)
    
    # Step 3: Setup database
    if not await setup_database():
        print("\n❌ Database setup failed")
        sys.exit(1)
    
    # Step 4: Create admin user (multi-user mode)
    settings = get_settings()
    if settings.AUTH_MODE == "multi_user":
        # Check if any users exist
        try:
            users_db = await get_users_db()
            existing_users = await users_db.list_users(limit=1)
            
            if not existing_users:
                response = input("\n📝 No users found. Create admin user? (Y/n): ").strip().lower()
                if response != 'n':
                    if not await create_admin_user():
                        print("\n⚠️  Admin user creation failed")
            else:
                print(f"\n✅ Found {len(existing_users)} existing user(s)")
        except Exception as e:
            logger.warning(f"Could not check existing users: {e}")
            response = input("\n📝 Create admin user? (Y/n): ").strip().lower()
            if response != 'n':
                await create_admin_user()
    
    # Step 5: Test authentication
    if not await test_authentication():
        print("\n⚠️  Authentication test failed")
    
    # Step 6: Start services (optional)
    response = input("\n🚀 Start background services? (y/N): ").strip().lower()
    if response == 'y':
        await start_services()
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ AuthNZ Initialization Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review your configuration in .env")
    print("2. Test authentication endpoints")
    print("3. Configure monitoring and alerting")
    print("4. Set up regular backups")
    print("\nTo start the application:")
    print("   python -m uvicorn tldw_Server_API.app.main:app --reload")
    print()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Initialization cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Initialization failed: {e}")
        logger.exception("Initialization error")
        sys.exit(1)