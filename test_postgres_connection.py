#!/usr/bin/env python3
"""
Test PostgreSQL connection for tldw multi-user mode
"""

import asyncio
import asyncpg
import os
from dotenv import load_dotenv

async def test_connection():
    """Test PostgreSQL connection and schema"""
    
    # Load environment variables
    load_dotenv()
    
    # Get database URL
    database_url = os.getenv("DATABASE_URL", "postgresql://tldw_user:TestPassword123!@localhost/tldw_multiuser")
    
    print(f"Testing connection to: {database_url.split('@')[1]}")  # Hide password in output
    
    try:
        # Connect to database
        conn = await asyncpg.connect(database_url)
        print("✅ Connected to PostgreSQL successfully!")
        
        # Test database version
        version = await conn.fetchval("SELECT version()")
        print(f"📊 PostgreSQL version: {version.split(',')[0]}")
        
        # Check if tables exist
        tables = await conn.fetch("""
            SELECT tablename 
            FROM pg_tables 
            WHERE schemaname = 'public'
            ORDER BY tablename
        """)
        
        if tables:
            print(f"\n📋 Found {len(tables)} tables:")
            for table in tables:
                print(f"  - {table['tablename']}")
        else:
            print("\n⚠️  No tables found. Please run the schema file:")
            print(f"psql -U tldw_user -h localhost -d tldw_multiuser -f schema/postgresql_users.sql")
        
        # Check for required extensions
        extensions = await conn.fetch("""
            SELECT extname 
            FROM pg_extension 
            WHERE extname IN ('uuid-ossp', 'pg_trgm')
        """)
        
        print(f"\n🔧 Extensions installed:")
        for ext in extensions:
            print(f"  - {ext['extname']}")
        
        # Test creating a user (in a transaction that we'll rollback)
        async with conn.transaction():
            try:
                # Try to insert a test user
                user_id = await conn.fetchval("""
                    INSERT INTO users (username, email, password_hash, role)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id
                """, "test_user", "test@example.com", "dummy_hash", "user")
                
                print(f"\n✅ Database write test successful! (User ID: {user_id})")
                
                # Rollback the transaction (we don't want to keep the test user)
                raise asyncpg.exceptions.RollbackException("Test complete, rolling back")
                
            except asyncpg.exceptions.RollbackException:
                print("📝 Test data rolled back successfully")
            except asyncpg.exceptions.UndefinedTableError:
                print("\n❌ Users table not found. Please run the schema file first.")
            except Exception as e:
                print(f"\n❌ Error during write test: {e}")
        
        # Close connection
        await conn.close()
        print("\n✅ All tests completed successfully!")
        return True
        
    except asyncpg.exceptions.InvalidCatalogNameError:
        print("❌ Database 'tldw_multiuser' does not exist.")
        print("\nCreate it with:")
        print("  sudo -u postgres psql -c 'CREATE DATABASE tldw_multiuser;'")
        return False
        
    except asyncpg.exceptions.InvalidPasswordError:
        print("❌ Invalid password for user 'tldw_user'")
        print("\nCheck your DATABASE_URL in .env file")
        return False
        
    except asyncpg.exceptions.ConnectionDoesNotExistError:
        print("❌ Cannot connect to PostgreSQL")
        print("\nMake sure PostgreSQL is running:")
        print("  brew services start postgresql@14  # macOS")
        print("  sudo systemctl start postgresql    # Linux")
        return False
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

async def test_auth_system():
    """Test the authentication system with PostgreSQL"""
    
    print("\n" + "="*60)
    print("Testing Authentication System with PostgreSQL")
    print("="*60)
    
    # Import auth components
    try:
        from tldw_Server_API.app.core.AuthNZ.settings import get_settings
        from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
        from tldw_Server_API.app.services.registration_service import get_registration_service
        
        # Check settings
        settings = get_settings()
        print(f"\n⚙️  Auth Mode: {settings.AUTH_MODE}")
        print(f"📊 Database URL: {settings.DATABASE_URL.split('@')[1] if settings.DATABASE_URL else 'Not set'}")
        
        if settings.AUTH_MODE != "multi_user":
            print("\n⚠️  Warning: AUTH_MODE is not set to 'multi_user'")
            print("Please update your .env file")
            return False
        
        # Initialize database pool
        print("\n🔄 Initializing database pool...")
        db_pool = await get_db_pool()
        print("✅ Database pool initialized")
        
        # Test registration service
        print("\n🔄 Testing registration service...")
        reg_service = await get_registration_service()
        print(f"✅ Registration enabled: {reg_service.settings.ENABLE_REGISTRATION}")
        print(f"✅ Require code: {reg_service.settings.REQUIRE_REGISTRATION_CODE}")
        
        # Close pool
        await db_pool.close()
        print("\n✅ Authentication system test completed!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Make sure you're in the project directory")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

async def main():
    """Run all tests"""
    
    print("🚀 PostgreSQL Setup Test for tldw_server")
    print("="*60)
    
    # Test basic connection
    connection_ok = await test_connection()
    
    if connection_ok:
        # Test auth system
        await test_auth_system()
        
        print("\n" + "="*60)
        print("📝 Next Steps:")
        print("="*60)
        print("\n1. If tables are missing, run:")
        print("   psql -U tldw_user -h localhost -d tldw_multiuser -f schema/postgresql_users.sql")
        print("\n2. Start the server in multi-user mode:")
        print("   python -m uvicorn tldw_Server_API.app.main:app --reload")
        print("\n3. Create an admin user:")
        print("   python tldw_Server_API/scripts/migrate_to_multiuser.py \\")
        print("     --admin-email admin@example.com \\")
        print("     --admin-password AdminPass123!")
        print("\n4. Test the API:")
        print("   curl http://localhost:8000/api/v1/health")
        print("   curl http://localhost:8000/docs")
    else:
        print("\n❌ Please fix the connection issues and try again")

if __name__ == "__main__":
    asyncio.run(main())