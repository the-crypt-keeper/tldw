"""
Simple test to verify PostgreSQL test database setup works.
"""

import asyncio
import asyncpg
import os


async def test_database_connection():
    """Test that we can connect to PostgreSQL test database."""
    TEST_DB_HOST = os.getenv("TEST_DB_HOST", "localhost")
    TEST_DB_PORT = int(os.getenv("TEST_DB_PORT", "5432"))
    TEST_DB_USER = os.getenv("TEST_DB_USER", "tldw_user")
    TEST_DB_PASSWORD = os.getenv("TEST_DB_PASSWORD", "TestPassword123!")
    TEST_DB_NAME = "tldw_test"
    
    # Create test database if it doesn't exist
    conn = await asyncpg.connect(
        host=TEST_DB_HOST,
        port=TEST_DB_PORT,
        user=TEST_DB_USER,
        password=TEST_DB_PASSWORD,
        database="postgres"
    )
    
    try:
        # Check if database exists
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1",
            TEST_DB_NAME
        )
        
        if not exists:
            await conn.execute(f"CREATE DATABASE {TEST_DB_NAME}")
            print(f"Created test database: {TEST_DB_NAME}")
        else:
            print(f"Test database already exists: {TEST_DB_NAME}")
    finally:
        await conn.close()
    
    # Now connect to test database
    test_conn = await asyncpg.connect(
        host=TEST_DB_HOST,
        port=TEST_DB_PORT,
        user=TEST_DB_USER,
        password=TEST_DB_PASSWORD,
        database=TEST_DB_NAME
    )
    
    try:
        # Create users table
        await test_conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                uuid UUID UNIQUE NOT NULL,
                username VARCHAR(255) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role VARCHAR(50) NOT NULL DEFAULT 'user',
                is_active BOOLEAN DEFAULT TRUE,
                is_verified BOOLEAN DEFAULT FALSE,
                storage_quota_mb INTEGER DEFAULT 5120,
                storage_used_mb FLOAT DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        print("Created users table successfully")
        
        # Test insert
        import uuid
        test_uuid = str(uuid.uuid4())
        await test_conn.execute("""
            INSERT INTO users (uuid, username, email, password_hash, role)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (username) DO NOTHING
        """, test_uuid, "test_user", "test@example.com", "hash", "user")
        
        # Test query
        count = await test_conn.fetchval("SELECT COUNT(*) FROM users")
        print(f"Users in database: {count}")
        
        # Clean up
        await test_conn.execute("TRUNCATE TABLE users CASCADE")
        print("Cleaned up test data")
        
    finally:
        await test_conn.close()
    
    print("PostgreSQL test database setup verified successfully!")


if __name__ == "__main__":
    asyncio.run(test_database_connection())