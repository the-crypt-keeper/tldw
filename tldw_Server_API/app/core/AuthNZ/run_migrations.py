#!/usr/bin/env python3
# run_migrations.py
# Description: Script to run AuthNZ database migrations
#
# Usage:
#   python run_migrations.py [--db-path PATH] [--rollback VERSION] [--status]
#

import argparse
import sys
from pathlib import Path
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from tldw_Server_API.app.core.AuthNZ.migrations import (
    apply_authnz_migrations,
    rollback_authnz_migrations,
    check_migration_status,
    ensure_authnz_tables
)
from tldw_Server_API.app.core.AuthNZ.settings import get_settings

def main():
    """Main function to run migrations"""
    parser = argparse.ArgumentParser(description="Run AuthNZ database migrations")
    parser.add_argument(
        "--db-path",
        type=str,
        help="Path to the database file (default: from settings)"
    )
    parser.add_argument(
        "--rollback",
        type=int,
        metavar="VERSION",
        help="Rollback to a specific migration version"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check migration status without applying changes"
    )
    parser.add_argument(
        "--target",
        type=int,
        metavar="VERSION",
        help="Migrate to a specific version (default: latest)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force migration even if database appears up to date"
    )
    
    args = parser.parse_args()
    
    # Get database path
    if args.db_path:
        db_path = Path(args.db_path)
    else:
        settings = get_settings()
        # Extract database path from DATABASE_URL
        db_url = settings.DATABASE_URL
        if db_url.startswith("sqlite:///"):
            db_path = Path(db_url.replace("sqlite:///", ""))
        else:
            logger.error("Only SQLite databases are supported for migrations currently")
            sys.exit(1)
    
    # Ensure database directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check status
    if args.status:
        status = check_migration_status(db_path)
        print("\n=== Migration Status ===")
        print(f"Database: {db_path}")
        print(f"Current Version: {status['current_version']}")
        print(f"Latest Version: {status['latest_version']}")
        print(f"Up to Date: {'✅ Yes' if status['is_up_to_date'] else '❌ No'}")
        
        if status['pending_migrations']:
            print("\nPending Migrations:")
            for migration in status['pending_migrations']:
                print(f"  - Version {migration['version']}: {migration['name']}")
        
        sys.exit(0)
    
    # Rollback
    if args.rollback is not None:
        print(f"\n=== Rolling Back to Version {args.rollback} ===")
        print(f"Database: {db_path}")
        
        confirm = input("Are you sure you want to rollback? This may result in data loss. (yes/no): ")
        if confirm.lower() != 'yes':
            print("Rollback cancelled")
            sys.exit(0)
        
        try:
            rollback_authnz_migrations(db_path, args.rollback)
            print("✅ Rollback completed successfully")
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            sys.exit(1)
    
    # Apply migrations
    else:
        status = check_migration_status(db_path)
        
        if status['is_up_to_date'] and not args.force:
            print(f"\n✅ Database is already up to date (version {status['current_version']})")
            sys.exit(0)
        
        print("\n=== Applying Migrations ===")
        print(f"Database: {db_path}")
        print(f"Current Version: {status['current_version']}")
        print(f"Target Version: {args.target or status['latest_version']}")
        
        if status['pending_migrations']:
            print("\nMigrations to Apply:")
            for migration in status['pending_migrations']:
                if args.target and migration['version'] > args.target:
                    break
                print(f"  - Version {migration['version']}: {migration['name']}")
        
        try:
            apply_authnz_migrations(db_path, args.target)
            
            # Check final status
            final_status = check_migration_status(db_path)
            print(f"\n✅ Migrations completed successfully")
            print(f"Database is now at version {final_status['current_version']}")
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()