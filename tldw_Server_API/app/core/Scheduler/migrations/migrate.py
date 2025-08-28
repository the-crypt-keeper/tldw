"""
Migration tool for Scheduler database schemas.
"""

import asyncio
import argparse
from pathlib import Path
from loguru import logger

from ..config import SchedulerConfig
from ..backends import create_backend


async def migrate_up(config: SchedulerConfig) -> None:
    """
    Apply migrations to create/update schema.
    
    Args:
        config: Scheduler configuration
    """
    logger.info(f"Applying migrations to {config.database_url}")
    
    backend = create_backend(config)
    await backend.connect()
    
    try:
        # The backends already create their schemas on connect
        # This is just a placeholder for future migration logic
        logger.info("Schema initialized successfully")
        
        # Test the schema
        from ..base import Task
        test_task = Task(handler="migration_test")
        task_id = await backend.enqueue(test_task)
        task = await backend.get_task(task_id)
        
        if task:
            logger.info("Schema validation successful")
        else:
            logger.error("Schema validation failed")
        
    finally:
        await backend.disconnect()


async def migrate_down(config: SchedulerConfig) -> None:
    """
    Remove scheduler tables (destructive!).
    
    Args:
        config: Scheduler configuration
    """
    logger.warning(f"Dropping scheduler tables from {config.database_url}")
    
    backend = create_backend(config)
    await backend.connect()
    
    try:
        if config.is_postgresql:
            # Drop PostgreSQL tables
            await backend.execute("DROP TABLE IF EXISTS payloads CASCADE")
            await backend.execute("DROP TABLE IF EXISTS tasks CASCADE")
            await backend.execute("DROP TABLE IF EXISTS leader_election CASCADE")
            logger.info("PostgreSQL tables dropped")
            
        elif config.is_sqlite:
            # Drop SQLite tables
            await backend.execute("DROP TABLE IF EXISTS tasks")
            await backend.execute("DROP TABLE IF EXISTS leader_election")
            await backend.execute("DROP INDEX IF EXISTS idx_tasks_queue")
            await backend.execute("DROP INDEX IF EXISTS idx_tasks_status")
            await backend.execute("DROP INDEX IF EXISTS idx_tasks_scheduled")
            await backend.execute("DROP INDEX IF EXISTS idx_tasks_dependencies")
            await backend.execute("DROP INDEX IF EXISTS idx_tasks_lease")
            logger.info("SQLite tables dropped")
        
    finally:
        await backend.disconnect()


async def check_status(config: SchedulerConfig) -> None:
    """
    Check migration status.
    
    Args:
        config: Scheduler configuration
    """
    logger.info(f"Checking schema status for {config.database_url}")
    
    backend = create_backend(config)
    
    try:
        await backend.connect()
        
        # Try to query tasks table
        if config.is_postgresql:
            result = await backend.execute("""
                SELECT COUNT(*) as count FROM information_schema.tables 
                WHERE table_name = 'tasks'
            """)
            exists = result > 0
        else:
            try:
                await backend.execute("SELECT COUNT(*) FROM tasks")
                exists = True
            except:
                exists = False
        
        if exists:
            # Get counts
            task_count = await backend.execute("SELECT COUNT(*) FROM tasks")
            logger.info(f"Schema exists with {task_count} tasks")
        else:
            logger.info("Schema does not exist")
        
    except Exception as e:
        logger.error(f"Failed to check status: {e}")
    finally:
        await backend.disconnect()


def main():
    """Main migration CLI."""
    parser = argparse.ArgumentParser(description="Scheduler database migrations")
    parser.add_argument(
        "action",
        choices=["up", "down", "status"],
        help="Migration action to perform"
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Database URL (overrides config)"
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Path to scheduler config"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config_kwargs = {}
    if args.database_url:
        config_kwargs['database_url'] = args.database_url
    
    config = SchedulerConfig(**config_kwargs)
    
    # Run migration
    if args.action == "up":
        asyncio.run(migrate_up(config))
    elif args.action == "down":
        confirm = input("This will DELETE all scheduler data. Continue? [y/N]: ")
        if confirm.lower() == 'y':
            asyncio.run(migrate_down(config))
        else:
            print("Migration cancelled")
    elif args.action == "status":
        asyncio.run(check_status(config))


if __name__ == "__main__":
    main()