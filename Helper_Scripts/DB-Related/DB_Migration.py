# DB_Migration.py
# Purpose: This file contains the functions to migrate the database from the old.db to the new.db.
# Regular migration
# python migrate_db.py --source old.db --target new.db --export-path ./export

# Verify previous migration
# python migrate_db.py --source old.db --target new.db --export-path ./export --verify-only

# Clean up incomplete migration
# python migrate_db.py --source old.db --target new.db --export-path ./export --cleanup

# Generate migration report
# python migrate_db.py --source old.db --target new.db --export-path ./export --report

# Imports
import sqlite3
from sqlite3 import Connection, Cursor
import logging
import json
import hashlib
import time
import sys
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class MigrationState:
    """Track migration progress and state."""
    completed_tables: Set[str]
    completed_chats: Set[int]
    last_successful_id: Dict[str, int]
    total_rows: Dict[str, int]
    migrated_rows: Dict[str, int]
    checksum: Dict[str, str]


class EnhancedDatabaseMigrator:
    def __init__(self, source_db_path: str, target_db_path: str, export_path: str, batch_size: int = 1000):
        self.source_db_path = source_db_path
        self.target_db_path = target_db_path
        self.export_path = Path(export_path)
        self.state_file = self.export_path / 'migration_state.json'
        self.batch_size = batch_size
        self.setup_logging()

    def setup_logging(self) -> None:
        """Configure detailed logging with rotation."""
        self.export_path.mkdir(parents=True, exist_ok=True)
        log_file = self.export_path / 'migration.log'

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def load_migration_state(self) -> MigrationState:
        """Load or initialize migration state."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                data = json.load(f)
                return MigrationState(
                    completed_tables=set(data['completed_tables']),
                    completed_chats=set(data['completed_chats']),
                    last_successful_id=data['last_successful_id'],
                    total_rows=data['total_rows'],
                    migrated_rows=data['migrated_rows'],
                    checksum=data['checksum']
                )
        return MigrationState(set(), set(), {}, {}, {}, {})

    def save_migration_state(self, state: MigrationState) -> None:
        """Save current migration state."""
        with open(self.state_file, 'w') as f:
            json.dump({
                'completed_tables': list(state.completed_tables),
                'completed_chats': list(state.completed_chats),
                'last_successful_id': state.last_successful_id,
                'total_rows': state.total_rows,
                'migrated_rows': state.migrated_rows,
                'checksum': state.checksum
            }, f, indent=2)

    def sanitize_filename(self, name: str) -> str:
        """Create a safe filename from a string."""
        # Replace invalid filename characters
        invalid_chars = '<>:"/\\|?*'
        filename = ''.join(c if c not in invalid_chars else '_' for c in name)
        # Limit length and remove trailing spaces/dots
        return filename[:200].rstrip('. ')

    def format_chat_content(self, messages: List[Tuple[str, str, str]]) -> str:
        """Format chat messages into markdown content."""
        content: List[str] = []

        for sender, message, timestamp in messages:
            # Format timestamp
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S %Z')
            except ValueError:
                formatted_time = timestamp

            # Format message block
            content.extend([
                f"### {sender}",
                f"*{formatted_time}*",
                "",
                message,
                "",
                "---",
                ""
            ])

        return "\n".join(content)

    def calculate_table_checksum(self, conn: Connection, table_name: str) -> str:
        """Calculate a checksum for table structure and content."""
        cursor: Cursor = conn.cursor()

        try:
            # Get table schema
            cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            schema_row = cursor.fetchone()
            if not schema_row:
                raise ValueError(f"Table {table_name} not found")
            schema = schema_row[0]

            # Get row count and sample of data
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count_row = cursor.fetchone()
            if not count_row:
                raise ValueError(f"Could not get row count for {table_name}")
            row_count = count_row[0]

            # Sample some rows for checksum
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 100")
            sample_data = str(cursor.fetchall())

            # Combine and hash
            content = f"{schema}{row_count}{sample_data}"
            return hashlib.sha256(content.encode()).hexdigest()
        finally:
            cursor.close()

    def get_table_schema(self, conn: Connection) -> Dict[str, List[str]]:
        """Get schema information for all tables in the database."""
        cursor: Cursor = conn.cursor()
        schemas: Dict[str, List[str]] = {}

        try:
            # Get regular tables
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' 
                AND name NOT LIKE 'sqlite_%'
            """)
            tables = cursor.fetchall()

            for (table_name,) in tables:
                # Get column information
                col_cursor: Cursor = conn.cursor()
                try:
                    col_cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = col_cursor.fetchall()

                    # Format column definitions
                    column_defs: List[str] = []
                    for col in columns:
                        # col structure: (id, name, type, notnull, default_value, pk)
                        definition: List[str] = []
                        definition.append(f"{col[1]} {col[2]}")  # name and type

                        if col[5]:  # is primary key
                            definition.append("PRIMARY KEY")
                        if col[3]:  # not null
                            definition.append("NOT NULL")
                        if col[4] is not None:  # default value
                            definition.append(f"DEFAULT {col[4]}")

                        column_defs.append(" ".join(definition))

                    # Get foreign key constraints
                    fk_cursor: Cursor = conn.cursor()
                    try:
                        fk_cursor.execute(f"PRAGMA foreign_key_list({table_name})")
                        foreign_keys = fk_cursor.fetchall()
                        for fk in foreign_keys:
                            # fk structure: (id, seq, table, from, to, on_update, on_delete, match)
                            constraint = f"FOREIGN KEY ({fk[3]}) REFERENCES {fk[2]}({fk[4]})"
                            column_defs.append(constraint)
                    finally:
                        fk_cursor.close()

                    schemas[table_name] = column_defs
                finally:
                    col_cursor.close()

            return schemas
        finally:
            cursor.close()

    def validate_table_constraints(self, conn: Connection, table_name: str) -> List[str]:
        """Validate table constraints and data integrity."""
        cursor: Cursor = conn.cursor()
        errors: List[str] = []

        try:
            # Check for NULL in NOT NULL columns
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()

            null_check_cursor: Cursor = conn.cursor()
            try:
                for col in columns:
                    name, _, _, not_null = col[1:5]
                    if not_null:
                        null_check_cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {name} IS NULL")
                        count_row = null_check_cursor.fetchone()
                        if count_row and count_row[0] > 0:
                            errors.append(f"NULL values found in NOT NULL column: {name}")
            finally:
                null_check_cursor.close()

            # Check foreign key constraints
            fk_cursor: Cursor = conn.cursor()
            try:
                fk_cursor.execute(f"PRAGMA foreign_key_list({table_name})")
                foreign_keys = fk_cursor.fetchall()

                fk_check_cursor: Cursor = conn.cursor()
                try:
                    for fk in foreign_keys:
                        ref_table = fk[2]
                        from_col = fk[3]
                        to_col = fk[4]
                        fk_check_cursor.execute(f"""
                            SELECT COUNT(*) FROM {table_name} t 
                            LEFT JOIN {ref_table} r ON t.{from_col} = r.{to_col}
                            WHERE t.{from_col} IS NOT NULL AND r.{to_col} IS NULL
                        """)
                        count_row = fk_check_cursor.fetchone()
                        if count_row and count_row[0] > 0:
                            errors.append(f"Foreign key violation found in {table_name}.{from_col}")
                finally:
                    fk_check_cursor.close()
            finally:
                fk_cursor.close()

        except Exception as e:
            errors.append(f"Validation error for {table_name}: {str(e)}")
        finally:
            cursor.close()

        return errors

    def migrate_table_data(
            self,
            source_conn: Connection,
            target_conn: Connection,
            table_name: str,
            state: MigrationState
    ) -> None:
        """Migrate table data with progress tracking and resume capability."""
        source_cursor: Cursor = source_conn.cursor()

        try:
            # Get total rows
            source_cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count_row = source_cursor.fetchone()
            if not count_row:
                raise ValueError(f"Could not get row count for table {table_name}")
            total_rows = count_row[0]
            state.total_rows[table_name] = total_rows

            # Calculate start position
            last_id = state.last_successful_id.get(table_name, 0)

            with tqdm(total=total_rows, initial=state.migrated_rows.get(table_name, 0),
                      desc=f"Migrating {table_name}") as pbar:

                while True:
                    try:
                        # Get batch of rows
                        source_cursor.execute(f"""
                            SELECT * FROM {table_name}
                            WHERE rowid > ?
                            ORDER BY rowid
                            LIMIT ?
                        """, (last_id, self.batch_size))

                        rows = source_cursor.fetchall()
                        if not rows:
                            break

                        # Get column names
                        source_cursor.execute(f"PRAGMA table_info({table_name})")
                        columns = [col[1] for col in source_cursor.fetchall()]

                        # Insert batch
                        placeholders = ','.join(['?' for _ in columns])
                        insert_sql = f"""
                            INSERT OR REPLACE INTO {table_name} 
                            ({','.join(columns)}) 
                            VALUES ({placeholders})
                        """

                        target_cursor: Cursor = target_conn.cursor()
                        try:
                            target_cursor.executemany(insert_sql, rows)
                            target_conn.commit()
                        finally:
                            target_cursor.close()

                        # Update state
                        last_id = rows[-1][0]  # Assuming first column is ID
                        state.last_successful_id[table_name] = last_id
                        state.migrated_rows[table_name] = state.migrated_rows.get(table_name, 0) + len(rows)
                        self.save_migration_state(state)

                        pbar.update(len(rows))

                    except Exception as e:
                        logging.error(f"Error migrating {table_name} at ID {last_id}: {str(e)}")
                        time.sleep(1)  # Pause before retry
                        continue

        finally:
            source_cursor.close()

    def validate_schemas(self) -> Tuple[bool, List[str]]:
        """Enhanced schema validation with detailed checks."""
        errors: List[str] = []

        try:
            with sqlite3.connect(self.source_db_path) as source_conn, \
                    sqlite3.connect(self.target_db_path) as target_conn:

                source_conn.row_factory = sqlite3.Row
                target_conn.row_factory = sqlite3.Row

                source_schema = self.get_table_schema(source_conn)
                target_schema = self.get_table_schema(target_conn)

                # Basic schema validation
                for table, columns in source_schema.items():
                    if table in ['ChatConversations', 'ChatMessages']:
                        continue

                    if table not in target_schema:
                        errors.append(f"Missing table in target: {table}")
                        continue

                    # Compare columns
                    source_cols = set(col.lower() for col in columns)
                    target_cols = set(col.lower() for col in target_schema[table])

                    missing_cols = source_cols - target_cols
                    if missing_cols:
                        errors.append(f"Missing columns in target {table}: {missing_cols}")

                    # Validate table constraints
                    if table not in ['ChatConversations', 'ChatMessages']:
                        table_errors = self.validate_table_constraints(source_conn, table)
                        errors.extend(table_errors)

                # Validate target database integrity
                target_cursor: Cursor = target_conn.cursor()
                try:
                    target_cursor.execute("PRAGMA foreign_key_check")
                    if target_cursor.fetchone():
                        errors.append("Target database has foreign key violations")
                finally:
                    target_cursor.close()

        except Exception as e:
            errors.append(f"Schema validation error: {str(e)}")

        return len(errors) == 0, errors

    def export_single_conversation(
            self,
            conn: Connection,
            conv_id: int,
            media_id: Optional[int],
            media_name: Optional[str],
            conv_name: Optional[str],
            created_at: str
    ) -> None:
        """Export a single conversation to a markdown file."""
        cursor: Cursor = conn.cursor()

        try:
            # Get messages for this conversation
            cursor.execute("""
                SELECT sender, message, timestamp 
                FROM ChatMessages 
                WHERE conversation_id = ? 
                ORDER BY timestamp
            """, (conv_id,))
            messages = cursor.fetchall()

            if not messages:
                logging.warning(f"No messages found for conversation {conv_id}")
                return

            # Create chat export directory
            chat_path = self.export_path / 'chats'
            chat_path.mkdir(exist_ok=True)

            # Generate filename
            timestamp = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            safe_name = self.sanitize_filename(conv_name or f"conversation_{conv_id}")
            filename = f"chat_{timestamp.strftime('%Y%m%d_%H%M%S')}_{safe_name}.md"

            # Generate markdown content
            content = [
                f"# {conv_name or 'Untitled Conversation'}",
                "",
                "## Metadata",
                f"- Conversation ID: {conv_id}",
                f"- Created: {created_at}",
                f"- Media ID: {media_id or 'None'}",
                f"- Media Name: {media_name or 'None'}",
                "",
                "## Messages",
                "",
                self.format_chat_content(messages)
            ]

            # Write to file
            filepath = chat_path / filename
            filepath.write_text('\n'.join(content), encoding='utf-8')

            # Generate metadata file
            metadata = {
                'conversation_id': conv_id,
                'media_id': media_id,
                'media_name': media_name,
                'conversation_name': conv_name,
                'created_at': created_at,
                'message_count': len(messages),
                'export_timestamp': datetime.utcnow().isoformat(),
                'filename': filename
            }

            metadata_path = chat_path / f"{filename}.meta.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)

            logging.info(f"Exported conversation {conv_id} to {filename}")

        except Exception as e:
            logging.error(f"Error exporting conversation {conv_id}: {str(e)}")
            raise
        finally:
            cursor.close()

    def export_chats_to_markdown(self, conn: Connection, state: MigrationState) -> None:
        """Export chat conversations with progress tracking and resume capability."""
        cursor: Cursor = conn.cursor()

        try:
            # Get total conversations
            cursor.execute("SELECT COUNT(*) FROM ChatConversations")
            total_row = cursor.fetchone()
            if not total_row:
                raise ValueError("Could not get conversation count")
            total_conversations = total_row[0]

            # Get conversations not yet exported
            completed_chats_str = ','.join(map(str, state.completed_chats)) if state.completed_chats else 'NULL'
            cursor.execute(f"""
                SELECT id, media_id, media_name, conversation_name, created_at 
                FROM ChatConversations
                WHERE id NOT IN ({completed_chats_str})
                ORDER BY id
            """)
            conversations = cursor.fetchall()

            with tqdm(total=total_conversations, initial=len(state.completed_chats),
                      desc="Exporting chats") as pbar:
                for conv in conversations:
                    try:
                        conv_id, media_id, media_name, conv_name, created_at = conv

                        # Export conversation
                        self.export_single_conversation(
                            conn, conv_id, media_id, media_name, conv_name, created_at
                        )

                        # Update state
                        state.completed_chats.add(conv_id)
                        self.save_migration_state(state)
                        pbar.update(1)

                    except Exception as e:
                        logging.error(f"Error exporting conversation {conv[0]}: {str(e)}")
                        raise
        finally:
            cursor.close()

    def migrate_data(self) -> None:
        """Main migration function with resume capability."""
        state = self.load_migration_state()

        try:
            with sqlite3.connect(self.source_db_path) as source_conn, \
                    sqlite3.connect(self.target_db_path) as target_conn:

                # Enable foreign keys
                source_conn.execute("PRAGMA foreign_keys = ON")
                target_conn.execute("PRAGMA foreign_keys = ON")

                # Export chats first
                if not state.completed_chats:
                    self.export_chats_to_markdown(source_conn, state)

                # Get remaining tables to migrate
                cursor: Cursor = source_conn.cursor()
                try:
                    cursor.execute("""
                        SELECT name FROM sqlite_master 
                        WHERE type='table' 
                        AND name NOT IN ('ChatConversations', 'ChatMessages')
                        AND name NOT LIKE 'sqlite_%'
                    """)

                    tables = [t[0] for t in cursor.fetchall()
                              if t[0] not in state.completed_tables]
                finally:
                    cursor.close()

                # Migrate each table
                for table_name in tables:
                    try:
                        # Verify table integrity
                        current_checksum = self.calculate_table_checksum(source_conn, table_name)
                        if (table_name in state.checksum and
                                state.checksum[table_name] != current_checksum):
                            raise Exception(f"Table {table_name} has changed during migration")

                        state.checksum[table_name] = current_checksum

                        # Migrate table data
                        self.migrate_table_data(source_conn, target_conn, table_name, state)

                        # Mark table as completed
                        state.completed_tables.add(table_name)
                        self.save_migration_state(state)

                    except Exception as e:
                        logging.error(f"Error migrating table {table_name}: {str(e)}")
                        raise

        except Exception as e:
            logging.error(f"Migration failed: {str(e)}")
            raise


def main() -> None:
    """Main entry point with command line argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(description='Database Migration Tool')
    parser.add_argument('--source', required=True, help='Source database path')
    parser.add_argument('--target', required=True, help='Target database path')
    parser.add_argument('--export-path', required=True, help='Export directory path')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for migration')
    parser.add_argument('--verify-only', action='store_true', help='Only verify previous migration')
    parser.add_argument('--report', action='store_true', help='Generate migration report')

    args = parser.parse_args()

    try:
        migrator = EnhancedDatabaseMigrator(
            args.source,
            args.target,
            args.export_path,
            args.batch_size
        )

        if args.verify_only:
            logging.info("Verifying previous migration...")
            success, errors = migrator.verify_migration()
            if success:
                logging.info("Migration verification successful")
            else:
                logging.error("Migration verification failed:")
                for error in errors:
                    logging.error(f"  - {error}")
            return

        # Regular migration process
        logging.info("Starting database migration")

        # Validate schemas
        valid, errors = migrator.validate_schemas()
        if not valid:
            logging.error("Schema validation failed:")
            for error in errors:
                logging.error(f"  - {error}")
            return

        # Perform migration
        migrator.migrate_data()

        logging.info("Migration completed successfully")

    except Exception as e:
        logging.error(f"Migration failed: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.error("Migration interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unhandled exception: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)
