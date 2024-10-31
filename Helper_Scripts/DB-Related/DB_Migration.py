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


