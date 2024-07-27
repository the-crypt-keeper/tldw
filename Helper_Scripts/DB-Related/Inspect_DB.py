import sqlite3
import os


def inspect_database():
    # Specify the path to your database file
    db_name = os.getenv('DB_NAME', 'media_summary.db')
    if not os.path.exists(db_name):
        print(f"Database file {db_name} does not exist.")
        return

    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Check Media table
        cursor.execute("SELECT * FROM Media")
        media_rows = cursor.fetchall()
        print("Media Table:")
        for row in media_rows:
            print(row)

        # Check Keywords table
        cursor.execute("SELECT * FROM Keywords")
        keyword_rows = cursor.fetchall()
        print("Keywords Table:")
        for row in keyword_rows:
            print(row)

        # Check MediaKeywords table
        cursor.execute("SELECT * FROM MediaKeywords")
        media_keyword_rows = cursor.fetchall()
        print("MediaKeywords Table:")
        for row in media_keyword_rows:
            print(row)

        # Check media_fts table
        cursor.execute("SELECT * FROM media_fts")
        media_fts_rows = cursor.fetchall()
        print("media_fts Table:")
        for row in media_fts_rows:
            print(row)

        # Check keyword_fts table
        cursor.execute("SELECT * FROM keyword_fts")
        keyword_fts_rows = cursor.fetchall()
        print("keyword_fts Table:")
        for row in keyword_fts_rows:
            print(row)

        conn.close()
    except sqlite3.OperationalError as e:
        print(f"SQLite error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    inspect_database()


inspect_database()