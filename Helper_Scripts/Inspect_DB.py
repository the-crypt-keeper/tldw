import sqlite3


def inspect_database():
    conn = sqlite3.connect('../media_summary.db')
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

    conn.close()


inspect_database()