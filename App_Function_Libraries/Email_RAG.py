# rag_system.py

import sqlite3
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# SQLite database setup
DB_PATH = 'email_analysis.db'


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.enable_load_extension(True)
        conn.load_extension("sqlite-vss")
        conn.execute('''
            CREATE TABLE IF NOT EXISTS email_embeddings (
                id INTEGER PRIMARY KEY,
                email_id TEXT UNIQUE,
                embedding BLOB
            )
        ''')
        conn.execute('CREATE VIRTUAL TABLE IF NOT EXISTS vss_email_embeddings USING vss0(embedding(384))')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_email_id ON email_embeddings(email_id)')


def embed_single_email(email: Dict[str, Any]) -> Tuple[str, np.ndarray]:
    embedding = model.encode(email['parsed_email']['body'])
    return (email['parsed_email']['message_id'], embedding)


def embed_emails_batch(emails: List[Dict[str, Any]], batch_size: int = 100):
    with Pool(processes=cpu_count()) as pool:
        results = []
        for i in range(0, len(emails), batch_size):
            batch = emails[i:i + batch_size]
            results.extend(pool.map(embed_single_email, batch))

    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany('INSERT OR REPLACE INTO email_embeddings (email_id, embedding) VALUES (?, ?)',
                         [(email_id, embedding.tobytes()) for email_id, embedding in results])
        conn.executemany(
            'INSERT OR REPLACE INTO vss_email_embeddings (rowid, embedding) VALUES ((SELECT id FROM email_embeddings WHERE email_id = ?), ?)',
            [(email_id, embedding.tobytes()) for email_id, embedding in results])


def retrieve_relevant_emails(query: str, k: int = 5) -> List[str]:
    query_vector = model.encode([query])
    with sqlite3.connect(DB_PATH) as conn:
        results = conn.execute('''
            SELECT email_embeddings.email_id
            FROM vss_email_embeddings
            JOIN email_embeddings ON vss_email_embeddings.rowid = email_embeddings.id
            WHERE vss_search(vss_email_embeddings.embedding, ?)
            LIMIT ?
        ''', (query_vector[0].tobytes(), k)).fetchall()

    return [email_id for (email_id,) in results]


def generate_response(query: str, relevant_email_ids: List[str], api_key: str) -> str:
    import openai
    openai.api_key = api_key

    with sqlite3.connect(DB_PATH) as conn:
        relevant_emails = conn.execute('''
            SELECT email_id, subject, sender, recipient, date, body
            FROM emails
            WHERE email_id IN ({})
        '''.format(','.join('?' * len(relevant_email_ids))), relevant_email_ids).fetchall()

    context = "\n\n".join([
        f"Subject: {email[1]}\n"
        f"From: {email[2]}\n"
        f"To: {email[3]}\n"
        f"Date: {email[4]}\n"
        f"Body: {email[5][:500]}..."
        for email in relevant_emails
    ])

    prompt = f"""Based on the following email excerpts, please answer the question: "{query}"

Email Excerpts:
{context}

Please provide a concise and informative answer based solely on the information given in these email excerpts. If the answer cannot be determined from the given information, please state that.

Answer:"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant analyzing emails."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        return response.choices[0].message['content'].strip()
    except openai.error.OpenAIError as e:
        return f"Error generating response: {str(e)}"


# Initialize the database when the module is imported
init_db()