# RAG_Library.py
#########################################
# RAG Search & Related Functions Library
# This library is used to hold any/all RAG-related operations.
# Currently, all of this code was generated from Sonnet 3.5. 0_0
#
####




# Plain
# import os
# from typing import List, Tuple, Callable, Optional
# from contextlib import contextmanager
# import sqlite3
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import logging
# from dotenv import load_dotenv
#
# load_dotenv()
#
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
#
#
# class RAGException(Exception):
#     """Custom exception class for RAG-related errors"""
#     pass
#
#
# class BaseRAGSystem:
#     def __init__(self, db_path: str, model_name: Optional[str] = None):
#         """
#         Initialize the RAG system.
#
#         :param db_path: Path to the SQLite database
#         :param model_name: Name of the SentenceTransformer model to use
#         """
#         self.db_path = db_path
#         self.model_name = model_name or os.getenv('DEFAULT_MODEL_NAME', 'all-MiniLM-L6-v2')
#         try:
#             self.model = SentenceTransformer(self.model_name)
#             logger.info(f"Initialized SentenceTransformer with model: {self.model_name}")
#         except Exception as e:
#             logger.error(f"Failed to initialize SentenceTransformer: {e}")
#             raise RAGException(f"Model initialization failed: {e}")
#
#         self.init_db()
#
#     @contextmanager
#     def get_db_connection(self):
#         conn = sqlite3.connect(self.db_path)
#         try:
#             yield conn
#         finally:
#             conn.close()
#
#     def init_db(self):
#         try:
#             with self.get_db_connection() as conn:
#                 cursor = conn.cursor()
#                 cursor.execute('''
#                 CREATE TABLE IF NOT EXISTS documents (
#                     id INTEGER PRIMARY KEY,
#                     title TEXT,
#                     content TEXT,
#                     embedding BLOB
#                 )
#                 ''')
#                 conn.commit()
#             logger.info("Initialized database schema")
#         except sqlite3.Error as e:
#             logger.error(f"Failed to initialize database schema: {e}")
#             raise RAGException(f"Database schema initialization failed: {e}")
#
#     def add_documents(self, documents: List[Tuple[str, str]]):
#         try:
#             embeddings = self.model.encode([content for _, content in documents])
#             with self.get_db_connection() as conn:
#                 cursor = conn.cursor()
#                 cursor.executemany(
#                     'INSERT INTO documents (title, content, embedding) VALUES (?, ?, ?)',
#                     [(title, content, embedding.tobytes()) for (title, content), embedding in zip(documents, embeddings)]
#                 )
#                 conn.commit()
#             logger.info(f"Added {len(documents)} documents in batch")
#         except Exception as e:
#             logger.error(f"Failed to add documents in batch: {e}")
#             raise RAGException(f"Batch document addition failed: {e}")
#
#     def get_documents(self) -> List[Tuple[int, str, str, np.ndarray]]:
#         try:
#             with self.get_db_connection() as conn:
#                 cursor = conn.cursor()
#                 cursor.execute('SELECT id, title, content, embedding FROM documents')
#                 documents = [(id, title, content, np.frombuffer(embedding, dtype=np.float32))
#                              for id, title, content, embedding in cursor.fetchall()]
#             logger.info(f"Retrieved {len(documents)} documents")
#             return documents
#         except sqlite3.Error as e:
#             logger.error(f"Failed to retrieve documents: {e}")
#             raise RAGException(f"Document retrieval failed: {e}")
#
#     def close(self):
#         try:
#             self.conn.close()
#             logger.info("Closed database connection")
#         except sqlite3.Error as e:
#             logger.error(f"Error closing database connection: {e}")
#
#
# class StandardRAGSystem(BaseRAGSystem):
#     def get_relevant_documents(self, query: str, top_k: int = 3) -> List[Tuple[int, str, str, float]]:
#         try:
#             query_embedding = self.model.encode([query])[0]
#             documents = self.get_documents()
#             similarities = [
#                 (id, title, content, cosine_similarity([query_embedding], [doc_embedding])[0][0])
#                 for id, title, content, doc_embedding in documents
#             ]
#             similarities.sort(key=lambda x: x[3], reverse=True)
#             logger.info(f"Retrieved top {top_k} relevant documents for query")
#             return similarities[:top_k]
#         except Exception as e:
#             logger.error(f"Error in getting relevant documents: {e}")
#             raise RAGException(f"Retrieval of relevant documents failed: {e}")
#
#     def rag_query(self, query: str, llm_function: Callable[[str], str], top_k: int = 3) -> str:
#         try:
#             relevant_docs = self.get_relevant_documents(query, top_k)
#             context = "\n\n".join([f"Title: {title}\nContent: {content}" for _, title, content, _ in relevant_docs])
#
#             llm_prompt = f"Based on the following context, please answer the query:\n\nContext:\n{context}\n\nQuery: {query}"
#
#             response = llm_function(llm_prompt)
#             logger.info("Generated response for query")
#             return response
#         except Exception as e:
#             logger.error(f"Error in RAG query: {e}")
#             raise RAGException(f"RAG query failed: {e}")
#
#
# class HyDERAGSystem(BaseRAGSystem):
#     def generate_hypothetical_document(self, query: str, llm_function: Callable[[str], str]) -> str:
#         try:
#             prompt = f"Given the question '{query}', write a short paragraph that would answer this question. Do not include the question itself in your response."
#             hypothetical_doc = llm_function(prompt)
#             logger.info("Generated hypothetical document")
#             return hypothetical_doc
#         except Exception as e:
#             logger.error(f"Error generating hypothetical document: {e}")
#             raise RAGException(f"Hypothetical document generation failed: {e}")
#
#     def get_relevant_documents(self, query: str, llm_function: Callable[[str], str], top_k: int = 3) -> List[
#         Tuple[int, str, str, float]]:
#         try:
#             hypothetical_doc = self.generate_hypothetical_document(query, llm_function)
#             hyde_embedding = self.model.encode([hypothetical_doc])[0]
#
#             documents = self.get_documents()
#             similarities = [
#                 (id, title, content, cosine_similarity([hyde_embedding], [doc_embedding])[0][0])
#                 for id, title, content, doc_embedding in documents
#             ]
#             similarities.sort(key=lambda x: x[3], reverse=True)
#             logger.info(f"Retrieved top {top_k} relevant documents using HyDE")
#             return similarities[:top_k]
#         except Exception as e:
#             logger.error(f"Error in getting relevant documents with HyDE: {e}")
#             raise RAGException(f"HyDE retrieval of relevant documents failed: {e}")
#
#     def rag_query(self, query: str, llm_function: Callable[[str], str], top_k: int = 3) -> str:
#         try:
#             relevant_docs = self.get_relevant_documents(query, llm_function, top_k)
#             context = "\n\n".join([f"Title: {title}\nContent: {content}" for _, title, content, _ in relevant_docs])
#
#             llm_prompt = f"Based on the following context, please answer the query:\n\nContext:\n{context}\n\nQuery: {query}"
#
#             response = llm_function(llm_prompt)
#             logger.info("Generated response for query using HyDE")
#             return response
#         except Exception as e:
#             logger.error(f"Error in HyDE RAG query: {e}")
#             raise RAGException(f"HyDE RAG query failed: {e}")
#
#
# # Example usage with error handling
# def mock_llm(prompt: str) -> str:
#     if "write a short paragraph" in prompt:
#         return "Paris, the capital of France, is renowned for its iconic Eiffel Tower and rich cultural heritage."
#     else:
#         return f"This is a mock LLM response for the prompt: {prompt}"
#
#
# def main():
#     use_hyde = False  # Set this to True when you want to enable HyDE
#
#     try:
#         if use_hyde:
#             rag_system = HyDERAGSystem('rag_database.db')
#             logger.info("Using HyDE RAG System")
#         else:
#             rag_system = StandardRAGSystem('rag_database.db')
#             logger.info("Using Standard RAG System")
#
#         # Add sample documents in batch
#         sample_docs = [
#             ("Paris", "Paris is the capital of France and is known for the Eiffel Tower."),
#             ("London", "London is the capital of the United Kingdom and home to Big Ben."),
#             ("Tokyo", "Tokyo is the capital of Japan and is famous for its bustling city life.")
#         ]
#
#         for title, content in sample_docs:
#             rag_system.add_document(title, content)
#
#         query = "What is the capital of France?"
#         result = rag_system.rag_query(query, mock_llm)
#         print(f"Query: {query}")
#         print(f"Result: {result}")
#
#     except RAGException as e:
#         logger.error(f"RAG system error: {e}")
#         print(f"An error occurred: {e}")
#     except Exception as e:
#         logger.error(f"Unexpected error: {e}")
#         print(f"An unexpected error occurred: {e}")
#     finally:
#         if 'rag_system' in locals():
#             rag_system.close()
#
#
# if __name__ == "__main__":
#     main()
#
#

####################################################################################
# From async to non-async:
# Efficient parts retained:
#
# Numpy operations for handling embeddings
# SentenceTransformer with the same model choice
# Improved similarity search:
# Using numpy's argsort for efficient top-k retrieval
# Optimized database operations:
# Added an index on the embedding column
# Using a single SELECT query to fetch all document data
# Added caching:
#   LRU cache for embeddings to avoid recomputing frequently used embeddings
# Restructured code:
#   Removed object-oriented and async patterns
#   Organized into clear functional sections: database, embedding, RAG, and main functions
#
# To integrate this with Gradio, you would typically:
#
# Import the necessary functions (e.g., init_db, add_documents, rag_query) in your Gradio script.
# Call init_db() when your application starts.
# Use rag_query() in your Gradio interface function to process user queries.


import os
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import List, Tuple, Callable
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize global variables
DB_PATH = 'rag_database.db'
MODEL_NAME = os.getenv('DEFAULT_MODEL_NAME', 'all-MiniLM-L6-v2')
model = SentenceTransformer(MODEL_NAME)


# Database functions
def init_db():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                title TEXT,
                content TEXT,
                embedding BLOB
            )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_embedding ON documents(embedding)')
        logger.info("Initialized database schema")
    except sqlite3.Error as e:
        logger.error(f"Failed to initialize database schema: {e}")
        raise


def add_documents(documents: List[Tuple[str, str]]):
    try:
        embeddings = model.encode([content for _, content in documents])
        with sqlite3.connect(DB_PATH) as conn:
            conn.executemany(
                'INSERT INTO documents (title, content, embedding) VALUES (?, ?, ?)',
                [(title, content, embedding.tobytes()) for (title, content), embedding in zip(documents, embeddings)]
            )
        logger.info(f"Added {len(documents)} documents")
    except Exception as e:
        logger.error(f"Failed to add documents: {e}")
        raise


def get_documents() -> List[Tuple[int, str, str, np.ndarray]]:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute('SELECT id, title, content, embedding FROM documents')
            documents = [
                (id, title, content, np.frombuffer(embedding, dtype=np.float32))
                for id, title, content, embedding in cursor
            ]
        logger.info(f"Retrieved {len(documents)} documents")
        return documents
    except sqlite3.Error as e:
        logger.error(f"Failed to retrieve documents: {e}")
        raise


# Embedding and similarity functions
@lru_cache(maxsize=1000)
def get_embedding(text: str) -> np.ndarray:
    return model.encode([text])[0]


def compute_similarities(query_embedding: np.ndarray, document_embeddings: List[np.ndarray]) -> np.ndarray:
    return cosine_similarity([query_embedding], document_embeddings)[0]


# RAG functions
def get_relevant_documents(query: str, top_k: int = 3) -> List[Tuple[int, str, str, float]]:
    try:
        query_embedding = get_embedding(query)
        documents = get_documents()
        document_embeddings = [doc[3] for doc in documents]

        similarities = compute_similarities(query_embedding, document_embeddings)

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        relevant_docs = [
            (documents[i][0], documents[i][1], documents[i][2], similarities[i])
            for i in top_indices
        ]

        logger.info(f"Retrieved top {top_k} relevant documents for query")
        return relevant_docs
    except Exception as e:
        logger.error(f"Error in getting relevant documents: {e}")
        raise


def rag_query(query: str, llm_function: Callable[[str], str], top_k: int = 3) -> str:
    try:
        relevant_docs = get_relevant_documents(query, top_k)
        context = "\n\n".join([f"Title: {title}\nContent: {content}" for _, title, content, _ in relevant_docs])

        llm_prompt = f"Based on the following context, please answer the query:\n\nContext:\n{context}\n\nQuery: {query}"

        response = llm_function(llm_prompt)
        logger.info("Generated response for query")
        return response
    except Exception as e:
        logger.error(f"Error in RAG query: {e}")
        raise


# Example usage
def mock_llm(prompt: str) -> str:
    return f"This is a mock LLM response for the prompt: {prompt}"


def main():
    try:
        init_db()

        # Add sample documents
        sample_docs = [
            ("Paris", "Paris is the capital of France and is known for the Eiffel Tower."),
            ("London", "London is the capital of the United Kingdom and home to Big Ben."),
            ("Tokyo", "Tokyo is the capital of Japan and is famous for its bustling city life.")
        ]
        add_documents(sample_docs)

        query = "What is the capital of France?"
        result = rag_query(query, mock_llm)
        print(f"Query: {query}")
        print(f"Result: {result}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()



####################################################################################
# async:

# import os
# import asyncio
# from typing import List, Tuple, Callable, Optional
# import aiosqlite
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import logging
# from dotenv import load_dotenv
#
# load_dotenv()
#
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
#
#
# class RAGException(Exception):
#     """Custom exception class for RAG-related errors"""
#     pass
#
#
# class BaseRAGSystem:
#     def __init__(self, db_path: str, model_name: Optional[str] = None):
#         """
#         Initialize the RAG system.
#
#         :param db_path: Path to the SQLite database
#         :param model_name: Name of the SentenceTransformer model to use
#         """
#         self.db_path = db_path
#         self.model_name = model_name or os.getenv('DEFAULT_MODEL_NAME', 'all-MiniLM-L6-v2')
#         try:
#             self.model = SentenceTransformer(self.model_name)
#             logger.info(f"Initialized SentenceTransformer with model: {self.model_name}")
#         except Exception as e:
#             logger.error(f"Failed to initialize SentenceTransformer: {e}")
#             raise RAGException(f"Model initialization failed: {e}")
#
#     async def init_db(self):
#         try:
#             async with aiosqlite.connect(self.db_path) as db:
#                 await db.execute('''
#                 CREATE TABLE IF NOT EXISTS documents (
#                     id INTEGER PRIMARY KEY,
#                     title TEXT,
#                     content TEXT,
#                     embedding BLOB
#                 )
#                 ''')
#                 await db.commit()
#             logger.info("Initialized database schema")
#         except aiosqlite.Error as e:
#             logger.error(f"Failed to initialize database schema: {e}")
#             raise RAGException(f"Database schema initialization failed: {e}")
#
#     async def add_documents(self, documents: List[Tuple[str, str]]):
#         try:
#             embeddings = self.model.encode([content for _, content in documents])
#             async with aiosqlite.connect(self.db_path) as db:
#                 await db.executemany(
#                     'INSERT INTO documents (title, content, embedding) VALUES (?, ?, ?)',
#                     [(title, content, embedding.tobytes()) for (title, content), embedding in
#                      zip(documents, embeddings)]
#                 )
#                 await db.commit()
#             logger.info(f"Added {len(documents)} documents in batch")
#         except Exception as e:
#             logger.error(f"Failed to add documents in batch: {e}")
#             raise RAGException(f"Batch document addition failed: {e}")
#
# async def get_documents(self) -> List[Tuple[int, str, str, np.ndarray, str]]:
#     try:
#         async with aiosqlite.connect(self.db_path) as db:
#             async with db.execute('SELECT id, title, content, embedding, source FROM documents') as cursor:
#                 documents = [
#                     (id, title, content, np.frombuffer(embedding, dtype=np.float32), source)
#                     async for id, title, content, embedding, source in cursor
#                 ]
#         logger.info(f"Retrieved {len(documents)} documents")
#         return documents
#     except aiosqlite.Error as e:
#         logger.error(f"Failed to retrieve documents: {e}")
#         raise RAGException(f"Document retrieval failed: {e}")
#
#
# class AsyncStandardRAGSystem(BaseRAGSystem):
#     async def get_relevant_documents(self, query: str, top_k: int = 3) -> List[Tuple[int, str, str, float]]:
#         try:
#             query_embedding = self.model.encode([query])[0]
#             documents = await self.get_documents()
#             similarities = [
#                 (id, title, content, cosine_similarity([query_embedding], [doc_embedding])[0][0])
#                 for id, title, content, doc_embedding in documents
#             ]
#             similarities.sort(key=lambda x: x[3], reverse=True)
#             logger.info(f"Retrieved top {top_k} relevant documents for query")
#             return similarities[:top_k]
#         except Exception as e:
#             logger.error(f"Error in getting relevant documents: {e}")
#             raise RAGException(f"Retrieval of relevant documents failed: {e}")
#
#     async def rag_query(self, query: str, llm_function: Callable[[str], str], top_k: int = 3) -> str:
#         try:
#             relevant_docs = await self.get_relevant_documents(query, top_k)
#             context = "\n\n".join([f"Title: {title}\nContent: {content}\nSource: {source}" for _, title, content, _, source in relevant_docs])
#
#             llm_prompt = f"Based on the following context, please answer the query. Include citations in your response using [Source] format:\n\nContext:\n{context}\n\nQuery: {query}"
#
#             response = llm_function(llm_prompt)
#             logger.info("Generated response for query")
#             return response
#         except Exception as e:
#             logger.error(f"Error in RAG query: {e}")
#             raise RAGException(f"RAG query failed: {e}")
#
#
# class AsyncHyDERAGSystem(BaseRAGSystem):
#     async def generate_hypothetical_document(self, query: str, llm_function: Callable[[str], str]) -> str:
#         try:
#             prompt = f"Given the question '{query}', write a short paragraph that would answer this question. Do not include the question itself in your response."
#             hypothetical_doc = llm_function(prompt)
#             logger.info("Generated hypothetical document")
#             return hypothetical_doc
#         except Exception as e:
#             logger.error(f"Error generating hypothetical document: {e}")
#             raise RAGException(f"Hypothetical document generation failed: {e}")
#
#     async def get_relevant_documents(self, query: str, llm_function: Callable[[str], str], top_k: int = 3) -> List[
#         Tuple[int, str, str, float]]:
#         try:
#             hypothetical_doc = await self.generate_hypothetical_document(query, llm_function)
#             hyde_embedding = self.model.encode([hypothetical_doc])[0]
#
#             documents = await self.get_documents()
#             similarities = [
#                 (id, title, content, cosine_similarity([hyde_embedding], [doc_embedding])[0][0])
#                 for id, title, content, doc_embedding in documents
#             ]
#             similarities.sort(key=lambda x: x[3], reverse=True)
#             logger.info(f"Retrieved top {top_k} relevant documents using HyDE")
#             return similarities[:top_k]
#         except Exception as e:
#             logger.error(f"Error in getting relevant documents with HyDE: {e}")
#             raise RAGException(f"HyDE retrieval of relevant documents failed: {e}")
#
#     async def rag_query(self, query: str, llm_function: Callable[[str], str], top_k: int = 3) -> str:
#         try:
#             relevant_docs = await self.get_relevant_documents(query, llm_function, top_k)
#             context = "\n\n".join([f"Title: {title}\nContent: {content}" for _, title, content, _ in relevant_docs])
#
#             llm_prompt = f"Based on the following context, please answer the query:\n\nContext:\n{context}\n\nQuery: {query}"
#
#             response = llm_function(llm_prompt)
#             logger.info("Generated response for query using HyDE")
#             return response
#         except Exception as e:
#             logger.error(f"Error in HyDE RAG query: {e}")
#             raise RAGException(f"HyDE RAG query failed: {e}")
#
#
# # Example usage with error handling
# def mock_llm(prompt: str) -> str:
#     if "write a short paragraph" in prompt:
#         return "Paris, the capital of France, is renowned for its iconic Eiffel Tower and rich cultural heritage."
#     else:
#         return f"This is a mock LLM response for the prompt: {prompt}"
#
#
# async def main():
#     use_hyde = False  # Set this to True when you want to enable HyDE
#
#     try:
#         if use_hyde:
#             rag_system = AsyncHyDERAGSystem('rag_database.db')
#             logger.info("Using Async HyDE RAG System")
#         else:
#             rag_system = AsyncStandardRAGSystem('rag_database.db')
#             logger.info("Using Async Standard RAG System")
#
#         await rag_system.init_db()
#
#         # Add sample documents
#         sample_docs = [
#             ("Paris", "Paris is the capital of France and is known for the Eiffel Tower."),
#             ("London", "London is the capital of the United Kingdom and home to Big Ben."),
#             ("Tokyo", "Tokyo is the capital of Japan and is famous for its bustling city life.")
#         ]
#
#         await rag_system.add_documents(sample_docs)
#
#         query = "What is the capital of France?"
#         result = await rag_system.rag_query(query, mock_llm)
#         print(f"Query: {query}")
#         print(f"Result: {result}")
#
#     except RAGException as e:
#         logger.error(f"RAG system error: {e}")
#         print(f"An error occurred: {e}")
#     except Exception as e:
#         logger.error(f"Unexpected error: {e}")
#         print(f"An unexpected error occurred: {e}")
#
#
# if __name__ == "__main__":
#     asyncio.run(main())



#
# from fastapi import FastAPI, HTTPException
#
# app = FastAPI()
# rag_system = AsyncStandardRAGSystem('rag_database.db')
#
# @app.on_event("startup")
# async def startup_event():
#     await rag_system.init_db()
#
# @app.get("/query")
# async def query(q: str):
#     try:
#         result = await rag_system.rag_query(q, mock_llm)
#         return {"query": q, "result": result}
#     except RAGException as e:
#         raise HTTPException(status_code=500, detail=str(e))
#


############################################################################################
# Using FAISS
#
#
#
# Update DB
# async def init_db(self):
#     try:
#         async with aiosqlite.connect(self.db_path) as db:
#             await db.execute('''
#             CREATE TABLE IF NOT EXISTS documents (
#                 id INTEGER PRIMARY KEY,
#                 title TEXT,
#                 content TEXT,
#                 embedding BLOB,
#                 source TEXT
#             )
#             ''')
#             await db.commit()
#         logger.info("Initialized database schema")
#     except aiosqlite.Error as e:
#         logger.error(f"Failed to initialize database schema: {e}")
#         raise RAGException(f"Database schema initialization failed: {e}")
#
#

# import os
# import asyncio
# from typing import List, Tuple, Callable, Optional
# import aiosqlite
# import numpy as np
# from sentence_transformers import SentenceTransformer
# import faiss
# import logging
# from dotenv import load_dotenv
#
# load_dotenv()
#
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
#
#
# class RAGException(Exception):
#     """Custom exception class for RAG-related errors"""
#     pass
#
#
# class AsyncFAISSRAGSystem:
#     def __init__(self, db_path: str, model_name: Optional[str] = None):
#         self.db_path = db_path
#         self.model_name = model_name or os.getenv('DEFAULT_MODEL_NAME', 'all-MiniLM-L6-v2')
#         try:
#             self.model = SentenceTransformer(self.model_name)
#             logger.info(f"Initialized SentenceTransformer with model: {self.model_name}")
#         except Exception as e:
#             logger.error(f"Failed to initialize SentenceTransformer: {e}")
#             raise RAGException(f"Model initialization failed: {e}")
#
#         self.index = None
#         self.document_lookup = {}
#
#     async def init_db(self):
#         try:
#             async with aiosqlite.connect(self.db_path) as db:
#                 await db.execute('''
#                 CREATE TABLE IF NOT EXISTS documents (
#                     id INTEGER PRIMARY KEY,
#                     title TEXT,
#                     content TEXT
#                 )
#                 ''')
#                 await db.commit()
#             logger.info("Initialized database schema")
#         except aiosqlite.Error as e:
#             logger.error(f"Failed to initialize database schema: {e}")
#             raise RAGException(f"Database schema initialization failed: {e}")
#
# async def add_documents(self, documents: List[Tuple[str, str, str]]):
#     try:
#         embeddings = self.model.encode([content for _, content, _ in documents])
#         async with aiosqlite.connect(self.db_path) as db:
#             await db.executemany(
#                 'INSERT INTO documents (title, content, embedding, source) VALUES (?, ?, ?, ?)',
#                 [(title, content, embedding.tobytes(), source) for (title, content, source), embedding in
#                  zip(documents, embeddings)]
#             )
#             await db.commit()
#         logger.info(f"Added {len(documents)} documents in batch")
#     except Exception as e:
#         logger.error(f"Failed to add documents in batch: {e}")
#         raise RAGException(f"Batch document addition failed: {e}")
#
#     async def get_relevant_documents(self, query: str, top_k: int = 3) -> List[Tuple[int, str, str, float, str]]:
#         try:
#             query_embedding = self.model.encode([query])[0]
#             documents = await self.get_documents()
#             similarities = [
#                 (id, title, content, cosine_similarity([query_embedding], [doc_embedding])[0][0], source)
#                 for id, title, content, doc_embedding, source in documents
#             ]
#             similarities.sort(key=lambda x: x[3], reverse=True)
#             logger.info(f"Retrieved top {top_k} relevant documents for query")
#             return similarities[:top_k]
#         except Exception as e:
#             logger.error(f"Error in getting relevant documents: {e}")
#             raise RAGException(f"Retrieval of relevant documents failed: {e}")
#
#     async def rag_query(self, query: str, llm_function: Callable[[str], str], top_k: int = 3) -> str:
#         try:
#             relevant_docs = await self.get_relevant_documents(query, top_k)
#             context = "\n\n".join([f"Title: {title}\nContent: {content}" for _, title, content, _ in relevant_docs])
#
#             llm_prompt = f"Based on the following context, please answer the query:\n\nContext:\n{context}\n\nQuery: {query}"
#
#             response = llm_function(llm_prompt)
#             logger.info("Generated response for query")
#             return response
#         except Exception as e:
#             logger.error(f"Error in RAG query: {e}")
#             raise RAGException(f"RAG query failed: {e}")
#
#
# class AsyncFAISSHyDERAGSystem(AsyncFAISSRAGSystem):
#     async def generate_hypothetical_document(self, query: str, llm_function: Callable[[str], str]) -> str:
#         try:
#             prompt = f"Given the question '{query}', write a short paragraph that would answer this question. Do not include the question itself in your response."
#             hypothetical_doc = llm_function(prompt)
#             logger.info("Generated hypothetical document")
#             return hypothetical_doc
#         except Exception as e:
#             logger.error(f"Error generating hypothetical document: {e}")
#             raise RAGException(f"Hypothetical document generation failed: {e}")
#
#     async def get_relevant_documents(self, query: str, llm_function: Callable[[str], str], top_k: int = 3) -> List[
#         Tuple[int, str, str, float]]:
#         try:
#             hypothetical_doc = await self.generate_hypothetical_document(query, llm_function)
#             hyde_embedding = self.model.encode([hypothetical_doc])[0]
#
#             distances, indices = self.index.search(np.array([hyde_embedding]), top_k)
#
#             results = []
#             for i, idx in enumerate(indices[0]):
#                 doc_id = list(self.document_lookup.keys())[idx]
#                 title, content = self.document_lookup[doc_id]
#                 results.append((doc_id, title, content, distances[0][i]))
#
#             logger.info(f"Retrieved top {top_k} relevant documents using HyDE")
#             return results
#         except Exception as e:
#             logger.error(f"Error in getting relevant documents with HyDE: {e}")
#             raise RAGException(f"HyDE retrieval of relevant documents failed: {e}")
#
#
# # Example usage
# def mock_llm(prompt: str) -> str:
#     if "write a short paragraph" in prompt:
#         return "Paris, the capital of France, is renowned for its iconic Eiffel Tower and rich cultural heritage."
#     else:
#         return f"This is a mock LLM response for the prompt: {prompt}"
#
#
# async def main():
#     use_hyde = False  # Set this to True when you want to enable HyDE
#
#     try:
#         if use_hyde:
#             rag_system = AsyncFAISSHyDERAGSystem('rag_database.db')
#             logger.info("Using Async FAISS HyDE RAG System")
#         else:
#             rag_system = AsyncFAISSRAGSystem('rag_database.db')
#             logger.info("Using Async FAISS RAG System")
#
#         await rag_system.init_db()
#
#         # Add sample documents
#         sample_docs = [
#             ("Paris", "Paris is the capital of France and is known for the Eiffel Tower."),
#             ("London", "London is the capital of the United Kingdom and home to Big Ben."),
#             ("Tokyo", "Tokyo is the capital of Japan and is famous for its bustling city life.")
#         ]
#
#         await rag_system.add_documents(sample_docs)
#
#         query = "What is the capital of France?"
#         result = await rag_system.rag_query(query, mock_llm)
#         print(f"Query: {query}")
#         print(f"Result: {result}")
#
#     except RAGException as e:
#         logger.error(f"RAG system error: {e}")
#         print(f"An error occurred: {e}")
#     except Exception as e:
#         logger.error(f"Unexpected error: {e}")
#         print(f"An unexpected error occurred: {e}")
#
#
# if __name__ == "__main__":
#     asyncio.run(main())


"""
Key changes in this FAISS-integrated version:

We've replaced the cosine similarity search with FAISS indexing and search.
The add_documents method now adds embeddings to the FAISS index as well as storing documents in the SQLite database.
We maintain a document_lookup dictionary to quickly retrieve document content based on FAISS search results.
The get_relevant_documents method now uses FAISS for similarity search instead of computing cosine similarities manually.
We've kept the asynchronous structure for database operations, while FAISS operations remain synchronous (as FAISS doesn't have built-in async support).

Benefits of using FAISS:

Scalability: FAISS can handle millions of vectors efficiently, making it suitable for large document collections.
Speed: FAISS is optimized for fast similarity search, which can significantly improve query times as your dataset grows.
Memory Efficiency: FAISS provides various index types that can trade off between search accuracy and memory usage, allowing you to optimize for your specific use case.

Considerations:

This implementation uses a simple IndexFlatL2 FAISS index, which performs exact search. For larger datasets, you might want to consider approximate search methods like IndexIVFFlat for better scalability.
The current implementation keeps all document content in memory (in the document_lookup dictionary). For very large datasets, you might want to modify this to fetch document content from the database as needed.
If you're dealing with a very large number of documents, you might want to implement batch processing for adding documents to the FAISS index.

This FAISS-integrated version should provide better performance for similarity search, especially as your document collection grows larger
"""


###############################################################################################################
# Web Search
# Output from Sonnet 3.5 regarding how to add web searches to the RAG system
# Integrating web search into your RAG system can significantly enhance its capabilities by providing up-to-date information. Here's how you can modify your RAG system to include web search:
#
# First, you'll need to choose a web search API. Some popular options include:
#
# Google Custom Search API
# Bing Web Search API
# DuckDuckGo API
# SerpAPI (which can interface with multiple search engines)
#
#
#
# For this example, let's use the DuckDuckGo API, as it's free and doesn't require authentication.
#
# Install the required library:
# `pip install duckduckgo-search`
#
# Add a new method to your RAG system for web search:
# ```
# from duckduckgo_search import ddg
#
# class AsyncRAGSystem:
#     # ... (existing code) ...
#
#     async def web_search(self, query: str, num_results: int = 3) -> List[Dict[str, str]]:
#         try:
#             results = ddg(query, max_results=num_results)
#             return [{'title': r['title'], 'content': r['body'], 'source': r['href']} for r in results]
#         except Exception as e:
#             logger.error(f"Error in web search: {e}")
#             raise RAGException(f"Web search failed: {e}")
#
#     async def add_web_results_to_db(self, results: List[Dict[str, str]]):
#         try:
#             documents = [(r['title'], r['content'], r['source']) for r in results]
#             await self.add_documents(documents)
#             logger.info(f"Added {len(documents)} web search results to the database")
#         except Exception as e:
#             logger.error(f"Error adding web search results to database: {e}")
#             raise RAGException(f"Adding web search results failed: {e}")
#
#     async def rag_query_with_web_search(self, query: str, llm_function: Callable[[str], str], top_k: int = 3,
#                                         use_web_search: bool = True, num_web_results: int = 3) -> str:
#         try:
#             if use_web_search:
#                 web_results = await self.web_search(query, num_web_results)
#                 await self.add_web_results_to_db(web_results)
#
#             relevant_docs = await self.get_relevant_documents(query, top_k)
#             context = "\n\n".join([f"Title: {title}\nContent: {content}\nSource: {source}"
#                                    for _, title, content, _, source in relevant_docs])
#
#             llm_prompt = f"Based on the following context, please answer the query. Include citations in your response using [Source] format:\n\nContext:\n{context}\n\nQuery: {query}"
#
#             response = llm_function(llm_prompt)
#             logger.info("Generated response for query with web search")
#             return response
#         except Exception as e:
#             logger.error(f"Error in RAG query with web search: {e}")
#             raise RAGException(f"RAG query with web search failed: {e}")
# ```
#
# Update your main function to use the new web search capability:
# ```
# async def main():
#     use_hyde = False  # Set this to True when you want to enable HyDE
#     use_web_search = True  # Set this to False if you don't want to use web search
#
#     try:
#         if use_hyde:
#             rag_system = AsyncHyDERAGSystem('rag_database.db')
#             logger.info("Using Async HyDE RAG System")
#         else:
#             rag_system = AsyncStandardRAGSystem('rag_database.db')
#             logger.info("Using Async Standard RAG System")
#
#         await rag_system.init_db()
#
#         # Add sample documents
#         sample_docs = [
#             ("Paris", "Paris is the capital of France and is known for the Eiffel Tower.", "Local Database"),
#             ("London", "London is the capital of the United Kingdom and home to Big Ben.", "Local Database"),
#             ("Tokyo", "Tokyo is the capital of Japan and is famous for its bustling city life.", "Local Database")
#         ]
#
#         await rag_system.add_documents(sample_docs)
#
#         query = "What is the capital of France?"
#         result = await rag_system.rag_query_with_web_search(query, mock_llm, use_web_search=use_web_search)
#         print(f"Query: {query}")
#         print(f"Result: {result}")
#
#     except RAGException as e:
#         logger.error(f"RAG system error: {e}")
#         print(f"An error occurred: {e}")
#     except Exception as e:
#         logger.error(f"Unexpected error: {e}")
#         print(f"An unexpected error occurred: {e}")
# ```
#
#
# This implementation does the following:
#
# It adds a web_search method that uses the DuckDuckGo API to perform web searches.
# It adds an add_web_results_to_db method that adds the web search results to your existing database.
# It modifies the rag_query method (now called rag_query_with_web_search) to optionally perform a web search before retrieving relevant documents.
#
# When use_web_search is set to True, the system will:
#
# Perform a web search for the given query.
# Add the web search results to the database.
# Retrieve relevant documents (which now may include the newly added web search results).
# Use these documents to generate a response.
#
# This approach allows your RAG system to combine information from your existing database with fresh information from the web, potentially providing more up-to-date and comprehensive answers.
# Remember to handle rate limiting and respect the terms of service of the web search API you choose to use. Also, be aware that adding web search results to your database will increase its size over time, so you may need to implement a strategy to manage this growth (e.g., removing old web search results periodically).


