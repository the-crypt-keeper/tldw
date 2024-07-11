import os
from typing import List, Tuple, Callable, Optional
from contextlib import contextmanager
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RAGException(Exception):
    """Custom exception class for RAG-related errors"""
    pass


class BaseRAGSystem:
    def __init__(self, db_path: str, model_name: Optional[str] = None):
        """
        Initialize the RAG system.

        :param db_path: Path to the SQLite database
        :param model_name: Name of the SentenceTransformer model to use
        """
        self.db_path = db_path
        self.model_name = model_name or os.getenv('DEFAULT_MODEL_NAME', 'all-MiniLM-L6-v2')
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Initialized SentenceTransformer with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformer: {e}")
            raise RAGException(f"Model initialization failed: {e}")

        self.init_db()

    @contextmanager
    def get_db_connection(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def init_db(self):
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    title TEXT,
                    content TEXT,
                    embedding BLOB
                )
                ''')
                conn.commit()
            logger.info("Initialized database schema")
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize database schema: {e}")
            raise RAGException(f"Database schema initialization failed: {e}")

    def add_documents(self, documents: List[Tuple[str, str]]):
        try:
            embeddings = self.model.encode([content for _, content in documents])
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.executemany(
                    'INSERT INTO documents (title, content, embedding) VALUES (?, ?, ?)',
                    [(title, content, embedding.tobytes()) for (title, content), embedding in zip(documents, embeddings)]
                )
                conn.commit()
            logger.info(f"Added {len(documents)} documents in batch")
        except Exception as e:
            logger.error(f"Failed to add documents in batch: {e}")
            raise RAGException(f"Batch document addition failed: {e}")

    def get_documents(self) -> List[Tuple[int, str, str, np.ndarray]]:
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT id, title, content, embedding FROM documents')
                documents = [(id, title, content, np.frombuffer(embedding, dtype=np.float32))
                             for id, title, content, embedding in cursor.fetchall()]
            logger.info(f"Retrieved {len(documents)} documents")
            return documents
        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve documents: {e}")
            raise RAGException(f"Document retrieval failed: {e}")

    def close(self):
        try:
            self.conn.close()
            logger.info("Closed database connection")
        except sqlite3.Error as e:
            logger.error(f"Error closing database connection: {e}")


class StandardRAGSystem(BaseRAGSystem):
    def get_relevant_documents(self, query: str, top_k: int = 3) -> List[Tuple[int, str, str, float]]:
        try:
            query_embedding = self.model.encode([query])[0]
            documents = self.get_documents()
            similarities = [
                (id, title, content, cosine_similarity([query_embedding], [doc_embedding])[0][0])
                for id, title, content, doc_embedding in documents
            ]
            similarities.sort(key=lambda x: x[3], reverse=True)
            logger.info(f"Retrieved top {top_k} relevant documents for query")
            return similarities[:top_k]
        except Exception as e:
            logger.error(f"Error in getting relevant documents: {e}")
            raise RAGException(f"Retrieval of relevant documents failed: {e}")

    def rag_query(self, query: str, llm_function: Callable[[str], str], top_k: int = 3) -> str:
        try:
            relevant_docs = self.get_relevant_documents(query, top_k)
            context = "\n\n".join([f"Title: {title}\nContent: {content}" for _, title, content, _ in relevant_docs])

            llm_prompt = f"Based on the following context, please answer the query:\n\nContext:\n{context}\n\nQuery: {query}"

            response = llm_function(llm_prompt)
            logger.info("Generated response for query")
            return response
        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            raise RAGException(f"RAG query failed: {e}")


class HyDERAGSystem(BaseRAGSystem):
    def generate_hypothetical_document(self, query: str, llm_function: Callable[[str], str]) -> str:
        try:
            prompt = f"Given the question '{query}', write a short paragraph that would answer this question. Do not include the question itself in your response."
            hypothetical_doc = llm_function(prompt)
            logger.info("Generated hypothetical document")
            return hypothetical_doc
        except Exception as e:
            logger.error(f"Error generating hypothetical document: {e}")
            raise RAGException(f"Hypothetical document generation failed: {e}")

    def get_relevant_documents(self, query: str, llm_function: Callable[[str], str], top_k: int = 3) -> List[
        Tuple[int, str, str, float]]:
        try:
            hypothetical_doc = self.generate_hypothetical_document(query, llm_function)
            hyde_embedding = self.model.encode([hypothetical_doc])[0]

            documents = self.get_documents()
            similarities = [
                (id, title, content, cosine_similarity([hyde_embedding], [doc_embedding])[0][0])
                for id, title, content, doc_embedding in documents
            ]
            similarities.sort(key=lambda x: x[3], reverse=True)
            logger.info(f"Retrieved top {top_k} relevant documents using HyDE")
            return similarities[:top_k]
        except Exception as e:
            logger.error(f"Error in getting relevant documents with HyDE: {e}")
            raise RAGException(f"HyDE retrieval of relevant documents failed: {e}")

    def rag_query(self, query: str, llm_function: Callable[[str], str], top_k: int = 3) -> str:
        try:
            relevant_docs = self.get_relevant_documents(query, llm_function, top_k)
            context = "\n\n".join([f"Title: {title}\nContent: {content}" for _, title, content, _ in relevant_docs])

            llm_prompt = f"Based on the following context, please answer the query:\n\nContext:\n{context}\n\nQuery: {query}"

            response = llm_function(llm_prompt)
            logger.info("Generated response for query using HyDE")
            return response
        except Exception as e:
            logger.error(f"Error in HyDE RAG query: {e}")
            raise RAGException(f"HyDE RAG query failed: {e}")


# Example usage with error handling
def mock_llm(prompt: str) -> str:
    if "write a short paragraph" in prompt:
        return "Paris, the capital of France, is renowned for its iconic Eiffel Tower and rich cultural heritage."
    else:
        return f"This is a mock LLM response for the prompt: {prompt}"


def main():
    use_hyde = False  # Set this to True when you want to enable HyDE

    try:
        if use_hyde:
            rag_system = HyDERAGSystem('rag_database.db')
            logger.info("Using HyDE RAG System")
        else:
            rag_system = StandardRAGSystem('rag_database.db')
            logger.info("Using Standard RAG System")

        # Add sample documents in batch
        sample_docs = [
            ("Paris", "Paris is the capital of France and is known for the Eiffel Tower."),
            ("London", "London is the capital of the United Kingdom and home to Big Ben."),
            ("Tokyo", "Tokyo is the capital of Japan and is famous for its bustling city life.")
        ]

        for title, content in sample_docs:
            rag_system.add_document(title, content)

        query = "What is the capital of France?"
        result = rag_system.rag_query(query, mock_llm)
        print(f"Query: {query}")
        print(f"Result: {result}")

    except RAGException as e:
        logger.error(f"RAG system error: {e}")
        print(f"An error occurred: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"An unexpected error occurred: {e}")
    finally:
        if 'rag_system' in locals():
            rag_system.close()


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
#     async def get_documents(self) -> List[Tuple[int, str, str, np.ndarray]]:
#         try:
#             async with aiosqlite.connect(self.db_path) as db:
#                 async with db.execute('SELECT id, title, content, embedding FROM documents') as cursor:
#                     documents = [
#                         (id, title, content, np.frombuffer(embedding, dtype=np.float32))
#                         async for id, title, content, embedding in cursor
#                     ]
#             logger.info(f"Retrieved {len(documents)} documents")
#             return documents
#         except aiosqlite.Error as e:
#             logger.error(f"Failed to retrieve documents: {e}")
#             raise RAGException(f"Document retrieval failed: {e}")
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