# database_retrievers.py
"""
Database-specific retrievers for the RAG service.

This module provides specialized retrieval strategies for different data sources,
including media database, notes, prompts, and character cards.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime, timedelta

from loguru import logger
import numpy as np

from .types import Document, DataSource


@dataclass
class RetrievalConfig:
    """Configuration for database retrieval."""
    max_results: int = 20
    min_score: float = 0.0
    use_fts: bool = True
    use_vector: bool = True
    include_metadata: bool = True
    date_filter: Optional[Tuple[datetime, datetime]] = None
    tags_filter: Optional[List[str]] = None
    source_filter: Optional[List[str]] = None


class BaseRetriever(ABC):
    """Base class for database-specific retrievers."""
    
    def __init__(
        self,
        db_path: str,
        config: Optional[RetrievalConfig] = None
    ):
        """
        Initialize retriever.
        
        Args:
            db_path: Path to database
            config: Retrieval configuration
        
        Raises:
            ValueError: If db_path is invalid or contains path traversal
        """
        # Validate and sanitize the database path
        self.db_path = self._validate_path(db_path)
        self.config = config or RetrievalConfig()
    
    def _validate_path(self, path: str) -> str:
        """
        Validate and sanitize a file path to prevent path traversal attacks.
        
        Args:
            path: The path to validate
            
        Returns:
            Sanitized absolute path
            
        Raises:
            ValueError: If path is invalid or contains traversal attempts
        """
        try:
            # Convert to Path object for safe handling
            path_obj = Path(path)
            
            # Resolve to absolute path (follows symlinks and resolves ..)
            abs_path = path_obj.resolve()
            
            # Check if path contains suspicious patterns
            path_str = str(abs_path)
            suspicious_patterns = [
                '../',  # Parent directory traversal
                '..\\',  # Windows parent directory traversal
                '/etc/',  # System configuration
                '/proc/',  # Process information
                '/sys/',  # System information
                '\\System32\\',  # Windows system directory
                '\\Windows\\',  # Windows directory
            ]
            
            for pattern in suspicious_patterns:
                if pattern in path_str:
                    logger.warning(f"Suspicious path pattern detected: {pattern} in {path_str}")
                    raise ValueError(f"Invalid path: contains suspicious pattern '{pattern}'")
            
            # Ensure the path is within allowed directories (configurable)
            # For now, just ensure it's not trying to access system directories
            if abs_path.parts[0] == '/' and len(abs_path.parts) > 1:
                # Unix-like systems
                restricted_dirs = ['etc', 'proc', 'sys', 'dev', 'boot', 'root']
                if abs_path.parts[1] in restricted_dirs:
                    raise ValueError(f"Access to /{abs_path.parts[1]}/ directory is not allowed")
            
            # Check if parent directory exists (database might not exist yet)
            parent_dir = abs_path.parent
            if not parent_dir.exists():
                logger.warning(f"Parent directory does not exist: {parent_dir}")
                # Optionally create parent directory with restricted permissions
                # parent_dir.mkdir(parents=True, mode=0o700)
            
            return str(abs_path)
            
        except Exception as e:
            logger.error(f"Path validation error for '{path}': {e}")
            raise ValueError(f"Invalid database path: {e}")
    
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        **kwargs
    ) -> List[Document]:
        """Retrieve documents from database."""
        pass
    
    @abstractmethod
    async def get_metadata(self, doc_id: str) -> Dict[str, Any]:
        """Get metadata for a document."""
        pass
    
    def _execute_query(
        self,
        query: str,
        params: Tuple = ()
    ) -> List[Tuple]:
        """Execute SQL query and return results."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, params)
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Database query error: {e}")
            return []


class MediaDBRetriever(BaseRetriever):
    """Retriever for Media_DB (main content database)."""
    
    async def retrieve(
        self,
        query: str,
        media_type: Optional[str] = None,
        **kwargs
    ) -> List[Document]:
        """
        Retrieve from media database using FTS5.
        
        Args:
            query: Search query
            media_type: Optional media type filter
            
        Returns:
            List of retrieved documents
        """
        documents = []
        
        # Build FTS query
        fts_query = self._build_fts_query(query)
        
        # Build SQL with filters
        sql = """
            SELECT 
                m.id,
                m.title,
                m.content,
                m.media_type,
                m.url,
                m.created_at,
                m.transcription_model,
                bm25(media_fts) as rank
            FROM media_fts
            JOIN media m ON media_fts.rowid = m.id
            WHERE media_fts MATCH ?
        """
        
        params = [fts_query]
        
        # Add media type filter
        if media_type:
            sql += " AND m.media_type = ?"
            params.append(media_type)
        
        # Add date filter
        if self.config.date_filter:
            start_date, end_date = self.config.date_filter
            sql += " AND m.created_at BETWEEN ? AND ?"
            params.extend([start_date.isoformat(), end_date.isoformat()])
        
        # Add ordering and limit
        sql += " ORDER BY rank DESC LIMIT ?"
        params.append(self.config.max_results)
        
        # Execute query
        results = self._execute_query(sql, tuple(params))
        
        # Convert to documents
        for row in results:
            doc = Document(
                id=str(row["id"]),
                content=row["content"],
                metadata={
                    "title": row["title"],
                    "media_type": row["media_type"],
                    "url": row["url"],
                    "created_at": row["created_at"],
                    "transcription_model": row["transcription_model"],
                    "source": "media_db"
                },
                score=float(row["rank"]) if row["rank"] else 0.0
            )
            documents.append(doc)
        
        logger.debug(f"Retrieved {len(documents)} documents from Media_DB")
        
        return documents
    
    async def retrieve_with_keywords(
        self,
        query: str,
        keywords: List[str]
    ) -> List[Document]:
        """Retrieve with additional keyword filtering."""
        # Get base results
        documents = await self.retrieve(query)
        
        # Filter by keywords
        if keywords:
            filtered_docs = []
            for doc in documents:
                content_lower = doc.content.lower()
                if any(keyword.lower() in content_lower for keyword in keywords):
                    filtered_docs.append(doc)
            documents = filtered_docs
        
        return documents
    
    async def get_metadata(self, doc_id: str) -> Dict[str, Any]:
        """Get full metadata for a media item."""
        sql = """
            SELECT 
                m.*,
                GROUP_CONCAT(t.name) as tags,
                COUNT(DISTINCT ma.id) as analysis_count
            FROM media m
            LEFT JOIN media_tags mt ON m.id = mt.media_id
            LEFT JOIN tags t ON mt.tag_id = t.id
            LEFT JOIN media_analysis ma ON m.id = ma.media_id
            WHERE m.id = ?
            GROUP BY m.id
        """
        
        results = self._execute_query(sql, (doc_id,))
        
        if results:
            row = results[0]
            return {
                "id": row["id"],
                "title": row["title"],
                "media_type": row["media_type"],
                "url": row["url"],
                "tags": row["tags"].split(",") if row["tags"] else [],
                "analysis_count": row["analysis_count"],
                "created_at": row["created_at"]
            }
        
        return {}
    
    def _build_fts_query(self, query: str) -> str:
        """Build FTS5 query with proper escaping."""
        # Basic tokenization and escaping
        terms = query.split()
        
        # Quote phrases
        if len(terms) > 1:
            # Use phrase search for multi-word queries
            return f'"{query}"'
        else:
            # Single term with prefix matching
            return f'{query}*'


class NotesDBRetriever(BaseRetriever):
    """Retriever for notes database."""
    
    async def retrieve(
        self,
        query: str,
        notebook_id: Optional[int] = None,
        **kwargs
    ) -> List[Document]:
        """
        Retrieve from notes database.
        
        Args:
            query: Search query
            notebook_id: Optional notebook filter
            
        Returns:
            List of retrieved documents
        """
        documents = []
        
        # Build SQL query
        sql = """
            SELECT 
                n.id,
                n.title,
                n.content,
                n.notebook_id,
                n.created_at,
                n.updated_at,
                nb.name as notebook_name
            FROM notes n
            LEFT JOIN notebooks nb ON n.notebook_id = nb.id
            WHERE (n.title LIKE ? OR n.content LIKE ?)
        """
        
        params = [f"%{query}%", f"%{query}%"]
        
        # Add notebook filter
        if notebook_id:
            sql += " AND n.notebook_id = ?"
            params.append(notebook_id)
        
        # Add tag filter
        if self.config.tags_filter:
            sql += """ 
                AND n.id IN (
                    SELECT note_id FROM note_tags 
                    WHERE tag_id IN (
                        SELECT id FROM tags WHERE name IN ({})
                    )
                )
            """.format(",".join("?" * len(self.config.tags_filter)))
            params.extend(self.config.tags_filter)
        
        # Order and limit
        sql += " ORDER BY n.updated_at DESC LIMIT ?"
        params.append(self.config.max_results)
        
        # Execute query
        results = self._execute_query(sql, tuple(params))
        
        # Convert to documents
        for row in results:
            # Calculate simple relevance score
            title_match = query.lower() in row["title"].lower()
            content_match = query.lower() in row["content"].lower()
            score = (1.0 if title_match else 0.0) + (0.5 if content_match else 0.0)
            
            doc = Document(
                id=f"note_{row['id']}",
                content=f"# {row['title']}\n\n{row['content']}",
                metadata={
                    "title": row["title"],
                    "notebook": row["notebook_name"],
                    "notebook_id": row["notebook_id"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "source": "notes_db"
                },
                score=score
            )
            documents.append(doc)
        
        # Sort by score
        documents.sort(key=lambda x: x.score, reverse=True)
        
        logger.debug(f"Retrieved {len(documents)} documents from Notes_DB")
        
        return documents
    
    async def get_metadata(self, doc_id: str) -> Dict[str, Any]:
        """Get metadata for a note."""
        # Extract numeric ID
        note_id = doc_id.replace("note_", "")
        
        sql = """
            SELECT 
                n.*,
                nb.name as notebook_name,
                GROUP_CONCAT(t.name) as tags
            FROM notes n
            LEFT JOIN notebooks nb ON n.notebook_id = nb.id
            LEFT JOIN note_tags nt ON n.id = nt.note_id
            LEFT JOIN tags t ON nt.tag_id = t.id
            WHERE n.id = ?
            GROUP BY n.id
        """
        
        results = self._execute_query(sql, (note_id,))
        
        if results:
            row = results[0]
            return {
                "id": row["id"],
                "title": row["title"],
                "notebook": row["notebook_name"],
                "tags": row["tags"].split(",") if row["tags"] else [],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]
            }
        
        return {}


class PromptsDBRetriever(BaseRetriever):
    """Retriever for prompts database."""
    
    async def retrieve(
        self,
        query: str,
        category: Optional[str] = None,
        **kwargs
    ) -> List[Document]:
        """
        Retrieve from prompts database.
        
        Args:
            query: Search query
            category: Optional category filter
            
        Returns:
            List of retrieved documents
        """
        documents = []
        
        # Build SQL query
        sql = """
            SELECT 
                p.id,
                p.name,
                p.prompt,
                p.category,
                p.rating,
                p.times_used,
                p.created_at
            FROM prompts p
            WHERE (p.name LIKE ? OR p.prompt LIKE ?)
        """
        
        params = [f"%{query}%", f"%{query}%"]
        
        # Add category filter
        if category:
            sql += " AND p.category = ?"
            params.append(category)
        
        # Order by relevance and usage
        sql += " ORDER BY p.rating DESC, p.times_used DESC LIMIT ?"
        params.append(self.config.max_results)
        
        # Execute query
        results = self._execute_query(sql, tuple(params))
        
        # Convert to documents
        for row in results:
            # Calculate relevance score
            name_match = query.lower() in row["name"].lower()
            prompt_match = query.lower() in row["prompt"].lower()
            base_score = (1.0 if name_match else 0.0) + (0.5 if prompt_match else 0.0)
            
            # Boost by rating and usage
            rating_boost = (row["rating"] or 0) / 5.0 * 0.3
            usage_boost = min(row["times_used"] / 100.0, 1.0) * 0.2
            
            score = base_score + rating_boost + usage_boost
            
            doc = Document(
                id=f"prompt_{row['id']}",
                content=f"**{row['name']}**\n\n{row['prompt']}",
                metadata={
                    "name": row["name"],
                    "category": row["category"],
                    "rating": row["rating"],
                    "times_used": row["times_used"],
                    "created_at": row["created_at"],
                    "source": "prompts_db"
                },
                score=score
            )
            documents.append(doc)
        
        # Sort by score
        documents.sort(key=lambda x: x.score, reverse=True)
        
        logger.debug(f"Retrieved {len(documents)} documents from Prompts_DB")
        
        return documents
    
    async def get_metadata(self, doc_id: str) -> Dict[str, Any]:
        """Get metadata for a prompt."""
        prompt_id = doc_id.replace("prompt_", "")
        
        sql = "SELECT * FROM prompts WHERE id = ?"
        results = self._execute_query(sql, (prompt_id,))
        
        if results:
            row = results[0]
            return dict(row)
        
        return {}


class CharacterCardsRetriever(BaseRetriever):
    """Retriever for character cards and chats."""
    
    async def retrieve(
        self,
        query: str,
        include_chats: bool = True,
        **kwargs
    ) -> List[Document]:
        """
        Retrieve from character cards and chats.
        
        Args:
            query: Search query
            include_chats: Whether to include chat messages
            
        Returns:
            List of retrieved documents
        """
        documents = []
        
        # Search character cards
        card_sql = """
            SELECT 
                cc.id,
                cc.name,
                cc.description,
                cc.personality,
                cc.first_message,
                cc.scenario,
                cc.creator,
                cc.version
            FROM character_cards cc
            WHERE cc.name LIKE ? 
                OR cc.description LIKE ?
                OR cc.personality LIKE ?
                OR cc.scenario LIKE ?
            LIMIT ?
        """
        
        params = [f"%{query}%"] * 4 + [self.config.max_results // 2]
        
        card_results = self._execute_query(card_sql, params)
        
        # Convert character cards to documents
        for row in card_results:
            content = f"""# {row['name']}
            
**Description:** {row['description']}

**Personality:** {row['personality']}

**Scenario:** {row['scenario']}

**First Message:** {row['first_message']}"""
            
            # Calculate relevance
            matches = sum([
                query.lower() in (row[field] or "").lower()
                for field in ["name", "description", "personality", "scenario"]
            ])
            score = matches / 4.0
            
            doc = Document(
                id=f"character_{row['id']}",
                content=content,
                metadata={
                    "name": row["name"],
                    "creator": row["creator"],
                    "version": row["version"],
                    "type": "character_card",
                    "source": "character_cards"
                },
                score=score
            )
            documents.append(doc)
        
        # Search chat messages if requested
        if include_chats:
            chat_sql = """
                SELECT 
                    cm.id,
                    cm.message,
                    cm.sender,
                    cm.timestamp,
                    cs.character_id,
                    cc.name as character_name
                FROM chat_messages cm
                JOIN chat_sessions cs ON cm.session_id = cs.id
                LEFT JOIN character_cards cc ON cs.character_id = cc.id
                WHERE cm.message LIKE ?
                ORDER BY cm.timestamp DESC
                LIMIT ?
            """
            
            chat_params = [f"%{query}%", self.config.max_results // 2]
            chat_results = self._execute_query(chat_sql, chat_params)
            
            # Convert chat messages to documents
            for row in chat_results:
                content = f"[{row['sender']}]: {row['message']}"
                
                doc = Document(
                    id=f"chat_{row['id']}",
                    content=content,
                    metadata={
                        "sender": row["sender"],
                        "timestamp": row["timestamp"],
                        "character": row["character_name"],
                        "type": "chat_message",
                        "source": "character_chats"
                    },
                    score=0.5  # Lower base score for chat messages
                )
                documents.append(doc)
        
        # Sort by score
        documents.sort(key=lambda x: x.score, reverse=True)
        
        logger.debug(f"Retrieved {len(documents)} documents from Character Cards")
        
        return documents
    
    async def get_metadata(self, doc_id: str) -> Dict[str, Any]:
        """Get metadata for a character card or chat."""
        if doc_id.startswith("character_"):
            card_id = doc_id.replace("character_", "")
            sql = "SELECT * FROM character_cards WHERE id = ?"
            results = self._execute_query(sql, (card_id,))
        elif doc_id.startswith("chat_"):
            chat_id = doc_id.replace("chat_", "")
            sql = """
                SELECT cm.*, cs.character_id, cc.name as character_name
                FROM chat_messages cm
                JOIN chat_sessions cs ON cm.session_id = cs.id
                LEFT JOIN character_cards cc ON cs.character_id = cc.id
                WHERE cm.id = ?
            """
            results = self._execute_query(sql, (chat_id,))
        else:
            return {}
        
        if results:
            return dict(results[0])
        
        return {}


class MultiDatabaseRetriever:
    """Orchestrates retrieval across multiple databases."""
    
    def __init__(self, db_paths: Dict[str, str]):
        """
        Initialize multi-database retriever.
        
        Args:
            db_paths: Mapping of database names to paths
        """
        self.retrievers = {}
        
        # Initialize retrievers for available databases
        if "media_db" in db_paths:
            self.retrievers[DataSource.MEDIA_DB] = MediaDBRetriever(
                db_paths["media_db"]
            )
        
        if "notes_db" in db_paths:
            self.retrievers[DataSource.NOTES] = NotesDBRetriever(
                db_paths["notes_db"]
            )
        
        if "prompts_db" in db_paths:
            self.retrievers[DataSource.PROMPTS] = PromptsDBRetriever(
                db_paths["prompts_db"]
            )
        
        if "character_cards_db" in db_paths:
            self.retrievers[DataSource.CHARACTER_CARDS] = CharacterCardsRetriever(
                db_paths["character_cards_db"]
            )
    
    async def retrieve(
        self,
        query: str,
        sources: Optional[List[DataSource]] = None,
        config: Optional[RetrievalConfig] = None
    ) -> List[Document]:
        """
        Retrieve from multiple databases.
        
        Args:
            query: Search query
            sources: List of sources to search (None = all)
            config: Retrieval configuration
            
        Returns:
            Combined list of documents from all sources
        """
        sources = sources or list(self.retrievers.keys())
        
        # Configure retrievers
        if config:
            for retriever in self.retrievers.values():
                retriever.config = config
        
        # Parallel retrieval from all sources
        tasks = []
        for source in sources:
            if source in self.retrievers:
                task = self.retrievers[source].retrieve(query)
                tasks.append(task)
        
        # Gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine documents
        all_documents = []
        for result in results:
            if isinstance(result, list):
                all_documents.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Retrieval error: {result}")
        
        # Sort by score
        all_documents.sort(key=lambda x: x.score, reverse=True)
        
        # Apply global limit
        if config and config.max_results:
            all_documents = all_documents[:config.max_results]
        
        logger.info(
            f"Retrieved {len(all_documents)} documents from "
            f"{len(sources)} sources"
        )
        
        return all_documents
    
    async def retrieve_with_fusion(
        self,
        query: str,
        sources: Optional[List[DataSource]] = None,
        fusion_method: str = "rrf"
    ) -> List[Document]:
        """
        Retrieve with result fusion across databases.
        
        Args:
            query: Search query
            sources: List of sources to search
            fusion_method: Fusion method (rrf, weighted, max)
            
        Returns:
            Fused list of documents
        """
        # Get results from each source
        source_results = {}
        sources = sources or list(self.retrievers.keys())
        
        for source in sources:
            if source in self.retrievers:
                docs = await self.retrievers[source].retrieve(query)
                source_results[source] = docs
        
        # Apply fusion
        if fusion_method == "rrf":
            return self._reciprocal_rank_fusion(source_results)
        elif fusion_method == "weighted":
            return self._weighted_fusion(source_results)
        elif fusion_method == "max":
            return self._max_fusion(source_results)
        else:
            # Simple concatenation
            all_docs = []
            for docs in source_results.values():
                all_docs.extend(docs)
            return sorted(all_docs, key=lambda x: x.score, reverse=True)
    
    def _reciprocal_rank_fusion(
        self,
        source_results: Dict[DataSource, List[Document]],
        k: int = 60
    ) -> List[Document]:
        """Apply Reciprocal Rank Fusion."""
        doc_scores = {}
        
        for source, docs in source_results.items():
            for rank, doc in enumerate(docs, 1):
                if doc.id not in doc_scores:
                    doc_scores[doc.id] = {"doc": doc, "score": 0}
                
                # RRF formula
                doc_scores[doc.id]["score"] += 1.0 / (k + rank)
        
        # Sort by fused score
        fused_docs = [
            Document(
                id=item["doc"].id,
                content=item["doc"].content,
                metadata=item["doc"].metadata,
                score=item["score"]
            )
            for item in sorted(
                doc_scores.values(),
                key=lambda x: x["score"],
                reverse=True
            )
        ]
        
        return fused_docs
    
    def _weighted_fusion(
        self,
        source_results: Dict[DataSource, List[Document]],
        weights: Optional[Dict[DataSource, float]] = None
    ) -> List[Document]:
        """Apply weighted fusion based on source importance."""
        # Default weights
        if not weights:
            weights = {
                DataSource.MEDIA_DB: 1.0,
                DataSource.NOTES: 0.8,
                DataSource.PROMPTS: 0.6,
                DataSource.CHARACTER_CARDS: 0.5
            }
        
        doc_scores = {}
        
        for source, docs in source_results.items():
            weight = weights.get(source, 1.0)
            
            for doc in docs:
                if doc.id not in doc_scores:
                    doc_scores[doc.id] = {"doc": doc, "score": 0}
                
                doc_scores[doc.id]["score"] += doc.score * weight
        
        # Create fused documents
        fused_docs = [
            Document(
                id=item["doc"].id,
                content=item["doc"].content,
                metadata=item["doc"].metadata,
                score=item["score"]
            )
            for item in sorted(
                doc_scores.values(),
                key=lambda x: x["score"],
                reverse=True
            )
        ]
        
        return fused_docs
    
    def _max_fusion(
        self,
        source_results: Dict[DataSource, List[Document]]
    ) -> List[Document]:
        """Take maximum score for each document across sources."""
        doc_scores = {}
        
        for source, docs in source_results.items():
            for doc in docs:
                if doc.id not in doc_scores:
                    doc_scores[doc.id] = doc
                elif doc.score > doc_scores[doc.id].score:
                    doc_scores[doc.id] = doc
        
        return sorted(doc_scores.values(), key=lambda x: x.score, reverse=True)


# Pipeline integration function
async def retrieve_from_databases(context: Any, **kwargs) -> Any:
    """Retrieve documents from configured databases for pipeline."""
    config = context.config.get("database_retrieval", {})
    
    # Get database paths from config
    db_paths = config.get("db_paths", {})
    if not db_paths:
        logger.warning("No database paths configured")
        return context
    
    # Create multi-database retriever
    retriever = MultiDatabaseRetriever(db_paths)
    
    # Configure retrieval
    retrieval_config = RetrievalConfig(
        max_results=config.get("max_results", 20),
        min_score=config.get("min_score", 0.0),
        use_fts=config.get("use_fts", True),
        include_metadata=config.get("include_metadata", True)
    )
    
    # Get sources to search
    sources = config.get("sources")
    if sources:
        sources = [DataSource[s.upper()] for s in sources]
    
    # Retrieve with fusion if enabled
    if config.get("use_fusion", True):
        documents = await retriever.retrieve_with_fusion(
            query=context.query,
            sources=sources,
            fusion_method=config.get("fusion_method", "rrf")
        )
    else:
        documents = await retriever.retrieve(
            query=context.query,
            sources=sources,
            config=retrieval_config
        )
    
    # Update context
    context.documents = documents
    context.metadata["database_retrieval"] = {
        "sources_searched": [s.value for s in (sources or retriever.retrievers.keys())],
        "documents_retrieved": len(documents),
        "fusion_used": config.get("use_fusion", True)
    }
    
    logger.info(f"Retrieved {len(documents)} documents from databases")
    
    return context