# session_manager.py
# Description: Session management with Redis caching and automatic cleanup
#
# Imports
import json
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import asyncio
#
# 3rd-party imports
import redis
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger
#
# Local imports
from tldw_Server_API.app.core.AuthNZ.settings import Settings, get_settings
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    SessionError,
    InvalidSessionError,
    SessionRevokedException,
    DatabaseError
)

#######################################################################################################################
#
# Session Manager Class

class SessionManager:
    """Manages user sessions with database persistence and optional Redis caching"""
    
    def __init__(
        self,
        db_pool: Optional[DatabasePool] = None,
        settings: Optional[Settings] = None
    ):
        """Initialize session manager"""
        self.settings = settings or get_settings()
        self.db_pool = db_pool
        self.redis_client: Optional[redis.Redis] = None
        self.scheduler = AsyncIOScheduler()
        self._initialized = False
        
    async def initialize(self):
        """Initialize session manager and start cleanup scheduler"""
        if self._initialized:
            return
        
        # Get database pool
        if not self.db_pool:
            self.db_pool = await get_db_pool()
        
        # Initialize Redis if configured
        if self.settings.REDIS_URL:
            try:
                self.redis_client = redis.from_url(
                    self.settings.REDIS_URL,
                    decode_responses=True,
                    socket_connect_timeout=1,
                    socket_keepalive=True,
                    socket_keepalive_options={},
                    max_connections=self.settings.REDIS_MAX_CONNECTIONS,
                    health_check_interval=30
                )
                
                # Test connection
                self.redis_client.ping()
                logger.info("Redis connected for session caching")
                
            except (RedisConnectionError, RedisError) as e:
                logger.warning(f"Redis unavailable, using database only: {e}")
                self.redis_client = None
        
        # Schedule session cleanup
        if self.settings.SESSION_CLEANUP_INTERVAL_HOURS > 0:
            self.scheduler.add_job(
                self.cleanup_expired_sessions,
                trigger=IntervalTrigger(hours=self.settings.SESSION_CLEANUP_INTERVAL_HOURS),
                id='session_cleanup',
                replace_existing=True,
                max_instances=1
            )
            self.scheduler.start()
            logger.info(
                f"Session cleanup scheduled every {self.settings.SESSION_CLEANUP_INTERVAL_HOURS} hours"
            )
        
        self._initialized = True
        logger.info("SessionManager initialized")
    
    def hash_token(self, token: str) -> str:
        """Create SHA256 hash of token for storage"""
        return hashlib.sha256(token.encode()).hexdigest()
    
    async def create_session(
        self,
        user_id: int,
        access_token: str,
        refresh_token: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        device_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new session for a user
        
        Args:
            user_id: User's database ID
            access_token: JWT access token
            refresh_token: JWT refresh token
            ip_address: Client IP address
            user_agent: Client user agent string
            device_id: Optional device identifier
            
        Returns:
            Session information dictionary
        """
        if not self._initialized:
            await self.initialize()
        
        access_hash = self.hash_token(access_token)
        refresh_hash = self.hash_token(refresh_token)
        expires_at = datetime.utcnow() + timedelta(
            minutes=self.settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
        
        session_id = None
        
        try:
            async with self.db_pool.transaction() as conn:
                if hasattr(conn, 'fetchval'):
                    # PostgreSQL
                    session_id = await conn.fetchval(
                        """
                        INSERT INTO sessions (
                            user_id, token_hash, refresh_token_hash,
                            expires_at, ip_address, user_agent, device_id
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        RETURNING id
                        """,
                        user_id, access_hash, refresh_hash, expires_at,
                        ip_address, user_agent, device_id
                    )
                else:
                    # SQLite
                    cursor = await conn.execute(
                        """
                        INSERT INTO sessions (
                            user_id, token_hash, refresh_token_hash,
                            expires_at, ip_address, user_agent
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (user_id, access_hash, refresh_hash, 
                         expires_at.isoformat(), ip_address, user_agent)
                    )
                    session_id = cursor.lastrowid
                    await conn.commit()
                
                # Cache in Redis if available
                if self.redis_client:
                    await self._cache_session(
                        access_hash, 
                        user_id, 
                        session_id, 
                        expires_at
                    )
                
                logger.info(f"Created session {session_id} for user {user_id}")
                
                return {
                    "session_id": session_id,
                    "user_id": user_id,
                    "expires_at": expires_at.isoformat(),
                    "access_token": access_token,
                    "refresh_token": refresh_token
                }
                
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise SessionError(f"Failed to create session: {e}")
    
    async def validate_session(self, access_token: str) -> Optional[Dict[str, Any]]:
        """
        Validate a session by access token
        
        Args:
            access_token: JWT access token
            
        Returns:
            Session data if valid, None otherwise
        """
        if not self._initialized:
            await self.initialize()
        
        token_hash = self.hash_token(access_token)
        
        # Try Redis cache first
        if self.redis_client:
            session_data = await self._get_cached_session(token_hash)
            if session_data:
                return session_data
        
        # Database lookup
        try:
            async with self.db_pool.acquire() as conn:
                if hasattr(conn, 'fetchrow'):
                    # PostgreSQL
                    row = await conn.fetchrow(
                        """
                        SELECT s.id, s.user_id, s.expires_at, s.is_active,
                               s.revoked_at, u.username, u.role, u.is_active as user_active
                        FROM sessions s
                        JOIN users u ON s.user_id = u.id
                        WHERE s.token_hash = $1
                        AND s.is_active = TRUE
                        AND s.expires_at > CURRENT_TIMESTAMP
                        """,
                        token_hash
                    )
                    if row:
                        session_data = dict(row)
                else:
                    # SQLite
                    cursor = await conn.execute(
                        """
                        SELECT s.id, s.user_id, s.expires_at, s.is_active,
                               s.revoked_at, u.username, u.role, u.is_active as user_active
                        FROM sessions s
                        JOIN users u ON s.user_id = u.id
                        WHERE s.token_hash = ?
                        AND s.is_active = 1
                        AND datetime(s.expires_at) > datetime('now')
                        """,
                        (token_hash,)
                    )
                    row = await cursor.fetchone()
                    if row:
                        session_data = {
                            "id": row[0],
                            "user_id": row[1],
                            "expires_at": row[2],
                            "is_active": row[3],
                            "revoked_at": row[4],
                            "username": row[5],
                            "role": row[6],
                            "user_active": row[7]
                        }
                    else:
                        session_data = None
                
                if session_data:
                    # Check if user is still active
                    if not session_data.get('user_active'):
                        logger.warning(f"Session valid but user {session_data['user_id']} is inactive")
                        return None
                    
                    # Check if session was revoked
                    if session_data.get('revoked_at'):
                        logger.warning(f"Session {session_data['id']} was revoked")
                        raise SessionRevokedException()
                    
                    # Update last activity
                    await self._update_last_activity(session_data['id'], conn)
                    
                    # Cache the session
                    if self.redis_client:
                        expires_at = session_data.get('expires_at')
                        if isinstance(expires_at, str):
                            expires_at = datetime.fromisoformat(expires_at)
                        await self._cache_session(
                            token_hash,
                            session_data['user_id'],
                            session_data['id'],
                            expires_at
                        )
                    
                    return session_data
                
                return None
                
        except SessionRevokedException:
            raise
        except Exception as e:
            logger.error(f"Error validating session: {e}")
            return None
    
    async def revoke_session(
        self,
        session_id: int,
        revoked_by: Optional[int] = None,
        reason: Optional[str] = None
    ):
        """Revoke a specific session"""
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self.db_pool.transaction() as conn:
                if hasattr(conn, 'fetchrow'):
                    # PostgreSQL
                    await conn.execute(
                        """
                        UPDATE sessions
                        SET is_active = FALSE,
                            revoked_at = CURRENT_TIMESTAMP,
                            revoked_by = $2,
                            revoke_reason = $3
                        WHERE id = $1
                        """,
                        session_id, revoked_by, reason
                    )
                else:
                    # SQLite
                    await conn.execute(
                        """
                        UPDATE sessions
                        SET is_active = 0,
                            revoked_at = datetime('now')
                        WHERE id = ?
                        """,
                        (session_id,)
                    )
                    await conn.commit()
                
                # Clear from cache
                if self.redis_client:
                    await self._clear_session_cache(session_id)
                
                logger.info(f"Revoked session {session_id}")
                
        except Exception as e:
            logger.error(f"Failed to revoke session: {e}")
            raise SessionError(f"Failed to revoke session: {e}")
    
    async def revoke_all_user_sessions(
        self,
        user_id: int,
        except_session_id: Optional[int] = None
    ):
        """Revoke all sessions for a user, optionally except one"""
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self.db_pool.transaction() as conn:
                if hasattr(conn, 'fetchrow'):
                    # PostgreSQL
                    if except_session_id:
                        await conn.execute(
                            """
                            UPDATE sessions
                            SET is_active = FALSE,
                                revoked_at = CURRENT_TIMESTAMP
                            WHERE user_id = $1 AND id != $2
                            """,
                            user_id, except_session_id
                        )
                    else:
                        await conn.execute(
                            """
                            UPDATE sessions
                            SET is_active = FALSE,
                                revoked_at = CURRENT_TIMESTAMP
                            WHERE user_id = $1
                            """,
                            user_id
                        )
                else:
                    # SQLite
                    if except_session_id:
                        await conn.execute(
                            """
                            UPDATE sessions
                            SET is_active = 0,
                                revoked_at = datetime('now')
                            WHERE user_id = ? AND id != ?
                            """,
                            (user_id, except_session_id)
                        )
                    else:
                        await conn.execute(
                            """
                            UPDATE sessions
                            SET is_active = 0,
                                revoked_at = datetime('now')
                            WHERE user_id = ?
                            """,
                            (user_id,)
                        )
                    await conn.commit()
                
                # Clear from cache
                if self.redis_client:
                    await self._clear_user_sessions_cache(user_id)
                
                logger.info(f"Revoked all sessions for user {user_id}")
                
        except Exception as e:
            logger.error(f"Failed to revoke user sessions: {e}")
            raise SessionError(f"Failed to revoke sessions: {e}")
    
    async def refresh_session(
        self,
        refresh_token: str,
        new_access_token: str,
        new_refresh_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Refresh a session with new tokens"""
        if not self._initialized:
            await self.initialize()
        
        old_refresh_hash = self.hash_token(refresh_token)
        new_access_hash = self.hash_token(new_access_token)
        new_refresh_hash = self.hash_token(new_refresh_token) if new_refresh_token else None
        
        try:
            async with self.db_pool.transaction() as conn:
                # Find session by refresh token
                if hasattr(conn, 'fetchrow'):
                    # PostgreSQL
                    session = await conn.fetchrow(
                        """
                        SELECT id, user_id FROM sessions
                        WHERE refresh_token_hash = $1
                        AND is_active = TRUE
                        """,
                        old_refresh_hash
                    )
                    
                    if not session:
                        raise InvalidSessionError()
                    
                    # Update session with new tokens
                    expires_at = datetime.utcnow() + timedelta(
                        minutes=self.settings.ACCESS_TOKEN_EXPIRE_MINUTES
                    )
                    
                    if new_refresh_hash:
                        await conn.execute(
                            """
                            UPDATE sessions
                            SET token_hash = $2,
                                refresh_token_hash = $3,
                                expires_at = $4,
                                last_activity = CURRENT_TIMESTAMP
                            WHERE id = $1
                            """,
                            session['id'], new_access_hash, new_refresh_hash, expires_at
                        )
                    else:
                        await conn.execute(
                            """
                            UPDATE sessions
                            SET token_hash = $2,
                                expires_at = $3,
                                last_activity = CURRENT_TIMESTAMP
                            WHERE id = $1
                            """,
                            session['id'], new_access_hash, expires_at
                        )
                    
                    session_data = dict(session)
                    
                else:
                    # SQLite
                    cursor = await conn.execute(
                        """
                        SELECT id, user_id FROM sessions
                        WHERE refresh_token_hash = ?
                        AND is_active = 1
                        """,
                        (old_refresh_hash,)
                    )
                    row = await cursor.fetchone()
                    
                    if not row:
                        raise InvalidSessionError()
                    
                    session_data = {"id": row[0], "user_id": row[1]}
                    
                    # Update session
                    expires_at = datetime.utcnow() + timedelta(
                        minutes=self.settings.ACCESS_TOKEN_EXPIRE_MINUTES
                    )
                    
                    if new_refresh_hash:
                        await conn.execute(
                            """
                            UPDATE sessions
                            SET token_hash = ?,
                                refresh_token_hash = ?,
                                expires_at = ?,
                                last_activity = datetime('now')
                            WHERE id = ?
                            """,
                            (new_access_hash, new_refresh_hash, 
                             expires_at.isoformat(), session_data['id'])
                        )
                    else:
                        await conn.execute(
                            """
                            UPDATE sessions
                            SET token_hash = ?,
                                expires_at = ?,
                                last_activity = datetime('now')
                            WHERE id = ?
                            """,
                            (new_access_hash, expires_at.isoformat(), session_data['id'])
                        )
                    await conn.commit()
                
                # Update cache
                if self.redis_client:
                    await self._cache_session(
                        new_access_hash,
                        session_data['user_id'],
                        session_data['id'],
                        expires_at
                    )
                
                logger.info(f"Refreshed session {session_data['id']}")
                
                return {
                    "session_id": session_data['id'],
                    "user_id": session_data['user_id'],
                    "expires_at": expires_at.isoformat()
                }
                
        except InvalidSessionError:
            raise
        except Exception as e:
            logger.error(f"Failed to refresh session: {e}")
            raise SessionError(f"Failed to refresh session: {e}")
    
    async def update_session_tokens(
        self,
        session_id: int,
        access_token: str,
        refresh_token: str
    ):
        """Update session with actual tokens after creation"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Hash tokens for storage
            access_token_hash = hashlib.sha256(access_token.encode()).hexdigest()
            refresh_token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
            
            if self.db_pool.pool:
                # PostgreSQL
                await self.db_pool.execute(
                    """
                    UPDATE sessions
                    SET token_hash = $1,
                        refresh_token_hash = $2
                    WHERE id = $3
                    """,
                    access_token_hash, refresh_token_hash, session_id
                )
            else:
                # SQLite
                await self.db_pool.execute(
                    """
                    UPDATE sessions
                    SET token_hash = ?,
                        refresh_token_hash = ?
                    WHERE id = ?
                    """,
                    access_token_hash, refresh_token_hash, session_id
                )
            
            logger.debug(f"Updated session {session_id} with token hashes")
            
        except Exception as e:
            logger.error(f"Failed to update session tokens: {e}")
            raise SessionError(f"Failed to update session tokens: {e}")
    
    async def is_token_blacklisted(self, token: str) -> bool:
        """
        Check if a token has been blacklisted/revoked
        
        Args:
            token: JWT token to check
            
        Returns:
            True if token is blacklisted, False otherwise
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Hash the token for storage/comparison
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            
            # Check Redis cache first if available
            if self.redis_client:
                try:
                    blacklisted = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.redis_client.get,
                        f"blacklist:{token_hash}"
                    )
                    if blacklisted:
                        return True
                except RedisError:
                    pass  # Fall back to database
            
            # Check database for revoked sessions
            if self.db_pool.pool:
                # PostgreSQL
                result = await self.db_pool.fetchval(
                    """
                    SELECT COUNT(*) 
                    FROM sessions 
                    WHERE token_hash = $1 AND is_revoked = true
                    """,
                    token_hash
                )
            else:
                # SQLite
                result = await self.db_pool.fetchval(
                    """
                    SELECT COUNT(*) 
                    FROM sessions 
                    WHERE token_hash = ? AND is_revoked = 1
                    """,
                    token_hash
                )
            
            return result > 0
            
        except Exception as e:
            logger.error(f"Error checking token blacklist: {e}")
            # Fail open - if we can't check, assume not blacklisted
            return False
    
    async def get_user_sessions(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all sessions for a user (alias for get_active_sessions)"""
        return await self.get_active_sessions(user_id)
    
    async def get_active_sessions(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all active sessions for a user"""
        if not self._initialized:
            await self.initialize()
        
        try:
            sessions = await self.db_pool.fetchall(
                """
                SELECT id, ip_address, user_agent, device_id,
                       created_at, last_activity, expires_at
                FROM sessions
                WHERE user_id = ? AND is_active = ?
                ORDER BY last_activity DESC
                """,
                user_id, True
            )
            
            return sessions
            
        except Exception as e:
            logger.error(f"Failed to get active sessions: {e}")
            return []
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions from database and cache"""
        if not self._initialized:
            await self.initialize()
        
        try:
            logger.info("Starting session cleanup...")
            
            async with self.db_pool.transaction() as conn:
                if hasattr(conn, 'fetchval'):
                    # PostgreSQL
                    deleted = await conn.fetchval(
                        """
                        DELETE FROM sessions
                        WHERE expires_at < CURRENT_TIMESTAMP - INTERVAL '1 day'
                        OR (is_active = FALSE AND revoked_at < CURRENT_TIMESTAMP - INTERVAL '7 days')
                        RETURNING COUNT(*)
                        """
                    )
                else:
                    # SQLite
                    cursor = await conn.execute(
                        """
                        DELETE FROM sessions
                        WHERE datetime(expires_at) < datetime('now', '-1 day')
                        OR (is_active = 0 AND datetime(revoked_at) < datetime('now', '-7 days'))
                        """
                    )
                    deleted = cursor.rowcount
                    await conn.commit()
                
                if deleted:
                    logger.info(f"Cleaned up {deleted} expired sessions")
                
                # Clean Redis cache
                if self.redis_client:
                    await self._cleanup_redis_cache()
                    
        except Exception as e:
            logger.error(f"Session cleanup failed: {e}")
    
    # Redis cache helpers
    async def _cache_session(
        self,
        token_hash: str,
        user_id: int,
        session_id: int,
        expires_at: datetime
    ):
        """Cache session in Redis"""
        if not self.redis_client:
            return
        
        try:
            cache_data = {
                "user_id": user_id,
                "session_id": session_id,
                "expires_at": expires_at.isoformat()
            }
            
            # Calculate TTL
            ttl = int((expires_at - datetime.utcnow()).total_seconds())
            if ttl > 0:
                # Cache session data
                self.redis_client.setex(
                    f"session:{token_hash}",
                    ttl,
                    json.dumps(cache_data)
                )
                
                # Add to user's session set
                self.redis_client.sadd(f"user:{user_id}:sessions", session_id)
                self.redis_client.expire(f"user:{user_id}:sessions", ttl)
                
        except RedisError as e:
            logger.warning(f"Failed to cache session: {e}")
    
    async def _get_cached_session(self, token_hash: str) -> Optional[Dict[str, Any]]:
        """Get session from Redis cache"""
        if not self.redis_client:
            return None
        
        try:
            cached = self.redis_client.get(f"session:{token_hash}")
            if cached:
                data = json.loads(cached)
                expires = datetime.fromisoformat(data['expires_at'])
                if expires > datetime.utcnow():
                    return data
                else:
                    # Expired, remove from cache
                    self.redis_client.delete(f"session:{token_hash}")
            return None
            
        except RedisError:
            return None
    
    async def _clear_session_cache(self, session_id: int):
        """Clear specific session from cache"""
        if not self.redis_client:
            return
        
        try:
            # Find and delete session by scanning (not ideal but necessary)
            for key in self.redis_client.scan_iter("session:*"):
                data = self.redis_client.get(key)
                if data:
                    session_data = json.loads(data)
                    if session_data.get('session_id') == session_id:
                        self.redis_client.delete(key)
                        break
                        
        except RedisError:
            pass
    
    async def _clear_user_sessions_cache(self, user_id: int):
        """Clear all sessions for a user from cache"""
        if not self.redis_client:
            return
        
        try:
            # Get user's sessions
            session_ids = self.redis_client.smembers(f"user:{user_id}:sessions")
            
            # Clear each session
            for key in self.redis_client.scan_iter("session:*"):
                data = self.redis_client.get(key)
                if data:
                    session_data = json.loads(data)
                    if session_data.get('user_id') == user_id:
                        self.redis_client.delete(key)
            
            # Clear user's session set
            self.redis_client.delete(f"user:{user_id}:sessions")
            
        except RedisError:
            pass
    
    async def _cleanup_redis_cache(self):
        """Clean up expired sessions from Redis"""
        if not self.redis_client:
            return
        
        try:
            count = 0
            for key in self.redis_client.scan_iter("session:*"):
                ttl = self.redis_client.ttl(key)
                if ttl == -1:  # No expiry set
                    self.redis_client.delete(key)
                    count += 1
            
            if count:
                logger.info(f"Cleaned {count} sessions from Redis cache")
                
        except RedisError as e:
            logger.warning(f"Redis cache cleanup failed: {e}")
    
    async def _update_last_activity(self, session_id: int, conn):
        """Update last activity timestamp for a session"""
        try:
            if hasattr(conn, 'execute'):
                # PostgreSQL
                await conn.execute(
                    "UPDATE sessions SET last_activity = CURRENT_TIMESTAMP WHERE id = $1",
                    session_id
                )
            else:
                # SQLite
                await conn.execute(
                    "UPDATE sessions SET last_activity = datetime('now') WHERE id = ?",
                    (session_id,)
                )
        except Exception:
            # Don't fail on activity update
            pass
    
    async def shutdown(self):
        """Shutdown session manager and cleanup"""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
        
        if self.redis_client:
            self.redis_client.close()
        
        logger.info("SessionManager shutdown complete")


#######################################################################################################################
#
# Module Functions

# Global instance
_session_manager: Optional[SessionManager] = None


async def get_session_manager() -> SessionManager:
    """Get session manager singleton instance"""
    global _session_manager
    if not _session_manager:
        _session_manager = SessionManager()
        await _session_manager.initialize()
    return _session_manager


async def reset_session_manager():
    """Reset session manager (for testing)"""
    global _session_manager
    if _session_manager:
        await _session_manager.shutdown()
    _session_manager = None


#
# End of session_manager.py
#######################################################################################################################