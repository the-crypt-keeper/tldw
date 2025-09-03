"""
Per-user rate limiting for Evaluations module.

Implements tiered rate limiting based on user subscription levels
with support for burst traffic and cost-based limits.
"""

import time
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import asyncio
from pathlib import Path

# Import configuration management
from tldw_Server_API.app.core.Evaluations.config_manager import get_rate_limit_config, get_config
# Import connection pool
from tldw_Server_API.app.core.Evaluations.connection_pool import get_connection
import sqlite3


class UserTier(Enum):
    """User subscription tiers."""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


@dataclass
class RateLimitConfig:
    """Rate limit configuration for a user tier."""
    tier: UserTier
    evaluations_per_minute: int
    batch_evaluations_per_minute: int
    evaluations_per_day: int
    total_tokens_per_day: int
    burst_size: int
    max_cost_per_day: float
    max_cost_per_month: float
    
    @classmethod
    def for_tier(cls, tier: UserTier) -> "RateLimitConfig":
        """Get configuration for a tier from external config."""
        # Try to get configuration from external config first
        tier_config = get_rate_limit_config(tier.value)
        
        if tier_config:
            return cls(
                tier=tier,
                evaluations_per_minute=tier_config.evaluations_per_minute,
                batch_evaluations_per_minute=tier_config.batch_evaluations_per_minute,
                evaluations_per_day=tier_config.evaluations_per_day,
                total_tokens_per_day=tier_config.total_tokens_per_day,
                burst_size=tier_config.burst_size,
                max_cost_per_day=tier_config.max_cost_per_day,
                max_cost_per_month=tier_config.max_cost_per_month
            )
        
        # Fallback to hardcoded defaults if external config not available
        fallback_configs = {
            UserTier.FREE: cls(
                tier=UserTier.FREE,
                evaluations_per_minute=10,
                batch_evaluations_per_minute=2,
                evaluations_per_day=100,
                total_tokens_per_day=100_000,
                burst_size=5,
                max_cost_per_day=1.0,
                max_cost_per_month=10.0
            ),
            UserTier.BASIC: cls(
                tier=UserTier.BASIC,
                evaluations_per_minute=30,
                batch_evaluations_per_minute=5,
                evaluations_per_day=1000,
                total_tokens_per_day=1_000_000,
                burst_size=10,
                max_cost_per_day=10.0,
                max_cost_per_month=100.0
            ),
            UserTier.PREMIUM: cls(
                tier=UserTier.PREMIUM,
                evaluations_per_minute=100,
                batch_evaluations_per_minute=20,
                evaluations_per_day=10000,
                total_tokens_per_day=10_000_000,
                burst_size=25,
                max_cost_per_day=100.0,
                max_cost_per_month=1000.0
            ),
            UserTier.ENTERPRISE: cls(
                tier=UserTier.ENTERPRISE,
                evaluations_per_minute=500,
                batch_evaluations_per_minute=100,
                evaluations_per_day=100000,
                total_tokens_per_day=100_000_000,
                burst_size=100,
                max_cost_per_day=1000.0,
                max_cost_per_month=10000.0
            ),
            UserTier.CUSTOM: cls(
                tier=UserTier.CUSTOM,
                evaluations_per_minute=10,
                batch_evaluations_per_minute=2,
                evaluations_per_day=100,
                total_tokens_per_day=100_000,
                burst_size=5,
                max_cost_per_day=1.0,
                max_cost_per_month=10.0
            )
        }
        
        logger.warning(f"Using fallback configuration for tier {tier.value}")
        return fallback_configs.get(tier, fallback_configs[UserTier.FREE])


class UserRateLimiter:
    """Per-user rate limiter with tier-based limits."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize user rate limiter.
        
        Args:
            db_path: Path to rate limiting database
        """
        if db_path is None:
            db_dir = Path(__file__).parent.parent.parent.parent / "Databases"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "evaluations.db"
        
        self.db_path = str(db_path)
        self._init_database()
        
        # In-memory cache for rate limit data (with TTL)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = 60  # seconds
    
    def _init_database(self):
        """Initialize rate limiting tables."""
        with sqlite3.connect(self.db_path) as conn:
            # User rate limits table (created in migration)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_rate_limits (
                    user_id TEXT PRIMARY KEY,
                    tier TEXT NOT NULL DEFAULT 'free',
                    evaluations_per_minute INTEGER DEFAULT 10,
                    batch_evaluations_per_minute INTEGER DEFAULT 2,
                    evaluations_per_day INTEGER DEFAULT 100,
                    total_tokens_per_day INTEGER DEFAULT 100000,
                    burst_size INTEGER DEFAULT 5,
                    max_cost_per_day REAL DEFAULT 10.0,
                    max_cost_per_month REAL DEFAULT 100.0,
                    custom_limits TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    notes TEXT
                )
            """)
            
            # Rate limit tracking table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rate_limit_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    tokens_used INTEGER DEFAULT 0,
                    cost REAL DEFAULT 0.0
                )
            """)
            
            # Create indexes separately
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tracking_user ON rate_limit_tracking(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tracking_time ON rate_limit_tracking(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tracking_endpoint ON rate_limit_tracking(endpoint)")
            
            # Daily usage summary
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_usage (
                    user_id TEXT NOT NULL,
                    date DATE NOT NULL,
                    total_evaluations INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    total_cost REAL DEFAULT 0.0,
                    PRIMARY KEY (user_id, date)
                )
            """)
            
            conn.commit()
    
    async def check_rate_limit(
        self,
        user_id: str,
        endpoint: str,
        is_batch: bool = False,
        tokens_requested: int = 0,
        estimated_cost: float = 0.0
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if user can make request based on their tier limits.
        
        Args:
            user_id: User identifier
            endpoint: API endpoint being accessed
            is_batch: Whether this is a batch evaluation
            tokens_requested: Estimated tokens for request
            estimated_cost: Estimated cost for request
            
        Returns:
            Tuple of (is_allowed, metadata)
            - is_allowed: Whether request is allowed
            - metadata: Rate limit information and headers
        """
        # Get user's rate limit configuration
        config = await self._get_user_config(user_id)
        
        # Check per-minute limits
        minute_check = await self._check_minute_limit(user_id, endpoint, is_batch, config)
        if not minute_check[0]:
            return minute_check
        
        # Check daily limits
        daily_check = await self._check_daily_limits(user_id, tokens_requested, estimated_cost, config)
        if not daily_check[0]:
            return daily_check
        
        # Record the request
        await self._record_request(user_id, endpoint, tokens_requested, estimated_cost)
        
        # Return success with rate limit headers
        metadata = self._generate_rate_limit_headers(user_id, config, minute_check[1], daily_check[1])
        return True, metadata
    
    async def _get_user_config(self, user_id: str) -> RateLimitConfig:
        """Get user's rate limit configuration."""
        # Check cache first
        cache_key = f"config_{user_id}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if time.time() - cached["timestamp"] < self._cache_ttl:
                return cached["config"]
        
        # Query database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT tier, evaluations_per_minute, batch_evaluations_per_minute,
                       evaluations_per_day, total_tokens_per_day, burst_size,
                       max_cost_per_day, max_cost_per_month, expires_at
                FROM user_rate_limits
                WHERE user_id = ?
            """, (user_id,))
            
            row = cursor.fetchone()
            
            if row:
                # Check if custom limits have expired
                if row[8] and datetime.fromisoformat(row[8]) < datetime.now(timezone.utc):
                    # Reset to default tier
                    tier = UserTier.FREE
                    config = RateLimitConfig.for_tier(tier)
                    
                    # Update database
                    cursor.execute("""
                        UPDATE user_rate_limits
                        SET tier = ?, expires_at = NULL
                        WHERE user_id = ?
                    """, (tier.value, user_id))
                    conn.commit()
                else:
                    # Use stored configuration
                    config = RateLimitConfig(
                        tier=UserTier(row[0]),
                        evaluations_per_minute=row[1],
                        batch_evaluations_per_minute=row[2],
                        evaluations_per_day=row[3],
                        total_tokens_per_day=row[4],
                        burst_size=row[5],
                        max_cost_per_day=row[6],
                        max_cost_per_month=row[7]
                    )
            else:
                # Create default configuration for new user
                config = RateLimitConfig.for_tier(UserTier.FREE)
                
                cursor.execute("""
                    INSERT INTO user_rate_limits (
                        user_id, tier, evaluations_per_minute, batch_evaluations_per_minute,
                        evaluations_per_day, total_tokens_per_day, burst_size,
                        max_cost_per_day, max_cost_per_month
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id, config.tier.value, config.evaluations_per_minute,
                    config.batch_evaluations_per_minute, config.evaluations_per_day,
                    config.total_tokens_per_day, config.burst_size,
                    config.max_cost_per_day, config.max_cost_per_month
                ))
                conn.commit()
        
        # Cache the configuration
        self._cache[cache_key] = {
            "config": config,
            "timestamp": time.time()
        }
        
        return config
    
    async def _check_minute_limit(
        self,
        user_id: str,
        endpoint: str,
        is_batch: bool,
        config: RateLimitConfig
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check per-minute rate limits."""
        now = datetime.now(timezone.utc)
        minute_ago = now - timedelta(minutes=1)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Count requests in last minute
            cursor.execute("""
                SELECT COUNT(*) FROM rate_limit_tracking
                WHERE user_id = ? AND timestamp > ?
                AND endpoint LIKE ?
            """, (user_id, minute_ago, f"%{endpoint}%"))
            
            request_count = cursor.fetchone()[0]
            
            # Check against limit
            limit = config.batch_evaluations_per_minute if is_batch else config.evaluations_per_minute
            
            # Check burst allowance
            if request_count >= limit:
                # Check if within burst window
                burst_window = now - timedelta(seconds=10)
                cursor.execute("""
                    SELECT COUNT(*) FROM rate_limit_tracking
                    WHERE user_id = ? AND timestamp > ?
                    AND endpoint LIKE ?
                """, (user_id, burst_window, f"%{endpoint}%"))
                
                burst_count = cursor.fetchone()[0]
                
                if burst_count >= config.burst_size:
                    retry_after = 60 - (now - minute_ago).seconds
                    return False, {
                        "error": "Rate limit exceeded",
                        "retry_after": retry_after,
                        "limit": limit,
                        "window": "1 minute",
                        "tier": config.tier.value
                    }
            
            return True, {
                "requests_remaining": limit - request_count - 1,
                "limit": limit,
                "window": "1 minute"
            }
    
    async def _check_daily_limits(
        self,
        user_id: str,
        tokens_requested: int,
        estimated_cost: float,
        config: RateLimitConfig
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check daily usage limits."""
        today = datetime.now(timezone.utc).date()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get today's usage
            cursor.execute("""
                SELECT total_evaluations, total_tokens, total_cost
                FROM daily_usage
                WHERE user_id = ? AND date = ?
            """, (user_id, today))
            
            row = cursor.fetchone()
            
            if row:
                total_evaluations, total_tokens, total_cost = row
            else:
                total_evaluations = total_tokens = total_cost = 0
            
            # Check limits
            if total_evaluations >= config.evaluations_per_day:
                return False, {
                    "error": "Daily evaluation limit exceeded",
                    "limit": config.evaluations_per_day,
                    "used": total_evaluations,
                    "resets_at": (datetime.combine(today + timedelta(days=1), datetime.min.time())).isoformat()
                }
            
            if total_tokens + tokens_requested > config.total_tokens_per_day:
                return False, {
                    "error": "Daily token limit exceeded",
                    "limit": config.total_tokens_per_day,
                    "used": total_tokens,
                    "requested": tokens_requested,
                    "resets_at": (datetime.combine(today + timedelta(days=1), datetime.min.time())).isoformat()
                }
            
            if total_cost + estimated_cost > config.max_cost_per_day:
                return False, {
                    "error": "Daily cost limit exceeded",
                    "limit": config.max_cost_per_day,
                    "used": total_cost,
                    "requested": estimated_cost,
                    "resets_at": (datetime.combine(today + timedelta(days=1), datetime.min.time())).isoformat()
                }
            
            return True, {
                "evaluations_remaining": config.evaluations_per_day - total_evaluations - 1,
                "tokens_remaining": config.total_tokens_per_day - total_tokens - tokens_requested,
                "cost_remaining": config.max_cost_per_day - total_cost - estimated_cost
            }
    
    async def _record_request(
        self,
        user_id: str,
        endpoint: str,
        tokens_used: int,
        cost: float
    ):
        """Record a request for tracking."""
        now = datetime.now(timezone.utc)
        today = now.date()
        
        with sqlite3.connect(self.db_path) as conn:
            # Record in tracking table
            conn.execute("""
                INSERT INTO rate_limit_tracking (user_id, endpoint, timestamp, tokens_used, cost)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, endpoint, now, tokens_used, cost))
            
            # Update daily usage
            conn.execute("""
                INSERT INTO daily_usage (user_id, date, total_evaluations, total_tokens, total_cost)
                VALUES (?, ?, 1, ?, ?)
                ON CONFLICT(user_id, date) DO UPDATE SET
                    total_evaluations = total_evaluations + 1,
                    total_tokens = total_tokens + ?,
                    total_cost = total_cost + ?
            """, (user_id, today, tokens_used, cost, tokens_used, cost))
            
            conn.commit()
    
    def _generate_rate_limit_headers(
        self,
        user_id: str,
        config: RateLimitConfig,
        minute_metadata: Dict[str, Any],
        daily_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate rate limit headers for response."""
        return {
            "headers": {
                "X-RateLimit-Tier": config.tier.value,
                "X-RateLimit-Limit": str(minute_metadata.get("limit", 0)),
                "X-RateLimit-Remaining": str(minute_metadata.get("requests_remaining", 0)),
                "X-RateLimit-Reset": str(int(time.time()) + 60),
                "X-RateLimit-Daily-Limit": str(config.evaluations_per_day),
                "X-RateLimit-Daily-Remaining": str(daily_metadata.get("evaluations_remaining", 0)),
                "X-RateLimit-Tokens-Remaining": str(daily_metadata.get("tokens_remaining", 0)),
                "X-RateLimit-Cost-Remaining": f"{daily_metadata.get('cost_remaining', 0):.2f}"
            },
            "tier": config.tier.value,
            "limits": {
                "per_minute": minute_metadata,
                "daily": daily_metadata
            }
        }
    
    async def upgrade_user_tier(
        self,
        user_id: str,
        new_tier: UserTier,
        expires_at: Optional[datetime] = None,
        custom_limits: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Upgrade a user's tier.
        
        Args:
            user_id: User identifier
            new_tier: New tier to assign
            expires_at: Optional expiration for temporary upgrades
            custom_limits: Optional custom limit overrides
            
        Returns:
            True if upgrade successful
        """
        try:
            config = RateLimitConfig.for_tier(new_tier)
            
            # Apply custom limits if provided
            if custom_limits:
                for key, value in custom_limits.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE user_rate_limits
                    SET tier = ?, evaluations_per_minute = ?, batch_evaluations_per_minute = ?,
                        evaluations_per_day = ?, total_tokens_per_day = ?, burst_size = ?,
                        max_cost_per_day = ?, max_cost_per_month = ?,
                        expires_at = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                """, (
                    new_tier.value, config.evaluations_per_minute, config.batch_evaluations_per_minute,
                    config.evaluations_per_day, config.total_tokens_per_day, config.burst_size,
                    config.max_cost_per_day, config.max_cost_per_month,
                    expires_at, user_id
                ))
                
                if cursor.rowcount == 0:
                    # User doesn't exist, create new entry
                    cursor.execute("""
                        INSERT INTO user_rate_limits (
                            user_id, tier, evaluations_per_minute, batch_evaluations_per_minute,
                            evaluations_per_day, total_tokens_per_day, burst_size,
                            max_cost_per_day, max_cost_per_month, expires_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        user_id, new_tier.value, config.evaluations_per_minute,
                        config.batch_evaluations_per_minute, config.evaluations_per_day,
                        config.total_tokens_per_day, config.burst_size,
                        config.max_cost_per_day, config.max_cost_per_month,
                        expires_at
                    ))
                
                conn.commit()
            
            # Clear cache
            cache_key = f"config_{user_id}"
            if cache_key in self._cache:
                del self._cache[cache_key]
            
            logger.info(f"Upgraded user {user_id} to tier {new_tier.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upgrade user tier: {e}")
            return False
    
    async def get_usage_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get usage summary for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Usage summary with current limits and consumption
        """
        config = await self._get_user_config(user_id)
        today = datetime.now(timezone.utc).date()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get today's usage
            cursor.execute("""
                SELECT total_evaluations, total_tokens, total_cost
                FROM daily_usage
                WHERE user_id = ? AND date = ?
            """, (user_id, today))
            
            row = cursor.fetchone()
            
            if row:
                total_evaluations, total_tokens, total_cost = row
            else:
                total_evaluations = total_tokens = total_cost = 0
            
            # Get monthly cost
            month_start = today.replace(day=1)
            cursor.execute("""
                SELECT SUM(total_cost)
                FROM daily_usage
                WHERE user_id = ? AND date >= ?
            """, (user_id, month_start))
            
            monthly_cost = cursor.fetchone()[0] or 0.0
        
        return {
            "user_id": user_id,
            "tier": config.tier.value,
            "limits": {
                "per_minute": {
                    "evaluations": config.evaluations_per_minute,
                    "batch_evaluations": config.batch_evaluations_per_minute,
                    "burst_size": config.burst_size
                },
                "daily": {
                    "evaluations": config.evaluations_per_day,
                    "tokens": config.total_tokens_per_day,
                    "cost": config.max_cost_per_day
                },
                "monthly": {
                    "cost": config.max_cost_per_month
                }
            },
            "usage": {
                "today": {
                    "evaluations": total_evaluations,
                    "tokens": total_tokens,
                    "cost": total_cost
                },
                "month": {
                    "cost": monthly_cost
                }
            },
            "remaining": {
                "daily_evaluations": config.evaluations_per_day - total_evaluations,
                "daily_tokens": config.total_tokens_per_day - total_tokens,
                "daily_cost": config.max_cost_per_day - total_cost,
                "monthly_cost": config.max_cost_per_month - monthly_cost
            }
        }


# Global instance
user_rate_limiter = UserRateLimiter()