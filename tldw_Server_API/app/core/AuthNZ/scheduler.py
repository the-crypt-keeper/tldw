# scheduler.py
# Description: Scheduled jobs for AuthNZ maintenance tasks
#
# Imports
import asyncio
from datetime import datetime, timedelta
from typing import Optional
#
# 3rd-party imports
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from loguru import logger
#
# Local imports
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.AuthNZ.session_manager import get_session_manager
from tldw_Server_API.app.core.AuthNZ.api_key_manager import get_api_key_manager
from tldw_Server_API.app.core.AuthNZ.rate_limiter import get_rate_limiter
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

#######################################################################################################################
#
# Scheduled Jobs
#

class AuthNZScheduler:
    """Manages scheduled maintenance tasks for the AuthNZ module"""
    
    def __init__(self):
        """Initialize the scheduler"""
        self.scheduler = AsyncIOScheduler()
        self.settings = get_settings()
        self._started = False
        
    async def start(self):
        """Start the scheduler and register all jobs"""
        if self._started:
            logger.warning("AuthNZ scheduler already started")
            return
        
        # Register cleanup jobs
        self._register_session_cleanup()
        self._register_api_key_cleanup()
        self._register_rate_limit_cleanup()
        self._register_audit_log_cleanup()
        self._register_expired_registration_cleanup()
        
        # Register monitoring jobs
        self._register_auth_failure_monitor()
        self._register_api_usage_monitor()
        self._register_rate_limit_monitor()
        
        # Start the scheduler
        self.scheduler.start()
        self._started = True
        logger.info("AuthNZ scheduler started with all jobs registered")
    
    async def stop(self):
        """Stop the scheduler"""
        if not self._started:
            return
        
        self.scheduler.shutdown(wait=True)
        self._started = False
        logger.info("AuthNZ scheduler stopped")
    
    def _register_session_cleanup(self):
        """Register job to clean up expired sessions"""
        self.scheduler.add_job(
            self._cleanup_expired_sessions,
            trigger=IntervalTrigger(
                hours=self.settings.SESSION_CLEANUP_INTERVAL_HOURS
            ),
            id='session_cleanup',
            name='Clean up expired sessions',
            replace_existing=True,
            max_instances=1
        )
        logger.debug(f"Registered session cleanup job (every {self.settings.SESSION_CLEANUP_INTERVAL_HOURS} hours)")
    
    def _register_api_key_cleanup(self):
        """Register job to clean up expired API keys"""
        self.scheduler.add_job(
            self._cleanup_expired_api_keys,
            trigger=CronTrigger(
                hour=2,  # Run at 2 AM daily
                minute=0
            ),
            id='api_key_cleanup',
            name='Clean up expired API keys',
            replace_existing=True,
            max_instances=1
        )
        logger.debug("Registered API key cleanup job (daily at 2 AM)")
    
    def _register_rate_limit_cleanup(self):
        """Register job to clean up old rate limit entries"""
        self.scheduler.add_job(
            self._cleanup_old_rate_limits,
            trigger=IntervalTrigger(hours=6),  # Every 6 hours
            id='rate_limit_cleanup',
            name='Clean up old rate limit entries',
            replace_existing=True,
            max_instances=1
        )
        logger.debug("Registered rate limit cleanup job (every 6 hours)")
    
    def _register_audit_log_cleanup(self):
        """Register job to prune old audit logs"""
        self.scheduler.add_job(
            self._prune_audit_logs,
            trigger=CronTrigger(
                day=1,  # First day of month
                hour=3,
                minute=0
            ),
            id='audit_log_cleanup',
            name='Prune old audit logs',
            replace_existing=True,
            max_instances=1
        )
        logger.debug("Registered audit log cleanup job (monthly)")
    
    def _register_expired_registration_cleanup(self):
        """Register job to clean up expired registration codes"""
        self.scheduler.add_job(
            self._cleanup_expired_registration_codes,
            trigger=CronTrigger(
                hour=1,  # Run at 1 AM daily
                minute=30
            ),
            id='registration_cleanup',
            name='Clean up expired registration codes',
            replace_existing=True,
            max_instances=1
        )
        logger.debug("Registered registration code cleanup job (daily at 1:30 AM)")
    
    def _register_auth_failure_monitor(self):
        """Register job to monitor authentication failures"""
        self.scheduler.add_job(
            self._monitor_auth_failures,
            trigger=IntervalTrigger(minutes=5),  # Every 5 minutes
            id='auth_failure_monitor',
            name='Monitor authentication failures',
            replace_existing=True,
            max_instances=1
        )
        logger.debug("Registered auth failure monitor (every 5 minutes)")
    
    def _register_api_usage_monitor(self):
        """Register job to monitor API key usage patterns"""
        self.scheduler.add_job(
            self._monitor_api_usage,
            trigger=IntervalTrigger(hours=1),  # Every hour
            id='api_usage_monitor',
            name='Monitor API key usage',
            replace_existing=True,
            max_instances=1
        )
        logger.debug("Registered API usage monitor (hourly)")
    
    def _register_rate_limit_monitor(self):
        """Register job to monitor rate limit violations"""
        self.scheduler.add_job(
            self._monitor_rate_limits,
            trigger=IntervalTrigger(minutes=15),  # Every 15 minutes
            id='rate_limit_monitor',
            name='Monitor rate limit violations',
            replace_existing=True,
            max_instances=1
        )
        logger.debug("Registered rate limit monitor (every 15 minutes)")
    
    # Cleanup Jobs
    
    async def _cleanup_expired_sessions(self):
        """Clean up expired sessions from the database"""
        try:
            session_manager = await get_session_manager()
            count = await session_manager.cleanup_expired_sessions()
            if count > 0:
                logger.info(f"Cleaned up {count} expired sessions")
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {e}")
    
    async def _cleanup_expired_api_keys(self):
        """Mark expired API keys as expired"""
        try:
            api_key_manager = await get_api_key_manager()
            await api_key_manager.cleanup_expired_keys()
            logger.info("Completed API key expiration check")
        except Exception as e:
            logger.error(f"Failed to cleanup expired API keys: {e}")
    
    async def _cleanup_old_rate_limits(self):
        """Remove old rate limit entries"""
        try:
            rate_limiter = await get_rate_limiter()
            await rate_limiter.cleanup_old_entries(hours=24)  # Remove entries older than 24 hours
            logger.info("Cleaned up old rate limit entries")
        except Exception as e:
            logger.error(f"Failed to cleanup rate limits: {e}")
    
    async def _prune_audit_logs(self):
        """Prune audit logs older than retention period"""
        try:
            db_pool = await get_db_pool()
            retention_days = self.settings.AUDIT_LOG_RETENTION_DAYS
            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
            
            async with db_pool.transaction() as conn:
                if hasattr(conn, 'execute'):
                    # PostgreSQL
                    result = await conn.execute(
                        "DELETE FROM audit_logs WHERE created_at < $1",
                        cutoff_date
                    )
                    # Extract count from result
                    count = int(result.split()[-1]) if isinstance(result, str) else 0
                else:
                    # SQLite
                    cursor = await conn.execute(
                        "DELETE FROM audit_logs WHERE created_at < ?",
                        (cutoff_date.isoformat(),)
                    )
                    count = cursor.rowcount
                    await conn.commit()
            
            if count > 0:
                logger.info(f"Pruned {count} audit log entries older than {retention_days} days")
        except Exception as e:
            logger.error(f"Failed to prune audit logs: {e}")
    
    async def _cleanup_expired_registration_codes(self):
        """Clean up expired registration codes"""
        try:
            db_pool = await get_db_pool()
            
            async with db_pool.transaction() as conn:
                if hasattr(conn, 'execute'):
                    # PostgreSQL
                    result = await conn.execute(
                        """
                        UPDATE registration_codes 
                        SET is_active = FALSE 
                        WHERE is_active = TRUE 
                        AND expires_at < $1
                        """,
                        datetime.utcnow()
                    )
                    count = int(result.split()[-1]) if isinstance(result, str) else 0
                else:
                    # SQLite
                    cursor = await conn.execute(
                        """
                        UPDATE registration_codes 
                        SET is_active = 0 
                        WHERE is_active = 1 
                        AND expires_at < ?
                        """,
                        (datetime.utcnow().isoformat(),)
                    )
                    count = cursor.rowcount
                    await conn.commit()
            
            if count > 0:
                logger.info(f"Deactivated {count} expired registration codes")
        except Exception as e:
            logger.error(f"Failed to cleanup registration codes: {e}")
    
    # Monitoring Jobs
    
    async def _monitor_auth_failures(self):
        """Monitor and alert on authentication failures"""
        try:
            db_pool = await get_db_pool()
            threshold = 10  # Alert if more than 10 failures in 5 minutes
            time_window = datetime.utcnow() - timedelta(minutes=5)
            
            # Check for failed login attempts
            result = await db_pool.fetchone(
                """
                SELECT COUNT(*) as failure_count,
                       COUNT(DISTINCT ip_address) as unique_ips
                FROM audit_logs 
                WHERE action IN ('login_failed', 'invalid_api_key', 'invalid_token')
                AND created_at > ?
                """,
                time_window.isoformat() if not hasattr(db_pool, 'fetchval') else time_window
            )
            
            if result:
                failure_count = result['failure_count'] or 0
                unique_ips = result['unique_ips'] or 0
                
                if failure_count > threshold:
                    logger.warning(
                        f"⚠️ High authentication failure rate detected: "
                        f"{failure_count} failures from {unique_ips} unique IPs in last 5 minutes"
                    )
                    
                    # Here you would trigger actual alerts (email, Slack, etc.)
                    await self._send_security_alert(
                        "High Authentication Failure Rate",
                        f"{failure_count} failures from {unique_ips} IPs"
                    )
        except Exception as e:
            logger.error(f"Failed to monitor auth failures: {e}")
    
    async def _monitor_api_usage(self):
        """Monitor API key usage patterns"""
        try:
            db_pool = await get_db_pool()
            
            # Get API usage statistics for the last hour
            time_window = datetime.utcnow() - timedelta(hours=1)
            
            results = await db_pool.fetchall(
                """
                SELECT 
                    k.id,
                    k.name,
                    k.user_id,
                    COUNT(l.id) as usage_count,
                    k.rate_limit
                FROM api_keys k
                LEFT JOIN api_key_audit_log l ON k.id = l.api_key_id
                WHERE k.status = 'active'
                AND l.created_at > ?
                GROUP BY k.id, k.name, k.user_id, k.rate_limit
                HAVING COUNT(l.id) > 0
                ORDER BY usage_count DESC
                LIMIT 10
                """,
                time_window.isoformat() if not hasattr(db_pool, 'fetchval') else time_window
            )
            
            for row in results:
                usage = row['usage_count']
                rate_limit = row['rate_limit'] or 60  # Default rate limit
                
                # Alert if usage is approaching rate limit
                if usage > rate_limit * 0.8:  # 80% of rate limit
                    logger.warning(
                        f"API key '{row['name']}' (ID: {row['id']}) "
                        f"approaching rate limit: {usage}/{rate_limit} requests/hour"
                    )
            
            # Log summary
            if results:
                total_usage = sum(r['usage_count'] for r in results)
                logger.info(f"API usage monitoring: {total_usage} total requests in last hour")
                
        except Exception as e:
            logger.error(f"Failed to monitor API usage: {e}")
    
    async def _monitor_rate_limits(self):
        """Monitor rate limit violations"""
        try:
            db_pool = await get_db_pool()
            time_window = datetime.utcnow() - timedelta(minutes=15)
            
            # Find IPs/users hitting rate limits
            results = await db_pool.fetchall(
                """
                SELECT 
                    identifier,
                    endpoint,
                    SUM(request_count) as total_requests,
                    COUNT(*) as window_count
                FROM rate_limits
                WHERE window_start > ?
                GROUP BY identifier, endpoint
                HAVING SUM(request_count) > ?
                ORDER BY total_requests DESC
                LIMIT 20
                """,
                time_window.isoformat() if not hasattr(db_pool, 'fetchval') else time_window,
                self.settings.RATE_LIMIT_PER_MINUTE * 15  # 15 minute threshold
            )
            
            if results:
                logger.warning(f"Rate limit violations detected for {len(results)} identifiers")
                
                for row in results:
                    logger.warning(
                        f"Rate limit violation: {row['identifier']} on {row['endpoint']} "
                        f"({row['total_requests']} requests in 15 minutes)"
                    )
                
                # Send alert if there are many violations
                if len(results) > 10:
                    await self._send_security_alert(
                        "Multiple Rate Limit Violations",
                        f"{len(results)} identifiers exceeding rate limits"
                    )
                    
        except Exception as e:
            logger.error(f"Failed to monitor rate limits: {e}")
    
    async def _send_security_alert(self, subject: str, message: str):
        """
        Send security alert (placeholder for actual alerting)
        
        In production, this would integrate with:
        - Email service (SendGrid, SES, etc.)
        - Slack/Discord webhooks
        - PagerDuty or similar incident management
        - SIEM systems
        """
        logger.critical(f"🚨 SECURITY ALERT: {subject} - {message}")
        
        # TODO: Implement actual alerting based on your infrastructure
        # Example integrations:
        # - await send_email_alert(subject, message)
        # - await send_slack_alert(subject, message)
        # - await trigger_pagerduty(subject, message)


#######################################################################################################################
#
# Module Functions
#

# Global scheduler instance
_scheduler: Optional[AuthNZScheduler] = None

async def get_authnz_scheduler() -> AuthNZScheduler:
    """Get the AuthNZ scheduler singleton"""
    global _scheduler
    if not _scheduler:
        _scheduler = AuthNZScheduler()
    return _scheduler

async def start_authnz_scheduler():
    """Start the AuthNZ scheduler"""
    scheduler = await get_authnz_scheduler()
    await scheduler.start()
    logger.info("AuthNZ scheduled jobs started")

async def stop_authnz_scheduler():
    """Stop the AuthNZ scheduler"""
    scheduler = await get_authnz_scheduler()
    await scheduler.stop()
    logger.info("AuthNZ scheduled jobs stopped")

#
# End of scheduler.py
#######################################################################################################################