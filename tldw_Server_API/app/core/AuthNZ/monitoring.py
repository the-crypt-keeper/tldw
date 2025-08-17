# monitoring.py
# Description: Monitoring and metrics collection for AuthNZ module
#
# Imports
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum
import json
from collections import defaultdict
#
# 3rd-party imports
from loguru import logger
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed - metrics will be logged only")
#
# Local imports
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.settings import get_settings

#######################################################################################################################
#
# Metrics Types
#

class MetricType(Enum):
    """Types of metrics we track"""
    AUTH_ATTEMPT = "auth_attempt"
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    API_KEY_VALIDATION = "api_key_validation"
    API_KEY_CREATED = "api_key_created"
    API_KEY_ROTATED = "api_key_rotated"
    API_KEY_REVOKED = "api_key_revoked"
    RATE_LIMIT_HIT = "rate_limit_hit"
    SESSION_CREATED = "session_created"
    SESSION_EXPIRED = "session_expired"
    REGISTRATION_ATTEMPT = "registration_attempt"
    REGISTRATION_SUCCESS = "registration_success"
    PASSWORD_RESET = "password_reset"
    TWO_FACTOR_ENABLED = "2fa_enabled"
    TWO_FACTOR_DISABLED = "2fa_disabled"
    SECURITY_ALERT = "security_alert"

#######################################################################################################################
#
# Prometheus Metrics (if available)
#

if PROMETHEUS_AVAILABLE:
    # Counters
    auth_attempts_total = Counter(
        'authnz_auth_attempts_total',
        'Total number of authentication attempts',
        ['method', 'status']
    )
    
    api_key_operations_total = Counter(
        'authnz_api_key_operations_total',
        'Total number of API key operations',
        ['operation']
    )
    
    rate_limit_violations_total = Counter(
        'authnz_rate_limit_violations_total',
        'Total number of rate limit violations',
        ['endpoint']
    )
    
    security_alerts_total = Counter(
        'authnz_security_alerts_total',
        'Total number of security alerts',
        ['alert_type']
    )
    
    # Histograms
    auth_duration_seconds = Histogram(
        'authnz_auth_duration_seconds',
        'Authentication request duration in seconds',
        ['method']
    )
    
    password_strength_score = Histogram(
        'authnz_password_strength_score',
        'Password strength scores',
        buckets=[0, 20, 40, 60, 80, 100]
    )
    
    # Gauges
    active_sessions_count = Gauge(
        'authnz_active_sessions_count',
        'Number of active sessions'
    )
    
    active_api_keys_count = Gauge(
        'authnz_active_api_keys_count',
        'Number of active API keys'
    )
    
    failed_auth_rate = Gauge(
        'authnz_failed_auth_rate_5min',
        'Failed authentication rate over 5 minutes'
    )

#######################################################################################################################
#
# Monitoring Manager
#

class AuthNZMonitor:
    """Centralized monitoring for AuthNZ operations"""
    
    def __init__(self):
        """Initialize the monitor"""
        self.settings = get_settings()
        self._metrics_buffer = defaultdict(list)
        self._alert_thresholds = {
            'failed_auth_5min': 10,
            'rate_limit_violations_15min': 50,
            'api_key_abuse_hourly': 1000,
            'suspicious_ips': 5,
            'password_reset_flood': 10
        }
        
    async def record_metric(
        self,
        metric_type: MetricType,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record a metric
        
        Args:
            metric_type: Type of metric to record
            value: Metric value (default 1 for counters)
            labels: Labels for Prometheus metrics
            metadata: Additional metadata to store
        """
        try:
            # Update Prometheus metrics if available
            if PROMETHEUS_AVAILABLE:
                await self._update_prometheus_metric(metric_type, value, labels)
            
            # Store in database for analysis
            await self._store_metric_in_db(metric_type, value, labels, metadata)
            
            # Check if this metric triggers any alerts
            await self._check_alert_conditions(metric_type, value, metadata)
            
        except Exception as e:
            logger.error(f"Failed to record metric {metric_type}: {e}")
    
    async def _update_prometheus_metric(
        self,
        metric_type: MetricType,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """Update Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        labels = labels or {}
        
        # Update appropriate Prometheus metric
        if metric_type in [MetricType.AUTH_ATTEMPT, MetricType.AUTH_SUCCESS, MetricType.AUTH_FAILURE]:
            method = labels.get('method', 'unknown')
            status = 'success' if metric_type == MetricType.AUTH_SUCCESS else 'failure'
            auth_attempts_total.labels(method=method, status=status).inc(value)
            
        elif metric_type in [MetricType.API_KEY_CREATED, MetricType.API_KEY_ROTATED, MetricType.API_KEY_REVOKED]:
            operation = metric_type.value.replace('api_key_', '')
            api_key_operations_total.labels(operation=operation).inc(value)
            
        elif metric_type == MetricType.RATE_LIMIT_HIT:
            endpoint = labels.get('endpoint', 'unknown')
            rate_limit_violations_total.labels(endpoint=endpoint).inc(value)
            
        elif metric_type == MetricType.SECURITY_ALERT:
            alert_type = labels.get('alert_type', 'unknown')
            security_alerts_total.labels(alert_type=alert_type).inc(value)
    
    async def _store_metric_in_db(
        self,
        metric_type: MetricType,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Store metric in database for historical analysis"""
        try:
            db_pool = await get_db_pool()
            
            # Prepare data
            metric_data = {
                'type': metric_type.value,
                'value': value,
                'labels': labels or {},
                'metadata': metadata or {},
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Store in audit log for now (could have dedicated metrics table)
            async with db_pool.transaction() as conn:
                if hasattr(conn, 'execute'):
                    # PostgreSQL
                    await conn.execute(
                        """
                        INSERT INTO audit_logs (action, details, created_at)
                        VALUES ($1, $2, $3)
                        """,
                        f"metric_{metric_type.value}",
                        json.dumps(metric_data),
                        datetime.utcnow()
                    )
                else:
                    # SQLite
                    await conn.execute(
                        """
                        INSERT INTO audit_logs (action, details, created_at)
                        VALUES (?, ?, ?)
                        """,
                        (f"metric_{metric_type.value}",
                         json.dumps(metric_data),
                         datetime.utcnow().isoformat())
                    )
                    await conn.commit()
                    
        except Exception as e:
            logger.error(f"Failed to store metric in database: {e}")
    
    async def _check_alert_conditions(
        self,
        metric_type: MetricType,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Check if metric triggers alert conditions"""
        # Add to buffer for rate calculations
        self._metrics_buffer[metric_type].append({
            'value': value,
            'timestamp': datetime.utcnow(),
            'metadata': metadata
        })
        
        # Clean old entries from buffer (keep last hour)
        cutoff = datetime.utcnow() - timedelta(hours=1)
        for key in self._metrics_buffer:
            self._metrics_buffer[key] = [
                m for m in self._metrics_buffer[key]
                if m['timestamp'] > cutoff
            ]
        
        # Check specific alert conditions
        if metric_type == MetricType.AUTH_FAILURE:
            await self._check_auth_failure_rate()
        elif metric_type == MetricType.RATE_LIMIT_HIT:
            await self._check_rate_limit_violations()
        elif metric_type == MetricType.API_KEY_VALIDATION:
            await self._check_api_key_abuse()
    
    async def _check_auth_failure_rate(self):
        """Check if authentication failure rate is too high"""
        five_min_ago = datetime.utcnow() - timedelta(minutes=5)
        recent_failures = [
            m for m in self._metrics_buffer[MetricType.AUTH_FAILURE]
            if m['timestamp'] > five_min_ago
        ]
        
        if len(recent_failures) > self._alert_thresholds['failed_auth_5min']:
            # Extract unique IPs
            unique_ips = set()
            for failure in recent_failures:
                if failure['metadata'] and 'ip_address' in failure['metadata']:
                    unique_ips.add(failure['metadata']['ip_address'])
            
            await self.trigger_alert(
                'high_auth_failure_rate',
                f"{len(recent_failures)} authentication failures in 5 minutes from {len(unique_ips)} unique IPs",
                severity='high',
                metadata={
                    'failure_count': len(recent_failures),
                    'unique_ips': list(unique_ips)
                }
            )
    
    async def _check_rate_limit_violations(self):
        """Check if rate limit violations are excessive"""
        fifteen_min_ago = datetime.utcnow() - timedelta(minutes=15)
        recent_violations = [
            m for m in self._metrics_buffer[MetricType.RATE_LIMIT_HIT]
            if m['timestamp'] > fifteen_min_ago
        ]
        
        if len(recent_violations) > self._alert_thresholds['rate_limit_violations_15min']:
            await self.trigger_alert(
                'excessive_rate_limiting',
                f"{len(recent_violations)} rate limit violations in 15 minutes",
                severity='medium',
                metadata={'violation_count': len(recent_violations)}
            )
    
    async def _check_api_key_abuse(self):
        """Check for API key abuse patterns"""
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_validations = [
            m for m in self._metrics_buffer[MetricType.API_KEY_VALIDATION]
            if m['timestamp'] > one_hour_ago
        ]
        
        # Group by API key
        key_usage = defaultdict(int)
        for validation in recent_validations:
            if validation['metadata'] and 'key_id' in validation['metadata']:
                key_usage[validation['metadata']['key_id']] += 1
        
        # Check for abuse
        for key_id, count in key_usage.items():
            if count > self._alert_thresholds['api_key_abuse_hourly']:
                await self.trigger_alert(
                    'api_key_abuse',
                    f"API key {key_id} used {count} times in one hour",
                    severity='high',
                    metadata={'key_id': key_id, 'usage_count': count}
                )
    
    async def trigger_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = 'medium',
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Trigger a security alert
        
        Args:
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity (low, medium, high, critical)
            metadata: Additional alert metadata
        """
        # Log the alert
        log_method = {
            'low': logger.info,
            'medium': logger.warning,
            'high': logger.error,
            'critical': logger.critical
        }.get(severity, logger.warning)
        
        log_method(f"🚨 SECURITY ALERT [{severity.upper()}]: {alert_type} - {message}")
        
        # Record the alert as a metric
        await self.record_metric(
            MetricType.SECURITY_ALERT,
            labels={'alert_type': alert_type, 'severity': severity},
            metadata=metadata
        )
        
        # In production, integrate with alerting services
        # await self._send_to_alerting_service(alert_type, message, severity, metadata)
    
    async def get_metrics_summary(
        self,
        time_range_minutes: int = 60
    ) -> Dict[str, Any]:
        """
        Get a summary of metrics for the specified time range
        
        Args:
            time_range_minutes: Time range in minutes
            
        Returns:
            Dictionary with metrics summary
        """
        try:
            db_pool = await get_db_pool()
            cutoff = datetime.utcnow() - timedelta(minutes=time_range_minutes)
            
            # Get authentication metrics
            auth_metrics = await db_pool.fetchone(
                """
                SELECT 
                    COUNT(CASE WHEN action = 'metric_auth_success' THEN 1 END) as successful_auths,
                    COUNT(CASE WHEN action = 'metric_auth_failure' THEN 1 END) as failed_auths,
                    COUNT(CASE WHEN action = 'metric_rate_limit_hit' THEN 1 END) as rate_limit_hits
                FROM audit_logs
                WHERE created_at > ?
                AND action LIKE 'metric_%'
                """,
                cutoff.isoformat() if not hasattr(db_pool, 'fetchval') else cutoff
            )
            
            # Get active sessions count
            sessions_count = await db_pool.fetchone(
                """
                SELECT COUNT(*) as active_sessions
                FROM sessions
                WHERE expires_at > ? AND is_revoked = 0
                """,
                datetime.utcnow().isoformat() if not hasattr(db_pool, 'fetchval') else datetime.utcnow()
            )
            
            # Get active API keys count
            api_keys_count = await db_pool.fetchone(
                """
                SELECT COUNT(*) as active_keys
                FROM api_keys
                WHERE status = 'active'
                """
            )
            
            # Calculate success rate
            total_auths = (auth_metrics['successful_auths'] or 0) + (auth_metrics['failed_auths'] or 0)
            success_rate = ((auth_metrics['successful_auths'] or 0) / total_auths * 100) if total_auths > 0 else 100
            
            return {
                'time_range_minutes': time_range_minutes,
                'authentication': {
                    'successful': auth_metrics['successful_auths'] or 0,
                    'failed': auth_metrics['failed_auths'] or 0,
                    'success_rate': round(success_rate, 2)
                },
                'rate_limiting': {
                    'violations': auth_metrics['rate_limit_hits'] or 0
                },
                'sessions': {
                    'active': sessions_count['active_sessions'] or 0
                },
                'api_keys': {
                    'active': api_keys_count['active_keys'] or 0
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def get_security_dashboard(self) -> Dict[str, Any]:
        """
        Get a comprehensive security dashboard
        
        Returns:
            Dictionary with security metrics and alerts
        """
        try:
            # Get metrics for different time ranges
            last_hour = await self.get_metrics_summary(60)
            last_24h = await self.get_metrics_summary(24 * 60)
            
            # Get recent alerts
            db_pool = await get_db_pool()
            recent_alerts = await db_pool.fetchall(
                """
                SELECT action, details, created_at
                FROM audit_logs
                WHERE action = 'metric_security_alert'
                ORDER BY created_at DESC
                LIMIT 10
                """
            )
            
            alerts_list = []
            for alert in recent_alerts:
                try:
                    details = json.loads(alert['details'])
                    alerts_list.append({
                        'type': details.get('labels', {}).get('alert_type'),
                        'severity': details.get('labels', {}).get('severity'),
                        'timestamp': alert['created_at']
                    })
                except:
                    pass
            
            return {
                'metrics': {
                    'last_hour': last_hour,
                    'last_24_hours': last_24h
                },
                'recent_alerts': alerts_list,
                'health_status': self._calculate_health_status(last_hour),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get security dashboard: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _calculate_health_status(self, metrics: Dict[str, Any]) -> str:
        """Calculate overall health status based on metrics"""
        if 'error' in metrics:
            return 'unknown'
        
        # Check various health indicators
        auth_success_rate = metrics.get('authentication', {}).get('success_rate', 100)
        rate_violations = metrics.get('rate_limiting', {}).get('violations', 0)
        
        if auth_success_rate < 50 or rate_violations > 100:
            return 'critical'
        elif auth_success_rate < 80 or rate_violations > 50:
            return 'warning'
        else:
            return 'healthy'


#######################################################################################################################
#
# Module Functions
#

# Global monitor instance
_monitor: Optional[AuthNZMonitor] = None

async def get_authnz_monitor() -> AuthNZMonitor:
    """Get the AuthNZ monitor singleton"""
    global _monitor
    if not _monitor:
        _monitor = AuthNZMonitor()
    return _monitor

async def record_auth_attempt(
    method: str,
    success: bool,
    ip_address: Optional[str] = None,
    user_id: Optional[int] = None,
    reason: Optional[str] = None
):
    """Convenience function to record authentication attempt"""
    monitor = await get_authnz_monitor()
    
    metric_type = MetricType.AUTH_SUCCESS if success else MetricType.AUTH_FAILURE
    await monitor.record_metric(
        metric_type,
        labels={'method': method},
        metadata={
            'ip_address': ip_address,
            'user_id': user_id,
            'reason': reason
        }
    )

async def record_api_key_operation(
    operation: str,
    key_id: int,
    user_id: int
):
    """Convenience function to record API key operation"""
    monitor = await get_authnz_monitor()
    
    metric_map = {
        'created': MetricType.API_KEY_CREATED,
        'rotated': MetricType.API_KEY_ROTATED,
        'revoked': MetricType.API_KEY_REVOKED
    }
    
    metric_type = metric_map.get(operation)
    if metric_type:
        await monitor.record_metric(
            metric_type,
            metadata={'key_id': key_id, 'user_id': user_id}
        )

async def record_rate_limit_violation(
    identifier: str,
    endpoint: str,
    ip_address: Optional[str] = None
):
    """Convenience function to record rate limit violation"""
    monitor = await get_authnz_monitor()
    
    await monitor.record_metric(
        MetricType.RATE_LIMIT_HIT,
        labels={'endpoint': endpoint},
        metadata={'identifier': identifier, 'ip_address': ip_address}
    )

#
# End of monitoring.py
#######################################################################################################################