"""
Queue backend implementations.
"""

from .factory import (
    create_backend,
    test_backend_connection,
    get_backend_info,
    BackendManager
)

from .sqlite_backend import SQLiteBackend
from .memory_backend import MemoryBackend

# PostgreSQL backend is optional
try:
    from .postgresql_backend import PostgreSQLBackend
    __all__ = [
        'create_backend',
        'test_backend_connection', 
        'get_backend_info',
        'BackendManager',
        'SQLiteBackend',
        'PostgreSQLBackend',
        'MemoryBackend'
    ]
except ImportError:
    __all__ = [
        'create_backend',
        'test_backend_connection',
        'get_backend_info', 
        'BackendManager',
        'SQLiteBackend',
        'MemoryBackend'
    ]