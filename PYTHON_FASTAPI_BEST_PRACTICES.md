Python FastAPI Best Practices for Web Apps (mid-2025 Edition by Jeffrey Emanuel)

* uv and a venv targeting only python 3.13 and higher (NOT pip/poetry/conda!); key commands to use for this are:
	- uv venv --python 3.13
	- uv lock --upgrade
	- uv sync --all-extras

* pyproject.toml with hatchling build system; ruff for linter and mypy for type checking;

* .envrc file containing `source .venv/bin/activate` (for direnv)

* setup.sh script for automating all that stuff targeting ubuntu 25

* All settings handled via .env file using the python-decouple library; key pattern to always use:

	```python
	from decouple import Config as DecoupleConfig, RepositoryEnv
	decouple_config = DecoupleConfig(RepositoryEnv(".env"))
	POSTGRES_URL = decouple_config("DATABASE_URL")
	```

* fastapi for backend; automatic generation of openapi.json file so we can do automatic client/model generation for the separate frontend (a different project entirely using nextjs 15). Fastapi routes must be fully documented and use response models (using sqlmodel library) 

* sqlmodel/sqlalchemy for database connecting to postgres; alembic for db migrations. Database operations should be as efficient as possible; batch operations should use batch insertions where possible (same with reads); we should create all relevant database indexes to optimize the access patterns we care about, and create views where it simplifies the code and improves performance.

* where it would help a lot and make sense in the overall flow of logic and be complementary, we should liberally use redis to speed things up.

* typer library used for any CLI (including detailed help)

* rich library used for all console output; really leveraging all the many powerful features to make everything looks extremely slick, polished, colorful, detailed; syntax highlighting for json, progress bars, rounded boxes, live panels (be careful about having more than one live panel at once!), etc.

* uvicorn with uvloop for serving (to be reverse proxied from NGINX)

* For key functionality in the app and key dependencies (e.g., postgres database, redis, elastic search, openai API, etc) we want to "fail fast" so we can address core bugs and problems, not hide issues and try to recover gracefully from everything.

* Async for absolutely everything: all network activity (use httpx); all file access (use aiofiles); all database operations (sqlmodel/sqlalchemy/psycopg2); etc.

* No unit tests or mocks; no fake/generated data; always REAL data, REAL API calls, and REAL, REALISTIC, ACTUAL END TO END INTEGRATION TESTS. All integration tests should feature super detailed and informative logging using the rich library. 

* Aside from the allowed ruff exceptions specified in the pyproject.toml file, we must always strive for ZERO ruff linter warnings/errors in the entire project, as well as ZERO mypy warnings/errors!

* Network requests (especial API calls to third party services like OpenAI, Gemini, Anthropic, etc. should be properly rate limited with semaphores and use robust retry with exponential backoff and random jitter. Where possible, we should always try to do network requests in parallel using asyncio.gather() and similar design patterns (using the semaphores to prevent rate limiting issues automatically).

* Usage of AI APIs should either get precise token length estimates using official APIs or should use the tiktoken library and the relevant tokenizer, never estimate using simplistic rules of thumb. We should always carefully track and monitor and report (using rich console output) the total costs of using APIs since the last startup of the app, for the most recent operations, etc. and track approximate run-rate of spend per day using extrapolation.

* Code should be sensibly organized by functional areas using customary and typical code structures to make it easy and familiar to navigate. But we don't want extreme fragmentation and proliferation of tiny code files! It's about striking the right balance so we don't end up with excessively long and monolithic code files but so we also don't have dozens and dozens of code files with under 50 lines each!


Here is a sample complete pyproject.toml file showing the basic structure of an example application:

```
# pyproject.toml - SmartEdgar.ai Configuration

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "smartedgar"
version = "0.1.0"
description = "SEC EDGAR filing downloader and processor with API"
readme = "README.md"
requires-python = ">=3.13"
license = { text = "MIT" }
authors = [
    { name = "SmartEdgar Team", email = "info@smartedgar.ai" },
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Financial and Insurance Industry",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.13",
    "Topic :: Office/Business :: Financial :: Investment",
    "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
    "Topic :: Text Processing :: Indexing",
    "Framework :: FastAPI",
    "Typing :: Typed",
]

# Core dependencies
dependencies = [
    # Web framework and server
    "fastapi >= 0.120.0",
    "uvicorn[standard] >= 0.35.0",
    # Async operations and HTTP
    "aiofiles >= 23.2.0",
    "aiohttp[speedups] >= 3.9.0",
    "aiohttp-retry >= 2.8.0",
    "aioh2 >= 0.2.0",
    "aiolimiter >= 1.1.0",
    "aiosqlite >= 0.19.0",
    "httpx[http2] >= 0.25.0",
    # Data processing and validation
    "beautifulsoup4 >= 4.12.0",
    "lxml >= 4.9.0",
    "html2text >= 2020.1.0",
    "html5lib >= 1.1",
    "pydantic >= 2.7.0",
    "python-decouple>=3.8",
    "pandas >= 2.0.0",
    # Database and ORM
    "sqlalchemy >= 2.0.41",
    "sqlmodel >= 0.0.15",
    # Text processing and NLP
    "tiktoken >= 0.5.0",
    "nltk >= 3.8.0",
    "fuzzywuzzy >= 0.18.0",
    "python-Levenshtein >= 0.20.0",
    "tenacity >= 8.2.0",
    # PDF and document processing
    "PyMuPDF >= 1.23.0",
    "PyPDF2 >= 3.0.0",
    "pdf2image >= 1.16.0",
    "Pillow >= 10.0.0",
    # Word document processing
    "python-docx >= 1.1.0",
    "mammoth >= 1.8.0",
    # PowerPoint processing
    "python-pptx >= 1.0.0",
    # RTF processing
    "striprtf >= 0.0.26",
    # Text encoding detection
    "chardet >= 5.2.0",
    # Excel and data formats
    "openpyxl >= 3.1.0",
    "xlsx2html >= 0.4.0",
    "markitdown >= 0.1.0",
    # XBRL processing
    "arelle-release >= 2.37.0",
    "tidyxbrl >= 1.2.0",
    # Caching and performance
    "redis[hiredis] >= 5.3.0",
    "cachetools >= 5.3.0",
    # Console output and CLI
    "rich>=13.7.0",
    "typer >= 0.15.0",
    "prompt_toolkit >= 3.0.0",
    "colorama >= 0.4.0",
    "termcolor >= 2.3.0",
    # Progress and utilities
    "tqdm >= 4.66.0",
    "psutil >= 5.9.0",
    "tabulate >= 0.9.0",
    "structlog >= 23.0.0",
    # Networking and scraping
    "scrapling >= 0.2.0",
    "sec-cik-mapper >= 2.1.0",
    # Machine learning and AI
    "torch >= 2.1.0",
    "transformers >= 4.35.0",
    "aisuite[all] >= 0.1.0",
    # Development and code quality
    "ruff>=0.9.0",
    "mypy >= 1.7.0",
    # Monitoring and profiling
    "yappi >= 1.4.0",
    "nvidia-ml-py3 >= 7.352.0",
    # Integration and protocols
    "mcp[cli] >= 1.5.0",
    "fastapi-mcp>=0.3.4",
    "google-genai",
    "tiktoken",
    "scipy>=1.15.3",
    "numpy>=2.2.6",
    "cryptography>=45.0.3",
    "pyyaml>=6.0.2",
    "watchdog>=6.0.0",
    "pytrends",
    "pandas-ta>=0.3.14b0",
    "scikit-learn",
    "statsmodels",
    "backtesting",
    "defusedxml",
    "ciso8601",
    "holidays",
    "matplotlib>=3.5.0",
    "seaborn>=0.11.0",
    "plotly>=5.0.0",
    "networkx",
    "authlib>=1.5.2",
    "jinja2>=3.1.6",
    "itsdangerous>=2.2.0",
    "openai",
    "elasticsearch>=9.0.0,<10.0.0",
    "pyjwt",
    "httpx-oauth",
    "arelle>=2.2",
    "alembic",
    "brotli>=1.1.0",
    "psycopg2-binary>=2.9.10",
    "sqlalchemy-utils>=0.41.2",
    "pgcli>=4.3.0",
    "asyncpg>=0.30.0",
    "user-agents>=2.2.0",
    "types-aiofiles>=24.1.0.20250606",
    "types-pyyaml>=6.0.12.20250516",
    "types-cachetools>=6.0.0.20250525",
    "orjson>=3.11.2,<4",
    "opentelemetry-instrumentation-fastapi>=0.45",
    "testcontainers>=4.0",
]

[project.optional-dependencies]
# Heavy ML dependencies (optional for basic functionality)
ml = [
    "ray >= 2.40.0",
    "flashinfer-python < 0.2.3",
]

# Interactive tools
interactive = [
    "streamlit >= 1.22.0",
    "ipython >= 8.0.0",
]

# Development dependencies
dev = [
    "pytest >= 7.4.0",
    "pytest-asyncio >= 0.21.0",
    "pytest-cov >= 4.1.0",
    "black >= 23.0.0",
    "pre-commit >= 3.0.0",
]

# All optional dependencies
all = [
    "smartedgar[ml,interactive,dev]",
]

[project.scripts]
smartedgar = "smartedgar.cli.main:main"

[project.urls]
Homepage = "https://github.com/Dicklesworthstone/smartedgar"
Repository = "https://github.com/Dicklesworthstone/smartedgar"
Issues = "https://github.com/Dicklesworthstone/smartedgar/issues"
Documentation = "https://github.com/Dicklesworthstone/smartedgar#readme"

# Configure Hatchling
[tool.hatch.build.targets.wheel]
packages = ["smartedgar"]

[tool.hatch.build.targets.sdist]
exclude = [
    "/.venv",
    "/.vscode", 
    "/.git",
    "/.github",
    "/__pycache__",
    "/*.pyc",
    "/*.pyo", 
    "/*.pyd",
    "*.db",
    "*.db-journal",
    "*.db-wal",
    "*.db-shm",
    ".env",
    "tests/*",
    "docs/*", 
    "*.log",
    "sec_filings/*",  # Exclude downloaded filings
    "logs/*",         # Exclude logs
    "old_code/*",     # Exclude archived code
    "*.gz",           # Exclude gzipped files
    ".DS_Store",
    "cache.db",
    "fonts/*",        # Exclude font files
    "static/*.html",  # Exclude demo files
]

# --- Tool Configurations ---

[tool.ruff]
line-length = 150
target-version = "py313"

[tool.ruff.lint]
select = [
    "E",     # pycodestyle errors
    "W",     # pycodestyle warnings  
    "F",     # pyflakes
    "I",     # isort
    "C4",    # flake8-comprehensions
    "B",     # flake8-bugbear
    "A",     # flake8-builtins
    "RUF",   # Ruff-specific rules
    "ASYNC", # flake8-async
    "FA",    # flake8-future-annotations
    "SIM",   # flake8-simplify
    "TID",   # flake8-tidy-imports
    "PTH",   # flake8-use-pathlib
    "RUF100", # Ruff-specific rule for unused noqa
]
extend-select = [
    "A005",   # stdlib-module-shadowing
    "A006",   # builtin-lambda-argument-shadowing  
    "FURB188", # slice-to-remove-prefix-or-suffix
    "PLR1716", # boolean-chained-comparison
    "RUF032",  # decimal-from-float-literal
    "RUF033",  # post-init-default
    "RUF034",  # useless-if-else
]
ignore = [
    "E501",  # Line too long (handled by formatter)
    "E402",  # Module level import not at top of file
    "B008",  # Do not perform function calls in argument defaults
    "B007",  # Loop control variable not used
    "A003",  # Class attribute shadowing builtin (needed for pydantic)
    "SIM108", # Use ternary operator (sometimes less readable)
    "W293",  # Blank lines contain whitespace
    "RUF003", # Ambiguous characters in comments
]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
docstring-code-format = true
# 2025 style guide presets
line-ending = "lf"
skip-magic-trailing-comma = false
docstring-code-line-length = "dynamic"

[tool.ruff.lint.isort]
known-first-party = ["smartedgar"]
combine-as-imports = true

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["S101", "I001"]  # Allow assert in tests and ignore import formatting in tests
"old_code/*" = ["E", "W", "F", "I", "C4", "B", "A", "RUF"]  # Ignore archived code

# Black configuration removed - use Ruff format instead

[tool.mypy]
python_version = "3.13"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true
strict_optional = true
disallow_untyped_defs = false  # Start permissive for large codebase
disallow_incomplete_defs = true
check_untyped_defs = true
no_implicit_optional = true

# Performance optimizations (mypy 1.16+)
sqlite_cache = true
cache_fine_grained = true
incremental = true

[[tool.mypy.overrides]]
module = "streamlit.*"
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = [
    "--strict-markers",
    "--cov=smartedgar",
    "--cov-report=term-missing",
    "--cov-report=html",
]

[tool.coverage.run]
source = ["smartedgar"]
omit = ["tests/*", "*/conftest.py", "old_code/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError", 
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]

# --- uv specific configuration ---
[tool.uv]

[dependency-groups]
dev = [
    "types-aiofiles>=24.1.0.20250606",
    "types-cachetools>=6.0.0.20250525",
    "types-python-dateutil>=2.9.0.20250516",
    "types-pyyaml>=6.0.12.20250516",
]
# uv will use the Python version from requires-python by default

# ---------------------------------------------------------------
# Installation Instructions with uv:
# ---------------------------------------------------------------
# 1. Update .python-version to use Python 3.13:
#    echo "3.13" > .python-version
#
# 2. Create virtual environment:
#    uv venv --python 3.13
#
#    # For experimental free-threaded Python (GIL-less, mid-2025):
#    uv python install 3.13.0-ft
#    uv venv .venv --python 3.13.0-ft
#
# 3. Activate virtual environment:
#    source .venv/bin/activate  # On Unix/macOS
#
# 4. Install the project:
#    uv sync                    # Basic install
#    uv sync --extra interactive # Add Streamlit/IPython
#    uv sync --extra ml         # Add ML dependencies  
#    uv sync --all-extras       # Install everything
#
#    # Run tools without installing in .venv:
#    uvx ruff check .           # Linting via uvx (alias: uv tool run)
#    uvx black .                # Formatting without polluting environment
#
# 5. Set up environment variables:
#    cp .env.example .env
#    # Edit .env with your configuration
#
# 6. Initialize the system:
#    smartedgar setup
#
# 7. Run the API server:
#    smartedgar api
#    # Or: uvicorn smartedgar.api.main:app --reload
#
# 8. Other commands:
#    smartedgar --help          # Show all commands
#    smartedgar status          # Check system health
#    smartedgar download        # Download SEC filings
#    smartedgar dashboard       # Start Streamlit dashboard
# ---------------------------------------------------------------

``` 

## Advanced uv Features

**Leverage uv's tool functionality** to install and run linters/formatters in isolation without polluting your app's virtual environment:

```bash
uv tool install ruff
uv tool install black
uv tool run ruff check .
```

**For monorepo projects**, uv provides workspace management capabilities. It supports constraint dependencies and override dependencies for complex dependency scenarios. The tool can also automatically install Python versions including experimental JIT and free-threaded builds for Python 3.13+.

• **Workspace initialization**: Use `uv workspace init` to create a Cargo-style workspace for multi-package monorepos, allowing shared dependencies and coordinated builds across packages.
• **Free-threaded Python**: Install GIL-less CPython with `uv python install 3.13.0-ft` to experiment with true parallelism in CPU-bound workloads (requires careful testing as not all packages are compatible yet).

## Project Structure for Scale

When your application grows beyond a few modules, consider **domain-based organization**:

```
smartedgar/
├── api/
│   ├── main.py          # FastAPI app creation
│   └── middleware.py    # Custom ASGI middleware
├── filings/
│   ├── router.py        # Domain-specific routes
│   ├── models.py        # SQLModel definitions
│   ├── service.py       # Business logic
│   └── repository.py    # Database queries
├── analytics/
│   ├── router.py
│   ├── models.py
│   └── service.py
└── common/
    └── dependencies.py  # Shared FastAPI dependencies
```

Each domain package contains its own router, models, and services, keeping related code together while maintaining clear boundaries.

## FastAPI Performance Optimizations

**Use lifespan context manager** instead of deprecated `@app.on_event` decorators (FastAPI 0.120.0+):

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await connect_to_database()
    await initialize_redis_pool()
    yield
    # Shutdown
    await disconnect_database()
    await close_redis_pool()

app = FastAPI(lifespan=lifespan)
```

**Avoid BaseHTTPMiddleware** - it introduces 2-3x performance overhead under load. Use raw ASGI middleware instead:

```python
async def timing_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

app.add_middleware(timing_middleware)
```

**Use orjson for 3x faster JSON serialization**:

```python
from fastapi.responses import ORJSONResponse

app = FastAPI(default_response_class=ORJSONResponse)
```

**Optimize response models** to reduce serialization overhead:

```python
@router.get("/items", 
    response_model_exclude_unset=True,
    response_model_by_alias=False  # For internal APIs
)
async def list_items():
    # Only serializes set fields
```

**For streaming AI responses** with token counting:

```python
@router.post("/chat", response_class=StreamingResponse)
async def chat_stream(request: ChatRequest):
    async def generate():
        async for chunk in ai_client.stream(
            messages=request.messages,
            stream_options={"include_usage": True}
        ):
            yield f"data: {json.dumps(chunk)}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

## Advanced Database Patterns

**Use SQLModel's async-first patterns** (v0.0.15+):

```python
from sqlmodel import SQLModel, create_engine
from sqlmodel.ext.asyncio.session import AsyncSession, AsyncEngine
from sqlalchemy.ext.asyncio import create_async_engine

# Create async engine using SQLModel patterns
engine = AsyncEngine(create_async_engine(
    "postgresql+asyncpg://user:pass@localhost/db",
    echo=False,
    future=True
))

# Use AsyncSession from sqlmodel.ext.asyncio
async def get_session() -> AsyncSession:
    async with AsyncSession(engine) as session:
        yield session
```

**Configure connection pool** for optimal async performance:

```python
from sqlalchemy.pool import AsyncAdaptedQueuePool

engine = create_async_engine(
    "postgresql+asyncpg://...",
    poolclass=AsyncAdaptedQueuePool,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600,
    query_cache_size=1200,
    connect_args={
        "server_settings": {
            "application_name": "smartedgar",
            "jit": "off"  # Disable for OLTP workloads
        },
        "command_timeout": 60,
        # asyncpg 0.30+ defaults to statement_cache_size=256
        # Set to 0 only for dynamic SQL edge cases
        "statement_cache_size": 256,  # ~15% throughput gain
    }
)
```

**Use PostgreSQL COPY for bulk operations** achieving 100,000-500,000 rows/second:

```python
async def bulk_insert_with_copy(df: pd.DataFrame, table_name: str):
    from io import StringIO
    import csv
    
    buffer = StringIO()
    df.to_csv(buffer, index=False, header=False, quoting=csv.QUOTE_MINIMAL)
    buffer.seek(0)
    
    async with engine.raw_connection() as conn:
        async with conn.cursor() as cursor:
            await cursor.copy_expert(
                f"COPY {table_name} FROM STDIN WITH CSV",
                buffer
            )
```

**For bulk updates, use the UNNEST pattern**:

```python
async def bulk_update_filings(updates: list[dict]):
    stmt = text("""
        UPDATE filings
        SET status = updates.status
        FROM (
            SELECT * FROM UNNEST(
                :ids::integer[],
                :statuses::text[]
            ) AS t(id, status)
        ) AS updates
        WHERE filings.id = updates.id
    """)
    
    await session.execute(stmt, {
        "ids": [u["id"] for u in updates],
        "statuses": [u["status"] for u in updates]
    })
```

**Leverage PostgreSQL's JSON capabilities** for complex aggregations:

```python
# Return nested JSON directly from database
query = text("""
    SELECT json_build_object(
        'company', c.name,
        'filings', json_agg(
            json_build_object(
                'id', f.id,
                'type', f.type,
                'filed_at', f.filed_at
            ) ORDER BY f.filed_at DESC
        )
    ) as data
    FROM companies c
    LEFT JOIN filings f ON c.id = f.company_id
    GROUP BY c.id
""")
```

**Use AsyncAttrs mixin** for safe lazy loading in async contexts:

```python
from sqlalchemy.ext.asyncio import AsyncAttrs

class Base(AsyncAttrs, SQLModel):
    pass

class Filing(Base, table=True):
    # Now supports await filing.awaitable_attrs.company
    company: Company = Relationship(back_populates="filings")
```

## Redis Advanced Patterns

**Note: redis-py now includes async support** (aioredis is deprecated):

```python
from redis import asyncio as redis

# Connection pool with circuit breaker
class RedisClient:
    def __init__(self):
        self.pool = redis.ConnectionPool(
            host="localhost",
            max_connections=50,
            socket_keepalive=True,
            socket_keepalive_options={
                1: 1,  # TCP_KEEPIDLE
                2: 2,  # TCP_KEEPINTVL
                3: 3,  # TCP_KEEPCNT
            }
        )
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60
        )
    
    async def get(self, key: str):
        if not self.circuit_breaker.is_closed:
            return None  # Fail gracefully
            
        try:
            async with redis.Redis(connection_pool=self.pool) as r:
                return await r.get(key)
        except redis.RedisError:
            self.circuit_breaker.record_failure()
            raise
```

**Use Lua scripts for atomic operations**:

```python
# Sliding window rate limiter with microsecond precision
rate_limit_script = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

redis.call('ZREMRANGEBYSCORE', key, 0, now - window * 1000000)
local count = redis.call('ZCARD', key)

if count < limit then
    redis.call('ZADD', key, now, now)
    redis.call('EXPIRE', key, window)
    return 1
else
    return 0
end
"""

async def check_rate_limit(user_id: str, limit: int = 100, window: int = 60):
    async with redis.Redis(connection_pool=redis_pool) as r:
        allowed = await r.eval(
            rate_limit_script,
            1,
            f"rate:{user_id}",
            limit,
            window,
            int(time.time() * 1000000)
        )
        return bool(allowed)
```

**Leverage Redis Streams** for event-driven architectures:

```python
async def publish_event(stream: str, event: dict):
    async with redis.Redis(connection_pool=redis_pool) as r:
        await r.xadd(stream, {"data": json.dumps(event)})

async def consume_events(stream: str, consumer_group: str):
    async with redis.Redis(connection_pool=redis_pool) as r:
        while True:
            messages = await r.xreadgroup(
                consumer_group, "consumer1", {stream: ">"}, count=10
            )
            for stream_name, stream_messages in messages:
                for msg_id, data in stream_messages:
                    yield json.loads(data[b"data"])
                    await r.xack(stream_name, consumer_group, msg_id)
```

## Rich Console Advanced Features

**Optimize Live displays** with appropriate refresh rates:

```python
from rich.live import Live
from rich.table import Table

# Use fixed console width to prevent recalculation
console = Console(width=120)

# Configure refresh rate for smooth updates without overwhelming resources
with Live(
    table,
    refresh_per_second=4,  # 4-10 FPS is optimal
    transient=True,
    console=console
) as live:
    for update in data_stream:
        table.add_row(update)
        live.update(table)
```

**Memory-efficient rendering** for large datasets:

```python
from rich.console import Console
from rich.text import Text

console = Console()

# Use pagination for large outputs
with console.pager():
    for line in massive_dataset:
        console.print(line)

# Stream output instead of building entire string
def stream_json_pretty(data: dict):
    for line in json.dumps(data, indent=2).splitlines():
        yield Text(line, style="json")
        
console.print(stream_json_pretty(large_json))
```

## Integration Testing Patterns

**Use TestContainers v4.0+** for real service testing (mid-2025 improvements):
- v4.0 adds typed async APIs and parallel network mode
- ~40% faster test suite execution on GitHub Actions
- Module-scoped Container.start() context manager for better resource management

```python
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer
import pytest
```

**Transaction-per-test pattern** for test isolation:

```python
@pytest.fixture
async def db_session(test_db):
    """Each test runs in a transaction that's rolled back"""
    async with test_db.begin() as transaction:
        yield test_db
        await transaction.rollback()
```

**Parallel test execution** with pytest-xdist:

```bash
# Run tests in parallel with 4 workers
pytest -n 4 --dist loadscope

# Configure in pyproject.toml
[tool.pytest.ini_options]
addopts = [
    "-n", "auto",  # Auto-detect CPU count
    "--dist", "loadscope",  # Group by module
]
```

## AI API Token Management

**Provider-specific token counting**:

```python
# OpenAI with tiktoken
encoding_gpt4o = tiktoken.encoding_for_model("gpt-4o")  # o200k_base
encoding_gpt4 = tiktoken.encoding_for_model("gpt-4")    # cl100k_base

# Anthropic (use their API)
async def count_anthropic_tokens(text: str):
    response = await anthropic_client.count_tokens(
        model="claude-3-opus",
        messages=[{"role": "user", "content": text}]
    )
    return response.usage.input_tokens

# Google Gemini (character-based)
def estimate_gemini_tokens(text: str):
    # Gemini uses ~4 characters per token on average
    return len(text) / 4
```

**Request coalescing** for similar queries:

```python
from collections import defaultdict
import asyncio

class AIRequestCoalescer:
    def __init__(self, wait_time: float = 0.1):
        self.pending = defaultdict(list)
        self.wait_time = wait_time
    
    async def request(self, prompt: str, callback):
        prompt_hash = hash(prompt)
        
        # Add to pending requests
        future = asyncio.Future()
        self.pending[prompt_hash].append(future)
        
        # If first request for this prompt, process after wait_time
        if len(self.pending[prompt_hash]) == 1:
            asyncio.create_task(self._process_batch(prompt_hash, prompt))
        
        return await future
    
    async def _process_batch(self, prompt_hash: str, prompt: str):
        await asyncio.sleep(self.wait_time)
        
        # Make single API call
        response = await ai_client.complete(prompt)
        
        # Resolve all waiting futures
        for future in self.pending[prompt_hash]:
            future.set_result(response)
        
        del self.pending[prompt_hash]
```

## Production Deployment

**Gunicorn configuration** with uvicorn workers:

```python
# gunicorn.conf.py
import multiprocessing

bind = "0.0.0.0:8000"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000

# Restart workers after this many requests to prevent memory leaks
max_requests = 1000
max_requests_jitter = 50

# Graceful shutdown
graceful_timeout = 30
timeout = 60

# Access logs with timing
accesslog = "-"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'
```

**Graceful shutdown handling**:

```python
import signal
import asyncio
from contextlib import asynccontextmanager

shutdown_event = asyncio.Event()

def signal_handler(signum, frame):
    shutdown_event.set()

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown
    # Wait for ongoing requests to complete
    await asyncio.sleep(0.5)
    
    # Close connections gracefully
    await redis_pool.disconnect()
    await engine.dispose()

app = FastAPI(lifespan=lifespan)
```

**Health check endpoints** with dependency verification:

```python
from datetime import datetime

startup_time = datetime.now()

@app.get("/health/liveness")
async def liveness():
    return {
        "status": "alive",
        "uptime": (datetime.now() - startup_time).total_seconds()
    }

@app.get("/health/readiness")
async def readiness(
    db: AsyncSession = Depends(get_db),
    redis_client = Depends(get_redis)
):
    checks = {
        "database": "unknown",
        "redis": "unknown",
        "external_api": "unknown"
    }
    
    # Check database
    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = "healthy"
    except Exception:
        checks["database"] = "unhealthy"
    
    # Check Redis
    try:
        await redis_client.ping()
        checks["redis"] = "healthy"
    except Exception:
        checks["redis"] = "unhealthy"
    
    # Check external API
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get("https://api.openai.com/v1/models")
            checks["external_api"] = "healthy" if resp.status_code == 200 else "degraded"
    except Exception:
        checks["external_api"] = "unhealthy"
    
    all_healthy = all(v == "healthy" for v in checks.values())
    return {
        "status": "ready" if all_healthy else "degraded",
        "checks": checks
    }
```

## Dockerization

**Multi-stage build** for lean production images (mid-2025 pattern):

```dockerfile
# Dockerfile
# Stage 1: Builder
FROM python:3.13-slim-bookworm as builder

# Set up the environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    UV_SYSTEM_PYTHON=true

WORKDIR /app

# Install uv
RUN pip install uv

# Copy only the dependency configuration files
COPY pyproject.toml uv.lock* ./

# Install dependencies into a virtual environment
RUN uv venv .venv && \
    source .venv/bin/activate && \
    uv sync --all-extras

# ---
# Stage 2: Final Image
FROM python:3.13-slim-bookworm as final

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-root user for security
RUN addgroup --system app && adduser --system --group app
USER app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv ./.venv

# Copy the application source code
COPY ./smartedgar ./smartedgar
COPY gunicorn.conf.py .

# Make the virtual environment's executables available
ENV PATH="/app/.venv/bin:$PATH"

# Expose the port the app runs on
EXPOSE 8000

# The command to run the application using gunicorn
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-c", "gunicorn.conf.py", "smartedgar.api.main:app"]
```

## CI/CD with GitHub Actions

**Automated testing and linting** for every commit:

```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]

jobs:
  test-and-lint:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.13", "3.14-dev"]  # Pre-test Python 3.14 beta (Oct 2025)
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Cache uv dependencies
        uses: actions/cache@v3
        with:
          path: |
            ~/.cache/uv
            .venv
          key: ${{ runner.os }}-uv-${{ matrix.python-version }}-${{ hashFiles('uv.lock') }}
          restore-keys: |
            ${{ runner.os }}-uv-${{ matrix.python-version }}-

      - name: Install uv
        run: pip install uv

      - name: Install dependencies
        run: uv sync --all-extras

      - name: Lint with Ruff
        run: uvx ruff check .

      - name: Format check with Ruff
        run: uvx ruff format . --check

      - name: Type check with MyPy
        run: uv run mypy .

      - name: Run tests with Pytest
        run: uv run pytest

      - name: Security audit
        run: uv run pip-audit
```

## Monitoring and Observability

**Structured logging with correlation IDs**:

```python
from asgi_correlation_id import CorrelationIdMiddleware
import structlog

# Add correlation ID middleware
app.add_middleware(CorrelationIdMiddleware)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

# Use in routes
logger = structlog.get_logger()

@app.get("/process")
async def process_filing(filing_id: int):
    logger.info("processing_filing", filing_id=filing_id)
    # Correlation ID automatically included in all logs
```

**Prometheus metrics integration**:

```python
from prometheus_client import Counter, Histogram, Gauge
import prometheus_client

# Define metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

http_request_duration = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    ["method", "endpoint"]
)

active_connections = Gauge(
    "active_connections",
    "Number of active connections"
)

# Track metrics in middleware
@app.middleware("http")
async def track_metrics(request: Request, call_next):
    active_connections.inc()
    
    with http_request_duration.labels(
        method=request.method,
        endpoint=request.url.path
    ).time():
        response = await call_next(request)
    
    http_requests_total.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    active_connections.dec()
    return response

# Expose metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(
        prometheus_client.generate_latest(),
        media_type="text/plain"
    )
```

## Security Enhancements

**Regular dependency auditing**:

```bash
# Add to your CI/CD pipeline
pip install safety
safety check

# Or use pip-audit
pip install pip-audit
pip-audit
```

**Secure headers middleware**:

```python
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.sessions import SessionMiddleware

# Security headers
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

# Trusted host validation
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["example.com", "*.example.com"]
)
```

## Type Checking Enhancements

**Mypy 1.16+ performance improvements** (2.2x faster with orjson cache):

```toml
[tool.mypy]
python_version = "3.13"
strict = true
plugins = [
    "pydantic.mypy",
    "sqlalchemy.ext.mypy.plugin"
]

# Performance optimizations
cache_fine_grained = true
incremental = true
sqlite_cache = true
```

**Find and use type stub libraries where available!**

For many popular libraries, there are already community provided type stub libraries available from PyPi for projects that don't yet provide types directly in the library itself. Here is a sampling of some of these:

```
    "asyncpg-stubs >=0",
    "elasticsearch-stubs",
    "pandas-stubs>=2.2.3.250527",
    "sqlalchemy2-stubs >=0",
    "types-aiofiles>=24.1.0.20250606",
    "types-beautifulsoup4>=4.12.0",
    "types-cachetools>=6.0.0.20250525",
    "types-lxml>=2024.8.7",
    "types-openpyxl>=3.1.5",
    "types-passlib>=1.7.7",
    "types-psutil>=7.0.0",
    "types-python-dateutil>=2.9.0.20250516",
    "types-python-jose>=3.5.0.20250531",
    "types-pyyaml>=6.0.12.20250516",
    "types-pytz>0",
    "types-redis>=4.6.0.20241004",
    "types-reportlab",
    "types-requests>=2.32.4.20250611",
    "types-setuptools>=80.9.0.20250529",
    "types-sqlalchemy>=1.4.53.38",
    "types-tabulate>=0.9.0",
    "types-termcolor",
    "types-tqdm>=4.67.0",
```

Wherever it is relevant and such libraries exist (you always need to search online to verify that a proposed library actually is available under that exact name in PyPi!), you should use them rather than try to create your own ad-hoc, incomplete type specifications.


## Ruff 2025 Style Guide Updates

**Adopt the new Ruff 2025 style guide** (v0.9.0+) for enhanced readability:

```python
# F-string formatting - now breaks expressions intelligently
print(f"Processing {
    very_long_variable_name_that_exceeds_line_limit
} items")

# Implicit string concatenation - merges single-line strings
message = "This is a long message that continues seamlessly."

# Assertion messages - wraps message in parentheses
assert condition, (
    "Detailed error message stays together"
)
```

**Enable newly stabilized lint rules**:
```toml
[tool.ruff.lint]
extend-select = [
    "A005",   # stdlib-module-shadowing
    "A006",   # builtin-lambda-argument-shadowing  
    "FURB188", # slice-to-remove-prefix-or-suffix
    "PLR1716", # boolean-chained-comparison
    "RUF032",  # decimal-from-float-literal
    "RUF033",  # post-init-default
    "RUF034",  # useless-if-else
]
```

## FastAPI Router-Level Configuration

**Reduce endpoint duplication** with router-level dependencies:

```python
# Common authentication for all endpoints in router
admin_router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin_role)],
    responses={403: {"description": "Insufficient permissions"}}
)

# All routes automatically require admin role
@admin_router.get("/users")
async def list_users():
    # No need to add Depends(require_admin_role) here
    return await get_all_users()
```

## Service Layer Architecture Pattern

**Implement clean separation of concerns** beyond simple routers:

```python
# services/filing_service.py - Business logic layer
class FilingService:
    def __init__(self, db: AsyncSession, redis: Redis):
        self.db = db
        self.redis = redis
        self.repository = FilingRepository(db)
    
    async def process_filing(self, filing_id: int) -> Filing:
        # Orchestrate complex business logic
        filing = await self.repository.get(filing_id)
        
        # Check cache first
        cached = await self.redis.get(f"filing:{filing_id}:processed")
        if cached:
            return Filing.parse_raw(cached)
        
        # Process with multiple steps
        filing = await self._validate_filing(filing)
        filing = await self._enrich_filing(filing)
        filing = await self._calculate_metrics(filing)
        
        # Update database and cache
        await self.repository.update(filing)
        await self.redis.setex(
            f"filing:{filing_id}:processed",
            3600,
            filing.json()
        )
        return filing

# api/endpoints/filings.py - Presentation layer
@router.post("/filings/{filing_id}/process")
async def process_filing(
    filing_id: int,
    service: FilingService = Depends(get_filing_service)
):
    # Router only handles HTTP concerns
    result = await service.process_filing(filing_id)
    return FilingResponse.from_orm(result)
```

## Advanced Async Patterns

**Structured concurrency with asyncio.TaskGroup** (Python 3.11+):

```python
import asyncio

# Replace ad-hoc gather() with TaskGroup for better error handling
async def process_many_items(items: list) -> list:
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(process_item(item)) for item in items]
    
    # All tasks complete successfully or TaskGroup cancels all on first failure
    return [task.result() for task in tasks]

# Compare with the old pattern:
# results = await asyncio.gather(*tasks, return_exceptions=True)  # Continues even if some fail
```

**Create a utility wrapper around asyncio.Runner** for CLI entry points:

```python
# common/async_utils.py
import asyncio
from typing import Coroutine, TypeVar

T = TypeVar('T')

def run_async(coro: Coroutine[None, None, T]) -> T:
    """Run async code with proper context propagation (Python 3.13+)"""
    with asyncio.Runner() as runner:
        return runner.run(coro)

# Use in Typer CLI instead of asyncio.run()
import typer
from .common.async_utils import run_async

app = typer.Typer()

@app.command()
def process(file: str):
    result = run_async(process_file_async(file))
    typer.echo(f"Processed: {result}")
```

**Rate limiting with semaphores** for external APIs:

```python
# Limit concurrent API calls
api_semaphore = asyncio.Semaphore(10)

async def call_external_api(endpoint: str):
    async with api_semaphore:  # Automatically queues if limit reached
        async with httpx.AsyncClient() as client:
            return await client.get(f"https://api.example.com/{endpoint}")

# Process many requests without overwhelming the API
results = await asyncio.gather(*[
    call_external_api(f"item/{i}") for i in range(100)
])
```

**Graceful error handling** in concurrent operations:

```python
async def process_batch_with_failures(items: list):
    tasks = [process_item(item) for item in items]
    
    # Don't fail everything if one task fails
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successful = []
    failed = []
    
    for item, result in zip(items, results):
        if isinstance(result, Exception):
            failed.append((item, str(result)))
        else:
            successful.append(result)
    
    logger.info(f"Processed {len(successful)} successfully, {len(failed)} failed")
    return successful, failed
```

**Integrate blocking code** safely:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Create a thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=4)

async def process_pdf(pdf_path: str):
    loop = asyncio.get_event_loop()
    
    # Run blocking PDF processing in thread pool
    result = await loop.run_in_executor(
        executor,
        extract_text_from_pdf,  # Blocking function
        pdf_path
    )
    
    return result
```

## Task Queue with arq

**Production-ready background jobs** with arq (async-native alternative to Celery):

```python
# worker.py
from arq import Worker
from arq.connections import RedisSettings

async def send_email(ctx, user_id: int, subject: str, body: str):
    """Background task to send emails"""
    user = await get_user(user_id)
    await email_client.send(
        to=user.email,
        subject=subject,
        body=body
    )
    logger.info(f"Email sent to {user.email}")

async def generate_report(ctx, report_id: int):
    """Long-running report generation"""
    redis: Redis = ctx["redis"]
    
    # Update progress in Redis
    await redis.set(f"report:{report_id}:status", "processing")
    
    # Generate report...
    result = await create_complex_report(report_id)
    
    await redis.set(f"report:{report_id}:status", "completed")
    return result

# Worker configuration
class WorkerSettings:
    functions = [send_email, generate_report]
    redis_settings = RedisSettings(host="localhost", port=6379)
    max_jobs = 10
    job_timeout = 300

# api/endpoints/reports.py - Enqueue jobs
from arq import create_pool

@router.post("/reports")
async def create_report(request: ReportRequest):
    pool = await create_pool(RedisSettings())
    
    # Enqueue job and return immediately
    job = await pool.enqueue_job(
        "generate_report",
        report_id=request.id,
        _job_try=3  # Retry up to 3 times
    )
    
    return {
        "job_id": job.job_id,
        "status": "queued"
    }
```

**Run the worker**:
```bash
arq worker.WorkerSettings
```

## WebSocket + Redis Pub/Sub for Real-time Features

**Scalable WebSocket implementation** across multiple servers:

```python
from fastapi import WebSocket
import aioredis
from broadcaster import Broadcast

# Initialize broadcaster with Redis backend
broadcast = Broadcast("redis://localhost:6379")

# Use lifespan context manager instead of deprecated @app.on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await broadcast.connect()
    yield
    # Shutdown
    await broadcast.disconnect()

app = FastAPI(lifespan=lifespan)

# WebSocket endpoint
@app.websocket("/ws/{channel}")
async def websocket_endpoint(websocket: WebSocket, channel: str):
    await websocket.accept()
    
    async def receive_broadcasts():
        async with broadcast.subscribe(channel) as subscriber:
            async for event in subscriber:
                await websocket.send_json(event.message)
    
    async def receive_websocket():
        async for data in websocket.iter_json():
            # Broadcast to all subscribers
            await broadcast.publish(channel, data)
    
    # Run both concurrently
    await asyncio.gather(
        receive_broadcasts(),
        receive_websocket()
    )

# Publish events from regular endpoints
@app.post("/notify/{channel}")
async def send_notification(channel: str, message: dict):
    await broadcast.publish(channel, message)
    return {"status": "sent"}
```

## OpenTelemetry Distributed Tracing

**Full observability stack with zero-instrumentation shims** (mid-2025):

Note: OpenTelemetry instrumentation shims stabilized in spring 2025. One-line `instrument_app()` now covers FastAPI, SQLAlchemy async, redis-py, and httpx, giving you distributed traces without custom middleware. Latency impact is <1 µs/request when the exporter batches.

```python
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure OpenTelemetry
def setup_telemetry(app: FastAPI):
    # Set up OTLP exporter (for Jaeger, Tempo, etc.)
    otlp_exporter = OTLPSpanExporter(
        endpoint="http://localhost:4317",
        insecure=True
    )
    
    # Configure tracer
    provider = TracerProvider()
    processor = BatchSpanProcessor(otlp_exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    
    # Auto-instrument libraries with zero LOC overhead (2025 pattern)
    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
    SQLAlchemyInstrumentor().instrument(engine=engine)
    RedisInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument()

# Custom spans for business logic
tracer = trace.get_tracer(__name__)

async def process_complex_operation(data: dict):
    with tracer.start_as_current_span("process_operation") as span:
        span.set_attribute("operation.type", data["type"])
        span.set_attribute("operation.size", len(data["items"]))
        
        # Nested span for sub-operation
        with tracer.start_as_current_span("validate_data"):
            validated = await validate(data)
        
        with tracer.start_as_current_span("save_to_database"):
            result = await save(validated)
            span.set_attribute("db.rows_affected", result.rowcount)
        
        return result
```

## Alembic Team Collaboration Best Practices

**Prevent migration conflicts** in team environments:

```bash
# Always pull latest migrations before creating new ones
git pull origin main
alembic current  # Check current state
alembic history  # Review migration chain

# Create migration with descriptive message
alembic revision --autogenerate -m "add_filing_status_index"

# ALWAYS review autogenerated migrations
# Alembic can miss: table renames, column type changes, complex constraints
```

**Migration review checklist**:
```python
"""Add filing status index

Revision ID: abc123
Revises: def456
Create Date: 2025-01-15 10:00:00

CHECKLIST:
[ ] Reviewed autogenerated SQL
[ ] Added missing operations (renames, custom types)
[ ] Tested upgrade AND downgrade
[ ] Considered performance impact
[ ] Added concurrent index creation for large tables
"""

def upgrade():
    # Use CONCURRENTLY for zero-downtime index creation
    op.execute(
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
        "ix_filings_status ON filings(status)"
    )

def downgrade():
    op.execute("DROP INDEX IF EXISTS ix_filings_status")
```

## Production Deployment Architecture

**Three-layer deployment pattern** explanation:

```nginx
# nginx.conf - Layer 1: Edge proxy
upstream app_servers {
    # Unix socket for same-machine communication (faster than TCP)
    server unix:/tmp/gunicorn.sock fail_timeout=0;
}

server {
    listen 80;
    listen 443 ssl http2;  # HTTP/2 support with Uvicorn 0.35.0+
    server_name api.example.com;
    
    # Security headers
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    
    # Optimize for API traffic
    client_max_body_size 50M;
    client_body_timeout 60s;
    
    # Compression
    gzip on;
    gzip_types application/json;
    gzip_min_length 1000;
    
    location / {
        proxy_pass http://app_servers;
        proxy_set_header Host $http_host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts for long-running requests
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
```

**systemd service** for production:
```ini
# /etc/systemd/system/smartedgar.service
[Unit]
Description=SmartEdgar API
After=network.target postgresql.service redis.service

[Service]
Type=notify
User=smartedgar
Group=smartedgar
WorkingDirectory=/opt/smartedgar
Environment="PATH=/opt/smartedgar/.venv/bin"
ExecStart=/opt/smartedgar/.venv/bin/gunicorn \
    -c /opt/smartedgar/gunicorn.conf.py \
    smartedgar.api.main:app
    
# Restart policy
Restart=always
RestartSec=10

# Security hardening
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

## Advanced LLM Development Patterns

**Context management for LLM-assisted coding**:

```python
class LLMContext:
    """Manage context window efficiently"""
    
    def __init__(self, max_tokens: int = 8000):
        self.max_tokens = max_tokens
        self.encoder = tiktoken.encoding_for_model("gpt-4")
    
    def build_context(
        self,
        task: str,
        relevant_code: list[str],
        error_messages: list[str] = None,
        examples: list[dict] = None
    ) -> str:
        """Build optimal context within token limits"""
        
        sections = []
        
        # Always include task
        sections.append(f"TASK: {task}")
        
        # Add code context
        if relevant_code:
            sections.append("RELEVANT CODE:")
            for code in relevant_code:
                sections.append(f"```python\n{code}\n```")
        
        # Add errors if present
        if error_messages:
            sections.append("ERRORS TO FIX:")
            sections.extend(error_messages)
        
        # Add examples if space allows
        if examples:
            sections.append("EXAMPLES:")
            for ex in examples:
                test_context = "\n\n".join(sections + [str(ex)])
                if self.count_tokens(test_context) < self.max_tokens:
                    sections.append(str(ex))
                else:
                    break
        
        return "\n\n".join(sections)
    
    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))
```

**Multi-model orchestration** with LangChain:

```python
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.schema import SystemMessage, HumanMessage

class MultiModelOrchestrator:
    def __init__(self):
        self.fast_model = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0
        )
        self.smart_model = ChatAnthropic(
            model="claude-3-opus",
            temperature=0
        )
        self.code_model = ChatOpenAI(
            model="gpt-4",
            temperature=0
        )
    
    async def process(self, task: str) -> str:
        # Route to appropriate model based on task
        if "debug" in task or "fix" in task:
            return await self.smart_model.ainvoke([
                SystemMessage("You are an expert debugger."),
                HumanMessage(task)
            ])
        elif "code" in task or "implement" in task:
            return await self.code_model.ainvoke([
                SystemMessage("You are an expert Python developer."),
                HumanMessage(task)
            ])
        else:
            # Use fast model for simple queries
            return await self.fast_model.ainvoke([HumanMessage(task)])
```

## Documentation Workflow (mid-2025)

**Use MkDocs ≥ 1.6 with material theme** for comprehensive project documentation:

```toml
# mkdocs.yml
site_name: SmartEdgar Documentation
theme:
  name: material
  features:
    - navigation.instant
    - navigation.tracking
    - search.suggest
    - content.code.copy
  palette:
    scheme: slate
    primary: indigo

plugins:
  - search
  - autorefs
  - mkdocstrings:
      handlers:
        python:
          options:
            show_source: true

nav:
  - Home: index.md
  - Architecture: architecture.md
  - API Reference: api.md
  - ADRs: adrs/index.md
```

**Architectural Decision Records (ADRs)** to codify technical decisions:

```markdown
# docs/adrs/001-free-threaded-python.md
# ADR-001: Experimental Free-Threaded Python Support

## Status
Experimental

## Context
Python 3.13 introduced an experimental free-threaded (GIL-less) build that allows true parallelism for CPU-bound workloads. This could significantly improve performance for our data processing pipelines.

## Decision
We will offer an optional free-threaded build configuration for power users who want to experiment with improved parallelism, while maintaining the standard build as the default.

## Consequences
- Potential 2-4x performance improvement for CPU-bound tasks
- Some C extensions may not be compatible
- Requires careful testing before production use
- Must maintain both build configurations
```

This approach ensures future contributors understand the "why" behind technical choices and won't inadvertently reverse important decisions.