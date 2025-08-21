# Prompt Studio Implementation Plan

## Executive Summary
This plan addresses 41 failing tests and implements missing functionality to achieve >95% test coverage. The implementation is divided into 4 phases over an estimated 3-4 weeks of development.

## Phase 1: Critical Infrastructure & Quick Fixes (2-3 days)
**Goal**: Fix database schema, authentication, and registration issues to enable 30+ more tests to pass

### 1.1 Database Schema Updates (4 hours)
```python
# Location: app/core/DB_Management/PromptStudioDatabase.py
# Add to _create_tables() method

"""
ALTER TABLE prompt_studio_job_queue 
ADD COLUMN project_id INTEGER REFERENCES prompt_studio_projects(id);

CREATE INDEX idx_job_queue_project ON prompt_studio_job_queue(project_id);
"""
```

**Tasks**:
- [ ] Add migration script for existing databases
- [ ] Update job queue table schema
- [ ] Add project_id to job creation methods
- [ ] Fix SQL parameter binding in test_golden_test_cases

### 1.2 API Endpoint Registration (2 hours)
```python
# Location: app/main.py
# Add to application startup

from app.api.v1.endpoints import prompt_studio_evaluations
from app.api.v1.endpoints import prompt_studio_optimizations  
from app.api.v1.endpoints import prompt_studio_websocket

app.include_router(
    prompt_studio_evaluations.router,
    prefix="/api/v1/prompt-studio",
    tags=["prompt-studio-evaluations"]
)

app.include_router(
    prompt_studio_optimizations.router,
    prefix="/api/v1/prompt-studio",
    tags=["prompt-studio-optimizations"]
)

# WebSocket endpoint
app.add_websocket_route(
    "/api/v1/prompt-studio/ws",
    prompt_studio_websocket.websocket_endpoint
)
```

### 1.3 Authentication Test Mode (3 hours)
```python
# Location: app/api/v1/API_Deps/auth_deps.py

def get_current_user_test_mode():
    """Test mode authentication bypass"""
    if os.getenv("TEST_MODE") == "true":
        return User(
            id="test-user",
            username="testuser",
            is_authenticated=True
        )
    return get_current_user()

# Location: tests/prompt_studio/conftest.py
@pytest.fixture(autouse=True)
def enable_test_mode(monkeypatch):
    """Enable test mode for all tests"""
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("CSRF_ENABLED", "false")
```

### 1.4 Fix TestCaseManager Mock Issues (2 hours)
```python
# Location: tests/prompt_studio/unit/test_test_case_manager.py

@pytest.fixture
def mock_test_case():
    """Properly structured test case dict"""
    return {
        "id": 1,
        "uuid": "test-uuid",
        "project_id": 1,
        "name": "Test Case",
        "description": "Description",
        "inputs": '{"input": "value"}',
        "expected_outputs": '{"output": "result"}',
        "tags": "tag1,tag2",
        "is_golden": 1,
        "metadata": '{}'
    }

def test_get_test_case(self, manager, mock_db, mock_test_case):
    mock_db.row_to_dict.return_value = mock_test_case
    # ... rest of test
```

## Phase 2: Core Missing Endpoints (3-4 days)
**Goal**: Implement evaluation, optimization, and WebSocket endpoints

### 2.1 Evaluation Endpoints (1 day)
```python
# Location: app/api/v1/endpoints/prompt_studio_evaluations.py

from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
from ..schemas.prompt_studio_schemas import (
    EvaluationCreate, EvaluationResponse, 
    EvaluationList, EvaluationMetrics
)

router = APIRouter()

@router.post("/evaluations", response_model=EvaluationResponse)
async def create_evaluation(
    evaluation: EvaluationCreate,
    db: PromptStudioDatabase = Depends(get_prompt_studio_db),
    user: User = Depends(get_current_user)
):
    """Create a new evaluation"""
    # Implementation
    
@router.get("/evaluations", response_model=EvaluationList)
async def list_evaluations(
    project_id: int,
    limit: int = 100,
    offset: int = 0,
    db: PromptStudioDatabase = Depends(get_prompt_studio_db)
):
    """List evaluations for a project"""
    # Implementation

@router.get("/evaluations/{evaluation_id}", response_model=EvaluationResponse)
async def get_evaluation(
    evaluation_id: int,
    db: PromptStudioDatabase = Depends(get_prompt_studio_db)
):
    """Get evaluation details"""
    # Implementation
```

### 2.2 Optimization Endpoints (1 day)
```python
# Location: app/api/v1/endpoints/prompt_studio_optimizations.py

@router.post("/optimizations", response_model=OptimizationJobResponse)
async def start_optimization(
    optimization: OptimizationCreate,
    background_tasks: BackgroundTasks,
    db: PromptStudioDatabase = Depends(get_prompt_studio_db)
):
    """Start optimization job"""
    job_id = str(uuid.uuid4())
    
    # Create job in database
    job = db.create_job(
        project_id=optimization.project_id,
        job_type="optimization",
        status="pending",
        config=optimization.dict()
    )
    
    # Queue background task
    background_tasks.add_task(
        run_optimization,
        job_id=job_id,
        config=optimization
    )
    
    return OptimizationJobResponse(
        id=job_id,
        status="pending",
        type="optimization"
    )

@router.get("/optimizations/{job_id}", response_model=OptimizationStatusResponse)
async def get_optimization_status(
    job_id: str,
    db: PromptStudioDatabase = Depends(get_prompt_studio_db)
):
    """Get optimization job status"""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    
    return OptimizationStatusResponse(
        id=job_id,
        status=job["status"],
        progress=job.get("progress", 0),
        current_iteration=job.get("current_iteration"),
        best_score=job.get("best_score")
    )
```

### 2.3 WebSocket Implementation (1-2 days)
```python
# Location: app/api/v1/endpoints/prompt_studio_websocket.py

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set
import asyncio
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Set[str]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
    
    def disconnect(self, client_id: str):
        del self.active_connections[client_id]
        # Clean up subscriptions
    
    async def broadcast(self, message: dict, project_id: str = None):
        """Broadcast to relevant connections"""
        for client_id, websocket in self.active_connections.items():
            if project_id is None or project_id in self.subscriptions.get(client_id, set()):
                await websocket.send_json(message)

manager = ConnectionManager()

async def websocket_endpoint(websocket: WebSocket):
    client_id = str(uuid.uuid4())
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "subscribe":
                project_id = data["project_id"]
                if client_id not in manager.subscriptions:
                    manager.subscriptions[client_id] = set()
                manager.subscriptions[client_id].add(str(project_id))
                
                await websocket.send_json({
                    "type": "subscribed",
                    "project_id": project_id
                })
            
            elif data["type"] == "subscribe_job":
                # Handle job subscription
                pass
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)
```

## Phase 3: Advanced Feature Implementation (5-7 days)
**Goal**: Implement missing prompt generation strategies and validation

### 3.1 Chain-of-Thought Implementation (1 day)
```python
# Location: app/core/Prompt_Management/prompt_studio/prompt_generator.py

def _generate_chain_of_thought(self, task: str, examples: List = None) -> str:
    """Generate chain-of-thought prompt"""
    base_prompt = f"{task}\n\nLet's think step by step:"
    
    if examples:
        base_prompt = f"{task}\n\nExamples of step-by-step reasoning:\n"
        for ex in examples:
            base_prompt += f"\nProblem: {ex['problem']}\n"
            base_prompt += f"Step-by-step solution:\n{ex['solution']}\n"
        base_prompt += f"\nNow, let's solve this step by step:"
    
    return base_prompt

def generate(self, prompt_type: PromptType, task: str, **kwargs):
    if prompt_type == PromptType.CHAIN_OF_THOUGHT:
        return self._generate_chain_of_thought(task, kwargs.get('examples'))
```

### 3.2 Few-Shot Formatting (1 day)
```python
# Location: app/core/Prompt_Management/prompt_studio/prompt_generator.py

def _format_few_shot_examples(self, examples: List[Dict], 
                              input_label: str = "Input",
                              output_label: str = "Output") -> str:
    """Format few-shot examples"""
    if not examples:
        return ""
    
    formatted = "\n\nExamples:\n"
    for ex in examples:
        formatted += f"{input_label}: {ex.get('input', '')}\n"
        formatted += f"{output_label}: {ex.get('output', '')}\n\n"
    
    return formatted

def generate(self, prompt_type: PromptType, task: str, **kwargs):
    if prompt_type == PromptType.FEW_SHOT:
        examples = kwargs.get('examples', [])
        formatted_examples = self._format_few_shot_examples(examples)
        return f"{formatted_examples}{task}"
```

### 3.3 ReAct Pattern Implementation (1 day)
```python
# Location: app/core/Prompt_Management/prompt_studio/prompt_generator.py

def _generate_react_prompt(self, task: str, tools: List[str] = None) -> str:
    """Generate ReAct (Reasoning + Acting) prompt"""
    prompt = f"""Task: {task}

You will solve this task using the following format:

Thought: [Your reasoning about what to do next]
Action: [The action you want to take]
Observation: [The result of the action]
... (repeat Thought/Action/Observation as needed)
Thought: [Final reasoning]
Answer: [Final answer]
"""
    
    if tools:
        prompt += "\nAvailable tools:\n"
        for tool in tools:
            prompt += f"- {tool}\n"
    
    prompt += "\nBegin:\nThought: "
    return prompt
```

### 3.4 Generation Strategies (1-2 days)
```python
# Location: app/core/Prompt_Management/prompt_studio/prompt_generator.py

class GenerationStrategy(Enum):
    CONCISE = "concise"
    DETAILED = "detailed"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    TECHNICAL = "technical"

def _apply_strategy(self, prompt: str, strategy: GenerationStrategy) -> str:
    """Apply generation strategy to prompt"""
    
    if strategy == GenerationStrategy.DETAILED:
        additions = [
            "Provide comprehensive details and explanations.",
            "Include relevant examples and illustrations.",
            "Break down complex concepts into understandable parts.",
            "Address potential edge cases and considerations."
        ]
        return f"{prompt}\n\n" + "\n".join(additions)
    
    elif strategy == GenerationStrategy.CREATIVE:
        additions = [
            "Be creative and think outside the box.",
            "Use vivid descriptions and imaginative scenarios.",
            "Explore unconventional approaches.",
            "Incorporate metaphors and analogies where appropriate."
        ]
        return f"{prompt}\n\n" + "\n".join(additions)
    
    elif strategy == GenerationStrategy.CONCISE:
        return f"{prompt}\n\nBe concise and to the point."
    
    elif strategy == GenerationStrategy.ANALYTICAL:
        additions = [
            "Analyze systematically and logically.",
            "Provide data-driven insights.",
            "Consider multiple perspectives.",
            "Draw evidence-based conclusions."
        ]
        return f"{prompt}\n\n" + "\n".join(additions)
    
    return prompt
```

### 3.5 Input Validation (1 day)
```python
# Location: app/core/Prompt_Management/prompt_studio/prompt_generator.py

def _validate_prompt_type(self, prompt_type: Any) -> PromptType:
    """Validate prompt type"""
    if isinstance(prompt_type, str):
        try:
            return PromptType(prompt_type)
        except ValueError:
            raise ValueError(f"Invalid prompt type: {prompt_type}")
    elif isinstance(prompt_type, PromptType):
        return prompt_type
    else:
        raise ValueError(f"Prompt type must be string or PromptType, got {type(prompt_type)}")

def _validate_variables(self, template: str, variables: Dict) -> None:
    """Validate all required variables are provided"""
    import re
    required_vars = re.findall(r'\{(\w+)\}', template)
    missing = [var for var in required_vars if var not in variables]
    
    if missing:
        raise ValueError(f"Missing required variables: {', '.join(missing)}")

def _validate_few_shot_format(self, examples: Any) -> List[Dict]:
    """Validate few-shot examples format"""
    if not isinstance(examples, list):
        raise ValueError("Examples must be a list")
    
    for i, ex in enumerate(examples):
        if not isinstance(ex, dict):
            raise ValueError(f"Example {i} must be a dictionary")
        if 'input' not in ex or 'output' not in ex:
            raise ValueError(f"Example {i} must have 'input' and 'output' keys")
    
    return examples
```

### 3.6 Dynamic Example Selection (1 day)
```python
# Location: app/core/Prompt_Management/prompt_studio/prompt_generator.py

def _select_dynamic_examples(self, 
                            task: str, 
                            example_pool: List[Dict],
                            n_examples: int = 3,
                            selection_strategy: str = "similarity") -> List[Dict]:
    """Dynamically select best examples for the task"""
    
    if selection_strategy == "similarity":
        # Use embeddings to find most similar examples
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        task_embedding = model.encode([task])
        example_embeddings = model.encode([ex['input'] for ex in example_pool])
        
        # Calculate similarities
        similarities = []
        for i, ex_emb in enumerate(example_embeddings):
            similarity = np.dot(task_embedding[0], ex_emb)
            similarities.append((similarity, example_pool[i]))
        
        # Sort and select top N
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [ex for _, ex in similarities[:n_examples]]
    
    elif selection_strategy == "diversity":
        # Select diverse examples
        import random
        return random.sample(example_pool, min(n_examples, len(example_pool)))
    
    elif selection_strategy == "difficulty":
        # Select by difficulty progression
        sorted_examples = sorted(example_pool, 
                               key=lambda x: x.get('difficulty', 0))
        return sorted_examples[:n_examples]
    
    return example_pool[:n_examples]
```

## Phase 4: Performance & Test Infrastructure (2-3 days)
**Goal**: Optimize performance and improve test infrastructure

### 4.1 Batch Processing Optimization (1 day)
```python
# Location: app/core/Prompt_Management/prompt_studio/prompt_improver.py

async def improve_batch_async(self, 
                             prompts: List[str], 
                             strategy: ImprovementStrategy) -> List[ImprovementResult]:
    """Optimized batch improvement using async"""
    import asyncio
    
    # Process in parallel with limited concurrency
    semaphore = asyncio.Semaphore(5)  # Max 5 concurrent
    
    async def improve_with_limit(prompt):
        async with semaphore:
            return await self.improve_async(prompt, strategy)
    
    tasks = [improve_with_limit(p) for p in prompts]
    results = await asyncio.gather(*tasks)
    
    return results

def improve_batch(self, prompts: List[str], strategy: ImprovementStrategy) -> List[ImprovementResult]:
    """Batch improvement with caching"""
    results = []
    
    # Check cache first
    cached_results = []
    uncached_prompts = []
    
    for prompt in prompts:
        cache_key = f"{prompt}:{strategy.value}"
        if self.enable_cache and cache_key in self.cache:
            cached_results.append((prompt, self.cache[cache_key]))
        else:
            uncached_prompts.append(prompt)
    
    # Process uncached in batch
    if uncached_prompts:
        # Use vectorized operations if possible
        new_results = [self.improve(p, strategy) for p in uncached_prompts]
        results.extend(new_results)
    
    # Add cached results
    results.extend([r for _, r in cached_results])
    
    return results
```

### 4.2 Database Concurrency Improvements (1 day)
```python
# Location: app/core/DB_Management/PromptStudioDatabase.py

def __init__(self, db_path: str, client_id: str):
    """Initialize with WAL mode for better concurrency"""
    super().__init__(db_path, client_id)
    
    # Enable WAL mode for better concurrent access
    conn = self.get_connection()
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")  # 5 second timeout
    conn.commit()
    
def get_connection_pool(self):
    """Connection pool for concurrent access"""
    if not hasattr(self, '_pool'):
        from queue import Queue
        self._pool = Queue(maxsize=10)
        
        # Pre-populate pool
        for _ in range(5):
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            self._pool.put(conn)
    
    return self._pool
```

### 4.3 Test Infrastructure Improvements (1 day)
```python
# Location: tests/prompt_studio/conftest.py

import pytest
import tempfile
import os
from pathlib import Path

@pytest.fixture(scope="session")
def test_db_path():
    """Session-scoped test database"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_prompt_studio.db"
        yield str(db_path)

@pytest.fixture
def isolated_db(test_db_path):
    """Isolated database for each test"""
    from app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase
    
    # Create unique database for this test
    import uuid
    unique_path = f"{test_db_path}_{uuid.uuid4()}.db"
    db = PromptStudioDatabase(unique_path, "test-client")
    
    yield db
    
    # Cleanup
    if os.path.exists(unique_path):
        os.unlink(unique_path)

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset any singleton instances between tests"""
    # Reset connection managers, caches, etc.
    pass
```

## Implementation Timeline

### Week 1
- **Day 1-2**: Phase 1 - Critical Infrastructure
  - Database schema updates
  - API endpoint registration
  - Authentication test mode
  - Fix mock issues
  
- **Day 3-5**: Phase 2 (Part 1) - Core Endpoints
  - Evaluation endpoints
  - Optimization endpoints

### Week 2
- **Day 6-7**: Phase 2 (Part 2) - WebSocket
  - WebSocket implementation
  - Connection manager
  - Event broadcasting
  
- **Day 8-10**: Phase 3 (Part 1) - Prompt Generation
  - Chain-of-thought
  - Few-shot formatting
  - ReAct pattern

### Week 3
- **Day 11-13**: Phase 3 (Part 2) - Advanced Features
  - Generation strategies
  - Input validation
  - Dynamic example selection
  
- **Day 14-15**: Phase 4 - Performance & Testing
  - Batch optimization
  - Database concurrency
  - Test infrastructure

## Success Metrics

### Phase 1 Complete
- Database tests pass (6 fixed)
- Authentication works in tests (11 fixed)
- Mock issues resolved (5 fixed)
- **Total: 22 tests fixed (71.7% → 85.5%)**

### Phase 2 Complete
- Evaluation endpoints work (2 fixed)
- Optimization endpoints work (2 fixed)
- WebSocket connects (2 fixed)
- **Total: 6 tests fixed (85.5% → 89.3%)**

### Phase 3 Complete
- Prompt generation strategies work (8 fixed)
- Input validation works (4 fixed)
- **Total: 12 tests fixed (89.3% → 96.9%)**

### Phase 4 Complete
- Performance test passes (1 fixed)
- **Total: 1 test fixed (96.9% → 97.5%)**

### Final Target
- **155+ tests passing out of 159 (>97% pass rate)**
- All critical functionality implemented
- Performance benchmarks met

## Risk Mitigation

### Technical Risks
1. **WebSocket complexity**: Use existing FastAPI WebSocket support
2. **Database migrations**: Create backup before schema changes
3. **Performance regressions**: Add benchmarks before optimization

### Schedule Risks
1. **Underestimated complexity**: Phase 3 has buffer time
2. **Integration issues**: Test each phase independently
3. **Dependencies**: All phases can be worked on partially in parallel

## Testing Strategy

### Unit Tests First
- Write unit tests for each new feature
- Aim for >90% code coverage
- Use TDD where possible

### Integration Tests
- Test each endpoint with real database
- Test WebSocket with mock clients
- Test authentication flow end-to-end

### Performance Tests
- Benchmark batch operations
- Test concurrent database access
- Monitor memory usage

## Documentation Requirements

### Code Documentation
- Docstrings for all new functions
- Type hints for all parameters
- Example usage in docstrings

### API Documentation
- OpenAPI schemas for all endpoints
- WebSocket protocol documentation
- Authentication flow diagram

### User Documentation
- Update README with new features
- Add usage examples
- Create troubleshooting guide

## Rollout Plan

### Phase 1 Rollout
1. Deploy database migration script
2. Update test environment
3. Verify existing functionality

### Phase 2 Rollout
1. Deploy new endpoints
2. Test with Postman/curl
3. Update API documentation

### Phase 3 Rollout
1. Feature flag for new strategies
2. A/B testing with users
3. Gradual rollout

### Phase 4 Rollout
1. Performance monitoring
2. Load testing
3. Production deployment

## Conclusion

This implementation plan will:
1. Fix all 41 failing tests
2. Implement all missing functionality
3. Improve test infrastructure
4. Optimize performance
5. Achieve >97% test pass rate

The phased approach ensures:
- Quick wins in Phase 1
- Core functionality in Phase 2
- Advanced features in Phase 3
- Polish and optimization in Phase 4

Total estimated time: 15-20 working days
Expected test pass rate: >97%
Risk level: Low to Medium
Confidence level: High
