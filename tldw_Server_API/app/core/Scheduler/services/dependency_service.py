"""
Stateless dependency resolution service.
Efficiently resolves task dependencies without maintaining state.
"""

from typing import List, Dict, Set, Optional
from loguru import logger

from ..base.queue_backend import QueueBackend
from ..base import Task, TaskStatus
from ..base.exceptions import DependencyError


class DependencyService:
    """
    Resolves task dependencies without maintaining any state.
    
    All dependency information is queried from the database on-demand.
    Uses efficient algorithms to minimize database queries.
    """
    
    def __init__(self, backend: QueueBackend):
        """
        Initialize dependency service.
        
        Args:
            backend: Queue backend for database operations
        """
        self.backend = backend
    
    async def get_ready_tasks(self, queue_name: Optional[str] = None) -> List[str]:
        """
        Get tasks that are ready to run (all dependencies satisfied).
        
        This is a stateless operation that queries the database.
        
        Args:
            queue_name: Optional queue filter
            
        Returns:
            List of task IDs ready to run
        """
        return await self.backend.get_ready_tasks()
    
    async def check_dependencies(self, task_id: str) -> bool:
        """
        Check if a task's dependencies are satisfied.
        
        Args:
            task_id: Task ID to check
            
        Returns:
            True if all dependencies are completed
        """
        task = await self.backend.get_task(task_id)
        if not task:
            raise DependencyError(f"Task {task_id} not found")
        
        if not task.depends_on:
            return True
        
        # Check each dependency
        for dep_id in task.depends_on:
            dep_task = await self.backend.get_task(dep_id)
            if not dep_task:
                logger.warning(f"Dependency {dep_id} not found for task {task_id}")
                return False
            
            if dep_task.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    async def get_dependency_graph(self, root_task_id: str) -> Dict[str, List[str]]:
        """
        Build dependency graph starting from a root task.
        
        Args:
            root_task_id: Root task ID
            
        Returns:
            Dictionary mapping task IDs to their dependencies
        """
        graph: Dict[str, List[str]] = {}
        visited: Set[str] = set()
        
        async def traverse(task_id: str):
            if task_id in visited:
                return
            
            visited.add(task_id)
            task = await self.backend.get_task(task_id)
            
            if not task:
                return
            
            graph[task_id] = task.depends_on or []
            
            # Recursively traverse dependencies
            for dep_id in task.depends_on or []:
                await traverse(dep_id)
        
        await traverse(root_task_id)
        return graph
    
    async def detect_circular_dependencies(self, task_id: str) -> bool:
        """
        Check if adding a task would create circular dependencies.
        
        Args:
            task_id: Task ID to check
            
        Returns:
            True if circular dependency detected
        """
        task = await self.backend.get_task(task_id)
        if not task or not task.depends_on:
            return False
        
        # Use DFS to detect cycles
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        
        async def has_cycle(current_id: str) -> bool:
            visited.add(current_id)
            rec_stack.add(current_id)
            
            current_task = await self.backend.get_task(current_id)
            if current_task and current_task.depends_on:
                for dep_id in current_task.depends_on:
                    if dep_id not in visited:
                        if await has_cycle(dep_id):
                            return True
                    elif dep_id in rec_stack:
                        return True
            
            rec_stack.remove(current_id)
            return False
        
        return await has_cycle(task_id)
    
    async def get_dependent_tasks(self, task_id: str) -> List[str]:
        """
        Get tasks that depend on the given task.
        
        This requires scanning all tasks, so use sparingly.
        
        Args:
            task_id: Task ID
            
        Returns:
            List of dependent task IDs
        """
        # This would be more efficient with a reverse index
        # For now, we'd need to add a method to the backend
        logger.warning("get_dependent_tasks not fully implemented")
        return []
    
    async def validate_dependencies(self, task: Task) -> List[str]:
        """
        Validate that all dependencies exist and return any missing ones.
        
        Args:
            task: Task to validate
            
        Returns:
            List of missing dependency IDs
        """
        if not task.depends_on:
            return []
        
        missing = []
        for dep_id in task.depends_on:
            dep_task = await self.backend.get_task(dep_id)
            if not dep_task:
                missing.append(dep_id)
        
        return missing
    
    async def get_execution_order(self, task_ids: List[str]) -> List[str]:
        """
        Get optimal execution order for a set of tasks based on dependencies.
        
        Uses topological sort to determine order.
        
        Args:
            task_ids: List of task IDs
            
        Returns:
            Ordered list of task IDs
        """
        # Build adjacency list
        graph: Dict[str, List[str]] = {}
        in_degree: Dict[str, int] = {}
        
        for task_id in task_ids:
            task = await self.backend.get_task(task_id)
            if not task:
                continue
            
            graph[task_id] = []
            if task_id not in in_degree:
                in_degree[task_id] = 0
            
            if task.depends_on:
                for dep_id in task.depends_on:
                    if dep_id in task_ids:  # Only consider tasks in our set
                        if dep_id not in graph:
                            graph[dep_id] = []
                        graph[dep_id].append(task_id)
                        in_degree[task_id] = in_degree.get(task_id, 0) + 1
                        if dep_id not in in_degree:
                            in_degree[dep_id] = 0
        
        # Topological sort using Kahn's algorithm
        queue = [node for node in in_degree if in_degree[node] == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            for neighbor in graph.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for cycles
        if len(result) != len(in_degree):
            raise DependencyError("Circular dependency detected")
        
        return result