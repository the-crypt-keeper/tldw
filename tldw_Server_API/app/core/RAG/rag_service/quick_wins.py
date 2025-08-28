# quick_wins.py
"""
Quick win features for the RAG service.

This module provides immediately useful features like spell checking,
result highlighting, cost tracking, and query templates.
"""

import os
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
from collections import defaultdict

from loguru import logger
from spellchecker import SpellChecker
import tiktoken


# 1. QUERY SPELL CHECKING
class QuerySpellChecker:
    """Spell checker for queries."""
    
    def __init__(self, custom_dictionary: Optional[List[str]] = None):
        """
        Initialize spell checker.
        
        Args:
            custom_dictionary: Custom words to add to dictionary
        """
        self.spell_checker = SpellChecker()
        
        # Add custom dictionary
        if custom_dictionary:
            self.spell_checker.word_frequency.load_words(custom_dictionary)
        
        # Add common technical terms
        self.add_technical_terms()
    
    def add_technical_terms(self):
        """Add common technical terms to dictionary."""
        terms = [
            "api", "json", "xml", "html", "css", "javascript", "python",
            "database", "sql", "nosql", "mongodb", "redis", "elasticsearch",
            "docker", "kubernetes", "microservices", "serverless",
            "machine", "learning", "neural", "network", "embedding",
            "vector", "rag", "llm", "gpt", "bert", "transformer"
        ]
        self.spell_checker.word_frequency.load_words(terms)
    
    def check_query(self, query: str) -> Dict[str, Any]:
        """
        Check query for spelling errors.
        
        Args:
            query: Query to check
            
        Returns:
            Dictionary with corrections and suggestions
        """
        words = query.split()
        misspelled = self.spell_checker.unknown(words)
        
        corrections = {}
        for word in misspelled:
            # Get correction suggestions
            suggestions = self.spell_checker.candidates(word)
            if suggestions:
                # Get most likely correction
                correction = self.spell_checker.correction(word)
                corrections[word] = {
                    "correction": correction,
                    "suggestions": list(suggestions)[:5]
                }
        
        # Generate corrected query
        corrected_query = query
        for word, data in corrections.items():
            corrected_query = corrected_query.replace(word, data["correction"])
        
        return {
            "original": query,
            "corrected": corrected_query if corrections else query,
            "has_errors": bool(corrections),
            "corrections": corrections
        }
    
    def auto_correct(self, query: str, confidence_threshold: float = 0.8) -> str:
        """
        Automatically correct query if confidence is high.
        
        Args:
            query: Query to correct
            confidence_threshold: Minimum confidence for auto-correction
            
        Returns:
            Corrected query
        """
        result = self.check_query(query)
        
        if result["has_errors"]:
            # For now, return corrected query
            # In production, would calculate confidence
            return result["corrected"]
        
        return query


# 2. RESULT HIGHLIGHTING
class ResultHighlighter:
    """Highlights matched terms in search results."""
    
    def __init__(self, highlight_tag: str = "**", case_sensitive: bool = False):
        """
        Initialize highlighter.
        
        Args:
            highlight_tag: Tag to use for highlighting (e.g., "**" for markdown)
            case_sensitive: Whether to match case
        """
        self.highlight_tag = highlight_tag
        self.case_sensitive = case_sensitive
    
    def highlight_document(
        self,
        document: str,
        query_terms: List[str],
        context_window: int = 50
    ) -> Dict[str, Any]:
        """
        Highlight query terms in document.
        
        Args:
            document: Document content
            query_terms: Terms to highlight
            context_window: Characters of context around matches
            
        Returns:
            Highlighted document with metadata
        """
        highlighted = document
        matches = []
        
        for term in query_terms:
            # Find all occurrences
            pattern = re.escape(term)
            flags = 0 if self.case_sensitive else re.IGNORECASE
            
            for match in re.finditer(pattern, document, flags):
                matches.append({
                    "term": term,
                    "start": match.start(),
                    "end": match.end(),
                    "context": self._extract_context(document, match.start(), match.end(), context_window)
                })
        
        # Sort matches by position (reverse for replacement)
        matches.sort(key=lambda x: x["start"], reverse=True)
        
        # Apply highlights
        for match in matches:
            start = match["start"]
            end = match["end"]
            highlighted = (
                highlighted[:start] +
                f"{self.highlight_tag}{highlighted[start:end]}{self.highlight_tag}" +
                highlighted[end:]
            )
        
        return {
            "highlighted_text": highlighted,
            "matches": len(matches),
            "snippets": self._extract_snippets(document, matches, context_window)
        }
    
    def _extract_context(
        self,
        document: str,
        start: int,
        end: int,
        window: int
    ) -> str:
        """Extract context around a match."""
        context_start = max(0, start - window)
        context_end = min(len(document), end + window)
        
        context = document[context_start:context_end]
        
        # Add ellipsis if truncated
        if context_start > 0:
            context = "..." + context
        if context_end < len(document):
            context = context + "..."
        
        return context
    
    def _extract_snippets(
        self,
        document: str,
        matches: List[Dict],
        window: int
    ) -> List[str]:
        """Extract relevant snippets from document."""
        snippets = []
        seen_ranges = []
        
        for match in matches[:5]:  # Limit to 5 snippets
            start = match["start"]
            end = match["end"]
            
            # Check if this range overlaps with seen ranges
            overlap = False
            for seen_start, seen_end in seen_ranges:
                if start < seen_end and end > seen_start:
                    overlap = True
                    break
            
            if not overlap:
                snippet = self._extract_context(document, start, end, window)
                snippets.append(snippet)
                seen_ranges.append((start - window, end + window))
        
        return snippets


# 3. COST TRACKING
@dataclass
class LLMCost:
    """LLM API cost information."""
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    input_cost_per_1k: float
    output_cost_per_1k: float
    total_cost: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CostTracker:
    """Tracks LLM API costs."""
    
    # Pricing per 1K tokens (update as needed)
    PRICING = {
        "openai": {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        },
        "anthropic": {
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        },
        "groq": {
            "mixtral-8x7b": {"input": 0.00027, "output": 0.00027},
            "llama-70b": {"input": 0.00059, "output": 0.00079},
        }
    }
    
    def __init__(self):
        """Initialize cost tracker."""
        self.costs: List[LLMCost] = []
        self.total_by_provider = defaultdict(float)
        self.total_by_model = defaultdict(float)
        
        # Initialize tokenizer for counting
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
    
    def track_cost(
        self,
        provider: str,
        model: str,
        input_text: str,
        output_text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LLMCost:
        """
        Track cost for an LLM call.
        
        Args:
            provider: LLM provider
            model: Model name
            input_text: Input text
            output_text: Output text
            metadata: Additional metadata
            
        Returns:
            Cost information
        """
        # Count tokens
        input_tokens = self.count_tokens(input_text)
        output_tokens = self.count_tokens(output_text)
        
        # Get pricing
        pricing = self.PRICING.get(provider, {}).get(model, {"input": 0.001, "output": 0.001})
        input_cost_per_1k = pricing["input"]
        output_cost_per_1k = pricing["output"]
        
        # Calculate cost
        input_cost = (input_tokens / 1000) * input_cost_per_1k
        output_cost = (output_tokens / 1000) * output_cost_per_1k
        total_cost = input_cost + output_cost
        
        # Create cost record
        cost = LLMCost(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost_per_1k=input_cost_per_1k,
            output_cost_per_1k=output_cost_per_1k,
            total_cost=total_cost,
            metadata=metadata or {}
        )
        
        # Track
        self.costs.append(cost)
        self.total_by_provider[provider] += total_cost
        self.total_by_model[f"{provider}/{model}"] += total_cost
        
        logger.debug(
            f"LLM cost: {provider}/{model} - "
            f"Input: {input_tokens} tokens (${input_cost:.4f}), "
            f"Output: {output_tokens} tokens (${output_cost:.4f}), "
            f"Total: ${total_cost:.4f}"
        )
        
        return cost
    
    def get_summary(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """Get cost summary."""
        costs_to_analyze = self.costs[-last_n:] if last_n else self.costs
        
        if not costs_to_analyze:
            return {"total_cost": 0, "call_count": 0}
        
        total_cost = sum(c.total_cost for c in costs_to_analyze)
        total_input_tokens = sum(c.input_tokens for c in costs_to_analyze)
        total_output_tokens = sum(c.output_tokens for c in costs_to_analyze)
        
        return {
            "total_cost": total_cost,
            "call_count": len(costs_to_analyze),
            "avg_cost_per_call": total_cost / len(costs_to_analyze),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "by_provider": dict(self.total_by_provider),
            "by_model": dict(self.total_by_model),
            "last_updated": self.costs[-1].timestamp.isoformat() if self.costs else None
        }


# 4. DEBUG MODE
class DebugMode:
    """Debug mode controller."""
    
    def __init__(self):
        """Initialize debug mode from environment."""
        self.enabled = os.getenv("RAG_DEBUG", "false").lower() == "true"
        self.verbose_components = os.getenv("RAG_DEBUG_COMPONENTS", "").split(",")
        
        if self.enabled:
            logger.info("🐛 Debug mode enabled")
            if self.verbose_components:
                logger.info(f"Verbose components: {self.verbose_components}")
    
    def is_enabled(self, component: Optional[str] = None) -> bool:
        """Check if debug is enabled for component."""
        if not self.enabled:
            return False
        
        if component:
            return not self.verbose_components or component in self.verbose_components
        
        return True
    
    def log(self, component: str, message: str, data: Optional[Any] = None):
        """Log debug message if enabled for component."""
        if self.is_enabled(component):
            if data:
                logger.debug(f"[{component}] {message}: {json.dumps(data, default=str)}")
            else:
                logger.debug(f"[{component}] {message}")


# 6. QUERY TEMPLATES
class QueryTemplate:
    """Predefined query template."""
    
    def __init__(self, name: str, pattern: str, variables: List[str], description: str = ""):
        """
        Initialize query template.
        
        Args:
            name: Template name
            pattern: Template pattern with {variables}
            variables: List of variable names
            description: Template description
        """
        self.name = name
        self.pattern = pattern
        self.variables = variables
        self.description = description
    
    def format(self, **kwargs) -> str:
        """Format template with variables."""
        missing = [v for v in self.variables if v not in kwargs]
        if missing:
            raise ValueError(f"Missing variables: {missing}")
        
        return self.pattern.format(**kwargs)
    
    def validate(self, query: str) -> Tuple[bool, Dict[str, str]]:
        """Check if query matches template and extract variables."""
        # Simple pattern matching (would be more sophisticated in production)
        # Convert template pattern to regex
        regex_pattern = self.pattern
        for var in self.variables:
            regex_pattern = regex_pattern.replace(f"{{{var}}}", f"(?P<{var}>.*?)")
        
        match = re.match(regex_pattern, query)
        if match:
            return True, match.groupdict()
        
        return False, {}


class QueryTemplateLibrary:
    """Library of query templates."""
    
    def __init__(self):
        """Initialize with default templates."""
        self.templates = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default query templates."""
        templates = [
            QueryTemplate(
                "comparison",
                "What is the difference between {item1} and {item2}?",
                ["item1", "item2"],
                "Compare two items"
            ),
            QueryTemplate(
                "definition",
                "What is {term}?",
                ["term"],
                "Define a term"
            ),
            QueryTemplate(
                "how_to",
                "How do I {action}?",
                ["action"],
                "How-to question"
            ),
            QueryTemplate(
                "troubleshooting",
                "Why is {system} {problem}?",
                ["system", "problem"],
                "Troubleshooting question"
            ),
            QueryTemplate(
                "best_practice",
                "What are the best practices for {topic}?",
                ["topic"],
                "Best practices question"
            ),
            QueryTemplate(
                "explanation",
                "Explain {concept} in {context}",
                ["concept", "context"],
                "Explanation request"
            ),
            QueryTemplate(
                "timeline",
                "What happened to {subject} between {start_date} and {end_date}?",
                ["subject", "start_date", "end_date"],
                "Timeline query"
            ),
            QueryTemplate(
                "pros_cons",
                "What are the pros and cons of {option}?",
                ["option"],
                "Pros and cons analysis"
            )
        ]
        
        for template in templates:
            self.templates[template.name] = template
    
    def add_template(self, template: QueryTemplate):
        """Add a custom template."""
        self.templates[template.name] = template
    
    def match_query(self, query: str) -> Optional[Tuple[QueryTemplate, Dict[str, str]]]:
        """Find matching template for query."""
        for template in self.templates.values():
            matches, variables = template.validate(query)
            if matches:
                return template, variables
        
        return None
    
    def suggest_templates(self, partial_query: str) -> List[QueryTemplate]:
        """Suggest templates based on partial query."""
        suggestions = []
        
        for template in self.templates.values():
            # Simple substring matching
            if any(word in template.pattern.lower() for word in partial_query.lower().split()):
                suggestions.append(template)
        
        return suggestions[:5]


# 8. BATCH JOB WEBHOOKS
class WebhookNotifier:
    """Sends webhook notifications for batch jobs."""
    
    def __init__(self, default_timeout: int = 30):
        """
        Initialize webhook notifier.
        
        Args:
            default_timeout: Default timeout for webhook calls
        """
        self.default_timeout = default_timeout
        self.webhook_history = []
    
    async def send_notification(
        self,
        webhook_url: str,
        event_type: str,
        data: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Send webhook notification.
        
        Args:
            webhook_url: URL to send notification to
            event_type: Type of event
            data: Event data
            headers: Optional headers
            
        Returns:
            Success status
        """
        import aiohttp
        
        payload = {
            "event": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=payload,
                    headers=headers or {},
                    timeout=aiohttp.ClientTimeout(total=self.default_timeout)
                ) as response:
                    success = response.status < 400
                    
                    # Log history
                    self.webhook_history.append({
                        "url": webhook_url,
                        "event": event_type,
                        "status": response.status,
                        "timestamp": datetime.now(),
                        "success": success
                    })
                    
                    if success:
                        logger.info(f"Webhook sent successfully to {webhook_url}")
                    else:
                        logger.warning(f"Webhook failed with status {response.status}")
                    
                    return success
                    
        except Exception as e:
            logger.error(f"Webhook error: {e}")
            
            self.webhook_history.append({
                "url": webhook_url,
                "event": event_type,
                "error": str(e),
                "timestamp": datetime.now(),
                "success": False
            })
            
            return False
    
    async def notify_batch_complete(
        self,
        webhook_url: str,
        job_id: str,
        success_rate: float,
        total_queries: int,
        processing_time: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Send batch completion notification."""
        await self.send_notification(
            webhook_url,
            "batch_complete",
            {
                "job_id": job_id,
                "success_rate": success_rate,
                "total_queries": total_queries,
                "processing_time": processing_time,
                "metadata": metadata or {}
            }
        )
    
    async def notify_batch_failed(
        self,
        webhook_url: str,
        job_id: str,
        error: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Send batch failure notification."""
        await self.send_notification(
            webhook_url,
            "batch_failed",
            {
                "job_id": job_id,
                "error": error,
                "metadata": metadata or {}
            }
        )


# Global instances
_spell_checker = None
_highlighter = None
_cost_tracker = None
_debug_mode = None
_template_library = None
_webhook_notifier = None


def get_spell_checker() -> QuerySpellChecker:
    """Get global spell checker instance."""
    global _spell_checker
    if _spell_checker is None:
        _spell_checker = QuerySpellChecker()
    return _spell_checker


def get_highlighter() -> ResultHighlighter:
    """Get global highlighter instance."""
    global _highlighter
    if _highlighter is None:
        _highlighter = ResultHighlighter()
    return _highlighter


def get_cost_tracker() -> CostTracker:
    """Get global cost tracker instance."""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker


def get_debug_mode() -> DebugMode:
    """Get global debug mode instance."""
    global _debug_mode
    if _debug_mode is None:
        _debug_mode = DebugMode()
    return _debug_mode


def get_template_library() -> QueryTemplateLibrary:
    """Get global template library instance."""
    global _template_library
    if _template_library is None:
        _template_library = QueryTemplateLibrary()
    return _template_library


def get_webhook_notifier() -> WebhookNotifier:
    """Get global webhook notifier instance."""
    global _webhook_notifier
    if _webhook_notifier is None:
        _webhook_notifier = WebhookNotifier()
    return _webhook_notifier


# Pipeline integration functions

async def spell_check_query(context: Any, **kwargs) -> Any:
    """Spell check and correct query in pipeline."""
    if not context.config.get("spell_check", {}).get("enabled", True):
        return context
    
    checker = get_spell_checker()
    debug = get_debug_mode()
    
    result = checker.check_query(context.query)
    
    if result["has_errors"]:
        debug.log("spell_check", "Found spelling errors", result["corrections"])
        
        if context.config.get("spell_check", {}).get("auto_correct", True):
            context.metadata["original_query_before_correction"] = context.query
            context.query = result["corrected"]
            context.metadata["spell_corrections"] = result["corrections"]
            logger.info(f"Auto-corrected query: {context.query}")
    
    return context


async def highlight_results(context: Any, **kwargs) -> Any:
    """Highlight query terms in results."""
    if not context.config.get("highlighting", {}).get("enabled", True):
        return context
    
    highlighter = get_highlighter()
    
    # Extract query terms
    query_terms = context.query.lower().split()
    
    # Highlight each document
    for doc in context.documents:
        if hasattr(doc, "content"):
            result = highlighter.highlight_document(
                doc.content,
                query_terms,
                context_window=context.config.get("highlighting", {}).get("context_window", 50)
            )
            
            # Add highlighted version to metadata
            doc.metadata["highlighted"] = result["highlighted_text"]
            doc.metadata["match_count"] = result["matches"]
            doc.metadata["snippets"] = result["snippets"]
    
    return context


async def track_llm_cost(context: Any, **kwargs) -> Any:
    """Track LLM costs in pipeline."""
    if not context.config.get("cost_tracking", {}).get("enabled", True):
        return context
    
    tracker = get_cost_tracker()
    
    # Check if we have generation info
    if "generation" in context.metadata:
        gen_info = context.metadata["generation"]
        
        # Track cost
        cost = tracker.track_cost(
            provider=gen_info.get("provider", "unknown"),
            model=gen_info.get("model", "unknown"),
            input_text=context.query,
            output_text=context.response if hasattr(context, "response") else "",
            metadata={"query_id": context.metadata.get("query_id")}
        )
        
        context.metadata["cost"] = {
            "total": cost.total_cost,
            "input_tokens": cost.input_tokens,
            "output_tokens": cost.output_tokens
        }
    
    return context