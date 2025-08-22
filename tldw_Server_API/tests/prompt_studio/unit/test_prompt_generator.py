# test_prompt_generator.py
# Unit tests for prompt generator

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import json

from tldw_Server_API.app.core.Prompt_Management.prompt_studio.prompt_generator import (
    PromptGenerator, PromptTemplate, PromptType, GenerationStrategy
)

########################################################################################################################
# Test PromptTemplate Model

class TestPromptTemplate:
    """Test the PromptTemplate model."""
    
    def test_template_creation(self):
        """Test creating a PromptTemplate."""
        template = PromptTemplate(
            name="Test Template",
            type=PromptType.CHAIN_OF_THOUGHT,
            system_template="You are a helpful assistant.",
            user_template="Please {action} the following: {input}",
            variables=["action", "input"],
            metadata={"version": "1.0"}
        )
        
        assert template.name == "Test Template"
        assert template.type == PromptType.CHAIN_OF_THOUGHT
        assert "action" in template.variables
        assert template.metadata["version"] == "1.0"
    
    def test_template_defaults(self):
        """Test PromptTemplate default values."""
        template = PromptTemplate(
            name="Minimal",
            type=PromptType.BASIC,
            user_template="Simple prompt"
        )
        
        assert template.system_template == ""
        assert template.variables == []
        assert template.metadata == {}
        assert template.few_shot_examples == []
    
    def test_template_with_few_shot(self):
        """Test template with few-shot examples."""
        examples = [
            {"input": "2+2", "output": "4"},
            {"input": "3*3", "output": "9"}
        ]
        
        template = PromptTemplate(
            name="Math Template",
            type=PromptType.FEW_SHOT,
            user_template="Calculate: {input}",
            few_shot_examples=examples
        )
        
        assert len(template.few_shot_examples) == 2
        assert template.few_shot_examples[0]["output"] == "4"
    
    def test_template_validation(self):
        """Test template validation."""
        # Valid template
        valid = PromptTemplate(
            name="Valid",
            type=PromptType.BASIC,
            user_template="Test {var}",
            variables=["var"]
        )
        assert valid.validate()
        
        # Invalid - missing variable in template
        with pytest.raises(ValueError):
            invalid = PromptTemplate(
                name="Invalid",
                type=PromptType.BASIC,
                user_template="Test {missing}",
                variables=["different"]
            )
            invalid.validate()

########################################################################################################################
# Test PromptGenerator

class TestPromptGenerator:
    """Test the PromptGenerator class."""
    
    @pytest.fixture
    def generator(self):
        """Create a PromptGenerator instance."""
        return PromptGenerator()
    
    def test_generator_initialization(self):
        """Test PromptGenerator initialization."""
        generator = PromptGenerator()
        assert hasattr(generator, 'templates')
        assert hasattr(generator, 'strategies')
        assert len(generator.templates) > 0  # Should have built-in templates
    
    def test_generate_basic_prompt(self, generator):
        """Test generating a basic prompt."""
        prompt = generator.generate(
            type=PromptType.BASIC,
            task_description="Summarize text",
            variables={"text": "Long article content..."}
        )
        
        assert "system" in prompt
        assert "user" in prompt
        assert isinstance(prompt["system"], str)
        assert isinstance(prompt["user"], str)
    
    def test_generate_chain_of_thought(self, generator):
        """Test generating chain-of-thought prompt."""
        prompt = generator.generate(
            type=PromptType.CHAIN_OF_THOUGHT,
            task_description="Solve a complex problem",
            variables={"problem": "Calculate compound interest"}
        )
        
        assert "step by step" in prompt["user"].lower() or "think through" in prompt["user"].lower()
        assert len(prompt["user"]) > len("Calculate compound interest")
    
    def test_generate_few_shot(self, generator):
        """Test generating few-shot prompt."""
        examples = [
            {"input": "Hello", "output": "Hi there!"},
            {"input": "How are you?", "output": "I'm doing well, thanks!"}
        ]
        
        prompt = generator.generate(
            type=PromptType.FEW_SHOT,
            task_description="Respond to greetings: {input}",
            variables={"input": "Good morning"},
            few_shot_examples=examples
        )
        
        assert "Hello" in prompt["user"]
        assert "Hi there!" in prompt["user"]
        assert "Good morning" in prompt["user"] or "Respond to greetings" in prompt["user"]
    
    def test_generate_react_prompt(self, generator):
        """Test generating ReAct prompt."""
        prompt = generator.generate(
            type=PromptType.REACT,
            task_description="Research a topic",
            variables={"topic": "quantum computing"}
        )
        
        assert "thought" in prompt["user"].lower() or "action" in prompt["user"].lower()
        assert "observation" in prompt["user"].lower() or "result" in prompt["user"].lower()
    
    def test_generate_with_custom_template(self, generator):
        """Test generating with custom template."""
        custom_template = PromptTemplate(
            name="Custom",
            type=PromptType.CUSTOM,
            system_template="You are {role}.",
            user_template="Please {action}: {content}",
            variables=["role", "action", "content"]
        )
        
        generator.add_template(custom_template)
        
        prompt = generator.generate(
            type=PromptType.CUSTOM,
            template_name="Custom",
            variables={
                "role": "a code reviewer",
                "action": "review",
                "content": "this Python function"
            }
        )
        
        assert "You are a code reviewer" in prompt["system"]
        assert "Please review: this Python function" in prompt["user"]
    
    def test_generate_with_strategy(self, generator):
        """Test generating with specific strategy."""
        prompt = generator.generate(
            type=PromptType.BASIC,
            task_description="Write a story",
            strategy=GenerationStrategy.DETAILED,
            variables={"theme": "adventure"}
        )
        
        # Detailed strategy should produce longer prompts
        assert len(prompt["user"]) > 50
    
    def test_generate_with_constraints(self, generator):
        """Test generating with constraints."""
        constraints = [
            "Output must be valid JSON",
            "Include error handling",
            "Maximum 100 tokens"
        ]
        
        prompt = generator.generate(
            type=PromptType.BASIC,
            task_description="Generate configuration",
            constraints=constraints
        )
        
        assert "JSON" in prompt["user"]
        assert "error" in prompt["user"].lower()
        assert "100 tokens" in prompt["user"] or "100" in prompt["user"]
    
    def test_generate_with_modules(self, generator):
        """Test generating with prompt modules."""
        modules = [
            {"type": "thinking", "content": "Consider edge cases"},
            {"type": "format", "content": "Output as markdown"}
        ]
        
        prompt = generator.generate(
            type=PromptType.MODULAR,
            task_description="Analyze code",
            modules=modules
        )
        
        assert "edge cases" in prompt["user"].lower()
        assert "markdown" in prompt["user"].lower()
    
    def test_template_caching(self, generator):
        """Test template caching for performance."""
        # First generation
        prompt1 = generator.generate(
            type=PromptType.BASIC,
            task_description="Task 1",
            use_cache=True
        )
        
        # Second generation with same parameters
        prompt2 = generator.generate(
            type=PromptType.BASIC,
            task_description="Task 1",
            use_cache=True
        )
        
        # Should return cached result (same object)
        assert prompt1 == prompt2
    
    def test_batch_generation(self, generator):
        """Test batch prompt generation."""
        tasks = [
            {"description": "Summarize", "variables": {"text": "Article 1"}},
            {"description": "Translate", "variables": {"text": "Hello"}},
            {"description": "Explain", "variables": {"concept": "AI"}}
        ]
        
        prompts = generator.generate_batch(
            type=PromptType.BASIC,
            tasks=tasks
        )
        
        assert len(prompts) == 3
        assert all("system" in p and "user" in p for p in prompts)
    
    def test_template_composition(self, generator):
        """Test composing multiple templates."""
        composed = generator.compose_templates(
            templates=[
                PromptType.CHAIN_OF_THOUGHT,
                PromptType.FEW_SHOT
            ],
            task_description="Complex reasoning task",
            few_shot_examples=[{"input": "1", "output": "2"}]
        )
        
        assert "step by step" in composed["user"].lower()
        assert "example" in composed["user"].lower()
    
    def test_validate_variables(self, generator):
        """Test variable validation."""
        # Valid variables
        valid = generator.validate_variables(
            template_vars=["var1", "var2"],
            provided_vars={"var1": "value1", "var2": "value2"}
        )
        assert valid is True
        
        # Missing variable
        invalid = generator.validate_variables(
            template_vars=["var1", "var2"],
            provided_vars={"var1": "value1"}
        )
        assert invalid is False
    
    def test_optimize_prompt_length(self, generator):
        """Test prompt length optimization."""
        long_content = "Very " * 1000 + "long content"
        
        prompt = generator.generate(
            type=PromptType.BASIC,
            task_description="Summarize",
            variables={"content": long_content},
            max_length=500
        )
        
        # Should truncate to fit max_length
        total_length = len(prompt["system"]) + len(prompt["user"])
        assert total_length <= 500
    
    def test_generate_with_persona(self, generator):
        """Test generating with specific persona."""
        prompt = generator.generate(
            type=PromptType.BASIC,
            task_description="Explain concept",
            persona="expert data scientist",
            variables={"concept": "neural networks"}
        )
        
        assert "data scientist" in prompt["system"].lower() or "expert" in prompt["system"].lower()
    
    def test_generate_structured_output(self, generator):
        """Test generating prompts for structured output."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            }
        }
        
        prompt = generator.generate(
            type=PromptType.STRUCTURED,
            task_description="Extract information",
            output_schema=schema,
            variables={"text": "John is 30 years old"}
        )
        
        assert "JSON" in prompt["user"] or "json" in prompt["user"]
        assert "name" in prompt["user"]
        assert "age" in prompt["user"]

########################################################################################################################
# Test Generation Strategies

class TestGenerationStrategies:
    """Test different generation strategies."""
    
    @pytest.fixture
    def generator(self):
        """Create generator instance."""
        return PromptGenerator()
    
    def test_concise_strategy(self, generator):
        """Test concise generation strategy."""
        prompt = generator.generate(
            type=PromptType.BASIC,
            task_description="Summarize this text",
            strategy=GenerationStrategy.CONCISE
        )
        
        # Concise prompts should be brief
        assert len(prompt["user"]) < 100
    
    def test_detailed_strategy(self, generator):
        """Test detailed generation strategy."""
        prompt = generator.generate(
            type=PromptType.BASIC,
            task_description="Analyze this data",
            strategy=GenerationStrategy.DETAILED
        )
        
        # Detailed prompts should be comprehensive
        assert len(prompt["user"]) > 100
    
    def test_creative_strategy(self, generator):
        """Test creative generation strategy."""
        prompt = generator.generate(
            type=PromptType.BASIC,
            task_description="Write a story",
            strategy=GenerationStrategy.CREATIVE
        )
        
        creative_keywords = ["imagine", "creative", "unique", "original"]
        assert any(word in prompt["user"].lower() for word in creative_keywords)
    
    def test_analytical_strategy(self, generator):
        """Test analytical generation strategy."""
        prompt = generator.generate(
            type=PromptType.BASIC,
            task_description="Analyze trends",
            strategy=GenerationStrategy.ANALYTICAL
        )
        
        analytical_keywords = ["analyze", "examine", "evaluate", "assess"]
        assert any(word in prompt["user"].lower() for word in analytical_keywords)

########################################################################################################################
# Test Error Handling

class TestErrorHandling:
    """Test error handling in PromptGenerator."""
    
    @pytest.fixture
    def generator(self):
        """Create generator instance."""
        return PromptGenerator()
    
    def test_invalid_prompt_type(self, generator):
        """Test handling invalid prompt type."""
        with pytest.raises(ValueError):
            generator.generate(
                type="invalid_type",
                task_description="Test"
            )
    
    def test_missing_required_variables(self, generator):
        """Test handling missing required variables."""
        with pytest.raises(ValueError):
            generator.generate(
                type=PromptType.BASIC,
                task_description="Process {input}",
                variables={}  # Missing 'input' variable
            )
    
    def test_invalid_template_name(self, generator):
        """Test handling invalid template name."""
        with pytest.raises(ValueError):
            generator.generate(
                type=PromptType.CUSTOM,
                template_name="NonexistentTemplate",
                task_description="Test"
            )
    
    def test_invalid_few_shot_format(self, generator):
        """Test handling invalid few-shot examples."""
        with pytest.raises(ValueError):
            generator.generate(
                type=PromptType.FEW_SHOT,
                task_description="Test",
                few_shot_examples=["invalid", "format"]  # Should be dicts
            )

########################################################################################################################
# Test Template Library

class TestTemplateLibrary:
    """Test built-in template library."""
    
    @pytest.fixture
    def generator(self):
        """Create generator instance."""
        return PromptGenerator()
    
    def test_list_available_templates(self, generator):
        """Test listing available templates."""
        templates = generator.list_templates()
        
        assert len(templates) > 0
        assert all(hasattr(t, "name") for t in templates)
        assert all(hasattr(t, "type") for t in templates)
    
    def test_get_template_by_name(self, generator):
        """Test getting template by name."""
        # Assuming there's a built-in "summarization" template
        template = generator.get_template("summarization")
        
        if template:
            assert template.name == "summarization"
            assert template.type in [t for t in PromptType]
    
    def test_register_custom_template(self, generator):
        """Test registering custom template."""
        custom = PromptTemplate(
            name="MyCustomTemplate",
            type=PromptType.CUSTOM,
            system_template="Custom system",
            user_template="Custom user {var}",
            variables=["var"]
        )
        
        generator.register_template(custom)
        retrieved = generator.get_template("MyCustomTemplate")
        
        assert retrieved is not None
        assert retrieved.name == "MyCustomTemplate"
    
    def test_remove_template(self, generator):
        """Test removing template."""
        custom = PromptTemplate(
            name="ToRemove",
            type=PromptType.CUSTOM,
            user_template="Test"
        )
        
        generator.register_template(custom)
        removed = generator.remove_template("ToRemove")
        
        assert removed is True
        assert generator.get_template("ToRemove") is None

########################################################################################################################
# Test Advanced Features

class TestAdvancedFeatures:
    """Test advanced prompt generation features."""
    
    @pytest.fixture
    def generator(self):
        """Create generator instance."""
        return PromptGenerator()
    
    def test_dynamic_example_selection(self, generator):
        """Test dynamic few-shot example selection."""
        all_examples = [
            {"input": "Math: 2+2", "output": "4"},
            {"input": "Math: 3*3", "output": "9"},
            {"input": "Text: Hello", "output": "Hi"},
            {"input": "Text: Goodbye", "output": "Bye"},
        ]
        
        # Should select relevant examples based on input
        prompt = generator.generate(
            type=PromptType.FEW_SHOT,
            task_description="Solve math problem",
            variables={"input": "Math: 5+5"},
            few_shot_examples=all_examples,
            dynamic_selection=True,
            max_examples=2
        )
        
        # Should preferentially include math examples
        assert "2+2" in prompt["user"] or "3*3" in prompt["user"]
    
    def test_prompt_chaining(self, generator):
        """Test chaining multiple prompts."""
        chain = generator.create_chain([
            {"type": PromptType.BASIC, "task": "Extract key points"},
            {"type": PromptType.CHAIN_OF_THOUGHT, "task": "Analyze implications"},
            {"type": PromptType.STRUCTURED, "task": "Format as JSON"}
        ])
        
        assert len(chain) == 3
        assert all("system" in p and "user" in p for p in chain)
    
    def test_conditional_prompt_generation(self, generator):
        """Test conditional prompt generation."""
        conditions = {
            "language": "python",
            "complexity": "high",
            "output_format": "json"
        }
        
        prompt = generator.generate_conditional(
            base_type=PromptType.BASIC,
            task_description="Code review",
            conditions=conditions
        )
        
        assert "python" in prompt["user"].lower()
        assert "json" in prompt["user"].lower()
    
    def test_prompt_mutation(self, generator):
        """Test prompt mutation for optimization."""
        original = generator.generate(
            type=PromptType.BASIC,
            task_description="Summarize text"
        )
        
        mutations = generator.mutate_prompt(
            original,
            strategies=["rephrase", "add_detail", "simplify"],
            count=3
        )
        
        assert len(mutations) == 3
        assert all(m != original for m in mutations)
        assert all("system" in m and "user" in m for m in mutations)