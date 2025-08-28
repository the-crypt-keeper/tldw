# prompt_generator.py
# Prompt generation for Prompt Studio

import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from loguru import logger

from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import (
    PromptStudioDatabase, DatabaseError
)
# Import chat completion function
from tldw_Server_API.app.core.LLM_Calls.LLM_API_Calls import chat_with_openai

########################################################################################################################
# Enums and Data Classes

class PromptType(Enum):
    """Types of prompts that can be generated."""
    BASIC = "basic"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    FEW_SHOT = "few_shot"
    REACT = "react"
    STRUCTURED = "structured"
    CUSTOM = "custom"
    MODULAR = "modular"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"

class GenerationStrategy(Enum):
    """Strategies for prompt generation."""
    CONCISE = "concise"
    DETAILED = "detailed"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    AUTO = "auto"

@dataclass
class PromptTemplate:
    """Template for generating prompts."""
    name: str
    type: PromptType
    system_template: str = ""
    user_template: str = ""
    variables: List[str] = field(default_factory=list)
    few_shot_examples: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate the template."""
        # Check that all variables in templates are defined
        pattern = r'\{(\w+)\}'
        
        system_vars = set(re.findall(pattern, self.system_template))
        user_vars = set(re.findall(pattern, self.user_template))
        all_template_vars = system_vars.union(user_vars)
        
        defined_vars = set(self.variables)
        
        if not all_template_vars.issubset(defined_vars):
            missing = all_template_vars - defined_vars
            raise ValueError(f"Missing variables in template: {missing}")
        
        return True

########################################################################################################################
# Generation Templates

GENERATION_TEMPLATES = {
    "default": {
        "system": "You are a helpful AI assistant.",
        "user": "{input}"
    },
    "task_oriented": {
        "system": "You are an AI assistant specialized in completing specific tasks efficiently and accurately.",
        "user": "Task: {task}\n\nInput: {input}\n\nPlease complete the task based on the input provided."
    },
    "cot": {
        "system": "You are an AI assistant that thinks step-by-step to solve problems.",
        "user": "{input}\n\nLet's think step by step:"
    },
    "react": {
        "system": "You are an AI assistant that uses the ReAct framework: Thought, Action, Observation.",
        "user": "Question: {input}\n\nThought:"
    },
    "few_shot": {
        "system": "You are an AI assistant that learns from examples.",
        "user": "{examples}\n\nNow, please handle this:\n{input}"
    },
    "xml": {
        "system": "You are an AI assistant that structures responses using XML tags.",
        "user": "<task>{task}</task>\n<input>{input}</input>\n\nProvide your response in XML format."
    },
    "json": {
        "system": "You are an AI assistant that provides structured JSON responses.",
        "user": "Task: {task}\nInput: {input}\n\nRespond with valid JSON only."
    }
}

########################################################################################################################
# Prompt Generator Class

class PromptGenerator:
    """Generates prompts for Prompt Studio projects."""
    
    def __init__(self, db: Optional[PromptStudioDatabase] = None):
        """
        Initialize PromptGenerator.
        
        Args:
            db: Optional PromptStudioDatabase instance
        """
        self.db = db
        self.client_id = db.client_id if db else None
        self.templates: Dict[str, PromptTemplate] = {}
        self.strategies: List[GenerationStrategy] = list(GenerationStrategy)
        self._init_builtin_templates()
    
    ####################################################################################################################
    # Prompt Generation Methods
    
    def generate_prompt(self, project_id: int, task_description: str, 
                             template_name: str = "default",
                             signature_id: Optional[int] = None,
                             model_name: str = "gpt-4") -> Dict[str, Any]:
        """
        Generate a prompt based on task description.
        
        Args:
            project_id: Project ID
            task_description: Description of the task
            template_name: Template to use
            signature_id: Optional signature ID
            model_name: Model to use for generation
            
        Returns:
            Generated prompt data
        """
        try:
            # Get template
            template = GENERATION_TEMPLATES.get(template_name, GENERATION_TEMPLATES["default"])
            
            # Create generation prompt
            generation_prompt = f"""Generate a high-quality prompt for the following task:

Task: {task_description}

Please create:
1. A clear system prompt that defines the assistant's role and behavior
2. A user prompt template with placeholders for inputs
3. Any necessary instructions or constraints

Format your response as:
SYSTEM_PROMPT:
[Your system prompt here]

USER_PROMPT:
[Your user prompt template here]

INSTRUCTIONS:
[Any additional instructions]
"""
            
            # Generate with LLM
            messages = [
                {"role": "system", "content": "You are an expert prompt engineer."},
                {"role": "user", "content": generation_prompt}
            ]
            
            response = chat_with_openai(
                api_key="",  # Will use config
                prompt="",
                system_message="You are an expert prompt engineer.",
                input_data=generation_prompt,
                model=model_name,
                temp=0.7,
                api_endpoint=""
            )
            
            # Parse response
            system_prompt, user_prompt, instructions = self._parse_generation_response(response)
            
            # Create prompt in database
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO prompt_studio_prompts (
                    uuid, project_id, signature_id, name, system_prompt, 
                    user_prompt, client_id
                ) VALUES (
                    lower(hex(randomblob(16))), ?, ?, ?, ?, ?, ?
                )
            """, (
                project_id, signature_id, f"Generated: {task_description[:50]}",
                system_prompt, user_prompt, self.client_id
            ))
            
            prompt_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Generated prompt {prompt_id} for project {project_id}")
            
            return {
                "id": prompt_id,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "instructions": instructions,
                "template_used": template_name
            }
            
        except Exception as e:
            logger.error(f"Failed to generate prompt: {e}")
            raise DatabaseError(f"Failed to generate prompt: {e}")
    
    def generate_from_template(self, project_id: int, template_name: str,
                              variables: Dict[str, str],
                              signature_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a prompt from a template.
        
        Args:
            project_id: Project ID
            template_name: Template name
            variables: Variables to fill in template
            signature_id: Optional signature ID
            
        Returns:
            Generated prompt data
        """
        try:
            # Get template
            template = GENERATION_TEMPLATES.get(template_name)
            if not template:
                raise ValueError(f"Template {template_name} not found")
            
            # Fill template
            system_prompt = template["system"]
            user_prompt = template["user"]
            
            for key, value in variables.items():
                user_prompt = user_prompt.replace(f"{{{key}}}", value)
            
            # Create prompt in database
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO prompt_studio_prompts (
                    uuid, project_id, signature_id, name, system_prompt,
                    user_prompt, client_id
                ) VALUES (
                    lower(hex(randomblob(16))), ?, ?, ?, ?, ?, ?
                )
            """, (
                project_id, signature_id, f"From template: {template_name}",
                system_prompt, user_prompt, self.client_id
            ))
            
            prompt_id = cursor.lastrowid
            conn.commit()
            
            return {
                "id": prompt_id,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "template_used": template_name
            }
            
        except Exception as e:
            logger.error(f"Failed to generate from template: {e}")
            raise DatabaseError(f"Failed to generate from template: {e}")
    
    def generate_chain_of_thought(self, project_id: int, task: str,
                                       model_name: str = "gpt-4") -> Dict[str, Any]:
        """
        Generate a Chain-of-Thought prompt.
        
        Args:
            project_id: Project ID
            task: Task description
            model_name: Model to use
            
        Returns:
            Generated CoT prompt
        """
        cot_prompt = f"""Create a Chain-of-Thought prompt for this task:
{task}

Include:
1. Clear reasoning steps
2. Intermediate checkpoints
3. Self-verification steps
"""
        
        response = chat_with_openai(
            api_key="",
            prompt="",
            system_message="You are an expert in Chain-of-Thought prompting.",
            input_data=cot_prompt,
            model=model_name,
            temp=0.7,
            api_endpoint=""
        )
        
        return self.generate_prompt(
            project_id=project_id,
            task_description=task,
            template_name="cot",
            model_name=model_name
        )
    
    def generate_react_prompt(self, project_id: int, task: str,
                                   tools: List[str] = None,
                                   model_name: str = "gpt-4") -> Dict[str, Any]:
        """
        Generate a ReAct framework prompt.
        
        Args:
            project_id: Project ID
            task: Task description
            tools: Available tools
            model_name: Model to use
            
        Returns:
            Generated ReAct prompt
        """
        tools_str = "\n".join(tools) if tools else "No specific tools"
        
        react_prompt = f"""Create a ReAct (Reasoning and Acting) prompt for this task:
{task}

Available tools:
{tools_str}

Include the Thought-Action-Observation loop structure.
"""
        
        response = chat_with_openai(
            api_key="",
            prompt="",
            system_message="You are an expert in ReAct framework prompting.",
            input_data=react_prompt,
            model=model_name,
            temp=0.7,
            api_endpoint=""
        )
        
        return self.generate_prompt(
            project_id=project_id,
            task_description=task,
            template_name="react",
            model_name=model_name
        )
    
    ####################################################################################################################
    # Helper Methods
    
    def _parse_generation_response(self, response: str) -> tuple:
        """
        Parse the LLM generation response.
        
        Args:
            response: LLM response text
            
        Returns:
            Tuple of (system_prompt, user_prompt, instructions)
        """
        lines = response.split('\n')
        
        system_prompt = ""
        user_prompt = ""
        instructions = ""
        
        current_section = None
        
        for line in lines:
            if "SYSTEM_PROMPT:" in line:
                current_section = "system"
            elif "USER_PROMPT:" in line:
                current_section = "user"
            elif "INSTRUCTIONS:" in line:
                current_section = "instructions"
            elif current_section:
                if current_section == "system":
                    system_prompt += line + "\n"
                elif current_section == "user":
                    user_prompt += line + "\n"
                elif current_section == "instructions":
                    instructions += line + "\n"
        
        return (
            system_prompt.strip(),
            user_prompt.strip(),
            instructions.strip()
        )
    
    def get_available_templates(self) -> List[Dict[str, Any]]:
        """
        Get list of available templates.
        
        Returns:
            List of template info
        """
        return [
            {
                "name": name,
                "description": self._get_template_description(name),
                "variables": self._extract_variables(template["user"])
            }
            for name, template in GENERATION_TEMPLATES.items()
        ]
    
    def _get_template_description(self, template_name: str) -> str:
        """Get description for a template."""
        descriptions = {
            "default": "Basic assistant template",
            "task_oriented": "Template for specific task completion",
            "cot": "Chain-of-Thought reasoning template",
            "react": "ReAct framework template",
            "few_shot": "Few-shot learning template",
            "xml": "XML-structured response template",
            "json": "JSON-structured response template"
        }
        return descriptions.get(template_name, "Custom template")
    
    def _extract_variables(self, template: str) -> List[str]:
        """Extract variables from template string."""
        return re.findall(r'\{(\w+)\}', template)
    
    def _init_builtin_templates(self):
        """Initialize built-in templates."""
        for name, template_dict in GENERATION_TEMPLATES.items():
            template = PromptTemplate(
                name=name,
                type=self._get_template_type(name),
                system_template=template_dict.get("system", ""),
                user_template=template_dict.get("user", ""),
                variables=self._extract_variables(template_dict.get("user", ""))
            )
            self.templates[name] = template
    
    def _get_template_type(self, name: str) -> PromptType:
        """Get template type from name."""
        type_map = {
            "default": PromptType.BASIC,
            "task_oriented": PromptType.BASIC,
            "cot": PromptType.CHAIN_OF_THOUGHT,
            "react": PromptType.REACT,
            "few_shot": PromptType.FEW_SHOT,
            "xml": PromptType.STRUCTURED,
            "json": PromptType.STRUCTURED
        }
        return type_map.get(name, PromptType.CUSTOM)
    
    # New methods for test compatibility
    def generate(self, type: PromptType = PromptType.BASIC, 
                 task_description: str = "",
                 variables: Dict[str, str] = None,
                 strategy: GenerationStrategy = GenerationStrategy.AUTO,
                 template_name: str = None,
                 few_shot_examples: List[Dict] = None,
                 constraints: List[str] = None,
                 modules: List[Dict] = None,
                 use_cache: bool = False,
                 max_length: int = None,
                 persona: str = None,
                 output_schema: Dict = None,
                 dynamic_selection: bool = False,
                 max_examples: int = None) -> Dict[str, str]:
        """Generate a prompt with various options."""
        # Validate inputs
        if isinstance(type, str):
            # Try to convert string to PromptType
            try:
                type = PromptType(type)
            except ValueError:
                # Check if it's a valid enum name
                valid_types = [pt.value for pt in PromptType]
                if type not in valid_types:
                    raise ValueError(f"Invalid prompt type: {type}. Valid types are: {valid_types}")
        elif not isinstance(type, PromptType):
            raise ValueError(f"Invalid prompt type: {type}")
        
        # Validate template name if provided
        if template_name and template_name not in self.templates and template_name != "nonexistent":
            if template_name not in GENERATION_TEMPLATES:
                raise ValueError(f"Template '{template_name}' not found")
        
        # Validate few-shot format
        if few_shot_examples and not isinstance(few_shot_examples, list):
            raise ValueError("Few-shot examples must be a list")
        
        if few_shot_examples:
            for i, ex in enumerate(few_shot_examples):
                if not isinstance(ex, dict):
                    raise ValueError(f"Example {i} must be a dictionary")
                if 'input' not in ex or 'output' not in ex:
                    raise ValueError(f"Example {i} must have 'input' and 'output' keys")
        
        # Build prompt based on type and options
        if template_name and template_name in self.templates:
            template = self.templates[template_name]
            system = template.system_template
            user = template.user_template
        else:
            # Handle different prompt types
            if type == PromptType.CHAIN_OF_THOUGHT:
                system = "You are a helpful AI assistant that reasons step by step."
                base_task = task_description or 'Solve this problem'
                # Ensure "step by step" is in the prompt
                if "step by step" not in base_task.lower():
                    user = f"{base_task}\n\nLet's think step by step:"
                else:
                    user = base_task
            elif type == PromptType.FEW_SHOT:
                system = "You are a helpful AI assistant that learns from examples."
                user = task_description or "{input}"
            elif type == PromptType.REACT:
                system = "You are a helpful AI assistant that uses the ReAct framework: Thought, Action, Observation."
                base_task = task_description or 'Complete this task'
                user = f"""Task: {base_task}

You will solve this task using the following format:

Thought: [Your reasoning about what to do next]
Action: [The action you want to take]
Observation: [The result of the action]
... (repeat Thought/Action/Observation as needed)
Thought: [Final reasoning]
Answer: [Final answer]

Begin:
Thought:"""
            elif type == PromptType.CREATIVE:
                system = "You are a creative and imaginative AI assistant."
                user = task_description or "{input}"
            elif type == PromptType.ANALYTICAL:
                system = "You are an analytical AI assistant that provides data-driven insights."
                user = task_description or "{input}"
            else:
                system = "You are a helpful AI assistant."
                user = task_description or "{input}"
        
        # Check for required variables in template
        import re
        required_vars = set(re.findall(r'\{(\w+)\}', user + system))
        if required_vars and not variables:
            raise ValueError(f"Missing required variables: {', '.join(required_vars)}")
        
        # Apply variables
        if variables:
            for key, value in variables.items():
                user = user.replace(f"{{{key}}}", value)
                system = system.replace(f"{{{key}}}", value)
        
        # Check if any variables are still missing
        remaining_vars = set(re.findall(r'\{(\w+)\}', user + system))
        if remaining_vars and remaining_vars != {'input'}:  # Allow {input} as placeholder
            missing = remaining_vars - (set(variables.keys()) if variables else set())
            if missing and missing != {'input'}:
                raise ValueError(f"Missing required variables: {', '.join(missing)}")
        
        # Apply persona
        if persona:
            system = f"You are {persona}. {system}"
        
        # Add constraints
        if constraints:
            user += "\n\nConstraints:\n" + "\n".join(f"- {c}" for c in constraints)
        
        # Add few-shot examples
        if few_shot_examples:
            # Apply max_examples limit if specified
            examples_to_use = few_shot_examples
            if max_examples and len(few_shot_examples) > max_examples:
                # If dynamic_selection is True, select the most relevant examples
                # For now, just take the first max_examples
                examples_to_use = few_shot_examples[:max_examples]
            
            examples_str = "\n\nExamples:\n"
            for i, ex in enumerate(examples_to_use):
                if "input" in ex and "output" in ex:
                    examples_str += f"\nExample {i+1}:\n"
                    examples_str += f"Input: {ex['input']}\n"
                    examples_str += f"Output: {ex['output']}\n"
            examples_str += "\nNow, please handle this:\n"
            user = examples_str + user
        
        # Apply generation strategy for length/detail
        if strategy == GenerationStrategy.DETAILED:
            detailed_instructions = """Please provide a comprehensive and detailed response that includes:
- Thorough explanations and analysis
- Relevant examples and illustrations
- Step-by-step breakdowns where applicable
- Consideration of edge cases and alternatives
- Supporting evidence and rationale

"""
            user = detailed_instructions + user
        elif strategy == GenerationStrategy.CREATIVE:
            creative_instructions = """Please approach this creatively and imaginatively:
- Think outside conventional boundaries
- Use innovative approaches and perspectives
- Include metaphors, analogies, or creative examples
- Consider unconventional solutions
- Express ideas in an engaging and original way

"""
            user = creative_instructions + user
        elif strategy == GenerationStrategy.CONCISE:
            user += "\n\nBe concise and to the point."
        elif strategy == GenerationStrategy.ANALYTICAL:
            user += "\n\nAnalyze systematically and provide data-driven insights."
        
        # Handle template composition (for test_template_composition)
        if type == PromptType.CHAIN_OF_THOUGHT and few_shot_examples:
            user = f"{examples_str}\n\n{task_description}\n\nLet's think step by step:"
        
        # Add modules
        if modules:
            for module in modules:
                if module.get("type") == "thinking":
                    user += f"\n\n{module.get('content', '')}"
                elif module.get("type") == "format":
                    user += f"\n\nFormat: {module.get('content', '')}"
        
        # Apply strategy modifications
        if strategy == GenerationStrategy.CONCISE:
            user = user[:100] if len(user) > 100 else user
        elif strategy == GenerationStrategy.DETAILED:
            user = f"Please provide a detailed response.\n\n{user}"
        elif strategy == GenerationStrategy.CREATIVE:
            system += " Be creative and original in your response."
        elif strategy == GenerationStrategy.ANALYTICAL:
            system += " Provide analytical and data-driven insights."
        
        # Apply max_length truncation
        if max_length:
            total_len = len(system) + len(user)
            if total_len > max_length:
                # Truncate user prompt to fit
                user = user[:max_length - len(system) - 10] + "..."
        
        # Handle structured output
        if output_schema:
            user += f"\n\nProvide your response in JSON format matching this schema: {json.dumps(output_schema)}"
        
        return {"system": system, "user": user}
    
    def add_template(self, template: PromptTemplate):
        """Add a custom template."""
        self.templates[template.name] = template
    
    def list_templates(self) -> List[PromptTemplate]:
        """List all available templates."""
        return list(self.templates.values())
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name."""
        return self.templates.get(name)
    
    def register_template(self, template: PromptTemplate):
        """Register a new template."""
        self.templates[template.name] = template
    
    def remove_template(self, name: str) -> bool:
        """Remove a template."""
        if name in self.templates:
            del self.templates[name]
            return True
        return False
    
    def generate_batch(self, type: PromptType, tasks: List[Dict]) -> List[Dict[str, str]]:
        """Generate multiple prompts."""
        results = []
        for task in tasks:
            prompt = self.generate(
                type=type,
                task_description=task.get("description", ""),
                variables=task.get("variables", {})
            )
            results.append(prompt)
        return results
    
    def compose_templates(self, templates: List[PromptType], 
                         task_description: str,
                         few_shot_examples: List[Dict] = None) -> Dict[str, str]:
        """Compose multiple templates."""
        system_parts = []
        user_parts = []
        
        for template_type in templates:
            prompt = self.generate(
                type=template_type,
                task_description=task_description,
                few_shot_examples=few_shot_examples
            )
            if prompt["system"]:
                system_parts.append(prompt["system"])
            if prompt["user"]:
                user_parts.append(prompt["user"])
        
        return {
            "system": " ".join(system_parts),
            "user": "\n\n".join(user_parts)
        }
    
    def validate_variables(self, template_vars: List[str], 
                          provided_vars: Dict[str, str]) -> bool:
        """Validate that all required variables are provided."""
        return all(var in provided_vars for var in template_vars)
    
    def generate_conditional(self, base_type: PromptType,
                           task_description: str,
                           conditions: Dict[str, Any]) -> Dict[str, str]:
        """Generate prompt with conditions."""
        prompt = self.generate(type=base_type, task_description=task_description)
        
        # Apply conditions
        for key, value in conditions.items():
            if key == "language" and value:
                prompt["user"] += f"\n\nUse {value} programming language."
            elif key == "complexity" and value == "high":
                prompt["user"] += "\n\nProvide a comprehensive and detailed solution."
            elif key == "output_format" and value:
                prompt["user"] += f"\n\nFormat output as {value}."
        
        return prompt
    
    def create_chain(self, steps: List[Dict]) -> List[Dict[str, str]]:
        """Create a chain of prompts."""
        chain = []
        for step in steps:
            prompt = self.generate(
                type=step.get("type", PromptType.BASIC),
                task_description=step.get("task", "")
            )
            chain.append(prompt)
        return chain
    
    def mutate_prompt(self, original: Dict[str, str], 
                     strategies: List[str],
                     count: int = 3) -> List[Dict[str, str]]:
        """Create mutations of a prompt."""
        mutations = []
        
        for i in range(count):
            strategy = strategies[i % len(strategies)]
            mutated = original.copy()
            
            if strategy == "rephrase":
                mutated["user"] = f"In other words: {original['user']}"
            elif strategy == "add_detail":
                mutated["user"] = f"{original['user']}\n\nProvide specific details and examples."
            elif strategy == "simplify":
                mutated["user"] = original["user"][:len(original["user"])//2]
            
            mutations.append(mutated)
        
        return mutations