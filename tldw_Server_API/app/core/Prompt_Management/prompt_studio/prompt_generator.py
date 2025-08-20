# prompt_generator.py
# Prompt generation for Prompt Studio

import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger

from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import (
    PromptStudioDatabase, DatabaseError
)
# Import chat completion function
from tldw_Server_API.app.core.LLM_Calls.LLM_API_Calls import chat_with_openai

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
    
    def __init__(self, db: PromptStudioDatabase):
        """
        Initialize PromptGenerator.
        
        Args:
            db: PromptStudioDatabase instance
        """
        self.db = db
        self.client_id = db.client_id
    
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
        import re
        return re.findall(r'\{(\w+)\}', template)