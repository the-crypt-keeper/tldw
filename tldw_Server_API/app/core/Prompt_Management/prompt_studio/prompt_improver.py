# prompt_improver.py
# Prompt improvement and optimization for Prompt Studio

import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from loguru import logger

from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import (
    PromptStudioDatabase, DatabaseError
)
from tldw_Server_API.app.core.LLM_Calls.LLM_API_Calls import chat_with_openai

########################################################################################################################
# Improvement Strategies

IMPROVEMENT_STRATEGIES = {
    "clarity": {
        "focus": "Improve clarity and reduce ambiguity",
        "techniques": [
            "Use specific, concrete language",
            "Define technical terms",
            "Break complex instructions into steps",
            "Add examples where helpful"
        ]
    },
    "structure": {
        "focus": "Improve prompt structure and organization",
        "techniques": [
            "Use XML or markdown formatting",
            "Create clear sections",
            "Add numbered steps",
            "Use consistent formatting"
        ]
    },
    "specificity": {
        "focus": "Make instructions more specific",
        "techniques": [
            "Add explicit constraints",
            "Specify output format",
            "Include success criteria",
            "Define edge cases"
        ]
    },
    "robustness": {
        "focus": "Make prompt more robust to edge cases",
        "techniques": [
            "Add error handling instructions",
            "Include fallback behaviors",
            "Handle missing inputs",
            "Add validation steps"
        ]
    },
    "efficiency": {
        "focus": "Reduce token usage while maintaining quality",
        "techniques": [
            "Remove redundant instructions",
            "Consolidate similar points",
            "Use concise language",
            "Eliminate unnecessary examples"
        ]
    }
}

########################################################################################################################
# Prompt Improver Class

class PromptImprover:
    """Improves and optimizes prompts for Prompt Studio projects."""
    
    def __init__(self, db: PromptStudioDatabase):
        """
        Initialize PromptImprover.
        
        Args:
            db: PromptStudioDatabase instance
        """
        self.db = db
        self.client_id = db.client_id
    
    ####################################################################################################################
    # Improvement Methods
    
    def improve_prompt(self, prompt_id: int, strategies: List[str] = None,
                            model_name: str = "gpt-4") -> Dict[str, Any]:
        """
        Improve an existing prompt using specified strategies.
        
        Args:
            prompt_id: ID of prompt to improve
            strategies: List of improvement strategies to apply
            model_name: Model to use for improvement
            
        Returns:
            Improved prompt data
        """
        try:
            # Get existing prompt
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, project_id, signature_id, name, system_prompt, 
                       user_prompt, few_shot_examples
                FROM prompt_studio_prompts
                WHERE id = ? AND deleted = 0
            """, (prompt_id,))
            
            prompt_data = cursor.fetchone()
            if not prompt_data:
                raise ValueError(f"Prompt {prompt_id} not found")
            
            # Default strategies if none specified
            if not strategies:
                strategies = ["clarity", "structure", "specificity"]
            
            # Apply improvements
            improved_system = self._improve_text(
                prompt_data[4],  # system_prompt
                strategies,
                "system prompt",
                model_name
            )
            
            improved_user = self._improve_text(
                prompt_data[5],  # user_prompt
                strategies,
                "user prompt",
                model_name
            )
            
            # Create new version
            cursor.execute("""
                INSERT INTO prompt_studio_prompts (
                    uuid, project_id, signature_id, name, system_prompt,
                    user_prompt, parent_version_id, change_description,
                    version_number, client_id
                ) VALUES (
                    lower(hex(randomblob(16))), ?, ?, ?, ?, ?, ?, ?,
                    (SELECT COALESCE(MAX(version_number), 0) + 1 
                     FROM prompt_studio_prompts 
                     WHERE project_id = ? AND name = ?),
                    ?
                )
            """, (
                prompt_data[1],  # project_id
                prompt_data[2],  # signature_id
                prompt_data[3],  # name
                improved_system,
                improved_user,
                prompt_id,
                f"Improved using strategies: {', '.join(strategies)}",
                prompt_data[1], prompt_data[3],  # for version subquery
                self.client_id
            ))
            
            new_prompt_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Created improved version {new_prompt_id} of prompt {prompt_id}")
            
            return {
                "id": new_prompt_id,
                "parent_id": prompt_id,
                "system_prompt": improved_system,
                "user_prompt": improved_user,
                "strategies_applied": strategies,
                "improvements": self._analyze_improvements(
                    prompt_data[4], improved_system,
                    prompt_data[5], improved_user
                )
            }
            
        except Exception as e:
            logger.error(f"Failed to improve prompt: {e}")
            raise DatabaseError(f"Failed to improve prompt: {e}")
    
    def analyze_prompt(self, prompt_id: int, model_name: str = "gpt-4") -> Dict[str, Any]:
        """
        Analyze a prompt for potential improvements.
        
        Args:
            prompt_id: ID of prompt to analyze
            model_name: Model to use for analysis
            
        Returns:
            Analysis results
        """
        try:
            # Get prompt
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT system_prompt, user_prompt
                FROM prompt_studio_prompts
                WHERE id = ? AND deleted = 0
            """, (prompt_id,))
            
            prompt_data = cursor.fetchone()
            if not prompt_data:
                raise ValueError(f"Prompt {prompt_id} not found")
            
            # Analyze with LLM
            analysis_prompt = f"""Analyze this prompt for potential improvements:

SYSTEM PROMPT:
{prompt_data[0]}

USER PROMPT:
{prompt_data[1]}

Please identify:
1. Strengths of the current prompt
2. Potential weaknesses or areas for improvement
3. Specific suggestions for each weakness
4. Priority of improvements (high/medium/low)

Format as JSON.
"""
            
            response = chat_with_openai(
                api_key="",
                prompt="",
                system_message="You are an expert prompt analyst.",
                input_data=analysis_prompt,
                model=model_name,
                temp=0.7,
                api_endpoint=""
            )
            
            # Parse response
            try:
                analysis = json.loads(response)
            except json.JSONDecodeError:
                analysis = {"raw_analysis": response}
            
            return {
                "prompt_id": prompt_id,
                "analysis": analysis,
                "recommended_strategies": self._recommend_strategies(analysis)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze prompt: {e}")
            raise DatabaseError(f"Failed to analyze prompt: {e}")
    
    def standardize_to_xml(self, prompt_id: int) -> Dict[str, Any]:
        """
        Convert a prompt to XML format for better structure.
        
        Args:
            prompt_id: ID of prompt to standardize
            
        Returns:
            Standardized prompt data
        """
        try:
            # Get prompt
            conn = self.db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT project_id, signature_id, name, system_prompt, user_prompt
                FROM prompt_studio_prompts
                WHERE id = ? AND deleted = 0
            """, (prompt_id,))
            
            prompt_data = cursor.fetchone()
            if not prompt_data:
                raise ValueError(f"Prompt {prompt_id} not found")
            
            # Convert to XML format
            xml_system = f"""<role>
    You are an AI assistant that follows structured instructions precisely.
</role>

<capabilities>
    {prompt_data[3]}
</capabilities>"""
            
            xml_user = self._convert_to_xml_format(prompt_data[4])
            
            # Create new version
            cursor.execute("""
                INSERT INTO prompt_studio_prompts (
                    uuid, project_id, signature_id, name, system_prompt,
                    user_prompt, parent_version_id, change_description,
                    version_number, client_id
                ) VALUES (
                    lower(hex(randomblob(16))), ?, ?, ?, ?, ?, ?, ?,
                    (SELECT COALESCE(MAX(version_number), 0) + 1 
                     FROM prompt_studio_prompts 
                     WHERE project_id = ? AND name = ?),
                    ?
                )
            """, (
                prompt_data[0], prompt_data[1], prompt_data[2],
                xml_system, xml_user, prompt_id,
                "Standardized to XML format",
                prompt_data[0], prompt_data[2],
                self.client_id
            ))
            
            new_prompt_id = cursor.lastrowid
            conn.commit()
            
            return {
                "id": new_prompt_id,
                "parent_id": prompt_id,
                "system_prompt": xml_system,
                "user_prompt": xml_user,
                "format": "xml"
            }
            
        except Exception as e:
            logger.error(f"Failed to standardize prompt: {e}")
            raise DatabaseError(f"Failed to standardize prompt: {e}")
    
    ####################################################################################################################
    # Helper Methods
    
    def _improve_text(self, text: str, strategies: List[str],
                           text_type: str, model_name: str) -> str:
        """
        Improve a text using specified strategies.
        
        Args:
            text: Text to improve
            strategies: Improvement strategies
            text_type: Type of text (for context)
            model_name: Model to use
            
        Returns:
            Improved text
        """
        # Build improvement instructions
        instructions = []
        for strategy in strategies:
            if strategy in IMPROVEMENT_STRATEGIES:
                strat_info = IMPROVEMENT_STRATEGIES[strategy]
                instructions.append(f"\n{strat_info['focus']}:")
                for technique in strat_info['techniques']:
                    instructions.append(f"  - {technique}")
        
        improvement_prompt = f"""Improve this {text_type} using these strategies:
{''.join(instructions)}

Original text:
{text}

Provide only the improved text, no explanations.
"""
        
        response = chat_with_openai(
            api_key="",
            prompt="",
            system_message="You are an expert prompt engineer focused on improvement.",
            input_data=improvement_prompt,
            model=model_name,
            temp=0.7,
            api_endpoint=""
        )
        
        return response.strip()
    
    def _analyze_improvements(self, old_system: str, new_system: str,
                            old_user: str, new_user: str) -> Dict[str, Any]:
        """
        Analyze the improvements made.
        
        Args:
            old_system: Original system prompt
            new_system: Improved system prompt
            old_user: Original user prompt
            new_user: Improved user prompt
            
        Returns:
            Analysis of improvements
        """
        return {
            "system_prompt": {
                "length_change": len(new_system) - len(old_system),
                "length_ratio": len(new_system) / max(len(old_system), 1)
            },
            "user_prompt": {
                "length_change": len(new_user) - len(old_user),
                "length_ratio": len(new_user) / max(len(old_user), 1)
            },
            "total_length_change": (len(new_system) + len(new_user)) - (len(old_system) + len(old_user))
        }
    
    def _recommend_strategies(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Recommend improvement strategies based on analysis.
        
        Args:
            analysis: Prompt analysis results
            
        Returns:
            List of recommended strategies
        """
        recommendations = []
        
        # Simple heuristic-based recommendations
        if "weaknesses" in analysis:
            weaknesses = analysis.get("weaknesses", [])
            if isinstance(weaknesses, list):
                for weakness in weaknesses:
                    weakness_lower = str(weakness).lower()
                    if "unclear" in weakness_lower or "ambiguous" in weakness_lower:
                        recommendations.append("clarity")
                    if "structure" in weakness_lower or "organization" in weakness_lower:
                        recommendations.append("structure")
                    if "specific" in weakness_lower or "vague" in weakness_lower:
                        recommendations.append("specificity")
                    if "edge" in weakness_lower or "error" in weakness_lower:
                        recommendations.append("robustness")
                    if "long" in weakness_lower or "verbose" in weakness_lower:
                        recommendations.append("efficiency")
        
        # Remove duplicates while preserving order
        seen = set()
        return [x for x in recommendations if not (x in seen or seen.add(x))]
    
    def _convert_to_xml_format(self, text: str) -> str:
        """
        Convert text to XML format.
        
        Args:
            text: Text to convert
            
        Returns:
            XML-formatted text
        """
        # Simple conversion - could be enhanced with more sophisticated parsing
        lines = text.split('\n')
        xml_parts = ["<instructions>"]
        
        for line in lines:
            line = line.strip()
            if line:
                if line.endswith(':'):
                    # Section header
                    tag = line[:-1].lower().replace(' ', '_')
                    xml_parts.append(f"  <{tag}>")
                else:
                    # Content
                    xml_parts.append(f"    {line}")
        
        xml_parts.append("</instructions>")
        
        return '\n'.join(xml_parts)