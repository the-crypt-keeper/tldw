# test_runner.py
# Runs test cases against prompts

import json
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger

from ....core.Chat.Chat_Functions import chat_api_call

class TestRunner:
    """Runs test cases against prompts using LLM."""
    
    def __init__(self, db_manager):
        """
        Initialize test runner.
        
        Args:
            db_manager: Database manager instance
        """
        self.db = db_manager
    
    async def run_test_case(
        self,
        prompt_id: int,
        test_case_id: int,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Run a single test case against a prompt.
        
        Args:
            prompt_id: ID of the prompt
            test_case_id: ID of the test case
            model: LLM model to use
            temperature: Temperature setting
            max_tokens: Maximum tokens
            
        Returns:
            Test run result
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        # Get prompt
        cursor.execute("""
            SELECT system_prompt, user_prompt, project_id
            FROM prompt_studio_prompts
            WHERE id = ? AND deleted = 0
        """, (prompt_id,))
        
        prompt = cursor.fetchone()
        if not prompt:
            raise ValueError(f"Prompt {prompt_id} not found")
        
        # Get test case
        cursor.execute("""
            SELECT inputs, expected_outputs
            FROM prompt_studio_test_cases
            WHERE id = ? AND deleted = 0
        """, (test_case_id,))
        
        test_case = cursor.fetchone()
        if not test_case:
            raise ValueError(f"Test case {test_case_id} not found")
        
        inputs = json.loads(test_case[0]) if test_case[0] else {}
        expected = json.loads(test_case[1]) if test_case[1] else {}
        
        # Format prompt with inputs
        user_prompt = prompt[1]
        for key, value in inputs.items():
            user_prompt = user_prompt.replace(f"{{{key}}}", str(value))
        
        # Call LLM
        try:
            response = await asyncio.to_thread(
                chat_api_call,
                api_endpoint="openai",
                model=model,
                messages=[
                    {"role": "system", "content": prompt[0]},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            actual_output = {"response": response[0] if response else ""}
            
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            actual_output = {"error": str(e)}
        
        # Create test run record
        cursor.execute("""
            INSERT INTO prompt_studio_test_runs (
                uuid, project_id, test_case_id, prompt_id, model_name,
                inputs, outputs, expected_outputs,
                client_id
            ) VALUES (
                lower(hex(randomblob(16))), ?, ?, ?, ?,
                ?, ?, ?,
                ?
            )
        """, (
            prompt[2],  # project_id (index 2 now)
            test_case_id, prompt_id, model,
            json.dumps(inputs), json.dumps(actual_output), json.dumps(expected),
            self.db.client_id
        ))
        
        run_id = cursor.lastrowid
        conn.commit()
        
        return {
            "id": run_id,
            "test_case_id": test_case_id,
            "prompt_id": prompt_id,
            "inputs": inputs,
            "expected": expected,
            "actual": actual_output,
            "model": model
        }
    
    async def run_multiple_tests(
        self,
        prompt_id: int,
        test_case_ids: List[int],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        parallel: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Run multiple test cases.
        
        Args:
            prompt_id: ID of the prompt
            test_case_ids: List of test case IDs
            model: LLM model to use
            temperature: Temperature setting
            max_tokens: Maximum tokens
            parallel: Run tests in parallel
            
        Returns:
            List of test run results
        """
        if parallel:
            # Run tests concurrently
            tasks = [
                self.run_test_case(prompt_id, test_id, model, temperature, max_tokens)
                for test_id in test_case_ids
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Test case {test_case_ids[i]} failed: {result}")
                    processed_results.append({
                        "test_case_id": test_case_ids[i],
                        "error": str(result)
                    })
                else:
                    processed_results.append(result)
            
            return processed_results
        else:
            # Run tests sequentially
            results = []
            for test_id in test_case_ids:
                try:
                    result = await self.run_test_case(
                        prompt_id, test_id, model, temperature, max_tokens
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"Test case {test_id} failed: {e}")
                    results.append({
                        "test_case_id": test_id,
                        "error": str(e)
                    })
            
            return results