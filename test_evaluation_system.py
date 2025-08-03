#!/usr/bin/env python3
"""
Test script for the evaluation system implementation.

Tests:
1. G-Eval for summarization
2. RAG evaluation
3. Response quality evaluation
4. Batch evaluation
5. History and comparison
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "tldw_Server_API"))

# Import test utilities
import httpx
from loguru import logger


class EvaluationSystemTester:
    """Test harness for evaluation system"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_prefix = "/api/v1"
        
    async def test_geval(self) -> Dict[str, Any]:
        """Test G-Eval summarization evaluation"""
        logger.info("Testing G-Eval endpoint...")
        
        test_data = {
            "source_text": """
            Artificial Intelligence (AI) has transformed numerous industries in recent years. 
            Machine learning algorithms can now process vast amounts of data to identify patterns 
            and make predictions. Deep learning, a subset of machine learning, uses neural networks 
            with multiple layers to tackle complex problems. Natural language processing enables 
            computers to understand and generate human language. Computer vision allows machines 
            to interpret visual information. These technologies are being applied in healthcare 
            for disease diagnosis, in finance for fraud detection, and in transportation for 
            autonomous vehicles.
            """,
            "summary": """
            AI has revolutionized many sectors through technologies like machine learning, 
            deep learning, NLP, and computer vision. Applications include healthcare diagnostics, 
            financial fraud detection, and self-driving cars.
            """,
            "metrics": ["fluency", "consistency", "relevance", "coherence"],
            "api_name": "openai",
            "save_results": False
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}{self.api_prefix}/evaluations/geval",
                json=test_data,
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.success(f"G-Eval test passed. Average score: {result['average_score']:.2f}")
                return {"status": "success", "data": result}
            else:
                logger.error(f"G-Eval test failed: {response.status_code} - {response.text}")
                return {"status": "error", "message": response.text}
    
    async def test_rag_evaluation(self) -> Dict[str, Any]:
        """Test RAG system evaluation"""
        logger.info("Testing RAG evaluation endpoint...")
        
        test_data = {
            "query": "What are the benefits of renewable energy?",
            "retrieved_contexts": [
                "Solar power reduces electricity bills and carbon emissions.",
                "Wind energy is clean and sustainable, requiring no fuel.",
                "Renewable sources help combat climate change and create jobs."
            ],
            "generated_response": """
            Renewable energy offers several benefits including reduced carbon emissions, 
            lower electricity costs, sustainability, and job creation in the green sector.
            """,
            "ground_truth": "Benefits include environmental protection, cost savings, and economic growth.",
            "metrics": ["relevance", "faithfulness", "answer_similarity"],
            "api_name": "openai"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}{self.api_prefix}/evaluations/rag",
                json=test_data,
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.success(f"RAG evaluation test passed. Overall score: {result['overall_score']:.2f}")
                return {"status": "success", "data": result}
            else:
                logger.error(f"RAG evaluation test failed: {response.status_code} - {response.text}")
                return {"status": "error", "message": response.text}
    
    async def test_response_quality(self) -> Dict[str, Any]:
        """Test response quality evaluation"""
        logger.info("Testing response quality endpoint...")
        
        test_data = {
            "prompt": "Write a haiku about artificial intelligence",
            "response": """
            Silicon minds think
            Algorithms learn and grow
            Future unfolds now
            """,
            "expected_format": "Three lines with 5-7-5 syllable pattern",
            "evaluation_criteria": {
                "creativity": "How creative and original is the haiku",
                "theme_adherence": "How well does it capture AI themes"
            },
            "api_name": "openai"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}{self.api_prefix}/evaluations/response-quality",
                json=test_data,
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.success(f"Response quality test passed. Overall quality: {result['overall_quality']:.2f}")
                return {"status": "success", "data": result}
            else:
                logger.error(f"Response quality test failed: {response.status_code} - {response.text}")
                return {"status": "error", "message": response.text}
    
    async def test_batch_evaluation(self) -> Dict[str, Any]:
        """Test batch evaluation"""
        logger.info("Testing batch evaluation endpoint...")
        
        test_data = {
            "evaluation_type": "geval",
            "items": [
                {
                    "source_text": "The Earth orbits the Sun once every 365.25 days.",
                    "summary": "Earth takes about a year to orbit the Sun."
                },
                {
                    "source_text": "Water freezes at 0 degrees Celsius and boils at 100 degrees.",
                    "summary": "Water's freezing point is 0°C and boiling point is 100°C."
                }
            ],
            "api_name": "openai",
            "parallel_workers": 2
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}{self.api_prefix}/evaluations/batch",
                json=test_data,
                timeout=60.0
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.success(f"Batch evaluation test passed. Processed {result['successful']} items")
                return {"status": "success", "data": result}
            else:
                logger.error(f"Batch evaluation test failed: {response.status_code} - {response.text}")
                return {"status": "error", "message": response.text}
    
    async def test_evaluation_history(self) -> Dict[str, Any]:
        """Test evaluation history endpoint"""
        logger.info("Testing evaluation history endpoint...")
        
        test_data = {
            "evaluation_type": "all",
            "limit": 10,
            "offset": 0
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}{self.api_prefix}/evaluations/history",
                json=test_data,
                timeout=10.0
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.success(f"History test passed. Found {result['total_count']} evaluations")
                return {"status": "success", "data": result}
            else:
                logger.error(f"History test failed: {response.status_code} - {response.text}")
                return {"status": "error", "message": response.text}
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all evaluation system tests"""
        logger.info("Starting evaluation system tests...")
        
        results = {}
        
        # Note: These tests require the API server to be running
        # and appropriate API keys configured
        
        tests = [
            ("G-Eval", self.test_geval),
            ("RAG Evaluation", self.test_rag_evaluation),
            ("Response Quality", self.test_response_quality),
            ("Batch Evaluation", self.test_batch_evaluation),
            ("History", self.test_evaluation_history)
        ]
        
        for test_name, test_func in tests:
            try:
                logger.info(f"\nRunning {test_name} test...")
                result = await test_func()
                results[test_name] = result
            except Exception as e:
                logger.error(f"{test_name} test failed with exception: {e}")
                results[test_name] = {"status": "error", "message": str(e)}
        
        # Summary
        passed = sum(1 for r in results.values() if r["status"] == "success")
        total = len(results)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluation System Test Summary: {passed}/{total} tests passed")
        logger.info(f"{'='*60}")
        
        return results


async def main():
    """Run evaluation system tests"""
    # First, check if we can import the evaluation modules
    try:
        from tldw_Server_API.app.core.Evaluations.ms_g_eval import run_geval
        from tldw_Server_API.app.core.Evaluations.rag_evaluator import RAGEvaluator
        from tldw_Server_API.app.core.Evaluations.response_quality_evaluator import ResponseQualityEvaluator
        from tldw_Server_API.app.core.Evaluations.evaluation_manager import EvaluationManager
        logger.success("✓ All evaluation modules imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import evaluation modules: {e}")
        return
    
    # Check schemas
    try:
        from tldw_Server_API.app.api.v1.schemas.evaluation_schema import (
            GEvalRequest, RAGEvaluationRequest, ResponseQualityRequest
        )
        logger.success("✓ Evaluation schemas imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import evaluation schemas: {e}")
        return
    
    # Check endpoint
    try:
        from tldw_Server_API.app.api.v1.endpoints.evals import router
        logger.success("✓ Evaluation endpoint imported successfully")
    except ImportError as e:
        logger.error(f"✗ Failed to import evaluation endpoint: {e}")
        return
    
    logger.info("\nAll imports successful! The evaluation system is properly integrated.")
    
    # If API server is running, test the endpoints
    logger.info("\nTo test the API endpoints, ensure the server is running:")
    logger.info("  python -m uvicorn tldw_Server_API.app.main:app --reload")
    
    # Uncomment to run API tests
    # tester = EvaluationSystemTester()
    # results = await tester.run_all_tests()


if __name__ == "__main__":
    from typing import Dict, Any
    asyncio.run(main())