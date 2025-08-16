#!/usr/bin/env python3
"""
Basic test script to verify Evaluations module functionality.
Tests the OpenAI-compatible evaluation API without embeddings.
"""

import asyncio
import json
import httpx
from datetime import datetime

# Test configuration
BASE_URL = "http://localhost:8000"
API_KEY = "default-secret-key-for-single-user"  # Default dev key

# Test data
test_evaluation = {
    "name": "test_summarization_eval",
    "description": "Test evaluation for summarization",
    "eval_type": "model_graded",
    "eval_spec": {
        "sub_type": "summarization",
        "evaluator_model": "openai",
        "metrics": ["fluency", "consistency", "relevance", "coherence"],
        "threshold": 0.7
    },
    "dataset": [
        {
            "input": {
                "source_text": "The quick brown fox jumps over the lazy dog. This is a classic pangram that contains all letters of the English alphabet. It has been used for decades to test typewriters and keyboards.",
                "summary": "A pangram containing all letters used for testing typing equipment."
            }
        }
    ]
}

async def test_create_evaluation():
    """Test creating an evaluation"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/v1/evals",
            json=test_evaluation,
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if response.status_code == 201:
            print("✅ Evaluation created successfully")
            eval_data = response.json()
            print(f"   ID: {eval_data['id']}")
            print(f"   Name: {eval_data['name']}")
            return eval_data['id']
        else:
            print(f"❌ Failed to create evaluation: {response.status_code}")
            print(f"   Response: {response.text}")
            return None

async def test_list_evaluations():
    """Test listing evaluations"""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/v1/evals",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Listed {len(data['data'])} evaluations")
            for eval in data['data'][:3]:  # Show first 3
                print(f"   - {eval['name']} ({eval['id']})")
        else:
            print(f"❌ Failed to list evaluations: {response.status_code}")

async def test_create_run(eval_id: str):
    """Test creating an evaluation run"""
    run_request = {
        "target_model": "gpt-3.5-turbo",
        "config": {
            "temperature": 0.0,
            "max_workers": 1,
            "timeout_seconds": 60
        }
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/v1/evals/{eval_id}/runs",
            json=run_request,
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if response.status_code == 202:
            print("✅ Evaluation run started")
            run_data = response.json()
            print(f"   Run ID: {run_data['id']}")
            print(f"   Status: {run_data['status']}")
            return run_data['id']
        else:
            print(f"❌ Failed to start run: {response.status_code}")
            print(f"   Response: {response.text}")
            return None

async def test_check_run_status(run_id: str):
    """Check the status of a run"""
    async with httpx.AsyncClient() as client:
        # Check status periodically
        for _ in range(10):  # Check up to 10 times
            response = await client.get(
                f"{BASE_URL}/v1/runs/{run_id}",
                headers={"Authorization": f"Bearer {API_KEY}"}
            )
            
            if response.status_code == 200:
                run_data = response.json()
                print(f"   Run status: {run_data['status']}")
                
                if run_data['status'] in ['completed', 'failed']:
                    return run_data['status']
                    
            await asyncio.sleep(2)  # Wait 2 seconds before checking again
        
        return "timeout"

async def test_exact_match_evaluation():
    """Test a simpler exact match evaluation that doesn't need LLM"""
    exact_match_eval = {
        "name": "test_exact_match",
        "description": "Test exact match evaluation",
        "eval_type": "exact_match",
        "eval_spec": {
            "threshold": 1.0
        },
        "dataset": [
            {
                "input": {"output": "hello world"},
                "expected": {"output": "hello world"}
            },
            {
                "input": {"output": "foo bar"},
                "expected": {"output": "foo bar"}
            }
        ]
    }
    
    async with httpx.AsyncClient() as client:
        # Create evaluation
        response = await client.post(
            f"{BASE_URL}/v1/evals",
            json=exact_match_eval,
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        
        if response.status_code == 201:
            eval_id = response.json()['id']
            print(f"✅ Exact match evaluation created: {eval_id}")
            
            # Run evaluation
            run_response = await client.post(
                f"{BASE_URL}/v1/evals/{eval_id}/runs",
                json={
                    "target_model": "none",  # No model needed for exact match
                    "config": {"temperature": 0}
                },
                headers={"Authorization": f"Bearer {API_KEY}"}
            )
            
            if run_response.status_code == 202:
                run_id = run_response.json()['id']
                print(f"✅ Exact match run started: {run_id}")
                
                # Wait and check results
                await asyncio.sleep(2)
                status = await test_check_run_status(run_id)
                
                if status == "completed":
                    # Get results
                    results_response = await client.get(
                        f"{BASE_URL}/v1/runs/{run_id}/results",
                        headers={"Authorization": f"Bearer {API_KEY}"}
                    )
                    
                    if results_response.status_code == 200:
                        results = results_response.json()
                        print("✅ Exact match evaluation completed successfully")
                        print(f"   Results: {json.dumps(results['results']['aggregate'], indent=2)}")
                    else:
                        print(f"❌ Failed to get results: {results_response.status_code}")
                else:
                    print(f"❌ Run failed or timed out: {status}")
            else:
                print(f"❌ Failed to start exact match run: {run_response.status_code}")
        else:
            print(f"❌ Failed to create exact match evaluation: {response.status_code}")

async def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Evaluations Module")
    print("=" * 60)
    print()
    
    # Test basic CRUD operations
    print("1. Testing evaluation creation...")
    eval_id = await test_create_evaluation()
    print()
    
    print("2. Testing evaluation listing...")
    await test_list_evaluations()
    print()
    
    # Test exact match (doesn't need LLM)
    print("3. Testing exact match evaluation (no LLM required)...")
    await test_exact_match_evaluation()
    print()
    
    # Only test LLM-based evaluation if we created one successfully
    if eval_id:
        print("4. Testing evaluation run (requires LLM API key)...")
        print("   Note: This will fail without a valid OpenAI API key")
        run_id = await test_create_run(eval_id)
        
        if run_id:
            print("   Checking run status...")
            final_status = await test_check_run_status(run_id)
            
            if final_status == "completed":
                print("   ✅ Run completed successfully")
            elif final_status == "failed":
                print("   ⚠️ Run failed (likely due to missing API key)")
            else:
                print("   ⏱️ Run timed out or still running")
    
    print()
    print("=" * 60)
    print("Testing complete!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())