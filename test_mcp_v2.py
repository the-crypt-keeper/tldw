#!/usr/bin/env python3
"""
Test script for MCP v2 server
"""

import asyncio
import json
import httpx
from typing import Dict, Any

BASE_URL = "http://localhost:8000/api/v1/mcp/v2"

async def test_mcp_v2():
    """Test MCP v2 endpoints"""
    
    async with httpx.AsyncClient() as client:
        print("Testing MCP v2 Server...")
        print("=" * 50)
        
        # Test 1: Server Status
        print("\n1. Testing server status...")
        try:
            response = await client.get(f"{BASE_URL}/status")
            if response.status_code == 200:
                status = response.json()
                print(f"✓ Server status: {status['status']}")
                print(f"  Version: {status['version']}")
                print(f"  Modules: {status['modules']['total']} registered")
                for module in status['modules']['registrations']:
                    print(f"    - {module['name']} v{module['version']} ({module['status']})")
            else:
                print(f"✗ Failed to get status: {response.status_code}")
        except Exception as e:
            print(f"✗ Error: {e}")
        
        # Test 2: List Tools
        print("\n2. Testing tool listing...")
        try:
            response = await client.get(f"{BASE_URL}/tools")
            if response.status_code == 200:
                tools_response = response.json()
                print(f"✓ Found {tools_response['count']} tools")
                for tool in tools_response['tools'][:5]:  # Show first 5
                    print(f"    - {tool['name']}: {tool['description']}")
            else:
                print(f"✗ Failed to list tools: {response.status_code}")
        except Exception as e:
            print(f"✗ Error: {e}")
        
        # Test 3: MCP Request via HTTP
        print("\n3. Testing MCP request (initialize)...")
        try:
            mcp_request = {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "clientInfo": {
                        "name": "Test Client",
                        "version": "1.0"
                    }
                },
                "id": "test-1"
            }
            response = await client.post(
                f"{BASE_URL}/request",
                json=mcp_request,
                params={"client_id": "test_client"}
            )
            if response.status_code == 200:
                result = response.json()
                print(f"✓ MCP initialized: {result['result']['serverInfo']['name']}")
                print(f"  Protocol version: {result['result']['protocolVersion']}")
            else:
                print(f"✗ Failed to initialize: {response.status_code}")
        except Exception as e:
            print(f"✗ Error: {e}")
        
        # Test 4: Execute a tool
        print("\n4. Testing tool execution (search_media)...")
        try:
            mcp_request = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "media.search_media",
                    "arguments": {
                        "query": "test",
                        "limit": 3
                    }
                },
                "id": "test-2"
            }
            response = await client.post(
                f"{BASE_URL}/request",
                json=mcp_request,
                params={"client_id": "test_client"}
            )
            if response.status_code == 200:
                result = response.json()
                if "result" in result:
                    print(f"✓ Tool executed successfully")
                    # Print first part of result
                    result_str = str(result['result'])[:200]
                    print(f"  Result preview: {result_str}...")
                elif "error" in result:
                    print(f"⚠ Tool execution returned error: {result['error']['message']}")
            else:
                print(f"✗ Failed to execute tool: {response.status_code}")
        except Exception as e:
            print(f"✗ Error: {e}")
        
        # Test 5: List modules
        print("\n5. Testing module listing...")
        try:
            response = await client.get(f"{BASE_URL}/modules")
            if response.status_code == 200:
                modules = response.json()
                print(f"✓ Found {modules['count']} modules")
                for module in modules['modules']:
                    print(f"    - {module['id']}: {module['name']} ({module['department']})")
                    print(f"      Capabilities: {', '.join(module['capabilities'])}")
            else:
                print(f"✗ Failed to list modules: {response.status_code}")
        except Exception as e:
            print(f"✗ Error: {e}")
        
        print("\n" + "=" * 50)
        print("MCP v2 tests complete!")

if __name__ == "__main__":
    print("MCP v2 Test Script")
    print("Make sure the tldw server is running on http://localhost:8000")
    print()
    asyncio.run(test_mcp_v2())