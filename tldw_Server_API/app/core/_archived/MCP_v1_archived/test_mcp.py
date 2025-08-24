# test_mcp.py - Simple MCP Server Test
"""
Simple test script to verify MCP server functionality.

Run this after starting the FastAPI server to test the MCP implementation.
"""

import asyncio
import json
import websockets
import uuid
from datetime import datetime


async def test_mcp_connection():
    """Test basic MCP server connection and operations"""
    
    uri = "ws://localhost:8000/api/v1/mcp/ws"
    
    async with websockets.connect(uri) as websocket:
        print("Connected to MCP server")
        
        # 1. Send CONNECT message
        connect_msg = {
            "id": str(uuid.uuid4()),
            "type": "connect",
            "version": "1.0",
            "timestamp": datetime.utcnow().isoformat(),
            "method": "connect",
            "params": {
                "client_id": "test_client_001",
                "client_info": {
                    "name": "Test Client",
                    "version": "1.0"
                }
            }
        }
        
        await websocket.send(json.dumps(connect_msg))
        response = await websocket.recv()
        connect_response = json.loads(response)
        print(f"Connect response: {json.dumps(connect_response, indent=2)}")
        
        # 2. List available tools
        list_tools_msg = {
            "id": str(uuid.uuid4()),
            "type": "list_tools",
            "version": "1.0",
            "timestamp": datetime.utcnow().isoformat(),
            "method": "list_tools"
        }
        
        await websocket.send(json.dumps(list_tools_msg))
        response = await websocket.recv()
        tools_response = json.loads(response)
        print(f"\nAvailable tools: {json.dumps(tools_response, indent=2)}")
        
        # 3. Execute a simple tool
        execute_msg = {
            "id": str(uuid.uuid4()),
            "type": "execute_tool",
            "version": "1.0",
            "timestamp": datetime.utcnow().isoformat(),
            "method": "execute_tool",
            "params": {
                "tool_name": "echo",
                "arguments": {
                    "message": "Hello from MCP test!"
                }
            }
        }
        
        await websocket.send(json.dumps(execute_msg))
        response = await websocket.recv()
        execute_response = json.loads(response)
        print(f"\nTool execution response: {json.dumps(execute_response, indent=2)}")
        
        # 4. Create a context
        create_context_msg = {
            "id": str(uuid.uuid4()),
            "type": "update_context",
            "version": "1.0",
            "timestamp": datetime.utcnow().isoformat(),
            "method": "update_context",
            "params": {
                "updates": {
                    "test_key": "test_value",
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        }
        
        await websocket.send(json.dumps(create_context_msg))
        response = await websocket.recv()
        context_response = json.loads(response)
        print(f"\nContext created: {json.dumps(context_response, indent=2)}")
        
        # Extract context ID
        context_id = context_response.get("result", {}).get("id")
        
        # 5. Get the context
        if context_id:
            get_context_msg = {
                "id": str(uuid.uuid4()),
                "type": "get_context",
                "version": "1.0",
                "timestamp": datetime.utcnow().isoformat(),
                "method": "get_context",
                "params": {
                    "context_id": context_id
                }
            }
            
            await websocket.send(json.dumps(get_context_msg))
            response = await websocket.recv()
            get_context_response = json.loads(response)
            print(f"\nRetrieved context: {json.dumps(get_context_response, indent=2)}")
        
        # 6. Send disconnect
        disconnect_msg = {
            "id": str(uuid.uuid4()),
            "type": "disconnect",
            "version": "1.0",
            "timestamp": datetime.utcnow().isoformat(),
            "method": "disconnect"
        }
        
        await websocket.send(json.dumps(disconnect_msg))
        response = await websocket.recv()
        disconnect_response = json.loads(response)
        print(f"\nDisconnect response: {json.dumps(disconnect_response, indent=2)}")


async def test_rest_endpoints():
    """Test MCP REST endpoints"""
    import aiohttp
    
    base_url = "http://localhost:8000/api/v1"
    
    async with aiohttp.ClientSession() as session:
        # Test server status
        async with session.get(f"{base_url}/mcp/status") as resp:
            status = await resp.json()
            print(f"Server status: {json.dumps(status, indent=2)}")
        
        # List tools
        async with session.get(f"{base_url}/mcp/tools") as resp:
            tools = await resp.json()
            print(f"\nTools via REST: {json.dumps(tools[:2], indent=2)}...")  # Show first 2
        
        # Create a context via REST
        context_data = {
            "name": "Test Context",
            "content": {"key": "value"},
            "metadata": {"created_by": "test"}
        }
        async with session.post(f"{base_url}/mcp/contexts", json=context_data) as resp:
            context = await resp.json()
            print(f"\nCreated context via REST: {json.dumps(context, indent=2)}")


async def main():
    """Run all tests"""
    print("=== Testing MCP WebSocket Connection ===")
    try:
        await test_mcp_connection()
    except Exception as e:
        print(f"WebSocket test failed: {e}")
    
    print("\n\n=== Testing MCP REST Endpoints ===")
    try:
        await test_rest_endpoints()
    except Exception as e:
        print(f"REST test failed: {e}")


if __name__ == "__main__":
    # Note: Make sure the FastAPI server is running first!
    # Start with: python -m uvicorn tldw_Server_API.app.main:app --reload
    print("Make sure the FastAPI server is running on localhost:8000")
    print("Starting MCP tests...\n")
    asyncio.run(main())