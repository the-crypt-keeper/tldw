#!/usr/bin/env python3
"""
Test script for MCP authentication functionality.

This script demonstrates:
1. Creating API clients
2. Authenticating with different methods
3. Accessing protected endpoints
4. WebSocket connections with authentication
"""

import asyncio
import json
import httpx
import websockets
from loguru import logger


class MCPAuthTester:
    """Test harness for MCP authentication"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_prefix = "/api/v1/mcp"
        
    async def test_authentication_flow(self):
        """Test complete authentication flow"""
        logger.info("Testing MCP Authentication Flow")
        
        async with httpx.AsyncClient() as client:
            # 1. First, authenticate with default admin credentials
            logger.info("\n1. Authenticating as admin...")
            auth_response = await client.post(
                f"{self.base_url}{self.api_prefix}/auth/token",
                json={
                    "api_key": "admin-secret-key"  # Default admin key
                }
            )
            
            if auth_response.status_code != 200:
                logger.error(f"Admin auth failed: {auth_response.text}")
                return
            
            admin_token = auth_response.json()["access_token"]
            logger.success(f"Admin authenticated successfully")
            
            # 2. Create a new client
            logger.info("\n2. Creating a new client...")
            headers = {"Authorization": f"Bearer {admin_token}"}
            
            create_response = await client.post(
                f"{self.base_url}{self.api_prefix}/auth/clients",
                headers=headers,
                json={
                    "name": "Test Client",
                    "role": "user",
                    "permissions": ["tools:read", "tools:execute", "context:read", "context:write"],
                    "allowed_tools": ["echo", "timestamp", "search_media"],
                    "rate_limit": 100
                }
            )
            
            if create_response.status_code != 200:
                logger.error(f"Client creation failed: {create_response.text}")
                return
            
            client_data = create_response.json()
            new_client_id = client_data["client"]["client_id"]
            new_api_key = client_data["api_key"]
            logger.success(f"Created client: {new_client_id}")
            logger.info(f"API Key: {new_api_key}")
            
            # 3. Authenticate with the new client's API key
            logger.info("\n3. Authenticating with new client's API key...")
            client_auth_response = await client.post(
                f"{self.base_url}{self.api_prefix}/auth/token",
                json={
                    "api_key": new_api_key
                }
            )
            
            if client_auth_response.status_code != 200:
                logger.error(f"Client auth failed: {client_auth_response.text}")
                return
            
            client_token = client_auth_response.json()["access_token"]
            logger.success("Client authenticated successfully")
            
            # 4. Test accessing protected endpoints
            logger.info("\n4. Testing protected endpoints...")
            
            # Get current client info
            me_response = await client.get(
                f"{self.base_url}{self.api_prefix}/auth/me",
                headers={"Authorization": f"Bearer {client_token}"}
            )
            
            if me_response.status_code == 200:
                logger.success(f"Current client: {me_response.json()['name']}")
            else:
                logger.error(f"Failed to get client info: {me_response.text}")
            
            # Try to list all clients (should fail for non-admin)
            list_response = await client.get(
                f"{self.base_url}{self.api_prefix}/auth/clients",
                headers={"Authorization": f"Bearer {client_token}"}
            )
            
            if list_response.status_code == 403:
                logger.info("✓ Non-admin correctly denied access to list clients")
            else:
                logger.warning(f"Unexpected response: {list_response.status_code}")
            
            # 5. Test WebSocket connection with authentication
            logger.info("\n5. Testing WebSocket connection with auth token...")
            await self.test_websocket_auth(client_token)
            
            # 6. Test rate limiting
            logger.info("\n6. Testing rate limiting...")
            await self.test_rate_limiting(client_token)
            
            # 7. Clean up - delete the test client
            logger.info("\n7. Cleaning up...")
            delete_response = await client.delete(
                f"{self.base_url}{self.api_prefix}/auth/clients/{new_client_id}",
                headers=headers  # Use admin token
            )
            
            if delete_response.status_code == 200:
                logger.success("Test client deleted")
            
    async def test_websocket_auth(self, auth_token: str):
        """Test WebSocket connection with authentication"""
        ws_url = f"ws://localhost:8000{self.api_prefix}/ws"
        
        try:
            async with websockets.connect(ws_url) as websocket:
                # Send CONNECT message with auth token
                connect_msg = {
                    "id": "connect-1",
                    "type": "connect",
                    "method": "connect",
                    "params": {
                        "client_id": "test-client",
                        "auth_token": auth_token,
                        "client_info": {
                            "name": "Test WebSocket Client",
                            "version": "1.0"
                        }
                    }
                }
                
                await websocket.send(json.dumps(connect_msg))
                
                # Receive response
                response = await websocket.recv()
                response_data = json.loads(response)
                
                if response_data.get("type") == "connect":
                    logger.success("WebSocket authenticated successfully")
                    session_id = response_data["result"]["session_id"]
                    logger.info(f"Session ID: {session_id}")
                    
                    # Test listing tools
                    list_tools_msg = {
                        "id": "list-1",
                        "type": "list_tools",
                        "method": "list_tools",
                        "params": {}
                    }
                    
                    await websocket.send(json.dumps(list_tools_msg))
                    tools_response = await websocket.recv()
                    tools_data = json.loads(tools_response)
                    
                    if "result" in tools_data:
                        tool_count = len(tools_data["result"]["tools"])
                        logger.success(f"Listed {tool_count} available tools")
                    
                    # Disconnect
                    disconnect_msg = {
                        "id": "disconnect-1",
                        "type": "disconnect",
                        "method": "disconnect",
                        "params": {}
                    }
                    await websocket.send(json.dumps(disconnect_msg))
                    
                else:
                    logger.error(f"WebSocket auth failed: {response_data}")
                    
        except Exception as e:
            logger.error(f"WebSocket test failed: {e}")
    
    async def test_rate_limiting(self, auth_token: str):
        """Test rate limiting functionality"""
        logger.info("Testing rate limiting...")
        
        # This would make many rapid requests to test rate limiting
        # For now, just demonstrate the concept
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {auth_token}"}
            
            # Make several requests
            for i in range(5):
                response = await client.get(
                    f"{self.base_url}{self.api_prefix}/status",
                    headers=headers
                )
                
                if response.status_code == 200:
                    logger.info(f"Request {i+1}: Success")
                elif response.status_code == 429:
                    logger.warning(f"Request {i+1}: Rate limited")
                    break
                else:
                    logger.error(f"Request {i+1}: Error {response.status_code}")
                
                await asyncio.sleep(0.1)  # Small delay between requests
    
    async def test_unauthorized_access(self):
        """Test unauthorized access attempts"""
        logger.info("\nTesting unauthorized access...")
        
        async with httpx.AsyncClient() as client:
            # Try to access protected endpoint without auth
            response = await client.get(
                f"{self.base_url}{self.api_prefix}/auth/me"
            )
            
            if response.status_code == 401:
                logger.success("✓ Unauthorized access correctly denied")
            else:
                logger.error(f"Unexpected response: {response.status_code}")
            
            # Try with invalid token
            response = await client.get(
                f"{self.base_url}{self.api_prefix}/auth/me",
                headers={"Authorization": "Bearer invalid-token"}
            )
            
            if response.status_code == 401:
                logger.success("✓ Invalid token correctly rejected")
            else:
                logger.error(f"Unexpected response: {response.status_code}")


async def main():
    """Run MCP authentication tests"""
    logger.info("MCP Authentication Test Suite")
    logger.info("=" * 60)
    
    tester = MCPAuthTester()
    
    # Make sure the server is running
    logger.info("Checking if server is running...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{tester.base_url}{tester.api_prefix}/health")
            if response.status_code != 200:
                logger.error("Server is not running. Start it with:")
                logger.error("  python -m uvicorn tldw_Server_API.app.main:app --reload")
                return
    except Exception as e:
        logger.error(f"Cannot connect to server: {e}")
        logger.error("Start the server with:")
        logger.error("  python -m uvicorn tldw_Server_API.app.main:app --reload")
        return
    
    logger.success("Server is running")
    
    # Run tests
    await tester.test_authentication_flow()
    await tester.test_unauthorized_access()
    
    logger.info("\n" + "=" * 60)
    logger.info("Authentication tests completed")


if __name__ == "__main__":
    asyncio.run(main())