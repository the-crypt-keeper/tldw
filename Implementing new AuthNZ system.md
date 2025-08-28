Implementing new AuthNZ system



⏺ Complete Guide: Implementing Multi-User AuthNZ in Your Endpoints

  1. Environment Configuration

  First, set up your environment for multi-user mode:

  # .env file
  AUTH_MODE=multi_user
  DATABASE_URL=postgresql://user:password@localhost/tldw_db
  JWT_SECRET_KEY=your-very-secure-secret-key-at-least-32-chars
  ENABLE_REGISTRATION=true
  REQUIRE_REGISTRATION_CODE=false  # Set to true for invite-only

  2. Basic Endpoint Protection

  Here's how to protect your endpoints with authentication:

  # app/api/v1/endpoints/your_endpoint.py
  from fastapi import APIRouter, Depends, HTTPException, status
  from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User
  from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_current_active_user

  router = APIRouter(prefix="/your-resource", tags=["your-resource"])

  # Method 1: Basic authentication required
  @router.get("/protected")
  async def protected_endpoint(current_user: User = Depends(get_request_user)):
      """Any authenticated user can access this"""
      return {"message": f"Hello {current_user.username}"}

  # Method 2: Active user required (checks is_active flag)
  @router.get("/active-only")
  async def active_users_only(current_user: User = Depends(get_current_active_user)):
      """Only active users can access this"""
      return {"user_id": current_user.id, "username": current_user.username}

  # Method 3: Role-based access control
  async def require_admin(current_user: User = Depends(get_current_active_user)):
      if current_user.role != "admin":
          raise HTTPException(
              status_code=status.HTTP_403_FORBIDDEN,
              detail="Admin access required"
          )
      return current_user

  @router.post("/admin-only")
  async def admin_endpoint(admin_user: User = Depends(require_admin)):
      """Only admins can access this"""
      return {"message": "Admin action performed"}

  3. Advanced Authorization Patterns

  # app/api/v1/deps/authorization.py
  from typing import List
  from fastapi import Depends, HTTPException, status
  from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user

  def require_roles(allowed_roles: List[str]):
      """Factory function for role-based access control"""
      async def role_checker(current_user: User = Depends(get_request_user)):
          if current_user.role not in allowed_roles:
              raise HTTPException(
                  status_code=status.HTTP_403_FORBIDDEN,
                  detail=f"Requires one of these roles: {allowed_roles}"
              )
          return current_user
      return role_checker

  def require_permissions(required_permissions: List[str]):
      """Check specific permissions"""
      async def permission_checker(current_user: User = Depends(get_request_user)):
          # Fetch user permissions from database
          user_permissions = await get_user_permissions(current_user.id)

          for permission in required_permissions:
              if permission not in user_permissions:
                  raise HTTPException(
                      status_code=status.HTTP_403_FORBIDDEN,
                      detail=f"Missing required permission: {permission}"
                  )
          return current_user
      return permission_checker

  # Usage examples
  @router.get("/moderator-content")
  async def moderator_content(
      user: User = Depends(require_roles(["admin", "moderator"]))
  ):
      return {"message": "Moderator or admin content"}

  @router.post("/create-item")
  async def create_item(
      user: User = Depends(require_permissions(["items.create"]))
  ):
      return {"message": "Item created"}

  4. Resource Ownership Protection

  # Protect resources based on ownership
  @router.get("/my-items/{item_id}")
  async def get_user_item(
      item_id: int,
      current_user: User = Depends(get_request_user),
      db = Depends(get_db)
  ):
      # Check if item belongs to user
      item = await db.fetchone(
          "SELECT * FROM items WHERE id = $1 AND user_id = $2",
          item_id, current_user.id
      )

      if not item:
          raise HTTPException(
              status_code=status.HTTP_404_NOT_FOUND,
              detail="Item not found or access denied"
          )

      return item

  # Allow owner or admin access
  @router.put("/items/{item_id}")
  async def update_item(
      item_id: int,
      current_user: User = Depends(get_request_user),
      db = Depends(get_db)
  ):
      # Check ownership or admin role
      if current_user.role != "admin":
          item = await db.fetchone(
              "SELECT user_id FROM items WHERE id = $1",
              item_id
          )
          if not item or item["user_id"] != current_user.id:
              raise HTTPException(
                  status_code=status.HTTP_403_FORBIDDEN,
                  detail="Access denied"
              )

      # Update item...
      return {"message": "Item updated"}

  5. Rate Limiting Per User

  from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_user_rate_limit

  @router.post("/expensive-operation")
  async def expensive_operation(
      current_user: User = Depends(get_request_user),
      _: None = Depends(check_user_rate_limit)
  ):
      """Rate limited per user"""
      # Perform expensive operation
      return {"result": "completed"}

  6. Audit Logging

  from tldw_Server_API.app.core.Audit.unified_audit_service import (
      get_unified_audit_service, AuditEventType, AuditContext
  )

  @router.delete("/sensitive-data/{id}")
  async def delete_sensitive_data(
      id: int,
      current_user: User = Depends(get_request_user),
      request: Request
  ):
      # Log the action
      audit_service = await get_unified_audit_service()
      await audit_service.log_event(
          event_type=AuditEventType.DATA_DELETION,
          details={
              "resource_id": id,
              "resource_type": "sensitive_data",
              "action": "delete"
          },
          user_id=current_user.id,
          context=AuditContext(
              ip_address=request.client.host,
              user_agent=request.headers.get("User-Agent")
          )
      )

      # Perform deletion
      return {"message": "Data deleted"}

  7. Client-Side Implementation

  # client.py - Python client example
  import httpx
  from typing import Optional

  class TLDWClient:
      def __init__(self, base_url: str = "http://localhost:8000"):
          self.base_url = base_url
          self.client = httpx.AsyncClient()
          self.access_token: Optional[str] = None
          self.refresh_token: Optional[str] = None

      async def login(self, username: str, password: str):
          response = await self.client.post(
              f"{self.base_url}/api/v1/auth/login",
              data={"username": username, "password": password}
          )
          response.raise_for_status()
          data = response.json()
          self.access_token = data["access_token"]
          self.refresh_token = data["refresh_token"]
          return data

      async def request(self, method: str, path: str, **kwargs):
          headers = kwargs.pop("headers", {})
          if self.access_token:
              headers["Authorization"] = f"Bearer {self.access_token}"

          response = await self.client.request(
              method, f"{self.base_url}{path}", headers=headers, **kwargs
          )

          # Auto-refresh on 401
          if response.status_code == 401 and self.refresh_token:
              await self._refresh_token()
              headers["Authorization"] = f"Bearer {self.access_token}"
              response = await self.client.request(
                  method, f"{self.base_url}{path}", headers=headers, **kwargs
              )

          return response

      async def _refresh_token(self):
          response = await self.client.post(
              f"{self.base_url}/api/v1/auth/refresh",
              json={"refresh_token": self.refresh_token}
          )
          response.raise_for_status()
          data = response.json()
          self.access_token = data["access_token"]

  # Usage
  client = TLDWClient()
  await client.login("username", "password")
  response = await client.request("GET", "/api/v1/protected-endpoint")

  8. Testing Your Protected Endpoints

  # tests/test_protected_endpoints.py
  import pytest
  from httpx import AsyncClient

  @pytest.fixture
  async def authenticated_client(test_user):
      """Fixture that provides an authenticated client"""
      client = AsyncClient(app=app, base_url="http://test")

      # Login
      response = await client.post("/api/v1/auth/login", data={
          "username": test_user["username"],
          "password": test_user["password"]
      })
      token = response.json()["access_token"]

      # Set auth header
      client.headers["Authorization"] = f"Bearer {token}"
      return client

  async def test_protected_endpoint(authenticated_client):
      response = await authenticated_client.get("/api/v1/protected")
      assert response.status_code == 200

  async def test_unauthorized_access():
      client = AsyncClient(app=app, base_url="http://test")
      response = await client.get("/api/v1/protected")
      assert response.status_code == 401

  9. Migration from Single-User to Multi-User

  # migration_helper.py
  async def migrate_to_multi_user():
      """Helper to migrate existing single-user data to multi-user"""

      # 1. Create default admin user
      from tldw_Server_API.app.services.registration_service import get_registration_service

      reg_service = await get_registration_service()
      admin_user = await reg_service.register_user(
          username="admin",
          email="admin@example.com",
          password="SecureAdminPass123!",
          role="admin"
      )

      # 2. Assign existing data to admin user
      db = await get_db_pool()
      async with db.transaction() as conn:
          # Update all existing records with the admin user_id
          await conn.execute(
              "UPDATE media_items SET user_id = $1 WHERE user_id IS NULL",
              admin_user.id
          )

      print(f"Migration complete. Admin user created: {admin_user.username}")

  Key Points:

  1. Always use dependency injection for user authentication
  2. Check both authentication AND authorization (logged in vs. allowed)
  3. Use role-based or permission-based access for fine-grained control
  4. Log sensitive operations for audit trails
  5. Implement rate limiting for expensive operations
  6. Test both authorized and unauthorized access scenarios

  The AuthNZ module provides all the building blocks - you just need to apply them consistently across your endpoints!