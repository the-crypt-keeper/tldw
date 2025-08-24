# ⚠️ ARCHIVED MODULES - DO NOT USE

## These modules have been archived and replaced by the unified MCP module

### Archived Modules
- **MCP_v1_archived/** - Original MCP implementation (had hardcoded secrets)
- **MCP_v2_archived/** - Second MCP implementation (had security vulnerabilities)

### Replacement
All functionality has been consolidated into the new **MCP_unified** module with:
- ✅ No hardcoded secrets
- ✅ Enhanced security (JWT, RBAC, rate limiting)
- ✅ Production-ready features
- ✅ Better performance and monitoring

### Migration
Use the new unified module instead:

```python
# OLD - DO NOT USE
from tldw_Server_API.app.core.MCP import MCPServer
from tldw_Server_API.app.core.MCP_v2 import MCPServer

# NEW - USE THIS
from tldw_Server_API.app.core.MCP_unified import MCPServer, get_mcp_server
```

### Why Archived?
1. **Security Issues**: Hardcoded JWT secrets and API keys
2. **No Rate Limiting**: Vulnerable to abuse
3. **Poor Error Handling**: Generic errors, no proper codes
4. **No Health Checks**: No monitoring capabilities
5. **No Tests**: Zero test coverage

### Important
These modules are kept for reference only. **DO NOT** restore or use them in any deployment.

**Archived Date**: 2024-08-24
**Archived By**: Security Review Team
**Reason**: Critical security vulnerabilities and replaced by unified implementation