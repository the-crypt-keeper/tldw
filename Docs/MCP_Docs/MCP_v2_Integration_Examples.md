# MCP v2 Integration Examples

## Table of Contents
1. [Python Integration](#python-integration)
2. [JavaScript/TypeScript Integration](#javascripttypescript-integration)
3. [Go Integration](#go-integration)
4. [Rust Integration](#rust-integration)
5. [Shell/CLI Integration](#shellcli-integration)
6. [AI Assistant Integration](#ai-assistant-integration)
7. [IDE Extension Integration](#ide-extension-integration)
8. [Complete Application Examples](#complete-application-examples)

## Python Integration

### Basic Client Library

```python
# mcp_client.py
import asyncio
import httpx
import websockets
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MCPConfig:
    base_url: str = "http://localhost:8001/api/v1/mcp/v2"
    ws_url: str = "ws://localhost:8001/api/v1/mcp/v2/ws"
    token: Optional[str] = None
    timeout: int = 30

class MCPClient:
    """Python client for MCP v2 protocol"""
    
    def __init__(self, config: MCPConfig = MCPConfig()):
        self.config = config
        self.client = httpx.AsyncClient(timeout=config.timeout)
        self.request_id = 0
        
    def _next_id(self) -> str:
        """Generate unique request ID"""
        self.request_id += 1
        return f"py-{self.request_id}"
    
    def _headers(self) -> Dict[str, str]:
        """Build request headers"""
        headers = {"Content-Type": "application/json"}
        if self.config.token:
            headers["Authorization"] = f"Bearer {self.config.token}"
        return headers
    
    async def request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send MCP request via HTTP"""
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": self._next_id()
        }
        
        response = await self.client.post(
            f"{self.config.base_url}/request",
            json=request,
            headers=self._headers()
        )
        response.raise_for_status()
        
        result = response.json()
        if "error" in result:
            raise Exception(f"MCP Error: {result['error']}")
        
        return result.get("result", {})
    
    async def initialize(self, client_name: str = "Python Client") -> Dict[str, Any]:
        """Initialize MCP session"""
        return await self.request("initialize", {
            "clientInfo": {
                "name": client_name,
                "version": "1.0.0"
            }
        })
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools"""
        result = await self.request("tools/list")
        return result.get("tools", [])
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a specific tool"""
        return await self.request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
    
    async def list_resources(self) -> List[Dict[str, Any]]:
        """List available resources"""
        result = await self.request("resources/list")
        return result.get("resources", [])
    
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read a resource"""
        return await self.request("resources/read", {"uri": uri})
    
    async def close(self):
        """Close the client"""
        await self.client.aclose()

# Example usage
async def main():
    client = MCPClient()
    
    try:
        # Initialize
        await client.initialize("My Python App")
        
        # List tools
        tools = await client.list_tools()
        print(f"Available tools: {len(tools)}")
        
        # Search media
        results = await client.call_tool("media.search_media", {
            "query": "python programming",
            "limit": 5
        })
        print(f"Search results: {results}")
        
        # Create a note
        note = await client.call_tool("notes.create_note", {
            "title": "Python Integration Test",
            "content": "Successfully integrated with MCP v2!",
            "tags": ["python", "integration", "test"]
        })
        print(f"Created note: {note}")
        
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### WebSocket Streaming Client

```python
# mcp_websocket.py
import asyncio
import websockets
import json
from typing import AsyncGenerator, Dict, Any

class MCPWebSocketClient:
    """WebSocket client for streaming MCP operations"""
    
    def __init__(self, ws_url: str = "ws://localhost:8001/api/v1/mcp/v2/ws"):
        self.ws_url = ws_url
        self.websocket = None
        self.request_id = 0
        
    async def connect(self):
        """Connect to MCP WebSocket"""
        self.websocket = await websockets.connect(self.ws_url)
        
        # Initialize
        await self.send({
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "clientInfo": {
                    "name": "Python WebSocket Client",
                    "version": "1.0.0"
                }
            },
            "id": "init"
        })
        
        # Wait for initialization response
        response = await self.receive()
        if "error" in response:
            raise Exception(f"Initialization failed: {response['error']}")
            
    async def send(self, message: Dict[str, Any]):
        """Send message through WebSocket"""
        await self.websocket.send(json.dumps(message))
        
    async def receive(self) -> Dict[str, Any]:
        """Receive message from WebSocket"""
        message = await self.websocket.recv()
        return json.loads(message)
        
    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo"
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion"""
        self.request_id += 1
        
        await self.send({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "chat.stream_completion",
                "arguments": {
                    "messages": messages,
                    "model": model,
                    "stream": True
                }
            },
            "id": f"stream-{self.request_id}"
        })
        
        while True:
            response = await self.receive()
            
            if "result" in response:
                content = response["result"].get("content")
                if content:
                    yield content
                    
                if response["result"].get("done"):
                    break
                    
            elif "error" in response:
                raise Exception(f"Stream error: {response['error']}")
                
    async def close(self):
        """Close WebSocket connection"""
        if self.websocket:
            await self.websocket.close()

# Example usage
async def stream_example():
    client = MCPWebSocketClient()
    
    try:
        await client.connect()
        
        # Stream chat completion
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain quantum computing in simple terms."}
        ]
        
        print("Assistant: ", end="")
        async for chunk in client.stream_chat(messages):
            print(chunk, end="", flush=True)
        print()  # New line at end
        
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(stream_example())
```

### Research Assistant Example

```python
# research_assistant.py
import asyncio
from mcp_client import MCPClient
from typing import List, Dict, Any

class ResearchAssistant:
    """AI-powered research assistant using MCP"""
    
    def __init__(self, mcp_client: MCPClient):
        self.mcp = mcp_client
        
    async def research_topic(self, topic: str) -> Dict[str, Any]:
        """Comprehensive research on a topic"""
        research_data = {
            "topic": topic,
            "media_results": [],
            "notes": [],
            "summary": None
        }
        
        # Search for relevant media
        print(f"Searching for media about '{topic}'...")
        media_results = await self.mcp.call_tool("media.search_media", {
            "query": topic,
            "limit": 10
        })
        research_data["media_results"] = media_results.get("results", [])
        
        # Perform vector search for semantic matches
        print("Performing semantic search...")
        vector_results = await self.mcp.call_tool("rag.vector_search", {
            "query": topic,
            "top_k": 5
        })
        
        # Get transcripts for top results
        transcripts = []
        for item in research_data["media_results"][:3]:
            print(f"Getting transcript for: {item.get('title', 'Unknown')}")
            transcript = await self.mcp.call_tool("media.get_transcript", {
                "media_id": item["id"]
            })
            if transcript.get("success"):
                transcripts.append(transcript["transcript"])
        
        # Generate contextual summary
        if transcripts:
            print("Generating summary...")
            context = "\n\n".join(transcripts[:2])  # Use first 2 transcripts
            
            summary = await self.mcp.call_tool("chat.chat_completion", {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a research assistant. Summarize the key points about the topic based on the provided context."
                    },
                    {
                        "role": "user",
                        "content": f"Topic: {topic}\n\nContext:\n{context[:3000]}\n\nProvide a comprehensive summary."
                    }
                ],
                "max_tokens": 500
            })
            research_data["summary"] = summary.get("content")
        
        # Create research note
        print("Creating research note...")
        note = await self.mcp.call_tool("notes.create_note", {
            "title": f"Research: {topic}",
            "content": self._format_research_note(research_data),
            "tags": ["research", topic.lower().replace(" ", "-")]
        })
        
        return research_data
    
    def _format_research_note(self, data: Dict[str, Any]) -> str:
        """Format research data as markdown note"""
        note = f"# Research: {data['topic']}\n\n"
        
        if data["summary"]:
            note += f"## Summary\n\n{data['summary']}\n\n"
        
        note += "## Sources\n\n"
        for item in data["media_results"][:5]:
            note += f"- **{item.get('title', 'Unknown')}**\n"
            if item.get("url"):
                note += f"  - URL: {item['url']}\n"
            if item.get("description"):
                note += f"  - {item['description']}\n"
            note += "\n"
        
        return note

# Example usage
async def research_example():
    client = MCPClient()
    assistant = ResearchAssistant(client)
    
    try:
        await client.initialize("Research Assistant")
        
        # Research a topic
        research = await assistant.research_topic("artificial intelligence ethics")
        
        print("\n" + "="*50)
        print("RESEARCH COMPLETE")
        print("="*50)
        print(f"Found {len(research['media_results'])} media items")
        if research["summary"]:
            print(f"\nSummary:\n{research['summary']}")
            
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(research_example())
```

## JavaScript/TypeScript Integration

### TypeScript Client Library

```typescript
// mcp-client.ts
interface MCPConfig {
  baseUrl?: string;
  wsUrl?: string;
  token?: string;
  timeout?: number;
}

interface MCPRequest {
  jsonrpc: "2.0";
  method: string;
  params?: any;
  id: string | number;
}

interface MCPResponse {
  jsonrpc: "2.0";
  result?: any;
  error?: {
    code: number;
    message: string;
    data?: any;
  };
  id: string | number;
}

class MCPClient {
  private config: Required<MCPConfig>;
  private requestId = 0;

  constructor(config: MCPConfig = {}) {
    this.config = {
      baseUrl: config.baseUrl || "http://localhost:8001/api/v1/mcp/v2",
      wsUrl: config.wsUrl || "ws://localhost:8001/api/v1/mcp/v2/ws",
      token: config.token || "",
      timeout: config.timeout || 30000,
    };
  }

  private nextId(): string {
    return `js-${++this.requestId}`;
  }

  private getHeaders(): HeadersInit {
    const headers: HeadersInit = {
      "Content-Type": "application/json",
    };
    
    if (this.config.token) {
      headers["Authorization"] = `Bearer ${this.config.token}`;
    }
    
    return headers;
  }

  async request(method: string, params?: any): Promise<any> {
    const request: MCPRequest = {
      jsonrpc: "2.0",
      method,
      params: params || {},
      id: this.nextId(),
    };

    const response = await fetch(`${this.config.baseUrl}/request`, {
      method: "POST",
      headers: this.getHeaders(),
      body: JSON.stringify(request),
      signal: AbortSignal.timeout(this.config.timeout),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result: MCPResponse = await response.json();
    
    if (result.error) {
      throw new Error(`MCP Error: ${result.error.message}`);
    }

    return result.result;
  }

  async initialize(clientName = "JavaScript Client"): Promise<any> {
    return this.request("initialize", {
      clientInfo: {
        name: clientName,
        version: "1.0.0",
      },
    });
  }

  async listTools(): Promise<any[]> {
    const result = await this.request("tools/list");
    return result.tools || [];
  }

  async callTool(name: string, args: any): Promise<any> {
    return this.request("tools/call", {
      name,
      arguments: args,
    });
  }

  async searchMedia(query: string, limit = 10): Promise<any> {
    return this.callTool("media.search_media", { query, limit });
  }

  async createNote(title: string, content: string, tags: string[] = []): Promise<any> {
    return this.callTool("notes.create_note", { title, content, tags });
  }
}

// WebSocket client for streaming
class MCPWebSocketClient {
  private ws: WebSocket | null = null;
  private config: Required<MCPConfig>;
  private requestId = 0;
  private handlers = new Map<string, (response: any) => void>();

  constructor(config: MCPConfig = {}) {
    this.config = {
      baseUrl: config.baseUrl || "http://localhost:8001/api/v1/mcp/v2",
      wsUrl: config.wsUrl || "ws://localhost:8001/api/v1/mcp/v2/ws",
      token: config.token || "",
      timeout: config.timeout || 30000,
    };
  }

  async connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      const url = this.config.token 
        ? `${this.config.wsUrl}?token=${this.config.token}`
        : this.config.wsUrl;
        
      this.ws = new WebSocket(url);

      this.ws.onopen = async () => {
        // Initialize
        await this.send({
          jsonrpc: "2.0",
          method: "initialize",
          params: {
            clientInfo: {
              name: "JavaScript WebSocket Client",
              version: "1.0.0",
            },
          },
          id: "init",
        });
        
        resolve();
      };

      this.ws.onerror = (error) => {
        reject(error);
      };

      this.ws.onmessage = (event) => {
        const response = JSON.parse(event.data);
        const handler = this.handlers.get(response.id);
        
        if (handler) {
          handler(response);
        }
      };
    });
  }

  private send(message: any): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
        reject(new Error("WebSocket not connected"));
        return;
      }
      
      this.ws.send(JSON.stringify(message));
      resolve();
    });
  }

  async streamChat(
    messages: Array<{ role: string; content: string }>,
    onChunk: (chunk: string) => void
  ): Promise<void> {
    const id = `stream-${++this.requestId}`;
    
    return new Promise((resolve, reject) => {
      this.handlers.set(id, (response) => {
        if (response.error) {
          reject(new Error(response.error.message));
          return;
        }
        
        if (response.result?.content) {
          onChunk(response.result.content);
        }
        
        if (response.result?.done) {
          this.handlers.delete(id);
          resolve();
        }
      });

      this.send({
        jsonrpc: "2.0",
        method: "tools/call",
        params: {
          name: "chat.stream_completion",
          arguments: {
            messages,
            stream: true,
          },
        },
        id,
      });
    });
  }

  close(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

// Example usage
async function example() {
  const client = new MCPClient();
  
  try {
    // Initialize
    await client.initialize("My App");
    
    // List tools
    const tools = await client.listTools();
    console.log(`Available tools: ${tools.length}`);
    
    // Search media
    const results = await client.searchMedia("javascript tutorials");
    console.log(`Found ${results.count} results`);
    
    // Create note
    const note = await client.createNote(
      "JavaScript Integration",
      "Successfully integrated with MCP v2!",
      ["javascript", "integration"]
    );
    console.log("Note created:", note);
    
  } catch (error) {
    console.error("Error:", error);
  }
}

// Streaming example
async function streamExample() {
  const wsClient = new MCPWebSocketClient();
  
  try {
    await wsClient.connect();
    
    const messages = [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "Tell me a short story." }
    ];
    
    process.stdout.write("Assistant: ");
    await wsClient.streamChat(messages, (chunk) => {
      process.stdout.write(chunk);
    });
    console.log(); // New line
    
  } finally {
    wsClient.close();
  }
}

export { MCPClient, MCPWebSocketClient };
```

### React Integration

```tsx
// useMCP.tsx - React Hook for MCP
import { useState, useEffect, useCallback } from 'react';
import { MCPClient } from './mcp-client';

interface UseMCPOptions {
  autoInitialize?: boolean;
  token?: string;
}

export function useMCP(options: UseMCPOptions = {}) {
  const [client] = useState(() => new MCPClient({ token: options.token }));
  const [initialized, setInitialized] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    if (options.autoInitialize) {
      client.initialize()
        .then(() => setInitialized(true))
        .catch(setError);
    }
  }, []);

  const callTool = useCallback(async (name: string, args: any) => {
    setLoading(true);
    setError(null);
    
    try {
      const result = await client.callTool(name, args);
      return result;
    } catch (err) {
      setError(err as Error);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [client]);

  return {
    client,
    initialized,
    loading,
    error,
    callTool,
  };
}

// MediaSearch.tsx - React Component
import React, { useState } from 'react';
import { useMCP } from './useMCP';

export function MediaSearch() {
  const { callTool, loading, error } = useMCP({ autoInitialize: true });
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<any[]>([]);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    
    try {
      const response = await callTool('media.search_media', {
        query,
        limit: 10,
      });
      setResults(response.results || []);
    } catch (err) {
      console.error('Search failed:', err);
    }
  };

  return (
    <div>
      <form onSubmit={handleSearch}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search media..."
          disabled={loading}
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Searching...' : 'Search'}
        </button>
      </form>

      {error && <div className="error">Error: {error.message}</div>}

      <div className="results">
        {results.map((item) => (
          <div key={item.id} className="result-item">
            <h3>{item.title}</h3>
            <p>{item.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
```

## Go Integration

```go
// mcp_client.go
package mcp

import (
    "bytes"
    "context"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "sync/atomic"
    "time"
)

type Config struct {
    BaseURL string
    Token   string
    Timeout time.Duration
}

type Request struct {
    JSONRPC string      `json:"jsonrpc"`
    Method  string      `json:"method"`
    Params  interface{} `json:"params,omitempty"`
    ID      interface{} `json:"id"`
}

type Response struct {
    JSONRPC string          `json:"jsonrpc"`
    Result  json.RawMessage `json:"result,omitempty"`
    Error   *Error          `json:"error,omitempty"`
    ID      interface{}     `json:"id"`
}

type Error struct {
    Code    int         `json:"code"`
    Message string      `json:"message"`
    Data    interface{} `json:"data,omitempty"`
}

type Client struct {
    config    Config
    client    *http.Client
    requestID int64
}

func NewClient(config Config) *Client {
    if config.BaseURL == "" {
        config.BaseURL = "http://localhost:8001/api/v1/mcp/v2"
    }
    if config.Timeout == 0 {
        config.Timeout = 30 * time.Second
    }

    return &Client{
        config: config,
        client: &http.Client{
            Timeout: config.Timeout,
        },
    }
}

func (c *Client) nextID() string {
    id := atomic.AddInt64(&c.requestID, 1)
    return fmt.Sprintf("go-%d", id)
}

func (c *Client) Request(ctx context.Context, method string, params interface{}) (json.RawMessage, error) {
    req := Request{
        JSONRPC: "2.0",
        Method:  method,
        Params:  params,
        ID:      c.nextID(),
    }

    body, err := json.Marshal(req)
    if err != nil {
        return nil, fmt.Errorf("marshal request: %w", err)
    }

    httpReq, err := http.NewRequestWithContext(
        ctx,
        "POST",
        c.config.BaseURL+"/request",
        bytes.NewReader(body),
    )
    if err != nil {
        return nil, fmt.Errorf("create request: %w", err)
    }

    httpReq.Header.Set("Content-Type", "application/json")
    if c.config.Token != "" {
        httpReq.Header.Set("Authorization", "Bearer "+c.config.Token)
    }

    resp, err := c.client.Do(httpReq)
    if err != nil {
        return nil, fmt.Errorf("send request: %w", err)
    }
    defer resp.Body.Close()

    respBody, err := io.ReadAll(resp.Body)
    if err != nil {
        return nil, fmt.Errorf("read response: %w", err)
    }

    var response Response
    if err := json.Unmarshal(respBody, &response); err != nil {
        return nil, fmt.Errorf("unmarshal response: %w", err)
    }

    if response.Error != nil {
        return nil, fmt.Errorf("MCP error %d: %s", response.Error.Code, response.Error.Message)
    }

    return response.Result, nil
}

func (c *Client) Initialize(ctx context.Context, clientName string) error {
    params := map[string]interface{}{
        "clientInfo": map[string]string{
            "name":    clientName,
            "version": "1.0.0",
        },
    }

    _, err := c.Request(ctx, "initialize", params)
    return err
}

func (c *Client) CallTool(ctx context.Context, name string, arguments interface{}) (json.RawMessage, error) {
    params := map[string]interface{}{
        "name":      name,
        "arguments": arguments,
    }

    return c.Request(ctx, "tools/call", params)
}

// Example usage
func Example() {
    client := NewClient(Config{
        Token: "your-token",
    })

    ctx := context.Background()

    // Initialize
    if err := client.Initialize(ctx, "Go Client"); err != nil {
        panic(err)
    }

    // Search media
    result, err := client.CallTool(ctx, "media.search_media", map[string]interface{}{
        "query": "golang tutorials",
        "limit": 5,
    })
    if err != nil {
        panic(err)
    }

    fmt.Printf("Search results: %s\n", result)
}
```

## Rust Integration

```rust
// mcp_client.rs
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use reqwest::{Client, header};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct MCPConfig {
    pub base_url: String,
    pub token: Option<String>,
    pub timeout: std::time::Duration,
}

impl Default for MCPConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:8001/api/v1/mcp/v2".to_string(),
            token: None,
            timeout: std::time::Duration::from_secs(30),
        }
    }
}

#[derive(Debug, Serialize)]
struct MCPRequest {
    jsonrpc: String,
    method: String,
    params: Option<Value>,
    id: String,
}

#[derive(Debug, Deserialize)]
struct MCPResponse {
    jsonrpc: String,
    result: Option<Value>,
    error: Option<MCPError>,
    id: String,
}

#[derive(Debug, Deserialize)]
struct MCPError {
    code: i32,
    message: String,
    data: Option<Value>,
}

pub struct MCPClient {
    config: MCPConfig,
    client: Client,
    request_id: Arc<AtomicU64>,
}

impl MCPClient {
    pub fn new(config: MCPConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let mut headers = header::HeaderMap::new();
        headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/json"),
        );
        
        if let Some(token) = &config.token {
            headers.insert(
                header::AUTHORIZATION,
                header::HeaderValue::from_str(&format!("Bearer {}", token))?,
            );
        }

        let client = Client::builder()
            .timeout(config.timeout)
            .default_headers(headers)
            .build()?;

        Ok(Self {
            config,
            client,
            request_id: Arc::new(AtomicU64::new(0)),
        })
    }

    fn next_id(&self) -> String {
        let id = self.request_id.fetch_add(1, Ordering::SeqCst);
        format!("rust-{}", id)
    }

    pub async fn request(
        &self,
        method: &str,
        params: Option<Value>,
    ) -> Result<Value, Box<dyn std::error::Error>> {
        let request = MCPRequest {
            jsonrpc: "2.0".to_string(),
            method: method.to_string(),
            params,
            id: self.next_id(),
        };

        let response = self
            .client
            .post(format!("{}/request", self.config.base_url))
            .json(&request)
            .send()
            .await?;

        let mcp_response: MCPResponse = response.json().await?;

        if let Some(error) = mcp_response.error {
            return Err(format!("MCP Error {}: {}", error.code, error.message).into());
        }

        Ok(mcp_response.result.unwrap_or(Value::Null))
    }

    pub async fn initialize(&self, client_name: &str) -> Result<(), Box<dyn std::error::Error>> {
        let params = json!({
            "clientInfo": {
                "name": client_name,
                "version": "1.0.0"
            }
        });

        self.request("initialize", Some(params)).await?;
        Ok(())
    }

    pub async fn call_tool(
        &self,
        name: &str,
        arguments: Value,
    ) -> Result<Value, Box<dyn std::error::Error>> {
        let params = json!({
            "name": name,
            "arguments": arguments
        });

        self.request("tools/call", Some(params)).await
    }
}

// Example usage
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = MCPClient::new(MCPConfig::default())?;

    // Initialize
    client.initialize("Rust Client").await?;

    // Search media
    let result = client.call_tool(
        "media.search_media",
        json!({
            "query": "rust programming",
            "limit": 5
        }),
    ).await?;

    println!("Search results: {:?}", result);

    Ok(())
}
```

## Shell/CLI Integration

### Bash Script Client

```bash
#!/bin/bash
# mcp_cli.sh - Simple MCP CLI client

MCP_BASE_URL="${MCP_BASE_URL:-http://localhost:8001/api/v1/mcp/v2}"
MCP_TOKEN="${MCP_TOKEN:-}"
REQUEST_ID=0

# Generate next request ID
next_id() {
    REQUEST_ID=$((REQUEST_ID + 1))
    echo "bash-$REQUEST_ID"
}

# Send MCP request
mcp_request() {
    local method="$1"
    local params="${2:-{}}"
    local id=$(next_id)
    
    local auth_header=""
    if [ -n "$MCP_TOKEN" ]; then
        auth_header="-H \"Authorization: Bearer $MCP_TOKEN\""
    fi
    
    local request=$(cat <<EOF
{
    "jsonrpc": "2.0",
    "method": "$method",
    "params": $params,
    "id": "$id"
}
EOF
    )
    
    curl -s -X POST \
        -H "Content-Type: application/json" \
        $auth_header \
        -d "$request" \
        "$MCP_BASE_URL/request" | jq .
}

# Initialize session
mcp_init() {
    local client_name="${1:-Bash Client}"
    mcp_request "initialize" "{\"clientInfo\": {\"name\": \"$client_name\", \"version\": \"1.0.0\"}}"
}

# Call a tool
mcp_tool() {
    local tool_name="$1"
    shift
    local args="$@"
    
    mcp_request "tools/call" "{\"name\": \"$tool_name\", \"arguments\": $args}"
}

# Search media
mcp_search() {
    local query="$1"
    local limit="${2:-10}"
    
    mcp_tool "media.search_media" "{\"query\": \"$query\", \"limit\": $limit}"
}

# Create note
mcp_note() {
    local title="$1"
    local content="$2"
    local tags="${3:-[]}"
    
    mcp_tool "notes.create_note" "{\"title\": \"$title\", \"content\": \"$content\", \"tags\": $tags}"
}

# List tools
mcp_list_tools() {
    mcp_request "tools/list" | jq -r '.result.tools[].name'
}

# Interactive mode
mcp_interactive() {
    echo "MCP Interactive Shell"
    echo "Type 'help' for commands, 'quit' to exit"
    
    # Initialize
    mcp_init "Interactive Shell" > /dev/null
    echo "Session initialized"
    
    while true; do
        read -p "mcp> " cmd args
        
        case "$cmd" in
            help)
                echo "Commands:"
                echo "  search <query> - Search media"
                echo "  note <title> <content> - Create note"
                echo "  tools - List available tools"
                echo "  call <tool> <args> - Call any tool"
                echo "  quit - Exit"
                ;;
            search)
                mcp_search "$args" | jq -r '.result.results[] | "\(.title): \(.description)"'
                ;;
            note)
                read -p "Title: " title
                read -p "Content: " content
                mcp_note "$title" "$content" | jq .
                ;;
            tools)
                mcp_list_tools
                ;;
            call)
                tool=$(echo "$args" | cut -d' ' -f1)
                json_args=$(echo "$args" | cut -d' ' -f2-)
                mcp_tool "$tool" "$json_args" | jq .
                ;;
            quit|exit)
                echo "Goodbye!"
                break
                ;;
            *)
                echo "Unknown command: $cmd (type 'help' for commands)"
                ;;
        esac
    done
}

# Main script
if [ "$1" = "interactive" ]; then
    mcp_interactive
else
    # Example usage
    echo "Initializing MCP session..."
    mcp_init
    
    echo -e "\nSearching for 'python tutorials'..."
    mcp_search "python tutorials" 3
    
    echo -e "\nCreating a note..."
    mcp_note "Test Note" "This is a test note from bash" "[\"bash\", \"test\"]"
fi
```

### PowerShell Client

```powershell
# mcp_client.ps1 - PowerShell MCP client

$global:MCP_BASE_URL = $env:MCP_BASE_URL ?? "http://localhost:8001/api/v1/mcp/v2"
$global:MCP_TOKEN = $env:MCP_TOKEN ?? ""
$global:RequestId = 0

function Get-NextId {
    $global:RequestId++
    return "ps-$($global:RequestId)"
}

function Invoke-MCPRequest {
    param(
        [string]$Method,
        [hashtable]$Params = @{}
    )
    
    $headers = @{
        "Content-Type" = "application/json"
    }
    
    if ($global:MCP_TOKEN) {
        $headers["Authorization"] = "Bearer $($global:MCP_TOKEN)"
    }
    
    $request = @{
        jsonrpc = "2.0"
        method = $Method
        params = $Params
        id = Get-NextId
    } | ConvertTo-Json -Depth 10
    
    $response = Invoke-RestMethod `
        -Uri "$($global:MCP_BASE_URL)/request" `
        -Method POST `
        -Headers $headers `
        -Body $request
    
    if ($response.error) {
        throw "MCP Error: $($response.error.message)"
    }
    
    return $response.result
}

function Initialize-MCPSession {
    param(
        [string]$ClientName = "PowerShell Client"
    )
    
    Invoke-MCPRequest -Method "initialize" -Params @{
        clientInfo = @{
            name = $ClientName
            version = "1.0.0"
        }
    }
}

function Invoke-MCPTool {
    param(
        [string]$Name,
        [hashtable]$Arguments
    )
    
    Invoke-MCPRequest -Method "tools/call" -Params @{
        name = $Name
        arguments = $Arguments
    }
}

function Search-MCPMedia {
    param(
        [string]$Query,
        [int]$Limit = 10
    )
    
    Invoke-MCPTool -Name "media.search_media" -Arguments @{
        query = $Query
        limit = $Limit
    }
}

function New-MCPNote {
    param(
        [string]$Title,
        [string]$Content,
        [string[]]$Tags = @()
    )
    
    Invoke-MCPTool -Name "notes.create_note" -Arguments @{
        title = $Title
        content = $Content
        tags = $Tags
    }
}

# Example usage
Write-Host "Initializing MCP session..." -ForegroundColor Green
Initialize-MCPSession

Write-Host "`nSearching for 'powershell tutorials'..." -ForegroundColor Yellow
$results = Search-MCPMedia -Query "powershell tutorials" -Limit 5
$results.results | ForEach-Object {
    Write-Host "- $($_.title)" -ForegroundColor Cyan
}

Write-Host "`nCreating a note..." -ForegroundColor Yellow
$note = New-MCPNote `
    -Title "PowerShell Integration" `
    -Content "Successfully integrated PowerShell with MCP v2!" `
    -Tags @("powershell", "integration", "test")
Write-Host "Note created with ID: $($note.note_id)" -ForegroundColor Green
```

## AI Assistant Integration

### Claude Desktop Integration

```json
// claude_desktop_config.json
{
  "mcpServers": {
    "tldw": {
      "command": "node",
      "args": ["./tldw-mcp-bridge.js"],
      "env": {
        "MCP_BASE_URL": "http://localhost:8001/api/v1/mcp/v2",
        "MCP_TOKEN": "your-token-here"
      }
    }
  }
}
```

```javascript
// tldw-mcp-bridge.js - Bridge for Claude Desktop
const { Server } = require('@modelcontextprotocol/sdk/server/index.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
const axios = require('axios');

const MCP_BASE_URL = process.env.MCP_BASE_URL || 'http://localhost:8001/api/v1/mcp/v2';
const MCP_TOKEN = process.env.MCP_TOKEN || '';

class TLDWMCPBridge {
  constructor() {
    this.server = new Server(
      {
        name: 'tldw-mcp-bridge',
        version: '1.0.0',
      },
      {
        capabilities: {
          tools: {},
          resources: {},
        },
      }
    );
    
    this.setupHandlers();
  }
  
  async mcpRequest(method, params = {}) {
    const response = await axios.post(
      `${MCP_BASE_URL}/request`,
      {
        jsonrpc: '2.0',
        method,
        params,
        id: `bridge-${Date.now()}`,
      },
      {
        headers: {
          'Content-Type': 'application/json',
          ...(MCP_TOKEN && { Authorization: `Bearer ${MCP_TOKEN}` }),
        },
      }
    );
    
    if (response.data.error) {
      throw new Error(response.data.error.message);
    }
    
    return response.data.result;
  }
  
  setupHandlers() {
    // List available tools
    this.server.setRequestHandler('tools/list', async () => {
      const result = await this.mcpRequest('tools/list');
      return { tools: result.tools };
    });
    
    // Execute tool
    this.server.setRequestHandler('tools/call', async (request) => {
      const { name, arguments: args } = request.params;
      const result = await this.mcpRequest('tools/call', {
        name,
        arguments: args,
      });
      return result;
    });
    
    // List resources
    this.server.setRequestHandler('resources/list', async () => {
      const result = await this.mcpRequest('resources/list');
      return { resources: result.resources };
    });
    
    // Read resource
    this.server.setRequestHandler('resources/read', async (request) => {
      const { uri } = request.params;
      const result = await this.mcpRequest('resources/read', { uri });
      return result;
    });
  }
  
  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('tldw MCP bridge started');
  }
}

// Start the bridge
const bridge = new TLDWMCPBridge();
bridge.run().catch(console.error);
```

## IDE Extension Integration

### VS Code Extension

```typescript
// extension.ts - VS Code extension using MCP
import * as vscode from 'vscode';
import { MCPClient } from './mcp-client';

export function activate(context: vscode.ExtensionContext) {
    const client = new MCPClient({
        baseUrl: vscode.workspace.getConfiguration('tldw').get('mcpUrl'),
        token: vscode.workspace.getConfiguration('tldw').get('mcpToken'),
    });

    // Command: Search documentation
    const searchCommand = vscode.commands.registerCommand(
        'tldw.searchDocs',
        async () => {
            const query = await vscode.window.showInputBox({
                prompt: 'Search documentation',
                placeHolder: 'Enter search query...',
            });

            if (!query) return;

            try {
                await client.initialize('VS Code Extension');
                const results = await client.callTool('media.search_media', {
                    query,
                    limit: 10,
                });

                // Show results in quickpick
                const items = results.results.map((item: any) => ({
                    label: item.title,
                    description: item.description,
                    detail: item.url,
                    item,
                }));

                const selected = await vscode.window.showQuickPick(items, {
                    placeHolder: 'Select item to view',
                });

                if (selected) {
                    // Get transcript and show in new document
                    const transcript = await client.callTool('media.get_transcript', {
                        media_id: selected.item.id,
                    });

                    const doc = await vscode.workspace.openTextDocument({
                        content: transcript.transcript,
                        language: 'markdown',
                    });
                    await vscode.window.showTextDocument(doc);
                }
            } catch (error: any) {
                vscode.window.showErrorMessage(`Search failed: ${error.message}`);
            }
        }
    );

    // Command: Create note from selection
    const createNoteCommand = vscode.commands.registerCommand(
        'tldw.createNote',
        async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;

            const selection = editor.document.getText(editor.selection);
            if (!selection) {
                vscode.window.showWarningMessage('Please select text to create a note');
                return;
            }

            const title = await vscode.window.showInputBox({
                prompt: 'Note title',
                placeHolder: 'Enter note title...',
            });

            if (!title) return;

            try {
                await client.initialize('VS Code Extension');
                const note = await client.callTool('notes.create_note', {
                    title,
                    content: selection,
                    tags: ['vscode', 'code-snippet'],
                });

                vscode.window.showInformationMessage(
                    `Note created: ${note.note_id}`
                );
            } catch (error: any) {
                vscode.window.showErrorMessage(
                    `Failed to create note: ${error.message}`
                );
            }
        }
    );

    // Status bar item showing connection status
    const statusBar = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Right,
        100
    );
    statusBar.text = '$(cloud) tldw: Disconnected';
    statusBar.command = 'tldw.connect';
    statusBar.show();

    const connectCommand = vscode.commands.registerCommand(
        'tldw.connect',
        async () => {
            try {
                await client.initialize('VS Code Extension');
                statusBar.text = '$(cloud) tldw: Connected';
                vscode.window.showInformationMessage('Connected to tldw MCP server');
            } catch (error: any) {
                statusBar.text = '$(cloud-offline) tldw: Error';
                vscode.window.showErrorMessage(
                    `Failed to connect: ${error.message}`
                );
            }
        }
    );

    context.subscriptions.push(
        searchCommand,
        createNoteCommand,
        connectCommand,
        statusBar
    );
}

export function deactivate() {}
```

## Complete Application Examples

### Research Dashboard (Next.js)

```tsx
// pages/api/mcp/[...path].ts - API Proxy
import { NextApiRequest, NextApiResponse } from 'next';
import httpProxy from 'http-proxy-middleware';

const proxy = httpProxy.createProxyMiddleware({
    target: process.env.MCP_BASE_URL || 'http://localhost:8001',
    changeOrigin: true,
    pathRewrite: {
        '^/api/mcp': '/api/v1/mcp/v2',
    },
});

export default function handler(req: NextApiRequest, res: NextApiResponse) {
    return proxy(req, res);
}

// hooks/useMCPQuery.ts - React Query integration
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { MCPClient } from '@/lib/mcp-client';

const client = new MCPClient();

export function useMCPQuery(tool: string, args: any, options = {}) {
    return useQuery({
        queryKey: ['mcp', tool, args],
        queryFn: () => client.callTool(tool, args),
        ...options,
    });
}

export function useMCPMutation(tool: string) {
    const queryClient = useQueryClient();
    
    return useMutation({
        mutationFn: (args: any) => client.callTool(tool, args),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['mcp'] });
        },
    });
}

// components/ResearchDashboard.tsx
import { useState } from 'react';
import { useMCPQuery, useMCPMutation } from '@/hooks/useMCPQuery';

export function ResearchDashboard() {
    const [topic, setTopic] = useState('');
    const [selectedMedia, setSelectedMedia] = useState<any>(null);
    
    // Query for search results
    const { data: searchResults, isLoading } = useMCPQuery(
        'media.search_media',
        { query: topic, limit: 20 },
        { enabled: !!topic }
    );
    
    // Mutation for creating notes
    const createNote = useMCPMutation('notes.create_note');
    
    // Query for transcript
    const { data: transcript } = useMCPQuery(
        'media.get_transcript',
        { media_id: selectedMedia?.id },
        { enabled: !!selectedMedia }
    );
    
    const handleCreateNote = async () => {
        if (!selectedMedia || !transcript) return;
        
        await createNote.mutateAsync({
            title: `Research: ${selectedMedia.title}`,
            content: transcript.transcript,
            tags: ['research', topic],
        });
    };
    
    return (
        <div className="dashboard">
            <div className="search-section">
                <input
                    type="text"
                    value={topic}
                    onChange={(e) => setTopic(e.target.value)}
                    placeholder="Research topic..."
                />
            </div>
            
            <div className="results-section">
                {isLoading && <div>Searching...</div>}
                
                {searchResults?.results?.map((item: any) => (
                    <div
                        key={item.id}
                        className="result-card"
                        onClick={() => setSelectedMedia(item)}
                    >
                        <h3>{item.title}</h3>
                        <p>{item.description}</p>
                    </div>
                ))}
            </div>
            
            {transcript && (
                <div className="transcript-section">
                    <h2>{selectedMedia.title}</h2>
                    <pre>{transcript.transcript}</pre>
                    <button onClick={handleCreateNote}>
                        Save as Note
                    </button>
                </div>
            )}
        </div>
    );
}
```

---

These examples demonstrate various ways to integrate with the MCP v2 API across different programming languages and use cases. Each example includes proper error handling, authentication, and follows best practices for the respective language/framework.