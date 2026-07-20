"""Lightweight MCP (Model Context Protocol) client for Dev Studio.

Supports stdio transports and simple HTTP/SSE URL transports.
MCP server definitions are read from:
  - dev-studio/mcp.json
  - ~/.cursor/mcp.json (if present)

Each entry under `mcpServers` can be:
  { "command": "npx", "args": [...] }   # stdio
  { "url": "https://.../mcp" }           # HTTP/SSE (basic)
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional


_lock = threading.Lock()
_stdio_sessions: dict[str, dict] = {}


def _config_paths() -> list[Path]:
    return [
        Path(__file__).resolve().parent.parent / "mcp.json",
        Path.home() / ".cursor" / "mcp.json",
    ]


def load_mcp_config() -> dict[str, dict]:
    """Load and merge MCP server configs. Later files override earlier ones."""
    merged: dict[str, dict] = {}
    for p in _config_paths():
        if not p.exists():
            continue
        try:
            cfg = json.loads(p.read_text())
            for name, spec in cfg.get("mcpServers", {}).items():
                merged[name] = dict(spec)
        except Exception:
            continue
    return merged


def list_servers() -> list[dict]:
    """Return server names and transport summaries (no tool discovery yet)."""
    out = []
    for name, spec in load_mcp_config().items():
        transport = "url" if "url" in spec else "stdio"
        out.append({
            "name": name,
            "transport": transport,
            "command": " ".join([spec.get("command", "")] + spec.get("args", [])) if transport == "stdio" else None,
            "url": spec.get("url"),
        })
    return out


def _rpc_message(method: str, params: dict) -> str:
    return json.dumps({"jsonrpc": "2.0", "id": str(uuid.uuid4().hex[:8]), "method": method, "params": params})


class MCPStdioClient:
    """JSON-RPC over stdio for an MCP server."""

    def __init__(self, name: str, spec: dict):
        self.name = name
        self.command = spec.get("command")
        self.args = spec.get("args", [])
        self.env = os.environ.copy()
        for k, v in spec.get("env", {}).items():
            self.env[k] = v
        self.proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._pending: dict[str, dict] = {}
        self._reader: Optional[threading.Thread] = None
        self._initialized = False

    def start(self, timeout: float = 30) -> None:
        if self.proc and self.proc.poll() is None:
            return
        if not self.command:
            raise RuntimeError(f"MCP server {self.name} has no command")
        self.proc = subprocess.Popen(
            [self.command, *self.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self.env,
            bufsize=1,
        )
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._call("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "dev-studio", "version": "0.5.0"},
        }, timeout=timeout)
        # send initialized notification
        self._send_raw(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        self._initialized = True

    def _send_raw(self, msg: str) -> None:
        if not self.proc or self.proc.stdin is None:
            raise RuntimeError("MCP process not running")
        self.proc.stdin.write(msg + "\n")
        self.proc.stdin.flush()

    def _read_loop(self) -> None:
        if not self.proc or self.proc.stdout is None:
            return
        for line in iter(self.proc.stdout.readline, ""):
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except Exception:
                continue
            msg_id = msg.get("id")
            with self._lock:
                entry = self._pending.get(msg_id) if msg_id else None
            if entry:
                entry["result"] = msg
                entry["event"].set()

    def _call(self, method: str, params: dict, timeout: float = 60) -> dict:
        if not self.proc or self.proc.poll() is not None:
            raise RuntimeError(f"MCP server {self.name} is not running")
        msg_id = str(uuid.uuid4().hex[:8])
        event = threading.Event()
        with self._lock:
            self._pending[msg_id] = {"event": event, "result": None}
        self._send_raw(_rpc_message(method, params))
        if not event.wait(timeout=timeout):
            with self._lock:
                self._pending.pop(msg_id, None)
            raise TimeoutError(f"MCP call {method} timed out")
        with self._lock:
            result = self._pending.pop(msg_id, {}).get("result")
        if result is None:
            raise RuntimeError(f"MCP call {method} got no response")
        if "error" in result:
            raise RuntimeError(result["error"].get("message", str(result["error"])))
        return result.get("result", {})

    def list_tools(self) -> list[dict]:
        self.start()
        res = self._call("tools/list", {})
        return res.get("tools", [])

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        self.start()
        return self._call("tools/call", {"name": tool_name, "arguments": arguments})

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=3)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass


class MCPHTTPClient:
    """Very basic HTTP client for URL-based MCP servers.

    Assumes a JSON endpoint that accepts {tool, args} and returns {result, error}.
    This is NOT a full SSE MCP implementation; it's a pragmatic bridge for
    servers that expose a simple HTTP tool surface.
    """

    def __init__(self, name: str, spec: dict):
        self.name = name
        self.url = spec.get("url")
        if not self.url:
            raise RuntimeError(f"MCP server {self.name} URL missing")
        self.headers = spec.get("headers", {})

    def list_tools(self) -> list[dict]:
        import requests
        r = requests.get(f"{self.url}/tools", headers=self.headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        return data.get("tools", [])

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        import requests
        r = requests.post(
            f"{self.url}/tools/{tool_name}",
            json=arguments,
            headers=self.headers,
            timeout=120,
        )
        r.raise_for_status()
        return r.json()


def _client_for(server_name: str) -> Any:
    cfg = load_mcp_config()
    spec = cfg.get(server_name)
    if not spec:
        raise RuntimeError(f"MCP server '{server_name}' not found in config")
    if "url" in spec:
        return MCPHTTPClient(server_name, spec)
    return MCPStdioClient(server_name, spec)


def get_cached_stdio_client(server_name: str) -> MCPStdioClient:
    with _lock:
        client = _stdio_sessions.get(server_name)
        if client is None:
            cfg = load_mcp_config()
            spec = cfg.get(server_name)
            if not spec:
                raise RuntimeError(f"MCP server '{server_name}' not found")
            if "url" in spec:
                raise RuntimeError(f"MCP server '{server_name}' is HTTP, not stdio")
            client = MCPStdioClient(server_name, spec)
            _stdio_sessions[server_name] = client
        return client


def list_tools(server_name: str) -> list[dict]:
    return _client_for(server_name).list_tools()


def call_tool(server_name: str, tool_name: str, arguments: dict) -> dict:
    return _client_for(server_name).call_tool(tool_name, arguments)


def close_stdio_sessions() -> None:
    with _lock:
        for client in list(_stdio_sessions.values()):
            client.stop()
        _stdio_sessions.clear()
