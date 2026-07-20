"""Run multi-harness agents (cerebras/agy/devin/codex/claude) from worker threads.

Mirrors server._run_export_agent without importing FastAPI.
"""
from __future__ import annotations

from typing import Optional

from lib import llm
from lib.planner import resolve_planner
from lib import planner as planmod


def run_agent(
    agent_id: str,
    prompt: str,
    *,
    run_id: Optional[str] = None,
    purpose: str = "agent",
    max_tokens: int = 4096,
    temperature: float = 0.35,
    timeout_secs: int = 300,
) -> str:
    resolved = resolve_planner(agent_id)
    harness = resolved["harness"]
    model = resolved["model"]
    if harness == "none":
        return ""
    if harness == "cerebras":
        return llm.call_cerebras_sync(
            prompt,
            model=model or "gemma-4-31b",
            max_tokens=max_tokens,
            run_id=run_id,
            purpose=purpose,
            temperature=temperature,
        )
    if harness == "openrouter":
        return llm.call_openrouter_sync(
            prompt,
            model=model or "openrouter/free",
            max_tokens=max_tokens,
            run_id=run_id,
            purpose=purpose,
            temperature=temperature,
        )
    if harness == "agy":
        return planmod._run_agy(prompt, model=model or "Gemini 3.1 Pro (High)", timeout_secs=timeout_secs)
    if harness == "devin":
        return planmod._run_devin(prompt, model=model or "swe-1-7", timeout_secs=timeout_secs)
    if harness == "codex":
        return planmod._run_codex(prompt, model=model or "gpt-5.6-sol", timeout_secs=timeout_secs)
    if harness == "claude":
        return planmod._run_claude(prompt, model=model or "opus", timeout_secs=timeout_secs)
    raise RuntimeError(f"unknown agent harness: {harness}")
