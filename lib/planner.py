"""Goal → evolution brief expansion via configurable planner harnesses.

Planner takes the user's goal field and produces a richer brief that Cerebras
workers then use for create / evaluate / build prompts.

Harness options (from cli-subagents skill defaults):
  - cerebras  — same API as workers (fast, free-tier models)
  - agy       — Gemini via Antigravity CLI
  - devin     — Devin CLI (swe-1-7, glm-5-2, …)
  - codex     — OpenAI Codex CLI
  - claude    — Claude Code CLI
  - none      — skip expansion; raw goal only
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from lib import llm


# Catalog for the Evolve UI (planner that writes prompts for Cerebras workers).
PLANNER_OPTIONS: list[dict[str, Any]] = [
    {
        "id": "none",
        "harness": "none",
        "label": "None — use raw goal as-is",
        "model": "",
    },
    {
        "id": "cerebras:gemma-4-31b",
        "harness": "cerebras",
        "label": "Cerebras · gemma-4-31b (fast planner)",
        "model": "gemma-4-31b",
    },
    {
        "id": "cerebras:gpt-oss-120b",
        "harness": "cerebras",
        "label": "Cerebras · gpt-oss-120b",
        "model": "gpt-oss-120b",
    },
    {
        "id": "cerebras:zai-glm-4.7",
        "harness": "cerebras",
        "label": "Cerebras · zai-glm-4.7",
        "model": "zai-glm-4.7",
    },
    {
        "id": "agy:gemini-3.1-pro-high",
        "harness": "agy",
        "label": "agy · Gemini 3.1 Pro (High)",
        # agy --model requires the exact display string from `agy models`
        "model": "Gemini 3.1 Pro (High)",
    },
    {
        "id": "agy:gemini-3.1-pro-low",
        "harness": "agy",
        "label": "agy · Gemini 3.1 Pro (Low)",
        "model": "Gemini 3.1 Pro (Low)",
    },
    {
        "id": "agy:gemini-3.5-flash-medium",
        "harness": "agy",
        "label": "agy · Gemini 3.5 Flash (Medium)",
        "model": "Gemini 3.5 Flash (Medium)",
    },
    {
        "id": "agy:gemini-3.5-flash-high",
        "harness": "agy",
        "label": "agy · Gemini 3.5 Flash (High)",
        "model": "Gemini 3.5 Flash (High)",
    },
    {
        "id": "agy:gemini-3.5-flash-low",
        "harness": "agy",
        "label": "agy · Gemini 3.5 Flash (Low)",
        "model": "Gemini 3.5 Flash (Low)",
    },
    # Devin free models only in the dropdown for now (verified live at boot via
    # `devin models list`). Today: GLM-5.2 High + SWE-1.7 Max.
    {
        "id": "devin:glm-5-2",
        "harness": "devin",
        "label": "Devin · GLM-5.2 High (free)",
        "model": "glm-5-2",
    },
    {
        "id": "devin:swe-1-7",
        "harness": "devin",
        "label": "Devin · SWE-1.7 Max (free)",
        "model": "swe-1-7",
    },
    {
        "id": "codex:gpt-5.6-sol",
        "harness": "codex",
        "label": "Codex · gpt-5.6-sol (max effort)",
        "model": "gpt-5.6-sol",
    },
    {
        "id": "claude:opus",
        "harness": "claude",
        "label": "Claude Code · opus (max effort)",
        "model": "opus",
    },
]


# agy CLI expects exact display names from `agy models`, not slug ids.
AGY_MODEL_ALIASES: dict[str, str] = {
    "gemini-3.5-flash-medium": "Gemini 3.5 Flash (Medium)",
    "gemini-3.5-flash-high": "Gemini 3.5 Flash (High)",
    "gemini-3.5-flash-low": "Gemini 3.5 Flash (Low)",
    "gemini-3.1-pro-high": "Gemini 3.1 Pro (High)",
    "gemini-3.1-pro-low": "Gemini 3.1 Pro (Low)",
    "claude-sonnet-4.6-thinking": "Claude Sonnet 4.6 (Thinking)",
    "claude-opus-4.6-thinking": "Claude Opus 4.6 (Thinking)",
    "gpt-oss-120b-medium": "GPT-OSS 120B (Medium)",
}


def openrouter_planner_options() -> list[dict[str, Any]]:
    """OpenRouter free models for planner / director dropdowns (when key present)."""
    if not llm.has_openrouter_key():
        return []
    out: list[dict[str, Any]] = []
    for entry in llm.openrouter_free_models():
        oid = entry.get("openrouter_id") or str(entry.get("id") or "").replace("openrouter:", "", 1)
        full_id = f"openrouter:{oid}"
        out.append({
            "id": full_id,
            "harness": "openrouter",
            "label": entry.get("label") or f"OpenRouter · {oid}",
            "model": oid,
            "provider": "openrouter",
            "free": True,
        })
    return out


def all_planner_options() -> list[dict[str, Any]]:
    """Static PLANNER_OPTIONS + live OpenRouter free entries."""
    return list(PLANNER_OPTIONS) + openrouter_planner_options()


def parse_planner_id(planner_id: str) -> tuple[str, str]:
    """Return (harness, model_token). Prefer resolve_planner() for real CLI model names."""
    planner_id = (planner_id or "none").strip()
    if planner_id == "none" or not planner_id:
        return "none", ""
    # openrouter:google/foo:free  (model id may contain colons)
    if planner_id.startswith("openrouter:"):
        return "openrouter", planner_id.split(":", 1)[1].strip()
    if ":" in planner_id:
        h, m = planner_id.split(":", 1)
        return h.strip(), m.strip()
    # bare cerebras model id
    if planner_id in llm.CEREBRAS_MODEL_QUOTAS:
        return "cerebras", planner_id
    # bare openrouter free id
    if planner_id.endswith(":free") or planner_id == "openrouter/free" or planner_id.startswith("openrouter/"):
        return "openrouter", planner_id
    return planner_id, ""


def resolve_planner(planner_id: str) -> dict[str, str]:
    """Map a planner catalog id → harness + CLI model name.

    Important: for agy, the id slug (e.g. gemini-3.5-flash-medium) is NOT accepted by
    `agy --model`; the catalog `model` field / AGY_MODEL_ALIASES holds the display name.
    """
    planner_id = (planner_id or "none").strip() or "none"
    for opt in all_planner_options():
        if opt.get("id") == planner_id:
            harness = str(opt.get("harness") or "none")
            model = str(opt.get("model") or "")
            if harness == "agy" and model and model not in AGY_MODEL_ALIASES.values():
                # if someone stored a slug in model, still alias it
                model = AGY_MODEL_ALIASES.get(model, model)
            return {
                "planner_id": planner_id,
                "harness": harness,
                "model": model,
                "label": str(opt.get("label") or planner_id),
            }
    harness, model = parse_planner_id(planner_id)
    if harness == "agy":
        model = AGY_MODEL_ALIASES.get(model, model)
        # bare display name already
        if model and " " not in model and model not in AGY_MODEL_ALIASES.values():
            # still a slug with no alias — leave as-is; CLI will error with available list
            pass
    return {
        "planner_id": planner_id,
        "harness": harness,
        "model": model,
        "label": planner_id,
    }


def expand_goal(
    goal: str,
    *,
    planner_id: str = "none",
    output_type: str = "auto",
    build_software: bool = True,
    run_id: Optional[str] = None,
    timeout_secs: int = 180,
) -> dict[str, Any]:
    """Expand a short goal into an evolution brief for worker prompts."""
    resolved = resolve_planner(planner_id)
    harness = resolved["harness"]
    model = resolved["model"]
    if harness == "none" or not goal.strip():
        return {
            "ok": True,
            "planner_id": "none",
            "harness": "none",
            "model": "",
            "brief": goal.strip(),
            "raw": goal.strip(),
            "duration_secs": 0,
        }

    prompt = _planner_prompt(goal, output_type=output_type, build_software=build_software)
    start = time.time()
    try:
        if harness == "cerebras":
            raw = llm.call_cerebras_sync(
                prompt,
                model=model or "gemma-4-31b",
                max_tokens=4096,
                run_id=run_id,
                purpose="plan_goal",
                temperature=0.4,
            )
        elif harness == "openrouter":
            raw = llm.call_openrouter_sync(
                prompt,
                model=model or "openrouter/free",
                max_tokens=4096,
                run_id=run_id,
                purpose="plan_goal",
                temperature=0.5,
            )
        elif harness == "agy":
            raw = _run_agy(prompt, model=model or "Gemini 3.1 Pro (High)", timeout_secs=timeout_secs)
        elif harness == "devin":
            raw = _run_devin(prompt, model=model or "swe-1-7", timeout_secs=timeout_secs)
        elif harness == "codex":
            raw = _run_codex(prompt, model=model or "gpt-5.6-sol", timeout_secs=timeout_secs)
        elif harness == "claude":
            raw = _run_claude(prompt, model=model or "opus", timeout_secs=timeout_secs)
        else:
            raise RuntimeError(f"unknown planner harness: {harness}")
        brief = raw.strip()
        return {
            "ok": True,
            "planner_id": planner_id,
            "harness": harness,
            "model": model,
            "brief": brief,
            "raw": raw,
            "duration_secs": round(time.time() - start, 2),
        }
    except Exception as e:
        return {
            "ok": False,
            "planner_id": planner_id,
            "harness": harness,
            "model": model,
            "brief": goal.strip(),  # fallback to raw goal
            "raw": "",
            "error": str(e),
            "duration_secs": round(time.time() - start, 2),
        }


def _planner_prompt(goal: str, output_type: str, build_software: bool) -> str:
    from lib import deployer_lens as dlens

    build_note = (
        "The evolution loop will scaffold and improve real source code each generation."
        if build_software
        else "Architecture-only evolution (no code build)."
    )
    return f"""You are the evolution planner for Dev Studio — writing for a DEPLOYER who ships products to make money.

{dlens.DEPLOYER_LENS}

The user wrote a short goal. Expand it into a precise EVOLUTION BRIEF that will be
handed to many cheap Cerebras worker calls (create genomes, score them, build software).

User goal:
{goal}

Output type preference: {output_type}
Build mode: {build_note}

Write a concise brief (plain text, 300–700 words) with these sections:
1) Product outcome — what shippable software means for the deployer
2) Must-have capabilities (bullet list) — include user-visible surface
3) Innovation angles (2–4 non-generic technical or product twists)
4) Authenticity + volume — real content strategy + how it scales
5) Architecture cells to explore (roles + responsibilities; include growth/monetization + deployer)
6) Build plan across generations (gen0 scaffold → later improve; always ship a money path surface)
7) Evaluation criteria for workers (what “good” looks like — monetization + shippability + goal_fit)
{dlens.planner_sections_extra()}
11) Risks / anti-patterns (generic CRUD, empty placeholders, products with no monetization, spammy content)

Do NOT wrap in JSON. Do NOT include secrets. Be concrete and implementable.
"""


def _which(name: str) -> Optional[str]:
    return shutil.which(name)


def _run_agy(prompt: str, model: str, timeout_secs: int) -> str:
    bin_path = _which("agy") or str(Path.home() / ".local/bin/agy")
    if not Path(bin_path).exists():
        raise RuntimeError("agy binary not found")
    cmd = [
        bin_path, "--print", prompt,
        "--model", model,
        "--mode", "accept-edits",
        "--dangerously-skip-permissions",
        "--print-timeout", f"{max(1, timeout_secs // 60)}m",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_secs)
    out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    if proc.returncode != 0 and not proc.stdout.strip():
        raise RuntimeError(f"agy failed ({proc.returncode}): {out[-800:]}")
    return proc.stdout.strip() or out.strip()


def _run_devin(
    prompt: str,
    model: str,
    timeout_secs: int,
    *,
    run_id: Optional[str] = None,
    purpose: str = "devin",
) -> str:
    bin_path = _which("devin") or str(Path.home() / ".local/bin/devin")
    if not Path(bin_path).exists():
        raise RuntimeError("devin binary not found")
    # Soft throttle vs estimated free-tier (includes external host processes)
    try:
        from lib import devin_usage as dusage
        throttle, reason = dusage.USAGE.should_throttle()
        if throttle:
            # Wait up to 90s for 3h window pressure to ease (external CLIs may finish)
            waited = 0.0
            while waited < 90 and throttle:
                time.sleep(min(5.0, 90 - waited))
                waited += 5.0
                throttle, reason = dusage.USAGE.should_throttle()
            if throttle:
                raise RuntimeError(
                    f"devin free-tier pressure high ({reason}); "
                    "wait for external Devin CLIs or the ~3h window to cool"
                )
    except RuntimeError:
        raise
    except Exception:
        pass

    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory(prefix="evo-plan-") as td:
            pf = Path(td) / "prompt.txt"
            pf.write_text(prompt, encoding="utf-8")
            cmd = [
                bin_path, "-p",
                "--model", model,
                "--permission-mode", "dangerous",
                "--prompt-file", str(pf),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_secs, cwd=td)
            out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
            text = proc.stdout.strip() or out.strip()
            ok = proc.returncode == 0 or bool(proc.stdout.strip())
            try:
                from lib import devin_usage as dusage
                dusage.record_studio_call(
                    model,
                    prompt,
                    text if ok else "",
                    ok=ok,
                    purpose=purpose,
                    run_id=run_id,
                    duration_secs=round(time.time() - t0, 3),
                    error=None if ok else f"rc={proc.returncode}: {out[-400:]}",
                )
            except Exception:
                pass
            if proc.returncode != 0 and not proc.stdout.strip():
                raise RuntimeError(f"devin failed ({proc.returncode}): {out[-800:]}")
            return text
    except Exception as e:
        try:
            from lib import devin_usage as dusage
            dusage.record_studio_call(
                model,
                prompt,
                "",
                ok=False,
                purpose=purpose,
                run_id=run_id,
                duration_secs=round(time.time() - t0, 3),
                error=str(e),
            )
        except Exception:
            pass
        raise


def _run_codex(prompt: str, model: str, timeout_secs: int) -> str:
    bin_path = _which("codex") or str(Path.home() / ".local/bin/codex")
    if not Path(bin_path).exists():
        raise RuntimeError("codex binary not found")
    with tempfile.TemporaryDirectory(prefix="evo-plan-codex-") as td:
        pf = Path(td) / "prompt.txt"
        pf.write_text(prompt, encoding="utf-8")
        last = Path(td) / "last.txt"
        cmd = [
            bin_path, "exec",
            "--json",
            "-m", model,
            "-c", 'model_reasoning_effort="max"',
            "-C", td,
            "-s", "workspace-write",
            "--dangerously-bypass-approvals-and-sandbox",
            "-o", str(last),
            "-",
        ]
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout_secs)
        if last.exists() and last.read_text(encoding="utf-8", errors="replace").strip():
            return last.read_text(encoding="utf-8", errors="replace").strip()
        out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
        if proc.returncode != 0 and not out.strip():
            raise RuntimeError(f"codex failed ({proc.returncode})")
        return out.strip()


def _run_claude(prompt: str, model: str, timeout_secs: int) -> str:
    bin_path = _which("claude")
    if not bin_path:
        raise RuntimeError("claude binary not found")
    cmd = [
        bin_path, "-p", prompt,
        "--model", model,
        "--effort", "max",
        "--dangerously-skip-permissions",
        "--output-format", "text",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_secs)
    out = (proc.stdout or "").strip()
    if proc.returncode != 0 and not out:
        raise RuntimeError(f"claude failed ({proc.returncode}): {(proc.stderr or '')[-800:]}")
    return out or (proc.stderr or "").strip()
