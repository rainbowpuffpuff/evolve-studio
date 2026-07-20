#!/usr/bin/env python3
"""Evolve Studio — product evolution (extracted from Dev Studio)."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from lib import projects as pm
from lib import mcp_client as mcp
from lib import llm
from lib.evolution import (
    EvolutionConfig,
    EvolutionEngine,
    EVOLUTION_LLM_MODEL,
    EVOLUTION_LLM_MODELS,
    EVOLUTION_PROVIDER_OPTIONS,
    resolve_worker_model_pool,
    worker_pool_catalog,
)
from lib import llm as llm_mod
from lib import system_stats
from lib.planner import PLANNER_OPTIONS
from lib import evolution_export as evo_export

# ── paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent  # evolve-studio/
STUDIO = ROOT
DATA_DIR = Path(os.environ.get("STUDIO_DATA_DIR", str(STUDIO))).resolve()
PROJECTS_ROOT = DATA_DIR / "projects"
STATIC = STUDIO / "static"
JOBS = DATA_DIR / "jobs"
LESSONS_FILE = DATA_DIR / "lessons.json"
ARCHIVES = DATA_DIR / "archives"
ELICIT_RESEARCH = ROOT / "research" / "elicit" / "studio-searches"

# load local .env if present (and home config for optional tools)
for _envfile in (STUDIO / ".env", Path.home() / ".config" / "elicit" / "env", ROOT / ".env.elicit"):
    if _envfile.is_file():
        for _line in _envfile.read_text().splitlines():
            if "=" in _line and not _line.strip().startswith("#"):
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip("'\""))

# Reload Cerebras multi-key pool after .env is in place (llm may have imported earlier)
try:
    from lib import llm as _llm_keys
    _n = _llm_keys.KEY_POOL.reload()
    if _n:
        print(f"cerebras keys: {_n} loaded ({', '.join(_llm_keys.KEY_POOL.fingerprints())})", flush=True)
except Exception as _ke:
    print(f"cerebras keys: reload failed: {_ke}", flush=True)

GROK_BIN = Path(shutil.which("grok") or Path.home() / ".grok" / "bin" / "grok")
DEVIN_BIN = Path(shutil.which("devin") or Path.home() / ".local" / "bin" / "devin")
AGY_BIN = Path(shutil.which("agy") or Path.home() / ".local" / "bin" / "agy")
PI_BIN = Path(shutil.which("pi") or Path.home() / ".local" / "npm-global" / "bin" / "pi")
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"
ELICIT_CLI = Path.home() / ".grok" / "skills" / "elicit" / "scripts" / "elicit_cli.py"
ELICIT_DIGEST = Path.home() / ".grok" / "skills" / "elicit" / "scripts" / "search_and_digest.py"

MEMORY_ROOT = Path("/home/q/Downloads/memory")
MEMORY_CLI = MEMORY_ROOT / "easy-memory"
MEMORY_VAULT = MEMORY_ROOT / "obsidian-vault"

for d in (PROJECTS_ROOT, JOBS, STATIC, ARCHIVES):
    pass  # placeholder
for d in (PROJECTS_ROOT, JOBS, STATIC, ARCHIVES):
    d.mkdir(parents=True, exist_ok=True)

# Serialize Devin jobs — the Devin CLI SIGPIPEs (exit -13) when too many
# instances run concurrently. A semaphore caps Devin at 3 simultaneous runs.
_DEVIN_SEMAPHORE = threading.Semaphore(3)


# Track running subprocesses and in-process (SDK) jobs so the user can stop them.
_running_procs: dict[str, subprocess.Popen] = {}
_running_procs_lock = threading.Lock()
# Cerebras SDK jobs don't have a PID; we set a cancel flag the worker polls.
_cancel_flags: set[str] = set()
_cancel_flags_lock = threading.Lock()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reconcile_orphaned_jobs() -> None:
    """On startup, mark any jobs left in 'running'/'queued' as failed."""
    for p in JOBS.glob("*.json"):
        try:
            j = json.loads(p.read_text())
            if j.get("status") in ("running", "queued"):
                j["status"] = "failed"
                j["error"] = "orphaned by server restart"
                j["exit_code"] = -15
                j["updated_at"] = utcnow()
                p.write_text(json.dumps(j, indent=2))
        except Exception:
            pass


def _reconcile_orphaned_evolutions(evolutions_root: Path) -> int:
    """On startup, mark evolutions stuck in running/queued as interrupted.

    Evolution workers are in-process threads. A server restart kills them while
    evolution.json can still say status=running — the UI then lies forever.

    NEVER deletes evolution.json, candidates, llm_calls, or exports/*.html.
    Status becomes 'failed' with a clear error so Resume can pick up later.
    """
    n = 0
    if not evolutions_root.exists():
        return 0
    for p in evolutions_root.glob("*/evolution.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            st = (data.get("status") or "").lower()
            if st not in ("running", "queued", "starting"):
                continue
            # Preserve any product HTML flags for the UI
            exports = p.parent / "exports"
            has_html = False
            if exports.is_dir():
                has_html = any(
                    f.is_file() and f.suffix == ".html" and f.stat().st_size > 50
                    for f in exports.iterdir()
                )
            data["status"] = "failed"
            data["error"] = (
                "interrupted by server restart — worker thread died while status was still "
                f"'{st}'. All on-disk progress kept (evolution.json, gen*/candidates, "
                f"exports HTML{' including PRODUCT-latest.html' if has_html else ''}). "
                "Use Resume to continue."
            )
            data["updated_at"] = utcnow()
            data["interrupted_by_restart"] = True
            data["has_product_html"] = has_html
            events = data.get("events") or []
            events.append({
                "ts": utcnow(),
                "generation": data.get("current_generation") or 0,
                "type": "error",
                "message": data["error"],
            })
            data["events"] = events
            p.write_text(json.dumps(data, indent=2), encoding="utf-8")
            n += 1
        except Exception:
            continue
    return n


_reconcile_orphaned_jobs()
PM = pm.ProjectManager(PROJECTS_ROOT)

EVOLUTIONS_ROOT = DATA_DIR / "evolutions"
EVOLUTIONS_ROOT.mkdir(parents=True, exist_ok=True)
_orphaned_evos = _reconcile_orphaned_evolutions(EVOLUTIONS_ROOT)
if _orphaned_evos:
    print(f"reconciled {_orphaned_evos} orphaned evolution(s) after restart", flush=True)
EVO_PM = pm.ProjectManager(EVOLUTIONS_ROOT)
EVOLUTION_ENGINE = EvolutionEngine(EVO_PM, real_pm=PM)

# Continuous Gemma maintainer — digests traces, evolves prompt banks, files tasks
from lib.maintainer import get_maintainer
from lib import trace_digest as _trace_digest

MAINTAINER = get_maintainer(DATA_DIR, EVOLUTIONS_ROOT)
try:
    if MAINTAINER.status().get("enabled"):
        MAINTAINER.start()
        print("maintainer: started (Cerebras gemma background digests)", flush=True)
except Exception as _me:
    print(f"maintainer: failed to start: {_me}", flush=True)

# Devin free-model catalog + host process scanner (quota investigation)
_DEVIN_CATALOG: dict = {}
try:
    from lib import devin_usage as _devin_usage
    _DEVIN_CATALOG = _devin_usage.bootstrap_catalog()
    n_free = len((_DEVIN_CATALOG or {}).get("free_models") or [])
    n_drop = len((_DEVIN_CATALOG or {}).get("dropdown_models") or [])
    print(
        f"devin models: ok={_DEVIN_CATALOG.get('ok')} free={n_free} "
        f"dropdown={n_drop} err={_DEVIN_CATALOG.get('error')}",
        flush=True,
    )
except Exception as _e_devin:
    print(f"devin catalog bootstrap failed: {_e_devin}", flush=True)
    _DEVIN_CATALOG = {"ok": False, "error": str(_e_devin), "dropdown_models": [], "free_models": []}

try:
    if (PROJECTS_ROOT / "default").exists():
        PM.ensure_skill_defaults("default", PM.load_state("default"))
except Exception:
    pass

# ── app ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="Dev Studio", version="0.4.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# If STUDIO_PASSWORD env var is set, all mutating API calls require a matching
# password in the X-Studio-Password header (or Authorization: Bearer <pw>).
_STUDIO_PASSWORD = os.environ.get("STUDIO_PASSWORD", "").strip()
_AUTH_EXEMPT_PATHS = {"/", "/api/health", "/api/auth/check"}
_AUTH_EXEMPT_PREFIXES = ("/static/", "/projects/", "/research/", "/.well-known/")


@app.middleware("http")
async def _auth_middleware(request: Request, call_next):
    if not _STUDIO_PASSWORD:
        return await call_next(request)
    path = request.url.path
    if path in _AUTH_EXEMPT_PATHS or path.startswith(_AUTH_EXEMPT_PREFIXES):
        return await call_next(request)
    if path.startswith("/api/jobs/") and path.endswith("/stream"):
        token = request.query_params.get("token", "")
        if token == _STUDIO_PASSWORD:
            return await call_next(request)
        return JSONResponse({"detail": "unauthorized"}, status_code=401)
    if path.startswith("/api/"):
        token = request.headers.get("x-studio-password") or ""
        if not token:
            auth = request.headers.get("authorization", "")
            if auth.lower().startswith("bearer "):
                token = auth[7:].strip()
        if token != _STUDIO_PASSWORD:
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
    return await call_next(request)


@app.get("/api/auth/check")
def api_auth_check():
    if not _STUDIO_PASSWORD:
        return {"auth_required": False, "valid": True}
    return {"auth_required": True, "valid": False}


_jobs_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}

CELL_FIELDS = ("name", "role", "goal", "params", "environment", "tools", "enabled", "status")


def set_job(job_id: str, **kwargs) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            p = JOBS / f"{job_id}.json"
            if p.exists():
                try:
                    job = json.loads(p.read_text())
                except Exception:
                    job = None
            if job is None:
                job = {"id": job_id, "created_at": utcnow()}
            _jobs[job_id] = job
        job.update(kwargs)
        job["updated_at"] = utcnow()
        (JOBS / f"{job_id}.json").write_text(json.dumps(job, indent=2))


def get_job(job_id: str) -> dict:
    with _jobs_lock:
        if job_id in _jobs:
            return dict(_jobs[job_id])
    p = JOBS / f"{job_id}.json"
    if p.exists():
        return json.loads(p.read_text())
    raise HTTPException(404, "job not found")


def run_subprocess_job(job_id: str, cmd: list[str], cwd: Path, env: Optional[dict] = None) -> None:
    set_job(job_id, status="running", cmd=cmd, cwd=str(cwd))
    log_path = JOBS / f"{job_id}.log"
    cancelled = False
    timed_out = False
    start = time.monotonic()
    TIMEOUT_SECS = 60 * 25
    try:
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        for keyfile in (Path.home() / ".config" / "elicit" / "env", ROOT / ".env.elicit", STUDIO / ".env"):
            if keyfile.is_file():
                for line in keyfile.read_text().splitlines():
                    if "=" in line and not line.strip().startswith("#"):
                        k, v = line.split("=", 1)
                        full_env.setdefault(k.strip(), v.strip().strip("'\""))
        with open(log_path, "w") as log:
            log.write(f"$ {' '.join(cmd)}\n\n")
            log.flush()
            proc = subprocess.Popen(cmd, cwd=str(cwd), env=full_env, stdout=log,
                                    stderr=subprocess.STDOUT, text=True)
            set_job(job_id, pid=proc.pid)
            with _running_procs_lock:
                _running_procs[job_id] = proc
            while True:
                rc = proc.poll()
                if rc is not None:
                    break
                with _cancel_flags_lock:
                    cancelled = job_id in _cancel_flags
                if cancelled:
                    try: proc.terminate()
                    except Exception: pass
                    try: proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try: proc.kill()
                        except Exception: pass
                    log.write("\n\n[stopped by user]\n")
                    break
                if time.monotonic() - start > TIMEOUT_SECS:
                    timed_out = True
                    try: proc.terminate()
                    except Exception: pass
                    try: proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try: proc.kill()
                        except Exception: pass
                    log.write("\n\n[timeout]\n")
                    break
                time.sleep(0.3)
        if cancelled:
            set_job(job_id, status="failed", exit_code=-15, error="stopped by user",
                    log_path=str(log_path))
        elif timed_out:
            set_job(job_id, status="failed", error="timeout", log_path=str(log_path))
        else:
            rc = proc.returncode if proc.returncode is not None else -15
            set_job(job_id, status="completed" if rc == 0 else "failed",
                    exit_code=rc, log_path=str(log_path))
    except Exception as e:
        set_job(job_id, status="failed", error=str(e), log_path=str(log_path))
    finally:
        with _running_procs_lock:
            _running_procs.pop(job_id, None)
        with _cancel_flags_lock:
            _cancel_flags.discard(job_id)


# ── models ───────────────────────────────────────────────────────────────────


class FactoryCreate(BaseModel):
    name: str
    description: str = ""
    template: str = "blank"  # blank | hello-world
    goal: str = ""


class StateUpdate(BaseModel):
    goal: Optional[str] = None
    order: Optional[list[str]] = None
    cells: Optional[list[dict]] = None
    environment: Optional[list[str]] = None
    tools: Optional[list[str]] = None


class CellUpdate(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    goal: Optional[str] = None
    params: Optional[Any] = None
    environment: Optional[str] = None
    tools: Optional[list[str]] = None
    enabled: Optional[bool] = None
    status: Optional[str] = None
    page_file: Optional[str] = None
    slide: Optional[str] = None


class BuildRequest(BaseModel):
    runner: str = "grok"  # grok | devin
    artifacts: list[str] = Field(default_factory=lambda: ["Dockerfile", "docker-compose.yml", "README.md"])
    instruction: str = "Generate the requested artifacts from the factory context. Write all files into the current directory."
    max_turns: int = Field(default=12, ge=1, le=40)
    model: str = "glm-5.2-high"
    permission_mode: str = "dangerous"
    include_memory: bool = True
    include_skill: bool = True


class AgentInspectRequest(BaseModel):
    instruction: str
    cell_ids: Optional[list[str]] = None
    auto_apply: bool = False
    include_memory: bool = True
    include_skill: bool = True


class GrokRunRequest(BaseModel):
    prompt: str
    max_turns: int = Field(default=12, ge=1, le=40)
    always_approve: bool = True
    purpose: str = "general"
    include_memory: bool = True
    include_skill: bool = True


class DevinRunRequest(BaseModel):
    prompt: str
    model: str = "glm-5.2-high"  # glm-5.2-high | swe-1.7-max
    permission_mode: str = "dangerous"  # auto | accept-edits | smart | dangerous
    purpose: str = "general"
    include_memory: bool = True
    include_skill: bool = True


class AgyRunRequest(BaseModel):
    prompt: str
    model: str = "gemini-2.5-pro"
    mode: str = "accept-edits"  # accept-edits | plan
    purpose: str = "general"
    include_memory: bool = True
    include_skill: bool = True


class CerebrasRunRequest(BaseModel):
    prompt: str
    model: str = "gemma-4-31b"
    purpose: str = "general"
    include_memory: bool = True
    include_skill: bool = True


class PiRunRequest(BaseModel):
    prompt: str
    model: str = "gpt-oss-120b"
    purpose: str = "general"
    include_memory: bool = True
    include_skill: bool = True


class OrchestrateRequest(BaseModel):
    goal: str
    output_type: str = "auto"  # auto | factory | app | factory-factory
    name: Optional[str] = None
    budget_usd: Optional[float] = None
    providers: list[str] = Field(default_factory=list)
    mcp_servers: list[str] = Field(default_factory=list)
    deployment_target: Optional[str] = None


class EvolveRequest(BaseModel):
    goal: str = ""  # optional when continue_run_id or seed_from provides the goal
    # product = ship toward the goal prompt (HTML/app/report); factories remain available
    output_type: str = "product"  # product | app | factory | factory-factory | auto
    name: Optional[str] = None
    # Raised cap: was le=12 (caused errors for larger fleets). Soft-max 64.
    population_size: int = Field(default=4, ge=1, le=64)
    # Raised cap: was le=20 (blocked multi-hour runs). Soft-max 200.
    generations: int = Field(default=3, ge=1, le=200)
    mutation_rate: float = Field(default=0.35, ge=0.0, le=1.0)
    attrition_rate: float = Field(default=0.5, ge=0.0, le=0.9)
    innovation_rate: float = Field(default=0.25, ge=0.0, le=1.0)
    benchmark_weights: dict[str, float] = Field(default_factory=dict)
    budget_usd: Optional[float] = None
    providers: list[str] = Field(default_factory=list)
    mcp_servers: list[str] = Field(default_factory=list)
    deployment_target: Optional[str] = None
    run_tests: bool = False
    promote_best: bool = True
    llm_model: str = Field(default="")  # Cerebras worker model id; empty → server default
    build_software: bool = True  # build/improve real source files each generation
    build_depth: str = "implement"  # scaffold | implement
    # Planner expands the goal into a brief before workers run (agy/devin/codex/claude/cerebras/none)
    # Default: cheap/fast Cerebras gemma-4-31b (UI also remembers last choice in localStorage)
    planner_id: str = Field(default="cerebras:gemma-4-31b")
    # Decision maker / product director (score + cooperation brief + product HTML)
    decision_maker_id: str = Field(default="cerebras:zai-glm-4.7")
    produce_product: bool = True
    use_git: bool = True
    cooperation: bool = True
    director_fitness_blend: float = 0.45
    # Iterate from a prior run's product HTML (new run seeded from that artifact)
    seed_from: Optional[str] = None
    seed_gen: Optional[int] = None
    # Continue an existing run id instead of starting a new one
    continue_run_id: Optional[str] = None
    extra_generations: Optional[int] = Field(default=None, ge=1, le=200)
    # Research harness: web search + fetch + Cerebras synth before workers (default on)
    research_enabled: bool = True
    # Population members use a mix of models (Cerebras HT+LT + OpenRouter free)
    diverse_workers: bool = True
    include_low_throughput_workers: bool = True
    include_openrouter_workers: bool = True
    worker_models: list[str] = Field(default_factory=list)


class EvolveReportRequest(BaseModel):
    """Generate investigative PDF/EDA/zips; optional agent writes a narrative."""
    # agent_id uses planner catalog ids: cerebras:gemma-4-31b, agy:..., devin:..., codex:..., claude:..., none
    agent_id: str = Field(default="cerebras:gemma-4-31b")
    model: str = Field(default="", description="Deprecated alias for agent_id / bare cerebras model")
    include_narrative: bool = True
    make_pdf: bool = True
    make_full_zip: bool = True
    make_bundle_zip: bool = True


class EvolveLearnRequest(BaseModel):
    """Deploy a learning agent over EDA/report to distill lessons / prompt strategy / skill notes."""
    agent_id: str = Field(default="cerebras:gemma-4-31b")
    focus: str = Field(default="lessons")  # lessons | prompts | skill | all


class EvolveLearnSummaryRequest(BaseModel):
    """Deploy a summary agent over all saved learning reports for this evolution."""
    agent_id: str = Field(default="cerebras:gemma-4-31b")
    include_lessons_json: bool = True


class DeployRequest(BaseModel):
    target: str  # local-docker | github-repo | github-pages | aws-lambda | aws-ecs | cloudflare-worker
    options: dict = Field(default_factory=dict)
    runner: str = "devin"  # devin | grok | cerebras | pi
    model: Optional[str] = None


class MCPToolCallRequest(BaseModel):
    tool: str
    arguments: dict = Field(default_factory=dict)


class SkillUpdate(BaseModel):
    text: str


class ReferenceCreate(BaseModel):
    name: str
    text: str


class ReferenceUpdate(BaseModel):
    text: str


class SkillRegenerateRequest(BaseModel):
    pass


class MemorySearchRequest(BaseModel):
    query: str
    limit: int = 5


class MemoryRememberRequest(BaseModel):
    task_id: str = ""
    title: str = ""
    notes: str = ""
    project: str = "default"


class MemorySyncRequest(BaseModel):
    pid: str = "default"


class NoteCreate(BaseModel):
    selector: str
    element_html: str = ""
    element_text: str = ""
    element_tag: str = ""
    page_context: dict = {}
    note: str
    severity: str = "bug"  # bug | nit | idea | question
    images: list[str] = []


class NoteUpdate(BaseModel):
    note: Optional[str] = None
    status: Optional[str] = None  # open | resolved | wontfix
    severity: Optional[str] = None
    images: Optional[list[str]] = None


class RegenRequest(BaseModel):
    prompt: Optional[str] = None
    notes: Optional[str] = None


class TTSOpenRouterRequest(BaseModel):
    cell_id: str
    model: str
    voice: str
    text_override: Optional[str] = None
    response_format: str = "mp3"
    speed: float = 1.0
    emotion: Optional[str] = None


class EmotionRequest(BaseModel):
    text: str
    context: str = ""


class VideoBuildRequest(BaseModel):
    cell_ids: Optional[list[str]] = None
    hold_seconds: float = Field(default=1.2, ge=0.0, le=10.0)
    gap_seconds: float = Field(default=0.25, ge=0.0, le=3.0)
    use_selected_audio: bool = True
    name: str = "preview"


class HyperagentBuildRequest(BaseModel):
    instruction: str = "Rebuild the video from the current cells."
    video_backend: str = "auto"  # veo | heygen | ffmpeg | auto
    thread_id: Optional[str] = None
    new_thread: bool = False
    model: str = "zai/glm-5.2-fast"


class ElicitSearchRequest(BaseModel):
    query: str
    max_results: int = Field(default=10, ge=1, le=50)
    min_year: Optional[int] = None


class AudioSelectRequest(BaseModel):
    audio_id: Optional[str] = None


class UsefulRequest(BaseModel):
    useful: bool


class EscalateRequest(BaseModel):
    agent: str
    model: Optional[str] = None
    extra_context: Optional[str] = None


class RecommendReasonsRequest(BaseModel):
    note_id: str
    job_id: Optional[str] = None
    model: str = "gemma-4-31b"


# ── routes: health + index ───────────────────────────────────────────────────



@app.get("/")
def index():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)



@app.get("/api/health")
def health():
    return {
        "ok": True,
        "app": "Evolve Studio",
        "root": str(ROOT),
        "data_dir": str(DATA_DIR),
        "evolutions_dir": str(EVOLUTIONS_ROOT),
        "auth_required": bool(_STUDIO_PASSWORD),
        "openrouter_key": llm.has_openrouter_key(),
        "cerebras_key": llm.has_cerebras_key(),
        "cerebras_keys": llm.cerebras_key_count(),
        "cerebras_key_ids": llm.KEY_POOL.fingerprints(),
        "cerebras_throughput": llm.throughput_status(),
        "devin": str(DEVIN_BIN) if DEVIN_BIN.exists() else None,
        "agy": str(AGY_BIN) if AGY_BIN.exists() else None,
        "time": utcnow(),
    }


# ── routes: factories (projects) ─────────────────────────────────────────────


def _planner_options_with_live_devin() -> list[dict]:
    """Static planner catalog, but Devin entries come from live free-model discovery when available."""
    from lib import devin_usage as dusage
    cat = dusage.USAGE.get_catalog() or _DEVIN_CATALOG or {}
    live_devin = cat.get("dropdown_models") or []
    if not live_devin:
        # Fall back to static free pair (GLM-5.2 + SWE-1.7)
        live_devin = [
            {"id": "devin:glm-5-2", "harness": "devin", "model": "glm-5-2",
             "label": "Devin · GLM-5.2 High (free)", "is_free": True},
            {"id": "devin:swe-1-7", "harness": "devin", "model": "swe-1-7",
             "label": "Devin · SWE-1.7 Max (free)", "is_free": True},
        ]
    out = []
    for opt in PLANNER_OPTIONS:
        if str(opt.get("harness")) == "devin" or str(opt.get("id") or "").startswith("devin:"):
            continue
        out.append(opt)
    out.extend(live_devin)
    return out


@app.get("/api/evolve/options")
def api_evolve_options():
    """Models + provider choices for the Evolve form (dropdowns)."""
    from lib.planner import all_planner_options, openrouter_planner_options
    planner_opts = _planner_options_with_live_devin()
    # Append OpenRouter free models for planner + director dropdowns
    or_plan = openrouter_planner_options()
    seen_plan = {p.get("id") for p in planner_opts}
    for opt in or_plan:
        if opt.get("id") not in seen_plan:
            planner_opts.append(opt)
            seen_plan.add(opt.get("id"))
    worker_catalog = worker_pool_catalog()
    # Primary worker dropdown: Cerebras + OpenRouter free (full catalog)
    primary_models = list(EVOLUTION_LLM_MODELS)
    for entry in worker_catalog:
        if entry.get("provider") == "openrouter" or str(entry.get("id") or "").startswith("openrouter:"):
            if not any(m.get("id") == entry.get("id") for m in primary_models):
                primary_models.append(entry)
    default_pool = resolve_worker_model_pool(
        EVOLUTION_LLM_MODEL,
        diverse=True,
        include_low_throughput=True,
        include_openrouter=True,
    )
    return {
        "ok": True,
        "default_llm_model": EVOLUTION_LLM_MODEL,
        "llm_models": primary_models,  # Primary worker dropdown (CB + OR free)
        "worker_models_catalog": worker_catalog,
        "default_worker_pool": default_pool,
        "openrouter_enabled": llm_mod.has_openrouter_key(),
        "openrouter_free_only": bool(getattr(llm_mod, "OPENROUTER_FREE_ONLY", True)),
        "planner_options": planner_opts,
        "default_planner_id": "cerebras:gemma-4-31b",
        "decision_maker_options": planner_opts,
        "default_decision_maker_id": "cerebras:zai-glm-4.7",
        "providers": EVOLUTION_PROVIDER_OPTIONS,
        "population_max": 64,
        "generations_max": 200,
        "saved_by_default": True,
        "evolutions_dir": str(EVOLUTIONS_ROOT),
        "cerebras_quotas": llm_mod.CEREBRAS_MODEL_QUOTAS,
        "devin_catalog": {
            "ok": (_DEVIN_CATALOG or {}).get("ok"),
            "fetched_at": (_DEVIN_CATALOG or {}).get("fetched_at"),
            "free_models": (_DEVIN_CATALOG or {}).get("free_models") or [],
            "dropdown_models": (_DEVIN_CATALOG or {}).get("dropdown_models") or [],
            "error": (_DEVIN_CATALOG or {}).get("error"),
        },
    }



@app.get("/api/cerebras/usage")
def api_cerebras_usage(run_id: Optional[str] = None):
    """Session Cerebras token/request usage vs published quotas (local estimate)."""
    snap = llm_mod.USAGE.snapshot(run_id=run_id)
    snap["keys"] = llm_mod.KEY_POOL.status()
    return {"ok": True, **snap}



@app.get("/api/free-models/usage")
def api_free_models_usage(run_id: Optional[str] = None, hours: float = 48.0, bucket_mins: int = 15):
    """Single integrative free-tier dashboard: Cerebras + Devin + OpenRouter free."""
    from lib import devin_usage as dusage
    from lib.usage_history import DEVIN_HISTORY, CEREBRAS_HISTORY

    # Ensure KEY_POOL sees multi-key .env (comma-separated CEREBRAS_API_KEY)
    try:
        llm_mod.KEY_POOL.reload()
    except Exception:
        pass
    payload = llm_mod.free_models_unified_snapshot(run_id=run_id)
    # Attach disk histories for charts
    try:
        payload["cerebras"]["history"] = CEREBRAS_HISTORY.history(
            hours=float(hours), bucket_mins=int(bucket_mins)
        )
    except Exception as e:
        payload["cerebras"]["history"] = {"error": str(e), "series": []}
    try:
        payload["devin"]["history"] = DEVIN_HISTORY.history(
            hours=float(hours), bucket_mins=int(bucket_mins)
        )
    except Exception as e:
        payload["devin"]["history"] = {"error": str(e), "series": []}
    # Live free Devin catalog if empty
    try:
        if not (payload.get("devin") or {}).get("catalog"):
            cat = dusage.USAGE.catalog if hasattr(dusage.USAGE, "catalog") else None
            if not cat:
                global _DEVIN_CATALOG
                if not (_DEVIN_CATALOG or {}).get("ok"):
                    _DEVIN_CATALOG = dusage.discover_models()
                    dusage.USAGE.set_catalog(_DEVIN_CATALOG)
                cat = _DEVIN_CATALOG
            if cat:
                payload["devin"]["catalog"] = cat
    except Exception:
        pass
    return payload



@app.get("/api/devin/usage")
def api_devin_usage(run_id: Optional[str] = None, hours: float = 48.0, bucket_mins: int = 15):
    """Devin usage estimate: studio calls + any host `devin` processes + 3h soft caps + history."""
    from lib import devin_usage as dusage
    from lib.usage_history import DEVIN_HISTORY
    snap = dusage.USAGE.snapshot(run_id=run_id)
    # Allow client to request different history windows
    try:
        snap["history"] = DEVIN_HISTORY.history(hours=float(hours), bucket_mins=int(bucket_mins))
    except Exception as e:
        snap["history"] = snap.get("history") or {"error": str(e), "series": []}
    return {"ok": True, **snap}



@app.get("/api/devin/models")
def api_devin_models(refresh: bool = False):
    """Live Devin model catalog (free vs paid) from `devin models list`."""
    from lib import devin_usage as dusage
    global _DEVIN_CATALOG
    if refresh or not (_DEVIN_CATALOG or {}).get("ok"):
        _DEVIN_CATALOG = dusage.discover_models()
        dusage.USAGE.set_catalog(_DEVIN_CATALOG)
    return {"ok": True, **(_DEVIN_CATALOG or {})}



@app.get("/api/usage/history")
def api_usage_history(provider: str = "devin", hours: float = 48.0, bucket_mins: int = 15):
    """Historical throughput series from disk (devin | cerebras)."""
    from lib.usage_history import DEVIN_HISTORY, CEREBRAS_HISTORY
    store = DEVIN_HISTORY if provider == "devin" else CEREBRAS_HISTORY
    if provider not in ("devin", "cerebras"):
        raise HTTPException(400, "provider must be devin or cerebras")
    return {"ok": True, **store.history(hours=float(hours), bucket_mins=int(bucket_mins))}



@app.get("/api/system/stats")
def api_system_stats():
    """CPU / RAM / load for the Evolve monitor strip."""
    return {"ok": True, **system_stats.snapshot()}


def _evolve_data(evo_id: str) -> dict:
    run = EVOLUTION_ENGINE.get_run(evo_id)
    if run:
        return run._to_dict()
    data = EVOLUTION_ENGINE.load_disk_dict(evo_id, EVOLUTIONS_ROOT)
    if not data:
        raise HTTPException(404, "evolution run not found")
    return data



@app.get("/api/evolve/{evo_id}/eda")
def api_evolve_eda(evo_id: str):
    """Exploratory analysis of generation traces, software lineage, semantic drift."""
    data = _evolve_data(evo_id)
    root = EVOLUTIONS_ROOT / evo_id
    eda = evo_export.analyze_run(data, root=root)
    return {"ok": True, "eda": eda}



@app.post("/api/evolve/{evo_id}/export")
def api_evolve_export(evo_id: str, body: Optional[EvolveReportRequest] = None):
    """Generate EDA + markdown + PDF + zips (full run and/or transcript bundle).

    If include_narrative, a Cerebras model writes an investigative story of the run.
    """
    body = body or EvolveReportRequest()
    data = _evolve_data(evo_id)
    root = EVOLUTIONS_ROOT / evo_id
    eda = evo_export.analyze_run(data, root=root)
    narrative = None
    agent_id = (body.agent_id or body.model or "none").strip() or "none"
    # bare model id → cerebras
    if agent_id and ":" not in agent_id and agent_id.lower() != "none":
        agent_id = f"cerebras:{agent_id}"
    narrative_meta: dict = {"tools_enabled": False, "tool_rounds": 0}
    if body.include_narrative and agent_id.lower() != "none":
        try:
            def _call_llm(prompt: str, purpose: str = "export_narrative") -> str:
                return _run_export_agent(agent_id, prompt, run_id=evo_id, purpose=purpose)

            # Multi-round tool-using report agent (prefetched pack + dig-deeper tools)
            narrative = evo_export.run_narrative_agent(
                _call_llm,
                data,
                eda,
                root=root,
                max_tool_rounds=4,
            )
            narrative_meta["tools_enabled"] = True
        except Exception as e:
            try:
                # Fallback: single-shot rich context pack
                prompt = evo_export.narrative_prompt(data, eda, root=root)
                narrative = _run_export_agent(agent_id, prompt, run_id=evo_id, purpose="export_narrative")
                narrative_meta["fallback"] = "oneshot"
            except Exception as e2:
                narrative = (
                    f"(Narrative generation failed: {e}; fallback: {e2})\n\n"
                    "See EDA JSON and REPORT.md for structured analysis."
                )
                narrative_meta["error"] = str(e2)

    files = evo_export.generate_all_exports(
        EVOLUTIONS_ROOT,
        evo_id,
        data=data,
        narrative=narrative,
        make_pdf=body.make_pdf,
        make_full_zip=body.make_full_zip,
        make_bundle_zip=body.make_bundle_zip,
    )
    urls = {}
    for k, rel in files.items():
        # charts_dir is a directory path — skip URL mapping
        if k == "charts_dir":
            continue
        name = Path(rel).name
        # nested paths like exports/REPORT-latest.html → use basename only (file endpoint is flat)
        urls[k] = f"/api/evolve/{evo_id}/export/file/{name}"
    return {
        "ok": True,
        "evolution_id": evo_id,
        "files": files,
        "urls": urls,
        "narrative_agent": agent_id,
        "narrative_meta": narrative_meta,
        "has_narrative": bool(narrative and not str(narrative).startswith("(Narrative generation failed")),
        "eda_summary": {
            "fitness_improvement": eda.get("fitness_improvement"),
            "n_generations": len(eda.get("series") or []),
            "final_best_id": (eda.get("final_best") or {}).get("id"),
            "llm_calls": (eda.get("llm_stats") or {}).get("n_calls"),
            "charts": len(list((EVOLUTIONS_ROOT / evo_id / "exports" / "charts").glob("*.png")))
            if (EVOLUTIONS_ROOT / evo_id / "exports" / "charts").exists() else 0,
        },
    }


def _run_export_agent(agent_id: str, prompt: str, *, run_id: str, purpose: str) -> str:
    """Run narrative/learning via planner harness catalog (cerebras/agy/devin/codex/claude)."""
    from lib.planner import resolve_planner
    from lib import planner as planmod
    resolved = resolve_planner(agent_id)
    harness = resolved["harness"]
    model = resolved["model"]
    if harness == "none":
        return ""
    # Narratives / tool rounds need more room; learning stays moderate
    max_tok = 4096
    if purpose.startswith("export_narrative"):
        max_tok = 8192
    elif purpose.startswith("learn"):
        max_tok = 6144
    if harness == "cerebras":
        return llm_mod.call_cerebras_sync(
            prompt,
            model=model or "gemma-4-31b",
            max_tokens=max_tok,
            run_id=run_id,
            purpose=purpose,
            temperature=0.35 if purpose.startswith("export_narrative") else 0.3,
        )
    # Reuse planner CLI runners with the full prompt as a "goal expansion" style task
    # by calling harness runners directly
    timeout = 300
    if harness == "agy":
        return planmod._run_agy(prompt, model=model or "Gemini 3.1 Pro (High)", timeout_secs=timeout)
    if harness == "devin":
        return planmod._run_devin(prompt, model=model or "swe-1-7", timeout_secs=timeout)
    if harness == "codex":
        return planmod._run_codex(prompt, model=model or "gpt-5.6-sol", timeout_secs=timeout)
    if harness == "claude":
        return planmod._run_claude(prompt, model=model or "opus", timeout_secs=timeout)
    raise RuntimeError(f"unknown agent harness: {harness}")


def _learnings_index_path(evo_id: str) -> Path:
    return evo_export.ensure_exports_dir(EVOLUTIONS_ROOT, evo_id) / "learnings-index.json"


def _load_learnings_index(evo_id: str) -> dict:
    """Load or rebuild the per-evolution learnings index from disk artifacts."""
    path = _learnings_index_path(evo_id)
    index: dict = {"evolution_id": evo_id, "items": [], "summaries": []}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                index.update(loaded)
                index.setdefault("items", [])
                index.setdefault("summaries", [])
                index["evolution_id"] = evo_id
        except Exception:
            pass
    # Rebuild missing entries from LEARNING-*.md / SUMMARY-*.md on disk
    exp = path.parent
    known_files = {it.get("filename") for it in index.get("items") or [] if isinstance(it, dict)}
    known_files |= {it.get("filename") for it in index.get("summaries") or [] if isinstance(it, dict)}
    for p in sorted(exp.glob("LEARNING-*.md")):
        if p.name in ("LEARNING-latest.md",) or p.name in known_files:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            text = ""
        stamp = p.stem.replace("LEARNING-", "", 1)
        index["items"].append({
            "id": uuid.uuid4().hex[:12],
            "kind": "learning",
            "filename": p.name,
            "stamp": stamp,
            "agent_id": _meta_from_learning_md(text, "agent") or "unknown",
            "focus": _meta_from_learning_md(text, "focus") or "lessons",
            "duration_secs": _meta_from_learning_md(text, "duration_secs"),
            "generated_at": _meta_from_learning_md(text, "generated") or "",
            "lessons_added": 0,
            "excerpt": _learning_excerpt(text),
            "chars": len(text),
            "url": f"/api/evolve/{evo_id}/export/file/{p.name}",
        })
        known_files.add(p.name)
    for p in sorted(exp.glob("SUMMARY-*.md")):
        if p.name in ("SUMMARY-latest.md",) or p.name in known_files:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            text = ""
        stamp = p.stem.replace("SUMMARY-", "", 1)
        index["summaries"].append({
            "id": uuid.uuid4().hex[:12],
            "kind": "summary",
            "filename": p.name,
            "stamp": stamp,
            "agent_id": _meta_from_learning_md(text, "agent") or "unknown",
            "sources": int(_meta_from_learning_md(text, "sources") or 0),
            "duration_secs": _meta_from_learning_md(text, "duration_secs"),
            "generated_at": _meta_from_learning_md(text, "generated") or "",
            "excerpt": _learning_excerpt(text),
            "chars": len(text),
            "url": f"/api/evolve/{evo_id}/export/file/{p.name}",
        })
        known_files.add(p.name)
    # newest first
    index["items"] = sorted(
        index.get("items") or [],
        key=lambda x: x.get("generated_at") or x.get("stamp") or "",
        reverse=True,
    )
    index["summaries"] = sorted(
        index.get("summaries") or [],
        key=lambda x: x.get("generated_at") or x.get("stamp") or "",
        reverse=True,
    )
    try:
        path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    except Exception:
        pass
    return index


def _save_learnings_index(evo_id: str, index: dict) -> None:
    path = _learnings_index_path(evo_id)
    index = dict(index or {})
    index["evolution_id"] = evo_id
    index["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(index, indent=2), encoding="utf-8")


def _meta_from_learning_md(text: str, key: str) -> Optional[str]:
    """Parse `- key: value` lines from learning/summary markdown headers."""
    if not text:
        return None
    m = re.search(rf"^-\s*{re.escape(key)}:\s*`?([^`\n]+)`?\s*$", text, re.I | re.M)
    if not m:
        return None
    return m.group(1).strip()


def _learning_excerpt(text: str, limit: int = 280) -> str:
    body = text or ""
    if "\n---\n" in body:
        body = body.split("\n---\n", 1)[1]
    body = re.sub(r"^#+\s*", "", body, flags=re.M)
    body = re.sub(r"\s+", " ", body).strip()
    return body[:limit]


def _read_learning_bodies(evo_id: str, max_items: int = 20, max_chars_each: int = 6000) -> list[dict]:
    """Load saved LEARNING-*.md bodies (newest first) for summary agent context."""
    index = _load_learnings_index(evo_id)
    exp = EVOLUTIONS_ROOT / evo_id / "exports"
    out = []
    for it in (index.get("items") or [])[:max_items]:
        fn = it.get("filename") or ""
        if not fn or fn == "LEARNING-latest.md":
            continue
        p = exp / fn
        if not p.exists():
            continue
        try:
            raw = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        out.append({
            **{k: it.get(k) for k in ("id", "filename", "agent_id", "focus", "generated_at", "stamp")},
            "text": raw[:max_chars_each],
        })
    return out



@app.get("/api/evolve/{evo_id}/learnings")
def api_evolve_learnings(evo_id: str):
    """List saved learning reports + summaries for this evolution (dashboard)."""
    _evolve_data(evo_id)  # 404 if missing
    index = _load_learnings_index(evo_id)
    # lessons from global lessons.json for this evo
    lessons = []
    if LESSONS_FILE.exists():
        try:
            all_lessons = json.loads(LESSONS_FILE.read_text(encoding="utf-8"))
            lessons = [L for L in all_lessons if isinstance(L, dict) and L.get("evolution_id") == evo_id]
            lessons = list(reversed(lessons[-80:]))
        except Exception:
            lessons = []
    exp = EVOLUTIONS_ROOT / evo_id / "exports"
    return {
        "ok": True,
        "evolution_id": evo_id,
        "items": index.get("items") or [],
        "summaries": index.get("summaries") or [],
        "lessons": lessons,
        "counts": {
            "learnings": len(index.get("items") or []),
            "summaries": len(index.get("summaries") or []),
            "lessons": len(lessons),
        },
        "urls": {
            "learning_latest": f"/api/evolve/{evo_id}/export/file/LEARNING-latest.md" if (exp / "LEARNING-latest.md").exists() else None,
            "summary_latest": f"/api/evolve/{evo_id}/export/file/SUMMARY-latest.md" if (exp / "SUMMARY-latest.md").exists() else None,
            "index": f"/api/evolve/{evo_id}/export/file/learnings-index.json" if (exp / "learnings-index.json").exists() else None,
        },
    }



@app.post("/api/evolve/{evo_id}/learn")
def api_evolve_learn(evo_id: str, body: Optional[EvolveLearnRequest] = None):
    """Deploy a learning agent on the run's EDA/report to distill reusable lessons.

    Saves exports/LEARNING-*.md, updates learnings-index.json, and can append lessons.json.
    Does not require the client to display the text — artifacts are the source of truth.
    """
    body = body or EvolveLearnRequest()
    data = _evolve_data(evo_id)
    root = EVOLUTIONS_ROOT / evo_id
    eda = evo_export.analyze_run(data, root=root)
    focus = (body.focus or "lessons").strip().lower()
    agent_id = (body.agent_id or "cerebras:gemma-4-31b").strip()
    if agent_id and ":" not in agent_id:
        agent_id = f"cerebras:{agent_id}"

    prompt = _learning_prompt(data, eda, focus=focus)
    import time as _time
    t0 = _time.time()
    text = _run_export_agent(agent_id, prompt, run_id=evo_id, purpose=f"learn_{focus}")
    duration = round(_time.time() - t0, 2)

    exp = evo_export.ensure_exports_dir(EVOLUTIONS_ROOT, evo_id)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    learn_id = uuid.uuid4().hex[:12]
    learn_path = exp / f"LEARNING-{stamp}.md"
    generated = datetime.now(timezone.utc).isoformat()
    header = (
        f"# Learning report — `{evo_id}`\n\n"
        f"- id: `{learn_id}`\n"
        f"- agent: `{agent_id}`\n"
        f"- focus: `{focus}`\n"
        f"- duration_secs: {duration}\n"
        f"- generated: {generated}\n\n---\n\n"
    )
    body_text = text or ""
    learn_path.write_text(header + body_text, encoding="utf-8")
    (exp / "LEARNING-latest.md").write_text(header + body_text, encoding="utf-8")

    lessons_added = 0
    if focus in ("lessons", "all", "prompts", "skill"):
        lessons_added = _append_learning_lessons(evo_id, agent_id, focus, body_text)

    # If focus includes prompts, try to merge tips into prompt-bank.json
    if focus in ("prompts", "all") and body_text:
        _merge_prompt_bank_from_learning(root, body_text)

    # Persist dashboard index entry
    index = _load_learnings_index(evo_id)
    entry = {
        "id": learn_id,
        "kind": "learning",
        "filename": learn_path.name,
        "stamp": stamp,
        "agent_id": agent_id,
        "focus": focus,
        "duration_secs": duration,
        "generated_at": generated,
        "lessons_added": lessons_added,
        "excerpt": _learning_excerpt(header + body_text),
        "chars": len(header + body_text),
        "url": f"/api/evolve/{evo_id}/export/file/{learn_path.name}",
    }
    items = [it for it in (index.get("items") or []) if it.get("filename") != learn_path.name]
    items.insert(0, entry)
    index["items"] = items[:50]
    _save_learnings_index(evo_id, index)

    from lib.planner import resolve_planner
    resolved = resolve_planner(agent_id)
    harness, model = resolved["harness"], resolved["model"]
    return {
        "ok": True,
        "evolution_id": evo_id,
        "learning_id": learn_id,
        "agent_id": agent_id,
        "harness": harness,
        "model": model,
        "focus": focus,
        "duration_secs": duration,
        "lessons_added": lessons_added,
        "saved": True,
        "filename": learn_path.name,
        "learning_text": body_text[:20000],
        "entry": entry,
        "counts": {
            "learnings": len(index.get("items") or []),
            "summaries": len(index.get("summaries") or []),
        },
        "urls": {
            "learning_md": f"/api/evolve/{evo_id}/export/file/{learn_path.name}",
            "learning_latest": f"/api/evolve/{evo_id}/export/file/LEARNING-latest.md",
            "index": f"/api/evolve/{evo_id}/export/file/learnings-index.json",
        },
    }



@app.post("/api/evolve/{evo_id}/learn/summary")
def api_evolve_learn_summary(evo_id: str, body: Optional[EvolveLearnSummaryRequest] = None):
    """Deploy a summary agent that synthesizes all saved learning-agent outputs for this run.

    Reads LEARNING-*.md (and optionally lessons.json rows), writes SUMMARY-*.md,
    and records the summary in the learnings dashboard index.
    """
    body = body or EvolveLearnSummaryRequest()
    data = _evolve_data(evo_id)
    agent_id = (body.agent_id or "cerebras:gemma-4-31b").strip()
    if agent_id and ":" not in agent_id:
        agent_id = f"cerebras:{agent_id}"

    learnings = _read_learning_bodies(evo_id, max_items=20, max_chars_each=5500)
    if not learnings:
        raise HTTPException(
            400,
            "No learning reports saved yet — deploy a learning agent first, then summarize.",
        )

    lesson_rows = []
    if body.include_lessons_json and LESSONS_FILE.exists():
        try:
            all_lessons = json.loads(LESSONS_FILE.read_text(encoding="utf-8"))
            lesson_rows = [
                L.get("lesson") for L in all_lessons
                if isinstance(L, dict) and L.get("evolution_id") == evo_id and L.get("lesson")
            ][-60:]
        except Exception:
            lesson_rows = []

    goal = (data.get("config") or {}).get("goal") or data.get("goal") or ""
    blocks = []
    for i, L in enumerate(learnings, 1):
        blocks.append(
            f"### Learning report {i}/{len(learnings)} — {L.get('filename')}\n"
            f"agent={L.get('agent_id')} · focus={L.get('focus')} · at={L.get('generated_at')}\n\n"
            f"{L.get('text')}\n"
        )
    lessons_block = ""
    if lesson_rows:
        lessons_block = (
            "\n## Extracted lessons.json bullets for this evolution\n"
            + "\n".join(f"- {x}" for x in lesson_rows[:50])
            + "\n"
        )

    prompt = (
        "You are a summary agent consolidating multiple learning-agent reports from one "
        "software-evolution run into a single actionable synthesis.\n\n"
        "Goals:\n"
        "1) Deduplicate and reconcile lessons across reports (note agreements & contradictions).\n"
        "2) Produce a prioritized playbook for the NEXT evolution run (what to lock / change / avoid).\n"
        "3) Propose concrete prompt-bank addenda (create / build / evaluate) if justified.\n"
        "4) End with a short checklist (max 10 items).\n"
        "Be specific: cite gen numbers, candidate ids, roles, filenames when present.\n\n"
        f"EVOLUTION ID: {evo_id}\n"
        f"GOAL: {goal}\n"
        f"NUMBER OF LEARNING REPORTS: {len(learnings)}\n\n"
        + "\n".join(blocks)
        + lessons_block
    )

    import time as _time
    t0 = _time.time()
    text = _run_export_agent(agent_id, prompt, run_id=evo_id, purpose="learn_summary")
    duration = round(_time.time() - t0, 2)

    exp = evo_export.ensure_exports_dir(EVOLUTIONS_ROOT, evo_id)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_id = uuid.uuid4().hex[:12]
    summary_path = exp / f"SUMMARY-{stamp}.md"
    generated = datetime.now(timezone.utc).isoformat()
    source_names = [L.get("filename") for L in learnings if L.get("filename")]
    header = (
        f"# Learning summary — `{evo_id}`\n\n"
        f"- id: `{summary_id}`\n"
        f"- agent: `{agent_id}`\n"
        f"- sources: `{len(learnings)}`\n"
        f"- source_files: `{', '.join(source_names)}`\n"
        f"- duration_secs: {duration}\n"
        f"- generated: {generated}\n\n---\n\n"
    )
    body_text = text or ""
    summary_path.write_text(header + body_text, encoding="utf-8")
    (exp / "SUMMARY-latest.md").write_text(header + body_text, encoding="utf-8")

    # Also fold high-signal bullets into lessons.json under source=evolution_summary
    lessons_added = _append_learning_lessons(evo_id, agent_id, "summary", body_text)

    index = _load_learnings_index(evo_id)
    entry = {
        "id": summary_id,
        "kind": "summary",
        "filename": summary_path.name,
        "stamp": stamp,
        "agent_id": agent_id,
        "sources": len(learnings),
        "source_files": source_names,
        "duration_secs": duration,
        "generated_at": generated,
        "lessons_added": lessons_added,
        "excerpt": _learning_excerpt(header + body_text),
        "chars": len(header + body_text),
        "url": f"/api/evolve/{evo_id}/export/file/{summary_path.name}",
    }
    summaries = [s for s in (index.get("summaries") or []) if s.get("filename") != summary_path.name]
    summaries.insert(0, entry)
    index["summaries"] = summaries[:30]
    _save_learnings_index(evo_id, index)

    from lib.planner import resolve_planner
    resolved = resolve_planner(agent_id)
    harness, model = resolved["harness"], resolved["model"]
    return {
        "ok": True,
        "evolution_id": evo_id,
        "summary_id": summary_id,
        "agent_id": agent_id,
        "harness": harness,
        "model": model,
        "duration_secs": duration,
        "sources": len(learnings),
        "source_files": source_names,
        "lessons_added": lessons_added,
        "saved": True,
        "filename": summary_path.name,
        "summary_text": body_text[:20000],
        "entry": entry,
        "counts": {
            "learnings": len(index.get("items") or []),
            "summaries": len(index.get("summaries") or []),
        },
        "urls": {
            "summary_md": f"/api/evolve/{evo_id}/export/file/{summary_path.name}",
            "summary_latest": f"/api/evolve/{evo_id}/export/file/SUMMARY-latest.md",
            "index": f"/api/evolve/{evo_id}/export/file/learnings-index.json",
        },
    }


def _learning_prompt(data: dict, eda: dict, focus: str = "lessons") -> str:
    focus_instr = {
        "lessons": "Extract 5–12 durable lessons for future evolution runs (what to do / avoid).",
        "prompts": "Propose improved create/build/evaluate prompt addenda that would have helped this run. Output concrete prompt text blocks.",
        "skill": "Write skill-package notes (SKILL.md style) capturing the successful product/build patterns for this goal.",
        "all": "Cover lessons, improved prompt bank text, and skill-package notes.",
    }.get(focus, "Extract durable lessons.")
    cfg = data.get("config") or {}
    out_type = eda.get("output_type") or cfg.get("output_type") or "product"
    product_focus = ""
    if out_type in ("product", "app"):
        product_focus = (
            "PRODUCT MODE: Lessons must be about how generations advanced the USER GOAL "
            "(e.g. HTML report quality/relevance if the goal was an HTML product). "
            "Do not invent factory-architecture lessons that ignore the goal.\n"
        )
    return (
        "You are a learning agent analyzing a multi-generation evolution run.\n"
        f"{product_focus}"
        f"Focus: {focus_instr}\n\n"
        "Use ONLY the EDA/report data provided. Be specific (gen numbers, candidate ids, file names, product HTML).\n"
        "Structure your answer with clear headings. End with a short checklist for the next run.\n\n"
        f"OUTPUT TYPE: {out_type}\n"
        f"GOAL: {eda.get('goal')}\n\n"
        f"FITNESS: {json.dumps(eda.get('fitness_improvement'))}\n"
        f"CHARTER: {json.dumps(eda.get('charter') or {}, ensure_ascii=False)[:3000]}\n"
        f"PROMPT BANK: {json.dumps(eda.get('prompt_bank_summary') or {}, ensure_ascii=False)}\n"
        f"ROLE DIFFS: {json.dumps(eda.get('role_diffs') or [], ensure_ascii=False)[:4000]}\n"
        f"FILE DIFFS: {json.dumps(eda.get('file_diffs') or [], ensure_ascii=False)[:4000]}\n"
        f"SEMANTIC DIFFS: {json.dumps(eda.get('semantic_diffs') or [], ensure_ascii=False)[:5000]}\n"
        f"SERIES: {json.dumps(eda.get('series') or [], ensure_ascii=False)[:12000]}\n"
        f"FINAL BEST: {json.dumps(eda.get('final_best') or {}, ensure_ascii=False)[:3000]}\n"
        f"LLM STATS: {json.dumps(eda.get('llm_stats') or {})}\n"
    )


def _append_learning_lessons(evo_id: str, agent_id: str, focus: str, text: str) -> int:
    """Append compact lessons extracted from learning text into lessons.json."""
    lessons_file = LESSONS_FILE
    lessons = []
    if lessons_file.exists():
        try:
            lessons = json.loads(lessons_file.read_text(encoding="utf-8"))
        except Exception:
            lessons = []
    # Split on numbered/bullet lines as crude lesson units
    bullets = []
    for line in (text or "").splitlines():
        s = line.strip()
        if re.match(r"^[-*•]|\d+[.)]\s+", s) and len(s) > 20:
            bullets.append(re.sub(r"^[-*•]\s*|\d+[.)]\s+", "", s)[:500])
    if not bullets:
        bullets = [(text or "")[:400]] if text else []
    added = 0
    for b in bullets[:12]:
        lessons.append({
            "id": uuid.uuid4().hex[:12],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": "evolution_learn",
            "evolution_id": evo_id,
            "agent_id": agent_id,
            "focus": focus,
            "lesson": b,
            "lesson_type": "evolution",
        })
        added += 1
    lessons = lessons[-150:]
    lessons_file.write_text(json.dumps(lessons, indent=2), encoding="utf-8")
    return added


def _merge_prompt_bank_from_learning(root: Path, text: str) -> None:
    """Best-effort: pull fenced or labeled prompt tips into prompt-bank.json."""
    pb_path = root / "prompt-bank.json"
    bank = {}
    if pb_path.exists():
        try:
            bank = json.loads(pb_path.read_text(encoding="utf-8"))
        except Exception:
            bank = {}
    # naive extraction of lines after "build:" / "create:" / "evaluate:"
    for key, label in (
        ("create_addendum", r"create[^:\n]*:\s*(.+)"),
        ("build_addendum", r"build[^:\n]*:\s*(.+)"),
        ("evaluate_addendum", r"evaluate[^:\n]*:\s*(.+)"),
    ):
        m = re.search(label, text, re.I)
        if m:
            tip = m.group(1).strip()[:400]
            prev = bank.get(key) or ""
            if tip and tip not in prev:
                bank[key] = (prev + " " + tip).strip() if prev else tip
    hist = list(bank.get("history") or [])
    hist.append({"ts": datetime.now(timezone.utc).isoformat(), "source": "learning_agent", "excerpt": (text or "")[:300]})
    bank["history"] = hist[-40:]
    pb_path.write_text(json.dumps(bank, indent=2), encoding="utf-8")



@app.get("/api/evolve/{evo_id}/product")
def api_evolve_product_index(evo_id: str):
    """List generational product HTML artifacts (genN/product + exports/PRODUCT-latest)."""
    root = EVOLUTIONS_ROOT / evo_id
    if not root.exists():
        raise HTTPException(404, "evolution not found")
    items = []
    seen_gens = set()
    for p in sorted(root.glob("gen*/product/index.html")):
        try:
            gen = p.parent.parent.name  # genN
            seen_gens.add(gen)
            items.append({
                "generation": gen,
                "path": str(p.relative_to(root)),
                "url": f"/api/evolve/{evo_id}/product/file/{p.relative_to(root).as_posix()}",
            })
        except Exception:
            continue
    exp_root = root / "exports"
    if exp_root.exists():
        for p in sorted(exp_root.glob("gen*-product.html")):
            gen_num = p.name.split("-")[0]
            if gen_num not in seen_gens:
                seen_gens.add(gen_num)
                items.append({
                    "generation": gen_num,
                    "path": str(p.relative_to(root)),
                    "url": f"/api/evolve/{evo_id}/product/file/{p.relative_to(root).as_posix()}",
                })
    latest = root / "exports" / "PRODUCT-latest.html"
    return {
        "ok": True,
        "evolution_id": evo_id,
        "items": items,
        "latest": f"/api/evolve/{evo_id}/export/file/PRODUCT-latest.html" if (latest.exists() or items) else None,
        "count": len(items),
    }



@app.get("/api/evolve/{evo_id}/product/file/{name:path}")
def api_evolve_product_file(evo_id: str, name: str):
    """Serve a file under the evolution root limited to gen*/product/** or exports/PRODUCT*."""
    raw = (name or "").strip().lstrip("/")
    if not raw or ".." in raw.split("/"):
        raise HTTPException(400, "invalid path")
    # Allow genN/product/... or exports/PRODUCT*
    ok = False
    parts = raw.split("/")
    if len(parts) >= 3 and parts[0].startswith("gen") and parts[1] == "product":
        ok = True
    if raw.startswith("exports/PRODUCT") or raw.startswith("exports/gen") and raw.endswith("-product.html"):
        ok = True
    if not ok:
        raise HTTPException(400, "path not allowed")
    path = (EVOLUTIONS_ROOT / evo_id / raw).resolve()
    root = (EVOLUTIONS_ROOT / evo_id).resolve()
    if not str(path).startswith(str(root)):
        raise HTTPException(404, "file not found")
    if not path.exists() or not path.is_file():
        # Fallback for genN/product/index.html if exports/genN-product.html or exports/PRODUCT-latest.html exists
        if len(parts) >= 3 and parts[0].startswith("gen") and parts[1] == "product" and parts[2] in ("index.html", "artifact.html"):
            gen_str = parts[0]
            alt1 = (root / "exports" / f"{gen_str}-product.html").resolve()
            alt2 = (root / "exports" / "PRODUCT-latest.html").resolve()
            if alt1.exists() and alt1.is_file():
                path = alt1
            elif alt2.exists() and alt2.is_file():
                path = alt2
            else:
                raise HTTPException(404, "file not found")
        else:
            raise HTTPException(404, "file not found")

    media = "text/html; charset=utf-8" if path.suffix.lower() in (".html", ".htm") else "text/plain; charset=utf-8"
    if path.suffix.lower() == ".md":
        media = "text/markdown; charset=utf-8"
    elif path.suffix.lower() == ".json":
        media = "application/json"
    # Inline display for HTML (iframe / new tab) — never force download
    lower = path.name.lower()
    if lower.endswith((".html", ".htm", ".png", ".svg", ".jpg", ".jpeg", ".gif", ".webp", ".css", ".js")):
        return FileResponse(path, media_type=media, content_disposition_type="inline")
    return FileResponse(path, media_type=media, filename=path.name, content_disposition_type="inline")



@app.get("/api/evolve/{evo_id}/export/file/{name:path}")
def api_evolve_export_download(evo_id: str, name: str):
    """Serve export artifacts. HTML/images open inline; zip/pdf may still download."""
    # Allow flat files or single-level charts/<file>.png under exports/
    raw = (name or "").strip().lstrip("/")
    if not raw or ".." in raw.split("/") or raw.startswith("."):
        raise HTTPException(400, "invalid name")
    parts = raw.split("/")
    if len(parts) > 2:
        raise HTTPException(400, "invalid name")
    if len(parts) == 2 and parts[0] not in ("charts",):
        raise HTTPException(400, "invalid name")
    path = (EVOLUTIONS_ROOT / evo_id / "exports" / raw).resolve()
    exp_root = (EVOLUTIONS_ROOT / evo_id / "exports").resolve()
    root = (EVOLUTIONS_ROOT / evo_id).resolve()
    if not str(path).startswith(str(exp_root)):
        raise HTTPException(404, "export file not found — generate export first")
    if not path.exists() or not path.is_file():
        if raw == "PRODUCT-latest.html":
            gen_exports = sorted(exp_root.glob("gen*-product.html")) if exp_root.exists() else []
            gen_prods = sorted(root.glob("gen*/product/index.html")) if root.exists() else []
            if gen_exports:
                path = gen_exports[-1]
            elif gen_prods:
                path = gen_prods[-1]
            else:
                raise HTTPException(404, "export file not found — generate export first")
        else:
            raise HTTPException(404, "export file not found — generate export first")
    media = "application/octet-stream"
    lower = path.name.lower()
    if lower.endswith(".pdf"):
        media = "application/pdf"
    elif lower.endswith(".zip"):
        media = "application/zip"
    elif lower.endswith(".md"):
        media = "text/markdown; charset=utf-8"
    elif lower.endswith(".json"):
        media = "application/json"
    elif lower.endswith(".html") or lower.endswith(".htm"):
        media = "text/html; charset=utf-8"
    elif lower.endswith(".png"):
        media = "image/png"
    elif lower.endswith(".svg"):
        media = "image/svg+xml"
    # HTML/images: display in browser/iframe. Zip/pdf: attachment is fine.
    if lower.endswith((".html", ".htm", ".png", ".svg", ".jpg", ".jpeg", ".gif", ".webp", ".md", ".json")):
        return FileResponse(path, media_type=media, content_disposition_type="inline")
    return FileResponse(path, media_type=media, filename=path.name)



@app.get("/api/evolve/{evo_id}/export/full.zip")
def api_evolve_export_full_zip(evo_id: str):
    """Convenience: build (if needed) and stream full-run-latest.zip."""
    data = _evolve_data(evo_id)
    exp = evo_export.ensure_exports_dir(EVOLUTIONS_ROOT, evo_id)
    latest = exp / "full-run-latest.zip"
    if not latest.exists():
        evo_export.generate_all_exports(
            EVOLUTIONS_ROOT, evo_id, data=data, narrative=None,
            make_pdf=True, make_full_zip=True, make_bundle_zip=True,
        )
    if not latest.exists():
        raise HTTPException(500, "failed to create full zip")
    return FileResponse(latest, media_type="application/zip", filename=f"{evo_id}-full-run.zip")



@app.get("/api/evolve/{evo_id}/export/bundle.zip")
def api_evolve_export_bundle_zip(evo_id: str):
    """Convenience: transcripts + report + best sources (bundle-latest.zip)."""
    data = _evolve_data(evo_id)
    exp = evo_export.ensure_exports_dir(EVOLUTIONS_ROOT, evo_id)
    latest = exp / "bundle-latest.zip"
    if not latest.exists():
        evo_export.generate_all_exports(
            EVOLUTIONS_ROOT, evo_id, data=data, narrative=None,
            make_pdf=True, make_full_zip=False, make_bundle_zip=True,
        )
    if not latest.exists():
        raise HTTPException(500, "failed to create bundle zip")
    return FileResponse(latest, media_type="application/zip", filename=f"{evo_id}-bundle.zip")



@app.get("/api/evolve/{evo_id}/export/report.pdf")
def api_evolve_export_pdf(evo_id: str):
    """Convenience: REPORT-latest.pdf (generates EDA/PDF if missing)."""
    data = _evolve_data(evo_id)
    exp = evo_export.ensure_exports_dir(EVOLUTIONS_ROOT, evo_id)
    latest = exp / "REPORT-latest.pdf"
    if not latest.exists():
        evo_export.generate_all_exports(
            EVOLUTIONS_ROOT, evo_id, data=data, narrative=None,
            make_pdf=True, make_full_zip=False, make_bundle_zip=False,
        )
    if not latest.exists():
        raise HTTPException(500, "failed to create pdf")
    return FileResponse(latest, media_type="application/pdf", filename=f"{evo_id}-report.pdf")



@app.get("/api/evolve/seeds")
def api_evolve_seeds(limit: int = 80):
    """Catalog of prior runs/product HTML topics for the Evolve start-from dropdown."""
    seeds = EVOLUTION_ENGINE.list_product_seeds(EVOLUTIONS_ROOT, limit=limit)
    return {"ok": True, "seeds": seeds, "count": len(seeds)}



@app.get("/api/evolve/gallery")
def api_evolve_gallery(limit: int = 100):
    """Product gallery: final HTMLs + short titles (for later Cloudflare/Stripe wiring)."""
    items = EVOLUTION_ENGINE.list_product_gallery(EVOLUTIONS_ROOT, limit=limit)
    return {
        "ok": True,
        "items": items,
        "count": len(items),
        "note": "deploy.cloudflare / deploy.stripe reserved for future auto-wiring",
    }



@app.get("/api/evolve/money-ideas")
def api_evolve_money_ideas_get():
    """Instant starter money-making product ideas (no LLM). Must be before /{evo_id}."""
    from lib import deployer_lens as dlens
    ideas = [{**x, "source": "starter"} for x in dlens.STARTER_MONEY_IDEAS]
    return {"ok": True, "ideas": ideas, "source": "starter", "count": len(ideas)}



@app.post("/api/evolve/money-ideas")
def api_evolve_money_ideas(body: Optional[MoneyIdeasRequest] = None):
    """Generate deployable monetizable goals — prefer OpenRouter free, then Cerebras."""
    from lib import deployer_lens as dlens
    import uuid as _uuid
    body = body or MoneyIdeasRequest()
    starters = [{**x, "source": "starter"} for x in dlens.STARTER_MONEY_IDEAS]
    if not body.refresh:
        return {"ok": True, "ideas": starters, "source": "starter", "count": len(starters)}

    prior = []
    try:
        for s in EVOLUTION_ENGINE.list_product_seeds(EVOLUTIONS_ROOT, limit=16):
            t = (s.get("topic") or s.get("goal") or "")[:100]
            if t:
                prior.append(t)
    except Exception:
        pass

    # Prefer OpenRouter free (smarter variety); fall back to Cerebras gemma
    prefer_or = llm.has_openrouter_key()
    model = (body.model or "").strip()
    if not model:
        model = "openrouter:openrouter/free" if prefer_or else "gemma-4-31b"
    # Nudge novelty each click
    hint = (body.hint or "").strip()
    novelty = f"session={_uuid.uuid4().hex[:8]} · invent NEW niches not in prior list"
    prompt = dlens.money_ideas_prompt(
        hint=(hint + "\n" + novelty).strip(),
        prior_topics=prior,
    )
    raw = ""
    err = None
    used_model = model
    try:
        raw = llm.call_worker_sync(
            prompt, model=model, max_tokens=2800, purpose="money_ideas", temperature=0.95,
        )
        ideas = dlens.parse_money_ideas_json(raw)
    except Exception as e:
        err = str(e)
        ideas = []
        # Cross-provider fallback
        try:
            alt = "gemma-4-31b" if str(model).startswith("openrouter") else "openrouter:openrouter/free"
            if (alt.startswith("openrouter") and llm.has_openrouter_key()) or (
                not alt.startswith("openrouter") and llm.has_cerebras_key()
            ):
                used_model = alt
                raw = llm.call_worker_sync(
                    prompt, model=alt, max_tokens=2800, purpose="money_ideas", temperature=0.95,
                )
                ideas = dlens.parse_money_ideas_json(raw)
                err = None
        except Exception as e2:
            err = f"{err}; fallback: {e2}"

    if not ideas:
        return {
            "ok": True, "ideas": starters, "source": "starter", "count": len(starters),
            "error": err, "note": "LLM failed — starter ideas", "model": used_model,
        }

    seen = {i.get("title", "").lower() for i in ideas}
    for s in starters:
        if s["title"].lower() not in seen:
            ideas.append({**s, "source": "starter"})
    source = "openrouter" if str(used_model).startswith("openrouter") else "cerebras"
    for idea in ideas:
        if not idea.get("source"):
            idea["source"] = source
    ideas = ideas[:12]

    return {
        "ok": True, "ideas": ideas, "source": source, "count": len(ideas),
        "model": used_model,
        "prior_topics_used": len(prior),
    }



@app.post("/api/evolve/improve-goal")
def api_evolve_improve_goal(body: ImproveGoalRequest):
    """Refine a deployer goal into an evolution brief (planner) without starting a run."""
    from lib.planner import expand_goal
    if not (body.goal or "").strip():
        raise HTTPException(400, "goal required")
    result = expand_goal(
        body.goal.strip(),
        planner_id=body.planner_id or "cerebras:gemma-4-31b",
        output_type=body.output_type or "product",
        build_software=bool(body.build_software),
    )
    if not result.get("ok") and result.get("error"):
        return {
            "ok": False, "error": result.get("error"),
            "brief": result.get("brief") or body.goal,
            "planner_id": body.planner_id, "goal": body.goal,
        }
    return {
        "ok": True, "brief": result.get("brief") or body.goal,
        "planner_id": result.get("planner_id") or body.planner_id,
        "goal": body.goal, "duration_secs": result.get("duration_secs"),
    }



@app.post("/api/evolve")
def api_evolve(body: EvolveRequest):
    """Start an evolutionary design run for a factory/app/factory-factory.

    The engine generates a population of candidate genomes, scores them on
    benchmarks/KPIs, applies attrition, mutates/breeds survivors, and may
    inject innovation cells. The best candidate can be promoted to a real project.
    Every run is saved under data-*/evolutions/<id>/ by default.

    Optional:
      - continue_run_id: continue that run (+ extra_generations)
      - seed_from / seed_gen: new run iterating from that run's product HTML
    """
    # Path A: continue an existing run (same id, more gens)
    if body.continue_run_id:
        result = EVOLUTION_ENGINE.continue_generations(
            body.continue_run_id.strip(),
            EVOLUTIONS_ROOT,
            extra_generations=int(body.extra_generations or body.generations or 2),
            goal_addendum=(body.goal or "").strip() if body.goal and body.goal.strip() != "continue" else "",
        )
        if not result.get("ok"):
            err = result.get("error") or "could not continue"
            code = 404 if "not found" in err.lower() else 409
            raise HTTPException(code, err)
        return result

    llm_model = (body.llm_model or "").strip() or EVOLUTION_LLM_MODEL
    allowed = {m["id"] for m in EVOLUTION_LLM_MODELS}
    if llm_model not in allowed:
        # Allow unknown models (forward-compat) but prefer known list
        pass
    goal = (body.goal or "").strip()
    if not goal and body.seed_from:
        # Inherit goal from parent seed when user picks a topic without rewriting
        parent = EVOLUTION_ENGINE.load_disk_dict(body.seed_from, EVOLUTIONS_ROOT)
        if parent:
            goal = ((parent.get("config") or {}).get("goal") or "").strip()
    if not goal:
        raise HTTPException(400, "goal required (or pick a seed topic)")
    cfg = EvolutionConfig(
        goal=goal,
        output_type=body.output_type,
        name=body.name,
        population_size=body.population_size,
        generations=body.generations,
        mutation_rate=body.mutation_rate,
        attrition_rate=body.attrition_rate,
        innovation_rate=body.innovation_rate,
        benchmark_weights=body.benchmark_weights,
        budget_usd=body.budget_usd,
        providers=body.providers,
        mcp_servers=body.mcp_servers or [s["name"] for s in mcp.list_servers()],
        deployment_target=body.deployment_target,
        run_tests=body.run_tests,
        promote_best=body.promote_best,
        llm_model=llm_model,
        build_software=body.build_software,
        build_depth=(body.build_depth or "implement").strip() or "implement",
        planner_id=(body.planner_id or "cerebras:gemma-4-31b").strip() or "cerebras:gemma-4-31b",
        decision_maker_id=(body.decision_maker_id or "cerebras:zai-glm-4.7").strip() or "cerebras:zai-glm-4.7",
        produce_product=bool(body.produce_product),
        use_git=bool(body.use_git),
        cooperation=bool(body.cooperation),
        director_fitness_blend=float(body.director_fitness_blend if body.director_fitness_blend is not None else 0.45),
        research_enabled=bool(getattr(body, "research_enabled", True)),
        diverse_workers=bool(getattr(body, "diverse_workers", True)),
        include_low_throughput_workers=bool(getattr(body, "include_low_throughput_workers", True)),
        include_openrouter_workers=bool(getattr(body, "include_openrouter_workers", True)),
        worker_models=list(getattr(body, "worker_models", None) or []),
    )
    seed_from = (body.seed_from or "").strip() or None
    seed_gen = body.seed_gen
    run = EVOLUTION_ENGINE.start(
        cfg, EVOLUTIONS_ROOT, seed_from=seed_from, seed_gen=seed_gen,
    )
    pool = resolve_worker_model_pool(
        cfg.llm_model,
        explicit=list(cfg.worker_models or []) or None,
        diverse=bool(cfg.diverse_workers),
        include_low_throughput=bool(cfg.include_low_throughput_workers),
        include_openrouter=bool(cfg.include_openrouter_workers),
    )
    return {
        "ok": True,
        "evolution_id": run.id,
        "status": run.status,
        "llm_model": run.llm_model,
        "worker_pool": pool,
        "diverse_workers": cfg.diverse_workers,
        "planner_id": cfg.planner_id,
        "build_software": cfg.build_software,
        "build_depth": cfg.build_depth,
        "seed_from": seed_from,
        "seed_gen": seed_gen,
        "saved": True,
        "path": str(EVOLUTIONS_ROOT / run.id),
    }



@app.get("/api/evolve")
def api_evolve_list(full: bool = False):
    """History of all saved evolutions (disk + in-memory). Default: compact summaries for sidebar."""
    return {
        "ok": True,
        "evolutions": EVOLUTION_ENGINE.list_runs(EVOLUTIONS_ROOT, full=full),
        "count": len(list(EVOLUTIONS_ROOT.glob("*/evolution.json"))),
        "dir": str(EVOLUTIONS_ROOT),
        "note": "Runs and product HTML under data/evolutions/<id>/ survive refresh and server restart.",
    }


@app.get("/api/evolve/runs")
def api_evolve_runs_alias(full: bool = False):
    """Alias for /api/evolve — frontend hash #/evolve/runs maps here; must not hit {evo_id}."""
    return api_evolve_list(full=full)


@app.get("/api/evolve/{evo_id}")
def api_evolve_status(evo_id: str):
    # Reserved path segments that are list/meta endpoints (never treat as run ids)
    if evo_id in ("runs", "options", "gallery", "seeds", "money-ideas", "list"):
        if evo_id == "runs":
            return api_evolve_list()
        raise HTTPException(404, f"use /api/evolve/{evo_id} static route")
    run = EVOLUTION_ENGINE.get_run(evo_id)
    if not run:
        # Fall back to on-disk evolution.json (survives server restarts)
        data = EVOLUTION_ENGINE.load_disk_dict(evo_id, EVOLUTIONS_ROOT)
        if data:
            data["persisted"] = True
            data["disk_path"] = str(EVOLUTIONS_ROOT / evo_id)
            # Surface product HTML even if status was interrupted mid-run
            latest = EVOLUTIONS_ROOT / evo_id / "exports" / "PRODUCT-latest.html"
            if latest.is_file():
                data["has_product_html"] = True
                data["product_url"] = f"/api/evolve/{evo_id}/export/file/PRODUCT-latest.html"
            return data
        raise HTTPException(404, "evolution run not found")
    return run._to_dict()



@app.get("/api/evolve/{evo_id}/digest")
def api_evolve_digest(evo_id: str, refresh: bool = False, gemma: bool = False):
    """Sense-making digest for an evolution (structural always; Gemma optional).

    Structural is free/instant. With gemma=1, runs a cheap Cerebras narrative +
    prompt-bank / studio-task proposals (quota-gated).
    """
    root = EVOLUTIONS_ROOT / evo_id
    run = EVOLUTION_ENGINE.get_run(evo_id)
    if run:
        data = run._to_dict()
    else:
        data = EVOLUTION_ENGINE.load_disk_dict(evo_id, EVOLUTIONS_ROOT)
    if not data:
        raise HTTPException(404, "evolution run not found")

    if not refresh and not gemma:
        cached = _trace_digest.load_latest_digest(root)
        if cached:
            # Refresh structural counts if stale vs current call count
            cur_n = len(data.get("llm_calls") or [])
            dig_n = ((cached.get("counts") or cached.get("structural", {}).get("counts") or {}).get("llm_calls"))
            if dig_n == cur_n:
                return {"ok": True, "cached": True, "digest": cached}

    if gemma:
        try:
            digest = _trace_digest.digest_for_run_root(root, data, with_gemma=True)
        except Exception as e:
            structural = _trace_digest.build_structural_digest(data)
            _trace_digest.save_digest(root, structural)
            return {"ok": False, "error": str(e), "digest": structural}
    else:
        digest = _trace_digest.digest_for_run_root(root, data, with_gemma=False)
    return {"ok": True, "cached": False, "digest": digest}


class EvolveDigestRequest(BaseModel):
    gemma: bool = True


class MaintainerConfigRequest(BaseModel):
    enabled: Optional[bool] = None
    interval_secs: Optional[float] = None
    model: Optional[str] = None
    auto_apply_prompt_patches: Optional[bool] = None
    prefer_running_runs: Optional[bool] = None
    absorb_learnings: Optional[bool] = None


class MaintainerTickRequest(BaseModel):
    evolution_id: Optional[str] = None
    evo_id: Optional[str] = None
    gemma: bool = True


class MaintainerMindPatch(BaseModel):
    mission: Optional[str] = None
    focus: Optional[str] = None
    notes: Optional[str] = None
    goals: Optional[list] = None
    plans: Optional[list] = None


class MaintainerGoalCreate(BaseModel):
    title: str
    detail: str = ""
    priority: str = "med"


class MaintainerGoalUpdate(BaseModel):
    title: Optional[str] = None
    detail: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[str] = None


class MaintainerPlanCreate(BaseModel):
    title: str
    steps: Optional[list] = None
    focus: str = ""


class MaintainerPlanUpdate(BaseModel):
    title: Optional[str] = None
    steps: Optional[list] = None
    focus: Optional[str] = None
    status: Optional[str] = None


class MaintainerMemoryCreate(BaseModel):
    content: str
    kind: str = "note"
    tags: Optional[list] = None
    source: str = "user"
    evolution_id: Optional[str] = None


class MaintainerMemoryUpdate(BaseModel):
    content: Optional[str] = None
    kind: Optional[str] = None
    tags: Optional[list] = None
    source: Optional[str] = None


class MaintainerTaskStatus(BaseModel):
    status: str



@app.post("/api/evolve/{evo_id}/digest")
def api_evolve_digest_post(evo_id: str, body: Optional[EvolveDigestRequest] = None):
    """Force Gemma trace analysis for one run (also used by maintainer)."""
    gemma = True if body is None else bool(body.gemma)
    return api_evolve_digest(evo_id, refresh=True, gemma=gemma)



@app.get("/api/maintainer")
def api_maintainer_status():
    """Compact status + recent tasks (backward compatible)."""
    return {"ok": True, **MAINTAINER.status(), "tasks": MAINTAINER.list_tasks(40)}



@app.get("/api/maintainer/monitor")
def api_maintainer_monitor():
    """Full mind dump for the monitor UI: goals, plans, memories, learnings, log, tasks."""
    return MAINTAINER.snapshot()



@app.post("/api/maintainer")
def api_maintainer_configure(body: Optional[MaintainerConfigRequest] = None):
    body = body or MaintainerConfigRequest()
    return {"ok": True, **MAINTAINER.configure(
        enabled=body.enabled,
        interval_secs=body.interval_secs,
        model=body.model,
        auto_apply_prompt_patches=body.auto_apply_prompt_patches,
        prefer_running_runs=body.prefer_running_runs,
        absorb_learnings=body.absorb_learnings,
    )}



@app.post("/api/maintainer/tick")
def api_maintainer_tick(body: Optional[MaintainerTickRequest] = None):
    """Run one maintainer cycle now (optionally force a specific evolution)."""
    body = body or MaintainerTickRequest()
    result = MAINTAINER.tick(
        force_evo_id=body.evolution_id or body.evo_id,
        with_gemma=body.gemma,
    )
    return {"ok": True, "result": result, "status": MAINTAINER.status(), "snapshot": MAINTAINER.snapshot()}



@app.get("/api/maintainer/mind")
def api_maintainer_mind_get():
    return {"ok": True, "mind": MAINTAINER.get_mind()}



@app.put("/api/maintainer/mind")

@app.post("/api/maintainer/mind")
def api_maintainer_mind_put(body: Optional[MaintainerMindPatch] = None):
    body = body or MaintainerMindPatch()
    mind = MAINTAINER.update_mind(body.model_dump(exclude_none=True))
    return {"ok": True, "mind": mind}



@app.post("/api/maintainer/goals")
def api_maintainer_goal_add(body: MaintainerGoalCreate):
    g = MAINTAINER.add_goal(body.title, body.detail, body.priority)
    return {"ok": True, "goal": g, "mind": MAINTAINER.get_mind()}



@app.patch("/api/maintainer/goals/{goal_id}")
def api_maintainer_goal_patch(goal_id: str, body: MaintainerGoalUpdate):
    g = MAINTAINER.update_goal(goal_id, **body.model_dump(exclude_none=True))
    if not g:
        raise HTTPException(404, "goal not found")
    return {"ok": True, "goal": g}



@app.delete("/api/maintainer/goals/{goal_id}")
def api_maintainer_goal_del(goal_id: str):
    if not MAINTAINER.remove_goal(goal_id):
        raise HTTPException(404, "goal not found")
    return {"ok": True, "mind": MAINTAINER.get_mind()}



@app.post("/api/maintainer/plans")
def api_maintainer_plan_add(body: MaintainerPlanCreate):
    p = MAINTAINER.add_plan(body.title, body.steps, body.focus)
    return {"ok": True, "plan": p, "mind": MAINTAINER.get_mind()}



@app.patch("/api/maintainer/plans/{plan_id}")
def api_maintainer_plan_patch(plan_id: str, body: MaintainerPlanUpdate):
    p = MAINTAINER.update_plan(plan_id, **body.model_dump(exclude_none=True))
    if not p:
        raise HTTPException(404, "plan not found")
    return {"ok": True, "plan": p}



@app.delete("/api/maintainer/plans/{plan_id}")
def api_maintainer_plan_del(plan_id: str):
    if not MAINTAINER.remove_plan(plan_id):
        raise HTTPException(404, "plan not found")
    return {"ok": True, "mind": MAINTAINER.get_mind()}



@app.get("/api/maintainer/memories")
def api_maintainer_memories(kind: Optional[str] = None, limit: int = 200):
    return {"ok": True, "memories": MAINTAINER.list_memories(kind=kind, limit=limit)}



@app.post("/api/maintainer/memories")
def api_maintainer_memory_add(body: MaintainerMemoryCreate):
    try:
        m = MAINTAINER.add_memory(
            body.content,
            kind=body.kind,
            tags=body.tags,
            source=body.source,
            evolution_id=body.evolution_id,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "memory": m}



@app.patch("/api/maintainer/memories/{memory_id}")
def api_maintainer_memory_patch(memory_id: str, body: MaintainerMemoryUpdate):
    m = MAINTAINER.update_memory(memory_id, **body.model_dump(exclude_none=True))
    if not m:
        raise HTTPException(404, "memory not found")
    return {"ok": True, "memory": m}



@app.delete("/api/maintainer/memories/{memory_id}")
def api_maintainer_memory_del(memory_id: str):
    if not MAINTAINER.remove_memory(memory_id):
        raise HTTPException(404, "memory not found")
    return {"ok": True}



@app.post("/api/maintainer/memories/clear")
def api_maintainer_memories_clear(kind: Optional[str] = None, source: Optional[str] = None):
    n = MAINTAINER.clear_memories(kind=kind, source=source)
    return {"ok": True, "removed": n}



@app.get("/api/maintainer/learnings")
def api_maintainer_learnings(limit: int = 40):
    return {"ok": True, "learnings": MAINTAINER.list_learnings(limit)}



@app.get("/api/maintainer/log")
def api_maintainer_log(limit: int = 80):
    return {"ok": True, "log": MAINTAINER.list_log(limit)}



@app.patch("/api/maintainer/tasks/{task_id}")
def api_maintainer_task_status(task_id: str, body: MaintainerTaskStatus):
    if not MAINTAINER.update_task_status(task_id, body.status):
        raise HTTPException(404, "task not found")
    return {"ok": True, "tasks": MAINTAINER.list_tasks(60)}



@app.post("/api/evolve/{evo_id}/stop")
def api_evolve_stop(evo_id: str):
    """Cooperatively stop a live evolution. Progress is always saved on disk.

    The worker finishes the current LLM call if one is in flight, then exits at
    the next checkpoint with status=stopped (not failed). Candidates, llm_calls,
    charter, and generation snapshots remain available for inspection / promote.
    """
    result = EVOLUTION_ENGINE.stop(evo_id, EVOLUTIONS_ROOT)
    if not result.get("ok"):
        raise HTTPException(404, result.get("error") or "evolution run not found")
    return result



@app.post("/api/evolve/{evo_id}/resume")
def api_evolve_resume(evo_id: str):
    """Resume a stopped/failed/orphaned evolution from the next incomplete generation.

    Loads candidates, charter, prompt bank, and completed gen summaries from disk.
    Skips gen0 create when gen0 already exists. Does not resume a live running worker.
    """
    result = EVOLUTION_ENGINE.resume(evo_id, EVOLUTIONS_ROOT)
    if not result.get("ok"):
        # 409 for already completed / busy; 404 for missing
        err = result.get("error") or "could not resume"
        code = 404 if "not found" in err.lower() else 409
        raise HTTPException(code, err)
    return result


class EvolveContinueRequest(BaseModel):
    extra_generations: int = Field(default=2, ge=1, le=200)
    goal_addendum: str = ""


class EvolveCloneRequest(BaseModel):
    flavor: str
    generations: int = Field(default=3, ge=1, le=200)
    population_size: Optional[int] = Field(default=None, ge=1, le=64)
    name: Optional[str] = None


class ThroughputRequest(BaseModel):
    enabled: bool = False


class ImproveGoalRequest(BaseModel):
    goal: str
    planner_id: str = "cerebras:gemma-4-31b"
    output_type: str = "product"
    build_software: bool = True


class MoneyIdeasRequest(BaseModel):
    hint: str = ""
    model: str = "gemma-4-31b"
    refresh: bool = True  # False → starters only (no LLM)



@app.get("/api/cerebras/throughput")
def api_cerebras_throughput_get():
    """High-throughput multi-key mode (off by default)."""
    return {"ok": True, **llm.throughput_status()}



@app.post("/api/cerebras/throughput")
def api_cerebras_throughput_set(body: ThroughputRequest):
    st = llm.set_high_throughput(bool(body.enabled))
    return {"ok": True, **st}



@app.post("/api/evolve/{evo_id}/continue")
def api_evolve_continue(evo_id: str, body: Optional[EvolveContinueRequest] = None):
    """Push more generations on the same run (works after completed too)."""
    body = body or EvolveContinueRequest()
    result = EVOLUTION_ENGINE.continue_generations(
        evo_id,
        EVOLUTIONS_ROOT,
        extra_generations=body.extra_generations,
        goal_addendum=body.goal_addendum or "",
    )
    if not result.get("ok"):
        err = result.get("error") or "could not continue"
        code = 404 if "not found" in err.lower() else 409
        raise HTTPException(code, err)
    return result



@app.post("/api/evolve/{evo_id}/clone")
def api_evolve_clone(evo_id: str, body: EvolveCloneRequest):
    """Clone a flavor branch: same product line, different monetizable variant."""
    result = EVOLUTION_ENGINE.clone_flavor(
        evo_id,
        EVOLUTIONS_ROOT,
        flavor=body.flavor,
        generations=body.generations,
        population_size=body.population_size,
        name=body.name,
    )
    if not result.get("ok"):
        err = result.get("error") or "could not clone"
        code = 404 if "not found" in err.lower() else 400
        raise HTTPException(code, err)
    return result



@app.get("/api/evolve/{evo_id}/answers")
def api_evolve_answers(evo_id: str):
    """Full model answers document for an evolution run (all prompts + responses)."""
    run = EVOLUTION_ENGINE.get_run(evo_id)
    root = EVOLUTIONS_ROOT / evo_id
    if run:
        try:
            EVOLUTION_ENGINE._write_llm_transcript(run)
        except Exception:
            pass
        data = run._to_dict()
        md_path = root / "model-answers.md"
        return {
            "ok": True,
            "evolution_id": evo_id,
            "goal": data.get("config", {}).get("goal"),
            "llm_model": data.get("llm_model"),
            "calls": data.get("llm_calls") or [],
            "markdown": md_path.read_text(encoding="utf-8") if md_path.exists() else None,
            "best": data.get("best"),
            "promoted_project_id": data.get("promoted_project_id"),
        }
    # disk fallback
    calls_path = root / "llm-calls.json"
    md_path = root / "model-answers.md"
    evo_path = root / "evolution.json"
    if not root.exists():
        raise HTTPException(404, "evolution run not found")
    calls = []
    if calls_path.exists():
        try:
            calls = json.loads(calls_path.read_text(encoding="utf-8"))
        except Exception:
            calls = []
    evo = {}
    if evo_path.exists():
        try:
            evo = json.loads(evo_path.read_text(encoding="utf-8"))
            if not calls:
                calls = evo.get("llm_calls") or []
        except Exception:
            pass
    return {
        "ok": True,
        "evolution_id": evo_id,
        "goal": (evo.get("config") or {}).get("goal"),
        "llm_model": evo.get("llm_model"),
        "calls": calls,
        "markdown": md_path.read_text(encoding="utf-8") if md_path.exists() else None,
        "best": evo.get("best"),
        "promoted_project_id": evo.get("promoted_project_id"),
    }


def _hydrate_candidate_cells(cand: dict) -> dict:
    """Load cells + build artifacts from candidate path when the summary snapshot omitted them."""
    if not cand:
        return cand or {}
    path = cand.get("path")
    if not path:
        return cand
    root = Path(path)
    state_path = root / "state.json"
    out = dict(cand)
    if state_path.exists() and not out.get("cells"):
        try:
            genome = json.loads(state_path.read_text(encoding="utf-8"))
            cells = [
                {
                    "id": c.get("id"),
                    "role": c.get("role") or "cell",
                    "name": c.get("name") or c.get("role") or c.get("id"),
                    "goal": c.get("goal") or "",
                    "tools": c.get("tools") or [],
                    "environment": c.get("environment"),
                    "status": c.get("status") or "ready",
                    "enabled": c.get("enabled", True),
                }
                for c in (genome.get("cells") or [])
                if isinstance(c, dict)
            ]
            out.update({
                "cells": cells,
                "cell_count": len(cells),
                "cell_roles": [c.get("role", "") for c in cells],
                "description": genome.get("description") or out.get("description"),
                "template": genome.get("template") or out.get("template"),
                "order": genome.get("order") or [c.get("id") for c in cells],
            })
            if genome.get("build") and not out.get("build"):
                out["build"] = genome["build"]
            if genome.get("artifacts") and not out.get("artifacts"):
                out["artifacts"] = genome["artifacts"]
            if genome.get("innovation_thesis"):
                out["innovation_thesis"] = genome["innovation_thesis"]
        except Exception:
            pass
    # build-manifest.json is authoritative for software files
    man = root / "build-manifest.json"
    if man.exists() and not out.get("build"):
        try:
            out["build"] = json.loads(man.read_text(encoding="utf-8"))
            out["artifacts"] = out["build"].get("files") or out.get("artifacts") or []
        except Exception:
            pass
    elif man.exists() and out.get("build") and not out.get("artifacts"):
        try:
            m = json.loads(man.read_text(encoding="utf-8"))
            out["artifacts"] = m.get("files") or []
        except Exception:
            pass
    return out



@app.get("/api/evolve/{evo_id}/compare")
def api_evolve_compare(evo_id: str, gen_a: Optional[int] = None, gen_b: Optional[int] = None):
    """Side-by-side generation comparison: architecture (cells) + prompts/responses.

    Returns every generation with hydrated best-candidate genome and LLM calls
    scoped to that generation. Optional gen_a/gen_b highlight a pair.
    """
    run = EVOLUTION_ENGINE.get_run(evo_id)
    data = run._to_dict() if run else EVOLUTION_ENGINE.load_disk_dict(evo_id, EVOLUTIONS_ROOT)
    if not data:
        raise HTTPException(404, "evolution run not found")

    all_calls = data.get("llm_calls") or []
    generations_out = []
    for g in data.get("generations") or []:
        cands = [_hydrate_candidate_cells(dict(c)) for c in (g.get("candidates") or [])]
        # sort by fitness desc if not already
        cands_sorted = sorted(cands, key=lambda c: c.get("fitness") or 0, reverse=True)
        best = cands_sorted[0] if cands_sorted else None
        gen_num = g.get("generation")
        # prompts for this generation: evaluate calls at this gen + create for gen0 candidates evaluated here
        prompts = []
        for call in all_calls:
            cg = call.get("generation")
            # include calls tagged with this generation
            if cg == gen_num:
                prompts.append(call)
                continue
            # also attach create_initial that produced candidates still in this gen's population
            if call.get("purpose") == "create_initial" and best and call.get("candidate_id") == best.get("id"):
                if not any(p.get("id") == call.get("id") for p in prompts):
                    prompts.append(call)
        generations_out.append({
            "generation": gen_num,
            "best_fitness": g.get("best_fitness"),
            "avg_fitness": g.get("avg_fitness"),
            "survivors": g.get("survivors"),
            "population": g.get("population"),
            "brilliant": g.get("brilliant"),
            "brilliant_count": g.get("brilliant_count") or (len(g.get("brilliant") or [])),
            "survivors_ids": g.get("survivors_ids") or [],
            "eliminated_ids": g.get("eliminated_ids") or [],
            "candidates": cands_sorted,
            "best": best,
            "architecture": {
                "cell_count": (best or {}).get("cell_count") or len((best or {}).get("cells") or []),
                "cell_roles": (best or {}).get("cell_roles") or [c.get("role") for c in ((best or {}).get("cells") or [])],
                "cells": (best or {}).get("cells") or [],
                "description": (best or {}).get("description"),
                "template": (best or {}).get("template"),
            },
            "prompts": prompts,
            "prompt_count": len(prompts),
        })

    # fitness series for charting
    series = [
        {
            "generation": g["generation"],
            "best_fitness": g.get("best_fitness"),
            "avg_fitness": g.get("avg_fitness"),
            "roles": (g.get("architecture") or {}).get("cell_roles") or [],
        }
        for g in generations_out
    ]

    # architecture diff between consecutive gens
    diffs = []
    for i in range(1, len(generations_out)):
        a, b = generations_out[i - 1], generations_out[i]
        roles_a = set((a.get("architecture") or {}).get("cell_roles") or [])
        roles_b = set((b.get("architecture") or {}).get("cell_roles") or [])
        ids_a = {c.get("id") for c in (a.get("candidates") or [])}
        ids_b = {c.get("id") for c in (b.get("candidates") or [])}
        diffs.append({
            "from_gen": a.get("generation"),
            "to_gen": b.get("generation"),
            "fitness_delta": round((b.get("best_fitness") or 0) - (a.get("best_fitness") or 0), 4),
            "roles_added": sorted(roles_b - roles_a),
            "roles_removed": sorted(roles_a - roles_b),
            "candidates_new": sorted(ids_b - ids_a),
            "candidates_dropped": sorted(ids_a - ids_b),
            "best_changed": (a.get("best") or {}).get("id") != (b.get("best") or {}).get("id"),
        })

    pair = None
    if gen_a is not None and gen_b is not None:
        ga = next((g for g in generations_out if g.get("generation") == gen_a), None)
        gb = next((g for g in generations_out if g.get("generation") == gen_b), None)
        if ga and gb:
            pair = {"gen_a": ga, "gen_b": gb}

    return {
        "ok": True,
        "evolution_id": evo_id,
        "goal": (data.get("config") or {}).get("goal"),
        "llm_model": data.get("llm_model") or (data.get("config") or {}).get("llm_model"),
        "generations": generations_out,
        "series": series,
        "diffs": diffs,
        "pair": pair,
        "all_prompts": all_calls,
    }



@app.get("/api/evolve/{evo_id}/genome")
def api_evolve_genome(evo_id: str, candidate_id: Optional[str] = None):
    """Return genome cells for visualization (best candidate by default)."""
    run = EVOLUTION_ENGINE.get_run(evo_id)
    data = None
    if run:
        data = run._to_dict()
    else:
        disk = EVOLUTIONS_ROOT / evo_id / "evolution.json"
        if disk.exists():
            try:
                data = json.loads(disk.read_text(encoding="utf-8"))
            except Exception:
                data = None
    if not data:
        raise HTTPException(404, "evolution run not found")

    cand = None
    if candidate_id:
        for c in data.get("candidates") or []:
            if c.get("id") == candidate_id:
                cand = c
                break
        if not cand:
            for g in data.get("generations") or []:
                for c in g.get("candidates") or []:
                    if c.get("id") == candidate_id:
                        cand = c
                        break
                if cand:
                    break
    if not cand:
        cand = data.get("best") or {}

    cells = cand.get("cells") or []
    # Disk fallback: load state.json from candidate path
    if not cells and cand.get("path"):
        state_path = Path(cand["path"]) / "state.json"
        if state_path.exists():
            try:
                genome = json.loads(state_path.read_text(encoding="utf-8"))
                cells = genome.get("cells") or []
                cand = {
                    **cand,
                    "cells": [
                        {
                            "id": c.get("id"),
                            "role": c.get("role") or "cell",
                            "name": c.get("name") or c.get("role") or c.get("id"),
                            "goal": c.get("goal") or "",
                            "tools": c.get("tools") or [],
                            "environment": c.get("environment"),
                            "status": c.get("status") or "ready",
                            "enabled": c.get("enabled", True),
                        }
                        for c in cells if isinstance(c, dict)
                    ],
                    "description": genome.get("description") or cand.get("description"),
                    "template": genome.get("template") or cand.get("template"),
                    "order": genome.get("order") or [c.get("id") for c in cells if isinstance(c, dict)],
                }
                cells = cand["cells"]
            except Exception:
                pass

    return {
        "ok": True,
        "evolution_id": evo_id,
        "candidate_id": cand.get("id"),
        "goal": (data.get("config") or {}).get("goal"),
        "fitness": cand.get("fitness"),
        "brilliant": cand.get("brilliant"),
        "description": cand.get("description"),
        "template": cand.get("template"),
        "order": cand.get("order"),
        "cells": cells,
        "cell_count": len(cells),
        "rationale": cand.get("rationale"),
        "scores": cand.get("scores"),
    }



@app.get("/api/evolve/{evo_id}/answers/{call_id}")
def api_evolve_answer_one(evo_id: str, call_id: str):
    """Single LLM call (full prompt + response) from an evolution run."""
    payload = api_evolve_answers(evo_id)
    for c in payload.get("calls") or []:
        if str(c.get("id")) == call_id or str(c.get("ts")) == call_id:
            return {"ok": True, "call": c, "evolution_id": evo_id}
    # index fallback: 0-based numeric id
    try:
        idx = int(call_id)
        calls = payload.get("calls") or []
        if 0 <= idx < len(calls):
            return {"ok": True, "call": calls[idx], "evolution_id": evo_id}
    except Exception:
        pass
    raise HTTPException(404, "llm call not found")



@app.post("/api/evolve/{evo_id}/promote")
def api_evolve_promote(evo_id: str):
    """Promote the best candidate of a completed evolution to a real project."""
    run = EVOLUTION_ENGINE.get_run(evo_id)
    if not run:
        raise HTTPException(404, "evolution run not found")
    if run.status != "completed":
        raise HTTPException(400, f"evolution is {run.status}; wait for completion")
    meta = EVOLUTION_ENGINE.best_to_project(run, PM)
    if not meta:
        raise HTTPException(500, "no best candidate to promote")
    return {
        "ok": True,
        "project_id": meta["id"],
        "project": meta,
        "evolution_id": evo_id,
        "answers_available": True,
    }



@app.get("/api/mcp/servers")
def api_mcp_servers():
    """List configured MCP servers (from dev-studio/mcp.json and ~/.cursor/mcp.json)."""
    return mcp.list_servers()



@app.post("/api/mcp/{server}/tools")
def api_mcp_tools(server: str):
    """List tools exposed by a configured MCP server."""
    try:
        return {"ok": True, "server": server, "tools": mcp.list_tools(server)}
    except Exception as e:
        raise HTTPException(500, str(e))



@app.post("/api/mcp/{server}/call")
def api_mcp_call(server: str, body: MCPToolCallRequest):
    """Call an MCP tool on a configured server."""
    try:
        return {"ok": True, "server": server, "result": mcp.call_tool(server, body.tool, body.arguments)}
    except Exception as e:
        raise HTTPException(500, str(e))


# ── deploy routes ──────────────────────────────────────────────────────────────


def _build_factory_context(pid: str, state: dict) -> str:
    """Return a concise string describing the factory / app state for prompts."""
    cells = [c for c in state.get("cells", []) if c.get("enabled", True)]
    return f"""Factory / project: {pid}
Type: {state.get('type', 'factory')}
Template: {state.get('template', 'blank')}
Goal: {state.get('goal', '')}
Description: {state.get('description', '')}
Environment: {', '.join(state.get('environment', ['local']))}
Tools: {', '.join(state.get('tools', ['git', 'python3']))}
Cells ({len(cells)}):
{json.dumps(cells, indent=2)}
"""


def _deploy_prompt(pid: str, target: str, options: dict, state: dict, skill_text: str) -> str:
    ctx = _build_factory_context(pid, state)
    opts = json.dumps(options, indent=2)
    return (
        f"## Factory / project context\n{ctx}\n\n"
        f"## Skill package\n{skill_text}\n\n"
        f"Deploy this project to target: {target}\n"
        f"Options: {opts}\n\n"
        "Implement the deployment: write all required files (Dockerfile, docker-compose.yml, CI config, CDK, CloudFormation, etc.) "
        "into the project directory. Do NOT print long prose; only output shell commands, file paths, and short status lines. "
        "If the target is local-docker, build and run the container. If github-repo, create/push a repo. "
        "Use available tools (docker, git, gh, aws, mcp) via shell commands."
    )






def _cleanup_zombie_jobs() -> None:
    """Mark jobs stuck in 'running' for >30min as failed."""
    now = datetime.now(timezone.utc)
    for p in JOBS.glob("*.json"):
        try:
            j = json.loads(p.read_text())
            if j.get("status") != "running":
                continue
            created = j.get("updated_at") or j.get("created_at")
            if not created:
                continue
            ct = datetime.fromisoformat(created)
            age_min = (now - ct).total_seconds() / 60
            if age_min > 30:
                set_job(j["id"], status="failed", error=f"zombie: running for {age_min:.0f}min, process likely died")
        except Exception:
            pass



def _agent_status_from_result(job_id: str, result_text: str) -> None:
    """Agent CLIs can be killed by SIGPIPE (exit -13) even after committing work."""
    job = get_job(job_id)
    if job.get("status") != "failed":
        return
    exit_code = job.get("exit_code")
    killed_by_signal = isinstance(exit_code, int) and exit_code < 0
    if not killed_by_signal:
        return
    success_markers = [
        r"COMMITTED:\s*[a-f0-9]{7,40}",
        r"git push.*origin",
        r"pushed.*to.*origin",
        r"fix.*verified",
        r"fix.*confirmed",
        r"successfully.*fixed",
        r"fix.*committed",
    ]
    for pattern in success_markers:
        if re.search(pattern, result_text or "", re.IGNORECASE):
            set_job(job_id, status="completed", exit_code=exit_code,
                    note="process killed by signal but agent completed its work")
            return



def _capture_lesson(job: dict) -> None:
    if not job or not job.get("result_text"):
        return
    if job.get("purpose") != "bugfix":
        return
    lessons_file = LESSONS_FILE
    lessons = []
    if lessons_file.exists():
        try: lessons = json.loads(lessons_file.read_text())
        except: lessons = []
    result = job.get("result_text", "")
    approach = result.split("COMMITTED:")[0].split("FIXED_NOT_COMMITTED:")[0].split("CANNOT_FIX:")[0].strip()
    approach = approach[:500]
    lesson = {
        "id": uuid.uuid4().hex[:8],
        "job_id": job.get("id"),
        "kind": job.get("kind"),
        "model": job.get("model"),
        "purpose": job.get("purpose"),
        "approach": approach,
        "lesson_type": "success",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "selector_hint": (job.get("prompt") or "")[:200],
    }
    lessons.append(lesson)
    lessons = lessons[-100:]
    lessons_file.write_text(json.dumps(lessons, indent=2))



def _capture_cannot_fix_lesson(job: dict) -> bool:
    if not job or not job.get("result_text"):
        return False
    result = job.get("result_text", "")
    m = re.search(r"CANNOT_FIX:\s*(.+)", result, re.IGNORECASE | re.DOTALL)
    if not m:
        return False
    reason = m.group(1).strip().splitlines()[0][:500] if m.group(1).strip() else ""
    approach = result.split("CANNOT_FIX:")[0].strip()[-400:]
    lessons_file = LESSONS_FILE
    lessons = []
    if lessons_file.exists():
        try: lessons = json.loads(lessons_file.read_text())
        except: lessons = []
    if any(l.get("job_id") == job.get("id") and l.get("lesson_type") == "failure" for l in lessons):
        return False
    lesson = {
        "id": uuid.uuid4().hex[:8],
        "job_id": job.get("id"),
        "kind": job.get("kind"),
        "model": job.get("model"),
        "purpose": job.get("purpose"),
        "approach": approach,
        "reason": reason,
        "lesson_type": "failure",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "selector_hint": (job.get("prompt") or "")[:200],
    }
    lessons.append(lesson)
    lessons = lessons[-100:]
    lessons_file.write_text(json.dumps(lessons, indent=2))
    return True



def _maybe_capture_cannot_fix_lesson(job_id: str) -> None:
    try:
        job = get_job(job_id)
        if _capture_cannot_fix_lesson(job):
            set_job(job_id, cannot_fix_lesson_captured=True)
    except Exception:
        pass



def _notes_context_for_prompt(pid: str) -> str:
    """Build a context block of open notes to inject into agent/devin/grok prompts.
    Also includes recent lessons from verified fixes."""
    notes = PM.open_notes(pid)
    if not notes:
        return ""
    lessons_block = ""
    if LESSONS_FILE.exists():
        try:
            lessons = json.loads(LESSONS_FILE.read_text())
            recent = lessons[-5:]
            if recent:
                lessons_block = "\n## Recent lessons from verified fixes (learn from these patterns)\n"
                for l in recent:
                    lessons_block += f"- [{l.get('kind')}/{l.get('model')}] {l.get('approach','')[:200]}\n"
        except: pass
    lines = [
        "## Open bug notes (user-annotated on the page — address these & infer previous context)",
        "- Note to agent: Infer relationships and context across these notes (e.g. if multiple notes mention component sizing, position, tabs, or UI state, combine this context)."
    ]
    for i, n in enumerate(notes, 1):
        ctx = n.get("page_context") or {}
        ctx_str = ""
        if ctx.get("tab"): ctx_str += f" tab={ctx['tab']}"
        if ctx.get("cell_id"): ctx_str += f" cell={ctx['cell_id']}"
        if ctx.get("project"): ctx_str += f" project={ctx['project']}"
        html_snip = (n.get("element_html") or "")[:200].replace("\n", " ")
        imgs = n.get("images") or []
        img_note = f"\n   images: {len(imgs)} screenshot(s) attached" if imgs else ""
        lines.append(f"{i}. [{n.get('severity','bug')}] {n.get('note','')}"
                     f"\n   selector: {n.get('selector','')}"
                     f"\n   element: <{n.get('element_tag','')}> {(n.get('element_text','') or '')[:80]}"
                     f"\n   html: {html_snip}"
                     f"\n   context:{ctx_str} (note id {n.get('id')}){img_note}")
        for j, img in enumerate(imgs[:3]):
            if isinstance(img, str) and img.startswith("data:image/"):
                lines.append(f"   [image {j+1}]: {img[:8192]}{'...(truncated)' if len(img) > 8192 else ''}")
    return "\n".join(lines) + "\n\n" + lessons_block



def _skill_context_for_prompt(pid: str) -> str:
    return PM.build_skill_context(pid)



def _memory_context_for_prompt(query: str = "", pid: str = "default") -> str:
    search_q = query or f"dev-studio factory {pid} agent orchestration"
    memories = search_easy_memory(search_q, limit=3)
    if not memories:
        return ""
    lines = ["## Easy Memory Context (from /home/q/Downloads/memory)"]
    for m in memories:
        lines.append(f"- **{m.get('title')}** (Relevance: {m.get('score', 0):.2f}) [{m.get('type')}]")
        snip = m.get("snippet", "").strip().replace("\n", " ")
        if snip:
            lines.append(f"  {snip[:300]}...")
    return "\n".join(lines) + "\n\n"



def _devin_worker(job_id: str, cmd: list[str], cwd: Path) -> None:
    set_job(job_id, status="queued", note="waiting for Devin slot (max 3 concurrent)")
    t0 = time.time()
    job_meta = get_job(job_id) or {}
    model = job_meta.get("model") or "unknown"
    purpose = job_meta.get("purpose") or "devin_job"
    # Soft throttle against estimated free-tier + external CLIs
    try:
        from lib import devin_usage as dusage
        throttle, reason = dusage.USAGE.should_throttle()
        if throttle:
            set_job(job_id, note=f"waiting: devin quota pressure ({reason})")
            waited = 0.0
            while waited < 120 and throttle:
                time.sleep(5)
                waited += 5
                throttle, reason = dusage.USAGE.should_throttle()
            if throttle:
                set_job(
                    job_id,
                    status="failed",
                    error=f"devin free-tier pressure high ({reason}); try later",
                    exit_code=-1,
                )
                return
    except Exception:
        pass
    with _DEVIN_SEMAPHORE:
        set_job(job_id, status="running", note="")
        run_subprocess_job(job_id, cmd, cwd)
    log_path = JOBS / f"{job_id}.log"
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    parts = text.split("\n\n", 1)
    body = parts[1] if len(parts) > 1 else text
    set_job(job_id, result_text=body[-20000:])
    _agent_status_from_result(job_id, body)
    _maybe_capture_cannot_fix_lesson(job_id)
    try:
        from lib import devin_usage as dusage
        final = get_job(job_id) or {}
        ok = final.get("status") == "completed"
        prompt_est = ""
        pp = JOBS / f"{job_id}.prompt.txt"
        if pp.exists():
            prompt_est = pp.read_text(encoding="utf-8", errors="replace")[:50000]
        dusage.record_studio_call(
            model,
            prompt_est,
            body if ok else "",
            ok=ok,
            purpose=purpose,
            run_id=job_id,
            duration_secs=round(time.time() - t0, 3),
            error=None if ok else str(final.get("error") or final.get("status")),
        )
    except Exception:
        pass


# ── Agy (Google Antigravity CLI) ─────────────────────────────────────────────



def _agy_worker(job_id: str, cmd: list[str], cwd: Path) -> None:
    run_subprocess_job(job_id, cmd, cwd)
    log_path = JOBS / f"{job_id}.log"
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    parts = text.split("\n\n", 1)
    body = parts[1] if len(parts) > 1 else text
    set_job(job_id, result_text=body[-20000:])
    _agent_status_from_result(job_id, body)
    _maybe_capture_cannot_fix_lesson(job_id)


# ── Cerebras (free, fast inference) ──────────────────────────────────────────



def _cerebras_worker(job_id: str, prompt: str, model: str) -> None:
    set_job(job_id, status="running")
    log_path = JOBS / f"{job_id}.log"
    cancelled = False
    try:
        client, key_id = llm.make_cerebras_client()
        system_msg = ("You are a fast, focused coding agent fixing bugs in a web app. "
                      "Read the code, diagnose the root cause, propose the smallest correct fix, "
                      "and output the fix as a unified diff or clear instructions. Be concise.")
        with open(log_path, "w") as log:
            log.write(f"$ cerebras chat model={model} key={key_id}\n\n")
            log.flush()
            stream = client.chat.completions.create(
                messages=[{"role": "system", "content": system_msg},
                          {"role": "user", "content": prompt}],
                model=model, stream=True, max_completion_tokens=8192, temperature=0.2,
            )
            full = []
            in_reasoning = False
            for chunk in stream:
                with _cancel_flags_lock:
                    if job_id in _cancel_flags:
                        cancelled = True
                        break
                choice = chunk.choices[0]
                delta = choice.delta
                reasoning = getattr(delta, "reasoning_content", None) or ""
                content = getattr(delta, "content", None) or ""
                if reasoning:
                    if not in_reasoning:
                        log.write("\n[reasoning] ")
                        in_reasoning = True
                    log.write(reasoning)
                    log.flush()
                    full.append(reasoning)
                if content:
                    if in_reasoning:
                        log.write("\n[response] ")
                        in_reasoning = False
                    log.write(content)
                    log.flush()
                    full.append(content)
            text = "".join(full)
            if cancelled:
                log.write("\n\n[stopped by user]\n")
        if cancelled:
            set_job(job_id, status="failed", error="stopped by user",
                    result_text=text[-20000:])
        else:
            set_job(job_id, status="completed", result_text=text[-20000:])
            _maybe_capture_cannot_fix_lesson(job_id)
    except Exception as e:
        set_job(job_id, status="failed", error=str(e)[:500])
        with open(log_path, "a") as log:
            log.write(f"\n\nERROR: {e}\n")
    finally:
        with _cancel_flags_lock:
            _cancel_flags.discard(job_id)



def _pi_worker(job_id: str, cmd: list[str], prompt: str) -> None:
    set_job(job_id, status="running", cmd=cmd, cwd=str(ROOT))
    log_path = JOBS / f"{job_id}.log"
    try:
        full_env = os.environ.copy()
        for keyfile in (Path.home() / ".config" / "elicit" / "env", ROOT / ".env.elicit", STUDIO / ".env"):
            if keyfile.is_file():
                for line in keyfile.read_text().splitlines():
                    if "=" in line and not line.strip().startswith("#"):
                        k, v = line.split("=", 1)
                        full_env.setdefault(k.strip(), v.strip().strip("'\""))
        with open(log_path, "w") as log:
            log.write(f"$ {' '.join(cmd)} < prompt.txt\n\n")
            log.flush()
            proc = subprocess.run(cmd, cwd=str(ROOT), env=full_env, stdout=log,
                                  stderr=subprocess.STDOUT, text=True, input=prompt,
                                  timeout=60 * 10)
        text = log_path.read_text(encoding="utf-8", errors="replace")
        parts = text.split("\n\n", 1)
        result = parts[1] if len(parts) > 1 else text
        set_job(job_id, status="completed" if proc.returncode == 0 else "failed",
                exit_code=proc.returncode, result_text=result[-20000:])
        _agent_status_from_result(job_id, result)
        _maybe_capture_cannot_fix_lesson(job_id)
    except subprocess.TimeoutExpired:
        set_job(job_id, status="failed", error="timeout", log_path=str(log_path))
    except Exception as e:
        set_job(job_id, status="failed", error=str(e)[:500])


# ── costs ────────────────────────────────────────────────────────────────────


@app.post("/api/projects/{pid}/devin/run")
def api_devin_run(pid: str, body: DevinRunRequest):
    if not DEVIN_BIN.exists():
        raise HTTPException(500, "devin binary not found")
    job_id = uuid.uuid4().hex[:12]
    skill_ctx = _skill_context_for_prompt(pid) if getattr(body, "include_skill", True) else ""
    notes_ctx = _notes_context_for_prompt(pid)
    mem_ctx = _memory_context_for_prompt(body.prompt, pid) if getattr(body, "include_memory", True) else ""
    full_ctx = (skill_ctx + notes_ctx + mem_ctx).strip()
    prompt = (full_ctx + "\n\n" + body.prompt.strip()) if full_ctx else body.prompt.strip()
    prompt_path = JOBS / f"{job_id}.prompt.txt"
    prompt_path.write_text(prompt)
    cwd = PROJECTS_ROOT / pid
    cmd = [
        str(DEVIN_BIN), "--print",
        "--prompt-file", str(prompt_path),
        "--model", body.model,
        "--permission-mode", body.permission_mode,
    ]
    set_job(job_id, status="queued", kind="devin", project=pid, purpose=body.purpose,
            model=body.model, permission_mode=body.permission_mode,
            notes_injected=len(PM.open_notes(pid)), skill_injected=len(skill_ctx))
    threading.Thread(target=_devin_worker, args=(job_id, cmd, cwd), daemon=True).start()
    return {"job_id": job_id, "status": "queued", "model": body.model,
            "notes_injected": len(PM.open_notes(pid)), "skill_injected": len(skill_ctx)}


def _agent_status_from_result(job_id: str, result_text: str) -> None:
    """Agent CLIs can be killed by SIGPIPE (exit -13) even after committing work."""
    job = get_job(job_id)
    if job.get("status") != "failed":
        return
    exit_code = job.get("exit_code")
    killed_by_signal = isinstance(exit_code, int) and exit_code < 0
    if not killed_by_signal:
        return
    success_markers = [
        r"COMMITTED:\s*[a-f0-9]{7,40}",
        r"git push.*origin",
        r"pushed.*to.*origin",
        r"fix.*verified",
        r"fix.*confirmed",
        r"successfully.*fixed",
        r"fix.*committed",
    ]
    for pattern in success_markers:
        if re.search(pattern, result_text or "", re.IGNORECASE):
            set_job(job_id, status="completed", exit_code=exit_code,
                    note="process killed by signal but agent completed its work")
            return


def _devin_worker(job_id: str, cmd: list[str], cwd: Path) -> None:
    set_job(job_id, status="queued", note="waiting for Devin slot (max 3 concurrent)")
    t0 = time.time()
    job_meta = get_job(job_id) or {}
    model = job_meta.get("model") or "unknown"
    purpose = job_meta.get("purpose") or "devin_job"
    # Soft throttle against estimated free-tier + external CLIs
    try:
        from lib import devin_usage as dusage
        throttle, reason = dusage.USAGE.should_throttle()
        if throttle:
            set_job(job_id, note=f"waiting: devin quota pressure ({reason})")
            waited = 0.0
            while waited < 120 and throttle:
                time.sleep(5)
                waited += 5
                throttle, reason = dusage.USAGE.should_throttle()
            if throttle:
                set_job(
                    job_id,
                    status="failed",
                    error=f"devin free-tier pressure high ({reason}); try later",
                    exit_code=-1,
                )
                return
    except Exception:
        pass
    with _DEVIN_SEMAPHORE:
        set_job(job_id, status="running", note="")
        run_subprocess_job(job_id, cmd, cwd)
    log_path = JOBS / f"{job_id}.log"
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    parts = text.split("\n\n", 1)
    body = parts[1] if len(parts) > 1 else text
    set_job(job_id, result_text=body[-20000:])
    _agent_status_from_result(job_id, body)
    _maybe_capture_cannot_fix_lesson(job_id)
    try:
        from lib import devin_usage as dusage
        final = get_job(job_id) or {}
        ok = final.get("status") == "completed"
        prompt_est = ""
        pp = JOBS / f"{job_id}.prompt.txt"
        if pp.exists():
            prompt_est = pp.read_text(encoding="utf-8", errors="replace")[:50000]
        dusage.record_studio_call(
            model,
            prompt_est,
            body if ok else "",
            ok=ok,
            purpose=purpose,
            run_id=job_id,
            duration_secs=round(time.time() - t0, 3),
            error=None if ok else str(final.get("error") or final.get("status")),
        )
    except Exception:
        pass


# ── Agy (Google Antigravity CLI) ─────────────────────────────────────────────


@app.post("/api/projects/{pid}/agy/run")
def api_agy_run(pid: str, body: AgyRunRequest):
    if not AGY_BIN.exists():
        raise HTTPException(500, "agy binary not found")
    job_id = uuid.uuid4().hex[:12]
    skill_ctx = _skill_context_for_prompt(pid) if getattr(body, "include_skill", True) else ""
    notes_ctx = _notes_context_for_prompt(pid)
    mem_ctx = _memory_context_for_prompt(body.prompt, pid) if getattr(body, "include_memory", True) else ""
    full_ctx = (skill_ctx + notes_ctx + mem_ctx).strip()
    prompt = (full_ctx + "\n\n" + body.prompt.strip()) if full_ctx else body.prompt.strip()
    prompt_path = JOBS / f"{job_id}.prompt.txt"
    prompt_path.write_text(prompt)
    cwd = PROJECTS_ROOT / pid
    cmd = [
        str(AGY_BIN), "--print", prompt,
        "--model", body.model,
        "--mode", body.mode,
        "--dangerously-skip-permissions",
        "--print-timeout", "25m",
    ]
    set_job(job_id, status="queued", kind="agy", project=pid, purpose=body.purpose,
            model=body.model, mode=body.mode,
            notes_injected=len(PM.open_notes(pid)), skill_injected=len(skill_ctx))
    threading.Thread(target=_agy_worker, args=(job_id, cmd, cwd), daemon=True).start()
    return {"job_id": job_id, "status": "queued", "model": body.model,
            "notes_injected": len(PM.open_notes(pid)), "skill_injected": len(skill_ctx)}


def _agy_worker(job_id: str, cmd: list[str], cwd: Path) -> None:
    run_subprocess_job(job_id, cmd, cwd)
    log_path = JOBS / f"{job_id}.log"
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    parts = text.split("\n\n", 1)
    body = parts[1] if len(parts) > 1 else text
    set_job(job_id, result_text=body[-20000:])
    _agent_status_from_result(job_id, body)
    _maybe_capture_cannot_fix_lesson(job_id)


# ── Cerebras (free, fast inference) ──────────────────────────────────────────


@app.post("/api/projects/{pid}/cerebras/run")
def api_cerebras_run(pid: str, body: CerebrasRunRequest):
    if not llm.has_cerebras_key():
        raise HTTPException(500, "CEREBRAS_API_KEY not set in dev-studio/.env")
    job_id = uuid.uuid4().hex[:12]
    skill_ctx = _skill_context_for_prompt(pid) if getattr(body, "include_skill", True) else ""
    notes_ctx = _notes_context_for_prompt(pid)
    mem_ctx = _memory_context_for_prompt(body.prompt, pid) if getattr(body, "include_memory", True) else ""
    full_ctx = (skill_ctx + notes_ctx + mem_ctx).strip()
    prompt = (full_ctx + "\n\n" + body.prompt.strip()) if full_ctx else body.prompt.strip()
    (JOBS / f"{job_id}.prompt.txt").write_text(prompt)
    set_job(job_id, status="queued", kind="cerebras", project=pid, purpose=body.purpose,
            model=body.model, notes_injected=len(PM.open_notes(pid)), skill_injected=len(skill_ctx))
    threading.Thread(target=_cerebras_worker, args=(job_id, prompt, body.model), daemon=True).start()
    return {"job_id": job_id, "status": "queued", "model": body.model,
            "notes_injected": len(PM.open_notes(pid)), "skill_injected": len(skill_ctx)}


def _cerebras_worker(job_id: str, prompt: str, model: str) -> None:
    set_job(job_id, status="running")
    log_path = JOBS / f"{job_id}.log"
    cancelled = False
    try:
        client, key_id = llm.make_cerebras_client()
        system_msg = ("You are a fast, focused coding agent fixing bugs in a web app. "
                      "Read the code, diagnose the root cause, propose the smallest correct fix, "
                      "and output the fix as a unified diff or clear instructions. Be concise.")
        with open(log_path, "w") as log:
            log.write(f"$ cerebras chat model={model} key={key_id}\n\n")
            log.flush()
            stream = client.chat.completions.create(
                messages=[{"role": "system", "content": system_msg},
                          {"role": "user", "content": prompt}],
                model=model, stream=True, max_completion_tokens=8192, temperature=0.2,
            )
            full = []
            in_reasoning = False
            for chunk in stream:
                with _cancel_flags_lock:
                    if job_id in _cancel_flags:
                        cancelled = True
                        break
                choice = chunk.choices[0]
                delta = choice.delta
                reasoning = getattr(delta, "reasoning_content", None) or ""
                content = getattr(delta, "content", None) or ""
                if reasoning:
                    if not in_reasoning:
                        log.write("\n[reasoning] ")
                        in_reasoning = True
                    log.write(reasoning)
                    log.flush()
                    full.append(reasoning)
                if content:
                    if in_reasoning:
                        log.write("\n[response] ")
                        in_reasoning = False
                    log.write(content)
                    log.flush()
                    full.append(content)
            text = "".join(full)
            if cancelled:
                log.write("\n\n[stopped by user]\n")
        if cancelled:
            set_job(job_id, status="failed", error="stopped by user",
                    result_text=text[-20000:])
        else:
            set_job(job_id, status="completed", result_text=text[-20000:])
            _maybe_capture_cannot_fix_lesson(job_id)
    except Exception as e:
        set_job(job_id, status="failed", error=str(e)[:500])
        with open(log_path, "a") as log:
            log.write(f"\n\nERROR: {e}\n")
    finally:
        with _cancel_flags_lock:
            _cancel_flags.discard(job_id)


@app.post("/api/jobs/{job_id}/check-and-learn")
def api_job_check_and_learn(job_id: str):
    if not llm.has_cerebras_key():
        raise HTTPException(500, "CEREBRAS_API_KEY not set in dev-studio/.env")
    job = get_job(job_id)
    log_path = JOBS / f"{job_id}.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace")[-20000:] if log_path.exists() else ""
    prompt_path = JOBS / f"{job_id}.prompt.txt"
    prompt_text = prompt_path.read_text(encoding="utf-8", errors="replace")[-8000:] if prompt_path.exists() else ""
    investigation = (
        "You are investigating a failed agentic job to extract a reusable lesson.\n\n"
        f"## Failed job\n"
        f"- id: {job.get('id')}\n"
        f"- kind: {job.get('kind')}\n"
        f"- purpose: {job.get('purpose')}\n"
        f"- model: {job.get('model')}\n"
        f"- status: {job.get('status')}\n"
        f"- exit_code: {job.get('exit_code')}\n"
        f"- error: {job.get('error') or ''}\n"
        f"- note: {job.get('note') or ''}\n\n"
        f"## Original prompt (truncated)\n{prompt_text}\n\n"
        f"## Job log (truncated)\n{log_text}\n\n"
        f"## Result text (truncated)\n{(job.get('result_text') or '')[-4000:]}\n\n"
        "Investigate and answer concisely:\n"
        "1. ROOT CAUSE: why did this job fail or zombie?\n"
        "2. LOG INTERPRETATION: what do the log lines actually tell us?\n"
        "3. LESSON: a one-line reusable lesson for the agentic harness.\n"
        "4. NEXT ACTION: a concrete next step the user should take.\n"
        "End with a line: LESSON: <one-line lesson>\n"
    )
    new_id = uuid.uuid4().hex[:12]
    (JOBS / f"{new_id}.prompt.txt").write_text(investigation)
    set_job(new_id, status="queued", kind="cerebras", project=job.get("project") or "",
            purpose=f"check-and-learn:{job_id}", model="gemma-4-31b",
            notes_injected=0, parent_job=job_id)
    threading.Thread(target=_cerebras_worker, args=(new_id, investigation, "gemma-4-31b"),
                     daemon=True).start()
    return {"job_id": new_id, "parent_job": job_id, "status": "queued", "model": "gemma-4-31b"}


# ── Pi agent (mariozechner/pi-coding-agent, Cerebras-backed) ─────────────────


@app.post("/api/projects/{pid}/pi/run")
def api_pi_run(pid: str, body: PiRunRequest):
    if not PI_BIN.exists():
        raise HTTPException(500, "pi binary not found at " + str(PI_BIN))
    if not llm.has_cerebras_key():
        raise HTTPException(500, "CEREBRAS_API_KEY not set in dev-studio/.env (pi uses Cerebras)")
    job_id = uuid.uuid4().hex[:12]
    skill_ctx = _skill_context_for_prompt(pid) if getattr(body, "include_skill", True) else ""
    notes_ctx = _notes_context_for_prompt(pid)
    mem_ctx = _memory_context_for_prompt(body.prompt, pid) if getattr(body, "include_memory", True) else ""
    full_ctx = (skill_ctx + notes_ctx + mem_ctx).strip()
    prompt = (full_ctx + "\n\n" + body.prompt.strip()) if full_ctx else body.prompt.strip()
    prompt_path = JOBS / f"{job_id}.prompt.txt"
    prompt_path.write_text(prompt)
    cmd = [
        str(PI_BIN), "--print", "--provider", "cerebras", "--model", body.model,
        "--no-tools", "--no-session", "--no-extensions", "--no-skills",
    ]
    set_job(job_id, status="queued", kind="pi", project=pid, purpose=body.purpose,
            model=body.model, notes_injected=len(PM.open_notes(pid)), skill_injected=len(skill_ctx))
    threading.Thread(target=_pi_worker, args=(job_id, cmd, prompt), daemon=True).start()
    return {"job_id": job_id, "status": "queued", "model": body.model,
            "notes_injected": len(PM.open_notes(pid)), "skill_injected": len(skill_ctx)}


def _pi_worker(job_id: str, cmd: list[str], prompt: str) -> None:
    set_job(job_id, status="running", cmd=cmd, cwd=str(ROOT))
    log_path = JOBS / f"{job_id}.log"
    try:
        full_env = os.environ.copy()
        for keyfile in (Path.home() / ".config" / "elicit" / "env", ROOT / ".env.elicit", STUDIO / ".env"):
            if keyfile.is_file():
                for line in keyfile.read_text().splitlines():
                    if "=" in line and not line.strip().startswith("#"):
                        k, v = line.split("=", 1)
                        full_env.setdefault(k.strip(), v.strip().strip("'\""))
        with open(log_path, "w") as log:
            log.write(f"$ {' '.join(cmd)} < prompt.txt\n\n")
            log.flush()
            proc = subprocess.run(cmd, cwd=str(ROOT), env=full_env, stdout=log,
                                  stderr=subprocess.STDOUT, text=True, input=prompt,
                                  timeout=60 * 10)
        text = log_path.read_text(encoding="utf-8", errors="replace")
        parts = text.split("\n\n", 1)
        result = parts[1] if len(parts) > 1 else text
        set_job(job_id, status="completed" if proc.returncode == 0 else "failed",
                exit_code=proc.returncode, result_text=result[-20000:])
        _agent_status_from_result(job_id, result)
        _maybe_capture_cannot_fix_lesson(job_id)
    except subprocess.TimeoutExpired:
        set_job(job_id, status="failed", error="timeout", log_path=str(log_path))
    except Exception as e:
        set_job(job_id, status="failed", error=str(e)[:500])


# ── costs ────────────────────────────────────────────────────────────────────


@app.get("/api/jobs/archive")
def api_jobs_archive():
    """Return all jobs with their logs + prompts inlined."""
    _cleanup_zombie_jobs()
    files = sorted(JOBS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for p in files:
        try:
            j = json.loads(p.read_text())
        except Exception:
            continue
        if "prompt" not in j:
            pp = JOBS / f"{j['id']}.prompt.txt"
            if pp.exists():
                j["prompt"] = pp.read_text(encoding="utf-8", errors="replace")
        log_path = JOBS / f"{j['id']}.log"
        if log_path.exists():
            j["log"] = log_path.read_text(encoding="utf-8", errors="replace")[-50000:]
        out.append(j)
    return out


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    return get_job(job_id)


@app.get("/api/jobs")
def api_jobs():
    _cleanup_zombie_jobs()
    files = sorted(JOBS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:80]
    out = []
    for p in files:
        j = json.loads(p.read_text())
        if "prompt" not in j:
            pp = JOBS / f"{j['id']}.prompt.txt"
            if pp.exists():
                j["prompt"] = pp.read_text(encoding="utf-8", errors="replace")[:300]
        out.append(j)
    return out


@app.post("/api/jobs/clear")
def api_jobs_clear():
    """Remove completed/failed job files (keeps running + queued)."""
    removed = 0
    for p in JOBS.glob("*.json"):
        try:
            j = json.loads(p.read_text())
            if j.get("status") in ("completed", "failed"):
                p.unlink()
                for ext in (".log", ".prompt.txt"):
                    fp = JOBS / f"{j['id']}{ext}"
                    if fp.exists(): fp.unlink()
                removed += 1
        except Exception:
            pass
    return {"removed": removed}


@app.post("/api/jobs/{job_id}/verify")
def api_job_verify(job_id: str):
    """Manually mark a job's bug as actually fixed and capture a lesson."""
    job = get_job(job_id)
    set_job(job_id, verified=True, status="completed",
            note="user verified this fix is correct")
    _capture_lesson(job)
    return {"ok": True, "job_id": job_id, "verified": True}


@app.post("/api/jobs/{job_id}/stop")
def api_job_stop(job_id: str):
    """Stop a running job. Kills the underlying subprocess or signals the
    in-process Cerebras SDK stream to abort at the next chunk."""
    job = get_job(job_id)
    if job.get("status") not in ("running", "queued"):
        raise HTTPException(400, f"job is {job.get('status')}, not running")
    with _cancel_flags_lock:
        _cancel_flags.add(job_id)
    with _running_procs_lock:
        proc = _running_procs.get(job_id)
    if proc and proc.poll() is None:
        try: proc.terminate()
        except Exception: pass
    set_job(job_id, note="stop requested by user")
    return {"ok": True, "job_id": job_id, "stopped": True}


class UsefulRequest(BaseModel):
    useful: bool


@app.post("/api/jobs/{job_id}/useful")
def api_job_useful(job_id: str, body: UsefulRequest):
    """Mark whether a completed agent job's prompt was actually useful."""
    get_job(job_id)
    set_job(job_id, useful=body.useful, useful_marked_at=utcnow())
    return {"ok": True, "job_id": job_id, "useful": body.useful}


@app.post("/api/jobs/{job_id}/unverify")
def api_job_unverify(job_id: str):
    """Mark a completed/verified job as NOT actually completed."""
    get_job(job_id)
    set_job(job_id, verified=False, status="not-completed",
            note="user marked this fix as not completed")
    return {"ok": True, "job_id": job_id, "verified": False,
            "status": "not-completed"}


class EscalateRequest(BaseModel):
    agent: str
    model: Optional[str] = None
    extra_context: Optional[str] = None


@app.post("/api/jobs/{job_id}/escalate")
def api_job_escalate(job_id: str, body: EscalateRequest):
    """Re-dispatch a not-completed/failed job's original prompt to a different agent."""
    job = get_job(job_id)
    pid = job.get("project") or ""
    if not pid:
        raise HTTPException(400, "original job has no project; cannot escalate")
    prompt_path = JOBS / f"{job_id}.prompt.txt"
    if not prompt_path.exists():
        raise HTTPException(400, "original prompt not found; cannot escalate")
    base_prompt = prompt_path.read_text(encoding="utf-8", errors="replace")
    marker = "## Factory skill context"
    if marker in base_prompt:
        base_prompt = base_prompt.split(marker, 1)[0].rstrip()
    if body.extra_context:
        base_prompt += (
            "\n\n## Why the previous attempt did not resolve the bug\n"
            f"{body.extra_context.strip()}\n"
            "Address this reason specifically and propose a different approach than whatever was tried before.\n"
        )
    agent = (body.agent or "").lower()
    new_id = uuid.uuid4().hex[:12]
    purpose = job.get("purpose") or "general"
    if agent == "devin":
        if not DEVIN_BIN.exists():
            raise HTTPException(500, "devin binary not found")
        model = body.model or "glm-5.2-high"
        np = JOBS / f"{new_id}.prompt.txt"
        np.write_text(base_prompt)
        cmd = [str(DEVIN_BIN), "--print", "--prompt-file", str(np),
               "--model", model, "--permission-mode", "dangerous"]
        set_job(new_id, status="queued", kind="devin", project=pid, purpose=purpose,
                model=model, permission_mode="dangerous",
                notes_injected=job.get("notes_injected", 0),
                skill_injected=job.get("skill_injected", 0),
                escalated_from=job_id)
        threading.Thread(target=_devin_worker, args=(new_id, cmd, PROJECTS_ROOT / pid), daemon=True).start()
    elif agent == "cerebras":
        if not llm.has_cerebras_key():
            raise HTTPException(500, "CEREBRAS_API_KEY not set in dev-studio/.env")
        model = body.model or "gemma-4-31b"
        (JOBS / f"{new_id}.prompt.txt").write_text(base_prompt)
        set_job(new_id, status="queued", kind="cerebras", project=pid, purpose=purpose,
                model=model, notes_injected=job.get("notes_injected", 0),
                skill_injected=job.get("skill_injected", 0),
                escalated_from=job_id)
        threading.Thread(target=_cerebras_worker, args=(new_id, base_prompt, model),
                         daemon=True).start()
    elif agent == "pi":
        if not PI_BIN.exists():
            raise HTTPException(500, "pi binary not found at " + str(PI_BIN))
        if not llm.has_cerebras_key():
            raise HTTPException(500, "CEREBRAS_API_KEY not set in dev-studio/.env (pi uses Cerebras)")
        model = body.model or "gpt-oss-120b"
        (JOBS / f"{new_id}.prompt.txt").write_text(base_prompt)
        cmd = [str(PI_BIN), "--print", "--provider", "cerebras", "--model", model,
               "--no-tools", "--no-session", "--no-extensions", "--no-skills"]
        set_job(new_id, status="queued", kind="pi", project=pid, purpose=purpose,
                model=model, notes_injected=job.get("notes_injected", 0),
                skill_injected=job.get("skill_injected", 0),
                escalated_from=job_id)
        threading.Thread(target=_pi_worker, args=(new_id, cmd, base_prompt),
                         daemon=True).start()
    else:
        raise HTTPException(400, f"unknown agent '{agent}' (expected devin|cerebras|pi)")
    set_job(job_id, escalated_to=new_id)
    return {"ok": True, "job_id": new_id, "escalated_from": job_id,
            "agent": agent, "model": body.model}


class RecommendReasonsRequest(BaseModel):
    note_id: str
    job_id: Optional[str] = None
    model: str = "gemma-4-31b"


@app.get("/api/jobs/{job_id}/log")
def api_job_log(job_id: str):
    p = JOBS / f"{job_id}.log"
    if not p.exists():
        raise HTTPException(404, "no log")
    return {"log": p.read_text(encoding="utf-8", errors="replace")[-50000:]}


@app.get("/api/jobs/{job_id}/stream")
def api_job_stream(job_id: str):
    """SSE endpoint that streams live log lines for a job."""
    log_path = JOBS / f"{job_id}.log"

    def event_stream():
        import time as _time
        offset = 0
        if log_path.exists():
            content = log_path.read_text(encoding="utf-8", errors="replace")
            if len(content) > 2000:
                offset = len(content) - 2000
                yield f"data: {json.dumps({'type': 'history', 'text': content[:2000]})}\n\n"
            else:
                offset = 0
        last_status = None
        empty_ticks = 0
        while True:
            try:
                job = get_job(job_id)
                status = job.get("status")
            except Exception:
                status = "unknown"
            if status != last_status:
                yield f"data: {json.dumps({'type': 'status', 'status': status})}\n\n"
                last_status = status
            if log_path.exists():
                try:
                    content = log_path.read_text(encoding="utf-8", errors="replace")
                    if len(content) > offset:
                        new_text = content[offset:]
                        offset = len(content)
                        yield f"data: {json.dumps({'type': 'log', 'text': new_text})}\n\n"
                        empty_ticks = 0
                    else:
                        empty_ticks += 1
                except Exception:
                    pass
            if status in ("completed", "failed"):
                empty_ticks += 1
                if empty_ticks > 3:
                    yield f"data: {json.dumps({'type': 'done', 'status': status})}\n\n"
                    break
            _time.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── lessons ──────────────────────────────────────────────────────────────────


@app.get("/api/lessons")
def api_lessons():
    if not LESSONS_FILE.exists():
        return {"lessons": []}
    try: return {"lessons": json.loads(LESSONS_FILE.read_text())}
    except: return {"lessons": []}


# ── notes (bug annotations) ──────────────────────────────────────────────────


def _notes_context_for_prompt(pid: str) -> str:
    """Build a context block of open notes to inject into agent/devin/grok prompts.
    Also includes recent lessons from verified fixes."""
    notes = PM.open_notes(pid)
    if not notes:
        return ""
    lessons_block = ""
    if LESSONS_FILE.exists():
        try:
            lessons = json.loads(LESSONS_FILE.read_text())
            recent = lessons[-5:]
            if recent:
                lessons_block = "\n## Recent lessons from verified fixes (learn from these patterns)\n"
                for l in recent:
                    lessons_block += f"- [{l.get('kind')}/{l.get('model')}] {l.get('approach','')[:200]}\n"
        except: pass
    lines = [
        "## Open bug notes (user-annotated on the page — address these & infer previous context)",
        "- Note to agent: Infer relationships and context across these notes (e.g. if multiple notes mention component sizing, position, tabs, or UI state, combine this context)."
    ]
    for i, n in enumerate(notes, 1):
        ctx = n.get("page_context") or {}
        ctx_str = ""
        if ctx.get("tab"): ctx_str += f" tab={ctx['tab']}"
        if ctx.get("cell_id"): ctx_str += f" cell={ctx['cell_id']}"
        if ctx.get("project"): ctx_str += f" project={ctx['project']}"
        html_snip = (n.get("element_html") or "")[:200].replace("\n", " ")
        imgs = n.get("images") or []
        img_note = f"\n   images: {len(imgs)} screenshot(s) attached" if imgs else ""
        lines.append(f"{i}. [{n.get('severity','bug')}] {n.get('note','')}"
                     f"\n   selector: {n.get('selector','')}"
                     f"\n   element: <{n.get('element_tag','')}> {(n.get('element_text','') or '')[:80]}"
                     f"\n   html: {html_snip}"
                     f"\n   context:{ctx_str} (note id {n.get('id')}){img_note}")
        for j, img in enumerate(imgs[:3]):
            if isinstance(img, str) and img.startswith("data:image/"):
                lines.append(f"   [image {j+1}]: {img[:8192]}{'...(truncated)' if len(img) > 8192 else ''}")
    return "\n".join(lines) + "\n\n" + lessons_block


@app.get("/api/projects/{pid}/notes")
def api_notes_list(pid: str):
    return {"notes": PM.load_notes(pid), "open_count": len(PM.open_notes(pid))}


@app.post("/api/projects/{pid}/notes")
def api_notes_create(pid: str, body: NoteCreate):
    note = PM.add_note(pid, body.model_dump())
    return note


@app.patch("/api/projects/{pid}/notes/{nid}")
def api_notes_update(pid: str, nid: str, body: NoteUpdate):
    updated = PM.update_note(pid, nid, body.model_dump(exclude_unset=True))
    if not updated:
        raise HTTPException(404, "note not found")
    return updated


@app.delete("/api/projects/{pid}/notes/{nid}")
def api_notes_delete(pid: str, nid: str):
    if not PM.delete_note(pid, nid):
        raise HTTPException(404, "note not found")
    return {"ok": True}


# ── skill package ─────────────────────────────────────────────────────────────


@app.get("/api/projects/{pid}/skill")
def api_skill_get(pid: str):
    return {"text": PM.load_skill(pid)}


@app.put("/api/projects/{pid}/skill")
def api_skill_put(pid: str, body: SkillUpdate):
    PM.save_skill(pid, body.text)
    return {"ok": True}


@app.post("/api/projects/{pid}/skill/regenerate")
def api_skill_regenerate(pid: str, body: SkillRegenerateRequest):
    state = PM.load_state(pid)
    PM.ensure_skill_defaults(pid, state)
    return {"text": PM.load_skill(pid), "references": PM.list_references(pid)}


@app.get("/api/projects/{pid}/skill/references")
def api_references_list(pid: str):
    return {"references": PM.list_references(pid)}


@app.get("/api/projects/{pid}/skill/references/{name}")
def api_reference_get(pid: str, name: str):
    return {"name": Path(name).name, "text": PM.load_reference(pid, Path(name).name)}


@app.put("/api/projects/{pid}/skill/references/{name}")
def api_reference_put(pid: str, name: str, body: ReferenceUpdate):
    PM.save_reference(pid, Path(name).name, body.text)
    return {"ok": True}


@app.delete("/api/projects/{pid}/skill/references/{name}")
def api_reference_delete(pid: str, name: str):
    PM.delete_reference(pid, Path(name).name)
    return {"ok": True}


# ── archive / auto-archive ───────────────────────────────────────────────────


def _archive_project(pid: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    zip_path = ARCHIVES / f"{pid}-{ts}.zip"
    pdir = PROJECTS_ROOT / pid
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        notes = pdir / "notes.json"
        if notes.exists():
            zf.write(notes, "notes.json")
        if ELICIT_RESEARCH.exists():
            for d in ELICIT_RESEARCH.iterdir():
                if d.is_dir() and d.name.startswith(f"{pid}-"):
                    for f in d.rglob("*"):
                        if f.is_file():
                            zf.write(f, f"elicit/{d.name}/{f.relative_to(d)}")
        for p in JOBS.glob("*.json"):
            try:
                j = json.loads(p.read_text())
            except Exception:
                continue
            if j.get("project") != pid:
                continue
            zf.write(p, f"jobs/{p.name}")
            for ext in (".log", ".prompt.txt"):
                fp = JOBS / f"{j['id']}{ext}"
                if fp.exists():
                    zf.write(fp, f"jobs/{fp.name}")
    return zip_path


def ensure_evo_project_workspace(evo_id: str) -> Path:
    """Materialize evolution product HTML into projects/{evo_id}/ for bugfix agents."""
    evo_id = (evo_id or "").strip()
    if not evo_id:
        raise HTTPException(400, "evolution/project id required")
    dest = PROJECTS_ROOT / evo_id
    dest.mkdir(parents=True, exist_ok=True)
    # copy latest product if present
    root = EVOLUTIONS_ROOT / evo_id
    candidates = [
        root / "exports" / "PRODUCT-latest.html",
    ]
    for g in sorted(root.glob("gen*/product/index.html"), reverse=True):
        candidates.append(g)
    src_html = next((p for p in candidates if p.exists()), None)
    if src_html:
        target = dest / "index.html"
        try:
            if not target.exists() or src_html.stat().st_mtime > target.stat().st_mtime:
                import shutil as _sh
                _sh.copy2(src_html, target)
        except Exception:
            pass
    # minimal project state for notes API
    try:
        if not (dest / "state.json").exists():
            PM.save_state(evo_id, {
                "id": evo_id,
                "name": evo_id,
                "goal": f"Evolution product {evo_id}",
                "template": "product",
                "description": "Workspace for bugfix agents on Evolve product HTML",
                "cells": [],
                "environment": ["local"],
                "tools": ["git", "python3"],
            })
        PM.ensure_skill_defaults(evo_id, PM.load_state(evo_id))
    except Exception:
        pass
    return dest


@app.post("/api/evolve/{evo_id}/workspace")
def api_evolve_workspace(evo_id: str):
    """Ensure a project workspace exists for bugfix agents (product HTML)."""
    path = ensure_evo_project_workspace(evo_id)
    return {"ok": True, "project_id": evo_id, "path": str(path), "has_index": (path / "index.html").exists()}


@app.post("/api/projects/{pid}/recommend-reasons")


def main():
    import uvicorn
    host = os.environ.get("EVOLVE_STUDIO_HOST", "0.0.0.0")
    port = int(os.environ.get("EVOLVE_STUDIO_PORT", "8771"))
    print(f"Evolve Studio -> http://{host}:{port}")
    print(f"  data: {DATA_DIR}")
    print(f"  evolutions: {EVOLUTIONS_ROOT}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
