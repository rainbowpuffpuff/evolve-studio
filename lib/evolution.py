"""Algorithmic evolution for factory-of-factories and app designs.

Core concepts:
- Population: a set of candidate factories/apps for one generation.
- Genome: the project `state` (cells, environment, tools, MCP/deployment context).
- Fitness: a composite score built from LLM-rated benchmarks / KPIs.
- Attrition: low-fitness candidates are removed each generation.
- Mutation / crossover: survivors are used to breed the next generation.
- Innovation: periodic injection of novel cells, tools, or MCP servers.
- Brilliance / intelligence: candidates that exceed a threshold or show novelty.
"""
from __future__ import annotations

import json
import os
import random
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from lib import llm
from lib import deployer_lens as dlens


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvolutionStopped(Exception):
    """Cooperative stop requested by the user — not a hard failure."""

    def __init__(self, reason: str = "user requested stop"):
        super().__init__(reason)
        self.reason = reason


EVOLUTION_LLM_MODEL = os.environ.get("EVOLUTION_LLM_MODEL", "gemma-4-31b")

# Preferred Cerebras high-throughput free-tier workers (≥20 req/min class).
WORKER_HT_PREFERENCE: list[str] = [
    "gemma-4-31b",
    "llama3.1-8b",
    "llama3.1-70b",
    "qwen-2.5-coder-32b",
]

# Low-throughput Cerebras workers (also used as population members when diverse).
WORKER_LT_PREFERENCE: list[str] = [
    "gpt-oss-120b",
    "zai-glm-4.7",
]

# Cerebras worker models (create / evaluate / build). Prefer catalog with quotas.
EVOLUTION_LLM_MODELS: list[dict[str, Any]] = [
    {
        "id": mid,
        "label": meta.get("label") or mid,
        "provider": "cerebras",
        "worker": True,
        "high_throughput": int((meta.get("requests") or {}).get("minute") or 0) >= 20,
        "quota": {
            "requests": meta.get("requests"),
            "tokens": meta.get("tokens"),
            "context": meta.get("context"),
            "max_completion": meta.get("max_completion"),
            "tier": meta.get("tier"),
        },
    }
    for mid, meta in llm.CEREBRAS_MODEL_QUOTAS.items()
]
# Prefer gemma first in UI lists
EVOLUTION_LLM_MODELS.sort(key=lambda m: (0 if m["id"] == "gemma-4-31b" else 1, m["id"]))


def resolve_worker_model_pool(
    primary: Optional[str] = None,
    *,
    explicit: Optional[list[str]] = None,
    diverse: bool = True,
    include_low_throughput: bool = True,
    include_openrouter: bool = True,
) -> list[str]:
    """Models assigned across population members (create/build/eval).

    Mixes:
      - Cerebras high-throughput free (gemma, llama/qwen when listed)
      - Cerebras low-throughput (gpt-oss, zai-glm) when include_low_throughput
      - OpenRouter free models when key present + include_openrouter

    diverse=False → primary only (legacy).
    """
    primary = (primary or EVOLUTION_LLM_MODEL).strip() or EVOLUTION_LLM_MODEL
    if not diverse:
        return [primary]

    known = {m["id"]: m for m in EVOLUTION_LLM_MODELS}
    # Prefer models that actually exist on this Cerebras account
    try:
        live = llm.list_live_cerebras_models()
    except Exception:
        live = set(known.keys())

    def _ok_cb(mid: str) -> bool:
        mid = (mid or "").strip()
        if not mid:
            return False
        if mid in known and live and mid not in live:
            return False
        return True

    if explicit:
        pool: list[str] = []
        for mid in explicit:
            m = str(mid).strip()
            if m and m not in pool:
                pool.append(m)
        if primary not in pool:
            pool.insert(0, primary)
        return pool or [primary]

    # Build tier lists then interleave so small populations still mix HT/LT/OR
    ht: list[str] = []
    lt: list[str] = []
    or_free: list[str] = []

    def _push(bucket: list[str], mid: str) -> None:
        mid = (mid or "").strip()
        if not mid or mid in bucket or mid in ht or mid in lt or mid in or_free:
            return
        bucket.append(mid)

    if _ok_cb(primary):
        # primary always first slot overall
        pass

    for mid in WORKER_HT_PREFERENCE:
        if mid in known and _ok_cb(mid):
            _push(ht, mid)
    for mid, meta in known.items():
        if meta.get("high_throughput") and _ok_cb(mid):
            _push(ht, mid)

    if include_low_throughput:
        for mid in WORKER_LT_PREFERENCE:
            if mid in known and _ok_cb(mid):
                _push(lt, mid)
        for mid, meta in known.items():
            if not meta.get("high_throughput") and _ok_cb(mid):
                _push(lt, mid)

    if include_openrouter and llm.has_openrouter_key():
        for entry in llm.openrouter_free_models():
            _push(or_free, entry["id"])

    # Ensure primary is first in its tier
    for bucket in (ht, lt):
        if primary in bucket:
            bucket.remove(primary)
            bucket.insert(0, primary)
            break
    else:
        if _ok_cb(primary):
            # unknown primary — treat as HT seed
            ht.insert(0, primary)

    tiers = [b for b in (ht, lt, or_free) if b]
    if not tiers:
        return [primary]

    pool = []
    # Round-robin across tiers so pop[0]=HT, pop[1]=LT, pop[2]=OR, pop[3]=HT…
    max_len = max(len(b) for b in tiers)
    for i in range(max_len):
        for b in tiers:
            if i < len(b) and b[i] not in pool:
                pool.append(b[i])
    # Primary always first individual index 0
    if primary in pool:
        pool.remove(primary)
    pool.insert(0, primary)
    return pool or [primary]


def worker_pool_catalog() -> list[dict[str, Any]]:
    """UI/API catalog: cerebras workers + openrouter free when keyed."""
    out = list(EVOLUTION_LLM_MODELS)
    if llm.has_openrouter_key():
        out.extend(llm.openrouter_free_models())
    return out

EVOLUTION_PROVIDER_OPTIONS: list[str] = [
    "cerebras",
    "grok",
    "devin",
    "agy",
    "pi",
    "aws",
    "github",
    "cloudflare",
    "openai",
    "anthropic",
]


@dataclass
class EvolutionConfig:
    goal: str
    # product = ship a user-facing artifact (HTML/app/report) toward the goal prompt
    # factory / factory-factory / app = architecture-oriented modes
    output_type: str = "product"  # product | app | factory | factory-factory | auto
    population_size: int = 4
    generations: int = 3
    mutation_rate: float = 0.35
    attrition_rate: float = 0.5
    innovation_rate: float = 0.4
    benchmark_weights: dict[str, float] = field(default_factory=dict)
    deployment_target: Optional[str] = None
    budget_usd: Optional[float] = None
    providers: list[str] = field(default_factory=list)
    mcp_servers: list[str] = field(default_factory=list)
    run_tests: bool = False
    promote_best: bool = True
    name: Optional[str] = None
    llm_model: str = ""
    # When True, each generation builds/improves real software files (not only architecture JSON).
    build_software: bool = True
    # scaffold = minimal runnable skeleton; implement = fuller multi-file app each gen
    build_depth: str = "implement"  # scaffold | implement
    # Planner expands the user goal into a brief before Cerebras workers run.
    # e.g. "none", "cerebras:gemma-4-31b", "agy:gemini-3.1-pro-high", "devin:swe-1-7", "codex:gpt-5.6-sol", "claude:opus"
    planner_id: str = "cerebras:gemma-4-31b"
    # Filled after planner runs (or equals goal if planner is none / fails).
    goal_brief: str = ""
    # Decision maker / product director: ranks candidates, picks champion, writes cooperation brief + product HTML.
    # Default: zai-glm-4.7 (lower TPM, good for 1 call/gen). Multi-harness id like planner.
    decision_maker_id: str = "cerebras:zai-glm-4.7"
    produce_product: bool = True  # generational HTML product every gen
    use_git: bool = True  # local git trees per candidate + product
    cooperation: bool = True  # after director pick, shared product workspace
    director_fitness_blend: float = 0.45  # fitness' = (1-b)*worker + b*director
    # Research harness: plan→web search→fetch→synthesize (Cerebras + public HTTP)
    research_enabled: bool = True
    # Assign different models to different population members (Cerebras HT+LT + OpenRouter free)
    diverse_workers: bool = True
    include_low_throughput_workers: bool = True  # gpt-oss / zai-glm as some individuals
    include_openrouter_workers: bool = True  # free OpenRouter models as some individuals
    # Optional explicit worker pool (ids). Empty → auto mix.
    worker_models: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.benchmark_weights:
            # Deployer lens: products must be shippable + monetizable; factories produce money lines.
            ot = (self.output_type or "product").strip().lower()
            if ot in ("product", "app") or ot == "auto":
                self.benchmark_weights = dict(dlens.PRODUCT_BENCHMARK_WEIGHTS)
            else:
                self.benchmark_weights = dict(dlens.FACTORY_BENCHMARK_WEIGHTS)
        if not (self.llm_model or "").strip():
            self.llm_model = EVOLUTION_LLM_MODEL
        if self.build_depth not in ("scaffold", "implement"):
            self.build_depth = "implement"
        if not (self.goal_brief or "").strip():
            self.goal_brief = self.goal
        try:
            self.director_fitness_blend = max(0.0, min(1.0, float(self.director_fitness_blend)))
        except Exception:
            self.director_fitness_blend = 0.45
        if not (self.decision_maker_id or "").strip():
            self.decision_maker_id = "cerebras:zai-glm-4.7"
        if self.worker_models is None:
            self.worker_models = []
        self.worker_models = [str(m).strip() for m in (self.worker_models or []) if str(m).strip()]
        ot = (self.output_type or "product").strip().lower()
        if ot not in ("product", "app", "factory", "factory-factory", "auto"):
            ot = "product"
        if ot == "auto":
            # Prefer product shipping unless the goal clearly asks for factories
            g = (self.goal or "").lower()
            if "factory-factory" in g or "factory of factories" in g:
                ot = "factory-factory"
            elif re.search(r"\bfactory\b", g) and "product" not in g:
                ot = "factory"
            else:
                ot = "product"
        self.output_type = ot
        # Product mode always produces generational HTML artifacts
        if self.output_type == "product":
            self.produce_product = True


@dataclass
class Candidate:
    id: str
    generation: int
    genome: dict
    meta: dict
    scores: dict[str, float] = field(default_factory=dict)
    fitness: float = 0.0
    brilliant: bool = False
    rationale: str = ""
    path: Optional[Path] = None


@dataclass
class GenerationSummary:
    generation: int
    best_fitness: float
    avg_fitness: float
    survivors: int
    population: int
    brilliant: list[str] = field(default_factory=list)
    brilliant_count: int = 0
    survivors_ids: list[str] = field(default_factory=list)
    eliminated_ids: list[str] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)
    events_count: int = 0


class EvolutionRun:
    """In-memory + on-disk record of an evolution run."""

    def __init__(self, evo_id: str, cfg: EvolutionConfig, root: Path):
        self.id = evo_id
        self.cfg = cfg
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.status = "queued"  # queued | running | stopping | stopped | completed | failed
        self.current_generation = 0
        self.generations: list[GenerationSummary] = []
        self.candidates: list[Candidate] = []
        self.best: Optional[Candidate] = None
        self.error: Optional[str] = None
        self.stop_reason: Optional[str] = None
        self.events: list[dict] = []
        self.llm_calls: list[dict] = []
        self.promoted_project_id: Optional[str] = None
        self.llm_model: str = (cfg.llm_model or EVOLUTION_LLM_MODEL).strip() or EVOLUTION_LLM_MODEL
        # Frozen architecture charter — generations must build on this, not replace it
        self.charter: dict = {}
        # Evolving prompt bank — best prompt addenda survive with high-fitness candidates
        self.prompt_bank: dict = {
            "create_addendum": "",
            "build_addendum": "",
            "evaluate_addendum": "",
            "history": [],  # [{generation, fitness, variant, candidate_id}]
        }
        self.created_at = utcnow()
        self.updated_at = utcnow()
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._save()

    def request_stop(self, reason: str = "user requested stop") -> None:
        """Signal the worker to exit cooperatively and keep progress on disk."""
        self.stop_reason = reason or "user requested stop"
        self._stop.set()
        with self.lock:
            if self.status in ("queued", "running", "starting"):
                self.status = "stopping"
                self.updated_at = utcnow()
                self._save()

    def should_stop(self) -> bool:
        return self._stop.is_set()

    def check_stop(self) -> None:
        if self._stop.is_set():
            raise EvolutionStopped(self.stop_reason or "user requested stop")

    def clear_stop(self) -> None:
        """Clear stop signal so a resumed worker can run again."""
        self._stop.clear()
        self.stop_reason = None

    def log_event(
        self,
        type: str,
        message: str,
        generation: Optional[int] = None,
        candidate_id: Optional[str] = None,
        model: Optional[str] = None,
        details: Optional[dict] = None,
        **extra,
    ) -> None:
        event = {
            "ts": utcnow(),
            "generation": generation if generation is not None else self.current_generation,
            "type": type,
            "message": message,
        }
        if candidate_id:
            event["candidate_id"] = candidate_id
        if model:
            event["model"] = model
        payload = {}
        if details:
            payload.update(details)
        if extra:
            payload.update(extra)
        if payload:
            event["details"] = payload
        with self.lock:
            self.events.append(event)
            self._save()

    def _to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "llm_model": self.llm_model,
            "stop_reason": self.stop_reason,
            "promoted_project_id": self.promoted_project_id,
            "config": {
                "goal": self.cfg.goal,
                "output_type": self.cfg.output_type,
                "population_size": self.cfg.population_size,
                "generations": self.cfg.generations,
                "mutation_rate": self.cfg.mutation_rate,
                "attrition_rate": self.cfg.attrition_rate,
                "innovation_rate": self.cfg.innovation_rate,
                "benchmark_weights": self.cfg.benchmark_weights,
                "deployment_target": self.cfg.deployment_target,
                "budget_usd": self.cfg.budget_usd,
                "providers": self.cfg.providers,
                "mcp_servers": self.cfg.mcp_servers,
                "run_tests": self.cfg.run_tests,
                "promote_best": self.cfg.promote_best,
                "name": self.cfg.name,
                "llm_model": self.cfg.llm_model or self.llm_model,
                "build_software": self.cfg.build_software,
                "build_depth": self.cfg.build_depth,
                "planner_id": self.cfg.planner_id,
                "goal_brief": self.cfg.goal_brief,
                "decision_maker_id": self.cfg.decision_maker_id,
                "produce_product": self.cfg.produce_product,
                "use_git": self.cfg.use_git,
                "cooperation": self.cfg.cooperation,
                "director_fitness_blend": self.cfg.director_fitness_blend,
                "research_enabled": getattr(self.cfg, "research_enabled", True),
                "diverse_workers": getattr(self.cfg, "diverse_workers", True),
                "include_low_throughput_workers": getattr(self.cfg, "include_low_throughput_workers", True),
                "include_openrouter_workers": getattr(self.cfg, "include_openrouter_workers", True),
                "worker_models": list(getattr(self.cfg, "worker_models", None) or []),
                "worker_pool": resolve_worker_model_pool(
                    self.cfg.llm_model or self.llm_model,
                    explicit=list(getattr(self.cfg, "worker_models", None) or []) or None,
                    diverse=bool(getattr(self.cfg, "diverse_workers", True)),
                    include_low_throughput=bool(getattr(self.cfg, "include_low_throughput_workers", True)),
                    include_openrouter=bool(getattr(self.cfg, "include_openrouter_workers", True)),
                ),
            },
            "current_generation": self.current_generation,
            "best": self._candidate_dict(self.best) if self.best else None,
            "candidates": [self._candidate_dict(c) for c in self.candidates],
            "generations": [self._gen_dict(g) for g in self.generations],
            "events": self.events,
            "llm_calls": self.llm_calls,
            "charter": self.charter,
            "prompt_bank": self.prompt_bank,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
        }

    def _slim_cells(self, cells: list) -> list[dict]:
        """Compact cell list for genome visualization (no huge params dumps)."""
        out = []
        for cell in cells or []:
            if not isinstance(cell, dict):
                continue
            out.append({
                "id": cell.get("id"),
                "role": cell.get("role") or "cell",
                "name": cell.get("name") or cell.get("role") or cell.get("id"),
                "goal": cell.get("goal") or "",
                "tools": cell.get("tools") or [],
                "environment": cell.get("environment"),
                "status": cell.get("status") or "ready",
                "enabled": cell.get("enabled", True),
            })
        return out

    def _candidate_dict(self, c: Optional[Candidate]) -> dict:
        if not c:
            return {}
        cells = c.genome.get("cells", [])
        slim = self._slim_cells(cells)
        return {
            "id": c.id,
            "generation": c.generation,
            "fitness": round(c.fitness, 4),
            "brilliant": c.brilliant,
            "rationale": c.rationale,
            "scores": c.scores,
            "meta": c.meta,
            "path": str(c.path) if c.path else None,
            "model": c.meta.get("model"),
            "cell_count": len(slim),
            "cell_roles": [cell.get("role", "") for cell in slim],
            "description": c.genome.get("description", ""),
            "template": c.genome.get("template"),
            "order": c.genome.get("order") or [cell.get("id") for cell in slim],
            # Full-enough cells for lysosome-style genome visualization
            "cells": slim,
            "build": c.meta.get("build") or c.genome.get("build"),
            "artifacts": (c.meta.get("build") or {}).get("files") or c.genome.get("artifacts") or [],
            "prompt_variant": c.meta.get("prompt_variant") or c.genome.get("prompt_variant"),
            "lineage": c.meta.get("lineage"),
            "continuity": c.meta.get("continuity"),
        }

    def _gen_dict(self, g: GenerationSummary) -> dict:
        return {
            "generation": g.generation,
            "best_fitness": round(g.best_fitness, 4),
            "avg_fitness": round(g.avg_fitness, 4),
            "survivors": g.survivors,
            "population": g.population,
            "brilliant": g.brilliant,
            "brilliant_count": g.brilliant_count,
            "survivors_ids": g.survivors_ids,
            "eliminated_ids": g.eliminated_ids,
            "candidates": g.candidates,
            "events_count": g.events_count,
        }

    def _save(self) -> None:
        (self.root / "evolution.json").write_text(json.dumps(self._to_dict(), indent=2))

    def update(
        self,
        status: Optional[str] = None,
        generation: Optional[int] = None,
        candidates: Optional[list[Candidate]] = None,
        best: Optional[Candidate] = None,
        gen_summary: Optional[GenerationSummary] = None,
        error: Optional[str] = None,
    ) -> None:
        with self.lock:
            if status:
                self.status = status
            if generation is not None:
                self.current_generation = generation
            if candidates is not None:
                self.candidates = candidates
            if best is not None:
                self.best = best
            if gen_summary is not None:
                self.generations.append(gen_summary)
            if error is not None:
                self.error = error
            self.updated_at = utcnow()
            self._save()


class EvolutionEngine:
    """Run an evolutionary design loop for factories/apps."""

    def __init__(
        self,
        pm_factory: Any,  # ProjectManager instance for candidates
        call_llm: Callable[[str, str], str] = llm.call_worker_sync,
        build_tester: Optional[Callable[[Candidate], dict]] = None,
        real_pm: Optional[Any] = None,
    ):
        self.pm = pm_factory
        self.real_pm = real_pm
        self.call_llm = call_llm
        self.build_tester = build_tester
        self.runs: dict[str, EvolutionRun] = {}
        self._runs_lock = threading.Lock()

    def _worker_pool(self, cfg: EvolutionConfig) -> list[str]:
        return resolve_worker_model_pool(
            cfg.llm_model or EVOLUTION_LLM_MODEL,
            explicit=list(cfg.worker_models or []) or None,
            diverse=bool(getattr(cfg, "diverse_workers", True)),
            include_low_throughput=bool(getattr(cfg, "include_low_throughput_workers", True)),
            include_openrouter=bool(getattr(cfg, "include_openrouter_workers", True)),
        )

    def _pick_worker_model(
        self,
        cfg: EvolutionConfig,
        index: int,
        *,
        parent_model: Optional[str] = None,
    ) -> str:
        """Sticky model for one population individual. Round-robin + optional parent inherit."""
        pool = self._worker_pool(cfg)
        if not pool:
            return cfg.llm_model or EVOLUTION_LLM_MODEL
        # ~40% chance offspring keeps a parent model (lineage stability)
        if parent_model and parent_model in pool and random.random() < 0.4:
            return parent_model
        return pool[index % len(pool)]

    def _candidate_model(self, cand: Optional[Candidate], run: EvolutionRun) -> str:
        if cand is not None:
            m = (cand.meta or {}).get("model") or (cand.meta or {}).get("llm_model")
            if m:
                return str(m)
        return run.llm_model or run.cfg.llm_model or EVOLUTION_LLM_MODEL

    def _call_llm_tracked(
        self,
        run: EvolutionRun,
        prompt: str,
        purpose: str,
        candidate_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """Call the LLM and persist the full prompt + response for later inspection."""
        run.check_stop()
        if not model and candidate_id:
            # Prefer sticky model on the candidate
            for c in run.candidates or []:
                if c.id == candidate_id:
                    model = self._candidate_model(c, run)
                    break
            if not model:
                # During create, candidate may not be on run.candidates yet
                pass
        model = (model or run.llm_model or EVOLUTION_LLM_MODEL).strip() or EVOLUTION_LLM_MODEL
        start = time.time()
        ok = True
        error_msg = None
        text = ""
        # Cap stored bodies so evolution.json stays manageable while still showing full answers in UI.
        max_store = int(os.environ.get("EVOLUTION_LLM_STORE_CHARS", "120000"))
        try:
            # Tag usage to this evolution run for quota dashboards
            llm.set_usage_context(run_id=run.id, purpose=purpose)
            # Prefer signature that accepts run_id when available (lib.llm.call_worker_sync)
            try:
                text = self.call_llm(prompt, model, run_id=run.id, purpose=purpose) or ""  # type: ignore[call-arg]
            except TypeError:
                text = self.call_llm(prompt, model) or ""
        except Exception as e:
            ok = False
            error_msg = str(e)
            raise e
        finally:
            llm.clear_usage_context()
            call_id = uuid.uuid4().hex[:10]
            stored_prompt = prompt if len(prompt) <= max_store else prompt[:max_store] + "\n…[truncated]"
            stored_response = text if len(text) <= max_store else text[:max_store] + "\n…[truncated]"
            # Pull latest usage event for this model if present
            tok = {}
            try:
                snap = llm.USAGE.snapshot(run_id=run.id)
                recent = snap.get("recent_events") or []
                for ev in reversed(recent):
                    if ev.get("model") == model and ev.get("run_id") == run.id:
                        tok = {
                            "prompt_tokens": ev.get("prompt_tokens"),
                            "completion_tokens": ev.get("completion_tokens"),
                            "total_tokens": ev.get("total_tokens"),
                        }
                        break
            except Exception:
                pass
            record = {
                "id": call_id,
                "ts": utcnow(),
                "purpose": purpose,
                "model": model,
                "candidate_id": candidate_id,
                "generation": run.current_generation,
                "prompt_chars": len(prompt),
                "response_chars": len(text),
                "prompt_preview": prompt[:400],
                "response_preview": (text[:400] if text else ""),
                "prompt": stored_prompt,
                "response": stored_response,
                "duration_secs": round(time.time() - start, 2),
                "ok": ok,
                "error": error_msg,
                **tok,
            }
            with run.lock:
                run.llm_calls.append(record)
                # Per-call files (easy to open / archive)
                calls_dir = run.root / "llm_calls"
                calls_dir.mkdir(parents=True, exist_ok=True)
                (calls_dir / f"{len(run.llm_calls):03d}-{purpose}-{call_id}.json").write_text(
                    json.dumps(record, indent=2), encoding="utf-8"
                )
                run._save()
                self._write_llm_transcript(run)
        return text

    def _write_llm_transcript(self, run: EvolutionRun) -> None:
        """Write a human-readable document of every model prompt/answer for this run."""
        lines = [
            f"# Evolution model answers — {run.id}",
            "",
            f"- **Status:** {run.status}",
            f"- **Goal:** {run.cfg.goal}",
            f"- **Model:** {run.llm_model}",
            f"- **Calls:** {len(run.llm_calls)}",
            f"- **Updated:** {utcnow()}",
            "",
            "---",
            "",
        ]
        for i, c in enumerate(run.llm_calls, 1):
            ok = "ok" if c.get("ok", True) and not c.get("error") else "error"
            lines.extend([
                f"## {i}. {c.get('purpose', 'llm')}  ·  {ok}",
                "",
                f"- **id:** `{c.get('id', '')}`",
                f"- **model:** `{c.get('model', '')}`",
                f"- **candidate:** `{c.get('candidate_id') or '—'}`",
                f"- **generation:** {c.get('generation')}",
                f"- **ts:** {c.get('ts', '')}",
                f"- **duration:** {c.get('duration_secs', '—')}s",
                f"- **prompt chars / response chars:** {c.get('prompt_chars', 0)} / {c.get('response_chars', 0)}",
                "",
            ])
            if c.get("error"):
                lines.extend(["### Error", "", "```", str(c["error"]), "```", ""])
            lines.extend([
                "### Prompt",
                "",
                "```",
                c.get("prompt") or c.get("prompt_preview") or "(empty)",
                "```",
                "",
                "### Response",
                "",
                "```",
                c.get("response") or c.get("response_preview") or ("(no response)" if not c.get("error") else "(failed)"),
                "```",
                "",
                "---",
                "",
            ])
        (run.root / "model-answers.md").write_text("\n".join(lines), encoding="utf-8")
        (run.root / "llm-calls.json").write_text(json.dumps(run.llm_calls, indent=2), encoding="utf-8")

    def _write_trace_digest(
        self,
        run: EvolutionRun,
        *,
        generation: Optional[int] = None,
        with_gemma: bool = False,
    ) -> Optional[dict]:
        """Write structural (and optional Gemma) digest for UI sense-making."""
        try:
            from lib import trace_digest
        except Exception:
            return None
        data = run._to_dict()
        dig = trace_digest.digest_for_run_root(
            run.root, data, with_gemma=with_gemma, model=run.llm_model or EVOLUTION_LLM_MODEL
        )
        gen = generation if generation is not None else run.current_generation
        try:
            run.log_event(
                "maintain",
                f"Trace digest updated · {dig.get('headline') or dig.get('kind')}",
                generation=gen,
                details={
                    "kind": dig.get("kind"),
                    "counts": dig.get("counts") or (dig.get("structural") or {}).get("counts"),
                },
            )
        except Exception:
            pass
        return dig

    def start(
        self,
        cfg: EvolutionConfig,
        evolutions_root: Path,
        *,
        seed_from: Optional[str] = None,
        seed_gen: Optional[int] = None,
    ) -> EvolutionRun:
        evo_id = uuid.uuid4().hex[:12]
        root = evolutions_root / evo_id
        run = EvolutionRun(evo_id, cfg, root)
        if seed_from:
            try:
                self._seed_run_from_parent(run, seed_from, Path(evolutions_root), seed_gen=seed_gen)
            except Exception as e:
                run.log_event("error", f"Seed from {seed_from} failed: {e}")
        with self._runs_lock:
            self.runs[evo_id] = run
        threading.Thread(target=self._run_loop, args=(run,), daemon=True).start()
        return run

    def list_product_seeds(self, evolutions_root: Path, limit: int = 80) -> list[dict[str, Any]]:
        """Dropdown catalog: prior runs with product HTML / topics to iterate from."""
        root = Path(evolutions_root)
        if not root.exists():
            return []
        items: list[dict[str, Any]] = []
        for p in sorted(root.glob("*/evolution.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            evo_id = data.get("id") or p.parent.name
            cfg = data.get("config") or {}
            goal = (cfg.get("goal") or "").strip()
            name = (cfg.get("name") or "").strip()
            # Prefer short topic from name, else first line of goal
            topic = name or (goal.split("\n")[0].strip()[:72] if goal else evo_id)
            if len(topic) > 72:
                topic = topic[:71] + "…"
            parent = p.parent
            latest = parent / "exports" / "PRODUCT-latest.html"
            gen_products = sorted(
                [d for d in parent.glob("gen*/product/index.html") if d.is_file()],
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            has_html = latest.exists() or bool(gen_products)
            # Highest gen number with a product (not mtime — gallery is "last gen only")
            latest_gen = None
            for gp in gen_products:
                try:
                    g = int(gp.parent.parent.name.replace("gen", ""))
                    if latest_gen is None or g > latest_gen:
                        latest_gen = g
                except Exception:
                    continue
            gens_done = len(data.get("generations") or [])
            if latest_gen is None and gens_done:
                latest_gen = max(0, gens_done - 1)
            best = data.get("best") or {}
            items.append({
                "id": evo_id,
                "topic": topic,
                "label": f"{topic} · {data.get('status') or '?'} · gen{latest_gen if latest_gen is not None else gens_done} · {evo_id[:8]}",
                "name": name or None,
                "goal": goal[:400],
                "status": data.get("status"),
                "has_product_html": has_html,
                "product_url": f"/api/evolve/{evo_id}/export/file/PRODUCT-latest.html" if has_html else None,
                "latest_gen": latest_gen,
                "generations_done": gens_done,
                "generations_cfg": cfg.get("generations"),
                "best_fitness": best.get("fitness"),
                "updated_at": data.get("updated_at") or data.get("created_at"),
                "llm_model": data.get("llm_model") or cfg.get("llm_model"),
                "output_type": cfg.get("output_type"),
            })
            if len(items) >= limit * 2:
                break
        # Prefer runs that already have product HTML (iterable topics first)
        items.sort(
            key=lambda s: (
                0 if s.get("has_product_html") else 1,
                -(1 if s.get("updated_at") else 0),
                s.get("updated_at") or "",
            ),
        )
        # reverse chronological within groups — re-sort with mtime preference
        with_html = [s for s in items if s.get("has_product_html")]
        without = [s for s in items if not s.get("has_product_html")]
        with_html.sort(key=lambda s: s.get("updated_at") or "", reverse=True)
        without.sort(key=lambda s: s.get("updated_at") or "", reverse=True)
        return (with_html + without)[:limit]

    def list_product_gallery(self, evolutions_root: Path, limit: int = 100) -> list[dict[str, Any]]:
        """Gallery of last-gen product HTMLs only (one card per evolution) + titles for deploy wiring."""
        seeds = self.list_product_seeds(evolutions_root, limit=limit * 2)
        gallery: list[dict[str, Any]] = []
        root = Path(evolutions_root)
        for s in seeds:
            if not s.get("has_product_html"):
                continue
            evo_id = s["id"]
            parent = root / evo_id
            # Prefer highest genN/product; fall back to PRODUCT-latest
            last_gen: Optional[int] = None
            html_path: Optional[Path] = None
            product_rel = "exports/PRODUCT-latest.html"
            gen_dirs: list[tuple[int, Path]] = []
            for d in parent.glob("gen*/product"):
                idx = d / "index.html"
                if not idx.is_file():
                    continue
                try:
                    g = int(d.parent.name.replace("gen", ""))
                except Exception:
                    continue
                gen_dirs.append((g, idx))
            if gen_dirs:
                gen_dirs.sort(key=lambda x: x[0], reverse=True)
                last_gen, html_path = gen_dirs[0]
                product_rel = str(html_path.relative_to(parent))
                product_url = f"/api/evolve/{evo_id}/product/file/{product_rel}"
            else:
                latest = parent / "exports" / "PRODUCT-latest.html"
                if latest.exists():
                    html_path = latest
                    product_rel = "exports/PRODUCT-latest.html"
                    product_url = f"/api/evolve/{evo_id}/export/file/PRODUCT-latest.html"
                    last_gen = s.get("latest_gen")
                else:
                    continue  # no last-gen product
            # If PRODUCT-latest is newer than gen folder, use it for meta (same content, fresher export)
            latest_export = parent / "exports" / "PRODUCT-latest.html"
            meta_path = html_path
            if latest_export.exists() and html_path and html_path != latest_export:
                try:
                    if latest_export.stat().st_mtime >= html_path.stat().st_mtime:
                        meta_path = latest_export
                except Exception:
                    pass
            title, blurb, has_monetization = self._extract_product_meta(
                meta_path or html_path, fallback_title=s.get("topic") or evo_id
            )
            gens_done = int(s.get("generations_done") or 0)
            gens_cfg = s.get("generations_cfg")
            try:
                gens_cfg_i = int(gens_cfg) if gens_cfg is not None else None
            except Exception:
                gens_cfg_i = None
            # last gen index: prefer folder gen number; else gens_done-1
            if last_gen is None and gens_done:
                last_gen = max(0, gens_done - 1)
            gallery.append({
                "id": evo_id,
                "title": title,
                "blurb": blurb,
                "topic": s.get("topic"),
                "goal": s.get("goal"),
                "status": s.get("status"),
                "last_gen": last_gen,
                "latest_gen": last_gen,  # alias for older UI
                "generations_done": gens_done,
                "generations_cfg": gens_cfg_i,
                "best_fitness": s.get("best_fitness"),
                "updated_at": s.get("updated_at"),
                "worker_model": s.get("llm_model"),
                "output_type": s.get("output_type"),
                "has_monetization_board": has_monetization,
                "product_url": product_url,
                "product_path": product_rel,
                "is_last_gen_only": True,
                # Reserved for future Cloudflare / Stripe wiring
                "deploy": {
                    "cloudflare": None,
                    "stripe": None,
                    "status": "not_wired",
                },
            })
            if len(gallery) >= limit:
                break
        return gallery

    @staticmethod
    def _extract_product_meta(html_path: Path, fallback_title: str = "") -> tuple[str, str, bool]:
        """Pull <title> / h1 and a short blurb from product HTML."""
        title = (fallback_title or "Untitled product").strip()[:100]
        blurb = ""
        has_mon = False
        if not html_path or not Path(html_path).exists():
            return title, blurb, has_mon
        try:
            raw = Path(html_path).read_text(encoding="utf-8", errors="replace")[:80_000]
        except Exception:
            return title, blurb, has_mon
        has_mon = "monetization-setup" in raw or "Monetization setup" in raw
        def _is_boilerplate_title(t: str) -> bool:
            return bool(re.match(
                r"(?i)^(gen(eration)?\s*\d+\s*product|untitled|product\s*html)(\s*[·\-|:].*)?$",
                (t or "").strip(),
            ))

        m = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw)
        if m:
            t = re.sub(r"\s+", " ", m.group(1)).strip()
            # strip "Gen N product · id" / "Generation 1 product" boilerplate when possible
            if t and not _is_boilerplate_title(t):
                title = t[:100]
        if title == (fallback_title or "").strip()[:100] or _is_boilerplate_title(title):
            m2 = re.search(r"(?is)<h1[^>]*>(.*?)</h1>", raw)
            if m2:
                t = re.sub(r"<[^>]+>", "", m2.group(1))
                t = re.sub(r"\s+", " ", t).strip()
                if t and not _is_boilerplate_title(t):
                    title = t[:100]
            # Prefer topic / goal over leftover boilerplate
            if _is_boilerplate_title(title) and fallback_title:
                title = str(fallback_title).strip()[:100]
        # first meaningful paragraph
        for m in re.finditer(r"(?is)<p[^>]*>(.*?)</p>", raw):
            t = re.sub(r"<[^>]+>", "", m.group(1))
            t = re.sub(r"\s+", " ", t).strip()
            if len(t) < 40:
                continue
            if "signed artifact" in t.lower() or "product provenance" in t.lower():
                continue
            blurb = t[:180]
            break
        if not blurb and fallback_title:
            blurb = str(fallback_title)[:180]
        return title, blurb, has_mon

    def _resolve_seed_product_dir(
        self,
        parent_root: Path,
        seed_gen: Optional[int] = None,
    ) -> Optional[Path]:
        """Pick genN/product or exports PRODUCT-latest as seed directory/file source."""
        if seed_gen is not None:
            d = parent_root / f"gen{int(seed_gen)}" / "product"
            if (d / "index.html").exists():
                return d
        # highest gen with product
        gens = []
        for d in parent_root.glob("gen*/product"):
            if (d / "index.html").exists():
                try:
                    g = int(d.parent.name.replace("gen", ""))
                    gens.append((g, d))
                except Exception:
                    continue
        if gens:
            gens.sort(key=lambda x: x[0], reverse=True)
            return gens[0][1]
        # fallback: exports only
        latest = parent_root / "exports" / "PRODUCT-latest.html"
        if latest.exists():
            return parent_root / "exports"
        return None

    def _seed_run_from_parent(
        self,
        run: EvolutionRun,
        seed_from: str,
        evolutions_root: Path,
        *,
        seed_gen: Optional[int] = None,
    ) -> None:
        """Copy charter, prompt bank, and latest product HTML into a new run as iteration base."""
        parent_root = Path(evolutions_root) / seed_from
        parent_json = parent_root / "evolution.json"
        if not parent_json.exists():
            raise FileNotFoundError(f"seed evolution not found: {seed_from}")
        data = json.loads(parent_json.read_text(encoding="utf-8"))
        if data.get("charter"):
            run.charter = data["charter"]
        if data.get("prompt_bank"):
            run.prompt_bank = data["prompt_bank"]
        seed_dir = run.root / "seed"
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "parent-id.txt").write_text(seed_from, encoding="utf-8")
        if seed_gen is not None:
            (seed_dir / "parent-gen.txt").write_text(str(seed_gen), encoding="utf-8")
        prod_src = self._resolve_seed_product_dir(parent_root, seed_gen=seed_gen)
        copied: list[str] = []
        if prod_src and prod_src.exists():
            # Copy product tree (or single export file) into seed/product
            dest = seed_dir / "product"
            dest.mkdir(parents=True, exist_ok=True)
            if prod_src.name == "exports":
                src_html = prod_src / "PRODUCT-latest.html"
                if src_html.exists():
                    shutil.copy2(src_html, dest / "index.html")
                    copied.append("index.html")
                md = prod_src / "PRODUCT-latest.md"
                if md.exists():
                    shutil.copy2(md, dest / "PRODUCT.md")
                    copied.append("PRODUCT.md")
            else:
                for f in prod_src.rglob("*"):
                    if not f.is_file():
                        continue
                    rel = f.relative_to(prod_src)
                    if any(p.startswith(".") for p in rel.parts):
                        continue
                    out = dest / rel
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, out)
                    copied.append(rel.as_posix())
        # Also stash path for bootstrap inject
        run._seed_product_dir = str(seed_dir / "product") if copied else None  # type: ignore[attr-defined]
        run._seed_from = seed_from  # type: ignore[attr-defined]
        run._seed_gen = seed_gen  # type: ignore[attr-defined]
        run.log_event(
            "status",
            f"Seeded from {seed_from}"
            + (f" gen{seed_gen}" if seed_gen is not None else " (latest product)")
            + f" · {len(copied)} files",
            details={"parent_id": seed_from, "seed_gen": seed_gen, "files": copied[:40]},
        )
        run._save()

    def _inject_seed_product_into_population(self, run: EvolutionRun, population: list) -> None:
        """After gen0 scaffold, merge parent product HTML into candidates so they improve it."""
        seed_path = getattr(run, "_seed_product_dir", None)
        if not seed_path:
            return
        seed = Path(seed_path)
        if not seed.exists():
            return
        from lib import evolution_product as eprod
        from lib import evolution_git as egit
        injected = 0
        for cand in population:
            if not cand.path:
                continue
            try:
                got = eprod.soft_inject_product(seed, Path(cand.path))
                # Always ensure index.html from seed if missing
                idx = Path(cand.path) / "index.html"
                seed_idx = seed / "index.html"
                if seed_idx.exists() and (not idx.exists() or idx.stat().st_size < 200):
                    shutil.copy2(seed_idx, idx)
                    got = list(got or []) + ["index.html"]
                if got:
                    injected += 1
                    cand.meta.setdefault("build", {})
                    files = list((cand.meta.get("build") or {}).get("files") or [])
                    for f in got:
                        if f not in files:
                            files.append(f)
                    cand.meta["build"]["files"] = files
                    cand.meta["seed_from"] = getattr(run, "_seed_from", None)
                    if run.cfg.use_git:
                        egit.commit(Path(cand.path), f"seed product from parent {getattr(run, '_seed_from', '')}", allow_empty=True)
            except Exception:
                continue
        if injected:
            run.log_event(
                "status",
                f"Injected seed product into {injected}/{len(population)} candidates",
                generation=0,
            )

    @staticmethod
    def config_from_disk(cfg_d: Optional[dict]) -> EvolutionConfig:
        cfg_d = cfg_d or {}
        return EvolutionConfig(
            goal=str(cfg_d.get("goal") or "resumed evolution"),
            output_type=str(cfg_d.get("output_type") or "auto"),
            name=cfg_d.get("name"),
            population_size=int(cfg_d.get("population_size") or 4),
            generations=int(cfg_d.get("generations") or 3),
            mutation_rate=float(cfg_d.get("mutation_rate") if cfg_d.get("mutation_rate") is not None else 0.35),
            attrition_rate=float(cfg_d.get("attrition_rate") if cfg_d.get("attrition_rate") is not None else 0.5),
            innovation_rate=float(cfg_d.get("innovation_rate") if cfg_d.get("innovation_rate") is not None else 0.4),
            benchmark_weights=dict(cfg_d.get("benchmark_weights") or {}),
            budget_usd=cfg_d.get("budget_usd"),
            providers=list(cfg_d.get("providers") or []),
            mcp_servers=list(cfg_d.get("mcp_servers") or []),
            deployment_target=cfg_d.get("deployment_target"),
            run_tests=bool(cfg_d.get("run_tests")),
            promote_best=bool(cfg_d.get("promote_best", True)),
            llm_model=str(cfg_d.get("llm_model") or EVOLUTION_LLM_MODEL),
            build_software=bool(cfg_d.get("build_software", True)),
            build_depth=str(cfg_d.get("build_depth") or "implement"),
            planner_id=str(cfg_d.get("planner_id") or "cerebras:gemma-4-31b"),
            goal_brief=str(cfg_d.get("goal_brief") or cfg_d.get("goal") or ""),
            decision_maker_id=str(cfg_d.get("decision_maker_id") or "cerebras:zai-glm-4.7"),
            produce_product=bool(cfg_d.get("produce_product", True)),
            use_git=bool(cfg_d.get("use_git", True)),
            cooperation=bool(cfg_d.get("cooperation", True)),
            director_fitness_blend=float(
                cfg_d.get("director_fitness_blend") if cfg_d.get("director_fitness_blend") is not None else 0.45
            ),
            research_enabled=bool(cfg_d.get("research_enabled", True)),
            diverse_workers=bool(cfg_d.get("diverse_workers", True)),
            include_low_throughput_workers=bool(cfg_d.get("include_low_throughput_workers", True)),
            include_openrouter_workers=bool(cfg_d.get("include_openrouter_workers", True)),
            worker_models=list(cfg_d.get("worker_models") or []),
        )

    def _candidate_from_dict(self, d: dict, run: EvolutionRun) -> Optional[Candidate]:
        if not d or not d.get("id"):
            return None
        cid = str(d["id"])
        path = Path(d["path"]) if d.get("path") else None
        if path and not path.is_absolute():
            path = run.root / path
        if (not path or not path.exists()) and (run.root / "gen0" / cid).exists():
            path = run.root / "gen0" / cid
        # Search gen folders
        if not path or not path.exists():
            for gdir in sorted(run.root.glob("gen*")):
                if not gdir.is_dir():
                    continue
                cand = gdir / cid
                if cand.is_dir():
                    path = cand
                    break
        genome = {}
        meta = dict(d.get("meta") or {})
        if path and (path / "state.json").exists():
            try:
                genome = json.loads((path / "state.json").read_text(encoding="utf-8"))
            except Exception:
                genome = {}
        if not genome:
            genome = {
                "cells": d.get("cells") or [],
                "description": d.get("description") or "",
                "template": d.get("template") or "blank",
                "order": d.get("order") or [],
            }
        if d.get("build") and not meta.get("build"):
            meta["build"] = d.get("build")
        if d.get("prompt_variant") and not meta.get("prompt_variant"):
            meta["prompt_variant"] = d.get("prompt_variant")
        if d.get("lineage") and not meta.get("lineage"):
            meta["lineage"] = d.get("lineage")
        cand = Candidate(
            id=cid,
            generation=int(d.get("generation") or 0),
            genome=genome,
            meta=meta,
            scores=dict(d.get("scores") or {}),
            fitness=float(d.get("fitness") or 0),
            brilliant=bool(d.get("brilliant")),
            rationale=str(d.get("rationale") or ""),
            path=path,
        )
        return cand

    def _load_population_from_disk(self, run: EvolutionRun, gen: int) -> list[Candidate]:
        """Rebuild Candidate objects from gen{N}/ folders (and optional summary scores)."""
        gdir = run.root / f"gen{gen}"
        score_by_id: dict[str, dict] = {}
        for g in run.generations:
            if g.generation == gen:
                for cd in g.candidates or []:
                    if isinstance(cd, dict) and cd.get("id"):
                        score_by_id[str(cd["id"])] = cd
                break
        pop: list[Candidate] = []
        if gdir.exists():
            for d in sorted(gdir.iterdir()):
                if not d.is_dir() or d.name in ("product",):
                    continue
                if d.name.startswith("."):
                    continue
                blob = {"id": d.name, "path": str(d), "generation": gen}
                if d.name in score_by_id:
                    blob = {**score_by_id[d.name], **blob, "path": str(d)}
                c = self._candidate_from_dict(blob, run)
                if c:
                    pop.append(c)
        if not pop and score_by_id:
            for cid, blob in score_by_id.items():
                c = self._candidate_from_dict({**blob, "id": cid}, run)
                if c:
                    pop.append(c)
        # Prefer survivors order if summary has them
        for g in run.generations:
            if g.generation == gen and g.survivors_ids:
                order = {sid: i for i, sid in enumerate(g.survivors_ids)}
                pop.sort(key=lambda c: (0 if c.id in order else 1, order.get(c.id, 999), -(c.fitness or 0)))
                break
        else:
            pop.sort(key=lambda c: c.fitness, reverse=True)
        return pop

    def _hydrate_run_from_data(self, run: EvolutionRun, data: dict) -> None:
        run.status = str(data.get("status") or "stopped")
        run.current_generation = int(data.get("current_generation") or 0)
        run.error = data.get("error")
        run.stop_reason = data.get("stop_reason")
        run.events = list(data.get("events") or [])
        run.llm_calls = list(data.get("llm_calls") or [])
        run.promoted_project_id = data.get("promoted_project_id")
        run.llm_model = data.get("llm_model") or run.llm_model
        run.charter = data.get("charter") or {}
        run.prompt_bank = data.get("prompt_bank") or run.prompt_bank
        run.created_at = data.get("created_at") or run.created_at
        run.updated_at = data.get("updated_at") or run.updated_at
        # Generation summaries
        gens: list[GenerationSummary] = []
        for g in data.get("generations") or []:
            if not isinstance(g, dict):
                continue
            gens.append(GenerationSummary(
                generation=int(g.get("generation") or 0),
                best_fitness=float(g.get("best_fitness") or 0),
                avg_fitness=float(g.get("avg_fitness") or 0),
                survivors=int(g.get("survivors") or 0),
                population=int(g.get("population") or 0),
                brilliant=list(g.get("brilliant") or []),
                brilliant_count=int(g.get("brilliant_count") or len(g.get("brilliant") or [])),
                survivors_ids=list(g.get("survivors_ids") or []),
                eliminated_ids=list(g.get("eliminated_ids") or []),
                candidates=list(g.get("candidates") or []),
                events_count=int(g.get("events_count") or 0),
            ))
        run.generations = gens
        # Candidates
        cands: list[Candidate] = []
        for cd in data.get("candidates") or []:
            if isinstance(cd, dict):
                c = self._candidate_from_dict(cd, run)
                if c:
                    cands.append(c)
        if not cands:
            # try latest completed gen, then gen0
            last_gen = max((g.generation for g in gens), default=0)
            cands = self._load_population_from_disk(run, last_gen) if last_gen else []
            if not cands:
                cands = self._load_population_from_disk(run, 0)
        run.candidates = cands
        best = None
        if data.get("best") and isinstance(data["best"], dict):
            best = self._candidate_from_dict(data["best"], run)
        if not best and cands:
            best = sorted(cands, key=lambda c: c.fitness, reverse=True)[0]
        run.best = best

    def resume(self, evo_id: str, evolutions_root: Path) -> dict[str, Any]:
        """Resume a stopped/failed/orphaned evolution from disk progress.

        Continues from the next generation after the last fully completed
        GenerationSummary. Gen0 create is skipped if gen0 candidates exist.
        """
        live = self.get_run(evo_id)
        if live and live.status in ("running", "queued", "stopping", "starting"):
            return {
                "ok": False,
                "evolution_id": evo_id,
                "error": f"run is currently {live.status}; stop it first or wait",
            }

        root = Path(evolutions_root) / evo_id
        path = root / "evolution.json"
        if not path.exists():
            return {"ok": False, "evolution_id": evo_id, "error": "evolution not found on disk"}

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            return {"ok": False, "evolution_id": evo_id, "error": f"could not read evolution.json: {e}"}

        st = (data.get("status") or "").lower()
        if st == "completed":
            # Allow resume only when generations target was raised (continue_generations)
            cfg_peek = data.get("config") or {}
            gens_done = len(data.get("generations") or [])
            target = int(cfg_peek.get("generations") or 0)
            if target <= gens_done:
                return {
                    "ok": False,
                    "evolution_id": evo_id,
                    "status": "completed",
                    "error": "run already completed — use Continue (+gens) or Clone flavor",
                }
        if st in ("running", "queued", "stopping", "starting"):
            # Orphaned on disk (no live worker) — treat as resumable
            pass

        cfg = self.config_from_disk(data.get("config") or {})
        # Reuse existing EvolutionRun object if present but finished
        if live and live.status in ("stopped", "failed"):
            run = live
            run.cfg = cfg
            run.llm_model = cfg.llm_model or run.llm_model
            self._hydrate_run_from_data(run, data)
        else:
            run = EvolutionRun(evo_id, cfg, root)
            self._hydrate_run_from_data(run, data)

        run.clear_stop()
        run.error = None

        gens_done = len(run.generations)
        # Next gen to execute (1-based loop). If no gen summaries yet, still may have gen0.
        has_gen0 = any((root / "gen0").glob("*/state.json")) or any(
            (root / "gen0").iterdir()
        ) if (root / "gen0").exists() else False
        # More precise gen0 check
        if (root / "gen0").exists():
            has_gen0 = any(
                p.is_dir() and p.name != "product" and (p / "state.json").exists()
                for p in (root / "gen0").iterdir()
            )
        resume_from_gen = gens_done + 1  # after last completed summary
        if resume_from_gen < 1:
            resume_from_gen = 1

        if gens_done >= cfg.generations:
            # All gens recorded but status not completed — finalize
            run.update(status="completed")
            run.log_event("status", "Resume: all generations already complete — marked completed")
            return {
                "ok": True,
                "evolution_id": evo_id,
                "status": "completed",
                "already_finished": True,
                "message": "all generations already present; marked completed",
                "saved": True,
                "path": str(root),
            }

        # Load population to continue with
        if gens_done > 0:
            population = self._load_population_from_disk(run, gens_done)
            # Prefer survivors only when available
            for g in run.generations:
                if g.generation == gens_done and g.survivors_ids:
                    by_id = {c.id: c for c in population}
                    survivors = [by_id[sid] for sid in g.survivors_ids if sid in by_id]
                    if survivors:
                        population = survivors
                    break
        elif has_gen0:
            population = self._load_population_from_disk(run, 0)
        else:
            population = []

        run.candidates = population
        if population:
            try:
                run.best = sorted(population, key=lambda c: c.fitness, reverse=True)[0]
            except Exception:
                run.best = population[0]

        run._resume_from_gen = resume_from_gen  # type: ignore[attr-defined]
        run._skip_bootstrap = bool(has_gen0 or gens_done > 0)  # type: ignore[attr-defined]
        run._resume_population = population  # type: ignore[attr-defined]

        with self._runs_lock:
            self.runs[evo_id] = run

        run.log_event(
            "status",
            f"Resume requested · skip_bootstrap={bool(has_gen0 or gens_done > 0)} · "
            f"from gen {resume_from_gen}/{cfg.generations} · "
            f"{len(population)} candidates loaded · prior gens done={gens_done}",
            generation=run.current_generation,
            details={
                "resume_from_gen": resume_from_gen,
                "gens_done": gens_done,
                "has_gen0": has_gen0,
                "candidate_ids": [c.id for c in population],
            },
        )
        run.update(status="queued")
        threading.Thread(target=self._run_loop, args=(run,), daemon=True).start()
        return {
            "ok": True,
            "evolution_id": evo_id,
            "status": "queued",
            "resumed": True,
            "resume_from_gen": resume_from_gen,
            "gens_done": gens_done,
            "candidates_loaded": len(population),
            "message": f"resuming from generation {resume_from_gen}",
            "saved": True,
            "path": str(root),
        }

    def continue_generations(
        self,
        evo_id: str,
        evolutions_root: Path,
        *,
        extra_generations: int = 2,
        goal_addendum: str = "",
    ) -> dict[str, Any]:
        """Add more generations to an existing run (including completed ones).

        Bumps config.generations and resumes the worker from the next gen so the
        deployer can push the product further without starting from scratch.
        """
        extra = max(1, min(200, int(extra_generations or 2)))
        root = Path(evolutions_root) / evo_id
        path = root / "evolution.json"
        if not path.exists():
            return {"ok": False, "error": "evolution not found", "evolution_id": evo_id}

        live = self.get_run(evo_id)
        if live and live.status in ("running", "queued", "stopping", "starting"):
            return {
                "ok": False,
                "evolution_id": evo_id,
                "error": f"run is {live.status}; stop it before continuing",
            }

        data = json.loads(path.read_text(encoding="utf-8"))
        cfg_d = dict(data.get("config") or {})
        gens_done = len(data.get("generations") or [])
        current_target = int(cfg_d.get("generations") or gens_done or 3)
        new_target = max(current_target, gens_done) + extra
        cfg_d["generations"] = new_target
        if goal_addendum and goal_addendum.strip():
            base = (cfg_d.get("goal") or "").strip()
            add = goal_addendum.strip()
            cfg_d["goal"] = f"{base}\n\n[Continue focus] {add}"
            brief = (cfg_d.get("goal_brief") or base).strip()
            cfg_d["goal_brief"] = f"{brief}\n\n## Continue focus\n{add}"
        data["config"] = cfg_d
        data["status"] = "stopped"  # make resume accept it
        data["error"] = None
        data["updated_at"] = utcnow()
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        result = self.resume(evo_id, evolutions_root)
        if result.get("ok"):
            result["continued"] = True
            result["extra_generations"] = extra
            result["new_generation_target"] = new_target
            result["message"] = f"continuing for {extra} more gen(s) → target gen {new_target}"
        return result

    def clone_flavor(
        self,
        evo_id: str,
        evolutions_root: Path,
        *,
        flavor: str,
        generations: int = 3,
        population_size: Optional[int] = None,
        name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Fork a completed/stopped run into a new evolution with a flavor twist.

        Copies charter + prompt bank + best candidate seed so more compute
        explores a different monetizable flavor of the same product line.
        """
        flavor = (flavor or "").strip()
        if not flavor:
            return {"ok": False, "error": "flavor text required"}
        root = Path(evolutions_root) / evo_id
        path = root / "evolution.json"
        if not path.exists():
            return {"ok": False, "error": "source evolution not found", "evolution_id": evo_id}
        data = json.loads(path.read_text(encoding="utf-8"))
        cfg_d = dict(data.get("config") or {})
        base_goal = (cfg_d.get("goal") or "").strip()
        new_goal = (
            f"{base_goal}\n\n"
            f"[FLAVOR BRANCH] Explore this variant while keeping the same product line & monetization:\n"
            f"{flavor}\n"
            f"Ship a distinct but related product surface the deployer can sell or cross-link."
        )
        cfg = self.config_from_disk({
            **cfg_d,
            "goal": new_goal,
            "goal_brief": new_goal,
            "name": name or f"{(cfg_d.get('name') or evo_id)[:40]}-flavor",
            "generations": max(1, min(200, int(generations or 3))),
            "population_size": int(population_size or cfg_d.get("population_size") or 4),
        })
        run = self.start(cfg, evolutions_root)
        # Seed charter / prompt bank from parent
        try:
            if data.get("charter"):
                run.charter = data["charter"]
            if data.get("prompt_bank"):
                run.prompt_bank = data["prompt_bank"]
            # Copy latest product HTML as starting artifact reference
            src_prod = root / "exports" / "PRODUCT-latest.html"
            if src_prod.exists():
                seed_dir = run.root / "seed"
                seed_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_prod, seed_dir / "parent-product.html")
                (seed_dir / "parent-id.txt").write_text(evo_id, encoding="utf-8")
                (seed_dir / "flavor.txt").write_text(flavor, encoding="utf-8")
            run.log_event(
                "status",
                f"Cloned flavor from {evo_id}: {flavor[:120]}",
                details={"parent_id": evo_id, "flavor": flavor[:300]},
            )
            run._save()
        except Exception as e:
            run.log_event("error", f"clone seed warning: {e}")
        return {
            "ok": True,
            "evolution_id": run.id,
            "parent_id": evo_id,
            "flavor": flavor,
            "status": run.status,
            "path": str(run.root),
            "message": "flavor clone started",
        }

    def stop(self, evo_id: str, evolutions_root: Optional[Path] = None) -> dict[str, Any]:
        """Request a cooperative stop. Always keeps on-disk progress.

        If the worker is live, it exits at the next safe checkpoint (between
        candidates / generations / LLM calls) and finalizes status=stopped.
        If only disk state remains (zombie running), mark stopped immediately.
        """
        run = self.get_run(evo_id)
        if run:
            if run.status in ("completed", "failed", "stopped"):
                return {
                    "ok": True,
                    "evolution_id": evo_id,
                    "status": run.status,
                    "already_finished": True,
                    "message": f"run already {run.status}",
                    "saved": True,
                    "path": str(run.root),
                }
            run.request_stop("user requested stop")
            run.log_event(
                "status",
                "Stop requested — finishing current step then saving progress",
                generation=run.current_generation,
            )
            return {
                "ok": True,
                "evolution_id": evo_id,
                "status": run.status,
                "stop_requested": True,
                "message": "stop requested; progress will be saved",
                "saved": True,
                "path": str(run.root),
                "current_generation": run.current_generation,
                "llm_calls": len(run.llm_calls),
                "best_id": run.best.id if run.best else None,
            }

        # Disk-only (orphaned / restarted): mark stopped without a live worker
        if evolutions_root:
            path = Path(evolutions_root) / evo_id / "evolution.json"
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception as e:
                    return {"ok": False, "evolution_id": evo_id, "error": f"could not read run: {e}"}
                st = (data.get("status") or "").lower()
                if st in ("completed", "failed", "stopped"):
                    return {
                        "ok": True,
                        "evolution_id": evo_id,
                        "status": st,
                        "already_finished": True,
                        "saved": True,
                        "path": str(path.parent),
                    }
                data["status"] = "stopped"
                data["stop_reason"] = "user requested stop (worker not active — marked on disk)"
                data["updated_at"] = utcnow()
                events = data.get("events") or []
                events.append({
                    "ts": utcnow(),
                    "generation": data.get("current_generation") or 0,
                    "type": "status",
                    "message": "Stopped (no live worker) — prior progress kept on disk",
                })
                data["events"] = events
                path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                return {
                    "ok": True,
                    "evolution_id": evo_id,
                    "status": "stopped",
                    "stop_requested": True,
                    "disk_only": True,
                    "message": "no live worker; marked stopped on disk",
                    "saved": True,
                    "path": str(path.parent),
                }
        return {"ok": False, "evolution_id": evo_id, "error": "evolution run not found"}

    def _finalize_stopped(self, run: EvolutionRun, population: Optional[list] = None) -> None:
        """Persist partial progress and set status=stopped (never lose work)."""
        pop: list = list(population or [])
        if not pop and run.candidates:
            pop = list(run.candidates)
        best = None
        if pop:
            try:
                pop = sorted(pop, key=lambda c: getattr(c, "fitness", 0) or 0, reverse=True)
                best = pop[0]
            except Exception:
                best = pop[0] if pop else None
        if best is None:
            best = run.best
        reason = run.stop_reason or "user requested stop"
        run.stop_reason = reason
        run.update(candidates=pop if pop else None, best=best, status="stopped")
        # Ensure status sticks even if update skipped empty candidates
        with run.lock:
            run.status = "stopped"
            run.updated_at = utcnow()
            run._save()
        try:
            self._write_llm_transcript(run)
        except Exception:
            pass
        best_id = getattr(best, "id", None) if best else None
        best_fit = getattr(best, "fitness", None) if best else None
        run.log_event(
            "status",
            f"Stopped — progress saved · gen {run.current_generation} · "
            f"{len(run.llm_calls)} llm calls · best={best_id or '—'} "
            f"(fit={round(best_fit, 2) if best_fit is not None else '—'}) · {reason}",
            generation=run.current_generation,
            candidate_id=best_id,
            details={
                "stop_reason": reason,
                "generations_done": len(run.generations),
                "llm_calls": len(run.llm_calls),
                "candidates": len(pop),
                "best_id": best_id,
                "best_fitness": best_fit,
                "charter_roles": (run.charter or {}).get("roles"),
            },
        )

    def _maybe_plan_goal(self, run: EvolutionRun) -> None:
        """Optional planner step: expand user goal into a brief for Cerebras workers."""
        cfg = run.cfg
        if not cfg.planner_id or cfg.planner_id == "none":
            cfg.goal_brief = cfg.goal
            return
        run.check_stop()
        try:
            from lib import planner as planmod
            run.log_event("plan", f"Expanding goal via planner {cfg.planner_id}", generation=0)
            result = planmod.expand_goal(
                cfg.goal,
                planner_id=cfg.planner_id,
                output_type=cfg.output_type,
                build_software=cfg.build_software,
                run_id=run.id,
            )
            cfg.goal_brief = (result.get("brief") or cfg.goal).strip()
            run.log_event(
                "plan",
                f"Planner {'ok' if result.get('ok') else 'failed'} · {cfg.planner_id}"
                + (f" · {result.get('error')}" if result.get("error") else ""),
                generation=0,
                model=result.get("model") or None,
                details={
                    "ok": result.get("ok"),
                    "harness": result.get("harness"),
                    "duration_secs": result.get("duration_secs"),
                    "brief_chars": len(cfg.goal_brief or ""),
                },
            )
            # Persist brief next to evolution.json
            try:
                (run.root / "goal-brief.md").write_text(
                    f"# Goal brief\n\nPlanner: `{cfg.planner_id}`\n\n## User goal\n\n{cfg.goal}\n\n## Expanded brief\n\n{cfg.goal_brief}\n",
                    encoding="utf-8",
                )
            except Exception:
                pass
            run._save()
        except Exception as e:
            cfg.goal_brief = cfg.goal
            run.log_event("error", f"Planner error (using raw goal): {e}", generation=0)

    def _maybe_research_goal(self, run: EvolutionRun) -> None:
        """Optional web research harness — makes workers smarter without full Pi browser."""
        cfg = run.cfg
        if not getattr(cfg, "research_enabled", True):
            return
        try:
            from lib import research_harness as rh
            topic = (cfg.goal_brief or cfg.goal or "")[:1500]
            research = rh.research_topic(
                topic,
                run_id=run.id,
                model=cfg.llm_model or EVOLUTION_LLM_MODEL,
            )
            if research.get("ok") and research.get("brief"):
                run.meta = getattr(run, "meta", None) or {}
                # stash on run object (not always in _to_dict — also write file)
                run._research = research  # type: ignore[attr-defined]
                (run.root / "research-brief.md").write_text(
                    research["brief"] + "\n\n## Sources\n"
                    + "\n".join(f"- {s.get('url')}" for s in (research.get("sources") or [])),
                    encoding="utf-8",
                )
                # Append condensed research into goal_brief for workers
                extra = rh.format_brief_for_prompt(research, max_chars=2800)
                if extra and extra not in (cfg.goal_brief or ""):
                    cfg.goal_brief = (cfg.goal_brief or cfg.goal) + "\n\n" + extra
                run.log_event(
                    "plan",
                    f"Research harness complete · {len(research.get('sources') or [])} sources · model={research.get('model')}",
                    generation=0,
                    details={"harness": research.get("harness"), "queries": research.get("queries")},
                )
        except Exception as e:
            run.log_event("error", f"Research harness failed: {e}", generation=0)

    def _maintainer_learnings_for_product(self, run: EvolutionRun) -> tuple[str, list[str]]:
        """Pull recent maintainer memories/learnings to improve product HTML."""
        snippets: list[str] = []
        try:
            from lib.maintainer import get_maintainer
            # DATA_DIR is parent of evolutions root
            data_dir = run.root.parent.parent if run.root else None
            if not data_dir:
                return "", []
            m = get_maintainer(data_dir, run.root.parent)
            for mem in m.list_memories(limit=30):
                kind = mem.get("kind") or ""
                if kind in ("failed", "prompt_smell", "product_next", "worked", "lesson", "constraint"):
                    snippets.append(f"[{kind}] {mem.get('content')}")
            for L in m.list_learnings(limit=8):
                for w in (L.get("what_failed") or [])[:2]:
                    snippets.append(f"[learning-fail] {w}")
                for w in (L.get("product_next_steps") or [])[:2]:
                    snippets.append(f"[product-next] {w}")
            snippets = snippets[-12:]
            text = "\n".join(f"- {s}" for s in snippets)
            return text, snippets
        except Exception:
            return "", []

    def get_run(self, evo_id: str) -> Optional[EvolutionRun]:
        with self._runs_lock:
            return self.runs.get(evo_id)

    @staticmethod
    def _run_summary(data: dict, evolutions_root: Optional[Path] = None) -> dict:
        """Compact history entry for the Evolve sidebar (always persisted on disk)."""
        cfg = data.get("config") or {}
        best = data.get("best") or {}
        gens = data.get("generations") or []
        eid = data.get("id") or ""
        has_html = False
        product_url = None
        exports_dir = None
        if evolutions_root and eid:
            root = Path(evolutions_root) / eid
            latest = root / "exports" / "PRODUCT-latest.html"
            if latest.is_file() and latest.stat().st_size > 50:
                has_html = True
                product_url = f"/api/evolve/{eid}/export/file/PRODUCT-latest.html"
                exports_dir = str(root / "exports")
            else:
                # any genN/product/index.html or exports gen*-product.html
                for cand in sorted((root / "exports").glob("*.html")) if (root / "exports").is_dir() else []:
                    if cand.stat().st_size > 50:
                        has_html = True
                        product_url = f"/api/evolve/{eid}/export/file/{cand.name}"
                        exports_dir = str(root / "exports")
                        break
                if not has_html:
                    for cand in root.glob("gen*/product/index.html"):
                        if cand.is_file() and cand.stat().st_size > 50:
                            has_html = True
                            rel = cand.relative_to(root).as_posix()
                            product_url = f"/api/evolve/{eid}/product/file/{rel}"
                            break
        return {
            "id": eid,
            "status": data.get("status"),
            "goal": cfg.get("goal") or "",
            "name": cfg.get("name"),
            "llm_model": data.get("llm_model") or cfg.get("llm_model") or EVOLUTION_LLM_MODEL,
            "providers": cfg.get("providers") or [],
            "output_type": cfg.get("output_type"),
            "population_size": cfg.get("population_size"),
            "generations_cfg": cfg.get("generations"),
            "current_generation": data.get("current_generation"),
            "generations_done": len(gens),
            "best_fitness": best.get("fitness"),
            "best_id": best.get("id"),
            "brilliant": best.get("brilliant"),
            "promoted_project_id": data.get("promoted_project_id"),
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
            "error": data.get("error"),
            "stop_reason": data.get("stop_reason"),
            "llm_calls": len(data.get("llm_calls") or []),
            "cell_count": best.get("cell_count") or len(best.get("cells") or []),
            "has_product_html": has_html,
            "product_url": product_url,
            "exports_dir": exports_dir,
            "persisted": True,
            "disk_path": str(Path(evolutions_root) / eid) if evolutions_root and eid else None,
        }

    def list_runs(self, evolutions_root: Optional[Path] = None, full: bool = False) -> list[dict]:
        """List all evolutions. By default merges on-disk history (always saved) with in-memory runs."""
        by_id: dict[str, dict] = {}
        root = Path(evolutions_root) if evolutions_root else None
        if root and root.exists():
            for p in root.glob("*/evolution.json"):
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    eid = data.get("id") or p.parent.name
                    data["id"] = eid
                    by_id[eid] = data if full else self._run_summary(data, root)
                except Exception:
                    continue
        with self._runs_lock:
            for r in self.runs.values():
                data = r._to_dict()
                by_id[r.id] = data if full else self._run_summary(data, root)
        runs = list(by_id.values())
        runs.sort(key=lambda d: d.get("updated_at") or d.get("created_at") or "", reverse=True)
        return runs

    def load_disk_dict(self, evo_id: str, evolutions_root: Path) -> Optional[dict]:
        path = Path(evolutions_root) / evo_id / "evolution.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data["id"] = data.get("id") or evo_id
            return data
        except Exception:
            return None

    def _run_loop(self, run: EvolutionRun) -> None:
        population: list = list(getattr(run, "_resume_population", None) or run.candidates or [])
        skip_bootstrap = bool(getattr(run, "_skip_bootstrap", False))
        resume_from_gen = int(getattr(run, "_resume_from_gen", 1) or 1)
        try:
            run.clear_stop()
            run.update(status="running")
            cfg = run.cfg
            if skip_bootstrap:
                run.log_event(
                    "status",
                    f"Evolution resumed — continuing from gen {resume_from_gen}/{cfg.generations}",
                    generation=max(0, resume_from_gen - 1),
                )
                if not population:
                    # try load gen0 or last gen
                    population = self._load_population_from_disk(run, max(0, resume_from_gen - 1))
                if not population:
                    # cannot continue without individuals — fall back to full bootstrap
                    run.log_event("status", "Resume: no candidates on disk — re-running bootstrap", generation=0)
                    skip_bootstrap = False
                    resume_from_gen = 1
                else:
                    run.update(candidates=population)
                    if not run.charter:
                        self._ensure_charter(run, population)
                    if population:
                        try:
                            ranked = sorted(population, key=lambda c: c.fitness, reverse=True)
                            run.update(best=ranked[0])
                        except Exception:
                            pass
                    # If gen0 never got a product and we're resuming into gen1, optional refresh skipped
            if not skip_bootstrap:
                run.log_event("status", "Evolution run started", generation=0)
                run.check_stop()
                self._maybe_plan_goal(run)
                run.check_stop()
                self._maybe_research_goal(run)
                run.check_stop()
                population = self._create_initial_population(run)
                run.check_stop()
                # If started from prior HTML/product, inject it so gen0 improves that artifact
                try:
                    self._inject_seed_product_into_population(run, population)
                except Exception as e:
                    run.log_event("error", f"Seed inject failed: {e}", generation=0)
                run.check_stop()
                # Freeze architecture charter from the strongest gen0 design after first scaffold/eval setup
                # (charter is refined after gen 1 scores if still empty)
                self._ensure_charter(run, population)
                run.update(candidates=population)
                if population:
                    try:
                        ranked = sorted(population, key=lambda c: c.fitness, reverse=True)
                        run.update(best=ranked[0])
                    except Exception:
                        pass
                # Gen0 product board (after scaffolds) — same pipeline as later gens
                if cfg.produce_product and population:
                    try:
                        run.check_stop()
                        # Pre-seed gen0 product workspace from parent HTML when iterating
                        seed_path = getattr(run, "_seed_product_dir", None)
                        if seed_path and Path(seed_path).exists():
                            from lib import evolution_product as eprod
                            from lib import evolution_git as egit
                            pdir = eprod.product_dir(run.root, 0)
                            eprod.seed_product_from_champion(Path(seed_path), pdir)
                            if cfg.use_git:
                                egit.init_repo(pdir)
                                egit.commit(pdir, f"seed product from {getattr(run, '_seed_from', 'parent')}", allow_empty=True)
                        d0 = self._director_review(run, population, 0)
                        self._apply_director_scores(population, d0, cfg.director_fitness_blend)
                        self._cooperation_and_product(run, population, d0, 0)
                        run.update(best=population[0], candidates=population)
                    except EvolutionStopped:
                        raise
                    except Exception as e:
                        run.log_event("product", f"Gen0 product phase failed: {e}", generation=0)

            for gen in range(max(1, resume_from_gen), cfg.generations + 1):
                run.check_stop()
                run.update(generation=gen)
                run.log_event(
                    "status",
                    f"Generation {gen} started — building on charter + parent artifacts"
                    + (" (architecture + software)" if cfg.build_software else " (architecture only)"),
                    details={"charter_roles": (run.charter or {}).get("roles"), "prompt_bank": {
                        k: (v[:80] + "…") if isinstance(v, str) and len(v) > 80 else v
                        for k, v in (run.prompt_bank or {}).items() if k != "history"
                    }},
                )
                # Align genomes to charter before building (soft repair, not full rewrite)
                for cand in population:
                    run.check_stop()
                    self._align_genome_to_charter(cand, run)

                # Build / improve software before scoring (so fitness sees real code)
                if cfg.build_software:
                    for cand in population:
                        run.check_stop()
                        built = (cand.meta.get("build") or {}).get("ok")
                        last_built_gen = (cand.meta.get("build") or {}).get("generation")
                        if last_built_gen == gen and built:
                            continue
                        has_files = bool(self._read_candidate_sources(Path(cand.path)) if cand.path else {})
                        mode = "improve" if (built or has_files) else "scaffold"
                        try:
                            self._build_software(cand, run, mode=mode)
                        except EvolutionStopped:
                            raise
                        except Exception as e:
                            run.log_event("error", f"Build failed: {e}", candidate_id=cand.id, generation=gen)

                # evaluate (re-score whenever we built/improved software this generation)
                for cand in population:
                    run.check_stop()
                    built_this_gen = (cand.meta.get("build") or {}).get("generation") == gen
                    if cand.scores and not built_this_gen:
                        # elite survivor without rebuild: still re-score continuity lightly every gen
                        if run.charter:
                            cont = self._score_continuity(cand, run)
                            cand.meta["continuity"] = cont
                            # blend continuity into fitness without full re-eval
                            if cand.scores:
                                cand.scores["continuity"] = cont
                                weights = {**cfg.benchmark_weights, "continuity": 0.12}
                                tw = sum(weights.values()) or 1
                                cand.fitness = sum(
                                    max(0, min(100, float(cand.scores.get(k, 50)))) * (w / tw)
                                    for k, w in weights.items()
                                )
                        continue
                    if built_this_gen:
                        cand.scores = {}
                    self._evaluate(cand, run)

                # Decision maker: rank, champion, cooperation brief (1 call/gen)
                product_meta: dict = {}
                director: dict = {}
                if cfg.produce_product or (cfg.decision_maker_id and cfg.decision_maker_id != "none"):
                    run.check_stop()
                    director = self._director_review(run, population, gen)
                    self._apply_director_scores(
                        population, director, getattr(cfg, "director_fitness_blend", 0.45)
                    )
                else:
                    population.sort(key=lambda c: c.fitness, reverse=True)

                # Shared generational product (HTML + merges) under genN/product/
                if cfg.produce_product:
                    run.check_stop()
                    try:
                        product_meta = self._cooperation_and_product(run, population, director or {
                            "champion_id": population[0].id if population else None,
                            "product_direction": cfg.goal,
                            "must_have": [],
                            "merge_plan": [],
                        }, gen)
                    except EvolutionStopped:
                        raise
                    except Exception as e:
                        run.log_event("product", f"Product phase failed: {e}", generation=gen)

                # sort already done by director; reaffirm
                if not director:
                    population.sort(key=lambda c: c.fitness, reverse=True)

                # attrition after director + product so cooperation saw full pop
                survivors = self._attrition(population, cfg.attrition_rate)

                eliminated = [c for c in population if c not in survivors]
                run.log_event(
                    "attrition",
                    f"Kept {len(survivors)} survivors, eliminated {len(eliminated)}",
                    details={"survivors": [c.id for c in survivors], "eliminated": [c.id for c in eliminated]},
                )

                # Freeze/update charter from gen1 winner if not set; always evolve prompt bank from survivors
                if not run.charter and population:
                    self._freeze_charter_from_candidate(run, population[0], generation=gen)
                self._evolve_prompt_bank(run, survivors, generation=gen)

                # record generation summary
                avg = sum(c.fitness for c in population) / len(population) if population else 0
                brilliant = [c.id for c in population if c.brilliant]
                cand_dicts = [run._candidate_dict(c) for c in population]
                summary = GenerationSummary(
                    generation=gen,
                    best_fitness=population[0].fitness if population else 0,
                    avg_fitness=avg,
                    survivors=len(survivors),
                    population=len(population),
                    brilliant=brilliant,
                    brilliant_count=len(brilliant),
                    survivors_ids=[c.id for c in survivors],
                    eliminated_ids=[c.id for c in eliminated],
                    candidates=cand_dicts,
                    events_count=len([e for e in run.events if e.get("generation") == gen])
                )
                if population:
                    run.log_event(
                        "generation_snapshot",
                        f"Gen {gen} best={population[0].id} fitness={round(population[0].fitness, 2)} "
                        f"roles={cand_dicts[0].get('cell_roles') if cand_dicts else []} "
                        f"champion={director.get('champion_id') if director else population[0].id} "
                        f"product={product_meta.get('product_path') if product_meta else '—'}",
                        generation=gen,
                        candidate_id=population[0].id,
                        details={
                            "best_id": population[0].id,
                            "best_fitness": population[0].fitness,
                            "avg_fitness": avg,
                            "cell_roles": cand_dicts[0].get("cell_roles") if cand_dicts else [],
                            "survivors": [c.id for c in survivors],
                            "eliminated": [c.id for c in eliminated],
                            "charter_roles": (run.charter or {}).get("roles"),
                            "prompt_variant": (population[0].meta or {}).get("prompt_variant"),
                            "build_files": ((population[0].meta or {}).get("build") or {}).get("files"),
                            "champion_id": (director or {}).get("champion_id") or population[0].id,
                            "decision_maker_id": cfg.decision_maker_id,
                            "product": product_meta,
                            "director_rankings": (director or {}).get("rankings"),
                        },
                    )
                run.update(best=population[0], gen_summary=summary, candidates=population)

                # Structural digest after every gen (instant sense-making for UI + maintainer)
                try:
                    self._write_trace_digest(run, generation=gen, with_gemma=False)
                except Exception as _de:
                    run.log_event("maintain", f"digest write failed: {_de}", generation=gen)

                if gen >= cfg.generations:
                    population = survivors  # final population
                    break

                run.check_stop()
                # breed next generation — inherit code + prompt variants + charter alignment
                population = self._breed(survivors, cfg.population_size, gen, run)

            # final evaluation of survivors if not done
            for cand in population:
                run.check_stop()
                if not cand.scores:
                    self._evaluate(cand, run)

            population.sort(key=lambda c: c.fitness, reverse=True)
            best = population[0] if population else None
            run.update(candidates=population, best=best)
            # End-of-run structural digest; Gemma analysis is left to the background maintainer
            try:
                self._write_trace_digest(run, generation=run.current_generation, with_gemma=False)
            except Exception:
                pass
            if best:
                run.update(status="completed")
                run.log_event("status", "Evolution run completed", details={"best_id": best.id, "fitness": best.fitness})
                # auto-promote the brilliant winner if requested
                if cfg.promote_best and self.real_pm:
                    try:
                        meta = self.best_to_project(run, self.real_pm)
                        if meta:
                            pid = meta.get("id") if isinstance(meta, dict) else str(meta)
                            run.promoted_project_id = pid
                            run.log_event("promote", f"Auto-promoted best candidate to project {pid}", candidate_id=best.id)
                            run._save()
                    except Exception as e:
                        run.log_event("error", f"Auto-promote failed: {e}")
            else:
                run.update(status="completed")
                run.log_event("status", "Evolution run completed without any valid candidates")
        except EvolutionStopped as e:
            # User stop — keep every candidate / call / gen already on disk
            try:
                self._finalize_stopped(run, population)
            except Exception as fin_e:
                run.update(status="stopped", error=f"stopped (finalize warning: {fin_e})")
                run.log_event("status", f"Stopped with finalize warning: {fin_e}")
        except Exception as e:
            if run.should_stop():
                try:
                    self._finalize_stopped(run, population)
                except Exception:
                    run.update(status="stopped", error=str(e))
                    run.log_event("status", f"Stopped during error path: {e}")
            else:
                run.update(status="failed", error=str(e))
                run.log_event("error", f"Evolution run failed: {e}")

    def _create_initial_population(self, run: EvolutionRun) -> list[Candidate]:
        cfg = run.cfg
        pop = []
        base_name = (cfg.name or cfg.goal[:40]).strip().lower().replace(" ", "-") or "evo"
        base_name = "".join(c if c.isalnum() or c == "-" else "-" for c in base_name).strip("-")
        pool = self._worker_pool(cfg)
        run.log_event(
            "status",
            f"Worker pool ({len(pool)} models): {', '.join(pool[:12])}"
            + ("…" if len(pool) > 12 else ""),
            details={
                "worker_pool": pool,
                "diverse_workers": bool(getattr(cfg, "diverse_workers", True)),
                "include_low_throughput_workers": bool(getattr(cfg, "include_low_throughput_workers", True)),
                "include_openrouter_workers": bool(getattr(cfg, "include_openrouter_workers", True)),
            },
        )

        for i in range(cfg.population_size):
            run.check_stop()
            cand_id = f"{base_name}-g0-c{i}"
            cand_dir = run.root / "gen0" / cand_id
            if cand_dir.exists():
                shutil.rmtree(cand_dir)
            cand_dir.mkdir(parents=True, exist_ok=True)

            worker_model = self._pick_worker_model(cfg, i)
            # generate genome via LLM with a little randomness
            prompt = self._creation_prompt(cfg, i)
            try:
                raw = self._call_llm_tracked(run, prompt, "create_initial", cand_id, model=worker_model)
                text = llm.extract_json_block(raw)
                genome = json.loads(text)
                fallback = False
            except Exception:
                # Fallback: use a minimal valid genome
                genome = self._fallback_genome(cfg)
                fallback = True

            genome = self._normalize_genome(genome, cfg)
            # Each gen0 individual carries a prompt_variant — these evolve with fitness
            prompt_variant = self._seed_prompt_variant(cfg, seed_index=i)
            genome["prompt_variant"] = prompt_variant
            meta = {
                "name": f"{base_name}-{i}",
                "goal": cfg.goal,
                "template": genome.get("template", "direct-app"),
                "model": worker_model,
                "llm_model": worker_model,
                "prompt_variant": prompt_variant,
                "lineage": {"root_id": cand_id, "parent_ids": [], "generation": 0},
            }
            self._write_candidate(cand_dir, cand_id, genome, meta)
            cand = Candidate(
                id=cand_id,
                generation=0,
                genome=genome,
                meta=meta,
                path=cand_dir,
            )
            if cfg.use_git:
                self._git_commit_cand(cand, f"create: {cand_id}", run=run)
            pop.append(cand)
            # Keep run.candidates current so later tracked calls resolve model
            run.candidates = pop
            run.log_event(
                "create",
                f"Created initial candidate · worker={worker_model}",
                candidate_id=cand_id,
                model=worker_model,
                details={"fallback": fallback, "prompt_variant": prompt_variant, "worker_model": worker_model},
            )
            # Gen0 software scaffold (further improved each generation)
            if cfg.build_software:
                try:
                    self._build_software(cand, run, mode="scaffold")
                except Exception as e:
                    run.log_event("error", f"Initial scaffold failed: {e}", candidate_id=cand_id)
        run.update(candidates=pop)
        # Provisional charter from first non-fallback-looking candidate
        if pop:
            self._freeze_charter_from_candidate(run, pop[0], generation=0, provisional=True)
        return pop

    def _seed_prompt_variant(self, cfg: EvolutionConfig, seed_index: int = 0) -> dict:
        if cfg.output_type in ("product", "app"):
            emphases = [
                "Prefer a polished index.html (or report) humans can open over abstract architecture only.",
                "Prefer content quality + clear structure that advances the user's goal prompt.",
                "Prefer inheriting ancestor HTML and improving it (mutate + combine + invent).",
                "Prefer interactive lineage/scoreboard sections that show generational progress.",
                "Prefer small working backend only when it serves the product surface.",
                "Prefer end-to-end: research/implement → HTML product → tests that check the artifact exists.",
            ]
        else:
            emphases = [
                "Prefer cryptographic primitives and privacy proofs over CRUD wrappers.",
                "Prefer minimal runnable modules with tests that encode the goal invariants.",
                "Prefer ledger/payment settlement correctness and auditable event logs.",
                "Prefer federated orchestration and failure recovery over feature breadth.",
                "Prefer concrete library APIs (TenSEAL, NEAR SDK) and typed interfaces.",
                "Prefer end-to-end demo path: encrypt → train/aggregate → infer → pay.",
            ]
        e = emphases[seed_index % len(emphases)]
        return {
            "id": f"pv{seed_index}",
            "create_addendum": f"Prompt strategy: {e} Keep cells aligned to the product goal, not generic layers.",
            "build_addendum": f"When writing code: {e} Reuse and extend existing files; do not reinvent unrelated apps.",
            "evaluate_addendum": "Penalize architecture drift away from the frozen charter roles and abandoned modules.",
        }

    def _ensure_charter(self, run: EvolutionRun, population: list[Candidate]) -> None:
        if run.charter or not population:
            return
        # Prefer a candidate that already has a rich role set
        ranked = sorted(population, key=lambda c: len((c.genome or {}).get("cells") or []), reverse=True)
        self._freeze_charter_from_candidate(run, ranked[0], generation=0, provisional=True)

    def _freeze_charter_from_candidate(
        self, run: EvolutionRun, cand: Candidate, generation: int = 0, provisional: bool = False
    ) -> None:
        cells = (cand.genome or {}).get("cells") or []
        roles = [c.get("role") for c in cells if c.get("role")]
        role_goals = {c.get("role"): c.get("goal") for c in cells if c.get("role")}
        files = ((cand.meta or {}).get("build") or {}).get("files") or []
        run.charter = {
            "frozen_at_gen": generation,
            "provisional": provisional,
            "source_candidate_id": cand.id,
            "roles": roles,
            "role_goals": role_goals,
            "innovation_thesis": (cand.genome or {}).get("innovation_thesis") or "",
            "description": (cand.genome or {}).get("description") or "",
            "build_plan": (cand.genome or {}).get("build_plan") or [],
            "core_modules": files[:20],
            "goal": run.cfg.goal,
            "goal_brief": run.cfg.goal_brief or run.cfg.goal,
        }
        run.log_event(
            "charter",
            f"{'Provisional' if provisional else 'Frozen'} architecture charter from {cand.id}: roles={roles}",
            generation=generation,
            candidate_id=cand.id,
            details={"charter": run.charter},
        )
        try:
            (run.root / "charter.json").write_text(json.dumps(run.charter, indent=2), encoding="utf-8")
        except Exception:
            pass
        run._save()

    def _align_genome_to_charter(self, cand: Candidate, run: EvolutionRun) -> None:
        """Soft-align candidate roles to the frozen charter so gens don't abandon the initial architecture."""
        charter = run.charter or {}
        if not charter.get("roles"):
            return
        cells = list((cand.genome or {}).get("cells") or [])
        have = {c.get("role") for c in cells}
        missing = [r for r in charter["roles"] if r not in have]
        if not missing:
            return
        role_goals = charter.get("role_goals") or {}
        for role in missing:
            cells.append({
                "id": f"CR{random.randint(1000, 9999)}",
                "role": role,
                "name": f"Charter {role}",
                "goal": role_goals.get(role) or f"Uphold charter responsibility: {role} for {run.cfg.goal}",
                "params": {"from_charter": True},
                "environment": "local",
                "tools": ["python3", "git"],
                "enabled": True,
                "status": "ready",
            })
        cand.genome["cells"] = cells
        cand.genome["order"] = [c["id"] for c in cells]
        # Keep innovation thesis coherent with charter when empty
        if not cand.genome.get("innovation_thesis") and charter.get("innovation_thesis"):
            cand.genome["innovation_thesis"] = charter["innovation_thesis"]
        if cand.path:
            try:
                self._write_candidate(Path(cand.path), cand.id, cand.genome, cand.meta)
            except Exception:
                pass

    def _score_continuity(self, cand: Candidate, run: EvolutionRun) -> float:
        """0-100: how well this candidate preserves the architectural charter + prior modules."""
        charter = run.charter or {}
        if not charter:
            return 70.0
        cells = (cand.genome or {}).get("cells") or []
        roles = {c.get("role") for c in cells if c.get("role")}
        need = set(charter.get("roles") or [])
        if not need:
            return 70.0
        overlap = len(roles & need) / max(1, len(need))
        # extra roles ok; missing roles hurt
        missing_penalty = len(need - roles) / max(1, len(need))
        score = 100.0 * overlap - 40.0 * missing_penalty
        # thesis continuity
        ct = (charter.get("innovation_thesis") or "").lower().strip()
        it = ((cand.genome or {}).get("innovation_thesis") or "").lower().strip()
        if ct and it and (ct[:40] in it or it[:40] in ct or len(set(ct.split()) & set(it.split())) >= 3):
            score += 10
        # module continuity
        core = set(charter.get("core_modules") or [])
        have_files = set(((cand.meta or {}).get("build") or {}).get("files") or [])
        if cand.path:
            have_files |= set(self._read_candidate_sources(Path(cand.path)).keys())
        if core:
            mod_overlap = len(core & have_files) / max(1, len(core))
            score = 0.7 * score + 0.3 * (100.0 * mod_overlap)
        return max(0.0, min(100.0, score))

    def _evolve_prompt_bank(self, run: EvolutionRun, survivors: list[Candidate], generation: int) -> None:
        """Let high-fitness prompt variants survive into the shared prompt bank."""
        if not survivors:
            return
        top = survivors[0]
        variant = (top.meta or {}).get("prompt_variant") or (top.genome or {}).get("prompt_variant") or {}
        if not variant:
            return
        # Blend: keep bank if top is weak; otherwise promote top's addenda
        bank = run.prompt_bank or {"create_addendum": "", "build_addendum": "", "evaluate_addendum": "", "history": []}
        if top.fitness >= 70 or not bank.get("create_addendum"):
            bank["create_addendum"] = variant.get("create_addendum") or bank.get("create_addendum") or ""
            bank["build_addendum"] = variant.get("build_addendum") or bank.get("build_addendum") or ""
            bank["evaluate_addendum"] = variant.get("evaluate_addendum") or bank.get("evaluate_addendum") or ""
        hist = list(bank.get("history") or [])
        hist.append({
            "generation": generation,
            "fitness": top.fitness,
            "candidate_id": top.id,
            "variant": variant,
            "continuity": (top.meta or {}).get("continuity"),
        })
        bank["history"] = hist[-30:]
        run.prompt_bank = bank
        # Also mutate bank slightly toward goal keywords that appear in best rationale
        rationale = (top.rationale or "").lower()
        tips = []
        if "test" in rationale:
            tips.append("Always require tests that encode goal invariants.")
        if "near" in rationale or "ledger" in rationale:
            tips.append("Keep NEAR/ledger settlement paths explicit in code and cells.")
        if "he" in rationale or "encrypt" in rationale or "tenseal" in rationale:
            tips.append("Preserve HE/TenSEAL (or named crypto) modules across gens.")
        if tips and bank.get("build_addendum"):
            bank["build_addendum"] = bank["build_addendum"] + " " + " ".join(tips)
        try:
            (run.root / "prompt-bank.json").write_text(json.dumps(run.prompt_bank, indent=2), encoding="utf-8")
        except Exception:
            pass
        run.log_event(
            "prompt_evolve",
            f"Prompt bank updated from survivor {top.id} (fit={round(top.fitness, 2)})",
            generation=generation,
            candidate_id=top.id,
            details={"variant": variant, "fitness": top.fitness},
        )
        run._save()

    def _mutate_prompt_variant(self, parent_a: dict, parent_b: Optional[dict] = None, gen: int = 0) -> dict:
        a = dict(parent_a or {})
        b = dict(parent_b or {})
        # Crossover addenda
        out = {
            "id": f"pv-g{gen}-{uuid.uuid4().hex[:6]}",
            "create_addendum": random.choice([a.get("create_addendum"), b.get("create_addendum"), a.get("create_addendum")]) or a.get("create_addendum") or "",
            "build_addendum": random.choice([a.get("build_addendum"), b.get("build_addendum"), a.get("build_addendum")]) or a.get("build_addendum") or "",
            "evaluate_addendum": random.choice([a.get("evaluate_addendum"), b.get("evaluate_addendum"), a.get("evaluate_addendum")]) or a.get("evaluate_addendum") or "",
            "parents": [a.get("id"), b.get("id")],
        }
        muts = [
            " Prefer incremental edits over full rewrites.",
            " Name modules after charter roles and keep those filenames stable.",
            " Cite the frozen innovation thesis in README and module docstrings.",
            " Add a migration note explaining what changed from the parent generation.",
            " Reject generic TODO placeholders; implement simplified real logic.",
        ]
        if random.random() < 0.5:
            key = random.choice(["create_addendum", "build_addendum", "evaluate_addendum"])
            out[key] = (out.get(key) or "") + random.choice(muts)
        return out

    def _creation_prompt(self, cfg: EvolutionConfig, seed: int) -> str:
        mcp_hint = f"Available MCP servers: {', '.join(cfg.mcp_servers)}" if cfg.mcp_servers else ""
        deploy_hint = f"Deployment target: {cfg.deployment_target}" if cfg.deployment_target else ""
        budget_hint = f"Budget USD: {cfg.budget_usd}" if cfg.budget_usd is not None else ""
        provider_hint = f"Preferred providers: {', '.join(cfg.providers)}" if cfg.providers else ""
        output = cfg.output_type if cfg.output_type != "auto" else "product"
        is_product = output in ("product", "app")
        innov_seeds = [
            "clear user-facing narrative + progressive disclosure",
            "interactive scoreboard / lineage of prior generations",
            "privacy-preserving or federated multi-agent orchestration",
            "event-sourced collaboration with auditable history",
            "edge + cloud hybrid delivery of the same product surface",
            "self-improving content modules that cite ancestors",
            "MCP-native tool graph behind a simple UI",
            "lysosome-inspired recycle of failed attempts into lessons",
        ] if is_product else [
            "federated or multi-agent orchestration",
            "homomorphic / privacy-preserving computation",
            "event-sourced or CRDT collaboration",
            "edge + cloud hybrid runtime",
            "self-evolving skill packages",
            "MCP-native tool graphs",
            "zero-knowledge proofs for auditability",
            "biological / lysosome-inspired recycling of failed tasks",
        ]
        innov_hint = innov_seeds[seed % len(innov_seeds)]

        brief = (cfg.goal_brief or cfg.goal or "").strip()
        if is_product:
            mode_rules = dlens.create_mode_rules_product()
            template_hint = '"direct-app" or "hello-world"'
        else:
            mode_rules = (
                "MODE = monetizing factory / architecture design.\n"
                "- Optimize for multi-cell factories that RECYCLE modules into shippable product lines.\n"
                "- Every factory line must still have a path to monetization (SKU, ads, affiliates, portfolio CTAs).\n"
            )
            template_hint = '"direct-app" or "factory-factory" or "hello-world"'
        # prompt variant is attached after creation; bank may already exist mid-run
        return (
            "You are an evolutionary design engine working for a DEPLOYER who ships products for revenue.\n"
            "Later generations will BUILD ON this design — do not design a throwaway sketch.\n\n"
            f"{dlens.DEPLOYER_LENS}\n\n"
            f"User goal: {cfg.goal}\n"
            f"Evolution brief (from planner — follow this):\n{brief}\n\n"
            f"Output type: {output}\n"
            f"{mode_rules}"
            f"Candidate index: {seed} (vary tactics, but stay on the same product goal)\n"
            f"Innovation seed for this candidate: {innov_hint}\n"
            f"{deploy_hint}\n{budget_hint}\n{provider_hint}\n{mcp_hint}\n\n"
            "Return ONLY a JSON object with this shape:\n"
            '{\n'
            '  "description": "short description including the novel twist + money path",\n'
            f'  "template": {template_hint},\n'
            '  "innovation_thesis": "one sentence: non-obvious product or monetization angle",\n'
            '  "monetization": {"primary": "stripe|ads|affiliate|content_funnel", "secondary": "...", "sku_or_cta": "..."},\n'
            '  "cells": [\n'
            '    {"id": "C0", "role": "product-lead", "name": "Product lead", "goal": "...", "params": {}, "environment": "local|docker|browser", "tools": ["..."], "enabled": true, "status": "ready"}\n'
            '  ],\n'
            '  "order": ["C0"],\n'
            '  "environment": ["local"],\n'
            '  "tools": ["git", "python3"],\n'
            '  "build_plan": ["first concrete deliverable", "user-visible surface with monetization e.g. index.html pricing CTA"]\n'
            '}\n\n'
            "Rules:\n"
            "- Optimize for INNOVATION, AUTHENTICITY, VOLUME POTENTIAL, and MONETIZATION while serving the SAME product goal.\n"
            "- Include a growth/monetization cell (Stripe wiring notes, ads slots, affiliate links, or portfolio cross-sell content).\n"
            "- Include at least one cell whose goal is to write/run tests or generate code/HTML artifacts.\n"
            "- Include at least one deployer cell if a deployment target is set.\n"
            "- Include cells that use MCP servers where relevant (tools include 'mcp' and params.server/tool).\n"
            f"- Explicitly incorporate the innovation seed ({innov_hint}) or justify a better alternative in innovation_thesis.\n"
            "- Name roles specifically so later gens can keep them stable.\n"
            "- Keep designs concrete so later generations can extend the same modules rather than rewrite a new app.\n"
        )

    def _fallback_genome(self, cfg: EvolutionConfig) -> dict:
        return {
            "description": f"Fallback candidate for {cfg.goal}",
            "template": "direct-app" if cfg.output_type == "app" else "factory-factory" if cfg.output_type == "factory-factory" else "hello-world",
            "cells": [
                {"id": "C0", "role": "planner", "name": "Planner", "goal": cfg.goal, "params": {}, "environment": "local", "tools": ["git", "python3"], "enabled": True, "status": "ready"},
            ],
            "order": ["C0"],
            "environment": ["local"],
            "tools": ["git", "python3"],
        }

    def _normalize_genome(self, genome: dict, cfg: EvolutionConfig) -> dict:
        cells = genome.get("cells", [])
        for idx, c in enumerate(cells):
            c.setdefault("id", f"C{idx}")
            c.setdefault("enabled", True)
            c.setdefault("status", "ready")
        genome["cells"] = cells
        genome["order"] = genome.get("order", [c["id"] for c in cells])
        genome["environment"] = genome.get("environment", ["local"])
        genome["tools"] = genome.get("tools", ["git", "python3"])
        genome.setdefault("deployment_target", cfg.deployment_target)
        genome.setdefault("budget_usd", cfg.budget_usd)
        genome.setdefault("providers", cfg.providers)
        genome.setdefault("mcp_servers", cfg.mcp_servers)
        return genome

    def _write_candidate(self, cand_dir: Path, cand_id: str, genome: dict, meta: dict) -> None:
        ptype = self._type_for_template(genome.get("template", "blank"))
        cand_dir.mkdir(parents=True, exist_ok=True)
        (cand_dir / "project.json").write_text(json.dumps({
            "id": cand_id,
            "name": meta.get("name", cand_id),
            "description": genome.get("description", ""),
            "template": genome.get("template", "blank"),
            "type": ptype,
            "created_at": utcnow(),
            "goal": meta.get("goal", ""),
            "build": meta.get("build"),
        }, indent=2))
        genome["type"] = ptype
        (cand_dir / "state.json").write_text(json.dumps(genome, indent=2))
        costs = cand_dir / "costs.json"
        if not costs.exists():
            costs.write_text(json.dumps({"project": cand_id, "entries": [], "total_usd": 0}, indent=2))
        notes = cand_dir / "notes.json"
        if not notes.exists():
            notes.write_text(json.dumps({"project": cand_id, "notes": []}, indent=2))

    def _type_for_template(self, template: str) -> str:
        mapping = {"factory-factory": "meta-factory", "direct-app": "app"}
        return mapping.get(template, "factory")

    def _evaluate(self, cand: Candidate, run: EvolutionRun) -> None:
        cfg = run.cfg
        # LLM benchmark scoring (architecture + code)
        prompt = self._evaluate_prompt(cand, cfg, run)
        try:
            raw = self._call_llm_tracked(run, prompt, "evaluate", cand.id)
            text = llm.extract_json_block(raw)
            result = json.loads(text)
        except Exception:
            result = {}

        # Built-in scores from artifacts on disk (product-aware)
        impl = self._score_implementation(cand, cfg=cfg)
        for k, v in impl.items():
            if k in result and result[k] is not None:
                try:
                    result[k] = (float(result[k]) + float(v)) / 2.0
                except Exception:
                    result[k] = v
            else:
                result[k] = v

        # Continuity vs frozen charter (architecture goal survival)
        cont = self._score_continuity(cand, run)
        cand.meta["continuity"] = cont
        result["continuity"] = cont

        # Optional external build/test hook
        if self.build_tester:
            try:
                build_scores = self.build_tester(cand)
                result.update(build_scores)
            except Exception:
                pass

        # Compute weighted fitness
        weights = dict(cfg.benchmark_weights or {})
        if cfg.output_type in ("product", "app"):
            for k, v in dlens.PRODUCT_BENCHMARK_WEIGHTS.items():
                weights.setdefault(k, v)
        else:
            for k, v in dlens.FACTORY_BENCHMARK_WEIGHTS.items():
                weights.setdefault(k, v)
        total_weight = sum(weights.values()) or 1
        fitness = 0.0
        scores = {}
        for k, w in weights.items():
            v = result.get(k)
            try:
                scores[k] = max(0, min(100, float(v)))
            except Exception:
                scores[k] = 50.0
            fitness += scores[k] * (w / total_weight)

        # Hard soft-cap: no monetization path → cannot be brilliant
        mon = scores.get("monetization", 0)
        # Brilliance: high fitness + real money path + quality
        brilliant = (
            mon >= 55
            and (
                fitness >= 85
                or scores.get("goal_fit", 0) >= 88
                or scores.get("artifact_quality", 0) >= 88
                or (scores.get("innovation", 0) >= 85 and mon >= 70)
                or (scores.get("shippability", 0) >= 80 and mon >= 70)
            )
        )

        cand.scores = scores
        cand.fitness = fitness
        cand.brilliant = brilliant
        cand.rationale = result.get("rationale", "")
        focus = "goal" if cfg.output_type in ("product", "app") else "impl"
        run.log_event(
            "evaluate",
            f"Evaluated candidate (fitness: {round(fitness, 2)}, "
            f"{'goal_fit' if focus == 'goal' else 'impl'}: "
            f"{scores.get('goal_fit' if focus == 'goal' else 'implementation', '-')})",
            candidate_id=cand.id,
            model=run.llm_model,
            details={"scores": scores, "brilliant": brilliant, "build": cand.meta.get("build"), "mode": cfg.output_type},
        )

    def _evaluate_prompt(self, cand: Candidate, cfg: EvolutionConfig, run: Optional[EvolutionRun] = None) -> str:
        build = cand.meta.get("build") or {}
        file_list = list(build.get("files") or [])
        # Prefer HTML/product surfaces in previews when present
        if cand.path:
            try:
                for p in Path(cand.path).rglob("*.html"):
                    rel = p.relative_to(cand.path).as_posix()
                    if rel not in file_list and "node_modules" not in rel:
                        file_list.append(rel)
            except Exception:
                pass
        file_preview = ""
        prefer_html = sorted(file_list, key=lambda f: (0 if f.lower().endswith((".html", ".htm", ".css", ".md")) else 1, f))
        if cand.path and prefer_html:
            snippets = []
            for rel in prefer_html[:8]:
                path = Path(cand.path) / rel
                if path.exists() and path.is_file():
                    try:
                        body = path.read_text(encoding="utf-8", errors="replace")[:1200]
                        snippets.append(f"### {rel}\n{body}")
                    except Exception:
                        pass
            file_preview = "\n\n".join(snippets)
        brief = (cfg.goal_brief or cfg.goal or "").strip()
        charter = (run.charter if run else {}) or {}
        bank = (run.prompt_bank if run else {}) or {}
        pv = (cand.meta or {}).get("prompt_variant") or {}
        gen = getattr(run, "current_generation", None) if run else None
        parent_note = ""
        lineage = (cand.meta or {}).get("lineage") or {}
        if lineage.get("parent_ids"):
            parent_note = f"Parents: {lineage.get('parent_ids')}\n"

        if cfg.output_type in ("product", "app"):
            mon = cand.genome.get("monetization") or {}
            return (
                "You evaluate a PRODUCT for a DEPLOYER who wants to ship and make money.\n"
                "PRIMARY questions: (1) advances USER GOAL with authentic content; "
                "(2) has a concrete MONETIZATION path (Stripe/ads/affiliates/cross-sell); "
                "(3) is shippable.\n"
                "Score ONLY the benchmarks listed (0-100) and return ONLY JSON.\n\n"
                f"{dlens.DEPLOYER_LENS}\n\n"
                f"USER GOAL:\n{cfg.goal}\n\n"
                f"Evolution brief:\n{brief[:3500]}\n\n"
                f"Output type: {cfg.output_type}\n"
                f"Generation: {gen}\n"
                f"{parent_note}"
                f"Innovation thesis: {cand.genome.get('innovation_thesis', '')}\n"
                f"Declared monetization: {json.dumps(mon) if mon else '(none declared)'}\n"
                f"Build summary: {build.get('summary', 'no build yet')}\n"
                f"Files ({len(file_list)}): {', '.join(file_list[:25])}\n"
                f"Genome description: {cand.genome.get('description', '')}\n"
                f"Roles: {[c.get('role') for c in (cand.genome.get('cells') or []) if isinstance(c, dict)]}\n\n"
                + (f"Artifact / code previews (prefer HTML):\n{file_preview}\n\n" if file_preview else "Artifacts: (none yet)\n\n")
                + dlens.evaluate_benchmarks_product_text() + "\n\n"
                + (f"Charter roles: {charter.get('roles')}\n" if charter else "")
                + (f"Evaluate tip: {bank.get('evaluate_addendum', '')}\n")
                + (f"Lineage tip: {pv.get('evaluate_addendum', '')}\n\n")
                + "Return JSON exactly like:\n"
                '{"goal_fit": 80, "monetization": 75, "artifact_quality": 72, "shippability": 70, '
                '"innovation": 65, "authenticity": 70, "volume_potential": 60, '
                '"evolution_delta": 68, "implementation": 72, "continuity": 70, '
                '"rationale": "money path …; authenticity/volume …; vs ancestors …"}'
            )

        return (
            "You are evaluating a factory/architecture candidate for a DEPLOYER who monetizes product lines.\n"
            "Score these benchmarks (0-100 each) and return ONLY JSON.\n"
            "CRITICAL: later generations must build on earlier work — reward continuity of charter roles and modules.\n"
            "Factories must still produce monetizable shippable outputs (Stripe/ads/affiliates/portfolio CTAs).\n\n"
            f"{dlens.DEPLOYER_LENS}\n\n"
            f"User goal: {cfg.goal}\n"
            f"Evolution brief:\n{brief[:4000]}\n\n"
            f"Innovation thesis: {cand.genome.get('innovation_thesis', '')}\n"
            f"Build summary: {build.get('summary', 'no build yet')}\n"
            f"Files built ({len(file_list)}): {', '.join(file_list[:20])}\n"
            f"Candidate genome:\n{json.dumps({k: cand.genome.get(k) for k in ('description', 'template', 'innovation_thesis', 'monetization', 'cells', 'order', 'tools', 'build_plan')}, indent=2)}\n\n"
            + (f"Code previews:\n{file_preview}\n\n" if file_preview else "Code previews: (none)\n\n")
            + "Benchmarks:\n"
            "- correctness: does architecture + code address the goal?\n"
            "- completeness: planning through deploy, plus enough source files?\n"
            "- efficiency: lean cells and code without fluff?\n"
            "- deployability: realistic deploy path / Dockerfile / scripts?\n"
            "- monetization: clear product-line money path (SKU, ads, affiliates, cross-sell)\n"
            "- maintainability: clear modules, tests, docs?\n"
            "- innovation: non-obvious mechanisms, not generic CRUD layers?\n"
            "- implementation: quality of actual source files (0 if no code)\n"
            "- continuity: preserves the frozen architecture charter roles/modules from earlier gens\n"
            "- volume_potential: can this factory produce many sellable variants?\n\n"
            + (f"Frozen charter roles: {charter.get('roles')}\nCharter thesis: {charter.get('innovation_thesis')}\nCore modules: {charter.get('core_modules')}\n\n" if charter else "")
            + (f"Shared prompt bank evaluate tip: {bank.get('evaluate_addendum', '')}\n")
            + (f"This candidate prompt strategy: {pv.get('evaluate_addendum', '')}\n\n")
            + "Return JSON exactly like:\n"
            '{"correctness": 78, "completeness": 80, "efficiency": 70, "deployability": 65, '
            '"monetization": 70, "maintainability": 75, "innovation": 80, "implementation": 70, '
            '"continuity": 75, "volume_potential": 60, "rationale": "money path …; factory recycle …"}'
        )

    def _score_implementation(self, cand: Candidate, cfg: Optional[EvolutionConfig] = None) -> dict:
        """Heuristic scores from on-disk artifacts (product-aware when output_type is product/app)."""
        build = cand.meta.get("build") or {}
        files = list(build.get("files") or [])
        # Discover HTML even if not listed in build.files
        if cand.path and Path(cand.path).exists():
            try:
                for p in Path(cand.path).rglob("*"):
                    if not p.is_file():
                        continue
                    rel = p.relative_to(cand.path).as_posix()
                    if any(x in rel for x in (".git/", "node_modules/", "__pycache__/")):
                        continue
                    if rel not in files and p.suffix.lower() in {".html", ".htm", ".css", ".js", ".md", ".py"}:
                        files.append(rel)
            except Exception:
                pass

        product_mode = bool(cfg and cfg.output_type in ("product", "app"))
        if not files:
            base = {"implementation": 15.0, "innovation": 40.0, "monetization": 10.0, "volume_potential": 20.0}
            if product_mode:
                base.update({
                    "goal_fit": 20.0, "artifact_quality": 10.0, "evolution_delta": 25.0,
                    "completeness": 15.0, "shippability": 15.0, "authenticity": 25.0,
                })
            return base

        score = 30.0
        score += min(40.0, len(files) * 6.0)
        if any(f.endswith((".py", ".ts", ".js", ".go", ".rs")) for f in files):
            score += 10
        if any("test" in f.lower() or f.startswith("tests/") for f in files):
            score += 10
        if any(f.lower() in ("dockerfile", "docker-compose.yml", "compose.yaml") or f.endswith("Dockerfile") for f in files):
            score += 5
        if any(f.lower().endswith("readme.md") for f in files):
            score += 5
        if build.get("ok"):
            score += 5
        if build.get("innovations"):
            score += min(10.0, 3.0 * len(build.get("innovations") or []))
        py_ok = 0
        py_n = 0
        if cand.path:
            for rel in files:
                if not rel.endswith(".py"):
                    continue
                py_n += 1
                p = Path(cand.path) / rel
                if p.exists():
                    try:
                        compile(p.read_text(encoding="utf-8", errors="replace"), str(p), "exec")
                        py_ok += 1
                    except Exception:
                        pass
        if py_n:
            score += 10.0 * (py_ok / py_n)
        innov_boost = 55.0 + min(35.0, 5.0 * len(build.get("innovations") or []))
        if cand.genome.get("innovation_thesis"):
            innov_boost += 5

        out = {
            "implementation": max(0.0, min(100.0, score)),
            "innovation": max(35.0, min(100.0, innov_boost if files else 40.0)),
        }

        if product_mode:
            # HTML / product surface quality heuristics
            html_files = [f for f in files if f.lower().endswith((".html", ".htm"))]
            art = 15.0
            goal_fit = 35.0
            goal_tokens = set(re.findall(r"[a-z0-9]{4,}", (cfg.goal or "").lower())) if cfg else set()
            html_blob = ""
            code_blob = ""
            if cand.path and html_files:
                art = 40.0
                for rel in html_files[:4]:
                    p = Path(cand.path) / rel
                    if not p.exists():
                        continue
                    try:
                        body = p.read_text(encoding="utf-8", errors="replace")
                    except Exception:
                        continue
                    html_blob += "\n" + body.lower()
                    art += min(12.0, len(body) / 800.0)
                    if "<h1" in body.lower() or "<h2" in body.lower():
                        art += 4
                    if "style" in body.lower() or ".css" in " ".join(files).lower():
                        art += 3
                    if len(body) > 1500:
                        art += 5
                    if "lorem ipsum" in body.lower() or "todo" in body.lower() and body.lower().count("todo") > 3:
                        art -= 8
            # Sample a bit of non-HTML for monetization signals (JS/py)
            if cand.path:
                for rel in files[:20]:
                    if rel.lower().endswith((".js", ".ts", ".py", ".md", ".json")):
                        p = Path(cand.path) / rel
                        if p.exists() and p.is_file():
                            try:
                                code_blob += "\n" + p.read_text(encoding="utf-8", errors="replace")[:2000].lower()
                            except Exception:
                                pass
            # Goal keyword coverage in HTML + summary
            hay = (html_blob + " " + str(build.get("summary") or "") + " " + str(cand.genome.get("description") or "")).lower()
            if goal_tokens:
                hits = sum(1 for t in goal_tokens if t in hay)
                goal_fit = 30.0 + 70.0 * min(1.0, hits / max(3, min(12, len(goal_tokens))))
            if html_files:
                goal_fit = max(goal_fit, 45.0)
            # Evolution delta: inherited build improved
            evo_d = 50.0
            if build.get("mode") in ("improve", "improve_keep"):
                evo_d += 15
            if build.get("innovations"):
                evo_d += min(20.0, 5.0 * len(build.get("innovations") or []))
            if (cand.meta or {}).get("lineage", {}).get("parent_ids"):
                evo_d += 5
            if not html_files and not any(f.endswith((".py", ".js", ".ts")) for f in files):
                evo_d -= 15
            mon_blob = html_blob + "\n" + code_blob + "\n" + json.dumps(cand.genome.get("monetization") or {})
            mon = dlens.monetization_heuristic_score(mon_blob)
            # Authenticity: penalize lorem / spam; reward longer topical HTML
            authenticity = 55.0
            if "lorem ipsum" in html_blob:
                authenticity -= 30
            if html_blob and len(html_blob) > 2000:
                authenticity += 15
            if goal_tokens and sum(1 for t in goal_tokens if t in html_blob) >= 3:
                authenticity += 15
            # Shippability: HTML + monetization + optional backend
            ship = 35.0 + (20 if html_files else 0) + min(25.0, mon * 0.25)
            if any(f.endswith((".py", ".js", ".ts")) for f in files):
                ship += 10
            # Volume: content pages, catalog signals
            volume = 40.0
            if any(x in html_blob for x in ("blog", "article", "catalog", "collection", "newsletter")):
                volume += 20
            if mon >= 60:
                volume += 15
            out.update({
                "artifact_quality": max(0.0, min(100.0, art)),
                "goal_fit": max(0.0, min(100.0, goal_fit)),
                "evolution_delta": max(0.0, min(100.0, evo_d)),
                "completeness": max(0.0, min(100.0, 25.0 + min(50.0, len(files) * 5) + (15 if html_files else 0))),
                "monetization": max(0.0, min(100.0, mon)),
                "shippability": max(0.0, min(100.0, ship)),
                "authenticity": max(0.0, min(100.0, authenticity)),
                "volume_potential": max(0.0, min(100.0, volume)),
            })
        else:
            # Factory mode still gets a monetization heuristic from files + genome
            blob = json.dumps(cand.genome.get("monetization") or {}) + " " + " ".join(files)
            out["monetization"] = dlens.monetization_heuristic_score(blob)
            out["volume_potential"] = 45.0 + min(30.0, 3.0 * len(files))
        return out

    def _build_software(self, cand: Candidate, run: EvolutionRun, mode: str = "scaffold") -> None:
        """Generate or improve real source files for a candidate across generations.

        Improve mode is MERGE-ONLY: parent files are never wiped on LLM failure.
        """
        cfg = run.cfg
        if not cand.path:
            return
        cand_dir = Path(cand.path)
        existing = self._read_candidate_sources(cand_dir)
        # If improve has no sources, fall back to scaffold once
        if mode == "improve" and not existing:
            mode = "scaffold"
        prompt = self._build_prompt(cand, cfg, mode=mode, existing_files=existing, run=run)
        purpose = "build_scaffold" if mode == "scaffold" else "build_improve"
        llm_ok = True
        err_msg = None
        try:
            raw = self._call_llm_tracked(run, prompt, purpose, cand.id)
            text = llm.extract_json_block(raw)
            payload = json.loads(text)
        except Exception as e:
            llm_ok = False
            err_msg = str(e)
            if mode == "improve" and existing:
                # CRITICAL: do not destroy parent work on 429 / JSON errors
                prev = cand.meta.get("build") or {}
                build_meta = {
                    **prev,
                    "ok": True,
                    "mode": "improve_keep",
                    "generation": run.current_generation,
                    "files": sorted(existing.keys()),
                    "summary": f"Kept parent sources after build LLM failure: {err_msg}"[:240],
                    "innovations": prev.get("innovations") or [],
                    "depth": cfg.build_depth,
                    "model": self._candidate_model(cand, run),
                    "ts": utcnow(),
                    "kept_parent": True,
                    "error": err_msg,
                }
                cand.meta["build"] = build_meta
                cand.genome["build"] = build_meta
                cand.genome["artifacts"] = build_meta["files"]
                try:
                    self._write_candidate(cand_dir, cand.id, cand.genome, cand.meta)
                    (cand_dir / "build-manifest.json").write_text(json.dumps(build_meta, indent=2), encoding="utf-8")
                except Exception:
                    pass
                run.log_event(
                    "build",
                    f"improve_keep: preserved {len(existing)} parent files after LLM failure",
                    candidate_id=cand.id,
                    model=self._candidate_model(cand, run),
                    details={"files": list(existing.keys()), "error": err_msg},
                )
                return
            # scaffold-only fallback
            payload = self._fallback_build(cand, cfg)
            run.log_event("build", f"Build LLM failed ({e}); wrote fallback scaffold", candidate_id=cand.id)

        files = payload.get("files") or []
        written: list[str] = []
        keep_files = payload.get("keep_files") or []
        # Merge: start from existing paths when improving
        all_files = set(existing.keys()) if mode == "improve" else set()
        for item in files:
            if not isinstance(item, dict):
                continue
            rel = (item.get("path") or item.get("name") or "").strip().lstrip("/")
            content = item.get("content")
            if not rel or content is None:
                continue
            if ".." in rel.split("/"):
                continue
            # Skip empty / placeholder content that would erase real code
            cstr = str(content).strip()
            if mode == "improve" and rel in existing and len(cstr) < 20:
                continue
            out = cand_dir / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(str(content), encoding="utf-8")
            written.append(rel)
            all_files.add(rel)
        for rel in keep_files:
            if isinstance(rel, str) and rel in existing:
                all_files.add(rel)

        # Final file inventory on disk (post-merge)
        on_disk = sorted(self._read_candidate_sources(cand_dir, max_files=40, max_chars=200_000).keys())
        if not on_disk and written:
            on_disk = written
        elif on_disk:
            all_files |= set(on_disk)

        summary = payload.get("summary") or f"{mode} build with {len(written)} new/updated files"
        innovations = payload.get("innovations") or []
        build_meta = {
            "ok": bool(on_disk or written),
            "mode": mode,
            "generation": run.current_generation,
            "files": sorted(all_files) if all_files else written,
            "written_this_step": written,
            "summary": summary,
            "innovations": innovations,
            "depth": cfg.build_depth,
            "model": self._candidate_model(cand, run),
            "ts": utcnow(),
            "llm_ok": llm_ok,
            "error": err_msg,
        }
        # Grow charter core modules when we have a successful improve
        if run.charter is not None and build_meta["files"]:
            core = list(run.charter.get("core_modules") or [])
            for f in build_meta["files"]:
                if f not in core:
                    core.append(f)
            run.charter["core_modules"] = core[:40]

        cand.meta["build"] = build_meta
        cand.genome["build"] = build_meta
        cand.genome["artifacts"] = build_meta["files"]
        try:
            self._write_candidate(cand_dir, cand.id, cand.genome, cand.meta)
            (cand_dir / "build-manifest.json").write_text(json.dumps(build_meta, indent=2), encoding="utf-8")
        except Exception:
            pass
        if run.cfg.use_git:
            sha = self._git_commit_cand(
                cand,
                f"build g{run.current_generation} {mode}: {summary[:100]}",
                run=run,
            )
            if sha:
                build_meta["git_sha"] = sha
                cand.meta["build"] = build_meta
        # Individual artifact.html for director pack / lineage browsing
        if run.cfg.produce_product:
            try:
                self._write_candidate_artifact_html(cand, run)
            except Exception:
                pass
        run.log_event(
            "build",
            f"{mode}: wrote {len(written)} files (total {len(build_meta['files'])}) — {summary[:120]}",
            candidate_id=cand.id,
            model=self._candidate_model(cand, run),
            details={"files": build_meta["files"], "written": written, "innovations": innovations, "llm_ok": llm_ok},
        )

    def _read_candidate_sources(self, cand_dir: Path, max_files: int = 12, max_chars: int = 12000) -> dict[str, str]:
        out: dict[str, str] = {}
        if not cand_dir.exists():
            return out
        skip = {"state.json", "project.json", "costs.json", "notes.json", "build-manifest.json", "evolution.json"}
        total = 0
        for p in sorted(cand_dir.rglob("*")):
            if not p.is_file():
                continue
            rel = str(p.relative_to(cand_dir))
            if p.name in skip or rel.startswith("llm_calls"):
                continue
            if p.suffix.lower() not in {".py", ".ts", ".js", ".tsx", ".jsx", ".md", ".toml", ".yml", ".yaml", ".json", ".sh", ".txt", ".html", ".css", ".rs", ".go"} and p.name not in ("Dockerfile", "Makefile"):
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if total + len(text) > max_chars:
                text = text[: max(0, max_chars - total)]
            out[rel] = text
            total += len(text)
            if len(out) >= max_files or total >= max_chars:
                break
        return out

    def _build_prompt(
        self,
        cand: Candidate,
        cfg: EvolutionConfig,
        mode: str,
        existing_files: dict[str, str],
        run: Optional[EvolutionRun] = None,
    ) -> str:
        depth = cfg.build_depth
        existing_blob = ""
        if existing_files:
            parts = []
            for path, body in existing_files.items():
                parts.append(f"### {path}\n```\n{body[:2000]}\n```")
            existing_blob = "\n\n".join(parts)
        charter = (run.charter if run else {}) or {}
        bank = (run.prompt_bank if run else {}) or {}
        pv = (cand.meta or {}).get("prompt_variant") or {}
        if mode == "scaffold":
            task = (
                f"Create a {'rich multi-file implementation' if depth == 'implement' else 'minimal runnable scaffold'} "
                f"for this evolutionary candidate. Prefer real, working code over placeholders."
            )
        else:
            task = (
                "INCREMENTALLY improve the existing software from the parent generation.\n"
                "You are NOT starting a new project. Keep module names, public APIs, and charter roles stable.\n"
                "Fix bugs, fill gaps, add tests, and one concrete innovation — while preserving prior work."
            )
        n_files = 6 if depth == "implement" else 3
        brief = (cfg.goal_brief or cfg.goal or "").strip()
        product_extra = ""
        if cfg.output_type in ("product", "app"):
            product_extra = (
                "PRODUCT MODE: ship toward the user's goal prompt as a real artifact.\n"
                "- Always include or improve index.html (or a clear report HTML) that a human can open.\n"
                "- If the goal is a website → landing + structure. If a report/research topic → narrative HTML "
                "with learnings from ancestors + new findings.\n"
                "- Backend modules are fine when they support the product; avoid pure factory scaffolds.\n"
                "- Generational product workspace will also be built; your candidate still needs its own surface.\n\n"
            )
        return (
            "You are the software builder inside a multi-generation evolutionary loop. "
            f"{task}\n\n"
            f"{product_extra}"
            f"User goal: {cfg.goal}\n"
            f"Output type: {cfg.output_type}\n"
            f"Evolution brief:\n{brief[:3500]}\n\n"
            f"Frozen architecture charter roles: {charter.get('roles')}\n"
            f"Charter innovation thesis: {charter.get('innovation_thesis')}\n"
            f"Charter core modules to preserve: {charter.get('core_modules')}\n"
            f"Candidate innovation thesis: {cand.genome.get('innovation_thesis', '')}\n"
            f"Build plan: {json.dumps(cand.genome.get('build_plan') or [])}\n"
            f"Deployment target: {cfg.deployment_target or 'unspecified'}\n"
            f"Genome cells:\n{json.dumps(cand.genome.get('cells') or [], indent=2)}\n\n"
            f"Shared prompt-bank build tip: {bank.get('build_addendum', '')}\n"
            f"This lineage's prompt strategy: {pv.get('build_addendum', '')}\n\n"
            + (f"Existing files from parent generation (MUST build upon):\n{existing_blob}\n\n" if existing_blob else "Existing files: (none yet)\n\n")
            + "Return ONLY JSON:\n"
            "{\n"
            '  "summary": "what you improved vs parent",\n'
            '  "innovations": ["concrete novel mechanism 1"],\n'
            '  "keep_files": ["paths you intentionally leave unchanged"],\n'
            '  "files": [\n'
            '    {"path": "relative/path.py", "content": "full file content for NEW or CHANGED files only"}\n'
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            f"- For improve mode: only include files you change or add ({n_files // 2}–{n_files} changes is enough); omit unchanged files.\n"
            "- NEVER replace the whole tree with a new unrelated app.\n"
            "- NEVER output empty/tiny content for a path that already exists.\n"
            "- Preserve charter roles in code layout (package/module names can mirror roles).\n"
            "- Code must be self-contained and as runnable as possible without external secrets.\n"
            "- Include or extend at least one test when depth is implement.\n"
            "- Prefer Python or TypeScript unless the goal clearly demands another stack.\n"
            "- No ellipsis placeholders inside code — write real simplified logic.\n"
            "- innovations must name real techniques, not marketing words.\n"
        )

    def _fallback_build(self, cand: Candidate, cfg: EvolutionConfig) -> dict:
        slug = (cand.id or "app").replace("/", "-")[:40]
        main = (
            f'"""Auto-scaffold for {cfg.goal[:80]}"""\n'
            f"from __future__ import annotations\n\n"
            f"GOAL = {cfg.goal!r}\n"
            f"CANDIDATE = {cand.id!r}\n\n"
            f"def main() -> None:\n"
            f"    print('evolution scaffold', CANDIDATE)\n"
            f"    print('goal:', GOAL)\n"
            f"    # cells: {[c.get('role') for c in (cand.genome.get('cells') or [])]}\n\n"
            f"if __name__ == '__main__':\n"
            f"    main()\n"
        )
        readme = f"# {slug}\n\nGoal: {cfg.goal}\n\nGenerated by Dev Studio evolution (fallback scaffold).\n"
        return {
            "summary": "fallback minimal Python scaffold",
            "innovations": [],
            "files": [
                {"path": "README.md", "content": readme},
                {"path": "src/main.py", "content": main},
                {"path": "tests/test_smoke.py", "content": "def test_smoke():\n    assert True\n"},
            ],
        }

    def _attrition(self, population: list[Candidate], attrition_rate: float) -> list[Candidate]:
        keep = max(1, int(round(len(population) * (1 - attrition_rate))))
        return population[:keep]

    def _breed(self, survivors: list[Candidate], target_size: int, gen: int, run: EvolutionRun) -> list[Candidate]:
        cfg = run.cfg
        new_pop = survivors[:]  # elitism: keep survivors
        base_name = (cfg.name or cfg.goal[:40]).strip().lower().replace(" ", "-") or "evo"
        base_name = "".join(c if c.isalnum() or c == "-" else "-" for c in base_name).strip("-")

        parent_pairs = list(zip(survivors, reversed(survivors)))
        while len(new_pop) < target_size:
            run.check_stop()
            if not survivors:
                break
            p1, p2 = random.choice(parent_pairs) if len(survivors) > 1 else (survivors[0], survivors[0])
            offspring, event_details = self._crossover_and_mutate(p1, p2, gen, len(new_pop), cfg, run=run)
            cand_id = f"{base_name}-g{gen}-c{len(new_pop)}"
            cand_dir = run.root / f"gen{gen}" / cand_id
            if cand_dir.exists():
                shutil.rmtree(cand_dir)
            # Inherit parent source tree so the next gen can improve real software
            parent_src = p1 if p1.fitness >= p2.fitness else p2
            inherited = False
            git_cloned = False
            if parent_src.path and Path(parent_src.path).exists():
                if run.cfg.use_git:
                    from lib import evolution_git as egit
                    git_cloned = egit.clone_local(Path(parent_src.path), cand_dir)
                if not git_cloned:
                    cand_dir.mkdir(parents=True, exist_ok=True)
                    self._copy_build_tree(Path(parent_src.path), cand_dir)
                inherited = True
                event_details["inherited_build_from"] = parent_src.id
                event_details["git_cloned"] = git_cloned
                event_details["inherited_files"] = list(self._read_candidate_sources(cand_dir, max_files=40).keys())
            else:
                cand_dir.mkdir(parents=True, exist_ok=True)
            # Prompt variant crossover + mutation (prompt survival)
            pv1 = (p1.meta or {}).get("prompt_variant") or {}
            pv2 = (p2.meta or {}).get("prompt_variant") or {}
            # Bias toward shared bank when present
            if run.prompt_bank and run.prompt_bank.get("build_addendum"):
                base_pv = {
                    "id": "bank",
                    "create_addendum": run.prompt_bank.get("create_addendum", ""),
                    "build_addendum": run.prompt_bank.get("build_addendum", ""),
                    "evaluate_addendum": run.prompt_bank.get("evaluate_addendum", ""),
                }
                prompt_variant = self._mutate_prompt_variant(pv1 or base_pv, pv2 or base_pv, gen=gen)
            else:
                prompt_variant = self._mutate_prompt_variant(pv1, pv2, gen=gen)
            offspring["prompt_variant"] = prompt_variant
            root_id = ((parent_src.meta or {}).get("lineage") or {}).get("root_id") or parent_src.id
            parent_model = (parent_src.meta or {}).get("model") or (p1.meta or {}).get("model")
            worker_model = self._pick_worker_model(cfg, len(new_pop), parent_model=parent_model)
            meta = {
                "name": f"{base_name}-{gen}-{len(new_pop)}",
                "goal": cfg.goal,
                "template": offspring.get("template", "blank"),
                "model": worker_model,
                "llm_model": worker_model,
                "prompt_variant": prompt_variant,
                "lineage": {
                    "root_id": root_id,
                    "parent_ids": [p1.id, p2.id],
                    "generation": gen,
                },
                "build": dict(parent_src.meta.get("build") or {}) if parent_src.meta.get("build") else None,
            }
            if meta.get("build"):
                meta["build"] = {
                    **meta["build"],
                    "inherited": True,
                    "inherited_from": parent_src.id,
                    "generation": None,  # force improve pass next gen
                }
            self._write_candidate(cand_dir, cand_id, offspring, meta)
            cand = Candidate(
                id=cand_id,
                generation=gen,
                genome=offspring,
                meta=meta,
                path=cand_dir,
            )
            # Align to charter immediately so architecture goal persists
            self._align_genome_to_charter(cand, run)
            if run.cfg.use_git:
                msg = f"inherit from {parent_src.id} · breed g{gen}" if inherited else f"breed g{gen} new"
                self._git_commit_cand(cand, msg, run=run)
            new_pop.append(cand)
            event_details["worker_model"] = worker_model
            event_details["parent_model"] = parent_model
            run.log_event(
                "breed",
                f"Offspring {cand_id} from {p1.id[:16]}×{p2.id[:16]}"
                + (f" · inherited build from {parent_src.id[:20]}" if inherited else " · no parent build")
                + f" · worker={worker_model}",
                candidate_id=cand_id,
                model=worker_model,
                details=event_details,
            )
        return new_pop

    def _git_commit_cand(self, cand: Candidate, message: str, *, run: Optional[EvolutionRun] = None) -> Optional[str]:
        if not cand.path:
            return None
        try:
            from lib import evolution_git as egit
            return egit.commit(Path(cand.path), message)
        except Exception as e:
            if run:
                run.log_event("git", f"commit failed for {cand.id}: {e}", candidate_id=cand.id)
            return None

    def _write_candidate_artifact_html(self, cand: Candidate, run: EvolutionRun) -> None:
        if not cand.path:
            return
        from lib import evolution_product as eprod
        pack = eprod.build_candidate_pack(cand, run_goal=run.cfg.goal)
        html = eprod.template_product_html(
            goal=run.cfg.goal,
            gen=run.current_generation,
            evo_id=run.id,
            director={
                "champion_id": cand.id,
                "product_direction": pack.get("build_summary") or pack.get("rationale") or cand.id,
                "must_have": pack.get("innovations") or [],
                "merge_plan": [],
            },
            packs=[pack],
            charter=run.charter or {},
        )
        (Path(cand.path) / "artifact.html").write_text(html, encoding="utf-8")

    def _director_review(self, run: EvolutionRun, population: list, gen: int) -> dict:
        """One decision-maker call per generation: rank, champion, cooperation brief."""
        from lib import evolution_product as eprod
        from lib import agent_runner

        packs = [eprod.build_candidate_pack(c, run_goal=run.cfg.goal) for c in population]
        gdir = eprod.gen_dir(run.root, gen)
        agent_id = (run.cfg.decision_maker_id or "cerebras:zai-glm-4.7").strip()
        director: dict = {}
        raw = ""
        try:
            run.check_stop()
            prompt = eprod.director_prompt(
                run.cfg.goal,
                run.cfg.goal_brief or run.cfg.goal,
                run.cfg.output_type,
                gen,
                packs,
                run.charter or {},
            )
            raw = agent_runner.run_agent(
                agent_id,
                prompt,
                run_id=run.id,
                purpose=f"director_review_g{gen}",
                max_tokens=4096,
                temperature=0.3,
            )
            director = eprod.parse_director_json(raw)
        except EvolutionStopped:
            raise
        except Exception as e:
            run.log_event(
                "director",
                f"Director failed ({agent_id}): {e} — falling back to worker fitness",
                generation=gen,
                details={"agent_id": agent_id, "error": str(e)},
            )
            director = {}

        if not director.get("champion_id") and not director.get("rankings"):
            director = eprod.fallback_director(packs)
            run.log_event("director", "Using fitness fallback director ranking", generation=gen)
        else:
            run.log_event(
                "director",
                f"Director {agent_id} picked {director.get('champion_id')} · "
                f"{len(director.get('rankings') or [])} rankings",
                generation=gen,
                model=agent_id,
                details={
                    "champion_id": director.get("champion_id"),
                    "html_kind": director.get("html_kind"),
                    "fallback": director.get("fallback"),
                },
            )

        # Persist review + brief
        try:
            (gdir / "director-review.json").write_text(
                json.dumps({"agent_id": agent_id, "generation": gen, "director": director, "packs": packs}, indent=2),
                encoding="utf-8",
            )
            brief = eprod.cooperation_brief_md(director, gen, run.cfg.goal)
            (gdir / "cooperation-brief.md").write_text(brief, encoding="utf-8")
        except Exception:
            pass

        director["_packs"] = packs
        director["_agent_id"] = agent_id
        return director

    def _apply_director_scores(self, population: list, director: dict, blend: float) -> None:
        by_id = {
            str(r.get("id")): r
            for r in (director.get("rankings") or [])
            if isinstance(r, dict) and r.get("id")
        }
        blend = max(0.0, min(1.0, float(blend or 0)))
        for cand in population:
            row = by_id.get(cand.id) or {}
            dscore = row.get("director_score")
            try:
                dscore_f = float(dscore) if dscore is not None else float(cand.fitness or 0)
            except Exception:
                dscore_f = float(cand.fitness or 0)
            dscore_f = max(0.0, min(100.0, dscore_f))
            worker = float(cand.fitness or 0)
            cand.scores = dict(cand.scores or {})
            cand.scores["director_score"] = dscore_f
            cand.scores["worker_fitness"] = worker
            cand.fitness = (1.0 - blend) * worker + blend * dscore_f
            if row.get("why"):
                cand.meta = dict(cand.meta or {})
                cand.meta["director_why"] = str(row.get("why"))[:400]

        # Put champion first when present
        champ = director.get("champion_id")
        if champ:
            population.sort(
                key=lambda c: (0 if c.id == champ else 1, -(c.fitness or 0)),
            )
        else:
            population.sort(key=lambda c: c.fitness, reverse=True)

    def _cooperation_and_product(self, run: EvolutionRun, population: list, director: dict, gen: int) -> dict:
        """Build shared genN/product from champion + peer merges + HTML."""
        from lib import evolution_product as eprod
        from lib import evolution_git as egit
        from lib import agent_runner

        packs = director.get("_packs") or [eprod.build_candidate_pack(c, run_goal=run.cfg.goal) for c in population]
        product = eprod.product_dir(run.root, gen)
        champ_id = director.get("champion_id")
        champ = next((c for c in population if c.id == champ_id), None) or (population[0] if population else None)

        # Seed product workspace from champion
        if champ and champ.path:
            eprod.seed_product_from_champion(Path(champ.path), product)
        if run.cfg.use_git:
            egit.init_repo(product)
            egit.commit(product, f"product seed from {champ.id if champ else 'none'} g{gen}", allow_empty=True)

        # Research + maintainer learnings for smarter product HTML
        research = getattr(run, "_research", None) or {}
        research_brief = ""
        try:
            from lib import research_harness as rh
            if research:
                research_brief = rh.format_brief_for_prompt(research)
            elif (run.root / "research-brief.md").exists():
                research_brief = (run.root / "research-brief.md").read_text(encoding="utf-8", errors="replace")[:3500]
        except Exception:
            pass
        learn_text, learn_snips = self._maintainer_learnings_for_product(run)

        # Cooperation: one synthesis call (or template) for product files
        written: list[str] = []
        if run.cfg.cooperation and (run.cfg.decision_maker_id or "").strip() not in ("", "none"):
            try:
                run.check_stop()
                existing = eprod.list_source_files(product, max_files=50)
                prompt = eprod.product_html_prompt(
                    run.cfg.goal,
                    gen,
                    director,
                    packs,
                    run.charter or {},
                    existing,
                    research_brief=research_brief,
                    maintainer_learnings=learn_text,
                )
                raw = agent_runner.run_agent(
                    run.cfg.decision_maker_id,
                    prompt,
                    run_id=run.id,
                    purpose=f"product_html_g{gen}",
                    max_tokens=8192,
                    temperature=0.35,
                )
                from lib import llm as llm_mod
                text = llm_mod.extract_json_block(raw)
                payload = json.loads(text)
                written = eprod.write_files(product, payload.get("files") or [])
                if run.cfg.use_git and written:
                    egit.commit(product, f"product build g{gen}: {payload.get('summary', 'html')[:100]}")
                run.log_event(
                    "product",
                    f"Product agent wrote {len(written)} files for gen {gen}",
                    generation=gen,
                    details={"files": written, "summary": (payload.get("summary") or "")[:200]},
                )
            except EvolutionStopped:
                raise
            except Exception as e:
                run.log_event("product", f"Product agent failed: {e} — using template HTML", generation=gen)

        # Always ensure index.html exists
        idx = product / "index.html"
        if not idx.exists():
            html = eprod.template_product_html(
                goal=run.cfg.goal,
                gen=gen,
                evo_id=run.id,
                director=director,
                packs=packs,
                charter=run.charter or {},
            )
            idx.write_text(html, encoding="utf-8")
            written.append("index.html")
            if run.cfg.use_git:
                egit.commit(product, f"product template HTML g{gen}")
        # Director / coordinator monetization board + provenance colophon
        if idx.exists():
            try:
                body = idx.read_text(encoding="utf-8", errors="replace")
                patched = eprod.ensure_monetization_board(body, director, run.cfg.goal)
                colo_meta = {
                    "evo_id": run.id,
                    "generation": gen,
                    "generations_cfg": run.cfg.generations,
                    "generations_done": len(run.generations),
                    "population_size": run.cfg.population_size,
                    "llm_calls": len(run.llm_calls),
                    "worker_model": run.llm_model or run.cfg.llm_model,
                    "director_model": run.cfg.decision_maker_id,
                    "planner_id": run.cfg.planner_id,
                    "research_harness": (research.get("harness") if research else None)
                    or ("cerebras+web" if research_brief else "off"),
                    "maintainer_learnings": len(learn_snips),
                    "learning_snippets": [s[:180] for s in learn_snips[:5]],
                    "seed_from": getattr(run, "_seed_from", None),
                    "best_fitness": (run.best.fitness if run.best else None),
                    "status": run.status,
                    "signed_at": utcnow(),
                }
                patched = eprod.ensure_product_colophon(patched, colo_meta)
                if patched != body:
                    idx.write_text(patched, encoding="utf-8")
                    if run.cfg.use_git:
                        egit.commit(product, f"monetization board + colophon g{gen}")
                    run.log_event(
                        "product",
                        f"Signed product HTML gen {gen} · director={run.cfg.decision_maker_id} · learnings={len(learn_snips)}",
                        generation=gen,
                    )
            except Exception as e:
                run.log_event("product", f"Monetization/colophon inject failed: {e}", generation=gen)
        if not (product / "PRODUCT.md").exists():
            (product / "PRODUCT.md").write_text(
                eprod.cooperation_brief_md(director, gen, run.cfg.goal),
                encoding="utf-8",
            )

        # Soft-inject into top survivors so breed carries cooperation
        injected = []
        if run.cfg.cooperation and population:
            for c in population[: max(1, len(population) // 2 + 1)]:
                if c.path:
                    got = eprod.soft_inject_product(product, Path(c.path))
                    if got:
                        injected.append({"id": c.id, "files": got})
                        if run.cfg.use_git:
                            self._git_commit_cand(c, f"product_seed inject g{gen}", run=run)

        exports = eprod.publish_product_exports(run.root, gen, product)

        # Lineage map
        lineage = {
            "generation": gen,
            "champion_id": champ.id if champ else None,
            "product_head": egit.head_sha(product) if run.cfg.use_git else None,
            "candidates": {
                c.id: egit.snapshot(Path(c.path)) if c.path else {}
                for c in population
            },
            "exports": exports,
            "injected": injected,
        }
        try:
            (eprod.gen_dir(run.root, gen) / "git-lineage.json").write_text(
                json.dumps(lineage, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

        run.log_event(
            "product",
            f"Gen {gen} product ready · champion={champ.id if champ else '—'} · "
            f"html={exports.get('latest_html', 'gen' + str(gen) + '/product/index.html')}",
            generation=gen,
            candidate_id=champ.id if champ else None,
            details=lineage,
        )
        return {
            "product_path": str(product.relative_to(run.root)),
            "exports": exports,
            "champion_id": champ.id if champ else None,
            "lineage": lineage,
            "written": written,
        }

    def _copy_build_tree(self, src: Path, dst: Path) -> None:
        """Copy software artifacts from parent candidate, skipping evolution metadata."""
        skip_names = {"state.json", "project.json", "costs.json", "notes.json", "build-manifest.json"}
        for p in src.rglob("*"):
            if not p.is_file():
                continue
            if p.name in skip_names:
                continue
            rel = p.relative_to(src)
            if rel.parts and rel.parts[0] == "llm_calls":
                continue
            # Skip .git object bulk when doing plain copy (clone_local preferred)
            if rel.parts and rel.parts[0] == ".git":
                continue
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(p, out)
            except Exception:
                pass

    def _crossover_and_mutate(
        self,
        p1: Candidate,
        p2: Candidate,
        gen: int,
        idx: int,
        cfg: EvolutionConfig,
        run: Optional[EvolutionRun] = None,
    ) -> tuple[dict, dict]:
        # Start from the fitter parent — preserve architecture lineage
        base = p1.genome if p1.fitness >= p2.fitness else p2.genome
        offspring = json.loads(json.dumps(base))  # deep copy
        event_details = {
            "parent1": p1.id,
            "parent2": p2.id,
            "base": p1.id if p1.fitness >= p2.fitness else p2.id,
            "mutations": [],
            "innovations": [],
        }

        # Mutate cells: refine goals/tools — do NOT drop charter roles
        cells = offspring.get("cells", [])
        charter_roles = set(((run.charter if run else {}) or {}).get("roles") or [])
        if random.random() < cfg.mutation_rate and cells:
            # Prefer mutating non-charter extras, else refine a charter cell goal carefully
            mutable = [c for c in cells if c.get("role") not in charter_roles] or cells
            cell = random.choice(mutable)
            if random.random() < 0.5:
                cell["goal"] = self._mutate_text(cell.get("goal", ""))
                event_details["mutations"].append(f"Refined goal of cell {cell.get('id')} ({cell.get('role')})")
            else:
                cell["tools"] = self._mutate_tools(cell.get("tools", []), cfg.mcp_servers)
                event_details["mutations"].append(f"Mutated tools of cell {cell.get('id')}")
            if cfg.deployment_target and random.random() < 0.3:
                cell["environment"] = random.choice(["local", "docker", "browser"])
                event_details["mutations"].append(f"Mutated environment of cell {cell.get('id')}")

        # Innovation: ADD cells / MCP — never replace the whole role set
        if random.random() < cfg.innovation_rate:
            roll = random.random()
            if roll < 0.45:
                new_cell = self._innovation_cell(cfg)
                # avoid role collisions with same id
                cells.append(new_cell)
                event_details["innovations"].append(f"Added cell {new_cell.get('id')} ({new_cell.get('role')})")
            elif roll < 0.75:
                cells = self._add_mcp_cell(cells, cfg)
                event_details["innovations"].append("Added MCP cell")
            else:
                # Soft thesis extension (append), not replacement
                thesis = offspring.get("innovation_thesis") or ((run.charter if run else {}) or {}).get("innovation_thesis") or ""
                twists = [
                    " + verifiable contribution receipts",
                    " + threshold key ceremony for HE aggregation",
                    " + self-healing worker recycle on failed jobs",
                    " + auditable event-sourced settlement",
                ]
                offspring["innovation_thesis"] = (thesis + random.choice(twists)).strip()
                event_details["innovations"].append("Extended innovation_thesis (kept core)")

        # Ensure builder/test cells for software evolution
        if cfg.build_software and not any(
            c.get("role") in ("developer", "builder", "tester", "implementer", "cryptographer") for c in cells
        ):
            cells.append({
                "id": f"CB{random.randint(1000, 9999)}",
                "role": "developer",
                "name": "Software builder",
                "goal": f"Implement and test software for: {cfg.goal}",
                "params": {},
                "environment": "local",
                "tools": ["python3", "git", "pytest"],
                "enabled": True,
                "status": "ready",
            })
            event_details["innovations"].append("Ensured builder cell")

        # Crossover: import a useful cell from the other parent without dropping base
        other = p2 if (p1.fitness >= p2.fitness) else p1
        if random.random() < 0.35 and other.genome.get("cells"):
            donor = random.choice(other.genome["cells"])
            if donor.get("id") not in {c.get("id") for c in cells}:
                cells.append(json.loads(json.dumps(donor)))
                event_details["mutations"].append(f"Crossover: imported cell {donor.get('id')} ({donor.get('role')})")

        offspring["cells"] = cells
        offspring["order"] = [c["id"] for c in cells]
        # Preserve parent description if empty
        if not offspring.get("description"):
            offspring["description"] = base.get("description")
        return self._normalize_genome(offspring, cfg), event_details

    def _mutate_text(self, text: str) -> str:
        suffixes = [
            " with tighter scope",
            " using an MCP tool where possible",
            " with observability built in",
            " optimized for the chosen deployment target",
            " with stronger test coverage",
            " and emit concrete source modules each generation",
            " using an innovative protocol rather than a plain REST CRUD API",
        ]
        return f"{text}{random.choice(suffixes)}"

    def _mutate_tools(self, tools: list[str], mcp_servers: list[str]) -> list[str]:
        base = list(set(tools))
        if random.random() < 0.4 and mcp_servers and "mcp" not in base:
            base.append("mcp")
        additions = ["docker", "pytest", "github-actions", "aws", "cloudflare", "blender", "node"]
        if random.random() < 0.3:
            base.append(random.choice(additions))
        return list(set(base))

    def _innovation_cell(self, cfg: EvolutionConfig) -> dict:
        innovations = [
            {"role": "innovator", "name": "Innovation probe", "goal": "Probe the design space for a novel protocol, data structure, or MCP integration that improves the goal.", "params": {}, "environment": "local", "tools": ["mcp", "llm"]},
            {"role": "benchmarker", "name": "Benchmark runner", "goal": "Write and run KPI scripts that score candidate builds each generation.", "params": {}, "environment": "local", "tools": ["python3", "pytest"]},
            {"role": "builder", "name": "Code synthesizer", "goal": f"Write production-shaped modules implementing: {cfg.goal}", "params": {}, "environment": "local", "tools": ["python3", "git"]},
            {"role": "tester", "name": "Test gardener", "goal": "Grow property/unit tests that force the next generation's code to stay correct.", "params": {}, "environment": "local", "tools": ["pytest", "python3"]},
            {"role": "simplifier", "name": "Simplifier", "goal": "Remove redundant cells and consolidate overlapping code modules.", "params": {}, "environment": "local", "tools": ["llm"]},
            {"role": "attrition", "name": "Attrition judge", "goal": "Kill low-fitness modules and keep only code paths that pass tests.", "params": {}, "environment": "local", "tools": ["python3", "llm"]},
        ]
        cell = random.choice(innovations)
        cell["id"] = f"CX{random.randint(1000, 9999)}"
        cell["enabled"] = True
        cell["status"] = "ready"
        return cell

    def _add_mcp_cell(self, cells: list[dict], cfg: EvolutionConfig) -> list[dict]:
        if not cfg.mcp_servers:
            return cells
        server = random.choice(cfg.mcp_servers)
        cell = {
            "id": f"CM{random.randint(1000, 9999)}",
            "role": "mcp-integrator",
            "name": f"{server} integration",
            "goal": f"Use the {server} MCP server to perform a task relevant to the goal.",
            "params": {"server": server, "tool": "list_tools"},
            "environment": "local",
            "tools": ["mcp"],
            "enabled": True,
            "status": "ready",
        }
        return cells + [cell]

    def best_to_project(self, run: EvolutionRun, real_pm: Any) -> Optional[dict]:
        """Copy the best candidate into the real projects tree and return its meta.

        Also copies the full model-answers transcript so opening the promoted
        project still shows every LLM prompt/response from the evolution run.
        """
        if not run.best or not run.best.path:
            return None
        import shutil
        src = run.best.path
        goal_slug = (run.cfg.name or run.cfg.goal[:40]).strip().lower().replace(" ", "-")
        goal_slug = "".join(c if c.isalnum() or c == "-" else "-" for c in goal_slug).strip("-") or "evolved"
        pid = f"{goal_slug}-{run.id}"
        # if exists, append counter
        dst = Path(real_pm.root) / pid
        i = 2
        while dst.exists():
            pid = f"{goal_slug}-{run.id}-{i}"
            dst = Path(real_pm.root) / pid
            i += 1
        shutil.copytree(src, dst)
        # rewrite project id
        pj = dst / "project.json"
        if pj.exists():
            meta = json.loads(pj.read_text())
            meta["id"] = pid
            meta["evolved_from"] = run.id
            meta["evolution_llm_model"] = run.llm_model
            meta["evolution_llm_calls"] = len(run.llm_calls)
            pj.write_text(json.dumps(meta, indent=2))
        sj = dst / "state.json"
        if sj.exists():
            state = json.loads(sj.read_text())
            state["evolved_from"] = run.id
            state["fitness"] = run.best.fitness
            state["scores"] = run.best.scores
            sj.write_text(json.dumps(state, indent=2))
        # Persist model-answer document + structured calls on the project
        try:
            self._write_llm_transcript(run)
            for name in ("model-answers.md", "llm-calls.json", "evolution.json"):
                src_doc = run.root / name
                if src_doc.exists():
                    shutil.copy2(src_doc, dst / name)
            # Compact answers index for UI
            answers = []
            for c in run.llm_calls:
                answers.append({
                    "id": c.get("id"),
                    "ts": c.get("ts"),
                    "purpose": c.get("purpose"),
                    "model": c.get("model"),
                    "candidate_id": c.get("candidate_id"),
                    "generation": c.get("generation"),
                    "ok": c.get("ok"),
                    "error": c.get("error"),
                    "prompt": c.get("prompt") or c.get("prompt_preview"),
                    "response": c.get("response") or c.get("response_preview"),
                    "duration_secs": c.get("duration_secs"),
                })
            (dst / "evolution-answers.json").write_text(json.dumps({
                "evolution_id": run.id,
                "project_id": pid,
                "goal": run.cfg.goal,
                "llm_model": run.llm_model,
                "best": run._candidate_dict(run.best) if run.best else None,
                "calls": answers,
            }, indent=2), encoding="utf-8")
        except Exception:
            pass
        return real_pm._load_meta(pid)
