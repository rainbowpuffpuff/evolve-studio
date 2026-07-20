"""Trace digests + Gemma analysis for evolution runs.

Goal: turn 50–200 raw events / LLM calls into something a human (or a coordinator)
can actually use — generation rollups, failures, hotspots, prompt smells, and
actionable next steps — without dumping every expanded chat body.

Two layers:
  1. Structural (free, pure Python) — always available, instant.
  2. Gemma narrative (cheap Cerebras) — optional, background / on demand.
"""
from __future__ import annotations

import json
import re
import threading
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from lib import llm


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _purpose_family(purpose: str) -> str:
    p = (purpose or "").lower()
    if "director" in p or "product_html" in p or p.startswith("product"):
        return "director"
    if "plan" in p:
        return "plan"
    if "create" in p or "initial" in p or "breed" in p:
        return "create"
    if "build" in p or "scaffold" in p or "improve" in p:
        return "build"
    if "eval" in p:
        return "evaluate"
    if "charter" in p:
        return "charter"
    if "prompt" in p:
        return "prompt"
    if "trace" in p or "digest" in p or "maintain" in p:
        return "maintain"
    return "other"


def _clip(s: Any, n: int = 180) -> str:
    t = re.sub(r"\s+", " ", str(s or "")).strip()
    return t if len(t) <= n else t[: n - 1] + "…"


def build_structural_digest(data: dict[str, Any]) -> dict[str, Any]:
    """Instant, no-LLM sense-making over evolution.json-shaped dicts."""
    events = list(data.get("events") or [])
    calls = list(data.get("llm_calls") or [])
    gens = list(data.get("generations") or [])
    best = data.get("best") or {}
    cfg = data.get("config") or {}

    by_type = Counter((e.get("type") or "unknown") for e in events)
    by_purpose = Counter((c.get("purpose") or "unknown") for c in calls)
    by_family = Counter(_purpose_family(c.get("purpose") or "") for c in calls)
    by_gen_calls: dict[str, int] = defaultdict(int)
    by_gen_err: dict[str, int] = defaultdict(int)
    tokens_total = 0
    duration_total = 0.0
    errors: list[dict[str, Any]] = []
    empty_answers: list[dict[str, Any]] = []
    hot_tokens: list[dict[str, Any]] = []

    for c in calls:
        g = str(c.get("generation") if c.get("generation") is not None else "?")
        by_gen_calls[g] += 1
        tok = int(c.get("total_tokens") or 0)
        tokens_total += tok
        try:
            duration_total += float(c.get("duration_secs") or 0)
        except (TypeError, ValueError):
            pass
        ok = c.get("ok", True) is not False and not c.get("error")
        resp = c.get("response") or c.get("response_preview") or ""
        if not ok:
            by_gen_err[g] += 1
            errors.append({
                "id": c.get("id"),
                "purpose": c.get("purpose"),
                "generation": c.get("generation"),
                "error": _clip(c.get("error") or "failed", 200),
                "candidate_id": c.get("candidate_id"),
            })
        elif len(str(resp).strip()) < 8:
            empty_answers.append({
                "id": c.get("id"),
                "purpose": c.get("purpose"),
                "generation": c.get("generation"),
            })
        hot_tokens.append({
            "id": c.get("id"),
            "purpose": c.get("purpose"),
            "generation": c.get("generation"),
            "tokens": tok,
            "duration_secs": c.get("duration_secs"),
        })

    hot_tokens.sort(key=lambda x: int(x.get("tokens") or 0), reverse=True)

    # Milestone events (high signal only)
    milestone_types = {
        "status", "plan", "director", "product", "generation_snapshot",
        "error", "promote", "charter", "prompt_evolve", "maintain",
    }
    milestones = []
    for e in events:
        if (e.get("type") or "") in milestone_types:
            milestones.append({
                "ts": e.get("ts"),
                "type": e.get("type"),
                "generation": e.get("generation"),
                "message": _clip(e.get("message"), 160),
                "candidate_id": e.get("candidate_id"),
            })
    # Keep last 24 milestones
    milestones = milestones[-24:]

    gen_rollups = []
    for g in gens:
        gen_n = g.get("generation")
        gen_rollups.append({
            "generation": gen_n,
            "best_fitness": g.get("best_fitness"),
            "avg_fitness": g.get("avg_fitness"),
            "survivors": g.get("survivors"),
            "population": g.get("population"),
            "brilliant_count": g.get("brilliant_count"),
            "calls": by_gen_calls.get(str(gen_n), 0),
            "errors": by_gen_err.get(str(gen_n), 0),
        })
    # Also surface gens that only appear in calls
    known = {r["generation"] for r in gen_rollups}
    for g_key in sorted(by_gen_calls.keys(), key=lambda x: (x == "?", x)):
        try:
            gi = int(g_key)
        except ValueError:
            continue
        if gi not in known:
            gen_rollups.append({
                "generation": gi,
                "best_fitness": None,
                "avg_fitness": None,
                "survivors": None,
                "population": None,
                "brilliant_count": None,
                "calls": by_gen_calls[g_key],
                "errors": by_gen_err.get(g_key, 0),
            })
    gen_rollups.sort(key=lambda r: (r.get("generation") is None, r.get("generation") or 0))

    # Recent compact turns (for list UI) — newest last for chronological read
    recent_turns = []
    for i, c in enumerate(calls[-40:]):
        recent_turns.append({
            "idx": max(0, len(calls) - 40) + i,
            "id": c.get("id"),
            "purpose": c.get("purpose"),
            "family": _purpose_family(c.get("purpose") or ""),
            "generation": c.get("generation"),
            "ok": c.get("ok", True) is not False and not c.get("error"),
            "error": _clip(c.get("error"), 120) if c.get("error") else None,
            "preview": _clip(c.get("response_preview") or c.get("response") or c.get("prompt_preview") or "", 140),
            "tokens": c.get("total_tokens"),
            "duration_secs": c.get("duration_secs"),
            "ts": c.get("ts"),
            "candidate_id": c.get("candidate_id"),
            "model": c.get("model"),
        })

    product_events = [e for e in events if (e.get("type") or "") == "product"]
    error_events = [e for e in events if (e.get("type") or "") == "error"]

    headline_bits = []
    status = data.get("status") or "unknown"
    headline_bits.append(f"status={status}")
    if best:
        headline_bits.append(
            f"best={_clip(best.get('id') or best.get('name'), 40)} "
            f"fit={best.get('fitness')}"
        )
    if errors:
        headline_bits.append(f"{len(errors)} failed LLM turns")
    if empty_answers:
        headline_bits.append(f"{len(empty_answers)} empty answers")
    if gen_rollups:
        last = gen_rollups[-1]
        headline_bits.append(
            f"gen{last.get('generation')}: best_fit={last.get('best_fitness')} "
            f"calls={last.get('calls')}"
        )

    digest = {
        "version": 1,
        "kind": "structural",
        "ts": utcnow(),
        "evolution_id": data.get("id"),
        "status": status,
        "goal": _clip(cfg.get("goal"), 240),
        "headline": " · ".join(headline_bits),
        "counts": {
            "events": len(events),
            "llm_calls": len(calls),
            "errors": len(errors),
            "empty_answers": len(empty_answers),
            "generations_recorded": len(gen_rollups),
            "tokens_total": tokens_total,
            "duration_secs_total": round(duration_total, 2),
        },
        "by_event_type": dict(by_type.most_common()),
        "by_purpose": dict(by_purpose.most_common(20)),
        "by_family": dict(by_family.most_common()),
        "generation_rollups": gen_rollups,
        "errors": errors[:20],
        "empty_answers": empty_answers[:15],
        "top_token_calls": hot_tokens[:8],
        "milestones": milestones,
        "recent_turns": recent_turns,
        "product_event_count": len(product_events),
        "error_event_count": len(error_events),
        "best": {
            "id": best.get("id"),
            "fitness": best.get("fitness"),
            "description": _clip(best.get("description") or best.get("name"), 160),
        } if best else None,
        "suggestions_heuristic": _heuristic_suggestions(
            errors=errors,
            empty=empty_answers,
            by_family=by_family,
            status=status,
            best=best,
            gens=gen_rollups,
        ),
    }
    return digest


def _heuristic_suggestions(
    *,
    errors: list,
    empty: list,
    by_family: Counter,
    status: str,
    best: dict,
    gens: list,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if errors:
        out.append({
            "priority": "high",
            "area": "reliability",
            "action": "Investigate failed LLM turns; tighten prompts or retry policy for those purposes.",
            "evidence": f"{len(errors)} failed calls",
        })
    if empty:
        out.append({
            "priority": "high",
            "area": "prompting",
            "action": "Empty answers usually mean model truncation or JSON-only failures — add format enforcement + shorter max scope.",
            "evidence": f"{len(empty)} empty responses",
        })
    build_n = by_family.get("build", 0)
    eval_n = by_family.get("evaluate", 0)
    if build_n and eval_n and build_n > eval_n * 2:
        out.append({
            "priority": "med",
            "area": "loop_balance",
            "action": "Build calls dominate evaluate — consider cheaper build depth or batch evaluate.",
            "evidence": f"build={build_n} evaluate={eval_n}",
        })
    if status in ("failed", "stopped") and not best:
        out.append({
            "priority": "high",
            "area": "resume",
            "action": "Run has no best candidate — resume or restart with smaller population.",
            "evidence": f"status={status}",
        })
    if gens and all((g.get("best_fitness") or 0) < 40 for g in gens if g.get("best_fitness") is not None):
        out.append({
            "priority": "med",
            "area": "fitness",
            "action": "Fitness stuck low — evolve goal brief / product must-haves, not just mutate genomes.",
            "evidence": "all gen best_fitness < 40",
        })
    if not out:
        out.append({
            "priority": "low",
            "area": "continue",
            "action": "Trace looks healthy — keep generating; run Gemma analysis for deeper prompt evolution.",
            "evidence": "no major structural red flags",
        })
    return out


def save_digest(root: Path, digest: dict[str, Any], *, name: str = "trace-digest.json") -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    dig_dir = root / "digests"
    dig_dir.mkdir(parents=True, exist_ok=True)
    path = dig_dir / name
    path.write_text(json.dumps(digest, indent=2), encoding="utf-8")
    # Always refresh latest pointer
    (dig_dir / "latest.json").write_text(json.dumps(digest, indent=2), encoding="utf-8")
    (root / "trace-digest.json").write_text(json.dumps(digest, indent=2), encoding="utf-8")
    return path


def load_latest_digest(root: Path) -> Optional[dict[str, Any]]:
    root = Path(root)
    for p in (root / "digests" / "latest.json", root / "trace-digest.json"):
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
    return None


def run_gemma_trace_analysis(
    data: dict[str, Any],
    *,
    structural: Optional[dict[str, Any]] = None,
    model: str = "gemma-4-31b",
    max_tokens: int = 2500,
) -> dict[str, Any]:
    """Cheap Gemma pass: narrative + prompt mutations + studio tasks.

    Returns a dict suitable for digests/ and maintainer queue.
    """
    structural = structural or build_structural_digest(data)
    evo_id = data.get("id") or structural.get("evolution_id") or "unknown"
    # Compact payload for the model — never dump full prompts
    payload = {
        "goal": (data.get("config") or {}).get("goal"),
        "status": data.get("status"),
        "headline": structural.get("headline"),
        "counts": structural.get("counts"),
        "by_family": structural.get("by_family"),
        "by_purpose": structural.get("by_purpose"),
        "generation_rollups": structural.get("generation_rollups"),
        "errors": structural.get("errors")[:10],
        "empty_answers": structural.get("empty_answers")[:8],
        "milestones": structural.get("milestones")[-16:],
        "best": structural.get("best"),
        "heuristic_suggestions": structural.get("suggestions_heuristic"),
        "prompt_bank_keys": list((data.get("prompt_bank") or {}).keys())[:20],
        "recent_turn_previews": [
            {
                "purpose": t.get("purpose"),
                "gen": t.get("generation"),
                "ok": t.get("ok"),
                "preview": t.get("preview"),
            }
            for t in (structural.get("recent_turns") or [])[-12:]
        ],
    }

    prompt = f"""You are the continuous maintainer for Dev Studio Evolve — a product that evolves apps/products using LLM agents.

Your job: turn a structural trace digest into HIGH-SIGNAL sense-making and SELF-EVOLUTION actions.
You run on cheap Gemma; be concise and actionable. Never invent metrics not in the data.

Structural digest JSON:
```json
{json.dumps(payload, indent=2)[:12000]}
```

Return ONLY a JSON object with this shape:
{{
  "narrative": "3-6 sentence story of what happened in this run (for humans)",
  "what_worked": ["..."],
  "what_failed": ["..."],
  "prompt_smells": [
    {{"purpose": "create_initial|build|evaluate|director|...", "issue": "...", "fix": "concrete prompt change"}}
  ],
  "prompt_bank_patches": [
    {{"key": "create|build|evaluate|director|cooperation|...", "instruction": "new or improved prompt fragment to merge into the bank"}}
  ],
  "product_next_steps": ["what the product HTML/app should do next gen"],
  "studio_tasks": [
    {{
      "id": "short-slug",
      "area": "frontend|backend|prompting|product|ops",
      "priority": "high|med|low",
      "title": "one line",
      "detail": "what a Cerebras maintainer should implement",
      "auto_safe": true
    }}
  ],
  "coordinator_decisions": [
    "only decisions that need a human/Grok-level call — max 3, else empty"
  ],
  "confidence": 0.0
}}
"""
    text = llm.call_cerebras_sync(
        prompt,
        model=model,
        max_tokens=max_tokens,
        run_id=str(evo_id),
        purpose="trace_digest",
        temperature=0.2,
    )
    analysis: dict[str, Any] = {}
    try:
        raw = llm.extract_json_block(text)
        analysis = json.loads(raw)
    except Exception:
        analysis = {
            "narrative": _clip(text, 600),
            "what_worked": [],
            "what_failed": [],
            "prompt_smells": [],
            "prompt_bank_patches": [],
            "product_next_steps": [],
            "studio_tasks": [],
            "coordinator_decisions": [],
            "confidence": 0.3,
            "parse_error": True,
            "raw": _clip(text, 1200),
        }

    return {
        "version": 1,
        "kind": "gemma",
        "ts": utcnow(),
        "evolution_id": evo_id,
        "model": model,
        "structural": structural,
        "analysis": analysis,
        "headline": structural.get("headline"),
        "narrative": analysis.get("narrative") or structural.get("headline"),
        "suggestions": list(structural.get("suggestions_heuristic") or [])
        + [
            {
                "priority": t.get("priority") or "med",
                "area": t.get("area") or "studio",
                "action": t.get("title") or t.get("detail"),
                "evidence": "gemma",
                "task": t,
            }
            for t in (analysis.get("studio_tasks") or [])
        ],
    }


def merge_prompt_bank_patches(
    prompt_bank: Optional[dict],
    patches: list[dict[str, Any]],
    *,
    max_history: int = 40,
) -> dict[str, Any]:
    """Apply Gemma prompt_bank_patches into a mutable bank (returns new bank)."""
    bank = dict(prompt_bank or {})
    history = list(bank.get("history") or [])
    for p in patches or []:
        key = str(p.get("key") or "").strip()
        instruction = str(p.get("instruction") or "").strip()
        if not key or not instruction:
            continue
        prev = bank.get(key)
        bank[key] = instruction if not prev else f"{prev}\n\n# maintainer patch\n{instruction}"
        # Cap size per key
        if isinstance(bank[key], str) and len(bank[key]) > 6000:
            bank[key] = bank[key][-6000:]
        history.append({
            "ts": utcnow(),
            "key": key,
            "instruction": _clip(instruction, 300),
            "source": "gemma_trace",
        })
    bank["history"] = history[-max_history:]
    bank["updated_at"] = utcnow()
    return bank


def digest_for_run_root(root: Path, data: dict[str, Any], *, with_gemma: bool = False, model: str = "gemma-4-31b") -> dict[str, Any]:
    """Build (+ optional gemma), save, return digest."""
    structural = build_structural_digest(data)
    save_digest(root, structural, name=f"structural-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json")
    if not with_gemma:
        return structural
    full = run_gemma_trace_analysis(data, structural=structural, model=model)
    save_digest(root, full, name=f"gemma-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json")
    return full
