"""Export evolution runs: EDA, PDF report, full zip, transcript/main bundle.

Produces investigative artifacts for a completed (or in-progress) evolution:
- generation fitness trajectories
- architecture / role drift
- software file lineage
- semantic diffs between gen bests
- optional LLM narrative report
- zip packages (full | bundle)
"""
from __future__ import annotations

import json
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_run(evolutions_root: Path, evo_id: str, engine_get=None) -> dict:
    if engine_get:
        run = engine_get(evo_id)
        if run:
            return run._to_dict()
    p = evolutions_root / evo_id / "evolution.json"
    if not p.exists():
        raise FileNotFoundError(f"evolution {evo_id} not found")
    data = json.loads(p.read_text(encoding="utf-8"))
    data["id"] = data.get("id") or evo_id
    return data


def run_dir(evolutions_root: Path, evo_id: str) -> Path:
    return evolutions_root / evo_id


# ── EDA ──────────────────────────────────────────────────────────────────────


def analyze_run(data: dict, root: Optional[Path] = None) -> dict[str, Any]:
    """Exploratory analysis of generation traces, software, and semantic drift."""
    cfg = data.get("config") or {}
    gens = data.get("generations") or []
    calls = data.get("llm_calls") or []
    events = data.get("events") or []
    best = data.get("best") or {}
    charter = data.get("charter") or {}
    prompt_bank = data.get("prompt_bank") or {}

    series = []
    role_timeline = []
    build_timeline = []
    for g in gens:
        cands = sorted(g.get("candidates") or [], key=lambda c: c.get("fitness") or 0, reverse=True)
        top = cands[0] if cands else {}
        roles = top.get("cell_roles") or [c.get("role") for c in (top.get("cells") or [])]
        b = top.get("build") or {}
        scores = top.get("scores") if isinstance(top.get("scores"), dict) else {}
        series.append({
            "generation": g.get("generation"),
            "best_fitness": g.get("best_fitness"),
            "avg_fitness": g.get("avg_fitness"),
            "survivors": g.get("survivors"),
            "population": g.get("population"),
            "best_id": top.get("id"),
            "roles": roles,
            "scores": scores,
            "continuity": (top.get("continuity") if isinstance(top.get("continuity"), (int, float))
                           else scores.get("continuity")),
            "implementation": scores.get("implementation"),
            "innovation": scores.get("innovation"),
            "correctness": scores.get("correctness"),
            "completeness": scores.get("completeness"),
            "efficiency": scores.get("efficiency"),
            "deployability": scores.get("deployability"),
            "maintainability": scores.get("maintainability"),
            "build_files": b.get("files") or top.get("artifacts") or [],
            "build_mode": b.get("mode"),
            "build_summary": b.get("summary"),
            "rationale": (top.get("rationale") or "")[:400],
            "description": (top.get("description") or "")[:300],
            "prompt_variant": top.get("prompt_variant"),
            "candidate_count": len(cands),
            "top_fitnesses": [c.get("fitness") for c in cands[:5]],
        })
        role_timeline.append({"generation": g.get("generation"), "roles": roles, "best_id": top.get("id")})
        build_timeline.append({
            "generation": g.get("generation"),
            "files": b.get("files") or top.get("artifacts") or [],
            "mode": b.get("mode"),
            "summary": b.get("summary"),
            "n_files": len(b.get("files") or top.get("artifacts") or []),
        })

    # Role drift between consecutive bests
    role_diffs = []
    for i in range(1, len(role_timeline)):
        a, b = set(role_timeline[i - 1]["roles"] or []), set(role_timeline[i]["roles"] or [])
        role_diffs.append({
            "from_gen": role_timeline[i - 1]["generation"],
            "to_gen": role_timeline[i]["generation"],
            "added": sorted(b - a),
            "removed": sorted(a - b),
            "stable": sorted(a & b),
            "jaccard": round(len(a & b) / max(1, len(a | b)), 3),
        })

    # File set diffs between consecutive builds
    file_diffs = []
    for i in range(1, len(build_timeline)):
        a, b = set(build_timeline[i - 1]["files"] or []), set(build_timeline[i]["files"] or [])
        file_diffs.append({
            "from_gen": build_timeline[i - 1]["generation"],
            "to_gen": build_timeline[i]["generation"],
            "added": sorted(b - a),
            "removed": sorted(a - b),
            "kept": sorted(a & b),
            "jaccard": round(len(a & b) / max(1, len(a | b)), 3),
        })

    # Semantic tokens from descriptions / rationales / build summaries
    def tokens(text: str) -> set[str]:
        return {t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_\\-]{2,}", (text or "").lower())
                if t not in _STOP}

    semantic_diffs = []
    for i in range(1, len(series)):
        prev, cur = series[i - 1], series[i]
        ta = tokens(" ".join([prev.get("description") or "", prev.get("rationale") or "", prev.get("build_summary") or ""]))
        tb = tokens(" ".join([cur.get("description") or "", cur.get("rationale") or "", cur.get("build_summary") or ""]))
        semantic_diffs.append({
            "from_gen": prev["generation"],
            "to_gen": cur["generation"],
            "new_terms": sorted(list(tb - ta))[:40],
            "lost_terms": sorted(list(ta - tb))[:40],
            "overlap_terms": sorted(list(ta & tb))[:40],
            "jaccard": round(len(ta & tb) / max(1, len(ta | tb)), 3),
            "fitness_delta": round((cur.get("best_fitness") or 0) - (prev.get("best_fitness") or 0), 4),
        })

    call_purposes = Counter(c.get("purpose") for c in calls)
    call_by_gen = Counter(c.get("generation") for c in calls)
    event_types = Counter(e.get("type") for e in events)
    tokens_total = sum(int(c.get("total_tokens") or 0) for c in calls)
    prompt_tokens = sum(int(c.get("prompt_tokens") or 0) for c in calls)
    completion_tokens = sum(int(c.get("completion_tokens") or 0) for c in calls)

    # Lineage / grassroots: survivors across gens
    survivor_sets = []
    for g in gens:
        survivor_sets.append({
            "generation": g.get("generation"),
            "survivors": g.get("survivors_ids") or [],
            "eliminated": g.get("eliminated_ids") or [],
        })

    # Disk software inventory if root available
    disk_inventory = {}
    if root and root.exists():
        for gdir in sorted(root.glob("gen*")):
            kids = []
            for cdir in sorted(gdir.iterdir()):
                if not cdir.is_dir():
                    continue
                man = cdir / "build-manifest.json"
                files = []
                if man.exists():
                    try:
                        files = (json.loads(man.read_text(encoding="utf-8")).get("files") or [])
                    except Exception:
                        pass
                if not files:
                    files = [str(p.relative_to(cdir)) for p in cdir.rglob("*")
                             if p.is_file() and p.name not in
                             ("state.json", "project.json", "costs.json", "notes.json", "build-manifest.json")]
                kids.append({"id": cdir.name, "n_files": len(files), "files": files[:30]})
            disk_inventory[gdir.name] = kids

    fitness_vals = [s.get("best_fitness") for s in series if s.get("best_fitness") is not None]
    improvement = None
    if len(fitness_vals) >= 2:
        improvement = {
            "first_best": fitness_vals[0],
            "last_best": fitness_vals[-1],
            "delta": round(fitness_vals[-1] - fitness_vals[0], 4),
            "max_best": max(fitness_vals),
            "min_best": min(fitness_vals),
        }

    return {
        "evolution_id": data.get("id"),
        "status": data.get("status"),
        "goal": cfg.get("goal"),
        "goal_brief": (cfg.get("goal_brief") or "")[:2000],
        "output_type": cfg.get("output_type") or "product",
        "decision_maker_id": cfg.get("decision_maker_id"),
        "produce_product": cfg.get("produce_product"),
        "llm_model": data.get("llm_model") or cfg.get("llm_model"),
        "planner_id": cfg.get("planner_id"),
        "build_software": cfg.get("build_software"),
        "population_size": cfg.get("population_size"),
        "generations_cfg": cfg.get("generations"),
        "charter": charter,
        "prompt_bank_summary": {
            "create_addendum": (prompt_bank.get("create_addendum") or "")[:300],
            "build_addendum": (prompt_bank.get("build_addendum") or "")[:300],
            "evaluate_addendum": (prompt_bank.get("evaluate_addendum") or "")[:300],
            "history_len": len(prompt_bank.get("history") or []),
        },
        "final_best": {
            "id": best.get("id"),
            "fitness": best.get("fitness"),
            "roles": best.get("cell_roles"),
            "description": best.get("description"),
            "rationale": best.get("rationale"),
            "artifacts": best.get("artifacts") or (best.get("build") or {}).get("files"),
            "scores": best.get("scores"),
            "path": best.get("path"),
        },
        "series": series,
        "role_timeline": role_timeline,
        "role_diffs": role_diffs,
        "build_timeline": build_timeline,
        "file_diffs": file_diffs,
        "semantic_diffs": semantic_diffs,
        "survivor_lineage": survivor_sets,
        "llm_stats": {
            "n_calls": len(calls),
            "by_purpose": dict(call_purposes),
            "by_generation": {str(k): v for k, v in call_by_gen.items()},
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": tokens_total,
        },
        "event_stats": {"n_events": len(events), "by_type": dict(event_types)},
        "fitness_improvement": improvement,
        "disk_inventory": disk_inventory,
        "analyzed_at": utcnow(),
    }


_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "into", "using", "used",
    "have", "has", "was", "were", "are", "is", "been", "being", "will", "can",
    "via", "not", "but", "all", "any", "our", "its", "also", "than", "then",
    "when", "where", "which", "while", "over", "under", "each", "more", "most",
    "such", "only", "other", "some", "these", "those", "their", "them", "they",
    "json", "true", "false", "null", "none", "file", "files", "code", "cell",
}


# ── Report text ──────────────────────────────────────────────────────────────


def render_markdown_report(data: dict, eda: dict, narrative: Optional[str] = None) -> str:
    cfg = data.get("config") or {}
    out_type = eda.get("output_type") or cfg.get("output_type") or "product"
    product_mode = out_type in ("product", "app")
    lines = [
        f"# Evolution investigation report — `{eda.get('evolution_id')}`",
        "",
        f"- **Generated:** {utcnow()}",
        f"- **Status:** {eda.get('status')}",
        f"- **Output type:** {out_type}",
        f"- **Worker model:** {eda.get('llm_model')}",
        f"- **Planner:** {eda.get('planner_id') or 'none'}",
        f"- **Decision maker:** {eda.get('decision_maker_id') or cfg.get('decision_maker_id') or '—'}",
        f"- **Population × generations:** {eda.get('population_size')} × {eda.get('generations_cfg')}",
        f"- **Build software:** {eda.get('build_software')}",
        "",
        "## Goal (what evolution was optimizing for)",
        "",
        eda.get("goal") or cfg.get("goal") or "(none)",
        "",
    ]
    if product_mode:
        lines += [
            "> This was a **product** run. The report judges how generations evolved **toward this goal** "
            "(e.g. HTML/report quality and relevance) — not generic factory architecture.",
            "",
        ]
    if eda.get("goal_brief") and eda.get("goal_brief") != eda.get("goal"):
        lines += ["## Planner brief (excerpt)", "", eda["goal_brief"][:1500], ""]

    if narrative:
        lines += ["## Model narrative (goal evolution)", "", narrative.strip(), ""]

    lines += ["## Goal progress trajectory", ""]
    imp = eda.get("fitness_improvement") or {}
    if imp:
        lines.append(
            f"Best goal-oriented fitness moved **{imp.get('first_best')} → {imp.get('last_best')}** "
            f"(Δ **{imp.get('delta')}**, max {imp.get('max_best')})."
        )
        lines.append("")
    if product_mode:
        lines.append("| Gen | Best fit | Avg | Champion / best | Goal-facing files | Roles |")
        lines.append("|-----|----------|-----|-----------------|-------------------|-------|")
        for s in eda.get("series") or []:
            files = s.get("build_files") or []
            htmlish = [f for f in files if str(f).lower().endswith((".html", ".htm", ".css", ".md"))]
            lines.append(
                f"| {s.get('generation')} | {s.get('best_fitness')} | {s.get('avg_fitness')} | "
                f"`{(s.get('best_id') or '')[:28]}` | {len(htmlish)} html/md of {len(files)} | "
                f"{', '.join(s.get('roles') or [])} |"
            )
    else:
        lines.append("| Gen | Best fit | Avg | Best candidate | Roles | Files | Continuity |")
        lines.append("|-----|----------|-----|----------------|-------|-------|------------|")
        for s in eda.get("series") or []:
            lines.append(
                f"| {s.get('generation')} | {s.get('best_fitness')} | {s.get('avg_fitness')} | "
                f"`{(s.get('best_id') or '')[:28]}` | {', '.join(s.get('roles') or [])} | "
                f"{len(s.get('build_files') or [])} | {s.get('continuity')} |"
            )
    lines.append("")

    charter = eda.get("charter") or {}
    if charter:
        lines += [
            "## Architecture charter",
            "",
            f"- **Frozen at gen:** {charter.get('frozen_at_gen')} (provisional={charter.get('provisional')})",
            f"- **Roles:** {', '.join(charter.get('roles') or [])}",
            f"- **Thesis:** {charter.get('innovation_thesis') or '—'}",
            f"- **Core modules:** {', '.join((charter.get('core_modules') or [])[:20])}",
            "",
        ]

    lines += ["## Role drift (best-of-gen)", ""]
    for d in eda.get("role_diffs") or []:
        lines.append(
            f"- **G{d['from_gen']}→G{d['to_gen']}** (jaccard={d['jaccard']}): "
            f"+{d['added'] or '∅'} −{d['removed'] or '∅'} stable={d['stable'] or '∅'}"
        )
    if not eda.get("role_diffs"):
        lines.append("_No multi-gen role diffs._")
    lines.append("")

    lines += ["## Software file lineage", ""]
    for d in eda.get("file_diffs") or []:
        lines.append(
            f"- **G{d['from_gen']}→G{d['to_gen']}** (jaccard={d['jaccard']}): "
            f"+{d['added'][:8]} −{d['removed'][:8]} kept={len(d['kept'])}"
        )
    if not eda.get("file_diffs"):
        lines.append("_No multi-gen file diffs (or no builds)._")
    lines.append("")

    lines += ["## Semantic drift (descriptions / rationales / build summaries)", ""]
    for d in eda.get("semantic_diffs") or []:
        lines.append(
            f"### G{d['from_gen']} → G{d['to_gen']} (jaccard={d['jaccard']}, Δfit={d['fitness_delta']})"
        )
        lines.append(f"- **New terms:** {', '.join(d['new_terms'][:25]) or '—'}")
        lines.append(f"- **Lost terms:** {', '.join(d['lost_terms'][:25]) or '—'}")
        lines.append("")
    if not eda.get("semantic_diffs"):
        lines.append("_Insufficient generations for semantic comparison._")
        lines.append("")

    lines += ["## Grassroots / survivor lineage", ""]
    for s in eda.get("survivor_lineage") or []:
        lines.append(
            f"- **Gen {s.get('generation')} survivors ({len(s.get('survivors') or [])}):** "
            + ", ".join(f"`{x}`" for x in (s.get("survivors") or [])[:12])
        )
        if s.get("eliminated"):
            lines.append(
                f"  - eliminated: {', '.join(f'`{x}`' for x in (s.get('eliminated') or [])[:8])}"
            )
    lines.append("")

    fb = eda.get("final_best") or {}
    lines += [
        "## Final best candidate",
        "",
        f"- **ID:** `{fb.get('id')}`",
        f"- **Fitness:** {fb.get('fitness')}",
        f"- **Roles:** {', '.join(fb.get('roles') or [])}",
        f"- **Description:** {fb.get('description') or '—'}",
        f"- **Rationale:** {fb.get('rationale') or '—'}",
        f"- **Artifacts:** {', '.join(fb.get('artifacts') or []) or '—'}",
        f"- **Scores:** `{json.dumps(fb.get('scores') or {}, ensure_ascii=False)}`",
        "",
    ]

    ls = eda.get("llm_stats") or {}
    lines += [
        "## Trace / LLM stats",
        "",
        f"- Calls: **{ls.get('n_calls')}**",
        f"- Tokens: prompt={ls.get('prompt_tokens')} completion={ls.get('completion_tokens')} total={ls.get('total_tokens')}",
        f"- By purpose: `{json.dumps(ls.get('by_purpose') or {})}`",
        f"- By generation: `{json.dumps(ls.get('by_generation') or {})}`",
        f"- Events: `{json.dumps((eda.get('event_stats') or {}).get('by_type') or {})}`",
        "",
        "## Prompt bank (evolved)",
        "",
        f"- create: {(eda.get('prompt_bank_summary') or {}).get('create_addendum') or '—'}",
        f"- build: {(eda.get('prompt_bank_summary') or {}).get('build_addendum') or '—'}",
        f"- evaluate: {(eda.get('prompt_bank_summary') or {}).get('evaluate_addendum') or '—'}",
        f"- history entries: {(eda.get('prompt_bank_summary') or {}).get('history_len')}",
        "",
        "---",
        "_Report generated by Dev Studio evolution_export._",
        "",
    ]
    return "\n".join(lines)


# ── Report / narrative agent tools ───────────────────────────────────────────


REPORT_TOOL_SPECS: list[dict[str, Any]] = [
    {"name": "get_overview", "args": {}, "desc": "High-level run KPIs, fitness improvement, final best."},
    {"name": "get_generation", "args": {"gen": "int"}, "desc": "One generation: fitness, survivors, candidate shortlist."},
    {"name": "get_candidate", "args": {"candidate_id": "str"}, "desc": "Candidate scores, roles, build, rationale."},
    {"name": "get_score_series", "args": {}, "desc": "Per-gen best fitness + benchmark score components."},
    {"name": "get_file_lineage", "args": {}, "desc": "Build file sets and diffs across generations."},
    {"name": "get_semantic_drift", "args": {}, "desc": "Semantic term gain/loss between gen bests."},
    {"name": "get_role_drift", "args": {}, "desc": "Role add/remove/jaccard across gens."},
    {"name": "get_charter", "args": {}, "desc": "Frozen architecture charter."},
    {"name": "get_prompt_bank", "args": {}, "desc": "Evolved prompt addenda + history."},
    {"name": "get_llm_stats", "args": {}, "desc": "LLM call counts, tokens, by purpose/generation."},
    {"name": "list_llm_calls", "args": {"purpose": "str?", "generation": "int?", "limit": "int?"},
     "desc": "Short list of LLM calls (id, purpose, gen, tokens, previews)."},
    {"name": "get_llm_call", "args": {"call_id": "str"}, "desc": "Full prompt/response for one call (truncated)."},
    {"name": "list_best_files", "args": {}, "desc": "Files under the final-best candidate path."},
    {"name": "read_best_file", "args": {"path": "str", "max_chars": "int?"},
     "desc": "Read a source file from the best candidate (relative path)."},
    {"name": "get_learnings", "args": {}, "desc": "Saved LEARNING/SUMMARY reports for this run if any."},
    {"name": "get_survivors", "args": {}, "desc": "Survivor / eliminated ids per generation."},
    {"name": "get_disk_inventory", "args": {}, "desc": "Per-gen candidate file inventories on disk."},
]


def _find_candidate(data: dict, candidate_id: str) -> Optional[dict]:
    cid = (candidate_id or "").strip()
    if not cid:
        return None
    for g in data.get("generations") or []:
        for c in g.get("candidates") or []:
            if c.get("id") == cid:
                return c
    best = data.get("best") or {}
    if best.get("id") == cid:
        return best
    return None


def execute_report_tool(
    name: str,
    args: Optional[dict],
    *,
    data: dict,
    eda: dict,
    root: Optional[Path] = None,
) -> Any:
    """Execute one report-agent tool and return a JSON-serializable payload."""
    args = args or {}
    name = (name or "").strip()
    root = Path(root) if root else None

    if name == "get_overview":
        return {
            "evolution_id": eda.get("evolution_id"),
            "status": eda.get("status"),
            "goal": eda.get("goal"),
            "llm_model": eda.get("llm_model"),
            "planner_id": eda.get("planner_id"),
            "population_size": eda.get("population_size"),
            "generations_cfg": eda.get("generations_cfg"),
            "build_software": eda.get("build_software"),
            "fitness_improvement": eda.get("fitness_improvement"),
            "final_best": eda.get("final_best"),
            "n_series": len(eda.get("series") or []),
            "llm_stats": eda.get("llm_stats"),
        }
    if name == "get_generation":
        gen = args.get("gen")
        try:
            gen_i = int(gen)
        except Exception:
            return {"error": "gen must be int"}
        for g in data.get("generations") or []:
            if g.get("generation") == gen_i:
                cands = sorted(g.get("candidates") or [], key=lambda c: c.get("fitness") or 0, reverse=True)
                return {
                    "generation": gen_i,
                    "best_fitness": g.get("best_fitness"),
                    "avg_fitness": g.get("avg_fitness"),
                    "survivors": g.get("survivors"),
                    "population": g.get("population"),
                    "survivors_ids": g.get("survivors_ids"),
                    "eliminated_ids": g.get("eliminated_ids"),
                    "candidates": [
                        {
                            "id": c.get("id"),
                            "fitness": c.get("fitness"),
                            "roles": c.get("cell_roles"),
                            "scores": c.get("scores"),
                            "description": (c.get("description") or "")[:240],
                            "build_files": (c.get("build") or {}).get("files") or c.get("artifacts") or [],
                            "prompt_variant": c.get("prompt_variant"),
                        }
                        for c in cands[:12]
                    ],
                }
        # also from eda series
        for s in eda.get("series") or []:
            if s.get("generation") == gen_i:
                return s
        return {"error": f"generation {gen_i} not found"}
    if name == "get_candidate":
        c = _find_candidate(data, str(args.get("candidate_id") or ""))
        if not c:
            return {"error": "candidate not found"}
        return {
            "id": c.get("id"),
            "generation": c.get("generation"),
            "fitness": c.get("fitness"),
            "scores": c.get("scores"),
            "roles": c.get("cell_roles"),
            "description": c.get("description"),
            "rationale": (c.get("rationale") or "")[:2000],
            "build": c.get("build"),
            "artifacts": c.get("artifacts"),
            "prompt_variant": c.get("prompt_variant"),
            "lineage": c.get("lineage"),
            "continuity": c.get("continuity"),
            "cells": [
                {"id": x.get("id"), "role": x.get("role"), "name": x.get("name"), "goal": (x.get("goal") or "")[:200]}
                for x in (c.get("cells") or [])[:20]
            ],
            "path": c.get("path"),
        }
    if name == "get_score_series":
        return [
            {
                "generation": s.get("generation"),
                "best_id": s.get("best_id"),
                "best_fitness": s.get("best_fitness"),
                "avg_fitness": s.get("avg_fitness"),
                "scores": s.get("scores") or {
                    k: s.get(k) for k in (
                        "correctness", "completeness", "efficiency", "deployability",
                        "maintainability", "innovation", "implementation", "continuity",
                    ) if s.get(k) is not None
                },
            }
            for s in (eda.get("series") or [])
        ]
    if name == "get_file_lineage":
        return {"build_timeline": eda.get("build_timeline"), "file_diffs": eda.get("file_diffs")}
    if name == "get_semantic_drift":
        return eda.get("semantic_diffs") or []
    if name == "get_role_drift":
        return {"role_timeline": eda.get("role_timeline"), "role_diffs": eda.get("role_diffs")}
    if name == "get_charter":
        return eda.get("charter") or data.get("charter") or {}
    if name == "get_prompt_bank":
        pb = data.get("prompt_bank") or {}
        return {
            "create_addendum": pb.get("create_addendum"),
            "build_addendum": pb.get("build_addendum"),
            "evaluate_addendum": pb.get("evaluate_addendum"),
            "history": (pb.get("history") or [])[-12:],
            "summary": eda.get("prompt_bank_summary"),
        }
    if name == "get_llm_stats":
        return eda.get("llm_stats") or {}
    if name == "list_llm_calls":
        purpose = args.get("purpose")
        gen = args.get("generation")
        limit = int(args.get("limit") or 20)
        out = []
        for c in data.get("llm_calls") or []:
            if purpose and c.get("purpose") != purpose:
                continue
            if gen is not None and c.get("generation") != gen:
                try:
                    if int(c.get("generation")) != int(gen):
                        continue
                except Exception:
                    continue
            out.append({
                "id": c.get("id"),
                "purpose": c.get("purpose"),
                "generation": c.get("generation"),
                "candidate_id": c.get("candidate_id"),
                "model": c.get("model"),
                "total_tokens": c.get("total_tokens"),
                "duration_secs": c.get("duration_secs"),
                "ok": c.get("ok"),
                "prompt_preview": (c.get("prompt_preview") or c.get("prompt") or "")[:220],
                "response_preview": (c.get("response_preview") or c.get("response") or "")[:220],
            })
            if len(out) >= limit:
                break
        return out
    if name == "get_llm_call":
        cid = str(args.get("call_id") or "")
        for c in data.get("llm_calls") or []:
            if c.get("id") == cid:
                return {
                    "id": c.get("id"),
                    "purpose": c.get("purpose"),
                    "generation": c.get("generation"),
                    "candidate_id": c.get("candidate_id"),
                    "model": c.get("model"),
                    "tokens": {
                        "prompt": c.get("prompt_tokens"),
                        "completion": c.get("completion_tokens"),
                        "total": c.get("total_tokens"),
                    },
                    "prompt": (c.get("prompt") or "")[:12000],
                    "response": (c.get("response") or "")[:12000],
                    "ok": c.get("ok"),
                    "error": c.get("error"),
                }
        return {"error": "call not found"}
    if name == "list_best_files":
        best = data.get("best") or {}
        bpath = best.get("path") or (eda.get("final_best") or {}).get("path")
        files = (best.get("artifacts") or (best.get("build") or {}).get("files")
                 or (eda.get("final_best") or {}).get("artifacts") or [])
        disk = []
        if bpath and Path(bpath).exists():
            bp = Path(bpath)
            for p in sorted(bp.rglob("*")):
                if p.is_file() and p.name not in (
                    "state.json", "project.json", "costs.json", "notes.json", "build-manifest.json",
                ):
                    disk.append(str(p.relative_to(bp)))
        return {"path": bpath, "manifest_files": files, "disk_files": disk[:80]}
    if name == "read_best_file":
        rel = str(args.get("path") or "").lstrip("/")
        if not rel or ".." in rel.split("/"):
            return {"error": "invalid path"}
        best = data.get("best") or {}
        bpath = best.get("path") or (eda.get("final_best") or {}).get("path")
        if not bpath:
            return {"error": "no best candidate path"}
        fp = Path(bpath) / rel
        if not fp.exists() or not fp.is_file():
            return {"error": f"file not found: {rel}"}
        max_chars = int(args.get("max_chars") or 8000)
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return {"error": str(e)}
        return {"path": rel, "chars": len(text), "content": text[:max_chars]}
    if name == "get_learnings":
        if not root:
            return {"error": "no root"}
        exp = root / "exports"
        items = []
        if exp.exists():
            for p in sorted(exp.glob("LEARNING-*.md")) + sorted(exp.glob("SUMMARY-*.md")):
                if p.name.endswith("-latest.md"):
                    continue
                try:
                    t = p.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                items.append({"filename": p.name, "chars": len(t), "excerpt": t[:1500]})
        return items[:12]
    if name == "get_survivors":
        return eda.get("survivor_lineage") or []
    if name == "get_disk_inventory":
        return eda.get("disk_inventory") or {}
    return {"error": f"unknown tool: {name}"}


def build_report_context_pack(data: dict, eda: dict, root: Optional[Path] = None) -> dict[str, Any]:
    """Pre-fetch the high-value tool results so 1-shot agents still write good reports."""
    pack = {
        "overview": execute_report_tool("get_overview", {}, data=data, eda=eda, root=root),
        "score_series": execute_report_tool("get_score_series", {}, data=data, eda=eda, root=root),
        "file_lineage": execute_report_tool("get_file_lineage", {}, data=data, eda=eda, root=root),
        "semantic_drift": execute_report_tool("get_semantic_drift", {}, data=data, eda=eda, root=root),
        "role_drift": execute_report_tool("get_role_drift", {}, data=data, eda=eda, root=root),
        "charter": execute_report_tool("get_charter", {}, data=data, eda=eda, root=root),
        "prompt_bank": execute_report_tool("get_prompt_bank", {}, data=data, eda=eda, root=root),
        "llm_stats": execute_report_tool("get_llm_stats", {}, data=data, eda=eda, root=root),
        "survivors": execute_report_tool("get_survivors", {}, data=data, eda=eda, root=root),
        "best_files": execute_report_tool("list_best_files", {}, data=data, eda=eda, root=root),
        "recent_calls": execute_report_tool("list_llm_calls", {"limit": 12}, data=data, eda=eda, root=root),
        "learnings": execute_report_tool("get_learnings", {}, data=data, eda=eda, root=root),
    }
    # Include first + last generation detail when available
    gens = [s.get("generation") for s in (eda.get("series") or []) if s.get("generation") is not None]
    if gens:
        pack["first_generation"] = execute_report_tool(
            "get_generation", {"gen": gens[0]}, data=data, eda=eda, root=root
        )
        if gens[-1] != gens[0]:
            pack["last_generation"] = execute_report_tool(
                "get_generation", {"gen": gens[-1]}, data=data, eda=eda, root=root
            )
    return pack


def narrative_prompt(data: dict, eda: dict, root: Optional[Path] = None) -> str:
    """Rich 1-shot prompt with pre-fetched tool pack (fallback / non-tool harnesses)."""
    pack = build_report_context_pack(data, eda, root=root)
    cfg = data.get("config") or {}
    out_type = eda.get("output_type") or cfg.get("output_type") or "product"
    product_mode = out_type in ("product", "app")
    if product_mode:
        structure = (
            "Required structure (use these exact H2 headings):\n"
            "## Executive summary\n"
            "## Goal progress across generations\n"
            "## How the product evolved (HTML / artifacts)\n"
            "## What improved vs ancestors\n"
            "## Decision-maker / cooperation effects\n"
            "## Where the run drifted from the goal\n"
            "## Prompting strategies that advanced the goal\n"
            "## Recommendations for the next run\n"
            "## Next-run checklist\n\n"
            "FOCUS RULES (critical):\n"
            "- The user goal is the only north star. If they asked for an HTML report/site on X, "
            "judge whether generations improved THAT product — not factories, not unrelated architecture.\n"
            "- Trace gen0 → genN: what the product looked like, what changed, who championed, what HTML/files show progress.\n"
            "- Prefer genN/product/index.html, PRODUCT-latest, build file lists, director rankings, fitness scores tied to goal_fit/artifact_quality.\n"
            "- Do NOT pad with generic multi-agent / factory theory unless it served the goal.\n"
        )
    else:
        structure = (
            "Required structure (use these exact H2 headings):\n"
            "## Executive summary\n"
            "## Goal & charter assessment\n"
            "## Fitness trajectory\n"
            "## Software & architecture evolution\n"
            "## Semantic & role drift\n"
            "## Prompting strategies that worked\n"
            "## Risks & failure modes\n"
            "## Recommendations for the next run\n"
            "## Next-run checklist\n\n"
        )
    return (
        "You are an evolution scientist writing an INVESTIGATIVE REPORT for a PDF.\n"
        "Write GitHub-flavored Markdown that will be properly rendered (headings, tables, lists, bold).\n\n"
        f"{structure}"
        "Rules:\n"
        "- 700–1400 words total. Use short paragraphs and bullet lists.\n"
        "- Be specific: generation numbers, candidate ids, file names, score values from the data.\n"
        "- Do NOT invent files, scores, or candidates that are not present.\n"
        "- Include at least one markdown table (e.g. gen × best fitness × goal progress).\n"
        "- End checklist with 5–10 concrete action items.\n"
        "- Do not wrap the whole answer in a code fence.\n\n"
        f"OUTPUT TYPE: {out_type}\n"
        f"GOAL: {eda.get('goal')}\n\n"
        f"CONTEXT PACK (tool outputs):\n{json.dumps(pack, indent=2, ensure_ascii=False)[:55000]}\n"
    )


def _parse_tool_requests(text: str) -> list[dict]:
    """Extract tool call requests from model output."""
    if not text:
        return []
    # Prefer fenced block
    blocks = re.findall(r"```(?:tools|json)\s*([\s\S]*?)```", text, re.I)
    candidates = [b.strip() for b in blocks]
    # bare JSON array with tool-like objects
    if not candidates:
        m = re.search(r"\[\s*\{[\s\S]*?\}\s*\]", text)
        if m:
            candidates.append(m.group(0).strip())
    for cleaned in candidates:
        try:
            obj = json.loads(cleaned)
        except Exception:
            continue
        if isinstance(obj, dict) and "tools" in obj:
            obj = obj["tools"]
        if isinstance(obj, dict) and "name" in obj:
            obj = [obj]
        if isinstance(obj, list):
            out = []
            for item in obj:
                if isinstance(item, dict) and item.get("name"):
                    out.append({
                        "name": item.get("name"),
                        "args": item.get("args") or item.get("arguments") or {},
                    })
            if out:
                return out
    return []


def _extract_final_report(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"```(?:report|markdown|md)\s*([\s\S]*?)```", text, re.I)
    if m:
        return m.group(1).strip()
    # If it looks like a finished markdown report (has executive summary heading)
    if re.search(r"^##\s+Executive summary", text, re.I | re.M) or re.search(
        r"^#\s+", text, re.M
    ):
        # strip tool blocks if any
        cleaned = re.sub(r"```(?:tools|json)\s*[\s\S]*?```", "", text).strip()
        if len(cleaned) > 200:
            return cleaned
    return None


def run_narrative_agent(
    call_llm,
    data: dict,
    eda: dict,
    root: Optional[Path] = None,
    *,
    max_tool_rounds: int = 4,
) -> str:
    """Run a report agent with optional multi-round tools.

    call_llm(prompt: str, *, purpose: str) -> str
    """
    root = Path(root) if root else None
    pack = build_report_context_pack(data, eda, root=root)
    tool_catalog = "\n".join(
        f"- {t['name']}{json.dumps(t['args'])}: {t['desc']}" for t in REPORT_TOOL_SPECS
    )
    out_type = eda.get("output_type") or (data.get("config") or {}).get("output_type") or "product"
    product_focus = (
        "PRODUCT FOCUS: Judge only how generations evolved toward the user GOAL artifact "
        "(HTML/report/app). Do not center the report on factories or abstract architecture "
        "unless the goal explicitly asked for them.\n"
        if out_type in ("product", "app")
        else ""
    )
    system = (
        "You are an evolution investigation agent producing a PDF-ready Markdown report.\n"
        f"{product_focus}"
        "You may call tools to dig deeper, then write the final report.\n\n"
        "TOOL PROTOCOL:\n"
        "1) To request tools, output ONLY a fenced JSON block:\n"
        "```tools\n"
        '[{"name":"get_generation","args":{"gen":1}}, {"name":"read_best_file","args":{"path":"main.py"}}]\n'
        "```\n"
        "2) When ready for the final narrative, output Markdown under:\n"
        "```report\n"
        "## Executive summary\n...\n"
        "```\n"
        "Required H2 sections: Executive summary; Goal & charter assessment; Fitness trajectory; "
        "Software & architecture evolution; Semantic & role drift; Prompting strategies that worked; "
        "Risks & failure modes; Recommendations for the next run; Next-run checklist.\n"
        "Use tables/lists; cite gens, candidate ids, files, scores. Do not invent data.\n\n"
        f"AVAILABLE TOOLS:\n{tool_catalog}\n"
    )
    messages_blob = (
        f"{system}\n\nGOAL: {eda.get('goal')}\n\n"
        f"PREFETCHED CONTEXT PACK:\n{json.dumps(pack, indent=2, ensure_ascii=False)[:45000]}\n\n"
        "Either request tools (if you need deeper detail than the pack) or write the final ```report now."
    )
    tool_trace: list[dict] = []
    last_text = ""
    for round_i in range(max_tool_rounds + 1):
        purpose = "export_narrative" if round_i == 0 else f"export_narrative_tool_r{round_i}"
        last_text = call_llm(messages_blob, purpose=purpose) or ""
        final = _extract_final_report(last_text)
        if final and not _parse_tool_requests(last_text):
            return final
        # if both tools and report, prefer report after tools resolved
        reqs = _parse_tool_requests(last_text)
        if not reqs:
            if final:
                return final
            # treat whole text as report if substantial
            if len(last_text.strip()) > 400 and "```tools" not in last_text:
                return last_text.strip()
            # force final on last round
            if round_i >= max_tool_rounds:
                break
            messages_blob = (
                f"{system}\n\nYou did not request tools and did not produce a ```report block.\n"
                "Write the final investigation report now as ```report markdown.\n\n"
                f"CONTEXT PACK:\n{json.dumps(pack, indent=2, ensure_ascii=False)[:40000]}\n"
            )
            continue
        results = []
        for req in reqs[:6]:
            res = execute_report_tool(
                req.get("name") or "",
                req.get("args") or {},
                data=data,
                eda=eda,
                root=root,
            )
            results.append({"name": req.get("name"), "args": req.get("args"), "result": res})
            tool_trace.append({"name": req.get("name"), "args": req.get("args")})
        messages_blob = (
            f"{system}\n\nTool round {round_i + 1} results:\n"
            f"{json.dumps(results, indent=2, ensure_ascii=False)[:50000]}\n\n"
            "If you need more tools, request them; otherwise write the final ```report now.\n"
            f"Original overview: {json.dumps(pack.get('overview'), ensure_ascii=False)[:4000]}\n"
        )
    # Last-chance forced report
    force_prompt = (
        narrative_prompt(data, eda, root=root)
        + "\n\nWrite the final report now. Do not request tools."
    )
    try:
        last_text = call_llm(force_prompt, purpose="export_narrative_final") or last_text
    except Exception:
        pass
    final = _extract_final_report(last_text) or last_text.strip()
    if tool_trace:
        final = final + f"\n\n---\n_Agent used tools: {', '.join(t['name'] for t in tool_trace)}_\n"
    return final


# ── Charts + PDF (HTML/WeasyPrint with matplotlib figures) ───────────────────


def _mpl_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#cfe0cb",
        "axes.labelcolor": "#1f241f",
        "xtick.color": "#45644c",
        "ytick.color": "#45644c",
        "text.color": "#1f241f",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.color": "#cfe0cb",
    })
    return plt


def generate_report_charts(eda: dict, charts_dir: Path) -> list[dict[str, str]]:
    """Render chart PNGs into charts_dir; return [{id,title,path,filename}]."""
    plt = _mpl_setup()
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    series = eda.get("series") or []
    out: list[dict[str, str]] = []

    def save(fig, chart_id: str, title: str):
        fn = f"{chart_id}.png"
        p = charts_dir / fn
        fig.savefig(p, dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        out.append({"id": chart_id, "title": title, "path": str(p), "filename": fn})

    if series:
        gens = [s.get("generation") for s in series]
        best = [s.get("best_fitness") for s in series]
        avg = [s.get("avg_fitness") for s in series]
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        ax.plot(gens, best, "o-", color="#2f5d3a", linewidth=2.2, markersize=7, label="Best fitness")
        ax.plot(gens, avg, "s--", color="#3d4f8f", linewidth=1.6, markersize=6, label="Avg fitness")
        ax.fill_between(gens, avg, best, color="#d7e676", alpha=0.25)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title("Fitness trajectory")
        ax.legend(frameon=False)
        save(fig, "fitness_trajectory", "Fitness trajectory")

        # Score components (multi-line)
        score_keys = [
            "correctness", "completeness", "efficiency", "deployability",
            "maintainability", "innovation", "implementation", "continuity",
        ]
        colors = ["#2f5d3a", "#3d4f8f", "#c47b2b", "#8b4d6b", "#45644c", "#5b7c99", "#6b8f3c", "#a65d3f"]
        present = [k for k in score_keys if any((s.get("scores") or {}).get(k) is not None or s.get(k) is not None for s in series)]
        if present:
            fig, ax = plt.subplots(figsize=(7.2, 3.8))
            for i, k in enumerate(present):
                ys = []
                for s in series:
                    sc = s.get("scores") or {}
                    ys.append(sc.get(k) if sc.get(k) is not None else s.get(k))
                ax.plot(gens, ys, "o-", color=colors[i % len(colors)], linewidth=1.5, label=k, markersize=5)
            ax.set_xlabel("Generation")
            ax.set_ylabel("Score")
            ax.set_ylim(0, 105)
            ax.set_title("Benchmark score components (best-of-gen)")
            ax.legend(frameon=False, ncol=2, fontsize=8)
            save(fig, "score_components", "Benchmark score components")

        # Roles vs files
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        x = list(range(len(gens)))
        n_roles = [len(s.get("roles") or []) for s in series]
        n_files = [len(s.get("build_files") or []) for s in series]
        w = 0.36
        ax.bar([i - w / 2 for i in x], n_roles, width=w, color="#45644c", label="# roles")
        ax.bar([i + w / 2 for i in x], n_files, width=w, color="#d7e676", edgecolor="#2f5d3a", label="# build files")
        ax.set_xticks(x)
        ax.set_xticklabels([str(g) for g in gens])
        ax.set_xlabel("Generation")
        ax.set_title("Architecture size vs software footprint")
        ax.legend(frameon=False)
        save(fig, "roles_vs_files", "Architecture size vs software footprint")

        # Population fitness spread (top candidates)
        if any(s.get("top_fitnesses") for s in series):
            fig, ax = plt.subplots(figsize=(7.2, 3.4))
            for s in series:
                tops = [v for v in (s.get("top_fitnesses") or []) if v is not None]
                if not tops:
                    continue
                g = s.get("generation")
                ax.scatter([g] * len(tops), tops, s=36, color="#2f5d3a", alpha=0.55, zorder=3)
                ax.plot([g, g], [min(tops), max(tops)], color="#8dbf6a", linewidth=2, zorder=2)
            ax.set_xlabel("Generation")
            ax.set_ylabel("Candidate fitness")
            ax.set_title("Population fitness spread (top candidates)")
            save(fig, "fitness_spread", "Population fitness spread")

    # Semantic / role jaccard
    sem = eda.get("semantic_diffs") or []
    role_d = eda.get("role_diffs") or []
    if sem or role_d:
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        if sem:
            labels = [f"G{d['from_gen']}→{d['to_gen']}" for d in sem]
            ax.plot(labels, [d.get("jaccard") for d in sem], "o-", color="#3d4f8f", label="Semantic jaccard")
        if role_d:
            labels_r = [f"G{d['from_gen']}→{d['to_gen']}" for d in role_d]
            ax.plot(labels_r, [d.get("jaccard") for d in role_d], "s--", color="#c47b2b", label="Role jaccard")
        ax.set_ylim(0, 1.05)
        ax.set_title("Continuity of meaning & roles between gens")
        ax.set_ylabel("Jaccard similarity")
        ax.legend(frameon=False)
        fig.autofmt_xdate(rotation=15)
        save(fig, "drift_jaccard", "Semantic & role continuity")

    # LLM purpose pie / bar
    by_purpose = (eda.get("llm_stats") or {}).get("by_purpose") or {}
    if by_purpose:
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        labels = list(by_purpose.keys())
        vals = [by_purpose[k] for k in labels]
        colors = plt.cm.YlGn([0.35 + 0.5 * i / max(1, len(labels) - 1) for i in range(len(labels))])
        ax.barh(labels, vals, color=colors, edgecolor="#2f5d3a")
        ax.set_xlabel("Calls")
        ax.set_title("LLM calls by purpose")
        fig.tight_layout()
        save(fig, "llm_by_purpose", "LLM calls by purpose")

    by_gen = (eda.get("llm_stats") or {}).get("by_generation") or {}
    if by_gen:
        fig, ax = plt.subplots(figsize=(6.2, 3.4))
        items = sorted(by_gen.items(), key=lambda kv: str(kv[0]))
        ax.bar([str(k) for k, _ in items], [v for _, v in items], color="#8dbf6a", edgecolor="#2f5d3a")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Calls")
        ax.set_title("LLM calls by generation")
        save(fig, "llm_by_generation", "LLM calls by generation")

    return out


def md_to_html(md_text: str) -> str:
    """Render GitHub-flavored markdown to HTML for the PDF body."""
    text = md_text or ""
    try:
        from markdown_it import MarkdownIt
        md = MarkdownIt("commonmark", {"breaks": True, "html": False})
        try:
            md = md.enable("table")
        except Exception:
            pass
        try:
            md = md.enable("strikethrough")
        except Exception:
            pass
        return md.render(text)
    except Exception:
        # very small fallback
        esc = (
            text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        parts = []
        for line in esc.splitlines():
            if line.startswith("# "):
                parts.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith("## "):
                parts.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith("### "):
                parts.append(f"<h3>{line[4:]}</h3>")
            elif line.startswith("- "):
                parts.append(f"<li>{line[2:]}</li>")
            elif not line.strip():
                parts.append("<br/>")
            else:
                parts.append(f"<p>{line}</p>")
        return "\n".join(parts)


def _html_escape(s: Any) -> str:
    t = str(s if s is not None else "")
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def render_html_report(
    data: dict,
    eda: dict,
    narrative: Optional[str] = None,
    charts: Optional[list[dict[str, str]]] = None,
) -> str:
    """Full styled HTML document for WeasyPrint PDF generation."""
    charts = charts or []
    fb = eda.get("final_best") or {}
    imp = eda.get("fitness_improvement") or {}
    ls = eda.get("llm_stats") or {}
    series = eda.get("series") or []

    kpis = [
        ("Status", eda.get("status") or "—"),
        ("Best fit", fb.get("fitness") if fb.get("fitness") is not None else "—"),
        ("Δ fitness", imp.get("delta") if imp.get("delta") is not None else "—"),
        ("Gens", len(series)),
        ("LLM calls", ls.get("n_calls") or 0),
        ("Tokens", ls.get("total_tokens") or 0),
        ("Model", eda.get("llm_model") or "—"),
        ("Planner", eda.get("planner_id") or "none"),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><div class="k">{_html_escape(k)}</div><div class="v">{_html_escape(v)}</div></div>'
        for k, v in kpis
    )

    chart_html = ""
    if charts:
        cells = []
        for c in charts:
            p = Path(c["path"])
            if not p.exists():
                continue
            cells.append(
                f'<figure class="chart"><img src="{_html_escape(p.resolve().as_uri())}" alt="{_html_escape(c["title"])}"/>'
                f'<figcaption>{_html_escape(c["title"])}</figcaption></figure>'
            )
        chart_html = f'<section class="charts"><h2>Visual analysis</h2><div class="chart-grid">{"".join(cells)}</div></section>'

    narrative_html = ""
    if narrative and not str(narrative).startswith("(Narrative generation failed"):
        narrative_html = (
            f'<section class="narrative-sec"><h2>Investigative narrative</h2>'
            f'<div class="narrative md">{md_to_html(narrative)}</div></section>'
        )
    elif narrative:
        narrative_html = f'<section class="narrative-sec"><h2>Narrative</h2><p class="warn">{_html_escape(narrative)}</p></section>'

    # Fitness table
    rows = []
    for s in series:
        roles = ", ".join(s.get("roles") or [])
        rows.append(
            "<tr>"
            f"<td>{_html_escape(s.get('generation'))}</td>"
            f"<td><b>{_html_escape(s.get('best_fitness'))}</b></td>"
            f"<td>{_html_escape(s.get('avg_fitness'))}</td>"
            f"<td><code>{_html_escape((s.get('best_id') or '')[:28])}</code></td>"
            f"<td>{_html_escape(roles)}</td>"
            f"<td>{len(s.get('build_files') or [])}</td>"
            f"<td>{_html_escape(s.get('continuity'))}</td>"
            f"<td>{_html_escape(s.get('innovation'))}</td>"
            "</tr>"
        )
    fitness_table = (
        "<section><h2>Fitness trajectory table</h2>"
        "<table><thead><tr>"
        "<th>Gen</th><th>Best</th><th>Avg</th><th>Best id</th><th>Roles</th><th>Files</th><th>Cont.</th><th>Innov.</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></section>"
    )

    # Role / file / semantic sections as compact HTML lists
    def bullets(items: list[str]) -> str:
        if not items:
            return "<p class='muted'>None</p>"
        return "<ul>" + "".join(f"<li>{_html_escape(x)}</li>" for x in items) + "</ul>"

    role_bits = []
    for d in eda.get("role_diffs") or []:
        role_bits.append(
            f"<li><b>G{d.get('from_gen')}→G{d.get('to_gen')}</b> (j={d.get('jaccard')}): "
            f"+{_html_escape(d.get('added') or '∅')} −{_html_escape(d.get('removed') or '∅')}</li>"
        )
    file_bits = []
    for d in eda.get("file_diffs") or []:
        file_bits.append(
            f"<li><b>G{d.get('from_gen')}→G{d.get('to_gen')}</b> (j={d.get('jaccard')}): "
            f"+{_html_escape((d.get('added') or [])[:8])} −{_html_escape((d.get('removed') or [])[:8])} "
            f"kept={len(d.get('kept') or [])}</li>"
        )
    sem_bits = []
    for d in eda.get("semantic_diffs") or []:
        sem_bits.append(
            f"<div class='sem-card'><h3>G{d.get('from_gen')} → G{d.get('to_gen')} "
            f"<span class='muted'>(j={d.get('jaccard')}, Δfit={d.get('fitness_delta')})</span></h3>"
            f"<p><b>New:</b> {_html_escape(', '.join((d.get('new_terms') or [])[:20]) or '—')}</p>"
            f"<p><b>Lost:</b> {_html_escape(', '.join((d.get('lost_terms') or [])[:20]) or '—')}</p></div>"
        )

    charter = eda.get("charter") or {}
    charter_html = ""
    if charter:
        charter_html = (
            "<section><h2>Architecture charter</h2>"
            f"<p><b>Frozen at gen:</b> {_html_escape(charter.get('frozen_at_gen'))} "
            f"(provisional={_html_escape(charter.get('provisional'))})</p>"
            f"<p><b>Roles:</b> {_html_escape(', '.join(charter.get('roles') or []))}</p>"
            f"<p><b>Thesis:</b> {_html_escape(charter.get('innovation_thesis') or '—')}</p>"
            f"<p><b>Core modules:</b> {_html_escape(', '.join((charter.get('core_modules') or [])[:20]))}</p>"
            "</section>"
        )

    fb_html = (
        "<section><h2>Final best candidate</h2>"
        f"<p><b>ID:</b> <code>{_html_escape(fb.get('id'))}</code> · "
        f"<b>Fitness:</b> {_html_escape(fb.get('fitness'))}</p>"
        f"<p><b>Roles:</b> {_html_escape(', '.join(fb.get('roles') or []))}</p>"
        f"<p><b>Description:</b> {_html_escape(fb.get('description') or '—')}</p>"
        f"<p><b>Rationale:</b> {_html_escape((fb.get('rationale') or '—')[:1200])}</p>"
        f"<p><b>Artifacts:</b> {_html_escape(', '.join(fb.get('artifacts') or []) or '—')}</p>"
        f"<pre class='scores'>{_html_escape(json.dumps(fb.get('scores') or {}, indent=2))}</pre>"
        "</section>"
    )

    goal = _html_escape(eda.get("goal") or (data.get("config") or {}).get("goal") or "")
    css = """
    @page { size: A4; margin: 16mm 14mm 18mm 14mm;
      @bottom-center { content: "Dev Studio · Evolution report · " counter(page); font-size: 8pt; color: #6b7a6b; }
    }
    * { box-sizing: border-box; }
    body { font-family: "DejaVu Sans", "Noto Sans", system-ui, sans-serif; color: #1f241f;
      font-size: 10pt; line-height: 1.45; }
    h1 { color: #2f5d3a; font-size: 20pt; margin: 0 0 6pt; }
    h2 { color: #2f5d3a; font-size: 13pt; margin: 16pt 0 6pt; border-bottom: 1.5px solid #cfe0cb; padding-bottom: 3pt; }
    h3 { color: #45644c; font-size: 11pt; margin: 10pt 0 4pt; }
    .muted { color: #6b7a6b; }
    .cover { background: linear-gradient(135deg, #e7f3ea, #f7faf4); border: 1px solid #cfe0cb;
      border-radius: 10pt; padding: 16pt; margin-bottom: 12pt; }
    .cover .sub { color: #45644c; font-size: 9.5pt; margin-top: 4pt; }
    .kpis { display: flex; flex-wrap: wrap; gap: 6pt; margin: 10pt 0 4pt; }
    .kpi { background: #fff; border: 1px solid #cfe0cb; border-radius: 7pt; padding: 6pt 8pt; min-width: 72pt; }
    .kpi .k { font-size: 7.5pt; text-transform: uppercase; letter-spacing: .04em; color: #6b7a6b; }
    .kpi .v { font-size: 11pt; font-weight: 700; color: #1f241f; margin-top: 1pt; }
    .goal-box { background: #fff; border-left: 4px solid #2f5d3a; padding: 8pt 10pt; margin-top: 8pt; }
    .chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10pt; }
    .chart { margin: 0; background: #fff; border: 1px solid #e3eedd; border-radius: 8pt; padding: 6pt; break-inside: avoid; }
    .chart img { width: 100%; height: auto; }
    .chart figcaption { font-size: 8pt; color: #6b7a6b; text-align: center; margin-top: 3pt; }
    .narrative { background: #f7faf4; border: 1px solid #cfe0cb; border-radius: 8pt; padding: 10pt 12pt; }
    .narrative h1, .narrative h2, .narrative h3 { margin-top: 8pt; }
    .narrative table { font-size: 9pt; }
    .narrative ul, .narrative ol { margin: 4pt 0 6pt 14pt; padding: 0; }
    .narrative p { margin: 4pt 0; }
    table { width: 100%; border-collapse: collapse; font-size: 8.5pt; margin: 6pt 0 10pt; }
    th, td { border: 1px solid #cfe0cb; padding: 4pt 5pt; text-align: left; vertical-align: top; }
    th { background: #e7f3ea; color: #2f5d3a; font-weight: 700; }
    tr:nth-child(even) td { background: #fbfcf9; }
    code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.5pt; background: #eef4ea; padding: 0 2pt; border-radius: 2pt; }
    pre.scores { background: #f3f6f1; border: 1px solid #e3eedd; border-radius: 6pt; padding: 8pt; font-size: 8pt; white-space: pre-wrap; }
    .sem-card { background: #fff; border: 1px solid #e3eedd; border-radius: 6pt; padding: 6pt 8pt; margin: 4pt 0; break-inside: avoid; }
    .warn { color: #a33; }
    .footer-note { color: #6b7a6b; font-size: 8pt; margin-top: 14pt; }
    section { break-inside: avoid; }
    """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Evolution report · {_html_escape(eda.get('evolution_id'))}</title>
<style>{css}</style>
</head>
<body>
  <header class="cover">
    <h1>Evolution investigation report</h1>
    <div class="sub">
      <code>{_html_escape(eda.get('evolution_id'))}</code>
      · generated {_html_escape(utcnow().replace('T', ' ')[:19])} UTC
      · pop {_html_escape(eda.get('population_size'))} × {_html_escape(eda.get('generations_cfg'))} gens
      · build={_html_escape(eda.get('build_software'))}
    </div>
    <div class="kpis">{kpi_html}</div>
    <div class="goal-box"><b>Goal</b><br/>{goal}</div>
  </header>

  {chart_html}
  {narrative_html}
  {fitness_table}
  {charter_html}

  <section>
    <h2>Role drift</h2>
    {"<ul>" + "".join(role_bits) + "</ul>" if role_bits else "<p class='muted'>No multi-gen role diffs.</p>"}
  </section>
  <section>
    <h2>Software file lineage</h2>
    {"<ul>" + "".join(file_bits) + "</ul>" if file_bits else "<p class='muted'>No multi-gen file diffs (or no builds).</p>"}
  </section>
  <section>
    <h2>Semantic drift</h2>
    {"".join(sem_bits) if sem_bits else "<p class='muted'>Insufficient generations for semantic comparison.</p>"}
  </section>
  {fb_html}
  <section>
    <h2>Trace / LLM stats</h2>
    <p>Calls: <b>{_html_escape(ls.get('n_calls'))}</b> ·
       tokens prompt={_html_escape(ls.get('prompt_tokens'))}
       completion={_html_escape(ls.get('completion_tokens'))}
       total={_html_escape(ls.get('total_tokens'))}</p>
    <p>By purpose: <code>{_html_escape(json.dumps(ls.get('by_purpose') or {}))}</code></p>
    <p>By generation: <code>{_html_escape(json.dumps(ls.get('by_generation') or {}))}</code></p>
  </section>
  <p class="footer-note">Generated by Dev Studio evolution_export · charts via matplotlib · PDF via WeasyPrint · markdown rendered (not raw).</p>
</body>
</html>
"""


def write_pdf_report(
    path: Path,
    data: dict,
    eda: dict,
    narrative: Optional[str] = None,
    *,
    charts_dir: Optional[Path] = None,
) -> Path:
    """Multi-page visual PDF: charts + rendered markdown narrative (WeasyPrint).

    Falls back to an improved matplotlib multipage PDF if WeasyPrint is unavailable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    charts_dir = Path(charts_dir) if charts_dir else path.parent / "charts"
    charts = generate_report_charts(eda, charts_dir)

    html = render_html_report(data, eda, narrative=narrative, charts=charts)
    html_path = path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")

    try:
        from weasyprint import HTML
        HTML(filename=str(html_path), base_url=str(path.parent)).write_pdf(str(path))
        return path
    except Exception as e_wp:
        # Fallback: matplotlib pages, but still try to render narrative less raw
        return _write_pdf_report_matplotlib_fallback(
            path, data, eda, narrative=narrative, charts=charts, error=str(e_wp)
        )


def _write_pdf_report_matplotlib_fallback(
    path: Path,
    data: dict,
    eda: dict,
    narrative: Optional[str] = None,
    charts: Optional[list[dict]] = None,
    error: str = "",
) -> Path:
    """Fallback PDF when WeasyPrint fails — still embeds chart PNGs and wraps text."""
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    import textwrap
    from PIL import Image

    charts = charts or []
    path = Path(path)

    def text_pages(title: str, body: str, fontsize: float = 9.0):
        wrapped = []
        for para in (body or "").split("\n"):
            if not para.strip():
                wrapped.append("")
            else:
                wrapped.extend(textwrap.wrap(para, width=95) or [""])
        lines_per = 48
        chunks = [wrapped[i:i + lines_per] for i in range(0, max(1, len(wrapped)), lines_per)] or [[]]
        figs = []
        for ci, chunk in enumerate(chunks):
            fig = plt.figure(figsize=(8.5, 11))
            fig.patch.set_facecolor("white")
            ax = fig.add_axes([0.07, 0.05, 0.86, 0.9])
            ax.axis("off")
            ttl = title if ci == 0 else f"{title} (cont. {ci + 1})"
            ax.text(0, 0.98, ttl, va="top", ha="left", fontsize=13, fontweight="bold",
                    color="#2f5d3a", transform=ax.transAxes)
            ax.text(0, 0.93, "\n".join(chunk), va="top", ha="left", fontsize=fontsize,
                    family="DejaVu Sans", color="#1f241f", transform=ax.transAxes, linespacing=1.3)
            figs.append(fig)
        return figs

    with PdfPages(path) as pdf:
        cover = (
            f"Evolution ID: {eda.get('evolution_id')}\n"
            f"Status: {eda.get('status')}  Model: {eda.get('llm_model')}\n"
            f"Planner: {eda.get('planner_id')}  Pop×gens: {eda.get('population_size')}×{eda.get('generations_cfg')}\n"
            f"Generated: {utcnow()}\n"
            f"(WeasyPrint fallback: {error[:120]})\n\n"
            f"GOAL\n{textwrap.fill(eda.get('goal') or '', 90)}\n\n"
            f"FINAL BEST: {(eda.get('final_best') or {}).get('id')}  "
            f"fitness={(eda.get('final_best') or {}).get('fitness')}"
        )
        for f in text_pages("Dev Studio · Evolution investigation", cover, 10):
            pdf.savefig(f); plt.close(f)

        for c in charts:
            p = Path(c["path"])
            if not p.exists():
                continue
            fig = plt.figure(figsize=(8.5, 11))
            fig.patch.set_facecolor("white")
            ax = fig.add_axes([0.06, 0.2, 0.88, 0.7])
            ax.axis("off")
            fig.text(0.06, 0.93, c["title"], fontsize=13, fontweight="bold", color="#2f5d3a")
            try:
                img = Image.open(p)
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, f"(could not load {p.name})", ha="center")
            pdf.savefig(fig); plt.close(fig)

        if narrative:
            # light markdown strip for headings readability
            plain = re.sub(r"^#+\s*", "", narrative, flags=re.M)
            plain = re.sub(r"\*\*(.+?)\*\*", r"\1", plain)
            plain = re.sub(r"`([^`]+)`", r"\1", plain)
            for f in text_pages("Investigative narrative", plain, 9):
                pdf.savefig(f); plt.close(f)

        md = render_markdown_report(data, eda, narrative=None)
        plain = re.sub(r"^#+\s*", "", md, flags=re.M)
        plain = re.sub(r"[|`*]", "", plain)
        for f in text_pages("Structured EDA", plain, 8):
            pdf.savefig(f); plt.close(f)

    return path


# ── Zips ─────────────────────────────────────────────────────────────────────


def write_full_zip(evolutions_root: Path, evo_id: str, out_path: Path) -> Path:
    """Zip everything under the evolution directory (traces, gens, calls, exports)."""
    root = run_dir(evolutions_root, evo_id)
    if not root.exists():
        raise FileNotFoundError(evo_id)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in root.rglob("*"):
            if p.is_file():
                # skip nesting other export zips of ourselves if re-exporting
                if p.resolve() == out_path.resolve():
                    continue
                zf.write(p, arcname=str(p.relative_to(root.parent)))
    return out_path


def write_bundle_zip(
    evolutions_root: Path,
    evo_id: str,
    out_path: Path,
    *,
    data: dict,
    eda: dict,
    narrative: Optional[str] = None,
    include_best_sources: bool = True,
) -> Path:
    """Zip transcripts + main artifacts + report (not full gen trees unless best)."""
    root = run_dir(evolutions_root, evo_id)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    md = render_markdown_report(data, eda, narrative=narrative)
    eda_path_name = "eda.json"
    report_name = "REPORT.md"
    if narrative:
        narrative_name = "NARRATIVE.md"
    else:
        narrative_name = None

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{evo_id}/{report_name}", md)
        zf.writestr(f"{evo_id}/{eda_path_name}", json.dumps(eda, indent=2))
        if narrative:
            zf.writestr(f"{evo_id}/NARRATIVE.md", narrative)

        # main evolution artifacts
        for name in (
            "evolution.json",
            "model-answers.md",
            "llm-calls.json",
            "charter.json",
            "prompt-bank.json",
            "goal-brief.md",
        ):
            p = root / name
            if p.exists():
                zf.write(p, arcname=f"{evo_id}/{name}")

        # all llm_calls traces
        calls_dir = root / "llm_calls"
        if calls_dir.exists():
            for p in calls_dir.rglob("*"):
                if p.is_file():
                    zf.write(p, arcname=f"{evo_id}/llm_calls/{p.relative_to(calls_dir)}")

        # PDF / HTML / charts if already generated
        exp = root / "exports"
        if exp.exists():
            for p in exp.glob("*.pdf"):
                zf.write(p, arcname=f"{evo_id}/exports/{p.name}")
            for p in exp.glob("*.html"):
                zf.write(p, arcname=f"{evo_id}/exports/{p.name}")
            charts = exp / "charts"
            if charts.exists():
                for p in charts.glob("*"):
                    if p.is_file():
                        zf.write(p, arcname=f"{evo_id}/exports/charts/{p.name}")

        # best candidate sources
        if include_best_sources:
            best = data.get("best") or {}
            bpath = best.get("path")
            if bpath and Path(bpath).exists():
                bp = Path(bpath)
                for p in bp.rglob("*"):
                    if p.is_file():
                        zf.write(p, arcname=f"{evo_id}/best_candidate/{p.relative_to(bp)}")

        # final_best summary json
        zf.writestr(
            f"{evo_id}/final_best.json",
            json.dumps(eda.get("final_best") or data.get("best") or {}, indent=2),
        )

    return out_path


def ensure_exports_dir(evolutions_root: Path, evo_id: str) -> Path:
    d = run_dir(evolutions_root, evo_id) / "exports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def generate_all_exports(
    evolutions_root: Path,
    evo_id: str,
    *,
    data: dict,
    narrative: Optional[str] = None,
    make_pdf: bool = True,
    make_full_zip: bool = True,
    make_bundle_zip: bool = True,
) -> dict[str, str]:
    """Write report artifacts under evolutions/<id>/exports/ and return relative paths."""
    root = run_dir(evolutions_root, evo_id)
    exp = ensure_exports_dir(evolutions_root, evo_id)
    eda = analyze_run(data, root=root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out: dict[str, str] = {}

    eda_path = exp / f"eda-{stamp}.json"
    eda_path.write_text(json.dumps(eda, indent=2), encoding="utf-8")
    # also stable name
    (exp / "eda-latest.json").write_text(json.dumps(eda, indent=2), encoding="utf-8")
    out["eda"] = str(eda_path.relative_to(root))

    md = render_markdown_report(data, eda, narrative=narrative)
    md_path = exp / f"REPORT-{stamp}.md"
    md_path.write_text(md, encoding="utf-8")
    (exp / "REPORT-latest.md").write_text(md, encoding="utf-8")
    out["report_md"] = str(md_path.relative_to(root))

    if narrative:
        npath = exp / f"NARRATIVE-{stamp}.md"
        npath.write_text(narrative, encoding="utf-8")
        (exp / "NARRATIVE-latest.md").write_text(narrative, encoding="utf-8")
        out["narrative"] = str(npath.relative_to(root))

    if make_pdf:
        charts_dir = exp / "charts"
        pdf_path = exp / f"REPORT-{stamp}.pdf"
        write_pdf_report(pdf_path, data, eda, narrative=narrative, charts_dir=charts_dir)
        # stable copy of PDF + HTML companion
        latest_pdf = exp / "REPORT-latest.pdf"
        latest_pdf.write_bytes(pdf_path.read_bytes())
        html_src = pdf_path.with_suffix(".html")
        if html_src.exists():
            (exp / "REPORT-latest.html").write_bytes(html_src.read_bytes())
            out["report_html"] = str(html_src.relative_to(root))
            out["report_html_latest"] = "exports/REPORT-latest.html"
        out["report_pdf"] = str(pdf_path.relative_to(root))
        out["report_pdf_latest"] = str(latest_pdf.relative_to(root))
        if charts_dir.exists():
            out["charts_dir"] = "exports/charts"

    if make_bundle_zip:
        bpath = exp / f"bundle-{stamp}.zip"
        write_bundle_zip(evolutions_root, evo_id, bpath, data=data, eda=eda, narrative=narrative)
        (exp / "bundle-latest.zip").write_bytes(bpath.read_bytes())
        out["bundle_zip"] = str(bpath.relative_to(root))
        out["bundle_zip_latest"] = "exports/bundle-latest.zip"

    if make_full_zip:
        fpath = exp / f"full-run-{stamp}.zip"
        write_full_zip(evolutions_root, evo_id, fpath)
        (exp / "full-run-latest.zip").write_bytes(fpath.read_bytes())
        out["full_zip"] = str(fpath.relative_to(root))
        out["full_zip_latest"] = "exports/full-run-latest.zip"

    manifest = {
        "evolution_id": evo_id,
        "created_at": utcnow(),
        "files": out,
        "has_narrative": bool(narrative),
    }
    (exp / "export-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    out["manifest"] = "exports/export-manifest.json"
    return out
