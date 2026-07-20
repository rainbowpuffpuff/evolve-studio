"""Multi-project (factory) management for the Dev Studio.

Each factory lives at dev-studio/projects/<slug>/ with:
  state.json       — factory goal, cells, order, environment, tools
  project.json     — metadata (name, description, template)
  costs.json       — running cost log for this factory
  notes.json       — bug annotations and open notes
  generated/       — generated cell images
  audio/raw/       — audio clips
  audio/manifest.json
  video/           — rendered videos
  skill/           — factory skill package (SKILL.md + references/)
"""
from __future__ import annotations

import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "project"


DEFAULT_HELLO_WORLD_CELLS = [
    {
        "id": "C0",
        "role": "planner",
        "name": "Planner",
        "goal": "Design a tiny containerized hello-world web app with a single endpoint, tests, and a Dockerfile",
        "params": {"app_name": "hello_world", "framework": "fastapi", "port": 8000},
        "environment": "docker",
        "tools": ["python3", "pip", "docker"],
        "enabled": True,
        "status": "ready",
    },
    {
        "id": "C1",
        "role": "backend",
        "name": "Backend",
        "goal": "Implement the FastAPI app and tests in the project directory",
        "params": {"entry": "main.py", "test": "test_main.py"},
        "environment": "docker",
        "tools": ["python3", "fastapi", "uvicorn", "pytest", "httpx"],
        "enabled": True,
        "status": "ready",
    },
    {
        "id": "C2",
        "role": "frontend",
        "name": "Frontend",
        "goal": "Create a minimal static index.html that fetches /hello and shows the response",
        "params": {"api_path": "/hello"},
        "environment": "browser",
        "tools": ["html", "javascript", "fetch"],
        "enabled": False,
        "status": "ready",
    },
    {
        "id": "C3",
        "role": "scraper",
        "name": "Scraper",
        "goal": "Stub a scraper cell that can fetch a remote status page if enabled",
        "params": {"target_url": "https://example.com", "interval_minutes": 60},
        "environment": "local",
        "tools": ["requests", "curl"],
        "enabled": False,
        "status": "ready",
    },
]


class ProjectManager:
    def __init__(self, projects_root: Path, comic_pages: Optional[Path] = None,
                 cast: Optional[Path] = None, style: Optional[Path] = None):
        self.root = projects_root
        self.root.mkdir(parents=True, exist_ok=True)
        self.comic_pages = comic_pages or Path()
        self.cast = cast or Path()
        self.style = style or Path()

    def _pdir(self, pid: str) -> Path:
        return self.root / pid

    def list_projects(self) -> list[dict]:
        out = []
        for d in sorted(self.root.iterdir()):
            if not d.is_dir():
                continue
            meta = self._load_meta(d.name)
            if meta:
                out.append(meta)
        return out

    def _load_meta(self, pid: str) -> Optional[dict]:
        p = self._pdir(pid) / "project.json"
        if not p.exists():
            return None
        meta = json.loads(p.read_text())
        meta["id"] = pid
        state = self.load_state(pid)
        meta["cell_count"] = len(state.get("cells", []))
        costs = self._load_costs(pid)
        meta["total_cost_usd"] = costs.get("total_usd", 0)
        return meta

    def get_project(self, pid: str) -> dict:
        meta = self._load_meta(pid)
        if not meta:
            raise HTTPException(404, f"factory {pid} not found")
        return meta

    def create_project(self, name: str, description: str = "", template: str = "blank",
                       goal: str = "") -> dict:
        pid = slugify(name)
        base = pid
        i = 2
        while self._pdir(pid).exists():
            pid = f"{base}-{i}"
            i += 1
        d = self._pdir(pid)
        d.mkdir(parents=True, exist_ok=True)
        (d / "generated").mkdir(parents=True, exist_ok=True)
        (d / "audio" / "raw").mkdir(parents=True, exist_ok=True)
        (d / "video").mkdir(parents=True, exist_ok=True)

        template_path = Path(__file__).resolve().parent.parent / "templates" / f"{template}.json"
        tpl: dict = {"meta": {}, "state": {}}
        if template_path.exists():
            try:
                tpl = json.loads(template_path.read_text())
            except Exception:
                pass

        meta = {
            "id": pid,
            "name": name,
            "description": description,
            "template": template,
            "type": tpl.get("meta", {}).get("type", "factory"),
            "created_at": utcnow(),
            "goal": goal,
        }
        (d / "project.json").write_text(json.dumps(meta, indent=2))

        tpl_state = tpl.get("state", {})
        cells = [dict(c) for c in tpl_state.get("cells", [])]
        state = {
            "title": name,
            "goal": goal,
            "description": description,
            "type": meta["type"],
            "template": template,
            "cells": cells,
            "order": [c["id"] for c in cells],
            "environment": tpl_state.get("environment", ["local"]),
            "tools": tpl_state.get("tools", ["git", "python3"]),
        }
        (d / "state.json").write_text(json.dumps(state, indent=2))
        (d / "costs.json").write_text(json.dumps({"project": pid, "entries": [], "total_usd": 0}, indent=2))
        (d / "notes.json").write_text(json.dumps({"project": pid, "notes": []}, indent=2))
        (d / "audio" / "manifest.json").write_text(json.dumps({"audio": [], "video": []}, indent=2))
        return self._load_meta(pid)

    def delete_project(self, pid: str) -> None:
        d = self._pdir(pid)
        if d.exists():
            shutil.rmtree(d)

    # ── state ──
    def _state_path(self, pid: str) -> Path:
        return self._pdir(pid) / "state.json"

    def load_state(self, pid: str) -> dict:
        p = self._state_path(pid)
        if not p.exists():
            raise HTTPException(404, f"state.json missing for {pid}")
        return json.loads(p.read_text())

    def save_state(self, pid: str, state: dict) -> None:
        self._state_path(pid).write_text(json.dumps(state, indent=2))

    def cell_by_id(self, pid: str, cell_id: str) -> dict:
        state = self.load_state(pid)
        for c in state["cells"]:
            if c["id"] == cell_id:
                return c
        raise HTTPException(404, f"cell {cell_id} not found in {pid}")

    # ── media paths ──
    def generated(self, pid: str) -> Path:
        return self._pdir(pid) / "generated"

    def video_out(self, pid: str) -> Path:
        return self._pdir(pid) / "video"

    def audio_raw(self, pid: str) -> Path:
        return self._pdir(pid) / "audio" / "raw"

    def audio_manifest(self, pid: str) -> Path:
        return self._pdir(pid) / "audio" / "manifest.json"

    def load_audio_manifest(self, pid: str) -> dict:
        p = self.audio_manifest(pid)
        if not p.exists():
            return {"audio": [], "video": []}
        return json.loads(p.read_text())

    def save_audio_manifest(self, pid: str, man: dict) -> None:
        self.audio_manifest(pid).write_text(json.dumps(man, indent=2))

    def audio_local_path(self, pid: str, asset_id: str) -> Optional[Path]:
        for p in self.audio_raw(pid).iterdir():
            if p.name.startswith(asset_id + "_"):
                return p
        return None

    def page_path_for(self, pid: str, cell: dict) -> Path:
        """Prefer generated override, then any cell page_file hint."""
        gen = self.generated(pid) / f"{cell['id']}.png"
        if gen.exists():
            return gen
        pf = cell.get("page_file")
        if pf:
            p = self.comic_pages / pf
            if p.exists():
                return p
        return gen

    # ── costs ──
    def _load_costs(self, pid: str) -> dict:
        p = self._pdir(pid) / "costs.json"
        if not p.exists():
            return {"project": pid, "entries": [], "total_usd": 0}
        return json.loads(p.read_text())

    def add_cost(self, pid: str, entry: dict) -> dict:
        costs = self._load_costs(pid)
        entry["timestamp"] = utcnow()
        entry["id"] = uuid.uuid4().hex[:12]
        costs.setdefault("entries", []).append(entry)
        costs["total_usd"] = round(sum(e.get("cost_usd", 0) for e in costs["entries"]), 6)
        (self._pdir(pid) / "costs.json").write_text(json.dumps(costs, indent=2))
        return entry

    def get_costs(self, pid: str) -> dict:
        return self._load_costs(pid)

    # ── notes (bug annotations) ──
    def notes_path(self, pid: str) -> Path:
        return self._pdir(pid) / "notes.json"

    NOTE_STALE_DAYS = 7

    def load_notes(self, pid: str) -> list[dict]:
        p = self.notes_path(pid)
        if not p.exists():
            return []
        notes = json.loads(p.read_text()).get("notes", [])
        if self._auto_resolve_stale(notes):
            self.save_notes(pid, notes)
        return notes

    def _auto_resolve_stale(self, notes: list[dict]) -> bool:
        now = datetime.now(timezone.utc)
        changed = False
        for n in notes:
            if n.get("status") != "open":
                continue
            ts = n.get("updated_at") or n.get("created_at")
            if not ts:
                continue
            try:
                t = datetime.fromisoformat(ts)
            except Exception:
                continue
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            if (now - t).days >= self.NOTE_STALE_DAYS:
                n["status"] = "resolved"
                n["auto_resolved"] = True
                n["updated_at"] = utcnow()
                changed = True
        return changed

    def save_notes(self, pid: str, notes: list[dict]) -> None:
        path = self.notes_path(pid)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"project": pid, "notes": notes}, indent=2))
        tmp.replace(path)

    def add_note(self, pid: str, note: dict) -> dict:
        notes = self.load_notes(pid)
        note["id"] = uuid.uuid4().hex[:10]
        note["created_at"] = utcnow()
        note["status"] = note.get("status", "open")
        note.setdefault("images", [])
        notes.append(note)
        self.save_notes(pid, notes)
        return note

    def update_note(self, pid: str, nid: str, patch: dict) -> Optional[dict]:
        notes = self.load_notes(pid)
        for n in notes:
            if n["id"] == nid:
                n.update({k: v for k, v in patch.items() if k in ("note", "status", "severity", "images")})
                n["updated_at"] = utcnow()
                self.save_notes(pid, notes)
                return n
        return None

    def delete_note(self, pid: str, nid: str) -> bool:
        notes = self.load_notes(pid)
        new = [n for n in notes if n["id"] != nid]
        if len(new) == len(notes):
            return False
        self.save_notes(pid, new)
        return True

    def open_notes(self, pid: str) -> list[dict]:
        return [n for n in self.load_notes(pid) if n.get("status") != "resolved"]

    # ── skill package ──
    def skill_dir(self, pid: str) -> Path:
        d = self._pdir(pid) / "skill"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def skill_path(self, pid: str) -> Path:
        return self.skill_dir(pid) / "SKILL.md"

    def references_dir(self, pid: str) -> Path:
        d = self.skill_dir(pid) / "references"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def load_skill(self, pid: str) -> str:
        p = self.skill_path(pid)
        if not p.exists():
            return ""
        return p.read_text(encoding="utf-8")

    def save_skill(self, pid: str, text: str) -> None:
        self.skill_path(pid).write_text(text, encoding="utf-8")

    def list_references(self, pid: str) -> list[dict]:
        d = self.references_dir(pid)
        out = []
        for p in sorted(d.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not p.is_file():
                continue
            try:
                stat = p.stat()
                snippet = p.read_text(encoding="utf-8", errors="ignore")[:240]
                out.append({
                    "name": p.name,
                    "size": stat.st_size,
                    "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                    "snippet": snippet,
                })
            except Exception:
                pass
        return out

    def load_reference(self, pid: str, name: str) -> str:
        p = self.references_dir(pid) / Path(name).name
        if not p.exists():
            raise HTTPException(404, f"reference {name} not found")
        return p.read_text(encoding="utf-8")

    def save_reference(self, pid: str, name: str, text: str) -> None:
        p = self.references_dir(pid) / Path(name).name
        p.write_text(text, encoding="utf-8")

    def delete_reference(self, pid: str, name: str) -> None:
        p = self.references_dir(pid) / Path(name).name
        if not p.exists():
            raise HTTPException(404, f"reference {name} not found")
        p.unlink()

    def build_skill_context(self, pid: str, max_chars: int = 15000) -> str:
        skill_text = self.load_skill(pid)
        if not skill_text:
            return ""
        header = "## Factory skill context\n\n"
        budget = max(0, max_chars - len(header))
        chunks = []
        skill_chunk = f"### SKILL.md\n{skill_text}\n\n"
        if len(skill_chunk) > budget:
            skill_chunk = skill_chunk[:budget - 3] + "..." if budget > 3 else ""
        chunks.append(skill_chunk)
        budget -= len(chunks[-1])

        refs = []
        ref_dir = self.references_dir(pid)
        for suffix in ("*.md", "*.txt", "*.json"):
            refs.extend(ref_dir.glob(suffix))
        refs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        for ref in refs:
            if budget <= 0:
                break
            try:
                txt = ref.read_text(encoding="utf-8", errors="ignore")
                label = ref.suffix.lstrip(".") if ref.suffix.lower() != ".md" else ref.name
                ref_chunk = f"### {ref.name} ({label})\n{txt}\n\n"
                if len(ref_chunk) > budget:
                    ref_chunk = ref_chunk[:budget - 3] + "..." if budget > 3 else ""
                chunks.append(ref_chunk)
                budget -= len(ref_chunk)
            except Exception:
                pass

        if not chunks or not chunks[0]:
            return ""
        return header + "".join(chunks)

    def ensure_skill_defaults(self, pid: str, state: Optional[dict] = None) -> None:
        d = self.skill_dir(pid)
        refs = self.references_dir(pid)
        if state is None:
            try:
                state = self.load_state(pid)
            except Exception:
                state = {}

        meta = self._load_meta(pid) or {}
        factory_name = state.get("title") or meta.get("name") or pid
        goal = state.get("goal") or meta.get("goal") or ""
        description = state.get("description") or meta.get("description") or ""

        def _keywords(text: str) -> list[str]:
            words = [w.lower() for w in re.findall(r"[a-zA-Z0-9]+", text or "") if len(w) > 3 and w.lower() not in ("with", "from", "that", "this", "into", "your", "their", "they", "will", "should", "could", "would")]
            seen = set()
            out = []
            for w in words:
                if w not in seen:
                    seen.add(w); out.append(w)
            return out[:5]

        triggers = [factory_name] + _keywords(goal or description)

        cells = state.get("cells", [])
        order = state.get("order", [c["id"] for c in cells])
        ordered = []
        by_id = {c["id"]: c for c in cells}
        for cid in order:
            if cid in by_id:
                ordered.append(by_id[cid])
        for c in cells:
            if c["id"] not in {x["id"] for x in ordered}:
                ordered.append(c)

        envs = state.get("environment", ["local"])
        tools = state.get("tools", ["git", "python3"])
        deployment_target = state.get("deployment_target")
        budget_usd = state.get("budget_usd")
        providers = state.get("providers", [])
        mcp_servers = state.get("mcp_servers", [])

        open_notes = self.open_notes(pid)
        anti_notes = [n for n in open_notes if n.get("severity") in ("bug", "nit")]
        anti_patterns = []
        for n in anti_notes:
            note_text = n.get("note", "").strip()
            if not note_text:
                continue
            # Use the first sentence as a concise summary, keep the full note for context
            first_sentence = note_text.split(".")[0].strip()
            if len(note_text) > 200:
                note_text = note_text[:200] + "..."
            anti_patterns.append(f"- **[{n.get('severity','bug').upper()} {n.get('id','')}] {first_sentence}** — {note_text}")

        anti_section = "\n".join(anti_patterns) if anti_patterns else "- None recorded yet."

        contract_rules = []
        for c in ordered:
            enabled = "enabled" if c.get("enabled", True) else "disabled"
            contract_rules.append(f"- Cell `{c['id']}` ({c.get('role', '')} / {c.get('name', '')}, {enabled}): {c.get('goal', '')}")
        contract_rules.append(f"- Environments: {', '.join(envs)}")
        contract_rules.append(f"- Tools: {', '.join(tools)}")
        if deployment_target:
            contract_rules.append(f"- Deployment target: {deployment_target}")
        if budget_usd is not None:
            contract_rules.append(f"- Budget USD: {budget_usd}")
        if providers:
            contract_rules.append(f"- Preferred providers: {', '.join(providers)}")
        if mcp_servers:
            contract_rules.append(f"- MCP servers: {', '.join(mcp_servers)}")

        skill_md = f"""---
name: {factory_name}
version: 0.1.0
description: {goal}
triggers: {json.dumps(triggers)}
---

## Purpose
{goal}

## When to use
Use this factory when building `{factory_name}` — a {description or goal or 'software project'}.

## Reference files
- `references/factory-state.md` — goal, environment, tools, and cells
- `references/anti-patterns.md` — rules to avoid
- `references/architecture.md` — high-level architecture
- `references/mcp-servers.md` — available MCP servers and sample calls

## Modes
- **plan** — reason about the factory goal and cell layout
- **scaffold** — generate the minimal project skeleton
- **build** — implement cells in order, using the allowed tools and environments; use MCP tools when a cell's tools include `mcp`
- **test** — run any tests or checks defined by the cells
- **deploy** — produce runnable artifacts and deploy to `{deployment_target or 'the chosen target'}`
- **fix** — consult anti-patterns and open notes, then make the smallest correct change

## Deployment & operations
- Target environment: `{deployment_target or 'not specified'}`
- Budget USD: `{budget_usd if budget_usd is not None else 'not set'}`
- Preferred providers: `{', '.join(providers) if providers else 'any'}`
- MCP servers available: `{', '.join(mcp_servers) if mcp_servers else 'none configured'}`

## Contract / rules
{chr(10).join(contract_rules)}

## Workflow
1. Read `references/factory-state.md` for the current goal and cells.
2. Respect each cell's role, environment, tools, and `enabled` flag.
3. Run enabled cells in order.
4. Before output, check `references/anti-patterns.md`.
5. Hand off completed artifacts to the next stage.

## Output handoff
Generate the files and instructions needed to run `{factory_name}` in the target environment. Prefer small, tested changes over large refactors.

## Anti-patterns
{anti_section}
""".strip() + "\n"

        self.save_skill(pid, skill_md)

        # reference: factory state snapshot
        state_md = f"""# Factory state snapshot

**Goal:** {goal}

**Environments:** {', '.join(envs)}

**Tools:** {', '.join(tools)}

**Cells ({len(ordered)}):**
""".strip() + "\n\n"
        for c in ordered:
            state_md += f"- `{c['id']}` ({c.get('role','')} / {c.get('name','')}) — {c.get('goal','')}\n"
            state_md += f"  - environment: {c.get('environment','local')}\n"
            state_md += f"  - tools: {', '.join(c.get('tools', []))}\n"
            state_md += f"  - enabled: {c.get('enabled', True)}\n"
            params = c.get("params")
            if params:
                state_md += f"  - params: {json.dumps(params, indent=2)}\n"

        # reference: anti-patterns
        anti_md = "# Anti-patterns\n\n"
        if anti_patterns:
            anti_md += "Rules derived from open bug/nit notes:\n\n" + "\n".join(anti_patterns) + "\n"
        else:
            anti_md += "No active anti-patterns.\n"

        # reference: architecture
        arch = ["# Architecture\n"]
        arch.append(f"Factory `{factory_name}` is built from {len(ordered)} ordered cells:")
        arch.append("")
        for c in ordered:
            arch.append(f"- **{c['id']}** `{c.get('role','')}` — {c.get('goal','')} ({'enabled' if c.get('enabled', True) else 'disabled'})")
        if envs:
            arch.append(f"\nEnvironments: {', '.join(envs)}")
        if tools:
            arch.append(f"Tools: {', '.join(tools)}")
        arch.append(f"\nDeployment target: `{deployment_target or 'not set'}`")
        if budget_usd is not None:
            arch.append(f"Budget USD: {budget_usd}")
        if providers:
            arch.append(f"Preferred providers: {', '.join(providers)}")
        if mcp_servers:
            arch.append(f"MCP servers: {', '.join(mcp_servers)}")
        arch.append("\nThe build flows through enabled cells in the listed order, then produces deployment artifacts.")
        arch_md = "\n".join(arch) + "\n"

        mcp_md = "# MCP servers\n\n"
        if mcp_servers:
            mcp_md += "The following MCP servers may be used by cells with `tools: [\"mcp\"]`:\n\n"
            for s in mcp_servers:
                mcp_md += f"- `{s}` — call via `POST /api/mcp/{s}/call` from the Dev Studio backend.\n"
        else:
            mcp_md += "No MCP servers configured. Add them to `dev-studio/mcp.json` or `~/.cursor/mcp.json`.\n"

        (refs / "factory-state.md").write_text(state_md, encoding="utf-8")
        (refs / "anti-patterns.md").write_text(anti_md, encoding="utf-8")
        (refs / "architecture.md").write_text(arch_md, encoding="utf-8")
        (refs / "mcp-servers.md").write_text(mcp_md, encoding="utf-8")
