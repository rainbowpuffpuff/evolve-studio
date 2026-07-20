"""Continuous Cerebras (Gemma) maintainer loop for Dev Studio.

Philosophy:
  - Grok (or a human) is the coordinator: high-level goals only.
  - Gemma runs cheaply in the background: digests traces, evolves prompt banks,
    files safe studio tasks, and keeps products improving without manual theater.

The maintainer keeps a durable *mind* you can inspect and edit:
  - goals / mission
  - active plans
  - memories (add / remove / edit)
  - learnings from each analysis tick
  - tasks + activity log

Persists under DATA_DIR/maintainer/.
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from lib import llm
from lib import trace_digest


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


DEFAULT_MODEL = os.environ.get("MAINTAINER_MODEL", "gemma-4-31b")
DEFAULT_INTERVAL = float(os.environ.get("MAINTAINER_INTERVAL_SECS", "90"))

DEFAULT_MIND: dict[str, Any] = {
    "version": 1,
    "mission": (
        "Serve the DEPLOYER: continuously improve evolutions so products are shippable and "
        "monetizable (Stripe/checkout, ads, affiliates, portfolio cross-links). Recycle factory "
        "outputs into revenue-ready surfaces. Digest traces with Gemma, evolve prompts, file "
        "tasks. Escalate only coordinator-level decisions."
    ),
    "goals": [
        {
            "id": "g-sense",
            "title": "Make every evolution run legible",
            "detail": "Never leave humans staring at 80 raw chats — always maintain digests + learnings.",
            "status": "active",
            "priority": "high",
            "created_at": None,
        },
        {
            "id": "g-money",
            "title": "Every product has a money path",
            "detail": "Reject or patch candidates without Stripe/ads/affiliate/portfolio CTAs; raise monetization scores.",
            "status": "active",
            "priority": "high",
            "created_at": None,
        },
        {
            "id": "g-prompt",
            "title": "Self-improve prompt bank from traces",
            "detail": "Absorb failure modes into create/build/evaluate/director prompts (esp. weak monetization).",
            "status": "active",
            "priority": "high",
            "created_at": None,
        },
        {
            "id": "g-product",
            "title": "Push product HTML toward ship + revenue each gen",
            "detail": "Track product gaps: authenticity, volume, payments; file concrete next-step tasks.",
            "status": "active",
            "priority": "med",
            "created_at": None,
        },
    ],
    "plans": [
        {
            "id": "p-loop",
            "title": "Background analysis loop",
            "steps": [
                "Pick run with new activity",
                "Structural digest (free)",
                "Gemma narrative + prompt patches + studio tasks",
                "Flag missing monetization / weak authenticity",
                "Absorb learnings into memory",
                "Update plan focus for next ticks",
            ],
            "status": "active",
            "focus": "Digest runs; enforce deployer monetization lens",
            "created_at": None,
            "updated_at": None,
        }
    ],
    "focus": "Watch evolutions for shippability + monetization; absorb failures into memory.",
    "notes": "Deployer lens: innovation + volume + authenticity + money path on every product.",
    "updated_at": None,
}


class MaintainerService:
    def __init__(self, data_dir: Path, evolutions_root: Path):
        self.data_dir = Path(data_dir)
        self.evolutions_root = Path(evolutions_root)
        self.root = self.data_dir / "maintainer"
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.tasks_path = self.root / "tasks.jsonl"
        self.log_path = self.root / "log.jsonl"
        self.memories_path = self.root / "memories.json"
        self.mind_path = self.root / "mind.json"
        self.learnings_path = self.root / "learnings.jsonl"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._state = self._load_state()
        self._mind = self._load_mind()
        self._memories = self._load_memories()
        # stamp default goal timestamps once
        changed = False
        for g in self._mind.get("goals") or []:
            if not g.get("created_at"):
                g["created_at"] = utcnow()
                changed = True
        for p in self._mind.get("plans") or []:
            if not p.get("created_at"):
                p["created_at"] = utcnow()
                p["updated_at"] = p["created_at"]
                changed = True
        if changed:
            self._save_mind()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load_state(self) -> dict[str, Any]:
        if self.state_path.is_file():
            try:
                return json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "enabled": os.environ.get("MAINTAINER_ENABLED", "1") not in ("0", "false", "no"),
            "interval_secs": DEFAULT_INTERVAL,
            "model": DEFAULT_MODEL,
            "last_tick_at": None,
            "last_error": None,
            "last_result": None,
            "ticks": 0,
            "analyzed_runs": {},
            "auto_apply_prompt_patches": True,
            "prefer_running_runs": True,
            "absorb_learnings": True,
        }

    def _save_state(self) -> None:
        self.state_path.write_text(json.dumps(self._state, indent=2), encoding="utf-8")

    def _load_mind(self) -> dict[str, Any]:
        if self.mind_path.is_file():
            try:
                data = json.loads(self.mind_path.read_text(encoding="utf-8"))
                # merge missing default keys (don't wipe user edits)
                out = dict(DEFAULT_MIND)
                out.update(data)
                # Ensure deployer monetization goal exists
                goals = list(out.get("goals") or [])
                if not any(g.get("id") == "g-money" for g in goals):
                    for dg in DEFAULT_MIND.get("goals") or []:
                        if dg.get("id") == "g-money":
                            g = dict(dg)
                            g["created_at"] = utcnow()
                            goals.insert(0, g)
                            break
                    out["goals"] = goals
                return out
            except Exception:
                pass
        mind = json.loads(json.dumps(DEFAULT_MIND))
        mind["updated_at"] = utcnow()
        return mind

    def _save_mind(self) -> None:
        self._mind["updated_at"] = utcnow()
        self.mind_path.write_text(json.dumps(self._mind, indent=2), encoding="utf-8")

    def _load_memories(self) -> list[dict[str, Any]]:
        if self.memories_path.is_file():
            try:
                data = json.loads(self.memories_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return data
                if isinstance(data, dict) and isinstance(data.get("memories"), list):
                    return data["memories"]
            except Exception:
                pass
        return []

    def _save_memories(self) -> None:
        self.memories_path.write_text(
            json.dumps({"version": 1, "updated_at": utcnow(), "memories": self._memories}, indent=2),
            encoding="utf-8",
        )

    def _append_learning(self, row: dict[str, Any]) -> None:
        row = {"ts": utcnow(), **row}
        with self.learnings_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def list_learnings(self, limit: int = 40) -> list[dict[str, Any]]:
        if not self.learnings_path.is_file():
            return []
        rows = []
        for line in self.learnings_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        return rows[-limit:]

    def list_log(self, limit: int = 80) -> list[dict[str, Any]]:
        if not self.log_path.is_file():
            return []
        rows = []
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        return rows[-limit:]

    # ── public mind / memory API ─────────────────────────────────────────────

    def get_mind(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._mind))

    def update_mind(self, patch: dict[str, Any]) -> dict[str, Any]:
        """Patch mission, focus, notes; replace goals/plans if provided as lists."""
        with self._lock:
            for k in ("mission", "focus", "notes"):
                if k in patch and patch[k] is not None:
                    self._mind[k] = str(patch[k])
            if isinstance(patch.get("goals"), list):
                self._mind["goals"] = patch["goals"]
            if isinstance(patch.get("plans"), list):
                self._mind["plans"] = patch["plans"]
            self._save_mind()
            mind = json.loads(json.dumps(self._mind))
        self._log("mind", "mind updated", keys=list(patch.keys()))
        return mind

    def add_goal(self, title: str, detail: str = "", priority: str = "med") -> dict[str, Any]:
        goal = {
            "id": f"g-{uuid.uuid4().hex[:8]}",
            "title": (title or "").strip() or "Untitled goal",
            "detail": (detail or "").strip(),
            "status": "active",
            "priority": priority or "med",
            "created_at": utcnow(),
        }
        with self._lock:
            goals = list(self._mind.get("goals") or [])
            goals.append(goal)
            self._mind["goals"] = goals
            self._save_mind()
        self._log("mind", f"goal added: {goal['title']}", goal_id=goal["id"])
        return goal

    def update_goal(self, goal_id: str, **fields) -> Optional[dict[str, Any]]:
        with self._lock:
            goals = list(self._mind.get("goals") or [])
            found = None
            for g in goals:
                if g.get("id") == goal_id:
                    for k in ("title", "detail", "status", "priority"):
                        if k in fields and fields[k] is not None:
                            g[k] = fields[k]
                    g["updated_at"] = utcnow()
                    found = dict(g)
                    break
            if not found:
                return None
            self._mind["goals"] = goals
            self._save_mind()
        return found

    def remove_goal(self, goal_id: str) -> bool:
        with self._lock:
            goals = list(self._mind.get("goals") or [])
            new = [g for g in goals if g.get("id") != goal_id]
            if len(new) == len(goals):
                return False
            self._mind["goals"] = new
            self._save_mind()
        self._log("mind", f"goal removed: {goal_id}")
        return True

    def add_plan(self, title: str, steps: Optional[list] = None, focus: str = "") -> dict[str, Any]:
        plan = {
            "id": f"p-{uuid.uuid4().hex[:8]}",
            "title": (title or "").strip() or "Untitled plan",
            "steps": steps if isinstance(steps, list) else [],
            "status": "active",
            "focus": (focus or "").strip(),
            "created_at": utcnow(),
            "updated_at": utcnow(),
        }
        with self._lock:
            plans = list(self._mind.get("plans") or [])
            plans.append(plan)
            self._mind["plans"] = plans
            self._save_mind()
        self._log("mind", f"plan added: {plan['title']}", plan_id=plan["id"])
        return plan

    def update_plan(self, plan_id: str, **fields) -> Optional[dict[str, Any]]:
        with self._lock:
            plans = list(self._mind.get("plans") or [])
            found = None
            for p in plans:
                if p.get("id") == plan_id:
                    for k in ("title", "status", "focus"):
                        if k in fields and fields[k] is not None:
                            p[k] = fields[k]
                    if "steps" in fields and fields["steps"] is not None:
                        p["steps"] = fields["steps"] if isinstance(fields["steps"], list) else p.get("steps")
                    p["updated_at"] = utcnow()
                    found = dict(p)
                    break
            if not found:
                return None
            self._mind["plans"] = plans
            self._save_mind()
        return found

    def remove_plan(self, plan_id: str) -> bool:
        with self._lock:
            plans = list(self._mind.get("plans") or [])
            new = [p for p in plans if p.get("id") != plan_id]
            if len(new) == len(plans):
                return False
            self._mind["plans"] = new
            self._save_mind()
        self._log("mind", f"plan removed: {plan_id}")
        return True

    def list_memories(self, *, kind: Optional[str] = None, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            mems = list(self._memories)
        if kind:
            mems = [m for m in mems if (m.get("kind") or "") == kind]
        return mems[-limit:]

    def add_memory(
        self,
        content: str,
        *,
        kind: str = "note",
        tags: Optional[list] = None,
        source: str = "user",
        evolution_id: Optional[str] = None,
        meta: Optional[dict] = None,
    ) -> dict[str, Any]:
        content = (content or "").strip()
        if not content:
            raise ValueError("memory content required")
        mem = {
            "id": f"m-{uuid.uuid4().hex[:10]}",
            "ts": utcnow(),
            "kind": (kind or "note").strip() or "note",
            "content": content,
            "tags": list(tags or []),
            "source": source or "user",
            "evolution_id": evolution_id,
            "meta": meta or {},
        }
        with self._lock:
            self._memories.append(mem)
            # cap store
            if len(self._memories) > 500:
                self._memories = self._memories[-500:]
            self._save_memories()
        self._log("memory", f"memory added ({mem['kind']})", memory_id=mem["id"])
        return mem

    def update_memory(self, memory_id: str, **fields) -> Optional[dict[str, Any]]:
        with self._lock:
            found = None
            for m in self._memories:
                if m.get("id") == memory_id:
                    for k in ("content", "kind", "source"):
                        if k in fields and fields[k] is not None:
                            m[k] = fields[k]
                    if "tags" in fields and fields["tags"] is not None:
                        m["tags"] = list(fields["tags"]) if isinstance(fields["tags"], list) else m.get("tags")
                    m["updated_at"] = utcnow()
                    found = dict(m)
                    break
            if not found:
                return None
            self._save_memories()
        return found

    def remove_memory(self, memory_id: str) -> bool:
        with self._lock:
            before = len(self._memories)
            self._memories = [m for m in self._memories if m.get("id") != memory_id]
            if len(self._memories) == before:
                return False
            self._save_memories()
        self._log("memory", f"memory removed: {memory_id}")
        return True

    def clear_memories(self, *, kind: Optional[str] = None, source: Optional[str] = None) -> int:
        with self._lock:
            before = len(self._memories)
            kept = []
            for m in self._memories:
                if kind and m.get("kind") != kind:
                    kept.append(m)
                    continue
                if source and m.get("source") != source:
                    kept.append(m)
                    continue
                if not kind and not source:
                    continue  # clear all
                if kind and m.get("kind") == kind:
                    continue
                if source and m.get("source") == source:
                    continue
                kept.append(m)
            if not kind and not source:
                kept = []
            removed = before - len(kept)
            self._memories = kept
            self._save_memories()
        self._log("memory", f"cleared {removed} memories", kind=kind, source=source)
        return removed

    def update_task_status(self, task_id: str, status: str) -> bool:
        """Rewrite tasks.jsonl with updated status for matching id (last wins)."""
        rows = self.list_tasks(limit=5000)
        found = False
        out = []
        for r in rows:
            if r.get("id") == task_id:
                r = dict(r)
                r["status"] = status
                r["updated_at"] = utcnow()
                found = True
            out.append(r)
        if not found:
            return False
        with self.tasks_path.open("w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        self._log("task", f"task {task_id} → {status}")
        return True

    def snapshot(self) -> dict[str, Any]:
        """Full monitor payload for the UI."""
        with self._lock:
            st = dict(self._state)
            mind = json.loads(json.dumps(self._mind))
            memories = list(self._memories)
            running = bool(self._thread and self._thread.is_alive())
        return {
            "ok": True,
            "status": {
                **st,
                "running": running,
                "tasks_pending": self._count_tasks(status="pending"),
                "tasks_total": self._count_tasks(),
                "memories_count": len(memories),
                "goals_count": len(mind.get("goals") or []),
                "plans_count": len(mind.get("plans") or []),
                "root": str(self.root),
            },
            "mind": mind,
            "memories": memories[-200:],
            "learnings": self.list_learnings(40),
            "tasks": self.list_tasks(60),
            "log": self.list_log(60),
            "memory_kinds": sorted({m.get("kind") or "note" for m in memories}),
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            st = dict(self._state)
            st["running"] = bool(self._thread and self._thread.is_alive())
            st["tasks_pending"] = self._count_tasks(status="pending")
            st["tasks_total"] = self._count_tasks()
            st["memories_count"] = len(self._memories)
            st["goals_count"] = len((self._mind.get("goals") or []))
            st["plans_count"] = len((self._mind.get("plans") or []))
            st["focus"] = self._mind.get("focus")
            st["mission"] = self._mind.get("mission")
            st["root"] = str(self.root)
            return st

    def configure(self, **kwargs) -> dict[str, Any]:
        with self._lock:
            for k in (
                "enabled", "interval_secs", "model",
                "auto_apply_prompt_patches", "prefer_running_runs",
                "absorb_learnings",
            ):
                if k in kwargs and kwargs[k] is not None:
                    self._state[k] = kwargs[k]
            if "interval_secs" in kwargs and kwargs["interval_secs"] is not None:
                self._state["interval_secs"] = max(30.0, float(kwargs["interval_secs"]))
            self._save_state()
            enabled = bool(self._state.get("enabled"))
        if enabled:
            self.start()
        else:
            self.stop()
        return self.status()

    def start(self) -> dict[str, Any]:
        with self._lock:
            self._state["enabled"] = True
            self._save_state()
            if self._thread and self._thread.is_alive():
                return self.status()
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="studio-maintainer", daemon=True)
            self._thread.start()
        self._log("status", "maintainer started")
        return self.status()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        with self._lock:
            self._state["enabled"] = False
            self._save_state()
        self._log("status", "maintainer stop requested")
        return self.status()

    def _log(self, kind: str, message: str, **extra) -> None:
        row = {"ts": utcnow(), "kind": kind, "message": message, **extra}
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except Exception:
            pass

    def _count_tasks(self, status: Optional[str] = None) -> int:
        if not self.tasks_path.is_file():
            return 0
        n = 0
        try:
            for line in self.tasks_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                if status is None:
                    n += 1
                else:
                    try:
                        if json.loads(line).get("status") == status:
                            n += 1
                    except Exception:
                        pass
        except Exception:
            return 0
        return n

    def list_tasks(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.tasks_path.is_file():
            return []
        rows = []
        for line in self.tasks_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        return rows[-limit:]

    def append_task(self, task: dict[str, Any]) -> None:
        task = {
            "ts": utcnow(),
            "status": task.get("status") or "pending",
            "source": task.get("source") or "maintainer",
            **task,
        }
        # avoid exact-id duplicates still pending
        existing = self.list_tasks(limit=500)
        for e in existing:
            if e.get("id") == task.get("id") and e.get("status") == "pending":
                return
        with self.tasks_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(task) + "\n")

    def _memory_context_for_prompt(self, limit: int = 12) -> str:
        with self._lock:
            mems = list(self._memories)[-limit:]
            mind = self._mind
        lines = [
            f"MISSION: {mind.get('mission') or ''}",
            f"FOCUS: {mind.get('focus') or ''}",
            "ACTIVE GOALS:",
        ]
        for g in (mind.get("goals") or []):
            if g.get("status") in (None, "active", "in_progress"):
                lines.append(f"- [{g.get('priority')}] {g.get('title')}: {g.get('detail') or ''}")
        lines.append("RECENT MEMORIES:")
        for m in mems:
            lines.append(f"- ({m.get('kind')}) {m.get('content')}")
        return "\n".join(lines)[:4000]

    def _absorb_from_analysis(
        self,
        *,
        evolution_id: str,
        structural: dict[str, Any],
        analysis: dict[str, Any],
        narrative: Optional[str],
    ) -> dict[str, int]:
        """Turn Gemma output into durable memories + mind focus updates."""
        added = 0
        learnings = 0

        if narrative:
            self.add_memory(
                narrative,
                kind="narrative",
                tags=["trace", evolution_id],
                source="gemma",
                evolution_id=evolution_id,
            )
            added += 1

        for item in (analysis.get("what_worked") or [])[:6]:
            self.add_memory(
                str(item),
                kind="worked",
                tags=["learning", evolution_id],
                source="gemma",
                evolution_id=evolution_id,
            )
            added += 1
        for item in (analysis.get("what_failed") or [])[:6]:
            self.add_memory(
                str(item),
                kind="failed",
                tags=["learning", evolution_id],
                source="gemma",
                evolution_id=evolution_id,
            )
            added += 1
        for smell in (analysis.get("prompt_smells") or [])[:8]:
            purpose = smell.get("purpose") or "?"
            issue = smell.get("issue") or ""
            fix = smell.get("fix") or ""
            self.add_memory(
                f"[{purpose}] {issue} → {fix}",
                kind="prompt_smell",
                tags=["prompt", purpose, evolution_id],
                source="gemma",
                evolution_id=evolution_id,
                meta=smell if isinstance(smell, dict) else {},
            )
            added += 1
        for step in (analysis.get("product_next_steps") or [])[:5]:
            self.add_memory(
                str(step),
                kind="product_next",
                tags=["product", evolution_id],
                source="gemma",
                evolution_id=evolution_id,
            )
            added += 1

        # Learning log entry
        learning = {
            "evolution_id": evolution_id,
            "headline": structural.get("headline"),
            "narrative": narrative,
            "what_worked": analysis.get("what_worked") or [],
            "what_failed": analysis.get("what_failed") or [],
            "prompt_smells": analysis.get("prompt_smells") or [],
            "product_next_steps": analysis.get("product_next_steps") or [],
            "studio_tasks": analysis.get("studio_tasks") or [],
            "coordinator_decisions": analysis.get("coordinator_decisions") or [],
            "confidence": analysis.get("confidence"),
        }
        self._append_learning(learning)
        learnings = 1

        # Update plan focus from product next steps or narrative
        focus_bits = []
        if analysis.get("product_next_steps"):
            focus_bits.append(str(analysis["product_next_steps"][0])[:160])
        if analysis.get("what_failed"):
            focus_bits.append("Fix: " + str(analysis["what_failed"][0])[:120])
        if analysis.get("coordinator_decisions"):
            focus_bits.append("Coord: " + str(analysis["coordinator_decisions"][0])[:120])
        if focus_bits:
            with self._lock:
                self._mind["focus"] = " · ".join(focus_bits)[:400]
                plans = list(self._mind.get("plans") or [])
                if plans:
                    plans[0]["focus"] = self._mind["focus"]
                    plans[0]["updated_at"] = utcnow()
                    self._mind["plans"] = plans
                self._save_mind()

        return {"memories_added": added, "learnings": learnings}

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                enabled = bool(self._state.get("enabled"))
                interval = float(self._state.get("interval_secs") or DEFAULT_INTERVAL)
            if not enabled:
                break
            try:
                self.tick()
            except Exception as e:
                self._log("error", f"tick failed: {e}", traceback=traceback.format_exc()[-1500:])
                with self._lock:
                    self._state["last_error"] = str(e)
                    self._save_state()
            end = time.time() + max(30.0, interval)
            while time.time() < end and not self._stop.is_set():
                time.sleep(1.0)

    def tick(self, *, force_evo_id: Optional[str] = None, with_gemma: bool = True) -> dict[str, Any]:
        """One maintainer cycle: pick a run, digest, optional gemma, absorb mind."""
        with self._lock:
            model = self._state.get("model") or DEFAULT_MODEL
            auto_apply = bool(self._state.get("auto_apply_prompt_patches", True))
            absorb = bool(self._state.get("absorb_learnings", True))

        target = force_evo_id or self._pick_run()
        if not target:
            result = {"ok": True, "action": "idle", "reason": "no evolution runs to analyze"}
            with self._lock:
                self._state["last_tick_at"] = utcnow()
                self._state["ticks"] = int(self._state.get("ticks") or 0) + 1
                self._state["last_result"] = result
                self._state["last_error"] = None
                self._save_state()
            self._log("tick", "idle — no runs")
            return result

        root = self.evolutions_root / target
        evo_path = root / "evolution.json"
        if not evo_path.is_file():
            return {"ok": False, "error": f"missing evolution.json for {target}"}

        data = json.loads(evo_path.read_text(encoding="utf-8"))
        call_n = len(data.get("llm_calls") or [])
        with self._lock:
            prev = (self._state.get("analyzed_runs") or {}).get(target) or {}
        structural = trace_digest.build_structural_digest(data)
        trace_digest.save_digest(root, structural)

        gemma_out = None
        did_gemma = False
        if with_gemma and (force_evo_id or call_n != prev.get("llm_calls") or data.get("status") != prev.get("status")):
            try:
                # Inject mind context into analysis via structural side-channel
                structural = dict(structural)
                structural["maintainer_mind_context"] = self._memory_context_for_prompt()
                gemma_out = trace_digest.run_gemma_trace_analysis(data, structural=structural, model=model)
                trace_digest.save_digest(root, gemma_out)
                did_gemma = True
            except Exception as e:
                self._log("error", f"gemma analysis failed for {target}: {e}")
                gemma_out = {"error": str(e), "structural": structural}

        applied_patches = 0
        analysis = (gemma_out or {}).get("analysis") or {}
        patches = analysis.get("prompt_bank_patches") or []
        if auto_apply and patches and did_gemma:
            try:
                bank = data.get("prompt_bank") or {}
                new_bank = trace_digest.merge_prompt_bank_patches(bank, patches)
                data["prompt_bank"] = new_bank
                data["updated_at"] = utcnow()
                evo_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                (root / "prompt-bank.json").write_text(json.dumps(new_bank, indent=2), encoding="utf-8")
                applied_patches = len(patches)
                self.add_memory(
                    f"Applied {applied_patches} prompt-bank patches on {target}",
                    kind="action",
                    tags=["prompt_bank", target],
                    source="maintainer",
                    evolution_id=target,
                )
            except Exception as e:
                self._log("error", f"prompt patch apply failed: {e}")

        enqueued = 0
        for t in (analysis.get("studio_tasks") or []):
            self.append_task({
                "evolution_id": target,
                "id": t.get("id") or f"task-{uuid.uuid4().hex[:8]}",
                "area": t.get("area"),
                "priority": t.get("priority") or "med",
                "title": t.get("title"),
                "detail": t.get("detail"),
                "auto_safe": bool(t.get("auto_safe")),
                "status": "pending",
            })
            enqueued += 1

        if not enqueued:
            for s in (structural.get("suggestions_heuristic") or []):
                if s.get("priority") == "high":
                    self.append_task({
                        "evolution_id": target,
                        "id": f"heuristic-{s.get('area')}-{target[:6]}",
                        "area": s.get("area"),
                        "priority": "high",
                        "title": s.get("action"),
                        "detail": s.get("evidence"),
                        "auto_safe": False,
                        "status": "pending",
                        "source": "heuristic",
                    })
                    enqueued += 1

        absorb_stats = {"memories_added": 0, "learnings": 0}
        if absorb and did_gemma and analysis:
            try:
                absorb_stats = self._absorb_from_analysis(
                    evolution_id=target,
                    structural=structural,
                    analysis=analysis,
                    narrative=(gemma_out or {}).get("narrative") or analysis.get("narrative"),
                )
            except Exception as e:
                self._log("error", f"absorb failed: {e}")

        # Even without gemma, remember structural headline as a light memory occasionally
        if absorb and not did_gemma and structural.get("headline"):
            try:
                # Only if headline changed vs last memory for this evo
                with self._lock:
                    last = None
                    for m in reversed(self._memories):
                        if m.get("evolution_id") == target and m.get("kind") == "headline":
                            last = m
                            break
                if not last or last.get("content") != structural.get("headline"):
                    self.add_memory(
                        structural["headline"],
                        kind="headline",
                        tags=["structural", target],
                        source="structural",
                        evolution_id=target,
                    )
                    absorb_stats["memories_added"] = absorb_stats.get("memories_added", 0) + 1
            except Exception:
                pass

        result = {
            "ok": True,
            "action": "analyzed",
            "evolution_id": target,
            "llm_calls": call_n,
            "did_gemma": did_gemma,
            "applied_prompt_patches": applied_patches,
            "tasks_enqueued": enqueued,
            "headline": structural.get("headline"),
            "narrative": (gemma_out or {}).get("narrative") if gemma_out else None,
            "coordinator_decisions": analysis.get("coordinator_decisions") or [],
            "absorb": absorb_stats,
            "focus": self.get_mind().get("focus"),
        }
        with self._lock:
            ar = dict(self._state.get("analyzed_runs") or {})
            ar[target] = {
                "ts": utcnow(),
                "llm_calls": call_n,
                "status": data.get("status"),
                "did_gemma": did_gemma,
            }
            if len(ar) > 40:
                for k in list(ar.keys())[:-40]:
                    del ar[k]
            self._state["analyzed_runs"] = ar
            self._state["last_tick_at"] = utcnow()
            self._state["ticks"] = int(self._state.get("ticks") or 0) + 1
            self._state["last_result"] = result
            self._state["last_error"] = None
            self._save_state()
        self._log(
            "tick",
            f"analyzed {target}",
            **{k: result[k] for k in result if k not in ("ok", "absorb")},
            absorb=absorb_stats,
        )
        return result

    def _pick_run(self) -> Optional[str]:
        if not self.evolutions_root.is_dir():
            return None
        candidates = []
        for p in self.evolutions_root.glob("*/evolution.json"):
            try:
                st = p.stat()
                data = json.loads(p.read_text(encoding="utf-8"))
                evo_id = data.get("id") or p.parent.name
                call_n = len(data.get("llm_calls") or [])
                status = data.get("status") or ""
                with self._lock:
                    prev = (self._state.get("analyzed_runs") or {}).get(evo_id) or {}
                stale = call_n != prev.get("llm_calls") or status != prev.get("status")
                score = st.st_mtime
                if status == "running":
                    score += 1e12
                if stale:
                    score += 1e9
                candidates.append((score, evo_id, stale))
            except Exception:
                continue
        if not candidates:
            return None
        candidates.sort(reverse=True)
        for score, evo_id, stale in candidates:
            if stale:
                return evo_id
        return candidates[0][1]


_SERVICE: Optional[MaintainerService] = None


def get_maintainer(data_dir: Path, evolutions_root: Path) -> MaintainerService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = MaintainerService(data_dir, evolutions_root)
    return _SERVICE
