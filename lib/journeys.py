"""Mouse journey / heatmap sessions — automation reports injected into agent prompts.

Like bug notes (F), but J records pointer path + clicks, builds a heat summary,
and stores a text "video report" for the next prompt.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def journeys_dir(data_dir: Path) -> Path:
    d = Path(data_dir) / "journeys"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _index_path(data_dir: Path) -> Path:
    return journeys_dir(data_dir) / "index.json"


def load_index(data_dir: Path) -> list[dict[str, Any]]:
    p = _index_path(data_dir)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return list(data.get("sessions") or [])
    except Exception:
        return []


def save_index(data_dir: Path, sessions: list[dict[str, Any]]) -> None:
    p = _index_path(data_dir)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"sessions": sessions, "updated_at": utcnow()}, indent=2), encoding="utf-8")
    tmp.replace(p)


def session_path(data_dir: Path, sid: str) -> Path:
    return journeys_dir(data_dir) / f"{sid}.json"


def build_report(session: dict[str, Any]) -> str:
    """Human-readable automation / 'video' report for LLM injection."""
    meta = session.get("meta") or {}
    clicks = session.get("clicks") or []
    samples = session.get("path") or []
    dwell = session.get("hotspots") or []
    vw = meta.get("viewport_w") or meta.get("width") or "?"
    vh = meta.get("viewport_h") or meta.get("height") or "?"
    dur = meta.get("duration_ms") or session.get("duration_ms") or 0
    dur_s = round(float(dur) / 1000.0, 1) if dur else 0
    label = session.get("label") or session.get("note") or "Untitled journey"
    page = meta.get("page") or meta.get("url") or meta.get("hash") or ""
    tab = meta.get("tab") or ""

    lines = [
        f"### Journey report: {label}",
        f"- id: {session.get('id')}",
        f"- duration: {dur_s}s · viewport: {vw}×{vh}",
        f"- path samples: {len(samples)} · clicks: {len(clicks)}",
    ]
    if page:
        lines.append(f"- page: {page}")
    if tab:
        lines.append(f"- tab: {tab}")
    if session.get("note"):
        lines.append(f"- user note: {session.get('note')}")

    if clicks:
        lines.append("\n#### Click sequence (order matters — user intent trail)")
        for i, c in enumerate(clicks[:40], 1):
            sel = c.get("selector") or c.get("target") or "?"
            tag = c.get("tag") or ""
            txt = (c.get("text") or "")[:80].replace("\n", " ")
            x, y = c.get("x"), c.get("y")
            t = c.get("t_ms")
            t_s = f"{round(t/1000,1)}s" if isinstance(t, (int, float)) else "?"
            lines.append(
                f"{i}. t={t_s} @ ({x},{y}) <{tag}> {sel}"
                + (f' text="{txt}"' if txt else "")
            )

    if dwell:
        lines.append("\n#### Heat hotspots (high dwell / path density)")
        for i, h in enumerate(dwell[:15], 1):
            lines.append(
                f"{i}. region ~({h.get('x')},{h.get('y')}) "
                f"r={h.get('r') or 24} intensity={h.get('intensity') or h.get('count') or '?'}"
                + (f" near={h.get('selector')}" if h.get("selector") else "")
            )

    # Path summary: start / mid / end
    if samples:
        def _pt(p: dict) -> str:
            return f"({p.get('x')},{p.get('y')})"
        lines.append("\n#### Path skeleton")
        lines.append(f"- start: {_pt(samples[0])}")
        if len(samples) > 2:
            mid = samples[len(samples) // 2]
            lines.append(f"- mid: {_pt(mid)}")
        lines.append(f"- end: {_pt(samples[-1])}")

    lines.append(
        "\n#### Agent instructions from this journey\n"
        "- Treat this as a silent usability / intent recording (not a single bug click).\n"
        "- Infer friction: dead ends, repeated clicks, long dwell without progress, rage clicks.\n"
        "- Prefer UX/flow fixes that match the click order and hotspots.\n"
        "- If the user left a note, prioritize that narrative over raw coordinates.\n"
    )
    return "\n".join(lines)


def create_session(data_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    sid = uuid.uuid4().hex[:12]
    # Downsample path if huge
    path = list(payload.get("path") or [])
    if len(path) > 2000:
        step = max(1, len(path) // 1500)
        path = path[::step]
    clicks = list(payload.get("clicks") or [])[:200]
    session = {
        "id": sid,
        "created_at": utcnow(),
        "status": payload.get("status") or "open",  # open → inject; resolved → ignore
        "project": payload.get("project") or "evolve-ui",
        "label": (payload.get("label") or payload.get("note") or "Journey")[:120],
        "note": (payload.get("note") or "")[:2000],
        "meta": payload.get("meta") or {},
        "duration_ms": payload.get("duration_ms") or (payload.get("meta") or {}).get("duration_ms"),
        "path": path,
        "clicks": clicks,
        "hotspots": payload.get("hotspots") or [],
        "heatmap_data_url": (payload.get("heatmap_data_url") or "")[:500_000],  # cap
        "kind": "journey",
    }
    session["report"] = build_report(session)
    session_path(data_dir, sid).write_text(json.dumps(session, indent=2), encoding="utf-8")
    # index summary (no huge path/heatmap)
    idx = load_index(data_dir)
    idx.insert(0, {
        "id": sid,
        "created_at": session["created_at"],
        "status": session["status"],
        "project": session["project"],
        "label": session["label"],
        "note": session["note"][:200],
        "duration_ms": session.get("duration_ms"),
        "clicks": len(clicks),
        "path_samples": len(path),
        "has_heatmap": bool(session.get("heatmap_data_url")),
    })
    save_index(data_dir, idx[:200])
    return session


def get_session(data_dir: Path, sid: str) -> Optional[dict[str, Any]]:
    p = session_path(data_dir, sid)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def update_session(data_dir: Path, sid: str, patch: dict[str, Any]) -> Optional[dict[str, Any]]:
    s = get_session(data_dir, sid)
    if not s:
        return None
    for k in ("status", "label", "note"):
        if k in patch and patch[k] is not None:
            s[k] = patch[k]
    s["updated_at"] = utcnow()
    if "label" in patch or "note" in patch or "status" in patch:
        s["report"] = build_report(s)
    session_path(data_dir, sid).write_text(json.dumps(s, indent=2), encoding="utf-8")
    idx = load_index(data_dir)
    for e in idx:
        if e.get("id") == sid:
            e["status"] = s.get("status")
            e["label"] = s.get("label")
            e["note"] = (s.get("note") or "")[:200]
            e["updated_at"] = s["updated_at"]
            break
    save_index(data_dir, idx)
    return s


def delete_session(data_dir: Path, sid: str) -> bool:
    p = session_path(data_dir, sid)
    if p.exists():
        p.unlink()
    idx = [e for e in load_index(data_dir) if e.get("id") != sid]
    save_index(data_dir, idx)
    return True


def open_sessions(data_dir: Path, project: Optional[str] = None, limit: int = 20) -> list[dict[str, Any]]:
    out = []
    for e in load_index(data_dir):
        if e.get("status") == "resolved":
            continue
        if project and e.get("project") not in (project, "evolve-ui", None):
            # still include evolve-ui global sessions for any project
            if e.get("project") != "evolve-ui":
                continue
        full = get_session(data_dir, e["id"])
        if full:
            out.append(full)
        if len(out) >= limit:
            break
    return out


def prompt_context(data_dir: Path, project: Optional[str] = None, limit: int = 5) -> str:
    """Markdown block injected into the next agent / evolve prompt."""
    sessions = open_sessions(data_dir, project=project, limit=limit)
    if not sessions:
        return ""
    parts = [
        "## Open journey / heat-map reports (user pressed J — mouse + clicks recorded)",
        "- These are automation/usability trails, not single bug clicks (F).",
        "- Use them to fix flows, reduce friction, and align UI with observed intent.",
        "",
    ]
    for i, s in enumerate(sessions, 1):
        report = s.get("report") or build_report(s)
        parts.append(f"### [{i}] {report}")
        # optional tiny heatmap pointer (don't dump full data URL into every prompt)
        if s.get("heatmap_data_url"):
            parts.append(
                f"(heatmap image available at session {s.get('id')} — "
                f"prefer coordinates/clicks above for reasoning)"
            )
        parts.append("")
    return "\n".join(parts).strip() + "\n\n"
