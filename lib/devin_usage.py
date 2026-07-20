"""Devin CLI usage investigation + local quota estimates.

Devin does not publish a clean billing API for free-tier limits. This module:

1. Discovers models (and Free vs paid cost) via `devin models list --format json`
   at server boot — free set changes over time.
2. Tracks **our** Devin CLI invocations (planner / export / project jobs).
3. Scans the whole machine for **any** running `devin` processes so external
   CLIs (not part of evolution) still count against the hidden free-tier
   pressure (user reports a ~3 hour rolling quota we must not burn).
4. Exposes a Cerebras-like snapshot: rpm/tpm-style windows + 3h window,
   soft 90% / hard ~98% of estimated caps.

All numbers for 3h caps are **estimates** (env-overridable) until Cognition
documents real limits.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# Hidden free-tier style estimates (unknown true caps — stay conservative).
# Soft gate targets 90% of these; hard ≈ 98.1%.
DEVIN_3H_REQUESTS = int(os.environ.get("DEVIN_QUOTA_3H_REQUESTS", "40"))
DEVIN_3H_TOKENS = int(os.environ.get("DEVIN_QUOTA_3H_TOKENS", "800000"))
DEVIN_MIN_REQUESTS = int(os.environ.get("DEVIN_QUOTA_MIN_REQUESTS", "8"))
DEVIN_MIN_TOKENS = int(os.environ.get("DEVIN_QUOTA_MIN_TOKENS", "120000"))
DEVIN_HOUR_REQUESTS = int(os.environ.get("DEVIN_QUOTA_HOUR_REQUESTS", "20"))
DEVIN_HOUR_TOKENS = int(os.environ.get("DEVIN_QUOTA_HOUR_TOKENS", "400000"))

QUOTA_SOFT_RATIO = float(os.environ.get("DEVIN_QUOTA_SOFT", "0.90"))
QUOTA_OVERSHOOT = float(os.environ.get("DEVIN_QUOTA_OVERSHOOT", "0.09"))
QUOTA_HARD_RATIO = QUOTA_SOFT_RATIO * (1.0 + QUOTA_OVERSHOOT)

# Dropdown free models we pin for now (still verified against live catalog).
PINNED_FREE_MODELS = ("glm-5-2", "swe-1-7")  # GLM 5.2 High · SWE 1.7 Max


def _which_devin() -> Optional[str]:
    return shutil.which("devin") or (
        str(Path.home() / ".local/bin/devin")
        if (Path.home() / ".local/bin/devin").exists()
        else None
    )


def discover_models(*, timeout_secs: float = 45.0) -> dict[str, Any]:
    """Run `devin models list --format json` and parse free vs paid catalog."""
    bin_path = _which_devin()
    out: dict[str, Any] = {
        "ok": False,
        "bin": bin_path,
        "fetched_at": utcnow(),
        "families": [],
        "all_models": [],
        "free_models": [],
        "paid_models": [],
        "dropdown_models": [],
        "error": None,
    }
    if not bin_path:
        out["error"] = "devin binary not found"
        return out
    try:
        proc = subprocess.run(
            [bin_path, "models", "list", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=timeout_secs,
        )
        raw = (proc.stdout or "") + (proc.stderr or "")
        start = raw.find("{")
        if start < 0:
            out["error"] = f"no JSON in models list (rc={proc.returncode}): {raw[-400:]}"
            return out
        data = json.loads(raw[start:])
        families = data.get("families") or []
        out["families"] = families
        all_m: list[dict] = []
        free_m: list[dict] = []
        paid_m: list[dict] = []
        for fam in families:
            for v in fam.get("variants") or []:
                entry = {
                    "id": v.get("model_uid") or v.get("id") or "",
                    "label": v.get("label") or v.get("model_uid"),
                    "family": fam.get("family_label") or fam.get("slug"),
                    "family_slug": fam.get("slug"),
                    "cost_tier": v.get("cost_tier"),
                    "cost_summary": v.get("cost_summary"),
                    "context": v.get("max_context_tokens"),
                    "max_output": v.get("max_output_tokens"),
                    "is_beta": bool(v.get("is_beta")),
                    "is_free": str(v.get("cost_tier") or "").lower() == "free",
                }
                if not entry["id"]:
                    continue
                all_m.append(entry)
                if entry["is_free"]:
                    free_m.append(entry)
                else:
                    paid_m.append(entry)
        out["all_models"] = all_m
        out["free_models"] = free_m
        out["paid_models"] = paid_m
        # Dropdown: prefer pinned free ids that still exist; fall back to all free
        free_by_id = {m["id"]: m for m in free_m}
        dropdown = []
        for pid in PINNED_FREE_MODELS:
            if pid in free_by_id:
                m = free_by_id[pid]
                dropdown.append({
                    "id": f"devin:{m['id']}",
                    "harness": "devin",
                    "model": m["id"],
                    "label": f"Devin · {m['label']} (free)",
                    "cost_tier": "Free",
                    "context": m.get("context"),
                    "is_free": True,
                })
        if not dropdown:
            for m in free_m[:6]:
                dropdown.append({
                    "id": f"devin:{m['id']}",
                    "harness": "devin",
                    "model": m["id"],
                    "label": f"Devin · {m['label']} (free)",
                    "cost_tier": "Free",
                    "context": m.get("context"),
                    "is_free": True,
                })
        out["dropdown_models"] = dropdown
        out["ok"] = True
        return out
    except Exception as e:
        out["error"] = str(e)
        return out


def _pct(used: float, limit: Optional[float]) -> Optional[float]:
    if not limit:
        return None
    return round(100.0 * used / limit, 2)


@dataclass
class _Evt:
    ts: float
    model: str
    requests: int
    tokens: int
    source: str  # studio | external
    ok: bool = True
    purpose: Optional[str] = None
    run_id: Optional[str] = None
    pid: Optional[int] = None
    error: Optional[str] = None


class DevinUsageTracker:
    """Process-local Devin usage ledger + host process scanner."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started_at = utcnow()
        self._events: list[_Evt] = []
        self._seen_external_pids: dict[int, dict[str, Any]] = {}
        self.catalog: dict[str, Any] = {}
        self._catalog_lock = threading.Lock()
        # Background scanner
        self._scan_stop = threading.Event()
        self._scan_thread: Optional[threading.Thread] = None

    def set_catalog(self, catalog: dict[str, Any]) -> None:
        with self._catalog_lock:
            self.catalog = catalog

    def get_catalog(self) -> dict[str, Any]:
        with self._catalog_lock:
            return dict(self.catalog) if self.catalog else {}

    def record(
        self,
        model: str,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        ok: bool = True,
        purpose: Optional[str] = None,
        run_id: Optional[str] = None,
        source: str = "studio",
        pid: Optional[int] = None,
        error: Optional[str] = None,
        duration_secs: Optional[float] = None,
    ) -> dict[str, Any]:
        now = time.time()
        toks = max(0, int(prompt_tokens)) + max(0, int(completion_tokens))
        evt = _Evt(
            ts=now,
            model=model or "unknown",
            requests=1,
            tokens=toks,
            source=source,
            ok=ok,
            purpose=purpose,
            run_id=run_id,
            pid=pid,
            error=error,
        )
        with self._lock:
            self._events.append(evt)
            # keep 24h in memory
            cutoff = now - 86400
            self._events = [e for e in self._events if e.ts >= cutoff]
        out = {
            "ts": utcnow(),
            "ts_unix": now,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": toks,
            "tokens": toks,
            "requests": 1,
            "ok": ok,
            "purpose": purpose,
            "run_id": run_id,
            "source": source,
            "pid": pid,
            "duration_secs": duration_secs,
            "error": error,
        }
        # Persist every event for historical analysis
        try:
            from lib.usage_history import DEVIN_HISTORY
            DEVIN_HISTORY.append_event(out)
        except Exception:
            pass
        return out

    def _window_totals(self, seconds: float, *, model: Optional[str] = None, source: Optional[str] = None) -> tuple[int, int]:
        now = time.time()
        cutoff = now - seconds
        reqs = 0
        toks = 0
        with self._lock:
            for e in self._events:
                if e.ts < cutoff:
                    continue
                if model and e.model != model and model != "*":
                    continue
                if source and e.source != source:
                    continue
                reqs += e.requests
                toks += e.tokens
        return reqs, toks

    def scan_host_processes(self) -> list[dict[str, Any]]:
        """Find all running devin CLIs on this machine (any tty / cwd)."""
        found: list[dict[str, Any]] = []
        try:
            # ps: pid, etimes (seconds), rss, args
            proc = subprocess.run(
                ["ps", "-eo", "pid,etimes,rss,args", "--no-headers"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in (proc.stdout or "").splitlines():
                line = line.strip()
                if not line:
                    continue
                # crude parse
                parts = line.split(None, 3)
                if len(parts) < 4:
                    continue
                pid_s, etimes_s, rss_s, args = parts
                # Match real devin agent binary (not bash wrappers that only mention the word)
                first = args.split()[0] if args else ""
                base = Path(first).name
                is_devin = base == "devin" or bool(re.search(r"(^|\s)devin(\s|$)", args))
                if not is_devin:
                    continue
                # skip meta / discovery commands
                if "models list" in args or " models " in f" {args} ":
                    if "--format" in args or "list" in args.split():
                        # allow interactive `devin` only; skip list/catalog
                        if "models" in args:
                            continue
                if "ps -eo" in args or "GROK_AGENT" in args:
                    continue
                try:
                    pid = int(pid_s)
                    etimes = int(etimes_s)
                    rss_kb = int(rss_s)
                except ValueError:
                    continue
                model = None
                m = re.search(r"--model[=\s]+([^\s]+)", args)
                if m:
                    model = m.group(1).strip()
                entry = {
                    "pid": pid,
                    "etimes_secs": etimes,
                    "rss_mb": round(rss_kb / 1024.0, 1),
                    "cmdline": args[:300],
                    "model": model,
                    "source": "external",
                }
                found.append(entry)
                # First time we see a PID → record a session-start style event
                with self._lock:
                    if pid not in self._seen_external_pids:
                        self._seen_external_pids[pid] = {
                            **entry,
                            "first_seen": utcnow(),
                            "first_seen_ts": time.time(),
                        }
                        # Count as 1 request with rough token estimate for pressure
                        est_tok = 8000  # unknown; treat as moderate burn
                        self._events.append(_Evt(
                            ts=time.time(),
                            model=model or "external-unknown",
                            requests=1,
                            tokens=est_tok,
                            source="external",
                            ok=True,
                            purpose="external_process_seen",
                            pid=pid,
                        ))
                    else:
                        self._seen_external_pids[pid].update(entry)
                # Drop dead pids
            live = {f["pid"] for f in found}
            with self._lock:
                dead = [p for p in self._seen_external_pids if p not in live]
                for p in dead:
                    del self._seen_external_pids[p]
        except Exception:
            pass
        return found

    def start_scanner(self, interval_secs: float = 15.0) -> None:
        if self._scan_thread and self._scan_thread.is_alive():
            return

        def _loop() -> None:
            while not self._scan_stop.wait(interval_secs):
                try:
                    self.scan_host_processes()
                    # Periodic disk snapshot for historical throughput graphs
                    self._persist_snapshot()
                except Exception:
                    pass

        self._scan_thread = threading.Thread(target=_loop, name="devin-proc-scan", daemon=True)
        self._scan_thread.start()

    def _persist_snapshot(self, *, force: bool = False) -> None:
        try:
            from lib.usage_history import DEVIN_HISTORY
            r3, t3 = self._window_totals(3 * 3600)
            rs, ts = self._window_totals(3 * 3600, source="studio")
            rext, text = self._window_totals(3 * 3600, source="external")
            rmin, tmin = self._window_totals(60)
            with self._lock:
                n_live = len(self._seen_external_pids)
            DEVIN_HISTORY.maybe_snapshot({
                "live_process_count": n_live,
                "windows": {
                    "minute": {"requests": rmin, "tokens": tmin},
                    "three_hours": {
                        "requests": r3, "tokens": t3,
                        "studio_requests": rs, "studio_tokens": ts,
                        "external_requests": rext, "external_tokens": text,
                    },
                },
                "requests": r3,
                "tokens": t3,
                "studio_requests": rs,
                "external_requests": rext,
            }, force=force)
        except Exception:
            pass

    def stop_scanner(self) -> None:
        self._scan_stop.set()

    def snapshot(self, run_id: Optional[str] = None) -> dict[str, Any]:
        # Always refresh process list for the snapshot
        processes = self.scan_host_processes()
        catalog = self.get_catalog()

        def win(seconds: float, req_lim: int, tok_lim: int) -> dict[str, Any]:
            r_all, t_all = self._window_totals(seconds)
            r_studio, t_studio = self._window_totals(seconds, source="studio")
            r_ext, t_ext = self._window_totals(seconds, source="external")
            soft_r = int(req_lim * QUOTA_SOFT_RATIO)
            soft_t = int(tok_lim * QUOTA_SOFT_RATIO)
            hard_r = int(req_lim * QUOTA_HARD_RATIO)
            hard_t = int(tok_lim * QUOTA_HARD_RATIO)
            return {
                "requests": r_all,
                "tokens": t_all,
                "studio_requests": r_studio,
                "studio_tokens": t_studio,
                "external_requests": r_ext,
                "external_tokens": t_ext,
                "req_limit": req_lim,
                "tok_limit": tok_lim,
                "req_soft": soft_r,
                "tok_soft": soft_t,
                "req_hard": hard_r,
                "tok_hard": hard_t,
                "req_pct": _pct(r_all, req_lim),
                "tok_pct": _pct(t_all, tok_lim),
                "req_pct_of_soft": _pct(r_all, soft_r),
                "tok_pct_of_soft": _pct(t_all, soft_t),
            }

        by_model: dict[str, dict] = defaultdict(lambda: {
            "requests": 0, "tokens": 0, "errors": 0, "studio": 0, "external": 0,
        })
        with self._lock:
            recent = []
            for e in self._events[-80:]:
                by_model[e.model]["requests"] += e.requests
                by_model[e.model]["tokens"] += e.tokens
                if not e.ok:
                    by_model[e.model]["errors"] += 1
                if e.source == "studio":
                    by_model[e.model]["studio"] += 1
                else:
                    by_model[e.model]["external"] += 1
                recent.append({
                    "ts": datetime.fromtimestamp(e.ts, tz=timezone.utc).isoformat(),
                    "model": e.model,
                    "tokens": e.tokens,
                    "ok": e.ok,
                    "purpose": e.purpose,
                    "run_id": e.run_id,
                    "source": e.source,
                    "pid": e.pid,
                    "error": e.error,
                })
            recent = list(reversed(recent[-40:]))
            seen_ext = list(self._seen_external_pids.values())

        run_usage = None
        if run_id:
            r_req = r_tok = 0
            with self._lock:
                for e in self._events:
                    if e.run_id == run_id:
                        r_req += e.requests
                        r_tok += e.tokens
            run_usage = {"requests": r_req, "tokens": r_tok, "run_id": run_id}

        # Pressure signal: any of min/hour/3h near soft
        windows = {
            "minute": win(60, DEVIN_MIN_REQUESTS, DEVIN_MIN_TOKENS),
            "hour": win(3600, DEVIN_HOUR_REQUESTS, DEVIN_HOUR_TOKENS),
            "three_hours": win(3 * 3600, DEVIN_3H_REQUESTS, DEVIN_3H_TOKENS),
        }
        hot = False
        for w in windows.values():
            if (w.get("req_pct_of_soft") or 0) >= 100 or (w.get("tok_pct_of_soft") or 0) >= 100:
                hot = True

        try:
            self._persist_snapshot()
        except Exception:
            pass

        history: dict[str, Any] = {}
        try:
            from lib.usage_history import DEVIN_HISTORY
            history = DEVIN_HISTORY.history(hours=48, bucket_mins=15)
        except Exception as e:
            history = {"error": str(e), "series": [], "totals": {}}

        return {
            "started_at": self.started_at,
            "note": (
                "Devin free-tier 3h quota is not officially published. "
                "Caps below are local estimates; external CLIs on this host are included. "
                "Stay under soft (90%) of the 3h window. "
                "History is saved under data-*/usage/ for later analysis."
            ),
            "by_model": dict(by_model),
            "windows": windows,
            "gate": {
                "soft_ratio": QUOTA_SOFT_RATIO,
                "overshoot": QUOTA_OVERSHOOT,
                "hard_ratio": QUOTA_HARD_RATIO,
                "estimated_caps": {
                    "minute": {"requests": DEVIN_MIN_REQUESTS, "tokens": DEVIN_MIN_TOKENS},
                    "hour": {"requests": DEVIN_HOUR_REQUESTS, "tokens": DEVIN_HOUR_TOKENS},
                    "three_hours": {"requests": DEVIN_3H_REQUESTS, "tokens": DEVIN_3H_TOKENS},
                },
                "hot": hot,
            },
            "live_processes": processes,
            "seen_external": seen_ext,
            "recent_events": recent,
            "run": run_usage,
            "history": history,
            "catalog": {
                "ok": catalog.get("ok"),
                "fetched_at": catalog.get("fetched_at"),
                "free_models": catalog.get("free_models") or [],
                "dropdown_models": catalog.get("dropdown_models") or [],
                "error": catalog.get("error"),
            },
        }

    def should_throttle(self) -> tuple[bool, str]:
        """True if any estimated window is over soft cap (incl. external)."""
        snap = self.snapshot()
        for name, w in (snap.get("windows") or {}).items():
            if (w.get("req_pct_of_soft") or 0) >= 100:
                return True, f"{name} requests at {w.get('requests')}/{w.get('req_soft')} (soft)"
            if (w.get("tok_pct_of_soft") or 0) >= 100:
                return True, f"{name} tokens at {w.get('tokens')}/{w.get('tok_soft')} (soft)"
        return False, ""


USAGE = DevinUsageTracker()


def estimate_tokens(text: str) -> int:
    return max(1, (len(text or "") + 3) // 4)


def record_studio_call(
    model: str,
    prompt: str,
    response: str,
    *,
    ok: bool = True,
    purpose: Optional[str] = None,
    run_id: Optional[str] = None,
    duration_secs: Optional[float] = None,
    error: Optional[str] = None,
) -> dict[str, Any]:
    return USAGE.record(
        model,
        prompt_tokens=estimate_tokens(prompt),
        completion_tokens=estimate_tokens(response) if ok else 0,
        ok=ok,
        purpose=purpose,
        run_id=run_id,
        source="studio",
        duration_secs=duration_secs,
        error=error,
    )


def bootstrap_catalog() -> dict[str, Any]:
    """Call once at server startup."""
    cat = discover_models()
    USAGE.set_catalog(cat)
    USAGE.start_scanner(interval_secs=12.0)
    return cat
