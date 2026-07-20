"""Synchronous LLM helpers for Dev Studio + Cerebras usage tracking.

Intentionally synchronous so background evolution threads can call it without
pulling the whole server into asyncio.

Quota policy (local gate, process-wide):
  - Soft target = 90% of published Cerebras limits (minute / hour / day).
  - Hard ceiling = soft × (1 + 9%) ≈ 98.1% of published — may briefly overshoot
    the soft target by up to 9% of that soft max, but we never *deploy* a new
    call that would push past the hard ceiling.
  - When over soft (or projected over soft), callers wait in a fair queue
    instead of firing and getting 429s.
"""
from __future__ import annotations

import os
import re
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


# Published Cerebras free-tier style quotas (user-provided 2026-07-20).
# Used for local estimate dashboards — not a substitute for Cerebras billing UI.
CEREBRAS_MODEL_QUOTAS: dict[str, dict[str, Any]] = {
    "gemma-4-31b": {
        "label": "gemma-4-31b (Preview)",
        "context": 131072,
        "max_completion": 40000,
        "requests": {"minute": 100, "hour": 6000, "day": 144000},
        "tokens": {"minute": 100_000, "hour": 6_000_000, "day": 144_000_000},
        "images_per_request": 2,
        "tier": "preview",
    },
    "gpt-oss-120b": {
        "label": "gpt-oss-120b (Production)",
        "context": 65536,
        "max_completion": 32768,
        "requests": {"minute": 5, "hour": 150, "day": 2400},
        "tokens": {"minute": 30_000, "hour": 1_000_000, "day": 1_000_000},
        "tier": "production",
    },
    "zai-glm-4.7": {
        "label": "zai-glm-4.7 (Preview)",
        "context": 8192,
        "max_completion": 8192,
        "requests": {"minute": 5, "hour": 150, "day": 2400},
        "tokens": {"minute": 30_000, "hour": 1_000_000, "day": 1_000_000},
        "tier": "preview",
    },
    # retained for compatibility if still available on account
    "qwen-2.5-coder-32b": {
        "label": "qwen-2.5-coder-32b",
        "context": 131072,
        "max_completion": 32768,
        "requests": {"minute": 30, "hour": 900, "day": 10000},
        "tokens": {"minute": 60_000, "hour": 2_000_000, "day": 20_000_000},
        "tier": "legacy",
    },
    "llama3.1-70b": {
        "label": "llama3.1-70b",
        "context": 131072,
        "max_completion": 32768,
        "requests": {"minute": 30, "hour": 900, "day": 10000},
        "tokens": {"minute": 60_000, "hour": 2_000_000, "day": 20_000_000},
        "tier": "legacy",
    },
    "llama3.1-8b": {
        "label": "llama3.1-8b",
        "context": 8192,
        "max_completion": 8192,
        "requests": {"minute": 30, "hour": 900, "day": 10000},
        "tokens": {"minute": 60_000, "hour": 2_000_000, "day": 20_000_000},
        "tier": "legacy",
    },
}

# Soft = 85% of published by default (never intentionally hit the wall).
# Hard = soft + small overshoot only for in-flight estimate error — acquire never
# starts a NEW call past soft; wait in queue instead.
QUOTA_SOFT_RATIO = float(os.environ.get("CEREBRAS_QUOTA_SOFT", "0.85"))
QUOTA_OVERSHOOT = float(os.environ.get("CEREBRAS_QUOTA_OVERSHOOT", "0.05"))  # of soft max
QUOTA_HARD_RATIO = QUOTA_SOFT_RATIO * (1.0 + QUOTA_OVERSHOOT)
# Cap how long one call will wait for headroom (seconds). Evolution can retry later.
QUOTA_MAX_WAIT_SECS = float(os.environ.get("CEREBRAS_QUOTA_MAX_WAIT", "900"))
# Don't reserve the full max_completion (40k) — that starves the queue. Reserve a
# conservative expected completion, then re-check on actuals.
QUOTA_RESERVE_COMPLETION_CAP = int(os.environ.get("CEREBRAS_QUOTA_RESERVE_COMPLETION", "8000"))
# Never fire past soft after timeout (queue-first). Set CEREBRAS_QUOTA_ALLOW_HARD_AFTER_WAIT=1
# to restore old behavior that could enter the overshoot band after max_wait.
QUOTA_ALLOW_HARD_AFTER_WAIT = os.environ.get("CEREBRAS_QUOTA_ALLOW_HARD_AFTER_WAIT", "0") in (
    "1", "true", "yes",
)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class UsageBucket:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    errors: int = 0

    def add(self, prompt: int, completion: int, ok: bool = True) -> None:
        self.requests += 1
        if not ok:
            self.errors += 1
        self.prompt_tokens += max(0, prompt)
        self.completion_tokens += max(0, completion)
        self.total_tokens += max(0, prompt) + max(0, completion)

    def as_dict(self) -> dict[str, int]:
        return {
            "requests": self.requests,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "errors": self.errors,
        }


class CerebrasUsageTracker:
    """Process-local Cerebras usage ledger + pre-flight quota gate.

    Sliding windows (minute / hour / day) track completed calls. In-flight
    reservations prevent parallel threads from bursting past the soft cap.
    """

    # (seconds, reqs_key, toks_key) for published quota dicts
    _WINDOWS = (
        (60, "minute", "minute"),
        (3600, "hour", "hour"),
        (86400, "day", "day"),
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self.global_usage: dict[str, UsageBucket] = defaultdict(UsageBucket)
        self.by_run: dict[str, dict[str, UsageBucket]] = defaultdict(lambda: defaultdict(UsageBucket))
        self.events: list[dict[str, Any]] = []  # recent call log
        self.started_at = utcnow()
        # (ts, model, reqs, tokens) — completed calls (keep 24h for day window)
        self._window: list[tuple[float, str, int, int]] = []
        # In-flight reservations: id -> (ts, model, reqs, tokens)
        self._reserved: dict[str, tuple[float, str, int, int]] = {}
        self._waiters = 0
        self._total_waits = 0
        self._total_wait_secs = 0.0
        self._last_wait: Optional[dict[str, Any]] = None
        self._gate_enabled = os.environ.get("CEREBRAS_QUOTA_GATE", "1") not in ("0", "false", "no")

    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        ok: bool = True,
        run_id: Optional[str] = None,
        purpose: Optional[str] = None,
        duration_secs: Optional[float] = None,
        error: Optional[str] = None,
        reservation_id: Optional[str] = None,
        queued_secs: Optional[float] = None,
    ) -> dict[str, Any]:
        now = time.time()
        with self._cond:
            if reservation_id and reservation_id in self._reserved:
                del self._reserved[reservation_id]
            self.global_usage[model].add(prompt_tokens, completion_tokens, ok=ok)
            if run_id:
                self.by_run[run_id][model].add(prompt_tokens, completion_tokens, ok=ok)
            total = max(0, prompt_tokens) + max(0, completion_tokens)
            evt = {
                "ts": utcnow(),
                "ts_unix": now,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total,
                "tokens": total,
                "requests": 1,
                "source": "studio",
                "ok": ok,
                "run_id": run_id,
                "purpose": purpose,
                "duration_secs": duration_secs,
                "queued_secs": queued_secs,
                "error": error,
            }
            self.events.append(evt)
            if len(self.events) > 500:
                self.events = self.events[-500:]
            # Only count successful (or any completed) token burn toward windows.
            # Failed calls with 0 completion still count as a request attempt.
            self._window.append((now, model, 1, total))
            cutoff = now - 86400
            self._window = [w for w in self._window if w[0] >= cutoff]
            self._cond.notify_all()
        # Persist outside lock for historical throughput analysis
        try:
            from lib.usage_history import CEREBRAS_HISTORY
            CEREBRAS_HISTORY.append_event(evt)
        except Exception:
            pass
        return evt

    def release_reservation(self, reservation_id: Optional[str]) -> None:
        if not reservation_id:
            return
        with self._cond:
            if reservation_id in self._reserved:
                del self._reserved[reservation_id]
                self._cond.notify_all()

    def _window_totals_unlocked(self, model: str, seconds: float, *, include_reserved: bool = True) -> tuple[int, int]:
        now = time.time()
        cutoff = now - seconds
        reqs = 0
        toks = 0
        for ts, m, r, t in self._window:
            if ts >= cutoff and (model == "*" or m == model):
                reqs += r
                toks += t
        if include_reserved:
            for _rid, (ts, m, r, t) in self._reserved.items():
                # Reservations are "now"; always count them for short windows
                if model == "*" or m == model:
                    reqs += r
                    toks += t
        return reqs, toks

    def _window_totals(self, model: str, seconds: float) -> tuple[int, int]:
        with self._lock:
            return self._window_totals_unlocked(model, seconds, include_reserved=True)

    def _prune_stale_reservations_unlocked(self) -> None:
        # Safety: drop reservations older than 15 min (crashed callers)
        cutoff = time.time() - 900
        stale = [rid for rid, (ts, *_rest) in self._reserved.items() if ts < cutoff]
        for rid in stale:
            del self._reserved[rid]

    def _headroom_report_unlocked(
        self,
        model: str,
        need_reqs: int,
        need_toks: int,
        *,
        ratio: float,
    ) -> dict[str, Any]:
        """Check whether adding need_* stays under ratio * published limit for all windows."""
        q = CEREBRAS_MODEL_QUOTAS.get(model) or {}
        req_limits = q.get("requests") or {}
        tok_limits = q.get("tokens") or {}
        # Multi-key: each key has its own free-tier budget → scale published limits
        try:
            n_keys = max(1, KEY_POOL.size())
        except Exception:
            n_keys = 1
        blockers: list[dict[str, Any]] = []
        ok = True
        for secs, rkey, tkey in self._WINDOWS:
            used_r, used_t = self._window_totals_unlocked(model, secs, include_reserved=True)
            rlim = req_limits.get(rkey)
            tlim = tok_limits.get(tkey)
            rlim_s = int(rlim * n_keys) if rlim else None
            tlim_s = int(tlim * n_keys) if tlim else None
            soft_r = int(rlim_s * ratio) if rlim_s else None
            soft_t = int(tlim_s * ratio) if tlim_s else None
            proj_r = used_r + need_reqs
            proj_t = used_t + need_toks
            if soft_r is not None and proj_r > soft_r:
                ok = False
                blockers.append({
                    "window": rkey, "metric": "requests",
                    "used": used_r, "need": need_reqs, "projected": proj_r,
                    "soft_limit": soft_r, "published": rlim, "keys": n_keys,
                    "pct_of_published": _pct(proj_r, rlim_s),
                })
            if soft_t is not None and proj_t > soft_t:
                ok = False
                blockers.append({
                    "window": tkey, "metric": "tokens",
                    "used": used_t, "need": need_toks, "projected": proj_t,
                    "soft_limit": soft_t, "published": tlim, "keys": n_keys,
                    "pct_of_published": _pct(proj_t, tlim_s),
                })
        return {"ok": ok, "blockers": blockers, "ratio": ratio, "key_scale": n_keys}

    def _seconds_until_soft_room_unlocked(
        self,
        model: str,
        need_reqs: int,
        need_toks: int,
    ) -> float:
        """Estimate sleep until the tightest window frees enough capacity under soft ratio."""
        q = CEREBRAS_MODEL_QUOTAS.get(model) or {}
        req_limits = q.get("requests") or {}
        tok_limits = q.get("tokens") or {}
        try:
            n_keys = max(1, KEY_POOL.size())
        except Exception:
            n_keys = 1
        now = time.time()
        waits: list[float] = []
        for secs, rkey, tkey in self._WINDOWS:
            soft_r = int(req_limits[rkey] * QUOTA_SOFT_RATIO * n_keys) if req_limits.get(rkey) else None
            soft_t = int(tok_limits[tkey] * QUOTA_SOFT_RATIO * n_keys) if tok_limits.get(tkey) else None
            # Walk completed events oldest-first; find when enough fall out of window
            events = [(ts, r, t) for ts, m, r, t in self._window if m == model and ts >= now - secs]
            events.sort(key=lambda x: x[0])
            used_r = sum(e[1] for e in events) + sum(
                r for _rid, (_ts, m, r, _t) in self._reserved.items() if m == model
            )
            used_t = sum(e[2] for e in events) + sum(
                t for _rid, (_ts, m, _r, t) in self._reserved.items() if m == model
            )
            # Drop oldest events until projected fits (reservations stay)
            res_r = sum(r for _rid, (_ts, m, r, _t) in self._reserved.items() if m == model)
            res_t = sum(t for _rid, (_ts, m, _r, t) in self._reserved.items() if m == model)
            drop_r = 0
            drop_t = 0
            wait_for = 0.0
            for ts, r, t in events:
                if soft_r is not None and (used_r - drop_r + need_reqs) <= soft_r:
                    if soft_t is None or (used_t - drop_t + need_toks) <= soft_t:
                        break
                if soft_t is not None and soft_r is None and (used_t - drop_t + need_toks) <= soft_t:
                    break
                # need to drop this event
                drop_r += r
                drop_t += t
                wait_for = max(wait_for, (ts + secs) - now + 0.05)
            # If still over solely due to reservations, short poll
            proj_r = res_r + (used_r - drop_r - res_r) + need_reqs
            # simpler: if after dropping all historical still over soft due to reserved, wait 1s
            hist_r = sum(e[1] for e in events)
            hist_t = sum(e[2] for e in events)
            if soft_r is not None and res_r + need_reqs > soft_r:
                wait_for = max(wait_for, 1.0)
            if soft_t is not None and res_t + need_toks > soft_t:
                wait_for = max(wait_for, 1.0)
            if soft_r is not None and hist_r + res_r + need_reqs > soft_r:
                waits.append(wait_for or 1.0)
            elif soft_t is not None and hist_t + res_t + need_toks > soft_t:
                waits.append(wait_for or 1.0)
            _ = (proj_r,)  # silence lint
        return max(waits) if waits else 0.5

    def acquire(
        self,
        model: str,
        *,
        est_prompt_tokens: int,
        est_completion_tokens: int,
        purpose: Optional[str] = None,
        run_id: Optional[str] = None,
        max_wait_secs: Optional[float] = None,
    ) -> dict[str, Any]:
        """Block until a call is safe under soft quota, then reserve capacity.

        Returns reservation metadata. Always pair with record() or release_reservation().
        """
        if not self._gate_enabled or model not in CEREBRAS_MODEL_QUOTAS:
            return {
                "reservation_id": None,
                "queued_secs": 0.0,
                "skipped": True,
                "reason": "gate_disabled" if not self._gate_enabled else "unknown_model",
            }

        need_reqs = 1
        need_toks = max(1, int(est_prompt_tokens) + int(est_completion_tokens))
        max_wait = QUOTA_MAX_WAIT_SECS if max_wait_secs is None else float(max_wait_secs)
        t0 = time.time()
        reservation_id = uuid.uuid4().hex[:12]
        blockers: list[dict[str, Any]] = []

        with self._cond:
            self._waiters += 1
            self._total_waits += 1
            try:
                while True:
                    self._prune_stale_reservations_unlocked()
                    # Prefer soft; allow fire only if under hard when soft is tight but
                    # we've waited and still only slightly over soft? Policy:
                    # never acquire if projected > hard; wait while projected > soft.
                    hard = self._headroom_report_unlocked(
                        model, need_reqs, need_toks, ratio=QUOTA_HARD_RATIO
                    )
                    soft = self._headroom_report_unlocked(
                        model, need_reqs, need_toks, ratio=QUOTA_SOFT_RATIO
                    )
                    if soft["ok"]:
                        # Reserve under soft
                        self._reserved[reservation_id] = (
                            time.time(), model, need_reqs, need_toks
                        )
                        queued = round(time.time() - t0, 3)
                        self._total_wait_secs += queued
                        info = {
                            "reservation_id": reservation_id,
                            "queued_secs": queued,
                            "est_tokens": need_toks,
                            "model": model,
                            "purpose": purpose,
                            "run_id": run_id,
                            "soft_ratio": QUOTA_SOFT_RATIO,
                            "hard_ratio": QUOTA_HARD_RATIO,
                            "waited": queued > 0.05,
                        }
                        self._last_wait = {**info, "blockers": blockers[-3:] if blockers else []}
                        return info

                    # Over soft → always wait (never start a new call into the red zone).
                    # Optional escape hatch: CEREBRAS_QUOTA_ALLOW_HARD_AFTER_WAIT=1 may fire
                    # under hard ceiling after max_wait (not recommended).
                    blockers = soft.get("blockers") or hard.get("blockers") or []
                    elapsed = time.time() - t0
                    if elapsed >= max_wait:
                        if QUOTA_ALLOW_HARD_AFTER_WAIT and hard["ok"]:
                            self._reserved[reservation_id] = (
                                time.time(), model, need_reqs, need_toks
                            )
                            queued = round(elapsed, 3)
                            self._total_wait_secs += queued
                            info = {
                                "reservation_id": reservation_id,
                                "queued_secs": queued,
                                "est_tokens": need_toks,
                                "model": model,
                                "purpose": purpose,
                                "run_id": run_id,
                                "soft_ratio": QUOTA_SOFT_RATIO,
                                "hard_ratio": QUOTA_HARD_RATIO,
                                "waited": True,
                                "forced_after_timeout": True,
                                "blockers": blockers[:4],
                            }
                            self._last_wait = info
                            return info
                        raise TimeoutError(
                            f"cerebras quota gate: waited {queued_fmt(elapsed)} for {model} "
                            f"soft headroom ({QUOTA_SOFT_RATIO:.0%} of published × keys); "
                            f"queue full — not firing past soft. blockers={_fmt_blockers(blockers)}"
                        )

                    sleep_for = self._seconds_until_soft_room_unlocked(model, need_reqs, need_toks)
                    sleep_for = max(0.25, min(sleep_for, 5.0, max_wait - elapsed))
                    self._last_wait = {
                        "model": model,
                        "purpose": purpose,
                        "run_id": run_id,
                        "waiting": True,
                        "elapsed": round(elapsed, 2),
                        "sleep_for": round(sleep_for, 2),
                        "blockers": blockers[:4],
                        "soft_ratio": QUOTA_SOFT_RATIO,
                        "hard_ratio": QUOTA_HARD_RATIO,
                        "waiters": self._waiters,
                    }
                    self._cond.wait(timeout=sleep_for)
            finally:
                self._waiters = max(0, self._waiters - 1)

    def snapshot(self, run_id: Optional[str] = None) -> dict[str, Any]:
        with self._lock:
            by_model = {}
            models = set(self.global_usage.keys()) | set(CEREBRAS_MODEL_QUOTAS.keys()) | {
                m for _rid, (_ts, m, _r, _t) in self._reserved.items()
            }
            for model in sorted(models):
                bucket = self.global_usage.get(model) or UsageBucket()
                q = CEREBRAS_MODEL_QUOTAS.get(model, {})
                rpm, tpm = self._window_totals_unlocked(model, 60, include_reserved=True)
                rph, tph = self._window_totals_unlocked(model, 3600, include_reserved=True)
                rpd, tpd = self._window_totals_unlocked(model, 86400, include_reserved=True)
                # completed-only (for display of true burn)
                rpm_c, tpm_c = self._window_totals_unlocked(model, 60, include_reserved=False)
                sess = bucket.as_dict()
                res_r = sum(r for _rid, (_ts, m, r, _t) in self._reserved.items() if m == model)
                res_t = sum(t for _rid, (_ts, m, _r, t) in self._reserved.items() if m == model)

                def win(used_r, used_t, rlim, tlim, completed_r=None, completed_t=None):
                    soft_r = int(rlim * QUOTA_SOFT_RATIO) if rlim else None
                    soft_t = int(tlim * QUOTA_SOFT_RATIO) if tlim else None
                    hard_r = int(rlim * QUOTA_HARD_RATIO) if rlim else None
                    hard_t = int(tlim * QUOTA_HARD_RATIO) if tlim else None
                    return {
                        "requests": used_r,
                        "tokens": used_t,
                        "completed_requests": completed_r if completed_r is not None else used_r,
                        "completed_tokens": completed_t if completed_t is not None else used_t,
                        "reserved_requests": res_r,
                        "reserved_tokens": res_t,
                        "req_limit": rlim,
                        "tok_limit": tlim,
                        "req_soft": soft_r,
                        "tok_soft": soft_t,
                        "req_hard": hard_r,
                        "tok_hard": hard_t,
                        "req_pct": _pct(used_r, rlim),
                        "tok_pct": _pct(used_t, tlim),
                        "req_pct_of_soft": _pct(used_r, soft_r),
                        "tok_pct_of_soft": _pct(used_t, soft_t),
                    }

                by_model[model] = {
                    **sess,
                    "quota": q,
                    "window": {
                        "minute": win(
                            rpm, tpm,
                            (q.get("requests") or {}).get("minute"),
                            (q.get("tokens") or {}).get("minute"),
                            rpm_c, tpm_c,
                        ),
                        "hour": win(
                            rph, tph,
                            (q.get("requests") or {}).get("hour"),
                            (q.get("tokens") or {}).get("hour"),
                        ),
                        "day": win(
                            rpd, tpd,
                            (q.get("requests") or {}).get("day"),
                            (q.get("tokens") or {}).get("day"),
                        ),
                        "session_vs_day": {
                            "requests": sess["requests"],
                            "tokens": sess["total_tokens"],
                            "req_limit": (q.get("requests") or {}).get("day"),
                            "tok_limit": (q.get("tokens") or {}).get("day"),
                            "req_pct": _pct(sess["requests"], (q.get("requests") or {}).get("day")),
                            "tok_pct": _pct(sess["total_tokens"], (q.get("tokens") or {}).get("day")),
                            "req_soft": int(((q.get("requests") or {}).get("day") or 0) * QUOTA_SOFT_RATIO) or None,
                            "tok_soft": int(((q.get("tokens") or {}).get("day") or 0) * QUOTA_SOFT_RATIO) or None,
                        },
                    },
                }
            run_usage = None
            if run_id and run_id in self.by_run:
                run_usage = {m: b.as_dict() for m, b in self.by_run[run_id].items()}
            # Session totals for history snapshot
            sess_req = 0
            sess_tok = 0
            for _m, b in by_model.items():
                sess_req += int(b.get("requests") or 0)
                sess_tok += int(b.get("total_tokens") or 0)
            gate = {
                "enabled": self._gate_enabled,
                "soft_ratio": QUOTA_SOFT_RATIO,
                "overshoot": QUOTA_OVERSHOOT,
                "hard_ratio": QUOTA_HARD_RATIO,
                "max_wait_secs": QUOTA_MAX_WAIT_SECS,
                "waiters": self._waiters,
                "total_waits": self._total_waits,
                "total_wait_secs": round(self._total_wait_secs, 2),
                "reserved": {
                    rid: {"model": m, "reqs": r, "tokens": t, "age_secs": round(time.time() - ts, 2)}
                    for rid, (ts, m, r, t) in self._reserved.items()
                },
                "last_wait": self._last_wait,
            }
            # minute window aggregate across models
            rpm = tpm = 0
            for _m, b in by_model.items():
                w = (b.get("window") or {}).get("minute") or {}
                rpm += int(w.get("requests") or 0)
                tpm += int(w.get("tokens") or 0)

        try:
            from lib.usage_history import CEREBRAS_HISTORY
            CEREBRAS_HISTORY.maybe_snapshot({
                "requests": sess_req,
                "tokens": sess_tok,
                "studio_requests": sess_req,
                "external_requests": 0,
                "windows": {
                    "minute": {"requests": rpm, "tokens": tpm},
                    "three_hours": {
                        "requests": sess_req, "tokens": sess_tok,
                        "studio_requests": sess_req, "studio_tokens": sess_tok,
                        "external_requests": 0, "external_tokens": 0,
                    },
                },
            })
            history = CEREBRAS_HISTORY.history(hours=48, bucket_mins=15)
        except Exception as e:
            history = {"error": str(e), "series": [], "totals": {}}

        return {
            "started_at": self.started_at,
            "by_model": by_model,
            "run": run_usage,
            "recent_events": list(self.events[-40:]),
            "models_catalog": CEREBRAS_MODEL_QUOTAS,
            "gate": gate,
            "history": history,
            "note": "Cerebras usage history saved under data-*/usage/ for throughput analysis.",
        }


def queued_fmt(secs: float) -> str:
    if secs < 60:
        return f"{secs:.1f}s"
    return f"{secs / 60:.1f}m"


def _fmt_blockers(blockers: list[dict[str, Any]]) -> str:
    parts = []
    for b in (blockers or [])[:4]:
        parts.append(
            f"{b.get('window')}/{b.get('metric')} {b.get('projected')}/{b.get('soft_limit')} "
            f"(pub {b.get('published')})"
        )
    return "; ".join(parts) if parts else "none"


def estimate_prompt_tokens(prompt: str) -> int:
    # ~4 chars/token for English/code mix; pad slightly so we reserve early
    return max(1, (len(prompt) + 3) // 4)


def estimate_reserve_completion(max_tokens: int, model: str) -> int:
    q = CEREBRAS_MODEL_QUOTAS.get(model) or {}
    cap = int(q.get("max_completion") or max_tokens)
    # Reserve expected completion, not full theoretical max
    return max(256, min(int(max_tokens), cap, QUOTA_RESERVE_COMPLETION_CAP))


def _clamp_completion_to_headroom(model: str, max_tokens: int, est_prompt: int) -> int:
    """Shrink max_completion so prompt+completion stays under soft minute headroom."""
    q = CEREBRAS_MODEL_QUOTAS.get(model) or {}
    tlim = (q.get("tokens") or {}).get("minute")
    if not tlim:
        return max_tokens
    try:
        n_keys = max(1, KEY_POOL.size())
    except Exception:
        n_keys = 1
    soft = int(tlim * QUOTA_SOFT_RATIO * n_keys)
    # completed-only minute burn (reservation already holds our slot)
    with USAGE._lock:
        _rpm, tpm = USAGE._window_totals_unlocked(model, 60, include_reserved=False)
    # Leave a small cushion for other in-flight reserved tokens besides ours
    room = max(256, soft - tpm - est_prompt)
    return max(256, min(int(max_tokens), int(room)))


def _pct(used: int, limit: Optional[int]) -> Optional[float]:
    if not limit:
        return None
    return round(100.0 * used / limit, 2)


USAGE = CerebrasUsageTracker()

# Thread-local so evolution can tag calls with run_id without changing every signature site.
_tls = threading.local()


class CerebrasKeyPool:
    """Round-robin multi-key pool to multiply free-tier throughput.

    Env:
      CEREBRAS_API_KEY          — primary
      CEREBRAS_API_KEYS         — comma/newline/semicolon-separated extras
      CEREBRAS_API_KEY_2.._9    — additional numbered keys

    High-throughput mode is OFF by default: only the primary key is used.
    Turning it on enables rotation across all loaded keys and scales local
    quota soft-caps by key count.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rr = 0
        self._cooldown_until: dict[str, float] = {}  # key_id -> unix ts
        self._keys: list[tuple[str, str]] = []  # (key_id, secret)
        # OFF by default — user enables higher throughput from the Evolve UI
        self._high_throughput = False
        self.reload()

    def reload(self) -> int:
        raw: list[str] = []
        # Accept comma/space/semicolon lists in CEREBRAS_API_KEY (users often paste
        # two free-tier keys on one line) as well as CEREBRAS_API_KEYS / _KEY_2..
        primary = (os.environ.get("CEREBRAS_API_KEY") or "").strip()
        if primary:
            for part in re.split(r"[\s,;]+", primary):
                p = part.strip()
                if p:
                    raw.append(p)
        multi = os.environ.get("CEREBRAS_API_KEYS") or ""
        for part in re.split(r"[\s,;]+", multi):
            p = part.strip()
            if p:
                raw.append(p)
        for i in range(2, 10):
            k = (os.environ.get(f"CEREBRAS_API_KEY_{i}") or "").strip()
            if k:
                for part in re.split(r"[\s,;]+", k):
                    p = part.strip()
                    if p:
                        raw.append(p)
        # dedupe preserve order
        seen: set[str] = set()
        keys: list[tuple[str, str]] = []
        for k in raw:
            if k in seen:
                continue
            # skip obvious non-keys
            if len(k) < 8:
                continue
            seen.add(k)
            kid = f"k{len(keys)}…{k[-4:]}" if len(k) >= 4 else f"k{len(keys)}"
            keys.append((kid, k))
        with self._lock:
            self._keys = keys
            self._rr = 0
        return len(keys)

    def set_high_throughput(self, enabled: bool) -> dict[str, Any]:
        with self._lock:
            self._high_throughput = bool(enabled)
            ht = self._high_throughput
            n = len(self._keys)
        return self.status()

    def high_throughput(self) -> bool:
        with self._lock:
            return bool(self._high_throughput)

    def status(self) -> dict[str, Any]:
        with self._lock:
            n = len(self._keys)
            ht = bool(self._high_throughput)
            fps = [kid for kid, _ in self._keys]
        active = n if ht else (1 if n else 0)
        return {
            "high_throughput": ht,
            "keys_configured": n,
            "keys_active": active,
            "key_ids": fps,
            "quota_scale": max(1, active),
            "note": (
                "High throughput rotates across all keys and multiplies soft quota."
                if ht else
                "Default: primary key only. Enable high throughput to use all keys."
            ),
        }

    def size(self) -> int:
        """Keys that count toward quota scaling (1 unless high-throughput)."""
        with self._lock:
            n = len(self._keys)
            ht = self._high_throughput
        if n == 0:
            n = self.reload()
            with self._lock:
                ht = self._high_throughput
        if n == 0:
            return 0
        return n if ht else 1

    def configured_count(self) -> int:
        with self._lock:
            n = len(self._keys)
        if n == 0:
            return self.reload()
        return n

    def has_keys(self) -> bool:
        return self.configured_count() > 0

    def fingerprints(self) -> list[str]:
        with self._lock:
            return [kid for kid, _ in self._keys]

    def mark_rate_limited(self, key_id: str, secs: float = 45.0) -> None:
        with self._lock:
            self._cooldown_until[key_id] = time.time() + max(5.0, secs)

    def pick(self) -> tuple[str, str]:
        """Return (key_id, api_key). Single primary unless high-throughput is on."""
        if not self._keys:
            self.reload()
        with self._lock:
            if not self._keys:
                raise RuntimeError(
                    "No Cerebras API keys configured. Set CEREBRAS_API_KEY "
                    "and optional CEREBRAS_API_KEYS / CEREBRAS_API_KEY_2 in .env"
                )
            now = time.time()
            # Default: always primary (index 0) unless high-throughput
            if not self._high_throughput:
                return self._keys[0]
            n = len(self._keys)
            for offset in range(n):
                idx = (self._rr + offset) % n
                kid, secret = self._keys[idx]
                until = self._cooldown_until.get(kid, 0)
                if until <= now:
                    self._rr = (idx + 1) % n
                    return kid, secret
            best_i = min(
                range(n),
                key=lambda i: self._cooldown_until.get(self._keys[i][0], 0),
            )
            kid, secret = self._keys[best_i]
            wait = max(0.0, self._cooldown_until.get(kid, 0) - now)
            self._rr = (best_i + 1) % n
        if wait > 0.05:
            time.sleep(min(wait, 15.0))
        return kid, secret

    def client(self) -> tuple[Any, str]:
        """Build a Cerebras SDK client bound to the next key. Returns (client, key_id)."""
        from cerebras.cloud.sdk import Cerebras
        kid, secret = self.pick()
        # SDK accepts api_key=; also set env for any code that reads it
        try:
            client = Cerebras(api_key=secret)
        except TypeError:
            # older SDK: only env
            os.environ["CEREBRAS_API_KEY"] = secret
            client = Cerebras()
        return client, kid


KEY_POOL = CerebrasKeyPool()


def cerebras_key_count() -> int:
    return KEY_POOL.configured_count()


_LIVE_CEREBRAS_MODELS: Optional[set[str]] = None
_LIVE_CEREBRAS_MODELS_TS: float = 0.0


def list_live_cerebras_models(force: bool = False) -> set[str]:
    """Best-effort set of model ids available on this Cerebras account (cached ~10 min)."""
    global _LIVE_CEREBRAS_MODELS, _LIVE_CEREBRAS_MODELS_TS
    now = time.time()
    if (
        not force
        and _LIVE_CEREBRAS_MODELS is not None
        and (now - _LIVE_CEREBRAS_MODELS_TS) < 600
    ):
        return set(_LIVE_CEREBRAS_MODELS)
    out: set[str] = set()
    try:
        if not KEY_POOL.has_keys():
            KEY_POOL.reload()
        if not KEY_POOL.has_keys():
            return set(CEREBRAS_MODEL_QUOTAS.keys())
        client, _ = KEY_POOL.client()
        models = client.models.list()
        data = getattr(models, "data", None) or models
        for m in data:
            mid = getattr(m, "id", None)
            if mid is None and isinstance(m, dict):
                mid = m.get("id")
            if mid:
                out.add(str(mid))
    except Exception:
        # Fall back to catalog so we still assign known ids
        out = set(CEREBRAS_MODEL_QUOTAS.keys())
    if out:
        _LIVE_CEREBRAS_MODELS = out
        _LIVE_CEREBRAS_MODELS_TS = now
    return set(out)


def has_cerebras_key() -> bool:
    return KEY_POOL.has_keys()


def make_cerebras_client() -> tuple[Any, str]:
    """Public helper for server workers — (client, key_id)."""
    return KEY_POOL.client()


def set_high_throughput(enabled: bool) -> dict[str, Any]:
    return KEY_POOL.set_high_throughput(enabled)


def throughput_status() -> dict[str, Any]:
    return KEY_POOL.status()


def set_usage_context(run_id: Optional[str] = None, purpose: Optional[str] = None) -> None:
    _tls.run_id = run_id
    _tls.purpose = purpose


def clear_usage_context() -> None:
    _tls.run_id = None
    _tls.purpose = None


def call_cerebras_sync(
    prompt: str,
    model: str = "gemma-4-31b",
    max_tokens: int = 8192,
    *,
    run_id: Optional[str] = None,
    purpose: Optional[str] = None,
    temperature: float = 0.3,
) -> str:
    """Call Cerebras synchronously and return the generated text.

    Pre-flight quota gate: waits in queue until the call fits under the soft
    cap (90% of published limits × key count; hard ~98.1%). Rotates API keys
    for higher throughput when multiple keys are configured.
    """
    if not KEY_POOL.has_keys():
        raise RuntimeError("CEREBRAS_API_KEY not set (and no CEREBRAS_API_KEYS extras)")
    run_id = run_id or getattr(_tls, "run_id", None)
    purpose = purpose or getattr(_tls, "purpose", None)

    q = CEREBRAS_MODEL_QUOTAS.get(model) or {}
    cap = int(q.get("max_completion") or max_tokens)
    max_tokens = min(max_tokens, cap)

    est_prompt = estimate_prompt_tokens(prompt)
    est_completion = estimate_reserve_completion(max_tokens, model)

    try:
        slot = USAGE.acquire(
            model,
            est_prompt_tokens=est_prompt,
            est_completion_tokens=est_completion,
            purpose=purpose,
            run_id=run_id,
        )
    except TimeoutError as e:
        # Count as a failed/skipped deploy so the dashboard shows the block
        USAGE.record(
            model,
            est_prompt,
            0,
            ok=False,
            run_id=run_id,
            purpose=purpose,
            duration_secs=0,
            error=str(e),
        )
        raise RuntimeError(str(e)) from e

    reservation_id = slot.get("reservation_id")
    queued_secs = float(slot.get("queued_secs") or 0)
    # Clamp completion to remaining soft-token headroom so a single reply
    # cannot blow past the minute soft cap even if max_tokens is huge.
    max_tokens = _clamp_completion_to_headroom(model, max_tokens, est_prompt)
    start = time.time()
    prompt_tokens = 0
    completion_tokens = 0
    last_key_id = ""

    # 429 / transient: rotate key, release reservation, wait, re-acquire, retry
    max_attempts = max(4, KEY_POOL.size() + 2)
    last_err: Optional[Exception] = None
    for attempt in range(max_attempts):
        try:
            client, last_key_id = KEY_POOL.client()
            resp = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                max_completion_tokens=max_tokens,
                temperature=temperature,
            )
            usage = getattr(resp, "usage", None)
            if usage is not None:
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            else:
                prompt_tokens = est_prompt
                completion_tokens = 0
            text = resp.choices[0].message.content or ""
            if not completion_tokens and text:
                completion_tokens = max(1, len(text) // 4)
            evt = USAGE.record(
                model,
                prompt_tokens,
                completion_tokens,
                ok=True,
                run_id=run_id,
                purpose=purpose,
                duration_secs=round(time.time() - start, 3),
                reservation_id=reservation_id,
                queued_secs=queued_secs,
            )
            if isinstance(evt, dict):
                evt["key_id"] = last_key_id
            return text
        except Exception as e:
            last_err = e
            err_s = str(e).lower()
            is_rate = (
                "429" in err_s
                or "too_many_requests" in err_s
                or "rate" in err_s
                or "queue_exceeded" in err_s
                or "high traffic" in err_s
            )
            if is_rate and attempt < max_attempts - 1:
                if last_key_id:
                    KEY_POOL.mark_rate_limited(last_key_id, secs=30.0 + 10.0 * attempt)
                # Free reservation so others can proceed; cool down; re-acquire
                USAGE.release_reservation(reservation_id)
                reservation_id = None
                # Multi-key: try next key immediately; single-key: backoff
                if KEY_POOL.size() <= 1:
                    backoff = min(60.0, 5.0 * (2 ** attempt))
                    time.sleep(backoff)
                try:
                    slot = USAGE.acquire(
                        model,
                        est_prompt_tokens=est_prompt,
                        est_completion_tokens=est_completion,
                        purpose=purpose,
                        run_id=run_id,
                    )
                    reservation_id = slot.get("reservation_id")
                    queued_secs += float(slot.get("queued_secs") or 0)
                    continue
                except TimeoutError as te:
                    last_err = te
                    break
            # Non-rate error or final attempt
            break

    USAGE.record(
        model,
        prompt_tokens or est_prompt,
        completion_tokens or 0,
        ok=False,
        run_id=run_id,
        purpose=purpose,
        duration_secs=round(time.time() - start, 3),
        reservation_id=reservation_id,
        queued_secs=queued_secs,
        error=str(last_err) if last_err else "unknown",
    )
    raise RuntimeError(f"cerebras call failed: {last_err}")


# ── OpenRouter free-tier workers ─────────────────────────────────────────────

OPENROUTER_API_BASE = (os.environ.get("OPENROUTER_API_BASE") or "https://openrouter.ai/api/v1").rstrip("/")
OPENROUTER_FREE_ONLY = os.environ.get("OPENROUTER_FREE_ONLY", "1") in ("1", "true", "yes")

# Statistical free-tier estimates for OpenRouter (unpublished / varies).
# Soft gate style: stay under soft_ratio of these. Env-overridable.
OPENROUTER_STAT_QUOTAS: dict[str, Any] = {
    "requests": {
        "minute": int(os.environ.get("OPENROUTER_FREE_MIN_REQUESTS", "8")),
        "hour": int(os.environ.get("OPENROUTER_FREE_HOUR_REQUESTS", "30")),
        "day": int(os.environ.get("OPENROUTER_FREE_DAY_REQUESTS", "50")),
        "three_hours": int(os.environ.get("OPENROUTER_FREE_3H_REQUESTS", "20")),
    },
    "tokens": {
        "minute": int(os.environ.get("OPENROUTER_FREE_MIN_TOKENS", "80000")),
        "hour": int(os.environ.get("OPENROUTER_FREE_HOUR_TOKENS", "400000")),
        "day": int(os.environ.get("OPENROUTER_FREE_DAY_TOKENS", "1000000")),
        "three_hours": int(os.environ.get("OPENROUTER_FREE_3H_TOKENS", "300000")),
    },
    "note": (
        "OpenRouter free-model rate limits are not fixed per model; these are "
        "local statistical estimates so the dashboard can show pressure bars. "
        "Prefer free-only routing; spend caps are account-side."
    ),
    "source": "statistical_estimate",
}

# Curated free models for evolution workers (ids as OpenRouter expects).
# Prefer code/general chat; skip pure embedding / TTS / image models.
OPENROUTER_FREE_WORKER_MODELS: list[str] = [
    "openrouter/free",  # auto router across free catalog
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "tencent/hy3:free",
    "openai/gpt-oss-20b:free",
    "poolside/laguna-m.1:free",
    "poolside/laguna-xs-2.1:free",
    "cohere/north-mini-code:free",
]


def openrouter_api_key() -> str:
    return (os.environ.get("OPENROUTER_API_KEY") or "").strip()


def has_openrouter_key() -> bool:
    return bool(openrouter_api_key())


def openrouter_free_models() -> list[dict[str, Any]]:
    """Catalog entries for UI / worker pool (free only)."""
    out = []
    for mid in OPENROUTER_FREE_WORKER_MODELS:
        out.append({
            "id": f"openrouter:{mid}",
            "openrouter_id": mid,
            "label": f"OpenRouter · {mid}",
            "provider": "openrouter",
            "free": True,
            "high_throughput": mid in ("openrouter/free", "google/gemma-4-31b-it:free", "google/gemma-4-26b-a4b-it:free"),
            "worker": True,
            "quota": {
                **OPENROUTER_STAT_QUOTAS,
                "tier": "free_statistical",
            },
        })
    return out


def openrouter_usage_snapshot(run_id: Optional[str] = None) -> dict[str, Any]:
    """Session usage for openrouter:* models + statistical free-tier bars."""
    snap = USAGE.snapshot(run_id=run_id)
    by_model = snap.get("by_model") or {}
    or_models = {
        mid: data for mid, data in by_model.items()
        if str(mid).startswith("openrouter:") or ":free" in str(mid)
    }
    # Also surface catalog models with zero usage so UI lists all free options
    catalog = openrouter_free_models() if has_openrouter_key() else []
    for entry in catalog:
        mid = entry["id"]
        if mid not in or_models:
            or_models[mid] = {
                "requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "errors": 0,
                "quota": entry.get("quota") or OPENROUTER_STAT_QUOTAS,
                "window": {},
                "unused": True,
            }
    # Aggregate statistical windows for OR overall
    total_req = sum(int(m.get("requests") or 0) for m in or_models.values())
    total_tok = sum(int(m.get("total_tokens") or 0) for m in or_models.values())
    q = OPENROUTER_STAT_QUOTAS
    soft = QUOTA_SOFT_RATIO

    def _win(used_r: int, used_t: int, rlim: int, tlim: int) -> dict[str, Any]:
        return {
            "requests": used_r,
            "tokens": used_t,
            "req_limit": rlim,
            "tok_limit": tlim,
            "req_soft": int(rlim * soft),
            "tok_soft": int(tlim * soft),
            "req_pct": round(100.0 * used_r / rlim, 1) if rlim else None,
            "tok_pct": round(100.0 * used_t / tlim, 1) if tlim else None,
            "source": "statistical_estimate",
        }

    # Session totals vs day/3h estimates (best-effort; not true rolling windows for OR yet)
    windows = {
        "minute": _win(0, 0, q["requests"]["minute"], q["tokens"]["minute"]),
        "hour": _win(0, 0, q["requests"]["hour"], q["tokens"]["hour"]),
        "three_hours": _win(
            min(total_req, q["requests"]["three_hours"]),
            min(total_tok, q["tokens"]["three_hours"]),
            q["requests"]["three_hours"],
            q["tokens"]["three_hours"],
        ),
        "day": _win(total_req, total_tok, q["requests"]["day"], q["tokens"]["day"]),
        "session": {"requests": total_req, "tokens": total_tok},
    }
    day_pct = windows["day"]["req_pct"] or 0
    hot = day_pct >= soft * 100
    return {
        "ok": True,
        "enabled": has_openrouter_key(),
        "free_only": OPENROUTER_FREE_ONLY,
        "by_model": or_models,
        "catalog": catalog,
        "windows": windows,
        "statistical_quotas": OPENROUTER_STAT_QUOTAS,
        "gate": {
            "soft_ratio": soft,
            "hot": hot,
            "note": OPENROUTER_STAT_QUOTAS["note"],
        },
        "recent_events": [
            e for e in (snap.get("recent_events") or [])
            if str(e.get("model") or "").startswith("openrouter:")
        ][-40:],
        "totals": {"requests": total_req, "tokens": total_tok},
    }


def free_models_unified_snapshot(run_id: Optional[str] = None) -> dict[str, Any]:
    """Integrative free-tier view: Cerebras + Devin estimates + OpenRouter free."""
    from lib import devin_usage as dusage

    cerebras = USAGE.snapshot(run_id=run_id)
    # Drop openrouter rows from cerebras section (shown under OR)
    cb_by = {
        mid: data for mid, data in (cerebras.get("by_model") or {}).items()
        if not str(mid).startswith("openrouter:")
    }
    # Ensure every published Cerebras free/catalog model appears
    for mid, qmeta in CEREBRAS_MODEL_QUOTAS.items():
        if mid not in cb_by:
            cb_by[mid] = {
                "requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "errors": 0,
                "quota": qmeta,
                "window": {},
                "unused": True,
            }
    cerebras["by_model"] = cb_by
    cerebras["keys"] = KEY_POOL.status()
    try:
        cerebras["live_models"] = sorted(list_live_cerebras_models())
    except Exception:
        cerebras["live_models"] = sorted(CEREBRAS_MODEL_QUOTAS.keys())

    try:
        devin = dusage.USAGE.snapshot(run_id=run_id)
    except Exception as e:
        devin = {"ok": False, "error": str(e), "by_model": {}, "windows": {}, "gate": {}}

    openrouter = openrouter_usage_snapshot(run_id=run_id)

    # Summary pressure across providers
    def _hot_cb() -> bool:
        for mid, m in cb_by.items():
            w = (m.get("window") or {}).get("minute") or {}
            soft = float((cerebras.get("gate") or {}).get("soft_ratio") or QUOTA_SOFT_RATIO)
            if (w.get("req_pct") or 0) >= soft * 100 or (w.get("tok_pct") or 0) >= soft * 100:
                return True
        return bool((cerebras.get("gate") or {}).get("waiters"))

    summary = {
        "cerebras_hot": _hot_cb(),
        "devin_hot": bool((devin.get("gate") or {}).get("hot")),
        "openrouter_hot": bool((openrouter.get("gate") or {}).get("hot")),
        "cerebras_keys": (cerebras.get("keys") or {}).get("keys_configured") or 0,
        "cerebras_keys_active": (cerebras.get("keys") or {}).get("keys_active") or 0,
        "high_throughput": bool((cerebras.get("keys") or {}).get("high_throughput")),
        "openrouter_enabled": openrouter.get("enabled"),
        "models_cerebras": len(cb_by),
        "models_devin_free": len((devin.get("catalog") or {}).get("free_models") or devin.get("by_model") or {}),
        "models_openrouter_free": len(openrouter.get("catalog") or openrouter.get("by_model") or {}),
    }
    summary["any_hot"] = summary["cerebras_hot"] or summary["devin_hot"] or summary["openrouter_hot"]

    return {
        "ok": True,
        "started_at": cerebras.get("started_at"),
        "summary": summary,
        "cerebras": cerebras,
        "devin": devin,
        "openrouter": openrouter,
        "note": (
            "Integrative free-model quotas. Cerebras uses published limits × key scale. "
            "Devin 3h caps and OpenRouter free limits are statistical estimates when unpublished."
        ),
    }


def parse_worker_model(model: str) -> tuple[str, str]:
    """Return (provider, provider_model_id).

    Accepts:
      gemma-4-31b
      cerebras:gemma-4-31b
      openrouter:google/gemma-4-31b-it:free
      openrouter/free  (treated as openrouter)
      google/foo:free  (treated as openrouter when free-only)
    """
    m = (model or "").strip()
    if not m:
        return "cerebras", "gemma-4-31b"
    low = m.lower()
    if low.startswith("openrouter:"):
        return "openrouter", m.split(":", 1)[1].strip()
    if low.startswith("cerebras:"):
        return "cerebras", m.split(":", 1)[1].strip()
    if low.startswith("openrouter/") or ":free" in low or low.endswith("/free"):
        return "openrouter", m
    return "cerebras", m


def call_openrouter_sync(
    prompt: str,
    model: str = "openrouter/free",
    max_tokens: int = 8192,
    *,
    run_id: Optional[str] = None,
    purpose: Optional[str] = None,
    temperature: float = 0.3,
) -> str:
    """Call OpenRouter chat completions. Forces free models when OPENROUTER_FREE_ONLY=1."""
    import json as _json
    import urllib.error
    import urllib.request

    key = openrouter_api_key()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    provider, mid = parse_worker_model(model)
    if provider != "openrouter":
        mid = model
    # Strip accidental openrouter: prefix
    if mid.lower().startswith("openrouter:"):
        mid = mid.split(":", 1)[1].strip()

    if OPENROUTER_FREE_ONLY:
        if mid != "openrouter/free" and not mid.endswith(":free") and "/free" not in mid:
            # hard redirect to free router — never spend paid capacity under $0.01 cap
            mid = "openrouter/free"

    run_id = run_id or getattr(_tls, "run_id", None)
    purpose = purpose or getattr(_tls, "purpose", None)
    track_model = f"openrouter:{mid}"

    body = {
        "model": mid,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = _json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{OPENROUTER_API_BASE}/chat/completions",
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER") or "http://localhost:8767",
            "X-Title": os.environ.get("OPENROUTER_APP_TITLE") or "Dev Studio Evolve",
        },
    )
    start = time.time()
    prompt_tokens = estimate_prompt_tokens(prompt)
    completion_tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=float(os.environ.get("OPENROUTER_TIMEOUT", "120"))) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        payload = _json.loads(raw)
        choices = payload.get("choices") or []
        text = ""
        if choices:
            msg = (choices[0] or {}).get("message") or {}
            text = msg.get("content") or ""
            if isinstance(text, list):
                text = "".join(
                    (p.get("text") if isinstance(p, dict) else str(p)) for p in text
                )
        usage = payload.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        if not completion_tokens and text:
            completion_tokens = max(1, len(text) // 4)
        # Record under synthetic model id so dashboards show openrouter traffic
        try:
            USAGE.record(
                track_model,
                prompt_tokens,
                completion_tokens,
                ok=True,
                run_id=run_id,
                purpose=purpose,
                duration_secs=round(time.time() - start, 3),
            )
        except Exception:
            pass
        return text or ""
    except Exception as e:
        err = e
        if isinstance(e, urllib.error.HTTPError):
            try:
                err_body = e.read().decode("utf-8", errors="replace")[:400]
                err = RuntimeError(f"openrouter HTTP {e.code}: {err_body}")
            except Exception:
                err = RuntimeError(f"openrouter HTTP {getattr(e, 'code', '?')}: {e}")
        try:
            USAGE.record(
                track_model,
                prompt_tokens,
                0,
                ok=False,
                run_id=run_id,
                purpose=purpose,
                duration_secs=round(time.time() - start, 3),
                error=str(err),
            )
        except Exception:
            pass
        raise RuntimeError(f"openrouter call failed: {err}") from e


def call_worker_sync(
    prompt: str,
    model: str = "gemma-4-31b",
    max_tokens: int = 8192,
    *,
    run_id: Optional[str] = None,
    purpose: Optional[str] = None,
    temperature: float = 0.3,
) -> str:
    """Route a population-worker call to Cerebras or OpenRouter by model id."""
    provider, mid = parse_worker_model(model)
    if provider == "openrouter":
        return call_openrouter_sync(
            prompt,
            mid,
            max_tokens=max_tokens,
            run_id=run_id,
            purpose=purpose,
            temperature=temperature,
        )
    return call_cerebras_sync(
        prompt,
        mid,
        max_tokens=max_tokens,
        run_id=run_id,
        purpose=purpose,
        temperature=temperature,
    )


def extract_json_block(text: str) -> str:
    """Try to pull a JSON object out of a markdown code block or raw text."""
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    return text.strip()
