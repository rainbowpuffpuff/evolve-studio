"""Persistent usage history for Devin / Cerebras (and future providers).

Append-only JSONL under STUDIO_DATA_DIR/usage/:
  - {provider}-events.jsonl   — every call / external sighting
  - {provider}-snapshots.jsonl — periodic window snapshots for charts

Also keeps a compact rollup JSON for quick dashboard loads.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _data_dir() -> Path:
    return Path(os.environ.get("STUDIO_DATA_DIR", ".")).resolve()


def usage_dir() -> Path:
    d = _data_dir() / "usage"
    d.mkdir(parents=True, exist_ok=True)
    return d


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class UsageHistoryStore:
    """Thread-safe JSONL append + snapshot series for one provider."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        self._lock = threading.Lock()
        self._last_snapshot_ts = 0.0
        self._snapshot_interval = float(os.environ.get("USAGE_SNAPSHOT_INTERVAL_SECS", "60"))
        # Max lines kept when compacting (not on every write)
        self._max_events = int(os.environ.get("USAGE_MAX_EVENTS", "50000"))
        self._max_snapshots = int(os.environ.get("USAGE_MAX_SNAPSHOTS", "20000"))

    def _events_path(self) -> Path:
        return usage_dir() / f"{self.provider}-events.jsonl"

    def _snapshots_path(self) -> Path:
        return usage_dir() / f"{self.provider}-snapshots.jsonl"

    def _rollup_path(self) -> Path:
        return usage_dir() / f"{self.provider}-rollup.json"

    def append_event(self, event: dict[str, Any]) -> None:
        row = {
            "ts": event.get("ts") or utcnow(),
            "ts_unix": event.get("ts_unix") or time.time(),
            **{k: v for k, v in event.items() if k not in ("ts", "ts_unix")},
        }
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with self._lock:
            try:
                with self._events_path().open("a", encoding="utf-8") as f:
                    f.write(line)
            except Exception:
                pass

    def maybe_snapshot(self, snapshot: dict[str, Any], *, force: bool = False) -> bool:
        """Write a periodic compact snapshot for historical graphs."""
        now = time.time()
        with self._lock:
            if not force and (now - self._last_snapshot_ts) < self._snapshot_interval:
                return False
            self._last_snapshot_ts = now
            row = {
                "ts": utcnow(),
                "ts_unix": now,
                "provider": self.provider,
                **snapshot,
            }
            try:
                with self._snapshots_path().open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            except Exception:
                return False
            # Update rollup for quick stats
            try:
                roll = {
                    "provider": self.provider,
                    "updated_at": utcnow(),
                    "last_snapshot": row,
                    "events_path": str(self._events_path()),
                    "snapshots_path": str(self._snapshots_path()),
                }
                self._rollup_path().write_text(json.dumps(roll, indent=2), encoding="utf-8")
            except Exception:
                pass
            return True

    def _read_jsonl(self, path: Path, *, since_unix: Optional[float] = None, limit: int = 5000) -> list[dict]:
        if not path.exists():
            return []
        out: list[dict] = []
        try:
            # Read tail efficiently for large files
            with path.open("r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            for line in lines[-limit:]:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                ts = row.get("ts_unix")
                if since_unix is not None and ts is not None and float(ts) < since_unix:
                    continue
                out.append(row)
        except Exception:
            return []
        return out

    def history(
        self,
        *,
        hours: float = 24.0,
        bucket_mins: int = 15,
        max_points: int = 200,
    ) -> dict[str, Any]:
        """Bucketed series for charts: requests/tokens over time (studio vs external)."""
        since = time.time() - hours * 3600
        snaps = self._read_jsonl(self._snapshots_path(), since_unix=since, limit=8000)
        events = self._read_jsonl(self._events_path(), since_unix=since, limit=20000)

        bucket = max(60, int(bucket_mins) * 60)
        series_map: dict[int, dict[str, float]] = {}

        def bucket_key(ts: float) -> int:
            return int(ts // bucket) * bucket

        # Prefer snapshots when available (already windowed)
        for s in snaps:
            ts = float(s.get("ts_unix") or 0)
            if ts < since:
                continue
            k = bucket_key(ts)
            cell = series_map.setdefault(k, {
                "ts_unix": k,
                "requests": 0,
                "tokens": 0,
                "studio_requests": 0,
                "external_requests": 0,
                "studio_tokens": 0,
                "external_tokens": 0,
                "live_processes": 0,
                "n": 0,
            })
            # Snapshot fields may be nested under windows
            thr = (s.get("windows") or {}).get("three_hours") or s.get("three_hours") or {}
            # Use cumulative-ish last values — for snapshots we store absolute window counts
            # Chart shows last snapshot in each bucket (average if many)
            cell["requests"] = float(thr.get("requests") or s.get("requests") or 0)
            cell["tokens"] = float(thr.get("tokens") or s.get("tokens") or 0)
            cell["studio_requests"] = float(thr.get("studio_requests") or s.get("studio_requests") or 0)
            cell["external_requests"] = float(thr.get("external_requests") or s.get("external_requests") or 0)
            cell["studio_tokens"] = float(thr.get("studio_tokens") or s.get("studio_tokens") or 0)
            cell["external_tokens"] = float(thr.get("external_tokens") or s.get("external_tokens") or 0)
            cell["live_processes"] = float(s.get("live_process_count") or 0)
            cell["n"] += 1

        # Event-derived cumulative throughput (delta per bucket) when few snapshots
        if len(snaps) < 3 and events:
            series_map = {}
            for e in events:
                ts = float(e.get("ts_unix") or 0)
                if ts < since:
                    continue
                k = bucket_key(ts)
                cell = series_map.setdefault(k, {
                    "ts_unix": k,
                    "requests": 0,
                    "tokens": 0,
                    "studio_requests": 0,
                    "external_requests": 0,
                    "studio_tokens": 0,
                    "external_tokens": 0,
                    "live_processes": 0,
                    "n": 0,
                })
                req = float(e.get("requests") or 1)
                tok = float(e.get("tokens") or e.get("total_tokens") or 0)
                src = e.get("source") or "studio"
                cell["requests"] += req
                cell["tokens"] += tok
                if src == "external":
                    cell["external_requests"] += req
                    cell["external_tokens"] += tok
                else:
                    cell["studio_requests"] += req
                    cell["studio_tokens"] += tok
                cell["n"] += 1

        points = [series_map[k] for k in sorted(series_map.keys())]
        if len(points) > max_points:
            # downsample
            step = max(1, len(points) // max_points)
            points = points[::step][:max_points]

        for p in points:
            p["ts"] = datetime.fromtimestamp(p["ts_unix"], tz=timezone.utc).isoformat()

        totals = {
            "requests": sum(int(e.get("requests") or 1) for e in events),
            "tokens": sum(int(e.get("tokens") or e.get("total_tokens") or 0) for e in events),
            "studio_requests": sum(
                int(e.get("requests") or 1) for e in events
                if (e.get("source") or "studio") != "external"
            ),
            "external_requests": sum(
                int(e.get("requests") or 1) for e in events if e.get("source") == "external"
            ),
            "events": len(events),
            "snapshots": len(snaps),
        }

        return {
            "provider": self.provider,
            "hours": hours,
            "bucket_mins": bucket_mins,
            "since": datetime.fromtimestamp(since, tz=timezone.utc).isoformat(),
            "series": points,
            "totals": totals,
            "paths": {
                "events": str(self._events_path()),
                "snapshots": str(self._snapshots_path()),
                "rollup": str(self._rollup_path()),
            },
        }


# Shared stores
DEVIN_HISTORY = UsageHistoryStore("devin")
CEREBRAS_HISTORY = UsageHistoryStore("cerebras")
