"""Local git trees for evolution candidates and generational products.

All operations are best-effort: missing git or commit failures never kill a run.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional


_GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "evolution",
    "GIT_AUTHOR_EMAIL": "evolution@dev-studio.local",
    "GIT_COMMITTER_NAME": "evolution",
    "GIT_COMMITTER_EMAIL": "evolution@dev-studio.local",
    "GIT_TERMINAL_PROMPT": "0",
}


def git_available() -> bool:
    return bool(shutil.which("git"))


def _run(path: Path, args: list[str], *, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(path),
        capture_output=True,
        text=True,
        env=_GIT_ENV,
        timeout=60,
        check=check,
    )


def is_repo(path: Path) -> bool:
    return (path / ".git").exists()


def init_repo(path: Path) -> bool:
    if not git_available() or not path.exists():
        return False
    if is_repo(path):
        return True
    try:
        r = _run(path, ["init"])
        if r.returncode != 0:
            return False
        _run(path, ["config", "user.email", "evolution@dev-studio.local"])
        _run(path, ["config", "user.name", "evolution"])
        # Ignore heavy / volatile paths if present later
        ig = path / ".gitignore"
        if not ig.exists():
            ig.write_text(
                "__pycache__/\n*.pyc\n.pytest_cache/\nnode_modules/\n.env\n",
                encoding="utf-8",
            )
        return True
    except Exception:
        return False


def commit(path: Path, message: str, *, allow_empty: bool = False) -> Optional[str]:
    """Stage all and commit. Returns short SHA or None."""
    if not git_available() or not path.exists():
        return None
    try:
        if not is_repo(path):
            if not init_repo(path):
                return None
        _run(path, ["add", "-A"])
        args = ["commit", "-m", message[:200]]
        if allow_empty:
            args.append("--allow-empty")
        r = _run(path, args)
        if r.returncode != 0:
            # nothing to commit is ok
            if "nothing to commit" in (r.stdout or "") + (r.stderr or ""):
                return head_sha(path)
            return None
        return head_sha(path)
    except Exception:
        return None


def head_sha(path: Path) -> Optional[str]:
    if not is_repo(path):
        return None
    try:
        r = _run(path, ["rev-parse", "--short", "HEAD"])
        if r.returncode == 0:
            return (r.stdout or "").strip() or None
    except Exception:
        pass
    return None


def log(path: Path, n: int = 8) -> list[str]:
    if not is_repo(path):
        return []
    try:
        r = _run(path, ["log", f"-{n}", "--oneline", "--no-decorate"])
        if r.returncode != 0:
            return []
        return [ln for ln in (r.stdout or "").splitlines() if ln.strip()]
    except Exception:
        return []


def diff_stat(path: Path, against: str = "HEAD~1") -> str:
    if not is_repo(path):
        return ""
    try:
        r = _run(path, ["diff", "--stat", against, "HEAD"])
        if r.returncode != 0:
            # first commit
            r2 = _run(path, ["show", "--stat", "--oneline", "HEAD"])
            return (r2.stdout or "")[:1500]
        return (r.stdout or "")[:1500]
    except Exception:
        return ""


def short_diff(path: Path, against: str = "HEAD~1", max_chars: int = 2500) -> str:
    if not is_repo(path):
        return ""
    try:
        r = _run(path, ["diff", against, "HEAD", "--", ".", ":(exclude).git"])
        if r.returncode != 0:
            return ""
        text = r.stdout or ""
        if len(text) > max_chars:
            return text[:max_chars] + "\n…[diff truncated]"
        return text
    except Exception:
        return ""


def clone_local(parent: Path, child: Path) -> bool:
    """Prefer local clone to preserve history; fall back to False for caller to copy files."""
    if not git_available() or not is_repo(parent):
        return False
    try:
        if child.exists():
            shutil.rmtree(child)
        r = subprocess.run(
            ["git", "clone", "--local", str(parent), str(child)],
            capture_output=True,
            text=True,
            env=_GIT_ENV,
            timeout=90,
        )
        return r.returncode == 0 and is_repo(child)
    except Exception:
        return False


def snapshot(path: Path) -> dict[str, Any]:
    return {
        "is_repo": is_repo(path) if path else False,
        "head": head_sha(path) if path else None,
        "log": log(path, 6) if path else [],
        "diff_stat": diff_stat(path) if path else "",
    }
