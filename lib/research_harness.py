"""Lightweight research harness for Cerebras workers.

Cerebras itself is a completion API with no tools. To "be smart like Pi" we wrap
it in a small agent loop:

  1) Plan search queries (Cerebras, queued)
  2) Fetch public web snippets (stdlib HTTP — no browser required)
  3) Synthesize findings (Cerebras, queued)

This is intentionally cheap and quota-aware: every LLM call goes through
llm.call_cerebras_sync (soft-quota gate).

Note on CL4R1T4S: that repo is a transparency archive of third-party system
prompts. We do NOT copy proprietary leaked prompts. We use our own open
deployer research system prompt inspired by good agent-loop structure
(plan → tool → synthesize → cite).
"""
from __future__ import annotations

import html as html_lib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

from lib import llm


# Open research system style (not third-party leaked prompts)
RESEARCH_SYSTEM = """You are a careful research assistant for a product deployer.
Be accurate, cite sources by URL when available, prefer primary facts, and mark uncertainty.
Never invent pricing, regulations, or company claims without source support.
Focus on what helps ship and monetize a real product this week.
""".strip()


def _http_get(url: str, timeout: float = 12.0, max_bytes: int = 120_000) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "DevStudioResearchHarness/1.0 (+local; research for product evolution)",
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read(max_bytes)
        charset = "utf-8"
        try:
            charset = resp.headers.get_content_charset() or "utf-8"
        except Exception:
            pass
        return data.decode(charset, errors="replace")


def _strip_html(text: str) -> str:
    text = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", text)
    text = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html_lib.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def duckduckgo_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Best-effort DDG HTML results (no API key)."""
    q = urllib.parse.quote_plus(query[:200])
    url = f"https://html.duckduckgo.com/html/?q={q}"
    try:
        raw = _http_get(url, timeout=10.0)
    except Exception as e:
        return [{"title": "search_error", "url": "", "snippet": str(e)[:200]}]
    results: list[dict[str, str]] = []
    # DDG html: result blocks
    for m in re.finditer(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</(?:a|td|div)',
        raw,
        flags=re.I | re.S,
    ):
        href, title, snip = m.group(1), _strip_html(m.group(2)), _strip_html(m.group(3))
        # unwrap ddg redirect
        if "uddg=" in href:
            try:
                href = urllib.parse.unquote(href.split("uddg=")[1].split("&")[0])
            except Exception:
                pass
        results.append({"title": title[:160], "url": href[:400], "snippet": snip[:320]})
        if len(results) >= max_results:
            break
    if not results:
        # looser fallback
        for m in re.finditer(r'href="(https?://[^"]+)"[^>]*class="result__a"', raw, flags=re.I):
            results.append({"title": m.group(1)[:80], "url": m.group(1)[:400], "snippet": ""})
            if len(results) >= max_results:
                break
    return results


def fetch_page_excerpt(url: str, max_chars: int = 2500) -> dict[str, str]:
    if not url or not url.startswith("http"):
        return {"url": url or "", "excerpt": "", "error": "bad url"}
    try:
        raw = _http_get(url, timeout=12.0)
        text = _strip_html(raw)
        return {"url": url, "excerpt": text[:max_chars], "error": ""}
    except Exception as e:
        return {"url": url, "excerpt": "", "error": str(e)[:200]}


def research_topic(
    topic: str,
    *,
    run_id: Optional[str] = None,
    model: str = "gemma-4-31b",
    max_queries: int = 3,
    max_pages: int = 3,
) -> dict[str, Any]:
    """Plan → search → fetch → synthesize. Returns structured brief for workers."""
    topic = (topic or "").strip()
    if not topic:
        return {"ok": False, "error": "empty topic", "brief": "", "sources": []}

    # 1) Plan queries
    plan_prompt = (
        f"{RESEARCH_SYSTEM}\n\n"
        f"Topic / product goal:\n{topic[:1200]}\n\n"
        "Propose JSON only:\n"
        '{"queries": ["search query 1", "query 2", "query 3"], '
        '"what_we_need": ["facts needed for a shippable product"]}\n'
        f"At most {max_queries} short web search queries focused on market, competitors, pricing norms, and authentic content angles."
    )
    queries: list[str] = [topic[:120]]
    what_we_need: list[str] = []
    try:
        raw = llm.call_worker_sync(
            plan_prompt,
            model=model or "gemma-4-31b",
            max_tokens=600,
            run_id=run_id,
            purpose="research_plan",
            temperature=0.3,
        )
        block = llm.extract_json_block(raw)
        data = json.loads(block)
        qs = data.get("queries") or []
        if isinstance(qs, list) and qs:
            queries = [str(q)[:160] for q in qs[:max_queries]]
        what_we_need = [str(x)[:160] for x in (data.get("what_we_need") or [])[:8]]
    except Exception as e:
        what_we_need = [f"plan_error: {e}"]

    # 2) Search + fetch
    hits: list[dict[str, str]] = []
    for q in queries:
        hits.extend(duckduckgo_search(q, max_results=3))
    # dedupe urls
    seen = set()
    uniq = []
    for h in hits:
        u = h.get("url") or ""
        if u in seen:
            continue
        seen.add(u)
        uniq.append(h)
    pages = []
    for h in uniq[:max_pages]:
        if h.get("url"):
            pages.append(fetch_page_excerpt(h["url"]))

    # 3) Synthesize
    pack = {
        "queries": queries,
        "what_we_need": what_we_need,
        "search_hits": uniq[:8],
        "pages": [
            {"url": p.get("url"), "excerpt": (p.get("excerpt") or "")[:1200], "error": p.get("error")}
            for p in pages
        ],
    }
    synth_prompt = (
        f"{RESEARCH_SYSTEM}\n\n"
        f"Product goal:\n{topic[:1200]}\n\n"
        f"Raw research pack (JSON):\n{json.dumps(pack)[:10000]}\n\n"
        "Write a concise RESEARCH BRIEF (plain text, 250–450 words) with sections:\n"
        "1) Market / audience facts (with URLs when known)\n"
        "2) Competitor / alternative patterns\n"
        "3) Monetization realities (what actually converts)\n"
        "4) Content authenticity tips for this niche\n"
        "5) Risks / unknowns\n"
        "6) Actionable product must-haves for HTML ship this week\n"
    )
    brief = ""
    try:
        brief = llm.call_worker_sync(
            synth_prompt,
            model=model or "gemma-4-31b",
            max_tokens=1200,
            run_id=run_id,
            purpose="research_synth",
            temperature=0.25,
        ).strip()
    except Exception as e:
        brief = f"(research synth failed: {e})\nHits: " + "; ".join(
            f"{h.get('title')}: {h.get('url')}" for h in uniq[:5]
        )

    sources = [{"title": h.get("title"), "url": h.get("url")} for h in uniq[:10] if h.get("url")]
    return {
        "ok": True,
        "brief": brief,
        "sources": sources,
        "queries": queries,
        "model": model,
        "harness": "cerebras+web",
    }


def format_brief_for_prompt(research: Optional[dict[str, Any]], max_chars: int = 3500) -> str:
    if not research or not research.get("brief"):
        return ""
    sources = research.get("sources") or []
    src_lines = "\n".join(f"- {s.get('title') or s.get('url')}: {s.get('url')}" for s in sources[:8])
    body = (
        "WEB RESEARCH BRIEF (from research harness — use facts, do not invent sources):\n"
        f"{research.get('brief')}\n\n"
        f"Sources:\n{src_lines or '(none)'}\n"
    )
    return body[:max_chars]
