"""Generational product HTML, director packs, and cooperation workspace helpers."""
from __future__ import annotations

import html
import json
import re
import shutil
from pathlib import Path
from typing import Any, Optional

from lib import evolution_git as egit


def ensure_exports(run_root: Path) -> Path:
    exp = run_root / "exports"
    exp.mkdir(parents=True, exist_ok=True)
    return exp


def gen_dir(run_root: Path, gen: int) -> Path:
    d = run_root / f"gen{gen}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def product_dir(run_root: Path, gen: int) -> Path:
    d = gen_dir(run_root, gen) / "product"
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_source_files(cand_path: Optional[Path], max_files: int = 40) -> list[str]:
    if not cand_path or not Path(cand_path).exists():
        return []
    skip = {".git", "__pycache__", "node_modules", ".pytest_cache"}
    meta = {
        "project.json", "state.json", "costs.json", "notes.json",
        "build-manifest.json", "artifact.html",
    }
    out: list[str] = []
    root = Path(cand_path)
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in skip for part in rel.parts):
            continue
        if rel.name in meta or rel.as_posix() in meta:
            continue
        out.append(rel.as_posix())
        if len(out) >= max_files:
            break
    return out


def build_candidate_pack(cand: Any, *, run_goal: str = "") -> dict[str, Any]:
    """Compact summary for the decision-maker (one entry per candidate)."""
    path = Path(cand.path) if getattr(cand, "path", None) else None
    build = (getattr(cand, "meta", None) or {}).get("build") or {}
    git = egit.snapshot(path) if path else {}
    genome = getattr(cand, "genome", None) or {}
    return {
        "id": getattr(cand, "id", None),
        "generation": getattr(cand, "generation", None),
        "fitness": round(float(getattr(cand, "fitness", 0) or 0), 3),
        "scores": getattr(cand, "scores", None) or {},
        "brilliant": bool(getattr(cand, "brilliant", False)),
        "rationale": (getattr(cand, "rationale", None) or "")[:400],
        "description": (genome.get("description") or "")[:300],
        "monetization": genome.get("monetization") or {},
        "roles": [c.get("role") for c in (genome.get("cells") or []) if isinstance(c, dict)][:12],
        "build_summary": (build.get("summary") or "")[:400],
        "build_files": (build.get("files") or list_source_files(path))[:30],
        "innovations": (build.get("innovations") or [])[:8],
        "git_head": git.get("head"),
        "git_log": git.get("log") or [],
        "git_diff_stat": (git.get("diff_stat") or "")[:800],
        "git_diff": egit.short_diff(path, max_chars=1800) if path else "",
    }


def director_prompt(goal: str, goal_brief: str, output_type: str, gen: int, packs: list[dict], charter: dict) -> str:
    from lib import deployer_lens as dlens

    pack_json = json.dumps(packs, indent=2)[:14000]
    roles = (charter or {}).get("roles") or []
    thesis = (charter or {}).get("innovation_thesis") or ""
    return f"""You are the DECISION MAKER / product director for a DEPLOYER who ships products for revenue (generation {gen}).

{dlens.DEPLOYER_LENS}

User goal:
{goal}

Brief:
{(goal_brief or goal)[:2500]}

Output type preference: {output_type}
Charter roles: {roles}
Thesis: {thesis[:400]}

Each individual competed this generation. Packs (fitness, builds, git diffs):
{pack_json}

Your job:
1) Score each candidate 0–100 for deployer product value: goal fit + monetization + shippability + authenticity (not architecture beauty alone).
2) Pick a champion_id (must be one of the ids) — prefer the closest path to money + ship.
3) Write a cooperation brief so the WHOLE generation builds one shared monetizable product next.
4) Say what the generational HTML product should be (landing, report, dashboard, catalog, blog funnel, etc.).
{dlens.director_job_extra()}

Return ONLY JSON:
{{
  "rankings": [{{"id": "...", "director_score": 0-100, "why": "..."}}],
  "champion_id": "...",
  "product_direction": "1-2 paragraphs: what we build this gen together and how it makes money",
  "must_have": ["include at least one monetization item", "..."],
  "monetization": {{
    "primary": "stripe|ads|affiliate|content_funnel|aws_resale|mcp_commerce",
    "details": "what is for sale and why it is real",
    "setup_steps": [
      "concrete step the deployer must do (e.g. create Stripe product, connect AWS MCP affiliate feed)"
    ],
    "blockers": ["what will not work until setup is done — be honest"]
  }},
  "merge_plan": ["from <id> take <file or idea>", "..."],
  "html_kind": "website|report|dashboard|catalog|blog|other",
  "html_outline": ["section titles — include Pricing, Monetization setup, Related products when relevant"]
}}
"""


def parse_director_json(text: str) -> dict[str, Any]:
    if not text:
        return {}
    raw = text.strip()
    if "```" in raw:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if m:
            raw = m.group(1).strip()
    try:
        return json.loads(raw)
    except Exception:
        # try first {...}
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
    return {}


def fallback_director(packs: list[dict]) -> dict[str, Any]:
    ranked = sorted(packs, key=lambda p: float(p.get("fitness") or 0), reverse=True)
    rankings = [
        {
            "id": p["id"],
            "director_score": float(p.get("fitness") or 0),
            "why": "fallback: worker fitness (director unavailable)",
        }
        for p in ranked
    ]
    champ = ranked[0]["id"] if ranked else None
    return {
        "rankings": rankings,
        "champion_id": champ,
        "product_direction": (
            "Synthesize the best individual work into a shippable generational product HTML "
            "with a clear monetization path for the deployer (pricing CTA, ads, affiliates, or portfolio links)."
        ),
        "must_have": [
            "summary of goal",
            "champion solution",
            "monetization surface (Stripe/pricing CTA or ads or affiliate or cross-product links)",
            "learnings from peers",
            "next steps for the deployer",
        ],
        "monetization": {"primary": "content_funnel", "details": "pricing CTA + portfolio cross-links"},
        "merge_plan": [f"start from {champ}"] if champ else [],
        "html_kind": "website",
        "html_outline": ["Goal", "Product value", "Pricing / Get access", "Related products", "Lineage", "Next generation"],
        "fallback": True,
    }


def cooperation_brief_md(director: dict, gen: int, goal: str) -> str:
    must = director.get("must_have") or []
    merge = director.get("merge_plan") or []
    outline = director.get("html_outline") or []
    lines = [
        f"# Cooperation brief — generation {gen}",
        "",
        f"**Goal:** {goal}",
        "",
        f"**Champion:** `{director.get('champion_id')}`",
        "",
        "## Product direction",
        "",
        str(director.get("product_direction") or "").strip() or "(none)",
        "",
        "## Must-have",
        "",
    ]
    lines += [f"- {m}" for m in must] or ["- (none)"]
    lines += ["", "## Merge plan", ""]
    lines += [f"- {m}" for m in merge] or ["- (none)"]
    lines += ["", "## HTML outline", ""]
    lines += [f"- {m}" for m in outline] or ["- Goal", "- Results", "- Lineage"]
    lines += ["", f"## Kind: {director.get('html_kind') or 'report'}", ""]
    return "\n".join(lines)


def product_html_prompt(
    goal: str,
    gen: int,
    director: dict,
    packs: list[dict],
    charter: dict,
    existing_product_files: list[str],
    *,
    research_brief: str = "",
    maintainer_learnings: str = "",
) -> str:
    from lib import deployer_lens as dlens

    extra_ctx = ""
    if research_brief:
        extra_ctx += f"\n{research_brief}\n"
    if maintainer_learnings:
        extra_ctx += f"\nMAINTAINER LEARNINGS (apply these improvements):\n{maintainer_learnings[:2500]}\n"

    return f"""You redesign/build the generational PRODUCT for a DEPLOYER (evolution gen {gen}).

{dlens.DEPLOYER_LENS}

Goal: {goal}
Product direction: {director.get('product_direction')}
Must-have: {director.get('must_have')}
Monetization plan: {director.get('monetization') or director.get('must_have')}
HTML kind: {director.get('html_kind')}
Outline: {director.get('html_outline')}
Champion: {director.get('champion_id')}
Charter roles: {(charter or {}).get('roles')}
Candidate packs (compact): {json.dumps(packs, indent=2)[:8000]}
Existing product files: {existing_product_files[:40]}
{extra_ctx}
Write a complete self-contained product as JSON:
{{
  "files": [
    {{"path": "index.html", "content": "<!DOCTYPE html>... full polished HTML ..."}},
    {{"path": "PRODUCT.md", "content": "markdown summary including money path for the deployer"}}
  ],
  "summary": "what this product is and how it makes money"
}}

Rules:
- ALWAYS include index.html (beautiful, readable; inline CSS; no external CDN required if possible).
- If website goal: landing + structure. If research/report: narrative sections + scoreboard + CTAs.
- Include a lineage/scoreboard table of candidates.
- No empty placeholders. Real authentic content from packs / goal / research brief.
- Do NOT invent URLs or market stats not supported by the research brief when present.
{dlens.product_html_rules_extra()}
- CRITICAL: include a visible section id="monetization-setup" titled "Monetization setup (deployer)" that:
  * Lists what is for sale / how money is made (honest — not fake live checkout)
  * Numbered setup checklist (Stripe products, ad network, affiliate program, AWS MCP product feeds, etc.)
  * Marks links as "needs credentials" if they are not wired yet
  * Never pretends a referral/news feed works without a sellable offer behind it
- Leave room at the bottom of body for an automatic provenance footer (do not invent fake signatures).
"""


def template_product_html(
    *,
    goal: str,
    gen: int,
    evo_id: str,
    director: dict,
    packs: list[dict],
    charter: dict,
) -> str:
    rows = ""
    for p in packs:
        rows += (
            f"<tr><td><code>{html.escape(str(p.get('id')))}</code></td>"
            f"<td>{html.escape(str(p.get('fitness')))}</td>"
            f"<td>{html.escape(str((p.get('scores') or {}).get('director_score', '—')))}</td>"
            f"<td>{html.escape(str(p.get('build_summary') or p.get('rationale') or '')[:160])}</td></tr>\n"
        )
    roles = ", ".join((charter or {}).get("roles") or []) or "—"
    champ = director.get("champion_id") or "—"
    direction = html.escape(str(director.get("product_direction") or ""))
    must = "".join(f"<li>{html.escape(str(m))}</li>" for m in (director.get("must_have") or []))
    merge = "".join(f"<li>{html.escape(str(m))}</li>" for m in (director.get("merge_plan") or []))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Gen {gen} product · {html.escape(evo_id)}</title>
<style>
  :root {{ --bg:#f4f7f1; --ink:#1f241f; --hero:#2f5d3a; --line:#d0d8cc; --card:#fff; --muted:#667066; }}
  body {{ margin:0; font-family: system-ui,Segoe UI,sans-serif; background:var(--bg); color:var(--ink); line-height:1.45; }}
  header {{ background:linear-gradient(120deg,#2f5d3a,#3d4f8f); color:#fff; padding:1.4rem 1.5rem; }}
  header h1 {{ margin:0 0 .35rem; font-size:1.35rem; }}
  header p {{ margin:0; opacity:.9; max-width:52rem; }}
  main {{ max-width:960px; margin:0 auto; padding:1.2rem 1rem 3rem; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:1rem 1.1rem; margin:0 0 1rem; box-shadow:0 6px 18px rgba(0,0,0,.04); }}
  h2 {{ margin:.1rem 0 .55rem; color:var(--hero); font-size:1.05rem; }}
  table {{ width:100%; border-collapse:collapse; font-size:.88rem; }}
  th, td {{ text-align:left; padding:.4rem .35rem; border-bottom:1px solid #eef2ea; vertical-align:top; }}
  th {{ color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.04em; }}
  code {{ font-size:.82em; }}
  ul {{ margin:.3rem 0 .3rem 1.1rem; }}
  .meta {{ color:var(--muted); font-size:.82rem; }}
  .pill {{ display:inline-block; background:#e7f3ea; color:var(--hero); border:1px solid #b7d2bf; border-radius:999px; padding:.1rem .5rem; font-size:.75rem; font-weight:700; }}
</style>
</head>
<body>
<header>
  <h1>Generation {gen} product</h1>
  <p>{html.escape(goal[:500])}</p>
  <p class="meta" style="margin-top:.5rem;opacity:.85">run <code>{html.escape(evo_id)}</code> · champion <strong>{html.escape(str(champ))}</strong></p>
</header>
<main>
  <div class="card">
    <h2>Product direction</h2>
    <p>{direction or '<em>Synthesize the best work this generation into a shared artifact.</em>'}</p>
    <p class="meta">Charter roles: {html.escape(roles)}</p>
  </div>
  <div class="card">
    <h2>Must-have</h2>
    <ul>{must or '<li>—</li>'}</ul>
    <h2>Merge plan</h2>
    <ul>{merge or '<li>—</li>'}</ul>
  </div>
  <div class="card">
    <h2>Scoreboard <span class="pill">compete + cooperate</span></h2>
    <table>
      <thead><tr><th>Candidate</th><th>Worker fit</th><th>Director</th><th>Summary</th></tr></thead>
      <tbody>
      {rows or '<tr><td colspan="4">No candidates</td></tr>'}
      </tbody>
    </table>
  </div>
  {monetization_board_html(director, goal)}
  <div class="card">
    <h2>Lineage note</h2>
    <p class="meta">Individuals keep their own git trees. This product workspace is the shared generational artifact — improved from the champion and peer merges for the deployer.</p>
  </div>
</main>
<!-- colophon injected by ensure_product_colophon -->
</body>
</html>
"""


def write_files(target: Path, files: list[dict]) -> list[str]:
    written: list[str] = []
    for item in files or []:
        if not isinstance(item, dict):
            continue
        rel = (item.get("path") or item.get("name") or "").strip().lstrip("/")
        content = item.get("content")
        if not rel or content is None or ".." in rel.split("/"):
            continue
        out = target / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(str(content), encoding="utf-8")
        written.append(rel)
    return written


def monetization_board_html(director: dict, goal: str = "") -> str:
    """Visual deployer checklist injected into product HTML (honest setup, not fake live money)."""
    mon = director.get("monetization") if isinstance(director.get("monetization"), dict) else {}
    primary = html.escape(str(mon.get("primary") or "unset"))
    details = html.escape(str(mon.get("details") or director.get("product_direction") or "")[:500])
    steps = mon.get("setup_steps") or director.get("must_have") or []
    blockers = mon.get("blockers") or [
        "Checkout / affiliate links will not convert until credentials and offers are configured",
        "A news feed or content wall without a paid offer or catalog is not a business",
    ]
    steps_li = "".join(f"<li>{html.escape(str(s))}</li>" for s in steps[:12]) or (
        "<li>Create a Stripe product + price; paste publishable key into the checkout form</li>"
        "<li>Or wire AWS MCP / affiliate catalog and replace placeholder SKUs</li>"
        "<li>Or define one portfolio product this page cross-sells to</li>"
    )
    blockers_li = "".join(f"<li>{html.escape(str(b))}</li>" for b in blockers[:8])
    return f"""
<section id="monetization-setup" style="margin:1.5rem 0;padding:1.1rem 1.2rem;border:2px solid #c7d0f0;border-radius:14px;background:linear-gradient(180deg,#f7f8fd,#fff);box-shadow:0 8px 24px rgba(61,79,143,.08)">
  <div style="display:flex;flex-wrap:wrap;gap:.5rem;align-items:center;justify-content:space-between;margin-bottom:.55rem">
    <h2 style="margin:0;color:#3d4f8f;font-size:1.1rem">💰 Monetization setup <span style="font-size:.72rem;font-weight:700;background:#eef1fb;border:1px solid #c7d0f0;border-radius:999px;padding:.12rem .5rem;color:#3d4f8f">deployer checklist</span></h2>
    <span style="font-size:.75rem;font-weight:800;text-transform:uppercase;letter-spacing:.04em;color:#667066">primary: {primary}</span>
  </div>
  <p style="margin:.2rem 0 .65rem;color:#1f241f;font-size:.9rem;line-height:1.45"><strong>What this product is trying to sell / capture:</strong> {details or html.escape((goal or '')[:280]) or '—'}</p>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:.65rem">
    <div style="background:#fff;border:1px solid #d0d8cc;border-radius:10px;padding:.65rem .75rem">
      <h3 style="margin:0 0 .35rem;font-size:.82rem;color:#2f5d3a;text-transform:uppercase;letter-spacing:.04em">Setup steps (do these for real)</h3>
      <ol style="margin:.2rem 0 0 1.1rem;padding:0;font-size:.84rem;line-height:1.4">{steps_li}</ol>
    </div>
    <div style="background:#fdf6f5;border:1px solid #efc1b8;border-radius:10px;padding:.65rem .75rem">
      <h3 style="margin:0 0 .35rem;font-size:.82rem;color:#a94438;text-transform:uppercase;letter-spacing:.04em">Will not work until wired</h3>
      <ul style="margin:.2rem 0 0 1.1rem;padding:0;font-size:.84rem;line-height:1.4">{blockers_li}</ul>
    </div>
  </div>
  <p style="margin:.7rem 0 0;font-size:.78rem;color:#667066">Coordinator note: referral placeholders and content feeds are <em>not</em> revenue until an offer (SKU, subscription, or resale catalog via AWS MCP / affiliate) is live. This board is the generation's honest monetization plan for you, the deployer.</p>
</section>
"""


def ensure_monetization_board(html_text: str, director: dict, goal: str = "") -> str:
    """Inject monetization board if missing from product HTML."""
    if not html_text:
        return monetization_board_html(director, goal)
    if "id=\"monetization-setup\"" in html_text or "id='monetization-setup'" in html_text:
        return html_text
    board = monetization_board_html(director, goal)
    # Prefer inject before </main> or </body>
    for marker in ("</main>", "</body>", "</html>"):
        if marker in html_text:
            return html_text.replace(marker, board + "\n" + marker, 1)
    return html_text + board


def product_colophon_html(meta: dict) -> str:
    """Footer signature: who edited, models, gens, maintainer learnings count, etc."""
    def esc(s: Any) -> str:
        return html.escape(str(s if s is not None else "—"))

    rows = [
        ("Run id", meta.get("evo_id")),
        ("Generation", meta.get("generation")),
        ("Generations configured", meta.get("generations_cfg")),
        ("Generations completed", meta.get("generations_done")),
        ("Population size", meta.get("population_size")),
        ("LLM calls this run", meta.get("llm_calls")),
        ("Worker model (create/build/eval)", meta.get("worker_model")),
        ("Director / main editor (low RPM)", meta.get("director_model")),
        ("Planner", meta.get("planner_id")),
        ("Research harness", meta.get("research_harness") or "off"),
        ("Maintainer learnings applied", meta.get("maintainer_learnings") or 0),
        ("Seed from", meta.get("seed_from") or "—"),
        ("Best fitness", meta.get("best_fitness")),
        ("Status", meta.get("status")),
        ("Signed at (UTC)", meta.get("signed_at")),
    ]
    trs = "".join(
        f"<tr><th style='text-align:left;padding:.25rem .4rem;color:#667066;font-weight:650;width:42%'>{esc(k)}</th>"
        f"<td style='padding:.25rem .4rem;word-break:break-word'><code style='font-size:.78em'>{esc(v)}</code></td></tr>"
        for k, v in rows
    )
    learnings = meta.get("learning_snippets") or []
    learn_html = ""
    if learnings:
        lis = "".join(f"<li style='margin:.15rem 0'>{esc(x)[:220]}</li>" for x in learnings[:6])
        learn_html = f"<p style='margin:.5rem 0 .2rem;font-size:.78rem;font-weight:700;color:#3d4f8f'>Maintainer learnings folded into this ship</p><ul style='margin:.2rem 0 0 1.1rem;font-size:.78rem;color:#1f241f'>{lis}</ul>"
    return f"""
<footer id="product-colophon" style="margin:2rem 0 0;padding:1rem 1.1rem;border-top:2px solid #cfe0cb;background:#f7faf4;border-radius:0 0 12px 12px;font-size:.8rem;color:#1f241f">
  <div style="display:flex;flex-wrap:wrap;gap:.4rem;align-items:baseline;justify-content:space-between;margin-bottom:.45rem">
    <strong style="color:#2f5d3a;font-size:.9rem">Product provenance · Dev Studio Evolve</strong>
    <span style="font-size:.72rem;color:#667066">signed artifact · not a legal attestation</span>
  </div>
  <table style="width:100%;border-collapse:collapse;font-size:.78rem">{trs}</table>
  {learn_html}
  <p style="margin:.55rem 0 0;font-size:.72rem;color:#667066;line-height:1.35">
    <strong>Main editor</strong> = decision-maker / director model that ranks candidates and writes the shared product HTML.
    <strong>Workers</strong> = high-throughput Cerebras models for create / build / evaluate.
    Research harness (when on) = plan→web search→fetch→synthesize via Cerebras + public HTTP, not a full browser agent.
  </p>
</footer>
"""


def ensure_product_colophon(html_text: str, meta: dict) -> str:
    """Append or replace product colophon footer."""
    colo = product_colophon_html(meta)
    if not html_text:
        return colo
    # replace existing
    if 'id="product-colophon"' in html_text or "id='product-colophon'" in html_text:
        html_text = re.sub(
            r'(?is)<footer[^>]*id=["\']product-colophon["\'][\s\S]*?</footer>',
            colo,
            html_text,
            count=1,
        )
        return html_text
    for marker in ("</body>", "</html>"):
        if marker in html_text:
            return html_text.replace(marker, colo + "\n" + marker, 1)
    return html_text + colo


def publish_product_exports(run_root: Path, gen: int, product: Path) -> dict[str, str]:
    exp = ensure_exports(run_root)
    out: dict[str, str] = {}
    idx = product / "index.html"
    if idx.exists():
        dest = exp / f"gen{gen}-product.html"
        shutil.copy2(idx, dest)
        latest = exp / "PRODUCT-latest.html"
        shutil.copy2(idx, latest)
        out["gen_html"] = str(dest.relative_to(run_root))
        out["latest_html"] = str(latest.relative_to(run_root))
    md = product / "PRODUCT.md"
    if md.exists():
        shutil.copy2(md, exp / "PRODUCT-latest.md")
        out["latest_md"] = "exports/PRODUCT-latest.md"
    return out


def seed_product_from_champion(champion_path: Optional[Path], product: Path) -> None:
    """Copy useful files from champion into product (skip evolution meta)."""
    if not champion_path or not Path(champion_path).exists():
        return
    skip_names = {
        "project.json", "state.json", "costs.json", "notes.json",
        "build-manifest.json",
    }
    skip_dirs = {".git", "__pycache__", "node_modules"}
    src = Path(champion_path)
    for p in src.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(src)
        if any(part in skip_dirs for part in rel.parts):
            continue
        if rel.name in skip_names:
            continue
        # Prefer html/md/css/js and source for product seed
        dst = product / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(p, dst)
        except Exception:
            pass


def soft_inject_product(product: Path, survivor_path: Path, max_files: int = 12) -> list[str]:
    """Copy product index + a few core files into a survivor so cooperation sticks genetically."""
    if not product.exists() or not survivor_path.exists():
        return []
    copied: list[str] = []
    prefer = ["index.html", "PRODUCT.md", "README.md"]
    for name in prefer:
        src = product / name
        if src.exists():
            dst = survivor_path / "product_seed" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(f"product_seed/{name}")
    # a few extra sources
    n = 0
    for p in sorted(product.rglob("*")):
        if n >= max_files:
            break
        if not p.is_file():
            continue
        rel = p.relative_to(product)
        if rel.as_posix() in copied or rel.name in ("index.html", "PRODUCT.md"):
            continue
        if any(x in rel.parts for x in (".git", "__pycache__")):
            continue
        if p.suffix.lower() not in {".py", ".js", ".ts", ".css", ".html", ".md", ".json"}:
            continue
        dst = survivor_path / "product_seed" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(p, dst)
            copied.append(f"product_seed/{rel.as_posix()}")
            n += 1
        except Exception:
            pass
    return copied
