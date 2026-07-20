"""Deployer lens: every evolution is for an operator who ships products to make money.

The human (deployer) runs factories to recycle ideas into shippable, monetizable
surfaces — Stripe/checkout, ads, affiliate/outbound links, and cross-sell blog
posts that point at other products in the portfolio.

Import this module into planner / create / evaluate / director / product HTML
prompts so workers share one north star.
"""
from __future__ import annotations

# Short block injected into LLM prompts (keep under ~1.2k chars).
DEPLOYER_LENS = """
DEPLOYER LENS (you work for the human operator who ships products for revenue):
- Perspective: the DEPLOYER, not a student demo. Success = something shippable that can make money.
- Monetization is MANDATORY in every product — at least one concrete mechanism:
  (a) Stripe / checkout / subscription / one-time payment UI + wiring notes,
  (b) ads slots (placements + network-agnostic placeholders),
  (c) affiliate / outbound commerce links with UTM-ready URLs,
  (d) content funnel (blog/report posts that cross-link OTHER products in our portfolio).
- Factories exist to RECYCLE and MONETIZE: recombine prior candidates into products
  with innovation + volume potential + authenticity (real content, not empty CRUD shells).
- Prefer embeddable payment and growth surfaces over pure architecture diagrams.
- Score and design for: innovativeness, authenticity of content, volume/scale potential,
  and clear path from visitor → value → money.
""".strip()

# Default product-mode fitness weights (sum ~1.0)
PRODUCT_BENCHMARK_WEIGHTS: dict[str, float] = {
    "goal_fit": 0.18,
    "monetization": 0.20,       # payments / ads / affiliates / cross-sell present & credible
    "artifact_quality": 0.14,
    "shippability": 0.12,       # can deploy & take money this week
    "innovation": 0.10,
    "authenticity": 0.08,       # real content / trust / not spam template
    "volume_potential": 0.08,   # can scale distribution or repeat sales
    "evolution_delta": 0.05,
    "implementation": 0.03,
    "continuity": 0.02,
}

FACTORY_BENCHMARK_WEIGHTS: dict[str, float] = {
    "correctness": 0.10,
    "completeness": 0.08,
    "efficiency": 0.06,
    "deployability": 0.10,
    "monetization": 0.16,       # factory should produce monetizable product lines
    "maintainability": 0.08,
    "innovation": 0.14,
    "implementation": 0.12,
    "continuity": 0.10,
    "volume_potential": 0.06,
}

MONETIZATION_KEYWORDS = (
    "stripe", "checkout", "payment", "paywall", "subscribe", "subscription",
    "pricing", "buy now", "add to cart", "billing", "invoice",
    "adsense", "ad slot", "advertisement", "sponsored",
    "affiliate", "utm_", "referral", "partner link",
    "cross-sell", "upsell", "portfolio", "our other", "related product",
    "cta", "call to action", "free trial", "premium",
)


def planner_sections_extra() -> str:
    return """
8) Monetization plan (REQUIRED) — pick primary + secondary:
   - Stripe/checkout or subscriptions (what SKU, price band, where UI lives)
   - Ads placements (where, what inventory)
   - Affiliate / outbound commerce links
   - Content funnel: blog or report posts that link to other products of ours
9) Deployer success metrics — what makes this worth shipping for money
   (conversion path, authenticity, volume lever)
10) Factory recycle angle — what prior work/modules can be reused or cross-sold
""".strip()


def create_mode_rules_product() -> str:
    return (
        "MODE = SHIPPABLE MONETIZABLE PRODUCT (deployer lens).\n"
        "- Design for a human DEPLOYER who wants revenue, not a homework demo.\n"
        "- Cells should include product + growth + monetization roles "
        "(product-lead, frontend, backend, content, growth/monetization, deployer).\n"
        "- build_plan MUST end with a user-visible surface that includes a money path "
        "(Stripe/checkout UI, pricing, ads slot, affiliate links, or cross-product blog CTA).\n"
        "- Prefer authenticity (real topical content) + volume potential + innovation.\n"
        "- Do NOT invent a pure architecture factory unless the goal explicitly asks for factories.\n"
    )


def evaluate_benchmarks_product_text() -> str:
    return """
Benchmarks (DEPLOYER / PRODUCT MODE — score 0–100):
- goal_fit: serves the stated product goal with real content (not a random factory)
- monetization: REQUIRED — clear Stripe/checkout OR ads OR affiliate links OR cross-product CTAs;
  zero monetization surface = score ≤ 25
- artifact_quality: polish of HTML/app surface (structure, readability, trust)
- shippability: could the deployer take this live and accept money / capture leads this week?
- innovation: non-obvious product/monetization angle (not generic "AI wrapper")
- authenticity: real topical content; not spam, lorem, or hollow marketing chrome
- volume_potential: path to many users/sales (SEO, shareability, repeat purchase, catalog)
- evolution_delta: improvement vs parents toward revenue-ready product
- implementation: real files exist (HTML/JS/API stubs for payments ok)
- continuity: reuses useful ancestor modules rather than random rewrite

Penalize hard if: no monetization element, pure factory diagrams when goal is a product,
empty placeholders, or untrustworthy spammy content.
In rationale: one sentence on money path + one on authenticity/volume + one on goal progress.
""".strip()


def director_job_extra() -> str:
    return """
DEPLOYER constraints for scoring & product direction:
- Prefer champions that are closest to a shippable, money-making surface.
- must_have MUST include at least one monetization item (Stripe/checkout, ads, affiliate, or portfolio cross-links).
- product_direction must describe how the shared product makes or enables money for the deployer.
- html_outline should include Pricing / Buy or Ads or Related products (portfolio) section when relevant.
- Reward authenticity + innovation + volume potential over pretty but dead demos.
""".strip()


def product_html_rules_extra() -> str:
    return """
Monetization rules for product HTML (REQUIRED):
- Include at least one: pricing/checkout CTA (Stripe-ready form or clear embed notes),
  ad placement blocks, affiliate/outbound product links, OR a "More from us" portfolio section
  that cross-links other products (even as placeholder routes with real copy).
- Use authentic, specific content tied to the goal — no lorem ipsum walls.
- Make the deployer success path obvious: visitor → value → money or lead.
- Inline CSS; self-contained; shippable static page preferred.
""".strip()


# Instant starter ideas (no LLM) — always shown; Cerebras refreshes richer set.
STARTER_MONEY_IDEAS: list[dict[str, str]] = [
    {
        "id": "niche-guide-stripe",
        "kind": "website",
        "title": "Niche guide + Stripe pack",
        "goal": (
            "Ship a focused niche HTML guide site (authentic, specific topic) with a paid digital pack "
            "via Stripe (PDF/checklist/templates). Include pricing page, checkout CTA, email capture, "
            "and a monetization-setup board. Cross-link 2–3 related portfolio products."
        ),
    },
    {
        "id": "video-explainer-funnel",
        "kind": "video",
        "title": "Explainer video funnel",
        "goal": (
            "Build a monetizable explainer video product: narration script + frame storyboard + "
            "landing page that sells the full course/pack. Use Cronos-style beats for video; "
            "landing has Stripe CTA and ad slots. Setup board for voice model + Stripe keys."
        ),
    },
    {
        "id": "comparison-hub-ads",
        "kind": "website",
        "title": "Comparison hub + ads/affiliates",
        "goal": (
            "Ship an authentic comparison hub (tools/models/APIs) with editorial tables, "
            "affiliate outbound links (UTM-ready), ad placements, and a premium 'pro sheet' Stripe unlock. "
            "Honest monetization-setup: which affiliate programs + ad network to wire."
        ),
    },
    {
        "id": "aws-resale-catalog",
        "kind": "catalog",
        "title": "AWS MCP resale catalog page",
        "goal": (
            "Ship a product catalog HTML page designed for AWS MCP / marketplace resale hooks: "
            "product cards, SKU placeholders, 'needs AWS credentials' banners, Stripe fallback, "
            "and setup steps to connect catalog feeds. Cross-sell our other tools."
        ),
    },
    {
        "id": "micro-saas-landing",
        "kind": "app",
        "title": "Micro-SaaS landing + waitlist",
        "goal": (
            "Ship a polished micro-SaaS landing (problem → demo → pricing) with waitlist and "
            "early-bird Stripe subscription. Include docs snippet and monetization-setup for "
            "billing + domain deploy (GitHub Pages or Cloudflare)."
        ),
    },
    {
        "id": "report-series",
        "kind": "content",
        "title": "Research report series",
        "goal": (
            "Ship a research-style HTML report series homepage with free sample article + "
            "paid full report (Stripe). Include scoreboard of sources, authenticity over fluff, "
            "and portfolio links to our related products."
        ),
    },
]


def money_ideas_prompt(hint: str = "", prior_topics: list[str] | None = None) -> str:
    prior = prior_topics or []
    prior_s = "\n".join(f"- {t}" for t in prior[:12]) or "(none yet)"
    return f"""You invent FRESH shippable, monetizable product ideas for a solo deployer.

Proven money patterns (REUSE the pattern, CHANGE the niche every time — these are infinitely recyclable):
1) Niche guide / checklist / templates + Stripe digital pack
2) Explainer video funnel + landing + Stripe course/pack
3) Comparison hub (tools/APIs/hardware) + ads + affiliate + optional pro unlock
4) Catalog / marketplace page for tools or MCP resale + Stripe fallback
5) Micro-SaaS landing + waitlist + early-bird Stripe
6) Research/report series: free sample article + paid full report
7) Interactive calculator / quiz lead-magnet → paid unlock
8) Portfolio of related mini-products that cross-sell each other

You have: Dev Studio Evolve (HTML products), Cerebras/OpenRouter free for copy, Cronos for video, AWS/GitHub/Cloudflare deploy.

User hint (optional): {hint or "(open — invent best ROI niches right now)"}

Prior product topics already shipped (do NOT repeat titles; new niches only):
{prior_s}

Return ONLY JSON:
{{
  "ideas": [
    {{
      "id": "short-slug",
      "kind": "website|video|app|catalog|content|tool",
      "title": "≤6 words — specific niche, not generic",
      "goal": "1–3 sentences: exact niche, who pays, money path (Stripe/ads/affiliate), what HTML/video artifact to ship this week",
      "why_cheap": "why free LLMs + static HTML is enough",
      "money": "stripe|ads|affiliate|subscription|resale",
      "pattern": "which proven pattern (1–8) you remixed"
    }}
  ]
}}

Rules:
- Exactly 8 ideas. Each uses a different niche (be specific: "kombucha homebrew", "VPS for AI agents", "EU e-invoicing for freelancers" — never "AI tool" or "niche guide").
- Recycle the money *patterns* above 10,000 times — but every title/goal must feel new.
- Mix kinds (not all websites). Prefer authenticity + clear SKU.
- If user hint is set, bias 4+ ideas toward that hint.
"""


def parse_money_ideas_json(text: str) -> list[dict]:
    import json
    import re
    if not text:
        return []
    raw = text.strip()
    if "```" in raw:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if m:
            raw = m.group(1).strip()
    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except Exception:
            return []
    ideas = data.get("ideas") if isinstance(data, dict) else data
    out = []
    for i, idea in enumerate(ideas or []):
        if not isinstance(idea, dict):
            continue
        title = str(idea.get("title") or "").strip()
        goal = str(idea.get("goal") or "").strip()
        if not title or not goal:
            continue
        out.append({
            "id": str(idea.get("id") or f"idea-{i}"),
            "kind": str(idea.get("kind") or "website"),
            "title": title[:80],
            "goal": goal[:800],
            "why_cheap": str(idea.get("why_cheap") or "")[:200],
            "money": str(idea.get("money") or "")[:80],
            "source": "cerebras",
        })
    return out


def monetization_heuristic_score(text: str) -> float:
    """0–100-ish from keyword presence in HTML/code blob."""
    t = (text or "").lower()
    if not t.strip():
        return 12.0
    hits = sum(1 for k in MONETIZATION_KEYWORDS if k in t)
    score = 18.0 + min(70.0, hits * 9.0)
    # Strong signals
    if "stripe" in t or "checkout" in t:
        score += 12
    if "pricing" in t or "subscribe" in t:
        score += 6
    if "affiliate" in t or "utm_" in t:
        score += 5
    if any(x in t for x in ("ad-slot", "adsense", "advertisement", "sponsored")):
        score += 5
    if any(x in t for x in ("related product", "our other", "portfolio", "cross-sell")):
        score += 5
    return max(0.0, min(100.0, score))
