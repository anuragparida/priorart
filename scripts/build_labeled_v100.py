"""Generate the Phase 1 labeled benchmark (``evals/labeled_v100.jsonl``).

What this does
---------------
Builds a 100-record benchmark from a curated list of anchor
companies with confirmed direct-description match in the live
corpus, plus 30 long-tail "novel" ideas and 30 "adversarial"
records (paraphrase-with-pivot, market-overlap, same-tech-different-
domain, temporal-evolution).

Construction policy (per ``docs/EVAL.md`` "Construction policy"):

- 40 known-duplicate: pick 20 confirmed YC anchors, write 2
  paraphrasings of each as a "new idea." Each paraphrasing labels
  ``is_duplicate=true`` and lists the anchor's ``company_id`` in
  ``expected_top_ids``.
- 30 known-novel: long-tail ideas with no plausible YC match.
  ``is_duplicate=false``, ``expected_top_ids=[]``.
- 30 adversarial: slight pivots, market overlap, similar-tech-
  different-domain.

Why we generate this from a Python script (and don't hand-edit
the JSONL)
----------------------------------------------------------------
The 40 known-duplicate records share a structured shape
(``idea`` text + ``expected_top_ids=[id]``). Encoding the shape
once and emitting the records makes the labels auditable: each
record's anchor + paraphrase is in one Python data structure,
the JSONL is just the serialised form, and the audit README can
be regenerated from the same source.

The 30 novel and 30 adversarial records are also encoded here as
plain data structures. They are harder to audit but no harder
to read than a hand-typed JSONL would be, and the README links
back to the anchor company for each paraphrase so a reader can
verify by running ``/search`` themselves.

Run
---
``uv run python scripts/build_labeled_v100.py`` (or
``source .venv/bin/activate && python scripts/build_labeled_v100.py``).
The script writes ``evals/labeled_v100.jsonl``.

No LLM labels. Every paraphrase is hand-written by Anurag at the
time this script is updated — see ``evals/labeled_v100.README.md``
for the labeling policy and the change log.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "evals" / "labeled_v100.jsonl"

LABELER = "anurag"
LABELED_AT = "2026-06-28T12:30:00Z"


# ---------------------------------------------------------------------------
# Anchor companies — confirmed top-1 direct-description match in the live
# corpus (see ``docs/EVAL.md`` for what "confirmed" means here). The
# (id, name, description) tuple is exactly what's stored in the
# ``companies`` table for each anchor.
# ---------------------------------------------------------------------------
ANCHORS = [
    (1660, "Draftwise", "Contract and negotiation AI built for lawyers"),
    (1019, "Ironclad", "AI-powered contract management"),
    (3659, "Brex", "Business accounts, corporate cards, and spend management for fast-growing businesses"),
    (3713, "Retool", "Build internal tools fast."),
    (4761, "Dioptra", "Automated contract review with AI"),
    (2635, "Sweetspot", "AI for Government Contracting."),
    (5544, "Toolify", "Build internal tools with AI"),
    (5253, "Resend", "Email for developers"),
    (3203, "Mailgun", "Email API service."),
    (3655, "Bitrise", "Hosted Continuous Integration and Delivery for mobile apps"),
    (3858, "SafetyWing", "Health insurance for remote teams"),
    (3959, "Deel", "The all-in-one HR and payroll platform for global teams"),
    (4314, "Workpay", "Payroll, Benefits and HR solutions for local and remote teams"),
    (1788, "SuperTokens", "Open source alternative to Auth0 / AWS Cognito / Firebase Auth"),
    (4850, "Loops", "Email sending for startups"),
    (5483, "Patchwork", "AI powered communication tool for teams"),
    (2922, "Alter", "Secure access control and authorization platform"),
    (382, "Cubic", "AI-powered code review platform"),
    (2570, "Minded", "Cursor for AI Agents"),
    (2471, "Briefer", "The collaborative data platform."),
]

# Each anchor gets 2 paraphrasings → 40 known-duplicate records.
# The paraphrases are written as if a *founder* were pitching the
# idea cold to a YC partner — the partner would say "this is
# similar to {anchor}" because the underlying product is the same.
PARAPHRASES = {
    1660: [  # Draftwise — contract AI for lawyers
        "AI tool that helps lawyers draft and redline contracts faster",
        "Negotiation copilot for legal teams working on commercial agreements",
    ],
    1019: [  # Ironclad — AI contract lifecycle
        "End-to-end contract management platform with AI-assisted review",
        "Smart contract automation system for in-house legal departments",
    ],
    3659: [  # Brex — corporate cards for startups
        "Corporate charge card and expense platform built for venture-backed companies",
        "All-in-one business banking with cards and spend controls for tech startups",
    ],
    3713: [  # Retool — internal tools builder
        "Low-code platform for building custom internal admin dashboards",
        "Drag-and-drop tool that lets ops teams build their own internal apps",
    ],
    4761: [  # Dioptra — AI contract review
        "AI assistant that reviews legal contracts and flags risky clauses",
        "Automated contract analysis tool for procurement teams",
    ],
    2635: [  # Sweetspot — AI for gov contracting
        "AI platform for winning and managing government contract proposals",
        "Capture management software for federal contractors using AI",
    ],
    5544: [  # Toolify — internal tools with AI
        "AI-native internal tools builder for ops and support teams",
        "Build custom admin panels using natural-language prompts",
    ],
    5253: [  # Resend — email API for devs
        "Developer-first transactional email service with great DX",
        "Modern email API platform for SaaS applications",
    ],
    3203: [  # Mailgun — email API
        "Email delivery and validation service for application developers",
        "SMTP relay and email parsing API for product teams",
    ],
    3655: [  # Bitrise — CI/CD for mobile
        "Mobile-focused continuous integration and delivery platform",
        "Cloud CI/CD service built specifically for iOS and Android apps",
    ],
    3858: [  # SafetyWing — health insurance for remote
        "Global health insurance designed for remote workers and digital nomads",
        "Affordable international medical coverage for distributed teams",
    ],
    3959: [  # Deel — global payroll/HR
        "All-in-one platform to hire, pay, and manage international contractors and employees",
        "Global payroll and compliance service for companies hiring across borders",
    ],
    4314: [  # Workpay — payroll for local + remote
        "Payroll, benefits, and HR platform for companies with mixed local and remote staff",
        "Cloud HR and payroll solution for SMBs in emerging markets",
    ],
    1788: [  # SuperTokens — open source auth
        "Open-source drop-in replacement for Auth0, Firebase Auth, and AWS Cognito",
        "Self-hosted authentication and session management for developers",
    ],
    4850: [  # Loops — email for startups
        "Email marketing platform built specifically for early-stage SaaS startups",
        "Behavioral email tool for product-led growth teams",
    ],
    5483: [  # Patchwork — AI team comms
        "AI-powered internal communication tool for distributed teams",
        "Async-first team messaging with smart summarization built in",
    ],
    2922: [  # Alter — access control
        "Authorization and access-control platform for AI agents and APIs",
        "Fine-grained permissions and policy engine for modern SaaS applications",
    ],
    382: [  # Cubic — AI code review
        "AI code reviewer that catches bugs and style issues on every PR",
        "Automated pull-request review tool powered by large language models",
    ],
    2570: [  # Minded — Cursor for AI agents
        "IDE-like workspace for building and debugging AI agent workflows",
        "Visual development environment for autonomous agent code",
    ],
    2471: [  # Briefer — collaborative data
        "Collaborative notebook for teams that want to query, visualize, and share data",
        "Team-friendly data workspace combining SQL, Python, and dashboards",
    ],
}


# 30 known-novel ideas — long-tail ideas with no plausible YC match.
# ``is_duplicate=False``, ``expected_top_ids=[]``.
NOVEL = [
    "AI tool for composing Persian poetry with traditional meter and rhyme",
    "Subscription service for renting vintage manual typewriters",
    "Online forum for left-handed electricians in Berlin to share tips",
    "Mobile app for ranking the best public drinking fountains in Rome",
    "Subscription box for hand-painted Ukrainian pysanky eggs",
    "Tool for tracking the migratory patterns of European swallows",
    "Marketplace for buying and selling antique Tibetan singing bowls",
    "AI assistant that translates medieval Latin manuscripts into modern English",
    "Platform for organising neighbourhood watch schemes in rural Wales",
    "Booking platform for sundial installation and maintenance",
    "Subscription service for receiving hand-written postcards from strangers",
    "Tool for ranking the best public restroom experiences worldwide",
    "Online course teaching classical Ottoman Turkish calligraphy",
    "Community for amateur mycologists who hunt morel mushrooms in the Pacific Northwest",
    "Marketplace for trading vintage Soviet-era space programme patches",
    "Tool for tracking the blooming cycles of cherry blossoms in Kyoto",
    "Platform connecting amateur astronomers for joint observation nights",
    "Subscription box for artisanal goat cheese from the Pyrenees",
    "Service that audits and certifies the fairness of local carnival games",
    "Tool for planning multi-generational family reunions across five continents",
    "Marketplace for hand-forged Damascus steel kitchen knives",
    "Online community for amateur speleologists exploring lava tubes in Iceland",
    "Subscription for hand-pressed wildflower seed paper stationery",
    "Booking platform for retired sea captains offering day-sail experiences",
    "Tool for cataloguing the locations of historical public telephone boxes in the UK",
    "Service that connects amateur luthiers with rare tonewood suppliers",
    "Marketplace for trading vintage Polaroid SX-70 Land Cameras",
    "Platform for organising community composting programmes in dense urban areas",
    "Subscription for hand-bound blank journals made from recycled agricultural waste",
    "Tool for tracking the migration of humpback whales along the Australian coast",
]


# 30 adversarial records — ideas that look like duplicates but
# aren't quite. Three sub-categories per docs/EVAL.md:
#   - paraphrase_with_pivot:    same pattern, novel target
#   - market_overlap:           adjacent market, not direct competitor
#   - same_tech_diff_domain:    same tech stack, different industry
#   - temporal_evolution:       existed historically, but market has moved
# The first three categories have ``is_duplicate=False`` (the system
# should be uncertain / return low confidence, NOT call it a
# duplicate). The ``temporal_evolution`` records stay
# ``is_duplicate=False`` because the system should flag them as
# adjacent, not direct duplicates.
ADVERSARIAL = [
    # paraphrase_with_pivot (10): "Uber for X" where X is novel.
    ("Uber for chartered bush-plane flights in rural Alaska", "adversarial_paraphrase", []),
    ("Uber for on-demand horse logging services in the Carpathians", "adversarial_paraphrase", []),
    ("Uber for licensed falconers offering pest-control services at vineyards", "adversarial_paraphrase", []),
    ("Uber for mobile sheep-shearing teams in New Zealand", "adversarial_paraphrase", []),
    ("Uber for vintage Vespa mechanics offering roadside repairs", "adversarial_paraphrase", []),
    ("Uber for licensed ice-carvers serving cocktails at corporate events", "adversarial_paraphrase", []),
    ("Uber for on-demand blacksmiths doing farm-equipment repair", "adversarial_paraphrase", []),
    ("Uber for mobile leather-repair workshops at farmers markets", "adversarial_paraphrase", []),
    ("Uber for licensed farriers serving rural horse owners", "adversarial_paraphrase", []),
    ("Uber for mobile violin-luthiers doing on-site instrument repair", "adversarial_paraphrase", []),
    # market_overlap (10): adjacent market, not direct competitor.
    ("Stripe for cross-border B2B payments in West Africa", "adversarial_market_overlap", []),
    ("Stripe for marketplace seller payouts with embedded KYC", "adversarial_market_overlap", []),
    ("Stripe for instant settlement on creator economy platforms", "adversarial_market_overlap", []),
    ("Stripe for invoice financing and working capital for SMBs", "adversarial_market_overlap", []),
    ("Stripe for stablecoin payment routing across chains", "adversarial_market_overlap", []),
    ("Stripe for tipping and gratuity infrastructure in hospitality", "adversarial_market_overlap", []),
    ("Stripe for high-value art auction escrow", "adversarial_market_overlap", []),
    ("Stripe for marketplace dispute resolution and chargeback handling", "adversarial_market_overlap", []),
    ("Stripe for invoice factoring with AI credit scoring", "adversarial_market_overlap", []),
    ("Stripe for split-tender payments at restaurants with multiple concepts", "adversarial_market_overlap", []),
    # same_tech_diff_domain (10): same tech, different vertical.
    ("GitHub for machine-learning model versioning and experiment tracking", "adversarial_same_tech_diff_domain", []),
    ("GitHub for legal contract version control with redlining", "adversarial_same_tech_diff_domain", []),
    ("GitHub for genomic data versioning in biotech research", "adversarial_same_tech_diff_domain", []),
    ("GitHub for design-asset versioning with Figma integration", "adversarial_same_tech_diff_domain", []),
    ("GitHub for 3D-scene versioning in animation studios", "adversarial_same_tech_diff_domain", []),
    ("GitHub for music sheet versioning with audio playback diff", "adversarial_same_tech_diff_domain", []),
    ("GitHub for clinical-trial protocol versioning with audit logs", "adversarial_same_tech_diff_domain", []),
    ("GitHub for geospatial dataset versioning in urban planning", "adversarial_same_tech_diff_domain", []),
    ("GitHub for construction-blueprint versioning on infrastructure projects", "adversarial_same_tech_diff_domain", []),
    ("GitHub for pharmaceutical-formulation versioning with regulatory hooks", "adversarial_same_tech_diff_domain", []),
]


# ---------------------------------------------------------------------------
# Build the records
# ---------------------------------------------------------------------------

def _next_id(idx: int) -> str:
    """Stable record id: ``ev-001`` .. ``ev-100``."""
    return f"ev-{idx:03d}"


def build_records() -> list[dict]:
    """Assemble the 100 records in deterministic order:
    40 dup + 30 novel + 30 adversarial.
    """
    records: list[dict] = []
    counter = 1

    # 40 known-duplicate: 2 paraphrases per anchor.
    for cid, name, _desc in ANCHORS:
        for paraphrase in PARAPHRASES[cid]:
            records.append(
                {
                    "id": _next_id(counter),
                    "idea": paraphrase,
                    "source": "yc",
                    "category": "duplicate",
                    "expected_top_ids": [cid],
                    "is_duplicate": True,
                    "labeler": LABELER,
                    "labeled_at": LABELED_AT,
                    "notes": f"Hand-written paraphrase of {name} (YC id={cid})",
                }
            )
            counter += 1

    assert counter == 41, f"expected 40 dup records, got {counter - 1}"

    # 30 known-novel: long-tail ideas with no plausible YC match.
    for idea in NOVEL:
        records.append(
            {
                "id": _next_id(counter),
                "idea": idea,
                "source": "synthetic",
                "category": "novel",
                "expected_top_ids": [],
                "is_duplicate": False,
                "labeler": LABELER,
                "labeled_at": LABELED_AT,
                "notes": "Long-tail idea with no plausible YC match.",
            }
        )
        counter += 1

    assert counter == 71, f"expected 70 records (40+30) so far, got {counter - 1}"

    # 30 adversarial.
    for idea, category, expected in ADVERSARIAL:
        records.append(
            {
                "id": _next_id(counter),
                "idea": idea,
                "source": "synthetic",
                "category": category,
                "expected_top_ids": expected,
                "is_duplicate": False,
                "labeler": LABELER,
                "labeled_at": LABELED_AT,
                "notes": f"Adversarial: {category}.",
            }
        )
        counter += 1

    assert counter == 101, f"expected 100 records, got {counter - 1}"
    return records


def main() -> None:
    records = build_records()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} records to {OUTPUT_PATH}")

    # Quick stats
    by_cat: dict[str, int] = {}
    for r in records:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    print("by category:", by_cat)


if __name__ == "__main__":
    main()