"""Business-category assignment for eval records (Phase 3.4).

What this is
------------
Per ``docs/PHASE-3.md`` §3.4, the eval set gains a *business category*
field per record — orthogonal to the existing ``category`` (which
encodes the eval-set taxonomy: ``duplicate`` / ``novel`` /
``adversarial_*``). The business categories are the 8 PHASE-3.md
buckets:

    b2b_saas, consumer, devtools, marketplace, fintech,
    healthcare, education, other

Each eval record needs exactly one ``business_category`` so the
per-category failure breakdown (also delivered by this card) has
something to group on.

Why a deterministic rule-based assigner (and not LLM)
-----------------------------------------------------
The Phase 3.3 / Phase 2.8 discipline is that the eval set is
**LLM-generated v2, pending Anurag hand-review.** If the category
field were also LLM-generated on top of that, the failure analysis
would be *double*-unverified: both the labels and the categories
would be unverified. Worse, with no ``ANTHROPIC_API_KEY`` in the
test host, an LLM-backed assigner either silently fails (placeholder
detection) or requires an external API call the harness can't make.

This module is therefore a **pure deterministic rule-based assigner.**
It uses regex-keyword matching against the record's ``idea`` text.
The rules are:

  - keyword sets per category, ordered from most-specific
    (e.g. ``"BAAI/bge"`` → ``devtools``) to most-generic (``SaaS`` →
    ``b2b_saas``).
  - First-match wins; if no rule fires, the record is labeled
    ``other``.
  - All rules are deterministic, single-pass, no retries — matches
    the card's "low temperature, single-pass, no retries" spirit
    for an LLM-backed assigner, while being reproducible offline.

The honest acknowledgment: rule-based assignment on natural-language
idea descriptions will not classify every record cleanly. Records
that match no rule land in ``BusinessCategory.OTHER``. The
``category_coverage`` function surfaces this distribution so a
reader can see how the rule set performed — the failure analysis
is honest about its own coverage.

Why not just hand-label
-----------------------
Anurag's standing rule (per Phase 2.8 / commit ``5c1c8fa``) is that
the hand-label pass is a follow-up to the LLM-generation step. The
business category is *also* a label on the same record, so the same
discipline applies: a deterministic rule-based assigner is the
honest v1, the hand-label pass upgrades it later.

Provenance stamp
----------------
Every record in the JSONL gets a ``business_category_provenance``
field. The deterministic rule-based assigner stamps
``deterministic-rule-based-v1-pending-anurag-hand-review``. Same
honesty pattern as the eval set's own ``provenance`` field.

Outputs
-------
- ``assign_business_category(idea, ...)`` → ``CategoryAssignment``.
- ``assign_business_categories(benchmark, ...)`` → returns a
  ``{record_id: BusinessCategory}`` dict, used by the eval-runner
  extension in ``src/eval/run.py``.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Public taxonomy — must stay in sync with docs/PHASE-3.md §3.4
# ---------------------------------------------------------------------------


class BusinessCategory(str, enum.Enum):
    """The 8 PHASE-3.md §3.4 business categories.

    Inheriting from ``str`` so the values JSON-serialise cleanly and
    can be compared with bare-string ``category`` values from the
    eval set (which are free-form in Phase 1 but include
    ``"duplicate"``, ``"novel"``, ``"adversarial_*"``).
    """

    B2B_SAAS = "b2b_saas"
    CONSUMER = "consumer"
    DEVTOOLS = "devtools"
    MARKETPLACE = "marketplace"
    FINTECH = "fintech"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    OTHER = "other"


# Ordered list (for stable iteration in the breakdown table).
BUSINESS_CATEGORIES: Tuple[BusinessCategory, ...] = (
    BusinessCategory.B2B_SAAS,
    BusinessCategory.CONSUMER,
    BusinessCategory.DEVTOOLS,
    BusinessCategory.MARKETPLACE,
    BusinessCategory.FINTECH,
    BusinessCategory.HEALTHCARE,
    BusinessCategory.EDUCATION,
    BusinessCategory.OTHER,
)


# Stable human label for the failure-breakdown PNG / MD.
CATEGORY_LABEL: Dict[BusinessCategory, str] = {
    BusinessCategory.B2B_SAAS: "B2B SaaS",
    BusinessCategory.CONSUMER: "Consumer",
    BusinessCategory.DEVTOOLS: "Devtools",
    BusinessCategory.MARKETPLACE: "Marketplace",
    BusinessCategory.FINTECH: "Fintech",
    BusinessCategory.HEALTHCARE: "Healthcare",
    BusinessCategory.EDUCATION: "Education",
    BusinessCategory.OTHER: "Other",
}


# Provenance stamp — matches the Phase 2.8 discipline (commit
# 5c1c8fa). The hand-label pass replaces this with ``anurag-v2``.
DEFAULT_PROVENANCE = "deterministic-rule-based-v1-pending-anurag-hand-review"


# ---------------------------------------------------------------------------
# Rule table — ordered from most-specific to most-generic
# ---------------------------------------------------------------------------
#
# Each rule is (compiled-regex, BusinessCategory). First match wins.
# The order matters: ``"neobank"`` must be checked before ``"bank"``,
# ``"marketplace"`` before ``"market"``, etc. Adding a rule here
# requires bumping the version suffix in DEFAULT_PROVENANCE because
# the rule set is part of the deterministic contract.


def _compile_rules() -> List[Tuple[re.Pattern[str], BusinessCategory]]:
    raw: List[Tuple[str, BusinessCategory]] = [
        # ---------------------------------------------------------------
        # DEVTOOLS — developer-facing tools first
        # ---------------------------------------------------------------
        (r"\b(llm|rag|gpt-4|gpt-3|embedding|vector\s+(?:database|db|search|store)|"
         r"transformer|diffusion|fine[\s-]?tun|inference\s+server|"
         r"model\s+context\s+protocol|model\s+context|mcp|"
         r"langchain|llamaindex|llama\s*index|hugging\s*face|pinecone|weaviate|"
         r"qdrant|chroma|milvus|"
         r"openai\s+api|anthropic\s+api|prompt\s+engineer|prompt\s+management|"
         r"llm\s+ops|llmops|mlops|ml\s+platform|"
         r"rag\s+for\s+\w+|retrieval[\s-]?augmented)\b", BusinessCategory.DEVTOOLS),
        (r"\b(devtool|developer\s+tool|developer\s+experience|developer[\s-]?first|"
         r"developer[\s-]?facing|developer\s+portal|"
         r"\bcli\b|\bsdk\b|api\s+gateway|webhook|"
         r"infrastructure[\s-]?as[\s-]?code|"
         r"terraform|kubernetes|\bdocker\b|ci[\s/]?cd|github\s+app|gitlab|"
         r"observability|monitoring|logging|tracing|"
         r"serverless|edge\s+function|edge\s+compute|"
         r"build\s+tool|package\s+manager|"
         r"\bcompiler\b|\binterpreter\b|runtime|\bdebugger\b|"
         r"\bide\b|vscode|jetbrains|"
         r"unit\s+test|integration\s+test|test\s+automation|static\s+analysis|"
         r"lint|format|pre[\s-]?commit|code\s+review|"
         r"data\s+pipeline|etl|\belt\b|orchestration|workflow\s+engine|"
         r"message\s+queue|event[\s-]?bus|kafka|rabbitmq|"
         r"feature\s+flag|feature\s+management|"
         r"version[\s-]?control|\bgit\b|github\s+action|"
         r"container\s+orchestration|container\s+security|"
         r"\bplaywright\b|\bselenium\b|"
         r"issue\s+tracker|bug\s+tracker|"
         r"code\s+editor|code\s+generation|"
         r"browser\s+automation|"
         r"dataops|devops|devsecops|"
         r"\bsast\b|\bdast\b|"
         r"feature\s+store|feature\s+engineering|"
         r"model\s+registry|model\s+deployment|"
         r"vector\s+store|vector\s+search|"
         r"semantic\s+search\s+(?:infrastructure|engine|platform)|"
         r"workflow\s+automation|automation\s+platform|"
         r"\brpa\b|robotic\s+process\s+automation|"
         r"no[\s-]?code|low[\s-]?code|"
         r"ci[\s/]?cd\s+platform|"
         r"data\s+quality|data\s+observability|"
         r"data\s+lineage|data\s+catalog|data\s+governance|"
         r"data\s+contract|"
         r"load\s+testing|performance\s+testing|"
         r"api\s+monitoring|api\s+observability|"
         r"api\s+documentation|api\s+client|graphql\s+client|"
         r"data\s+integration|data\s+synchronization|"
         r"reverse\s+etl|\bcdc\b|change\s+data\s+capture|"
         r"data\s+mesh|data\s+lake|lakehouse|"
         r"data\s+warehouse|data\s+platform|"
         r"sql\s+editor|query\s+engine|"
         r"service\s+mesh|microservice|"
         r"\biam\b|identity\s+and\s+access|zero\s+trust|"
         r"secrets?\s+management|secret\s+rotation|"
         r"\bsso\b|single\s+sign[\s-]?on|"
         r"\bmfa\b|multi[\s-]?factor\s+auth|"
         r"passwordless|passkey|"
         r"error\s+tracking|session\s+replay|"
         r"apm|application\s+performance\s+monitoring|"
         r"log\s+management|log\s+aggregation|"
         r"\bsiem\b|threat\s+detection|"
         r"\bedr\b|\bxdr\b|endpoint\s+detection|"
         r"vulnerability\s+management|vulnerability\s+scanning|"
         r"pen[\s-]?test|penetration\s+testing|"
         r"security\s+operations|security\s+orchestration|\bsoar\b|"
         r"network\s+monitoring|network\s+management|"
         r"\bsd[\s-]?wan\b|wan\s+optimization|"
         r"remote\s+access\s+(?:for|tool|platform)|vpn\s+for\s+\w+|"
         r"privilege[ds]?\s+access|\bpam\b|"
         r"customer\s+identity|\bciam\b|"
         r"directory\s+service|active\s+directory|"
         r"deployment\s+platform|"
         r"infrastructure\s+monitoring|"
         r"infrastructure\s+automation|"
         r"deploy\s+pipeline|build\s+and\s+deploy|"
         r"package\s+registry|artifact\s+registry|"
         r"dependency\s+management|"
         r"container\s+scan|"
         r"compliance\s+as\s+code|policy\s+as\s+code|"
         r"\bgitops\b|argo\s+cd|fluxcd|"
         r"infrastructure\s+management|"
         r"infrastructure\s+platform|"
         r"backend\s+as\s+a\s+service|baas\s+for\s+devs|"
         r"cloud\s+development\s+environment|cloud\s+ide|"
         r"web\s+framework|backend\s+framework|"
         r"frontend\s+framework|ui\s+framework|"
         r"component\s+library|ui\s+kit|"
         r"code\s+execution|sandbox\s+for\s+code|"
         r"workflow\s+engine\s+for\s+developers|"
         r"headless\s+backend|headless\s+api|"
         r"internal\s+developer\s+platform|"
         r"build\s+system|"
         r"code\s+coverage|"
         r"open[\s-]?source|"
         r"developer\s+centric|developer[\s-]?facing|"
         r"infrastructure\s+for\s+developers|"
         r"workflow\s+builder|workflow\s+designer|"
         r"ai\s+infra|ai\s+infrastructure|"
         r"ai\s+gateway|llm\s+gateway|"
         r"agent\s+framework|agent\s+platform|"
         r"ai\s+security\s+for\s+developers|"
         r"ai\s+governance|ai\s+compliance|ai\s+risk\s+management|"
         r"model\s+interpretability|explainable\s+ai|"
         r"ai\s+red\s+team|red\s+teaming\s+for\s+ai|"
         r"differential\s+privacy|federated\s+learning|"
         r"homomorphic\s+encryption|secure\s+multi[\s-]?party|"
         r"trusted\s+execution|confidential\s+computing|"
         r"smart\s+contract\s+audit|smart\s+contract\s+security|"
         r"api\s+for\s+(?:a\s+)?(?:developers|software|engineers))\b",
         BusinessCategory.DEVTOOLS),

        # ---------------------------------------------------------------
        # FINTECH — banks, payments, crypto, insurance, accounting
        # ---------------------------------------------------------------
        (r"\b(neobank|neo[\s-]?bank|banking\s+app|banking\s+platform|"
         r"core\s+banking|digital\s+bank|digital\s+banking|open\s+banking|open\s+finance|"
         r"payments?|payment\s+gateway|payment\s+processor|payment\s+platform|"
         r"stripe\s+competitor|stripe\s+alternative|stripe\s+for|"
         r"invoicing|accounts?\s+receivable|accounts?\s+payable|"
         r"bookkeep|accounting\s+software|accounting\s+platform|cfo\s+tool|"
         r"expense\s+management\s+(?:for\s+(?:a\s+)?(?:business|company|enterprise|team|smb|startup))|"
         r"expense\s+report\s+(?:for\s+(?:a\s+)?(?:business|company|enterprise|team))|"
         r"corporate\s+card|corporate\s+card\s+for|"
         r"credit\s+card\s+for|credit\s+score|credit\s+report|underwrit|"
         r"loan\s+platform|lending|lender|mortgage|"
         r"insurance|insurtech|claims\s+automation|"
         r"crypto|bitcoin|ethereum|web3|\bdefi\b|"
         r"stablecoin|\bnft\b|wallet\s+for\s+crypto|on[\s-]?chain|"
         r"trading\s+platform|trading\s+app|\bbrokerage\b|stock\s+app|"
         r"investment\s+app|wealth\s+management|robo[\s-]?advisor|"
         r"financial\s+planning|personal\s+finance|budget\s+app|"
         r"tax\s+preparation|tax\s+filing|payroll|payroll\s+platform|"
         r"treasury|treasury\s+management|remittance|cross[\s-]?border\s+payment|"
         r"\bkyc\b|\baml\b|embedded\s+finance|"
         r"banking[\s-]?as[\s-]?a[\s-]?service|\bbaas\b|"
         r"finance\s+app|financial\s+app|financial\s+tool|"
         r"trading\s+bot|algo[\s-]?trading|algorithmic\s+trading|"
         r"portfolio\s+management|portfolio\s+tracker|"
         r"asset\s+management|asset\s+manager|"
         r"hedge\s+fund|private\s+equity|venture\s+debt|"
         r"financial\s+advisor|financial\s+planner|"
         r"wealthtech|fintech|"
         r"credit\s+monitoring|credit\s+building|"
         r"debt\s+management|debt\s+payoff|"
         r"savings\s+app|savings\s+account|high\s+yield\s+savings|"
         r"checking\s+account|money\s+transfer|wire\s+transfer|"
         r"buy\s+now\s+pay\s+later|\bbnpl\b|"
         r"installment\s+loan|payday\s+loan|"
         r"tax\s+software|tax\s+optimizer|tax\s+planning|"
         r"audit\s+tool|financial\s+audit|"
         r"invoice\s+financing|invoice\s+factoring|"
         r"revenue\s+recognition|"
         r"card\s+processor|card\s+processing|"
         r"\bpos\b|point[\s-]?of[\s-]?sale|"
         r"merchant\s+services|merchant\s+account|"
         r"\batm\b|bitcoin\s+atm|crypto\s+exchange|"
         r"yield\s+farming|liquidity\s+pool|"
         r"tokenization|security\s+token|"
         r"programmable\s+payment|programmable\s+money|"
         r"cross[\s-]?border\s+remittance|"
         r"compliance\s+for\s+banking|regtech|"
         r"risk\s+management\s+for\s+financial|risk\s+modeling|"
         r"fraud\s+prevention\s+for\s+financial|fraud\s+detection\s+for\s+bank|"
         r"loan\s+origination|loan\s+servicing|loan\s+underwriting|"
         r"credit\s+decisioning|"
         r"financial\s+reporting|financial\s+close|"
         r"financial\s+operations|finops|"
         r"ap\s+automation|accounts\s+payable\s+automation|"
         r"ar\s+automation|accounts\s+receivable\s+automation|"
         r"expense\s+report\s+app|expense\s+automation|"
         r"reconciliation\s+tool|reconciliation\s+automation|"
         r"revenue\s+operations|revops|"
         r"pricing\s+for\s+saas|saas\s+pricing|"
         r"subscription\s+billing|recurring\s+billing|"
         r"billing\s+platform|billing\s+automation|"
         r"financial\s+close\s+automation|"
         r"tax\s+compliance|tax\s+filing\s+automation|"
         r"loan\s+application|loan\s+origination\s+system|"
         r"home\s+equity|reverse\s+mortgage|"
         r"wealth\s+advisory|family\s+office|"
         r"private\s+banking|private\s+wealth|"
         r"institutional\s+trading|prime\s+brokerage|"
         r"marketplace\s+lending|peer[\s-]?to[\s-]?peer\s+lending|"
         r"crowdfunding|rewards\s+program\s+for\s+credit|"
         r"credit\s+card\s+rewards|cashback\s+app|"
         r"money\s+management|money\s+tracking|"
         r"subscription\s+management|subscription\s+analytics|"
         r"financial\s+wellness|paycheck\s+advance|earned\s+wage|"
         r"tax\s+compliance\s+for\s+(?:a\s+)?(?:business|companies|teams|enterprise|smb|startup))\b",
         BusinessCategory.FINTECH),

        # ---------------------------------------------------------------
        # HEALTHCARE — clinical, pharma, providers, payers, mental health
        # ---------------------------------------------------------------
        (r"\b(clinical\s+decision|clinical\s+support|clinical\s+trial|"
         r"electronic\s+health\s+record|\behr\b|\bemr\b|epic\s+integration|"
         r"\bhl7\b|\bfhir\b|\bhipaa\b|phi\s+compliant|hipaa[\s-]?compliant|"
         r"hospital\s+manag|"
         r"practice\s+management\s+for\s+(?:a\s+)?(?:clinic|doctor|hospital|provider|vet|dentist|dental)|"
         r"medical\s+record|medical\s+billing|medical\s+device|medtech|"
         r"telemedicine|telehealth|\brpm\b|remote\s+patient\s+monitoring|"
         r"digital\s+therapeutic|\bdtx\b|pharma|pharmaceutical|drug\s+discovery|"
         r"drug\s+repurpos|drug\s+interaction|medication\s+adherence|"
         r"prescription|\brx\s+refill\b|pharmacy|diagnostic|imaging|radiology|"
         r"pathology|genomic|genomic\s+data|genetic\s+test|"
         r"mental\s+health|therapist\s+app|therapy\s+app|counseling\s+app|"
         r"depression\s+app|anxiety\s+app|sleep\s+app|mindfulness\s+app|"
         r"meditation\s+app|sobriety\s+app|addiction\s+app|"
         r"fitness\s+app|workout\s+app|wearable\s+health|health\s+wearable|"
         r"primary\s+care|urgent\s+care|specialist\s+referral|"
         r"nurse\s+staffing|doctor\s+scheduling|provider\s+directory|"
         r"value[\s-]?based\s+care|revenue\s+cycle|hospital\s+it|"
         r"health\s+plan\s+manag|"
         r"member\s+engagement\s+for\s+(?:a\s+)?(?:health\s+plan|payer|insurance)|"
         r"elder\s+care|senior\s+care|home\s+health|hospice|"
         r"fertility|\bivf\b|pregnancy|prenatal|postpartum|maternity|"
         r"pediatric|oncology|cardiology|dental|dentistry|"
         r"veterinary|vet\s+clinic|animal\s+hospital|pet\s+health|"
         r"clinical\s+research|biomedical|biotech|"
         r"cell\s+therapy|gene\s+therapy|"
         r"\bvaccine\b|antibody|biologics|"
         r"medical\s+imaging|"
         r"patient\s+monitoring|patient\s+engagement|"
         r"care\s+coordination|care\s+management|"
         r"medical\s+scheduler|"
         r"appointment\s+scheduling\s+for\s+(?:a\s+)?(?:clinic|doctor|hospital|dentist|vet)|"
         r"medical\s+supply|medical\s+equipment|"
         r"clinical\s+workflow|clinical\s+operations|"
         r"health\s+data|health\s+analytics|"
         r"population\s+health|social\s+determinants|"
         r"claims\s+processing|medical\s+claims|"
         r"patient\s+intake|patient\s+portal|"
         r"patient\s+experience|patient\s+satisfaction|"
         r"hospital\s+system|health\s+system|"
         r"clinical\s+intelligence|"
         r"genomic\s+sequencing|dna\s+test|"
         r"lab\s+test|laboratory|laboratory\s+information|"
         r"pathology\s+lab|radiology\s+workflow|"
         r"medication\s+management|medication\s+reconciliation|"
         r"\bdosing\b|prescription\s+app|"
         r"weight\s+loss|obesity\s+app|"
         r"diabetes\s+app|diabetes\s+management|hypertension\s+app|"
         r"cardiac\s+monitor|heart\s+monitor|"
         r"sleep\s+tracker|sleep\s+monitor|"
         r"fertility\s+tracker|ovulation\s+tracker|"
         r"menstrual\s+tracker|period\s+tracker|"
         r"baby\s+tracker|pregnancy\s+tracker|"
         r"nursing\s+app|lactation\s+consulting|"
         r"elderly\s+monitor|fall\s+detection|"
         r"workplace\s+wellness|corporate\s+wellness|"
         r"employee\s+assistance|\beap\b|"
         r"gym\s+management|"
         r"studio\s+management\s+for\s+(?:a\s+)?(?:yoga|pilates|fitness|personal\s+training)|"
         r"personal\s+training\s+app|\bpt\s+app\b|"
         r"physical\s+therapy\s+app|physiotherapy|"
         r"nutrition\s+app|meal\s+plan\s+app|calorie\s+counter|"
         r"running\s+app|cycling\s+app|training\s+plan|"
         r"yoga\s+app|meditation\s+platform|"
         r"\bdietitian\b|\bnutritionist\b|"
         r"first\s+responder|\bemt\b|\bparamedic\b|"
         r"\bsurgeon\b|surgical\s+platform|surgical\s+robotics|"
         r"nursing\s+home|assisted\s+living|"
         r"hospice\s+care|palliative\s+care|"
         r"substance\s+abuse|rehab\s+app|recovery\s+app|"
         r"\bautism\b|\badhd\b|behavioral\s+health|behavioral\s+health\s+platform|"
         r"speech\s+therapy|occupational\s+therapy|"
         r"rehabilitation\s+app|rehab\s+platform|"
         r"home\s+care\s+platform|home\s+care\s+app|"
         r"caregiver\s+app|caregiver\s+platform|"
         r"patient\s+transport|medical\s+transport|ambulance|"
         r"lab\s+result|lab\s+order|lab\s+integration|"
         r"doctor\s+app|provider\s+app|clinician\s+app|"
         r"nurse\s+app|nurse\s+scheduling|"
         r"medical\s+research|biomedical\s+research|"
         r"clinical\s+decision\s+support|"
         r"fitness\s+tracker|step\s+counter\s+app|"
         r"workout\s+plan|training\s+program\s+app|"
         r"running\s+coach|cycling\s+coach|swimming\s+app|"
         r"outdoor\s+app|hiking\s+app|climbing\s+app|"
         r"health\s+coach|wellness\s+coach|"
         r"medication\s+reminder|pill\s+reminder|"
         r"blood\s+pressure\s+app|bp\s+tracker|"
         r"heart\s+rate\s+app|hr\s+tracker|"
         r"health\s+screening|health\s+check\s+app|"
         r"symptom\s+checker|medical\s+symptom\s+checker|"
         r"first\s+aid\s+app|emergency\s+app|"
         r"doctor\s+on[\s-]?demand|teladoc\s+alternative|"
         r"therapy\s+marketplace|counseling\s+marketplace|"
         r"nutrition\s+tracker|food\s+diary|"
         r"sleep\s+coach|insomnia\s+app|"
         r"snoring\s+app|\bcpap\s+app\b|"
         r"hearing\s+aid|hearing\s+loss\s+app|"
         r"vision\s+app|eye\s+health|"
         r"dental\s+app|orthodontics|teeth\s+whitening|"
         r"contact\s+lens\s+app|glasses\s+app|"
         r"pet\s+care\s+app|pet\s+health\s+app|dog\s+walking\s+app)\b",
         BusinessCategory.HEALTHCARE),

        # ---------------------------------------------------------------
        # EDUCATION — schools, courses, tutoring, kids, languages
        # ---------------------------------------------------------------
        (r"\b(edtech|ed\s+tech|education\s+platform|learning\s+platform|"
         r"\bmooc\b|online\s+course|course\s+marketplace|"
         r"\blms\b|learning\s+management|student\s+information|sis\s+for|"
         r"k[\s-]?12|kindergarten|elementary\s+school|"
         r"high\s+school|college\s+app|college\s+counseling|"
         r"test\s+prep|sat\s+prep|act\s+prep|gre\s+prep|gmat\s+prep|"
         r"\btutor\b|tutoring|tutoring\s+platform|homework\s+help|study\s+app|study\s+tool|"
         r"flashcard|spaced\s+repetition|anki\s+alternative|"
         r"language\s+learning|duolingo\s+alternative|duolingo\s+competitor|"
         r"vocab\s+app|grammar\s+app|speak\s+a\s+second\s+language|"
         r"learn\s+(?:a\s+)?(?:new\s+)?\w+|teach(?:es)?\s+you|\blearner\b|"
         r"professional\s+development|corporate\s+training|employee\s+training|"
         r"upskilling|reskilling|certification\s+prep|coding\s+bootcamp|"
         r"trade\s+school|vocational|continuing\s+education|"
         r"educational\s+game|edutainment|kids\s+learning|"
         r"kids\s+education|children.{0,3}\s+education|"
         r"school\s+management|teacher\s+tool|classroom\s+tool|"
         r"student\s+app|student\s+engagement|student\s+success|"
         r"academic\s+planner|homework\s+tracker|"
         r"learning\s+app|learning\s+tool|learning\s+experience|"
         r"education\s+app|education\s+tool|education\s+content|"
         r"teaching\s+platform|teaching\s+tool|"
         r"university\s+app|school\s+app|college\s+platform|"
         r"course\s+app|course\s+builder|course\s+authoring|"
         r"training\s+platform|training\s+app|training\s+program|"
         r"onboarding\s+training|"
         r"skills\s+training|skill\s+development|skill\s+building|"
         r"career\s+training|career\s+coaching|career\s+development|"
         r"study\s+group|study\s+buddy|study\s+companion|"
         r"ai\s+tutor|ai\s+teacher|ai\s+study|ai\s+homework|"
         r"homework\s+helper|homework\s+solver|"
         r"exam\s+prep|exam\s+preparation|exam\s+cram|"
         r"\bielts\b|\btoefl\b|\btesol\b|"
         r"math\s+tutor|math\s+practice|math\s+app|"
         r"reading\s+app|reading\s+comprehension|"
         r"writing\s+app|writing\s+practice|essay\s+helper|"
         r"\bphonics\b|early\s+childhood|preschool|"
         r"stem\s+education|steam\s+education|"
         r"history\s+app|science\s+app|geography\s+app|"
         r"learn\s+to\s+code|learn\s+programming|learn\s+python|"
         r"learn\s+spanish|learn\s+french|learn\s+german|"
         r"learn\s+chinese|learn\s+japanese|"
         r"language\s+exchange|language\s+partner|"
         r"conversation\s+practice\s+app|pronunciation\s+app|"
         r"online\s+school|virtual\s+school|online\s+academy|"
         r"\bacademy\b|academy\s+for|"
         r"cohort[\s-]?based\s+course|self[\s-]?paced\s+course|"
         r"microcredential|nanodegree|"
         r"instructional\s+design|"
         r"lecture\s+capture|lecture\s+transcription|"
         r"ai\s+for\s+education|ai\s+in\s+education|"
         r"\bstudents?\b|"
         r"teach\s+kids|teach\s+children|"
         r"classroom\s+management|\bgradebook\b|"
         r"special\s+education|special\s+ed|"
         r"gifted\s+and\s+talented|"
         r"early\s+intervention|"
         r"literacy\s+app|"
         r"stem\s+kit|robotics\s+kit\s+for\s+kids|"
         r"coding\s+for\s+kids|coding\s+for\s+children)\b",
         BusinessCategory.EDUCATION),

        # ---------------------------------------------------------------
        # MARKETPLACE — two-sided, listings, gig, rentals, "Uber for X"
        # ---------------------------------------------------------------
        (r"\b(marketplace|two[\s-]?sided\s+market|peer[\s-]?to[\s-]?peer|"
         r"p2p\s+market|gig\s+economy|gig\s+platform|gig\s+worker|"
         r"freelancer\s+marketplace|freelance\s+marketplace|"
         r"listing\s+site|listing\s+platform|classifieds|"
         r"rental\s+marketplace|rental\s+platform|short[\s-]?term\s+rental|"
         r"vacation\s+rental|airbnb\s+competitor|airbnb\s+alternative|"
         r"etsy\s+competitor|etsy\s+alternative|"
         r"uber\s+competitor|uber\s+alternative|uber\s+for|"
         r"doordash\s+competitor|doordash\s+alternative|"
         r"instore\s+delivery|delivery\s+service|"
         r"food\s+delivery|grocery\s+delivery|courier\s+service|"
         r"home\s+services\s+marketplace|handyman\s+app|"
         r"taskrabbit\s+competitor|taskrabbit\s+alternative|"
         r"upwork\s+competitor|fiverr\s+competitor|"
         r"creator\s+marketplace|creator\s+economy\s+platform|"
         r"ticket\s+marketplace|resale\s+marketplace|resale\s+platform|"
         r"used\s+car\s+marketplace|car\s+marketplace|"
         r"b2b\s+marketplace|business\s+marketplace|wholesale\s+marketplace|"
         r"auction\s+platform|reverse\s+auction|"
         r"connects?\s+\w+\s+with\s+|"
         r"hire\s+a\s+\w+\s+(?:for|to)|"
         r"on[\s-]?demand\s+\w+\s+service|"
         r"on[\s-]?demand\s+delivery|on[\s-]?demand\s+rental|"
         r"rent\s+\w+\s+by\s+the\s+\w+|"
         r"buy\s+and\s+sell\s+\w+|"
         r"second[\s-]?hand|secondhand|"
         r"resale\s+app|reseller\s+app|"
         r"storefront\s+for|"
         r"commerce\s+platform|"
         r"vendor\s+marketplace|"
         r"service\s+marketplace|service\s+provider\s+marketplace|"
         r"professional\s+services\s+marketplace|"
         r"property\s+marketplace|real\s+estate\s+marketplace|"
         r"car\s+sharing|car\s+share|peer[\s-]?to[\s-]?peer\s+car|"
         r"bike\s+share|scooter\s+share|"
         r"machinery\s+rental|equipment\s+rental|"
         r"tool\s+rental|tool\s+library|tool\s+lending|"
         r"space\s+rental|desk\s+rental|office\s+space\s+rental|"
         r"coworking\s+platform|"
         r"parking\s+marketplace|storage\s+marketplace|"
         r"boat\s+rental|\brv\s+rental\b|"
         r"outdoor\s+gear\s+rental|"
         r"clothing\s+rental|fashion\s+rental|"
         r"toy\s+rental|book\s+rental|"
         r"furniture\s+rental|electronics\s+rental|"
         r"talent\s+marketplace|"
         r"influencer\s+marketplace|creator[\s-]?collaboration\s+platform|"
         r"service\s+booking\s+platform|booking\s+platform\s+for|"
         r"appointment\s+booking\s+platform|scheduling\s+platform\s+for\s+services|"
         r"directory\s+of\s+\w+|"
         r"crowd[\s-]?sourced\s+map|"
         r"hyperlocal\s+\w+|neighborhood\s+marketplace|"
         r"university\s+marketplace|campus\s+marketplace|"
         r"enterprise\s+marketplace|b2b\s+sourcing\s+platform|"
         r"\brecommerce\b|resale[\s-]?as[\s-]?a[\s-]?service|"
         r"recommerce\s+platform|"
         r"circular\s+economy|sustainability\s+marketplace|"
         r"local\s+marketplace|hyperlocal\s+marketplace|"
         r"online\s+auction|"
         r"warehouse\s+marketplace|fulfillment\s+marketplace|"
         r"logistics\s+marketplace|delivery\s+marketplace|"
         r"freight\s+marketplace|trucking\s+marketplace|"
         r"supply\s+marketplace|"
         r"managed\s+marketplace|vertical\s+marketplace|"
         r"marketplace\s+for\s+\w+|"
         r"platform\s+(?:that|for)\s+connects?\s+\w+\s+to\s+|"
         r"connects?\s+(?:homeowners|neighbors|residents|travelers|locals|customers|users|tenants|landlords|sellers|buyers|patients|doctors|workers|providers|clients)|"
         r"online\s+(?:marketplace|directory|store|auction)|"
         r"rent\s+(?:a|an)\s+\w+\s+by|"
         r"subscription\s+service\s+that\s+rents|"
         r"subscription\s+lending|"
         r"subscription\s+kit|"
         r"subscription\s+(?:box|service)\s+for\s+consumers)\b",
         BusinessCategory.MARKETPLACE),

        # ---------------------------------------------------------------
        # CONSUMER — apps, social, dating, content, gaming
        # ---------------------------------------------------------------
        (r"\b(social\s+app|social\s+network|social\s+platform|"
         r"dating\s+app|dating\s+platform|tinder\s+competitor|tinder\s+alternative|"
         r"bumble\s+competitor|hinge\s+competitor|"
         r"messaging\s+app|chat\s+app|group\s+chat|encrypted\s+messag|"
         r"video\s+chat|video\s+calling|conferencing\s+for\s+consumers|"
         r"photo\s+sharing|photo\s+app|instagram\s+competitor|instagram\s+alternative|"
         r"tiktok\s+competitor|tiktok\s+alternative|youtube\s+competitor|"
         r"youtube\s+alternative|short[\s-]?form\s+video|reels\s+app|"
         r"podcast\s+app|podcast\s+platform|audio\s+social|"
         r"creator\s+tools\s+for\s+consumers|content\s+creation\s+for\s+consumers|"
         r"short[\s-]?form\s+content|long[\s-]?form\s+content|"
         r"news\s+app|news\s+aggregator|news\s+feed|"
         r"content\s+recommendation|recommendation\s+for\s+consumers|"
         r"streaming\s+service|streaming\s+platform|video\s+streaming|"
         r"music\s+streaming|audiobook\s+app|audiobook\s+platform|"
         r"\bgaming\b|gaming\s+platform|mobile\s+game|esports|"
         r"consumer\s+app|\bb2c\s+app\b|direct[\s-]?to[\s-]?consumer|\bd2c\b|"
         r"subscription\s+box|subscription\s+for\s+consumers|"
         r"personal\s+crm|friendship\s+app|community\s+app|fan\s+app|"
         r"fitness\s+tracker|step\s+counter|calorie\s+tracker|"
         r"meal\s+planning\s+for\s+consumers|recipe\s+app|cooking\s+app|"
         r"travel\s+app|travel\s+planning|trip\s+planner|itinerary\s+app|"
         r"wallet\s+app|mobile\s+wallet|loyalty\s+program\s+for\s+consumers|"
         r"personal\s+assistant\s+for\s+consumers|ai\s+assistant\s+for\s+consumers|"
         r"smart\s+home|home\s+automation\s+for\s+consumers|"
         r"smart\s+speaker|voice\s+assistant\s+for\s+consumers|"
         r"\bwearable\b|fitness\s+band|smartwatch\s+app|"
         r"family\s+app|parenting\s+app|baby\s+tracker|pet\s+app|"
         r"social\s+network\s+for|"
         r"hobbyist\s+app|enthusiast\s+app|"
         r"citizen[\s-]?science\s+app|"
         r"app\s+for\s+\w+\s+to\s+(?:share|connect|meet|find|track|log|record)|"
         r"app\s+that\s+lets\s+(?:you|users|families|teams|people)|"
         r"app\s+for\s+\w+\s+(?:lovers|enthusiasts|fans|community|owners)|"
         r"app\s+that\s+routes|app\s+that\s+tracks|app\s+that\s+monitors|"
         r"app\s+that\s+helps\s+\w+\s+(?:identify|find|track|monitor|learn|discover)|"
         r"app\s+for\s+(?:a\s+)?(?:citizen|local|neighborhood|hobbyist|amateur)\w*|"
         r"app\s+that\s+analyzes|app\s+that\s+identifies|app\s+that\s+detects|"
         r"app\s+for\s+tourists|app\s+for\s+travelers|"
         r"app\s+for\s+pet\s+owners|app\s+for\s+dog\s+owners|app\s+for\s+cat\s+owners|"
         r"app\s+for\s+parents|app\s+for\s+families|app\s+for\s+couples|"
         r"app\s+for\s+photographers|app\s+for\s+musicians|app\s+for\s+artists|"
         r"app\s+for\s+writers|app\s+for\s+readers|app\s+for\s+book\s+lovers|"
         r"app\s+for\s+foodies|app\s+for\s+coffee\s+lovers|"
         r"app\s+for\s+wine\s+lovers|app\s+for\s+beer\s+enthusiasts|"
         r"app\s+for\s+plant\s+parents|app\s+for\s+gardeners|"
         r"app\s+for\s+hikers|app\s+for\s+climbers|app\s+for\s+skiers|"
         r"app\s+for\s+surfers|app\s+for\s+fishermen|"
         r"app\s+for\s+campers|app\s+for\s+backpackers|"
         r"app\s+for\s+birdwatchers|app\s+for\s+stargazers|"
         r"app\s+for\s+astronomers|app\s+for\s+amateur\s+\w+|"
         r"app\s+that\s+helps\s+\w+\s+with\s+\w+|"
         r"consumer\s+\w+\s+app|consumer\s+\w+\s+platform|"
         r"subscription\s+app\s+for\s+consumers|"
         r"free[\s-]?to[\s-]?play|premium\s+subscription|"
         r"in[\s-]?app\s+purchase|microtransaction|"
         r"loyalty\s+program|rewards\s+program|"
         r"cashback\s+app|rewards\s+app|"
         r"coupon\s+app|deals\s+app|"
         r"shopping\s+app|shopping\s+assistant|"
         r"fashion\s+app|outfit\s+planner|wardrobe\s+app|"
         r"beauty\s+app|makeup\s+app|skincare\s+app|"
         r"hair\s+app|nail\s+app|"
         r"sports?\s+app|game\s+app|"
         r"dating\s+app\s+for|matchmaking\s+app|"
         r"pen[\s-]?pal\s+app|"
         r"social\s+app\s+for|social\s+platform\s+for|"
         r"fan\s+app|fan\s+platform|fan\s+engagement\s+app|"
         r"club\s+app|membership\s+app\s+for\s+consumers|"
         r"group\s+chat\s+app|voice\s+chat\s+app|"
         r"ai\s+companion|ai\s+friend|ai\s+girlfriend|ai\s+boyfriend|"
         r"chatbot\s+for\s+consumers|ai\s+chatbot\s+for\s+consumers|"
         r"smart\s+assistant\s+for\s+consumers|"
         r"budget\s+app|spending\s+app|expense\s+tracker\s+app|"
         r"price\s+tracker|price\s+comparison\s+app|"
         r"smart\s+doorbell|smart\s+lock\s+app|smart\s+thermostat|"
         r"smart\s+light|smart\s+plug|smart\s+bulb|"
         r"home\s+security\s+app|home\s+security\s+system|"
         r"smart\s+pet|smart\s+feeder|pet\s+tracker|"
         r"kids\s+app|children's?\s+app|kid[\s-]?friendly\s+app|"
         r"baby\s+app|toddler\s+app|preschool\s+app|"
         r"video\s+app\s+for\s+consumers|"
         r"music\s+app\s+for\s+consumers|"
         r"art\s+app\s+for\s+consumers|"
         r"creative\s+app\s+for\s+consumers|"
         r"hobby\s+app|hobby\s+platform|"
         r"craft\s+app|knitting\s+app|crochet\s+app|"
         r"scrapbooking\s+app|journal\s+app|"
         r"planner\s+app|calendar\s+app\s+for\s+consumers|"
         r"to[\s-]?do\s+app\s+for\s+consumers|"
         r"reminder\s+app|alarm\s+app|"
         r"sleep\s+app|alarm\s+clock\s+app|"
         r"white\s+noise|ambient\s+sound|"
         r"mood\s+tracker|emotion\s+tracker|"
         r"gratitude\s+app|habit\s+tracker|"
         r"meditation\s+app\s+for\s+consumers|"
         r"dating\s+coach|relationship\s+coach|"
         r"language\s+exchange\s+app|"
         r"remote\s+work\s+app\s+for\s+consumers|"
         r"vpn\s+app\s+for\s+consumers|"
         r"password\s+manager\s+for\s+consumers|"
         r"file\s+storage\s+for\s+consumers|cloud\s+storage\s+for\s+consumers|"
         r"photo\s+backup|photo\s+storage|"
         r"media\s+player|video\s+player|audio\s+player|"
         r"streaming\s+app|streaming\s+device|"
         r"smart\s+tv\s+app|smart\s+tv\s+platform|"
         r"voice\s+memo|recording\s+app|"
         r"translation\s+app|translator\s+app|"
         r"kids\s+game|children's?\s+game|kid[\s-]?friendly\s+game|"
         r"online\s+game|multiplayer\s+game|"
         r"battle\s+royale|\bmoba\b|\bmmorpg\b|"
         r"word\s+game|trivia\s+game|puzzle\s+game|"
         r"online\s+multiplayer|"
         r"arcade\s+game|retro\s+game|"
         r"indie\s+game|game\s+studio|"
         r"game\s+marketplace|game\s+economy|"
         r"in[\s-]?game\s+purchase|skin\s+marketplace|"
         r"fan[\s-]?fiction|webtoon|webcomic|"
         r"creator\s+app|content\s+creator\s+platform|"
         r"newsletter\s+platform|newsletter\s+for\s+consumers|"
         r"\bsubreddit\b|community\s+forum\s+for\s+consumers|"
         r"social\s+news|aggregator\s+app|"
         r"weather\s+app|weather\s+radar\s+app|"
         r"compass\s+app|map\s+app\s+for\s+consumers|"
         r"navigation\s+app\s+for\s+consumers|"
         r"translation\s+earbuds|smart\s+glasses|"
         r"fitness\s+wearable|workout\s+wearable|"
         r"smart\s+ring|smart\s+watch\s+app|"
         r"audio\s+device|earbud\s+app|hearing\s+aid\s+app|"
         r"homework\s+app\s+for\s+children|"
         r"pet\s+dating\s+app|pet\s+playdate|"
         r"playdate\s+app|"
         r"app\s+for\s+\w+\s+to\s+meet|"
         r"app\s+for\s+neighbors|app\s+for\s+residents|"
         r"neighborhood\s+app|neighbourhood\s+app|"
         r"homeowner\s+app|condo\s+app|"
         r"app\s+for\s+renters|app\s+for\s+tenants|"
         r"landlord\s+app|"
         r"fan\s+engagement|fan\s+messaging|"
         r"creator\s+messaging|creator\s+fan|"
         r"subscriber\s+app|subscriber\s+engagement|"
         r"app\s+that\s+predicts|app\s+that\s+detects|"
         r"app\s+for\s+citizens|"
         r"app\s+that\s+scores|app\s+that\s+monitors\s+\w+|"
         r"(?:^|\.)\s*(?:subscription|service|app|tool|platform)\s+that\s+(?:rents|lets\s+you|allows\s+you|helps\s+you|enables\s+you|tracks|monitors|analyzes|identifies|detects|manages|connects|matches|ships|delivers|ships|provides|ranks|finds|captures|shares|logs|records|counts|tracks)\b)\b",
         BusinessCategory.CONSUMER),

        # ---------------------------------------------------------------
        # B2B SaaS — generic "software for businesses" catch-all
        # (after every more-specific bucket)
        # ---------------------------------------------------------------
        (r"\b(\bb2b\b|\bsaas\b|enterprise\s+software|workflow\s+software|"
         r"\bcrm\b|\berp\b|\bhrm\b|hr\s+platform|hr\s+software|hr\s+tool|"
         r"marketing\s+automation|marketing\s+platform|email\s+marketing|"
         r"\babm\s+platform\b|lead\s+generation|lead\s+scoring|"
         r"customer\s+success|customer\s+support\s+platform|"
         r"help\s+desk|ticketing\s+system|knowledge\s+base|"
         r"internal\s+tool|operations\s+platform|ops\s+tool|"
         r"team\s+collaboration|team\s+chat|project\s+management|"
         r"task\s+management|productivity\s+tool|productivity\s+app|note[\s-]?taking\s+app|"
         r"document\s+management|contract\s+management|contract\s+review|"
         r"contract\s+automation|proposal\s+software|sales\s+enablement|"
         r"sales\s+engagement|outreach\s+platform|salesforce\s+competitor|"
         r"salesforce\s+alternative|hubspot\s+competitor|hubspot\s+alternative|"
         r"recruiting\s+platform|recruiting\s+software|ats\s+system|"
         r"applicant\s+tracking|talent\s+platform|onboarding\s+software|"
         r"facilities\s+management|vendor\s+management|\bprocurement\b|"
         r"supply\s+chain|inventory\s+management\s+for|warehouse\s+management|"
         r"logistics\s+platform|logistics\s+software|"
         r"field\s+service\s+management|field\s+service\s+software|"
         r"csp\s+for|customer\s+data\s+platform|cdp\s+platform|"
         r"data\s+warehouse\s+for|analytics\s+platform|analytics\s+for\s+businesses|"
         r"business\s+intelligence|\bbi\s+platform\b|reporting\s+tool|"
         r"compliance\s+platform|compliance\s+software|\bgrc\b|"
         r"security\s+platform|security\s+tool|siem|threat\s+detection|"
         r"identity\s+and\s+access|\biam\b|zero\s+trust|"
         r"password\s+manager|password\s+management|secret\s+management|"
         r"legal\s+tech|legal\s+software|\blegaltech\b|"
         r"contract\s+analysis|contract\s+intelligence|"
         r"board\s+management|board\s+portal|equity\s+management|"
         r"cap\s+table|cap\s+table\s+management|"
         r"for\s+businesses|for\s+companies|for\s+teams|"
         r"for\s+smbs|for\s+small\s+business|for\s+enterprise|for\s+mid[\s-]?market|"
         r"\bspreadsheet\b|excel\s+plugin|excel\s+add[\s-]?in|"
         r"workplace\s+management|workspace\s+management\s+for\s+(?:a\s+)?(?:business|office|company)|"
         r"office\s+management\s+for|"
         r"intranet|knowledge\s+management\s+for\s+(?:a\s+)?(?:company|business|team)|"
         r"wiki\s+for\s+(?:a\s+)?(?:company|teams|enterprise)|"
         r"internal\s+wiki|internal\s+knowledge|"
         r"client\s+portal\s+for|customer\s+portal\s+for|"
         r"vendor\s+portal|partner\s+portal|"
         r"procurement\s+platform|sourcing\s+platform|"
         r"rfp\s+platform|rfq\s+platform|"
         r"spend\s+management|spend\s+analytics|"
         r"corporate\s+travel|"
         r"corporate\s+card\s+for|"
         r"business\s+banking|"
         r"b2b\s+sourcing|b2b\s+procurement|"
         r"headcount\s+planning|workforce\s+planning|"
         r"people\s+analytics|hr\s+analytics|"
         r"compensation\s+platform|comp\s+management|"
         r"performance\s+management|performance\s+review|"
         r"goal\s+tracking\s+for\s+(?:a\s+)?(?:teams|business|company)|"
         r"okr\s+tool|okr\s+platform|"
         r"engagement\s+survey\s+for\s+(?:a\s+)?(?:employees|team|business)|"
         r"feedback\s+tool\s+for\s+(?:a\s+)?(?:managers|teams|business)|"
         r"360\s+feedback|peer\s+review\s+tool|"
         r"employee\s+engagement\s+for\s+(?:a\s+)?(?:teams|companies|business)|"
         r"employee\s+experience\s+platform|"
         r"legal\s+ops\s+for|legal\s+operations\s+for|"
         r"document\s+automation\s+for\s+(?:a\s+)?(?:business|legal|companies|enterprise|teams)|"
         r"document\s+workflow\s+for\s+(?:a\s+)?(?:business|legal|companies|enterprise|teams)|"
         r"e[\s-]?signature|electronic\s+signature|"
         r"signature\s+platform|signature\s+tool|"
         r"form\s+builder|form\s+automation|"
         r"case\s+management\s+for|case\s+management\s+system|"
         r"complaint\s+management|grievance\s+management|"
         r"knowledge\s+management|knowledge\s+graph|"
         r"semantic\s+search\s+for\s+(?:a\s+)?(?:business|enterprise|companies|teams|legal|hr|finance|support)|"
         r"enterprise\s+search\s+for|"
         r"unified\s+search|"
         r"data\s+platform\s+for\s+(?:a\s+)?(?:business|enterprise|companies|teams|developers|manufacturers|retailers|operators)|"
         r"data\s+cleaning|data\s+quality\s+for\s+(?:a\s+)?(?:business|enterprise|companies|teams|developers)|"
         r"master\s+data|reference\s+data|"
         r"data\s+security|data\s+privacy\s+for\s+(?:a\s+)?(?:business|enterprise|companies|teams|developers)|"
         r"privacy\s+compliance|gdpr\s+compliance|ccpa\s+compliance|"
         r"data\s+protection|data\s+loss\s+prevention|\bdlp\b|"
         r"audit\s+trail\s+for\s+(?:a\s+)?(?:business|enterprise|companies|teams|legal|hr|finance|healthcare|pharma|banking|manufacturers)|"
         r"audit\s+log\s+for\s+(?:a\s+)?(?:business|enterprise|companies|teams|legal|hr|finance|healthcare|pharma|banking|manufacturers)|"
         r"iot\s+platform|iot\s+management|"
         r"iot\s+analytics|connected\s+device\s+platform|"
         r"smart\s+building\s+platform|"
         r"smart\s+factory|industry\s+4\.0|"
         r"predictive\s+maintenance|"
         r"energy\s+management|energy\s+monitoring|"
         r"sustainability\s+for\s+(?:a\s+)?(?:business|enterprise|companies|manufacturers|hotels|data\s+centers)|"
         r"esg\s+platform|esg\s+reporting|"
         r"carbon\s+accounting|carbon\s+tracking|carbon\s+credits|carbon\s+offset|"
         r"hotel\s+operations|restaurant\s+operations\s+for\s+(?:a\s+)?(?:chain|restaurant|operator|owner)|"
         r"property\s+management\s+for\s+(?:a\s+)?(?:landlord|property\s+manager|building\s+owner|real\s+estate)|"
         r"agile\s+tool|agile\s+platform|"
         r"sprint\s+planning|product\s+roadmap|"
         r"service\s+management|\bitsm\b|"
         r"field\s+service\s+for|"
         r"work\s+order\s+for|work\s+order\s+management|"
         r"maintenance\s+management\s+for|"
         r"fleet\s+management|fleet\s+maintenance|"
         r"asset\s+tracking\s+for\s+(?:a\s+)?(?:business|enterprise|companies)|"
         r"decision\s+automation|decision\s+intelligence|"
         r"rules\s+engine|business\s+rules|"
         r"manufacturing\s+operations|factory\s+operations|"
         r"retail\s+operations\s+for|"
         r"contract\s+workflow\s+for\s+(?:a\s+)?(?:business|companies|teams|enterprise|developers|engineers|data\s+teams|banking|finance|marketplace|legal|hr|sales|operations)|"
         r"contract\s+lifecycle|clm\s+for\s+(?:a\s+)?(?:business|companies|teams|enterprise|developers|engineers|data\s+teams|banking|finance|marketplace|legal|hr|sales|operations)|"
         r"contract\s+drafting\s+for\s+(?:a\s+)?(?:business|companies|teams|enterprise|developers|engineers|data\s+teams|banking|finance|marketplace|legal|hr|sales|operations)|"
         r"contract\s+review\s+for\s+(?:a\s+)?(?:business|companies|teams|enterprise|developers|engineers|data\s+teams|banking|finance|marketplace|legal|hr|sales|operations)|"
         r"contract\s+management\s+for\s+(?:a\s+)?(?:business|companies|teams|enterprise|developers|engineers|data\s+teams|banking|finance|marketplace|legal|hr|sales|operations)|"
         r"negotiation\s+for\s+(?:a\s+)?(?:business|companies|teams|enterprise|developers|engineers|data\s+teams|banking|finance|marketplace|legal|hr|sales|operations|procurement)|"
         r"redlining\s+for\s+(?:a\s+)?(?:business|companies|teams|enterprise|developers|engineers|data\s+teams|banking|finance|marketplace|legal|hr|sales|operations|procurement)|"
         r"redline\s+for\s+(?:a\s+)?(?:business|companies|teams|enterprise|developers|engineers|data\s+teams|banking|finance|marketplace|legal|hr|sales|operations|procurement)|"
         r"document\s+ai\s+for\s+(?:a\s+)?(?:business|companies|teams|enterprise|developers|engineers|data\s+teams|banking|finance|marketplace|legal|hr|sales|operations|procurement|insurance|healthcare|manufacturers|retailers|logistics|hotels|restaurants|agencies|architects|engineers|developers|operators|property\s+managers|landlords|hoteliers|restaurateurs)|"
         r"document\s+extraction\s+for\s+(?:a\s+)?(?:business|companies|teams|enterprise|developers|engineers|data\s+teams|banking|finance|marketplace|legal|hr|sales|operations|procurement|insurance|healthcare|manufacturers|retailers|logistics|hotels|restaurants|agencies|architects|engineers|developers|operators|property\s+managers|landlords|hoteliers|restaurateurs)|"
         r"document\s+parsing\s+for\s+(?:a\s+)?(?:business|companies|teams|enterprise|developers|engineers|data\s+teams|banking|finance|marketplace|legal|hr|sales|operations|procurement|insurance|healthcare|manufacturers|retailers|logistics|hotels|restaurants|agencies|architects|engineers|developers|operators|property\s+managers|landlords|hoteliers|restaurateurs)|"
         r"ocr\s+for\s+(?:a\s+)?(?:business|companies|teams|enterprise|developers|engineers|data\s+teams|banking|finance|marketplace|legal|hr|sales|operations|procurement|insurance|healthcare|manufacturers|retailers|logistics|hotels|restaurants|agencies|architects|engineers|developers|operators|property\s+managers|landlords|hoteliers|restaurateurs)|"
         r"invoice\s+processing\s+for\s+(?:a\s+)?(?:business|companies|teams|enterprise|developers|engineers|data\s+teams|banking|finance|marketplace|legal|hr|sales|operations|procurement|insurance|healthcare|manufacturers|retailers|logistics|hotels|restaurants|agencies|architects|engineers|developers|operators|property\s+managers|landlords|hoteliers|restaurateurs)|"
         r"invoice\s+automation\s+for\s+(?:a\s+)?(?:business|companies|teams|enterprise|developers|engineers|data\s+teams|banking|finance|marketplace|legal|hr|sales|operations|procurement|insurance|healthcare|manufacturers|retailers|logistics|hotels|restaurants|agencies|architects|engineers|developers|operators|property\s+managers|landlords|hoteliers|restaurateurs)|"
         r"ap\s+automation\s+for\s+(?:a\s+)?(?:business|companies|teams|enterprise|developers|engineers|data\s+teams|banking|finance|marketplace|legal|hr|sales|operations|procurement|insurance|healthcare|manufacturers|retailers|logistics|hotels|restaurants|agencies|architects|engineers|developers|operators|property\s+managers|landlords|hoteliers|restaurateurs)|"
         r"ar\s+automation\s+for\s+(?:a\s+)?(?:business|companies|teams|enterprise|developers|engineers|data\s+teams|banking|finance|marketplace|legal|hr|sales|operations|procurement|insurance|healthcare|manufacturers|retailers|logistics|hotels|restaurants|agencies|architects|engineers|developers|operators|property\s+managers|landlords|hoteliers|restaurateurs)|"
         r"payment\s+orchestration\s+for\s+(?:a\s+)?(?:business|companies|teams|enterprise|developers|engineers|data\s+teams|banking|finance|marketplace|legal|hr|sales|operations|procurement|insurance|healthcare|manufacturers|retailers|logistics|hotels|restaurants|agencies|architects|engineers|developers|operators|property\s+managers|landlords|hoteliers|restaurateurs)|"
         r"checkout\s+optimization\s+for\s+(?:a\s+)?(?:business|companies|teams|enterprise|developers|engineers|data\s+teams|banking|finance|marketplace|legal|hr|sales|operations|procurement|insurance|healthcare|manufacturers|retailers|logistics|hotels|restaurants|agencies|architects|engineers|developers|operators|property\s+managers|landlords|hoteliers|restaurateurs)|"
         r"checkout\s+for\s+(?:a\s+)?(?:business|companies|teams|enterprise|developers|engineers|data\s+teams|banking|finance|marketplace|legal|hr|sales|operations|procurement|insurance|healthcare|manufacturers|retailers|logistics|hotels|restaurants|agencies|architects|engineers|developers|operators|property\s+managers|landlords|hoteliers|restaurateurs))\b",
         BusinessCategory.B2B_SAAS),

        # ---------------------------------------------------------------
        # B2B SaaS — second-pass catch-all for natural-language
        # descriptions that don't include a "for X" / "platform for"
        # qualifier. Catches "AI co-pilot for sales teams",
        # "patient billing and collections platform that helps
        # medical practices", etc. Order matters — these
        # patterns are broad so they come AFTER the more
        # specific rules above to avoid mis-classifying
        # marketplace / consumer / healthcare records.
        # ---------------------------------------------------------------
        (r"\bai\s+co[\s-]?pilot\b|co[\s-]?pilot\s+for\s+",
         BusinessCategory.B2B_SAAS),
        (r"\bai\s+agent\s+for\s+",
         BusinessCategory.B2B_SAAS),
        (r"\bai\s+for\s+(?:sales|marketing|legal|hr|finance|operations|support|customer\s+support|manufacturing|retail|logistics|supply\s+chain|insurance|healthcare|pharma|developers|engineers|data\s+teams|legal\s+teams|hr\s+teams|finance\s+teams|sales\s+teams|marketing\s+teams|support\s+teams|operations\s+teams|manufacturing\s+teams|retail\s+teams|logistics\s+teams|supply\s+chain\s+teams|companies|teams|enterprise|business|smb|startups?)\b",
         BusinessCategory.B2B_SAAS),
        (r"\bagentic\s+(?:sales|marketing|support|legal|hr|finance|operations|workflow|ai|automation)\b",
         BusinessCategory.B2B_SAAS),
        (r"\bpatient\s+billing\s+and\s+collections\b|"
         r"\bbilling\s+and\s+collections\s+platform\b|"
         r"\bmedical\s+billing\s+and\s+collections\b|"
         r"\bmedical\s+billing\b|"
         r"\bmedical\s+claims\b|"
         r"\bhealthcare\s+billing\b",
         BusinessCategory.HEALTHCARE),
        (r"\bpersonal\s+stylist\b|"
         r"\boutfit\s+planner\b|"
         r"\bwardrobe\s+app\b|"
         r"\bfashion\s+app\b|"
         r"\bpersonal\s+shopper\b|"
         r"\bstylist\s+app\b",
         BusinessCategory.CONSUMER),
    ]
    return [(re.compile(pat, re.IGNORECASE), cat) for pat, cat in raw]


_RULES: List[Tuple[re.Pattern[str], BusinessCategory]] = _compile_rules()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CategoryAssignment:
    """The deterministic result for one record.

    Attributes
    ----------
    business_category : BusinessCategory
        The assigned category. Always one of the 8 PHASE-3.md §3.4
        buckets — never ``None``. Records that match no rule land
        in ``BusinessCategory.OTHER``.
    matched_rule_index : Optional[int]
        Index into the internal rule table of the rule that
        matched. ``None`` for the ``OTHER`` fallback. Useful in
        tests for asserting "this specific rule fired" rather
        than "the right category came out".
    """

    business_category: BusinessCategory
    matched_rule_index: Optional[int]


def assign_business_category(
    idea: str,
    *args: object,
    **kwargs: object,
) -> CategoryAssignment:
    """Assign a business category to a single idea.

    Signature is permissive (``*args, **kwargs``) so future
    extensions (e.g. expected_top_ids lookup against the
    ``companies`` table, source-aware hints) can be added without
    breaking the existing call site in ``run.py``. The current
    implementation ignores everything except ``idea``.

    Determinism contract
    --------------------
    - Single pass: we iterate the rule table once and return on
      the first match.
    - No randomness: the regex set is fixed at import time.
    - No external state: the only input is the ``idea`` text.
    - Re-runs of the same input yield the same output, byte-for-byte.

    The fallback (no rule fires) is ``BusinessCategory.OTHER`` — the
    card explicitly allows this; records that fall through are
    surfaced as "category = other" in the failure breakdown, which
    is a real signal that the rule set is missing coverage.
    """
    text = (idea or "").strip()
    if not text:
        return CategoryAssignment(
            business_category=BusinessCategory.OTHER,
            matched_rule_index=None,
        )
    for i, (pattern, cat) in enumerate(_RULES):
        if pattern.search(text):
            return CategoryAssignment(
                business_category=cat,
                matched_rule_index=i,
            )
    return CategoryAssignment(
        business_category=BusinessCategory.OTHER,
        matched_rule_index=None,
    )


def assign_business_categories(
    ideas: Iterable[Tuple[str, str]],
) -> Dict[str, BusinessCategory]:
    """Assign categories for a batch of ``(record_id, idea)`` pairs.

    Convenience wrapper used by the eval-set extension step
    (``scripts/build_business_categories.py`` or in-process during
    ``run.py``). Returns a dict keyed by ``record_id``.

    This function is *idempotent* — calling it twice with the same
    inputs yields the same dict. The deterministic contract holds
    across the whole batch, not just per record.
    """
    out: Dict[str, BusinessCategory] = {}
    for record_id, idea in ideas:
        if record_id in out:
            raise ValueError(f"duplicate record_id in batch: {record_id!r}")
        out[record_id] = assign_business_category(idea).business_category
    return out


# ---------------------------------------------------------------------------
# Coverage stats — used by the failure-breakdown tooling
# ---------------------------------------------------------------------------


def category_coverage(
    assignments: Sequence[BusinessCategory],
) -> Dict[BusinessCategory, int]:
    """Count records per category from a list of assignments.

    Used by the failure-breakdown writer to surface the
    ``n_records`` column in the per-category table. Returns counts
    in the canonical ``BUSINESS_CATEGORIES`` order (for stable
    column rendering).
    """
    counts: Dict[BusinessCategory, int] = {c: 0 for c in BUSINESS_CATEGORIES}
    for cat in assignments:
        counts[cat] = counts.get(cat, 0) + 1
    return counts


__all__ = [
    "BusinessCategory",
    "BUSINESS_CATEGORIES",
    "CATEGORY_LABEL",
    "DEFAULT_PROVENANCE",
    "CategoryAssignment",
    "assign_business_category",
    "assign_business_categories",
    "category_coverage",
]