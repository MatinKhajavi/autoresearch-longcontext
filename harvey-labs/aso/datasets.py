"""Curated task pools for the optimization loop.

We never run all 1,251 LAB tasks. Three disjoint pools:
  - SCREEN  : tiny set for fast pruning of proposed variants (decisions only).
  - DEV     : the optimization signal the researcher iterates against.
  - HOLDOUT : never used in ANY decision — the honest headline number + tier comparison.

All chosen from the small "analysis / compare" tasks (read-heavy, doc-grounded,
~23–37 criteria) so the loop is fast and coverage/validation behaviors matter.
OVERFLOW_STRESS holds oversized matters that reliably blow the context window —
used in Phase 5 to demonstrate the tool-result-clearing module.

Every id below was verified to have tasks/<id>/task.json.
"""

SCREEN = [
    "immigration/compare-uscis-filing-receipt-against-original-petition-submission",
    "tax/review-iss-tax-transaction-structure",
    "banking-finance/identify-issues-in-borrower-financial-statements",
]

DEV = [
    "employment-labor/identify-issues-in-counterparty-motion-brief",
    "litigation-dispute-resolution/identify-issues-in-counterparty-complaint",
    "bankruptcy-restructuring/identify-issues-in-counterparty-sale-objection",
    "tax/identify-tax-issues-in-counterpartys-opposition-brief",
    "trusts-estates-private-client/identify-issues-in-counterpartys-proposed-parenting-plan",
    "environmental-esg/identify-issues-in-draft-permit-application",
    "intellectual-property/identify-issues-in-counterpartys-proposed-jury-instructions",
    "insurance/identify-issues-in-coverage-denial-letter",
]

HOLDOUT = [
    "trusts-estates-private-client/compare-trust-documents-against-client-instructions",
    "immigration/compare-draft-eb",
    "international-trade-sanctions/review-draft-voluntary-self",
    "capital-markets/compare-closing-documents-against-closing-checklist",
    "corporate-governance/identify-issues-in-dissident-proxy-statement",
]

# Filled in Phase 5: oversized matters (>1MB docs) that exercise compaction.
OVERFLOW_STRESS: list[str] = []

ALL_POOLS = {"screen": SCREEN, "dev": DEV, "holdout": HOLDOUT}
