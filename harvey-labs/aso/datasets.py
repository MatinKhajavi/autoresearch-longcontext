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

# Oversized matters that blow a 200K context window — used to demonstrate the
# Tier-2 tool-result-clearing module. Real cl100k token counts (measured by
# running the harness parsers + tiktoken), document payload only; none are in
# SCREEN/DEV/HOLDOUT. The corpus has no PDFs and nothing >5MB raw, so overflow
# comes from docx-text expansion + many-file matters, not giant binaries.
OVERFLOW_STRESS: list[str] = [
    "funds-asset-management/respond-to-comment-memo",                 # ~917K tok, 10 docx — 4.5x window
    "tax/draft-cross-border-acquisition-tax-memo",                    # ~467K tok, 22 docx
    "corporate-ma/draft-acquisition-due-diligence",                  # ~208K tok, 31 files (docx+xlsx)
    "corporate-governance/assess-impact-of-ftc-noncompete-ban-on-existing-employment-agreements",  # ~213K tok, mixed
]

# Heavy / document-dense matters where coverage + memory management actually move
# the needle (the small analysis pools above don't overflow or need compaction).
# Used via `optimize --task-set heavy` so the long-context win is measurable.
SCREEN_HEAVY = ["tax/draft-cross-border-acquisition-tax-memo"]              # ~467K tok
DEV_HEAVY = ["tax/draft-cross-border-acquisition-tax-memo"]                 # ~467K tok (clear memory-mgmt win)
HOLDOUT_HEAVY = ["tax/draft-transfer-pricing-documentation"]               # ~291K tok (distinct tax-drafting holdout, ~1/3 the 917K)

TASK_SETS = {
    "default": (SCREEN, DEV, HOLDOUT),
    "heavy": (SCREEN_HEAVY, DEV_HEAVY, HOLDOUT_HEAVY),
}

ALL_POOLS = {"screen": SCREEN, "dev": DEV, "holdout": HOLDOUT}
