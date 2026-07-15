"""Shared reviewable instructions for coding agents connected to STEERING."""

AGENT_WORKFLOW_INSTRUCTIONS = """Consult STEERING:
- at the start of architecture planning;
- before major technology choices;
- during model training and evaluation design;
- when the user requests unconventional or newly encountered options;
- after previous approaches fail; and
- before declaring that reasonable options are exhausted.

Search or review the retained knowledge before recommending. Compare genuinely different strategy families,
cite retained source URLs for factual claims, and propose small reversible experiments with success and
failure criteria. Keep relevance, evidence strength, maturity, and publication time separate.
Treat experimental or unresolved records as options to validate, not established recommendations.
Record decisions or outcomes only
when the user asks you to preserve them."""
