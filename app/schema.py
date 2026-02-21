"""
Pydantic output models and LangGraph shared state.
"""
from __future__ import annotations

import operator
from typing import Annotated, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field


# ── Output models (Pydantic) ──────────────────────────────────────────────────

class RiskItem(BaseModel):
    snippet:  str
    severity: str   # "high" | "medium" | "low"
    section:  str


class FinancialMetrics(BaseModel):
    revenue:      Optional[str] = None
    net_income:   Optional[str] = None
    gross_margin: Optional[str] = None
    yoy_change:   Optional[str] = None
    raw_tables:   List[str]     = Field(default_factory=list)


class FilingSummary(BaseModel):
    filing_id:  str
    highlights: List[str]           # max 2, section-cited
    risks:      List[RiskItem]      # max 3, severity-ranked
    tone:       str                 # "positive" | "neutral" | "cautious"
    financials: FinancialMetrics
    sources:    List[dict]          # [{section, page, chunk_index}]


class FilingState(TypedDict):
    # ── Set by Orchestrator ──────────────────────────────────────────────────
    filing_id:      str
    filing_source:  str                                 # "markdown" | "pdf" | ""
    source_docs:    List[dict]                          # raw loader output
    sections:       Dict[str, List[dict]]               # {section_label: [docs]}

    # ── Written by Sentiment Agent ───────────────────────────────────────────
    tone:           str
    tone_reasoning: str

    # ── Written by Risk Agent ────────────────────────────────────────────────
    # Reducer: accumulated across retries / parallel updates
    risks:          Annotated[List[dict], operator.add]

    # ── Written by Financial Agent ───────────────────────────────────────────
    financials:     dict

    # ── Written by Summarizer Agent ─────────────────────────────────────────
    highlights:     List[str]

    # ── Accumulated by all agents ────────────────────────────────────────────
    sources:        Annotated[List[dict], operator.add]
    errors:         Annotated[List[str],  operator.add]

    # ── Written by Evaluator Agent ───────────────────────────────────────────
    eval_result:    dict
    retry_count:    int
