"""
LangChain @tool definitions — all pure functions, no LLM required.

These are the building blocks consumed by the agent nodes in agents.py.
Each tool is independently testable without an OpenAI key.
"""
from __future__ import annotations

import re
from typing import Dict, List

from langchain_core.tools import tool

from .chunker import chunk_documents
from .loader import LoaderFactory
from .sentiment_risk import (
    extract_risk_snippets,
    label_from_scores,
    score_sentiment,
)


# ── Tool 1: Load Filing ───────────────────────────────────────────────────────

@tool
def load_filing_tool(filing_id: str, source: str = "markdown") -> dict:
    """
    Load a financial filing by ID and return a list of section documents.

    Args:
        filing_id: Ticker-year identifier, e.g. 'HLSR-2024'.
        source:    'markdown' (synthetic) or 'pdf' (real 10-K/10-Q).

    Returns:
        {'documents': List[dict], 'section_count': int}
    """
    loader = LoaderFactory.get(source)
    docs = loader.load(filing_id)
    return {"documents": docs, "section_count": len(docs)}


# ── Tool 2: Extract Sections ──────────────────────────────────────────────────

@tool
def extract_sections_tool(documents: list, section_names: list) -> dict:
    """
    Filter a list of Documents to only those matching the given section names.

    Args:
        documents:     List of Document dicts from load_filing_tool.
        section_names: Sections to keep, e.g. ['Business', 'Risk Factors'].

    Returns:
        {'documents': List[dict], 'count': int, 'sections_found': List[str]}
    """
    name_set  = set(section_names)
    filtered  = [d for d in documents if d.get("metadata", {}).get("section") in name_set]
    found     = list({d["metadata"]["section"] for d in filtered})
    return {"documents": filtered, "count": len(filtered), "sections_found": found}


# ── Tool 3: Score Sentiment ───────────────────────────────────────────────────

@tool
def score_sentiment_tool(text: str) -> dict:
    """
    Rule-based sentiment scorer for financial text.
    Used as a fast fallback inside the Sentiment Agent.

    Returns:
        {'positive': int, 'negative': int, 'uncertain': int, 'tone': str}
    """
    pos, neg, unc = score_sentiment(text)
    return {
        "positive":  pos,
        "negative":  neg,
        "uncertain": unc,
        "tone":      label_from_scores(pos, neg, unc),
    }


# ── Tool 4: Extract Risk Factors ──────────────────────────────────────────────

@tool
def extract_risk_factors_tool(documents: list, top_k: int = 3) -> list:
    """
    Rank document chunks by risk-keyword density and return the top-k.
    Each result includes the first sentence + section citation.

    Args:
        documents: List of Document dicts (already chunked or raw sections).
        top_k:     Number of top risk snippets to return (default 3).

    Returns:
        List of {'snippet': str, 'severity': str, 'section': str}
    """
    # chunk_documents handles sections that are already small gracefully
    chunks  = chunk_documents(documents, chunk_size=1000, chunk_overlap=0)
    top     = extract_risk_snippets(chunks, k=top_k)
    results = []
    for ch in top:
        section  = ch["metadata"].get("section", "Unknown")
        sentence = ch["page_content"].split(".")[0].strip()
        # Heuristic severity from raw risk-word count
        raw_text = ch["page_content"].lower()
        score    = sum(raw_text.count(w) for w in
                       ["risk", "breach", "regulatory", "volatility", "churn", "penalties"])
        severity = "high" if score >= 4 else "medium" if score >= 2 else "low"
        if sentence:
            results.append({"snippet": f"{sentence}.", "severity": severity, "section": section})
    return results


# ── Tool 5: Extract Financial Tables ─────────────────────────────────────────

@tool
def extract_financial_tables_tool(documents: list) -> dict:
    """
    Pull key financial metrics from document chunks that contain Markdown tables.
    Looks for revenue, net income, gross margin, and YoY change patterns.

    Args:
        documents: List of Document dicts (table chunks preferred).

    Returns:
        {
          'revenue': str | None,
          'net_income': str | None,
          'gross_margin': str | None,
          'yoy_change': str | None,
          'raw_tables': List[str]
        }
    """
    table_docs = [d for d in documents if d.get("metadata", {}).get("chunk_type") == "table"
                  or "[TABLES]" in d.get("page_content", "")]

    raw_tables:  List[str] = []
    revenue:     str | None = None
    net_income:  str | None = None
    gross_margin: str | None = None
    yoy_change:  str | None = None

    for doc in table_docs:
        content = doc["page_content"]
        raw_tables.append(content)

        # Simple regex extraction from Markdown pipe tables
        rev_match = re.search(
            r'(?:total\s+revenue|net\s+revenue|revenues?)[^\|]*\|\s*([\$\d,\.]+[MBK]?)',
            content, re.IGNORECASE
        )
        ni_match = re.search(
            r'net\s+(?:income|loss)[^\|]*\|\s*([\(\$\d,\.]+[MBK]?\)?)',
            content, re.IGNORECASE
        )
        gm_match = re.search(
            r'gross\s+(?:margin|profit)[^\|]*\|\s*([\d\.]+%?)',
            content, re.IGNORECASE
        )
        yoy_match = re.search(
            r'(?:yoy|year.over.year|change)[^\|]*\|\s*([\+\-\d\.]+%)',
            content, re.IGNORECASE
        )

        if rev_match  and not revenue:      revenue      = rev_match.group(1).strip()
        if ni_match   and not net_income:   net_income   = ni_match.group(1).strip()
        if gm_match   and not gross_margin: gross_margin = gm_match.group(1).strip()
        if yoy_match  and not yoy_change:   yoy_change   = yoy_match.group(1).strip()

    return {
        "revenue":      revenue,
        "net_income":   net_income,
        "gross_margin": gross_margin,
        "yoy_change":   yoy_change,
        "raw_tables":   raw_tables,
    }


# ── Tool 6: Validate Output ───────────────────────────────────────────────────

@tool
def validate_output_tool(filing_summary: dict) -> dict:
    """
    Validate a FilingSummary dict against the required schema.

    Checks:
    - highlights: exactly 2 items
    - risks: 1-3 items, each has snippet/severity/section
    - tone: one of positive / neutral / cautious
    - financials: dict present

    Returns:
        {'valid': bool, 'errors': List[str]}
    """
    errors: List[str] = []

    highlights = filing_summary.get("highlights", [])
    if len(highlights) != 2:
        errors.append(f"highlights must have exactly 2 items, got {len(highlights)}")

    risks = filing_summary.get("risks", [])
    if not (1 <= len(risks) <= 3):
        errors.append(f"risks must have 1-3 items, got {len(risks)}")
    for i, r in enumerate(risks):
        for key in ("snippet", "severity", "section"):
            if key not in r:
                errors.append(f"risks[{i}] missing field '{key}'")

    tone = filing_summary.get("tone", "")
    if tone not in {"positive", "neutral", "cautious"}:
        errors.append(f"tone '{tone}' is not valid (must be positive/neutral/cautious)")

    if "financials" not in filing_summary:
        errors.append("financials key missing from summary")

    return {"valid": len(errors) == 0, "errors": errors}


# ── Tool 7: Groundedness Check ────────────────────────────────────────────────

@tool
def groundedness_check_tool(summary_text: str, source_text: str) -> dict:
    """
    Lexical overlap check: what fraction of summary tokens appear in the source?
    Score >= 0.70 is considered grounded.

    Returns:
        {'score': float, 'grounded': bool, 'total_tokens': int, 'hit_tokens': int}
    """
    tokens      = re.findall(r'\w+', summary_text.lower())
    source_lower = source_text.lower()
    hits        = sum(1 for t in tokens if t in source_lower)
    score       = hits / len(tokens) if tokens else 0.0
    return {
        "score":        round(score, 3),
        "grounded":     score >= 0.70,
        "hit_tokens":   hits,
        "total_tokens": len(tokens),
    }


# ── Tool registry (for agent binding) ────────────────────────────────────────

ALL_TOOLS: List = [
    load_filing_tool,
    extract_sections_tool,
    score_sentiment_tool,
    extract_risk_factors_tool,
    extract_financial_tables_tool,
    validate_output_tool,
    groundedness_check_tool,
]
