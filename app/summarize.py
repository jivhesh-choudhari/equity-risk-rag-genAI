"""
Legacy rule-based pipeline — backs the POST /summarize endpoint.

All extraction logic lives in app/tools.py (shared with agents.py).
This module is a thin orchestrator: load → chunk → call tools → format response.
No LLM involved; always available without an API key.
"""
from __future__ import annotations

from .config import cfg
from .debug import dlog
from .loader import LoaderFactory
from .chunker import chunk_documents
from .tools import (
    extract_risk_factors_tool,
    extract_financial_tables_tool,
    score_sentiment_tool,
)


def summarize_filing(filing_id: str, source: str = None) -> dict:
    """
    Rule-based summarisation pipeline.

    Returns a dict compatible with the legacy /summarize response schema:
        {filing_id, highlights, risks, tone, financials}
    """
    dlog("summarize", "=== summarize_filing START ===", {"filing_id": filing_id, "source": source})

    # ── Load + chunk ──────────────────────────────────────────────────────────
    loader = LoaderFactory.get(source)
    docs   = loader.load(filing_id)          # raises FileNotFoundError if missing
    dlog("summarize", "Loaded docs",
         {"count": len(docs),
          "sections": [d["metadata"].get("section") for d in docs]})

    chunks = chunk_documents(
        docs,
        chunk_size=cfg.chunker.chunk_size,
        chunk_overlap=cfg.chunker.chunk_overlap,
    )
    dlog("summarize", "Chunked", {"chunks": len(chunks)})

    # ── Highlights: first sentence of Business / Results sections ─────────────
    highlights: list[str] = []
    for ch in chunks:
        if len(highlights) >= 2:
            break
        sec = ch["metadata"].get("section", "")
        if sec in {"Business", "Results"}:
            sentence = ch["page_content"].split(".")[0].strip()
            if sentence:
                highlights.append(f"{sentence}. ({sec})")
    dlog("summarize", "Highlights extracted", {"highlights": highlights})

    # ── Risks: top-k keyword-density ranked snippets (via shared tool) ────────
    raw_risks = extract_risk_factors_tool.invoke({"documents": chunks, "top_k": 3})
    # Normalise to plain strings for legacy response compatibility
    risk_strings = [
        f"{r['snippet']} ({r['section']})" for r in raw_risks
    ]
    dlog("summarize", "Risks extracted",
         {"count": len(risk_strings), "risks": risk_strings})

    # ── Tone: aggregate keyword scoring across all chunks (via shared tool) ───
    all_text = " ".join(ch["page_content"] for ch in chunks)
    sentiment = score_sentiment_tool.invoke({"text": all_text})
    tone      = sentiment["tone"]
    dlog("summarize", "Tone scored",
         {"tone": tone, "positive": sentiment["positive"],
          "negative": sentiment["negative"], "uncertain": sentiment["uncertain"]})

    # ── Financials: regex extraction from table chunks (via shared tool) ──────
    financials = extract_financial_tables_tool.invoke({"documents": chunks})
    dlog("summarize", "Financials extracted",
         {k: v for k, v in financials.items() if k != "raw_tables"})

    result = {
        "filing_id":  filing_id,
        "highlights": highlights[:2],
        "risks":      risk_strings[:3],
        "tone":       tone,
        "financials": {
            k: v for k, v in financials.items() if k != "raw_tables"
        },
    }
    dlog("summarize", "=== summarize_filing END ===", {"filing_id": filing_id})
    return result
