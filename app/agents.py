"""
LangGraph agent node functions.

Each function:
  - Takes FilingState
  - Returns a partial state dict (only the keys it owns)
  - Uses LLM when OPENAI_API_KEY is set; falls back to rule-based tools otherwise
  - Never raises — errors go into state['errors']

Node execution order (see graph.py):
  orchestrator
       │
  ┌────┼────┐  (parallel)
  ▼    ▼    ▼
  sentiment  risk  financial
       │    │    │  (fan-in barrier)
       └────┼────┘
            ▼
        summarizer
            ▼
         evaluator
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

from .loader import LoaderFactory
from .chunker import chunk_documents
from .schema import FilingState
from .sentiment_risk import score_sentiment, label_from_scores
from .tools import (
    extract_risk_factors_tool,
    extract_financial_tables_tool,
    groundedness_check_tool,
    validate_output_tool,
)


# ── LLM helper ───────────────────────────────────────────────────────────────

def _get_llm():
    """Return ChatOpenAI instance or None if unavailable."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    mock    = os.getenv("MOCK_LLM", "false").lower() == "true"
    if mock or not api_key:
        return None
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model       = os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature = float(os.getenv("LLM_TEMPERATURE", "0")),
        )
    except ImportError:
        return None


def _llm_json(llm, prompt_text: str, fallback: dict) -> dict:
    """Invoke LLM with a text prompt, return parsed JSON or fallback."""
    try:
        from langchain_core.messages import HumanMessage
        from langchain_core.output_parsers import JsonOutputParser
        response = llm.invoke(
            [HumanMessage(content=prompt_text)],
            response_format={"type": "json_object"},
        )
        return JsonOutputParser().parse(response.content)
    except Exception as e:
        return {**fallback, "_llm_error": str(e)}


def _section_text(sections: Dict[str, List[dict]], names: list, max_chars: int = 14000) -> str:
    """Concatenate page_content from requested sections, capped at max_chars."""
    parts = []
    for name in names:
        for doc in sections.get(name, []):
            parts.append(doc.get("page_content", ""))
    return "\n\n".join(parts)[:max_chars]


def _read_prompt(filename: str) -> str:
    prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")
    path = os.path.join(prompts_dir, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


# ── Node 1: Orchestrator ─────────────────────────────────────────────────────

def orchestrator_node(state: FilingState) -> dict:
    """
    Load the filing, split into sections, initialise state.
    No LLM call — pure I/O and routing.
    """
    filing_id = state["filing_id"]
    try:
        loader    = LoaderFactory.get()
        docs      = loader.load(filing_id)
        chunks    = chunk_documents(docs, chunk_size=1000, chunk_overlap=150)

        # Build sections dict  {section_label: [chunk_docs]}
        sections: Dict[str, List[dict]] = {}
        for ch in chunks:
            sec = ch.get("metadata", {}).get("section", "Unknown")
            sections.setdefault(sec, []).append(ch)

        return {
            "source_docs":  docs,
            "sections":     sections,
            "retry_count":  0,
            "errors":       [],
            "risks":        [],
            "sources":      [],
        }
    except Exception as e:
        return {"errors": [f"orchestrator: {e}"], "source_docs": [], "sections": {}}


# ── Node 2: Sentiment Agent ───────────────────────────────────────────────────

def sentiment_node(state: FilingState) -> dict:
    """
    Classify overall filing tone as positive / neutral / cautious.
    LLM path: structured JSON output with reasoning.
    Fallback: rule-based keyword scorer from sentiment_risk.py.
    """
    sections = state.get("sections", {})
    text     = _section_text(sections, ["Business", "Results", "Unknown"])
    if not text:
        return {"tone": "neutral", "tone_reasoning": "No text found", "errors": []}

    llm = _get_llm()

    # ── LLM path ─────────────────────────────────────────────────────────────
    if llm:
        prompt_template = _read_prompt("sentiment_prompt.txt")
        prompt = (
            f"{prompt_template}\n\n"
            f"### FILING TEXT ###\n{text}\n\n"
            '### OUTPUT ###\n'
            'Return JSON: {"tone": "positive|neutral|cautious", "reasoning": "..."}'
        )
        result = _llm_json(llm, prompt,
                           fallback={"tone": "neutral", "reasoning": "LLM parse error"})
        tone   = result.get("tone", "neutral")
        if tone not in ("positive", "neutral", "cautious"):
            tone = "neutral"
        return {
            "tone":           tone,
            "tone_reasoning": result.get("reasoning", ""),
            "sources":        [{"section": "Business+Results", "agent": "sentiment"}],
        }

    # ── Rule-based fallback ───────────────────────────────────────────────────
    p = n = u = 0
    for docs in sections.values():
        for doc in docs:
            sp, sn, su = score_sentiment(doc.get("page_content", ""))
            p += sp; n += sn; u += su
    tone = label_from_scores(p, n, u)
    return {
        "tone":           tone,
        "tone_reasoning": f"Rule-based (pos={p} neg={n} unc={u})",
        "sources":        [{"section": "all", "agent": "sentiment_rule_based"}],
    }


# ── Node 3: Risk Agent ────────────────────────────────────────────────────────

def risk_node(state: FilingState) -> dict:
    """
    Extract top-3 risk factors from the Risk Factors section.
    LLM path: structured extraction with severity and citations.
    Fallback: keyword-density ranking from tools.py.
    """
    sections  = state.get("sections", {})
    risk_docs = (
        sections.get("Risk Factors", []) or
        sections.get("Risks", []) or
        sections.get("Unknown", [])
    )
    if not risk_docs:
        return {"risks": [], "errors": ["risk_node: no risk-factor section found"]}

    llm = _get_llm()

    # ── LLM path ─────────────────────────────────────────────────────────────
    if llm:
        text            = _section_text(sections, ["Risk Factors", "Risks", "Unknown"])
        prompt_template = _read_prompt("risk_prompt.txt")
        prompt = (
            f"{prompt_template}\n\n"
            f"### RISK FACTORS TEXT ###\n{text}\n\n"
            "### OUTPUT ###\n"
            'Return JSON: {"risks": [{"snippet": "...", "severity": "high|medium|low", "section": "..."}]}'
        )
        result = _llm_json(llm, prompt, fallback={"risks": []})
        risks  = result.get("risks", [])[:3]
        return {
            "risks":   risks,
            "sources": [{"section": "Risk Factors", "agent": "risk"}],
        }

    # ── Rule-based fallback ───────────────────────────────────────────────────
    risks = extract_risk_factors_tool.invoke({"documents": risk_docs, "top_k": 3})
    return {
        "risks":   risks,
        "sources": [{"section": "Risk Factors", "agent": "risk_rule_based"}],
    }


# ── Node 4: Financial Agent ───────────────────────────────────────────────────

def financial_node(state: FilingState) -> dict:
    """
    Extract key financial metrics from table chunks (income statements, etc.).
    LLM path: structured reading of Markdown pipe tables.
    Fallback: regex-based metric extraction from tools.py.
    """
    sections       = state.get("sections", {})
    all_docs       = [d for docs in sections.values() for d in docs]
    table_docs     = [d for d in all_docs if d.get("metadata", {}).get("chunk_type") == "table"
                      or "[TABLES]" in d.get("page_content", "")]
    target_docs    = table_docs or all_docs   # fall back to all if no tables found

    llm = _get_llm()

    # ── LLM path ─────────────────────────────────────────────────────────────
    if llm and table_docs:
        tables_text = "\n\n".join(d["page_content"] for d in table_docs)[:10000]
        prompt = (
            "You are a financial data extractor. Read the Markdown tables below and "
            "extract key metrics. Return JSON with keys: revenue, net_income, "
            "gross_margin, yoy_change. Use null for missing values. "
            "Values should be strings including units (e.g. '$12.4M', '8.3%', '-15%').\n\n"
            f"### TABLES ###\n{tables_text}\n\n"
            '### OUTPUT ###\n{"revenue": ..., "net_income": ..., "gross_margin": ..., "yoy_change": ...}'
        )
        result = _llm_json(llm, prompt,
                           fallback={"revenue": None, "net_income": None,
                                     "gross_margin": None, "yoy_change": None})
        result["raw_tables"] = [d["page_content"][:500] for d in table_docs[:3]]
        return {
            "financials": result,
            "sources":    [{"section": "Financial Statements", "agent": "financial"}],
        }

    # ── Rule-based fallback ───────────────────────────────────────────────────
    financials = extract_financial_tables_tool.invoke({"documents": target_docs})
    return {
        "financials": financials,
        "sources":    [{"section": "Financial Statements", "agent": "financial_rule_based"}],
    }


# ── Node 5: Summarizer Agent ──────────────────────────────────────────────────

def summarizer_node(state: FilingState) -> dict:
    """
    Produce 2 highlight bullets grounded in Business/Results sections.
    Has access to tone, risks, and financials from prior parallel agents.
    LLM path: synthesis with full context.
    Fallback: first-sentence extraction (legacy behaviour).
    """
    sections   = state.get("sections", {})
    tone       = state.get("tone", "neutral")
    risks      = state.get("risks", [])
    financials = state.get("financials", {})
    text       = _section_text(sections, ["Business", "Results"])

    if not text:
        return {"highlights": [], "errors": ["summarizer: no Business/Results text"]}

    llm = _get_llm()

    # ── LLM path ─────────────────────────────────────────────────────────────
    if llm:
        risk_summary  = "; ".join(r.get("snippet", "") for r in risks[:2])
        fin_summary   = (f"Revenue: {financials.get('revenue')}, "
                         f"Net income: {financials.get('net_income')}, "
                         f"Gross margin: {financials.get('gross_margin')}")
        prompt_template = _read_prompt("summary_prompt.txt")
        prompt = (
            f"{prompt_template}\n\n"
            f"Overall tone: {tone}\n"
            f"Key risks: {risk_summary}\n"
            f"Financial snapshot: {fin_summary}\n\n"
            f"### BUSINESS & RESULTS TEXT ###\n{text}\n\n"
            "### OUTPUT ###\n"
            'Return JSON: {"highlights": ["bullet 1 (Section)", "bullet 2 (Section)"]}'
        )
        result     = _llm_json(llm, prompt, fallback={"highlights": []})
        highlights = result.get("highlights", [])[:2]
        return {
            "highlights": highlights,
            "sources":    [{"section": "Business+Results", "agent": "summarizer"}],
        }

    # ── Rule-based fallback (legacy first-sentence extraction) ────────────────
    highlights = []
    for sec_name in ("Business", "Results"):
        for doc in sections.get(sec_name, []):
            if len(highlights) >= 2:
                break
            sentence = doc["page_content"].split(".")[0].strip()
            if sentence:
                highlights.append(f"{sentence}. ({sec_name})")
    return {
        "highlights": highlights[:2],
        "sources":    [{"section": "Business+Results", "agent": "summarizer_rule_based"}],
    }


# ── Node 6: Evaluator Agent ───────────────────────────────────────────────────

def evaluator_node(state: FilingState) -> dict:
    """
    Validate output schema and groundedness. Sets eval_result and increments
    retry_count on failure (graph.py routes back to summarizer if retry < 2).
    """
    highlights = state.get("highlights", [])
    risks      = state.get("risks", [])
    tone       = state.get("tone", "")
    financials = state.get("financials", {})
    source_docs = state.get("source_docs", [])

    # ── Schema validation ─────────────────────────────────────────────────────
    summary_dict = {
        "highlights": highlights,
        "risks":      risks,
        "tone":       tone,
        "financials": financials,
    }
    validation = validate_output_tool.invoke({"filing_summary": summary_dict})

    # ── Groundedness check ────────────────────────────────────────────────────
    source_text   = " ".join(d.get("page_content", "") for d in source_docs)
    summary_text  = " ".join(highlights + [r.get("snippet", "") for r in risks])
    groundedness  = groundedness_check_tool.invoke({
        "summary_text": summary_text,
        "source_text":  source_text,
    })

    passed = validation["valid"] and groundedness["grounded"]
    errors = validation["errors"]
    if not groundedness["grounded"]:
        errors.append(f"groundedness too low: {groundedness['score']:.2f} (min 0.70)")

    return {
        "eval_result":  {
            "valid":        passed,
            "validation":   validation,
            "groundedness": groundedness,
        },
        "retry_count":  state.get("retry_count", 0) + (0 if passed else 1),
        "errors":       errors,
    }
