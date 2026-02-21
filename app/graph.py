"""
LangGraph StateGraph — wires all agent nodes into the pipeline.

Graph topology:
                     START
                       │
                  orchestrator
                       │
         ┌─────────────┼─────────────┐  parallel fan-out
         ▼             ▼             ▼
   sentiment_node  risk_node  financial_node
         │             │             │
         └─────────────┼─────────────┘  fan-in barrier
                       ▼
                 summarizer_node
                       │
                 evaluator_node
                       │
              ┌────────┴────────┐
          pass│                 │fail (retry_count < 2)
              ▼                 ▼
             END          summarizer_node
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .agents import (
    evaluator_node,
    financial_node,
    orchestrator_node,
    risk_node,
    sentiment_node,
    summarizer_node,
)
from .config import cfg
from .debug import dlog
from .schema import FilingState


# ── Routing function ──────────────────────────────────────────────────────────

def _route_evaluator(state: FilingState) -> str:
    """
    Pass → END.
    Fail + retries remaining → back to summarizer for one more attempt.
    Fail + exhausted → END anyway (partial result is still returned).
    """
    result      = state.get("eval_result", {})
    retry_count = state.get("retry_count", 0)
    if result.get("valid", False) or retry_count >= cfg.graph.max_retries:
        decision = "end"
    else:
        decision = "retry"
    dlog("graph", f"_route_evaluator decision: {decision}",
         {"eval_valid": result.get("valid", False),
          "retry_count": retry_count,
          "max_retries": cfg.graph.max_retries})
    return decision


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph():
    """
    Compile and return the runnable LangGraph pipeline.

    Usage:
        graph = build_graph()
        result = graph.invoke({"filing_id": "HLSR-2024"})
    """
    builder = StateGraph(FilingState)

    # Register nodes
    builder.add_node("orchestrator",   orchestrator_node)
    builder.add_node("sentiment_node", sentiment_node)
    builder.add_node("risk_node",      risk_node)
    builder.add_node("financial_node", financial_node)
    builder.add_node("summarizer_node", summarizer_node)
    builder.add_node("evaluator_node", evaluator_node)

    # Entry
    builder.add_edge(START, "orchestrator")

    # Fan-out: orchestrator → three parallel analysis nodes
    # LangGraph executes these concurrently; each writes to a distinct state key
    builder.add_edge("orchestrator", "sentiment_node")
    builder.add_edge("orchestrator", "risk_node")
    builder.add_edge("orchestrator", "financial_node")

    # Fan-in barrier: all three must complete before summarizer runs
    builder.add_edge("sentiment_node",  "summarizer_node")
    builder.add_edge("risk_node",       "summarizer_node")
    builder.add_edge("financial_node",  "summarizer_node")

    # Summarizer → Evaluator
    builder.add_edge("summarizer_node", "evaluator_node")

    # Evaluator conditional: pass → END | fail → retry summarizer (max 2 retries)
    builder.add_conditional_edges(
        "evaluator_node",
        _route_evaluator,
        {"end": END, "retry": "summarizer_node"},
    )

    return builder.compile()


# ── Convenience wrapper ───────────────────────────────────────────────────────

def run_pipeline(filing_id: str) -> dict:
    """
    Run the full agentic pipeline for a single filing.

    Returns a dict matching FilingSummary schema:
        {filing_id, highlights, risks, tone, financials, sources, eval_result}
    """
    dlog("graph", "=== run_pipeline START ===", {"filing_id": filing_id})
    graph  = build_graph()
    dlog("graph", "Graph compiled, invoking")
    result = graph.invoke({"filing_id": filing_id})
    output = {
        "filing_id":  filing_id,
        "highlights": result.get("highlights", []),
        "risks":      result.get("risks", []),
        "tone":       result.get("tone", "neutral"),
        "financials": result.get("financials", {}),
        "sources":    result.get("sources", []),
        "eval_result": result.get("eval_result", {}),
        "errors":     result.get("errors", []),
    }
    dlog("graph", "=== run_pipeline END ===",
         {"filing_id": filing_id,
          "tone": output["tone"],
          "highlights_count": len(output["highlights"]),
          "risks_count": len(output["risks"]),
          "eval_passed": output["eval_result"].get("valid", False),
          "errors": output["errors"]})
    return output
