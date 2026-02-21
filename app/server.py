from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .summarize import summarize_filing   # legacy rule-based endpoint (preserved)

app = FastAPI(
    title='Financial Research Analyst Agent',
    version='0.2.0',
    description=(
        'POST /summarize  — legacy rule-based pipeline (always available).\n'
        'POST /analyze    — LangGraph agentic pipeline (requires langgraph install).'
    ),
)


class SumReq(BaseModel):
    filing_id: str
    source: str = "markdown"  # "markdown" or "pdf"


# ── Legacy endpoint (rule-based, no LLM) ─────────────────────────────────────

@app.post('/summarize')
def summarize(req: SumReq):
    """Original keyword-based pipeline. Always available without an LLM."""
    try:
        return summarize_filing(req.filing_id, source=req.source)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── Agentic endpoint (LangGraph pipeline) ────────────────────────────────────

@app.post('/analyze')
def analyze(req: SumReq):
    """
    Full LangGraph agentic pipeline.
    Returns FilingSummary with highlights, risks, tone, financials, sources.
    Falls back to rule-based agents if OPENAI_API_KEY is not set or MOCK_LLM=true.
    """
    try:
        from .graph import run_pipeline
    except ImportError as e:
        raise HTTPException(
            status_code=501,
            detail=f'langgraph not installed. Run: pip install langgraph. Error: {e}',
        )
    try:
        return run_pipeline(req.filing_id, source=req.source)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
