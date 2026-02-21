# Public API for the app package
from .config import cfg                  # project-wide configuration singleton
from .summarize import summarize_filing  # legacy rule-based pipeline (always available)

try:
    from .graph import run_pipeline      # agentic pipeline (requires langgraph)
    __all__ = ["run_pipeline", "summarize_filing", "cfg"]
except ImportError:
    __all__ = ["summarize_filing", "cfg"]
