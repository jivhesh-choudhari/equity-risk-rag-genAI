"""
app/config.py — Single source of truth for all project configuration.

Loads config.yml from the project root at import time.
Secrets (OPENAI_API_KEY) are read from .env via python-dotenv and are
never stored in config.yml.

Usage anywhere in the project:
    from app.config import cfg

    if cfg.llm.mock:
        ...
    chunk_size = cfg.chunker.chunk_size
    api_key    = cfg.secrets.openai_api_key
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

# Load .env first so os.getenv picks up OPENAI_API_KEY
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass  # python-dotenv optional; key can still be set in the shell environment

# ── Section dataclasses ───────────────────────────────────────────────────────

@dataclass
class LoaderConfig:
    source:         str  = "markdown"
    markdown_dir:   str  = "corpus/filings"
    pdf_dir:        str  = "corpus/pdfs"
    pdf_table_settings: dict = field(default_factory=lambda: {
        "vertical_strategy":   "lines",
        "horizontal_strategy": "lines",
        "snap_tolerance":       3,
    })


@dataclass
class ChunkerConfig:
    chunk_size:    int = 1000
    chunk_overlap: int = 150


@dataclass
class LLMConfig:
    mock:               bool  = False
    model:              str   = "gpt-4o-mini"
    temperature:        float = 0.0
    max_section_chars:  int   = 14000
    max_table_chars:    int   = 10000


@dataclass
class GraphConfig:
    max_retries: int = 2


@dataclass
class EvaluationConfig:
    groundedness_threshold: float = 0.70
    gold_labels_path:       str   = "evaluation/gold_labels.json"


@dataclass
class ServerConfig:
    host:   str  = "0.0.0.0"
    port:   int  = 9060
    reload: bool = True


@dataclass
class MCPConfig:
    transport: str = "stdio"


@dataclass
class DebugConfig:
    """Controls console debug output across the entire pipeline."""
    enabled: bool = False


@dataclass
class SecretsConfig:
    """Secrets are read from environment / .env — never from config.yml."""
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )

    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key != "sk-your-key-here")


# ── Root config ───────────────────────────────────────────────────────────────

@dataclass
class Config:
    loader:     LoaderConfig     = field(default_factory=LoaderConfig)
    chunker:    ChunkerConfig    = field(default_factory=ChunkerConfig)
    llm:        LLMConfig        = field(default_factory=LLMConfig)
    graph:      GraphConfig      = field(default_factory=GraphConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    server:     ServerConfig     = field(default_factory=ServerConfig)
    mcp:        MCPConfig        = field(default_factory=MCPConfig)
    debug:      DebugConfig      = field(default_factory=DebugConfig)
    secrets:    SecretsConfig    = field(default_factory=SecretsConfig)


# ── Loader ────────────────────────────────────────────────────────────────────

def _deep_update(dataclass_instance, data: dict):
    """Recursively apply a dict's values onto a dataclass instance."""
    for key, value in data.items():
        if not hasattr(dataclass_instance, key):
            continue
        current = getattr(dataclass_instance, key)
        if hasattr(current, '__dataclass_fields__') and isinstance(value, dict):
            _deep_update(current, value)
        else:
            setattr(dataclass_instance, key, value)


def _load_config() -> Config:
    config_path = Path(__file__).parent.parent / "config.yml"
    instance    = Config()
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        _deep_update(instance, raw)
    # Always refresh secrets from environment (never from file)
    instance.secrets = SecretsConfig()
    return instance


# ── Singleton — import this everywhere ───────────────────────────────────────

cfg: Config = _load_config()
