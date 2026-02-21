"""
MCP Server: filing-loader-mcp

Exposes three tools over the Model Context Protocol (stdio transport):
  - load_filing        → loads a filing, returns section documents
  - get_sections       → loads a filing and filters to named sections
  - chunk_filing       → loads, chunks, and separates text vs table chunks

Run with:
    python -m mcp_server.filing_loader_server

Or register in your MCP client config (e.g. Claude Desktop, VS Code MCP):
    {
      "mcpServers": {
        "filing-loader": {
          "command": "python",
          "args": ["-m", "mcp_server.filing_loader_server"],
          "cwd": "<repo_root>"
        }
      }
    }
"""
from __future__ import annotations

import os
import sys

# Make app/ importable when run as __main__
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mcp.server.fastmcp import FastMCP

from app.chunker import chunk_documents
from app.loader import LoaderFactory

mcp = FastMCP("filing-loader-mcp")


# ── Tool 1: Load Filing ───────────────────────────────────────────────────────

@mcp.tool()
def load_filing(filing_id: str, source: str = "markdown") -> dict:
    """
    Load a financial filing by its ID.

    Args:
        filing_id: Ticker-year identifier, e.g. 'HLSR-2024'.
        source:    'markdown' for synthetic corpus, 'pdf' for real 10-K/10-Q.

    Returns:
        {
          "documents": List[{page_content, metadata}],
          "section_count": int,
          "sections": List[str]
        }
    """
    loader = LoaderFactory.get(source)
    docs   = loader.load(filing_id)
    sections = list({d["metadata"].get("section", "Unknown") for d in docs})
    return {
        "documents":     docs,
        "section_count": len(docs),
        "sections":      sorted(sections),
    }


# ── Tool 2: Get Sections ──────────────────────────────────────────────────────

@mcp.tool()
def get_sections(filing_id: str, section_names: list[str],
                 source: str = "markdown") -> dict:
    """
    Load a filing and return only the documents that belong to the
    specified section names.

    Args:
        filing_id:     Ticker-year identifier, e.g. 'HLSR-2024'.
        section_names: e.g. ['Business', 'Risk Factors', 'Results'].
        source:        'markdown' or 'pdf'.

    Returns:
        {
          "documents": List[{page_content, metadata}],
          "count": int,
          "sections_found": List[str]
        }
    """
    loader   = LoaderFactory.get(source)
    docs     = loader.load(filing_id)
    name_set = set(section_names)
    filtered = [d for d in docs if d.get("metadata", {}).get("section") in name_set]
    found    = list({d["metadata"]["section"] for d in filtered})
    return {
        "documents":     filtered,
        "count":         len(filtered),
        "sections_found": found,
    }


# ── Tool 3: Chunk Filing ──────────────────────────────────────────────────────

@mcp.tool()
def chunk_filing(filing_id: str, source: str = "markdown",
                 chunk_size: int = 1000, chunk_overlap: int = 150) -> dict:
    """
    Load, chunk, and split a filing into text chunks and table chunks.

    Table chunks (from PDF pipe-table blocks) are kept intact and returned
    separately so financial agents can process them without splitting mid-row.

    Args:
        filing_id:     Ticker-year identifier.
        source:        'markdown' or 'pdf'.
        chunk_size:    Max chars per text chunk (default 1000).
        chunk_overlap: Overlap between adjacent text chunks (default 150).

    Returns:
        {
          "text_chunks":  List[{page_content, metadata}],
          "table_chunks": List[{page_content, metadata}],
          "total":        int
        }
    """
    loader = LoaderFactory.get(source)
    docs   = loader.load(filing_id)
    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    text_chunks  = [c for c in chunks if c.get("metadata", {}).get("chunk_type", "text") == "text"]
    table_chunks = [c for c in chunks if c.get("metadata", {}).get("chunk_type") == "table"]

    return {
        "text_chunks":  text_chunks,
        "table_chunks": table_chunks,
        "total":        len(chunks),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
