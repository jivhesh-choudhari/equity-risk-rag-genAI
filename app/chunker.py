from typing import List
import re

from .debug import dlog

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        _LANGCHAIN_AVAILABLE = True
    except ImportError:
        _LANGCHAIN_AVAILABLE = False

# Separator priority tuned for SEC filing structure:
#   section breaks > paragraphs > sentences > clauses > words
_FINANCIAL_SEPARATORS = [
    '\n\n\n',   # major section breaks
    '\n\n',     # paragraph breaks
    '\n',       # line breaks
    '. ',       # sentence endings
    '; ',       # clause separators common in legal/financial text
    ', ',       # sub-clause breaks
    ' ',        # word breaks
    '',         # character fallback
]


def chunk_documents(
    docs: List[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[dict]:
    """
    Splits documents into semantically coherent overlapping chunks.

    Strategy:
    - Uses RecursiveCharacterTextSplitter with SEC-filing-aware separators.
    - Table blocks (marked [TABLES]...) are NEVER split mid-row; each table
      block becomes its own chunk with chunk_type='table'.
    - Prose sections are chunked normally with chunk_type='text'.
    - Falls back to original fixed-size slicing if LangChain is not installed.

    Args:
        docs:          List of Documents from the loader.
        chunk_size:    Target max characters per chunk (default 1000).
        chunk_overlap: Overlap between adjacent text chunks (default 150).
    """
    dlog("chunker", f"chunk_documents called",
         {"input_docs": len(docs), "chunk_size": chunk_size,
          "chunk_overlap": chunk_overlap,
          "backend": "langchain" if _LANGCHAIN_AVAILABLE else "fallback"})
    if _LANGCHAIN_AVAILABLE:
        result = _langchain_chunk(docs, chunk_size, chunk_overlap)
    else:
        result = _fallback_chunk(docs, chunk_size)
    dlog("chunker", f"chunk_documents done",
         {"output_chunks": len(result),
          "table_chunks": sum(1 for c in result if c.get('metadata', {}).get('chunk_type') == 'table'),
          "text_chunks":  sum(1 for c in result if c.get('metadata', {}).get('chunk_type') == 'text')})
    return result


def _langchain_chunk(docs: List[dict], chunk_size: int, chunk_overlap: int) -> List[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_FINANCIAL_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )
    out = []
    for d in docs:
        text = d['page_content']
        meta = d.get('metadata', {})
        section = meta.get('section', 'Unknown')

        # --- Separate table blocks: keep them intact, never split mid-table ---
        table_match = re.search(r'\[TABLES\]\n', text, re.DOTALL)
        if table_match:
            prose      = text[:table_match.start()].strip()
            table_block = text[table_match.start():].strip()
        else:
            prose       = text
            table_block = None

        # --- Chunk prose with overlap ---
        prose_count = 0
        if prose.strip():
            splits = splitter.split_text(prose)
            for i, chunk_text in enumerate(splits):
                if chunk_text.strip():
                    out.append({
                        'page_content': chunk_text,
                        'metadata': {**meta, 'chunk_index': i, 'chunk_type': 'text'},
                    })
                    prose_count += 1

        # --- Table block: one chunk per page, never split ---
        table_count = 0
        if table_block:
            out.append({
                'page_content': table_block,
                'metadata': {**meta, 'chunk_index': 0, 'chunk_type': 'table'},
            })
            table_count = 1

        dlog("chunker", f"  Section '{section}': {prose_count} prose + {table_count} table chunks")

    return out


def _fallback_chunk(docs: List[dict], max_chars: int) -> List[dict]:
    """Original fixed-size fallback when LangChain is unavailable."""
    dlog("chunker", f"_fallback_chunk called",
         {"input_docs": len(docs), "max_chars": max_chars})
    out = []
    for d in docs:
        t = d['page_content']; m = d.get('metadata', {})
        if len(t) <= max_chars:
            out.append(d)
        else:
            for i in range(0, len(t), max_chars):
                out.append({
                    'page_content': t[i:i + max_chars],
                    'metadata': {**m, 'chunk_index': i // max_chars},
                })
    dlog("chunker", f"_fallback_chunk done", {"output_chunks": len(out)})
    return out
