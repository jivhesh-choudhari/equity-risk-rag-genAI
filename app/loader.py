from typing import List
import re, os


# ── Markdown Loader (synthetic corpus) ────────────────────────────────────────

class MarkdownFilingLoader:
    """Parses section-split Markdown filings into LangChain-style Documents."""

    def __init__(self, corpus_dir: str):
        self.corpus_dir = corpus_dir

    def load(self, filing_id: str) -> List[dict]:
        path = os.path.join(self.corpus_dir, f"{filing_id}.md")
        return _parse_markdown_file(path)


def _parse_markdown_file(path: str) -> List[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    parts = re.split(r'\n(?=## )', text)
    title = parts[0].splitlines()[0].replace('# ', '')
    docs = []; current = 'Preamble'; buf = []
    for p in parts:
        if p.startswith('## '):
            if buf:
                docs.append({'page_content': '\n'.join(buf).strip(),
                             'metadata': {'title': title, 'section': current, 'source': path}})
            lines = p.splitlines(); current = lines[0].replace('## ', '').strip(); buf = lines[1:]
        else:
            buf.extend(p.splitlines()[1:])
    if buf:
        docs.append({'page_content': '\n'.join(buf).strip(),
                     'metadata': {'title': title, 'section': current, 'source': path}})
    return docs


# ── PDF Loader (real 10-K / 10-Q filings) ─────────────────────────────────────

# Section header patterns common in SEC 10-K / 10-Q filings
_SECTION_PATTERNS = [
    (r'item\s+1a[\.\ s]', 'Risk Factors'),
    (r'item\s+1[\.\ s]',  'Business'),
    (r'item\s+7a[\.\ s]', 'Quantitative Disclosures'),
    (r'item\s+7[\.\ s]',  'Results'),
    (r'item\s+8[\.\ s]',  'Financial Statements'),
    (r'item\s+2[\.\ s]',  'Properties'),
    (r'item\s+3[\.\ s]',  'Legal Proceedings'),
    (r'item\s+9[\.\ s]',  'Disagreements with Accountants'),
]


def _detect_section(text: str) -> str:
    """Heuristically map the first 400 chars of a page to an SEC section label."""
    lower = text.lower()[:400]
    for pattern, label in _SECTION_PATTERNS:
        if re.search(pattern, lower):
            return label
    return 'Unknown'


def _table_to_markdown(table: list) -> str:
    """Convert a pdfplumber table (list-of-lists) to a Markdown pipe table."""
    rows = []
    for i, row in enumerate(table):
        cleaned = [str(c).strip().replace('\n', ' ') if c is not None else '' for c in row]
        rows.append('| ' + ' | '.join(cleaned) + ' |')
        if i == 0:                              # header separator after first row
            rows.append('|' + ' --- |' * len(row))
    return '\n'.join(rows)


class PDFFilingLoader:
    """
    Extracts text + tables from financial PDFs using pdfplumber.

    Why pdfplumber over PyPDFLoader:
    - Preserves table cell boundaries (critical for income statements, balance sheets)
    - Tables are serialised to Markdown pipe format so LLMs see structured data
    - Handles multi-column layouts better than basic PDF text extraction
    - Returns one Document per page with metadata: {title, section, page, source, has_tables}
    """

    def __init__(self, pdf_dir: str):
        self.pdf_dir = pdf_dir

    def load(self, filing_id: str) -> List[dict]:
        try:
            import pdfplumber
        except ImportError:
            raise ImportError(
                "pdfplumber is required for PDF loading. "
                "Install it with: pip install pdfplumber"
            )

        path = os.path.join(self.pdf_dir, f"{filing_id}.pdf")
        if not os.path.exists(path):
            raise FileNotFoundError(f"PDF not found: {path}")

        docs = []
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):

                # --- Extract tables and convert to Markdown ---
                raw_tables = page.extract_tables(
                    table_settings={
                        'vertical_strategy': 'lines',
                        'horizontal_strategy': 'lines',
                        'snap_tolerance': 3,
                    }
                )
                table_blocks = [
                    _table_to_markdown(t) for t in (raw_tables or []) if t
                ]

                # --- Extract prose text ---
                text = page.extract_text(x_tolerance=3, y_tolerance=3) or ''

                if not text.strip() and not table_blocks:
                    continue  # skip blank / image-only pages

                # --- Combine: prose first, then labelled table block ---
                parts = [text.strip()] if text.strip() else []
                if table_blocks:
                    parts.append('[TABLES]\n' + '\n\n'.join(table_blocks))
                combined = '\n\n'.join(parts)

                docs.append({
                    'page_content': combined,
                    'metadata': {
                        'title':      filing_id,
                        'section':    _detect_section(text),
                        'page':       page_num,
                        'source':     path,
                        'has_tables': bool(table_blocks),
                    }
                })

        return docs


# ── Factory ────────────────────────────────────────────────────────────────────

class LoaderFactory:
    """
    Returns the appropriate loader based on the FILING_SOURCE env var.

    FILING_SOURCE=markdown  →  MarkdownFilingLoader  (default, synthetic corpus)
    FILING_SOURCE=pdf       →  PDFFilingLoader        (real 10-K / 10-Q PDFs)
    """

    @staticmethod
    def get(source: str = None):
        source = source or os.getenv('FILING_SOURCE', 'markdown')
        if source == 'pdf':
            pdf_dir = os.getenv('PDF_DIR', os.path.join('corpus', 'pdfs'))
            return PDFFilingLoader(pdf_dir)
        corpus_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'corpus', 'filings'
        )
        return MarkdownFilingLoader(corpus_dir)


# ── Backward-compat shim (used by summarize.py) ────────────────────────────────

def load_markdown_as_documents(path: str) -> List[dict]:
    """Legacy helper — kept for backward compatibility with summarize.py."""
    return _parse_markdown_file(path)
