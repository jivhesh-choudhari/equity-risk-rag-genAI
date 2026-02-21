# Equity Filings Summarizer

Agentic pipeline for summarising equity filings (10-K / 10-Q). Uses LangGraph + OpenAI with a full rule-based fallback that works without any API key. Supports both the synthetic Markdown corpus and real PDF filings.

## Quick Start

```bash
pip install -r requirements.txt
uvicorn app.server:app --port 9060 --reload

# rule-based (no key needed)
curl -X POST http://localhost:9060/summarize -H "Content-Type: application/json" -d '{"filing_id":"ACMR-2022"}'

# LLM agentic (requires OPENAI_API_KEY in .env)
curl -X POST http://localhost:9060/analyze -H "Content-Type: application/json" -d '{"filing_id":"HLSR-2024"}'

# real PDF filing
curl -X POST http://localhost:9060/analyze -H "Content-Type: application/json" -d '{"filing_id":"my-10Q","source":"pdf"}'
```

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [How It Works](#how-it-works)
3. [Setup](#setup)
4. [Configuration](#configuration)
5. [Running the Server](#running-the-server)
6. [Using the API](#using-the-api)
7. [Running the Pipeline Directly](#running-the-pipeline-directly)
8. [Evaluation](#evaluation)
9. [Debug Mode](#debug-mode)
10. [MCP Server](#mcp-server)
11. [Corpus](#corpus)

---

## Project Structure

```
.
├── app/
│   ├── agents.py          # LangGraph node functions (6 nodes)
│   ├── chunker.py         # RecursiveCharacterTextSplitter + table isolation
│   ├── config.py          # Typed config singleton (loaded from config.yml + .env)
│   ├── debug.py           # dlog() helper — console trace gated by debug.enabled
│   ├── graph.py           # StateGraph wiring + run_pipeline()
│   ├── loader.py          # MarkdownFilingLoader, PDFFilingLoader, LoaderFactory
│   ├── schema.py          # FilingState TypedDict + Pydantic output models
│   ├── sentiment_risk.py  # Keyword-based sentiment / risk scorer (rule-based)
│   ├── server.py          # FastAPI app — /summarize + /analyze
│   ├── summarize.py       # Rule-based orchestrator (backs /summarize)
│   └── tools.py           # LangChain @tool functions
├── corpus/
│   ├── filings/           # 15 synthetic .md filings (5 tickers × 2022–2024)
│   └── pdfs/              # Drop real 10-K / 10-Q PDFs here
├── evaluation/
│   ├── eval_coherence_proxy.py
│   ├── eval_groundedness.py
│   ├── eval_sentiment_agreement.py
│   └── gold_labels.json
├── notebooks/
│   └── agent.ipynb
├── prompts/
│   ├── risk_prompt.txt
│   ├── sentiment_prompt.txt
│   └── summary_prompt.txt
├── config.yml
├── .env                   # OPENAI_API_KEY (not committed)
└── requirements.txt
```

---

## How It Works

### Two endpoints, same output schema

| Endpoint          | Pipeline                            | Needs API key? |
|-------------------|-------------------------------------|----------------|
| `POST /summarize` | Rule-based (keyword scoring, regex) | No             |
| `POST /analyze`   | LangGraph agentic (LLM + fallback)  | Optional       |

### Agentic pipeline graph

```
START
  |
orchestrator          ← loads filing, builds section map
  |
  +--- sentiment_node  ← tone: positive / neutral / cautious
  +--- risk_node        ← top-3 risk factors with severity
  +--- financial_node   ← revenue, net_income, gross_margin, yoy_change
  |        (parallel fan-out, LangGraph fan-in barrier)
  |
summarizer_node       ← 2 highlight bullets grounded in Business/Results
  |
evaluator_node        ← schema validation + groundedness check (≥0.70)
  |
  +-- pass → END
  +-- fail → summarizer_node (max 2 retries)
```

Every node has an **LLM path** (when `OPENAI_API_KEY` is set and `llm.mock: false`) and a **rule-based fallback** that always works.

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd "Equity Fillings Summarizer"

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure your API key (optional)

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-your-real-key-here
```

If you skip this step the pipeline runs entirely on rule-based fallbacks — no LLM calls are made.

### 4. Review `config.yml`

Open [config.yml](config.yml) and adjust settings if needed (defaults work out of the box):

```yaml
loader:
  source: markdown          # "markdown" for synthetic corpus, "pdf" for real filings

llm:
  mock: false               # set true to force rule-based path even with a key
  model: gpt-4o-mini

debug:
  enabled: false            # set true to print full pipeline trace to console
```

---

## Configuration

All runtime behaviour is controlled by [config.yml](config.yml). Key sections:

| Section      | Key settings                                                           |
|--------------|------------------------------------------------------------------------|
| `loader`     | `source` (`markdown`/`pdf`), `markdown_dir`, `pdf_dir`                 |
| `chunker`    | `chunk_size` (default 1000), `chunk_overlap` (default 150)             |
| `llm`        | `mock`, `model`, `temperature`, `max_section_chars`, `max_table_chars` |
| `graph`      | `max_retries` (default 2)                                              |
| `evaluation` | `groundedness_threshold` (default 0.70)                                |
| `server`     | `host`, `port` (default 9060), `reload`                                |
| `debug`      | `enabled` (default false)                                              |

---

## Running the Server

```bash
uvicorn app.server:app --host 0.0.0.0 --port 9060 --reload
```

Or let the config drive it:

```bash
python -m app.server
```

The server starts at `http://localhost:9060`. Swagger UI is at `http://localhost:9060/docs`.

---

## Using the API

### Rule-based endpoint — `POST /summarize`

Always available, no API key required.

```bash
# PowerShell
Invoke-RestMethod -Method Post -Uri http://localhost:9060/summarize `
  -ContentType "application/json" -Body '{"filing_id":"ACMR-2022"}'

# curl
curl -s -X POST http://localhost:9060/summarize \
  -H "Content-Type: application/json" \
  -d '{"filing_id":"ACMR-2022"}' | python -m json.tool
```

**Response:**

```json
{
  "filing_id": "ACMR-2022",
  "highlights": [
    "ACMR provides enterprise software... (Business)",
    "Revenue grew 15%... (Results)"
  ],
  "risks": [
    "- Cybersecurity incidents could result in penalties. (Risk Factors)"
  ],
  "tone": "neutral",
  "financials": {
    "revenue": null,
    "net_income": null,
    "gross_margin": null,
    "yoy_change": null
  }
}
```

### Agentic endpoint — `POST /analyze`

Uses LangGraph. Falls back to rule-based nodes if no API key is set.

**Request body:**

| Field        | Type   | Default      | Description                              |
|--------------|--------|--------------|------------------------------------------|
| `filing_id`  | string | required     | Filing ID (no extension)                 |
| `source`     | string | `"markdown"` | `"markdown"` for corpus, `"pdf"` for PDFs |

```bash
# markdown filing (synthetic corpus)
Invoke-RestMethod -Method Post -Uri http://localhost:9060/analyze `
  -ContentType "application/json" -Body '{"filing_id":"HLSR-2024"}'

# real PDF (place file in corpus/pdfs/my-10Q.pdf first)
Invoke-RestMethod -Method Post -Uri http://localhost:9060/analyze `
  -ContentType "application/json" -Body '{"filing_id":"my-10Q","source":"pdf"}'

# curl
curl -s -X POST http://localhost:9060/analyze \
  -H "Content-Type: application/json" \
  -d '{"filing_id":"HLSR-2024"}' | python -m json.tool
```

**Response:**

```json
{
  "filing_id": "HLSR-2024",
  "highlights": [
    "HLSR achieved 12% revenue growth driven by cloud migrations. (Results)",
    "Gross margin of 77% reflects strong cost management. (Business)"
  ],
  "risks": [
    {"snippet": "Cybersecurity incidents could result in penalties.", "severity": "high", "section": "Risk Factors"}
  ],
  "tone": "positive",
  "financials": {"revenue": null, "net_income": null, "gross_margin": null, "yoy_change": null, "raw_tables": []},
  "sources": [{"section": "Risk Factors", "agent": "risk"}],
  "eval_result": {
    "valid": true,
    "groundedness": {"score": 0.702, "grounded": true}
  },
  "errors": []
}
```

### Available filing IDs

The synthetic corpus contains 15 filings across 5 tickers, years 2022–2024:

```
ACMR-2022  ACMR-2023  ACMR-2024
HLSR-2022  HLSR-2023  HLSR-2024
LUMO-2022  LUMO-2023  LUMO-2024
NEOV-2022  NEOV-2023  NEOV-2024
ZYNT-2022  ZYNT-2023  ZYNT-2024
```

---

## Running the Pipeline Directly

```bash
# rule-based
python -c "from app.summarize import summarize_filing; import json; print(json.dumps(summarize_filing('ACMR-2022'), indent=2))"

# agentic (markdown)
python -c "from app.graph import run_pipeline; import json; print(json.dumps(run_pipeline('HLSR-2024'), indent=2))"

# agentic (PDF)
python -c "from app.graph import run_pipeline; import json; print(json.dumps(run_pipeline('my-10Q', source='pdf'), indent=2))"
```

Or open [notebooks/agent.ipynb](notebooks/agent.ipynb) for an interactive walkthrough.

---

## Evaluation

Three evaluation scripts run against all 15 filings and print aggregate metrics. The server must be running.

### Sentiment agreement (accuracy vs. gold labels)

```bash
python evaluation/eval_sentiment_agreement.py
```

Hits `/analyze` for all 15 filings, compares predicted tone against `gold_labels.json`, prints per-filing result and overall accuracy.

### Coherence proxy (schema + structure validation)

```bash
python evaluation/eval_coherence_proxy.py
```

Validates that every response has the correct shape: 2 highlights, 1–3 risks, valid tone, financials dict present. Prints pass rate.

### Groundedness (lexical overlap)

```bash
python evaluation/eval_groundedness.py
```

Checks that summary tokens come from the source filing text. Scores each filing 0–1, reports how many exceed the `groundedness_threshold` set in [config.yml](config.yml) (default 0.70).

---

## Debug Mode

Enable step-by-step pipeline tracing to the console. Every stage prints what it received, what path it took (LLM vs. rule-based), and what it produced.

**Option 1 — via `config.yml`** (persists across server restarts):

```yaml
debug:
  enabled: true
```

**Option 2 — at runtime in Python:**

```python
from app.config import cfg
cfg.debug.enabled = True
```

**Sample output:**

```
[DEBUG][summarize] === summarize_filing START ===
         {"filing_id": "ACMR-2022"}
[DEBUG][loader] LoaderFactory -> MarkdownFilingLoader
         {"markdown_dir": "...corpus/filings"}
[DEBUG][loader] Parsed 5 sections from 'ACMR-2022 Annual Filing (Synthetic)'
         {"sections": ["Business", "Results", "Risk Factors", "Liquidity", "Outlook"], "total_chars": 517}
[DEBUG][chunker] chunk_documents called
         {"input_docs": 5, "chunk_size": 1000, "chunk_overlap": 150, "backend": "langchain"}
[DEBUG][tool:score_sentiment] Done
         {"tone": "neutral", "pos": 4, "neg": 5, "unc": 6}
[DEBUG][summarize] === summarize_filing END ===
```

Debug stages: `loader`, `chunker`, `summarize`, `tool:load_filing`, `tool:extract_risks`, `tool:score_sentiment`, `tool:extract_financials`, `tool:validate`, `tool:groundedness`, `orchestrator_node`, `sentiment_node`, `risk_node`, `financial_node`, `summarizer_node`, `evaluator_node`, `graph`.

---

## MCP Server

Exposes three tools over stdio for use with MCP-compatible clients (e.g. Claude Desktop):

```bash
python -m mcp_server.filing_loader_server
```

Available tools:

| Tool           | Description                                    |
|----------------|------------------------------------------------|
| `load_filing`  | Load a filing by ID, returns section documents |
| `get_sections` | Filter documents to specific section names     |
| `chunk_filing` | Load and chunk a filing, returns chunk list    |

---

## Corpus

`corpus/filings/` — 15 synthetic Markdown filings (SEC 10-K style) across 5 tickers and 3 years:

```
ACMR  HLSR  LUMO  NEOV  ZYNT   ×   2022  2023  2024
```

Each filing has sections: Business, Results, Risk Factors, Liquidity, Outlook.

`corpus/pdfs/` — drop real PDF filings here. Filename = filing ID, e.g. `goldman-2024.pdf`.
Then call `/analyze` with `"source": "pdf"` and `"filing_id": "goldman-2024"`.

