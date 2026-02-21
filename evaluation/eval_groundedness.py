"""
Evaluation: Groundedness Proxy

Measures what fraction of summary tokens (highlights + risk snippets)
appear in the original source filing text.

Threshold is read from config.yml (evaluation.groundedness_threshold).
Runs across ALL filings and reports per-filing scores + overall average.

Usage:
    python evaluation/eval_groundedness.py
    python evaluation/eval_groundedness.py HLSR-2024
    APP_BASE_URL=http://localhost:9060 python evaluation/eval_groundedness.py
"""
from __future__ import annotations

import json
import os
import re
import sys

import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.config import cfg

APP       = os.getenv('APP_BASE_URL', f'http://{cfg.server.host}:{cfg.server.port}')
ENDPOINT  = os.getenv('EVAL_ENDPOINT', '/analyze')
THRESHOLD = cfg.evaluation.groundedness_threshold

gold_path   = os.path.join(os.path.dirname(__file__), '..', cfg.evaluation.gold_labels_path)
ALL_FILINGS = list(json.load(open(gold_path)).keys())

target_filings = [sys.argv[1]] if len(sys.argv) > 1 else ALL_FILINGS

# Resolve corpus root relative to this file
CORPUS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

scores = []
for fid in target_filings:
    try:
        resp = requests.post(
            f'{APP}{ENDPOINT}',
            json={'filing_id': fid},
            timeout=60,
        ).json()
    except Exception as e:
        print(f'[ERROR] {fid}: {e}')
        continue

    highlights = resp.get('highlights', [])
    risks_raw  = resp.get('risks', [])

    # risks can be list[str] (/summarize) or list[dict with 'snippet'] (/analyze)
    risk_texts = [
        r['snippet'] if isinstance(r, dict) else r
        for r in risks_raw
    ]
    summary_text = ' '.join(highlights + risk_texts)

    # Try markdown source first, then PDF
    source_text = ''
    for subdir in (cfg.loader.markdown_dir, cfg.loader.pdf_dir):
        for ext in ('.md', '.pdf'):
            candidate = os.path.join(CORPUS_ROOT, subdir, f'{fid}{ext}')
            if os.path.exists(candidate) and ext == '.md':
                with open(candidate, 'r', encoding='utf-8') as f:
                    source_text = f.read()
                break
        if source_text:
            break

    if not source_text:
        print(f'[SKIP] {fid}: source file not found')
        continue

    tokens = re.findall(r'\w+', summary_text.lower())
    if not tokens:
        print(f'[SKIP] {fid}: empty summary')
        continue

    hits  = sum(1 for t in tokens if t in source_text.lower())
    score = hits / len(tokens)
    passed = score >= THRESHOLD
    scores.append(score)
    status = '\u2713' if passed else '\u2717'
    print(f'{status} {fid:<15}  groundedness={score:.2f}  '
          f'({hits}/{len(tokens)} tokens)  threshold={THRESHOLD}')

if scores:
    avg = sum(scores) / len(scores)
    passed_count = sum(1 for s in scores if s >= THRESHOLD)
    print(f'\nGroundedness: avg={avg:.2f}  '
          f'passed={passed_count}/{len(scores)} (threshold={THRESHOLD})')
