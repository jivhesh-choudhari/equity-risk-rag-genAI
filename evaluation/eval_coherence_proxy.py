"""
Evaluation: Coherence / Output Shape Proxy

Validates that every filing's pipeline output has the correct structure:
  - exactly 2 highlights
  - 1 to 3 risks
  - tone is one of: positive / neutral / cautious
  - financials key is present

Runs across ALL filings and reports pass/fail per filing + overall pass rate.

Usage:
    python evaluation/eval_coherence_proxy.py
    APP_BASE_URL=http://localhost:9060 python evaluation/eval_coherence_proxy.py
"""
from __future__ import annotations

import json
import os
import sys

import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.config import cfg

APP      = os.getenv('APP_BASE_URL', f'http://{cfg.server.host}:{cfg.server.port}')
ENDPOINT = os.getenv('EVAL_ENDPOINT', '/analyze')

gold_path = os.path.join(os.path.dirname(__file__), '..', cfg.evaluation.gold_labels_path)
ALL_FILINGS = list(json.load(open(gold_path)).keys())

VALID_TONES = {'positive', 'neutral', 'cautious'}

passed_count = 0
results = []
for fid in ALL_FILINGS:
    try:
        resp = requests.post(
            f'{APP}{ENDPOINT}',
            json={'filing_id': fid},
            timeout=60,
        ).json()
    except Exception as e:
        print(f'[ERROR] {fid}: {e}')
        results.append({'filing_id': fid, 'passed': False, 'reason': str(e)})
        continue

    hi         = resp.get('highlights', [])
    risks_raw  = resp.get('risks', [])
    tone       = resp.get('tone', '')
    financials = resp.get('financials')

    # risks can be list[str] (/summarize) or list[dict] (/analyze)
    risk_count = len(risks_raw)

    errors = []
    if len(hi) != 2:
        errors.append(f'highlights={len(hi)} (expected 2)')
    if not (1 <= risk_count <= 3):
        errors.append(f'risks={risk_count} (expected 1-3)')
    if tone not in VALID_TONES:
        errors.append(f'tone="{tone}" not in valid set')
    if financials is None:
        errors.append('financials key missing')

    passed = len(errors) == 0
    passed_count += int(passed)
    status = '\u2713' if passed else '\u2717'
    results.append({'filing_id': fid, 'passed': passed, 'errors': errors})
    print(f'{status} {fid:<15}  tone={tone:<10}  h={len(hi)}  r={risk_count}'
          + (f'  ERRORS: {errors}' if errors else ''))

print(f'\nCoherence pass rate: {passed_count}/{len(ALL_FILINGS)}')
