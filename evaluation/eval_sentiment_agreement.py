"""
Evaluation: Tone / Sentiment Agreement

Compares predicted tone from the pipeline against gold labels.
Runs across ALL filings in gold_labels.json and reports aggregate accuracy.

Usage:
    # All filings (default)
    python evaluation/eval_sentiment_agreement.py

    # Single filing
    python evaluation/eval_sentiment_agreement.py HLSR-2024

    # Override endpoint
    APP_BASE_URL=http://localhost:9060 python evaluation/eval_sentiment_agreement.py
"""
from __future__ import annotations

import json
import os
import sys

import requests

# Allow running from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.config import cfg

APP      = os.getenv('APP_BASE_URL', f'http://{cfg.server.host}:{cfg.server.port}')
ENDPOINT = os.getenv('EVAL_ENDPOINT', '/analyze')   # swap to /summarize for legacy

# Load gold labels via cfg
gold_path = os.path.join(os.path.dirname(__file__), '..', cfg.evaluation.gold_labels_path)
gold      = json.load(open(gold_path))

# Single-filing mode vs full-batch mode
target_filings = [sys.argv[1]] if len(sys.argv) > 1 else list(gold.keys())

results = []
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

    pred      = resp.get('tone', 'neutral')
    gold_tone = gold.get(fid, {}).get('tone', 'N/A')
    match     = int(pred == gold_tone)
    results.append({'filing_id': fid, 'pred': pred, 'gold': gold_tone, 'match': match})
    print(f'{fid:<15}  predicted={pred:<10}  gold={gold_tone:<10}  agreement={match}')

if len(results) > 1:
    accuracy = sum(r['match'] for r in results) / len(results)
    print(f'\nOverall tone accuracy: {sum(r["match"] for r in results)}/{len(results)} = {accuracy:.0%}')
