# Financial Research Analyst Agent

This README is the Starter Kit Summary.

## What’s Inside
- corpus/filings — 15 synthetic filings (5 companies × 2022–2024)
- prompts/summary_prompt.txt
- app/ (loader, chunker, sentiment_risk, summarize, server)
- evaluation/ scripts and gold_labels.json

## Quick Start
pip install -r requirements.txt
uvicorn app.server:app --host 0.0.0.0 --port 9060 --reload

## Evaluate
python evaluation/eval_sentiment_agreement.py HLSR-2024
python evaluation/eval_coherence_proxy.py
python evaluation/eval_groundedness.py
