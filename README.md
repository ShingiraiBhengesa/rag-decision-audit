# RAG Decision Audit

A diagnostic audit of decision-layer components in state-of-the-art adaptive RAG systems.

## Research question

Do adaptive RAG systems actually learn what they claim to learn? We audit published systems — including Adaptive-RAG, Self-RAG, and CRAG — and find evidence that their decision components (classifiers, reflection tokens, evaluators) rely on dataset-level surface features rather than the semantic properties they're designed to model.

## Status

Work in progress. Paper in preparation.

## Structure

- `experiments/` — controlled diagnostic experiments
- `analysis/` — reproduction scripts and figures
- `results/` — curated experimental results
- `paper/` — LaTeX source

## Reproducibility

Experiments require access to baseline RAG system checkpoints and Wikipedia corpora (not included due to size). See individual experiment READMEs for setup.

## Authors

Shingirai Bhengesa, advised by Dr. Tourani. Saint Louis University.
