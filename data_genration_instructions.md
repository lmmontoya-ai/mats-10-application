# Data Generation Instructions

This document explains how to run the dataset generator, what it produces,
and how to verify data quality against the project requirements.

## Overview
- Goal: build a controlled long-context "needle-in-a-haystack" dataset for
  prompt-injection / instruction-override monitoring.
- Inputs: configuration in `configs/dataset_main.yaml` plus filler text in
  `data_raw/filler_segments.jsonl`.
- Outputs: JSONL datasets, split manifest CSV, QA reports, and a run record.
- Deterministic: same config + seeds => identical outputs (byte-for-byte when
  timestamps are kept out of dataset files).

## Prerequisites
- Create a Python environment and install dependencies:
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```
- Ensure the target tokenizer is available.
  - Default is `google/gemma-3-4b-it` in `configs/dataset_main.yaml`.
  - If the model is not cached, you need network access and (for Gemma)
    an authenticated HF token with license access.

## Required input data
The generator expects a local filler corpus:
- File: `data_raw/filler_segments.jsonl`
- Format: one JSON object per line, at minimum `id` and `text`:
```json
{"id": "seg_000001", "text": "Your paragraph text...", "source": "wikipedia"}
```

LLM fallback filler is disabled by default for reproducibility. If you enable
it (not recommended), set `filler.allow_llm_fallback: true` and optionally
record a prompt in `filler.llm_fallback_prompt` in the config.

## Run the generator
```bash
python scripts/generate_data.py --config configs/dataset_main.yaml
```

The script will:
- Load config and validate it.
- Load the tokenizer and template families.
- Assemble documents to match token-length buckets within tolerance.
- Assign splits by template family (train/val/test disjoint for needles).
- Write datasets and manifests.
- Run QA checks and write reports.
- Write a run record to `results/run_records/`.

If QA fails, the script raises a runtime error (after writing reports).

## Outputs
- Dataset JSONL:
  - Combined: `data/datasets/{dataset_name}_{dataset_version}.jsonl`
  - Per split: `data/datasets/{dataset_name}_{dataset_version}_{split}.jsonl`
- Split manifest:
  - `data/splits/{dataset_name}_{dataset_version}_manifest.csv`
- QA reports:
  - JSON: `data/qa/{dataset_name}_{dataset_version}_qa.json`
  - Text summary: `data/qa/{dataset_name}_{dataset_version}_qa.txt`
  - Spot-check samples: `data/qa/{dataset_name}_{dataset_version}_spot_check.txt`
- Run record:
  - `results/run_records/{dataset_name}_{dataset_version}_run_record.json`

## Dataset schema (JSONL)
Each example includes (not exhaustive):
- `id`, `split`, `length_bucket`, `length_tokens`
- `distractor_level`, `needle_present`, `needle_family`, `needle_text`
- `distractor_families`
- `text`
- `needle_char_span`, `needle_token_indices`
- `seed`, `meta`

`needle_token_indices` is sparse and maps directly to the needle span
using tokenizer offset mappings when available.

## QA checks (automatic)
The QA suite runs on every generation and writes a JSON + text summary.
Key checks:
- Family leakage: no needle family overlap between train/val/test.
- Length tolerance: token lengths within configured tolerance.
- Span integrity: needle spans and token indices are valid and aligned.
- Trivial baselines: keyword and regex baselines to detect trivial leakage.
- Spot-check: sample positives/negatives for manual inspection.

If the tokenizer lacks offset mappings, the QA report warns that fallback
span mapping was used.

## Quality checklist
- `data/qa/*_qa.txt` shows all checks PASSED.
- `data/qa/*_qa.json` has zero length/span violations.
- `data/qa/*_spot_check.txt` looks reasonable for 10 positives and 10 negatives.
- The trivial baseline AUROC is not near-perfect (otherwise templates may be
  too easy).
- Run record exists and matches the config, especially model/tokenizer IDs.

## Troubleshooting
- Missing filler file:
  - Create `data_raw/filler_segments.jsonl` with a few thousand paragraphs.
  - Or explicitly enable fallback (not recommended).
- Tokenizer download issues:
  - Confirm network access and HF token permissions.
  - Optionally set `tokenizer_id` and `tokenizer_revision` in the config to pin.
- QA failures:
  - Check `data/qa/*_qa.json` for the exact violation.
  - Typical fixes: adjust templates, increase filler diversity, or relax
    tolerance slightly (keep it small, e.g. 0.02).

## Adjusting dataset size
Total examples per split are:
- `n_split * len(length_buckets_tokens) * len(distractor_levels)`
For quick smoke tests, reduce `scale.n_train/n_val/n_test` or reduce the grid.

## Determinism
- Seeds are set in `randomness` in the config.
- With the same config and seeds, dataset outputs are reproducible.
- Timestamps are stored only in the run record, not in the JSONL dataset files.
