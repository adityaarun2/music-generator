#!/usr/bin/env bash
set -euo pipefail

CKPT=${1:-runs/l4_full/ckpt_best.pt}
RUN_NAME=${2:-l4_full_best}

python src/evaluate.py \
  --ckpt "$CKPT" \
  --tokenized_dir data/tokenized \
  --manifest data/tokenized/manifest.parquet \
  --index data/index.parquet \
  --splits data/splits.json \
  --out_dir "eval/$RUN_NAME" \
  --samples_per_genre 25 \
  --max_new_tokens 2048 \
  --temperature 1.0 \
  --top_p 0.9

python src/genre_classifier_eval.py \
  --index data/index.parquet \
  --splits data/splits.json \
  --generated_dir "eval/$RUN_NAME/generated" \
  --out_dir "eval/$RUN_NAME/genre_classifier"
