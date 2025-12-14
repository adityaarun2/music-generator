#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=str, default="artifacts/index.parquet")
    ap.add_argument("--out", type=str, default="artifacts/splits.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--test_frac", type=float, default=0.05)
    ap.add_argument("--label_col", type=str, default="genre_final")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    df = pd.read_parquet(args.index).copy()
    if args.label_col not in df.columns:
        raise ValueError(f"index must contain '{args.label_col}' column")

    # Ensure no missing labels
    df[args.label_col] = df[args.label_col].fillna("other")

    train_paths, val_paths, test_paths = [], [], []

    # Stratified split by label_col
    for label, gdf in df.groupby(args.label_col):
        idx = gdf.index.to_numpy()
        rng.shuffle(idx)

        n = len(idx)
        n_test = int(round(n * args.test_frac))
        n_val = int(round(n * args.val_frac))

        # clamp for very small groups (shouldn't happen much now)
        n_test = min(n_test, n)
        n_val = min(n_val, max(0, n - n_test))

        test_idx = idx[:n_test]
        val_idx = idx[n_test:n_test + n_val]
        train_idx = idx[n_test + n_val:]

        test_paths.extend(df.loc[test_idx, "midi_path"].tolist())
        val_paths.extend(df.loc[val_idx, "midi_path"].tolist())
        train_paths.extend(df.loc[train_idx, "midi_path"].tolist())

    out = {
        "seed": args.seed,
        "index_path": str(Path(args.index).resolve()),
        "label_col": args.label_col,
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "n_total": int(len(df)),
        "n_train": int(len(train_paths)),
        "n_val": int(len(val_paths)),
        "n_test": int(len(test_paths)),
        "paths": {
            "train": train_paths,
            "val": val_paths,
            "test": test_paths,
        },
        "label_counts": df[args.label_col].value_counts().to_dict(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print(f"Saved splits: {out_path.resolve()}")
    print(f"Total: {out['n_total']}, Train: {out['n_train']}, Val: {out['n_val']}, Test: {out['n_test']}")
    print("\nLabel distribution:")
    for k, v in sorted(out["label_counts"].items(), key=lambda kv: kv[1], reverse=True):
        print(f"{k:12s} {v}")


if __name__ == "__main__":
    main()
