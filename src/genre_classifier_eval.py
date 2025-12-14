#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pretty_midi

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ------------------------
# Feature extraction
# ------------------------

def midi_features(pm: pretty_midi.PrettyMIDI) -> np.ndarray:
    notes = []
    for inst in pm.instruments:
        notes.extend(inst.notes)

    if len(notes) == 0:
        return np.zeros(18, dtype=np.float32)

    pitches = np.array([n.pitch for n in notes])
    durations = np.array([max(0.0, n.end - n.start) for n in notes])

    end_time = max(pm.get_end_time(), 1e-6)
    density = len(notes) / end_time

    # polyphony estimate
    grid_hz = 10
    t = np.linspace(0, end_time, int(end_time * grid_hz) + 1)
    active = np.zeros_like(t)
    for n in notes:
        s = int(min(len(t) - 1, n.start * grid_hz))
        e = int(min(len(t) - 1, n.end * grid_hz))
        if e > s:
            active[s:e] += 1
    mean_polyphony = active.mean()

    # pitch class histogram
    pc_hist = np.zeros(12)
    for p in pitches:
        pc_hist[p % 12] += 1
    pc_hist /= max(pc_hist.sum(), 1)

    pitch_range = pitches.max() - pitches.min()
    mean_duration = durations.mean()

    feats = np.concatenate([
        pc_hist,
        np.array([
            density,
            mean_duration,
            mean_polyphony,
            pitch_range,
            len(notes),
        ])
    ])

    return feats.astype(np.float32)


def extract_dataset_features(midi_paths: List[Path]) -> np.ndarray:
    feats = []
    for p in midi_paths:
        try:
            pm = pretty_midi.PrettyMIDI(str(p))
            feats.append(midi_features(pm))
        except Exception:
            continue
    return np.stack(feats) if feats else np.empty((0, 18))


# ------------------------
# Dataset helpers
# ------------------------

def load_real_split(index_path: str, splits_path: str, split: str):
    df = pd.read_parquet(index_path)
    df["genre_final"] = df["genre_final"].fillna("other").astype(str)

    splits = json.loads(Path(splits_path).read_text())["paths"]
    allowed = set(splits[split])

    df = df[df["midi_path"].astype(str).isin(allowed)]
    paths = [Path(p) for p in df["midi_path"].astype(str)]
    labels = df["genre_final"].tolist()
    return paths, labels


def load_generated_by_genre(gen_dir: Path) -> Dict[str, List[Path]]:
    out = {}
    for gdir in gen_dir.iterdir():
        if gdir.is_dir():
            mids = list(gdir.glob("*.mid"))
            if mids:
                out[gdir.name] = mids
    return out


# ------------------------
# Main
# ------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--index", type=str, default="data/index.parquet")
    ap.add_argument("--splits", type=str, default="data/splits.json")
    ap.add_argument("--generated_dir", type=str, required=True)

    ap.add_argument("--out_dir", type=str, default="eval/genre_classifier")
    ap.add_argument("--n_estimators", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------
    # Load real data
    # ------------------------
    print("Loading real MIDI data...")
    train_paths, train_labels = load_real_split(args.index, args.splits, "train")
    test_paths, test_labels = load_real_split(args.index, args.splits, "test")

    X_train = extract_dataset_features(train_paths)
    y_train = np.array(train_labels)

    X_test = extract_dataset_features(test_paths)
    y_test = np.array(test_labels)

    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # ------------------------
    # Train classifier
    # ------------------------
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=args.n_estimators,
            random_state=args.seed,
            n_jobs=-1,
        ))
    ])

    clf.fit(X_train, y_train)

    # ------------------------
    # Evaluate on real test
    # ------------------------
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\nReal test accuracy:", acc)
    print(classification_report(y_test, y_pred))

    (out_dir / "real_test_report.txt").write_text(
        classification_report(y_test, y_pred)
    )

    # ------------------------
    # Evaluate on generated
    # ------------------------
    print("\nEvaluating generated samples...")
    gen_by_genre = load_generated_by_genre(Path(args.generated_dir))

    rows = []
    for genre, paths in gen_by_genre.items():
        Xg = extract_dataset_features(paths)
        if len(Xg) == 0:
            continue

        yhat = clf.predict(Xg)
        consistency = float(np.mean(yhat == genre))

        rows.append({
            "genre": genre,
            "n_generated": len(Xg),
            "predicted_genre_accuracy": consistency,
        })

        print(f"{genre:10s} | consistency: {consistency:.3f}")

    df_gen = pd.DataFrame(rows).sort_values("genre")
    df_gen.to_csv(out_dir / "generated_genre_consistency.csv", index=False)

    print("\nSaved:")
    print(" -", (out_dir / "real_test_report.txt").resolve())
    print(" -", (out_dir / "generated_genre_consistency.csv").resolve())


if __name__ == "__main__":
    main()
