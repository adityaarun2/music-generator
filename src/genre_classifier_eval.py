#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import pretty_midi

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# -----------------------------
# Feature extraction (simple)
# -----------------------------

def safe_pretty_midi_load(path: Path) -> Optional[pretty_midi.PrettyMIDI]:
    try:
        return pretty_midi.PrettyMIDI(str(path))
    except Exception:
        return None


def extract_features_from_pm(pm: pretty_midi.PrettyMIDI) -> Optional[np.ndarray]:
    """
    Very simple, stable features for genre classification.
    Returns feature vector shape (D,) or None if unusable.
    """
    notes = []
    for inst in pm.instruments:
        # ignore drums? (optional)
        # if inst.is_drum: continue
        notes.extend(inst.notes)

    if len(notes) < 10:
        return None

    end_time = float(max(pm.get_end_time(), 1e-6))

    pitches = np.array([n.pitch for n in notes], dtype=np.float64)
    durs = np.array([max(0.0, n.end - n.start) for n in notes], dtype=np.float64)

    # pitch class histogram (12 bins)
    pcs = (pitches.astype(int) % 12)
    pc_hist, _ = np.histogram(pcs, bins=12, range=(0, 12))
    pc_hist = pc_hist.astype(np.float64)
    pc_hist = pc_hist / max(pc_hist.sum(), 1.0)

    # note density (notes/sec)
    density = len(notes) / end_time

    # mean + std pitch
    pitch_mean = float(pitches.mean())
    pitch_std = float(pitches.std())

    # pitch range
    pitch_range = float(pitches.max() - pitches.min())

    # duration stats
    dur_mean = float(durs.mean())
    dur_std = float(durs.std())

    # polyphony estimate on a small grid
    grid_hz = 10.0
    t = np.linspace(0, end_time, int(end_time * grid_hz) + 1)
    active = np.zeros_like(t)
    for n in notes:
        s = int(np.clip(np.floor(n.start * grid_hz), 0, len(t) - 1))
        e = int(np.clip(np.ceil(n.end * grid_hz), 0, len(t) - 1))
        if e > s:
            active[s:e] += 1
    poly_mean = float(active.mean())
    poly_std = float(active.std())

    # tempo summary if present (pretty_midi may be noisy but ok)
    try:
        tempi, _ = pm.get_tempo_changes()
        tempo_mean = float(np.mean(tempi)) if len(tempi) else 120.0
        tempo_std = float(np.std(tempi)) if len(tempi) else 0.0
    except Exception:
        tempo_mean, tempo_std = 120.0, 0.0

    # build feature vector
    feats = np.concatenate([
        pc_hist,  # 12
        np.array([
            density,
            pitch_mean, pitch_std, pitch_range,
            dur_mean, dur_std,
            poly_mean, poly_std,
            tempo_mean, tempo_std,
            end_time,
            float(len(notes)),
        ], dtype=np.float64)
    ], axis=0)

    # guard against NaNs/infs
    if not np.all(np.isfinite(feats)):
        return None
    return feats.astype(np.float32)


def extract_features_from_midi(path: Path) -> Optional[np.ndarray]:
    pm = safe_pretty_midi_load(path)
    if pm is None:
        return None
    return extract_features_from_pm(pm)


# -----------------------------
# Data loading helpers
# -----------------------------

def load_splits(splits_path: str) -> Dict[str, set]:
    splits = json.loads(Path(splits_path).read_text())["paths"]
    return {
        "train": set(splits.get("train", [])),
        "val": set(splits.get("val", [])),
        "test": set(splits.get("test", [])),
    }


def build_real_dataset(
    index_path: str,
    splits_path: str,
    use_split_train: str = "train",
    use_split_test: str = "test",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Returns: X_train, y_train, X_test, y_test, classes (sorted unique genres)
    """
    df = pd.read_parquet(index_path).copy()
    if "genre_final" not in df.columns or "midi_path" not in df.columns:
        raise ValueError("index.parquet must contain 'midi_path' and 'genre_final'")

    df["genre_final"] = df["genre_final"].fillna("other").astype(str)
    df["midi_path"] = df["midi_path"].astype(str)

    splits = load_splits(splits_path)
    train_paths = splits[use_split_train]
    test_paths = splits[use_split_test]

    df_train = df[df["midi_path"].isin(train_paths)].reset_index(drop=True)
    df_test = df[df["midi_path"].isin(test_paths)].reset_index(drop=True)

    classes = sorted(df["genre_final"].unique().tolist())
    genre2id = {g: i for i, g in enumerate(classes)}

    # --- build X/y (aligned!) ---
    def build_xy(sub: pd.DataFrame, split_name: str) -> Tuple[np.ndarray, np.ndarray]:
        X_list: List[np.ndarray] = []
        y_list: List[int] = []
        skipped = 0

        for r in sub.itertuples(index=False):
            p = Path(getattr(r, "midi_path"))
            g = getattr(r, "genre_final")

            if not p.exists():
                skipped += 1
                continue

            feats = extract_features_from_midi(p)
            if feats is None:
                skipped += 1
                continue

            X_list.append(feats)
            y_list.append(genre2id[g])  # only append label when feats exists

        if len(X_list) == 0:
            raise RuntimeError(f"No valid MIDI samples after filtering for {split_name}")

        X = np.stack(X_list, axis=0)
        y = np.asarray(y_list, dtype=np.int64)

        print(f"{split_name} valid: {len(X)} | skipped: {skipped}")
        return X, y

    X_train, y_train = build_xy(df_train, "Train")
    X_test, y_test = build_xy(df_test, "Test")

    return X_train, y_train, X_test, y_test, classes


def iter_generated_midis(generated_dir: Path) -> List[Tuple[str, Path]]:
    """
    generated_dir/
      jazz/*.mid
      rock/*.mid
    Returns list of (conditioning_genre, midi_path)
    """
    out: List[Tuple[str, Path]] = []
    if not generated_dir.exists():
        return out

    for genre_dir in sorted([p for p in generated_dir.iterdir() if p.is_dir()]):
        genre = genre_dir.name
        for mid in sorted(genre_dir.glob("*.mid")):
            out.append((genre, mid))
    return out


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=str, required=True)
    ap.add_argument("--splits", type=str, required=True)
    ap.add_argument("--generated_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--model", type=str, default="logreg", choices=["logreg", "rf"])
    ap.add_argument("--max_generated_per_genre", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    print("Loading real MIDI data...")
    X_train, y_train, X_test, y_test, classes = build_real_dataset(args.index, args.splits)

    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    print("Class distribution (train):", dict(Counter([classes[i] for i in y_train]).most_common(10)))

    # Choose a simple sklearn classifier
    if args.model == "logreg":
        clf = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(
                max_iter=2000,
                n_jobs=-1,
                multi_class="auto",
                class_weight="balanced",
            )),
        ])
    else:
        clf = Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=300,
                random_state=args.seed,
                n_jobs=-1,
                class_weight="balanced_subsample",
            )),
        ])

    clf.fit(X_train, y_train)

    # Real test accuracy
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[REAL TEST] accuracy = {acc:.4f}")

    report = classification_report(
        y_test, y_pred,
        target_names=classes,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred)

    (out_dir / "real_test_classification_report.txt").write_text(report)
    np.save(out_dir / "real_test_confusion_matrix.npy", cm)

    # Also save a CSV confusion matrix
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.to_csv(out_dir / "real_test_confusion_matrix.csv", index=True)

    # Genre consistency on generated
    print("\nLoading generated MIDIs...")
    gen_root = Path(args.generated_dir)
    gen_items = iter_generated_midis(gen_root)

    if len(gen_items) == 0:
        print("No generated MIDIs found. Done.")
        return

    # cap per genre (so it runs fast)
    by_genre: Dict[str, List[Path]] = defaultdict(list)
    for g, p in gen_items:
        by_genre[g].append(p)

    rows = []
    total_valid = 0
    total_correct = 0

    for g, paths in sorted(by_genre.items()):
        # cap
        paths = paths[: args.max_generated_per_genre]

        Xg_list = []
        yg_true = []

        skipped = 0
        for p in paths:
            feats = extract_features_from_midi(p)
            if feats is None:
                skipped += 1
                continue
            Xg_list.append(feats)
            yg_true.append(g)

        if len(Xg_list) == 0:
            rows.append({"genre": g, "n": 0, "skipped": skipped, "consistency": 0.0})
            continue

        Xg = np.stack(Xg_list, axis=0)
        pred_ids = clf.predict(Xg)
        pred_genres = [classes[i] for i in pred_ids]

        correct = sum(1 for pg in pred_genres if pg == g)
        n = len(pred_genres)

        total_valid += n
        total_correct += correct

        rows.append({
            "genre": g,
            "n": n,
            "skipped": skipped,
            "consistency": correct / max(1, n),
        })

        print(f"[GEN] {g:12s} consistency={correct/max(1,n):.3f} (n={n}, skipped={skipped})")

    overall = total_correct / max(1, total_valid)
    print(f"\n[GEN OVERALL] genre consistency = {overall:.4f} (valid={total_valid})")

    df = pd.DataFrame(rows).sort_values("genre")
    df.to_csv(out_dir / "generated_genre_consistency.csv", index=False)

    summary = {
        "real_test_accuracy": float(acc),
        "generated_genre_consistency_overall": float(overall),
        "n_generated_valid": int(total_valid),
        "classifier": args.model,
        "outputs": {
            "real_report": str((out_dir / "real_test_classification_report.txt").resolve()),
            "real_confusion_csv": str((out_dir / "real_test_confusion_matrix.csv").resolve()),
            "generated_consistency_csv": str((out_dir / "generated_genre_consistency.csv").resolve()),
        },
        "classes": classes,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\nSaved:")
    print(" -", (out_dir / "real_test_classification_report.txt").resolve())
    print(" -", (out_dir / "real_test_confusion_matrix.csv").resolve())
    print(" -", (out_dir / "generated_genre_consistency.csv").resolve())
    print(" -", (out_dir / "summary.json").resolve())


if __name__ == "__main__":
    main()
