#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import pretty_midi

# project imports
from data import TokenNPYDataset
from model import GPT, GPTConfig

# reuse your generate logic
from generate import (
    get_device,
    load_tokenizer_from_tokenized_dir,
    generate_tokens,
    decode_to_midi,
    BOS_TOKEN,
)


# ------------------------
# Utils / Metrics
# ------------------------

@dataclass
class ModelEvalResult:
    split: str
    loss: float
    ppl: float


@dataclass
class GenreCompareResult:
    genre: str
    n_real: int
    n_gen: int
    # feature distances
    pitch_class_kl: float
    density_kl: float
    polyphony_kl: float
    duration_kl: float
    pitch_range_diff: float
    note_count_diff: float


def safe_hist(x: np.ndarray, bins: int, range_: Tuple[float, float]) -> np.ndarray:
    h, _ = np.histogram(x, bins=bins, range=range_, density=False)
    h = h.astype(np.float64)
    h = h / max(h.sum(), 1.0)
    return h


def kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def midi_features(pm: pretty_midi.PrettyMIDI) -> Dict[str, float]:
    # Flatten all instruments into note list
    notes = []
    for inst in pm.instruments:
        notes.extend(inst.notes)

    if len(notes) == 0:
        return {
            "note_count": 0,
            "pitch_range": 0,
            "note_density": 0.0,
            "mean_polyphony": 0.0,
            "mean_duration": 0.0,
        }

    pitches = np.array([n.pitch for n in notes], dtype=np.float64)
    durations = np.array([max(0.0, n.end - n.start) for n in notes], dtype=np.float64)

    # duration of piece
    end_time = pm.get_end_time()
    end_time = float(max(end_time, 1e-6))

    # density (notes/sec)
    density = float(len(notes) / end_time)

    # polyphony estimate: average active notes over time grid
    # (cheap + stable)
    grid_hz = 10.0
    t = np.linspace(0, end_time, int(end_time * grid_hz) + 1)
    active = np.zeros_like(t)
    for n in notes:
        s = int(min(len(t) - 1, max(0, math.floor(n.start * grid_hz))))
        e = int(min(len(t) - 1, max(0, math.ceil(n.end * grid_hz))))
        if e > s:
            active[s:e] += 1
    mean_poly = float(active.mean())

    return {
        "note_count": float(len(notes)),
        "pitch_range": float(pitches.max() - pitches.min()),
        "note_density": density,
        "mean_polyphony": mean_poly,
        "mean_duration": float(durations.mean()),
        # keep raw for hist building
        "_pitches": pitches,
        "_durations": durations,
    }


def aggregate_feature_hists(mid_paths: List[Path]) -> Dict[str, np.ndarray]:
    pcs = []
    dens = []
    poly = []
    dur = []
    prng = []
    ncnt = []

    for p in mid_paths:
        try:
            pm = pretty_midi.PrettyMIDI(str(p))
            f = midi_features(pm)

            if f["note_count"] <= 0:
                continue

            # pitch class histogram (0..11) from pitches
            pitches = f["_pitches"]
            pcs.extend(list((pitches.astype(int) % 12)))

            dens.append(f["note_density"])
            poly.append(f["mean_polyphony"])
            dur.extend(list(f["_durations"]))
            prng.append(f["pitch_range"])
            ncnt.append(f["note_count"])
        except Exception:
            continue

    pcs = np.array(pcs, dtype=np.float64) if len(pcs) else np.array([], dtype=np.float64)
    dens = np.array(dens, dtype=np.float64) if len(dens) else np.array([], dtype=np.float64)
    poly = np.array(poly, dtype=np.float64) if len(poly) else np.array([], dtype=np.float64)
    dur = np.array(dur, dtype=np.float64) if len(dur) else np.array([], dtype=np.float64)
    prng = np.array(prng, dtype=np.float64) if len(prng) else np.array([], dtype=np.float64)
    ncnt = np.array(ncnt, dtype=np.float64) if len(ncnt) else np.array([], dtype=np.float64)

    # build normalized hists
    out = {
        "pitch_class": safe_hist(pcs, bins=12, range_=(0, 12)),
        "density": safe_hist(dens, bins=30, range_=(0, 15)),       # tune ranges if needed
        "polyphony": safe_hist(poly, bins=30, range_=(0, 10)),
        "duration": safe_hist(dur, bins=40, range_=(0, 2.0)),
        "pitch_range_mean": np.array([float(prng.mean()) if prng.size else 0.0]),
        "note_count_mean": np.array([float(ncnt.mean()) if ncnt.size else 0.0]),
        "n_valid": np.array([float(len(dens))]),
    }
    return out


@torch.no_grad()
def eval_loss(model: GPT, loader: DataLoader, device: torch.device, autocast_ctx) -> float:
    model.eval()
    losses = []
    for batch in tqdm(loader, desc="eval", leave=False):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with autocast_ctx:
            _, loss = model(input_ids, labels)
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses)) if losses else float("inf")


def list_real_midis_by_genre(index_parquet: str, split_json: str, genre: str, split: str) -> List[Path]:
    df = pd.read_parquet(index_parquet)
    df["genre_final"] = df["genre_final"].fillna("other").astype(str)

    splits = json.loads(Path(split_json).read_text())["paths"]
    allowed = set(splits[split])

    sub = df[(df["genre_final"] == genre) & (df["midi_path"].astype(str).isin(allowed))]
    return [Path(p) for p in sub["midi_path"].astype(str).tolist() if Path(p).exists()]


def load_model(ckpt_path: str, device: torch.device) -> GPT:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = ckpt["config"]
    cfg = GPTConfig(**cfg_dict)
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--tokenized_dir", type=str, default="data/tokenized")
    ap.add_argument("--manifest", type=str, default="data/tokenized/manifest.parquet")

    ap.add_argument("--index", type=str, default="data/index.parquet")
    ap.add_argument("--splits", type=str, default="data/splits.json")

    ap.add_argument("--out_dir", type=str, default="eval")
    ap.add_argument("--genres", type=str, default="data/tokenized/genres.json")

    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--samples_per_genre", type=int, default=25)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)

    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "generated").mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print("Device:", device)

    # Load tokenizer artifacts (for decoding)
    tokenizer, special_tokens, special_tok2id, offset, genres_map = load_tokenizer_from_tokenized_dir(Path(args.tokenized_dir))

    # Load model
    model = load_model(args.ckpt, device=device)
    cfg = model.cfg

    # AMP autocast context for eval on CUDA
    if device.type == "cuda":
        autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = torch.autocast("cpu", enabled=False)

    # ------------------------
    # (1) Model metrics: loss/ppl on val + test
    # ------------------------
    results_model: List[ModelEvalResult] = []

    for split in ["val", "test"]:
        ds = TokenNPYDataset(args.manifest, split=split, seq_len=args.seq_len, random_crop=False, seed=args.seed)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )
        loss = eval_loss(model, loader, device, autocast_ctx)
        ppl = float(math.exp(min(20.0, loss)))  # cap for numeric sanity
        results_model.append(ModelEvalResult(split=split, loss=loss, ppl=ppl))
        print(f"[{split}] loss={loss:.4f} ppl={ppl:.2f}")

    # ------------------------
    # (2) Generate samples per genre
    # ------------------------
    genres = json.loads(Path(args.genres).read_text())
    gen_paths_by_genre: Dict[str, List[Path]] = {}

    for genre in genres:
        if genre not in genres_map:
            continue

        gdir = out_dir / "generated" / genre
        gdir.mkdir(parents=True, exist_ok=True)

        genre_tok = f"<GENRE={genre}>"
        genre_id = int(special_tok2id[genre_tok])
        bos_id = int(special_tok2id[BOS_TOKEN])

        prompt_tok_space = [genre_id, bos_id]
        prompt_model_space = [x + 1 for x in prompt_tok_space]  # dataset shift

        print(f"Generating {args.samples_per_genre} samples for genre={genre} ...")
        paths = []
        for i in tqdm(range(args.samples_per_genre), desc=f"gen:{genre}", leave=False):
            ids_model = generate_tokens(
                model=model,
                prompt_ids=prompt_model_space,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                max_id=cfg.vocab_size - 1,
            )
            out_mid = gdir / f"sample_{i:03d}.mid"
            try:
                decode_to_midi(tokenizer, ids_model, offset=offset, out_mid_path=out_mid)
                paths.append(out_mid)
            except Exception:
                # If a decode fails, skip; you can log these if needed
                continue

        gen_paths_by_genre[genre] = paths
        print(f"  -> decoded {len(paths)}/{args.samples_per_genre}")

    # ------------------------
    # (3) Compare REAL vs GENERATED per genre (distribution distances)
    # ------------------------
    rows: List[GenreCompareResult] = []
    for genre in genres:
        gen_paths = gen_paths_by_genre.get(genre, [])
        if len(gen_paths) == 0:
            continue

        # choose real midis from same genre and split="test" (so comparisons are fair-ish)
        real_paths = list_real_midis_by_genre(args.index, args.splits, genre=genre, split="test")
        if len(real_paths) == 0:
            continue

        # optional: cap to match generated count
        real_paths = real_paths[: len(gen_paths)]

        real_h = aggregate_feature_hists(real_paths)
        gen_h = aggregate_feature_hists(gen_paths)

        # compute KL(real || gen)
        pitch_class_kl = kl_div(real_h["pitch_class"], gen_h["pitch_class"])
        density_kl = kl_div(real_h["density"], gen_h["density"])
        polyphony_kl = kl_div(real_h["polyphony"], gen_h["polyphony"])
        duration_kl = kl_div(real_h["duration"], gen_h["duration"])

        pitch_range_diff = float(real_h["pitch_range_mean"][0] - gen_h["pitch_range_mean"][0])
        note_count_diff = float(real_h["note_count_mean"][0] - gen_h["note_count_mean"][0])

        rows.append(
            GenreCompareResult(
                genre=genre,
                n_real=len(real_paths),
                n_gen=len(gen_paths),
                pitch_class_kl=pitch_class_kl,
                density_kl=density_kl,
                polyphony_kl=polyphony_kl,
                duration_kl=duration_kl,
                pitch_range_diff=pitch_range_diff,
                note_count_diff=note_count_diff,
            )
        )

    df_cmp = pd.DataFrame([asdict(r) for r in rows]).sort_values("genre")
    df_cmp.to_csv(out_dir / "genre_compare.csv", index=False)

    # Save model metrics too
    df_model = pd.DataFrame([asdict(r) for r in results_model])
    df_model.to_csv(out_dir / "model_metrics.csv", index=False)

    # Save summary JSON
    summary = {
        "ckpt": args.ckpt,
        "tokenized_dir": args.tokenized_dir,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "samples_per_genre": args.samples_per_genre,
        "model_metrics": [asdict(r) for r in results_model],
        "genre_compare_path": str((out_dir / "genre_compare.csv").resolve()),
        "model_metrics_path": str((out_dir / "model_metrics.csv").resolve()),
        "generated_dir": str((out_dir / "generated").resolve()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\nSaved:")
    print(" -", (out_dir / "model_metrics.csv").resolve())
    print(" -", (out_dir / "genre_compare.csv").resolve())
    print(" -", (out_dir / "summary.json").resolve())
    print(" -", (out_dir / "generated").resolve())


if __name__ == "__main__":
    main()
