#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from miditok import REMI, TokenizerConfig


BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sha1_name(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def build_genre_tokens(genres: List[str]) -> List[str]:
    genres_sorted = sorted(set(genres))
    return [f"<GENRE={g}>" for g in genres_sorted]


def get_tokenizer_config(time_res: int, velocity_bins: int, max_bar_embedding: int) -> TokenizerConfig:
    """
    IMPORTANT:
    - max_bar_embedding must be large enough for long MIDIs.
      Otherwise you'll see KeyError('Bar_XX') when bar index exceeds the vocab.
    """
    return TokenizerConfig(
        beat_res={(0, 4): time_res},
        use_velocities=True,
        num_velocities=velocity_bins,
        use_note_duration=True,
        use_chords=False,
        use_rests=True,
        use_tempos=True,
        use_time_signatures=True,
        max_bar_embedding=max_bar_embedding,
    )


def json_safe_config(time_res: int, velocity_bins: int, max_bar_embedding: int) -> Dict:
    return {
        "tokenizer": "REMI",
        "time_res": int(time_res),
        "velocity_bins": int(velocity_bins),
        "max_bar_embedding": int(max_bar_embedding),
        "use_velocities": True,
        "use_note_duration": True,
        "use_rests": True,
        "use_tempos": True,
        "use_time_signatures": True,
        "note": "Stored JSON-safe summary (MidiTok config dict contains tuple keys e.g. beat_res).",
    }


def build_special_token_map(genres: List[str]) -> Tuple[List[str], Dict[str, int]]:
    """
    Special IDs live in [0..offset-1], and all MidiTok ids are shifted by +offset.
    We don't need PAD for tokenization; we add PAD later in the PyTorch collate.
    """
    genre_tokens = build_genre_tokens(genres)
    special_tokens = [BOS_TOKEN, EOS_TOKEN] + genre_tokens
    tok2id = {t: i for i, t in enumerate(special_tokens)}
    return special_tokens, tok2id


def extract_ids(tok_out: Any) -> List[int]:
    """
    MidiTok v3 tokenizer(path) returns:
      - TokSequence (has .ids)
      - list[TokSequence] (each has .ids)
    This function returns a single flattened list[int].
    """
    if hasattr(tok_out, "ids"):
        return [int(x) for x in list(tok_out.ids)]

    if isinstance(tok_out, list) and len(tok_out) > 0 and hasattr(tok_out[0], "ids"):
        out: List[int] = []
        for seq in tok_out:
            out.extend([int(x) for x in list(seq.ids)])
        return out

    if isinstance(tok_out, list) and (len(tok_out) == 0 or isinstance(tok_out[0], int)):
        return [int(x) for x in tok_out]

    raise TypeError(f"Unexpected tokenization output type: {type(tok_out)}")


def tokenize_one_file(
    midi_path: str,
    token_path: str,
    genre_token: str,
    special_tok2id: Dict[str, int],
    offset: int,
    time_res: int,
    velocity_bins: int,
    max_bar_embedding: int,
    add_eos: bool,
    max_len: Optional[int],
    overwrite: bool,
) -> Dict:
    out_path = Path(token_path)

    if out_path.exists() and not overwrite:
        try:
            arr = np.load(out_path)
            return {
                "midi_path": midi_path,
                "token_path": str(out_path.resolve()),
                "n_tokens": int(arr.shape[0]),
                "status": "skipped_existing",
            }
        except Exception:
            pass  # overwrite if corrupted

    try:
        cfg = get_tokenizer_config(
            time_res=time_res,
            velocity_bins=velocity_bins,
            max_bar_embedding=max_bar_embedding,
        )
        tokenizer = REMI(cfg)

        # Pass path directly -> symusic backend (preferred for MidiTok v3)
        tok_out = tokenizer(midi_path)
        base_ids = extract_ids(tok_out)

        # Reserve [0..offset-1] for our special tokens
        base_ids = [i + offset for i in base_ids]

        genre_id = special_tok2id[genre_token]
        bos_id = special_tok2id[BOS_TOKEN]
        eos_id = special_tok2id[EOS_TOKEN]

        ids = [genre_id, bos_id] + base_ids
        if add_eos:
            ids.append(eos_id)

        if max_len is not None and len(ids) > max_len:
            ids = ids[:max_len]

        arr = np.asarray(ids, dtype=np.int32)
        np.save(out_path, arr)

        return {
            "midi_path": midi_path,
            "token_path": str(out_path.resolve()),
            "n_tokens": int(arr.shape[0]),
            "status": "ok",
        }

    except Exception as e:
        return {
            "midi_path": midi_path,
            "token_path": "",
            "n_tokens": 0,
            "status": f"fail:{type(e).__name__}",
            "error": repr(e)[:500],
        }


def _worker_entry(job: Tuple[str, str, str], worker_kwargs: Dict) -> Dict:
    """
    Pickle-safe multiprocessing entrypoint:
    job = (midi_path, token_path, genre_token)
    """
    midi_path, token_path, genre_token = job
    return tokenize_one_file(
        midi_path=midi_path,
        token_path=token_path,
        genre_token=genre_token,
        **worker_kwargs,
    )


def _atomic_write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    """
    Atomic manifest write: write to temp then rename.
    Prevents half-written parquet on interrupt / crash.
    """
    tmp = out_path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=str, default="data/index.parquet")
    ap.add_argument("--splits", type=str, default="data/splits.json")
    ap.add_argument("--out_dir", type=str, default="data/tokenized")
    ap.add_argument("--time_res", type=int, default=8)
    ap.add_argument("--velocity_bins", type=int, default=16)
    ap.add_argument("--max_bar_embedding", type=int, default=1024)
    ap.add_argument("--add_eos", action="store_true")
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--num_workers", type=int, default=0)

    # NEW: hardening knobs
    ap.add_argument("--flush_every", type=int, default=500, help="Write manifest every N results")
    ap.add_argument("--maxtasksperchild", type=int, default=200, help="Recycle worker after N tasks (mp only)")

    args = ap.parse_args()

    index_path = Path(args.index)
    splits_path = Path(args.splits)
    out_dir = Path(args.out_dir)
    safe_mkdir(out_dir)
    safe_mkdir(out_dir / "tokens")

    df = pd.read_parquet(index_path).copy()
    if "genre_final" not in df.columns:
        raise ValueError("index.parquet must contain 'genre_final'")

    splits = json.loads(splits_path.read_text())["paths"]
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    # Deterministic order
    df = df.sort_values("midi_path").reset_index(drop=True)
    if args.limit is not None:
        df = df.head(args.limit)

    # IMPORTANT: build genre token set from FULL dataset (not limited debug subset)
    all_df = pd.read_parquet(index_path)
    genres = sorted(all_df["genre_final"].fillna("other").unique().tolist())

    special_tokens, special_tok2id = build_special_token_map(genres)
    offset = len(special_tokens)

    # Save reproducibility artifacts
    (out_dir / "tokenizer_config.json").write_text(
        json.dumps(json_safe_config(args.time_res, args.velocity_bins, args.max_bar_embedding), indent=2)
    )
    (out_dir / "special_tokens.json").write_text(json.dumps(special_tokens, indent=2))
    (out_dir / "special_token_to_id.json").write_text(json.dumps(special_tok2id, indent=2))
    (out_dir / "genres.json").write_text(json.dumps(genres, indent=2))
    (out_dir / "offset.json").write_text(json.dumps({"offset": offset}, indent=2))

    # Build jobs + metadata
    jobs: List[Tuple[str, str, str]] = []
    metas: List[Tuple[str, str]] = []  # (genre, split)

    for r in df.itertuples(index=False):
        midi_path = str(r.midi_path)
        genre = getattr(r, "genre_final", "other") or "other"
        genre_token = f"<GENRE={genre}>"

        if midi_path in train_set:
            split = "train"
        elif midi_path in val_set:
            split = "val"
        elif midi_path in test_set:
            split = "test"
        else:
            split = "unknown_split"

        token_name = sha1_name(midi_path) + ".npy"
        token_path = str((out_dir / "tokens" / token_name).resolve())

        jobs.append((midi_path, token_path, genre_token))
        metas.append((genre, split))

    worker_kwargs = dict(
        special_tok2id=special_tok2id,
        offset=offset,
        time_res=args.time_res,
        velocity_bins=args.velocity_bins,
        max_bar_embedding=args.max_bar_embedding,
        add_eos=args.add_eos,
        max_len=args.max_len,
        overwrite=args.overwrite,
    )

    manifest_path = out_dir / "manifest.parquet"

    # If manifest already exists, resume by loading it and skipping completed midi_path entries
    records: List[Dict] = []
    done_paths: set[str] = set()
    if manifest_path.exists():
        try:
            old = pd.read_parquet(manifest_path)
            records = old.to_dict("records")
            done_paths = set(old["midi_path"].astype(str).tolist())
            print(f"[resume] loaded {len(records)} rows from existing manifest, skipping completed paths")
        except Exception:
            print("[resume] manifest exists but failed to read; starting fresh")

    if done_paths:
        # filter jobs/metas to only unfinished
        new_jobs: List[Tuple[str, str, str]] = []
        new_metas: List[Tuple[str, str]] = []
        for (job, meta) in zip(jobs, metas):
            if job[0] not in done_paths:
                new_jobs.append(job)
                new_metas.append(meta)
        jobs, metas = new_jobs, new_metas

    if len(jobs) == 0:
        print("[done] nothing to do (everything already tokenized in manifest).")
        return

    # incremental flush helper
    def flush_manifest() -> None:
        dfm = pd.DataFrame(records)
        _atomic_write_parquet(dfm, manifest_path)

    if args.num_workers and args.num_workers > 0:
        # Build metadata map for unordered results
        meta_map = {job[0]: metas[i] for i, job in enumerate(jobs)}  # midi_path -> (genre, split)

        # IMPORTANT: maxtasksperchild hardening
        with Pool(processes=args.num_workers, maxtasksperchild=args.maxtasksperchild) as pool:
            it = pool.imap_unordered(partial(_worker_entry, worker_kwargs=worker_kwargs), jobs)

            try:
                for i, res in enumerate(tqdm(it, total=len(jobs), desc="Tokenizing (mp)"), start=1):
                    genre, split = meta_map.get(res["midi_path"], ("other", "unknown_split"))
                    res.update({"split": split, "genre_final": genre})
                    records.append(res)

                    # NEW: incremental flush
                    if args.flush_every and (i % args.flush_every == 0):
                        flush_manifest()

            except KeyboardInterrupt:
                print("\n[interrupt] Caught Ctrl+C. Writing manifest and exiting...")
                flush_manifest()
                raise

    else:
        for i, ((midi_path, token_path, genre_token), (genre, split)) in enumerate(
            tqdm(list(zip(jobs, metas)), total=len(jobs), desc="Tokenizing"), start=1
        ):
            res = tokenize_one_file(
                midi_path=midi_path,
                token_path=token_path,
                genre_token=genre_token,
                **worker_kwargs,
            )
            res.update({"split": split, "genre_final": genre})
            records.append(res)

            # NEW: incremental flush
            if args.flush_every and (i % args.flush_every == 0):
                flush_manifest()

    # final write
    flush_manifest()

    manifest = pd.DataFrame(records)
    print(f"\nSaved tokenized dataset to: {out_dir.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")
    print("\nStatus counts:")
    print(manifest["status"].value_counts().to_string())

    ok = manifest[manifest["status"] == "ok"]
    if len(ok) > 0:
        print("\nToken counts summary (ok only):")
        print(ok["n_tokens"].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_string())


if __name__ == "__main__":
    main()
