from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, get_worker_info


class TokenNPYDataset(Dataset):
    """
    Loads .npy token sequences from manifest.parquet.

    We reserve token id 0 for PAD in batching.
    Since tokenization emits ids starting at 0, we SHIFT ALL TOKENS by +1 here.
    """

    def __init__(
        self,
        manifest_path: str,
        split: str,
        seq_len: int = 1024,
        random_crop: bool = True,
        seed: int = 42,
        mmap: bool = True,          # NEW: use np.load(..., mmap_mode="r")
    ):
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.seq_len = int(seq_len)
        self.random_crop = bool(random_crop)
        self.seed = int(seed)
        self.mmap = bool(mmap)

        m = pd.read_parquet(self.manifest_path)
        m = m[m["split"] == split].copy()
        m = m[m["status"].isin(["ok", "skipped_existing"])].copy()
        m = m[m["token_path"].astype(str).str.len() > 0].reset_index(drop=True)

        if len(m) == 0:
            raise RuntimeError(f"No samples found for split={split} in {manifest_path}")

        self.m = m

        # Per-worker RNGs (initialized lazily inside workers)
        self._worker_rng: Optional[np.random.Generator] = None

    def __len__(self) -> int:
        return len(self.m)

    def _get_rng(self) -> np.random.Generator:
        """
        Ensure each DataLoader worker has its own RNG stream.
        """
        if self._worker_rng is not None:
            return self._worker_rng

        wi = get_worker_info()
        if wi is None:
            # main process
            self._worker_rng = np.random.default_rng(self.seed)
        else:
            # worker process: different seed per worker
            self._worker_rng = np.random.default_rng(self.seed + wi.id * 100003)
        return self._worker_rng

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.m.iloc[idx]
        path = row["token_path"]

        # Faster / lower-RAM reads
        arr = np.load(path, mmap_mode="r" if self.mmap else None)

        # Shift to reserve PAD=0
        # Do it as int64 for torch embeddings safely
        # (np.asarray avoids an extra copy if already int64 and not memmap)
        arr = np.asarray(arr, dtype=np.int64) + 1

        L = int(arr.shape[0])

        if L >= self.seq_len:
            if self.random_crop:
                rng = self._get_rng()
                start = int(rng.integers(0, L - self.seq_len + 1))
            else:
                start = 0
            x = arr[start : start + self.seq_len]
        else:
            x = np.zeros((self.seq_len,), dtype=np.int64)
            x[:L] = arr

        # next-token prediction
        # torch.from_numpy is fast; slicing produces a view (still OK)
        input_ids = torch.from_numpy(x[:-1].copy()).long()
        labels = torch.from_numpy(x[1:].copy()).long()

        return {"input_ids": input_ids, "labels": labels}
