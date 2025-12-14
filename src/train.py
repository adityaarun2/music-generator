#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import time

import torch
from torch.utils.data import DataLoader

from data import TokenNPYDataset
from model import GPT, GPTConfig


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def supports_bf16(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    try:
        return torch.cuda.is_bf16_supported()
    except Exception:
        return False


@torch.no_grad()
def evaluate(model: GPT, loader: DataLoader, device: torch.device, autocast_ctx) -> float:
    model.eval()
    losses = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with autocast_ctx:
            _, loss = model(input_ids, labels)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--manifest", type=str, default="artifacts/tokenized/manifest.parquet")
    ap.add_argument("--out_dir", type=str, default="runs/exp1")

    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--vocab_size", type=int, default=1412)

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--max_steps", type=int, default=20000)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--save_every", type=int, default=1000)

    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=6)
    ap.add_argument("--n_embd", type=int, default=384)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--clip_grad", type=float, default=1.0)

    # NEW: best-checkpoint options
    ap.add_argument("--save_best", action="store_true", help="Also save ckpt_best.pt when val improves.")
    ap.add_argument("--best_metric", type=str, default="val_loss", choices=["val_loss"])
    ap.add_argument("--best_mode", type=str, default="min", choices=["min", "max"])

    args = ap.parse_args()

    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    device = get_device()
    print("Device:", device)

    # Data
    train_ds = TokenNPYDataset(args.manifest, split="train", seq_len=args.seq_len, random_crop=True, seed=args.seed)
    val_ds = TokenNPYDataset(args.manifest, split="val", seq_len=args.seq_len, random_crop=False, seed=args.seed)

    pin = (device.type == "cuda")
    prefetch = 4 if (device.type == "cuda" and args.num_workers > 0) else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=prefetch if prefetch is not None else 2,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers // 2),
        pin_memory=pin,
        persistent_workers=(max(0, args.num_workers // 2) > 0),
    )

    # Model
    cfg = GPTConfig(
        vocab_size=args.vocab_size,
        block_size=args.seq_len - 1,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # AMP
    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if supports_bf16(device) else torch.float16

    # With bf16 you typically do NOT need GradScaler; with fp16 you do.
    use_scaler = use_amp and (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    if use_amp:
        print(f"AMP: enabled | dtype={amp_dtype} | GradScaler={'on' if use_scaler else 'off'}")
        autocast_ctx = torch.amp.autocast("cuda", dtype=amp_dtype)
    else:
        print("AMP: disabled (or non-CUDA device)")
        autocast_ctx = torch.autocast("cpu", enabled=False)  # no-op context

    # NEW: track best checkpoint
    best_path = out_dir / "ckpt_best.pt"
    if args.best_mode == "min":
        best_val = float("inf")
        def is_better(v: float, best: float) -> bool:
            return v < best
    else:
        best_val = -float("inf")
        def is_better(v: float, best: float) -> bool:
            return v > best
    best_step = -1

    # Throughput calc
    tokens_per_micro = args.batch_size * (args.seq_len - 1)
    model.train()

    step = 0
    last_log = time()
    running_loss = 0.0

    # IMPORTANT: create iterator up-front so grad_accum can pull "next" safely
    train_iter = iter(train_loader)

    while step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # One optimizer step == grad_accum micro-steps
        optim.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro in range(args.grad_accum):
            if micro > 0:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with autocast_ctx:
                _, loss = model(input_ids, labels)
                loss = loss / args.grad_accum

            accum_loss += float(loss.item())

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        # clip + step
        if use_scaler:
            scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        if use_scaler:
            scaler.step(optim)
            scaler.update()
        else:
            optim.step()

        step += 1
        running_loss = accum_loss

        # logging
        if step == 1 or step % 50 == 0:
            now = time()
            dt = max(1e-9, now - last_log)
            tok_s = (tokens_per_micro * args.grad_accum) / dt
            print(f"step {step}/{args.max_steps} | loss {running_loss:.4f} | tok/s {tok_s:.0f}")
            last_log = now

        # eval (+ best)
        if step % args.eval_every == 0:
            val_loss = evaluate(model, val_loader, device, autocast_ctx)
            print(f"[eval] step {step} | val_loss {val_loss:.4f}")

            if args.save_best and is_better(val_loss, best_val):
                best_val = float(val_loss)
                best_step = step
                ckpt_best = {
                    "model": model.state_dict(),
                    "config": cfg.__dict__,
                    "step": step,
                    "best_metric": args.best_metric,
                    "best_mode": args.best_mode,
                    "best_val_loss": best_val,
                }
                torch.save(ckpt_best, best_path)
                print(f"[best] val_loss {best_val:.4f} @ step {best_step} -> {best_path}")

        # save latest (and step snapshots)
        if step % args.save_every == 0:
            ckpt = {"model": model.state_dict(), "config": cfg.__dict__, "step": step}
            torch.save(ckpt, out_dir / f"ckpt_step{step}.pt")
            torch.save(ckpt, out_dir / "ckpt_latest.pt")
            print(f"[save] {out_dir/'ckpt_latest.pt'}")

        if step >= args.max_steps:
            break

    print("Done.")
    if args.save_best and best_step >= 0:
        print(f"Best checkpoint: val_loss {best_val:.4f} @ step {best_step} -> {best_path}")


if __name__ == "__main__":
    main()
