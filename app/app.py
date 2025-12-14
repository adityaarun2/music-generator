#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import gradio as gr
import torch

# allow imports from src/
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from generate import (
    load_tokenizer_from_tokenized_dir,
    generate_tokens,
    decode_to_midi,
    BOS_TOKEN,
    get_device,
)
from model import GPT, GPTConfig


DEFAULT_CKPT = str(ROOT / "runs" / "l4_full" / "ckpt_best.pt")
DEFAULT_TOKENIZED_DIR = str(ROOT / "data" / "tokenized")


def load_model(ckpt_path: str, device: torch.device) -> GPT:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


# cache heavy objects
_DEVICE = get_device()
_MODEL = None
_TOKENIZER = None
_SPECIAL_TOK2ID = None
_OFFSET = None
_GENRES = None


def ensure_loaded(ckpt_path: str, tokenized_dir: str):
    global _MODEL, _TOKENIZER, _SPECIAL_TOK2ID, _OFFSET, _GENRES

    tokenized_dir_p = Path(tokenized_dir)
    tokenizer, special_tokens, special_tok2id, offset, genres_map = load_tokenizer_from_tokenized_dir(tokenized_dir_p)

    if _MODEL is None or getattr(_MODEL, "_ckpt_path", None) != ckpt_path:
        model = load_model(ckpt_path, _DEVICE)
        model._ckpt_path = ckpt_path  # type: ignore[attr-defined]
        _MODEL = model

    _TOKENIZER = tokenizer
    _SPECIAL_TOK2ID = special_tok2id
    _OFFSET = offset
    _GENRES = sorted(list(genres_map.keys()))


def generate_midi(
    genre: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    ckpt_path: str,
    tokenized_dir: str,
    seed: int,
):
    ensure_loaded(ckpt_path, tokenized_dir)

    torch.manual_seed(seed)

    genre_tok = f"<GENRE={genre}>"
    genre_id = int(_SPECIAL_TOK2ID[genre_tok])
    bos_id = int(_SPECIAL_TOK2ID[BOS_TOKEN])

    # prompt in token-space then +1 shift for model-space
    prompt_model_space = [genre_id + 1, bos_id + 1]

    ids_model = generate_tokens(
        model=_MODEL,
        prompt_ids=prompt_model_space,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    # write to a temp file and return path for Gradio download
    tmpdir = Path(tempfile.mkdtemp())
    out_mid = tmpdir / f"gen_{genre}_t{temperature}_p{top_p}.mid"

    decode_to_midi(
        tokenizer=_TOKENIZER,
        ids_model_space=ids_model,
        offset=_OFFSET,
        out_mid_path=out_mid,
    )

    return str(out_mid)


def main():
    # preload to populate genres
    ensure_loaded(DEFAULT_CKPT, DEFAULT_TOKENIZED_DIR)

    with gr.Blocks(title="Music Generator (Genre-conditioned)") as demo:
        gr.Markdown("# 🎵 Genre-conditioned MIDI generator")

        with gr.Row():
            genre = gr.Dropdown(choices=_GENRES, value=_GENRES[0], label="Genre")
            seed = gr.Number(value=42, precision=0, label="Seed")

        with gr.Row():
            temperature = gr.Slider(0.2, 2.0, value=1.0, step=0.05, label="Temperature")
            top_p = gr.Slider(0.5, 1.0, value=0.9, step=0.01, label="Top-p")

        max_new = gr.Slider(256, 4096, value=2048, step=256, label="Max new tokens")

        with gr.Accordion("Advanced", open=False):
            ckpt = gr.Textbox(value=DEFAULT_CKPT, label="Checkpoint path")
            tokenized_dir = gr.Textbox(value=DEFAULT_TOKENIZED_DIR, label="Tokenized dir")

        btn = gr.Button("Generate MIDI")
        out_file = gr.File(label="Generated MIDI (.mid)")

        btn.click(
            fn=generate_midi,
            inputs=[genre, temperature, top_p, max_new, ckpt, tokenized_dir, seed],
            outputs=[out_file],
        )

        gr.Markdown(
            "Tip: try **temperature 0.8–1.2** and **top_p 0.85–0.95**. "
            "If output is too repetitive, raise temperature slightly."
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
