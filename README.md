# Genre-Conditioned Music Generation with REMI-GPT

This project trains a **genre-conditioned Transformer (GPT)** over **REMI tokens** to generate symbolic music (**MIDI**) in styles such as *jazz, classical, rock, pop,* and more.

It implements a **full research-grade pipeline**:
- MIDI preprocessing → REMI tokenization
- GPT training with AMP (mixed precision)
- Quantitative evaluation (loss, perplexity, feature statistics)
- Genre-consistency evaluation via a classifier
- Interactive UI for music generation

---

## Dataset

This project uses the **Lakh MIDI Dataset v0.1 (Clean MIDI subset)**, consisting of approximately 14,000 MIDI files, organized by artist name and annotated with genre labels.

**Source:** http://colinraffel.com/projects/lmd/

**Citation:**  
> Colin Raffel. "Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching". PhD Thesis, 2016.

---

## Generate Jazz Demo Using Model Checkpoint

If you already have a trained checkpoint:
```bash
pip install -r requirements.txt

python src/generate.py \
  --ckpt runs/l4_full/ckpt_best.pt \
  --tokenized_dir data/tokenized \
  --genre jazz \
  --out out/jazz.mid \
  --max_new_tokens 2048 \
  --temperature 1.0 \
  --top_p 0.9
```

**Output:** `out/jazz.mid` (open in any DAW or MIDI player)

---

## Interactive UI

Launch a simple Gradio UI:
```bash
pip install gradio
python app/app.py
```

**Features:**
- Genre selection
- Temperature & top-p sampling
- Seed control
- Download generated MIDI

---

## Setup

**Python:** 3.9 – 3.12 recommended
```bash
pip install -r requirements.txt
```

**Key dependencies:**
- PyTorch
- miditok
- pretty_midi
- numpy / pandas / scikit-learn
- matplotlib / seaborn
- gradio (UI)

---

## Data Preparation

### 1. Build dataset index
```bash
python src/build_index.py
```

**Creates:**
- `data/index.parquet`
- `data/splits.json`

### 2. Tokenize MIDI files (REMI)
```bash
python src/tokenize_dataset.py \
  --index data/index.parquet \
  --splits data/splits.json \
  --out_dir data/tokenized \
  --max_bar_embedding 1024 \
  --max_len 2048 \
  --num_workers 8
```

**Outputs:**
- `data/tokenized/tokens/*.npy`
- `data/tokenized/manifest.parquet`
- Tokenizer metadata (`offset.json`, `special_tokens.json`, etc.)

---

## Training

Train a GPT model with mixed precision (AMP) and automatic checkpointing:
```bash
python src/train.py \
  --manifest data/tokenized/manifest.parquet \
  --out_dir runs/l4_full \
  --vocab_size 1414 \
  --seq_len 1024 \
  --batch_size 16 \
  --amp
```

**Checkpoints:**
- `ckpt_latest.pt` – updated during training
- `ckpt_best.pt` – best validation loss (recommended for generation)

---

## Evaluation

### Model-level metrics (loss & perplexity)
```bash
python src/evaluate.py \
  --ckpt runs/l4_full/ckpt_best.pt \
  --tokenized_dir data/tokenized \
  --manifest data/tokenized/manifest.parquet \
  --index data/index.parquet \
  --splits data/splits.json \
  --out_dir eval/l4_full \
  --samples_per_genre 25
```

**Outputs:**
- `model_metrics.csv`
- `genre_compare.csv`
- Generated MIDI samples per genre

### Genre-consistency evaluation
```bash
python src/genre_classifier_eval.py \
  --index data/index.parquet \
  --splits data/splits.json \
  --generated_dir eval/l4_full/generated \
  --out_dir eval/l4_full/genre_classifier
```

---

## Analysis Notebook

Open:
```bash
notebooks/model_performance.ipynb
```

**This notebook:**
- Loads evaluation CSVs
- Plots pitch-class, density, polyphony, duration distributions
- Compares real vs generated music per genre
- Summarizes model strengths & limitations

---

## Example Results (Summary)

| Metric             | Value  |
|--------------------|--------|
| Validation Loss    | ~0.67  |
| Test Loss          | ~0.68  |
| Test Perplexity    | ~1.97  |

**Observations:**
- Strong genre-specific pitch-class structure
- Generated music is simpler (lower density, reduced polyphony)
- Genre classifier shows above-chance genre consistency

---

## Repository Layout
```
music-generator/
├── app/                    # Gradio UI
├── data/                   # Dataset + tokenized artifacts (ignored)
├── eval/                   # Evaluation outputs (ignored)
├── notebooks/              # Analysis notebooks
├── runs/                   # Model checkpoints (ignored)
├── src/                    # Training, generation, evaluation code
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Known Limitations

- MIDI-only (no audio rendering)
- No explicit long-term musical structure modeling
- Evaluation metrics are symbolic proxies (not human preference)
- Generated samples can be sparse at low temperatures

---

## Notes on Reproducibility

- Evaluation outputs (`eval/*.csv`) are fully reproducible
- Trained checkpoints are not committed by default
- See README for instructions to regenerate everything
