# SPdesign: Region-Aware Generative Design and Scoring of Signal Peptides

**SPdesign** is an end-to-end pipeline for *de novo* design of N-terminal signal peptides (SPs) with a region-aware fine-tuned ProtGPT2 language model, followed by systematic candidate scoring with SignalP 6.0 and an interpretable, literature-informed physicochemical score (HT-SP).

The pipeline consists of three stages:

1. **Fine-tuning** — `ProtGPT2_train_3seed.py`
2. **Sequence generation & validation** — `SPdesign.py`
3. **SignalP-gated scoring & ranking** — `SPscore.py`

---

## Repository Structure

```
SPdesign/
├── README.md
├── ProtGPT2_train_3seed.py        # Stage 1: region-aware weighted fine-tuning of ProtGPT2 (3 seeds)
├── SPdesign.py                    # Stage 2: conditional generation, validation, FASTA export
├── SPscore.py                     # Stage 3: SignalP 6.0 gate + HT-SP scoring and top-fraction selection
├── SP_selected.txt                # Top 30% high-scoring natural SPs (UniProt + HT-SP)
├── train_cdhit.txt                # Training set  (CD-HIT 0.90 split, 80%)
├── val_cdhit.txt                  # Validation set (CD-HIT 0.90 split, 10%)
└── test_cdhit.txt                 # Test set       (CD-HIT 0.90 split, 10%)
```

The fine-tuned checkpoints (~1.5 GB each, fp16) are hosted on [Hugging Face](https://huggingface.co/Yuanshichao19961010/SPdesign/tree/main) and are **not** bundled in this GitHub repository. See [Model Weights](#model-weights) for download instructions.

---

## Datasets

The repository ships with the natural signal-peptide datasets used for fine-tuning:

- **`SP_selected.txt`** — Full-length precursor sequences (SP + cargo) fetched from UniProt, then scored with `SPscore.py` (SignalP 6.0 gate + HT-SP score). Only the **top 30 %** highest-scoring sequences are retained. This file is the default input for `SPscore.py`.
- **`train_cdhit.txt` / `val_cdhit.txt` / `test_cdhit.txt`** — Region-tagged training sequences (`<Nregion>...<Hregion>...<Cregion>...<|endoftext|>`). The top-30 % SPs were clustered with **CD-HIT** at **90 % sequence identity**, and the resulting clusters were randomly partitioned into **train / validation / test = 8 : 1 : 1** (seed = 42). No cluster spans more than one split, preventing data leakage.

These files let you reproduce the fine-tuning without re-running SignalP or CD-HIT.

---

## How It Works

### Stage 1 — Region-aware fine-tuning (`ProtGPT2_train_3seed.py`)

Fine-tunes ProtGPT2 on natural signal peptides written as a single tagged line per sequence:

```
<Nregion>MKTAIIPLL<IIGFAL>
<Hregion>...
```

Full format (one sequence per line in the training files):

```
<Nregion>XXXX<Hregion>XXXX<Cregion>XXXX<|endoftext|>
```

Key features:

- **Region-aware weighted loss** — per-token cross-entropy is weighted by region identity (`REGION_WEIGHTS`: N-region ×6, H-region ×2, C-region ×5, EOS ×15, dedicated `<eos_sp>` ×25, region tags ×2) to counteract N-region collapse, C-region degeneration, and missing EOS.
- **Dedicated `<eos_sp>` termination token** inserted automatically before `<|endoftext|>`, with a cold-start weight boost during the first 2,000 steps.
- **Full seed control** (`SEEDS = [42, 123, 999]`) with deterministic cuDNN settings; per-seed checkpoints are written to `<OUTPUT_DIR>/seed_<seed>/checkpoint-*`, and a `FINAL_REPORT.json` aggregates test losses (overall and per-region).

Configuration (model path, data paths, weights, steps, LR, etc.) is set via constants at the top of the script.

### Stage 2 — Generation and validation (`SPdesign.py`)

Loads a fine-tuned checkpoint (by default: the largest-step `checkpoint-*` subdirectory under `MODEL_ROOT/seed_42`, or the seed directory itself if it already contains `model.safetensors`), generates candidates from the prompt `<Nregion>M`, and applies a validation funnel:

| Check | Rule |
|---|---|
| Tag structure | `<Nregion>` → `<Hregion>` → `<Cregion>` in order, all present |
| Initiator Met | N-region must start with **M**; H- and C-regions must **not** |
| Amino acid alphabet | Standard 20 aa only (non-standard residues removed) |
| Non-empty regions | After cleaning and truncation, no region may be empty |
| Total length | 10–80 aa |
| EOS handling | If the model emitted EOS, the C-region is trusted as-is; otherwise heuristic truncation caps C-region degeneration (homopolymer ≥5, 2-mer ×4, 3-mer ×3, C >15 aa, H >25 aa) |

Outputs per run:

- `SP1_weighted_<N>_seed<seed>_ckpt<step>.fasta` — accepted sequences with fully annotated headers:
  `>SPw_1|seed42|ckpt15000|N=18|H=12|C=5|len=35|AXA=strict(AQA)|EOS=yes|trunc=none`
- `..._report.txt` — acceptance rate, rejection-reason distribution, N/H/C length statistics, N-region positive-charge and hydrophobic-tail features, AXA cleavage-site motif rates, truncation usage, and an automated quality conclusion.

### Stage 3 — SignalP-gated scoring (`SPscore.py`)

Screens generated candidates in three steps:

1. **SignalP 6.0 gate** — candidates must be predicted as Sec/SPI with probability ≥ 0.90 (configurable) and have a parseable H-region; a parsed cleavage site is optional.
2. **HT-SP scoring** — an interpretable weighted sum of six modules (all components constrained to [0, 1]):

```
SP_score = 0.30·S_hydro + 0.25·S_cleavage + 0.15·S_HC
         + 0.10·S_N     + 0.10·S_length   + 0.10·S_SignalP

S_hydro     = 0.40·S_GRAVY_SP + 0.60·S_H_KD
S_cleavage  = 0.40·S_minus1 + 0.20·S_minus3 + 0.15·S_A_C + 0.15·S_P_C + 0.10·S_C_polarity
S_HC        = 0.50·S_turn_HC + 0.30·S_G_H + 0.20·S_Hend_charge
S_N         = 0.70·S_N_charge + 0.30·S_NH_hydrophobic_contrast
S_length    = 0.40·S_SP_length + 0.60·S_H_length
S_SignalP   = SignalP probability rescaled after the gate
```

3. **Top-fraction selection** — the highest-scoring fraction (default 30 %) is exported.

> **Note.** The HT-SP score is a *transparent heuristic* built from physicochemical feature classes highlighted in high-throughput secretion studies (e.g., Grasso et al., *ACS Synth Biol*, 2023, Bacillus subtilis/AmyQ). It does **not** reproduce the published 156-feature random-forest model, and its weights are not SHAP coefficients. It should be described as literature-informed candidate ranking, not as a validated predictor of secretion titre in other host/cargo systems.

Each input SP is fused to a fixed cargo peptide before SignalP scoring; scoring itself uses the SP-only sequence.

---

## Requirements

- Python ≥ 3.9
- [PyTorch](https://pytorch.org) (CUDA recommended for Stages 1–2)
- `transformers`, `datasets`, `numpy`, `pandas`, `tqdm`
- [SignalP 6.0](https://services.healthtech.dtu.dk/services/SignalP-6.0/) (Stage 3; academic license, downloadable from DTU)

```bash
pip install torch transformers datasets numpy pandas tqdm
```

The base ProtGPT2 weights default to [`nferruz/ProtGPT2`](https://huggingface.co/nferruz/ProtGPT2) on the Hugging Face Hub and are downloaded automatically on first run. To use a local copy instead, set `MODEL_NAME` in `ProtGPT2_train_3seed.py` to the local path.

---

## Model Weights

Fine-tuned ProtGPT2 checkpoints (region-aware weighted loss, 15,000 steps, fp16, ~1.5 GB per seed) are hosted on Hugging Face:

👉 https://huggingface.co/Yuanshichao19961010/SPdesign/tree/main

The repository contains three seed subdirectories (`seed42`, `seed123`, `seed999`), each with:

- `config.json`
- `generation_config.json`
- `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`

### Download and setup

Clone the Hugging Face repository (requires `git-lfs`):

```bash
git lfs install
git clone https://huggingface.co/Yuanshichao19961010/SPdesign SPdesign_model_3seed
```

Then place `SPdesign_model_3seed/` in the same directory as `SPdesign.py`:

```
SPdesign/
├── ProtGPT2_train_3seed.py
├── SPdesign.py
├── SPscore.py
└── SPdesign_model_3seed/
    ├── seed42/
    ├── seed123/
    └── seed999/
```

`MODEL_ROOT` in `SPdesign.py` defaults to the relative path `./SPdesign_model_3seed`:

```python
MODEL_ROOT = "./SPdesign_model_3seed"
```

Then generate directly — no retraining needed:

```bash
python SPdesign.py --N 100 --seed 42
```

The script auto-detects the checkpoint: the largest `checkpoint-*` subdirectory under `seed_<seed>/`, or the seed directory itself if it already contains `model.safetensors`.

---

## Usage

### 1. Fine-tune (optional — skip if `SPdesign_model_3seed/` is present)

This step is only required to reproduce the fine-tuning from scratch. If `SPdesign_model_3seed/` contains the checkpoints, go straight to **Step 2**.

The bundled `train_cdhit.txt`, `val_cdhit.txt`, and `test_cdhit.txt` are used by default (relative paths). Edit the configuration block at the top of `ProtGPT2_train_3seed.py` (`MODEL_NAME`, `REGION_WEIGHTS`, `MAX_STEPS`, ...) if needed, then:

```bash
python ProtGPT2_train_3seed.py
```

Training automatically resumes from the latest checkpoint if one exists. Output checkpoints are written to `./SPTRAIN_REGION_3SEED/`.

### 2. Generate validated SPs

```bash
python SPdesign.py --N 100 --seed 42
# force a specific checkpoint:
python SPdesign.py --N 100 --seed 42 --checkpoint checkpoint-15000
```

| Argument | Default | Description |
|---|---|---|
| `--N` | 100 | Number of *accepted* sequences to produce |
| `--seed` | 42 | Fine-tuning seed whose checkpoint is used (42 / 123 / 999) |
| `--batch` | 16 | Prompt batch size per forward pass (auto-halved on CUDA OOM) |
| `--per-prompt` | 1 | Sequences returned per prompt |
| `--attempt-max` | 50 | Maximum generation rounds |
| `--checkpoint` | latest | Force a specific checkpoint name |

Set `MODEL_ROOT` and `OUTPUT_DIR` at the top of the script.

### 3. Score and rank

By default `SPscore.py` reads the bundled `SP_selected.txt` as input:

```bash
# score the bundled SP_selected.txt with local SignalP
python SPscore.py \
  --outdir run1_score \
  --top_fraction 0.30 \
  --run_signalp \
  --signalp_cmd /path/to/signalp6

# or provide your own input
python SPscore.py \
  --input generated.fasta \
  --outdir run1_score \
  --top_fraction 0.30 \
  --run_signalp \
  --signalp_cmd /path/to/signalp6

# re-parse / re-score existing SignalP outputs without rerunning SignalP
python SPscore.py \
  --input SP_selected.txt \
  --outdir run1_score \
  --parse_only \
  --top_fraction 0.30
```

Key outputs in `--outdir`:

| File | Content |
|---|---|
| `SP_all_scored.csv` | All gate-passing sequences with raw features, module scores, per-module contributions, `HT_SP_score` |
| `SP_selected_scored.csv` | Top-fraction subset (same columns) |
| `SP_selected.fasta` / `SP_selected.txt` | Selected SP sequences only |
| `run_summary.json` | Gate settings, weights, counts |
| `SignalP6_parsed_results.csv` | Parsed SignalP predictions |

---

## Citation

If you use SPdesign, please cite the manuscript and the underlying resources:

- ProtGPT2: Ferruz, N., Schmidt, S. & Höcker, B. *ProtGPT2 is a deep unsupervised language model for protein design.* Nat Commun 13, 4348 (2022).
- SignalP 6.0: Teufel, F. et al. *SignalP 6.0 across all organisms predicts protein signals across all five types.* Nat Biotechnol (2022).
- HT-SP feature rationale: Grasso, V. et al. *Signal Peptide Efficiency: From High-Throughput Data to Prediction and Explanation.* ACS Synth Biol (2023).

*A dedicated citation for this pipeline will be added upon publication.*
