#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ROOT = "./SPdesign_model_3seed"
MODEL_SEED = 42
CHECKPOINT_NAME: Optional[str] = None
OUTPUT_DIR = "./output"

MAX_NEW_TOKENS = 80
TEMPERATURE   = 1.0
TOP_P         = 0.9
REP_PENALTY   = 1.0
DO_SAMPLE     = True
PROMPT_STRING = "<Nregion>M"

STANDARD_AA   = set("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC   = set("VILMFWYCA")

TRUST_EOS_IF_FOUND = True
APPLY_DEGEN_TRUNC_IF_NO_EOS = True
MIN_TOTAL_LEN_ACCEPT = 10
MAX_TOTAL_LEN_ACCEPT = 80

C_MAX_LEN_CAP = 15
C_DEGEN_RUN_SAME = 5
C_DEGEN_2MER_REPEAT = 4
H_MAX_LEN_CAP = 25

EXPECT_EOS_SP_TOKEN = False

N_TAG = "<Nregion>"
H_TAG = "<Hregion>"
C_TAG = "<Cregion>"
EOS_TOKEN_IDX = "<|endoftext|>"
EOS_SP_TOKEN_IDX = "<eos_sp>" if EXPECT_EOS_SP_TOKEN else None

SPECIAL_TOKENS = [N_TAG, H_TAG, C_TAG]
if EXPECT_EOS_SP_TOKEN:
    SPECIAL_TOKENS.append(EOS_SP_TOKEN_IDX)


def _resolve_seed_dir(seed: int) -> Path:
    for name in (f"seed_{seed}", f"seed{seed}"):
        p = Path(MODEL_ROOT) / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"seed directory not found: looked for seed_{seed} and seed{seed} under {MODEL_ROOT}"
    )


def _is_model_dir(p: Path) -> bool:
    return p.is_dir() and (
        (p / "model.safetensors").exists()
        or (p / "pytorch_model.bin").exists()
        or (p / "model.bin").exists()
    )


def find_latest_checkpoint(seed: int) -> Path:
    seed_dir = _resolve_seed_dir(seed)
    if _is_model_dir(seed_dir):
        return seed_dir
    ckpts = sorted(
        seed_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
    )
    if not ckpts:
        raise FileNotFoundError(
            f"no checkpoint-* subdirectories and no model files found under {seed_dir}"
        )
    return ckpts[-1]


def get_checkpoint_path(seed: int) -> Path:
    if CHECKPOINT_NAME is None:
        return find_latest_checkpoint(seed)
    seed_dir = _resolve_seed_dir(seed)
    p = seed_dir / CHECKPOINT_NAME
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    return p


def parse_regions(text: str) -> Optional[Tuple[str, str, str]]:
    in_idx = text.find(N_TAG)
    ih_idx = text.find(H_TAG)
    ic_idx = text.find(C_TAG)
    if in_idx == -1 or ih_idx == -1 or ic_idx == -1:
        return None
    if not (in_idx < ih_idx < ic_idx):
        return None
    n_start = in_idx + len(N_TAG)
    h_start = ih_idx + len(H_TAG)
    c_start = ic_idx + len(C_TAG)
    n_seq = text[n_start:ih_idx]
    h_seq = text[h_start:ic_idx]
    c_seq_raw = text[c_start:]

    for et in ([EOS_SP_TOKEN_IDX, EOS_TOKEN_IDX] if EXPECT_EOS_SP_TOKEN else [EOS_TOKEN_IDX]):
        if et and et in c_seq_raw:
            c_seq_raw = c_seq_raw[: c_seq_raw.index(et)]
    m = re.search(r"<[A-Za-z0-9_|]+>", c_seq_raw)
    if m:
        c_seq_raw = c_seq_raw[: m.start()]
    c_seq = c_seq_raw
    return n_seq, h_seq, c_seq


def clean_aa(s: str) -> str:
    return "".join(ch for ch in s.upper() if ch in STANDARD_AA)


def has_eos_marker(text: str) -> bool:
    if EOS_TOKEN_IDX in text:
        return True
    if EXPECT_EOS_SP_TOKEN and EOS_SP_TOKEN_IDX and EOS_SP_TOKEN_IDX in text:
        return True
    return False


def strip_eos_markers(text: str) -> str:
    out = text
    if EXPECT_EOS_SP_TOKEN and EOS_SP_TOKEN_IDX:
        if EOS_SP_TOKEN_IDX in out:
            out = out[: out.index(EOS_SP_TOKEN_IDX)]
    if EOS_TOKEN_IDX in out:
        out = out[: out.index(EOS_TOKEN_IDX)]
    return out


def find_c_degeneration_trunc_point(c_seq: str) -> int:
    n = len(c_seq)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and c_seq[j + 1] == c_seq[i]:
            j += 1
        run_len = j - i + 1
        if run_len >= C_DEGEN_RUN_SAME:
            return max(3, i)
        i = j + 1

    for k in range(0, n - 7 + 1):
        ok = True
        m1 = c_seq[k:k + 2]
        if len(set(m1)) < 2:
            continue
        for repeat_i in range(4):
            a, b = k + repeat_i * 2, k + repeat_i * 2 + 2
            if c_seq[a:b] != m1:
                ok = False
                break
        if ok:
            return max(3, k)

    for k in range(0, n - 9 + 1):
        ok = True
        m3 = c_seq[k:k + 3]
        if len(set(m3)) < 2:
            continue
        for repeat_i in range(3):
            a, b = k + repeat_i * 3, k + repeat_i * 3 + 3
            if c_seq[a:b] != m3:
                ok = False
                break
        if ok:
            return max(3, k)

    return n


def apply_heuristic_truncation(n_clean, h_clean, c_clean) -> Tuple[str, str, str, str]:
    notes = []
    if len(h_clean) > H_MAX_LEN_CAP:
        old = len(h_clean)
        h_clean = h_clean[:20]
        notes.append(f"H_capped({old}->{len(h_clean)})")

    c_trunc_degen = find_c_degeneration_trunc_point(c_clean)
    if c_trunc_degen < len(c_clean):
        old = len(c_clean)
        c_clean = c_clean[:c_trunc_degen]
        notes.append(f"C_degen_trunc({old}->{len(c_clean)})")

    if len(c_clean) > C_MAX_LEN_CAP:
        old = len(c_clean)
        c_clean = c_clean[:10]
        notes.append(f"C_len_cap({old}->{len(c_clean)})")
    return n_clean, h_clean, c_clean, ",".join(notes)


def check_axa(c_seq: str) -> str:
    if len(c_seq) < 3:
        return "no(too_short)"
    last = c_seq[-3:]
    p1, p2, p3 = last[0], last[1], last[2]
    strict_ok = (p1 in "AGSVLIT" and p2 in "GAST" and p3 in "AGSV" and p3 != "P")
    loose_ok  = (p1 in "AGSVLITCMFWY" and p3 in "AGSVLITCMFWY" and p3 != "P")
    if strict_ok:
        return f"strict({last})"
    if loose_ok:
        return f"loose({last})"
    return f"no({last})"


REJECT_REASON_COUNTER = Counter()
TRUNC_NOTE_COUNTER = Counter()


def process_one(raw_text: str):
    eos_found = has_eos_marker(raw_text)

    parsed = parse_regions(raw_text)
    if parsed is None:
        if raw_text.find(N_TAG) == -1:
            return False, None, {}, "MISSING_N_TAG"
        if raw_text.find(H_TAG) == -1:
            return False, None, {}, "MISSING_H_TAG"
        if raw_text.find(C_TAG) == -1:
            return False, None, {}, "MISSING_C_TAG"
        return False, None, {}, "TAG_WRONG_ORDER"

    n_seq, h_seq, c_seq = parsed

    n_clean = clean_aa(n_seq)
    h_clean = clean_aa(h_seq)
    c_clean = clean_aa(c_seq)

    if not n_clean:
        return False, None, {}, "EMPTY_N_AFTER_CLEAN"
    if not h_clean:
        return False, None, {}, "EMPTY_H_AFTER_CLEAN"
    if not c_clean:
        return False, None, {}, "EMPTY_C_AFTER_CLEAN"

    if n_clean[0] != "M":
        return False, None, {}, "N_NOT_START_M"
    if h_clean[0] == "M":
        return False, None, {}, "H_START_M"
    if c_clean[0] == "M":
        return False, None, {}, "C_START_M"

    trunc_note = ""
    if not (eos_found and TRUST_EOS_IF_FOUND):
        if APPLY_DEGEN_TRUNC_IF_NO_EOS:
            n_clean, h_clean, c_clean, trunc_note = apply_heuristic_truncation(
                n_clean, h_clean, c_clean
            )

    if trunc_note:
        TRUNC_NOTE_COUNTER[trunc_note] += 1

    if not n_clean or not h_clean or not c_clean:
        return False, None, {}, "EMPTY_AFTER_TRUNC"

    total = n_clean + h_clean + c_clean
    if len(total) < MIN_TOTAL_LEN_ACCEPT:
        return False, None, {}, f"TOTAL_LEN_UNDER_{MIN_TOTAL_LEN_ACCEPT}"
    if len(total) > MAX_TOTAL_LEN_ACCEPT:
        return False, None, {}, f"TOTAL_LEN_OVER_{MAX_TOTAL_LEN_ACCEPT}"

    header_extra = {
        "N_len": len(n_clean),
        "H_len": len(h_clean),
        "C_len": len(c_clean),
        "total_len": len(total),
        "AXA": check_axa(c_clean),
        "EOS": "yes" if eos_found else "no",
        "trunc": trunc_note if trunc_note else "none",
        "n_seq": n_clean,
        "h_seq": h_clean,
        "c_seq": c_clean,
    }
    return True, total, header_extra, None


def load_model_and_tokenizer(seed: int):
    ckpt = get_checkpoint_path(seed)
    print(f"[load] model_seed={seed}  checkpoint={ckpt}", flush=True)

    tok = AutoTokenizer.from_pretrained(str(ckpt), use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    extra_need = list(SPECIAL_TOKENS)
    missing = [t for t in extra_need if t not in tok.get_vocab()]
    if missing:
        print(f"[warn] missing special tokens in tokenizer: {missing}; adding them and resizing embeddings")
        tok.add_special_tokens({"additional_special_tokens": missing})

    model = AutoModelForCausalLM.from_pretrained(str(ckpt), torch_dtype=torch.float16)
    if missing:
        model.resize_token_embeddings(len(tok))

    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return tok, model


@torch.no_grad()
def generate_batch(tok, model, prompts, num_return_per_prompt):
    enc = tok(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        attn = attn.cuda()

    eos_list = [tok.eos_token_id]
    if EXPECT_EOS_SP_TOKEN and EOS_SP_TOKEN_IDX:
        es_id = tok.convert_tokens_to_ids(EOS_SP_TOKEN_IDX)
        if isinstance(es_id, int) and es_id != tok.unk_token_id:
            eos_list.append(es_id)

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REP_PENALTY,
        pad_token_id=tok.pad_token_id,
        eos_token_id=eos_list if len(eos_list) > 1 else eos_list[0],
        num_return_sequences=num_return_per_prompt,
    )
    prompt_lens = [int((x != tok.pad_token_id).sum().item()) for x in enc["input_ids"]]
    bsz = len(prompts)
    decoded = []
    for global_i, gen in enumerate(out):
        pi = global_i // num_return_per_prompt
        plen = prompt_lens[pi]
        tail = gen[plen:]
        decoded.append(tok.decode(tail, skip_special_tokens=False))
    return decoded


def write_report(path: Path, N: int, seed: int, attempt: int, accepted_rows,
                 reject_counter: Counter, trunc_counter: Counter):
    lines = []

    def add(s=""):
        lines.append(s)

    add("=" * 78)
    add(f"  SPdesign (weighted training) validation report    seed={seed}    target={N}    attempts={attempt}")
    add("=" * 78)
    add("")
    acc_rate = len(accepted_rows) / max(1, attempt) * 100
    add(f"Acceptance rate: {len(accepted_rows)} / {attempt} = {acc_rate:.2f}%")
    add("")

    add("-" * 70)
    add("Rejection reason statistics")
    add("-" * 70)
    if not reject_counter:
        add("  (no rejected samples)")
    for reason, c in reject_counter.most_common():
        add(f"  {reason:<28s}  {c:>5d}  {c / max(1, attempt) * 100:6.2f}%")
    add("")

    if accepted_rows:
        n_lens = np.array([r["header"]["N_len"] for r in accepted_rows])
        h_lens = np.array([r["header"]["H_len"] for r in accepted_rows])
        c_lens = np.array([r["header"]["C_len"] for r in accepted_rows])
        total_lens = np.array([r["header"]["total_len"] for r in accepted_rows])

        def _pct(n):
            return int(np.percentile(n, 5)), int(np.percentile(n, 95))

        add("-" * 70)
        add("Length distributions of accepted samples (N/H/C/total)")
        add("-" * 70)
        for name, arr in (("N", n_lens), ("H", h_lens), ("C", c_lens), ("TOTAL", total_lens)):
            p5, p95 = _pct(arr)
            add(f"  {name:<6s}  mean={arr.mean():.2f}   median={int(np.median(arr))}  "
                f"min={arr.min()}  max={arr.max()}  5~95%=[{p5},{p95}]")
        add("")

        eos_cnt = Counter(r["header"]["EOS"] for r in accepted_rows)
        axa_cnt = Counter(r["header"]["AXA"].split("(")[0] for r in accepted_rows)
        add("-" * 70)
        add("N-region positive charge / N-tail hydrophobicity / C-cleavage AXA / EOS rate")
        add("-" * 70)
        add(f"  EOS termination rate          (train set 100% ): "
            f"yes={eos_cnt.get('yes', 0)} ({eos_cnt.get('yes', 0) / len(accepted_rows) * 100:.1f}%)  "
            f"no={eos_cnt.get('no', 0)} ({eos_cnt.get('no', 0) / len(accepted_rows) * 100:.1f}%)")

        ratios = []
        n_tail_hydro = []
        n_1bare = 0
        for r in accepted_rows:
            n = r["header"]["n_seq"]
            if len(n) >= 2:
                head = n[1:5] if len(n) > 5 else n[1:]
                kr = sum(1 for ch in head if ch in "KR")
                ratios.append(kr / max(1, len(head)))
            if len(n) == 1:
                n_1bare += 1
            if len(n) >= 4:
                tail3 = n[-3:]
                n_tail_hydro.append(sum(1 for ch in tail3 if ch in HYDROPHOBIC) / 3)
        add(f"  N_len=1 (bare M) ratio  (train set 0.2%): "
            f"{n_1bare} / {len(accepted_rows)} = {n_1bare / len(accepted_rows) * 100:.1f}%")
        if ratios:
            add(f"  N-head (first 4 after M) KR fraction (train set 30.9%): "
                f"{np.mean(ratios) * 100:.1f}%")
        if n_tail_hydro:
            add(f"  N-tail last-3 hydrophobic fraction   (train set 54.7%): "
                f"{np.mean(n_tail_hydro) * 100:.1f}%")

        axa_strict = sum(1 for k in axa_cnt if k == "strict")
        axa_loose  = sum(1 for k in axa_cnt if k in ("strict", "loose"))
        strict_n = sum(c for k, c in Counter(r["header"]["AXA"] for r in accepted_rows).items()
                       if k.startswith("strict("))
        loose_n  = sum(c for k, c in Counter(r["header"]["AXA"] for r in accepted_rows).items()
                       if k.startswith("loose("))
        add(f"  C-cleavage loose AXA  (train set 56.2%): "
            f"{loose_n} / {len(accepted_rows)} = {loose_n / len(accepted_rows) * 100:.1f}%")
        add(f"  C-cleavage strict AXA (train set 16.0%): "
            f"{strict_n} / {len(accepted_rows)} = {strict_n / len(accepted_rows) * 100:.1f}%")
        add("")

        add("-" * 70)
        add("Heuristic truncation usage (lower is better; 0 means EOS was fully learned)")
        add("-" * 70)
        none_n = sum(1 for r in accepted_rows if r["header"]["trunc"] == "none")
        add(f"  Not truncated   : {none_n} / {len(accepted_rows)} = "
            f"{none_n / len(accepted_rows) * 100:.1f}%")
        if not trunc_counter:
            add("  (0 truncated samples)")
        for note, c in trunc_counter.most_common():
            add(f"  {note:<40s} {c:>5d}  {c / len(accepted_rows) * 100:5.1f}%")
        add("")

        add("=" * 70)
        add("  Automatic conclusion")
        add("=" * 70)
        issues = []
        if eos_cnt.get("yes", 0) / len(accepted_rows) < 0.7:
            issues.append("EOS termination rate <70%: model has not fully learned to stop after the C-region")
        if n_1bare / len(accepted_rows) > 0.15:
            issues.append("N_len=1 (bare M) >15%: N-region content still tends to collapse")
        if ratios and np.mean(ratios) < 0.2:
            issues.append("N-head KR <20%: N-region positive-charge feature not learned (train set 30.9%)")
        if loose_n / len(accepted_rows) < 0.4:
            issues.append("C loose AXA <40%: AXA cleavage-site pattern not fitted (train set 56.2%)")
        if none_n / len(accepted_rows) < 0.8:
            issues.append(">20% of samples heuristically truncated: C-region degeneration still frequent")
        if not issues:
            add("All core metrics passed. Weighted training is effective; the model has learned the true N/H/C distributions.")
        else:
            for iss in issues:
                add(iss)
            add("")
            add("Suggestions if these issues persist:")
            add("   (1) increase MAX_STEPS to 30000;")
            add("   (2) in REGION_WEIGHTS set C_content=4.5, EOS=20, N_content=3.0;")
            add("   (3) consider retraining with WITH_DEDICATED_EOS_SP=True.")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] report written: {path}")


def main():
    global MODEL_SEED, CHECKPOINT_NAME
    parser = argparse.ArgumentParser(description="Region-weighted signal peptide generator")
    parser.add_argument("--N", type=int, default=100, help="Number of validated sequences to generate")
    parser.add_argument("--seed", type=int, default=MODEL_SEED,
                        help="Model seed to use (42/123/999, default 42)")
    parser.add_argument("--batch", type=int, default=16, help="Prompt batch size per forward pass")
    parser.add_argument("--attempt-max", type=int, default=50,
                        help="Maximum number of generation rounds")
    parser.add_argument("--per-prompt", type=int, default=1, help="Number of returned sequences per prompt")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Force a specific checkpoint name (e.g. checkpoint-12000); default: latest")
    args = parser.parse_args()

    MODEL_SEED = args.seed
    if args.checkpoint:
        CHECKPOINT_NAME = args.checkpoint

    target_N = args.N
    batch_sz = args.batch
    per_prompt = args.per_prompt
    max_attempt_rounds = args.attempt_max

    ckpt_path = get_checkpoint_path(MODEL_SEED)
    ckpt_step = ckpt_path.name.split("-")[-1] if ckpt_path.name.startswith("checkpoint-") else ckpt_path.name
    out_prefix = f"SP1_weighted_{target_N}_seed{MODEL_SEED}_ckpt{ckpt_step}"
    fasta_path = Path(OUTPUT_DIR) / f"{out_prefix}.fasta"
    report_path = Path(OUTPUT_DIR) / f"{out_prefix}_report.txt"

    tok, model = load_model_and_tokenizer(MODEL_SEED)

    accepted_rows = []
    total_attempt = 0
    pbar = tqdm(total=target_N, desc="accepted", ncols=100, dynamic_ncols=True)

    round_i = 0
    try:
        while len(accepted_rows) < target_N and round_i < max_attempt_rounds:
            round_i += 1
            prompts = [PROMPT_STRING for _ in range(batch_sz)]
            try:
                generated = generate_batch(tok, model, prompts, per_prompt)
            except torch.cuda.OutOfMemoryError:
                print(f"[warn] OOM in round {round_i}; reducing batch size from {batch_sz} to {max(1, batch_sz // 2)} and retrying")
                batch_sz = max(1, batch_sz // 2)
                if batch_sz == 0:
                    raise
                continue

            for raw in generated:
                total_attempt += 1
                ok, seq, hdr_extra, reason = process_one(PROMPT_STRING + raw)
                if not ok:
                    REJECT_REASON_COUNTER[reason] += 1
                    continue
                accepted_rows.append({"seq": seq, "header": hdr_extra})
                pbar.update(1)
                if len(accepted_rows) >= target_N:
                    break
            pbar.set_postfix_str(f"attempts={total_attempt}  accept_rate={len(accepted_rows) / max(1, total_attempt) * 100:.1f}%")
    finally:
        pbar.close()

    print(f"[finish] target {target_N}, accepted {len(accepted_rows)}, total attempts {total_attempt}")

    fasta_lines = []
    for i, r in enumerate(accepted_rows, 1):
        h = r["header"]
        name = (f">SPw_{i}|seed{MODEL_SEED}|ckpt{ckpt_step}"
                f"|N={h['N_len']}|H={h['H_len']}|C={h['C_len']}"
                f"|len={h['total_len']}|AXA={h['AXA']}|EOS={h['EOS']}|trunc={h['trunc']}")
        fasta_lines.append(name)
        s = r["seq"]
        for k in range(0, len(s), 60):
            fasta_lines.append(s[k:k + 60])
    fasta_path.write_text("\n".join(fasta_lines) + "\n", encoding="utf-8")
    print(f"[done] FASTA written: {fasta_path}")

    write_report(report_path, target_N, MODEL_SEED, total_attempt,
                 accepted_rows, REJECT_REASON_COUNTER, TRUNC_NOTE_COUNTER)

    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
