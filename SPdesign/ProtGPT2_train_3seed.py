#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import re
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

MODEL_NAME = "/root/autodl-tmp/ProtGPT2"

TRAIN_PATH = "/root/autodl-tmp/SP/train_cdhit.txt"
VAL_PATH   = "/root/autodl-tmp/SP/val_cdhit.txt"
TEST_PATH  = "/root/autodl-tmp/SP/test_cdhit.txt"

OUTPUT_DIR = "/root/autodl-tmp/SP/SPTRAIN_REGION_3SEED"

SEEDS = [42, 123, 999]

REGION_WEIGHTS = {
    "N_content": 6.0,
    "H_content": 2.0,
    "C_content": 5.0,
    "EOS":       15.0,
    "EOS_SP":    25.0,
    "TAG":       2.0,
}

USE_WEIGHT_SCHEDULE = False
WEIGHT_SCHEDULE_RAMP_FROM_STEP = 0
WEIGHT_SCHEDULE_RAMP_TO_STEP   = 1500

EOS_SP_COLD_START_UNTIL_STEP = 2000
EOS_SP_COLD_START_SCALE      = 1.6

WITH_DEDICATED_EOS_SP = True

MAX_LENGTH = 80
MAX_STEPS = 15000

BATCH_SIZE = 2
GRAD_ACCUM = 16

LR = 2e-5
WEIGHT_DECAY = 0.05
WARMUP_STEPS = 600

BASE_SPECIAL_TOKENS = ["<Nregion>", "<Hregion>", "<Cregion>", "<|endoftext|>"]
DEDICATED_EOS_SP_TOKEN = "<eos_sp>"


def set_all_seeds(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


N_TAG_RE = re.escape("<Nregion>")
H_TAG_RE = re.escape("<Hregion>")
C_TAG_RE = re.escape("<Cregion>")

REGION_ID_PAD = 0
REGION_ID_N_CONTENT = 1
REGION_ID_H_CONTENT = 2
REGION_ID_C_CONTENT = 3
REGION_ID_EOS = 4
REGION_ID_TAG = 5
REGION_ID_EOS_SP = 6


def find_tag_positions(text: str):
    def _find(tag):
        i = text.find(tag)
        if i == -1:
            return None
        return i, i + len(tag)
    return {
        "N": _find("<Nregion>"),
        "H": _find("<Hregion>"),
        "C": _find("<Cregion>"),
        "EOS": _find("<|endoftext|>"),
        "EOS_SP": _find("<eos_sp>"),
    }


def build_char_region_ids(text: str):
    n = len(text)
    arr = [REGION_ID_PAD] * n
    pos = find_tag_positions(text)

    def _mark_tag(start, end, kind):
        if start is None:
            return
        for i in range(max(0, start), min(end, n)):
            arr[i] = kind

    if pos["N"] is not None:
        _mark_tag(pos["N"][0], pos["N"][1], REGION_ID_TAG)
    if pos["H"] is not None:
        _mark_tag(pos["H"][0], pos["H"][1], REGION_ID_TAG)
    if pos["C"] is not None:
        _mark_tag(pos["C"][0], pos["C"][1], REGION_ID_TAG)
    if pos["EOS"] is not None:
        _mark_tag(pos["EOS"][0], pos["EOS"][1], REGION_ID_EOS)
    if pos["EOS_SP"] is not None:
        _mark_tag(pos["EOS_SP"][0], pos["EOS_SP"][1], REGION_ID_EOS_SP)

    def _mark_content(after_tag_pos, before_tag_pos_or_eos, rid):
        if after_tag_pos is None:
            return
        start = after_tag_pos[1]
        if before_tag_pos_or_eos is None:
            end = n
        else:
            end = before_tag_pos_or_eos[0]
        for i in range(max(0, start), min(end, n)):
            arr[i] = rid

    eos_pos = pos["EOS_SP"] if pos["EOS_SP"] is not None else pos["EOS"]
    _mark_content(pos["N"], pos["H"], REGION_ID_N_CONTENT)
    _mark_content(pos["H"], pos["C"], REGION_ID_H_CONTENT)
    _mark_content(pos["C"], eos_pos, REGION_ID_C_CONTENT)
    return arr


def char_region_ids_to_token_region_ids(char_region_ids, tokenizer,
                                     encoded, original_text):
    token_ids_char = []
    for idx in range(len(encoded["input_ids"])):
        span = encoded.token_to_chars(idx)
        if span is None:
            token_ids_char.append(REGION_ID_PAD)
            continue
        s, e = span.start, span.end
        if e <= s:
            token_ids_char.append(REGION_ID_PAD)
            continue
        seg = char_region_ids[s:e]
        counts = defaultdict(int)
        for r in seg:
            counts[r] += 1
        best = max(counts.keys(), key=lambda k: (counts[k], -k))
        if any(r == REGION_ID_TAG for r in seg):
            best = REGION_ID_TAG
        elif any(r == REGION_ID_EOS for r in seg):
            best = REGION_ID_EOS
        elif any(r == REGION_ID_EOS_SP for r in seg):
            best = REGION_ID_EOS_SP
        token_ids_char.append(best)
    return token_ids_char


WEIGHT_MAP = {
    REGION_ID_N_CONTENT: REGION_WEIGHTS["N_content"],
    REGION_ID_H_CONTENT: REGION_WEIGHTS["H_content"],
    REGION_ID_C_CONTENT: REGION_WEIGHTS["C_content"],
    REGION_ID_EOS:       REGION_WEIGHTS["EOS"],
    REGION_ID_TAG:       REGION_WEIGHTS["TAG"],
    REGION_ID_EOS_SP:    REGION_WEIGHTS["EOS_SP"],
    REGION_ID_PAD:       0.0,
}


def compute_token_weights(token_region_ids_list: list[list[int]],
                          weight_scale: float = 1.0,
                          device: torch.device | None = None):
    max_len = max(len(x) for x in token_region_ids_list)
    bsz = len(token_region_ids_list)
    w = torch.ones(bsz, max_len, dtype=torch.float32, device=device)
    for bi, row in enumerate(token_region_ids_list):
        for ti, rid in enumerate(row):
            base = WEIGHT_MAP.get(rid, 1.0)
            if rid != REGION_ID_H_CONTENT and rid != REGION_ID_PAD:
                base = 1.0 + (base - 1.0) * weight_scale
            w[bi, ti] = base
    return w


_NORMALIZE_ENDOFLINE_RE = re.compile(r"<endoftext>")
_INSERT_EOS_SP_RE = re.compile(r"<Cregion>(.*?)<\|endoftext\|>")


def maybe_insert_eos_sp(line: str) -> str:
    line = _NORMALIZE_ENDOFLINE_RE.sub("<|endoftext|>", line)
    if not WITH_DEDICATED_EOS_SP:
        return line
    return _INSERT_EOS_SP_RE.sub(
        lambda m: f"<Cregion>{m.group(1)}<eos_sp><|endoftext|>",
        line
    )


def build_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    extra = list(BASE_SPECIAL_TOKENS)
    if WITH_DEDICATED_EOS_SP:
        extra.append(DEDICATED_EOS_SP_TOKEN)
    tok.add_special_tokens({"additional_special_tokens": extra})
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


from torch.utils.data import Dataset as TorchDataset


class RegionDataset(TorchDataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {
            "input_ids": list(self.samples[idx]["input_ids"]),
            "attention_mask": list(self.samples[idx]["attention_mask"]),
            "token_region_ids": list(self.samples[idx]["token_region_ids"]),
        }


def build_dataset(tok):
    with open(TRAIN_PATH) as f:
        train_txt = [maybe_insert_eos_sp(l.strip()) for l in f if l.strip()]
    with open(VAL_PATH) as f:
        val_txt   = [maybe_insert_eos_sp(l.strip()) for l in f if l.strip()]
    with open(TEST_PATH) as f:
        test_txt  = [maybe_insert_eos_sp(l.strip()) for l in f if l.strip()]

    def process_texts(texts):
        out_list = []
        for line in texts:
            enc = tok(
                line, truncation=True, max_length=MAX_LENGTH,
                return_offsets_mapping=True,
                return_attention_mask=True,
            )
            if not enc["input_ids"]:
                continue
            char_rids = build_char_region_ids(line)
            tok_rids = char_region_ids_to_token_region_ids(
                char_rids, tok, enc, line
            )
            d = {
                "input_ids": [int(x) for x in enc["input_ids"]],
                "attention_mask": [int(x) for x in enc["attention_mask"]],
                "token_region_ids": [int(x) for x in tok_rids],
            }
            assert len(d["input_ids"]) == len(d["attention_mask"]) == len(d["token_region_ids"]), \
                f"length mismatch {len(d['input_ids'])} {len(d['attention_mask'])} {len(d['token_region_ids'])}"
            out_list.append(d)
        return out_list

    ds_train = RegionDataset(process_texts(train_txt))
    ds_val   = RegionDataset(process_texts(val_txt))
    ds_test  = RegionDataset(process_texts(test_txt))
    print(f"[dataset] train={len(ds_train)}  val={len(ds_val)}  test={len(ds_test)}")
    return {"train": ds_train, "validation": ds_val, "test": ds_test}


class RegionAwareDataCollator:
    def __init__(self, tokenizer, mlm: bool = False,
                 pad_to_multiple_of: int | None = None):
        self.base = DataCollatorForLanguageModeling(
            tokenizer, mlm=mlm, pad_to_multiple_of=pad_to_multiple_of)
        self.tokenizer = tokenizer
        self.pad_value_rid = REGION_ID_PAD

    def __call__(self, features):
        rid_list = [f.pop("token_region_ids").tolist()
                    if torch.is_tensor(f["token_region_ids"])
                    else list(f.pop("token_region_ids"))
                    for f in features]

        batch = self.base(features)
        bsz, maxlen = batch["input_ids"].shape

        rid_pad = torch.full(
            (bsz, maxlen), fill_value=self.pad_value_rid, dtype=torch.long
        )
        for i, row in enumerate(rid_list):
            L = min(len(row), maxlen)
            rid_pad[i, :L] = torch.tensor(row[:L], dtype=torch.long)
        batch["token_region_ids"] = rid_pad
        return batch


class RegionWeightedTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_global_step_for_weight = 0

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        step = self.state.global_step if hasattr(self, "state") else 0
        token_rids = inputs.pop("token_region_ids", None)

        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")

        if logits is None or labels is None:
            return (outputs, outputs) if return_outputs else outputs

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        vocab = shift_logits.size(-1)
        loss_flat = F.cross_entropy(
            shift_logits.view(-1, vocab),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        )
        loss_flat = loss_flat.view(shift_labels.shape)

        if token_rids is None:
            token_w = torch.ones_like(loss_flat, dtype=torch.float32)
            scale = 1.0
        else:
            if torch.is_tensor(token_rids):
                shift_rids = token_rids[..., 1:].contiguous()
            else:
                shift_rids = torch.tensor(
                    [row[1:] for row in token_rids],
                    dtype=torch.long,
                    device=loss_flat.device,
                )
            if USE_WEIGHT_SCHEDULE:
                scale = self._weight_scale(step)
            else:
                scale = 1.0
            token_w = compute_token_weights(
                shift_rids.cpu().tolist(),
                weight_scale=scale,
                device=loss_flat.device,
            )
            if WITH_DEDICATED_EOS_SP and step < EOS_SP_COLD_START_UNTIL_STEP:
                eos_sp_mask = (shift_rids == REGION_ID_EOS_SP).to(token_w.dtype)
                token_w = token_w * (1.0 + (EOS_SP_COLD_START_SCALE - 1.0) * eos_sp_mask)

        mask = (shift_labels != -100).to(loss_flat.dtype)
        token_w = token_w * mask

        denom = token_w.sum().clamp(min=1.0)
        weighted_loss = (loss_flat * token_w).sum() / denom
        return (weighted_loss, outputs) if return_outputs else weighted_loss

    @staticmethod
    def _weight_scale(global_step: int) -> float:
        if not USE_WEIGHT_SCHEDULE:
            return 1.0
        a, b = WEIGHT_SCHEDULE_RAMP_FROM_STEP, WEIGHT_SCHEDULE_RAMP_TO_STEP
        if global_step <= a:
            return 0.0
        if global_step >= b:
            return 1.0
        return (global_step - a) / (b - a)


class RegionLossMonitorCallback(TrainerCallback):

    REGION_OF_INTEREST = [
        ("N_content", REGION_ID_N_CONTENT),
        ("H_content", REGION_ID_H_CONTENT),
        ("C_content", REGION_ID_C_CONTENT),
        ("EOS",       REGION_ID_EOS),
        ("EOS_SP",    REGION_ID_EOS_SP),
        ("TAG",       REGION_ID_TAG),
    ]

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        eval_dataloader = kwargs.get("eval_dataloader")
        if model is None or eval_dataloader is None:
            return
        device = args.device
        model.eval()

        totals = defaultdict(float)
        counts = defaultdict(int)
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(device) if torch.is_tensor(v) else v
                         for k, v in batch.items()}
                token_rids = batch.pop("token_region_ids", None)
                out = model(**batch)
                logits = out.logits
                labels = batch["labels"]

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                ce = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100, reduction="none",
                ).view(shift_labels.shape)
                shift_rids_raw = token_rids
                if shift_rids_raw is None:
                    shift_rids = None
                elif torch.is_tensor(shift_rids_raw):
                    shift_rids = shift_rids_raw[..., 1:].contiguous()
                else:
                    shift_rids = torch.tensor(
                        [row[1:] for row in shift_rids_raw],
                        dtype=torch.long,
                        device=device,
                    )

                for rname, rid in self.REGION_OF_INTEREST:
                    if shift_rids is None:
                        continue
                    m = (shift_rids == rid) & (shift_labels != -100)
                    c = m.sum().item()
                    if c <= 0:
                        continue
                    totals[rname] += ce[m].sum().item()
                    counts[rname] += c
        if counts:
            log_extra = {}
            for rname, _ in self.REGION_OF_INTEREST:
                c = counts[rname]
                if c > 0:
                    log_extra[f"val/{rname}_loss"] = totals[rname] / c
            if hasattr(state, "log_history"):
                pass
            try:
                self.control = control
                parts = [f"{k}={v:.4f}" for k, v in log_extra.items()]
                print("\n[RegionMonitor step=%d val_region_loss  " % (state.global_step,)
                      + "  ".join(parts))
            except Exception:
                pass
        model.train(mode=model.training)


def build_model(tok):
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tok))
    return model


def build_trainer(seed):
    set_all_seeds(seed)
    tok = build_tokenizer()
    data = build_dataset(tok)
    model = build_model(tok)

    collator = RegionAwareDataCollator(tok, mlm=False)
    out_dir = Path(OUTPUT_DIR) / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=MAX_STEPS,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=50,
        save_steps=300,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        load_best_model_at_end=False,
    )

    trainer = RegionWeightedTrainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        data_collator=collator,
        callbacks=[RegionLossMonitorCallback()],
    )
    return trainer, data, tok


def run(seed):
    trainer, data, tok = build_trainer(seed)

    out_dir = Path(OUTPUT_DIR) / f"seed_{seed}"
    last_checkpoint = None
    if out_dir.exists():
        checkpoints = sorted(out_dir.glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[-1]))
        if checkpoints:
            last_checkpoint = str(checkpoints[-1])
            print(f"[resume] seed_{seed}: resuming from {last_checkpoint}")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    test_eval = trainer.evaluate(data["test"])
    result = {
        "seed": seed,
        "test_loss": float(test_eval.get("eval_loss", float("nan"))),
    }
    from torch.utils.data import DataLoader
    device = trainer.args.device
    model = trainer.model
    collator = trainer.data_collator
    test_ds = data["test"]
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, collate_fn=collator,
                      shuffle=False, num_workers=0)

    totals = defaultdict(float)
    counts = defaultdict(int)
    rois = [
        ("N_content", REGION_ID_N_CONTENT),
        ("H_content", REGION_ID_H_CONTENT),
        ("C_content", REGION_ID_C_CONTENT),
        ("EOS",       REGION_ID_EOS),
        ("TAG",       REGION_ID_TAG),
    ]
    with torch.no_grad():
        model.eval()
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            rids = batch.pop("token_region_ids", None)
            out = model(**batch)
            logits = out.logits
            labels = batch["labels"]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1), ignore_index=-100,
                reduction="none",
            ).view(shift_labels.shape)
            shift_rids = rids[..., 1:].contiguous() if rids is not None else None
            for rname, rid in rois:
                if shift_rids is None: continue
                m = (shift_rids == rid) & (shift_labels != -100)
                c = m.sum().item()
                if c <= 0: continue
                totals[rname] += ce[m].sum().item()
                counts[rname] += c
    for rname, _ in rois:
        c = counts[rname]
        if c > 0:
            result[f"test_{rname}_loss"] = totals[rname] / c

    del trainer, data
    cleanup()
    return result


def main():
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    meta = {
        "REGION_WEIGHTS": REGION_WEIGHTS,
        "USE_WEIGHT_SCHEDULE": USE_WEIGHT_SCHEDULE,
        "WEIGHT_SCHEDULE": (WEIGHT_SCHEDULE_RAMP_FROM_STEP,
                            WEIGHT_SCHEDULE_RAMP_TO_STEP),
        "WITH_DEDICATED_EOS_SP": WITH_DEDICATED_EOS_SP,
        "MAX_LENGTH": MAX_LENGTH,
        "MAX_STEPS": MAX_STEPS,
        "BATCH_SIZE": BATCH_SIZE,
        "GRAD_ACCUM": GRAD_ACCUM,
        "LR": LR,
        "WARMUP_STEPS": WARMUP_STEPS,
        "SEEDS": SEEDS,
    }
    with open(Path(OUTPUT_DIR) / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    results = []
    for seed in SEEDS:
        print(f"===== SEED {seed} =====", flush=True)
        results.append(run(seed))

    summary = {"results": results}
    for r in results:
        pass
    ks = [k for k in results[0].keys() if k.endswith("_loss")]
    for k in ks:
        vals = [r[k] for r in results if k in r]
        summary[f"mean_{k}"] = float(np.mean(vals))
        summary[f"std_{k}"]  = float(np.std(vals))

    with open(Path(OUTPUT_DIR) / "FINAL_REPORT.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
