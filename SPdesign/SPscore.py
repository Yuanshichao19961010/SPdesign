#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

CARGO_DEFAULT = "ADRAVAPREKVTKKVTKVVTVKKKHPKKKPKQKVYKPQKLPMKAERKLE"
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

KD_SCALE = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5,
    "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8,
    "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2,
    "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5,
    "K": -3.9, "R": -4.5,
}

CLEAVAGE_RESIDUE_PREF = {
    "A": 1.00,
    "G": 0.85,
    "S": 0.85,
    "C": 0.80,
    "V": 0.70,
    "T": 0.70,
    "I": 0.35,
    "L": 0.30,
    "M": 0.30,
    "F": 0.20,
    "Y": 0.20,
    "W": 0.10,
    "N": 0.25,
    "Q": 0.20,
    "H": 0.15,
    "D": 0.05,
    "E": 0.05,
    "K": 0.05,
    "R": 0.05,
    "P": 0.00,
}


@dataclass
class Record:
    safe_id: str
    original_header: str
    original_seq: str
    cargo_seq: str


def clean_sequence(seq: str) -> str:
    seq = re.sub(r"[^A-Za-z]", "", seq.upper())
    return "".join(aa if aa in VALID_AA else "X" for aa in seq)


def read_fasta(path: Path, cargo: str) -> List[Record]:
    records: List[Record] = []
    header: Optional[str] = None
    seq_lines: List[str] = []

    def flush() -> None:
        nonlocal header, seq_lines
        if header is None:
            return
        seq = clean_sequence("".join(seq_lines))
        if seq:
            safe_id = f"SP{len(records) + 1:09d}"
            records.append(Record(safe_id, header, seq, seq + cargo))
        header = None
        seq_lines = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:].strip()
            else:
                seq_lines.append(line)
        flush()
    return records


def write_fasta(records: Sequence[Record], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(f">{r.safe_id}\n")
            for i in range(0, len(r.cargo_seq), 80):
                f.write(r.cargo_seq[i:i + 80] + "\n")


def split_batches(records: Sequence[Record], outdir: Path, batch_size: int) -> pd.DataFrame:
    batch_dir = outdir / "01_signalp_batches"
    signalp_out_root = outdir / "02_signalp_outputs"
    batch_dir.mkdir(parents=True, exist_ok=True)
    signalp_out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for batch_no, start in enumerate(range(0, len(records), batch_size), start=1):
        batch = records[start:start + batch_size]
        fasta = batch_dir / f"signalp6_batch_{batch_no:04d}.fasta"
        batch_outdir = signalp_out_root / f"batch_{batch_no:04d}"
        batch_outdir.mkdir(parents=True, exist_ok=True)
        write_fasta(batch, fasta)
        rows.append({
            "batch": batch_no,
            "fasta": str(fasta),
            "signalp_outdir": str(batch_outdir),
            "n_sequences": len(batch),
        })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(outdir / "batch_manifest.csv", index=False)
    return manifest


def write_input_table(records: Sequence[Record], outdir: Path) -> pd.DataFrame:
    df = pd.DataFrame([asdict(r) for r in records])
    df.to_csv(outdir / "input_sequences_with_cargo.csv", index=False)
    with open(outdir / "input_safe_id_mapping.tsv", "w", encoding="utf-8") as f:
        f.write("safe_id\toriginal_header\toriginal_seq\tcargo_seq\n")
        for r in records:
            f.write(f"{r.safe_id}\t{r.original_header}\t{r.original_seq}\t{r.cargo_seq}\n")
    return df


def shell_quote(x: Any) -> str:
    return shlex.quote(str(x))


def build_signalp_command(args: argparse.Namespace, fasta: Path, outdir: Path) -> str:
    if args.signalp_template:
        return args.signalp_template.format(
            signalp_cmd=args.signalp_cmd,
            fasta=str(fasta),
            outdir=str(outdir),
            organism=args.organism,
            mode=args.mode,
            format=args.signalp_format,
            signalp_format=args.signalp_format,
            signalp_bsize=args.signalp_bsize,
            torch_threads=args.torch_threads,
            write_procs=args.write_procs,
        )

    parts = [
        shell_quote(args.signalp_cmd),
        "--fastafile", shell_quote(fasta),
        "--output_dir", shell_quote(outdir),
        "--organism", shell_quote(args.organism),
        "--mode", shell_quote(args.mode),
        "--format", shell_quote(args.signalp_format),
    ]
    if args.signalp_bsize is not None:
        parts += ["--bsize", str(args.signalp_bsize)]
    if args.torch_threads is not None:
        parts += ["--torch_num_threads", str(args.torch_threads)]
    if args.write_procs is not None:
        parts += ["--write_procs", str(args.write_procs)]
    return " ".join(parts)


def has_parseable_signalp_output(outdir: Path) -> bool:
    for p in outdir.rglob("*"):
        if p.is_file() and (p.name in {"prediction_results.txt", "region_output.gff3", "output.json"}
                            or p.suffix.lower() in {".json", ".gff3", ".txt", ".tsv", ".csv"}):
            return True
    return False


def run_signalp_batches(manifest: pd.DataFrame, args: argparse.Namespace) -> None:
    for _, row in manifest.iterrows():
        batch = int(row["batch"])
        fasta = Path(row["fasta"])
        outdir = Path(row["signalp_outdir"])
        outdir.mkdir(parents=True, exist_ok=True)
        done = outdir / ".signalp_done"
        if done.exists() and not args.overwrite_signalp:
            print(f"[SignalP] batch {batch}: already done")
            if args.stop_after_first_signalp:
                break
            continue

        cmd = build_signalp_command(args, fasta, outdir)
        log = outdir / "signalp_run.log"
        max_attempts = 3
        succeeded = False
        for attempt in range(1, max_attempts + 1):
            print(f"[SignalP] batch {batch}: attempt {attempt}/{max_attempts}: {cmd}")
            with open(log, "w", encoding="utf-8") as lf:
                lf.write("COMMAND:\n" + cmd + "\n\n")
                proc = subprocess.run(cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT)
            if proc.returncode == 0 and has_parseable_signalp_output(outdir):
                succeeded = True
                break
            print(f"[SignalP] batch {batch}: attempt {attempt} failed (rc={proc.returncode})")
            if attempt < max_attempts:
                print(f"[SignalP] batch {batch}: retrying in 30s ...")
                time.sleep(30)
        if not succeeded:
            raise RuntimeError(
                f"SignalP failed or produced no parseable output for batch {batch} "
                f"after {max_attempts} attempts. Check: {log}\n"
                f"If your SignalP flags differ, rerun with --signalp_template."
            )
        done.write_text("done\n", encoding="utf-8")
        if args.stop_after_first_signalp:
            break


def extract_cleavage_site(text: str) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"(CS|cleavage|site|pos)[^0-9]{0,30}(\d+)(?:\s*-\s*(\d+))?", text, re.I)
    if m:
        return int(m.group(2))
    m = re.search(r"\b(\d+)\s*-\s*\d+\b", text)
    return int(m.group(1)) if m else None


def normalize_prediction_dict(d: Dict[str, str], path: Path) -> Optional[Dict[str, Any]]:
    lower = {k.lower(): k for k in d}

    def get(*names: str) -> Optional[str]:
        for n in names:
            if n.lower() in lower:
                return d[lower[n.lower()]]
        return None

    seq_id = get("ID", "Name", "Protein", "seq_id")
    if not seq_id:
        return None
    pred = get("Prediction", "pred", "class") or ""

    preferred: List[float] = []
    candidates: List[float] = []
    cs_probs: List[float] = []
    for k, v in d.items():
        try:
            x = float(v)
        except Exception:
            continue
        if not 0 <= x <= 1:
            continue
        lk = k.lower()
        if "sec/spi" in lk or "spi" in lk:
            preferred.append(x)
        if "prob" in lk or "pr" in lk or "signal" in lk or "/" in lk or "other" in lk:
            candidates.append(x)
        if any(t in lk for t in ["cleavage probability", "cs probability", "cs_prob", "cleavage_prob"]):
            cs_probs.append(x)

    prob = max(preferred) if preferred else (max(candidates) if candidates else None)
    cs = get("CS Position", "CS_pos", "Cleavage site", "Cleavage_site", "CS")
    out = {
        "safe_id": str(seq_id).split()[0],
        "prediction": pred,
        "signalp_prob": prob,
        "cleavage_site": extract_cleavage_site(cs or " ".join(d.values())),
        "prediction_source": str(path),
    }
    if cs_probs:
        out["cleavage_prob"] = max(cs_probs)
    return out


def parse_prediction_results(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    header: Optional[List[str]] = None
    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        if raw.startswith("#"):
            h = raw.lstrip("#").strip()
            if re.search(r"\b(ID|Prediction|OTHER|Sec/SPI|CS Position|CS_pos)\b", h, re.I):
                header = re.split(r"\t+|\s{2,}", h)
            continue

        if header:
            parts = raw.split("\t") if "\t" in raw else re.split(r"\s{2,}", raw)
            if len(parts) == len(header):
                row = normalize_prediction_dict({k.strip(): v.strip() for k, v in zip(header, parts)}, path)
                if row:
                    rows.append(row)
                continue

        parts = raw.split("\t") if "\t" in raw else re.split(r"\s+", raw)
        if len(parts) < 2 or parts[0].lower() in {"id", "name"}:
            continue
        probs = []
        for tok in parts[1:]:
            try:
                x = float(tok)
                if 0 <= x <= 1:
                    probs.append(x)
            except Exception:
                pass
        joined = " ".join(parts[1:])
        rows.append({
            "safe_id": parts[0],
            "prediction": joined,
            "signalp_prob": max(probs) if probs else None,
            "cleavage_site": extract_cleavage_site(joined),
            "prediction_source": str(path),
        })
    return rows


def parse_region_gff3(path: Path) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 9:
                continue
            seqid, _, ftype, start, end, _, _, _, attrs = parts[:9]
            text = (ftype + " " + attrs).lower()
            region = None
            if "n-region" in text or "n_region" in text or ftype.lower() == "n":
                region = "n"
            elif "h-region" in text or "h_region" in text or ftype.lower() == "h":
                region = "h"
            elif "c-region" in text or "c_region" in text or ftype.lower() == "c":
                region = "c"
            if not region:
                continue
            row = by_id.setdefault(seqid, {"safe_id": seqid, "region_source": str(path)})
            row[f"{region}_start"] = int(start)
            row[f"{region}_end"] = int(end)
    return list(by_id.values())


def find_first_key(obj: Any, names: Sequence[str]) -> Optional[Any]:
    if not isinstance(obj, dict):
        return None
    lower = {str(k).lower(): k for k in obj}
    for n in names:
        if n.lower() in lower:
            return obj[lower[n.lower()]]
    return None


def find_prob(obj: Any) -> Optional[float]:
    pref, any_vals = [], []

    def rec(o: Any, ctx: str = "") -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                nctx = (ctx + " " + str(k)).lower()
                if isinstance(v, (int, float)) and 0 <= float(v) <= 1:
                    if "sec/spi" in nctx or "spi" in nctx:
                        pref.append(float(v))
                    if any(t in nctx for t in ["prob", "pr", "score", "signal", "sec"]):
                        any_vals.append(float(v))
                else:
                    rec(v, nctx)
        elif isinstance(o, list):
            for x in o:
                rec(x, ctx)

    rec(obj)
    return max(pref) if pref else (max(any_vals) if any_vals else None)


def find_cleavage(obj: Any) -> Optional[int]:
    found: List[int] = []

    def rec(o: Any, ctx: str = "") -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                nctx = (ctx + " " + str(k)).lower()
                if any(t in nctx for t in ["cleavage", "cs", "cut"]):
                    if isinstance(v, int):
                        found.append(v)
                    elif isinstance(v, str):
                        x = extract_cleavage_site(v)
                        if x is not None:
                            found.append(x)
                rec(v, nctx)
        elif isinstance(o, list):
            for x in o:
                rec(x, ctx)

    rec(obj)
    return found[0] if found else None


def find_regions_in_obj(obj: Any) -> Dict[str, int]:
    out: Dict[str, int] = {}

    def parse_range_val(v: Any) -> Optional[Tuple[int, int]]:
        if isinstance(v, str):
            nums = re.findall(r"\d+", v)
            if len(nums) >= 2:
                return int(nums[0]), int(nums[1])
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            try:
                return int(v[0]), int(v[1])
            except Exception:
                return None
        if isinstance(v, dict):
            s = find_first_key(v, ["start", "begin", "from"])
            e = find_first_key(v, ["end", "stop", "to"])
            try:
                return (int(s), int(e)) if s is not None and e is not None else None
            except Exception:
                return None
        return None

    def norm_label(s: str) -> Optional[str]:
        sl = s.lower()
        if "n-region" in sl or "n_region" in sl or sl in {"n", "nregion"}:
            return "n"
        if "h-region" in sl or "h_region" in sl or sl in {"h", "hregion"}:
            return "h"
        if "c-region" in sl or "c_region" in sl or sl in {"c", "cregion"}:
            return "c"
        return None

    def rec(o: Any) -> None:
        if isinstance(o, dict):
            for lab in ["n", "h", "c"]:
                s = find_first_key(o, [f"{lab}_start", f"{lab}-start"])
                e = find_first_key(o, [f"{lab}_end", f"{lab}-end"])
                if s is not None and e is not None:
                    try:
                        out[f"{lab}_start"] = int(s)
                        out[f"{lab}_end"] = int(e)
                    except Exception:
                        pass
            for k, v in o.items():
                lab = norm_label(str(k))
                if lab:
                    rng = parse_range_val(v)
                    if rng:
                        out[f"{lab}_start"], out[f"{lab}_end"] = rng
                rec(v)
        elif isinstance(o, list):
            for x in o:
                rec(x)

    rec(obj)
    return out


def parse_output_json(path: Path) -> List[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return []
    rows: List[Dict[str, Any]] = []

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            keys_lower = {str(k).lower(): k for k in obj}
            id_key = next((keys_lower[k] for k in ["id", "name", "seq_id", "sequence_id", "protein"] if k in keys_lower), None)
            if id_key is not None:
                row = {"safe_id": str(obj[id_key]).split()[0], "json_source": str(path)}
                pred = find_first_key(obj, ["prediction", "pred", "class", "type"])
                if pred is not None:
                    row["prediction"] = str(pred)
                prob = find_prob(obj)
                if prob is not None:
                    row["signalp_prob"] = prob
                cs = find_cleavage(obj)
                if cs is not None:
                    row["cleavage_site"] = cs
                row.update(find_regions_in_obj(obj))
                if len(row) > 2:
                    rows.append(row)
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for x in obj:
                walk(x)

    walk(data)
    return rows


def parse_signalp_outputs(output_root: Path) -> pd.DataFrame:
    pred_rows: List[Dict[str, Any]] = []
    reg_rows: List[Dict[str, Any]] = []
    json_rows: List[Dict[str, Any]] = []

    for p in sorted(output_root.rglob("*")):
        if not p.is_file():
            continue
        if p.name == "prediction_results.txt":
            pred_rows.extend(parse_prediction_results(p))
        elif p.name == "region_output.gff3" or p.suffix.lower() == ".gff3":
            reg_rows.extend(parse_region_gff3(p))
        elif p.name == "output.json" or p.suffix.lower() == ".json":
            json_rows.extend(parse_output_json(p))
        elif p.suffix.lower() in {".txt", ".tsv", ".csv"} and "signalp_run" not in p.name:
            pred_rows.extend(parse_prediction_results(p))

    dfs = []
    if pred_rows:
        dfs.append(pd.DataFrame(pred_rows).drop_duplicates("safe_id", keep="first"))
    if reg_rows:
        dfs.append(pd.DataFrame(reg_rows).drop_duplicates("safe_id", keep="first"))
    if json_rows:
        dfs.append(pd.DataFrame(json_rows).drop_duplicates("safe_id", keep="first"))
    if not dfs:
        raise RuntimeError(f"No parseable SignalP output found under {output_root}")

    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.merge(d, on="safe_id", how="outer", suffixes=("", "__new"))
        for col in list(merged.columns):
            if col.endswith("__new"):
                base = col[:-5]
                if base in merged.columns:
                    merged[base] = merged[base].combine_first(merged[col])
                    merged.drop(columns=[col], inplace=True)
                else:
                    merged.rename(columns={col: base}, inplace=True)
    return merged


def get_subseq_1based(seq: str, start: Any, end: Any) -> str:
    try:
        s = int(float(start))
        e = int(float(end))
    except Exception:
        return ""
    if s < 1 or e < s:
        return ""
    return str(seq)[s - 1:e]


def mean_scale(seq: str, scale: Dict[str, float]) -> Optional[float]:
    vals = [scale[a] for a in str(seq) if a in scale]
    return sum(vals) / len(vals) if vals else None


def net_charge(seq: str) -> int:
    s = str(seq)
    return sum(1 for a in s if a in "KR") - sum(1 for a in s if a in "DE")


def gaussian_suitability(x: Any, center: float, sigma: float) -> Optional[float]:
    try:
        x = float(x)
    except Exception:
        return None
    if not math.isfinite(x) or sigma <= 0:
        return None
    return math.exp(-((x - center) ** 2) / (2.0 * sigma ** 2))


def clip01(x: Any) -> Optional[float]:
    try:
        x = float(x)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return max(0.0, min(1.0, x))


def h_region_charge_score(h_region: str, penalty_lambda: float) -> float:
    n_charged = sum(1 for aa in str(h_region) if aa in "DEKR")
    return math.exp(-penalty_lambda * n_charged)


def cleavage_motif_suitability(seq: str, cleavage_site: Any = None) -> float:
    try:
        cs = int(float(cleavage_site))
    except Exception:
        cs = None

    if cs is not None and 3 <= cs <= len(seq):
        minus3 = seq[cs - 3]
        minus1 = seq[cs - 1]
        return 0.5 * CLEAVAGE_RESIDUE_PREF.get(minus3, 0.0) + 0.5 * CLEAVAGE_RESIDUE_PREF.get(minus1, 0.0)

    tail = str(seq)[-12:]
    best = 0.0
    for i in range(max(0, len(tail) - 2)):
        a, b = tail[i], tail[i + 2]
        sc = 0.5 * CLEAVAGE_RESIDUE_PREF.get(a, 0.0) + 0.5 * CLEAVAGE_RESIDUE_PREF.get(b, 0.0)
        best = max(best, sc)
    return best


def c_region_polarity_score(c_region: str, args: argparse.Namespace) -> Optional[float]:
    kd = mean_scale(c_region, KD_SCALE)
    if kd is None:
        return None
    denom = args.c_kd_bad - args.c_kd_good
    if denom <= 0:
        raise ValueError("--c_kd_bad must be greater than --c_kd_good")
    return clip01((args.c_kd_bad - kd) / denom)


def residue_preference_score(seq: str, cleavage_site: Any, offset: int) -> float:
    try:
        cs = int(float(cleavage_site))
    except Exception:
        return 0.0
    pos_1based = cs + offset + 1
    if pos_1based < 1 or pos_1based > len(seq):
        return 0.0
    aa = str(seq)[pos_1based - 1]
    return float(CLEAVAGE_RESIDUE_PREF.get(aa, 0.0))


def aa_fraction(seq: str, aa: str) -> float:
    seq = str(seq)
    if not seq:
        return 0.0
    return seq.count(aa) / len(seq)


def saturation_score(value: Any, target: float) -> Optional[float]:
    try:
        value = float(value)
    except Exception:
        return None
    if target <= 0:
        return None
    return clip01(value / target)


def sigmoid_score(x: Any, midpoint: float = 1.0, slope: float = 1.4) -> Optional[float]:
    try:
        x = float(x)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    z = max(-60.0, min(60.0, slope * (x - midpoint)))
    return 1.0 / (1.0 + math.exp(-z))


def trapezoid_suitability(x: Any, low0: float, low1: float, high1: float, high0: float) -> Optional[float]:
    try:
        x = float(x)
    except Exception:
        return None
    if not (low0 <= low1 <= high1 <= high0):
        return None
    if x <= low0 or x >= high0:
        return 0.0
    if low1 <= x <= high1:
        return 1.0
    if x < low1:
        return (x - low0) / (low1 - low0) if low1 > low0 else 1.0
    return (high0 - x) / (high0 - high1) if high0 > high1 else 1.0


def hc_turn_score(h_region: str, c_region: str) -> float:
    boundary = str(h_region)[-3:] + str(c_region)[:3]
    n_turn = sum(1 for aa in boundary if aa in "GPSN")
    return min(1.0, n_turn / 2.0)


def h_gly_suitability(h_region: str, penalty_lambda: float = 0.7) -> float:
    n_gly = str(h_region).count("G")
    excess = max(0, n_gly - 1)
    return math.exp(-penalty_lambda * excess)


def h_end_positive_charge_score(h_region: str, window: int = 4, penalty_lambda: float = 1.0) -> float:
    tail = str(h_region)[-max(1, int(window)):]
    n_pos = sum(1 for aa in tail if aa in "KR")
    return math.exp(-penalty_lambda * n_pos)


def nh_hydrophobic_contrast_score(n_kd: Any, h_kd: Any, midpoint: float = 1.0, slope: float = 1.2) -> Optional[float]:
    try:
        contrast = float(h_kd) - float(n_kd)
    except Exception:
        return None
    return sigmoid_score(contrast, midpoint=midpoint, slope=slope)


def signalp_confidence_score(p: Any, min_prob: float) -> Optional[float]:
    try:
        p = float(p)
    except Exception:
        return None
    if p < min_prob:
        return 0.0
    if min_prob >= 1.0:
        return 1.0 if p >= 1.0 else 0.0
    return clip01((p - min_prob) / (1.0 - min_prob))


def is_sec_spi(pred: Any) -> bool:
    s = str(pred).lower().strip()
    return ("sec/spi" in s) or ("signal peptide" in s) or re.search(r"\bspi\b", s) is not None or s == "sp"


def compute_scores(input_df: pd.DataFrame, signalp_df: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = input_df.merge(signalp_df, on="safe_id", how="left")

    for col, default in [("signalp_prob", pd.NA), ("cleavage_prob", pd.NA), ("cleavage_site", pd.NA)]:
        if col not in df.columns:
            df[col] = default
    if "prediction" not in df.columns:
        df["prediction"] = ""

    df["signalp_prob"] = pd.to_numeric(df["signalp_prob"], errors="coerce")
    df["cleavage_prob"] = pd.to_numeric(df["cleavage_prob"], errors="coerce")
    df["is_sec_spi"] = df["prediction"].apply(is_sec_spi)

    if args.allow_region_only_signalp:
        eligible = df[df["signalp_prob"] >= args.min_signalp_prob].copy()
    else:
        eligible = df[(df["is_sec_spi"]) & (df["signalp_prob"] >= args.min_signalp_prob)].copy()

    for col in ["n_start", "n_end", "h_start", "h_end", "c_start", "c_end"]:
        if col not in eligible.columns:
            eligible[col] = pd.NA
        eligible[col] = pd.to_numeric(eligible[col], errors="coerce")

    eligible["n_region"] = eligible.apply(lambda r: get_subseq_1based(r["cargo_seq"], r["n_start"], r["n_end"]), axis=1)
    eligible["h_region"] = eligible.apply(lambda r: get_subseq_1based(r["cargo_seq"], r["h_start"], r["h_end"]), axis=1)
    eligible["c_region"] = eligible.apply(lambda r: get_subseq_1based(r["cargo_seq"], r["c_start"], r["c_end"]), axis=1)

    if args.require_h_region:
        eligible = eligible[eligible["h_region"].fillna("").astype(str).str.len() > 0].copy()
    if args.require_cleavage_site:
        eligible = eligible[pd.to_numeric(eligible["cleavage_site"], errors="coerce").notna()].copy()
    if eligible.empty:
        raise RuntimeError("No records remained after the SignalP gate.")

    eligible["sp_length"] = eligible["original_seq"].astype(str).str.len()
    eligible["n_length"] = eligible["n_region"].astype(str).str.len()
    eligible["h_length"] = eligible["h_region"].astype(str).str.len()
    eligible["c_length"] = eligible["c_region"].astype(str).str.len()

    eligible["GRAVY_SP"] = eligible["original_seq"].apply(lambda s: mean_scale(s, KD_SCALE))
    eligible["KD_N"] = eligible["n_region"].apply(lambda s: mean_scale(s, KD_SCALE))
    eligible["KD_H"] = eligible["h_region"].apply(lambda s: mean_scale(s, KD_SCALE))
    eligible["KD_C"] = eligible["c_region"].apply(lambda s: mean_scale(s, KD_SCALE))
    eligible["N_charge"] = eligible["n_region"].apply(net_charge)
    eligible["SP_charge"] = eligible["original_seq"].apply(net_charge)
    eligible["A_C_fraction"] = eligible["c_region"].apply(lambda s: aa_fraction(s, "A"))
    eligible["P_C_fraction"] = eligible["c_region"].apply(lambda s: aa_fraction(s, "P"))
    eligible["G_H_fraction"] = eligible["h_region"].apply(lambda s: aa_fraction(s, "G"))

    eligible["S_GRAVY_SP"] = eligible["GRAVY_SP"].apply(
        lambda x: gaussian_suitability(x, args.gravy_sp_center, args.gravy_sp_sigma)
    )
    eligible["S_H_KD"] = eligible["KD_H"].apply(
        lambda x: gaussian_suitability(x, args.h_kd_center, args.h_kd_sigma)
    )
    eligible["S_hydro"] = 0.40 * eligible["S_GRAVY_SP"] + 0.60 * eligible["S_H_KD"]

    eligible["S_minus1"] = [
        residue_preference_score(seq, cs, -1)
        for seq, cs in zip(eligible["original_seq"], eligible["cleavage_site"])
    ]
    eligible["S_minus3"] = [
        residue_preference_score(seq, cs, -3)
        for seq, cs in zip(eligible["original_seq"], eligible["cleavage_site"])
    ]
    eligible["S_A_C"] = eligible["A_C_fraction"].apply(lambda x: saturation_score(x, args.c_ala_target_fraction))
    eligible["S_P_C"] = eligible["P_C_fraction"].apply(lambda x: saturation_score(x, args.c_pro_target_fraction))
    eligible["S_C_polarity"] = eligible["c_region"].apply(lambda s: c_region_polarity_score(s, args))
    eligible["S_cleavage"] = (
        0.40 * eligible["S_minus1"]
        + 0.20 * eligible["S_minus3"]
        + 0.15 * eligible["S_A_C"]
        + 0.15 * eligible["S_P_C"]
        + 0.10 * eligible["S_C_polarity"]
    )

    eligible["S_turn_HC"] = [hc_turn_score(h, c) for h, c in zip(eligible["h_region"], eligible["c_region"])]
    eligible["S_G_H"] = eligible["h_region"].apply(lambda s: h_gly_suitability(s, args.h_gly_penalty))
    eligible["S_Hend_charge"] = eligible["h_region"].apply(
        lambda s: h_end_positive_charge_score(s, args.h_end_window, args.h_end_charge_penalty)
    )
    eligible["S_HC"] = (
        0.50 * eligible["S_turn_HC"]
        + 0.30 * eligible["S_G_H"]
        + 0.20 * eligible["S_Hend_charge"]
    )

    eligible["S_N_charge"] = eligible["N_charge"].apply(
        lambda x: sigmoid_score(x, midpoint=args.n_charge_midpoint, slope=args.n_charge_slope)
    )
    eligible["S_NH_contrast"] = [
        nh_hydrophobic_contrast_score(n, h, args.nh_contrast_midpoint, args.nh_contrast_slope)
        for n, h in zip(eligible["KD_N"], eligible["KD_H"])
    ]
    eligible["S_N"] = 0.70 * eligible["S_N_charge"] + 0.30 * eligible["S_NH_contrast"]

    eligible["S_SP_length"] = eligible["sp_length"].apply(
        lambda x: trapezoid_suitability(x, args.sp_len_low0, args.sp_len_low1, args.sp_len_high1, args.sp_len_high0)
    )
    eligible["S_H_length"] = eligible["h_length"].apply(
        lambda x: trapezoid_suitability(x, args.h_len_low0, args.h_len_low1, args.h_len_high1, args.h_len_high0)
    )
    eligible["S_length"] = 0.40 * eligible["S_SP_length"] + 0.60 * eligible["S_H_length"]

    eligible["S_SignalP"] = eligible["signalp_prob"].apply(
        lambda x: signalp_confidence_score(x, args.min_signalp_prob)
    )

    modules = ["S_hydro", "S_cleavage", "S_HC", "S_N", "S_length", "S_SignalP"]
    for col in modules:
        eligible[col] = pd.to_numeric(eligible[col], errors="coerce")
    scored = eligible.dropna(subset=modules).copy()
    if scored.empty:
        raise RuntimeError("No records remained after HT-SP component-score calculation.")

    weights = [float(x) for x in args.ht_weights.split(",")]
    if len(weights) != 6 or sum(weights) <= 0:
        raise ValueError("--ht_weights must contain six positive-sum values")
    weights = [w / sum(weights) for w in weights]
    w_hy, w_cl, w_hc, w_n, w_len, w_sp = weights

    scored["contrib_hydro"] = w_hy * scored["S_hydro"]
    scored["contrib_cleavage"] = w_cl * scored["S_cleavage"]
    scored["contrib_HC"] = w_hc * scored["S_HC"]
    scored["contrib_N"] = w_n * scored["S_N"]
    scored["contrib_length"] = w_len * scored["S_length"]
    scored["contrib_SignalP"] = w_sp * scored["S_SignalP"]

    scored["HT_SP_score"] = (
        scored["contrib_hydro"]
        + scored["contrib_cleavage"]
        + scored["contrib_HC"]
        + scored["contrib_N"]
        + scored["contrib_length"]
        + scored["contrib_SignalP"]
    ).clip(0.0, 1.0)
    scored["SP_score"] = scored["HT_SP_score"]

    scored = scored.sort_values("HT_SP_score", ascending=False).reset_index(drop=True)
    top_n = max(1, math.ceil(len(scored) * args.top_fraction))
    selected = scored.head(top_n).copy()
    return scored, selected

def write_outputs(scored: pd.DataFrame, selected: pd.DataFrame, outdir: Path, args: argparse.Namespace) -> None:
    scored.to_csv(outdir / "SP_all_scored.csv", index=False)
    selected.to_csv(outdir / "SP_selected_scored.csv", index=False)

    with open(outdir / "SP_selected.txt", "w", encoding="utf-8") as f:
        for seq in selected["original_seq"]:
            f.write(str(seq) + "\n")

    with open(outdir / "SP_selected.fasta", "w", encoding="utf-8") as f:
        for _, r in selected.iterrows():
            f.write(f">{r['original_header']}|safe_id={r['safe_id']}|HT_SP_score={r['HT_SP_score']:.6f}\n")
            seq = str(r["original_seq"])
            for i in range(0, len(seq), 80):
                f.write(seq[i:i + 80] + "\n")

    summary = {
        "scoring_model": "HT-SP_interpretable_high-throughput-informed_score_v1",
        "n_scored": int(len(scored)),
        "n_selected": int(len(selected)),
        "top_fraction_requested": float(args.top_fraction),
        "top_fraction_actual": float(len(selected) / len(scored)) if len(scored) else 0.0,
        "gate": {
            "min_signalp_prob": args.min_signalp_prob,
            "require_h_region": args.require_h_region,
            "require_cleavage_site": args.require_cleavage_site,
            "allow_region_only_signalp": args.allow_region_only_signalp,
        },
        "score_weights": dict(zip(["S_hydro", "S_cleavage", "S_HC", "S_N", "S_length", "S_SignalP"], [float(x) for x in args.ht_weights.split(",")])),
        "files": {
            "all_scored": "SP_all_scored.csv",
            "selected_csv": "SP_selected_scored.csv",
            "selected_fasta": "SP_selected.fasta",
            "selected_txt": "SP_selected.txt",
        },
    }
    (outdir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", default="SP_selected.txt", help="Input FASTA/TXT file of SP sequences (default: bundled SP_selected.txt)")
    p.add_argument("--outdir", default="SP_score_out", help="Output directory")
    p.add_argument("--cargo", default=CARGO_DEFAULT, help="Cargo peptide appended to each SP")
    p.add_argument("--batch_size", type=int, default=1000, help="Sequences per SignalP batch")

    p.add_argument("--run_signalp", action="store_true", help="Run local SignalP6")
    p.add_argument("--signalp_cmd", default="signalp6", help="SignalP6 executable")
    p.add_argument("--organism", default="eukarya", help="SignalP organism argument")
    p.add_argument("--mode", default="fast", choices=["fast", "slow", "slow-sequential"], help="SignalP mode")
    p.add_argument("--signalp_format", default="none", help="SignalP --format argument")
    p.add_argument("--signalp_bsize", type=int, default=64, help="SignalP --bsize")
    p.add_argument("--torch_threads", type=int, default=8, help="SignalP --torch_num_threads")
    p.add_argument("--write_procs", type=int, default=1, help="SignalP --write_procs")
    p.add_argument("--signalp_template", default=None, help="Custom SignalP command template")
    p.add_argument("--overwrite_signalp", action="store_true", help="Rerun batches even if done")
    p.add_argument("--stop_after_first_signalp", action="store_true", help="Test mode")
    p.add_argument("--prepare_only", action="store_true", help="Only create cargo FASTA batches")
    p.add_argument("--parse_only", action="store_true", help="Parse existing outputs only")

    p.add_argument("--min_signalp_prob", type=float, default=0.90, help="Minimum SignalP Sec/SPI probability")
    p.add_argument("--top_fraction", type=float, default=0.30, help="Top fraction to export")
    p.add_argument("--ht_weights", default="0.30,0.25,0.15,0.10,0.10,0.10",
                   help="S_hydro,S_cleavage,S_HC,S_N,S_length,S_SignalP")
    p.add_argument("--allow_region_only_signalp", action="store_true", help="Use probability+regions without explicit Sec/SPI text")
    p.add_argument("--no_require_h_region", action="store_false", dest="require_h_region", help="Do not require a parsed H-region")
    p.set_defaults(require_h_region=True)
    p.add_argument("--require_cleavage_site", action="store_true", default=False, help="Require a parsed cleavage site")

    p.add_argument("--gravy_sp_center", type=float, default=1.0, help="Center of whole-SP GRAVY suitability")
    p.add_argument("--gravy_sp_sigma", type=float, default=1.0, help="Width of whole-SP GRAVY suitability")
    p.add_argument("--h_kd_center", type=float, default=2.3, help="Center of H-region mean KD suitability")
    p.add_argument("--h_kd_sigma", type=float, default=0.9, help="Width of H-region mean KD suitability")

    p.add_argument("--c_ala_target_fraction", type=float, default=0.30, help="Ala fraction at which S_A_C saturates")
    p.add_argument("--c_pro_target_fraction", type=float, default=0.15, help="Pro fraction at which S_P_C saturates")
    p.add_argument("--c_kd_good", type=float, default=-0.5, help="C-region mean KD considered strongly polar/suitable")
    p.add_argument("--c_kd_bad", type=float, default=1.5, help="C-region mean KD considered too hydrophobic")

    p.add_argument("--h_gly_penalty", type=float, default=0.7, help="Penalty for Gly beyond the first Gly in H-region")
    p.add_argument("--h_end_window", type=int, default=4, help="Residues at H-region end checked for K/R")
    p.add_argument("--h_end_charge_penalty", type=float, default=1.0, help="Penalty per H-end K/R residue")

    p.add_argument("--n_charge_midpoint", type=float, default=1.0, help="Midpoint of saturating N-charge score")
    p.add_argument("--n_charge_slope", type=float, default=1.4, help="Slope of saturating N-charge score")
    p.add_argument("--nh_contrast_midpoint", type=float, default=1.0, help="Preferred H-vs-N KD contrast midpoint")
    p.add_argument("--nh_contrast_slope", type=float, default=1.2, help="Slope of H-vs-N KD contrast score")

    p.add_argument("--sp_len_low0", type=float, default=12.0)
    p.add_argument("--sp_len_low1", type=float, default=16.0)
    p.add_argument("--sp_len_high1", type=float, default=30.0)
    p.add_argument("--sp_len_high0", type=float, default=36.0)
    p.add_argument("--h_len_low0", type=float, default=5.0)
    p.add_argument("--h_len_low1", type=float, default=7.0)
    p.add_argument("--h_len_high1", type=float, default=15.0)
    p.add_argument("--h_len_high0", type=float, default=20.0)

    p.add_argument("--deltaG_table", default=None, help="Deprecated for scoring; retained for CLI compatibility")
    p.add_argument("--deltaG_id_col", default="safe_id", help="Deprecated compatibility option")
    p.add_argument("--deltaG_col", default="deltaG", help="Deprecated compatibility option")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.ht_weights.split(",")) != 6:
        raise ValueError("--ht_weights must be six comma-separated numbers: S_hydro,S_cleavage,S_HC,S_N,S_length,S_SignalP")
    if not (0 < args.top_fraction <= 1):
        raise ValueError("--top_fraction must be in (0, 1]")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input)

    if args.parse_only:
        input_csv = outdir / "input_sequences_with_cargo.csv"
        if input_csv.exists():
            input_df = pd.read_csv(input_csv)
        else:
            records = read_fasta(input_path, args.cargo)
            input_df = write_input_table(records, outdir)
    else:
        print("[1/5] Reading FASTA and appending cargo...")
        records = read_fasta(input_path, args.cargo)
        if not records:
            raise RuntimeError("No FASTA records found")
        input_df = write_input_table(records, outdir)
        print(f"      Loaded {len(records):,} sequences")

        print("[2/5] Splitting SignalP batches...")
        manifest = split_batches(records, outdir, args.batch_size)
        print(f"      Created {len(manifest):,} batches")

        if args.prepare_only:
            print("prepare_only: done after creating batches")
            return

        if args.run_signalp:
            print("[3/5] Running local SignalP6...")
            run_signalp_batches(manifest, args)
        else:
            print("[3/5] --run_signalp not set; will parse existing outputs")

    print("[4/5] Parsing SignalP outputs...")
    signalp_df = parse_signalp_outputs(outdir / "02_signalp_outputs")
    signalp_df.to_csv(outdir / "SignalP6_parsed_results.csv", index=False)
    print(f"      Parsed {len(signalp_df):,} SignalP records")

    print("[5/5] Computing literature-informed SP_score and exporting top fraction...")
    scored, selected = compute_scores(input_df, signalp_df, args)
    write_outputs(scored, selected, outdir, args)
    print(f"Done. Scored {len(scored):,}; selected {len(selected):,}")
    print(f"Output directory: {outdir.resolve()}")


if __name__ == "__main__":
    main()
