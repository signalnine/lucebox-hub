"""
Bench chain_spec (test_chain_spec) over HumanEval / GSM8K prompts.

Runs each prompt through the AR baseline (test_generate) and through
test_chain_spec in each CHAIN_VERIFY mode, reporting tok/s and avg
commit/round. The output lets us measure DDTree quality + throughput
against a real workload at scale (HE = 164 prompts, but we sample).

Paths resolve from the repo root by default. Override with env vars:
    DFLASH_TARGET   path to target .gguf
    DFLASH_DRAFT    path to draft .gguf  (dense; e.g. Qwen3.5-0.8B)
    DFLASH_BIN      path to build/test_chain_spec
    DFLASH_BIN_AR   path to build/test_generate

Usage:
    python3 scripts/bench_chain_spec.py                    # all datasets, all modes
    python3 scripts/bench_chain_spec.py --dataset he       # HumanEval only
    python3 scripts/bench_chain_spec.py --modes seq,tree_chain,ddtree
    python3 scripts/bench_chain_spec.py --n-sample 5

The AR baseline uses test_generate (single-token stepping of target), which
is the correct reference for "what the target would greedily produce" under
VEC numerics. DDTree output is MMA-greedy-equivalent — it can diverge from
AR on near-tie positions (see M3a). tok/s comparison is still fair.
"""

import argparse
import json
import os
import re
import struct
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TARGET = os.environ.get(
    "DFLASH_TARGET",
    "/mnt/ai/models/huggingface/qwen3.6-35b-a3b-GGUF/Qwen3.6-35B-A3B-MXFP4_MOE.gguf",
)
DRAFT = os.environ.get(
    "DFLASH_DRAFT",
    str(ROOT / "models" / "Qwen3.5-0.8B-BF16.gguf"),
)
CHAIN_BIN = os.environ.get("DFLASH_BIN", str(ROOT / "build" / "test_chain_spec"))
AR_BIN    = os.environ.get("DFLASH_BIN_AR", str(ROOT / "build" / "test_generate"))

DEFAULT_TOKENIZER = os.environ.get("DFLASH_TOKENIZER", "Qwen/Qwen3.5-0.8B")

BENCHES = [
    ("HumanEval", "openai_humaneval", None,            "test",
     lambda x: x["prompt"]),
    ("GSM8K",     "gsm8k",            "main",          "test",
     lambda x: f"Question: {x['question']}\nAnswer: "),
]


def tokenize(tok, prompt: str, path: Path) -> int:
    ids = tok.encode(prompt, add_special_tokens=False)
    with open(path, "wb") as f:
        for t in ids:
            f.write(struct.pack("<i", int(t)))
    return len(ids)


def run_ar(prompt_path: Path, n_gen: int) -> tuple[float, int]:
    """Returns (tok/s, 0). test_generate usage:
       test_generate <target.gguf> <prompt.bin> <n_gen> <out.bin>
    """
    t0 = time.monotonic()
    r = subprocess.run(
        [AR_BIN, TARGET, str(prompt_path), str(n_gen), "/tmp/ar_out.bin"],
        capture_output=True, text=True, timeout=600,
    )
    dt = time.monotonic() - t0
    if r.returncode != 0:
        return 0.0, 0
    # test_generate now prints TWO tok/s lines — the prefill rate and the
    # decode rate. We want decode. The decode line is marked with "[gen]".
    m = re.search(r"\[gen\][^\n]*?(\d+\.\d+)\s*tok/s", r.stdout)
    if m:
        return float(m.group(1)), 0
    # Legacy single-line format (pre-prefill-batching).
    m = re.search(r"(\d+\.\d+)\s*tok/s", r.stdout)
    if m:
        return float(m.group(1)), 0
    # Fallback: derive from wallclock.
    return (n_gen / dt if dt > 0 else 0.0), 0


def run_chain(prompt_path: Path, n_gen: int, n_spec: int, mode: str,
              ddtree_k: int = 8, ddtree_budget: int = 22,
              ddtree_temp: float = 1.0) -> tuple[float, float, dict]:
    """Returns (tok/s, avg_commit_per_round, extra_stats)."""
    env = {**os.environ, "CHAIN_VERIFY": mode}
    if mode == "ddtree":
        env["DDTREE_K"]      = str(ddtree_k)
        env["DDTREE_BUDGET"] = str(ddtree_budget)
        env["DDTREE_TEMP"]   = f"{ddtree_temp}"
    out = f"/tmp/chain_out_{mode}.bin"
    r = subprocess.run(
        [CHAIN_BIN, TARGET, DRAFT, str(prompt_path), str(n_gen), str(n_spec), out],
        capture_output=True, text=True, timeout=600, env=env,
    )
    if r.returncode != 0:
        return 0.0, 0.0, {"err": r.stderr.strip().splitlines()[-1] if r.stderr else "nonzero exit"}
    # Parse "[chain] N gen tokens in X.XXX s  →  Y.YY tok/s"
    m_tps = re.search(r"→\s+(\d+\.\d+)\s+tok/s", r.stdout)
    # Parse "avg_commit/round=X.XX"
    m_al  = re.search(r"avg_commit/round=(\d+\.\d+)", r.stdout)
    # DDTree sibling stats
    m_sib = re.search(r"sibling_walks=(\d+)\s+\(([\d.]+)%", r.stdout)
    stats = {}
    if m_sib:
        stats["sibling_walks"]  = int(m_sib.group(1))
        stats["sibling_rate"]   = float(m_sib.group(2))
    return (float(m_tps.group(1)) if m_tps else 0.0,
            float(m_al.group(1))  if m_al else 0.0,
            stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["he", "gsm8k", "all"], default="all")
    ap.add_argument("--modes", default="seq,batch,tree_chain,ddtree",
                    help="comma-separated verify modes")
    ap.add_argument("--n-sample", type=int, default=10)
    ap.add_argument("--n-gen", type=int, default=256)
    ap.add_argument("--n-spec", type=int, default=8)
    ap.add_argument("--ddtree-k", type=int, default=8)
    ap.add_argument("--ddtree-budget", type=int, default=22)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--skip-ar", action="store_true",
                    help="skip AR baseline (test_generate)")
    ap.add_argument("--out", default="/tmp/bench_chain_spec.json")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    print(f"[bench] target={TARGET}")
    print(f"[bench] draft ={DRAFT}")
    print(f"[bench] modes ={modes}  n_gen={args.n_gen}  n_spec={args.n_spec}")
    if "ddtree" in modes:
        print(f"[bench] ddtree K={args.ddtree_k} budget={args.ddtree_budget}")

    from datasets import load_dataset  # noqa: WPS433
    from transformers import AutoTokenizer  # noqa: WPS433
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    if args.dataset == "he":
        benches = [b for b in BENCHES if b[0] == "HumanEval"]
    elif args.dataset == "gsm8k":
        benches = [b for b in BENCHES if b[0] == "GSM8K"]
    else:
        benches = BENCHES

    all_results = {}
    for name, ds_name, cfg, split, extract in benches:
        print(f"\n[bench] ==== {name} (n={args.n_sample}) ====", flush=True)
        ds = load_dataset(ds_name, cfg, split=split).shuffle(seed=42).select(range(args.n_sample))
        per_mode = {m: {"tps": [], "al": [], "sib_rate": []} for m in modes}
        ar_list = []
        for i, s in enumerate(ds):
            prompt = extract(s)
            path = Path(f"/tmp/b_{name}_{i:02d}.bin")
            n = tokenize(tok, prompt, path)
            if n == 0 or n > 3500:
                continue
            line = f"  [{i+1:02d}/{args.n_sample}] n_tok={n:4d}  "
            if not args.skip_ar:
                ar, _ = run_ar(path, args.n_gen)
                if ar > 0:
                    ar_list.append(ar)
                    line += f"AR={ar:6.2f}  "
                else:
                    line += f"AR=FAIL   "
            for m in modes:
                tps, al, st = run_chain(path, args.n_gen, args.n_spec, m,
                                         args.ddtree_k, args.ddtree_budget)
                if tps > 0:
                    per_mode[m]["tps"].append(tps)
                    per_mode[m]["al"].append(al)
                    if "sibling_rate" in st:
                        per_mode[m]["sib_rate"].append(st["sibling_rate"])
                    sib = f" sib={st['sibling_rate']:.0f}%" if "sibling_rate" in st else ""
                    line += f"{m}={tps:6.2f}(AL={al:4.2f}{sib}) "
                else:
                    line += f"{m}=FAIL "
            print(line, flush=True)

        ar_m = sum(ar_list) / len(ar_list) if ar_list else 0.0
        summary = {"ar": ar_m}
        for m in modes:
            tps_m = sum(per_mode[m]["tps"]) / len(per_mode[m]["tps"]) if per_mode[m]["tps"] else 0.0
            al_m  = sum(per_mode[m]["al"])  / len(per_mode[m]["al"])  if per_mode[m]["al"]  else 0.0
            sib_m = sum(per_mode[m]["sib_rate"]) / len(per_mode[m]["sib_rate"]) if per_mode[m]["sib_rate"] else 0.0
            summary[m] = {
                "tps": tps_m,
                "al":  al_m,
                "sib_rate_pct": sib_m,
                "speedup_vs_ar": tps_m / ar_m if ar_m else 0.0,
            }
        all_results[name] = summary
        print(f"  {name} mean: AR={ar_m:6.2f}", end="", flush=True)
        for m in modes:
            s = summary[m]
            print(f"  {m}={s['tps']:6.2f} (AL={s['al']:.2f}, x{s['speedup_vs_ar']:.2f})", end="")
        print(flush=True)

    print("\n[bench] === SUMMARY ===", flush=True)
    header = f"{'Task':10s}  {'AR':>7s}"
    for m in modes:
        header += f"  {m:>11s}  {'AL':>5s}  {'x':>5s}"
    print(header)
    for name, r in all_results.items():
        row = f"{name:10s}  {r['ar']:7.2f}"
        for m in modes:
            s = r[m]
            row += f"  {s['tps']:11.2f}  {s['al']:5.2f}  {s['speedup_vs_ar']:5.2f}"
        print(row)

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[bench] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
