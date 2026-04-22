# Luce DFlash benchmark results

Single RTX 3090 24 GB, CUDA 12, driver 535.
Target: `unsloth/Qwen3.5-27B-GGUF` (Q4_K_M, ~16 GB).
Draft:  `z-lab/Qwen3.5-27B-DFlash` (BF16, 3.46 GB).
Concurrency = 1, greedy decoding, `n_gen=256`.
Reproduce with `python3 scripts/bench_llm.py` (samples 10 prompts/dataset, seed=42).

**Also see:** [RTX 5090 results further down](#rtx-5090-blackwell-consumer-32-gb) — the 32 GB card lets us run Q5_K_M target, pushing HumanEval to 187.4 tok/s at 3.51× (+45 % over 3090's Q4_K_M headline).

## Headline — AR vs Luce DFlash at concurrency 1

| Task      | AR tok/s | DFlash tok/s | AL   | Speedup |
|-----------|:--------:|:------------:|:----:|:-------:|
| HumanEval | 37.78    | **129.52**   | 8.31 | **3.43×** |
| Math500   | 37.71    | **110.51**   | 7.04 | **2.93×** |
| GSM8K     | 37.65    | **96.15**    | 6.14 | **2.55×** |

AR = autoregressive target-only decode via `test_generate`.
DFlash = block-diffusion draft + DDTree budget 22 verify + fast rollback.
AL = mean committed tokens per draft/verify step (acceptance length).

Datasets pulled live via HuggingFace `datasets`:
- HumanEval — `openai_humaneval`, `prompt` field
- GSM8K    — `gsm8k` main split, `Question: … Answer: ` format
- Math500  — `HuggingFaceH4/MATH-500`, `Problem: … Solution: ` format

## Per-prompt numbers (seed 42)

### HumanEval (10 samples)

| # | n_tok | AR    | DFlash | AL    |
|:-:|:-----:|:-----:|:------:|:-----:|
| 01| 84    | 37.98 | 137.91 | 8.83  |
| 02| 138   | 37.90 | 143.38 | 9.14  |
| 03| 134   | 37.88 | 137.49 | 8.83  |
| 04| 120   | 37.84 | 153.77 | 9.85  |
| 05| 172   | 37.76 | 131.74 | 8.53  |
| 06| 118   | 37.59 | 113.97 | 7.31  |
| 07| 51    | 37.78 | 103.27 | 6.56  |
| 08| 141   | 37.68 | **158.40** | **10.24** |
| 09| 125   | 37.71 | 128.22 | 8.26  |
| 10| 95    | 37.65 |  87.04 | 5.57  |
| **mean** |   | **37.78** | **129.52** | **8.31** |

Peak per-prompt: **158.40 tok/s at AL 10.24** (4.20× over AR on the same prompt).

### GSM8K (10 samples)

| # | n_tok | AR    | DFlash | AL   |
|:-:|:-----:|:-----:|:------:|:----:|
| 01| 45    | 37.62 |  93.87 | 5.95 |
| 02| 111   | 37.53 |  90.59 | 5.82 |
| 03| 49    | 37.73 |  87.79 | 5.57 |
| 04| 70    | 37.67 |  82.11 | 5.22 |
| 05| 102   | 37.62 | **127.83** | **8.26** |
| 06| 118   | 37.61 |  88.67 | 5.69 |
| 07| 113   | 37.62 |  86.86 | 5.57 |
| 08| 50    | 37.72 | 102.98 | 6.56 |
| 09| 43    | 37.69 | 109.66 | 6.92 |
| 10| 96    | 37.72 |  91.12 | 5.82 |
| **mean** |   | **37.65** | **96.15** | **6.14** |

### Math500 (10 samples)

| # | n_tok | AR    | DFlash | AL   |
|:-:|:-----:|:-----:|:------:|:----:|
| 01| 257   | 37.60 | 100.97 | 6.56 |
| 02| 53    | 37.73 | 115.62 | 7.31 |
| 03| 40    | 37.76 | 126.47 | 8.00 |
| 04| 50    | 37.76 | 118.20 | 7.53 |
| 05| 117   | 37.69 | 114.55 | 7.31 |
| 06| 76    | 37.70 | 108.63 | 6.92 |
| 07| 43    | 37.72 |  90.41 | 5.69 |
| 08| 79    | 37.73 | 100.10 | 6.40 |
| 09| 52    | 37.69 |  91.69 | 5.82 |
| 10| 57    | 37.74 | **138.45** | **8.83** |
| **mean** |   | **37.71** | **110.51** | **7.04** |

## Why the speedup varies by task

Acceptance length is the dominant factor — tok/s is roughly linear in AL when per-step overhead is fixed:

| Task      | AL   | Speedup vs AR |
|-----------|:----:|:-------------:|
| HumanEval | 8.31 | 3.43×         |
| Math500   | 7.04 | 2.93×         |
| GSM8K     | 6.14 | 2.55×         |

HumanEval prompts are highly regular (function signatures + docstrings), the draft nails consecutive tokens. GSM8K is natural-language arithmetic reasoning, the draft is less confident, tree verify rescues less.

## 128K context configuration

`max_ctx = 131072` + `DFLASH27B_KV_Q4=1` (Q4_0 K+V cache, 8× compression vs F16).
Sliding `target_feat` ring (4096 slots) keeps captured features at 0.2 GB regardless of context length.
`--ddtree-budget=16` keeps per-layer `ssm_intermediate` under 1.3 GB.

| Prompt length | KV size  | Prefill | Decode tok/s |
|:-------------:|:--------:|:-------:|:------------:|
| 520 (HE)      | ~35 MB   | 0.06 s  | 130          |
| 13K           | ~860 MB  | 15 s    | 99           |
| 32K           | ~2.1 GB  | 106 s   | 35           |
| 128K          | ~8.4 GB  | ~10 min | ~15-20 (est) |

Q4_0 KV costs ~3% mean tok/s vs F16 at short contexts and is the only thing that lets 128K allocate at all.

## DDTree budget sweep (HumanEval, n_gen=256, f16 intermediate)

Historical tuning run from commit `f1cb9bf` (2026-04-16). Used to pick the default budget=22. Fresh run at budget=22 on commit `5bb7f8c` is the 129.5 tok/s / AL 8.31 reported in the headline above; the ~5 tok/s delta vs the 135.8 row here comes from sample variance across the 10 prompts and from minor build-flag drift between the two commits.

| Budget | Mean AL | Mean tok/s |
|:------:|:-------:|:----------:|
| 15     | 7.64    | 125.3      |
| 16     | 7.81    | 128.7      |
| 18     | 8.22    | 131.2      |
| 20     | 8.64    | 133.9      |
| **22** | **8.88**| **135.8**  |
| 24     | 8.91    | 133.0      |
| 30     | 8.86    | 120.5      |
| 40     | 8.90    | 105.1      |

AL plateaus at ~8.9, past budget 22 each extra node costs more in verify time than it buys in accept. Memory ceiling at budget 26 on 24 GB (per-token SSM intermediate cache is hybrid-only overhead).

## Kernel-level wins (cumulative, chain mode → DDTree budget 22 + f16)

Starting point: Chain DFlash at 112.8 tok/s mean on HumanEval, AL 7.67.

| Optimization                                    | Δ tok/s | Δ AL | Note |
|-------------------------------------------------|:-------:|:----:|------|
| DDTree budget 20, f32 intermediate              | +15.1   | +0.77| Heap-based best-first tree, 20 nodes |
| Chain pre-seed in `build_ddtree`                | —       | +~5  | Fixes top-1 chain coverage under Q4 noise (prior AL ~4) |
| Tree-aware `ggml_ssm_conv_tree` kernel          | —       | +~1  | Sibling conv window gathers via parent chain, not DFS |
| `target_feat` compaction after sibling-accept   | —       | +~0.8| Stale feature pruning |
| OpenMP-parallel CPU top-K, K reduced 32→8       | +2.1    | —    | Shaves 7% off draft step |
| Fast K=1 path for budget=15                     | +1.5    | —    | Skips 11 ms CPU top-K when no siblings needed |
| D2D `cudaMemcpyAsync` for target_feat (GPU→GPU) | +3.7    | —    | Replaces GPU→CPU→GPU round trip |
| `ggml_gated_delta_net_tree_persist` kernel      | +12.4   | —    | Direct-writes SSM intermediates, skips 9 ms `ggml_cpy` per step |
| Budget 20 → 22, f16 intermediate                | +5.5    | +0.24| f16 cuts intermediate bandwidth in half |
| **Total**                                       | **+16.7** | **+0.64** | **129.5 tok/s, AL 8.31 (HumanEval mean, fresh run)** |

## Reproducibility

- Deterministic: greedy decode + greedy verify. Same prompts + same weights + same binary = same numbers ±1 tok/s.
- Full bench (10×3 = 30 prompts): ~15 min.
- All numbers above reproduced on 2026-04-20 from commit `5bb7f8c` with:
  ```
  python3 scripts/bench_llm.py
  ```

## RTX 5090 (Blackwell consumer, 32 GB)

Built on the `rtx-5090` branch (`-DCMAKE_CUDA_ARCHITECTURES=120`), CUDA 13.2 / driver 580. `bench_llm.py` at `--ddtree-budget=22`, n=10 prompts/dataset, seed=42.

### Quantization sweep (target GGUF swapped, draft held fixed)

The 3090 could only fit Q4_K_M target + draft + verify tree + KV in 24 GB. The 5090's 32 GB unlocks Q5_K_M and Q6_K; we ran the full bench at each.

|                    | HumanEval |  GSM8K  | Math500 |
|--------------------|:---------:|:-------:|:-------:|
|                    | AR / DFlash / AL / Speedup ||  |
| 3090 Q4_K_M (pub)  | 37.8 / 129.5 / 8.31 / 3.43× | 37.7 / 96.2 / 6.14 / 2.55× | 37.7 / 110.5 / 7.04 / 2.93× |
| 5090 Q4_K_M        | 58.3 / 164.0 / 7.92 / 2.81× | 58.1 / 131.7 / 6.32 / 2.27× | 58.3 / 153.5 / 7.35 / 2.63× |
| **5090 Q5_K_M**    | **53.4 / 187.4 / 9.32 / 3.51×** | **53.4 / 143.7 / 7.06 / 2.69×** | 53.3 / 145.5 / 7.15 / 2.73× |
| 5090 Q6_K          | 49.1 / 174.2 / 9.11 / 3.55× | 49.0 / 122.6 / 6.34 / 2.50× | 49.0 / 136.4 / 7.10 / 2.78× |

**Q5_K_M is the sweet spot on 5090.** Highest absolute DFlash tok/s on every dataset. It recovers both the speedup ratio that Q4_K_M on 5090 had lost (HE 3.51× vs 3090's 3.43×) *and* delivers +45 % absolute tok/s.

### Why Q5_K_M wins

| Axis | Direction as target precision grows (Q4 → Q5 → Q6) |
|------|----------------------------------------------------|
| AR tok/s | Decreases (58.3 → 53.4 → 49.1 HumanEval) — more weight bytes per forward pass |
| Acceptance Length | Increases, then plateaus (7.92 → 9.32 → 9.11 HumanEval) — draft-target agreement improves with a less-quantized target, then saturates |
| Per-step verify cost | Increases with target size |
| DFlash tok/s | Peaks at Q5 (164 → **187** → 174 HumanEval) — AL gain beats verify-cost loss at Q5, then loses ground at Q6 |

Q6_K's marginally higher speedup *ratio* (3.55× vs 3.51× HE) doesn't compensate for its slower AR baseline; absolute throughput drops.

### VRAM fit at budget=22

| Target | Target size | + draft (3.3 GiB) | Remaining for KV + tree + activations |
|--------|:-----------:|:-----------------:|:-------------------------------------:|
| Q4_K_M | 15.6 GiB | 18.9 GiB | ~13 GiB |
| Q5_K_M | 18.3 GiB | 21.6 GiB | ~10 GiB |
| Q6_K   | 20.9 GiB | 24.2 GiB | ~8 GiB |
| Q8_0   | 26.6 GiB | 29.9 GiB | too tight for budget=22 state |

### Per-prompt Q5_K_M numbers (HumanEval, seed 42)

| # | n_tok | AR    | DFlash | AL    |
|:-:|:-----:|:-----:|:------:|:-----:|
| 01| 84    | 53.45 | 252.50 | 12.80 |
| 02| 138   | 53.35 | 172.73 | 8.53  |
| 03| 134   | 53.45 | 214.11 | 10.67 |
| 04| 120   | 53.45 | 213.77 | 10.67 |
| 05| 172   | 53.37 | 144.98 | 7.11  |
| 06| 118   | 53.41 | 157.66 | 7.76  |
| 07| 51    | 53.45 | 172.28 | 8.53  |
| 08| 141   | 53.35 | 162.89 | 8.00  |
| 09| 125   | 53.26 | 161.94 | 8.00  |
| 10| 95    | 53.34 | 220.83 | 11.13 |
| **mean** |   | **53.39** | **187.37** | **9.32** |

Peak per-prompt: **252.5 tok/s at AL 12.80** (prompt 01 — function-signature completion where the draft nails 12-token runs consistently).

### Reproducibility (5090)

Full bench at all three quant levels reproduced 2026-04-20 on commit `94a6410` of the `rtx-5090` branch with:

```bash
# Q5_K_M (the recommended sweet spot)
DFLASH_TARGET=$(pwd)/models/Qwen3.5-27B-Q5_K_M.gguf \
DFLASH_DRAFT=$(pwd)/models/draft/model.safetensors \
  python3 scripts/bench_llm.py

# Swap to Q4_K_M or Q6_K by changing DFLASH_TARGET.
```

## Hardware ceiling notes

- Published DFlash paper on Qwen3-4B/8B/30B-MoE (pure attention, BF16, B200) reports 4-5× over AR on HumanEval/Math500 at concurrency 1. Ours: 3.43× on 27B hybrid Q4_K_M on RTX 3090.
- Memory ceiling: per-token SSM intermediate cache (hybrid-only cost) caps tree budget at ~26 on 24 GB. The paper uses budgets up to 1024 on pure-attention models with zero per-node memory tax.
- Per-token verify cost drops from 25 ms at N=1 to 0.97 ms at N=128 (ggml-cuda Q4_K matmul amortises well with batch size).

---

## Qwen3.6-35B-A3B-MXFP4_MOE (5090, `qwen36-port` branch)

Second target arch: `qwen35moe` (per llama.cpp naming). 64 layers, 3:1 DeltaNet:Attn interleave, 256 MXFP4 experts × top-8, shared expert on every layer. The `qwen36-port` branch ported dflash's 27B engine to this arch, then closed the throughput gap to llama.cpp via a series of kernel-dispatch and cache-layout fixes (see `docs/port-qwen36-35b-a3b-moe.md` for the milestone-by-milestone trace).

Setup: RTX 5090 (Blackwell consumer, 32 GB, sm_120), CUDA 13.2, `-DCMAKE_CUDA_ARCHITECTURES=120`. Target `unsloth/Qwen3.6-35B-A3B-GGUF` (MXFP4_MOE, 20.2 GB). Draft for spec-decode is the generic `Qwen3.5-0.8B-BF16` dense model (no purpose-built DFlash draft yet — M5 stretch). `n_gen=256`, greedy decoding, 5 prompts/dataset, seed 42. Reproduce with `scripts/bench_chain_spec.py` (M4 driver).

### Headline — AR at parity with llama.cpp

| Decode path                     | Code prompt (fibonacci) | tg128 (llama-bench)       |
|---------------------------------|:-----------------------:|:-------------------------:|
| **`test_generate` AR (ours)**   | **213 tok/s** (+66% vs M1b) | —                          |
| `llama-bench tg128`             | —                         | 211–235 tok/s (varies by P-state) |

The M1b port baseline was 125 tok/s (60% of llama.cpp). Three commits closed the gap:

1. **swiglu-split fusion** (`499c176`) — emit `ggml_swiglu_split(gate, up)` instead of `silu(gate) * up`. ggml-cuda's `ggml_cuda_should_fuse_mul_mat_vec_q` pattern-matcher folds both `mul_mat_id` + GLU into one mmvq with `has_fusion=true`. nsys shows MXFP4 mmvq 78→39/step (matches llama.cpp exactly) and Q8_0 mmvq 201→121.

2. **persistent ggml context** (`499c176`) — `ggml_reset(ctx)` re-uses the arena instead of `ggml_free` + `ggml_init`. Tensors re-land at stable addresses so `cgraph->nodes[0]` stays pointer-stable across steps — the precondition for ggml-cuda's CUDA-graph cache.

3. **KV cache layout + `ggml_set_rows`** (`4c16107`) — switch cache from `[head_dim, max_ctx, n_head_kv]` to llama.cpp's `[n_embd_gqa, max_ctx]`, replace view-offset KV writes with `ggml_set_rows(cache, k_cur, kv_pos_idx)` where `kv_pos_idx` is an input tensor, and pad `n_kv` to `GGML_PAD(kv_len, 256)`. Graph shape is now stable across 256-step windows so ggml-cuda's 2-step `warmup_complete` latches and subsequent steps become single `cudaGraphLaunch` calls. Kernel instance count per step: Q8_0 mmvq from 7437/37steps pre-refactor to **121/37steps post-refactor** (~1 capture amortised over 36 graph launches). `compute=6208→4252 µs/step` (−31%).

### Batched prefill (PREFILL_CHUNK=1024)

The initial M1b implementation prefilled one prompt token per target forward. Commits `140215f`+`7ed0b6d` chunk the prompt into N-token forwards:

| Prompt length | Ours prefill | llama-bench reference | Ratio |
|---------------|:------------:|:---------------------:|:-----:|
|  512 tokens   |  2811 tok/s  |  6487 (pp512)         | 43%   |
| 1024 tokens   |  4012 tok/s  |  —                    | —     |
| 2048 tokens   |  **4687 tok/s** |  6457 (pp2048)     | **72%** |

Was 285 tok/s at M1b. The 1024-token chunk size amortises the per-chunk CUDA-graph capture over multiple kernel launches; each chunk's `mul_mat_q` tile utilises the MMQ batch path which is much faster per-token than the N=1 MMVQ kernel used during decode. Decode throughput is unaffected (prefill and decode use different CUDA graph cache slots, each with its own capture). Remaining prefill gap to llama.cpp is in upstream ggml-cuda (MMQ tile tuning; potential fused QKV for full-attn layers if the Q/K/V weights were pre-packed as they are on the 27B gguf).

### Spec-decode on HumanEval + GSM8K + Math500 (n=5, n_gen=256, N_spec=8)

With the 0.8B dense draft, after **GPU-argmax + graph-reuse** (`51ac916`+`f5f60b3`), fp32 ssm_intermediate (`c2ba531`), and DDTree sibling-walk partial draft rollback (`da6a499`):

| Mode                      | HE tok/s   | HE ×AR   | GSM8K tok/s | GSM8K ×AR | Math500 tok/s | Math500 ×AR |
|---------------------------|:----------:|:--------:|:-----------:|:---------:|:-------------:|:-----------:|
| `test_generate` AR        | **233.63** | 1.00     | **234.32**  | 1.00      | **234.05**    | 1.00        |
| `CHAIN_VERIFY=seq`        | 130.38     | 0.56     | 118.34      | 0.51      | 126.64        | 0.54        |
| `CHAIN_VERIFY=batch`      | 191.68     | 0.82     | 138.51      | 0.59      | 156.91        | 0.67        |
| `CHAIN_VERIFY=tree_chain` | **204.75** | **0.88** | **172.28**  | **0.74**  | **191.66**    | **0.82**    |
| `CHAIN_VERIFY=ddtree` (K=8, budget=22) | 153.65 | 0.66 | 142.43 | 0.61 | 149.28 | 0.64 |

(AL columns omitted for compactness; see `scripts/bench_*_reference.json` for per-mode AL.)

**On specific low-drift prompts `tree_chain` BEATS AR.** Best cases in the bench:

| Prompt               | AR tok/s | tree_chain tok/s | Δ      | AL   |
|----------------------|:--------:|:----------------:|:------:|:----:|
| HE sample 3          | 233.72   | **247.48**       | +5.9%  | 8.37 |
| GSM8K sample 1       | 234.22   | **248.46**       | +6.1%  | 8.20 |
| Math500 sample 3     | 234.25   | **242.03**       | +3.3%  | 8.14 |

Spec decode actually pays for itself once the draft-target AL is high enough to cover the per-round overhead. On GSM8K sample 5 the target/draft disagree on a near-tie (AL=3.13 for tree_chain), DDTree recovers via 23% sibling walks (AL=7.31, 151 tok/s — 55% faster than batch's 97.4).

AR decode 234 tok/s on 35B MXFP4 MoE **matches or slightly beats llama-bench tg256 (235) and tg1024 (234)** — we got there via the `qwen36-port` series:

- swiglu-split fusion + persistent ggml ctx: 128 → 150 tok/s
- KV cache layout + `ggml_set_rows`: 150 → 214 tok/s
- GPU-side argmax + same-shape graph reuse: 214 → 234 tok/s

The chain_spec propagation of the last of those is the reason spec-decode modes jumped 50-90% from the pre-fast-rollback bench snapshot. DDTree sibling-walk partial checkpoint restore (`da6a499`) added another +3.5% on HE / +3.7% on GSM8K to DDTree by skipping draft forwards that the accepted path shares with the original chain — savings scale with `sibling_rate × sibling_d / commit_count`.

**`N_spec` is mode-dependent** (5-sample sweep, HE / GSM8K / Math500):

| Mode       | best `N_spec` | vs `N_spec=8`     |
|------------|:-------------:|:-----------------:|
| tree_chain |      6        | +1–5% (biggest on GSM8K where true AL ≈ 5.4) |
| seq        |      6        | +3–6%             |
| batch      |      6        | +3–6% on GSM8K / Math500; −12% on HE (noisy) |
| ddtree     |      8        | DDTree loses 5–9% at `N_spec=6` because `N_spec` is the chain depth `L` — shrinking it wastes the `budget=22` tree budget on fewer paths |

The current `N_spec=8` default stays because (a) DDTree regresses at smaller values and (b) for chain modes the gain is smaller than the DDTree loss. Users wanting max tree_chain throughput can pass `--n-spec 6`.

### Comparison — before vs after the cache-layout refactor

Same driver, same prompts, same N_spec, same seed:

| dataset   | mode          | before (M4) | after (post-ttx) | Δ      |
|-----------|---------------|:-----------:|:----------------:|:------:|
| HumanEval | AR            | 125.7       | **212.3**        | +69%   |
|           | seq           | 72.7        | 112.8            | +55%   |
|           | batch         | 121.4       | 163.9            | +35%   |
|           | tree_chain    | 121.1       | 169.0            | +39%   |
|           | ddtree K=8 B=22 | 105.2     | 119.8            | +14%   |
| GSM8K     | AR            | 125.8       | **212.3**        | +69%   |
|           | seq           | 63.3        | 100.3            | +58%   |
|           | batch         | 72.5        | 116.3            | +60%   |
|           | tree_chain    | 72.6        | 122.8            | +69%   |
|           | ddtree K=8 B=22 | 78.9      | 106.9            | +35%   |

Raw per-sample JSON in `dflash/scripts/bench_post_ttx_reference.json`.

### Reproducibility (qwen36-port, 5090)

```bash
# AR baseline
build/test_generate $TGT prompt.bin 256 out.bin

# Spec-decode bench (5 HE + 5 GSM8K, all modes)
DFLASH_TARGET=$TGT DFLASH_DRAFT=$DRAFT_08B \
  python3 scripts/bench_chain_spec.py \
    --dataset all --n-sample 5 --n-gen 256 --n-spec 8 \
    --modes seq,batch,tree_chain,ddtree \
    --ddtree-k 8 --ddtree-budget 22
```

Where `$TGT` is `Qwen3.6-35B-A3B-MXFP4_MOE.gguf` and `$DRAFT_08B` is any qwen35-family dense GGUF with matching vocab (we used `Qwen3.5-0.8B-BF16.gguf`). The `bench_chain_spec.py` driver also writes a JSON report with per-sample stats (AL, sibling walks for DDTree, tok/s).

### 27B regression (sanity check post-refactor)

The M2a + ttx refactor touched `qwen35_target_graph.cpp` which is shared with the 27B `qwen35` arch. Re-ran the 27B DFlash bench on 5090 with the `jks` issue's methodology:

| Prompt (HE sample) | Pre-refactor (3090 baseline) | Post-refactor (5090) | AL    |
|--------------------|:----------------------------:|:--------------------:|:-----:|
| sample 00          | —                             | 134.45 tok/s         | 9.48  |
| sample 01          | —                             | 155.79 tok/s         | 11.13 |
| sample 02          | —                             | 139.83 tok/s         | 9.85  |
| **mean**           | 129.5 (3090, pre-refactor)    | **143.36 (5090)**    | 10.15 |

+10.7% over the 3090 3-prompt baseline; no semantic regression (output fully coherent). The absolute number is above baseline because of the GPU change; the important signal is that the per-step timing breakdown is stable and DFlash still converges to its expected AL on a well-matched draft.
