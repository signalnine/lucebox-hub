# Benchmark Results

All benchmarks are **batch size 1, single-stream decode**, targeting local inference on consumer hardware. This is the llama.cpp/Ollama use case, not multi-tenant serving.

## Hardware

| Machine | GPU/Chip | Memory |
|---------|----------|--------|
| Lucebox (3090) | NVIDIA RTX 3090 (Ampere, sm_86) | 24 GB VRAM |
| Lucebox (5090) | NVIDIA RTX 5090 (Blackwell consumer, sm_120) | 32 GB VRAM |
| MacBook Pro | Apple M5 Max | 36 GB Unified |

## RTX 3090: pp520 tg128

| Method | pp520 (tok/s) | tg128 (tok/s) |
|--------|:---:|:---:|
| **Megakernel** | **37,800** | **413** |
| llama.cpp BF16 | 11,247 | 267 |
| PyTorch HF | 7,578 | 108 |

### Speedups

| | vs llama.cpp | vs PyTorch |
|---|:---:|:---:|
| **Decode (tg128)** | **1.55x** | **3.8x** |

## RTX 5090: pp520 tg128

Build on `rtx-5090` branch (`-arch=sm_120`, `NUM_BLOCKS=170`, `BLOCK_SIZE=1024`, parallel DeltaNet prefill path), CUDA 13.2 / PyTorch 2.11+cu130. Five-run mean, stock power limit 575 W:

| Method | pp520 (tok/s) | tg128 (tok/s) |
|--------|:---:|:---:|
| **Megakernel** | **34,650** | **715** |
| llama.cpp BF16 | 31,258 | 504 |
| PyTorch HF | 6,865 | 81 |

### Speedups

| | vs llama.cpp | vs PyTorch |
|---|:---:|:---:|
| **Prefill (pp520)** | **1.11×** | **5.05×** |
| **Decode (tg128)** | **1.42×** | **8.8×** |

### Prefill path rewrite (axp)

First-pass pp520 on 5090 was **15,050 tok/s** — a 2.5× regression from the 3090's published 37,800. `nsys` isolated the cause to `pf_deltanet_recurrence`, which launched only 16 blocks (one per DeltaNet head, 9 % SM occupancy on Blackwell vs 19 % on Ampere) and consumed 80 % of prefill time. The fix was algorithmic, landed in the same `rtx-5090` branch:

- Observation: the per-token serial dependency is only in the state update. The conv1d over the time axis has no true cross-token dependency — it's a standard 1D conv with kernel=4 whose history comes from the previous call (read-only within one prefill).
- Refactor: replace the single serial kernel with five parallel ones — a fully parallel `pf_conv1d_parallel` (grid ~5,700 blocks), a cheap `pf_conv_buf_save`, a per-(t, h) `pf_deltanet_norm_activate` (grid 8,320 blocks), a V-sharded `pf_deltanet_state` (grid 128 blocks for N\_V\_CHUNKS=8), and a per-(t, h) `pf_deltanet_gated_rmsnorm` (grid 8,320 blocks).
- Result: pp520 recovers to **34,650 tok/s** (+130 %), past llama.cpp's 31,258 on the same card. Decode is unaffected (still 715 tok/s). Correctness holds — `final_bench.py` output matches PyTorch HF token for token.

Remaining gap to the 3090's 37,800: likely in the state kernel itself (128 blocks × ~470 µs per call × 18 layers = 8.5 ms of prefill). Further sharding (N\_V\_CHUNKS=16 → 256 blocks, or a fundamentally different algorithm like chunkwise parallel prefix) could close it. Not pursued — we already beat llama.cpp on the 5090 and the decode story is the product.

### BLOCK_SIZE sweep on 5090 (3-run means, decode)

| BLOCK_SIZE | J_PER_WARP | tg128 (tok/s) |
|:---:|:---:|:---:|
| 256 | 16 | 653 |
| 512 | 8 | 708 |
| **1024** | **4** | **715** |

The README's historical `S_TILE=16` warning was about the 256 direction and stands — it still underperforms on 5090. The *productive* direction is bigger blocks: more warps per SM → more latency hiding. `BLOCK_SIZE=1024` is the default on the `rtx-5090` branch.

### Apples-to-apples on 5090 (Megakernel vs llama.cpp BF16 vs Apple M5 Max)

| Platform | tok/s | Draw | tok/J |
|---|:---:|:---:|:---:|
| RTX 5090 Megakernel @ PL 400 W (floor) | 714 | 114 W | **6.26** |
| RTX 5090 Megakernel @ PL 575 W (stock) | 715 | 128 W | 5.61 |
| RTX 3090 Megakernel @ 220 W (published) | 411 | 220 W | 1.87 |
| Apple M5 Max | 229 | ~130 W | 1.76 |

At the hardware PL floor the 5090 draws **12 % less power than an M5 Max while delivering 3.1× the throughput**.

## RTX 5090 Power Efficiency (DVFS)

Sweep across the full PL range (400 W is the hardware floor on this card). Power and SM clock sampled in-process via pynvml at 20 Hz. Per-iteration window trims 1.5 s of startup and 1.0 s of teardown. Note: pp numbers in this table are from the pre-axp build (15 k on 5090); the tg number and tok/J conclusions are unchanged by the prefill rewrite because decode time dominates `final_bench.py`.

| Power Limit | Actual Draw | SM Clock | pp520 | tg128 | tok/J |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **400 W** (min) | **114 W** | 2722 MHz | 14,988 | 714 | **6.26** |
| 450 W | 121 W | 2747 MHz | 15,002 | 714 | 5.91 |
| 500 W | 123 W | 2758 MHz | 15,030 | 715 | 5.79 |
| 550 W | 127 W | 2744 MHz | 15,044 | 714 | 5.63 |
| 575 W (stock) | 128 W | 2749 MHz | 15,052 | 715 | 5.61 |
| 600 W | 129 W | 2759 MHz | 15,052 | 715 | 5.56 |

**Megakernel decode on Qwen3.5-0.8B never saturates the 5090.** Actual draw caps at ~128 W on a 575 W budget; tok/s is flat across the entire PL range (714-715). The card has far more throughput headroom than this model can consume — a direct consequence of batch=1, 0.8B parameters, and decode's inherent memory-latency bound.

## Apple M5 Max

| Method | tok/s |
|--------|:---:|
| LM Studio (llama.cpp) BF16 | 229 |

## Power Efficiency (DVFS)

| Power Limit | Clock | Draw | tok/s | tok/J | vs Stock |
|---|---|---|---|---|---|
| 420W (stock) | 1980 MHz | 314W | 433 | 1.38 | baseline |
| 300W | 1935 MHz | 299W | 432 | 1.44 | 99.8% speed, 5% less power |
| **220W** | **1635 MHz** | **220W** | **411** | **1.87** | **95% speed, 30% less power** |
| 150W | 405 MHz | 150W | 194 | 1.29 | too aggressive |

Sweet spot: 220W, 1.87 tok/J.

Contrast with the 5090 DVFS table above: on the 3090 every 70 W drop bought a measurable tok/J gain (1.38 → 1.87 → knee) because the card was power-saturated. On the 5090 the floor PL=400 W only tightens draw from 128 W to 114 W (11 % savings) — the kernel never demands full power, so the DVFS curve is flat. tok/J improvement there comes from the 5090's better overall perf/W at low clocks, not from DVFS headroom.

## Methodology

- **Precision:** BF16 weights and activations, FP32 accumulation. No quantization. All baselines (llama.cpp, PyTorch HF) also run BF16 for apples-to-apples comparison.
- **Power measurement:** Accelerator power only via NVML energy counters (NVIDIA) and `powermetrics` (Apple Silicon), consistent with [Hazy Research's Intelligence Per Watt methodology](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles). Total system draw is higher for both platforms.
- **Correctness:** `bench_pp_tg.py` includes an end-to-end correctness check, comparing megakernel output (prefill + decode) against a token-by-token reference decode path. Both must produce identical token sequences.
- **Warm-up:** One warm-up run before timed measurements. Timing uses `torch.cuda.synchronize()` barriers with `time.perf_counter()`.
- **llama.cpp version:** Latest release at time of testing, BF16 mode, default settings.

## What this doesn't measure

- Batched throughput (batch size > 1)
- Quantized model performance (INT4/INT8)
- Models larger than 0.8B parameters
- Multi-GPU or tensor-parallel setups
- Total system power (CPU, RAM, PSU losses)
