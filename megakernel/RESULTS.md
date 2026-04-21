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

Fresh build on `rtx-5090` branch (`-arch=sm_120`, `NUM_BLOCKS=170`, `BLOCK_SIZE=1024`), CUDA 13.2 / PyTorch 2.11+cu130. Three-run mean, stock power limit 575 W:

| Method | pp520 (tok/s) | tg128 (tok/s) |
|--------|:---:|:---:|
| **Megakernel** | 15,050 | **715** |
| llama.cpp BF16 | **31,258** | 504 |
| PyTorch HF | 6,865 | 81 |

### Speedups

| | vs llama.cpp | vs PyTorch |
|---|:---:|:---:|
| **Decode (tg128)** | **1.42×** | **8.8×** |

### Prefill regression — known issue

On 5090 the megakernel's pp520 drops from 37,800 (3090) to 15,050 — a 2.5× regression on a card with 2× the SMs and bandwidth. `nsys` isolated the cause to `pf_deltanet_recurrence` in `prefill.cu`, which launches only 16 blocks (one per DeltaNet head) and therefore uses 16/170 = 9 % of SMs on Blackwell (vs 16/82 = 19 % on Ampere). The fix is algorithmic — shard along the V-dimension for 64+ blocks — and is tracked separately. Decode is unaffected because its megakernel uses all 170 SMs cooperatively.

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

Sweep across the full PL range (400 W is the hardware floor on this card). Power and SM clock sampled in-process via pynvml at 20 Hz. Per-iteration window trims 1.5 s of startup and 1.0 s of teardown.

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
