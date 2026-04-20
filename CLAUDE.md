# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository shape

`lucebox-hub` is a thin hub of self-contained LLM-inference optimization projects. Each subdirectory ships independently with its own build system, README, and benchmarks — **there is no top-level build**. The two current projects target the same hardware (RTX 3090, sm_86, CUDA 12+) but share no code.

- `megakernel/` — Qwen3.5-0.8B BF16 decode megakernel (all 24 hybrid DeltaNet/Attention layers in a single persistent CUDA dispatch). PyTorch C++ extension.
- `dflash/` — GGUF port of DFlash+DDTree speculative decoding for Qwen3.5-27B Q4_K_M. Standalone C++/CUDA on top of ggml, no libllama, no Python runtime.
- `assets/` — banners, SVG cards, diagrams referenced by READMEs.

## Subproject: megakernel

**Build (PyTorch extension):**
```bash
cd megakernel && pip install -e .          # compiles kernel.cu + prefill.cu + torch_bindings.cpp
```
`setup.py` hardcodes `-arch=sm_86`, `--use_fast_math`, and kernel geometry (`NUM_BLOCKS=82`, `BLOCK_SIZE=512`, `LM_NUM_BLOCKS=512`, `LM_BLOCK_SIZE=256`). Weights stream from HuggingFace (`Qwen/Qwen3.5-0.8B`) on first run.

**Benchmarks:**
```bash
python final_bench.py       # 10 warmup + 20 timed runs — use THIS for published numbers
python bench_pp_tg.py       # quick single-run + correctness check (not warmed; under-reports prefill)
```

**Files (hand-authored, small):** `kernel.cu` (decode megakernel), `prefill.cu` (prefill via cuBLAS + standalone kernels), `torch_bindings.cpp`, `model.py` (weight loader + Decoder API). Layer pattern is hardcoded in `model.py:LAYER_TYPE` (18 DeltaNet + 6 Attention, 3:1 ratio).

**Kernel constraints learned the hard way** (see `megakernel/README.md` "Lessons"):
- `grid.sync()` inside per-token recurrence loops deadlocks silently. Sync **between layers**, not within them.
- `S_TILE=16` silently spills registers → `S_TILE=8` is the sweet spot.
- Batch size 1 only; BF16 only; Qwen3.5-0.8B only. Kernel does not generalize without a rewrite.

## Subproject: dflash

**Submodule:** `dflash/deps/llama.cpp` is pinned to `Luce-Org/llama.cpp@luce-dflash`, which adds three tree-mode ggml ops (`ggml_ssm_conv_tree`, `ggml_gated_delta_net_tree`, `ggml_gated_delta_net_tree_persist`). **You must clone with `--recurse-submodules`** or run `git submodule update --init --recursive` — CMake will `FATAL_ERROR` if the submodule is missing. Only the `ggml/` subtree is built/linked; `llama.cpp/src/models/{qwen35,delta-net-base}.cpp` sit there as **read-only reference** for porting the qwen35 forward pass.

**Build:**
```bash
cd dflash
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test_dflash -j          # main driver (~3 min on sm_86)
```
`CMakeLists.txt` force-disables every non-CUDA backend (`GGML_METAL`, `GGML_VULKAN`, `GGML_BLAS`, `GGML_OPENCL`, `GGML_BACKEND_DL` all OFF). Do not re-enable.

**Targets worth knowing:** `test_dflash` is the real driver (spec-decode + DDTree + fast-rollback). `test_generate` is the AR baseline used by benches. Other test binaries (`smoke_*`, `test_vs_oracle`) are built by default when their source file exists — guarded by `if(EXISTS ...)` blocks in `CMakeLists.txt`.

**Models (both required, ~20 GB):**
```bash
huggingface-cli download unsloth/Qwen3.5-27B-GGUF Qwen3.5-27B-Q4_K_M.gguf --local-dir models/
huggingface-cli download z-lab/Qwen3.5-27B-DFlash model.safetensors --local-dir models/draft/
```

**Run:**
```bash
python3 scripts/run.py --prompt "def fibonacci(n):"              # streaming one-shot
python3 examples/chat.py                                         # multi-turn REPL
python3 scripts/server.py --port 8000                            # OpenAI-compatible HTTP
python3 scripts/bench_llm.py                                     # HE + GSM8K + Math500 paper bench
python3 scripts/bench_he.py --n-gen 256 --ddtree-budget 22       # minimal HE bench
```
Python wrappers respawn `build/test_dflash` per turn (~10 s first-token latency). `bench_llm.py` reads paths from env (`DFLASH_TARGET`, `DFLASH_DRAFT`, `DFLASH_BIN`, `DFLASH_BIN_AR`) and parses tok/s out of stdout.

**128K context:** set `DFLASH27B_KV_Q4=1` to enable Q4_0 KV cache + sliding `target_feat` ring. See `dflash/RESULTS.md` for the full sweep.

**Architecture facts not obvious from grepping:**
- Qwen3.5-27B is the `qwen35` arch in llama.cpp — **not a dense transformer**. 64 layers: every 4th is full softmax attention (`il+1 % 4 == 0`), the other ~3/4 are Gated DeltaNet. M-RoPE with rope_sections `[11,11,10,0]`. 24 Q heads / 4 KV heads / head_dim 256.
- `include/dflash27b.h` `DFLASH27B_TARGET_N_*` / `_HEAD_DIM` macros are **draft dimensions** (32 Q / 8 KV / 128 head_dim), *not* target. The target constants live in `src/internal.h`. Naming is historical; do not rename without updating `safetensors_draft.cpp` and `qwen3_dflash_graph.cpp`.
- Token embedding stays CPU-resident (`CpuEmbedder` in `internal.h`) because CUDA `get_rows` doesn't support k-quants — matches llama.cpp's behavior, avoids 682 MiB VRAM for the embedding table.
- DDTree is tree-verify on top of block-diffusion draft. `budget=22` is the sweet spot for RTX 3090 + Q4_K_M. `chain_seed=true` in `build_ddtree` rescued AL from ~4 to ~9 — don't remove that pre-seed without a bench.
- Greedy decode only: server accepts `temperature`/`top_p` but **ignores** them.

## Cross-cutting rules (from CONTRIBUTING.md)

- **Scope is single-user, batch-1 local inference.** Do not add multi-tenant batching. Do not add support for attention-only architectures (LLaMA, Mistral) — point users at vLLM/SGLang.
- **Numbers need methodology.** Any perf PR must bench before/after on the same hardware, same power limit, same warmup. Power measurements use NVML (NVIDIA) or `powermetrics` (Apple Silicon) — accelerator power only, following the Hazy Research Intelligence-Per-Watt methodology.
- **One concern per PR.** Kernel, docs, and build-config changes go in separate commits or separate PRs.
- **Conventional commits** with scope: `feat(megakernel): ...`, `fix(dflash): ...`, `docs(hub): ...`. Allowed types: `feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `bench`, `chore`, `ci`.
- Correctness checks exist and must not regress: `megakernel/bench_pp_tg.py` (end-to-end parity vs reference decode), `dflash/test/test_vs_oracle` (draft graph cos-sim 0.999812 vs PyTorch reference; target graph bit-identical to `test_generate` in AR mode).

## Hardware expectations

RTX 3090 (Ampere, sm_86) is the reference target. CUDA 12+, PyTorch 2.0+ for megakernel; CMake 3.18+ for dflash. Both subprojects hardcode `sm_86` — building for newer arches requires editing `setup.py` / passing `-DCMAKE_CUDA_ARCHITECTURES=...`. No Metal, no ROCm, no multi-GPU.
