# Porting plan: Qwen3.6-35B-A3B MoE (MXFP4) to the DFlash engine

**Target artifact:** `Qwen3.6-35B-A3B-MXFP4_MOE.gguf` — 35 B-parameter sparse MoE
with ~3 B active params per token, MXFP4-quantized expert weights, distributed
as a single GGUF.

**Hardware target:** single RTX 5090 (32 GB, sm_120). Secondary target: 3090
(24 GB) — expected *not* to fit without offload.

This plan is written from what the `dflash/` codebase does today for Qwen3.5-27B
hybrid + Q4_K_M and extends each layer with what the MoE/MXFP4 variant needs.

---

## Scope & non-goals

**In scope**
- Get `test_dflash` to decode the new target in autoregressive (AR) mode — parity
  with `test_generate` or llama.cpp on a held-out prompt.
- Land a working speculative-decoding path (chain first, DDTree after).
- Publish a reproducible `bench_llm.py` result at budget=22 (or whatever sweeps to).

**Out of scope, this iteration**
- Multi-GPU / tensor-parallel.
- Training a new draft model from scratch.
- Any megakernel work — the 35 B scale is squarely a per-layer-optimization
  regime, the persistent-kernel story from `megakernel/` does not transfer and
  should not be attempted here.
- Batch > 1, temperature/top-p sampling (greedy only, matches current engine).

---

## Pre-work (verification, before touching code)

These unknowns gate the plan; resolve them first.

1. **Confirm the architecture key** inside the GGUF.
   ```bash
   ./deps/llama.cpp/build/bin/gguf-dump --no-tensors \
       models/Qwen3.6-35B-A3B-MXFP4_MOE.gguf | head -50
   ```
   The `general.architecture` string routes to one of `qwen3moe`, `qwen35moe`,
   `qwen3next`, or a new `qwen36moe`/`qwen36` variant. Each has a different
   reference `llm_build_*` in `dflash/deps/llama.cpp/src/models/`. The plan
   branches here — see §3.

2. **Confirm hybrid vs pure attention.** Look at the GGUF metadata for
   `*.recurrent_layer_indices` or for `*.ssm.*` tensors. If present, it's
   DeltaNet-hybrid (like 27B) and we reuse our tree-mode ops. If absent, it's
   pure attention + MoE and we can *remove* a lot of our machinery.

3. **Dump shape constants.** Collect `hparams`:
   `n_embd`, `n_layer`, `n_head`, `n_head_kv`, `n_embd_head_k/v`,
   `n_ff`, `n_expert`, `n_expert_used`, `n_embd_head_v`, `rope_freq_base`,
   `rope_sections`, `rms_norm_eps`.
   These become the new `#define`s in `include/dflash27b.h` /
   `src/internal.h` (or a new `include/dflash36.h` if we decide to keep 27B
   shipping alongside).

4. **Sanity-check MXFP4 support in our submodule.**
   Already present in `dflash/deps/llama.cpp/ggml/src/ggml-common.h`
   (`block_mxfp4`, `QK_MXFP4=32`) and
   `ggml-cuda/convert.cu` (`dequantize_block_mxfp4`, registered for
   `GGML_TYPE_MXFP4`). Do a smoke test:
   ```bash
   ./deps/llama.cpp/build/bin/llama-cli -m <target>.gguf -p "hello" -n 8 -ngl 99
   ```
   If that runs, every ggml op we need (matmul, get_rows, softmax) already
   has the MXFP4 path. If it doesn't, stop and file upstream before touching
   the port.

5. **Pick the draft model.**
   z-lab has not published a DFlash draft for this target (as of April 2026).
   Options in order of preference:
   - (a) **Chain-speculative draft:** any small Qwen3 model with matching
     tokenizer — e.g. `Qwen/Qwen3-1.7B` or `Qwen/Qwen3-4B`. This is the
     fallback path even in the current DFlash engine (set
     `ddtree-budget=n_spec+1` and no branching).
   - (b) **Train a DFlash draft** on the new target using z-lab's recipe.
     ~3-5 days of H100 time; out of scope for the first milestone.
   - (c) If the MoE architecture is pure-attention, EAGLE-2/3 drafts may be
     easier to adapt — check the `openai-moe-iswa.cpp` reference in our fork
     for similar-shape models that already have EAGLE ports.

   **Decision point:** ship chain-speculative first with a small Qwen3 dense
   draft. It validates the whole pipeline (AR + spec) at ~2× speedup. Draft
   training for DDTree is a follow-on.

---

## Architecture port (§3) — two branches depending on pre-work #2

### 3a. If the GGUF arch is hybrid DeltaNet + MoE (like `qwen35moe`)

We keep almost everything from the current engine. The only per-layer change
is the FFN: dense MLP → expert routing + top-k expert FFN.

Reference: `deps/llama.cpp/src/models/qwen35moe.cpp` — read-only, shows the
exact graph we need to build.

Work:

1. Copy `src/qwen35_target_graph.cpp` to `src/qwen36_target_graph.cpp`.
2. In the DeltaNet block path, nothing changes — the existing tree-mode SSM ops
   (`ggml_ssm_conv_tree`, `ggml_gated_delta_net_tree[_persist]`) already work.
3. Replace the dense `w_gate/w_up/w_down` FFN with:
   - `build_moe_ffn` from llama.cpp (copy the pattern, don't link libllama).
     Looks like:
     ```
     router_logits = norm @ layers[il].ffn_gate_inp   // [n_tokens, n_expert]
     probs         = softmax(router_logits, dim=-1)
     topk_ids, topk_w = top_k(probs, k=n_expert_used)
     topk_w        = normalize(topk_w, dim=-1)
     y = 0
     for k in 0..n_expert_used:
         expert_id = topk_ids[:, k]
         expert_out = MLP(expert_id, cur)   // ggml_mul_mat_id
         y += expert_out * topk_w[:, k]
     return y
     ```
   - `ggml_mul_mat_id` is already in ggml and handles MXFP4 weight tensors
     (rank-3 tensor: `[n_embd_expert, n_embd, n_expert]`).
4. Extend `TargetLayer` in `src/internal.h`:
   - drop `w_gate, w_up, w_down` from the dense-FFN fields (or keep as a union)
   - add `ffn_gate_inp` (router), `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`
     (all MXFP4 rank-3 tensors).
5. Extend `src/gguf_target_loader.cpp`:
   - Look up `blk.{il}.ffn_gate_inp.weight`, `blk.{il}.ffn_gate_exps.weight`,
     `blk.{il}.ffn_up_exps.weight`, `blk.{il}.ffn_down_exps.weight`.
   - Tensor name schema is fixed in llama.cpp's `llm_tn` mapping; mirror it.

### 3b. If the GGUF arch is pure attention + MoE (like `qwen3moe`)

Bigger simplification: delete the SSM state machinery.

1. Drop the three tree-mode ggml ops from the critical path (they stay linked
   because other TargetLayer fields still reference them, but aren't called).
2. Drop `CpuEmbedder` if MXFP4 supports `ggml_get_rows` on CUDA — check
   `dflash/deps/llama.cpp/ggml/src/ggml-cuda/getrows.cu` for MXFP4 case. If
   absent, keep CpuEmbedder (cheap 4 MiB/layer win isn't worth breaking).
3. `delta_net_chunked.*` and all `ssm_*` bookkeeping in `test_dflash.cpp`
   become dead code for this target. Do **not** delete — keep the 27B path
   shipping. Gate on the arch.
4. MoE FFN port is the same as 3a, step 3.

The pure-attention variant is the easier port. Hope for it on pre-work #2.

---

## Engine changes (shared between 3a and 3b)

### 4. Draft graph

`src/safetensors_draft.cpp` + `src/qwen3_dflash_graph.cpp` assume the
z-lab DFlash-27B draft shape (5 layers, target-layer conditioning on ids
`{1,16,31,46,61}`). Two cases:

- **Chain-spec fallback draft** (a small Qwen3 dense): rewrite the draft
  loader to use a standard Qwen3 `llm_build_qwen3` forward pass on bf16
  safetensors (or reuse llama.cpp's Qwen3 graph verbatim — we already link
  ggml, not libllama, so we'd need to hand-port like we did for 27B).
- **DFlash-trained draft for 3.6** (later milestone): keep the existing
  5-layer block-diffusion draft structure, change the target-layer capture
  indices to match the new target's layer count (~64 for 35B depending on
  arch).

### 5. VRAM budget for 5090

Estimated VRAM at budget=22:

| Component                        | Size (GiB) | Notes |
|----------------------------------|:----------:|-------|
| Target weights (MXFP4, 35B)      | ~17.5      | 4 bits/weight + 1-byte microscale per 32 elements |
| Draft weights (Qwen3-1.7B bf16)  | ~3.4       | fallback |
| KV cache (F16, ~64 layers)       | 4-6        | depends on n_head_kv, ctx 8K |
| Verify tree state (budget=22)    | 1-2        | attention-only: per-tree-node KV slot; hybrid: +SSM intermediate |
| Activations + scratch            | ~1         | |
| **Total**                        | **~27-30** | Fits on 5090 (32 GB). 3090 (24 GB) does **not** fit without offload. |

### 6. CMake + build flags

- `CMakeLists.txt` hardcodes `CUDA_ARCHITECTURES "120"` on the `rtx-5090`
  branch — keep. For dual-arch builds re-add `-DCMAKE_CUDA_ARCHITECTURES=86;120`.
- Add `src/qwen36_target_graph.cpp` to the `dflash27b` static library target.
- New CLI flag on `test_dflash`: `--arch=qwen36` selects the new graph. Today
  the arch is implicit (qwen35 hybrid). Keep backward compat.

### 7. CPU embedder

If MXFP4 has no `get_rows` CUDA kernel for the output embedding, keep the
CpuEmbedder pattern from `src/internal.h`. Only ~70-150 MiB saved on 35B,
but matches llama.cpp behaviour and avoids bug classes. Grep to verify:
```bash
grep -n "GGML_TYPE_MXFP4" dflash/deps/llama.cpp/ggml/src/ggml-cuda/getrows.cu
```
(check before and after pre-work #1.)

---

## Validation strategy

Each step has a numeric gate. Don't advance past a failing gate.

1. **GGUF loads.** `test_dflash` init runs through, all tensor lookups succeed,
   no `set_last_error` fires. Gate: process exits 0 on a no-op run.

2. **AR parity.** Run `test_generate` against the new target on a canned prompt;
   run llama.cpp `llama-cli` with the same prompt, same seed (greedy).
   Gate: token sequences match bit-for-bit for the first 64 decoded tokens.

3. **Chain-spec first-token.** Run `test_dflash --no-ddtree --n-spec=8` with
   the small dense draft. Gate: generates coherent text at ≥1.5× AR tok/s
   on HumanEval prompt 01.

4. **DDTree enable.** Flip `--ddtree --ddtree-budget=22`. Gate: no crashes,
   tok/s ≥ chain-spec, AL ≥ 4 on HumanEval prompt 01.

5. **Full bench.** `scripts/bench_llm.py` with the new target and draft.
   Gate: HumanEval DFlash tok/s ≥ 2× AR, Math500 and GSM8K ≥ 1.5× AR.

6. **Correctness.** `test_vs_oracle` port. Gate: draft-graph cos-sim ≥ 0.999
   vs PyTorch reference forward at the same prompt (matches the existing
   27B check).

7. **Budget sweep.** Re-run §5 at budgets {16, 22, 28, 34, 40}. Gate: peak
   tok/s within the sweep becomes the committed default.

---

## Milestones

| Milestone | What ships | Est. effort |
|-----------|-----------|:-----------:|
| M0: Pre-work + arch confirm | This doc updated with concrete shapes, llama.cpp sanity on MXFP4 target | 0.5 day |
| M1: AR-only path for Qwen3.6 target | `test_generate` prints coherent text, bit-match with llama.cpp | 2-3 days |
| M2: Chain-spec with dense Qwen3 draft | `test_dflash --no-ddtree` at ≥1.5× AR | 1-2 days |
| M3: DDTree re-enabled | `--ddtree --ddtree-budget=22` passes gates 4-5 | 2-3 days |
| M4: Full bench + sweep + RESULTS_5090 | Published numbers | 1 day |
| M5: (stretch) Train a DFlash draft for 3.6 | Move chain → block-diffusion draft | 1-2 weeks, needs H100 time |

Total to M4: roughly **one to two working weeks** if the pre-work answers are
benign (arch is already in llama.cpp, MXFP4 CUDA path works end-to-end, a
usable small draft exists). M5 is a separate project.

---

## Risks & mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|:---:|:---:|---|
| `general.architecture` is new (not in our fork of llama.cpp) | medium | blocks M1 | cherry-pick the upstream llama.cpp arch commit onto our `luce-dflash` branch; or rebase our fork onto recent master |
| MXFP4 `mul_mat_id` (expert matmul) CUDA kernel missing | medium | blocks M1 | file upstream; chain-spec at AR speed is still shippable |
| No compatible small draft with matching tokenizer | low | slows M2 | Qwen3 family tokenizers are stable across sizes; fallback is Qwen3-0.6B |
| DDTree verify costs blow VRAM at budget=22 | medium | forces smaller budget | sweep starting at budget=16; 32 GB headroom is real but not infinite |
| DFlash draft training for 3.6 is uneconomical | high | kills M5 | skip M5; chain + DDTree still delivers 2-2.5× |
| MoE routing kernels are launch-latency bound at batch=1 | medium | lower-than-expected tok/s | accept; the prize here is absolute tok/s, not speedup ratio |

---

## What this port teaches us that the 27B didn't

- Whether the tree-mode SSM ops generalize to hybrid-MoE (likely yes) or
  are load-bearing only for the specific 27B shape (possible).
- Whether DDTree still pays at smaller *active* parameter counts (A3B means
  per-step target compute is much lighter; the budget vs tok/s curve may
  shift *up*, not *down*, because verify is cheaper).
- Whether MXFP4 per-position acceptance is closer to BF16 than Q4_K_M was
  (MXFP4's per-32-element microscales preserve more dynamic range).
- A real data point for the "Q5/Q6 was the 5090 unlock" story: if 35B in
  MXFP4 ≈ 17 GB, we can afford a Q8_0 or even BF16 *draft*, which may be
  the next knob to tune.

---

## Open questions (for the user / upstream)

1. Is "Qwen3.6" the confirmed release name, or a stand-in for Qwen3.5-MoE /
   Qwen3-Next / a yet-unreleased model? The GGUF metadata will answer.
2. Is there a tokenizer mismatch risk vs the existing DFlash 27B draft?
   (Qwen3.5 and Qwen3 share a 248320-vocab BPE; Qwen3-Next may differ.)
3. Budget for M5 (draft training)? If no H100 access, M5 stays deferred and
   chain+DDTree is the ship-it path.
