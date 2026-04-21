# Porting plan: Qwen3.6-35B-A3B MoE (MXFP4) to the DFlash engine

**Target artifact:** `Qwen3.6-35B-A3B-MXFP4_MOE.gguf` — 35 B-parameter sparse MoE
with ~3 B active params per token, MXFP4-quantized expert weights, distributed
as a single GGUF.

**Hardware target:** single RTX 5090 (32 GB, sm_120). Secondary target: 3090
(24 GB) — expected *not* to fit without offload.

This plan is written from what the `dflash/` codebase does today for Qwen3.5-27B
hybrid + Q4_K_M and extends each layer with what the MoE/MXFP4 variant needs.

---

## Status (`qwen36-port` branch on the `signalnine` fork)

| Milestone | State | Notes |
|---|---|---|
| **M0** — arch + MXFP4 CUDA sanity | ✅ | `general.architecture=qwen35moe`, hybrid DeltaNet + MoE. `llama-bench`: 6,472 pp512 / 210.7 tg128 via the pinned fork. |
| **M1a** — dual-arch scaffolding | ✅ | `TargetWeights.arch` tag, MoE fields on `TargetLayer`, shared loader for `qwen35` and `qwen35moe`, `build_target_graph()` dispatch. |
| **M1b** — 35B-A3B graph builder | ✅ | `qwen36_target_graph.cpp` with the full stack: Q-packed full-attn (IMROPE) + fused-op DeltaNet + MoE FFN (`ggml_mul_mat_id` on MXFP4 gate/up + normalized-weight sum) + shared expert (sigmoid-scalar gate). End-to-end coherent output on "The capital of France is" → "Paris, a city renowned for its iconic landmarks such as". **125.8 tok/s decode**, ~60 % of llama.cpp's 210.7 baseline. |
| **M2a** — runtime-dim qwen35 builder | ✅ | `build_full_attn_block` / `build_delta_net_block` / `create_target_cache` now read `w.n_*` / `w.ssm_*` at runtime, not q35:: constants. Loader accepts any qwen35 size and tied LM head. Rope type standardised to `GGML_ROPE_TYPE_IMROPE` for both arches. Qwen3.5-0.8B-BF16 now loads + runs through the same engine as 27B and 35B. |
| **M2b** — chain-spec orchestrator | ✅ (sequential verify) + 🐛 (batched verify bug, isolated) | New binary `test_chain_spec` loads both models via `load_target_gguf`, runs chain-spec with `snapshot_ssm_state` + catch-up rollback. Greedy output matches 35B AR byte-for-byte. Sequential verify throughput 45-63 tok/s depending on N_spec — slower than AR (125) because single-token verify + draft catch-up together cost more than they save at our draft/target speed ratio (0.8B = 300, 35B = 125, i.e. draft ~2.4× faster). The batched verify bug is now isolated to a specific n_tokens pattern (see below); it's NOT arch-specific and blocks M2b's actual speedup. |

### M2b batched-verify bug — diagnostic state

Isolated with a minimal reproducer (`CHAIN_DIAG=1 test_chain_spec ...`): sequential target decode of 8 tokens from "The capital of France is" gives `[11, 264, 3177, 34756, 364, 1141, 25438, 57902]` (= ", a city renowned for its iconic landmarks"). Feeding the SAME input sequence through one batched forward, we expect to recover those 8 argmax predictions exactly — both paths evaluate target greedy on the same (state, inputs). Instead:

```
seq         : 11   264  3177 34756  364  1141 25438 57902
batched N=1 : 11                                            (OK — VEC kernel)
batched N=2 : 11   264                                      (OK)
batched N=3 : 11   264  3177                                (OK)
batched N=4 : 13!  264  3177 34756                          (pos 0 WRONG, 1..3 correct)
batched N=5 : 13!  264  3177 34756  364                     (pos 0 WRONG)
batched N=6 : 13!  264  3177 34756  364  1141               (pos 0 WRONG)
batched N=7 : 13!  264  3177 34756  364  1141 25438         (pos 0 WRONG)
batched N=8 : 13!  264  3177 34756  364  1141 25438 57902   (pos 0 WRONG)
```

Observations that constrain the fix:

1. **Only position 0 is affected.** Positions 1..N-1 match sequential exactly at every N. So K/V writes, RoPE, SSM evolution, MoE FFN for positions 1..N-1 are all producing the RIGHT per-position logits.
2. **Threshold is N=4.** N=1,2,3 are correct; N>=4 is wrong at position 0.
3. **Deterministic.** Running batched twice back-to-back gives the same wrong pos-0 output — not a race.
4. **Not KV alignment (simple fix).** Tried padding kv_len to FATTN_KQ_STRIDE=256 (and matching mask width): that *didn't* fix N>=4 pos 0 AND broke N=2. (The padding-unaware mask now reaches slots with uninitialized K/V, so the kernel's picks a different VEC/MMA path mid-range.)
5. **Not SSM state or MoE.** The kernel selection logic in `fattn.cu:ggml_cuda_get_best_fattn_kernel` switches paths at n_tokens boundaries (VEC for n<=2 on Ada+ with quantized K/V, MMA_F16 above). Wrong pos-0 starts right at the MMA_F16 threshold.
6. **Only the FIRST new position.** Query at `pre_pos` reads cache slots [0..pre_pos] where slot `pre_pos` is just-written from input_0 (the same input as single-step's K for that token). Attention output for q=0 differs between batched-MMA and single-token-VEC despite identical mathematical inputs.

Next-session diagnostics (in order of decreasing hypothesis plausibility):

1. **MMA_F16 kernel bug at the "just-written" KV slot.** Add a one-liner to `build_full_attn_block` that forces VEC by pre-padding Qfa to n_tokens=2 and masking the extra slot; if batched then matches sequential, the bug is in the MMA kernel. If it doesn't, look elsewhere.
2. **cpy-before-read ordering.** Manually add an explicit data dependency between the K/V `ggml_cpy` ops and `ggml_flash_attn_ext` (e.g., make `Kfa` `src[1]` of a no-op after the cpy so the graph scheduler can see the dep). Rebuild, rerun.
3. **Bisect by head count.** GQA is 16/2 = 8. Try a build with n_head_kv=1 (hack the loader's check) to see if the broadcast from KV to Q heads is involved.
4. **Diff vs test_dflash's build_target_step flow.** test_dflash's chain verify path works at n_tokens up to 16 on 27B. Look for what differs — maybe the `capture_delta_intermediate=true` path forces a different-and-correct op selection.
| M3 — DDTree verify on MoE | ☐ | Tree-mode SSM ops already wired through the graph (parent_ids path); driver changes only. |
| M4 — full bench + throughput tuning | ☐ | Close 125 → 200+ tok/s gap (likely KV type, MoE routing microcode). |
| M5 — DFlash-trained draft | ☐ | Stretch; 3-5 days H100 time. |

### Numbers shipped so far

| Model (arch) | First-token correctness | Decode tok/s (our engine) | llama.cpp AR baseline |
|---|---|:-:|:-:|
| Qwen3.5-0.8B (qwen35 dense) | ✓ " Paris." | 300 | — |
| Qwen3.6-35B-A3B MXFP4 (qwen35moe) | ✓ " Paris, a city renowned for its iconic landmarks…" | 125.8 | 210.7 |

### Key lessons from the port so far

1. **Arch-family commonality is higher than I initially expected.** The DeltaNet
   + FullAttn layer dispatch, SSM state layout, RMS+gate output norm,
   sigmoid-gated Q-packed attention — all identical across 0.8B, 27B, and the
   35B-A3B MoE variant. The single hybrid builder with runtime dims handles
   all three.
2. **MXFP4 CUDA support is already in our pinned `Luce-Org/llama.cpp@luce-dflash`
   submodule.** No kernel work needed; `GGML_TYPE_MXFP4` has registered dequant
   and `ggml_mul_mat_id` paths. One fewer upstream dependency.
3. **Qwen3.5 and Qwen3.5-MoE both use `GGML_ROPE_TYPE_IMROPE` (40), not `MROPE`
   (8).** Our original 27B code used MROPE and produced coherent output, but
   the llama.cpp-canonical type is IMROPE. We've switched to it for all qwen35
   sizes; the 35B's coherent output confirms it's the right call.
4. **Tied LM head handling** (the 0.8B has no `output.weight`; it aliases
   `token_embd.weight`). Cheap to support — upload the embedding table to GPU
   *in addition to* the CpuEmbedder row-lookup path, and alias `w.output ←
   w.tok_embd`.
5. **Chain-spec with matching tokenizer requires a same-family model.** The
   natural-looking "small Qwen3-1.7B as draft" path fails because Qwen3.x uses
   a 151,643-vocab tokenizer while Qwen3.5+ uses 248,320. Only *Qwen3.5-0.8B*
   is publicly available in the target's tokenizer family. Our megakernel
   project target gets reused as the DFlash draft — one fewer model family to
   support.

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

## Pre-work (M0) — status: verified

GGUF on disk: `/mnt/ai/models/huggingface/qwen3.6-35b-a3b-GGUF/Qwen3.6-35B-A3B-MXFP4_MOE.gguf` (20.2 GiB).

### 1. Architecture ✓
```
general.architecture = qwen35moe
```
Same arch name as our reference `deps/llama.cpp/src/models/qwen35moe.cpp`. Our fork already has the forward-pass reference. **Plan branch §3a (hybrid DeltaNet + MoE) applies.**

### 2. Hybrid vs pure attention ✓
`qwen35moe.full_attention_interval = 4`, confirmed by tensor-level inspection:

| Layer indices | Type | Tensors |
|---|---|---|
| 0, 1, 2, 4, 5, 6, 8, 9, 10, … | DeltaNet (SSM) | `attn_qkv`, `attn_gate`, `ssm_a`, `ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_dt`, `ssm_norm`, `ssm_out` |
| 3, 7, 11, 15, 19, 23, 27, 31, 35, 39 | Full attention | `attn_q`, `attn_k`, `attn_v`, `attn_q_norm`, `attn_k_norm`, `attn_output` |

3 DeltaNet + 1 Attention, repeating — identical 3:1 ratio to Qwen3.5-0.8B and 3.5-27B. Our tree-mode SSM ops (`ggml_ssm_conv_tree`, `ggml_gated_delta_net_tree[_persist]`) apply directly.

Full-attention layers use **separate** Q/K/V (not fused like DeltaNet's `attn_qkv`), with per-head Q-norm and K-norm (standard Qwen3 pattern).

### 3. Shape constants ✓

```c
// Candidate header: include/dflash36.h  (or extend existing internal.h)
#define QWEN36_N_EMBD                 2048      // vs 27B's 5120
#define QWEN36_N_LAYER                40        // vs 27B's 64
#define QWEN36_N_HEAD                 16        // Q heads (per full-attn layer)
#define QWEN36_N_HEAD_KV              2
#define QWEN36_HEAD_DIM               256       // key_length == value_length
#define QWEN36_ROPE_DIM               64        // partial RoPE
#define QWEN36_ROPE_THETA             1e7f
#define QWEN36_RMS_EPS                1e-6f
#define QWEN36_N_EXPERT               256
#define QWEN36_N_EXPERT_USED          8         // top-k routing
#define QWEN36_N_FF_EXPERT            512       // per-expert FFN hidden
#define QWEN36_N_FF_SHEXP             512       // shared-expert FFN hidden
#define QWEN36_FULL_ATTN_INTERVAL     4         // (il+1)%4==0 → full attn
#define QWEN36_CTX_MAX                262144    // 256K
#define QWEN36_VOCAB                  248320    // same as 27B
// SSM state dims still need to be dumped from ssm_* tensor shapes
// (`ssm_a` is [32], so SSM heads = 32; key/value dim TBD from ssm_conv1d shape [4, 8192])
```

Active params per token (rough count): attention ≈ 19 M/layer, expert+shared FFN ≈ 19 M/layer → ~40 M/layer × 40 = ~1.6 B active (consistent with the "A3B" label, which typically bundles embedding lookup).

### 4. MXFP4 end-to-end sanity ✓

Tensor dtypes in the GGUF:
| Role | Dtype | Count | Shape |
|------|:-----:|:-----:|-------|
| `ffn_gate_exps`, `ffn_up_exps` | **MXFP4** | 78 | [2048, 512, 256] (rank-3 MoE) |
| `ffn_down_exps` | Q5_K (38) / Q6_K (4) | 42 | [512, 2048, 256] |
| `attn_qkv`, `attn_q/k/v`, `attn_output`, `attn_gate`, `ffn_{gate,up,down}_shexp`, `token_embd`, `output` | Q8_0 | 252 | dense |
| norms, `ssm_a`, `ssm_alpha/beta`, `ssm_conv1d`, `ssm_dt`, `ffn_gate_inp[_shexp]` | F32 | 361 | small |

MXFP4 CUDA dequant is already in our pinned submodule: `ggml-cuda/convert.cu:dequantize_block_mxfp4` registered for `GGML_TYPE_MXFP4`. `llama-bench` against the pinned `deps/llama.cpp` (rebuilt with `-DCMAKE_CUDA_ARCHITECTURES=120`) loads the MXFP4 GGUF and runs prefill + decode cleanly:

| Test | tok/s |
|:---:|:---:|
| `pp512` (batched prefill) | **6,472** |
| `pp8` (short prefill) | 395 |
| `tg128` (decode) | **210.7** |
| `tg16` (decode) | 192.3 |

**Reference numbers for the DFlash port to beat** — these are llama.cpp's own MXFP4 autoregressive baseline on our 5090. For comparison, Qwen3.5-27B Q4_K_M AR on the same 5090 is ~58 tok/s tg (see `dflash/RESULTS.md`); A3B's sparse activation gives 3.6× the AR throughput despite being nominally a larger model, because only ~3 B of the 34.66 B params are active per token.

Note: `llama-cli` was unusable against this GGUF (kept echoing empty `> ` prompts, probably a chat-template interaction). Not blocking — we bypass llama-cli entirely in our engine and only link ggml.

### 5. Draft model ✓ (decision)

No z-lab DFlash-trained draft for Qwen3.6. Ship M1-M4 with **chain-speculative + small Qwen3-family dense draft** (matching `gpt2` BPE tokenizer, 248320-vocab). Candidates:
- `Qwen/Qwen3-1.7B` or `Qwen/Qwen3-4B` (tokenizer verified to match Qwen3.5; assume carries to 3.6 — must confirm)
- Larger draft = higher per-step cost but better acceptance; sweep after M3.

M5 (DFlash-trained draft) remains deferred.

### Plan-doc deltas resolved by M0

| Open question (before M0) | Answered |
|---|---|
| Arch variant? | `qwen35moe` — use branch §3a |
| Hybrid? | Yes — 3 DeltaNet + 1 Attention per 4 layers, same as 27B |
| MXFP4 CUDA support? | Library path confirmed; runtime smoke test pending |
| Draft with matching tokenizer? | Qwen3-family (gpt2 BPE, 248320 vocab) — small dense draft for M1-M4 |
| Tokenizer ID | `gpt2` BPE |

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
