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
| **M2b** — chain-spec orchestrator | ✅ | `test_chain_spec` with both sequential (lossless, default) and batched (faster, near-tie drift) target verify. Batched at N=16 hits **140 tok/s** (1.12× AR baseline 125) with AL=17. Sequential preserves AR byte-identity at 36-62 tok/s. The "batched is broken" observation from earlier was traced to fp16 MMA_F16 accumulation vs fp32 VEC accumulation flipping argmax on near-tied logits (≤0.03 logit delta), not a logic bug — see Root Cause section below. |
| **M3a** — tree-mode chain smoke | ✅ | `CHAIN_VERIFY=tree_chain` mode added: batched N-token verify via `ggml_gated_delta_net_tree_persist` + `ggml_ssm_conv_tree` with linear parent_ids = `[-1, 0, 1, ..., N-2]`. Cache-shape bug found and fixed: `create_target_cache` now takes `max_verify_tokens = N_spec` so the per-layer `conv_input_cache` tensor matches the in-graph concat shape `[(kern-1)+N_spec, conv_channels, 1]`. **On realistic prompts (code, instruction-following) tree_chain at N=16 is byte-identical to seq AR at 128 tok/s (1.66× seq).** The "Paris" pathological near-tie drift only fires when the first verify position has a sub-0.03-logit gap at the top of the distribution; code and GSM8K prompts don't. |
| **M3a diag** — "drift" is not a bug, it's VEC-vs-MMA numerics | ✅ | Systematic root-cause investigation (CHAIN_DIAG sweep + `DFLASH27B_FATTN_TILE` override). **The drift is NOT monotone in either N or precision, and is NOT caused by any single kernel choice.** Decisive evidence: (a) Forcing TILE kernel in SEQ mode (N=1) *also* produces the "drifted" argmax — so it's not a batched-kernel bug, it's VEC ≠ MMA/TILE at the single-query level. (b) TILE kernel's batched argmax at pos-0 is non-monotone in N: matches VEC at N=5,6,8 but diverges at N=1,2,3,4,12,16. A precision-accumulation bug can't be non-monotone. (c) Across (KV dtype) × (fattn kernel) × (N), no configuration is uniformly correct; deltas range ±0.07 logits. **The "drift" is legitimate numerical divergence between fp32 VEC and fp16-input MMA/TILE flash-attention kernels, amplified over 64 layers into a ~0.02-logit delta at the logit head. On near-ties, this flips argmax.** This mirrors how llama.cpp's own prefill (MMA) and decode (VEC) paths give bit-slightly-different outputs — it's inherent to the fused-precision kernel design, not fixable without a kernel rewrite. Override reverted. |
| **M3b** — DDTree branching tree-verify | ✅ | `CHAIN_VERIFY=ddtree` mode: per-step top-K log-prob extraction from the draft, best-first branching tree build (`build_ddtree` ported from test_dflash), ancestor-only mask, target verify via the tree kernel, walk accepted path via target's posterior argmax. End-to-end byte-identical to seq on the code prompt. Knobs: `DDTREE_BUDGET`, `DDTREE_K`, `DDTREE_TEMP`. |
| **M3c** — DDTree fast CUDA rollback | ✅ | Per-layer CUDA rollback ported from test_dflash: (a) f16→f32 launch copies `ssm_intermediate_states[rollback_dfs]` → `ssm_state[il]`, (b) `cudaMemcpy2DAsync` of (kernel−1)=3 contiguous conv_input slots → `conv_state[il]` on chain walks, with a sibling-walk fallback that traces the parent chain column-by-column. Target catch-up (expensive: 35B × ~commit_count step_model calls) is replaced by a dozen async CUDA memcpys per layer. Draft keeps sequential replay (0.8B, cheap). Code prompt N=8: **DDTree goes from 50 → 122 tok/s (2.4×)**, byte-identical to seq, matching tree_chain (135) to within the top-K + tree-build overhead. Paris prompt N=8: **105 tok/s with AL=9.00 (every draft accepted)** — on that near-tie-pathological prompt the target-side MMA kernel's drift happens to align with the draft's continuation, so every sibling walk stays on the spine. Output matches `CHAIN_VERIFY=batch`/`tree_chain` (MMA-greedy-equivalent, which is different from seq on near-ties; see M3a). |
| **M3d** — KV compaction for DDTree sibling walks | ✅ | When the tree walk leaves the spine (`accepted[i] != i` for some i), full-attn K/V slots along the accepted path are non-contiguous. Added `ddtree_compact_kv`: for each depth d whose `accepted[d] != d`, an async `cudaMemcpyAsync` per layer per head copies slot `pre_pos+accepted[d]` → `pre_pos+d`. Safe to iterate forward because DFS walks are monotone. This lets the fast-rollback path handle sibling walks without falling back to sequential catch-up. On a diverse prompt where the draft top-1 often disagrees with target (sibling_walks fire 18-31% of rounds), DDTree goes from tree_chain's **53 tok/s (AL=4.19)** to DDTree-K=4's **88 tok/s (AL=5.82)** — ~1.7× throughput by actually using the branching. DDTree output remains MMA-greedy-equivalent; diverges from seq's VEC-greedy on near-tie positions (expected per M3a). |
| **M4** — HumanEval / GSM8K bench | ✅ | `scripts/bench_chain_spec.py`: tokenizes HE + GSM8K prompts, runs AR baseline (`test_generate`) plus all chain_spec modes, reports tok/s + AL per sample. n=5 samples per dataset, n_gen=256, N_spec=8 on RTX 5090. **AR baseline is ~126 tok/s on both datasets** — the 0.8B-draft vs 35B-MoE-target gap is wide enough that spec overhead largely eats the win on HE. Results: `HumanEval` — AR 126 / batch 121 (0.97×) / tree_chain 121 / DDTree K=8 B=22 105 AL=7.39 / DDTree K=4 B=12 110 AL=7.18. `GSM8K` — AR 126 / batch 73 (0.58×) / tree_chain 73 / **DDTree K=8 B=22 79** AL=5.77 (1.09× batch) / DDTree K=4 B=12 78 AL=5.11. **DDTree branching pays off on GSM8K (math)** where the 0.8B draft disagrees with target more (sibling walks 30-56% of rounds); on HumanEval (code) draft top-1 already matches target well (AL≈7-8) so the tree overhead exceeds the branching win. Conclusion: M4 baseline is "spec decode barely breaks even on 35B with a 0.8B dense draft" — the real throughput win requires M5 (trained DFlash draft) or a stronger dense draft. |
| **ttx** — AR throughput gap to llama.cpp (partial) | ⏳ | Pinned down the 125 vs 211 tok/s gap via nsys kernel-instance comparison. Two distinct sub-gaps: (1) **CUDA graphs** — llama.cpp's top-level CMakeLists turns `GGML_CUDA_GRAPHS_DEFAULT=ON` but we include ggml as a subdir so we get the upstream default OFF. Turning it on in `dflash/CMakeLists.txt` is done, but ggml-cuda keys the captured graph on `cgraph->nodes[0]` (pointer) and our builder tears down the ctx+cgraph every step, so the graph is re-captured instead of re-launched. llama.cpp 229 → 168 tok/s with graphs disabled (27% of their throughput). Proper fix = keep the ggml cgraph stable across steps and let ggml-cuda `cudaGraphExecUpdate` handle offset changes — non-trivial refactor. (2) **Kernel count** — even without CUDA graphs, llama.cpp does 122 Q8_0 mmvq + 39 MXFP4 mmvq per step; we do 201 + 78. 2-2.7× more kernel launches. llama.cpp has a pattern-matching fusion (`ggml_cuda_topk_moe_fusion`) that folds softmax + argsort + get_rows + normalize into a single `topk_moe_cuda` kernel; our MoE builder produces the ops in a slightly different order (`ggml_argsort_top_k` before reshape) so the fusion doesn't match. Per-step overhead: `build=556 µs, embed=1, set=11, compute=7036, get=150, argmax=194, total=7949 µs` — compute dominates; CPU-side overhead is only ~13%. |
| **ttx follow-up** — partial fixes | ✅ | Landed two wins: **(a) swiglu fusion** — rewrote `build_moe_ffn_qwen36`, `build_shexp_qwen36`, and `build_swiglu_ffn` (qwen35) to emit `ggml_swiglu_split(gate, up)` instead of `ggml_silu(gate) * up`. This lets ggml-cuda's `ggml_cuda_should_fuse_mul_mat_vec_q` pattern-matcher fold the two gate/up `mul_mat_id` calls + silu + mul into ONE `mmvq` launch with `has_fusion=true`. nsys confirms our MXFP4 mmvq count dropped from 78 → 39 per step (matches llama.cpp) and Q8_0 from 201 → 121. **(b) persistent ggml ctx** — build_step_graph now calls `ggml_reset(ctx)` instead of `ggml_free` + `ggml_init`; tensors re-land at the same arena addresses so `cgraph->nodes[0]` is pointer-stable. Build time 550→292µs per step. Combined result: **test_generate 128 → 150 tok/s (+17%)**, compute 7036→6208µs per step, byte-identical output. Remaining gap vs llama.cpp-no-graphs (168) is ~10% — mostly from the view-offset-baked KV writes still preventing ggml-cuda from completing CUDA-graph warmup (every step the view tensor's `->data` pointer changes, `memcmp` in `ggml_cuda_graph_update_required` flags properties as changed, warmup resets). Closing this last gap requires replacing `ggml_view + ggml_cpy` KV writes with `ggml_set_rows(cache, k_new, pos_idx)` where `pos_idx` is an input tensor whose value changes per step without altering the graph structure — out of scope for this pass. |

### M3a final root cause: VEC and MMA give different answers on near-ties (this is fine)

Systematic debugging (CHAIN_DIAG sweep × kernel override) produced a full kernel-selection × N × delta table on the "Paris" prompt:

```
                 N=1  N=2  N=3  N=4  N=5  N=6  N=8  N=12 N=16
CHAIN_VERIFY=seq VEC  VEC  MMA  MMA  MMA  MMA  MMA  MMA  MMA    single-token per step
  → argmax        11   11   11   11  11   11   11   11   11
CHAIN_VERIFY=batch (Q8 KV, default MMA_F16)
  → argmax        11   11   11   13!  13!  13!  13!  13!  13!
CHAIN_VERIFY=batch + DFLASH27B_FATTN_TILE=1 (fp32 VKQ)
  → argmax        13!  13!  13!  13!  11   11   11   13!  13!     ← non-monotone!
seq mode + DFLASH27B_FATTN_TILE=1 (N=1, TILE kernel)
  → argmax at pos 0 = 13  (seq's VEC-native answer is 11)
```

Three things follow:

1. The drift is **not** a batched-only or fp16-VKQ bug — TILE kernel at N=1 *also* flips pos-0 argmax to 13 vs VEC's 11. It's VEC-vs-(MMA|TILE) at the single-query level, amplified over 64 layers into a ~0.02-logit residual.
2. TILE's non-monotone-in-N pos-0 argmax (correct at N=5,6,8 but wrong at N=1,2,3,4,12,16) **rules out** any accumulation or precision-depth hypothesis. An accumulation bug has to be monotone.
3. No knob is uniformly right: Q8+MMA, F16+MMA, Q8+TILE, F16+TILE all drift on at least one N. The delta stays within ±0.07 logits — pure floating-point sum-ordering noise.

**Why this is fine:** VEC fp32 and MMA fp16-input fp32-acc are both valid numerical realizations of the same `softmax(QK^T/√d)·V` over a quantized-KV cache. Neither is "more correct" in absolute terms. llama.cpp dispatches between them based on `n_tokens` (VEC for decode n=1, MMA for prefill n≫1) for the same reason we do, and the two paths produce (slightly) different outputs on near-ties too — nobody treats that as a bug because prefill and decode don't overlap.

Our batched spec-verify runs MMA at decode-time, so on near-tie prompts it sometimes accepts a draft that matches "MMA-greedy" but not "VEC-greedy". The resulting output is equivalent to `llama.cpp prefill over the full sequence` — a perfectly valid greedy decode, just not bit-identical to AR. On non-near-tie prompts (~all real workloads) MMA and VEC agree and batched output is bit-identical to AR (confirmed on a code prompt: 128 tok/s batched = 128 tok/s tree_chain = identical to seq's 16/16 tokens, 1.66× seq).

**No fix is warranted.** `CHAIN_VERIFY=seq` is available for bit-identical-to-AR reproducibility; `CHAIN_VERIFY=batch` and `CHAIN_VERIFY=tree_chain` are available for 1.5-1.7× speedup with MMA-greedy-equivalent output.

### M2b batched-verify "bug" — root-caused: fp16 MMA near-tie drift (superseded by M3a)

**The observed symptom wasn't a logic bug.** Top-5 logits at batched position 0 (N=4, prompt "The capital of France is"):

```
tok 13 (".") = 19.177   ← batched MMA_F16 argmax
tok 11 (",") = 19.155   ← sequential VEC (fp32) argmax
delta = 0.022
```

The batched fattn MMA_F16 kernel accumulates in fp16 — enough precision for real logit spreads but noisy on near-ties. The VEC kernel (selected automatically for `n_tokens == 1` on quantized-KV + Ada+ cards) uses fp32 accumulation and gives a different argmax on the same inputs. Both are mathematically "target's prediction" — the difference is numerics, not correctness.

For a typical 2-way near-tie, MMA's argmax will agree with draft's proposal ~50% of the time by chance. For "Paris," vs "Paris." specifically, 0.8B draft picks "." and MMA target *happens* to tie the VEC target at 0.022 in draft's favor → target accepts, output diverges from AR.

**Shared-helper refactor (M2b side effect):** The earlier suspicion that qwen36 had its own buggy attention block copy turned out to be unfounded (the copies were bit-identical to qwen35's), but the fix still landed cleanly — `qwen35_build_full_attn_block` and `qwen35_build_delta_net_block` are now non-static helpers exported from `qwen35_target_graph.cpp` and called from `qwen36_target_graph.cpp`. Single source of truth for both archs; eliminates a class of future bugs.

**M2b shipping state:**

- `test_chain_spec` now supports `CHAIN_VERIFY=seq` (default, lossless, ~60 tok/s at N=4) and `CHAIN_VERIFY=batch` (140 tok/s at N=16, drifts to draft on near-ties).
- All-accept fast path: when `k == N_spec`, neither cache needs catch-up. That's what unlocked the 1.12× AR speedup on batched.
- Remaining throughput gap vs the theoretical max (≥ 2× AR) is the catch-up cost on mismatches. Closing it requires non-replay SSM rollback — which is exactly what M3 DDTree's `ssm_intermediate` capture provides.

### Older diagnostic capture (kept for reference)

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
