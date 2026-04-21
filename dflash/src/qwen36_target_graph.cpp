// Forward pass of Qwen3.6-35B-A3B (qwen35moe arch) in pure ggml.
//
// Companion to qwen35_target_graph.cpp. Same hybrid DeltaNet + Full-attention
// dispatch pattern, but with:
//   - smaller hparams (see q36:: namespace below and include/dflash36.h)
//   - MoE FFN: 256 experts × top-8 routed via ggml_mul_mat_id on MXFP4 expert
//     tensors, plus an always-on shared expert gated by a sigmoid-scalar
//   - RoPE type is IMROPE (interleaved, not M-RoPE); rope.dimension_sections
//     in the GGUF is [0] → effectively all-zero sections
//
// What is shared with qwen35 (27B) and therefore NOT re-implemented here:
//   - create_target_cache / free_target_cache / snapshot_ssm_state /
//     restore_ssm_state: arch-agnostic, use w.n_layer / w.ssm_* / etc.
//   - DeltaNet block math (ggml_ssm_conv + ggml_gated_delta_net): same op
//     with different dims
//   - Full-attention block math (flash_attn_ext + packed-Q gate): same op
//     with different dims

#include "internal.h"
#include "delta_net_chunked.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace dflash27b {

// ─── qwen35moe / Qwen3.6-35B-A3B constants ───────────────────────────
namespace q36 {
constexpr int HIDDEN        = DFLASH36_TARGET_HIDDEN;            // 2048
constexpr int N_HEAD        = DFLASH36_TARGET_N_HEAD;             // 16
constexpr int N_HEAD_KV     = DFLASH36_TARGET_N_HEAD_KV;          // 2
constexpr int HEAD_DIM      = DFLASH36_TARGET_HEAD_DIM;           // 256
constexpr int Q_DIM         = N_HEAD * HEAD_DIM;                  // 4096
constexpr int KV_DIM        = N_HEAD_KV * HEAD_DIM;               // 512

constexpr int SSM_D_INNER   = DFLASH36_TARGET_SSM_D_INNER;        // 4096
constexpr int SSM_D_STATE   = DFLASH36_TARGET_SSM_D_STATE;        // 128
constexpr int SSM_DT_RANK   = DFLASH36_TARGET_SSM_DT_RANK;        // 32
constexpr int SSM_N_GROUP   = DFLASH36_TARGET_SSM_N_GROUP;        // 16
constexpr int SSM_CONV_KERN = DFLASH36_TARGET_SSM_CONV_KERN;      // 4

// Derived
constexpr int HEAD_V_DIM    = SSM_D_INNER / SSM_DT_RANK;          // 128
constexpr int HEAD_K_DIM    = SSM_D_STATE;                         // 128
constexpr int CONV_CHANNELS = SSM_D_INNER + 2 * SSM_N_GROUP * SSM_D_STATE;  // 4096+4096=8192

// MoE
constexpr int N_EXPERT      = DFLASH36_TARGET_N_EXPERT;           // 256
constexpr int N_EXPERT_USED = DFLASH36_TARGET_N_EXPERT_USED;      // 8
constexpr int FFN_EXPERT    = DFLASH36_TARGET_FFN_EXPERT;         // 512
constexpr int FFN_SHEXP     = DFLASH36_TARGET_FFN_SHEXP;          // 512

constexpr float EPS         = DFLASH36_TARGET_RMS_EPS;            // 1e-6
constexpr float ROPE_THETA  = DFLASH36_TARGET_ROPE_THETA;         // 1e7
constexpr int   ROPE_DIM    = DFLASH36_TARGET_ROPE_DIM;           // 64
} // namespace q36

// Short internal helpers. Mirror qwen35_target_graph.cpp's local helpers so
// both graph builders read similarly; a subsequent cleanup could factor
// these into a shared header once both are stable.
static ggml_tensor * rms_norm_mul_(ggml_context * ctx, ggml_tensor * x,
                                   ggml_tensor * weight, float eps) {
    ggml_tensor * n = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, n, weight);
}

// ─── MoE FFN: 256 experts × top-8, routed via ggml_mul_mat_id ─────────
//
// Returns [HIDDEN, n_tokens].
//
// Ported (stripped) from llama.cpp's llm_graph_context::build_moe_ffn. We
// don't have LoRA / biases / expert groups / GroveMoE / llama4 special cases
// in qwen35moe, so the general path collapses to:
//
//   logits        = ffn_gate_inp @ cur          [N_EXPERT, n_tokens]
//   probs         = softmax(logits)
//   selected      = argsort_top_k(probs, K)     [K, n_tokens]  (K = N_EXPERT_USED = 8)
//   weights       = get_rows(probs, selected)   [K, n_tokens]
//   weights       = weights / sum(weights)      (norm_w = true; clamp sum)
//   gate          = silu(mul_mat_id(gate_exps, cur, selected))
//   up            =      mul_mat_id(up_exps,   cur, selected)
//   experts       = mul_mat_id(down_exps, gate*up, selected)
//                       [HIDDEN, K, n_tokens]
//   out           = sum_{k<K} experts[:, k, :] * weights[k, :]
static ggml_tensor * build_moe_ffn_qwen36(
    ggml_context * ctx,
    ggml_cgraph * gf,
    const TargetLayer & L,
    ggml_tensor * cur,            // [HIDDEN, n_tokens]
    int n_tokens)
{
    const int64_t n_embd        = q36::HIDDEN;
    const int64_t n_expert      = q36::N_EXPERT;
    const int64_t n_expert_used = q36::N_EXPERT_USED;

    // Router + top-k + normalize. The exact op sequence matters:
    // ggml-cuda's ggml_cuda_topk_moe_fusion pattern-matches
    //
    //   SOFTMAX -> RESHAPE -> ARGSORT -> VIEW -> GET_ROWS
    //   [ -> RESHAPE -> SUM_ROWS -> CLAMP -> DIV -> RESHAPE ]  (optional norm)
    //
    // into one fused topk_moe_cuda launch. The `src` links must be exact: the
    // RESHAPE after SOFTMAX is "probs_reshaped", the ARGSORT must take the
    // ORIGINAL softmax (src = nodes[node_idx - 2]), and GET_ROWS takes
    // (probs_reshaped, view(argsort)). Any divergence kills the fusion.

    // 1. Router logits
    ggml_tensor * logits = ggml_mul_mat(ctx, L.ffn_gate_inp, cur);
    ggml_set_name(logits, "moe_logits");

    // 2. Softmax → probs  (remains the "original" softmax tensor — argsort
    //    takes this below, fusion requires ARGSORT.src[0] == SOFTMAX).
    ggml_tensor * probs = ggml_soft_max(ctx, logits);
    ggml_set_name(probs, "moe_probs");

    // 3. RESHAPE of SOFTMAX — must appear in the graph immediately after
    //    softmax for the fusion matcher (walks cgraph->nodes sequentially).
    //    DFS traversal from any downstream user of probs_3d will append
    //    this right after SOFTMAX.
    ggml_tensor * probs_3d = ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens);
    ggml_set_name(probs_3d, "moe_probs_reshaped");

    // 4. Top-k selection on the ORIGINAL softmax (not the reshape). Expands
    //    to ARGSORT → VIEW; GET_ROWS picks up the VIEW.
    ggml_tensor * selected = ggml_argsort_top_k(ctx, probs, n_expert_used);
    ggml_set_name(selected, "moe_topk");

    // 5. GET_ROWS(probs_reshaped, view(argsort)) → raw top-k weights.
    ggml_tensor * weights = ggml_get_rows(ctx, probs_3d, selected);
    ggml_set_name(weights, "moe_weights_raw");

    // 6. Normalize: RESHAPE → SUM_ROWS → CLAMP → DIV → RESHAPE.
    weights = ggml_reshape_2d(ctx, weights, n_expert_used, n_tokens);
    ggml_tensor * wsum = ggml_sum_rows(ctx, weights);
    wsum = ggml_clamp(ctx, wsum, 6.103515625e-5f, INFINITY);
    weights = ggml_div(ctx, weights, wsum);
    weights = ggml_reshape_3d(ctx, weights, 1, n_expert_used, n_tokens);
    ggml_set_name(weights, "moe_weights_normed");
    ggml_build_forward_expand(gf, weights);

    // 6. Route cur through top-k experts' gate and up projections. The two
    //    mul_mat_id + swiglu_split triplet is pattern-matched in ggml-cuda
    //    (see ggml_cuda_should_fuse_mul_mat_vec_q) into ONE mmvq kernel with
    //    has_fusion=true, instead of two separate gate/up mmvq calls + a
    //    separate silu + mul. Using ggml_swiglu_split (GLU op) rather than
    //    silu(gate) * up is the key — the raw op form doesn't fuse.
    cur = ggml_reshape_3d(ctx, cur, n_embd, 1, n_tokens);
    ggml_tensor * gate_out = ggml_mul_mat_id(ctx, L.ffn_gate_exps, cur, selected);
    ggml_tensor * up_out   = ggml_mul_mat_id(ctx, L.ffn_up_exps,   cur, selected);
    ggml_set_name(gate_out, "moe_gate");
    ggml_set_name(up_out,   "moe_up");
    ggml_tensor * gu = ggml_swiglu_split(ctx, gate_out, up_out);
    ggml_set_name(gu, "moe_swiglu");

    // 7. Down projection → [n_embd, n_expert_used, n_tokens]
    ggml_tensor * experts = ggml_mul_mat_id(ctx, L.ffn_down_exps, gu, selected);
    ggml_set_name(experts, "moe_experts_down");

    // 8. Weighted sum over the top-k axis.
    //   experts  : [n_embd, n_expert_used, n_tokens]
    //   weights  : [1,       n_expert_used, n_tokens]  → broadcast on dim 0
    experts = ggml_mul(ctx, experts, weights);
    ggml_set_name(experts, "moe_experts_weighted");

    // Sum over dim 1 (the expert_used axis): permute so it's dim 0, then
    // sum_rows.
    ggml_tensor * perm   = ggml_cont(ctx, ggml_permute(ctx, experts, 1, 0, 2, 3));
    ggml_set_name(perm, "moe_experts_permuted");
    ggml_tensor * summed = ggml_sum_rows(ctx, perm);             // [1, n_embd, n_tokens]
    ggml_set_name(summed, "moe_experts_summed");
    ggml_tensor * moe_out = ggml_reshape_2d(ctx, summed, n_embd, n_tokens);
    ggml_set_name(moe_out, "moe_out");
    return moe_out;
}

// ─── Shared expert (always active on qwen35moe) ──────────────────────
// Standard SwiGLU on {ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp}, then
// scaled per-token by sigmoid(ffn_gate_inp_shexp @ cur).
static ggml_tensor * build_shexp_qwen36(ggml_context * ctx, const TargetLayer & L,
                                        ggml_tensor * cur, int /*n_tokens*/) {
    // Same gate/up/swiglu fusion trick as build_moe_ffn_qwen36 — using
    // ggml_swiglu_split lets ggml-cuda fuse the two mmvq's + silu + mul into
    // one kernel with has_fusion=true.
    ggml_tensor * gate = ggml_mul_mat(ctx, L.ffn_gate_shexp, cur);   // [FFN_SHEXP, n_tokens]
    ggml_tensor * up   = ggml_mul_mat(ctx, L.ffn_up_shexp,   cur);   // [FFN_SHEXP, n_tokens]
    ggml_tensor * gu   = ggml_swiglu_split(ctx, gate, up);
    ggml_tensor * out  = ggml_mul_mat(ctx, L.ffn_down_shexp, gu);    // [HIDDEN, n_tokens]

    // Shared-expert sigmoid gate (one scalar per token).
    ggml_tensor * sgate = ggml_mul_mat(ctx, L.ffn_gate_inp_shexp, cur); // [1, n_tokens]
    sgate = ggml_sigmoid(ctx, sgate);
    out = ggml_mul(ctx, out, sgate);                                 // broadcast
    return out;
}

// ─── Main graph builder ────────────────────────────────────────────────
QwenGraphOutputs build_qwen36_graph(
    ggml_context *          ctx,
    ggml_cgraph *           gf,
    const TargetWeights &   w,
    TargetCache &           cache,
    const QwenGraphInputs & in)
{
    const int n_tokens = in.n_tokens;
    ggml_tensor * inpL = in.inp_embed;

    int fa_idx = 0, dn_idx = 0;

    QwenGraphOutputs og{};
    if (in.capture_delta_intermediate) {
        const int n_full_attn = w.n_layer / w.full_attention_interval;
        const int n_delta     = w.n_layer - n_full_attn;
        og.delta_captures.resize(n_delta);
    }

    const float eps = q36::EPS;

    for (int il = 0; il < w.n_layer; il++) {
        const TargetLayer & L = w.layers[il];
        const bool is_attn = (((il + 1) % w.full_attention_interval) == 0);

        ggml_tensor * inpSA = inpL;

        // Pre-attention norm
        ggml_tensor * cur = rms_norm_mul_(ctx, inpL, L.attn_norm, eps);

        if (is_attn) {
            cur = qwen35_build_full_attn_block(ctx, gf, w, L, cur,
                                        in.positions, w.rope_sections,
                                        cache.attn_k[fa_idx], cache.attn_v[fa_idx],
                                        in.attn_mask, in.kv_start, n_tokens);
            fa_idx++;
        } else {
            DeltaNetCapture * cap_ptr = nullptr;
            if (in.capture_delta_intermediate) {
                cap_ptr = &og.delta_captures[dn_idx];
                cap_ptr->ssm_intermediate_states = cache.ssm_intermediate[dn_idx];
                cap_ptr->conv_input              = cache.conv_input_cache[dn_idx];
            }
            cur = qwen35_build_delta_net_block(ctx, gf, w, L, cur,
                                        cache.conv_state[dn_idx], cache.ssm_state[dn_idx],
                                        n_tokens, cap_ptr, in.parent_ids);
            dn_idx++;
        }

        cur = ggml_add(ctx, cur, inpSA);

        // Post-attn norm → MoE FFN (+ shared expert) → residual
        ggml_tensor * ffn_residual = cur;
        ggml_tensor * post = rms_norm_mul_(ctx, cur, L.attn_post_norm, eps);

        ggml_tensor * moe_out   = build_moe_ffn_qwen36(ctx, gf, L, post, n_tokens);
        ggml_tensor * shexp_out = build_shexp_qwen36  (ctx, L, post, n_tokens);
        ggml_tensor * ffn_out   = ggml_add(ctx, moe_out, shexp_out);
        cur = ggml_add(ctx, ffn_out, ffn_residual);

        // NB: DFlash target_feat capture indices were tuned for 27B's 64-layer
        // stack and z-lab's DFlash-trained draft. For Qwen3.6-35B we have no
        // trained draft yet (see docs/port-qwen36-35b-a3b-moe.md M5), so we
        // skip the capture path for now. The graph input's `capture_layers`
        // flag is respected for forward compatibility but writes nothing.

        inpL = cur;
    }

    ggml_tensor * out    = rms_norm_mul_(ctx, inpL, w.out_norm, q36::EPS);
    ggml_tensor * logits = ggml_mul_mat(ctx, w.output, out);
    ggml_set_name(logits, "logits");
    ggml_build_forward_expand(gf, logits);

    og.logits = logits;
    return og;
}

} // namespace dflash27b
