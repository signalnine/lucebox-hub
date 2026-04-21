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

    // 1. Router logits
    ggml_tensor * logits = ggml_mul_mat(ctx, L.ffn_gate_inp, cur);
    ggml_set_name(logits, "moe_logits");

    // 2. Softmax → probs
    ggml_tensor * probs = ggml_soft_max(ctx, logits);
    ggml_set_name(probs, "moe_probs");

    // 3. Top-k selection
    ggml_tensor * selected = ggml_argsort_top_k(ctx, probs, n_expert_used);
    ggml_set_name(selected, "moe_topk");

    // 4. Gather selected probabilities
    probs = ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens);
    ggml_tensor * weights = ggml_get_rows(ctx, probs, selected);
    ggml_set_name(weights, "moe_weights_raw");

    // 5. Normalize weights to sum to 1
    weights = ggml_reshape_2d(ctx, weights, n_expert_used, n_tokens);
    ggml_tensor * wsum = ggml_sum_rows(ctx, weights);
    wsum = ggml_clamp(ctx, wsum, 6.103515625e-5f, INFINITY);
    weights = ggml_div(ctx, weights, wsum);
    weights = ggml_reshape_3d(ctx, weights, 1, n_expert_used, n_tokens);
    ggml_set_name(weights, "moe_weights_normed");
    ggml_build_forward_expand(gf, weights);

    // 6. Route cur through top-k experts' gate and up projections
    cur = ggml_reshape_3d(ctx, cur, n_embd, 1, n_tokens);
    ggml_tensor * gate_out = ggml_mul_mat_id(ctx, L.ffn_gate_exps, cur, selected);
    ggml_tensor * up_out   = ggml_mul_mat_id(ctx, L.ffn_up_exps,   cur, selected);
    ggml_set_name(gate_out, "moe_gate");
    ggml_set_name(up_out,   "moe_up");

    gate_out = ggml_silu(ctx, gate_out);
    ggml_tensor * gu = ggml_mul(ctx, gate_out, up_out);
    ggml_set_name(gu, "moe_gate_silu_mul_up");

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
    ggml_tensor * gate = ggml_mul_mat(ctx, L.ffn_gate_shexp, cur);   // [FFN_SHEXP, n_tokens]
    gate = ggml_silu(ctx, gate);
    ggml_tensor * up   = ggml_mul_mat(ctx, L.ffn_up_shexp,   cur);   // [FFN_SHEXP, n_tokens]
    ggml_tensor * gu   = ggml_mul(ctx, gate, up);
    ggml_tensor * out  = ggml_mul_mat(ctx, L.ffn_down_shexp, gu);    // [HIDDEN, n_tokens]

    // Shared-expert sigmoid gate (one scalar per token).
    ggml_tensor * sgate = ggml_mul_mat(ctx, L.ffn_gate_inp_shexp, cur); // [1, n_tokens]
    sgate = ggml_sigmoid(ctx, sgate);
    out = ggml_mul(ctx, out, sgate);                                 // broadcast
    return out;
}

// ─── Full-attention block (same structure as qwen35 27B, q36 dims, IMROPE) ──
static ggml_tensor * build_full_attn_block(
    ggml_context * ctx,
    ggml_cgraph * gf,
    const TargetLayer & L,
    ggml_tensor * cur,              // [HIDDEN, n_tokens]
    ggml_tensor * positions,        // [n_tokens] i32
    const int * rope_sections,
    ggml_tensor * cache_k,          // [HEAD_DIM, max_ctx, N_HEAD_KV]
    ggml_tensor * cache_v,
    ggml_tensor * attn_mask,        // [kv_len, n_tokens_padded] f32 or nullptr
    int kv_start,
    int n_tokens
) {
    // Packed Q || gate: wq outputs [2 * Q_DIM, n_tokens]
    ggml_tensor * QG = ggml_mul_mat(ctx, L.wq, cur);
    QG = ggml_reshape_3d(ctx, QG, q36::HEAD_DIM * 2, q36::N_HEAD, n_tokens);

    ggml_tensor * Q = ggml_view_3d(ctx, QG,
        q36::HEAD_DIM, q36::N_HEAD, n_tokens,
        ggml_element_size(QG) * q36::HEAD_DIM * 2,
        ggml_element_size(QG) * q36::HEAD_DIM * 2 * q36::N_HEAD,
        0);
    Q = rms_norm_mul_(ctx, Q, L.q_norm, q36::EPS);

    ggml_tensor * gate = ggml_view_3d(ctx, QG,
        q36::HEAD_DIM, q36::N_HEAD, n_tokens,
        ggml_element_size(QG) * q36::HEAD_DIM * 2,
        ggml_element_size(QG) * q36::HEAD_DIM * 2 * q36::N_HEAD,
        ggml_element_size(QG) * q36::HEAD_DIM);
    gate = ggml_cont_2d(ctx, gate, q36::HEAD_DIM * q36::N_HEAD, n_tokens);

    ggml_tensor * Kcur = ggml_mul_mat(ctx, L.wk, cur);
    ggml_tensor * Vcur = ggml_mul_mat(ctx, L.wv, cur);

    Kcur = ggml_reshape_3d(ctx, Kcur, q36::HEAD_DIM, q36::N_HEAD_KV, n_tokens);
    Kcur = rms_norm_mul_(ctx, Kcur, L.k_norm, q36::EPS);
    Vcur = ggml_reshape_3d(ctx, Vcur, q36::HEAD_DIM, q36::N_HEAD_KV, n_tokens);

    // IMROPE (interleaved M-RoPE). sections all-zero for qwen35moe in the GGUF.
    int sections[4];
    for (int i = 0; i < 4; i++) sections[i] = rope_sections[i];

    Q = ggml_rope_multi(ctx, Q, positions, /*freq_factors=*/nullptr,
                        q36::ROPE_DIM, sections, GGML_ROPE_TYPE_IMROPE,
                        /*n_ctx_orig=*/0, q36::ROPE_THETA, 1.0f,
                        0.0f, 1.0f, 0.0f, 0.0f);
    Kcur = ggml_rope_multi(ctx, Kcur, positions, nullptr,
                           q36::ROPE_DIM, sections, GGML_ROPE_TYPE_IMROPE,
                           0, q36::ROPE_THETA, 1.0f,
                           0.0f, 1.0f, 0.0f, 0.0f);

    ggml_tensor * Kcur_T = ggml_permute(ctx, Kcur, 0, 2, 1, 3);
    ggml_tensor * Vcur_T = ggml_permute(ctx, Vcur, 0, 2, 1, 3);

    ggml_tensor * k_slot = ggml_view_3d(ctx, cache_k,
        q36::HEAD_DIM, n_tokens, q36::N_HEAD_KV,
        cache_k->nb[1], cache_k->nb[2],
        cache_k->nb[1] * kv_start);
    ggml_tensor * v_slot = ggml_view_3d(ctx, cache_v,
        q36::HEAD_DIM, n_tokens, q36::N_HEAD_KV,
        cache_v->nb[1], cache_v->nb[2],
        cache_v->nb[1] * kv_start);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, Kcur_T, k_slot));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, Vcur_T, v_slot));

    const int kv_len = kv_start + n_tokens;
    const int fattn_stride  = 1;
    const int kv_len_padded = ((kv_len + fattn_stride - 1) / fattn_stride) * fattn_stride;

    ggml_tensor * Qfa = ggml_permute(ctx, Q, 0, 2, 1, 3);
    Qfa = ggml_cont(ctx, Qfa);

    ggml_tensor * Kfa = ggml_view_3d(ctx, cache_k,
        q36::HEAD_DIM, kv_len_padded, q36::N_HEAD_KV,
        cache_k->nb[1], cache_k->nb[2], 0);
    ggml_tensor * Vfa = ggml_view_3d(ctx, cache_v,
        q36::HEAD_DIM, kv_len_padded, q36::N_HEAD_KV,
        cache_v->nb[1], cache_v->nb[2], 0);

    const float kq_scale = 1.0f / std::sqrt((float)q36::HEAD_DIM);
    ggml_tensor * attn = ggml_flash_attn_ext(ctx, Qfa, Kfa, Vfa, attn_mask,
                                             kq_scale, 0.0f, 0.0f);
    attn = ggml_reshape_2d(ctx, attn, q36::Q_DIM, n_tokens);

    ggml_tensor * gate_sig = ggml_sigmoid(ctx, gate);
    attn = ggml_mul(ctx, attn, gate_sig);

    attn = ggml_mul_mat(ctx, L.wo, attn);
    return attn;
}

// ─── Gated DeltaNet block (same structure as qwen35 27B, q36 dims) ────
static ggml_tensor * build_delta_net_block(
    ggml_context * ctx,
    ggml_cgraph * gf,
    const TargetLayer & L,
    ggml_tensor * cur,             // [HIDDEN, n_tokens]
    ggml_tensor * conv_state,
    ggml_tensor * ssm_state,
    int n_tokens,
    DeltaNetCapture * cap,
    ggml_tensor * parent_ids
) {
    const int num_k_heads  = q36::SSM_N_GROUP;   // 16
    const int num_v_heads  = q36::SSM_DT_RANK;   // 32
    const int head_v_dim   = q36::HEAD_V_DIM;    // 128
    const int head_k_dim   = q36::HEAD_K_DIM;    // 128
    const int n_seqs       = 1;
    const int n_seq_tokens = n_tokens;

    ggml_tensor * qkv_mixed = ggml_mul_mat(ctx, L.wqkv, cur);                  // [CONV_CHANNELS, n_tokens]
    qkv_mixed = ggml_reshape_3d(ctx, qkv_mixed, q36::CONV_CHANNELS, n_seq_tokens, n_seqs);

    ggml_tensor * z = ggml_mul_mat(ctx, L.wqkv_gate, cur);                     // [SSM_D_INNER, n_tokens]

    ggml_tensor * beta = ggml_mul_mat(ctx, L.ssm_beta, cur);
    beta = ggml_reshape_4d(ctx, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    beta = ggml_sigmoid(ctx, beta);

    ggml_tensor * alpha = ggml_mul_mat(ctx, L.ssm_alpha, cur);
    alpha = ggml_reshape_3d(ctx, alpha, num_v_heads, n_seq_tokens, n_seqs);
    alpha = ggml_add(ctx, alpha, L.ssm_dt_bias);
    alpha = ggml_softplus(ctx, alpha);
    ggml_tensor * g_tensor = ggml_mul(ctx, alpha, L.ssm_a);
    g_tensor = ggml_reshape_4d(ctx, g_tensor, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_r = ggml_reshape_3d(ctx, conv_state,
        q36::SSM_CONV_KERN - 1, q36::CONV_CHANNELS, n_seqs);

    ggml_tensor * qkv_T = ggml_transpose(ctx, qkv_mixed);
    ggml_tensor * conv_input = ggml_concat(ctx, conv_states_r, qkv_T, 0);

    if (cap && cap->conv_input) {
        ggml_build_forward_expand(gf, ggml_cpy(ctx, conv_input, cap->conv_input));
    }

    ggml_tensor * last_conv = ggml_view_3d(ctx, conv_input,
        q36::SSM_CONV_KERN - 1, q36::CONV_CHANNELS, n_seqs,
        conv_input->nb[1], conv_input->nb[2],
        (conv_input->ne[0] - (q36::SSM_CONV_KERN - 1)) * ggml_element_size(conv_input));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, last_conv, conv_state));

    ggml_tensor * conv_out = parent_ids
        ? ggml_ssm_conv_tree(ctx, conv_input, L.ssm_conv1d, parent_ids)
        : ggml_ssm_conv     (ctx, conv_input, L.ssm_conv1d);
    conv_out = ggml_silu(ctx, conv_out);

    const int64_t q_offset = 0;
    const int64_t k_offset = num_k_heads * head_k_dim;
    const int64_t v_offset = 2 * num_k_heads * head_k_dim;
    const size_t  elt      = ggml_element_size(conv_out);
    const size_t  row_size = q36::CONV_CHANNELS * elt;

    ggml_tensor * q_c = ggml_view_4d(ctx, conv_out,
        head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
        head_k_dim * elt, row_size, row_size * n_seq_tokens,
        q_offset * elt);
    ggml_tensor * k_c = ggml_view_4d(ctx, conv_out,
        head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
        head_k_dim * elt, row_size, row_size * n_seq_tokens,
        k_offset * elt);
    ggml_tensor * v_c = ggml_view_4d(ctx, conv_out,
        head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
        head_v_dim * elt, row_size, row_size * n_seq_tokens,
        v_offset * elt);

    q_c = ggml_l2_norm(ctx, q_c, q36::EPS);
    k_c = ggml_l2_norm(ctx, k_c, q36::EPS);

    if (num_k_heads != num_v_heads) {
        q_c = ggml_repeat_4d(ctx, q_c, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_c = ggml_repeat_4d(ctx, k_c, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    ggml_tensor * s = ggml_reshape_4d(ctx, ssm_state,
        head_v_dim, head_v_dim, num_v_heads, n_seqs);

    ggml_tensor * persist_inter = (parent_ids && cap && cap->ssm_intermediate_states)
        ? cap->ssm_intermediate_states : nullptr;

    ggml_tensor * result =
        persist_inter
            ? ggml_gated_delta_net_tree_persist(ctx, q_c, k_c, v_c, g_tensor, beta, s, parent_ids, persist_inter)
            : (parent_ids
                ? ggml_gated_delta_net_tree(ctx, q_c, k_c, v_c, g_tensor, beta, s, parent_ids)
                : ggml_gated_delta_net     (ctx, q_c, k_c, v_c, g_tensor, beta, s));

    ggml_tensor * output, * new_state;
    {
        const int64_t S_v = head_v_dim;
        const int64_t H_v = num_v_heads;
        const size_t r_elt = ggml_element_size(result);
        output = ggml_view_4d(ctx, result,
            S_v, H_v, n_seq_tokens, n_seqs,
            S_v * r_elt, S_v * H_v * r_elt,
            S_v * H_v * n_seq_tokens * r_elt, 0);
        new_state = ggml_view_4d(ctx, result,
            S_v, S_v, H_v, n_seqs,
            S_v * r_elt, S_v * S_v * r_elt, S_v * S_v * H_v * r_elt,
            S_v * H_v * n_seq_tokens * n_seqs * r_elt);

        ggml_build_forward_expand(gf, ggml_cpy(ctx, new_state, ssm_state));

        if (cap && cap->ssm_intermediate_states && !persist_inter) {
            const size_t inter_offset =
                  S_v * H_v * n_seq_tokens * n_seqs * r_elt
                + S_v * S_v * H_v * n_seqs * r_elt;
            ggml_tensor * inter_view = ggml_view_4d(ctx, result,
                S_v, S_v, H_v, n_seq_tokens,
                S_v * r_elt, S_v * S_v * r_elt, S_v * S_v * H_v * r_elt,
                inter_offset);
            ggml_build_forward_expand(gf,
                ggml_cpy(ctx, inter_view, cap->ssm_intermediate_states));
        }
    }

    ggml_tensor * z_4d    = ggml_reshape_4d(ctx, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor * out_n   = ggml_rms_norm(ctx, output, q36::EPS);
    out_n = ggml_mul(ctx, out_n, L.ssm_norm);
    ggml_tensor * z_silu  = ggml_silu(ctx, z_4d);
    out_n = ggml_mul(ctx, out_n, z_silu);

    ggml_tensor * flat = ggml_reshape_3d(ctx, out_n,
        head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor * out_final = ggml_mul_mat(ctx, L.ssm_out, flat);       // [HIDDEN, n_tokens]
    out_final = ggml_reshape_2d(ctx, out_final, q36::HIDDEN, n_seq_tokens * n_seqs);
    return out_final;
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
            cur = build_full_attn_block(ctx, gf, L, cur, in.positions, w.rope_sections,
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
            cur = build_delta_net_block(ctx, gf, L, cur,
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
