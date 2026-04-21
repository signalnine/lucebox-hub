// Forward pass of Qwen3.5-27B (qwen35 hybrid) in pure ggml.
//
// Translates llama.cpp's `src/models/qwen35.cpp` + `delta-net-base.cpp` into
// our standalone library, hardcoded for Qwen3.5-27B dimensions. No
// llama.cpp runtime is linked — only ggml ops.
//
// Architecture highlights:
//   - 64 layers; every 4th (il % 4 == 3) is full attention, rest are Gated DeltaNet
//   - Full-attention Q projection is PACKED with a gate (attn_q has width 2*q_dim)
//   - Full attention uses M-RoPE with sections [11,11,10,0]
//   - Flash attention is GQA 24/4, causal
//   - Delta-net uses ggml_ssm_conv for the 1D conv + ggml_gated_delta_net for the recurrence
//   - FFN is SwiGLU (w_gate * silu, element-wise multiply with w_up, then w_down)
//
// State (persisted in TargetCache across calls):
//   - attn_k[16], attn_v[16]     : KV cache for full-attn layers, f16
//   - conv_state[48]             : 1D conv recurrence state, f32
//   - ssm_state[48]              : delta-net recurrent state (head_v^2 × H_v), f32
//
// Key dimensions (all hardcoded via DFLASH27B_* macros):
//   n_embd           = 5120
//   n_head           = 24    head_dim = 256   q_dim = n_head * head_dim = 6144
//   n_head_kv        = 4     kv_dim = 4 * 256 = 1024
//   n_ff             = 17408
//   d_inner (ssm)    = 6144
//   d_state (ssm)    = 128
//   dt_rank (ssm)    = 48    (num_v_heads)
//   n_group (ssm)    = 16    (num_k_heads)
//   head_v_dim       = d_inner / dt_rank = 128
//   head_k_dim       = d_state           = 128
//   conv_kernel      = 4

#include "internal.h"
#include "delta_net_chunked.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace dflash27b {

// ─── qwen35 arch dims (runtime, from loaded GGUF) ──────────────────────
//
// The qwen35 hybrid arch spans multiple model sizes with the same op
// pattern but different dimensions:
//
//            Qwen3.5-0.8B   Qwen3.5-27B
//   n_embd          1024         5120
//   n_layer           24           64
//   n_head             8           24
//   n_head_kv          2            4
//   head_dim         256          256
//   n_ff            3584        17408
//   ssm_d_inner     2048         6144
//   ssm_dt_rank       16           48
//   ssm_n_group       16           16
//   ssm_d_state      128          128
//
// head_v_dim = ssm_d_inner / ssm_dt_rank = 128 for both.
// conv_channels = ssm_d_inner + 2 * ssm_n_group * ssm_d_state
//   (4096 for 0.8B, 10240 for 27B).
//
// Builders read these at runtime from TargetWeights w; the `q35::` names
// below are the shared constants that really ARE the same across the arch.
namespace q35 {
constexpr int SSM_CONV_KERN = 4;
constexpr float EPS         = 1e-6f;
constexpr float ROPE_THETA  = 10000000.0f;
}  // namespace q35

// Small helper: derive the 1D conv_channels from a TargetWeights.
static inline int q35_conv_channels(const TargetWeights & w) {
    return w.ssm_d_inner + 2 * w.ssm_n_group * w.ssm_d_state;
}

// ─── TargetCache allocation ─────────────────────────────────────────

bool create_target_cache(const TargetWeights & w,
                         int max_ctx,
                         int max_verify_tokens,
                         ggml_backend_t backend,
                         TargetCache & out) {
    out.backend = backend;
    out.max_ctx = max_ctx;
    out.cur_pos = 0;
    if (max_verify_tokens <= 0) {
        max_verify_tokens = DFLASH27B_DRAFT_BLOCK_SIZE;
    }

    const int n_full_attn = w.n_layer / w.full_attention_interval; // 16
    const int n_delta     = w.n_layer - n_full_attn;               // 48

    out.attn_k.assign(n_full_attn, nullptr);
    out.attn_v.assign(n_full_attn, nullptr);
    out.ssm_state.assign(n_delta, nullptr);
    out.conv_state.assign(n_delta, nullptr);
    out.ssm_state_snap.assign(n_delta, nullptr);
    out.conv_state_snap.assign(n_delta, nullptr);
    out.ssm_intermediate.assign(n_delta, nullptr);
    out.conv_input_cache.assign(n_delta, nullptr);

    // Size the cache ggml context to hold all state tensors.
    //   per full-attn layer  : 2 (K, V)
    //   per delta-net layer  : 6 (ssm, conv, ssm_snap, conv_snap,
    //                             ssm_intermediate, conv_input_cache)
    //   top-level            : 1 (target_feat)
    const int n_tensors = 2 * n_full_attn + 6 * n_delta + 1;
    ggml_init_params ip{};
    ip.mem_size   = (size_t)(n_tensors + 32) * ggml_tensor_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    out.ctx = ggml_init(ip);
    if (!out.ctx) { set_last_error("cache ggml_init failed"); return false; }

    // Create the KV cache tensors (one set per full-attn layer).
    //
    // Env overrides (checked in order; last wins):
    //   DFLASH27B_KV_F16=1  → f16 (regression baseline)
    //   DFLASH27B_KV_Q4=1   → Q4_0 (8× vs f16, required for 128K on 24 GB, ~3% AL hit)
    //
    // Default: Q8_0 — best quality/memory tradeoff at short context.
    ggml_type kv_k_type = GGML_TYPE_Q8_0;
    ggml_type kv_v_type = GGML_TYPE_Q8_0;
    if (const char * s = std::getenv("DFLASH27B_KV_F16")) {
        if (std::atoi(s) != 0) { kv_k_type = GGML_TYPE_F16; kv_v_type = GGML_TYPE_F16; }
    }
    if (const char * s = std::getenv("DFLASH27B_KV_Q4")) {
        if (std::atoi(s) != 0) { kv_k_type = GGML_TYPE_Q4_0; kv_v_type = GGML_TYPE_Q4_0; }
    }
    // Arch-agnostic dims: read from w.* (populated by load_target_gguf).
    const int cache_head_dim   = w.n_embd_head_k;                     // 256 for both qwen35 and qwen35moe
    const int cache_n_head_kv  = w.n_head_kv;                          // 4 (27B) / 2 (35B-A3B)
    const int cache_conv_ch    = w.ssm_d_inner + 2 * w.ssm_n_group * w.ssm_d_state;  // 10240 / 8192
    const int cache_conv_kern  = w.ssm_d_conv;                         // 4
    const int cache_head_v_dim = w.ssm_d_inner / w.ssm_dt_rank;        // 128
    const int cache_dt_rank    = w.ssm_dt_rank;                        // 48 / 32

    int fa_idx = 0, dn_idx = 0;
    for (int il = 0; il < w.n_layer; il++) {
        const bool is_attn = (((il + 1) % w.full_attention_interval) == 0);
        if (is_attn) {
            ggml_tensor * K = ggml_new_tensor_3d(out.ctx, kv_k_type,
                                                 cache_head_dim, max_ctx, cache_n_head_kv);
            ggml_tensor * V = ggml_new_tensor_3d(out.ctx, kv_v_type,
                                                 cache_head_dim, max_ctx, cache_n_head_kv);
            char name[64];
            std::snprintf(name, sizeof(name), "cache_k_%d", il); ggml_set_name(K, name);
            std::snprintf(name, sizeof(name), "cache_v_%d", il); ggml_set_name(V, name);
            out.attn_k[fa_idx] = K;
            out.attn_v[fa_idx] = V;
            fa_idx++;
        } else {
            ggml_tensor * S  = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F32,
                                                  cache_head_v_dim, cache_head_v_dim, cache_dt_rank);
            ggml_tensor * Sn = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F32,
                                                  cache_head_v_dim, cache_head_v_dim, cache_dt_rank);
            ggml_tensor * C  = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32,
                                                  cache_conv_kern - 1, cache_conv_ch);
            ggml_tensor * Cn = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32,
                                                  cache_conv_kern - 1, cache_conv_ch);
            ggml_tensor * Si = ggml_new_tensor_4d(out.ctx, GGML_TYPE_F16,
                                                  cache_head_v_dim, cache_head_v_dim,
                                                  cache_dt_rank, max_verify_tokens);
            ggml_tensor * Ci = ggml_new_tensor_3d(out.ctx, GGML_TYPE_F32,
                                                  (cache_conv_kern - 1) + max_verify_tokens,
                                                  cache_conv_ch, 1);
            char name[64];
            std::snprintf(name, sizeof(name), "ssm_state_%d", il);       ggml_set_name(S,  name);
            std::snprintf(name, sizeof(name), "conv_state_%d", il);      ggml_set_name(C,  name);
            std::snprintf(name, sizeof(name), "ssm_state_snap_%d", il);  ggml_set_name(Sn, name);
            std::snprintf(name, sizeof(name), "conv_state_snap_%d", il); ggml_set_name(Cn, name);
            std::snprintf(name, sizeof(name), "ssm_intermediate_%d", il); ggml_set_name(Si, name);
            std::snprintf(name, sizeof(name), "conv_input_cache_%d", il); ggml_set_name(Ci, name);
            out.ssm_state[dn_idx]       = S;
            out.conv_state[dn_idx]      = C;
            out.ssm_state_snap[dn_idx]  = Sn;
            out.conv_state_snap[dn_idx] = Cn;
            out.ssm_intermediate[dn_idx] = Si;
            out.conv_input_cache[dn_idx] = Ci;
            dn_idx++;
        }
    }

    // Rolling target_feat buffer: [5*hidden, target_feat_len] bf16.
    //
    // target_feat_len is capped (default 4096) instead of growing to max_ctx,
    // because the draft only ever reads the last DRAFT_CTX_MAX=2048 positions
    // (see test_dflash.cpp). Cap = 2 * DRAFT_CTX_MAX to leave margin for
    // prefill batching and replay. Writes use `slot = kv_start % cap`; reads
    // produce a contiguous view of the last `draft_ctx` entries by handling
    // the wrap-around on the host side.
    //
    // At max_ctx=131072 this shrinks target_feat from 6.6 GB to 0.2 GB —
    // the difference that makes long context fit.
    constexpr int TARGET_FEAT_CAP_DEFAULT = 4096;
    out.target_feat_cap = std::min(max_ctx, TARGET_FEAT_CAP_DEFAULT);
    {
        const int fc_in = DFLASH27B_DRAFT_N_TARGET_LAYERS * w.n_embd;  // 25600
        out.target_feat = ggml_new_tensor_2d(out.ctx, GGML_TYPE_BF16, fc_in, out.target_feat_cap);
        ggml_set_name(out.target_feat, "target_feat");
    }

    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) {
        set_last_error("ggml_backend_alloc_ctx_tensors failed for target cache");
        ggml_free(out.ctx);
        out.ctx = nullptr;
        return false;
    }

    // Zero-initialize all state tensors. We'll need a scratch zero buffer
    // since ggml_backend_tensor_memset isn't always available.
    // Use a big-enough zero buffer and iterate.
    std::vector<uint8_t> zeros(1 * 1024 * 1024, 0);
    for (ggml_tensor * t = ggml_get_first_tensor(out.ctx); t != nullptr;
         t = ggml_get_next_tensor(out.ctx, t)) {
        size_t nb = ggml_nbytes(t);
        size_t off = 0;
        while (off < nb) {
            size_t chunk = std::min(nb - off, zeros.size());
            ggml_backend_tensor_set(t, zeros.data(), off, chunk);
            off += chunk;
        }
    }

    return true;
}

void free_target_cache(TargetCache & c) {
    if (c.buf) { ggml_backend_buffer_free(c.buf); c.buf = nullptr; }
    if (c.ctx) { ggml_free(c.ctx); c.ctx = nullptr; }
    c.attn_k.clear();
    c.attn_v.clear();
    c.ssm_state.clear();
    c.conv_state.clear();
    c.ssm_state_snap.clear();
    c.conv_state_snap.clear();
    c.ssm_intermediate.clear();
    c.conv_input_cache.clear();
    c.target_feat = nullptr;
    c.cur_pos = 0;
}

// Snapshot/restore SSM+conv state for speculative rollback. Uses device-side
// tensor copy (ggml_backend_tensor_copy). Called outside of any compute graph.
void snapshot_ssm_state(TargetCache & c) {
    for (size_t i = 0; i < c.ssm_state.size(); i++) {
        ggml_backend_tensor_copy(c.ssm_state[i], c.ssm_state_snap[i]);
        ggml_backend_tensor_copy(c.conv_state[i], c.conv_state_snap[i]);
    }
}

void restore_ssm_state(TargetCache & c) {
    for (size_t i = 0; i < c.ssm_state.size(); i++) {
        ggml_backend_tensor_copy(c.ssm_state_snap[i], c.ssm_state[i]);
        ggml_backend_tensor_copy(c.conv_state_snap[i], c.conv_state[i]);
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────

static ggml_tensor * rms_norm_mul(ggml_context * ctx, ggml_tensor * x,
                                  ggml_tensor * weight, float eps) {
    ggml_tensor * n = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, n, weight);
}

static ggml_tensor * build_swiglu_ffn(ggml_context * ctx, ggml_tensor * cur,
                                      const TargetLayer & L) {
    // Use ggml_swiglu_split so ggml-cuda fuses the two gate/up mmvq's + silu +
    // mul into one mmvq-with-fusion launch (has_fusion=true). See ttx notes.
    ggml_tensor * gate = ggml_mul_mat(ctx, L.w_gate, cur);   // [inter, n_tokens]
    ggml_tensor * up   = ggml_mul_mat(ctx, L.w_up,   cur);   // [inter, n_tokens]
    ggml_tensor * gu   = ggml_swiglu_split(ctx, gate, up);
    return ggml_mul_mat(ctx, L.w_down, gu);                  // [hidden, n_tokens]
}

// Full-attention block (matches llama.cpp's build_layer_attn for qwen35)
//
// `cache_k` / `cache_v` are the persistent KV buffers for this layer
// (shape [head_dim, max_ctx, n_head_kv] f16). We write the new K/V for
// `n_tokens` new positions starting at `kv_start`, then run causal attention
// over [0..kv_start + n_tokens).
ggml_tensor * qwen35_build_full_attn_block(
    ggml_context * ctx,
    ggml_cgraph * gf,
    const TargetWeights & w,
    const TargetLayer & L,
    ggml_tensor * cur,              // [hidden, n_tokens]
    ggml_tensor * positions,        // [n_tokens] i32
    const int * rope_sections,
    ggml_tensor * cache_k,          // [head_dim, max_ctx, n_head_kv]
    ggml_tensor * cache_v,
    ggml_tensor * attn_mask,        // [kv_len, n_tokens] f32 or nullptr
    int kv_start,
    int n_tokens
) {
    const int head_dim  = w.n_embd_head_k;   // 256 for all qwen35 sizes
    const int n_head    = w.n_head;
    const int n_head_kv = w.n_head_kv;
    const int q_dim     = n_head * head_dim;

    // Packed Q || gate: wq outputs [2 * q_dim, n_tokens]
    ggml_tensor * QG = ggml_mul_mat(ctx, L.wq, cur);
    QG = ggml_reshape_3d(ctx, QG, head_dim * 2, n_head, n_tokens);

    ggml_tensor * Q = ggml_view_3d(ctx, QG,
        head_dim, n_head, n_tokens,
        ggml_element_size(QG) * head_dim * 2,
        ggml_element_size(QG) * head_dim * 2 * n_head,
        0);
    Q = rms_norm_mul(ctx, Q, L.q_norm, q35::EPS);

    ggml_tensor * gate = ggml_view_3d(ctx, QG,
        head_dim, n_head, n_tokens,
        ggml_element_size(QG) * head_dim * 2,
        ggml_element_size(QG) * head_dim * 2 * n_head,
        ggml_element_size(QG) * head_dim);
    gate = ggml_cont_2d(ctx, gate, head_dim * n_head, n_tokens);

    ggml_tensor * Kcur = ggml_mul_mat(ctx, L.wk, cur);
    ggml_tensor * Vcur = ggml_mul_mat(ctx, L.wv, cur);
    Kcur = ggml_reshape_3d(ctx, Kcur, head_dim, n_head_kv, n_tokens);
    Kcur = rms_norm_mul(ctx, Kcur, L.k_norm, q35::EPS);
    Vcur = ggml_reshape_3d(ctx, Vcur, head_dim, n_head_kv, n_tokens);

    // M-RoPE: rope_sections=[11,11,10,0] for 27B, all-zero for 35B MoE's
    // single-section GGUF. n_rot = rope.dimension_count = 64 for both.
    int n_rot = 64;
    int sections[4];
    for (int i = 0; i < 4; i++) sections[i] = rope_sections[i];
    // llama.cpp's llama_model_rope_type maps BOTH qwen35 and qwen35moe to
    // LLAMA_ROPE_TYPE_IMROPE (interleaved). Our earlier 27B code used
    // GGML_ROPE_TYPE_MROPE; that still produced coherent output at the
    // sample contexts we tested, but the correct type is IMROPE. Using it
    // here for all qwen35 sizes.
    const int rope_type = GGML_ROPE_TYPE_IMROPE;

    Q    = ggml_rope_multi(ctx, Q,    positions, nullptr,
                           n_rot, sections, rope_type,
                           0, q35::ROPE_THETA, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    Kcur = ggml_rope_multi(ctx, Kcur, positions, nullptr,
                           n_rot, sections, rope_type,
                           0, q35::ROPE_THETA, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    ggml_tensor * Kcur_T = ggml_permute(ctx, Kcur, 0, 2, 1, 3);
    ggml_tensor * Vcur_T = ggml_permute(ctx, Vcur, 0, 2, 1, 3);

    ggml_tensor * k_slot = ggml_view_3d(ctx, cache_k,
        head_dim, n_tokens, n_head_kv,
        cache_k->nb[1], cache_k->nb[2],
        cache_k->nb[1] * kv_start);
    ggml_tensor * v_slot = ggml_view_3d(ctx, cache_v,
        head_dim, n_tokens, n_head_kv,
        cache_v->nb[1], cache_v->nb[2],
        cache_v->nb[1] * kv_start);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, Kcur_T, k_slot));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, Vcur_T, v_slot));

    const int kv_len        = kv_start + n_tokens;
    const int fattn_stride  = 1;
    const int kv_len_padded = ((kv_len + fattn_stride - 1) / fattn_stride) * fattn_stride;

    ggml_tensor * Qfa = ggml_permute(ctx, Q, 0, 2, 1, 3);
    Qfa = ggml_cont(ctx, Qfa);

    ggml_tensor * Kfa = ggml_view_3d(ctx, cache_k,
        head_dim, kv_len_padded, n_head_kv,
        cache_k->nb[1], cache_k->nb[2], 0);
    ggml_tensor * Vfa = ggml_view_3d(ctx, cache_v,
        head_dim, kv_len_padded, n_head_kv,
        cache_v->nb[1], cache_v->nb[2], 0);

    const float kq_scale = 1.0f / std::sqrt((float)head_dim);
    ggml_tensor * attn = ggml_flash_attn_ext(ctx, Qfa, Kfa, Vfa, attn_mask,
                                             kq_scale, 0.0f, 0.0f);
    attn = ggml_reshape_2d(ctx, attn, q_dim, n_tokens);

    ggml_tensor * gate_sig = ggml_sigmoid(ctx, gate);
    attn = ggml_mul(ctx, attn, gate_sig);
    attn = ggml_mul_mat(ctx, L.wo, attn);
    return attn;
}

// Gated DeltaNet block using the fused ggml_gated_delta_net primitive.
//
// Matches the semantics of llama.cpp's build_layer_attn_linear + build_delta_net_fused.
// Updates cache->conv_state and cache->ssm_state in place.
//
// When `cap` is non-null, the function populates `cap->ssm_intermediate_states`
// with a view into the gated_delta_net result's per-step recurrent states and
// `cap->conv_input` with the concatenated conv input (old state + new tokens),
// both of which are marked as graph outputs so the caller can rollback SSM and
// conv state to any intermediate step commit_n-1 without a replay forward pass.
ggml_tensor * qwen35_build_delta_net_block(
    ggml_context * ctx,
    ggml_cgraph * gf,
    const TargetWeights & w,
    const TargetLayer & L,
    ggml_tensor * cur,            // [hidden, n_tokens]
    ggml_tensor * conv_state,     // [kernel-1, conv_channels] persistent
    ggml_tensor * ssm_state,      // [head_v_dim, head_v_dim, num_v_heads] persistent
    int n_tokens,
    DeltaNetCapture * cap,        // optional: populated on capture_delta_intermediate
    ggml_tensor * parent_ids      // optional [n_tokens] i32; tree mode when non-null
) {
    const int head_k_dim    = w.ssm_d_state;                      // 128
    const int num_k_heads   = w.ssm_n_group;                      // 16
    const int num_v_heads   = w.ssm_dt_rank;                      // 48 (27B) / 32 (35B) / 16 (0.8B)
    const int head_v_dim    = w.ssm_d_inner / w.ssm_dt_rank;      // 128
    const int conv_channels = q35_conv_channels(w);               // 10240 / 8192 / 4096
    const int conv_kern_m1  = w.ssm_d_conv - 1;                   // 3
    const int n_seqs        = 1;
    const int n_seq_tokens  = n_tokens;

    // qkv_mixed = wqkv @ cur                                   [conv_channels, n_tokens]
    ggml_tensor * qkv_mixed = ggml_mul_mat(ctx, L.wqkv, cur);
    qkv_mixed = ggml_reshape_3d(ctx, qkv_mixed, conv_channels, n_seq_tokens, n_seqs);

    // ── z = wqkv_gate @ cur            [inner, n_tokens]
    ggml_tensor * z = ggml_mul_mat(ctx, L.wqkv_gate, cur);

    // ── beta = ssm_beta @ cur          [dt_rank, n_tokens]
    ggml_tensor * beta = ggml_mul_mat(ctx, L.ssm_beta, cur);
    beta = ggml_reshape_4d(ctx, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    beta = ggml_sigmoid(ctx, beta);

    // ── alpha = ssm_alpha @ cur        [dt_rank, n_tokens]
    //    alpha = alpha + ssm_dt_bias          (per-head bias)
    //    alpha = softplus(alpha)
    //    g     = alpha * ssm_a                (-A_log.exp() * softplus)
    ggml_tensor * alpha = ggml_mul_mat(ctx, L.ssm_alpha, cur);
    alpha = ggml_reshape_3d(ctx, alpha, num_v_heads, n_seq_tokens, n_seqs);
    alpha = ggml_add(ctx, alpha, L.ssm_dt_bias);
    alpha = ggml_softplus(ctx, alpha);
    ggml_tensor * g_tensor = ggml_mul(ctx, alpha, L.ssm_a);
    g_tensor = ggml_reshape_4d(ctx, g_tensor, 1, num_v_heads, n_seq_tokens, n_seqs);

    // ── Fetch conv state [kernel-1, conv_channels] and prepend to qkv_mixed
    //    along the token axis to form the convolution input.
    ggml_tensor * conv_states_r = ggml_reshape_3d(ctx, conv_state,
        conv_kern_m1, conv_channels, n_seqs);

    // qkv_mixed currently is [conv_channels, n_tokens, n_seqs]; we need
    // [n_tokens, conv_channels, n_seqs] to concat on dim 0.
    ggml_tensor * qkv_T = ggml_transpose(ctx, qkv_mixed);

    ggml_tensor * conv_input = ggml_concat(ctx, conv_states_r, qkv_T, 0);
    // conv_input: [kernel-1 + n_tokens, conv_channels, n_seqs]

    // For spec-decode rollback: copy the full conv_input into the persistent
    // cache buffer via an in-graph ggml_cpy. This avoids marking conv_input as
    // a graph output (which would force the gallocr to preserve its memory
    // past graph_compute). After graph_compute, the cache buffer's data is
    // always valid; the rollback code slices it at commit_n.
    if (cap && cap->conv_input) {
        ggml_build_forward_expand(gf, ggml_cpy(ctx, conv_input, cap->conv_input));
    }

    // ── Save the last (kernel-1) steps back to conv_state
    ggml_tensor * last_conv = ggml_view_3d(ctx, conv_input,
        conv_kern_m1, conv_channels, n_seqs,
        conv_input->nb[1], conv_input->nb[2],
        (conv_input->ne[0] - conv_kern_m1) * ggml_element_size(conv_input));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, last_conv, conv_state));

    // ── 1D conv + silu
    //    Tree mode: use the parent-chain-aware variant so sibling nodes gather
    //    their conv window from their actual tree parent instead of the DFS
    //    predecessor. Without this, siblings get garbage logits (the conv
    //    output would mix unrelated branches).
    ggml_tensor * conv_out = parent_ids
        ? ggml_ssm_conv_tree(ctx, conv_input, L.ssm_conv1d, parent_ids)
        : ggml_ssm_conv     (ctx, conv_input, L.ssm_conv1d);
    conv_out = ggml_silu(ctx, conv_out);

    // conv_out: [conv_channels, n_tokens, n_seqs]
    const int64_t q_offset = 0;
    const int64_t k_offset = num_k_heads * head_k_dim;
    const int64_t v_offset = 2 * num_k_heads * head_k_dim;

    const size_t elt = ggml_element_size(conv_out);
    const size_t row_size = conv_channels * elt;

    ggml_tensor * q_c = ggml_view_4d(ctx, conv_out,
        head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
        head_k_dim * elt,
        row_size,
        row_size * n_seq_tokens,
        q_offset * elt);
    ggml_tensor * k_c = ggml_view_4d(ctx, conv_out,
        head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
        head_k_dim * elt,
        row_size,
        row_size * n_seq_tokens,
        k_offset * elt);
    ggml_tensor * v_c = ggml_view_4d(ctx, conv_out,
        head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
        head_v_dim * elt,
        row_size,
        row_size * n_seq_tokens,
        v_offset * elt);

    // L2 norm on Q and K
    q_c = ggml_l2_norm(ctx, q_c, q35::EPS);
    k_c = ggml_l2_norm(ctx, k_c, q35::EPS);

    // Repeat Q and K from num_k_heads to num_v_heads so they match V's layout
    // (only needed if not using the fused op's broadcast support).
    if (num_k_heads != num_v_heads) {
        q_c = ggml_repeat_4d(ctx, q_c, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_c = ggml_repeat_4d(ctx, k_c, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    // ── SSM state (recurrent): reshape to [S_v, S_v, H_v, n_seqs]
    ggml_tensor * s = ggml_reshape_4d(ctx, ssm_state,
        head_v_dim, head_v_dim, num_v_heads, n_seqs);

    // ── Fused Gated DeltaNet op — returns packed (output | new_state [| intermediates]).
    //    In tree mode, the kernel uses parent_ids to reload state at DFS
    //    branch transitions (ported from sglang's retrieve_parent_token path).
    //    When `cap->ssm_intermediate_states` is present AND we are in tree
    //    mode, use the _tree_persist variant: the kernel writes per-token
    //    intermediate states DIRECTLY into the persistent cache buffer,
    //    eliminating the downstream ggml_cpy that would otherwise copy them.
    //    Saves ~5-10 ms per verify step (memory-bandwidth bound) on 27B.
    ggml_tensor * persist_inter = (parent_ids && cap && cap->ssm_intermediate_states)
        ? cap->ssm_intermediate_states
        : nullptr;

    // Chunked delta-net path: chain-only (no parent_ids), no per-token
    // capture (no cap). Ported from llama.cpp
    // src/models/delta-net-base.cpp::build_delta_net_chunking. At n_tokens=16
    // and 48 delta-net layers it eliminates the serial per-token loop that
    // dominates target-verify compute at long ctx. Currently OFF by
    // default — port produces correct shape but slightly wrong final state,
    // causing AL degradation and loopy output. Set DFLASH27B_CHUNKED=1 to
    // opt in for A/B testing while debugging.
    bool use_chunked = false;
    if (!parent_ids && !cap && n_seq_tokens > 1) {
        if (const char * s_env = std::getenv("DFLASH27B_CHUNKED")) {
            use_chunked = (std::atoi(s_env) != 0);
        }
    }

    ggml_tensor * output = nullptr;
    ggml_tensor * new_state = nullptr;

    if (use_chunked) {
        auto r = build_delta_net_chunked(ctx, q_c, k_c, v_c, g_tensor, beta, s);
        output    = r.output;
        new_state = r.new_state;
        goto after_delta_net;
    }

    ggml_tensor * result;
    result =
        persist_inter
            ? ggml_gated_delta_net_tree_persist(ctx, q_c, k_c, v_c, g_tensor, beta, s, parent_ids, persist_inter)
            : (parent_ids
                ? ggml_gated_delta_net_tree(ctx, q_c, k_c, v_c, g_tensor, beta, s, parent_ids)
                : ggml_gated_delta_net     (ctx, q_c, k_c, v_c, g_tensor, beta, s));

    // Slice output and new_state out of the packed result
    {
    const int64_t S_v = head_v_dim;
    const int64_t H_v = num_v_heads;
    const size_t r_elt = ggml_element_size(result);
    output = ggml_view_4d(ctx, result,
        S_v, H_v, n_seq_tokens, n_seqs,
        S_v * r_elt,
        S_v * H_v * r_elt,
        S_v * H_v * n_seq_tokens * r_elt,
        0);
    new_state = ggml_view_4d(ctx, result,
        S_v, S_v, H_v, n_seqs,
        S_v * r_elt,
        S_v * S_v * r_elt,
        S_v * S_v * H_v * r_elt,
        S_v * H_v * n_seq_tokens * n_seqs * r_elt);

    // Persist new_state back to cache
    ggml_build_forward_expand(gf, ggml_cpy(ctx, new_state, ssm_state));

    // Expose per-step intermediate states for spec-decode rollback. The patched
    // ggml_gated_delta_net kernel appends an intermediate-states region to the
    // result tensor after the final-state slot. Layout in result->data:
    //   [ attn_out: S_v*H_v*n_seq_tokens*n_seqs floats
    //   | final_state: S_v*S_v*H_v*n_seqs floats
    //   | intermediate_states: S_v*S_v*H_v*n_seq_tokens*n_seqs floats ]
    //
    // Instead of marking the whole `result` tensor as a graph output (which
    // forces gallocr to preserve ~50 MB per layer × 48 layers of otherwise
    // transient memory and inflates graph_build by ~35 ms), we create a VIEW
    // into the intermediate region and ggml_cpy it into the persistent cache
    // buffer cap->ssm_intermediate_states. The gallocr is unaware of the
    // persistent cache, so verify_build stays cheap. Matches SGLang's
    // mamba_caches.intermediate_ssm pattern.
    if (cap && cap->ssm_intermediate_states && !persist_inter) {
        // Legacy cpy path: only used when the kernel wrote intermediates into
        // its own result region (i.e. when we did NOT use _tree_persist).
        // The _tree_persist variant writes directly to the cache buffer and
        // this cpy becomes redundant, saving ~5-10 ms per verify step.
        const size_t inter_offset =
            S_v * H_v * n_seq_tokens * n_seqs * r_elt        // attn output region
          + S_v * S_v * H_v * n_seqs * r_elt;                // final-state region
        ggml_tensor * inter_view = ggml_view_4d(ctx, result,
            S_v, S_v, H_v, n_seq_tokens,
            S_v * r_elt,
            S_v * S_v * r_elt,
            S_v * S_v * H_v * r_elt,
            inter_offset);
        ggml_build_forward_expand(gf,
            ggml_cpy(ctx, inter_view, cap->ssm_intermediate_states));
    }
    } // end of block started at `{` before `const int64_t S_v = head_v_dim;`

after_delta_net:
    // Chunked path writes directly into the same ssm_state slot via its 4D
    // view `s` (which is a live view over ssm_state), using the same cpy
    // pattern the sequential path uses for `new_state`. Sequential path's
    // cpy was already emitted above; guard this second cpy on use_chunked
    // so we don't double-write.
    if (use_chunked) {
        ggml_build_forward_expand(gf, ggml_cpy(ctx, new_state, s));
    }

    // ── Gated output norm: rms_norm(output) * silu(z_4d)
    ggml_tensor * z_4d = ggml_reshape_4d(ctx, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);
    ggml_tensor * output_n = ggml_rms_norm(ctx, output, q35::EPS);
    output_n = ggml_mul(ctx, output_n, L.ssm_norm);
    ggml_tensor * z_silu  = ggml_silu(ctx, z_4d);
    output_n = ggml_mul(ctx, output_n, z_silu);

    // Reshape to [d_inner, n_tokens]
    ggml_tensor * flat = ggml_reshape_3d(ctx, output_n,
        head_v_dim * num_v_heads, n_seq_tokens, n_seqs);

    // Output projection
    ggml_tensor * out = ggml_mul_mat(ctx, L.ssm_out, flat);
    out = ggml_reshape_2d(ctx, out, w.n_embd, n_seq_tokens * n_seqs);
    return out;
}

// ─── Main graph builder ─────────────────────────────────────────────

QwenGraphOutputs build_qwen35_graph(
    ggml_context *         ctx,
    ggml_cgraph *          gf,
    const TargetWeights &  w,
    TargetCache &          cache,
    const QwenGraphInputs & in) {

    const int n_tokens = in.n_tokens;

    // 1. Caller supplies pre-embedded inputs via in.inp_embed (CPU lookup done
    //    ahead of time, zero GPU cost for the embedding table).
    ggml_tensor * inpL = in.inp_embed;

    int fa_idx = 0, dn_idx = 0;

    // If the caller requested capture, size the output list to the total delta-
    // net layer count so we can index by dn_idx as we iterate the layers.
    QwenGraphOutputs og_early{};
    if (in.capture_delta_intermediate) {
        const int n_full_attn = w.n_layer / w.full_attention_interval;
        const int n_delta     = w.n_layer - n_full_attn;
        og_early.delta_captures.resize(n_delta);
    }

    // DFlash target layer IDs for feature capture: {1, 16, 31, 46, 61}
    // HF hidden_states[lid+1] convention — capture AFTER layer 'lid' runs.
    static const int CAPTURE_LAYERS[DFLASH27B_DRAFT_N_TARGET_LAYERS] =
        { 1, 16, 31, 46, 61 };

    const int hidden = w.n_embd;
    const float eps  = q35::EPS;

    for (int il = 0; il < w.n_layer; il++) {
        const TargetLayer & L = w.layers[il];
        const bool is_attn = (((il + 1) % w.full_attention_interval) == 0);

        ggml_tensor * inpSA = inpL;

        // Pre-attention norm
        ggml_tensor * cur = rms_norm_mul(ctx, inpL, L.attn_norm, eps);

        if (is_attn) {
            cur = qwen35_build_full_attn_block(ctx, gf, w, L, cur, in.positions, w.rope_sections,
                                        cache.attn_k[fa_idx], cache.attn_v[fa_idx],
                                        in.attn_mask, in.kv_start, n_tokens);
            fa_idx++;
        } else {
            DeltaNetCapture * cap_ptr = nullptr;
            if (in.capture_delta_intermediate) {
                cap_ptr = &og_early.delta_captures[dn_idx];
                cap_ptr->ssm_intermediate_states = cache.ssm_intermediate[dn_idx];
                cap_ptr->conv_input              = cache.conv_input_cache[dn_idx];
            }
            cur = qwen35_build_delta_net_block(ctx, gf, w, L, cur,
                                        cache.conv_state[dn_idx], cache.ssm_state[dn_idx],
                                        n_tokens, cap_ptr, in.parent_ids);
            dn_idx++;
        }

        // Residual
        cur = ggml_add(ctx, cur, inpSA);

        // Post-attention norm (before FFN)
        ggml_tensor * ffn_residual = cur;
        ggml_tensor * post = rms_norm_mul(ctx, cur, L.attn_post_norm, eps);

        // SwiGLU FFN
        ggml_tensor * ffn = build_swiglu_ffn(ctx, post, L);
        cur = ggml_add(ctx, ffn, ffn_residual);

        // ── DFlash layer feature capture ──
        // Write `cur` into the rolling target_feat buffer. The buffer is a
        // ring of `target_feat_cap` slots; position P maps to slot P%cap.
        // Within a single build call we may straddle the wrap boundary, so
        // we split the copy into up to two contiguous ggml_cpy ops.
        if (in.capture_layers && cache.target_feat) {
            int capture_idx = -1;
            for (int k = 0; k < DFLASH27B_DRAFT_N_TARGET_LAYERS; k++) {
                if (CAPTURE_LAYERS[k] == il) { capture_idx = k; break; }
            }
            if (capture_idx >= 0) {
                const size_t elt        = ggml_element_size(cache.target_feat);
                const size_t col_stride = cache.target_feat->nb[1];
                const int    cap        = cache.target_feat_cap;
                const int    slot_start = in.kv_start % cap;
                const int    pre_n      = std::min(n_tokens, cap - slot_start);
                const int    post_n    = n_tokens - pre_n;

                ggml_tensor * cur_2d = ggml_reshape_2d(ctx, cur, hidden, n_tokens);

                // First slice: [slot_start..slot_start+pre_n) in the ring.
                {
                    const size_t offset =
                        (size_t)slot_start * col_stride +
                        (size_t)capture_idx * hidden * elt;
                    ggml_tensor * slot = ggml_view_2d(ctx, cache.target_feat,
                        hidden, pre_n, col_stride, offset);
                    ggml_tensor * src  = ggml_view_2d(ctx, cur_2d,
                        hidden, pre_n, cur_2d->nb[1], 0);
                    ggml_build_forward_expand(gf, ggml_cpy(ctx, src, slot));
                }

                // Second slice: wrap-around at [0..post_n) if needed.
                if (post_n > 0) {
                    const size_t offset =
                        (size_t)capture_idx * hidden * elt;
                    ggml_tensor * slot = ggml_view_2d(ctx, cache.target_feat,
                        hidden, post_n, col_stride, offset);
                    ggml_tensor * src  = ggml_view_2d(ctx, cur_2d,
                        hidden, post_n, cur_2d->nb[1],
                        (size_t)pre_n * cur_2d->nb[1]);
                    ggml_build_forward_expand(gf, ggml_cpy(ctx, src, slot));
                }
            }
        }

        inpL = cur;
    }

    // 2. Final norm
    ggml_tensor * out = rms_norm_mul(ctx, inpL, w.out_norm, q35::EPS);

    // 3. LM head
    ggml_tensor * logits = ggml_mul_mat(ctx, w.output, out);
    ggml_set_name(logits, "logits");

    ggml_build_forward_expand(gf, logits);

    QwenGraphOutputs og = std::move(og_early);
    og.logits = logits;
    return og;
}

} // namespace dflash27b
