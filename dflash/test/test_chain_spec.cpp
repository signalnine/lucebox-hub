// Chain-speculative decode: small Qwen3.5 draft proposes N tokens, large
// target verifies, commit matching prefix + 1 bonus, roll back on mismatch.
//
// Simplest correct implementation: both models use single-token step graphs.
// Target verify is N sequential single-token forwards that short-circuit at
// the first mismatch, so target cache ends at the correct commit point
// naturally. Draft runs all N steps; on mismatch we restore draft's SSM state
// to a pre-round snapshot and re-run commit_count "catch-up" steps.
//
// This gives up the batched-attention win that a true multi-token target
// verify would provide. It's a correctness checkpoint first; M3 adds tree
// verify and M4 adds batched verify for real throughput wins.
//
// Usage:
//   test_chain_spec <target.gguf> <draft.gguf> <prompt.bin> <n_gen> <N_spec> <out.bin>

#include "dflash27b.h"
#include "internal.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cuda_runtime.h>

extern "C" void dflash27b_launch_f16_to_f32(const void * src,
                                            void * dst,
                                            size_t n_elems,
                                            cudaStream_t stream);

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <queue>
#include <unordered_map>
#include <vector>

using namespace dflash27b;

static std::vector<int32_t> read_int32_file(const char * path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::vector<int32_t> v;
    int32_t x;
    while (f.read((char*)&x, sizeof(x))) v.push_back(x);
    return v;
}
static bool write_int32_file(const char * path, const std::vector<int32_t> & v) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write((const char*)v.data(), sizeof(int32_t) * v.size());
    return (bool)f;
}

struct StepGraph {
    ggml_context *    ctx   = nullptr;
    ggml_cgraph *     gf    = nullptr;
    ggml_gallocr_t    alloc = nullptr;
    ggml_tensor *     inp_embed   = nullptr;
    ggml_tensor *     positions   = nullptr;
    ggml_tensor *     attn_mask   = nullptr;
    ggml_tensor *     parent_ids  = nullptr;   // tree-mode only
    ggml_tensor *     kv_pos_idx  = nullptr;   // i64 [n_tokens] for KV set_rows
    ggml_tensor *     logits      = nullptr;
    ggml_tensor *     argmax      = nullptr;   // i32 [n_tokens] — GPU-side argmax
    int               n_tokens    = 0;
    int               n_kv_padded = 0;
    bool              is_tree     = false;     // true = built by build_step_graph_tree
    // Tree-mode only: one entry per delta-net layer. Populated when
    // capture_delta_intermediate=true. Used by DDTree fast rollback.
    std::vector<dflash27b::DeltaNetCapture> delta_captures;
};
// Reset per-call graph state but KEEP the persistent sg.alloc (CUDA buffer)
// AND sg.ctx across calls. ggml_reset rewinds the ctx's arena bump pointer
// without freeing the underlying mem_buffer — so the next ggml_new_tensor_*
// calls re-use the same addresses, keeping cgraph->nodes[0] pointer-stable.
// That's what ggml-cuda's CUDA-graph cache keys on, and it's what turns
// per-step kernel launches into a single captured graph launch under
// GGML_CUDA_GRAPHS=ON. Freeing+reallocating sg.alloc also costs ~ms per
// step at multi-GB working set, which is significant here.
static void step_graph_free(StepGraph & sg) {
    if (sg.ctx) ggml_reset(sg.ctx);
    sg.gf = nullptr;
    sg.inp_embed   = nullptr;
    sg.positions   = nullptr;
    sg.attn_mask   = nullptr;
    sg.parent_ids  = nullptr;
    sg.kv_pos_idx  = nullptr;
    sg.logits      = nullptr;
    sg.argmax      = nullptr;
    sg.n_tokens    = 0;
    sg.n_kv_padded = 0;
    sg.is_tree     = false;
    sg.delta_captures.clear();
}
// Called at shutdown — fully tear down.
static void step_graph_destroy(StepGraph & sg) {
    if (sg.alloc) { ggml_gallocr_free(sg.alloc); sg.alloc = nullptr; }
    if (sg.ctx)   { ggml_free(sg.ctx); sg.ctx = nullptr; }
    step_graph_free(sg);
}

// Flash-attn mask alignment (test_dflash convention: 32 on both axes).
// Kv-axis padding is supposed to be tied to fattn_stride in
// build_full_attn_block, which still uses stride=1; this is part of the
// unresolved M2b batched-verify bug — see handoff notes in the port plan.
static constexpr int KQ_MASK_PAD     = 32;
static constexpr int KQ_STRIDE_PAD   = 32;
static constexpr uint16_t F16_ZERO    = 0x0000;
static constexpr uint16_t F16_NEG_INF = 0xFC00;

static int align_up(int x, int a) { return ((x + a - 1) / a) * a; }

// Build a causal mask padded to `kv_pad_override` on the K axis (caller
// usually passes sg.n_kv_padded so the view shape stays stable for CUDA
// graph reuse). Pad q rows duplicate the last real row's mask pattern so
// fattn never sees an all-(-inf) softmax row.
static void build_causal_mask_f16(
    std::vector<uint16_t> & out, int kv_len, int n_tokens, int kv_start,
    int kv_pad_override = -1)
{
    const int kv_pad = (kv_pad_override > 0)
        ? kv_pad_override
        : align_up(kv_len, KQ_STRIDE_PAD);
    const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);
    out.assign((size_t)kv_pad * q_pad, F16_NEG_INF);
    for (int q = 0; q < n_tokens; q++) {
        const int max_k = kv_start + q;
        for (int k = 0; k <= max_k && k < kv_pad; k++) {
            out[(size_t)q * kv_pad + k] = F16_ZERO;
        }
    }
    // Pad q rows: copy last real row's pattern.
    if (n_tokens < q_pad && n_tokens > 0) {
        for (int q = n_tokens; q < q_pad; q++) {
            std::memcpy(&out[(size_t)q * kv_pad],
                        &out[(size_t)(n_tokens - 1) * kv_pad],
                        sizeof(uint16_t) * kv_pad);
        }
    }
}

// ─── DDTree support (ported from test_dflash.cpp / liranringel/ddtree) ────

// Per-position top-K softmax extraction. Computes log-probabilities via a
// single pass over the vocab that also maintains top-K in a min-heap and
// computes logsumexp online.
static void extract_draft_topk(const float * logits,
                               int n_positions, int vocab, int K,
                               float * out_log_probs,
                               int32_t * out_token_ids,
                               float temperature = 1.0f) {
    struct Entry { float logit; int32_t id; };
    auto cmp_greater = [](const Entry & a, const Entry & b) { return a.logit > b.logit; };
    const float inv_t = 1.0f / std::max(1e-3f, temperature);

    for (int i = 0; i < n_positions; i++) {
        const float * li = logits + (size_t)i * vocab;
        std::vector<Entry> heap;
        heap.reserve(K);
        float running_max     = -INFINITY;
        float running_sum_exp = 0.0f;
        for (int j = 0; j < vocab; j++) {
            const float l = li[j] * inv_t;
            if (l > running_max) {
                if (running_max > -INFINITY) {
                    running_sum_exp = running_sum_exp * std::exp(running_max - l);
                }
                running_sum_exp += 1.0f;
                running_max = l;
            } else {
                running_sum_exp += std::exp(l - running_max);
            }
            if ((int)heap.size() < K) {
                heap.push_back({l, (int32_t)j});
                std::push_heap(heap.begin(), heap.end(), cmp_greater);
            } else if (l > heap.front().logit) {
                std::pop_heap(heap.begin(), heap.end(), cmp_greater);
                heap.back() = {l, (int32_t)j};
                std::push_heap(heap.begin(), heap.end(), cmp_greater);
            }
        }
        const float log_z = running_max + std::log(running_sum_exp);
        std::sort_heap(heap.begin(), heap.end(), cmp_greater);
        for (int k = 0; k < K; k++) {
            out_log_probs[(size_t)i * K + k] = heap[k].logit - log_z;
            out_token_ids[(size_t)i * K + k] = heap[k].id;
        }
    }
}

// Flat DFS-ordered tree.
struct DDTree {
    int                   n_nodes = 0;
    std::vector<int32_t>  token_ids;        // size n_nodes
    std::vector<int>      depths;           // size n_nodes (1..L)
    std::vector<int>      parents;          // size n_nodes + 1; parents[0]=-1
    std::vector<std::unordered_map<int32_t, int>> child_maps;  // size n_nodes + 1
    std::vector<uint8_t>  visibility;       // (1 + n_nodes)^2 row-major
};

// Best-first build of a DDTree from per-position top-K log-probs.
static DDTree build_ddtree(const float * top_log_probs,
                           const int32_t * top_token_ids,
                           int L, int K, int budget,
                           bool chain_seed = true) {
    DDTree tree;
    tree.parents.push_back(-1);
    tree.child_maps.emplace_back();
    if (budget <= 0 || L <= 0) {
        tree.visibility.assign(1, 1);
        return tree;
    }

    struct HeapEntry { float neg_logw; int parent_index; int depth; int rank; float logw; };
    struct HeapCmp   { bool operator()(const HeapEntry & a, const HeapEntry & b) const { return a.neg_logw > b.neg_logw; } };
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapCmp> heap;

    if (chain_seed) {
        const int chain_depth = std::min(L, budget);
        float cum_logw = 0.0f;
        int   prev_idx = 0;
        for (int d = 1; d <= chain_depth; d++) {
            const int32_t tok_id = top_token_ids[(size_t)(d - 1) * K + 0];
            cum_logw += top_log_probs[(size_t)(d - 1) * K + 0];
            const int cur_idx = tree.n_nodes + 1;
            tree.token_ids.push_back(tok_id);
            tree.depths.push_back(d);
            tree.parents.push_back(prev_idx);
            tree.child_maps.emplace_back();
            tree.child_maps[prev_idx][tok_id] = cur_idx;
            tree.n_nodes++;
            if (K > 1) {
                const float sibling_logw = cum_logw
                    - top_log_probs[(size_t)(d - 1) * K + 0]
                    + top_log_probs[(size_t)(d - 1) * K + 1];
                heap.push({ -sibling_logw, prev_idx, d, 1, sibling_logw });
            }
            prev_idx = cur_idx;
        }
    } else {
        const float root_logw = top_log_probs[0];
        heap.push({ -root_logw, 0, 1, 0, root_logw });
    }

    while (!heap.empty() && tree.n_nodes < budget) {
        HeapEntry top = heap.top(); heap.pop();
        const int depth_minus_1 = top.depth - 1;
        const int32_t token_id  = top_token_ids[(size_t)depth_minus_1 * K + top.rank];
        const int current_index = tree.n_nodes + 1;
        tree.token_ids.push_back(token_id);
        tree.depths.push_back(top.depth);
        tree.parents.push_back(top.parent_index);
        tree.child_maps.emplace_back();
        tree.child_maps[top.parent_index][token_id] = current_index;
        tree.n_nodes++;

        if (top.rank + 1 < K) {
            const float sibling_logw = top.logw
                - top_log_probs[(size_t)depth_minus_1 * K + top.rank]
                + top_log_probs[(size_t)depth_minus_1 * K + top.rank + 1];
            heap.push({ -sibling_logw, top.parent_index, top.depth, top.rank + 1, sibling_logw });
        }
        if (top.depth < L) {
            const float child_logw = top.logw
                + top_log_probs[(size_t)top.depth * K + 0];
            heap.push({ -child_logw, current_index, top.depth + 1, 0, child_logw });
        }
    }

    const int N = 1 + tree.n_nodes;
    tree.visibility.assign((size_t)N * N, 0);
    tree.visibility[0 * N + 0] = 1;
    for (int i = 1; i < N; i++) {
        const int p = tree.parents[i];
        for (int j = 0; j < i; j++) {
            tree.visibility[(size_t)i * N + j] = tree.visibility[(size_t)p * N + j];
        }
        tree.visibility[(size_t)i * N + i] = 1;
    }
    return tree;
}

// Walk verified tree following target's argmax at each node.
static std::vector<int> follow_verified_tree(const DDTree & tree,
                                             const int32_t * posterior,
                                             int & out_next_token) {
    std::vector<int> accepted;
    accepted.reserve(tree.n_nodes + 1);
    accepted.push_back(0);
    int current_index = 0;
    int next_token    = posterior[0];
    while (true) {
        const auto & children = tree.child_maps[current_index];
        auto it = children.find(next_token);
        if (it == children.end()) break;
        current_index = it->second;
        accepted.push_back(current_index);
        next_token = posterior[current_index];
    }
    out_next_token = next_token;
    return accepted;
}

// Build f16 ancestor-only attention mask.
static void build_tree_mask(const DDTree & tree, int past_length,
                            std::vector<uint16_t> & out_mask,
                            int kv_pad_override = -1) {
    const int N      = 1 + tree.n_nodes;
    const int kv_len = past_length + N;
    const int kv_pad = (kv_pad_override > 0)
        ? kv_pad_override
        : align_up(kv_len, KQ_STRIDE_PAD);
    const int q_pad  = align_up(N, KQ_MASK_PAD);
    out_mask.assign((size_t)kv_pad * q_pad, F16_NEG_INF);
    for (int q = 0; q < N; q++) {
        for (int k = 0; k < past_length && k < kv_pad; k++) {
            out_mask[(size_t)q * kv_pad + k] = F16_ZERO;
        }
        for (int j = 0; j < N; j++) {
            if (tree.visibility[(size_t)q * N + j] && past_length + j < kv_pad) {
                out_mask[(size_t)q * kv_pad + (past_length + j)] = F16_ZERO;
            }
        }
    }
    if (N < q_pad && N > 0) {
        for (int q = N; q < q_pad; q++) {
            std::memcpy(&out_mask[(size_t)q * kv_pad],
                        &out_mask[(size_t)(N - 1) * kv_pad],
                        sizeof(uint16_t) * kv_pad);
        }
    }
}

// Build a forward graph for n_tokens >= 1. When n_tokens == 1, no attn_mask
// is needed (a single query attending to all keys is trivially causal); for
// n_tokens > 1 we allocate an f16 mask tensor that the caller fills pre-compute.
static bool build_step_graph(
    StepGraph & sg,
    const TargetWeights & w,
    TargetCache & cache,
    ggml_backend_t backend,
    int kv_start,
    int n_tokens)
{
    // Same-shape reuse: see test_generate for rationale. If the graph is
    // already built at this n_tokens + n_kv_padded shape, return early —
    // ggml_reset + tensor-rebuild + gallocr walk costs ~280 µs/step.
    constexpr int KV_PAD_ALIGN = 256;
    const int kv_len       = kv_start + n_tokens;
    const int want_n_kv_pd = ((kv_len + KV_PAD_ALIGN - 1) / KV_PAD_ALIGN) * KV_PAD_ALIGN;
    if (sg.ctx && sg.gf && !sg.is_tree
        && sg.n_tokens == n_tokens && sg.n_kv_padded == want_n_kv_pd) {
        return true;
    }

    step_graph_free(sg);
    if (!sg.ctx) {
        ggml_init_params ip{};
        ip.mem_size = 512 * 1024 * 1024; ip.no_alloc = true;
        sg.ctx = ggml_init(ip);
        if (!sg.ctx) return false;
    }

    const int hidden = w.n_embd;
    sg.inp_embed = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, hidden, n_tokens, 1);
    sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, 4 * n_tokens);
    ggml_set_input(sg.inp_embed);
    ggml_set_input(sg.positions);

    // KV write indices + always-on f16 mask (see test_generate for the
    // rationale — 256-aligned n_kv_padded keeps the read view shape stable
    // across 256 steps, which is what ggml-cuda needs to reuse the captured
    // CUDA graph). q_pad matches KQ_MASK_PAD=32 for the fattn padded-q
    // convention.
    const int n_kv_padded = want_n_kv_pd;
    const int q_pad      = align_up(n_tokens, KQ_MASK_PAD);
    sg.attn_mask  = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F16, n_kv_padded, q_pad);
    ggml_set_input(sg.attn_mask);
    sg.kv_pos_idx = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I64, n_tokens);
    ggml_set_input(sg.kv_pos_idx);
    sg.n_kv_padded = n_kv_padded;

    sg.gf = ggml_new_graph_custom(sg.ctx, 16384, false);

    QwenGraphInputs gi{};
    gi.inp_embed    = sg.inp_embed;
    gi.positions    = sg.positions;
    gi.attn_mask    = sg.attn_mask;
    gi.kv_pos_idx   = sg.kv_pos_idx;
    gi.n_kv_padded  = n_kv_padded;
    gi.n_tokens     = n_tokens;
    gi.kv_start     = kv_start;
    gi.capture_layers = false;

    QwenGraphOutputs go = build_target_graph(sg.ctx, sg.gf, w, cache, gi);
    if (!go.logits) return false;
    ggml_set_output(go.logits);
    ggml_build_forward_expand(sg.gf, go.logits);
    sg.logits   = go.logits;
    sg.n_tokens = n_tokens;
    sg.is_tree  = false;

    // GPU-side argmax output. Saves ~200 µs/step on H2D copy + CPU argmax
    // for the 248K-vocab logit head. Callers that need full logits (e.g.
    // DDTree top-K extraction from the DRAFT) still read sg.logits; callers
    // that only need argmax read sg.argmax.
    sg.argmax = ggml_argmax(sg.ctx, go.logits);
    ggml_set_output(sg.argmax);
    ggml_build_forward_expand(sg.gf, sg.argmax);

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}

// Tree-mode graph: like the batched graph but with a parent_ids input wired
// to the DeltaNet kernel AND capture_delta_intermediate=true so SSM rollback
// doesn't need replay. Matches test_dflash::build_target_step_tree structure.
static bool build_step_graph_tree(
    StepGraph & sg,
    const TargetWeights & w,
    TargetCache & cache,
    ggml_backend_t backend,
    int kv_start,
    int n_tokens)
{
    constexpr int KV_PAD_ALIGN = 256;
    const int kv_len       = kv_start + n_tokens;
    const int want_n_kv_pd = ((kv_len + KV_PAD_ALIGN - 1) / KV_PAD_ALIGN) * KV_PAD_ALIGN;
    if (sg.ctx && sg.gf && sg.is_tree
        && sg.n_tokens == n_tokens && sg.n_kv_padded == want_n_kv_pd) {
        return true;
    }

    step_graph_free(sg);
    if (!sg.ctx) {
        ggml_init_params ip{};
        ip.mem_size = 512 * 1024 * 1024; ip.no_alloc = true;
        sg.ctx = ggml_init(ip);
        if (!sg.ctx) return false;
    }

    const int hidden = w.n_embd;
    sg.inp_embed = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, hidden, n_tokens, 1);
    sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, 4 * n_tokens);
    ggml_set_input(sg.inp_embed);
    ggml_set_input(sg.positions);

    const int n_kv_padded = want_n_kv_pd;
    const int q_pad       = align_up(n_tokens, KQ_MASK_PAD);
    sg.attn_mask = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F16, n_kv_padded, q_pad);
    ggml_set_input(sg.attn_mask);
    sg.kv_pos_idx = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I64, n_tokens);
    ggml_set_input(sg.kv_pos_idx);
    sg.n_kv_padded = n_kv_padded;

    sg.parent_ids = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(sg.parent_ids);

    sg.gf = ggml_new_graph_custom(sg.ctx, 16384, false);

    QwenGraphInputs gi{};
    gi.inp_embed                  = sg.inp_embed;
    gi.positions                  = sg.positions;
    gi.attn_mask                  = sg.attn_mask;
    gi.kv_pos_idx                 = sg.kv_pos_idx;
    gi.n_kv_padded                = n_kv_padded;
    gi.n_tokens                   = n_tokens;
    gi.kv_start                   = kv_start;
    gi.capture_layers             = false;
    gi.capture_delta_intermediate = true;
    gi.parent_ids                 = sg.parent_ids;

    QwenGraphOutputs go = build_target_graph(sg.ctx, sg.gf, w, cache, gi);
    if (!go.logits) return false;
    ggml_set_output(go.logits);
    ggml_build_forward_expand(sg.gf, go.logits);
    sg.logits         = go.logits;
    sg.delta_captures = std::move(go.delta_captures);
    sg.n_tokens       = n_tokens;
    sg.is_tree        = true;

    // GPU argmax per slot. sg.argmax has shape [n_tokens] i32.
    sg.argmax = ggml_argmax(sg.ctx, go.logits);
    ggml_set_output(sg.argmax);
    ggml_build_forward_expand(sg.gf, sg.argmax);

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}

// Single-token forward: consume `tok` at `pos`, return argmax (prediction
// for `pos+1`). Caller updates cache.cur_pos.
static int32_t step_model(
    const TargetWeights & w,
    TargetCache & cache,
    ggml_backend_t backend,
    StepGraph & sg,
    int32_t tok, int pos,
    std::vector<float> & embed_buf,
    std::vector<float> & logits_buf,
    bool want_logits = false)  // true only when caller needs full logits
{
    if (!build_step_graph(sg, w, cache, backend, pos, 1)) {
        std::fprintf(stderr, "build_step_graph(1) failed at pos=%d\n", pos);
        std::exit(1);
    }
    int32_t ids[1] = { tok };
    const int hidden = w.n_embd;
    if ((int)embed_buf.size() < hidden) embed_buf.assign(hidden, 0.f);
    if (!w.embedder.embed(ids, 1, embed_buf.data())) {
        std::fprintf(stderr, "embed failed tok=%d\n", tok); std::exit(1);
    }
    // Write only the first hidden floats (embed_buf may be larger — shared
    // with verify_batch which uses hidden*N_spec).
    ggml_backend_tensor_set(sg.inp_embed, embed_buf.data(), 0, sizeof(float) * hidden);
    int32_t p4[4] = { pos, pos, pos, pos };
    ggml_backend_tensor_set(sg.positions, p4, 0, sizeof(int32_t) * 4);
    // KV write row index + full causal mask (see build_step_graph).
    int64_t kv_idx[1] = { (int64_t)pos };
    ggml_backend_tensor_set(sg.kv_pos_idx, kv_idx, 0, sizeof(int64_t));
    {
        const int kv_len = pos + 1;
        const int q_pad  = align_up(1, KQ_MASK_PAD);
        std::vector<uint16_t> mask((size_t)sg.n_kv_padded * q_pad, F16_NEG_INF);
        for (int q = 0; q < q_pad; q++) {
            for (int k = 0; k < kv_len && k < sg.n_kv_padded; k++) {
                mask[q * sg.n_kv_padded + k] = F16_ZERO;
            }
        }
        ggml_backend_tensor_set(sg.attn_mask, mask.data(), 0, sizeof(uint16_t) * mask.size());
    }
    if (ggml_backend_graph_compute(backend, sg.gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "compute failed at pos=%d\n", pos); std::exit(1);
    }
    const int vocab = (int)w.embedder.n_vocab;
    // Fast path: read GPU-computed argmax (4 bytes) instead of the 1 MiB
    // logits vector. DDTree draft top-K extraction needs full logits, so
    // those callers pass want_logits=true.
    if (!want_logits) {
        int32_t best_i32 = 0;
        ggml_backend_tensor_get(sg.argmax, &best_i32, 0, sizeof(int32_t));
        return (int)best_i32;
    }
    if ((int)logits_buf.size() < vocab) logits_buf.assign(vocab, 0.f);
    ggml_backend_tensor_get(sg.logits, logits_buf.data(), 0, sizeof(float) * vocab);
    int best = 0; float bv = logits_buf[0];
    for (int i = 1; i < vocab; i++)
        if (logits_buf[i] > bv) { bv = logits_buf[i]; best = i; }
    return best;
}

// Batched forward: consume `toks[0..n_tokens-1]` at positions
// [pos, pos+1, ..., pos+n_tokens-1] in ONE graph compute. Fills `argmax_out`
// with n_tokens argmax predictions, one per input position. Caller updates
// cache.cur_pos by n_tokens.
static void verify_batch(
    const TargetWeights & w,
    TargetCache & cache,
    ggml_backend_t backend,
    StepGraph & sg,
    const int32_t * toks, int n_tokens, int pos,
    std::vector<float> & embed_buf,
    std::vector<float> & logits_buf,
    std::vector<uint16_t> & mask_buf,
    int32_t * argmax_out)
{
    if (!build_step_graph(sg, w, cache, backend, pos, n_tokens)) {
        std::fprintf(stderr, "build_step_graph(%d) failed at pos=%d\n", n_tokens, pos);
        std::exit(1);
    }
    // Embeddings for all n_tokens
    const int hidden = w.n_embd;
    if ((int)embed_buf.size() < hidden * n_tokens) embed_buf.assign(hidden * n_tokens, 0.f);
    if (!w.embedder.embed(toks, n_tokens, embed_buf.data())) {
        std::fprintf(stderr, "batch embed failed\n"); std::exit(1);
    }
    ggml_backend_tensor_set(sg.inp_embed, embed_buf.data(), 0,
                            sizeof(float) * hidden * n_tokens);

    // M-RoPE positions: layout is [axis, token] with axis 0 fastest in
    // token index. axes 0-2 = real position, axis 3 = 0 (matches
    // rope_sections=[11,11,10,0] convention used by test_dflash prefill).
    std::vector<int32_t> positions(4 * n_tokens);
    for (int i = 0; i < n_tokens; i++) {
        positions[0 * n_tokens + i] = pos + i;
        positions[1 * n_tokens + i] = pos + i;
        positions[2 * n_tokens + i] = pos + i;
        positions[3 * n_tokens + i] = 0;
    }
    ggml_backend_tensor_set(sg.positions, positions.data(), 0,
                            sizeof(int32_t) * positions.size());

    // KV write indices: pos..pos+n_tokens-1.
    std::vector<int64_t> kv_idxs(n_tokens);
    for (int i = 0; i < n_tokens; i++) kv_idxs[i] = (int64_t)(pos + i);
    ggml_backend_tensor_set(sg.kv_pos_idx, kv_idxs.data(), 0,
                            sizeof(int64_t) * kv_idxs.size());

    // Causal mask padded to n_kv_padded (256-aligned) so the read view shape
    // stays stable across 256 steps for CUDA-graph reuse.
    {
        const int kv_len = pos + n_tokens;
        const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);
        mask_buf.assign((size_t)sg.n_kv_padded * q_pad, F16_NEG_INF);
        for (int q = 0; q < n_tokens; q++) {
            const int max_k = pos + q;
            for (int k = 0; k <= max_k && k < sg.n_kv_padded; k++) {
                mask_buf[(size_t)q * sg.n_kv_padded + k] = F16_ZERO;
            }
        }
        // Pad q rows: duplicate the last real row's pattern so softmax
        // doesn't see an all-(-inf) row (NaN guard).
        for (int q = n_tokens; q < q_pad; q++) {
            const int max_k = pos + (n_tokens - 1);
            for (int k = 0; k <= max_k && k < sg.n_kv_padded; k++) {
                mask_buf[(size_t)q * sg.n_kv_padded + k] = F16_ZERO;
            }
        }
        ggml_backend_tensor_set(sg.attn_mask, mask_buf.data(), 0,
                                sizeof(uint16_t) * mask_buf.size());
    }

    if (ggml_backend_graph_compute(backend, sg.gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "batch compute failed at pos=%d n=%d\n", pos, n_tokens); std::exit(1);
    }

    // Debug: logits shape + top-5 logits at pos 0 for the FIRST N=4 call
    // (that's the smallest N where we see the pos-0 wrong bug).
    static int dumped = 0;
    if (!dumped && n_tokens == 4) {
        dumped = 1;
        std::printf("[debug] verify logits ne=[%ld,%ld,%ld,%ld]\n",
                    (long)sg.logits->ne[0], (long)sg.logits->ne[1],
                    (long)sg.logits->ne[2], (long)sg.logits->ne[3]);
        // Pull pos-0 logits now
        const int vc = (int)w.embedder.n_vocab;
        std::vector<float> pos0_logits(vc);
        ggml_backend_tensor_get(sg.logits, pos0_logits.data(), 0, sizeof(float) * vc);
        // top-5
        std::vector<int> idx(vc);
        for (int i = 0; i < vc; i++) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
            [&](int a, int b){ return pos0_logits[a] > pos0_logits[b]; });
        std::printf("[debug] pos0 top-5: ");
        for (int k = 0; k < 5; k++)
            std::printf("%d=%.3f ", idx[k], pos0_logits[idx[k]]);
        std::printf("\n[debug] pos0 tok 11=%.3f, tok 13=%.3f, delta=%.3f\n",
                    pos0_logits[11], pos0_logits[13],
                    pos0_logits[11] - pos0_logits[13]);
    }

    // GPU-side argmax per position: read n_tokens * 4 bytes instead of
    // n_tokens * 248K * 4 bytes. Saves ~200 µs/step on the batch verify
    // (multi-MiB H2D transfer for 35B's 248K-token vocab).
    ggml_backend_tensor_get(sg.argmax, argmax_out, 0,
                            sizeof(int32_t) * n_tokens);
    return;
    // (Unreachable old path preserved below for reference/debugging.)
    const int vocab = (int)w.embedder.n_vocab;
    const size_t nbytes = sizeof(float) * (size_t)vocab * (size_t)n_tokens;
    if (logits_buf.size() < (size_t)vocab * (size_t)n_tokens)
        logits_buf.assign((size_t)vocab * n_tokens, 0.f);
    ggml_backend_tensor_get(sg.logits, logits_buf.data(), 0, nbytes);

    for (int i = 0; i < n_tokens; i++) {
        const float * row = logits_buf.data() + (size_t)i * vocab;
        int best = 0; float bv = row[0];
        for (int k = 1; k < vocab; k++)
            if (row[k] > bv) { bv = row[k]; best = k; }
        argmax_out[i] = best;
    }
}

// Tree-mode batched forward. Like verify_batch but:
//   - builds graph with parent_ids + capture_delta_intermediate (tree-mode
//     DeltaNet kernel)
//   - caller supplies parent_ids[n_tokens]; linear chain is [-1, 0, 1, ..., N-2]
//     (GGML_GDN_TREE_ROOT_PARENT = -1 for the first token, else sequential)
//   - positions can be ABSOLUTE per-token depth offsets (for real trees) or
//     linear [pos, pos+1, ..., pos+N-1] (for chain DDTree)
//   - mask can be causal (linear chain) or ancestor-only (real tree)
//
// Returns N argmax predictions in argmax_out.
static void verify_tree(
    const TargetWeights & w,
    TargetCache & cache,
    ggml_backend_t backend,
    StepGraph & sg,
    const int32_t * toks,
    const int32_t * parent_ids_in,
    const int32_t * positions_in,  // 4*n_tokens, axis-major layout
    const uint16_t * mask_in,      // kv_pad * q_pad, f16 bits
    int n_tokens, int kv_start,
    std::vector<float> & embed_buf,
    std::vector<float> & logits_buf,
    int32_t * argmax_out)
{
    if (!build_step_graph_tree(sg, w, cache, backend, kv_start, n_tokens)) {
        std::fprintf(stderr, "build_step_graph_tree(%d) failed at pos=%d\n", n_tokens, kv_start);
        std::exit(1);
    }
    const int hidden = w.n_embd;
    if ((int)embed_buf.size() < hidden * n_tokens) embed_buf.assign(hidden * n_tokens, 0.f);
    if (!w.embedder.embed(toks, n_tokens, embed_buf.data())) {
        std::fprintf(stderr, "tree embed failed\n"); std::exit(1);
    }
    ggml_backend_tensor_set(sg.inp_embed, embed_buf.data(), 0,
                            sizeof(float) * hidden * n_tokens);
    ggml_backend_tensor_set(sg.positions, positions_in, 0, sizeof(int32_t) * 4 * n_tokens);
    ggml_backend_tensor_set(sg.parent_ids, parent_ids_in, 0, sizeof(int32_t) * n_tokens);

    // Caller's mask is expected to be padded to sg.n_kv_padded on the K axis.
    const int q_pad = align_up(n_tokens, KQ_MASK_PAD);
    ggml_backend_tensor_set(sg.attn_mask, mask_in, 0,
                            sizeof(uint16_t) * sg.n_kv_padded * q_pad);

    // KV write indices = sequential slots [kv_start..kv_start+n_tokens-1].
    std::vector<int64_t> kv_idxs(n_tokens);
    for (int i = 0; i < n_tokens; i++) kv_idxs[i] = (int64_t)(kv_start + i);
    ggml_backend_tensor_set(sg.kv_pos_idx, kv_idxs.data(), 0,
                            sizeof(int64_t) * kv_idxs.size());

    if (ggml_backend_graph_compute(backend, sg.gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "tree compute failed\n"); std::exit(1);
    }
    // GPU-side argmax per slot: sg.argmax has shape [n_tokens] i32. Skips
    // the multi-MiB logits H2D transfer.
    ggml_backend_tensor_get(sg.argmax, argmax_out, 0,
                            sizeof(int32_t) * n_tokens);
}

// ─── Draft SSM/conv checkpoint ring ─────────────────────────────────────
//
// Fast draft rollback: instead of `snapshot + replay commit_count forwards`
// after verify, we save ssm_state + conv_state to a per-draft-step slot
// DURING the initial drafting loop, then on rollback just cudaMemcpy the
// saved slot back. Turns commit_count×step_drf (each ~2 ms for 0.8B) into
// one ~400 µs batch of D2D copies per round.
//
// Valid only when the accepted path matches the draft's top-1 chain. If
// DDTree accepts a sibling (accepted[i] != i), the state at slot
// commit_count doesn't correspond to the accepted token sequence and we
// must fall back to sequential replay.
struct DraftCheckpoints {
    int                                 n_slots   = 0;
    int                                 n_delta   = 0;
    std::vector<size_t>                 ssm_bytes;
    std::vector<size_t>                 conv_bytes;
    std::vector<std::vector<void *>>    ssm_bufs;   // [n_slots][n_delta]
    std::vector<std::vector<void *>>    conv_bufs;
};

static bool draft_ckpt_alloc(DraftCheckpoints & ckpt, const TargetCache & c_drf, int n_slots) {
    ckpt.n_slots = n_slots;
    ckpt.n_delta = (int)c_drf.ssm_state.size();
    ckpt.ssm_bytes.resize(ckpt.n_delta);
    ckpt.conv_bytes.resize(ckpt.n_delta);
    ckpt.ssm_bufs.assign(n_slots, std::vector<void *>(ckpt.n_delta, nullptr));
    ckpt.conv_bufs.assign(n_slots, std::vector<void *>(ckpt.n_delta, nullptr));
    for (int l = 0; l < ckpt.n_delta; l++) {
        ckpt.ssm_bytes[l]  = ggml_nbytes(c_drf.ssm_state[l]);
        ckpt.conv_bytes[l] = ggml_nbytes(c_drf.conv_state[l]);
        for (int s = 0; s < n_slots; s++) {
            if (cudaMalloc(&ckpt.ssm_bufs[s][l],  ckpt.ssm_bytes[l])  != cudaSuccess) return false;
            if (cudaMalloc(&ckpt.conv_bufs[s][l], ckpt.conv_bytes[l]) != cudaSuccess) return false;
        }
    }
    return true;
}

static void draft_ckpt_save(const DraftCheckpoints & ckpt, const TargetCache & c_drf, int slot) {
    for (int l = 0; l < ckpt.n_delta; l++) {
        cudaMemcpyAsync(ckpt.ssm_bufs[slot][l],  c_drf.ssm_state[l]->data,
                        ckpt.ssm_bytes[l], cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(ckpt.conv_bufs[slot][l], c_drf.conv_state[l]->data,
                        ckpt.conv_bytes[l], cudaMemcpyDeviceToDevice);
    }
}

static void draft_ckpt_restore(const DraftCheckpoints & ckpt, TargetCache & c_drf, int slot) {
    for (int l = 0; l < ckpt.n_delta; l++) {
        cudaMemcpyAsync(c_drf.ssm_state[l]->data,  ckpt.ssm_bufs[slot][l],
                        ckpt.ssm_bytes[l], cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(c_drf.conv_state[l]->data, ckpt.conv_bufs[slot][l],
                        ckpt.conv_bytes[l], cudaMemcpyDeviceToDevice);
    }
}

// Full-attention KV compaction for DDTree sibling walks.
//
// The verify wrote each full-attn layer's K/V at DFS slots
// [pre_pos..pre_pos+N-1]. For the next round's verify to see the right
// committed prefix, slots [pre_pos..pre_pos+commit_count-1] must hold the
// K/V of the accepted path's tokens. For each committed depth d whose
// accepted DFS index != d, copy slot pre_pos+accepted[d] → slot pre_pos+d.
//
// Safety: accepted[] is monotone strictly increasing (DFS order + the walk
// always descends), so iterating d = 0..commit_count-1 never overwrites a
// slot we still need to read.
static bool ddtree_compact_kv(TargetCache & cache,
                              int pre_pos,
                              const std::vector<int> & accepted,
                              int commit_count)
{
    cudaStream_t stream = nullptr;
    const int n_full_attn = (int)cache.attn_k.size();
    for (int d = 0; d < commit_count; d++) {
        const int src_dfs = accepted[d];
        if (src_dfs == d) continue;
        for (int l = 0; l < n_full_attn; l++) {
            ggml_tensor * K = cache.attn_k[l];
            ggml_tensor * V = cache.attn_v[l];
            const size_t slot_bytes_K = K->nb[1];
            const size_t slot_bytes_V = V->nb[1];
            const int n_kv = (int)K->ne[2];
            for (int h = 0; h < n_kv; h++) {
                const size_t k_src = (size_t)(pre_pos + src_dfs) * slot_bytes_K + (size_t)h * K->nb[2];
                const size_t k_dst = (size_t)(pre_pos + d)       * slot_bytes_K + (size_t)h * K->nb[2];
                const size_t v_src = (size_t)(pre_pos + src_dfs) * slot_bytes_V + (size_t)h * V->nb[2];
                const size_t v_dst = (size_t)(pre_pos + d)       * slot_bytes_V + (size_t)h * V->nb[2];
                cudaError_t ce;
                ce = cudaMemcpyAsync((char *)K->data + k_dst,
                                     (const char *)K->data + k_src,
                                     slot_bytes_K, cudaMemcpyDeviceToDevice, stream);
                if (ce != cudaSuccess) {
                    std::fprintf(stderr, "kv compact K l=%d h=%d: %s\n",
                                 l, h, cudaGetErrorString(ce));
                    return false;
                }
                ce = cudaMemcpyAsync((char *)V->data + v_dst,
                                     (const char *)V->data + v_src,
                                     slot_bytes_V, cudaMemcpyDeviceToDevice, stream);
                if (ce != cudaSuccess) {
                    std::fprintf(stderr, "kv compact V l=%d h=%d: %s\n",
                                 l, h, cudaGetErrorString(ce));
                    return false;
                }
            }
        }
    }
    return true;
}

// Fast DDTree rollback: reconstruct target cache state to "after processing
// accepted path of length commit_count". Ported from test_dflash.cpp.
//
// Inputs (all on device):
//   sg.delta_captures[il].ssm_intermediate_states  f16 [S_v, S_v, H_v, N]
//   sg.delta_captures[il].conv_input               f32 [(K-1)+N, ch, 1]
//   cache.ssm_state[il]                            f32 [S_v, S_v, H_v]
//   cache.conv_state[il]                           f32 [(K-1), ch]
//
// Steps per delta-net layer:
//   (a) SSM: copy ssm_intermediate_states[:, :, :, rollback_dfs] → ssm_state[il]
//       via a single f16→f32 kernel launch (dflash27b_launch_f16_to_f32).
//   (b) Conv: copy the (K-1) most recent conv_input slots along the accepted
//       node's ANCESTRY into conv_state[il]. Pure-chain fast path uses one
//       cudaMemcpy2DAsync over 3 contiguous slots; sibling-accept path walks
//       the parent chain one column at a time.
//
// Caller must:
//   - Set cache.cur_pos = pre_pos + commit_count after this returns
//   - For sibling accepts also compact KV cache slots (not implemented here;
//     we detect and fall back to sequential catch-up)
//
// Returns true iff fast rollback was performed. On sibling-accept paths the
// caller may prefer the simpler sequential catch-up.
static bool ddtree_fast_rollback_target(
    StepGraph & sg, TargetCache & cache,
    const std::vector<int> & accepted,
    const DDTree & tree,
    int commit_count,
    bool walked_sibling)
{
    const int n_delta = (int)sg.delta_captures.size();
    if (n_delta == 0) return false;
    if (commit_count <= 0) return false;

    const int rollback_dfs = accepted[commit_count - 1];
    cudaStream_t stream = nullptr;  // default stream

    for (int il = 0; il < n_delta; il++) {
        const dflash27b::DeltaNetCapture & cap = sg.delta_captures[il];
        if (!cap.ssm_intermediate_states || !cap.conv_input) return false;

        // (a) SSM: slot `rollback_dfs` of ssm_intermediate_states (f16 or f32)
        //     → ssm_state[il] (f32).
        const size_t ssm_elems =
            (size_t)cache.ssm_state[il]->ne[0] *
            (size_t)cache.ssm_state[il]->ne[1] *
            (size_t)cache.ssm_state[il]->ne[2];
        const size_t ssm_src_offset =
            (size_t)rollback_dfs * cap.ssm_intermediate_states->nb[3];
        const void * ssm_src =
            (const char *)cap.ssm_intermediate_states->data + ssm_src_offset;
        if (cap.ssm_intermediate_states->type == GGML_TYPE_F16) {
            dflash27b_launch_f16_to_f32(ssm_src, cache.ssm_state[il]->data,
                                        ssm_elems, stream);
        } else {
            // F32: plain memcpy.
            cudaError_t ce = cudaMemcpyAsync(cache.ssm_state[il]->data, ssm_src,
                                             ssm_elems * sizeof(float),
                                             cudaMemcpyDeviceToDevice, stream);
            if (ce != cudaSuccess) {
                std::fprintf(stderr, "fast rollback ssm il=%d: %s\n",
                             il, cudaGetErrorString(ce));
                return false;
            }
        }

        // (b) Conv rollback: (K-1)=3 contiguous slots along ancestry.
        const int K_conv = 4;
        const int row_cnt = (int)cap.conv_input->ne[1];
        const size_t elt = ggml_element_size(cap.conv_input);
        const size_t dpitch = (size_t)(K_conv - 1) * elt;
        const size_t spitch = cap.conv_input->nb[1];

        if (!walked_sibling) {
            // Fast path: conv window is 3 contiguous slots ending at rollback_dfs.
            const int conv_off = rollback_dfs + 1;
            const void * conv_src =
                (const char *)cap.conv_input->data + (size_t)conv_off * elt;
            cudaError_t ce = cudaMemcpy2DAsync(cache.conv_state[il]->data, dpitch,
                                               conv_src, spitch,
                                               (size_t)(K_conv - 1) * elt, row_cnt,
                                               cudaMemcpyDeviceToDevice, stream);
            if (ce != cudaSuccess) {
                std::fprintf(stderr, "fast rollback conv il=%d: %s\n",
                             il, cudaGetErrorString(ce));
                return false;
            }
        } else {
            // Sibling path: walk parent chain for (K-1) predecessors.
            int virt[K_conv - 1];
            virt[K_conv - 2] = rollback_dfs;
            for (int m = K_conv - 3; m >= 0; m--) {
                const int prev = virt[m + 1];
                virt[m] = (prev >= 0) ? (int)tree.parents[prev] : (prev - 1);
            }
            for (int m = 0; m < K_conv - 1; m++) {
                const int sx_slot = (K_conv - 1) + virt[m];
                const void * src_col =
                    (const char *)cap.conv_input->data + (size_t)sx_slot * elt;
                char * dst_col =
                    (char *)cache.conv_state[il]->data + (size_t)m * elt;
                cudaError_t ce = cudaMemcpy2DAsync(dst_col, dpitch,
                                                   src_col, spitch,
                                                   elt, row_cnt,
                                                   cudaMemcpyDeviceToDevice, stream);
                if (ce != cudaSuccess) {
                    std::fprintf(stderr, "fast rollback sib il=%d m=%d: %s\n",
                                 il, m, cudaGetErrorString(ce));
                    return false;
                }
            }
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 7) {
        std::fprintf(stderr,
            "usage: %s <target.gguf> <draft.gguf> <prompt.bin> <n_gen> <N_spec> <out.bin>\n",
            argv[0]);
        return 1;
    }
    const char * target_path = argv[1];
    const char * draft_path  = argv[2];
    const char * prompt_path = argv[3];
    const int    n_gen       = std::atoi(argv[4]);
    const int    N_spec      = std::atoi(argv[5]);
    const char * out_path    = argv[6];
    if (N_spec < 1 || N_spec > 64) { std::fprintf(stderr, "N_spec out of range\n"); return 1; }

    // Verify mode:
    //   CHAIN_VERIFY=seq          sequential single-token target verify,
    //                             byte-identical to AR but slow (~50 tok/s)
    //   CHAIN_VERIFY=batch        batched N-token verify via plain
    //                             ggml_gated_delta_net. Fast (~140 tok/s at
    //                             N=16) but fp16 MMA accumulation resolves
    //                             near-tie argmax differently than VEC fp32
    //                             (see M2b notes).
    //   CHAIN_VERIFY=tree_chain   batched N-token verify via the tree-mode
    //                             kernel (ggml_gated_delta_net_tree_persist)
    //                             with linear parent_ids. Same op structure
    //                             as real DDTree but no branching. Tests
    //                             whether the tree kernel has different
    //                             numerics than the plain batched kernel.
    //   CHAIN_VERIFY=ddtree       Full DDTree: per-step top-K extraction,
    //                             best-first branching tree build (budget =
    //                             DDTREE_BUDGET env, default N_spec+8),
    //                             ancestor-only mask, walk accepted path.
    // Default = seq (lossless).
    enum VerifyMode { VERIFY_SEQ, VERIFY_BATCH, VERIFY_TREE_CHAIN, VERIFY_DDTREE };
    VerifyMode verify_mode = VERIFY_SEQ;
    if (const char * s = std::getenv("CHAIN_VERIFY")) {
        std::string m = s;
        if      (m == "batch")      verify_mode = VERIFY_BATCH;
        else if (m == "tree_chain") verify_mode = VERIFY_TREE_CHAIN;
        else if (m == "ddtree")     verify_mode = VERIFY_DDTREE;
    }
    const bool use_batched_verify =
        (verify_mode == VERIFY_BATCH || verify_mode == VERIFY_TREE_CHAIN ||
         verify_mode == VERIFY_DDTREE);

    int ddtree_budget = N_spec + 8;
    int ddtree_K      = 8;
    float ddtree_temp = 1.0f;
    if (const char * s = std::getenv("DDTREE_BUDGET")) ddtree_budget = std::atoi(s);
    if (const char * s = std::getenv("DDTREE_K"))      ddtree_K      = std::atoi(s);
    if (const char * s = std::getenv("DDTREE_TEMP"))   ddtree_temp   = std::atof(s);
    if (ddtree_budget < 1) ddtree_budget = 1;
    if (ddtree_K      < 1) ddtree_K      = 1;
    if (verify_mode == VERIFY_DDTREE) {
        std::printf("[ddtree] budget=%d  K=%d  temp=%.2f  L=%d\n",
                    ddtree_budget, ddtree_K, ddtree_temp, N_spec);
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "cuda init failed\n"); return 1; }

    TargetWeights w_tgt, w_drf;
    if (!load_target_gguf(target_path, backend, w_tgt)) {
        std::fprintf(stderr, "target: %s\n", dflash27b_last_error()); return 1;
    }
    std::printf("[target] %s\n", dflash27b_last_error());
    if (!load_target_gguf(draft_path, backend, w_drf)) {
        std::fprintf(stderr, "draft: %s\n", dflash27b_last_error()); return 1;
    }
    std::printf("[draft]  %s\n", dflash27b_last_error());
    if (w_tgt.embedder.n_vocab != w_drf.embedder.n_vocab) {
        std::fprintf(stderr, "tokenizer mismatch\n"); return 1;
    }

    const int max_ctx = 4096;
    TargetCache c_tgt, c_drf;
    // Size the target's verify-capture tensors (ssm_intermediate, conv_input_cache)
    // for the max n_tokens a single verify call will feed. For chain modes that's
    // N_spec; for DDTree it's 1 + ddtree_budget (root + tree nodes). The conv_input
    // capture cpy asserts ggml_nelements(a) == ggml_nelements(b), so the cache tensor
    // size must match exactly what the graph builds.
    const int verify_max_tokens = (verify_mode == VERIFY_DDTREE)
        ? std::max(N_spec, 1 + ddtree_budget)
        : N_spec;
    if (!create_target_cache(w_tgt, max_ctx, verify_max_tokens, backend, c_tgt)) {
        std::fprintf(stderr, "tgt cache: %s\n", dflash27b_last_error()); return 1;
    }
    if (!create_target_cache(w_drf, max_ctx, 0, backend, c_drf)) {
        std::fprintf(stderr, "drf cache: %s\n", dflash27b_last_error()); return 1;
    }

    auto prompt = read_int32_file(prompt_path);
    if (prompt.empty()) { std::fprintf(stderr, "empty prompt\n"); return 1; }
    std::printf("[prompt] %zu tokens: ", prompt.size());
    for (auto t : prompt) std::printf("%d ", t);
    std::printf("\n");
    if ((int)prompt.size() + n_gen + N_spec > max_ctx) {
        std::fprintf(stderr, "prompt+gen+spec exceeds max_ctx\n"); return 1;
    }

    std::vector<float> embed_tgt(w_tgt.n_embd), embed_drf(w_drf.n_embd);
    std::vector<float> logits_tgt, logits_drf;
    StepGraph sg_tgt, sg_drf;

    auto step_tgt = [&](int32_t tok, int pos) {
        return step_model(w_tgt, c_tgt, backend, sg_tgt, tok, pos, embed_tgt, logits_tgt,
                          /*want_logits=*/false);
    };
    auto step_drf = [&](int32_t tok, int pos, bool want_logits = false) {
        return step_model(w_drf, c_drf, backend, sg_drf, tok, pos, embed_drf, logits_drf,
                          want_logits);
    };

    // ── Minimal batched-verify diagnostic. BEFORE running any draft, run
    //    the target prefill via single-token step_tgt, snapshot SSM, do N
    //    single-step target predictions, then restore and do ONE batched
    //    verify. Compare the two sequences — they MUST agree token-for-token
    //    because they're both the target greedy-decoding the same sequence.
    if (const char * s = std::getenv("CHAIN_DIAG")) {
        (void)s;
        std::printf("[diag] starting batched-verify diagnostic\n");
        // Prefill target only on the full prompt.
        int32_t tp = -1;
        for (int i = 0; i < (int)prompt.size(); i++) {
            tp = step_tgt(prompt[i], i);
            c_tgt.cur_pos = i + 1;
        }
        const int P = (int)prompt.size();
        const int N = 16;
        int32_t seq_preds[16];
        int32_t batch_preds[16];

        // Snapshot target state after prefill.
        snapshot_ssm_state(c_tgt);

        // Sequential: feed tp, then each prediction as input to the next step.
        int32_t cur = tp;
        for (int i = 0; i < N; i++) {
            seq_preds[i] = step_tgt(cur, P + i);
            c_tgt.cur_pos = P + i + 1;
            cur = seq_preds[i];
        }
        std::printf("[diag] seq  : ");
        for (int i = 0; i < N; i++) std::printf("%d ", seq_preds[i]);
        std::printf("\n");

        // Restore target state + reset cur_pos.
        restore_ssm_state(c_tgt);
        c_tgt.cur_pos = P;

        // Batched at various N values (fresh snapshot+restore each time).
        int32_t all_inputs[16] = { tp };
        for (int i = 1; i < 16; i++) all_inputs[i] = seq_preds[i - 1];
        for (int try_n : { 1, 2, 3, 4, 5, 6, 8, 12, 16 }) {
            restore_ssm_state(c_tgt);
            c_tgt.cur_pos = P;
            StepGraph sg_try; std::vector<uint16_t> mbt;
            std::vector<float> ebt(w_tgt.n_embd * try_n), lbt;
            int32_t out_try[16];
            for (int i = 0; i < 16; i++) out_try[i] = -1;
            verify_batch(w_tgt, c_tgt, backend, sg_try,
                         all_inputs, try_n, P, ebt, lbt, mbt, out_try);
            step_graph_destroy(sg_try);
            std::printf("[diag] batched N=%d:", try_n);
            for (int i = 0; i < try_n; i++)
                std::printf(" %d%s", out_try[i], out_try[i] == seq_preds[i] ? "" : "!");
            std::printf("\n");
        }
        // Restore + batched N=4 on the same sequence.
        restore_ssm_state(c_tgt);
        c_tgt.cur_pos = P;
        StepGraph sg_batch;
        int32_t inputs[4] = { tp, seq_preds[0], seq_preds[1], seq_preds[2] };
        std::vector<uint16_t> mbuf;
        std::vector<float> embed_batch(w_tgt.n_embd * N), logits_batch;
        verify_batch(w_tgt, c_tgt, backend, sg_batch,
                     inputs, N, P, embed_batch, logits_batch, mbuf, batch_preds);
        std::printf("[diag] batch #1: ");
        for (int i = 0; i < N; i++) std::printf("%d ", batch_preds[i]);
        std::printf("\n");

        // Run it AGAIN without rebuild. K/V slots P..P+N-1 now contain the
        // correct writes from batch #1 (matching what sequential would have
        // written). If batch #2 gives correct pos 0, the bug is a write-before-
        // read ordering issue where flash_attn_ext reads slot P BEFORE the
        // cpy op for position 0 has landed.
        restore_ssm_state(c_tgt);
        c_tgt.cur_pos = P;
        int32_t batch_preds2[4];
        verify_batch(w_tgt, c_tgt, backend, sg_batch,
                     inputs, N, P, embed_batch, logits_batch, mbuf, batch_preds2);
        std::printf("[diag] batch #2: ");
        for (int i = 0; i < N; i++) std::printf("%d ", batch_preds2[i]);
        std::printf("\n");
        step_graph_destroy(sg_batch);
        for (int i = 0; i < N; i++) batch_preds[i] = batch_preds2[i];
        std::printf("[diag] batch: ");
        for (int i = 0; i < N; i++) std::printf("%d ", batch_preds[i]);
        std::printf("\n");

        bool match = true;
        for (int i = 0; i < N; i++) if (seq_preds[i] != batch_preds[i]) { match = false; break; }
        std::printf("[diag] %s\n", match ? "PASS: seq == batch" : "FAIL: batch != seq");
        return match ? 0 : 1;
    }

    // Prefill: run both models on the full prompt sequentially. After this,
    // both caches are at cur_pos = prompt.size(). last_tok_to_feed is the
    // target's prediction for position prompt.size() (first generation
    // position).
    int32_t last_tok = -1;
    for (int i = 0; i < (int)prompt.size(); i++) {
        (void)step_drf(prompt[i], i); c_drf.cur_pos = i + 1;
        int32_t t  = step_tgt(prompt[i], i); c_tgt.cur_pos = i + 1;
        if (i == (int)prompt.size() - 1) last_tok = t;
    }
    std::printf("[prefill] first-gen token = target argmax = %d\n", last_tok);

    // Main chain-spec loop.
    //
    // Semantics at the TOP of each round:
    //   last_tok is the token to be consumed at position pre_pos (NOT yet
    //   committed). Both caches are at cur_pos = pre_pos. pre_pos-1 worth of
    //   tokens are already committed (prompt + prior-round commits).
    //
    // Within a round, model.step(tok, pos) consumes `tok` at `pos` and
    // produces prediction for `pos+1` as the argmax. That matches how
    // test_generate.cpp drives the model.
    //
    // Draft: runs N iterations. Iter i consumes carry at pre_pos+i,
    //   producing drafts[i] = prediction for pre_pos+i+1. carry is last_tok
    //   (i=0) then drafts[i-1]. Draft cache ends at pre_pos+N.
    //
    // Target verify: runs similarly BUT stops at first mismatch. Iter i
    //   consumes carry at pre_pos+i, producing t_pred = prediction for
    //   pre_pos+i+1. Compare t_pred to drafts[i]. If match, continue;
    //   otherwise break. k = count of matches. Target cache ends at
    //   pre_pos + k + 1 (last iter consumed but didn't match — its cache
    //   position reflects consumption of k+1 tokens: last_tok + drafts[0..k-1]
    //   plus one extra). Wait — on mismatch at iter k (0-indexed),
    //   target consumed k+1 carry values: last_tok + drafts[0..k-1]. cur_pos
    //   = pre_pos + k + 1.
    //
    //   On k == N (all match): iter N-1 was the last, consumed drafts[N-2],
    //   produced t_pred matching drafts[N-1]. Target consumed last_tok +
    //   drafts[0..N-2] = N tokens. cur_pos = pre_pos + N.
    //
    // Commits this round:
    //   commit_count = min(k + 1, N).
    //     k <  N:  commit_count = k + 1 tokens:  last_tok + drafts[0..k-1].
    //     k == N:  commit_count = N tokens:      last_tok + drafts[0..N-2].
    //   next_last_tok:
    //     k <  N:  t_pred (target's replacement for the mismatch).
    //     k == N:  drafts[N-1] (all drafts agreed; this is the bonus).
    //
    // Target cache after the round is correctly at pre_pos + commit_count in
    // both cases. No target rollback needed.
    //
    // Draft cache after N iters is at pre_pos + N. Rollback to pre_pos +
    // commit_count. For the KV part this is a simple cur_pos decrement. For
    // the SSM state we `restore_ssm_state(c_drf)` (reverts to pre-round) and
    // re-run commit_count catch-up forwards to evolve state through the
    // committed tokens. This doubles draft cost per round (N + commit_count
    // draft forwards total) but is a dead-simple way to get correctness
    // first.

    std::vector<int32_t> gen;  gen.reserve(n_gen + N_spec);
    std::vector<int32_t> drafts(N_spec);
    std::vector<int32_t> verify_in(N_spec);   // [last_tok, drafts[0], ..., drafts[N-2]]
    std::vector<int32_t> verify_out(N_spec);
    std::vector<uint16_t> mask_buf;

    int n_rounds        = 0;
    long long n_drafted = 0;
    long long n_matched = 0;
    long long n_ddtree_sibling_walks = 0;
    long long n_ddtree_sibling_commits = 0;

    // Draft rollback checkpoints: one slot per draft step (0..N_spec), where
    // slot 0 is pre-round baseline and slot i is post-step_drf i-1. Enabled
    // whenever the draft has any delta-net layers (which is true for all
    // qwen35-family drafts). Memory: ~N_spec * sum(layer ssm_state sizes).
    DraftCheckpoints draft_ckpt{};
    const bool use_draft_ckpt = (!c_drf.ssm_state.empty());
    if (use_draft_ckpt) {
        if (!draft_ckpt_alloc(draft_ckpt, c_drf, N_spec + 1)) {
            std::fprintf(stderr, "draft_ckpt_alloc failed — falling back to sequential catch-up\n");
        }
    }

    auto t_start = std::chrono::steady_clock::now();
    while ((int)gen.size() < n_gen) {
        const int pre_pos = c_tgt.cur_pos;
        if (c_drf.cur_pos != pre_pos) {
            std::fprintf(stderr, "cache desync: drf=%d tgt=%d\n", c_drf.cur_pos, pre_pos);
            return 1;
        }

        // Snapshot both models' SSM state pre-round.
        snapshot_ssm_state(c_drf);
        snapshot_ssm_state(c_tgt);

        // Draft baseline checkpoint (slot 0 = pre-round state). This is what
        // we'd restore via the old snapshot+replay path; keeping a copy here
        // lets fast-rollback reach commit_count==0 cleanly (just a memcpy).
        if (use_draft_ckpt) draft_ckpt_save(draft_ckpt, c_drf, 0);

        // 1. Draft: N sequential forwards starting from last_tok.
        //    In DDTree mode we also capture per-position top-K log-probs from
        //    the draft's full logits vector for tree-building. We also save
        //    the draft's ssm+conv state to checkpoint slot i+1 after each
        //    step — the fast-rollback path restores from checkpoint[commit]
        //    instead of replaying commit_count step_drf forwards.
        const int L_ddtree = N_spec;
        std::vector<float>   ddtree_logp;
        std::vector<int32_t> ddtree_toks;
        if (verify_mode == VERIFY_DDTREE) {
            ddtree_logp.assign((size_t)L_ddtree * ddtree_K, 0.0f);
            ddtree_toks.assign((size_t)L_ddtree * ddtree_K, 0);
        }
        int32_t carry = last_tok;
        for (int i = 0; i < N_spec; i++) {
            drafts[i] = step_drf(carry, pre_pos + i, /*want_logits=*/verify_mode == VERIFY_DDTREE);
            c_drf.cur_pos = pre_pos + i + 1;
            if (use_draft_ckpt) draft_ckpt_save(draft_ckpt, c_drf, i + 1);
            if (verify_mode == VERIFY_DDTREE) {
                extract_draft_topk(logits_drf.data(), 1, (int)w_drf.embedder.n_vocab,
                                   ddtree_K,
                                   ddtree_logp.data() + (size_t)i * ddtree_K,
                                   ddtree_toks.data() + (size_t)i * ddtree_K,
                                   ddtree_temp);
            }
            carry = drafts[i];
        }
        n_drafted += N_spec;

        // 2. Target verify — one of four modes.
        int k = 0;                     // number of accepted draft matches (chain)
        int commit_count = 0;          // total tokens committed this round
        int32_t t_pred = -1;           // target's correction at first mismatch (chain)
        int32_t next_last_tok = -1;    // last_tok carry for next round
        std::vector<int32_t> accepted_tokens;  // tokens in order, length commit_count
        // DDTree scratch (populated only in VERIFY_DDTREE branch, used by
        // the fast-rollback path below).
        DDTree ddtree_tree;
        std::vector<int> ddtree_accepted;
        if (verify_mode == VERIFY_BATCH) {
            verify_in[0] = last_tok;
            for (int i = 1; i < N_spec; i++) verify_in[i] = drafts[i - 1];
            verify_batch(w_tgt, c_tgt, backend, sg_tgt,
                         verify_in.data(), N_spec, pre_pos,
                         embed_tgt, logits_tgt, mask_buf, verify_out.data());
            c_tgt.cur_pos = pre_pos + N_spec;
            for (int i = 0; i < N_spec; i++) {
                if (verify_out[i] == drafts[i]) k++;
                else break;
            }
            t_pred = verify_out[k < N_spec ? k : N_spec - 1];
            commit_count = std::min(k + 1, N_spec);
            accepted_tokens.assign(1, last_tok);
            for (int i = 0; i + 1 < commit_count; i++) accepted_tokens.push_back(drafts[i]);
            next_last_tok = (k < N_spec) ? t_pred : drafts[N_spec - 1];
        } else if (verify_mode == VERIFY_TREE_CHAIN) {
            // Linear chain via tree-mode kernel. parent_ids[0]=-1 (root
            // parent sentinel), parent_ids[i]=i-1 (sequential for i>0).
            // Positions are linear like batched; mask is standard causal.
            verify_in[0] = last_tok;
            for (int i = 1; i < N_spec; i++) verify_in[i] = drafts[i - 1];
            std::vector<int32_t> parent_ids(N_spec);
            parent_ids[0] = -1;  // GGML_GDN_TREE_ROOT_PARENT
            for (int i = 1; i < N_spec; i++) parent_ids[i] = i - 1;
            std::vector<int32_t> tree_positions(4 * N_spec);
            for (int i = 0; i < N_spec; i++) {
                tree_positions[0 * N_spec + i] = pre_pos + i;
                tree_positions[1 * N_spec + i] = pre_pos + i;
                tree_positions[2 * N_spec + i] = pre_pos + i;
                tree_positions[3 * N_spec + i] = 0;
            }
            const int kv_len = pre_pos + N_spec;
            const int n_kv_padded = ((kv_len + 255) / 256) * 256;
            build_causal_mask_f16(mask_buf, kv_len, N_spec, pre_pos, n_kv_padded);
            verify_tree(w_tgt, c_tgt, backend, sg_tgt,
                        verify_in.data(), parent_ids.data(),
                        tree_positions.data(), mask_buf.data(),
                        N_spec, pre_pos,
                        embed_tgt, logits_tgt, verify_out.data());
            c_tgt.cur_pos = pre_pos + N_spec;
            for (int i = 0; i < N_spec; i++) {
                if (verify_out[i] == drafts[i]) k++;
                else break;
            }
            t_pred = verify_out[k < N_spec ? k : N_spec - 1];
            commit_count = std::min(k + 1, N_spec);
            accepted_tokens.assign(1, last_tok);
            for (int i = 0; i + 1 < commit_count; i++) accepted_tokens.push_back(drafts[i]);
            next_last_tok = (k < N_spec) ? t_pred : drafts[N_spec - 1];
        } else if (verify_mode == VERIFY_DDTREE) {
            // Build best-first branching tree from draft's per-position top-K.
            ddtree_tree = build_ddtree(ddtree_logp.data(), ddtree_toks.data(),
                                       L_ddtree, ddtree_K, ddtree_budget, /*chain_seed=*/true);
            DDTree & tree = ddtree_tree;
            // Pad the tree to EXACTLY ddtree_budget non-root nodes so the
            // graph's n_tokens (=1+budget) is constant across rounds. That
            // constancy is what lets ggml-cuda reuse the captured CUDA graph
            // via cudaGraphExecUpdate (same op sequence, same shapes; only
            // tensor values change per step). Real trees with <budget nodes
            // happen when the heap exhausts or the best-first search picks
            // fewer siblings than allotted.
            //
            // Dummy leaves are parented at the root with a pad token_id (0).
            // The visibility rebuild below ensures: (a) real queries can't
            // see dummies (they weren't ancestors); (b) dummy queries only
            // see root + self, so they produce well-defined (but discarded)
            // output; (c) no NaN softmax rows.
            while (tree.n_nodes < ddtree_budget) {
                tree.token_ids.push_back(0);
                tree.depths.push_back(0);
                tree.parents.push_back(0);
                tree.child_maps.emplace_back();
                tree.n_nodes++;
            }
            // Rebuild visibility over the padded size. For real nodes this
            // recreates the original ancestor chain; for dummies (parent=0)
            // it sets visibility[dummy][0] and visibility[dummy][dummy].
            // Crucially, visibility[real][dummy] stays 0 because the j<i
            // loop never writes to j >= i; dummies are appended AFTER the
            // real nodes so their indices are > real indices.
            {
                const int N = 1 + tree.n_nodes;
                tree.visibility.assign((size_t)N * N, 0);
                tree.visibility[0] = 1;  // root sees itself
                for (int i = 1; i < N; i++) {
                    const int p = tree.parents[i];
                    for (int j = 0; j < i; j++) {
                        tree.visibility[(size_t)i * N + j] =
                            tree.visibility[(size_t)p * N + j];
                    }
                    tree.visibility[(size_t)i * N + i] = 1;
                }
            }
            const int N = 1 + tree.n_nodes;  // root + tree nodes (= 1 + budget)
            // Flat tokens: slot 0 = root (= last_tok), slots 1..N-1 = tree nodes.
            std::vector<int32_t> flat_tokens(N);
            flat_tokens[0] = last_tok;
            for (int i = 0; i < tree.n_nodes; i++) flat_tokens[1 + i] = tree.token_ids[i];
            // Positions (axis-major): committed + depth.
            std::vector<int32_t> pos4(4 * N);
            for (int i = 0; i < N; i++) {
                const int p = pre_pos + (i == 0 ? 0 : tree.depths[i - 1]);
                pos4[0 * N + i] = p;
                pos4[1 * N + i] = p;
                pos4[2 * N + i] = p;
                pos4[3 * N + i] = 0;
            }
            // Ancestor-only mask, padded to 256-aligned n_kv_padded so
            // the K-read view shape stays stable for CUDA-graph reuse.
            const int tree_kv_len    = pre_pos + N;
            const int tree_kv_padded = ((tree_kv_len + 255) / 256) * 256;
            build_tree_mask(tree, pre_pos, mask_buf, tree_kv_padded);
            // parent_ids (slot 0 = -1 root sentinel).
            std::vector<int32_t> parent_ids_tree(N);
            parent_ids_tree[0] = -1;
            for (int i = 1; i < N; i++) parent_ids_tree[i] = (int32_t)tree.parents[i];
            // Verify.
            std::vector<int32_t> posterior((size_t)N);
            verify_tree(w_tgt, c_tgt, backend, sg_tgt,
                        flat_tokens.data(), parent_ids_tree.data(),
                        pos4.data(), mask_buf.data(),
                        N, pre_pos,
                        embed_tgt, logits_tgt, posterior.data());
            c_tgt.cur_pos = pre_pos + N;
            // Walk tree following target's argmax at each slot.
            int next_tok_w = -1;
            ddtree_accepted = follow_verified_tree(tree, posterior.data(), next_tok_w);
            const int accept_depth = (int)ddtree_accepted.size();  // includes root
            k = accept_depth - 1;  // matched children (for stats)
            commit_count = accept_depth;
            accepted_tokens.resize(commit_count);
            for (int i = 0; i < commit_count; i++) {
                const int dfs_idx = ddtree_accepted[i];
                accepted_tokens[i] = (dfs_idx == 0) ? last_tok : tree.token_ids[dfs_idx - 1];
            }
            next_last_tok = next_tok_w;
        } else {
            // Sequential verify: short-circuit on first mismatch; target
            // cache naturally stops at pre_pos + commit_count.
            int32_t carry2 = last_tok;
            for (int i = 0; i < N_spec; i++) {
                t_pred = step_tgt(carry2, pre_pos + i);
                c_tgt.cur_pos = pre_pos + i + 1;
                if (t_pred == drafts[i]) { k++; carry2 = drafts[i]; }
                else                    { break; }
            }
            commit_count = std::min(k + 1, N_spec);
            accepted_tokens.assign(1, last_tok);
            for (int i = 0; i + 1 < commit_count; i++) accepted_tokens.push_back(drafts[i]);
            next_last_tok = (k < N_spec) ? t_pred : drafts[N_spec - 1];
        }
        n_matched += k;

        // Emit committed tokens.
        for (int i = 0; i < commit_count; i++) gen.push_back(accepted_tokens[i]);

        // 4. Rollback.
        //    Target side — batched path consumed all N_spec (or tree N) tokens
        //    into cache. Needs catch-up whenever the committed prefix diverges
        //    from the chain that batched/tree verify walked. In chain modes
        //    when commit_count == N_spec this is a no-op; in DDTree we always
        //    rebuild the cache to the accepted path (siblings in tree ≠ spine).
        //    Draft side — always consumed all N_spec; needs catch-up when the
        //    accepted path diverges from the draft top-1 chain, i.e. when any
        //    accepted_tokens[i] != drafts[i-1] for i in [1, commit_count).
        auto catch_up = [&](const TargetWeights & w, TargetCache & cache, StepGraph & sg,
                            std::vector<float> & eb, std::vector<float> & lb) {
            restore_ssm_state(cache);
            cache.cur_pos = pre_pos;
            for (int i = 0; i < commit_count; i++) {
                (void)step_model(w, cache, backend, sg, accepted_tokens[i], pre_pos + i, eb, lb);
                cache.cur_pos = pre_pos + i + 1;
            }
        };
        bool accepted_matches_chain = true;
        for (int i = 1; i < commit_count; i++) {
            if (accepted_tokens[i] != drafts[i - 1]) { accepted_matches_chain = false; break; }
        }

        if (verify_mode == VERIFY_DDTREE) {
            // DDTree rollback.
            //
            // Target (35B, expensive): the verify_tree step wrote KV/SSM/conv
            // state for all N = 1 + tree.n_nodes flat tree slots into cache.
            // We want the cache at state "processed accepted_tokens only".
            //
            //   - Chain walk (accepted[i] == i for i < commit_count AND no
            //     accepted DFS index exceeds L_ddtree = N_spec): spine KV
            //     slots [0..commit_count-1] are already the accepted tokens.
            //     Truncate cur_pos and use fast CUDA rollback for SSM + conv.
            //     Skips ~commit_count sequential step_model calls at target
            //     size — the whole DDTree throughput win.
            //   - Sibling walk (any accepted DFS index > L_ddtree OR any
            //     accepted[i] != i): KV slots along accepted path are
            //     scattered; compaction would work but isn't implemented.
            //     Fall back to sequential catch-up.
            // Sibling detection: any accepted[i] != i. The chain-seeded build
            // places spine at DFS slots [1..L], so a walk that stays on spine
            // has accepted[i] == i for every i. A sibling pop lands in a slot
            // beyond L, making accepted[i] != i at that depth.
            bool walked_sibling = false;
            int sibling_d = -1;
            for (size_t ii = 0; ii < ddtree_accepted.size(); ii++) {
                if (ddtree_accepted[ii] != (int)ii) {
                    walked_sibling = true;
                    if (sibling_d < 0) sibling_d = (int)ii;
                }
            }
            if (walked_sibling) {
                n_ddtree_sibling_walks++;
                // Count committed tokens that came from sibling path (d >= sibling_d).
                n_ddtree_sibling_commits += std::max(0, commit_count - sibling_d);
            }

            bool fast_ok = false;
            if (commit_count > 0) {
                // On sibling walks, compact full-attn KV slots so that
                // slot[pre_pos+d] holds accepted[d]'s K/V. SSM + conv rollback
                // uses the tree parent chain for the window; see
                // ddtree_fast_rollback_target.
                bool kv_ok = walked_sibling
                    ? ddtree_compact_kv(c_tgt, pre_pos, ddtree_accepted, commit_count)
                    : true;
                if (kv_ok) {
                    fast_ok = ddtree_fast_rollback_target(
                        sg_tgt, c_tgt, ddtree_accepted, ddtree_tree,
                        commit_count, walked_sibling);
                }
                if (fast_ok) c_tgt.cur_pos = pre_pos + commit_count;
            }
            if (!fast_ok) catch_up(w_tgt, c_tgt, sg_tgt, embed_tgt, logits_tgt);

            // Draft rollback (0.8B).
            //   - Chain-matched, commit_count in [0, N_spec]: slot commit_count
            //     was saved during the draft loop — restore it in one memcpy.
            //   - Chain-matched, commit_count == N_spec+1: extra step needed
            //     (the N_spec'th draft was never actually processed; only its
            //     argmax was captured). One step_model call.
            //   - Sibling walk (DDTree only): checkpoints don't match the
            //     accepted token sequence; fall back to sequential replay.
            if (accepted_matches_chain && use_draft_ckpt && commit_count <= N_spec) {
                draft_ckpt_restore(draft_ckpt, c_drf, commit_count);
                c_drf.cur_pos = pre_pos + commit_count;
            } else if (accepted_matches_chain && commit_count == N_spec + 1) {
                if (use_draft_ckpt) {
                    draft_ckpt_restore(draft_ckpt, c_drf, N_spec);
                    c_drf.cur_pos = pre_pos + N_spec;
                }
                (void)step_model(w_drf, c_drf, backend, sg_drf,
                                 drafts[N_spec - 1], pre_pos + N_spec,
                                 embed_drf, logits_drf);
                c_drf.cur_pos = pre_pos + N_spec + 1;
            } else {
                catch_up(w_drf, c_drf, sg_drf, embed_drf, logits_drf);
            }
        } else {
            // Chain modes: draft cache is at pre_pos + N_spec; target cache
            // is at pre_pos + N_spec (batched) or pre_pos + commit_count
            // (seq). Catch up iff commit_count differs from where the cache
            // sits, OR if the accepted prefix diverges from the linear draft
            // chain.
            const bool draft_needs_catchup =
                (commit_count < N_spec) || !accepted_matches_chain;
            const bool tgt_needs_catchup =
                use_batched_verify &&
                (commit_count < N_spec || !accepted_matches_chain);

            // Fast-path draft rollback via checkpoints — avoids replaying
            // commit_count sequential step_drf forwards (~2 ms each at 0.8B).
            if (draft_needs_catchup && accepted_matches_chain && use_draft_ckpt) {
                draft_ckpt_restore(draft_ckpt, c_drf, commit_count);
                c_drf.cur_pos = pre_pos + commit_count;
            } else if (draft_needs_catchup) {
                catch_up(w_drf, c_drf, sg_drf, embed_drf, logits_drf);
            }

            // Fast target rollback for tree_chain: its build_step_graph_tree
            // captures ssm_intermediate per slot, same as DDTree. For chain
            // mode the accepted DFS indices are [0..commit_count-1], so we
            // construct that synthetic "accepted" path and reuse DDTree's
            // fast rollback machinery. KV slots are already contiguous (chain
            // walk), no compaction needed. Saves commit_count × 35B step_model
            // calls (~5 ms each) per round, the dominant cost at 35B scale.
            // Batch mode uses build_step_graph (no capture) so it still needs
            // sequential catch-up.
            bool fast_tgt_done = false;
            if (tgt_needs_catchup && verify_mode == VERIFY_TREE_CHAIN
                && accepted_matches_chain && commit_count > 0)
            {
                std::vector<int> synth_chain(commit_count);
                for (int i = 0; i < commit_count; i++) synth_chain[i] = i;
                DDTree dummy_tree;  // only parents[] is read for !walked_sibling branch
                if (ddtree_fast_rollback_target(sg_tgt, c_tgt, synth_chain,
                                                dummy_tree, commit_count,
                                                /*walked_sibling=*/false)) {
                    c_tgt.cur_pos = pre_pos + commit_count;
                    fast_tgt_done = true;
                }
            }
            if (tgt_needs_catchup && !fast_tgt_done) {
                catch_up(w_tgt, c_tgt, sg_tgt, embed_tgt, logits_tgt);
            }
        }

        last_tok = next_last_tok;
        n_rounds++;

        if ((int)gen.size() >= n_gen) break;
    }
    auto t_end = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t_end - t_start).count();

    // Truncate to exactly n_gen output tokens.
    if ((int)gen.size() > n_gen) gen.resize(n_gen);
    double tps = gen.size() / std::max(1e-9, secs);
    double al  = n_rounds > 0 ? (double)(n_matched + n_rounds) / n_rounds : 0.0;  // commits/round

    std::printf("[chain] %zu gen tokens in %.3f s  →  %.2f tok/s\n",
                gen.size(), secs, tps);
    std::printf("[chain] rounds=%d  drafted=%lld  matched=%lld  avg_commit/round=%.2f (N_spec=%d)\n",
                n_rounds, n_drafted, n_matched, al, N_spec);
    if (verify_mode == VERIFY_DDTREE) {
        std::printf("[ddtree] sibling_walks=%lld (%.1f%% of rounds)  sibling_commits=%lld\n",
                    n_ddtree_sibling_walks,
                    n_rounds > 0 ? 100.0 * n_ddtree_sibling_walks / n_rounds : 0.0,
                    n_ddtree_sibling_commits);
    }

    // Emit full output (prompt + gen)
    std::vector<int32_t> all; all.reserve(prompt.size() + gen.size());
    all.insert(all.end(), prompt.begin(), prompt.end());
    all.insert(all.end(), gen.begin(), gen.end());
    write_int32_file(out_path, all);

    step_graph_destroy(sg_tgt); step_graph_destroy(sg_drf);
    free_target_cache(c_tgt); free_target_cache(c_drf);
    free_target_weights(w_tgt); free_target_weights(w_drf);
    ggml_backend_free(backend);
    return 0;
}
