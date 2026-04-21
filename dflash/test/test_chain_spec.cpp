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

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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
    ggml_tensor *     inp_embed = nullptr;
    ggml_tensor *     positions = nullptr;
    ggml_tensor *     attn_mask = nullptr;
    ggml_tensor *     logits    = nullptr;
    int               n_tokens  = 0;
};
static void step_graph_free(StepGraph & sg) {
    if (sg.alloc) { ggml_gallocr_free(sg.alloc); sg.alloc = nullptr; }
    if (sg.ctx)   { ggml_free(sg.ctx); sg.ctx = nullptr; }
    sg.gf = nullptr; sg.inp_embed = nullptr;
    sg.positions = nullptr; sg.attn_mask = nullptr;
    sg.logits = nullptr; sg.n_tokens = 0;
}

// Flash-attn mask alignment (KQ_MASK_PAD in test_dflash.cpp). Our pinned
// fattn kernel uses 32; TBQ path would bump this to 256.
static constexpr int KQ_MASK_PAD     = 32;
static constexpr int KQ_STRIDE_PAD   = 32;
static constexpr uint16_t F16_ZERO    = 0x0000;
static constexpr uint16_t F16_NEG_INF = 0xFC00;

static int align_up(int x, int a) { return ((x + a - 1) / a) * a; }

static void build_causal_mask_f16(
    std::vector<uint16_t> & out, int kv_len, int n_tokens, int kv_start)
{
    const int kv_pad = align_up(kv_len,  KQ_STRIDE_PAD);
    const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);
    out.assign((size_t)kv_pad * q_pad, F16_NEG_INF);
    for (int q = 0; q < n_tokens; q++) {
        const int max_k = kv_start + q;
        for (int k = 0; k <= max_k && k < kv_len; k++) {
            out[(size_t)q * kv_pad + k] = F16_ZERO;
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
    step_graph_free(sg);
    ggml_init_params ip{};
    ip.mem_size = 512 * 1024 * 1024; ip.no_alloc = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    const int hidden = w.n_embd;
    sg.inp_embed = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, hidden, n_tokens, 1);
    sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, 4 * n_tokens);
    ggml_set_input(sg.inp_embed);
    ggml_set_input(sg.positions);

    if (n_tokens > 1) {
        const int kv_len = kv_start + n_tokens;
        const int kv_pad = align_up(kv_len, KQ_STRIDE_PAD);
        const int q_pad  = align_up(n_tokens, KQ_MASK_PAD);
        sg.attn_mask = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F16, kv_pad, q_pad);
        ggml_set_input(sg.attn_mask);
    }

    sg.gf = ggml_new_graph_custom(sg.ctx, 16384, false);

    QwenGraphInputs gi{};
    gi.inp_embed = sg.inp_embed;
    gi.positions = sg.positions;
    gi.attn_mask = sg.attn_mask;   // nullptr for n_tokens==1
    gi.n_tokens  = n_tokens;
    gi.kv_start  = kv_start;
    gi.capture_layers = false;

    QwenGraphOutputs go = build_target_graph(sg.ctx, sg.gf, w, cache, gi);
    if (!go.logits) return false;
    ggml_set_output(go.logits);
    ggml_build_forward_expand(sg.gf, go.logits);
    sg.logits   = go.logits;
    sg.n_tokens = n_tokens;

    sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
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
    std::vector<float> & logits_buf)
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
    if (ggml_backend_graph_compute(backend, sg.gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "compute failed at pos=%d\n", pos); std::exit(1);
    }
    const int vocab = (int)w.embedder.n_vocab;
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

    // Causal mask
    const int kv_len = pos + n_tokens;
    build_causal_mask_f16(mask_buf, kv_len, n_tokens, pos);
    ggml_backend_tensor_set(sg.attn_mask, mask_buf.data(), 0,
                            sizeof(uint16_t) * mask_buf.size());

    if (ggml_backend_graph_compute(backend, sg.gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "batch compute failed at pos=%d n=%d\n", pos, n_tokens); std::exit(1);
    }

    // Debug: logits shape once
    static int dumped = 0;
    if (!dumped) {
        dumped = 1;
        std::printf("[debug] verify logits ne=[%ld,%ld,%ld,%ld]\n",
                    (long)sg.logits->ne[0], (long)sg.logits->ne[1],
                    (long)sg.logits->ne[2], (long)sg.logits->ne[3]);
    }

    // Logits shape: [vocab, n_tokens]. Extract per-position argmax.
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
    if (!create_target_cache(w_tgt, max_ctx, 0, backend, c_tgt)) {
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
        return step_model(w_tgt, c_tgt, backend, sg_tgt, tok, pos, embed_tgt, logits_tgt);
    };
    auto step_drf = [&](int32_t tok, int pos) {
        return step_model(w_drf, c_drf, backend, sg_drf, tok, pos, embed_drf, logits_drf);
    };

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

    auto t_start = std::chrono::steady_clock::now();
    while ((int)gen.size() < n_gen) {
        const int pre_pos = c_tgt.cur_pos;
        if (c_drf.cur_pos != pre_pos) {
            std::fprintf(stderr, "cache desync: drf=%d tgt=%d\n", c_drf.cur_pos, pre_pos);
            return 1;
        }

        // Snapshot draft SSM state pre-round. Target uses short-circuit
        // sequential verify so no target snapshot is needed below.
        snapshot_ssm_state(c_drf);

        // 1. Draft: N sequential forwards starting from last_tok.
        int32_t carry = last_tok;
        for (int i = 0; i < N_spec; i++) {
            drafts[i] = step_drf(carry, pre_pos + i);
            c_drf.cur_pos = pre_pos + i + 1;
            carry = drafts[i];
        }
        n_drafted += N_spec;

        // 2. Target sequential verify with short-circuit on first mismatch.
        //    Batched verify (one N-token forward) would be the real win here
        //    and is drafted in verify_batch() below, but there is an
        //    outstanding correctness bug: when called with target weights,
        //    verify_batch produces logits that match the draft's output
        //    instead of the target's. Single-step target_step at the same
        //    (last_tok, pos) gives the correct argmax, so the single-token
        //    path works. Needs a debug session with a logit-diff check on
        //    position 0 of the N-token forward against the single-step logit.
        int k = 0;
        int32_t t_pred = -1;
        carry = last_tok;
        for (int i = 0; i < N_spec; i++) {
            t_pred = step_tgt(carry, pre_pos + i);
            c_tgt.cur_pos = pre_pos + i + 1;
            if (t_pred == drafts[i]) { k++; carry = drafts[i]; }
            else                    { break; }
        }
        n_matched += k;

        const int commit_count = std::min(k + 1, N_spec);

        // Emit committed tokens: last_tok (always) + drafts[0..commit_count-2].
        gen.push_back(last_tok);
        for (int i = 0; i + 1 < commit_count; i++) gen.push_back(drafts[i]);

        int32_t next_last_tok = (k < N_spec) ? t_pred : drafts[N_spec - 1];

        // 3. Target cache is already at pre_pos + commit_count (short-circuit
        //    stopped feeding further). No target rollback needed.
        if (c_tgt.cur_pos != pre_pos + commit_count) {
            std::fprintf(stderr, "tgt desync: %d vs %d\n", c_tgt.cur_pos, pre_pos + commit_count);
            return 1;
        }

        // 4. Draft: restore_ssm_state + commit_count catch-up single forwards.
        restore_ssm_state(c_drf);
        c_drf.cur_pos = pre_pos;
        int32_t cur = last_tok;
        for (int i = 0; i < commit_count; i++) {
            (void)step_drf(cur, pre_pos + i);
            c_drf.cur_pos = pre_pos + i + 1;
            cur = (i + 1 < commit_count) ? drafts[i] : -1;
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

    // Emit full output (prompt + gen)
    std::vector<int32_t> all; all.reserve(prompt.size() + gen.size());
    all.insert(all.end(), prompt.begin(), prompt.end());
    all.insert(all.end(), gen.begin(), gen.end());
    write_int32_file(out_path, all);

    step_graph_free(sg_tgt); step_graph_free(sg_drf);
    free_target_cache(c_tgt); free_target_cache(c_drf);
    free_target_weights(w_tgt); free_target_weights(w_drf);
    ggml_backend_free(backend);
    return 0;
}
