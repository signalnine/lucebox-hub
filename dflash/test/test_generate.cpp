// End-to-end generation test for our qwen35 target forward.
//
// Reads a binary int32 token file (produced by scripts/tokenize_prompt.py),
// runs single-token decode over every token (no batched prefill), generates
// N new tokens via greedy argmax, and writes the resulting int32 token stream
// to an output file for Python-side detokenization.
//
// Also reports decode tok/s (generation only, prompt steps excluded).
//
// Usage:
//   test_generate <qwen35.gguf> <prompt_ids.bin> <n_gen> <out_ids.bin>

#include "dflash27b.h"
#include "internal.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <unistd.h>
#include <vector>

using namespace dflash27b;

struct StepGraph {
    ggml_context *    ctx = nullptr;
    ggml_cgraph *     gf  = nullptr;
    ggml_gallocr_t    alloc = nullptr;
    ggml_tensor *     inp_embed  = nullptr;
    ggml_tensor *     positions  = nullptr;
    ggml_tensor *     kv_pos_idx = nullptr;   // i64 [n_tokens] row indices for KV set_rows
    ggml_tensor *     attn_mask  = nullptr;   // f16 [n_kv_padded, n_tokens_padded]
    ggml_tensor *     logits     = nullptr;
    int               n_kv_padded = 0;
};

// Build a fresh single-token forward graph. We rebuild per step so that
// `kv_start` updates drive the correct KV cache slot. The graph is cheap to
// rebuild — all the weights + KV cache stay persistent.
// Build a per-step forward graph. Three persistent resources survive across
// steps, which matters for decode throughput:
//
//   1. ggml_context — `ggml_reset(ctx)` moves the arena bump pointer back
//      to 0 without freeing the underlying memory. After reset, the next
//      ggml_new_tensor_* calls land at the SAME addresses as last step, so
//      cgraph->nodes[0] is pointer-stable across steps. ggml-cuda keys its
//      captured CUDA graph on nodes[0] — pointer-stable keys are what let
//      it call cudaGraphExecUpdate on subsequent steps instead of re-
//      capturing. With GGML_CUDA_GRAPHS=ON this is the big win.
//   2. ggml_gallocr — skips cudaMalloc/cudaFree of the multi-GB working
//      set every step.
//   3. CUDA graph cache (inside ggml-cuda) — keyed by nodes[0]; reused
//      only because of (1).
static bool build_step_graph(
    StepGraph & sg,
    const TargetWeights & w,
    TargetCache & cache,
    ggml_backend_t backend,
    int kv_start,
    int n_tokens = 1
) {
    if (!sg.ctx) {
        ggml_init_params ip{};
        ip.mem_size   = 512 * 1024 * 1024;   // larger arena — batched prefill needs it
        ip.mem_buffer = nullptr;
        ip.no_alloc   = true;
        sg.ctx = ggml_init(ip);
        if (!sg.ctx) return false;
    } else {
        ggml_reset(sg.ctx);
    }

    const int hidden = w.n_embd;
    sg.inp_embed = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, hidden, n_tokens, 1);
    sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, 4 * n_tokens);
    ggml_set_input(sg.inp_embed);
    ggml_set_input(sg.positions);

    // KV write indices: one i64 per new token, values kv_start..kv_start+n_tokens-1.
    sg.kv_pos_idx = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I64, n_tokens);
    ggml_set_input(sg.kv_pos_idx);

    // Pad n_kv so the read view's shape stays stable across consecutive
    // steps (ggml-cuda's CUDA-graph cache is invalidated on any property
    // change). 256-aligned pad matches llama.cpp's get_n_kv policy. The
    // attention mask matches: unused positions filled with -inf.
    constexpr int KV_PAD_ALIGN = 256;
    const int kv_len     = kv_start + n_tokens;
    const int n_kv_padded = ((kv_len + KV_PAD_ALIGN - 1) / KV_PAD_ALIGN) * KV_PAD_ALIGN;
    // q_pad must cover n_tokens AND align to fattn's 32-row expectation.
    const int q_pad      = ((n_tokens + 31) / 32) * 32;
    sg.attn_mask   = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F16, n_kv_padded, q_pad);
    ggml_set_input(sg.attn_mask);
    sg.n_kv_padded = n_kv_padded;

    sg.gf = ggml_new_graph_custom(sg.ctx, 8192, false);

    QwenGraphInputs gi{};
    gi.inp_embed      = sg.inp_embed;
    gi.positions      = sg.positions;
    gi.attn_mask      = sg.attn_mask;
    gi.kv_pos_idx     = sg.kv_pos_idx;
    gi.n_kv_padded    = n_kv_padded;
    gi.n_tokens       = n_tokens;
    gi.kv_start       = kv_start;
    gi.capture_layers = false;

    QwenGraphOutputs go = build_target_graph(sg.ctx, sg.gf, w, cache, gi);
    if (!go.logits) return false;
    ggml_set_output(go.logits);
    ggml_build_forward_expand(sg.gf, go.logits);
    sg.logits = go.logits;

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    }
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}

static std::vector<int32_t> read_int32_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    auto sz = (size_t)f.tellg();
    f.seekg(0);
    std::vector<int32_t> out(sz / sizeof(int32_t));
    f.read((char *)out.data(), sz);
    return out;
}

static bool write_int32_file(const std::string & path, const std::vector<int32_t> & v) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write((const char *)v.data(), v.size() * sizeof(int32_t));
    return (bool)f;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::fprintf(stderr,
            "usage: %s <qwen35.gguf> <prompt_ids.bin> <n_gen> <out_ids.bin>\n", argv[0]);
        return 2;
    }
    const char * gguf_path   = argv[1];
    const char * prompt_path = argv[2];
    const int    n_gen       = std::atoi(argv[3]);
    const char * out_path    = argv[4];
    int stream_fd = -1;
    for (int i = 5; i < argc; i++) {
        if (std::strncmp(argv[i], "--stream-fd=", 12) == 0) {
            stream_fd = std::atoi(argv[i] + 12);
        }
    }
    auto stream_emit = [&](int32_t tok) {
        if (stream_fd < 0) return;
        int32_t v = tok;
        ssize_t n = ::write(stream_fd, &v, sizeof(v));
        (void)n;
    };

    // ── Load model and cache
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "cuda init failed\n"); return 1; }

    TargetWeights w;
    if (!load_target_gguf(gguf_path, backend, w)) {
        std::fprintf(stderr, "load: %s\n", dflash27b_last_error());
        return 1;
    }
    std::printf("[target] %s\n", dflash27b_last_error());

    const int max_ctx = 4096;
    TargetCache cache;
    if (!create_target_cache(w, max_ctx, /*max_verify_tokens=*/0, backend, cache)) {
        std::fprintf(stderr, "cache: %s\n", dflash27b_last_error());
        return 1;
    }

    auto prompt = read_int32_file(prompt_path);
    if (prompt.empty()) { std::fprintf(stderr, "empty prompt bin\n"); return 1; }
    std::printf("[prompt] %zu tokens: ", prompt.size());
    for (auto t : prompt) std::printf("%d ", t);
    std::printf("\n");

    if ((int)prompt.size() + n_gen > max_ctx) {
        std::fprintf(stderr, "prompt+gen exceeds max_ctx\n");
        return 1;
    }

    std::vector<int32_t> all_tokens = prompt;
    all_tokens.reserve(prompt.size() + n_gen);

    const int hidden = w.n_embd;
    std::vector<float> embed_buf(hidden);

    StepGraph sg;

    // Per-step timing accumulators (enabled with DFLASH27B_TTX=1).
    const bool tt_enable = [] {
        const char * s = std::getenv("DFLASH27B_TTX");
        return s && std::atoi(s) != 0;
    }();
    double tt_build = 0, tt_embed = 0, tt_set = 0, tt_compute = 0, tt_get = 0, tt_argmax = 0;
    long long tt_steps = 0;
    auto now_us = []() {
        return std::chrono::steady_clock::now();
    };

    // ── Helper: run one step given current token + absolute position
    auto run_step = [&](int32_t tok, int pos) -> int32_t {
        auto t0 = now_us();
        if (!build_step_graph(sg, w, cache, backend, pos)) {
            std::fprintf(stderr, "build_step_graph failed at pos=%d\n", pos);
            std::exit(1);
        }
        auto t1 = now_us();

        // CPU embed
        int32_t ids[1] = { tok };
        if (!w.embedder.embed(ids, 1, embed_buf.data())) {
            std::fprintf(stderr, "embed failed tok=%d\n", tok);
            std::exit(1);
        }
        auto t2 = now_us();
        ggml_backend_tensor_set(sg.inp_embed, embed_buf.data(), 0,
                                sizeof(float) * embed_buf.size());

        // M-RoPE positions: 4 copies of pos
        int32_t p4[4] = { pos, pos, pos, pos };
        ggml_backend_tensor_set(sg.positions, p4, 0, sizeof(int32_t) * 4);

        // KV write index: single row at absolute position.
        int64_t pos64 = pos;
        ggml_backend_tensor_set(sg.kv_pos_idx, &pos64, 0, sizeof(int64_t));

        // Attention mask. Causal over kv_len, -inf past. The n_tokens=1 decode
        // only uses q row 0; pad rows (1..q_pad-1) get the same row-0 pattern
        // so fattn never does softmax over an all-(-inf) row (NaN guard).
        const int kv_len = pos + 1;
        std::vector<uint16_t> mask((size_t)sg.n_kv_padded * 32, /*F16 -inf*/ 0xFC00);
        for (int q = 0; q < 32; q++) {
            for (int k = 0; k < kv_len && k < sg.n_kv_padded; k++) {
                mask[q * sg.n_kv_padded + k] = /*F16 zero*/ 0x0000;
            }
        }
        ggml_backend_tensor_set(sg.attn_mask, mask.data(), 0,
                                sizeof(uint16_t) * mask.size());
        auto t3 = now_us();

        auto st = ggml_backend_graph_compute(backend, sg.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "compute failed at pos=%d (%d)\n", pos, (int)st);
            std::exit(1);
        }
        auto t4 = now_us();

        // argmax on logits
        const int vocab = (int)w.embedder.n_vocab;
        std::vector<float> logits(vocab);
        ggml_backend_tensor_get(sg.logits, logits.data(), 0, sizeof(float) * vocab);
        auto t5 = now_us();
        int best = 0;
        float bv = logits[0];
        for (int i = 1; i < vocab; i++) {
            if (logits[i] > bv) { bv = logits[i]; best = i; }
        }
        auto t6 = now_us();
        if (tt_enable) {
            tt_build   += std::chrono::duration<double, std::micro>(t1 - t0).count();
            tt_embed   += std::chrono::duration<double, std::micro>(t2 - t1).count();
            tt_set     += std::chrono::duration<double, std::micro>(t3 - t2).count();
            tt_compute += std::chrono::duration<double, std::micro>(t4 - t3).count();
            tt_get     += std::chrono::duration<double, std::micro>(t5 - t4).count();
            tt_argmax  += std::chrono::duration<double, std::micro>(t6 - t5).count();
            tt_steps++;
        }
        return best;
    };

    // ── Batched prefill: process prompt in chunks of PREFILL_CHUNK so the
    //    target sees each chunk as a single multi-token forward. At N=256
    //    the forward amortises weight-load + quantize_q8_1 over 256 tokens,
    //    giving ~20x speedup vs the old N=1-per-prompt-token path.
    //    Only the LAST chunk's last position's logits matter (they seed the
    //    decode loop); logits from earlier chunks are just written-then-
    //    discarded.
    auto run_prefill_chunk = [&](const int32_t * toks, int start_pos, int chunk_n) -> int32_t {
        if (!build_step_graph(sg, w, cache, backend, start_pos, chunk_n)) {
            std::fprintf(stderr, "build_step_graph failed prefill pos=%d n=%d\n", start_pos, chunk_n);
            std::exit(1);
        }

        // CPU embed: chunk_n tokens laid out contiguously.
        std::vector<float> chunk_embed((size_t)hidden * chunk_n);
        if (!w.embedder.embed(toks, chunk_n, chunk_embed.data())) {
            std::fprintf(stderr, "prefill embed failed\n"); std::exit(1);
        }
        ggml_backend_tensor_set(sg.inp_embed, chunk_embed.data(), 0,
                                sizeof(float) * chunk_embed.size());

        // M-RoPE axis-major positions: [axis, token], first 3 axes = abs pos,
        // axis 3 = 0 for plain text.
        std::vector<int32_t> pos4((size_t)4 * chunk_n);
        for (int i = 0; i < chunk_n; i++) {
            const int p = start_pos + i;
            pos4[0 * chunk_n + i] = p;
            pos4[1 * chunk_n + i] = p;
            pos4[2 * chunk_n + i] = p;
            pos4[3 * chunk_n + i] = 0;
        }
        ggml_backend_tensor_set(sg.positions, pos4.data(), 0,
                                sizeof(int32_t) * pos4.size());

        // KV write indices: sequential slots start_pos..start_pos+chunk_n-1.
        std::vector<int64_t> kv_idxs(chunk_n);
        for (int i = 0; i < chunk_n; i++) kv_idxs[i] = (int64_t)(start_pos + i);
        ggml_backend_tensor_set(sg.kv_pos_idx, kv_idxs.data(), 0,
                                sizeof(int64_t) * kv_idxs.size());

        // Causal mask over the padded n_kv range. q_pad = align_up(chunk_n, 32);
        // pad rows copy the last real row (NaN guard) — same convention as
        // test_chain_spec.
        const int kv_len = start_pos + chunk_n;
        const int q_pad  = ((chunk_n + 31) / 32) * 32;
        std::vector<uint16_t> mask((size_t)sg.n_kv_padded * q_pad, /*F16 -inf*/ 0xFC00);
        for (int q = 0; q < chunk_n; q++) {
            const int max_k = start_pos + q;
            for (int k = 0; k <= max_k && k < sg.n_kv_padded; k++) {
                mask[(size_t)q * sg.n_kv_padded + k] = /*F16 zero*/ 0x0000;
            }
        }
        for (int q = chunk_n; q < q_pad; q++) {
            std::memcpy(&mask[(size_t)q * sg.n_kv_padded],
                        &mask[(size_t)(chunk_n - 1) * sg.n_kv_padded],
                        sizeof(uint16_t) * sg.n_kv_padded);
        }
        ggml_backend_tensor_set(sg.attn_mask, mask.data(), 0,
                                sizeof(uint16_t) * mask.size());

        auto st = ggml_backend_graph_compute(backend, sg.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "prefill compute failed (%d)\n", (int)st); std::exit(1);
        }

        // Only need the LAST position's logits (to seed decode). Logits
        // tensor shape is [vocab, chunk_n]; last row offset = (chunk_n-1)*vocab*4.
        const int vocab = (int)w.embedder.n_vocab;
        std::vector<float> last_logits(vocab);
        const size_t row_off = (size_t)(chunk_n - 1) * vocab * sizeof(float);
        ggml_backend_tensor_get(sg.logits, last_logits.data(), row_off,
                                sizeof(float) * vocab);
        int best = 0;
        float bv = last_logits[0];
        for (int i = 1; i < vocab; i++) {
            if (last_logits[i] > bv) { bv = last_logits[i]; best = i; }
        }
        return best;
    };

    constexpr int PREFILL_CHUNK = 1024;
    int next = -1;
    auto t_prefill_start = now_us();
    for (int start = 0; start < (int)prompt.size(); start += PREFILL_CHUNK) {
        const int chunk_n = std::min(PREFILL_CHUNK, (int)prompt.size() - start);
        next = run_prefill_chunk(prompt.data() + start, start, chunk_n);
    }
    auto t_prefill_end = now_us();
    const double prefill_ms = std::chrono::duration<double, std::milli>(
        t_prefill_end - t_prefill_start).count();
    std::printf("[prefill] %zu tokens in %.3f s → %.2f tok/s; last-token argmax=%d\n",
                prompt.size(), prefill_ms / 1000.0,
                prompt.size() * 1000.0 / std::max(1e-9, prefill_ms),
                next);

    // ── Generation loop
    auto t_start = std::chrono::steady_clock::now();
    int gen_start_pos = (int)prompt.size();
    for (int g = 0; g < n_gen; g++) {
        int32_t tok = next;
        all_tokens.push_back(tok);
        stream_emit(tok);
        next = run_step(tok, gen_start_pos + g);
    }
    auto t_end = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t_end - t_start).count();

    if (tt_enable && tt_steps > 0) {
        const double inv = 1.0 / (double)tt_steps;
        std::printf("[ttx] per-step us: build=%.0f embed=%.0f set=%.0f compute=%.0f get=%.0f argmax=%.0f total=%.0f\n",
                    tt_build * inv, tt_embed * inv, tt_set * inv,
                    tt_compute * inv, tt_get * inv, tt_argmax * inv,
                    (tt_build+tt_embed+tt_set+tt_compute+tt_get+tt_argmax) * inv);
    }
    double tps  = n_gen / std::max(1e-9, secs);

    // Also push the final next token so downstream sees it
    all_tokens.push_back(next);

    std::printf("[gen] %d new tokens in %.3f s  →  %.2f tok/s\n", n_gen, secs, tps);
    std::printf("[gen] tokens: ");
    for (int i = 0; i < n_gen; i++) std::printf("%d ", all_tokens[prompt.size() + i]);
    std::printf("\n");

    write_int32_file(out_path, all_tokens);
    std::printf("[out] wrote %zu tokens to %s\n", all_tokens.size(), out_path);

    if (sg.alloc) ggml_gallocr_free(sg.alloc);
    if (sg.ctx)   ggml_free(sg.ctx);
    free_target_cache(cache);
    free_target_weights(w);
    ggml_backend_free(backend);
    return 0;
}
