// Forward pass of Qwen3.6-35B-A3B (qwen35moe arch) in pure ggml. Companion
// to qwen35_target_graph.cpp — same hybrid DeltaNet + Full-attention layer
// dispatch, but with:
//   - smaller hparams (n_embd=2048, n_layer=40, head=16/2, ssm_d_inner=4096,
//     ssm_dt_rank=32; see dflash36.h)
//   - MoE FFN: 256 experts top-8 via ggml_mul_mat_id, MXFP4 expert weights,
//     Q5_K/Q6_K down projections
//   - always-active shared expert (ffn_*_shexp) with sigmoid gate
//   - plain RoPE (not M-RoPE) with dimension_count=64
//
// Status (M1 scaffolding): this file compiles and exposes the symbol so
// the library links cleanly. The actual graph is TODO — will be ported
// from deps/llama.cpp/src/models/qwen35moe.cpp in the next pass.

#include "internal.h"

#include <cstdio>

namespace dflash27b {

QwenGraphOutputs build_qwen36_graph(
    ggml_context *          /* ctx */,
    ggml_cgraph *           /* gf */,
    const TargetWeights &   /* w */,
    TargetCache &           /* cache */,
    const QwenGraphInputs & /* in */)
{
    set_last_error("build_qwen36_graph: qwen35moe graph builder not yet "
                   "implemented (M1-in-progress). See docs/port-qwen36-35b-a3b-moe.md");
    QwenGraphOutputs empty{};
    return empty;
}

} // namespace dflash27b
