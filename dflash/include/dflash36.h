// dflash36 — Qwen3.6-35B-A3B-MXFP4_MOE target constants. Lives alongside
// dflash27b.h; both share the same C++ library, same draft infra, same
// test_dflash driver; arch is detected at load time from the GGUF's
// general.architecture field.
//
// Differences from dflash27b (27B, qwen35 hybrid dense):
//   - arch is qwen35moe (MoE FFN, 256 experts top-8, plus shared expert)
//   - smaller hidden (2048 vs 5120), fewer layers (40 vs 64), smaller
//     attention (16/2 heads vs 24/4)
//   - SSM d_inner shrinks to 4096 (vs 6144), dt_rank 32 (vs 48)
//   - weights in MXFP4 (experts) + Q5_K/Q6_K (expert down) + Q8_0 (dense)

#ifndef DFLASH36_H
#define DFLASH36_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ─── Model config (Qwen3.6-35B-A3B target; draft dims live in dflash27b.h) ─

#define DFLASH36_TARGET_HIDDEN              2048
#define DFLASH36_TARGET_LAYERS              40
#define DFLASH36_TARGET_N_HEAD              16
#define DFLASH36_TARGET_N_HEAD_KV           2
#define DFLASH36_TARGET_HEAD_DIM            256
#define DFLASH36_TARGET_FULL_ATTN_INTERVAL  4     // (il+1) % 4 == 0 → attn layer
#define DFLASH36_TARGET_ROPE_DIM            64
#define DFLASH36_TARGET_ROPE_THETA          10000000.0f
#define DFLASH36_TARGET_RMS_EPS             1e-6f
#define DFLASH36_TARGET_VOCAB               248320  // shared with 27B

// MoE FFN
#define DFLASH36_TARGET_N_EXPERT            256
#define DFLASH36_TARGET_N_EXPERT_USED       8
#define DFLASH36_TARGET_FFN_EXPERT          512     // per-expert hidden
#define DFLASH36_TARGET_FFN_SHEXP           512     // shared-expert hidden

// SSM / DeltaNet
#define DFLASH36_TARGET_SSM_D_INNER         4096
#define DFLASH36_TARGET_SSM_D_STATE         128
#define DFLASH36_TARGET_SSM_DT_RANK         32
#define DFLASH36_TARGET_SSM_N_GROUP         16
#define DFLASH36_TARGET_SSM_CONV_KERN       4

// Max context the model supports; bench/serve configure their own max_ctx
// within this.
#define DFLASH36_TARGET_CTX_MAX             262144

#ifdef __cplusplus
}
#endif

#endif // DFLASH36_H
