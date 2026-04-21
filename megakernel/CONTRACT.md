# CONTRACT: shard `pf_deltanet_recurrence` along V-dim (issue axp)

Root cause from `02y`: the kernel launches `<<<16 blocks, 512 threads>>>` — one
block per DeltaNet head. On the 5090's 170 SMs that's 9 % occupancy and the
kernel becomes 80 % of prefill time at 1.52 ms per call. Refactor to shard the
128-element V-dim across multiple blocks per head so the kernel reaches 64+
blocks and actually uses the card.

## Required behavior (don't break anything)

- [ ] **Correctness: `bench_pp_tg.py` end-to-end check PASSES.**
      Verify: prefill→decode token sequence equals pure-decode token sequence
      for the prompt `"The capital of France is"`, first 30 tokens.
      Command: `python bench_pp_tg.py` → look for line `PASS: megakernel output matches reference decode path`.

- [ ] **Build succeeds for sm_120 with NUM_BLOCKS=170 BLOCK_SIZE=1024.**
      Verify: `pip install -e .` exits 0, resulting `.so` imports clean after `import torch`.

- [ ] **Decode tg128 does not regress.**
      Verify: `python final_bench.py` reports tg129 ≥ 700 tok/s (post-5090 baseline 715). ±1 % tolerance for noise.

## New behavior (the perf contract)

- [ ] **Prefill pp520 ≥ 25,000 tok/s on 5090** (base case for "meaningful recovery").
      Verify: `python final_bench.py` reports pp520 ≥ 25,000.
      Stretch: ≥ 30,000 (matches llama.cpp's 31,258 on 5090).
      Dream: ≥ 40,000 (exceeds 3090's 37,800 published number).

- [ ] **Block count ≥ 64 for `pf_deltanet_recurrence`.**
      Verify: grep the kernel launch in `prefill.cu`, confirm grid dim > 16.

- [ ] **No register spilling.**
      Verify: `cuobjdump --dump-resource-usage build/.../prefill.o` shows `STACK:0` and `LOCAL:0` for the new kernel.

## Out of scope

- Decode megakernel changes (decode path separate, already tuned).
- Full-attention `pf_causal_attn` kernel (6 % of prefill, diminishing returns).
- cuBLAS GEMM algo selection (already shown unimportant by nsys).
- 3090 performance on this path (we're on the `rtx-5090` branch; Ampere
  can be tuned separately if it regresses).

## Design sketch (informed by nsys + cuobjdump)

Current:
```
pf_deltanet_recurrence<<<16, 512>>>  // 1 block per head, __launch_bounds__(512, 1)
  for t in 0..S-1:                   // serial per-token, unavoidable
    conv1d + SiLU + L2norm  // shared s_q/s_k/s_v on shared mem
    for j in 0..127:                 // V-dim, parallel
      rank-1 state update
```

Proposed:
```
pf_deltanet_recurrence<<<16 * N_V_CHUNKS, BLOCK_SIZE>>>   // N_V_CHUNKS = 4 → 64 blocks
  h = blockIdx.x / N_V_CHUNKS
  v_chunk = blockIdx.x % N_V_CHUNKS
  for t in 0..S-1:
    conv1d + SiLU + L2norm    // REDUNDANT: all V-chunk blocks for the same head do this.
                              //   ~4× scalar work on this phase but it's fast (DN_KEY=128 small).
    for j in [v_chunk * 32, (v_chunk+1) * 32):   // owns V rows v_chunk*32 .. v_chunk*32+31
      rank-1 state update over this stripe
```

Notes:
- N_V_CHUNKS=4 keeps state-per-block at 32 V rows × 128 K cols × 4 B = 16 KB —
  far below shared mem budget. Registers per thread stay similar (~96) because
  each thread still handles CPW*RPL = 8*4 = 32 state entries (unchanged: warps
  per block × CPW must equal V-stripe width; NWARPS shrinks from 16 to 4).
- BLOCK_SIZE must match: NWARPS = BLOCK_SIZE/32 = DN_VAL/CPW. With V-stripe=32
  and CPW=8, NWARPS=4, BLOCK_SIZE=128.
- Conv state `conv_buf` is shared (same head), but the shift semantics
  (`conv_buf[k] = conv_buf[k+1]` ring) require ONE writer per head. Solution:
  only `v_chunk == 0` writes the shift/new sample; other chunks read-only. Needs
  cross-block sync. Since blocks on different SMs can't `__syncthreads()`,
  use `grid.sync()` or keep conv1d in a separate kernel.

Chosen approach: **two kernels**.
1. `pf_deltanet_conv` (new) — 16 blocks, one per head, does conv1d + SiLU + L2norm
   for all S tokens; writes `s_q[S,h,K], s_k[S,h,K], s_v[S,h,V], beta[S,h], decay[S,h]` to global.
2. `pf_deltanet_recurrence_shard` (new) — 64 blocks (16 heads × 4 V-chunks),
   reads precomputed per-token Q/K/V/beta/decay from global, owns a V-stripe of state.

Global memory cost of the intermediate buffer:
- S=520 tokens × 16 heads × (128+128+128) K/V = 520 × 16 × 384 × 4 B = 12.6 MiB FP32
  (or 6.3 MiB if stored BF16). Small.
- Plus beta/decay: S × H × 2 × 4 B = 520 × 16 × 8 = 65 KiB. Trivial.

## Risks

- Redundant conv work: we fan out conv1d results via a new global buffer, so
  NOT redundant across V-chunk blocks (good). But we *add* a global memory
  round-trip that the original kernel avoided by keeping s_q/s_k/s_v in
  shared mem. Needs measurement — on 5090 with 96 MB L2 this should hit cache.
- The `conv_buf` shift (current: each thread updates `conv_buf[c]` ring) must
  be kept serial-per-head. Keep in `pf_deltanet_conv` where there's only
  one writer per head.
- `__launch_bounds__` on the new recurrence kernel: with BLOCK_SIZE=128 and
  ~96 regs per thread, occupancy may allow 4+ blocks per SM. Leave unset
  and let the compiler decide.
