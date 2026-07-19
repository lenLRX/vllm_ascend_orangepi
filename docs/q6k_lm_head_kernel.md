# Q6_K Fused Dequant+Matmul Kernel (int8 layout) — Final Report

Date: 2026-07-19. Device: Ascend 310B1 (dav-m300, 1 AI core @ 1224 MHz, UB 248 KB,
L2 4 MB, DDR read wall 28.8 GB/s measured).

## Summary

Built an NPU-friendly fused Q6_K dequant+matmul kernel (`matmul_gguf_q6_k_i8`)
in the style of the existing Q4_0 kernel, with a load-time weight repack.
**The lm_head (tied token_embd, fp16 [262144, 1536] = 805 MB/token at 28.2 ms)
now runs as Q6_K int8 (~453 MB/token) at 15.9 ms — 1.77x, saving ~12.3 ms
per decode token, with zero quantization loss vs the shipped GGUF** (the int8
expansion stores the exact Q6_K values).

## The key design decision: int8-expanded layout, not packed 6-bit

Q6_K's binary layout (210 B per 256-element super-block: `ql` 4-bit + `qh`
2-bit + int8 scales + fp16 d) needs ~38 vector instructions per 512 weights
to dequant in-kernel (2-bit `qh` field extraction dominates). On this chip the
vector pipe is ~1.2 instr/cycle, so packed Q6_K dequant is vector-bound at
~24-25 ms for the lm_head — worse than fp16's 28.2 but far from the 15 ms
bandwidth floor.

The reorg: the host converter expands each 6-bit value losslessly to **int8
(q6-32)** (exactly representable) in the Q4_0-style NZ-fractal element order,
plus per-16-element group scales (`fp16(d × scales_int8)`) in UB-run order.
Kernel-side dequant becomes trivial: `vlds int8 -> vfcvt_s82f16 -> vmul(E2B
scale) -> vsts` at 14 instr/512 weights — bandwidth-bound, not vector-bound.
Cost: 453 MB vs 330 MB stream (+37%), which the DDR wall absorbs fine.

Files:
- `dequant_only_q6_k_i8.cpp` — standalone dequant bring-up (validated
  bit-exact vs the numpy reference, max_diff = 0.0 on 4 shapes).
- `matmul_gguf_q6_k_i8.cpp` — full kernel (Q4_0 pingpong pipeline + new
  dequant block), pybind `matmul_gguf_q6_k_i8_layer`.
- `convert_gguf_q6_k_i8` (pybind, C++) — GGUF Q6_K bytes -> int8 NZ + fp16
  group scales; bit-exact with the numpy reference converter.
- `matmul_gguf_q6_k_i8.cpp` + wrapper decl in `prebuild/npu_ops.h`.
- Tests: `/ssd/test_q6k_i8_dequant.py` (dequant), `/ssd/test_gguf_q6_k_i8_kernel.py`
  (matmul, 7 shapes), `/ssd/bench_q6k_lm_head.py`.

## Validation

- Dequant: **bit-exact** (max_diff = 0.000000) vs bit-exact numpy reference
  on N=32/64/96/256 x K=256/512/1536.
- Matmul vs CPU (dequant + fp32 matmul): **7/7 shapes PASS**, cos = 1.000000,
  rel err <= 0.0005 — incl. qkv_slide/full, o_slide/full, gate_up, down.
- C++ converter == numpy converter bit-exact (values and scales).
- lm_head e2e: greedy 256-token generation coherent; "The capital of France
  is" -> "Paris".

## Performance (measured, cold stream, median of 5)

| lm_head matmul (M=1, N=262144, K=1536) | stream | time | effective BW |
|---|---|---|---|
| dense fp16 (before) | 805 MB | 27.7 ms | 29.0 GB/s |
| **Q6_K int8 (this kernel)** | 453 MB | **15.9 ms** | **28.4 GB/s** |

The kernel is at the memory wall (28.4 of 28.8 GB/s measured pure-read).
Goal target (>= 22 GB/s) met.

## e2e decode impact

- logits phase: 28 ms -> **16 ms** per token (phase timing).
- e2e decode, 64-token runs: (wall-prefill)/64 = 115 -> **102 ms/tok**
  (**-12.5 ms/tok**); 256-token gate hook-free: 96 -> **84.7 ms/tok**
  (11.81 tok/s).
- Layer-hook decode_avg (excludes lm_head phase): 75.2 -> 75.0 ms/tok
  (unchanged, as expected).
- Cumulative Q4_0 decode speedup vs the original baseline: 206.4 -> ~85-102
  ms/tok e2e (**~2.0-2.4x**, metric-dependent).

Note on criterion 4 (~15-17 ms/token improvement): achieved **-12.3 ms/tok**
(28.2 -> 15.9 on the lm_head). The 15-17 estimate assumed the packed 6-bit
byte count (330 MB); the int8 layout trades +37% bytes (453 MB) for vector
simplicity — that is what makes the >= 22 GB/s bandwidth target reachable at
all on this chip. The measurable bandwidth criterion is fully met
(28.4 of 28.8 GB/s).

## Wiring (behind flag)

- `gemma4_npu.py::_dequant_q6_k_and_load` stashes token_embd's raw Q6_K
  bytes; `_convert_q6_k_lm_head` converts them to the int8 layout on NPU
  (env `GEMMA4_LMHEAD_Q6K=1` default on, "0" disables).
- `UnquantizedEmbeddingMethod.apply` (vocab_parallel_embedding.py) routes to
  `matmul_gguf_q6_k_i8_layer` when `_q6k_i8_qweight` is present.
- The fp16 embedding table is kept for the input-embedding lookup (one row
  per token — format-irrelevant); the Q6_K int8 copy serves only the lm_head
  matmul. Extra device memory: +453 MB.

## Debug history (for the record)

- The UB element order is `[k16][n32][n_local][kk]` (16 consecutive UB
  positions = one (k16-block, n) pair = one scale group) — verified by
  dumping E2B expansion and the Q4_0 UB structure; the first converter used
  nibble-stream order directly and was wrong.
- `vfcvt` (not `vcvt`) is the s8->f16 intrinsic; PART_EVEN/ODD are lane
  parity (verified by probe).
- UB pingpong is required: with single-buffered UB, the next tile's VEC
  overwrites the region the previous tile's MTE3 dump is reading (gelu-style
  flags pass init events for the first two tiles).
