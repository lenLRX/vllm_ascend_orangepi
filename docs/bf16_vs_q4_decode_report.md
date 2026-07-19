# bf16 vs Q4_0 Gemma4-E2B Decode — Kernel-Level Comparison (256-token input)

Date: 2026-07-18. Device: Ascend 310B1 (dav-m300, 1 AI core @ 1224 MHz, L2 4 MB).
Models: bf16 = `/ssd/models/gemma-4-E2B-it` (safetensors, bf16);
Q4_0 = `/ssd/models/gemma-4-E2B-it-qat-q4_0-gguf` (GGUF Q4_0 weights, fp16
activations). Both run the same `gemma4_npu.py` + current NPU kernels.

**Method**: prompt = exactly 256 tokens; warmup; 3×32-token greedy decode
runs, median. Scripts: `/ssd/e2e_256.py` (e2e wall), `/ssd/bench_decode_kernels.py`
(per-kernel micro-bench, cold weight rotation >> 4 MB L2, back-to-back
launches for small kernels). Layer-hook timing syncs per layer and inflates
forward wall ~10–15%; small-kernel per-call times are launch-rate-bound
(~18–27µs host floor) — device-side work of each is only a few µs.

## 1. End-to-end decode (median of 3)

| Metric | Q4_0 | bf16 | bf16 / Q4_0 |
|---|---|---|---|
| Forward decode (layer hooks) | 73.7 ms/tok | 154.1 ms/tok | **2.09x** |
| E2E decode per-token | 111.8 ms/tok | 190.8 ms/tok | **1.71x** |
| E2E rate | 8.95 tok/s | 5.24 tok/s | 1.71x |
| Prefill (256 tok) | 73 ms | 153 ms | 2.10x |
| Prefill rate | 3500 tok/s | 1670 tok/s | 2.10x |

E2E decode = forward + lm_head tail (~28 ms) + engine/sampling (~9–10 ms),
in both models.

## 2. Matmul kernels (device time, cold weights)

Per-token launch counts: 35 layers (28 sliding @ head_dim 256, 7 full @ 512);
PLE on all 35; lm_head = 1. Weight bytes are read from DDR once per token —
this is the decode bottleneck on both models.

### Q4_0 (nibbles + fp16 scales, `matmul_gguf_q4_0` kernel)

| projection | N | K | calls/tok | µs/call | ms/tok | GB/s | MB/tok |
|---|---|---|---|---|---|---|---|
| qkv (fused q\|k\|v) slide | 2560 | 1536 | 28 | 86.3 | 2.42 | 25.6 | 61.9 |
| qkv (fused) full | 5120 | 1536 | 7 | 171.8 | 1.20 | 25.7 | 31.0 |
| o_proj slide | 1536 | 2048 | 28 | 69.5 | 1.95 | 25.4 | 49.5 |
| o_proj full | 1536 | 4096 | 7 | 137.2 | 0.96 | 25.8 | 24.8 |
| gate_up (fused) | 12288 | 1536 | 35 | 437.7 | 15.32 | 24.3 | 371.6 |
| down_proj | 1536 | 6144 | 35 | 200.8 | 7.03 | 26.4 | 185.8 |
| ple_gate | 256 | 1536 | 35 | 22.2 | 0.78 | 9.9 | 7.7 |
| ple_proj | 1536 | 256 | 35 | 33.3 | 1.17 | 6.6 | 7.7 |
| **matmul total** | | | **210** | | **30.82** | ~25.5 | **740** |
| **lm_head (fp16 NZ)** | 262144 | 1536 | 1 | 28145 | **28.15** | 28.6 | 805 |

### bf16 (dense fp16/bf16 NZ, `matmul_nz` kernel)

| projection | N | K | calls/tok | µs/call | ms/tok | GB/s | MB/tok |
|---|---|---|---|---|---|---|---|
| qkv slide | 2560 | 1536 | 28 | 276.8 | 7.75 | 28.4 | 220.1 |
| qkv full | 5120 | 1536 | 7 | 554.2 | 3.88 | 28.4 | 110.1 |
| o_proj slide | 1536 | 2048 | 28 | 219.7 | 6.15 | 28.6 | 176.2 |
| o_proj full | 1536 | 4096 | 7 | 430.3 | 3.01 | 29.2 | 88.1 |
| gate_up | 12288 | 1536 | 35 | 1320.8 | 46.23 | 28.6 | 1321.2 |
| down_proj | 1536 | 6144 | 35 | 647.0 | 22.65 | 29.2 | 660.5 |
| ple_gate | 256 | 1536 | 35 | 32.4 | 1.14 | 24.3 | 27.5 |
| ple_proj | 1536 | 256 | 35 | 32.7 | 1.14 | 24.1 | 27.5 |
| **matmul total** | | | **210** | | **91.95** | ~28.7 | **2631** |
| **lm_head (bf16 NZ)** | 262144 | 1536 | 1 | 28145 | **28.15** | 28.6 | 805 |

Matmul-only comparison: **bf16/Q4_0 = 2.98x time** on 3.56x bytes — the Q4
kernel runs at 24.3–26.4 GB/s vs the dense kernel's 28.4–29.2 GB/s (~86%:
the Q4 kernel also dequantizes on the vector pipe, so it sits a bit further
from the wall). Both are at the DDR roofline: measured pure-read peak is
28.8 GB/s, D2D copy 42.5 GB/s.

## 3. Attention, norms, rope, gelu, and other kernels

Same code path in both models (activations are bf16 vs fp16 — cost is the
same). "Sustained/call" is back-to-back stream timing; everything here is
launch-rate-bound (host ~18–27µs/call) except page attention.

| kernel | calls/tok | µs/call | ms/tok | bytes/tok | GB/s | device-bound? |
|---|---|---|---|---|---|---|
| rmsnorm [1,1536] | 176 | 24.8 | 4.36 | 1.6 MB | — | no (launch) |
| qkv_norm fused (q,k,v) | 15 | 49.2 | 0.74 | 0.2 MB | — | no |
| qkv_norm q-only (KV-shared layers) | 20 | 23.8 | 0.48 | 0.1 MB | — | no |
| rope q+k fused | 15 | 34.1 | 0.51 | 0.2 MB | — | no |
| rope q-only (KV-shared) | 20 | 22.2 | 0.44 | 0.2 MB | — | no |
| split_qkv | 35 | 27.3 | 0.96 | 0.4 MB | — | no |
| add (residual/PLE) | 105 | 22.3 | 2.34 | 1.0 MB | — | no |
| mul_scalar (layer scale) | 35 | 20.6 | 0.72 | 0.3 MB | — | no |
| gated_gelu (batched) | 35 | 20.5 | 0.72 | 1.3 MB | — | no |
| gelu_pytorch_tanh (PLE) | 35 | 21.0 | 0.73 | 0.05 MB | — | no |
| mul (PLE) | 35 | 40.0 | 1.40 | 0.05 MB | — | no |
| ple_slice | 35 | 20.8 | 0.73 | 0.04 MB | — | no |
| kv cache memcpy (non-shared layers) | 30 | 11.6 | 0.35 | 0.03 MB | — | no |
| **page_attn dim256 (slide, ctx 288)** | 28 | 58.5 | 1.64 | 8.3 MB | 5.0 | **yes** |
| **page_attn dim512 (full, ctx 288)** | 7 | 160.8 | 1.13 | 4.1 MB | 3.7 | **yes** |
| **small-kernel subtotal** | ~630 | | **15.7** | | | mostly launch |
| **attention subtotal** | 35 | | **2.76** | 12.4 MB | | |
| sampling (argmax+log_softmax fp32 [1,262144]) | 1 | 150 | 0.15 | 2.1 MB | 14.0 | yes |

## 4. Wall vs. device reconciliation (per token)

| | Q4_0 | bf16 |
|---|---|---|
| matmul device | 30.8 | 92.0 |
| lm_head device | 28.2 | 28.2 |
| attention device | 2.8 | 2.8 |
| small-kernel device (few µs each) | ~2 | ~2 |
| sampling device | 0.2 | 0.2 |
| **device-busy total** | **~64** | **~125** |
| forward wall (measured) | 73.7 | 154.1 |
| forward host gap (wall − device) | ~10 | ~29 |
| E2E wall (measured) | 111.8 | 190.8 |
| E2E − device | ~48 | ~66 |

The host gap is the small-launch Python+enqueue cost that the device cannot
hide (device drains tiny kernels faster than the ~18–27µs/call host rate) plus
the serial lm_head tail and engine bookkeeping (in E2E). Both models pay the
same absolute host cost; it is a smaller *fraction* of the Q4 wall.

## 5. Takeaways

- Decode is memory-bound end to end: per-token weight traffic is 740 MB
  (Q4) + 805 MB (lm_head) vs 2631 MB + 805 MB (bf16). The lm_head (tied
  embedding, kept fp16) is 52% of the Q4 model's total traffic — it is the
  single largest remaining kernel in the Q4 model (28.2 ms, 38% of Q4
  forward device time).
- Q4 matmuls are ~3x faster than bf16 matmuls at ~86% of the dense kernel's
  bandwidth — the gap to the full 3.56x byte ratio is the dequant vector
  work in the Q4 kernel.
- Attention/norm/rope/gelu kernels are tiny and launch-bound; they cost
  ~16 ms/tok of host rate in both models and are what the forward host gap
  is made of.
- Correctness: 256-token greedy output on the evaluation prompt is coherent
  for both models (see `verify_q4_gate4.log`); the filler-prompt emoji
  output in `e2e_256_q4.log` is the model's response to meaningless
  repetitive input, not a regression (kernel unit tests 17/17 pass).
