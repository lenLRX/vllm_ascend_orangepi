# Gemma4 Q4_0 (GGUF) Decode — Baseline Report

Date: 2026-07-18. Device: Ascend 310B1 (dav-m300, 1 AI core @ 1224 MHz, UB 248 KB,
L1 1 MB, L0A/L0B 64 KB, L0C 128 KB, L2 4 MB). Model:
`/ssd/models/gemma-4-E2B-it-qat-q4_0-gguf/gemma-4-E2B_q4_0-it.gguf` (fp16 activations,
Q4_0 weights via custom `matmul_gguf_q4_0` CCE kernel).

All speedups in this effort are measured against this report: same scripts,
warm kernel cache, median of 3 runs.

## 1. End-to-end decode latency

Script: `/ssd/profile_gemma4_q4.py` (prompt = 464 tokens, greedy, 64 decode
tokens x 3 runs, per-layer hook timing).

| Metric | Value |
|---|---|
| **Decode** | **206.4 ms/tok (4.84 tok/s)** — runs: 206.4 / 205.9 / 206.7 |
| Prefill (464 tok) | 2.42–2.52 s (~190 tok/s) |

Correctness at baseline: output coherent (greedy 64 tokens).

## 2. Decode Q4_0 matmul kernel micro-benchmark (the "4x" metric)

Script: `/ssd/bench_q4_matmul_cold.py` — rotates through >=64 MB of weights per
shape so every iteration streams from DDR, matching real decode (724 MB of
weights per token, no L2 reuse). (`/ssd/bench_q4_matmul.py` is the L2-warm
variant; its numbers are flattered by up to 1.6x on big shapes.)

| shape (M=1) | N | K | layers x shards | cold ms | GB/s |
|---|---|---|---|---|---|
| q_slide | 2048 | 1536 | 28 | 0.0759 | 23.3 |
| kv_slide | 256 | 1536 | 56 | 0.0350 | 6.3 |
| q_full | 4096 | 1536 | 7 | 0.1459 | 24.3 |
| kv_full | 512 | 1536 | 14 | 0.0284 | 15.6 |
| o_slide | 1536 | 2048 | 28 | 0.0754 | 23.5 |
| o_full | 1536 | 4096 | 7 | 0.1462 | 24.2 |
| gate_up | 6144 | 1536 | 70 | 0.2591 | 20.5 |
| down | 1536 | 6144 | 35 | 0.2183 | 24.3 |

**Per-token cold Q4 matmul total: 34.4 ms** (L2-warm: 31.5 ms).

## 3. Measured DDR bandwidth (roofline)

| Probe | Value |
|---|---|
| D2D copy (read+write), 256MB–1GB | 42.5 GB/s |
| Pure read (`sum` over 1 GB fp16) | 28.8 GB/s |

Per-token Q4 weight traffic is **724 MB** (nibbles + fp16 scales for all
linear matmuls; irreducible for the format). Implications:

- Matmul floor at measured read BW: 724 MB / 28.8 GB/s ≈ **25 ms**
  (optimistically ~17–21 ms if the read path reaches copy-level 42 GB/s).
- **"4x on the decode matmul" (34.4 → 8.6 ms) would require ~84 GB/s of
  sustained weight reads — 2–3x the hardware's peak read bandwidth.** As
  written, that sub-target is not physically reachable on this chip for the
  aggregate. The reachable kernel win is bringing every shape to the BW wall:
  ~34.4 → ~21–25 ms (1.4–1.6x). We will pursue that and report honestly
  against the stop rule.

## 4. Host-vs-device split (the dominant lever)

Evidence:
- CANN task_time trace (8 decode steps): device busy ≈ **25% of wall**;
  ~2475 AICORE kernel launches per token in the profiled window.
- Launch-rate micro-test (`/tmp/launch_rate_test.py`): tiny kernel
  (rmsnorm) host enqueue **18 µs/launch**, device keeps up (backlog ~1.7 µs);
  q4 matmul enqueue 27 µs vs 254 µs device (device-bound).
- Launch census (`/ssd/count_q4_launches.py`, wrappers on all py_npu_ops):
  ~210 Q4 matmul calls + ~700–900 other launches per token; per-launch
  Python+pybind+acl cost 18–27 µs plus torch-level glue.

Estimated decode composition per token: device ≈ 45–55 ms (matmul 34.4 +
attention/norms/gelu/rope/etc.) vs wall 206 ms → **~150 ms (~75%) is
host-side**.

Structural host-time sources found (in impact order):

1. **35x full `torch.npu.synchronize()` per token** — one per decoder layer
   (`gemma4_npu.py:573-574`), there only to retain per-layer
   `page_table_npu` tensors until async kernels finish. Kills all overlap.
2. **35x blocking H2D `torch.tensor(page_table_list, device='npu')`** per
   token (one per layer), though the table is identical for all layers in a
   step and only grows every 64 tokens.
3. **Shard-matmul fan-out**: qkv as 3 matmul calls + `torch.cat`, gate_up as
   2 calls + cat (`linear.py:155-225`), plus per-call `torch.empty`,
   `reshape().contiguous()`, and a per-call `from ... import` inside
   `apply()`. GGUF raw shard bytes concatenate losslessly (block order is
   row-major over N), so one fused matmul per projection is possible.
4. **~300 `torch.empty` per token** (decode shapes are static at M=1 — can be
   pre-allocated once).
5. **Prefill-only**: `gated_gelu_layer` called per token in a Python loop
   (`gemma4_npu.py:245-248`) — 464 calls/layer at 464-token prefill
   (~16K launches). Decode impact is only 35 calls; fix separately.

## 5. Targets

- **T2 (host hiding)**: wall ≤ ~1.1x device-busy → decode ≈ 55–65 ms/tok
  (3.2–3.7x e2e). Primary work items 1–4 above.
- **T1 (matmul)**: drive shapes to the BW wall: ≥1.4x aggregate
  (34.4 → ≤25 ms); report vs the 4x stop rule.
- **T3 (e2e)**: falls out of T1+T2: 206 → ~45–55 ms/tok expected.

Correctness gate after every change: greedy 256-token generation coherent
and numerically close to the pre-change Q4_0 model.

---

## 6. Progress log (updated as the work lands)

| Round | Change | Decode ms/tok | Notes |
|---|---|---|---|
| baseline | — | 206.4 | report above |
| r1 | kill 35x/step full syncs; page-table content cache; qkv/gate_up shard fusion at load (GGUF raw concat, 245→140 matmul launches, no torch.cat) | 168.0 | output identical |
| r2 | `get_default_stream()` cached (was ~3000 calls/tok × ~120µs through torch._utils device-index resolution) | 95.7 | prefill 2.5→0.89s |
| r3 | matmul kernel k_tile 256→128 (n_tile 128→256, half the FIX drains); batched gated_gelu (m rows/launch); on-device greedy sampling (argmax on NPU, no full-logits D2H, no CPU softmax) | 89.9 | kernel unit tests 17/17 pass |
| r4 | preseed `_qweight_nz*` attrs on all linears (cheap `hasattr` in apply) | 87.9 | |
| r5 | buffer pooling: attention (10 bufs/layer), Q4 matmul outputs, RMSNorm, gelu | 78.7 | |
| r6 | + decoder-layer residuals & PLE & ple_slice pooling | 75.2 | |
| r7 | batched rope (q+k), fused qkv_norm (one call for q,k,v) | 75.2 | final state; see `q4_decode_optimization_final.md` |

**Final: 96 ms/tok e2e decode hook-free (10.42 tok/s, 256-token run), ~2.6x
vs the ~250 ms/tok hook-free baseline; 75.2 ms/tok forward (2.74x). Matmul
34.4 → 28.8 ms (1.19x, at the memory wall).** Note: every new process pays a
~50 s one-time TBE init on its first greedy step (argmax/log_softmax) — warm
up with a 2-token generate before timing, or un-warmed runs mislead
(~3.3 tok/s artifact).

Matmul kernel (fused shapes, cold): **34.4 → 28.8 ms/tok (1.19x)**; every
shape now 24.5–26.2 GB/s vs 28.8 GB/s measured pure-read wall — the kernel
is at the memory wall. vs the 4x target: see §3 — 4x needs ~84 GB/s, which
exceeds the hardware's read peak by ~2–3x; reported under the stop rule.

Known one-time cost: the on-device greedy path uses `torch.argmax` /
`log_softmax` on NPU → ~52s TBE cold compile ONCE (cached in
`/root/atc_data/kernel_cache` afterwards). If a zero-cold-compile deployment
matters, replace these with small CCE kernels (see jit_elimination_summary.md
for the pattern).
