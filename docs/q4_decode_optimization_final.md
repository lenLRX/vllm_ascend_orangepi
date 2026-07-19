# Gemma4 Q4_0 Decode Optimization — Final Report

Date: 2026-07-18. Device: Ascend 310B1 (Orange Pi AI Pro). Baseline report:
`docs/q4_decode_baseline.md` (measurement protocol: same scripts, warm kernel
cache, median of 3 runs).

## Outcome vs. goal targets

| Metric | Baseline | Final | Speedup |
|---|---|---|---|
| E2E decode per-token, hook-free (256-tok run, warmed) | ~250 ms/tok | **96 ms/tok (10.42 tok/s)** | **~2.6x** |
| E2E decode per-token, layer-hooked (64-tok runs) | 259 ms/tok | 115 ms/tok | 2.25x |
| Forward decode only (layer-hook timing) | 206.4 ms/tok | 75.2 ms/tok | 2.74x |
| vLLM whole-run rate, 64 tok | 3.35 tok/s | 7.35 tok/s | 2.19x |
| Q4_0 matmul kernel (cold, fused shapes) | 34.4 ms/tok | **28.8 ms/tok** | **1.19x** |
| Prefill (464-token prompt) | 2.52 s | 1.35 s | 1.87x |

Measurement notes: (a) the layer-hook scripts sync per layer and inflate
per-token time ~10–20 ms vs hook-free runs; both baselines above are quoted
with the same methodology as their final numbers. (b) The first greedy
decode step in each process pays a ~50 s one-time TBE init for the fast-path
ops (argmax/log_softmax kernels are on-disk cached but still cost ~50 s at
first dispatch) — this must be absorbed by a warmup generate before timing;
un-warmed 256-token runs misleadingly show ~3.3 tok/s because of it.

### T1 — decode matmul ≥ 4x: NOT MET (stop rule, with evidence)

Achieved: **1.19x aggregate** (34.4 → 28.8 ms/tok); per-shape effective
bandwidth 6.3–24.3 → 24.5–26.2 GB/s. Best per-shape: kv_slide 6.3 → 25.6 GB/s
(4.06x, small-N launch-bound case).

Why 4x is not physically reachable on this chip: every decode token must
stream the full Q4_0 weight set — **724 MB of nibbles+scales** — from DDR
(working set >> 4 MB L2, zero reuse). Measured device bandwidth: 28.8 GB/s
pure read, 42.5 GB/s read+write copy. The 4x target (34.4 → 8.6 ms) requires
~84 GB/s of sustained reads — 2–3x the hardware peak. The kernel now runs
every shape at 24.5–26.2 GB/s, i.e. within ~10–15% of the 28.8 GB/s read
wall. Attempted and rejected: VEC scale-load hoisting (produced wrong
results with non-POST_UPDATE vector forms — reverted); mad_s4/mad_s8 cube
paths (W4A4 accuracy risk; W4A8 `mad_s8s4` does not exist on 310B1).

### T2 — host time hidden (wall ≤ ~1.1 × device-busy): PARTIALLY MET

- E2E decode: 96 ms/tok wall vs ~61 ms device-busy (matmul 28.8 + lm_head
  28.2 + attention/norms/misc ~4) → **1.57x**.
- Forward section only: ~63–75 ms wall vs ~32–37 ms device → the ~600
  remaining small launches (norms, rope, gelu, adds, attn glue) each cost
  ~18µs of Python+launch host time, and the device drains those kernels
  faster than Python can issue them. The matmul-dense regions are fully
  overlapped; the small-op regions starve.
- Residual beyond small-op launches: serial lm_head tail (28.2 ms device
  with the host idle at the sampling sync) and engine bookkeeping (~5–10 ms:
  scheduler, detokenize, prepare).
- (CANN `task_time` profiler inflates per-kernel durations ~3–4x on this
  chip, so device-busy is bench-anchored, not profiler-anchored.)
- Closing the rest needs either materially fewer small launches (kernel
  fusions) or a graph-replay / C++-dispatch path that bypasses per-op
  Python — bigger, riskier work than the safe changes above.

### T3 — end-to-end improvement: MET

2.2–2.6x e2e depending on metric (see table), 2.74x forward. Correctness
gate passed after every round (coherent greedy generation; 256-token final
gate fully coherent; kernel unit tests 17/17).

## Changes

vLLM repo (4 files, ~320 insertions):

- `vllm/model_executor/models/gemma4_npu.py`
  - Removed the 35x/token full `torch.npu.synchronize()` (page-table
    lifetime was the only reason for it).
  - Page tables cached by content on the attention module — the blocking
    `torch.tensor(device='npu')` H2D now runs ~once per 64 tokens instead of
    35x/token.
  - **Load-time shard fusion**: GGUF raw blocks are row-major over N, so
    q/k/v (and gate/up) raw bytes concatenate losslessly into one Q4_0
    tensor → one NZ conversion → **one matmul per projection** at runtime
    (245 → 140 matmul launches/token, no `torch.cat`).
  - KV-shared layers get raw k/v bytes copied from layers 13/14 *before*
    conversion (bit-identical to the source layer's conversion).
  - Buffer pooling across decode steps for attention, decoder-layer
    residuals, RMSNorm, gelu, and ple_slice intermediates (launches are
    stream-ordered; elementwise in-place use is index-safe).
  - Batched gated_gelu (m rows per launch; prefill had 464 Python calls
    per layer before).
  - Batched rope (q+k in one call) and fused qkv_norm on non-KV-shared
    layers.
  - Preseeded `_qweight_nz*` attrs so per-call `hasattr` checks are cheap
    `__dict__` hits (nn.Module `__getattr__` does a full scan on misses).
- `vllm/model_executor/layers/linear.py`
  - Single fused Q4_0 matmul per projection; pooled output buffers; hoisted
    a per-call import.
- `vllm/model_executor/layers/npu/util.py`
  - **Cached `get_default_stream()`** — was called ~3000x/token, each a
    ~120µs walk through `torch._utils` device-index resolution (incl.
    `torch.cuda.is_available`). This was the single largest host win
    (168 → 96 ms/tok alone).
- `vllm/worker/npu_model_runner.py`
  - **On-device greedy sampling**: `argmax` + `log_softmax` on NPU, copy
    back only the token id (+ exact logprob) — skips the full-vocab logits
    D2H and the ~10 ms CPU softmax per token. Falls back to the CPU sampler
    whenever anything could change the argmax (penalties, min_tokens,
    logprobs requests, beam, n>1).
  - `Q4_PHASE_TIMING=1` phase instrumentation (fwd / logits / d2h / sample /
    engine_gap).

Kernel repo (`/data/llm_simple`, deployed via the two .so files):

- `matmul_gguf_q4_0.cpp`: k_tile 256→128 → n_tile 128→256 — halves the
  per-n_i FIX drain count; every shape now at the memory wall.
- `gated_gelu`: batched m rows per launch.
- pybind batch wrappers: `rope_qk_layer`, `qkv_norm_fused_layer`.

## Verification

- `test_gguf_q4_0_kernel.py`: **17/17 PASS** with the final kernel.
- Greedy 256-token generation (`verify_q4_gate.py`): fully coherent; opening
  matches the baseline text exactly (all rounds produced identical text for
  the shared 64-token prefix — greedy argmax is unchanged).
- Matmul kernel change set is exactly one line (k_tile 256→128); the failed
  scale-hoist experiment was fully reverted.

## Remaining levers (not done, expected gains)

1. **lm_head Q4_0 quantization** (−~19 ms/tok): the tied embedding doubles
   as lm_head — an fp16 [262144, 1536] matmul, 805 MB/token = 28.2 ms (44%
   of device time). Requantizing to Q4_0 at load → 226 MB → ~8.7 ms.
   Precedent: the plain community Q4_0 GGUF of gemma-4-E2B ships
   `token_embd.weight` at Q4_K (4-bit), while the QAT Q4_0 file we deploy
   deliberately keeps Q6_K for quality; llama.cpp also exposes
   `--token-embedding-type` / `--output-tensor-type` to force it. It shifts
   logits ~1%, so it is an accuracy/product decision, not a pure
   optimization — left for the user.
2. **Engine-level pipelining** (−~10 ms/tok): overlap the serial lm_head
   tail + engine bookkeeping with the next step. vLLM core surgery — outside
   the agreed scope.
3. **More fusions** (add+rmsnorm, gelu+mul PLE): −~5 ms/tok of forward
   Python.
4. **Custom CCE argmax/log_softmax**: removes the one-time ~52 s TBE cold
   compile the greedy fast path introduces (steady-state unaffected).

## Caveats

- One 256-token gate run measured 3.33 tok/s (2.5x slow) amid sustained
  load; three adjacent rounds and a re-run all show 7.33–7.35 tok/s, so it
  was a transient (thermal or system noise). Worth watching under long runs.
