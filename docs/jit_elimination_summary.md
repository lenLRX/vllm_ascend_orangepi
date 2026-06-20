# Gemma4 on Ascend 310B1 — TBE JIT-Compile Elimination

**Status: COMPLETE.** The full `gemma-4-E2B-it` (bf16) inference path on the
Ascend 310B1 Davinci NPU now compiles **zero TBE kernels cold**. Cold-cache
compile count over the effort: **44 → 0**.

## Why this mattered

Every PyTorch builtin op that reaches the NPU can trigger a TBE (Tensor Boost
Engine) JIT compile the first time it sees a new shape/dtype. On the 310B1 a
single cold compile costs tens of seconds (e.g. the last `te_Cast` alone added
~98 s of wall time to an 8-token run). The goal was to drive the model down to
**only custom CCE kernels** so a fresh deployment has no JIT stalls.

## Ground-truth measurement

`jit_check.py` (a `TorchDispatchMode` aten-op logger) is only a proxy and
**under-counts** — it labels `slice`/`contiguous`/`to.dtype` as "free," but on
310B1 those lower to real `te_StridedSliceD` / `te_Cast` kernels. The only
reliable gate is the **cold-cache diff**:

1. Back up `/root/atc_data/kernel_cache/Ascend310B1`, then empty it.
2. Run a short greedy generate (`max_tokens=8` is enough — per-token ops fire on
   the first decode step) with **`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`**
   set (cold TBE compiles SIGABRT with a protobuf `FatalException` without it).
3. `ls *.o` in the cache dir = the exact set of TBE kernels this workload
   compiles. Restore the backup afterward.

`.o` counts are **shape-keyed distinct kernels** — a per-token op on a constant
shape shows as 1. "1 cold kernel" does *not* imply "startup-only"; confirm
hot-vs-startup with a stack trace, not the count.

## Milestones (cold count)

| Commit | Change | te_* removed | Cold |
|--------|--------|--------------|------|
| (groundwork) | ~15 commits: drop `.item()` D2H syncs, replace scalar mul/add with `mul_scalar_layer`/`add_layer`, CPU-side PLE mask, `empty`+`copy_` instead of `torch.tensor(device=npu)`, pre-allocated buffers | per-step aten ops | (baseline 44) |
| `cfec72cd9` | PLE per-layer slice → custom **`ple_slice_layer`** kernel (layer index as runtime arg; 1 kernel serves all 35 layers) | 35× te_StridedSliceD | 44 → 9 |
| `1684a3295` | Zero NPU KV cache via **`fill_layer`** kernel instead of `torch.zeros(device=npu)` | te_ZerosLike + 2× te_Cast | 9 → 6 |
| `3d724d3d` | `layer_scalar`: `torch.empty(1)` not `torch.ones`, skip the load-time `fill_` (the NPU buffer is dead — forward uses a cached Python float) | te_OnesLike + te_Fill + te_Cast | 6 → 1 |
| `7f9a6b4e` | Build `categorized_sample_indices` as **int64** so the sampler's per-token `sample_indices.long()` is a no-op | last te_Cast | 1 → 0 |

The final te_Cast was **not** a startup op — it was a per-token int32→int64 cast
in the greedy sampler (`sampler.py:794`), pinned with a live `TorchDispatchMode`
cast trace. Fixed at the source in `sampling_metadata.py` (build the index
tensors as `torch.long`, matching `selected_token_indices` directly above); the
CPU tensor is created at the target dtype and `async_tensor_h2d` does a
device-only H2D copy, so no cast is dispatched.

## Custom CCE kernels introduced

`mul_scalar_layer`, `add_layer`, `ple_slice_layer`, `fill_layer`
(plus pre-existing `matmul_weight_transpose_layer`). The recurring pattern:
collapse N per-offset/per-shape TBE compiles into **one prebuilt CCE kernel** that
takes the varying value (layer index, fill value) as a **runtime argument**.

Kernel source + build lives in `/data/llm_simple` and
`/data/llm_simple/src/npu_ops` (separate repos). The npu_ops → vLLM build+wire
chain is documented in the project memory.

## Verification

Each change was verified with `/ssd/test_simple.py` (greedy, 1024 tokens — output
must stay coherent, no garbling) **and** a cold-cache diff confirming the target
kernel dropped to 0. The final A/B (8-token cold, empty cache):

| | te_Cast | total `.o` | wall time |
|---|---|---|---|
| before (int32) | 1 | 1 | 102 s |
| after (int64) | 0 | 0 | 4 s |

Identical output text. The 98 s gap was exactly the cold compile of that one
kernel.
