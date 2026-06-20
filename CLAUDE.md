# CLAUDE.md — vllm_ascend_orangepi

Goal: run `gemma-4-E2B-it` (bf16) inference on an **Ascend 310B1 Davinci NPU**
via vLLM using **only custom CCE kernels**, eliminating PyTorch builtin NPU ops
because each can trigger a TBE JIT compile (tens of seconds, cold, per shape).

Branch: `ascend_orangepi` (this is also the PR base — no separate main).

## Current state

The JIT-elimination effort is **COMPLETE**: cold TBE compile count is **44 → 0**.
The full forward + sampler path compiles zero TBE kernels cold. See
`docs/jit_elimination_summary.md` for the milestone-by-milestone arc. Further
work on this model is **latency/throughput**, a separate axis from JIT compiles.

## Running inference / tests

- `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` **must** be set for any run
  that compiles cold — without it, cold TBE/ATC compiles SIGABRT with
  `google::protobuf::FatalException: CHECK failed: file != nullptr`. Warm runs
  (everything cached) don't need it. Every working test script sets it.
- `/ssd/test_simple.py` — greedy 1024-token coherence check (output must not
  garble). Params: max_model_len=256, enforce_eager=True, block_size=64.

## Measuring TBE JIT compiles (the real gate)

`jit_check.py` is only a proxy and **under-counts** (labels slice/contiguous/cast
"free"; they lower to te_StridedSliceD/te_Cast on 310B1). Ground truth =
**cold-cache diff**:

1. Back up `/root/atc_data/kernel_cache/Ascend310B1` (a `_BACKUP_coldtest` copy
   already exists), then empty the live dir.
2. Run an 8-token greedy generate with the protobuf env var (per-token ops fire
   on the first decode step).
3. `ls *.o` = the exact TBE kernels this workload compiles. **Restore the warm
   cache afterward** so normal runs stay fast.

`.o` count is shape-keyed distinct kernels — "1 cold kernel" does NOT mean
"startup-only." Use a stack trace to tell hot-path from startup.

## Workflow conventions

- **One JIT trigger at a time.** Remove a trigger → verify (test_simple.py
  coherent + cold diff drops the target kernel) → commit separately. Commit
  messages are prefixed `jit-elim:`.
- **Surgical staging.** Stage only the file(s) you changed. Do **not** `git add
  -A` / `git add .`.
- The recurring kernel pattern: collapse N per-offset/per-shape TBE compiles into
  one prebuilt CCE kernel taking the varying value as a **runtime argument**
  (`ple_slice_layer`, `fill_layer`, `mul_scalar_layer`, `add_layer`).
- A kernel swap that produces correct values but garbled text usually needs to be
  **in-place** (out ptr == in ptr).

## Do NOT touch

- Untracked files `EOF` and `docs/pytorch_builtin_tbe_ops.md` predate this work —
  leave them out of commits.
- Kernel repos `/data/llm_simple` and `/data/llm_simple/src/npu_ops` carry
  pre-existing unrelated dirty files (`.gitignore`, `flash_attn_simple.cce`,
  `gather.cpp`, `qkv_norm_layer.cce/.cpp`, `freqs_cis_gather.cce/.cpp`,
  `gelu_pytorch_tanh.cce.orig`). Stage only the specific kernel files you intend.

## Custom kernel build/wire chain

Kernel source: `/data/llm_simple/src/npu_ops` → build npu_ops → wire into vLLM.
Full chain documented in the auto-memory
(`custom-kernel-runtime-offset-pattern`).
