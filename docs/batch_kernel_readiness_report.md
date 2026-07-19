# Gemma4 / Ascend 310B1 — Batch (bs>1) Kernel Readiness Report

Status: **investigation only, no code changed.**
Symptom: `bs=1` (`/ssd/test_simple.py`) is correct; `bs=2` (`/ssd/test_batch.py`)
garbles the **second** sequence while the **first** stays correct:

```
[0]: \n\nThe capital of France is Paris.        <- correct  (prompt 0)
[1]: \n\ncomputer what what? ... 🧐             <- garbage  (prompt 1)
```

## TL;DR

* The model does **not** rely on the kernels to be "batch aware." The Python
  wiring **packs all sequences into one flat `[total_tokens, hidden]` buffer**
  and either (a) calls elementwise/norm/matmul kernels **once** over the whole
  pack, or (b) calls the **position-dependent** kernels (RoPE, page-attention)
  **once per sequence** inside a `for batch_i in range(batch_size)` loop.
* Under that model, **almost every kernel is batch-safe by construction** and I
  verified it against the sources. The attn-metadata that drives the per-seq
  loop (`seq_lens / offsets / start_positions / block_tables`) is also built
  correctly for `bs>1`.
* **One concrete kernel-level defect exists:** `page_attn_gqa_dim512.cce`
  (the **full-attention** path) processes a hard-coded `m_tile = 32` rows
  instead of the real `curr_m`, so it reads/writes 32 rows even when a sequence
  contributes fewer (always true in decode, where `curr_seq_len == 1`). This is
  a real out-of-bounds read of `q` and out-of-bounds write of the attention
  output. The sliding-window twin `page_attn_dim256_gqa.cce` does **not** have
  this bug (it uses `curr_m_tile`).
* **Important caveat (do not over-claim):** because the per-seq calls run in
  increasing `flat_seq_offset` order on one serialized stream, and softmax is
  per-row, the dim512 overspill into a later sequence's rows is **overwritten by
  that sequence's own (correct) call**, and each sequence's one valid row is
  always computed correctly. So this bug is a genuine latent hazard and must be
  fixed, but static analysis says it is *largely self-healing* for in-buffer
  rows — it may not be the **sole** cause of the garbage. The residual risk it
  leaves is the OOB **past the end** of the packed output buffer (allocator-
  dependent corruption), which can differ between the `[1,·]` buffer of `bs=1`
  and the `[2,·]` buffer of `bs=2`.

Recommended order of action is in the last section: fix dim512 bounding first
(cheap, removes a real bug, gives a decisive re-test), then check the small set
of non-kernel suspects if garbage persists.

---

## 1. How batching actually flows through this port

`vllm/worker/npu_model_runner.py`
* `_prepare_prompt` (prefill): packs every prompt's tokens into one flat list;
  per sequence it records `offset=0`, `length=prompt_len`, `start_position=0`,
  `block_tables=(None, host_block_list)`. → `seq_lens=[L0,L1,…]`.
* `_prepare_decode`: one generation token per sequence; per sequence
  `offset = position = seq_len-1`, `length=1`, `start_position=offset`.
  → `seq_lens=[1,1,…]`.

Both are **correct for bs>1.** The flat layout means token row `t` belongs to
exactly one sequence, and the per-sequence metadata lists line up with the
`flat_seq_offset` walk in the model.

`vllm/model_executor/models/gemma4_npu.py`, `Gemma4Attention.forward`
* RoPE: `for batch_i …` → `rope_standard_layer` on
  `q_flat[token_offset : token_offset+seq_len]` with this seq's `start_pos`.
* Page-attention: `for batch_i …` → KV-cache memcpy into this seq's blocks, then
  `page_attn_*_layer` on `q[flat_seq_offset : flat_seq_offset+curr_seq_len]`
  writing to `attn_output[flat_seq_offset]`.

Everything else (`qkv_proj`, norms, residual adds, MLP, PLE, `o_proj`, `lm_head`)
runs once over the flat pack.

---

## 2. Kernel-by-kernel batch readiness

Legend: **PF** = prefill (M = packed prompt tokens, attn per-seq with
`seq_len = prompt_len`), **DEC** = decode (M = #sequences, attn per-seq with
`seq_len = 1`).

| Kernel (source) | Call pattern | PF | DEC | Verdict / evidence |
|---|---|---|---|---|
| `rmsnorm_layer` (`rmsnorm_layer.cce`) | flat, `first_dim=tokens` | ✅ | ✅ | loops `i<cur_size` in tiles w/ tail mask |
| `qkv_norm_with/no_weight` (`qkv_norm_layer.cce`) | flat, `token_num*heads` | ✅ | ✅ | `for token_idx<total_tokens`, tail-safe |
| `add_layer` (`add_layer.cce`) | flat, `numel` | ✅ | ✅ | `for i<total_size` tiled, tail mask |
| `mul_layer` / `mul_scalar_layer` | flat, `numel` | ✅ | ✅ | same tiled+tail pattern |
| `gated_gelu` (`gated_gelu.cce`) | **per-token** loop in MLP | ✅ | ✅ | one token/call, runtime `last_dim` |
| `gelu_pytorch_tanh` (`.cce`) | flat, `numel` (PLE gate) | ✅ | ✅ | tiled + tail mask |
| `split_qkv` (`split_qkv.cpp`) | flat, `token_num` | ✅ | ✅ | DMA loop over `batch` w/ `curr_tile` |
| `ple_slice_layer` (`ple_slice_layer.cpp`) | flat, `tokens` | ✅ | ✅ | `for t<tokens` DMA |
| `rope_standard` (`rope_standard.cce`) | **per-seq** | ✅ | ✅ | `for s<cur_size`, freq idx `s+start_pos`, writes only `cur_size` rows; no `start_pos==0` assumption |
| `matmul_nz_layer` (`matmul_layer_nz.cpp`) | flat, `m=tokens` | ✅ | ✅ | stores **`m_curr_tile`** rows (l.192/353); `m==1` GEMV path (l.518) writes 1 row |
| `matmul_bias_nz_layer` (`matmul_bias_layer_nz.cce`) | flat | ✅ | ✅ | stores `m_curr_tile` rows (l.180) |
| `matmul_weight_transpose` | weight load only | n/a | n/a | not on token hot-path |
| `gather_layer` (`gather.cpp`) — embed + logits | flat, exact count | ✅ | ✅ | `for i<index_num`; logits passes `first_dim=#selected` |
| `embedding_layer`/`gather_embedding` | (unused — embed goes via `gather_layer`) | — | — | same safe pattern if used |
| **`page_attn_dim256_gqa`** (sliding, `.cce`) | **per-seq** | ✅ | ✅ | **bounded by `curr_m_tile`** (l.80) incl. output write (l.533-534) |
| **`page_attn_gqa_dim512`** (full, `.cce`) | **per-seq** | ⚠️ | ⚠️ | **NOT bounded** — uses constant `m_tile=32`; `curr_m` (l.36) computed but never used. OOB read/write. |

---

## 3. The one concrete kernel defect — `page_attn_gqa_dim512.cce`

File: `/data/llm_simple/src/npu_ops/page_attn_gqa_dim512.cce`
(host wrapper `page_attn_gemma4.cpp:36`, launched `<<<1,nullptr,stream>>>`).

```c
34  for (int mi = 0; mi < m_loop; ++mi) {
35      int m_start = mi * m_tile;
36      int curr_m = (M - m_start < m_tile) ? (M - m_start) : m_tile;   // <-- computed
...                                                                     //     and NEVER used
111     copy_gm_to_cbuf_multi_nd2nz_b16(Q_l1, q_curr + k_start, 0, 1, m_tile, ...);   // reads 32 Q rows
131     mad(l0c, l0a, l0b, m_tile, curr_k, curr_n, ...);                              // 32 rows
155     for (uint16_t row = 0; row < 32; ++row) { ... }                              // softmax 32 rows
335     mad(l0c, l0a, l0b, m_tile, curr_n, N_out, ...);                              // PV 32 rows
413     for (int row = 0; row < m_tile; ++row)                                        // OUTPUT write
414         copy_ubuf_to_gm_align(out_curr + row * q_gm_stride, ...);                 //  32 rows always
```

`m_tile` is the constant **32**. For any sequence with fewer than 32 query rows
(every decode step has exactly 1; prefill tails are < 32) the kernel:

1. **reads** 32 `q` rows from `q[flat_seq_offset …]` — past this sequence's
   `curr_seq_len` rows into the next sequence's `q` (or past the buffer for the
   last sequence), and
2. **writes** 32 output rows from `attn_output[flat_seq_offset …]` — past this
   sequence's region.

The sliding-window kernel does the same job correctly: `page_attn_dim256_gqa.cce`
computes `curr_m_tile = min(m_tile, m - m_tile*i)` (l.80) and uses it for the Q
load, softmax, PV, **and** the output write (l.533-534). dim512 is simply missing
that bound.

### Why `bs=1` survives it and why this is still real
* `bs=1`: there is no neighbouring sequence; the 31 spill rows land past a
  `[1, q_size]` buffer and are never read by `o_proj` (which only consumes row
  0). Harmless in practice.
* `bs=2`: the spill from sequence 0 lands in sequence 1's rows — **but** the
  per-seq calls run in `flat_seq_offset` order on one serialized stream, and
  softmax is per-row, so:
  * each sequence's single valid row is computed only from its own valid `q`
    row → **correct**, and
  * sequence 1 **overwrites** sequence 0's spill with its own correct values
    when its call runs.
  So the in-buffer corruption is self-healing. The part that is **not** healed
  is the write *past the end of the whole `attn_output`* (last sequence's tail),
  whose neighbour in the allocator differs between the `bs=1` `[1,·]` buffer and
  the `bs=2` `[2,·]` buffer. That is undefined behaviour and the right thing to
  eliminate, but I cannot prove from static reading alone that it is the entire
  story.

**Fix (when we get to it):** bound the Q load / softmax / PV / output-write by
`curr_m` exactly as dim256 bounds by `curr_m_tile`. This removes a genuine
latent batch bug and makes the re-test decisive.

---

## 4. What is verified batch-safe (so we can stop suspecting it)

* All elementwise/norm kernels — single-core `<<<1>>>` kernels that loop over the
  runtime element/token count in tiles with a `plt_*` tail mask. No fixed token
  cap, no batch=1 assumption.
* RoPE — per-seq, honours `start_pos` and writes only `seq_len` rows.
* All matmuls and the gather/embedding path — store exactly `m_curr_tile` /
  `first_dim` rows; the only special case (`m==1`) is a separate GEMV kernel that
  writes exactly one row.
* Sliding-window page-attention (dim256) — fully `curr_m_tile`-bounded.
* Attn metadata construction for bs>1 — `seq_lens / offsets / start_positions /
  block_tables` are per-sequence and consistent with the flat token walk.

---

## 5. Suspects to check **if** fixing dim512 does not fix the garbage

These are outside the kernel set but are the only remaining places a per-second-
sequence corruption can hide; listed so the next session can go straight to them:

1. **Sampler row selection for bs>1.** `prepare_model_input` passes
   `seq_lens=None` into `SamplingMetadata.prepare(...)` on the **decode** path
   (`npu_model_runner.py:294-302`). Confirm `selected_token_indices` is `[0,1]`
   for a 2-way decode (and `[L0-1, L0+L1-1]` for a packed prefill) — a wrong
   index for seq 1 would feed seq 1's sampler the wrong logits row.
2. **Per-iteration `page_table_npu` allocation** inside the attn batch loop
   (`gemma4_npu.py:415`, `torch.tensor(..., device="npu")`) — verify the H2D and
   the kernel read stay correctly ordered for the 2nd sequence (same-stream
   ordering should make it safe, but it is the one fresh device alloc inside the
   loop).
3. **KV-shared layers (L13/L14) under bs>1** — shared layers read
   `kv_caches[local_src]` with the *current* sequence's block table
   (`gemma4_npu.py:869-880`); confirm both sequences' shared KV are read from the
   right physical blocks.

## 6. How to confirm empirically (ground truth)

1. Fix the dim512 bound, rebuild `npu_ops`, re-run `/ssd/test_batch.py`. If
   `[1]` becomes coherent → dim512 was the cause.
2. If still garbled, dump `attn_output[0]` vs `attn_output[1]` after the first
   decode step of a full-attention layer vs a sliding layer, and dump the
   sampler's `selected_token_indices`, to localize divergence to attention vs
   sampling.

(Per project convention set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` for
any cold run, and restore the warm kernel cache afterwards.)
