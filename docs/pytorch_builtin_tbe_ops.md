# PyTorch Builtin TBE Ops Used in vllm_ascend_orangepi (Gemma4)

Generated: 2026-06-12
Scope: `vllm/model_executor/models/gemma4_npu.py` and the surrounding NPU runner stack.

---

## 1. Executive Summary

Most of the compute-heavy Gemma4 operators are already implemented as custom CCE kernels in `/data/llm_simple/src/npu_ops` and exposed through `py_npu_ops`.  However, several **PyTorch builtin operators** are still on the hot path and each unique shape/dtype combination triggers a TBE JIT compilation via `libacl_op_compiler` → TBE → ccec.  This document lists them, maps them to the TBE kernels they produce, and identifies which existing custom kernels could replace them.

The goal is to **reduce TBE JIT recompilation** (and remove all PyTorch NPU operators) by routing every op through the existing custom CCE kernels.

---

## 2. Methodology

1. **Static review** of `gemma4_npu.py`, `npu_model_runner.py`, and `vllm/model_executor/layers/npu/*.py`.
2. **Empirical mapping** with `/ssd/map_ops.py` (clears ATC cache, runs one PyTorch op at a time, diffs the TBE cache to see which kernel was compiled).  See `/ssd/op_to_kernel_map.json` for raw data.
3. **Custom-kernel inventory** from `/data/llm_simple/src/npu_ops/pybind.cc` and the `.cpp`/`.cce` sources.

---

## 3. Currently Used PyTorch Builtin Ops

### 3.1 Embedding lookup

| PyTorch code | TBE kernel(s) | Location | Replacement available? |
|---|---|---|---|
| `self.embed_tokens(input_ids)` | **GatherV2** (torch.nn.Embedding maps to gather) | `Gemma4Model.get_input_embeddings` | **Yes** — `embedding_layer(output, weight, index, seqlen, hidden_dim, dt, stream)` |
| `self.embed_tokens_per_layer(per_layer_inputs_tokens)` | **GatherV2** | `Gemma4Model.get_per_layer_inputs` | **Yes** — same `embedding_layer` |

Notes:
- `VocabParallelEmbedding` ultimately calls `torch.nn.functional.embedding` → `GatherV2`.
- The custom `embedding_layer` kernel supports FP16/BF16, uses an int32 index, and copies rows one-by-one in UB.  It expects `index` on the NPU.

---

### 3.2 Element-wise / scalar multiplications

| PyTorch code | TBE kernel(s) | Location | Replacement available? |
|---|---|---|---|
| `hidden_states * self.normalizer` | **Mul** | `Gemma4Model.get_input_embeddings` | **Yes** — `mul_scalar_layer(output, input, total_size, scalar, dt, stream)` |
| `per_layer_embeds * self.embed_scale_per_layer` | **Mul** | `Gemma4Model.get_per_layer_inputs` | **Yes** — `mul_scalar_layer` |
| `per_layer_projection * self.per_layer_projection_scale` | **Mul** | `Gemma4Model.project_per_layer_inputs` | **Yes** — `mul_scalar_layer` |
| `combined * self.per_layer_input_scale` | **Mul** | `Gemma4Model.project_per_layer_inputs` | **Yes** — `mul_scalar_layer` |
| `gated * per_layer_input` (PLE gate) | **Mul** | `Gemma4DecoderLayer.forward` | **Already using** `mul_layer` ✓ |
| `hidden_states * layer_scalar` | **Mul** | `Gemma4DecoderLayer.forward` | **Already using** `mul_scalar_layer` ✓ |

Notes:
- `mul_scalar_layer` is a single custom kernel for `x * scalar`.
- `mul_layer` is already used for the PLE gate multiply.

---

### 3.3 Additions / residual connections

| PyTorch code | TBE kernel(s) | Location | Replacement available? |
|---|---|---|---|
| `hidden_states + residual` (post-attention) | **Add** | `Gemma4DecoderLayer.forward` | **Already using** `add_layer` ✓ |
| `hidden_states + residual` (post-MLP) | **Add** | `Gemma4DecoderLayer.forward` | **Already using** `add_layer` ✓ |
| `hidden_states + ple_out` | **Add** | `Gemma4DecoderLayer.forward` | **Already using** `add_layer` ✓ |
| `per_layer_projection + per_layer_inputs` | **Add** | `Gemma4Model.project_per_layer_inputs` | **Could use** `add_layer` (currently PyTorch) |

Notes:
- `add_layer` is already used for the three main residuals.  The PLE combine add is still a PyTorch op.

---

### 3.4 Linear projections (QKV, O, gate_up, down, PLE)

| PyTorch code | TBE kernel(s) | Location | Replacement available? |
|---|---|---|---|
| `self.qkv_proj(hidden_states)` | **MatMul** + possible **Cast**/**Add** | `Gemma4Attention.forward` | **Partially** — `matmul_layer`/`matmul_nz_layer` exist but require specific weight layouts (NZ/transposed) |
| `self.o_proj(attn_output)` | **MatMul** | `Gemma4Attention.forward` | **Partially** — same caveat |
| `self.gate_up_proj(x)` | **MatMul** | `Gemma4MLP.forward` | **Partially** |
| `self.down_proj(output)` | **MatMul** | `Gemma4MLP.forward` | **Partially** |
| `self.per_layer_input_gate(hidden_states)` | **MatMul** | `Gemma4DecoderLayer.forward` | **Partially** |
| `self.per_layer_model_projection(flat_embeds)` | **MatMul** | `Gemma4Model.project_per_layer_inputs` | **Partially** |
| `self.per_layer_projection(ple_mul)` | **MatMul** | `Gemma4DecoderLayer.forward` | **Partially** |

Notes:
- These go through vLLM's `ColumnParallelLinear`/`ReplicatedLinear`, which ultimately call `torch.nn.functional.linear` → **MatMul** (and bias Add if present).
- Custom kernels:
  - `matmul_layer(output, lhs, rhs, m, n, k, dt, stream)` — expects `lhs` [m,k] row-major and `rhs` [k,n] **transposed internally**?  Actually the CCE `matmul_layer.cpp` loads `rhs` with `load_cbuf_to_cb(..., inc=1)`, meaning it expects RHS already in transposed/NZ layout.  See Section 5.
  - `matmul_nz_layer` — same but assumes RHS is pre-converted to NZ.
  - `matmul_bias_nz_layer` — adds bias.
  - `matmul_weight_transpose_layer` — converts a [n,k] weight to the layout `matmul_layer` wants.
- Replacing these is **possible but requires weight-layout conversion** at load time or compile time.

---

### 3.5 Activation functions

| PyTorch code | TBE kernel(s) | Location | Replacement available? |
|---|---|---|---|
| `F.gelu_pytorch_tanh(gate)` | **Gelu** | `Gemma4DecoderLayer.forward` | **Already using** `gelu_pytorch_tanh_layer` ✓ |
| Gated GELU in MLP | **Mul** + **Gelu** (manual) or custom | `Gemma4MLP.forward` | **Already using** `gated_gelu_layer` ✓ |

Notes:
- `gated_gelu_layer` fuses `gate = gate_up[..., 0:h]; up = gate_up[..., h:2h]; out = gelu(gate) * up` into one kernel.

---

### 3.6 Normalization

| PyTorch code | TBE kernel(s) | Location | Replacement available? |
|---|---|---|---|
| `self.input_layernorm(x)` | **Rsqrt** + **Mul** + **ReduceMeanD** (manual RMSNorm = 4 kernels) or fused | `Gemma4DecoderLayer.forward` | **Already using** `rmsnorm_layer` ✓ |
| `self.post_attention_layernorm` | same | `Gemma4DecoderLayer.forward` | **Already using** `rmsnorm_layer` ✓ |
| `self.pre_feedforward_layernorm` | same | `Gemma4DecoderLayer.forward` | **Already using** `rmsnorm_layer` ✓ |
| `self.post_feedforward_layernorm` | same | `Gemma4DecoderLayer.forward` | **Already using** `rmsnorm_layer` ✓ |
| `self.norm(x)` (final) | same | `Gemma4Model.forward` | **Already using** `rmsnorm_layer` ✓ |
| `self.per_layer_projection_norm` | same | `Gemma4Model.project_per_layer_inputs` | **Already using** `rmsnorm_layer` ✓ |
| Q/K/V per-head norms | **LayerNorm** / RMSNorm pieces | `Gemma4Attention.forward` | **Already using** `qkv_norm_with_weight_layer` / `qkv_norm_no_weight_layer` ✓ |

---

### 3.7 Attention

| PyTorch code | TBE kernel(s) | Location | Replacement available? |
|---|---|---|---|
| Page attention (sliding, dim256) | custom page_attn_dim256_gqa | `Gemma4Attention.forward` | **Already using** `page_attn_dim256_gqa_layer` ✓ |
| Page attention (full, dim512) | custom page_attn_gqa_dim512 | `Gemma4Attention.forward` | **Already using** `page_attn_gqa_dim512_layer` ✓ |

---

### 3.8 RoPE

| PyTorch code | TBE kernel(s) | Location | Replacement available? |
|---|---|---|---|
| RoPE application per sequence | custom `rope_standard_layer` | `Gemma4Attention.forward` | **Already using** `rope_standard_layer` ✓ |

---

### 3.9 Split / reshape / indexing

| PyTorch code | TBE kernel(s) | Location | Replacement available? |
|---|---|---|---|
| `qkv.reshape(...)` | metadata only | `Gemma4Attention.forward` | No kernel triggered; cheap |
| `q.reshape(...)` | metadata only | `Gemma4Attention.forward` | No kernel triggered; cheap |
| `q_roped_flat[token_offset:token_offset+seq_len]` | **StridedSliceD** | `Gemma4Attention.forward` | Could avoid by passing pointer offset to kernel |
| `attn_output[flat_seq_offset, ...]` | **StridedSliceD** | `Gemma4Attention.forward` | Could avoid by passing pointer offset |
| `kv_cache[0, block, offset, 0]` indexing for memcpy | metadata / address calc | `Gemma4Attention.forward` | No TBE kernel; pure pointer arithmetic |

Notes:
- `StridedSliceD` is the most frequent kernel in the ATC cache (315+ instances).  Most are from these dynamic slices inside loops.  They can be eliminated by computing raw `data_ptr()` offsets and passing them directly to the custom kernels instead of slicing tensors.

---

### 3.10 Tensor creation / memory allocation

| PyTorch code | TBE kernel(s) | Location | Replacement available? |
|---|---|---|---|
| `torch.empty(..., device="npu")` | none | everywhere | Memory allocation; not a compute op |
| `torch.empty_like(x)` | none | everywhere | Memory allocation; not a compute op |
| `torch.tensor(page_table_list, dtype=torch.long, device="npu")` | **Cast** or H2D copy | `Gemma4Attention.forward` | H2D copy; unavoidable unless page table stays on NPU |
| `torch.zeros_like(input_ids)` | **ZerosLike** | `Gemma4Model.get_per_layer_inputs` | Could pre-allocate a zero buffer once |
| `torch.ones(1)` (layer_scalar buffer) | none | `Gemma4DecoderLayer.__init__` | Buffer; no runtime kernel |

---

### 3.11 Runner-level builtins (npu_model_runner.py)

| PyTorch code | TBE kernel(s) | Location | Notes |
|---|---|---|---|
| `torch.tensor(..., device=self.device)` | H2D copy / Cast | `_prepare_prompt`, `_prepare_decode` | Token/position tensors copied from CPU list to NPU |
| `torch.cat(all_hidden, dim=0)` | **ConcatD** | `execute_model` (batch > 1) | From serial batch merge; could be avoided if model outputs already contiguous |
| `logits.cpu()` | D2H copy | `execute_model` | Needed for sampling; not TBE |
| sampling ops (`top_k`, `top_p`, etc.) | various | `self.model.sample` | CPU-only after logits move; no NPU TBE |

---

## 4. TBE Kernel Mapping (from empirical test)

Run with `/ssd/map_ops.py` after clearing `/root/atc_data/kernel_cache/Ascend310B1`.

| PyTorch op | TBE kernel(s) |
|---|---|
| `a + b` / `torch.add` | **Add** |
| `a - b` / `torch.sub` | **Sub** |
| `a * b` / `torch.mul` | **Mul** |
| `a / b` / `torch.div` | **RealDiv** |
| `-a` | **Neg** |
| `torch.abs` | **Abs** |
| `torch.square` | **Square** |
| `torch.sqrt` | **Add** + **Sqrt** |
| `torch.rsqrt` | **Rsqrt** |
| `a ** n` / `torch.pow` | **Pow** |
| `a @ b` / `torch.matmul` / `torch.mm` | **MatMul** |
| `torch.sum` / `.sum(dim=...)` | **ReduceSumD** |
| `torch.mean` / `.mean(dim=...)` | **ReduceMeanD** |
| `torch.max` | **ReduceMaxD** |
| `torch.min` | **ReduceMinD** |
| `F.relu` | **Relu** |
| `F.gelu` | **Gelu** |
| `F.silu` | **Swish** |
| `torch.sigmoid` | **Sigmoid** |
| `torch.tanh` | **Tanh** |
| `F.softmax` | **SoftmaxV2** |
| `.bfloat16()` / `.half()` / `.float()` | **Cast** |
| `a > b` | **Greater** |
| `a < b` | **Less** |
| `a == b` | **Equal** |
| `a >= b` | **GreaterEqual** |
| `a <= b` | **LessEqual** |
| `a != b` | **NotEqual** |
| `torch.zeros_like` | **ZerosLike** |
| `torch.ones_like` | **Fill** |
| `torch.cat` | **ConcatD** |
| `torch.stack` | **Pack** |
| `torch.where` | **Greater** + **Select** |
| `torch.clamp` | **ClipByValueV2** |
| `torch.exp` | **Exp** |
| `torch.cos` / `torch.sin` | **Cos** / **Sin** |
| `torch.ceil` / `torch.floor` / `torch.round` | **Ceil** / **Floor** / **Round** |
| `reshape` / `view` / `transpose` / `permute` / `squeeze` / `unsqueeze` / `flatten` | (none — metadata only) |
| slicing `a[0:5]` / `a[:, :3]` | **StridedSliceD** |

---

## 5. Available Custom Kernels in `/data/llm_simple/src/npu_ops`

### Already used by Gemma4

| Kernel | Pybind name | Purpose |
|---|---|---|
| `add_layer` | `add_layer` | `output = lhs + rhs` |
| `mul_layer` | `mul_layer` | `output = lhs * rhs` |
| `mul_scalar_layer` | `mul_scalar_layer` | `output = input * scalar` |
| `rmsnorm_layer` | `rmsnorm_layer` | RMSNorm with weight |
| `qkv_norm_with_weight_layer` | `qkv_norm_with_weight_layer` | per-head Q/K norm with shared weight |
| `qkv_norm_no_weight_layer` | `qkv_norm_no_weight_layer` | per-head V norm (no weight) |
| `rope_standard_layer` | `rope_standard_layer` | standard RoPE (cos/sin table) |
| `gated_gelu_layer` | `gated_gelu_layer` | `gelu(gate) * up` for MLP |
| `gelu_pytorch_tanh_layer` | `gelu_pytorch_tanh_layer` | GELU for PLE gate |
| `split_qkv_layer` | `split_qkv_layer` | split fused QKV into Q/K/V |
| `page_attn_dim256_gqa_layer` | `page_attn_dim256_gqa_layer` | sliding-window page attention (hd=256) |
| `page_attn_gqa_dim512_layer` | `page_attn_gqa_dim512_layer` | full page attention (hd=512) |

### Available but NOT yet used in Gemma4

| Kernel | Pybind name | Purpose | Potential use in Gemma4 |
|---|---|---|---|
| `embedding_layer` | `embedding_layer` | embedding lookup | Replace `VocabParallelEmbedding` |
| `gather_layer` | `gather_layer` | generic gather | Could replace embedding-like gathers |
| `matmul_layer` | `matmul_layer` | `output = lhs @ rhs` (m,k) @ (k,n) | Replace linear projections |
| `matmul_nz_layer` | `matmul_nz_layer` | same, RHS already NZ | Replace linear projections |
| `matmul_bias_nz_layer` | `matmul_bias_nz_layer` | `output = lhs @ rhs + bias` | Replace biased linear |
| `matmul_weight_transpose_layer` | `matmul_weight_transpose_layer` | weight layout conversion | Prepare linear weights |
| `batch_matmul_layer` | `batch_matmul_layer` | batched matmul | Attention QK/VO if needed |
| `batch_matmul_trans_v_layer` | `batch_matmul_trans_v_layer` | batched matmul with V transposed | Attention |
| `batch_matmul_causual_layer` | `batch_matmul_causual_layer` | causal batched matmul | Attention |
| `batch_matmul_qk_trans_causual_layer` | `batch_matmul_qk_trans_causual_layer` | QK^T causal | Attention |
| `softmax_layer` | `softmax_layer` | softmax | Attention / sampling |
| `silu_mul_layer_vllm` | `silu_mul_layer_vllm` | `silu(x[0:h]) * x[h:2h]` | SwiGLU MLP (not used because Gemma uses GELU) |
| `partial_rope_layer` | `partial_rope_layer` | partial RoPE with factor | Could replace manual partial RoPE logic |
| `sliding_window_flash_attn_layer` | `sliding_window_flash_attn_layer` | non-paged sliding flash attn | Not needed (page attn used) |
| `flash_attn_layer` | `flash_attn_layer` | non-paged flash attn | Not needed |
| `convert_awq_4bit_qweight` | `convert_awq_4bit_qweight` | AWQ weight layout helper | Quantization |

---

## 6. Weight Layout Notes for Linear Replacement

The custom `matmul_layer` kernel does **not** use the same weight layout as PyTorch `nn.Linear`.  From `matmul_layer.cpp`:

- LHS (activation) is loaded as ND → NZ with `copy_gm_to_cbuf_multi_nd2nz_b16`.
- RHS (weight) is loaded with `copy_gm_to_cbuf_multi_nd2nz_b16` and then fed to `load_cbuf_to_cb(..., inc=1)`, which means it expects the weight to be **transposed** in a specific NZ format relative to standard PyTorch `[out_features, in_features]`.

To use `matmul_layer` for a PyTorch linear of shape `[out_features, in_features]`:

1. Option A: At load time, transpose the weight to `[in_features, out_features]` and then call `matmul_weight_transpose_layer` once to produce the layout the kernel wants.
2. Option B: Modify the kernel or add a wrapper that accepts standard PyTorch layout and performs the transpose inside the kernel (cost: extra memory traffic).

`matmul_nz_layer` expects the weight already in the final NZ layout, so it is the fastest option after a one-time conversion.

`matmul_bias_nz_layer` additionally adds a bias vector — useful for `ColumnParallelLinear` / `ReplicatedLinear` with bias.

---

## 7. Recommended Next Steps

The highest-impact replacements to eliminate PyTorch builtin TBE ops are:

1. **Embedding lookups** (`VocabParallelEmbedding` and `embed_tokens_per_layer`) → `embedding_layer`.
2. **Scalar multiplications** (normalizer, scale factors) → `mul_scalar_layer`.
3. **Residual/ple addition** (`per_layer_projection + per_layer_inputs`) → `add_layer`.
4. **Linear projections** (QKV, O, gate_up, down, PLE gates/projections) → `matmul_layer`/`matmul_nz_layer` after weight layout conversion.
5. **Eliminate dynamic slices** inside attention loops by passing raw pointer offsets to `rope_standard_layer` and the page-attention kernels instead of `tensor[off:off+len]`.
6. **Pre-allocate zero buffer** for `torch.zeros_like(input_ids)` instead of creating it per call.

## 9. Practical Test Results

A standalone test script was created at `/ssd/test_existing_kernels.py` to verify that some of the existing custom kernels produce the same result as the PyTorch reference.

**Result:** `embedding_layer`, `add_layer`, `mul_layer`, and `mul_scalar_layer` all pass against PyTorch reference with BF16 inputs.

### Important finding: not contiguity — async TBE JIT race condition

Initial testing suggested that `add_layer`/`mul_layer` failed when inputs were created with `torch.randn(..., device='npu')` but worked after `x.contiguous()` or CPU → NPU round-trip.  Further isolation showed that **contiguity is not the issue**: the failing tensors already report `is_contiguous() == True` and identical strides to the working ones.

The real root cause is **asynchronous TBE JIT compilation** triggered by `torch.randn(..., device='npu')`.  When the tensor is created, torch_npu queues a JIT compile for the random-fill kernel in the background.  The tensor handle is returned immediately, but the underlying NPU memory is not yet correctly initialized when the next custom kernel (`add_layer`) is launched.  This produces garbage / NaN / huge values in the output.

**Evidence:**
- Fresh cache + `torch.randn(device='npu')` + immediate `add_layer` → fail.
- Fresh cache + `torch.randn(device='npu')` + `torch.npu.synchronize()` + `add_layer` → **pass**.
- Fresh cache + `torch.empty(device='npu')` + CPU `copy_()` + sync + `add_layer` → **pass**.
- CPU-allocated tensor + `.npu()` → **pass** (the `.npu()` transfer synchronizes implicitly or avoids the randn JIT path).

**Implication for the model:** whenever a PyTorch op that may JIT-compile is used before a custom kernel, insert an explicit `torch.npu.synchronize()` before the custom-kernel call.  Better yet, avoid PyTorch NPU ops entirely on the hot path.

### Tested kernels

| Kernel | Test shape | Max diff vs PyTorch (BF16) | Status |
|---|---|---|---|
| `embedding_layer` | (5, 32) | 0.0000 | ✓ OK |
| `add_layer` | (16, 16) | 0.0078 | ✓ OK |
| `mul_layer` | (16, 16) | 0.0139 | ✓ OK |
| `mul_scalar_layer` | (16, 32) | 0.0000 | ✓ OK |
| `add_layer` | (11, 13) | 0.0156 | ✓ OK |

---

## 10. Open Questions / Next Work

- Weight-layout conversion for `matmul_layer`/`matmul_nz_layer`: a reproducible helper is needed to convert standard PyTorch `[out_features, in_features]` weights to the layout the custom matmul kernels expect.
- `matmul_bias_nz_layer` should be evaluated for bias-present linear layers.
- Dynamic `tensor[off:off+len]` slices in attention should be replaced by raw pointer arithmetic to eliminate `StridedSliceD` JIT triggers.
- Ensure every PyTorch op that may JIT-compile is followed by `torch.npu.synchronize()` before a custom kernel consumes its output, or eliminate the PyTorch op entirely.

---

*Document generated by Claude Code for the vllm_ascend_orangepi project.*