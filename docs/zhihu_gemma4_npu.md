# 在香橙派上跑 Gemma4：vLLM 适配昇腾 310B 的内核开发实录

> 硬件：Orange Pi AI Pro（昇腾 310B1，1 个 AI Core @ 1224 MHz，L2 4MB，实测 DDR 读带宽 28.8 GB/s）
> 软件：vLLM（fork）+ 自研算子库（kernel 以预编译形式随 wheel 发布）

## 最终效果

先把结果放在前面。Gemma4 E2B（GGUF Q4_0 量化）在香橙派 AI Pro 上：

| 指标 | 数值 |
|---|---|
| 端到端解码速度（预热后） | **85 ms/token（11.8 tok/s）** |
| 同模型 bf16 解码 | 191 ms/token（5.2 tok/s） |
| Q4_0 解码相对 bf16 | **约 2.2 倍** |
| 相对优化前的 Q4_0 基线（206 ms/token） | 约 2.4–2.7 倍（视统计口径） |
| 预填 464 token | 1.35 s（优化前 2.52 s） |
| 正确性 | 贪心 256 token 生成完全连贯；kernel 单元测试 17/17、Q6_K 7/7 通过 |

要达到这个数字，框架适配之外，热点算子几乎全部要自己实现。这篇文章记录一下过程：先简单介绍 Gemma4 的网络结构，然后列出适配 NPU 需要开发的 kernel 清单，挑几个重点 kernel 讲讲实现思路，最后分析当前的性能瓶颈，并展望下一步的提升空间。

## 一、Gemma4 网络结构简介

Gemma4 E2B 是一个 35 层的 decoder-only 模型，hidden size 1536，FFN 中间维 6144，词表 262144（26 万，非常大）。它有几个在算子适配时特别"折腾人"的特性：

**1. 混合注意力：滑窗 + 全局**

35 层里 28 层是滑窗注意力（sliding window=512，head_dim=256，本地 RoPE，频率基 10000），7 层是全局注意力（head_dim=512，RoPE 频率基 1M）。两种层交替排布。这意味着分页注意力内核至少要覆盖两种 head_dim，滑窗层还要额外处理窗口掩码。

**2. 极端 GQA：8 个 Q 头共享 1 个 KV 头**

num_attention_heads=8，num_key_value_heads=1，group_size=8。KV cache 很小，对低带宽设备友好，但 attention kernel 必须按 GQA 展开。

**3. KV 共享层**

后 20 层（num_kv_shared_layers=20）不写自己的 KV cache，而是复用第 13/14 层写入的 KV。实现上这些层只算 q，k/v 从共享 cache 读。模型加载时要把共享层的 k/v 投影权重特殊处理，attention 里也要区分"写 cache 的层"和"只读的层"。

**4. PLE（Per-Layer Embedding）**

除了主 embedding，还有一路 256 维的 per-layer 输入 embedding，每层各带两个小投影（256→1536、1536→256）和一个 gate。于是每层多出 4-5 个小算子——后文会看到，这些小算子是 host 侧开销的主要来源。

**5. QK-Norm 与 logit softcapping**

Q/K 在 attention 前各过一个 RMSNorm；最终 logits 有一个 30.0 的 softcap（tanh 缩放）。

**6. 权重共享的 lm_head**

lm_head 与输入 embedding 共享同一张 [262144, 1536] 的表。这张表在 bf16 下是 805 MB——每解码一个 token 都要完整读一遍，后面会看到它是最大的单点瓶颈。

量化方面，官方提供了 QAT 的 GGUF Q4_0 权重（embedding/lm_head 保留 Q6_K 保精度），也是我们部署的目标格式。

## 二、适配 NPU 需要开发哪些 kernel

vLLM 的 Python 侧适配（模型定义、调度、采样）之外，热点全部要落到自研 kernel 上。算子库总计约 9000 行，完整清单如下（按类别，代码量为约数）：

| 类别 | kernel | 代码量（约） | 说明 |
|---|---|---|---|
| 矩阵乘 | dense matmul（NZ 布局） | 1000 行 | bf16/fp16 稠密权重，权重加载时重排为 NZ 分形布局 |
| | matmul_weight_transpose | 70 行 | 权重转置工具 |
| | matmul_gguf_q4_0 | 330 行 | Q4_0 融合反量化+矩阵乘（核心 kernel） |
| | matmul_gguf_q6_k_i8 | 310 行 | Q6_K int8 融合反量化+矩阵乘（lm_head 专用） |
| | matmul_nz_awq_4bit(_bias) | 2200 行 | AWQ 4bit 权重支持 |
| | convert_*（host 工具） | — | Q4_0/Q6_K/AWQ 原始字节 → NZ 布局转换，含于对应文件 |
| 注意力 | page_attn_gqa（128/256/512） | 760 行 | 分页 KV cache 的 GQA 注意力，含 causal 变体 |
| | page_attn_dim256 | 90 行 | Gemma4 滑窗层专用入口（设备代码与上者共享） |
| | page_attn_gqa_dim512 | 670 行 | Gemma4 全局层专用（head_dim=512） |
| | flash_attn / sliding_window_flash_attn | 1500 行 | 连续 KV 的 prefill 路径 |
| | batch_matmul 系列 | 1100 行 | 因果/转置等批量矩阵乘 |
| 归一化 | rmsnorm | 70 行 | RMSNorm |
| | qkv_norm_fused | 70 行 | Q/K/V 三分支 RMSNorm 融合为一次发射 |
| | add | 130 行 | 残差相加 |
| 激活 | gated_gelu | 250 行 | GeGLU（gate ⊙ gelu(up)），支持批量行 |
| | silu_mul / gelu / mul / mul_scalar | 200 行 | 各类逐元素算子 |
| 位置编码 | rope / rope_qk / partial_rope / rope_standard | 500 行 | RoPE 及其 Q+K 批量融合变体 |
| 其他 | split_qkv | 90 行 | QKV 投影输出按 q/k/v 切分 |
| | embedding / gather / fill / softmax / ple_slice / softcap | 700 行 | 查表、收集、填充、归一化、切片、logit 缩放 |
| | dequant_only_q4_0 / q6_k_i8 | 300 行 | 纯反量化 kernel（用于 bring-up 与正确性校验） |

粗数一下，解码一个 token 要发射 600 多次 kernel（35 层 × 每层十几二十个算子），这个数字对后面的瓶颈分析很重要。

## 三、重点 kernel 实现方式

### 3.1 Q4_0 融合反量化+矩阵乘：一切设计围绕 DDR 带宽

**问题**：Q4_0 权重每个 32 元素块存 16 字节的 4-bit 数据加一个 fp16 scale。直接的做法是先反量化成 fp16 再做稠密矩阵乘——但这样每 token 要多读一遍 fp16 中间结果，带宽消耗直接翻倍。

**思路**：把反量化塞进矩阵乘的数据通路里，权重始终以 4-bit 形式从 DDR 流入，在芯片内部展开后立即参与乘加：

1. **加载时重排（一次性）**。GGUF 文件里的 Q4_0 块是行优先排列的。加载模型时在 host 侧把它重排成 NZ 分形布局（矩阵乘单元喜爱的数据排布），scale 也按计算顺序重排为每 32 元素一组。这个转换只做一次，之后每个 token 的权重流就是纯粹顺序读的 4-bit 字节流。
2. **流水化执行**。片上 UB 做乒乓双缓冲：DMA 引擎搬运下一块权重的同时，向量单元把当前块的 4-bit 数据展开成 fp16 并乘上对应 scale，随即送入乘加通路；结果块累积完成后写回 DDR。搬数、解量化、乘加三条流水线互相覆盖。
3. **tile 参数贴合硬件**。K 方向 tile 取 128、N 方向 tile 取 256，让每个 N 分块上累加结果的写回次数减半——这个一行改动把所有投影形状都推到了接近带宽上限的位置。
4. **加载期 shard 融合**。q/k/v 三个投影（以及 gate/up 两个投影）在 GGUF 里的原始字节按行拼接后仍然是一份合法的 Q4_0 权重，于是加载时直接拼成一张大权重、一次重排，运行时一次矩阵乘。每 token 的矩阵乘发射次数从 245 降到 140，还省掉了运行时的拼接算子。

**效果**：所有投影形状达到 24.3–26.4 GB/s（实测纯读带宽上限 28.8 GB/s 的 85–92%）。每 token 权重流量从 bf16 的 2631 MB 降到 740 MB，矩阵乘部分 92 ms → 30.8 ms，约 3 倍。剩余的差距来自向量单元解量化占用的时间——这是融合反量化必须付出的成本。

### 3.2 Q6_K int8 lm_head：用 37% 的字节换一条能走通的向量流水线

**问题**：lm_head 是 [262144, 1536] 的大矩阵，与 embedding 共享权重。QAT 模型里它是 Q6_K 格式：每 256 个权重 210 字节（4-bit 低位 + 2-bit 高位 + int8 子块 scale + 一个 fp16 总 scale）。

最初的想法是原样在 kernel 里解 packed 6-bit。算了一下发现走不通：2-bit 高位字段的抽取非常费向量指令，解 512 个权重要约 38 条向量指令，而这颗芯片的向量单元吞吐大约 1.2 条/周期——lm_head 的瓶颈会落在向量单元上，约 24–25 ms，远高于 15 ms 的带宽理论下限。

**思路**：既然瓶颈是指令数而不是字节数，那就反过来——**加载时把 6-bit 值无损展开成 int8**（Q6_K 的值域正好可以被 int8 精确表示，零精度损失），scale 预乘成 fp16 按组排好。这样 kernel 里的反量化只剩"读 int8 → 转 fp16 → 乘 scale"三步，每 512 个权重约 14 条向量指令，不再是瓶颈。

代价是多读 37% 的字节（453 MB vs packed 的 330 MB）——但反正到不了带宽下限以下，字节多一点没关系；真正贵的是向量指令。

**效果**：lm_head 从 fp16 稠密的 805 MB / 27.7 ms 降到 453 MB / 15.9 ms，实测 28.4 GB/s，基本达到 28.8 GB/s 的实测带宽上限。每 token 省下约 12 ms。

### 3.3 分页 GQA 注意力：页表、共享 KV 与编译期掩码

attention kernel 按页表（block table）寻址 KV cache：host 侧维护每个序列的物理块列表，kernel 按 (块号, 块内偏移) 取 K/V。几个要点：

- **GQA 展开**：8 个 Q 头共享 1 个 KV 头，kernel 内按 group_size=8 广播 K/V。
- **causal 掩码做成编译期开关**：同一份 kernel 源码编译出 causal / no-causal 两个变体，运行时按 bool 参数分发，避免在热点循环里做分支。
- **滑窗变体**：Gemma4 的 28 个滑窗层（window=512）在取 KV 时直接按窗口下界裁剪页表范围，head_dim=256 与全局层的 512 各自有专门实例，寄存器/UB 布局各自最优。
- **host 侧配合**：页表内容按内容缓存，只有内容变化时才重新做一次 H2D 拷贝——从每 token 35 次降到大约每 64 token 一次；每 token 35 次的 device 同步也因此被删掉。这一项是 host 侧最大的单项收益之一。

### 3.4 小算子与 host 侧优化：被低估的另一半

单看 device 时间，RMSNorm、RoPE、残差加、gelu 这些小算子每个只有几 µs，但它们每层十几个、每 token 600+ 次发射，每次发射的 Python + 下发成本约 18–27 µs——device 算得再快，也架不住 host 发得慢。围绕这个做了一组"无聊但有效"的优化：

- **流句柄缓存**：取默认 stream 的调用每 token 约 3000 次，每次要 ~120 µs 走一遍框架的设备索引解析。缓存结果后单项就把端到端从 168 拉到 96 ms/token，是全部优化里最大的一刀。
- **小算子批量/融合**：RoPE 的 q、k 合并为一次调用；Q/K/V 三分支 RMSNorm 融合成一次发射；gated_gelu 支持一次算多行（prefill 时每层曾要 464 次 Python 调用）。
- **缓冲区池化**：attention 输出、残差、norm/gelu 中间结果按 (层, M) 复用预分配 buffer，消除每 token 的重复分配。
- **on-device 贪心采样**：argmax + log_softmax 在 NPU 上算，只把 token id（和精确的 logprob）拷回 host，省掉每 token 整词表 logits 的 D2H 拷贝和约 10 ms 的 CPU softmax；遇到惩罚项、min_tokens、logprobs 请求等会改变 argmax 的情况自动回退 CPU 采样器。

## 四、当前性能瓶颈分析

### 4.1 先看上限在哪里

解码是自回归的：每生成一个 token，都要把全部权重从 DDR 完整读一遍（工作集远超 4 MB L2，没有复用）。所以解码性能的上限就是 **权重字节数 ÷ DDR 带宽**。

这颗 310B1 实测：纯读带宽 28.8 GB/s，读写拷贝 42.5 GB/s。

Q4_0 模型每 token 的流量：

| 部分 | 字节/token | device 时间 | 有效带宽 |
|---|---|---|---|
| 35 层矩阵乘（Q4_0 权重） | 740 MB | 30.8 ms | ~25.5 GB/s |
| lm_head（Q6_K int8） | 453 MB | 15.9 ms | 28.4 GB/s |
| 注意力（KV cache 读写） | ~12 MB | 2.8 ms | — |
| 小算子（norm/rope/gelu/...） | 几 MB | ~2 ms | — |
| 采样 | 2 MB | 0.2 ms | — |

device 合计约 52–61 ms，而端到端 wall 是 85–102 ms/token。差距在哪？

### 4.2 wall 与 device 的对账

| 构成 | 时间/token |
|---|---|
| 矩阵乘 device | 30.8 ms |
| lm_head device | 15.9 ms |
| 注意力 + 小算子 device | ~5 ms |
| **device-busy 合计** | **~52 ms** |
| 小算子发射的 host 缺口（600+ 次 × 18–27 µs 发射率） | ~10 ms |
| lm_head 串行收尾 + 采样同步 | 若干 ms |
| 引擎记账（调度、detokenize、prepare） | ~5–10 ms |
| **端到端 wall** | **85–102 ms** |

三个明确的剩余瓶颈：

1. **小算子发射率**。矩阵乘密集区已经能完全掩盖 host 时间；但每层十几个小算子，device 几 µs 就排空一个，Python 以 18–27 µs/个的速度根本喂不饱。要根治只有两条路：更多融合（add+rmsnorm、gelu+mul 之类）减少发射次数，或者走图回放 / C++ 派发绕过逐算子的 Python——工程量都明显更大。
2. **lm_head 的串行收尾**。lm_head 本身已接近带宽上限（28.4/28.8 GB/s），但它和采样是串行收尾的，这段时间 host 是空等的。要掩盖它需要引擎级流水（把下一步的准备工作和当前步的 lm_head 重叠），属于 vLLM 核心改造。
3. **lm_head 仍是最大的单矩阵**。它已经占了 Q4 模型总流量的大头。进一步把它量化到 Q4_0 可以再省 ~7 ms（453 MB → 226 MB），但 logits 会有约 1% 的偏移——这是精度/产品取舍，不是纯优化，我们没有默认开启。

### 4.3 为什么矩阵乘做不到 4 倍

立项时定过"解码矩阵乘 4 倍加速"的目标，最后实测只有 1.19 倍（34.4 → 28.8 ms/tok），但所有形状都达到了实测带宽上限的 85–92%。原因很简单：4 倍意味着约 84 GB/s 的持续读带宽，是这颗芯片实测峰值的 2–3 倍——**物理上就够不着**。当 kernel 已经贴近带宽上限时，剩下的唯一出路是继续减少字节数（更激进的量化），而不是再"优化"kernel。

## 五、展望：MTP 是解码性能的下一个倍增器

到这里，矩阵乘和 lm_head 都已贴近带宽上限，继续压榨 kernel 本身的空间很小了。但解码的带宽瓶颈恰恰给 **MTP（Multi-Token Prediction，多 token 预测/推测解码）** 留出了几乎是"免费"的提升空间，而且这块芯片是最适合吃这波红利的情形：

**为什么适合**：解码时每步都要完整流读全部权重（740 MB + 453 MB），但 M=1 的乘加计算量极小，AI Core 的算力大量闲置。MTP 的思路是每步用草稿（draft）一次提出 K 个候选 token，主模型做一次 M=K 的并行验证。**关键点是：验证 K 个 token 需要读取的权重字节数和验证 1 个完全一样**——权重只流读一遍，多出来的只是 M 从 1 变 K 的乘加，而这些本来闲着。也就是说，在这颗带宽瓶颈的芯片上，一次验证 K 个候选几乎不增加单步耗时，但只要平均接受 2–3 个 token，有效解码速度就直接翻 2–3 倍。GPU 上做 MTP 还要担心验证步的算力开销，在这里几乎是纯白捡。

**预期收益**：当前单步约 85 ms（权重流读 ~50 ms + host/引擎开销）。若平均每步接受 2.5 个 token，有效速度约 85/2.5 ≈ **34 ms/token（~29 tok/s）**，相对 bf16 基线是 5 倍以上。即便接受率只有 2，也有 40 ms/token 量级。相比继续抠 kernel，这是数量级更大的一档。

**要做的功课**：

- **草稿来源**：Gemma4 官方没有放出 MTP 头，需要一个轻量 draft（小模型、n-gram、或者自训练一个 MTP/EAGLE 头）。草稿质量直接决定接受率，是收益大小的核心变量。
- **引擎支持**：vLLM 的 speculative decoding 路径要在我们的 NPU runner 上打通，包括验证用的 M=K paged attention、KV cache 的回滚管理——按当前 kernel 基础，主要是适配工作而非新开发。
- **小算子发射率问题会被放大**：验证步算子数量随 K 不变，但 host 发射次数也不变，所以 4.2 节说的发射瓶颈占的比例会更高，融合/图回放的价值也会同步放大。

## 六、总结

在 28.8 GB/s 的板子上跑 2B 模型，打法其实就一句话：**把每个字节的搬运都算清楚，把每次 kernel 发射都算清楚**。字节侧靠量化格式 + 融合反量化 kernel（Q4_0 NZ 重排、Q6_K int8 展开）；发射侧靠融合、批处理、池化和能缓存就缓存。当矩阵乘和 lm_head 都贴近带宽上限之后，瓶颈就转移到了 600 多次小算子发射和引擎的串行收尾上——再往后，MTP 用几乎免费的并行验证把带宽瓶颈变成收益来源，是下一步最值得做的事情。

vLLM 适配代码已开源（kernel 以预编译库形式随 pip wheel 一起发布，在香橙派上直接安装即可复现本文数字），欢迎交流。
