# VLLM适配OrangePi
## 安装方式
[Release页面](https://github.com/lenLRX/vllm_ascend_orangepi/releases)下载wheel直接在香橙派上pip安装
## 使用方式
### 下载模型
vllm支持自动下载huggingface和modelscope的模型,默认使用huggingface可以通过环境变量改为modelscope
```
export VLLM_USE_MODELSCOPE=True
```
一般来说SD卡容量小，而且慢。建议通过环境变量将默认存储位置改为加装的固态硬盘。
```
export HF_HOME=/ssd/hf_home
export MODELSCOPE_CACHE=/ssd/hf_home
```
vllm也支持使用本地已经下载好的模型，直接设置绝对路径即可

> ⚠️ **当前版本可用性说明**：本 release 仅保证 Gemma4 系列可用（见下方
> 「Gemma4 支持」一节）。以下 Qwen2 系列模型（DeepSeek-R1-Distill-Qwen-1.5B、
> Qwen2.5-7B-Instruct-AWQ 等）在当前版本暂时不可用（能加载但输出异常），
> 正在修复中，将在后续版本恢复。

### OpenAI http api
```bash
# 自动下载模型
python -m vllm.entrypoints.openai.api_server --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
# 使用本地已下载模型
python -m vllm.entrypoints.openai.api_server --model /ssd/models/Qwen2.5-7B-Instruct-AWQ/
```
### python脚本
```python
from vllm import LLM, SamplingParams

sampling_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=16, seed=42)
llm = LLM(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

prompts = [
    "AI的未来是",
    "The weather today"
]

outputs = llm.generate(prompts, sampling_params)

for i, output in enumerate(outputs):
    print(f"Prompt {i+1}: {prompts[i]}")
    print(f"Generated: {output.outputs[0].text}\n")
```
### CherryStudio对接

<img width="1612" height="894" alt="image" src="https://github.com/user-attachments/assets/36796bb9-ddbc-4f9c-960e-a3cc071b730e" />
如果使用本地模型，"模型ID"填启动命令行的--model参数中给的路径

## Gemma4 支持 (GGUF Q4_0 / Q6_K / bf16)

支持在 Ascend 310B1 (Orange Pi AI Pro) 上运行 Gemma4 E2B/E4B：

- **bf16** safetensors 模型直接加载
- **GGUF Q4_0** 量化模型（含 Google QAT q4_0 gguf），自定义 CCE 融合反量化+矩阵乘内核（NZ 布局）
- **Q6_K lm_head** 融合反量化+矩阵乘内核（int8 布局，权重无损展开），由环境变量 `GEMMA4_LMHEAD_Q6K` 控制（默认开启，设 `0` 关闭）

GGUF 模型加载示例：
```python
from vllm import LLM, SamplingParams

model_dir = '/ssd/models/gemma-4-E2B-it-qat-q4_0-gguf'
llm = LLM(
    model=f'{model_dir}/gemma-4-E2B_q4_0-it.gguf',
    tokenizer=model_dir,
    trust_remote_code=True,
    dtype='float16', max_model_len=2048,
    gpu_memory_utilization=0.9, enforce_eager=True, block_size=64,
)
outputs = llm.generate(["The capital of France is"],
                       SamplingParams(temperature=0.0, max_tokens=32))
```

OpenAI http api 服务（GGUF Q4_0）：
```bash
python -m vllm.entrypoints.openai.api_server \
  --model /ssd/models/gemma-4-E2B-it-qat-q4_0-gguf/gemma-4-E2B_q4_0-it.gguf \
  --tokenizer /ssd/models/gemma-4-E2B-it-qat-q4_0-gguf \
  --served-model-name gemma-4-E2B-q4_0 \
  --trust-remote-code --dtype float16 --max-model-len 2048 \
  --gpu-memory-utilization 0.9 --enforce-eager --block-size 64 \
  --chat-template /ssd/models/gemma-4-E2B-it/chat_template.jinja \
  --port 8000
```

注意：GGUF 模型目录的 tokenizer 没有内嵌 chat template，使用
`/v1/chat/completions` 时必须通过 `--chat-template` 指定
（可复用 bf16 模型目录中的 `chat_template.jinja`，两者 tokenizer 一致），
否则 chat 接口会返回 400 错误。

另外，GGUF 目录还需放入 `generation_config.json`（内容同 bf16 模型，
关键是 `eos_token_id: [1, 106, 50]`）。vllm 对文件形式的模型会从
tokenizer 目录读取它；缺失时引擎只把 `<eos>`(1) 当作停止符，模型输出
`<turn|>`(106) 时不会停止，会陷入 `<turn|>` 无限重复。

解码性能（256 token 输入，贪心解码，Ascend 310B1，预热后）：

| 模型 | 解码速度 |
|---|---|
| gemma-4-E2B-it (bf16) | 5.2 tok/s (191 ms/tok) |
| gemma-4-E2B-it (GGUF Q4_0) | 11.8 tok/s (85 ms/tok) |

提示：每个进程首次贪心解码会触发一次 TBE 编译（argmax/log_softmax，约 50 秒），
之后命中磁盘缓存；压测前请先跑 1-2 个 token 预热。

内核源码在子模块 `/data/llm_simple/src/npu_ops`，构建产物部署到
`vllm/model_executor/layers/npu/`。详细优化与对比报告见：
[docs/q4_decode_baseline.md](docs/q4_decode_baseline.md)、
[docs/q4_decode_optimization_final.md](docs/q4_decode_optimization_final.md)、
[docs/bf16_vs_q4_decode_report.md](docs/bf16_vs_q4_decode_report.md)、
[docs/q6k_lm_head_kernel.md](docs/q6k_lm_head_kernel.md)。



<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/source/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/source/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://vllm.ai"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://discord.gg/jz7wjKhh6g"><b>Discord</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

---

*Latest News* 🔥
- [2024/11] We hosted [the seventh vLLM meetup](https://lu.ma/h0qvrajz) with Snowflake! Please find the meetup slides [here](https://docs.google.com/presentation/d/1e3CxQBV3JsfGp30SwyvS3eM_tW-ghOhJ9PAJGK6KR54/edit?usp=sharing).
- [2024/10] We have just created a developer slack ([slack.vllm.ai](https://slack.vllm.ai)) focusing on coordinating contributions and discussing features. Please feel free to join us there!
- [2024/10] Ray Summit 2024 held a special track for vLLM! Please find the opening talk slides from the vLLM team [here](https://docs.google.com/presentation/d/1B_KQxpHBTRa_mDF-tR6i8rWdOU5QoTZNcEg2MKZxEHM/edit?usp=sharing). Learn more from the [talks](https://raysummit.anyscale.com/flow/anyscale/raysummit2024/landing/page/sessioncatalog?tab.day=20241001&search.sessiontracks=1719251906298001uzJ2) from other vLLM contributors and users!
- [2024/09] We hosted [the sixth vLLM meetup](https://lu.ma/87q3nvnh) with NVIDIA! Please find the meetup slides [here](https://docs.google.com/presentation/d/1wrLGwytQfaOTd5wCGSPNhoaW3nq0E-9wqyP7ny93xRs/edit?usp=sharing).
- [2024/07] We hosted [the fifth vLLM meetup](https://lu.ma/lp0gyjqr) with AWS! Please find the meetup slides [here](https://docs.google.com/presentation/d/1RgUD8aCfcHocghoP3zmXzck9vX3RCI9yfUAB2Bbcl4Y/edit?usp=sharing).
- [2024/07] In partnership with Meta, vLLM officially supports Llama 3.1 with FP8 quantization and pipeline parallelism! Please check out our blog post [here](https://blog.vllm.ai/2024/07/23/llama31.html).
- [2024/06] We hosted [the fourth vLLM meetup](https://lu.ma/agivllm) with Cloudflare and BentoML! Please find the meetup slides [here](https://docs.google.com/presentation/d/1iJ8o7V2bQEi0BFEljLTwc5G1S10_Rhv3beed5oB0NJ4/edit?usp=sharing).
- [2024/04] We hosted [the third vLLM meetup](https://robloxandvllmmeetup2024.splashthat.com/) with Roblox! Please find the meetup slides [here](https://docs.google.com/presentation/d/1A--47JAK4BJ39t954HyTkvtfwn0fkqtsL8NGFuslReM/edit?usp=sharing).
- [2024/01] We hosted [the second vLLM meetup](https://lu.ma/ygxbpzhl) with IBM! Please find the meetup slides [here](https://docs.google.com/presentation/d/12mI2sKABnUw5RBWXDYY-HtHth4iMSNcEoQ10jDQbxgA/edit?usp=sharing).
- [2023/10] We hosted [the first vLLM meetup](https://lu.ma/first-vllm-meetup) with a16z! Please find the meetup slides [here](https://docs.google.com/presentation/d/1QL-XPFXiFpDBh86DbEegFXBXFXjix4v032GhShbKf3s/edit?usp=sharing).
- [2023/08] We would like to express our sincere gratitude to [Andreessen Horowitz](https://a16z.com/2023/08/30/supporting-the-open-source-ai-community/) (a16z) for providing a generous grant to support the open-source development and research of vLLM.
- [2023/06] We officially released vLLM! FastChat-vLLM integration has powered [LMSYS Vicuna and Chatbot Arena](https://chat.lmsys.org) since mid-April. Check out our [blog post](https://vllm.ai).

---
## About
vLLM is a fast and easy-to-use library for LLM inference and serving.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with **PagedAttention**
- Continuous batching of incoming requests
- Fast model execution with CUDA/HIP graph
- Quantizations: [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), INT4, INT8, and FP8.
- Optimized CUDA kernels, including integration with FlashAttention and FlashInfer.
- Speculative decoding
- Chunked prefill

**Performance benchmark**: We include a performance benchmark at the end of [our blog post](https://blog.vllm.ai/2024/09/05/perf-update.html). It compares the performance of vLLM against other LLM serving engines ([TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [SGLang](https://github.com/sgl-project/sglang) and [LMDeploy](https://github.com/InternLM/lmdeploy)). The implementation is under [nightly-benchmarks folder](.buildkite/nightly-benchmarks/) and you can [reproduce](https://github.com/vllm-project/vllm/issues/8176) this benchmark using our one-click runnable script.

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor parallelism and pipeline parallelism support for distributed inference
- Streaming outputs
- OpenAI-compatible API server
- Support NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, TPU, and AWS Neuron.
- Prefix caching support
- Multi-lora support

vLLM seamlessly supports most popular open-source models on HuggingFace, including:
- Transformer-like LLMs (e.g., Llama)
- Mixture-of-Expert LLMs (e.g., Mixtral)
- Embedding Models (e.g. E5-Mistral)
- Multi-modal LLMs (e.g., LLaVA)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with `pip` or [from source](https://vllm.readthedocs.io/en/latest/getting_started/installation.html#build-from-source):

```bash
pip install vllm
```

Visit our [documentation](https://vllm.readthedocs.io/en/latest/) to learn more.
- [Installation](https://vllm.readthedocs.io/en/latest/getting_started/installation.html)
- [Quickstart](https://vllm.readthedocs.io/en/latest/getting_started/quickstart.html)
- [Supported Models](https://vllm.readthedocs.io/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [CONTRIBUTING.md](./CONTRIBUTING.md) for how to get involved.

## Sponsors

vLLM is a community project. Our compute resources for development and testing are supported by the following organizations. Thank you for your support!

<!-- Note: Please sort them in alphabetical order. -->
<!-- Note: Please keep these consistent with docs/source/community/sponsors.md -->

- a16z
- AMD
- Anyscale
- AWS
- Crusoe Cloud
- Databricks
- DeepInfra
- Dropbox
- Google Cloud
- Lambda Lab
- NVIDIA
- Replicate
- Roblox
- RunPod
- Sequoia Capital
- Skywork AI
- Trainy
- UC Berkeley
- UC San Diego
- ZhenFund

We also have an official fundraising venue through [OpenCollective](https://opencollective.com/vllm). We plan to use the fund to support the development, maintenance, and adoption of vLLM.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):
```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

* For technical questions and feature requests, please use Github issues or discussions.
* For discussing with fellow users, please use Discord.
* For coordinating contributions and development, please use Slack.
* For security disclosures, please use Github's security advisory feature.
* For collaborations and partnerships, please contact us at vllm-questions AT lists.berkeley.edu.
