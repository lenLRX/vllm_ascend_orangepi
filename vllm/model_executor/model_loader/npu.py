"""Utilities for selecting and loading nPU models."""
import copy
import importlib
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import ModelConfig, ParallelConfig, SchedulerConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import (CompletionSequenceGroupOutput, Logprob,
                           SequenceOutput)


# Models supported by Neuron.
_NPU_SUPPORTED_MODELS: Dict[str, Tuple[str, str]] = {
    "LlamaForCausalLM": ("vllm.model_executor.models.llama_npu", "LlamaForCausalLM"),
}


class NPUCausalLM(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 on_device_sampling_disabled: bool = False) -> None:
        super().__init__()
        self.config = config
        self.logits_processor = LogitsProcessor(config.vocab_size,
                                                logits_as_input=True)

        self.on_device_sampling_disabled = on_device_sampling_disabled
        if self.on_device_sampling_disabled:
            # Use default sampler
            self.sampler = Sampler()

        # Lazy initialized
        self.model: nn.Module

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_block_ids: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.model(input_ids,
                            cache_ids=positions,
                            start_ids=input_block_ids)
        return logits

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(None, hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:

        if self.on_device_sampling_disabled:
            next_tokens = self.sampler(logits, sampling_metadata)
            return next_tokens

        # On-device sampling outputs the token ids directly.
        sampled_token_ids = logits.flatten()
        next_tokens = []
        sample_idx = 0
        for seq_group in sampling_metadata.seq_groups:
            samples = []
            for seq_id in seq_group.seq_ids:
                token_id = sampled_token_ids[sample_idx].item()
                samples.append(
                    SequenceOutput(parent_seq_id=seq_id,
                                   output_token=token_id,
                                   logprobs={token_id: Logprob(token_id)}))
                sample_idx += 1
            next_tokens.append(
                CompletionSequenceGroupOutput(samples=samples,
                                              prompt_logprobs=None))

        return SamplerOutput(outputs=next_tokens)

    def load_weights(self, model_name_or_path: str, **kwargs):
        arch = _get_model_architecture(self.config)
        npu_module_path, npu_model_cls_name = (
            _NPU_SUPPORTED_MODELS[arch])
        npu_module = importlib.import_module(npu_module_path)
        npu_model_cls = getattr(npu_module, npu_model_cls_name)

        self.model = npu_model_cls.from_pretrained(model_name_or_path,
                                                       **kwargs)


def _get_model_architecture(config: PretrainedConfig) -> str:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _NPU_SUPPORTED_MODELS:
            return arch
    raise ValueError(
        f"Model architectures {architectures} are not supported on NPU now "
        f"for now. Supported architectures: "
        f"{list(_NPU_SUPPORTED_MODELS.keys())}")


def _get_buckets(env: str, default_value: List[int]) -> List[int]:
    env_value = os.getenv(env)
    if env_value is None:
        return default_value
    buckets_remove_empty = filter(
        lambda x: x is not None and len(x.strip()) > 0, env_value.split(","))
    buckets_int = map(int, buckets_remove_empty)
    buckets_list = list(buckets_int)
    return buckets_list


def _get_neuron_on_device_generation_config(model_config: ModelConfig):
    if not _is_neuron_on_device_sampling_disabled(model_config):
        return copy.deepcopy(model_config.neuron_sampling_params)
    return None


def _get_neuron_config_after_override(default_neuron_config,
                                      overridden_neuron_config):
    from transformers_neuronx.config import NeuronConfig
    overridden_neuron_config = overridden_neuron_config or {}
    default_neuron_config.update(overridden_neuron_config)
    return NeuronConfig(**default_neuron_config)


def get_npu_model(model_config: ModelConfig,
                     parallel_config: ParallelConfig,
                     scheduler_config: SchedulerConfig) -> nn.Module:

    # Create a model instance.
    model = NPUCausalLM(
        model_config.hf_config,
        True)

    # Load the weights from the cached or downloaded files.
    model.load_weights(model_config.model,
                       tp_degree=parallel_config.tensor_parallel_size,
                       context_length_estimate=scheduler_config.max_model_len,
                       n_positions=scheduler_config.max_model_len,
                       batch_size=scheduler_config.max_num_seqs)

    return model.eval()

