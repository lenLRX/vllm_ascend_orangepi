import os
from dataclasses import dataclass
from importlib.util import find_spec
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set,
                    Tuple, Type, TypeVar, Union)

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader import get_model
from vllm.core.scheduler import SchedulerOutputs

from vllm.multimodal import (MULTIMODAL_REGISTRY, BatchedTensorInputs,
                             MultiModalKwargs)
from vllm.sequence import IntermediateTensors, SequenceGroupMetadata
from vllm.utils import is_pin_memory_available, make_tensor_with_pad
from vllm.worker.model_runner_base import ModelRunnerBase, ModelRunnerInputBase
from vllm.worker.model_runner_base import (
    ModelRunnerBase, ModelRunnerInputBase, ModelRunnerInputBuilderBase,
    _add_attn_metadata_broadcastable_dict,
    _add_sampling_metadata_broadcastable_dict,
    _init_attn_metadata_from_tensor_dict,
    _init_sampling_metadata_from_tensor_dict, dump_input_when_exception)

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

logger = init_logger(__name__)

@dataclass(frozen=False)
class NPUAttentionMetadata:
    offsets: Optional[List[int]] = None
    seq_lens: Optional[List[int]] = None


@dataclass(frozen=True)
class ModelInputForNPU(ModelRunnerInputBase):
    """
    This base class contains metadata needed for the base model forward pass
    but not metadata for possible additional steps, e.g., sampling. Model
    runners that run additional steps should subclass this method to add
    additional fields.
    """
    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    seq_lens: Optional[List[int]] = None
    query_lens: Optional[List[int]] = None
    #lora_mapping: Optional["LoRAMapping"] = None
    #lora_requests: Optional[Set[LoRARequest]] = None
    attn_metadata: Optional["AttentionMetadata"] = None
    #prompt_adapter_mapping: Optional[PromptAdapterMapping] = None
    #prompt_adapter_requests: Optional[Set[PromptAdapterRequest]] = None
    multi_modal_kwargs: Optional[BatchedTensorInputs] = None
    request_ids_to_seq_ids: Optional[Dict[str, List[int]]] = None
    finished_requests_ids: Optional[List[str]] = None
    virtual_engine: int = 0
    async_callback: Optional[Callable] = None
    #seq_group_metadata_list: Optional[List[SequenceGroupMetadata]] = None
    scheduler_outputs: Optional[SchedulerOutputs] = None
    sampling_metadata: Optional["SamplingMetadata"] = None


    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
            "lora_requests": self.lora_requests,
            "lora_mapping": self.lora_mapping,
            "multi_modal_kwargs": self.multi_modal_kwargs,
            "prompt_adapter_mapping": self.prompt_adapter_mapping,
            "prompt_adapter_requests": self.prompt_adapter_requests,
            "virtual_engine": self.virtual_engine,
            "request_ids_to_seq_ids": self.request_ids_to_seq_ids,
            "finished_requests_ids": self.finished_requests_ids,
        }
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)
        _add_sampling_metadata_broadcastable_dict(tensor_dict,
                                                  self.sampling_metadata)

        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
        cls,
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ):
        if attn_backend is not None:
            tensor_dict = _init_attn_metadata_from_tensor_dict(
                attn_backend, tensor_dict)
        return cls(**tensor_dict)

    # Exclude `async_callback` to be able to pickle this object
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["async_callback"]
        return state

    # TODO: What happens when we depickle this object?
    # How can we update this callback to properly pass it to the engine?
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__dict__.update({'async_callback': None})



class NPUModelRunner(ModelRunnerBase[ModelInputForNPU]):

    def __init__(
        self,
        vllm_config: VllmConfig,
    ):
        ModelRunnerBase.__init__(self, vllm_config)
        model_config = self.model_config
        if model_config is not None and model_config.get_sliding_window():
            logger.warning("Sliding window is not supported on NPU. "
                           "The model will run without sliding window.")
        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.multi_modal_input_mapper = self.mm_registry \
            .create_input_mapper(self.model_config)

        # Lazy initialization.
        self.model: nn.Module  # initialize after load_model.

        self._on_device_sampling_disabled = 0

        self._previous_batch_request_ids: List[str] = []


    def load_model(self) -> None:
        logger.info(f"Starting to load model {self.model_config.model} ...")
        logger.info(f"load_model config: {self.model_config}")
        self.model = get_model(vllm_config=self.vllm_config).npu()

        
    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int],
               BatchedTensorInputs]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        input_offsets: List[int] = []
        input_lengths: List[int] = []

        seq_lens: List[int] = []
        multi_modal_kwargs_list: List[MultiModalKwargs] = []
        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            seq_len = len(prompt_tokens)
            seq_lens.append(seq_len)

            input_offsets.append(0)
            input_lengths.append(seq_len)

            input_tokens.append(prompt_tokens)
            input_positions.append(list(range(seq_len)))

            assert seq_group_metadata.block_tables is not None
            block_table = seq_group_metadata.block_tables[seq_id]
            #assert len(block_table) == 1

            mm_data = seq_group_metadata.multi_modal_data
            if mm_data:
                if self.mm_registry.has_processor(self.model_config):
                    mm_kwargs = mm_data
                else:
                    mm_kwargs = self.multi_modal_input_mapper(
                        mm_data,
                        seq_group_metadata.mm_processor_kwargs,
                    )

                multi_modal_kwargs_list.append(mm_kwargs)

        max_seq_len = max(seq_lens)
        assert max_seq_len > 0
        assert len(input_tokens) == 1
        input_tokens = make_tensor_with_pad(input_tokens,
                                            pad=0,
                                            max_len=max_seq_len,
                                            dtype=torch.long,
                                            device=self.device)
        input_positions = make_tensor_with_pad(input_positions,
                                               pad=0,
                                               max_len=max_seq_len,
                                               dtype=torch.long,
                                               device=self.device)


        multi_modal_kwargs = MultiModalKwargs.batch(multi_modal_kwargs_list)

        return (input_tokens, input_positions, input_offsets, input_lengths, seq_lens,
                multi_modal_kwargs)

    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        input_offsets: List[int] = []
        input_lengths: List[int] = []
        context_lens: List[int] = []

        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt

            seq_ids = list(seq_group_metadata.seq_data.keys())

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append([generation_token])

                seq_len = seq_data.get_len()
                position = seq_len - 1
                input_positions.append([position])
                context_lens.append(seq_len)

                input_offsets.append(position)
                input_lengths.append(1)
        assert len(input_tokens) == 1
        input_tokens = make_tensor_with_pad(input_tokens,
                                            pad=0,
                                            max_len=1,
                                            dtype=torch.long,
                                            device=self.device)
        input_positions = make_tensor_with_pad(input_positions,
                                               pad=0,
                                               max_len=1,
                                               dtype=torch.long,
                                               device=self.device)
        context_lens = torch.tensor(context_lens,
                                    dtype=torch.int,
                                    device=self.device)

        return input_tokens, input_positions, input_offsets, input_lengths

    def make_model_input_from_broadcasted_tensor_dict(
            self, tensor_dict: Dict[str, Any]) -> ModelInputForNPU:
        return ModelInputForNPU.from_broadcasted_tensor_dict(tensor_dict)

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None
    ) -> ModelInputForNPU:
        multi_modal_kwargs = None
        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        is_prompt = seq_group_metadata_list[0].is_prompt
        # Prepare input tensors.
        if is_prompt:
            (input_tokens, input_positions, input_offsets, input_lengths, seq_lens,
             multi_modal_kwargs
             ) = self._prepare_prompt(seq_group_metadata_list)
        else:
            (input_tokens, input_positions,
             input_offsets, input_lengths) = self._prepare_decode(seq_group_metadata_list)
            seq_lens = None
        sampling_metadata = SamplingMetadata.prepare(
            seq_group_metadata_list,
            seq_lens,
            seq_lens,
            "cpu", #self.device,
            self.pin_memory,
            generators=self.get_generators(finished_requests_ids))
        sampling_metadata.selected_token_indices = sampling_metadata.selected_token_indices.npu()
        attn_metadata = NPUAttentionMetadata(offsets=input_offsets, seq_lens=input_lengths)

        return ModelInputForNPU(input_tokens=input_tokens,
                                input_positions=input_positions,
                                sampling_metadata=sampling_metadata,
                                attn_metadata=attn_metadata,
                                multi_modal_kwargs=multi_modal_kwargs)

    def _update_npu_sampling_params(self,
                                       sampling_metadata: SamplingMetadata):
        current_sampling_params = self.model_config.sampling_params
        assert current_sampling_params is not None, (
            f"Failed to update sampling_params, "
            f"current sampling params is {current_sampling_params}")

        top_k = current_sampling_params.top_k
        top_p = current_sampling_params.top_p
        temperature = current_sampling_params.temperature
        for index, sequence_group_to_sample in enumerate(
                sampling_metadata.seq_groups):
            top_k[index] = sequence_group_to_sample.sampling_params.top_k
            top_p[index] = sequence_group_to_sample.sampling_params.top_p
            temperature[index] = \
                sequence_group_to_sample.sampling_params.temperature

        self.model.model.update_generation_config(current_sampling_params)


    @torch.inference_mode()
    @dump_input_when_exception(exclude_args=[0], exclude_kwargs=["self"])
    def execute_model(
        self,
        model_input: ModelInputForNPU,
        kv_caches: Optional[List[torch.Tensor]] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:
        if num_steps > 1:
            raise ValueError(
                "NPUModelRunner does not support multi-step execution.")
        #logger.info(f"input_tokens {model_input.input_tokens}")
        #logger.info(f"input_positions {model_input.input_positions}")
        # TODO split batch and merge
        hidden_states = self.model(
            input_ids=model_input.input_tokens,
            positions=model_input.input_positions,
            kv_caches=kv_caches,
            attn_metadata=model_input.attn_metadata,
            intermediate_tensors=intermediate_tensors,
            **MultiModalKwargs.as_kwargs(model_input.multi_modal_kwargs or {},
                                         device=self.device),
        )
        #logger.info(f"inference_dome")

        # Compute the logits only if the on-device sampling is turned off as
        # on-device sampling outputs the token ids.
        logits = self.model.compute_logits(hidden_states,
                                               model_input.sampling_metadata)
        #logger.info(f"compute_logits done")
        logits = logits.cpu()
        # Sample the next token.
        output = self.model.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )
        #logger.info(f"sample done")
        return [output]

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()
