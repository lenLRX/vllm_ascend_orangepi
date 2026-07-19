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


def _greedy_fast_path_ok(sampling_metadata) -> bool:
    """Whether every sequence group can use the on-device greedy argmax.

    Falls back to the full CPU sampler when anything could change the argmax
    (penalties, min_tokens) or when logprob details are requested.
    top_k/top_p/min_p are safe: they never remove the global argmax.
    """
    if sampling_metadata is None:
        return False
    for sg in sampling_metadata.seq_groups:
        if not sg.do_sample:
            continue
        sp = sg.sampling_params
        if sp is None:
            return False
        if sp.temperature is None or sp.temperature > 1e-5:
            return False
        if getattr(sp, 'presence_penalty', 0.0) not in (0.0, None):
            return False
        if getattr(sp, 'frequency_penalty', 0.0) not in (0.0, None):
            return False
        if getattr(sp, 'repetition_penalty', 1.0) not in (1.0, None):
            return False
        if getattr(sp, 'logprobs', None) is not None:
            return False
        if getattr(sp, 'prompt_logprobs', None) is not None:
            return False
        if getattr(sp, 'min_tokens', 0):
            return False
        if getattr(sp, 'use_beam_search', False):
            return False
        if len(sg.seq_ids) != 1:
            return False
    return True


@dataclass(frozen=False)
class NPUAttentionMetadata:
    offsets: Optional[List[int]] = None
    seq_lens: Optional[List[int]] = None
    block_tables: Optional[List] = None
    is_prompt: bool = None
    start_positions: Optional[List[int]] = None


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
        self.device = self.device_config.device
        self.pin_memory = False

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
    ):
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[int] = []
        input_positions: List[List[int]] = []
        input_offsets: List[int] = []
        input_lengths: List[int] = []
        input_block_tables = []
        start_positions: List[int] = []

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
            start_positions.append(0)

            #input_tokens.append(prompt_tokens)
            input_tokens.extend(prompt_tokens)
            input_positions.extend(list(range(seq_len)))

            assert seq_group_metadata.block_tables is not None
            #logger.info(f"seq_ids {seq_ids}, block_tables: {seq_group_metadata.block_tables}")
            block_table = seq_group_metadata.block_tables[seq_id]
            # Block table NPU tensor is never used downstream — only the CPU
            # list is indexed. Pass (None, host_list) to eliminate .npu() JIT trigger.
            input_block_tables.append((None, block_table))
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
        # Use torch.empty + copy_ to avoid torch.tensor(..., device=npu) JIT trigger
        total_tokens = len(input_tokens)
        tokens_cpu = torch.tensor(input_tokens, dtype=torch.long, device="cpu")
        positions_cpu = torch.tensor(input_positions, dtype=torch.long, device="cpu")
        input_tokens = torch.empty(total_tokens, dtype=torch.long, device=self.device)
        input_positions = torch.empty(total_tokens, dtype=torch.long, device=self.device)
        input_tokens.copy_(tokens_cpu)
        input_positions.copy_(positions_cpu)
        #assert len(input_tokens) == 1
        #input_tokens = make_tensor_with_pad(input_tokens,
        #                                    pad=0,
        #                                    max_len=max_seq_len,
        #                                    dtype=torch.long,
        #                                    device=self.device)
        #input_positions = make_tensor_with_pad(input_positions,
        #                                       pad=0,
        #                                       max_len=max_seq_len,
        #                                       dtype=torch.long,
        #                                       device=self.device)


        multi_modal_kwargs = MultiModalKwargs.batch(multi_modal_kwargs_list)

        return (input_tokens, input_positions, input_offsets, input_lengths, seq_lens,
                multi_modal_kwargs, input_block_tables, start_positions)

    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ):
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        input_offsets: List[int] = []
        input_lengths: List[int] = []
        input_block_tables = []

        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt

            seq_ids = list(seq_group_metadata.seq_data.keys())

            assert seq_group_metadata.block_tables is not None
            #logger.info(f"seq_ids {seq_ids}, block_tables: {seq_group_metadata.block_tables}")

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append(generation_token)

                seq_len = seq_data.get_len()
                position = seq_len - 1
                input_positions.append(position)

                input_offsets.append(position)
                input_lengths.append(1)

                block_table = seq_group_metadata.block_tables[seq_id]
                # Block table NPU tensor is never used downstream — only the
                # CPU list is indexed. Pass (None, host_list) to eliminate .npu() JIT trigger.
                input_block_tables.append((None, block_table))

        # Use torch.empty + copy_ to avoid torch.tensor(..., device=npu) JIT trigger
        total_tokens = len(input_tokens)
        tokens_cpu = torch.tensor(input_tokens, dtype=torch.long, device="cpu")
        positions_cpu = torch.tensor(input_positions, dtype=torch.long, device="cpu")
        input_tokens = torch.empty(total_tokens, dtype=torch.long, device=self.device)
        input_positions = torch.empty(total_tokens, dtype=torch.long, device=self.device)
        input_tokens.copy_(tokens_cpu)
        input_positions.copy_(positions_cpu)

        return input_tokens, input_positions, input_offsets, input_lengths, input_block_tables

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
             multi_modal_kwargs, input_block_tables, start_positions
             ) = self._prepare_prompt(seq_group_metadata_list)
        else:
            (input_tokens, input_positions,
             input_offsets, input_lengths, input_block_tables) = self._prepare_decode(seq_group_metadata_list)
            seq_lens = None
            start_positions = input_offsets  # decode: start position = offset (which is seq_len - 1)
        sampling_metadata = SamplingMetadata.prepare(
            seq_group_metadata_list,
            seq_lens,
            seq_lens,
            self.device,
            self.pin_memory,
            generators=self.get_generators(finished_requests_ids))
        # Use torch.empty + copy_ to avoid .npu() JIT trigger
        sti_cpu = sampling_metadata.selected_token_indices
        sti_npu = torch.empty(sti_cpu.shape, dtype=sti_cpu.dtype, device=self.device)
        sti_npu.copy_(sti_cpu)
        sampling_metadata.selected_token_indices = sti_npu
        attn_metadata = NPUAttentionMetadata(offsets=input_offsets, seq_lens=input_lengths,
                                             block_tables=input_block_tables, is_prompt=is_prompt,
                                             start_positions=start_positions)
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
        _pt = os.environ.get("Q4_PHASE_TIMING", "0") == "1"
        if _pt:
            import time as _time
            _t0 = _time.perf_counter()
            _prev = getattr(self, '_phase_prev_ret', None)
            if _prev is not None:
                print(f"PHASE engine_gap={_t0 - _prev:.3f}s")
        hidden_states = self.model(
            input_ids=model_input.input_tokens,
            positions=model_input.input_positions,
            kv_caches=kv_caches,
            attn_metadata=model_input.attn_metadata,
            intermediate_tensors=intermediate_tensors,
            **MultiModalKwargs.as_kwargs(model_input.multi_modal_kwargs or {},
                                         device=self.device),
        )
        if _pt:
            torch.npu.synchronize()
            _t1 = _time.perf_counter()
        #logger.info(f"inference_dome")

        # Compute the logits only if the on-device sampling is turned off as
        # on-device sampling outputs the token ids.
        logits = self.model.compute_logits(hidden_states,
                                               model_input.sampling_metadata)
        if _pt:
            torch.npu.synchronize()
            _t2 = _time.perf_counter()
        # Greedy fast path: argmax on device, skip the full-vocab logits D2H
        # copy and the CPU softmax/log_softmax (~10ms/token on this host).
        output = None
        if _greedy_fast_path_ok(model_input.sampling_metadata):
            output = self._greedy_sample_on_device(
                logits, model_input.sampling_metadata)
        if output is None:
            #logger.info(f"compute_logits done")
            logits = logits.cpu()
            if _pt:
                _t3 = _time.perf_counter()
            # Sample the next token.
            output = self.model.sample(
                logits=logits,
                sampling_metadata=model_input.sampling_metadata,
            )
        elif _pt:
            _t3 = _time.perf_counter()
        if _pt:
            _t4 = _time.perf_counter()
            self._phase_prev_ret = _t4
            print(f"PHASE fwd={_t1 - _t0:.3f} logits={_t2 - _t1:.3f} "
                  f"d2h={_t3 - _t2:.3f} sample={_t4 - _t3:.3f}")
        #logger.info(f"sample done")
        return [output]

    def _greedy_sample_on_device(self, logits, sampling_metadata):
        """Greedy (temperature==0) sampling directly on the NPU.

        The default path moves the full [1, vocab] logits to the CPU and runs
        softmax/log_softmax over 262k entries there (~10ms/token). For greedy
        decoding only the argmax matters, so do it on device and copy back
        just the token ids (+ their exact logprobs, for output parity).
        """
        from vllm.sequence import (CompletionSequenceGroupOutput, Logprob,
                                   SequenceOutput)

        sti = sampling_metadata.selected_token_indices
        sel = logits[sti]  # [num_samples, vocab] on NPU
        greedy_ids = torch.argmax(sel, dim=-1)
        chosen_lp = torch.log_softmax(sel, dim=-1, dtype=torch.float32).gather(
            1, greedy_ids.unsqueeze(1)).squeeze(1)
        ids = greedy_ids.tolist()
        lps = chosen_lp.tolist()

        outputs = []
        idx = 0
        for sg in sampling_metadata.seq_groups:
            if not sg.do_sample:
                outputs.append(CompletionSequenceGroupOutput([], None))
                continue
            seq_id = sg.seq_ids[0]
            tid = int(ids[idx])
            logprobs = {tid: Logprob(logprob=float(lps[idx]), rank=1)}
            outputs.append(CompletionSequenceGroupOutput(
                [SequenceOutput(seq_id, tid, logprobs)], None))
            idx += 1
        return SamplerOutput(outputs=outputs)

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()
