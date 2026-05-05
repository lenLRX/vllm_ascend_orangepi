"""Gemma4 configuration for vLLM - bypasses transformers PretrainedConfig to avoid protobuf issues."""
import json
import os


class Gemma4TextConfig:
    """Minimal config for Gemma4 text model. Not a PretrainedConfig subclass
    to avoid protobuf issues with unrecognized model_type 'gemma4_text'."""
    model_type = "gemma4_text"
    is_encoder_decoder = False
    attribute_map = {}
    architectures = ["Gemma4ForCausalLM"]

    def __init__(self, **kwargs):
        # Core model params
        self.vocab_size = kwargs.get("vocab_size", 256000)
        self.hidden_size = kwargs.get("hidden_size", 1536)
        self.intermediate_size = kwargs.get("intermediate_size", 6144)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 35)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 1)
        self.head_dim = kwargs.get("head_dim", 256)
        self.global_head_dim = kwargs.get("global_head_dim", 512)
        self.hidden_activation = kwargs.get("hidden_activation", "gelu_pytorch_tanh")
        self.hidden_act = self.hidden_activation  # alias for vllm compat
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 131072)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.attention_bias = kwargs.get("attention_bias", False)
        self.attention_dropout = kwargs.get("attention_dropout", 0.0)
        self.rope_parameters = kwargs.get("rope_parameters", None)
        self.rope_theta = kwargs.get("rope_theta", 10000.0)
        self.rope_scaling = kwargs.get("rope_scaling", None)
        self.layer_types = kwargs.get("layer_types", None)
        self.sliding_window = kwargs.get("sliding_window", 512)
        self.num_kv_shared_layers = kwargs.get("num_kv_shared_layers", 0)
        self.use_double_wide_mlp = kwargs.get("use_double_wide_mlp", False)
        self.hidden_size_per_layer_input = kwargs.get("hidden_size_per_layer_input", 0)
        self.vocab_size_per_layer_input = kwargs.get(
            "vocab_size_per_layer_input",
            kwargs.get("vocab_size", 256000))
        self.attention_k_eq_v = kwargs.get("attention_k_eq_v", False)
        self.num_global_key_value_heads = kwargs.get("num_global_key_value_heads", None)
        self.enable_moe_block = kwargs.get("enable_moe_block", False)
        self.use_second_mlp_block = kwargs.get("use_second_mlp_block", False)
        self.num_experts = kwargs.get("num_experts", None)
        self.top_k_experts = kwargs.get("top_k_experts", None)
        self.expert_intermediate_size = kwargs.get("expert_intermediate_size", None)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", None)
        self.attn_logit_softcapping = kwargs.get("attn_logit_softcapping", None)
        self.final_logit_softcapping = kwargs.get("final_logit_softcapping", None)
        self.use_bidirectional_attention = kwargs.get("use_bidirectional_attention", None)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", False)
        self.pad_token_id = kwargs.get("pad_token_id", 0)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.use_cache = kwargs.get("use_cache", True)
        self.initializer_range = kwargs.get("initializer_range", 0.02)
        # Store any extra kwargs
        self._extra = {k: v for k, v in kwargs.items()
                       if not hasattr(self, k) and not k.startswith("_")}

    def __getattr__(self, name):
        if name in ("_extra",):
            raise AttributeError(name)
        if name in self._extra:
            return self._extra[name]
        if hasattr(self, "_extra") and name in self._extra:
            return self._extra[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def to_dict(self):
        d = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return d

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config_file = os.path.join(pretrained_model_name_or_path, "config.json")
        with open(config_file) as f:
            config_dict = json.load(f)
        # If there's a text_config, use that
        if "text_config" in config_dict:
            return cls(**config_dict["text_config"])
        return cls(**config_dict)


class Gemma4Config:
    """Wrapper config that extracts text_config for multimodal Gemma4 checkpoints."""
    model_type = "gemma4"
    is_encoder_decoder = False
    attribute_map = {}

    def __init__(self, **kwargs):
        text_config_dict = kwargs.get("text_config", None)
        if isinstance(text_config_dict, dict):
            self.text_config = Gemma4TextConfig(**text_config_dict)
        elif isinstance(text_config_dict, Gemma4TextConfig):
            self.text_config = text_config_dict
        else:
            # Use top-level keys for text_config
            self.text_config = Gemma4TextConfig(**kwargs)
        self.architectures = kwargs.get("architectures", ["Gemma4ForConditionalGeneration"])
        self.pad_token_id = kwargs.get("pad_token_id", 0)
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.tie_word_embeddings = self.text_config.tie_word_embeddings
        self.use_cache = True

    def to_dict(self):
        return {"text_config": self.text_config.to_dict(),
                "architectures": self.architectures}

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config_file = os.path.join(pretrained_model_name_or_path, "config.json")
        with open(config_file) as f:
            config_dict = json.load(f)
        return cls(**config_dict)
