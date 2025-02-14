"""PyTorch Pixtral Multimodal model for NXD inference."""

import copy
import json
import logging
import math
from types import SimpleNamespace
from typing import List, Optional, Tuple, Type, Union, Iterable, List, Mapping, Set
from functools import cached_property
from dataclasses import dataclass, fields
from PIL import Image

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torch_xla.core.xla_model as xm

from transformers import PixtralVisionConfig
from transformers.models.pixtral.image_processing_pixtral import (
    _num_image_tokens as _get_pixtral_hf_num_image_tokens)
from transformers.models.pixtral.modeling_pixtral import (
    PixtralRotaryEmbedding, apply_rotary_pos_emb, position_ids_in_meshgrid)
from transformers.generation import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding

from transformers import AutoTokenizer, GenerationConfig

from mistral_common.protocol.instruct.messages import ImageChunk

from vllm.attention import AttentionMetadata
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.inputs import INPUT_REGISTRY,InputContext

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import SamplerOutput

from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm.multimodal.utils import cached_get_tokenizer
from vllm.sequence import IntermediateTensors, SequenceData

from neuronx_distributed.parallel_layers.mappings import (
    _reduce_scatter_along_dim,
    gather_from_sequence_parallel_region,
)


from neuronx_distributed_inference.models.llama.modeling_llama import Llama3RotaryEmbedding
from neuronx_distributed_inference.modules.attention.attention_base import (
    FlashAttentionStrategy,
    NeuronAttentionBase,
)
from neuronx_distributed_inference.modules.attention.utils import (
    RotaryEmbedding,
    apply_rotary_polar_compatible,
    move_heads_front,
    precompute_freqs_cis,
)
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm  # noqa: F401
from neuronx_distributed_inference.modules.kvcache.multimodal_kv_cache_manager import (
    MultimodalKVCacheManager,
)
from neuronx_distributed_inference.utils.distributed import get_tp_group

from model_wrapper_mllama import ModelWrapperMllama

SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]

from neuronx_distributed.parallel_layers import parallel_state  # noqa: E402
from neuronx_distributed.parallel_layers.layers import (  # noqa: E402
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)

from neuronx_distributed_inference.models.config import (  # noqa: E402
    InferenceConfig,
    MultimodalVisionNeuronConfig,
    to_dict,
)

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig


from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaMLP  # noqa: E402
from neuronx_distributed_inference.models.model_base import (  # noqa: E402
    NeuronBaseForCausalLM,
    NeuronBaseModel,
    get_cache_size,
    mask_util,
    turn_2d_mask_to_4d,
)

from modeling_pixtral_vision import NeuronMllamaVisionModel  # noqa: E402

DEBUG = False

try:
    from xformers import ops as xops
    USE_XFORMERS_OPS = True
except ImportError:
    USE_XFORMERS_OPS = False

HF_CHECKPOINT = "HF"
META_CHECKPOINT = "META"

def get_rmsnorm_cls():
    # Initialize to the appropriate implementation of RMSNorm
    # If infer on NXD -> CustomRMSNorm
    # If infer on CPU -> HF's LlamaRMSNorm (CustomRMSNorm does not work on CPU)
    return CustomRMSNorm if parallel_state.get_tensor_model_parallel_size() > 1 else LlamaRMSNorm


class NeuronPixtralConfig(NeuronConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set any args/defaults


class PixtralInferenceConfig(InferenceConfig):
    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "pad_token_id",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "hidden_act",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[MultimodalVisionNeuronConfig]:
        return MultimodalVisionNeuronConfig





