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

DEBUG = False

try:
    from xformers import ops as xops
    USE_XFORMERS_OPS = True
except ImportError:
    USE_XFORMERS_OPS = False

from pixtral_utils import VisionTransformer

from modeling_mistral import NeuronLlamaModel

HF_CHECKPOINT = "HF"
META_CHECKPOINT = "META"

def get_rmsnorm_cls():
    # Initialize to the appropriate implementation of RMSNorm
    # If infer on NXD -> CustomRMSNorm
    # If infer on CPU -> HF's LlamaRMSNorm (CustomRMSNorm does not work on CPU)
    return CustomRMSNorm if parallel_state.get_tensor_model_parallel_size() > 1 else LlamaRMSNorm

class NeuronPixtralModel(NeuronBaseModel):
        def __init__(self, config: InferenceConfig):
            self.text_config = config.text_config
            self.vision_config = config.vision_config
            super().__init__(self.text_config, optimize_inference = False)

        def init_model(self, config: InferenceConfig):
            self.vision_model = VisionTransformer(self.vision_config)
            self.text_model = NeuronLlamaModel(self.text_config)

        def setup_attr_for_model(self, config: InferenceConfig):
            self.on_device_sampling = config.neuron_config.on_device_sampling_config
            self.tp_degree = config.neuron_config.tp_degree
            self.hidden_dim = self.text_config.hidden_dim
            self.n_heads = self.text_config.n_heads
            self.dim = self.text_config.dim
            self.n_layers = self.text_config.n_layers
            self.head_dim = self.text_config.head_dim
            self.n_kv_heads = self.text_config.n_kv_heads
            self.rope_theta = self.text_config.rope_theta
            self.norm_eps = self.text_config.norm_eps
            self.vocab_size = self.text_config.vocab_size


        def forward(self):
            print ('Forward pass is not yet implemented for NeuronPixtralModel')
    
class PixtralInferenceConfig(InferenceConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)    
        self.text_config = args[1]['text_config']
        self.vision_config = args[1]['vision_encoder']
        self.pad_token_id = args[1]['text_config']['pad_token_id']
        self.fused_spec_config = None # hard overwrite to pass application_base assert
        
        if not hasattr(self, "checkpoint"):
            self.checkpoint = kwargs.get("checkpoint", HF_CHECKPOINT)

        assert self.checkpoint in [
            HF_CHECKPOINT,
            META_CHECKPOINT,
        ], f"Uknown checkpoint: {self.checkpoint}"

        if hasattr(self, "text_config"):
            if isinstance(self.text_config, SimpleNamespace):
                self.text_config = vars(self.text_config)
            # replicating what's done in hf_adapter's load_config()
            self.text_config.pop("torch_dtype", None)
            self.text_config = InferenceConfig(self.neuron_config, **self.text_config)
            if not hasattr(self.text_config, "checkpoint"):
                setattr(self.text_config, "checkpoint", self.checkpoint)

        if hasattr(self, "vision_config"):
            if isinstance(self.vision_config, SimpleNamespace):
                self.vision_config = vars(self.vision_config)
            # replicating what's done in hf_adapter's load_config()
            self.vision_config.pop("torch_dtype", None)
            self.vision_config = InferenceConfig(self.neuron_config, **self.vision_config)
            if not hasattr(self.vision_config, "checkpoint"):
                setattr(self.vision_config, "checkpoint", self.checkpoint)


    def validate_config(self):
        """
        Validates that the config has all required attributes.
        """

        def hasattr_nested(obj, attr_chain):
            attrs = attr_chain.split(".")
            for attr in attrs:
                if isinstance(obj, dict):
                    if attr not in obj:
                        return False
                    obj = obj[attr]
                else:
                    if not hasattr(obj, attr):
                        return False
                    obj = getattr(obj, attr)
            return True

        assert (
            self.neuron_config.is_medusa is False and self.neuron_config.speculation_length == 0
        ), f"Speculative Decoding is not yet supported in this Model. \
                is_medusa was set to {self.neuron_config.is_medusa}. \
                speculation_length was set to {self.neuron_config.speculation_length}"
        assert (
            int(self.neuron_config.logical_neuron_cores) == 1
        ), "This model currently only support logical_neuron_cores=1"

    def to_json_string(self):
        config_copy = copy.deepcopy(self)
        config_dict = to_dict(config_copy)
        config_dict["text_config"].pop("neuron_config", None)
        config_dict["vision_config"].pop("neuron_config", None)
        return json.dumps(config_dict, indent=2, sort_keys=True)

    @classmethod
    def get_neuron_config_cls(cls) -> Type[MultimodalVisionNeuronConfig]:
        return MultimodalVisionNeuronConfig

class NeuronPixtralForConditionalGeneration(NeuronBaseForCausalLM):
    """
    This class extends LlamaForCausalLM to create traceable
    blocks for Neuron.

    Args:
        LlamaForCausalLM (_type_): _description_
    """

    _model_cls = NeuronPixtralModel

    @classmethod
    def get_config_cls(cls):
        return PixtralInferenceConfig

    @classmethod
    def get_neuron_config_cls(cls):
        return MultimodalVisionNeuronConfig

    @staticmethod
    def load_hf_model(model_path):
        raise Exception("HuggingFace checkpoint is not supported yet")

    def get_compiler_args(self) -> str:
        return "--enable-saturate-infinity --auto-cast=none --model-type=transformer \
                --tensorizer-options='--enable-ccop-compute-overlap \
                --cc-pipeline-tiling-factor=2 --vectorize-dge-dma --vectorize-strided-dma' -O1 \
                --hbm-scratchpad-page-size=1024"

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, inference_config: InferenceConfig
    ) -> dict:
        if inference_config.checkpoint == HF_CHECKPOINT:
            from .hf_state_dict_conversion import convert_hf_state_dict_to_neuron_state_dict

            return convert_hf_state_dict_to_neuron_state_dict(state_dict, inference_config)
        elif inference_config.checkpoint == META_CHECKPOINT:
            from .meta_state_dict_conversion import convert_meta_state_dict_to_neuron_state_dict

            return convert_meta_state_dict_to_neuron_state_dict(state_dict, inference_config)

    def get_model_wrapper_cls(self):
        return ModelWrapperMllama

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        pass