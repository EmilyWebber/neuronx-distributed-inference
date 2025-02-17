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

from pixtral_utils import VisionTransformer, VisionLanguageAdapter

from modeling_mistral import NeuronLlamaModel

from vllm_data import TokenInputs, token_inputs

from typing import (TYPE_CHECKING, Any, Callable, Mapping, NamedTuple,
                    Optional, Protocol, Union)

DecoderOnlyInputs = Union[TokenInputs, "MultiModalInputs"]

NestedTensors = Union[list["NestedTensors"], list[torch.Tensor], torch.Tensor, tuple[torch.Tensor, ...]]

HF_CHECKPOINT = "HF"
META_CHECKPOINT = "META"

class DummyData(NamedTuple):
    """Dummy data used for profiling."""

    seq_data: "SequenceData"
    multi_modal_data: Optional["MultiModalDataDict"] = None
    multi_modal_placeholders: Optional["MultiModalPlaceholderDict"] = None

def get_rmsnorm_cls():
    # Initialize to the appropriate implementation of RMSNorm
    # If infer on NXD -> CustomRMSNorm
    # If infer on CPU -> HF's LlamaRMSNorm (CustomRMSNorm does not work on CPU)
    return CustomRMSNorm if parallel_state.get_tensor_model_parallel_size() > 1 else LlamaRMSNorm


def make_empty_intermediate_tensors(batch_size: int,
                                    dtype: torch.dtype,
                                    device: torch.device,
                                    ) -> IntermediateTensors:
    return IntermediateTensors({
        key:
        torch.zeros((batch_size, hidden_size), dtype=dtype, device=device)
        for key in keys
    })


def get_max_pixtral_image_tokens(ctx: InputContext):
    tokenizer = cached_get_tokenizer(
        ctx.model_config.tokenizer,
        tokenizer_mode=ctx.model_config.tokenizer_mode)
    mm_encoder = tokenizer.instruct.mm_encoder

    image_config = mm_encoder.mm_config if hasattr(
        mm_encoder, "mm_config") else mm_encoder.image_config

    max_image_size = image_config.max_image_size
    image_patch_size = image_config.image_patch_size

    return ((max_image_size // image_patch_size)**2)


def dummy_data_for_pixtral(ctx: InputContext, seq_len: int,
                           mm_counts: Mapping[str, int]):
    tokenizer = cached_get_tokenizer(
        ctx.model_config.tokenizer,
        tokenizer_mode=ctx.model_config.tokenizer_mode)

    mm_encoder = tokenizer.mistral.instruct_tokenizer.mm_encoder
    image_token_id = mm_encoder.special_ids.img

    mm_config = ctx.get_mm_config()
    num_images = mm_config.limit_per_prompt.get("image", 1)

    # dummy size
    size = 256
    image = Image.new("RGB", (size, size), color=0)

    encoding = tokenizer.instruct.mm_encoder(ImageChunk(image=image))
    image_feature_size = len(encoding.tokens)
    num_image_tokens = image_feature_size * num_images
    seq_data = SequenceData.from_prompt_token_counts(
        (image_token_id, num_image_tokens),
        (0, seq_len - num_image_tokens),
    )

    mm_data = {"image": num_images * [image]}
    mm_placeholders = {
        "image":
        consecutive_placeholder_ranges(num_items=num_images,
                                       item_size=image_feature_size)
    }
    return DummyData(seq_data, mm_data, mm_placeholders)


def input_mapper_for_pixtral(ctx: InputContext, data: object):
    """Maps the input data to its MultiModalKwargs (if any).

    Args:
        ctx: Context of the loaded model.
        data: data potentially containing PIL images to be processed
            and mapped to `images`.

    Returns:
        MultiModalKwargs containing the stacked normalized images tensor or
        image embeddings.
    """
    model_config = ctx.model_config
    tokenizer = cached_get_tokenizer(
        model_config.tokenizer, tokenizer_mode=model_config.tokenizer_mode)

    data_list = data if isinstance(data, list) else [data]

    images = []
    image_tokens_list = []
    for image_data in data_list:
        image = ImageChunk(image=image_data)
        encoding = tokenizer.instruct.mm_encoder(image)
        image = torch.from_numpy(encoding.image).to(dtype=torch.float16)
        images.append(image)
        image_tokens_list.append(encoding.tokens)

    image_tokens = torch.tensor([
        token_id for image_tokens in image_tokens_list
        for token_id in image_tokens
    ])

    # return MultiModalKwargs({"images": images, "image_tokens": image_tokens})

    rt = {"images": images, "image_tokens": image_tokens}

    return rt    


def input_processor_for_pixtral(ctx: InputContext, inputs: DecoderOnlyInputs):
    multi_modal_data = inputs.get("multi_modal_data")
    if multi_modal_data is None or "image" not in multi_modal_data:
        return inputs

    prompt_token_ids = inputs.get("prompt_token_ids")
    prompt = inputs.get("prompt")
    tokenizer = cached_get_tokenizer(
        ctx.model_config.tokenizer,
        tokenizer_mode=ctx.model_config.tokenizer_mode)

    mm_encoder = tokenizer.mistral.instruct_tokenizer.mm_encoder
    image_token_id = mm_encoder.special_ids.img
    image_break_id = mm_encoder.special_ids.img_break
    image_end_id = mm_encoder.special_ids.img_end

    if image_token_id not in inputs['prompt_token_ids']:
        raise ValueError(
            f"You've passed {inputs=} without {image_token_id=}"
            " Make sure to process your input via mistral_common's"
            " tokenizer or pass a chat completion request. For more"
            " For more info, see: "
            "https://github.com/vllm-project/vllm/issues/8411.")

    # Get precise tracking of placeholder positions
    placeholder_ranges = []
    curr_offset = -1
    curr_length = 0
    for i in range(len(prompt_token_ids)):
        if prompt_token_ids[i] in (image_token_id, image_break_id):
            if curr_offset < 0:
                curr_offset = i
            curr_length += 1
        elif prompt_token_ids[i] == image_end_id:
            curr_length += 1
            placeholder_ranges.append(
                PlaceholderRange(offset=curr_offset, length=curr_length))
            curr_offset = -1
            curr_length = 0
        else:
            pass
    return token_inputs(prompt=prompt,
                        prompt_token_ids=prompt_token_ids,
                        multi_modal_data=multi_modal_data,
                        multi_modal_placeholders={"image": placeholder_ranges})

class NeuronPixtralModel(NeuronBaseModel):
        def __init__(self, config: InferenceConfig):
            self.text_config = config.text_config
            self.vision_config = config.vision_config
            super().__init__(self.text_config)
            
            self.config = config
            # self.multimodal_config = multimodal_config
    
            # dataclass_fields = {field.name for field in fields(config.vision_config)}
            
            # vision_args = {
            #     key: value
            #     for key, value in self.config.vision_config.to_dict().items()
            #     if key in dataclass_fields
            # }
    
            # if not ("image_break_token_id" in vision_args
            #         and "image_end_token_id" in vision_args):
            #     raise ValueError(
            #         "'image_break_token_id' and 'image_end_token_id' not found "
            #         "in the vision_encoder arguments. Please download the latest "
            #         "version of 'params.json' from the model repository.")

    
            self.vision_args = config.vision_config
    
        def init_model(self, config: InferenceConfig):
            self.vision_model = VisionTransformer(self.vision_config)
            self.text_model = NeuronLlamaModel(self.text_config)

            # init MistralForCausalLM
            # self.language_model = init_vllm_registered_model(
            #     vllm_config=vllm_config,
            #     hf_config=config.text_config,
            #     prefix=maybe_prefix(prefix, "language_model"),
            # )
    
            # self.vision_encoder = VisionTransformer(self.vision_args)
            self.vision_language_adapter = VisionLanguageAdapter(
                self.vision_config, dim=self.text_config.hidden_size)
    
            self.make_empty_intermediate_tensors = (make_empty_intermediate_tensors)

        def setup_attr_for_model(self, config: InferenceConfig):
            self.on_device_sampling = config.neuron_config.on_device_sampling_config
            self.tp_degree = config.neuron_config.tp_degree
            self.hidden_dim = self.text_config.hidden_dim
            self.num_key_value_heads = self.text_config.n_heads
            self.dim = self.text_config.dim
            self.n_layers = self.text_config.n_layers
            self.head_dim = self.text_config.head_dim
            self.n_kv_heads = self.text_config.n_kv_heads
            self.rope_theta = self.text_config.rope_theta
            self.norm_eps = self.text_config.norm_eps
            self.vocab_size = self.text_config.vocab_size

        def init_inference_optimization(self, config: InferenceConfig):
            super().init_inference_optimization(config)
            # only need one kv cache mgr
            self.kv_mgr = self.text_model.kv_mgr 

        @cached_property
        def sampler(self):
            if hasattr(self.language_model, "sampler"):
                return self.language_model.sampler
    
            return get_sampler()
    
        def get_multimodal_embeddings(self, **kwargs) -> Optional[NestedTensors]:
            image_input, image_tokens = self._parse_and_validate_image_input(
                **kwargs)
            if image_input is None:
                return None
    
            vision_embeddings = self._process_image_input(image_input)
    
            # NOTE: We patch the outputs of the vision encoder with embeddings
            # from `[IMG_BREAK]` and `[IMG_END]` tokens.
            image_embeds = self.language_model.get_input_embeddings(image_tokens)
            image_token_mask = image_tokens == self.vision_args.image_token_id
            image_embeds[image_token_mask] = vision_embeddings
    
            # NOTE: Image embeddings are split into separate tensors for each image
            # by the indices of `[IMG_END]` token.
            image_end_mask = image_tokens == self.vision_args.image_end_token_id
            split_indices = torch.where(image_end_mask)[0] + 1
            if len(split_indices) <= 1:
                # Do not split, return as tensor of shape [1, fs, hs]
                return image_embeds.unsqueeze(0)
    
            # If the last split index is the last index in image_tokens, we
            # ignore it to avoid empty split tensor
            if split_indices[-1] == len(image_tokens):
                split_indices = split_indices[:-1]
    
            image_embeds = image_embeds.tensor_split(split_indices.cpu())
            return image_embeds
    
        def get_input_embeddings(
            self,
            input_ids: torch.Tensor,
            multimodal_embeddings: Optional[NestedTensors] = None,
        ) -> torch.Tensor:
            inputs_embeds = self.language_model.get_input_embeddings(input_ids)
            if multimodal_embeddings is not None:
                inputs_embeds = merge_multimodal_embeddings(
                    input_ids, inputs_embeds, multimodal_embeddings, [
                        self.vision_args.image_token_id,
                        self.vision_args.image_break_token_id,
                        self.vision_args.image_end_token_id,
                    ])
            return inputs_embeds
    
        def forward(self,
                    input_ids: torch.Tensor,
                    positions: torch.Tensor,
                    kv_caches: List[torch.Tensor],
                    attn_metadata: AttentionMetadata,
                    intermediate_tensors: Optional[IntermediateTensors] = None,
                    inputs_embeds: Optional[torch.Tensor] = None,
                    **kwargs: object,
        ) -> Union[torch.Tensor, IntermediateTensors]:
            """Run forward pass for pixtral.
            """
            if intermediate_tensors is not None:
                inputs_embeds = None
    
            # NOTE: In v1, inputs_embeds is always generated at model runner, this
            # condition is for v0 compatibility.
            elif inputs_embeds is None:
                vision_embeddings = self.get_multimodal_embeddings(**kwargs)
                inputs_embeds = self.get_input_embeddings(input_ids,
                                                          vision_embeddings)
                input_ids = None
    
            hidden_states = self.text_model.forward(input_ids,
                                                  positions,
                                                  kv_caches,
                                                  attn_metadata,
                                                  intermediate_tensors,
                                                  inputs_embeds=inputs_embeds)
    
            return hidden_states
    
        def _parse_and_validate_image_input(
            self,
            images: Optional[Union[List[List[torch.Tensor]], List[torch.Tensor],
                                   torch.Tensor]] = None,
            image_tokens: Optional[torch.Tensor] = None,
        ) -> Tuple[Optional[List[torch.Tensor]], Optional[torch.Tensor]]:
            if images is None:
                return None, None
    
            if isinstance(images, torch.Tensor):
                # if passed as batch take all images
                N, B, C, W, H = images.shape
                images = images.reshape(N * B, C, W, H)
                images = [images[i] for i in range(images.size(0))]
            elif isinstance(images, list):
                # if passed as list flatten lists of tensors
                flatten_images = []
                for imgs_per_req in images:
                    imgs_per_req = [
                        imgs_per_req[i] for i in range(imgs_per_req.size(0))
                    ] if isinstance(imgs_per_req, torch.Tensor) else imgs_per_req
    
                    flatten_images.extend(imgs_per_req)
    
                images = flatten_images
    
            if isinstance(image_tokens, torch.Tensor):
                # image_tokens are batched
                image_tokens = image_tokens.flatten()
            elif isinstance(image_tokens, list):
                # image_tokens are of different lengths thus passed as a list
                image_tokens = torch.cat(image_tokens)
    
            assert image_tokens.dim() == 1
    
            return images, image_tokens
    
        def _process_image_input(self,
                                 image_input: List[torch.Tensor]) -> torch.Tensor:
            return self.vision_language_adapter(self.vision_encoder(image_input))
    
        def compute_logits(
            self,
            hidden_states: torch.Tensor,
            sampling_metadata: SamplingMetadata,
        ) -> Optional[torch.Tensor]:
            return self.language_model.compute_logits(hidden_states,
                                                      sampling_metadata)
    
        def sample(
            self,
            logits: torch.Tensor,
            sampling_metadata: SamplingMetadata,
        ) -> Optional[SamplerOutput]:
            return self.language_model.sample(logits, sampling_metadata)
    
        def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
    
            def is_vision_encoder_weights(weight: Tuple[str, torch.Tensor]):
                return weight[0].startswith("vision_encoder")
    
            def is_vision_lang_adapter_weights(weight: Tuple[str, torch.Tensor]):
                return weight[0].startswith("vision_language_adapter")
    
            # Get references to parameters for direct loading
            vision_encoder_dict = dict(self.vision_encoder.named_parameters())
            vision_lang_adapter_dict = dict(
                self.vision_language_adapter.named_parameters())
    
            def llm_weights_generator():
                # Single pass over weights
                for name, w in weights:
                    if is_vision_encoder_weights((name, w)):
                        # Load vision encoder weights directly
                        trimmed_name = '.'.join(name.split(".")[1:])
                        param = vision_encoder_dict[trimmed_name]
                        with torch.no_grad():
                            default_weight_loader(param, w)
                    elif is_vision_lang_adapter_weights((name, w)):
                        # Load vision-language adapter weights directly
                        trimmed_name = '.'.join(name.split(".")[1:])
                        param = vision_lang_adapter_dict[trimmed_name]
                        with torch.no_grad():
                            default_weight_loader(param, w)
                    else:
                        # LLM weights: yield them to be loaded
                        # by language_model.load_weights
                        yield (name, w)
    
            # Now we call the language model load with the generator
            self.language_model.load_weights(llm_weights_generator())

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

# this just wraps the model to create callable classes and traceble blocks for Neuron
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