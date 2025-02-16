from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Set, Tuple, Union

from torch import nn, Tensor
import torch

from vllm.model_executor.layers.layernorm import RMSNorm

from neuronx_distributed.parallel_layers import ColumnParallelLinear, RowParallelLinear

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

from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding

from neuronx_distributed.parallel_layers import parallel_state 

from neuronx_distributed_inference.models.config import InferenceConfig

@dataclass
class VisionEncoderArgs:
    hidden_size: int
    num_channels: int
    image_size: int
    patch_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    rope_theta: float  # for rope-2D
    image_token_id: int
    image_break_token_id: int
    image_end_token_id: int
    adapter_bias: bool = True

class FeedForward(nn.Module):

    def __init__(self, args: VisionEncoderArgs):
        super().__init__()
        assert args.intermediate_size is not None

        self.w1 = nn.Linear(args.hidden_size,
                                       args.intermediate_size,
                                       bias=False)

        self.w2 = nn.Linear(args.intermediate_size,
                                    args.hidden_size,
                                    bias=False)
        
        self.w3 = nn.Linear(args.hidden_size,
                            args.intermediate_size,
                            bias=False)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(NeuronAttentionBase):
    def __init__(
            self, config: VisionEncoderArgs):
        super().__init__()
        self.config = config
        self.neuron_config = config.neuron_config
        self.hidden_size = 1408
        self.num_attention_heads = 16
        
        # print ('To do need to confirm number of kv heads in vision encoder, is it really ', 8)
        self.num_key_value_heads = 8
        
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.padding_side = self.neuron_config.padding_side
        self.torch_dtype = self.neuron_config.torch_dtype
        self.is_medusa = self.neuron_config.is_medusa
        self.num_cores_per_group = config.num_cores_per_group
        # self.max_position_embeddings = config.neuron_config.max_position_embeddings

        self.attn_kernel_enabled = True

        if parallel_state.model_parallel_is_initialized():
            self.tp_degree = parallel_state.get_tensor_model_parallel_size()
        else:
            self.tp_degree = 1
            
        self.fused_qkv = getattr(self.neuron_config, "fused_qkv", False)
        self.clip_qkv = None
 
        self.sequence_parallel_enabled = self.neuron_config.sequence_parallel_enabled
        self.sequence_dimension = 1 if self.sequence_parallel_enabled else None

        self.init_gqa_properties()

        # self.init_rope()

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        print ('Attention forward pass is not implemented')
        # batch, patches, _ = x.shape

        # q, k, v = self.wq(x), self.wk(x), self.wv(x)
        # q = q.reshape(batch, patches, self.n_heads, self.head_dim)
        # k = k.reshape(batch, patches, self.n_heads, self.head_dim)
        # v = v.reshape(batch, patches, self.n_heads, self.head_dim)

        # q, k = apply_rotary_emb_vit(q, k, freqs_cis=freqs_cis)
        # out = xops.memory_efficient_attention(q, k, v, attn_bias=mask)
        # out = out.reshape(batch, patches, self.n_heads * self.head_dim)
        # return self.wo(out)

    def init_rope(self):
        if not hasattr(self.config, "rope_scaling") or self.config.rope_scaling is None:
            # TODO(yihsian): Check if we can just use our own implementation
            if self.is_medusa:
                self.rotary_emb = LlamaRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                )
            else:
                self.rotary_emb = RotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                )
        else:
            rope_type = self.config.rope_scaling.get(
                "rope_type", self.config.rope_scaling.get("type", None)
            )
            if rope_type == "llama3":
                self.rotary_emb = Llama3RotaryEmbedding(
                    dim=self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                    factor=self.config.rope_scaling["factor"],
                    low_freq_factor=self.config.rope_scaling["low_freq_factor"],
                    high_freq_factor=self.config.rope_scaling["high_freq_factor"],
                    original_max_position_embeddings=self.config.rope_scaling[
                        "original_max_position_embeddings"
                    ],
                )
            else:
                # LlamaRotaryEmbedding automatically chooses the correct scaling type from config.
                # Warning: The HF implementation may have precision issues when run on Neuron.
                # We include it here for compatibility with other scaling types.
                self.rotary_emb = LlamaRotaryEmbedding(self.config)

class TransformerBlock(nn.Module):

    def __init__(self, args: VisionEncoderArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.hidden_size, eps=1e-5)
        self.ffn_norm = RMSNorm(args.hidden_size, eps=1e-5)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x),
                                   mask=mask,
                                   freqs_cis=freqs_cis)
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out

class Transformer(nn.Module):

    def __init__(self, args: VisionEncoderArgs):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for _ in range(args.num_hidden_layers):
            self.layers.append(TransformerBlock(args))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        freqs_cis: Optional[torch.Tensor],
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask, freqs_cis=freqs_cis)
        return x

   
class VisionTransformer(nn.Module):

    def __init__(self, args: VisionEncoderArgs):
        super().__init__()
        self.args = args
        self.patch_conv = nn.Conv2d(
            in_channels=args.num_channels,
            out_channels=args.hidden_size,
            kernel_size=args.patch_size,
            stride=args.patch_size,
            bias=False,
        )
        self.ln_pre = RMSNorm(args.hidden_size, eps=1e-5)
        self.transformer = Transformer(args)

        head_dim = self.args.hidden_size // self.args.num_attention_heads
        assert head_dim % 2 == 0, "ROPE requires even head_dim"
        self._freqs_cis: Optional[torch.Tensor] = None

    @property
    def max_patches_per_side(self) -> int:
        return self.args.image_size // self.args.patch_size

    @property
    def device(self) -> torch.types.Device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def freqs_cis(self) -> torch.Tensor:
        if self._freqs_cis is None:
            self._freqs_cis = precompute_freqs_cis_2d(
                dim=self.args.hidden_size // self.args.num_attention_heads,
                height=self.max_patches_per_side,
                width=self.max_patches_per_side,
                theta=self.args.rope_theta,
            )

        if self._freqs_cis.device != self.device:
            self._freqs_cis = self._freqs_cis.to(device=self.device)

        return self._freqs_cis

    def forward(
        self,
        images: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            images: list of N_img images of variable sizes, 
                each of shape (C, H, W)
        Returns:
            image_features: tensor of token features for 
                all tokens of all images of shape (N_toks, D)
        """
        # pass images through initial convolution independently
        patch_embeds_list = [
            self.patch_conv(img.unsqueeze(0).to(self.dtype)) for img in images
        ]

        # flatten to a single sequence
        patch_embeds = torch.cat(
            [p.flatten(2).permute(0, 2, 1) for p in patch_embeds_list], dim=1)
        patch_embeds = self.ln_pre(patch_embeds)

        # positional embeddings
        positions = position_meshgrid(patch_embeds_list).to(self.device)
        freqs_cis = self.freqs_cis[positions[:, 0], positions[:, 1]]

        # pass through Transformer with a block diagonal mask delimiting images
        if USE_XFORMERS_OPS:
            mask = xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(
                [p.shape[-2] * p.shape[-1] for p in patch_embeds_list], )
        else:
            raise ImportError("Xformers is required for Pixtral inference "
                              "with the Mistral format")
        out = self.transformer(patch_embeds, mask=mask, freqs_cis=freqs_cis)

        # remove batch dimension of the single sequence
        return out.squeeze(0)

class VisionLanguageAdapter(nn.Module):

    def __init__(self, args: VisionEncoderArgs, dim: int):
        super().__init__()
        assert isinstance(args, VisionEncoderArgs)
        self.w_in = nn.Linear(
            args.hidden_size,
            dim,
            bias=args.adapter_bias,
        )
        self.gelu = nn.GELU()
        self.w_out = nn.Linear(dim, dim, bias=args.adapter_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(self.gelu(self.w_in(x)))
