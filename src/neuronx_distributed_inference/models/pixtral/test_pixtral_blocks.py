
from pixtral_utils import (VisionEncoderArgs,
                            FeedForward,
                            Attention,
                            TransformerBlock,
                            Transformer,
                            VisionTransformer
                            )

from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig

from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import MultimodalVisionNeuronConfig, OnDeviceSamplingConfig

from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config


# pull from params.json
params = {
  "dim": 12288,
  "n_layers": 88,
  "head_dim": 128,
  "hidden_dim": 28672,
  "n_heads": 96,
  "n_kv_heads": 8,
  "rope_theta": 1000000000.0,
  "norm_eps": 1e-05,
  "vocab_size": 32768,
  "vision_encoder": {
    "hidden_size": 1408,
    "num_channels": 3,
    "image_size": 1024,
    "patch_size": 16,
    "rope_theta": 10000.0,
    "intermediate_size": 6144,
    "num_hidden_layers": 40,
    "num_attention_heads": 16,
    "image_token_id": 10,
    "image_break_token_id": 14,
    "image_end_token_id": 15,
    "adapter_bias": False
  }
}

if __name__ == "__main__":

    print ('Testing out the builds')
    
    vision_args = VisionEncoderArgs(**params['vision_encoder'])

    ff = FeedForward(vision_args)

    attention = Attention(vision_args)

    transformer_block = TransformerBlock(vision_args)

    transformer = Transformer(vision_args)

    vision_transformer = VisionTransformer(vision_args)

    print ('Success!')

    
