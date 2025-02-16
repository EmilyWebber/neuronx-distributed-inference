
from pixtral_utils import (VisionEncoderArgs,
                            FeedForward,
                            Attention,
                            TransformerBlock,
                            Transformer,
                            VisionTransformer,
                            VisionLanguageAdapter
                            )

from neuronx_distributed_inference.models.config import MultimodalVisionNeuronConfig, OnDeviceSamplingConfig

from modeling_pixtral import (PixtralInferenceConfig,
                              NeuronPixtralForConditionalGeneration
                              )

from modeling_mistral import NeuronMistralForCausalLM

# pull from params.json and cast as nested dict
params = {'text_config' : {
  "dim": 12288,
  "n_layers": 88,
  "head_dim": 128,
  "hidden_dim": 28672,
  "hidden_size": 28672, # adding again with second string identified size vs dim
  "n_heads": 96,
  "n_kv_heads": 8,
  "rope_theta": 1000000000.0,
  "norm_eps": 1e-05,
  "vocab_size": 32768,
  "pad_token_id": 128004}, # the pad token id is added from llama 3.2 11b vision instruct
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
    "adapter_bias": False,
    "max_num_tiles": 4 #copying from mllama
  }
}

def test_all_obj_builds(params):
    print ('Testing out the builds')

    vision_args = VisionEncoderArgs(**params['vision_encoder'])

    ff = FeedForward(vision_args)

    attention = Attention(vision_args)

    transformer_block = TransformerBlock(vision_args)

    transformer = Transformer(vision_args)

    vision_transformer = VisionTransformer(vision_args)

    vision_adapter = VisionLanguageAdapter(vision_args, dim = params['dim'])
    
    print ('Success on the builds!') 

if __name__ == "__main__":

    batch_size = 1
    num_img_per_prompt = 1
    max_context_length = 1024
    seq_len = 2048

    checkpoint_path = '/home/ubuntu/models/pixtral'

    trace_path = '/home/ubuntu/models/traced_pixtral'
    
    neuron_config = MultimodalVisionNeuronConfig(
        tp_degree=32,
        batch_size=batch_size,
        max_context_length=max_context_length,
        seq_len=seq_len,
        on_device_sampling_config=OnDeviceSamplingConfig(dynamic=False),
        enable_bucketing=True,
        sequence_parallel_enabled=False,
        fused_qkv=False,
        async_mode=False,
    )
    
    pixtral_config = PixtralInferenceConfig(neuron_config, params)

    neuron_pixtral = NeuronPixtralForConditionalGeneration(checkpoint_path, pixtral_config)

    neuron_pixtral.compile(trace_path)
    