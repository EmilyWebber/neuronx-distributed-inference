
from modeling_pixtral import PixtralInferenceConfig
from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig

from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import MultimodalVisionNeuronConfig, OnDeviceSamplingConfig

from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

if __name__ == "__main__":
    print ('Now we will test imports and builds on key Pixtral building blocks')

    print ('Using a config from mLlama')
    model_path = '/home/ubuntu/models/mllama_90b'
    
    # Initialize configs and tokenizer.
    batch_size = 1
    num_img_per_prompt = 1
    max_context_length = 1024
    seq_len = 2048

    generation_config = GenerationConfig.from_pretrained(model_path)
    generation_config_kwargs = {
        "top_k": 1,
    }
    generation_config.update(**generation_config_kwargs)

    on_device_sampling_config=OnDeviceSamplingConfig(dynamic=True)

    neuron_config = MultimodalVisionNeuronConfig(
        tp_degree=32,
        batch_size=batch_size,
        max_context_length=max_context_length,
        seq_len=seq_len,
        on_device_sampling_config=on_device_sampling_config,
        enable_bucketing=True,
        sequence_parallel_enabled=False,
        fused_qkv=False,
        async_mode=False,
    )

    config = PixtralInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )