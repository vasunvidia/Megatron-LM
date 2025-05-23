# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.


import pytest
import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    get_transformer_layer_offset,
)
from tests.unit_tests.test_utilities import Utils


class TestParallelTransformerLayer:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True
        )
        self.parallel_transformer_layer = TransformerLayer(
            transformer_config, get_gpt_layer_with_transformer_engine_spec().submodules
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        parallel_transformer_layer = self.parallel_transformer_layer
        assert isinstance(parallel_transformer_layer, TransformerLayer)
        assert parallel_transformer_layer.layer_number == 1

        num_weights = sum([p.numel() for p in parallel_transformer_layer.parameters()])
        assert num_weights == 1884

    def test_gpu_forward(self):
        parallel_transformer_layer = self.parallel_transformer_layer
        config: TransformerConfig = parallel_transformer_layer.config
        sequence_length = 32
        micro_batch_size = 2
        parallel_transformer_layer.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((sequence_length, micro_batch_size, config.hidden_size))
        hidden_states = hidden_states.cuda()

        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

        hidden_states, context = parallel_transformer_layer(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        assert hidden_states.shape[0] == sequence_length
        assert hidden_states.shape[1] == micro_batch_size
        assert hidden_states.shape[2] == config.hidden_size

    def test_chunked_mlp(self):
        with torch.no_grad():

            def test(
                num_layers,
                hidden_size,
                num_attention_heads,
                mlp_chunks_for_prefill,
                hidden_states,
                inference_context,
            ):

                transformer_config = TransformerConfig(
                    num_layers=2,
                    hidden_size=12,
                    num_attention_heads=4,
                    mlp_chunks_for_prefill=4,
                    add_bias_linear=True,
                    use_cpu_initialization=True,
                )
                parallel_transformer_layer = TransformerLayer(
                    transformer_config, get_gpt_layer_with_transformer_engine_spec().submodules
                )

                parallel_transformer_layer.cuda()

                hidden_states, context = parallel_transformer_layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    inference_context=inference_context,
                )

                return hidden_states, context

            num_layers = 2
            hidden_size = 12
            num_attention_heads = 4

            sequence_length = 32
            micro_batch_size = 2

            # [sequence length, batch size, hidden size]
            input_hidden_states = torch.ones((sequence_length, micro_batch_size, hidden_size))
            input_hidden_states = input_hidden_states.cuda()

            attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()

            inference_context = StaticInferenceContext(
                max_batch_size=micro_batch_size, max_sequence_length=sequence_length
            )

            outputs = {}

            for mlp_chunks_for_prefill in [1, 4]:
                hidden_states, context = test(
                    num_layers,
                    hidden_size,
                    num_attention_heads,
                    mlp_chunks_for_prefill,
                    input_hidden_states,
                    inference_context,
                )
                assert hidden_states.shape[0] == sequence_length
                assert hidden_states.shape[1] == micro_batch_size
                assert hidden_states.shape[2] == hidden_size

                outputs[mlp_chunks_for_prefill] = (hidden_states, context)

        assert torch.equal(outputs[1][0], outputs[4][0])

    def test_get_layer_offset(self):
        config = self.parallel_transformer_layer.config
        assert get_transformer_layer_offset(config) == 0

    @pytest.mark.parametrize('order', ['tp-pp-dp', 'tp-dp-pp'])
    @pytest.mark.parametrize('tp_pp', [(4, 2), (1, 1), (8, 1), (2, 2)])
    def test_sharded_state_dict(self, tp_pp, order):
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(*tp_pp, order=order)

        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=128, num_attention_heads=8, use_cpu_initialization=True
        )
        parallel_transformer_layer = TransformerLayer(
            transformer_config, get_gpt_layer_with_transformer_engine_spec().submodules
        )

        sharded_state_dict = parallel_transformer_layer.sharded_state_dict()

        extra_states = {k: v for k, v in sharded_state_dict.items() if k.endswith('extra_state')}
        sharded_tensors = {
            k: v for k, v in sharded_state_dict.items() if not k.endswith('extra_state')
        }
        assert all(isinstance(t, ShardedObject) for t in extra_states.values())
        assert all(isinstance(t, ShardedTensor) for t in sharded_tensors.values())

        # Test all local shapes
        tensor_local_shapes = {k: v.local_shape for k, v in sharded_tensors.items()}
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        assert tensor_local_shapes == get_tensor_shapes_for_tp(transformer_config, tp_size)

        # Test all global shapes. Prepend num layers in front of expected shapes
        tensor_global_shapes = {k: v.global_shape for k, v in sharded_tensors.items()}
        expected_global_shapes = get_tensor_shapes_for_tp(transformer_config, 1)
        assert tensor_global_shapes == expected_global_shapes

        # Test ShardedTensor keys
        for state_dict_key, sh_ten in sharded_tensors.items():
            assert state_dict_key == sh_ten.key

        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(1, 1)


def get_tensor_shapes_for_tp(transformer_config, tp_size):
    hs = transformer_config.hidden_size
    return {
        'mlp.linear_fc1.layer_norm_weight': (hs,),
        'mlp.linear_fc1.layer_norm_bias': (hs,),
        'mlp.linear_fc1.weight': (hs * 4 // tp_size, hs),
        'mlp.linear_fc1.bias': (hs * 4 // tp_size,),
        'mlp.linear_fc2.weight': (hs, hs * 4 // tp_size),
        'mlp.linear_fc2.bias': (hs,),
        'self_attention.linear_proj.weight': (hs, hs // tp_size),
        'self_attention.linear_proj.bias': (hs,),
        'self_attention.linear_qkv.layer_norm_weight': (hs,),
        'self_attention.linear_qkv.layer_norm_bias': (hs,),
        'self_attention.linear_qkv.weight': (hs * 3 // tp_size, hs),
        'self_attention.linear_qkv.bias': (hs * 3 // tp_size,),
    }
