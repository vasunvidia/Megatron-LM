import os
import torch
from megatron.core.tensor_parallel.random import get_all_rng_states

def struct_copy_one(src):
    if isinstance(src, tuple):
        return tuple(struct_copy_one(i) for i in src)
    elif isinstance(src, list):
        return list(struct_copy_one(i) for i in src)
    elif isinstance(src, dict):
        return {k: struct_copy_one(src[k]) for k in src}
    elif isinstance(src, torch.Tensor):
        return src.clone().detach().cuda()
    else:
        return src


def struct_copy_two(tgt, src):
    if isinstance(src, tuple):
        raise Exception(f"Unsupported copy for tuple yet: {type(src)}")
    elif isinstance(src, list):
        for i in range(len(src)):
            if isinstance(src[i], (tuple, list, dict, torch.Tensor)):
                struct_copy_two(tgt[i], src[i])
            else:
                tgt[i] = src[i]
    elif isinstance(src, dict):
        for k in src:
            if isinstance(src[k], (tuple, list, dict, torch.Tensor)):
                struct_copy_two(tgt[k], src[k])
            else:
                tgt[k] = src[k]
    elif isinstance(src, torch.Tensor):
        tgt.copy_(src, non_blocking=True)
    else:
        raise Exception(f"Expect top-level as container type but got: {type(src)}")

class StaticBufferLoader:
    """Load data to static buffers."""

    static_buffers = {'training': [], 'validation': []}

    def __init__(self):
        self.stream = torch.cuda.Stream()

    def __call__(self, inputs, stage, microbatch):
        assert stage in ['training', 'validation']
        assert microbatch <= len(StaticBufferLoader.static_buffers[stage])
        if isinstance(inputs, tuple) and isinstance(inputs[0], dict):
            inputs = inputs[0]

        assert isinstance(inputs, dict)
        if microbatch == len(
            StaticBufferLoader.static_buffers[stage]
        ):
            with torch.cuda.stream(self.stream):
                StaticBufferLoader.static_buffers[stage].append(
                    struct_copy_one(inputs)
                )
        else:

            for k in inputs.keys():
                if (
                    k
                    not in StaticBufferLoader.static_buffers[stage][
                        microbatch
                    ]
                ):
                    StaticBufferLoader.static_buffers[stage][
                        microbatch
                    ][k] = torch.empty_like(inputs[k]).cuda()

            with torch.cuda.stream(self.stream):
                struct_copy_two(
                    StaticBufferLoader.static_buffers[stage][
                        microbatch
                    ],
                    inputs,
                )
        torch.cuda.current_stream().wait_stream(self.stream)
        return StaticBufferLoader.static_buffers[stage][microbatch]


class FullCGWrapper:

    curr_iteration = {'training': 0, 'validation': 0}
    cuda_graph = {'training': None, 'validation': None}
    result = {'training': None, 'validation': None}
    def __init__(
        self,
        forward_backward_func
    ):
        self.forward_backward_func = forward_backward_func
        self.static_loader = StaticBufferLoader()
        self.warmup_cuda_graph = int(os.getenv('FULL_CUDA_GRAPH_WARMUP_ITERS', '11'))

    def data_read(
        self,
        data_iterator,
        model,
        training,
        num_microbatches
    ):
        if not isinstance(model, list) or len(model) == 1:
            assert not isinstance(data_iterator, list) or len(data_iterator) == 1
            iterator0 = data_iterator if not isinstance(data_iterator, list) else data_iterator[0]
            data_list = []
            for b in range(num_microbatches):
                data_list.append(self.static_loader(next(iterator0), 'training' if training else 'validation', b))
            data_list = [iter(data_list)]
        else:
            assert isinstance(data_iterator, list) and len(data_iterator) == len(model)
            data_list = []
            for i in range(len(model)):
                data_list_i = []
                for b in range(num_microbatches):
                    data_list_i.append(self.static_loader(next(data_iterator[i]), 'training' if training else 'validation', b))
                data_list.append(iter(data_list_i))
        return data_list

    def __call__(
        self,
        *args,
        **kwargs):
        assert len(args)==0, 'forward_backward_func does not accept positional args'
        assert all([kwarg in kwargs for kwarg in ['model', 'data_iterator', 'num_microbatches', 'seq_length', 'forward_only']])
        model = kwargs['model']
        num_microbatches = kwargs['num_microbatches']

        training = not kwargs['forward_only']
        data_iterator = kwargs['data_iterator']
        data_list = self.data_read(data_iterator, model, training, num_microbatches)
        kwargs['data_iterator'] = data_list

        training_str = 'training' if training else 'validation'
        curr_iteration = self.cur_iter(training_str)
        if curr_iteration == self.warmup_cuda_graph:
            if torch.distributed.get_rank() == 0:
                print (f'Capture CUDA graph for {training_str}!!!')
            torch.distributed.barrier()
            assert FullCGWrapper.cuda_graph[training_str] is None
            FullCGWrapper.cuda_graph[training_str] = torch.cuda.CUDAGraph()
            for _, state in get_all_rng_states().items():
                FullCGWrapper.cuda_graph[training_str].register_generator_state(state)
            torch.cuda.synchronize()
            capture_stream = torch.cuda.Stream()
            with torch.cuda.graph(FullCGWrapper.cuda_graph[training_str], stream=capture_stream, capture_error_mode="thread_local"):
                FullCGWrapper.result[training_str] = self.forward_backward_func (*args, **kwargs)
            torch.cuda.synchronize()
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                print (f'CUDA graph capture done!!!')

        if FullCGWrapper.cuda_graph[training_str] is None:
            FullCGWrapper.result[training_str] = self.forward_backward_func (*args, **kwargs)
        else:
            FullCGWrapper.cuda_graph[training_str].replay()
            
        self.next_iter(training_str)
        return FullCGWrapper.result[training_str]

    def cur_iter(self, stage):
        return FullCGWrapper.curr_iteration[stage]
    def next_iter(self, stage):
        FullCGWrapper.curr_iteration[stage] += 1
