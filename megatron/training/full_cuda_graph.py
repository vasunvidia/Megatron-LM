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

    static_buffers = {}

    def __init__(self):
        self.stream = torch.cuda.Stream()

    def __call__(self, inputs, stage, microbatch):
        assert stage in ['training', 'validation']
        if stage not in StaticBufferLoader.static_buffers:
            StaticBufferLoader.static_buffers[stage] = []
        assert microbatch <= len(StaticBufferLoader.static_buffers[stage])
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

    def __init__(
        self,
        forward_backward_func,
        cuda_graph_warmup_iters = -1,
    ):
        self.forward_backward_func = forward_backward_func
        self.static_loader = StaticBufferLoader()
        self.warmup_cuda_graph = cuda_graph_warmup_iters
        self.curr_iteration = 0
        self.cuda_graph = None
        self.result = None

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
            print (f'dl {len(data_list)}-{[hex(data_list[b]["tokens"].data_ptr()) for b in range(num_microbatches)]}')
            data_list = [iter(data_list)]
        else:
            assert isinstance(data_iterator, list) and len(data_iterator) == len(model)
            data_list = []
            for i in range(len(model)):
                data_list_i = []
                for b in range(num_microbatches):
                    data_list_i.append(self.static_loader(next(data_iterator[i]), 'training' if training else 'validation', b))
                print (f'dl_{i} {len(data_list_i)}-{[hex(data_list_i[b]["tokens"].data_ptr()) for b in range(num_microbatches)]}')
                data_list.append(iter(data_list_i))
        return data_list

    def __call__(
        self,
        *args,
        **kwargs):
        assert len(args)==0, 'forward_backward_func does not accept positional args'
        assert all([kwarg in kwargs for kwarg in ['model', 'data_iterator', 'num_microbatches', 'seq_length']])
        model = kwargs['model']
        num_microbatches = kwargs['num_microbatches']

        if isinstance(model, list):
            training = model[0].training
        else:
            training = model.training
        data_iterator = kwargs['data_iterator']
        data_list = self.data_read(data_iterator, model, training, num_microbatches)
        kwargs['data_iterator'] = data_list

        #print (f'FullCGWrapper iteration {self.curr_iteration} call self.forward_backward_func args {type(args)} kwargs {type(kwargs)} training? {training} num_microbatches {num_microbatches} data_iterator {type(data_iterator)}')

        if self.curr_iteration == self.warmup_cuda_graph:
            print (f'Capture CUDA graph!!!')
            assert self.cuda_graph is None
            self.cuda_graph = torch.cuda.CUDAGraph()
            for _, state in get_all_rng_states().items():
                self.cuda_graph.register_generator_state(state)
            torch.cuda.synchronize()
            capture_stream = torch.cuda.Stream()
            with torch.cuda.graph(self.cuda_graph, stream=capture_stream, capture_error_mode="global"):
                self.result = self.forward_backward_func (*args, **kwargs)
            torch.cuda.synchronize()
            print (f'CUDA graph capture done!!!')

        if self.cuda_graph is None:
            if torch.distributed.get_rank() == 0:
                print (f'run self.forward_backward_func')
            self.result = self.forward_backward_func (*args, **kwargs)
        else:
            if torch.distributed.get_rank() == 0:
                print (f'CG replay')
            self.cuda_graph.replay()
            
        self.curr_iteration += 1
        return self.result
