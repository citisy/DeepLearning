import copy
import math
import torch
from torch import nn


class ModuleInfo:
    @classmethod
    def std_profile(cls, model, depth=None, human_readable=True):
        from .visualize import TextVisualize

        profile = cls.profile_per_layer(model, depth=depth)
        cols = ('name', 'module', 'params', 'grads', 'args')
        lens = [-1] * len(cols)
        infos = []
        for p in profile:
            info = (
                p[0],
                p[1],
                TextVisualize.num_to_human_readable_str(p[2]["params"]) if human_readable else p[2]["params"],
                TextVisualize.num_to_human_readable_str(p[2]["grads"]) if human_readable else p[2]["grads"],
                TextVisualize.dict_to_str(p[2]["args"])
            )
            infos.append(info)
            for i, s in enumerate(info):
                l = len(str(s))
                if lens[i] < l:
                    lens[i] = l

        template = ''
        for l in lens:
            template += f'%-{l + 3}s'

        s = 'module info: \n'
        s += template % cols + '\n'
        s += template % tuple('-' * l for l in lens) + '\n'

        for info in infos:
            s += template % info + '\n'

        params = sum([p[2]["params"] for p in profile])
        grads = sum([p[2]["grads"] for p in profile])
        if human_readable:
            params = TextVisualize.num_to_human_readable_str(params)
            grads = TextVisualize.num_to_human_readable_str(grads)

        s += template % tuple('-' * l for l in lens) + '\n'
        s += template % ('sum', '', params, grads, '')
        return s, infos

    @classmethod
    def profile_per_layer(cls, module: nn.Module, depth=None):
        profiles = []

        def cur(current_m, dep, prev_name=''):
            for name, m in current_m._modules.items():
                if m is None:
                    continue
                if dep <= 1 or len(m._modules) == 0:
                    name = f'{prev_name}.{name}'[1:]
                    profiles.append((name, str(type(m))[8:-2], cls.profile(m)))
                else:
                    cur(m, dep - 1, f'{prev_name}.{name}')

        depth = depth or float('inf')
        cur(module, depth)
        return profiles

    @classmethod
    def profile(cls, module):
        return dict(
            params=cls.profile_params(module),
            grads=cls.profile_grads(module),
            args=cls.profile_args(module)
        )

    @staticmethod
    def profile_params(module):
        return sum(x.numel() for x in module.parameters() if not isinstance(x, nn.UninitializedParameter))

    @staticmethod
    def profile_grads(module):
        return sum(x.numel() for x in module.parameters() if x.requires_grad and not isinstance(x, nn.UninitializedParameter))

    @staticmethod
    def profile_args(module):
        args = {}
        if hasattr(module, 'in_channels'):
            args['i_ch'] = module.in_channels
        if hasattr(module, 'out_channels'):
            args['o_ch'] = module.out_channels
        if hasattr(module, 'in_features'):
            args['i_f'] = module.in_features
        if hasattr(module, 'out_features'):
            args['o_f'] = module.out_features
        if hasattr(module, 'input_size'):
            args['i_size'] = module.input_size
        if hasattr(module, 'output_size'):
            args['o_size'] = module.output_size
        if hasattr(module, 'num_embeddings'):
            args['n_emb'] = module.num_embeddings
        if hasattr(module, 'embedding_dim'):
            args['emb_dim'] = module.embedding_dim
        if hasattr(module, 'kernel_size'):
            k = module.kernel_size
            if isinstance(k, (list, tuple)):
                k = k[0]
            args['k'] = k
        if hasattr(module, 'stride'):
            s = module.stride
            if isinstance(s, (list, tuple)):
                s = s[0]
            args['s'] = s
        if hasattr(module, 'padding'):
            p = module.padding
            if isinstance(p, (list, tuple)):
                p = p[0]
            args['p'] = p

        return args

    @staticmethod
    def profile_flops(module, input_size, *test_args):
        import thop

        p = next(module.parameters())
        test_data = torch.empty((1, *input_size), device=p.device)
        flops = thop.profile(copy.deepcopy(module), inputs=(test_data, *test_args), verbose=False)[0] / 1E9 * 2  # stride GFLOPs

        return flops


def initialize_layers(module, init_gain=0.02, init_type='normal'):
    """trace each module, initialize the variables
    if module has `initialize_layers`, use `module.initialize_layers()` to initialize"""

    def cur(current_m):
        for name, m in current_m._modules.items():
            if m is None:
                continue

            if hasattr(m, 'initialize_layers'):
                m.initialize_layers()
                continue

            t = type(m)

            if t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
                m.weight.data.normal_(1.0, init_gain)
                m.bias.data.fill_(0.)

            elif t is nn.LayerNorm:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True

            elif t in [nn.Conv2d, nn.Linear, nn.Embedding]:
                if init_type == 'normal':
                    nn.init.normal_(m.weight, 0.0, init_gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight, gain=init_gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, a=0)
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight, gain=init_gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

            elif t in [nn.ConvTranspose2d]:
                m.weight.data.copy_(bilinear_kernel(m.in_channels, m.out_channels, m.kernel_size[0]))

            if len(m._modules) != 0:
                cur(m)

    cur(module)


def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1), torch.arange(kernel_size).reshape(1, -1))
    f = (1 - torch.abs(og[0] - center) / factor) * (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size))
    ch = min(in_channels, out_channels)
    weight[range(ch), range(ch), :, :] = f
    return weight


class EarlyStopping:
    def __init__(self, thres=0.005, patience=None, min_epoch=0, ignore_min_score=-1,
                 verbose=True, stdout_method=print):
        self.thres = thres
        self.best_score = -1
        self.best_epoch = 0
        self.acc_epoch = 0
        self.last_epoch = 0
        self.min_epoch = min_epoch
        self.ignore_min_score = ignore_min_score
        self.patience = patience or float('inf')
        self.verbose = verbose
        self.stdout_method = stdout_method

    def __call__(self, epoch, score):
        if epoch < self.min_epoch or score < self.ignore_min_score:
            self.last_epoch = epoch
            return False

        if score - self.best_score > self.thres:
            self.acc_epoch = 0
        elif abs(self.best_score - score) <= self.thres:
            self.acc_epoch += epoch - self.last_epoch

        if score > self.best_score:
            self.best_epoch = epoch
            self.best_score = score

        self.last_epoch = epoch
        stop = self.acc_epoch >= self.patience
        if stop and self.verbose:
            self.stdout_method(f'Early Stopping training. Best results observed at epoch {self.best_epoch}, and best score is {self.best_score}')
        return stop

    def state_dict(self):
        return dict(
            last_epoch=self.last_epoch,
            acc_epoch=self.acc_epoch,
            best_epoch=self.best_epoch,
            best_score=self.best_score
        )

    def load_state_dict(self, items: dict):
        self.__dict__.update(items)


def export_formats():
    import pandas as pd

    x = [
        ['PyTorch', '-', '.pt', True],
        ['TorchScript', 'torchscript', '.torchscript', True],
        ['ONNX', 'onnx', '.onnx', True],
        ['OpenVINO', 'openvino', '_openvino_model', False],
        ['TensorRT', 'engine', '.engine', True],
        ['CoreML', 'coreml', '.mlmodel', False],
        ['TensorFlow SavedModel', 'saved_model', '_saved_model', True],
        ['TensorFlow GraphDef', 'pb', '.pb', True],
        ['TensorFlow Lite', 'tflite', '.tflite', False],
        ['TensorFlow Edge TPU', 'edgetpu', '_edgetpu.tflite', False],
        ['TensorFlow.js', 'tfjs', '_web_model', False], ]
    return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'GPU'])


class Export:
    @staticmethod
    def to_pickle(model):
        pass

    @staticmethod
    def to_torchscript(model, *trace_input, **export_kwargs):
        """note that, dynamic python script change to static c++ script, according to trace the code
        so, such as `if...else...`, 'for...in...`, etc., if trace in a dynamic variable,
        will cause some unexpectedly bugs"""
        with torch.no_grad():
            model.eval()
            # warmup, make sure that the model is initialized right
            model(*trace_input)
            jit_model = torch.jit.trace(model, trace_input, **export_kwargs)

        return jit_model

    @staticmethod
    def to_onnx(model, f, *trace_input, **export_kwargs):
        torch.onnx.export(model=model, f=f, args=trace_input, **export_kwargs)


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


class EMA:
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, cur_step=0, decay=0.9999, step_start_ema=0, tau=2000):
        """

        Args:
            cur_step:
            decay:
            step_start_ema:
            tau: growth factor
                when x is larger, y is larger
                x is closed to 0, y is closed to 0
                x is closed 3 * tau, y is closed to exp(-3)*decay=0.95*decay
                tau is closed to 0, y is closed to decay
        """
        super().__init__()
        self.cur_step = cur_step
        self.step_start_ema = step_start_ema
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))

    @staticmethod
    def copy(model):
        ema_model = copy.deepcopy(model)
        ema_model.requires_grad_(False)
        return ema_model

    @torch.no_grad()
    def update_attr(self, model, ema_model):
        # beta is larger, weights are closed to old model
        beta = self.decay(self.cur_step)

        msd = de_parallel(model).state_dict()  # model state_dict
        for k, v in ema_model.state_dict().items():
            if v.dtype.is_floating_point:
                v *= beta
                v += (1 - beta) * msd[k].detach()

    @torch.no_grad()
    def copy_attr(self, model, ema_model, include=(), exclude=()):
        for k, v in model.__dict__.items():
            if (len(include) and k not in include) or k.startswith('_') or k in exclude:
                continue
            else:
                setattr(ema_model, k, v)
        ema_model.load_state_dict(ema_model.state_dict())

    def step(self, model, ema_model):
        if self.cur_step < self.step_start_ema:
            self.copy_attr(model, ema_model)
        else:
            self.update_attr(model, ema_model)
        self.cur_step += 1
        return self.cur_step
