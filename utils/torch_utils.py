import re
import copy
import math
import pandas as pd
import numpy as np
import torch
from torch import nn
from pathlib import Path
from collections import OrderedDict


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

    @staticmethod
    def possible_device(module):
        """Returns the first found device in parameters, otherwise returns the first found device in tensors."""
        try:
            return next(module.parameters()).device
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5

            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = module._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].device

    @staticmethod
    def possible_dtype(module):
        """Returns the first found floating dtype in parameters if there is one, otherwise returns the last dtype it found."""
        last_dtype = None
        for t in module.parameters():
            last_dtype = t.dtype
            if t.is_floating_point():
                return t.dtype

        if last_dtype is not None:
            # if no floating dtype was found return whatever the first dtype is
            return last_dtype

        else:
            # For nn.DataParallel compatibility in PyTorch > 1.5
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = module._named_members(get_members_fn=find_tensor_attributes)
            last_tuple = None
            for tuple in gen:
                last_tuple = tuple
                if tuple[1].is_floating_point():
                    return tuple[1].dtype

            # fallback to the last dtype
            return last_tuple[1].dtype


class ModuleManager:
    @staticmethod
    def is_parallel(module):
        """Returns True if model is of type DP or DDP"""
        return type(module) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

    @classmethod
    def de_parallel(cls, module):
        """De-parallelize a model: returns single-GPU model if model is of type DP or DDP"""
        return module.module if cls.is_parallel(module) else module

    @staticmethod
    def freeze_module(module: nn.Module, allow_train=False):
        module.requires_grad_(False)
        if not allow_train:
            # module only be allowed to eval, does not change to train mode anymore
            module.eval()
            module.train = lambda self, mode=True: self

    @staticmethod
    def convert_to_quantized_module(module: nn.Module, trace_func=None, backend='fbgemm'):
        """note, can not convert Embedding layer"""
        module.eval()
        module.qconfig = torch.quantization.get_default_qconfig(backend)
        torch.quantization.prepare(module, inplace=True)
        if trace_func is not None:
            with torch.no_grad():
                # help to collect the running info for quantization
                trace_func(module)
        torch.quantization.convert(module, inplace=True)

    @staticmethod
    def low_memory_run(module: nn.Module, device, *args, **kwargs):
        """only send the module to gpu when the module need to be run,
        and the gpu will be released after running"""
        module.to(device)
        obj = module(*args, **kwargs)
        module.cpu()
        torch.cuda.empty_cache()
        return obj

    @staticmethod
    def assign_device_run(module: nn.Module, device, *args, **kwargs):
        """let module run in the assigned device"""
        module.to(device)
        args = [obj.to(device) if isinstance(obj, torch.Tensor) else obj for obj in args]
        kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

        return module(*args, **kwargs)

    @classmethod
    def initialize_layers(cls, module, init_gain=0.02, init_type='normal'):
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
                    m.weight.data.copy_(cls.bilinear_kernel(m.in_channels, m.out_channels, m.kernel_size[0]))

                if len(m._modules) != 0:
                    cur(m)

        cur(module)

    @staticmethod
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

    @staticmethod
    def get_module_by_name(module, key=None, include=(), exclude=(), is_last=False):
        """
        >>> module = ...
        >>> ModuleManager.get_module_by_name(module, key='q')
        >>> ModuleManager.get_module_by_name(module, include=('q', 'k', 'v'), exclude=('l0.q', 'l0.k', 'l0.v'))
        """
        def cur(current_m: nn.Module, prev_name=''):
            for name, m in current_m._modules.items():
                if m is None:
                    continue

                full_name = f'{prev_name}.{name}'[1:]

                if is_last:
                    for k in include:
                        if full_name.endswith(k):
                            r.append((current_m, name, full_name))

                elif len(m._modules) == 0:
                    if is_find(full_name):
                        r.append((current_m, name, full_name))

                if len(m._modules) > 0:
                    cur(m, f'{prev_name}.{name}')

        def is_find(name):
            flag = False
            for k in include:
                if k in name:
                    flag = True

            for k in exclude:
                if k in name:
                    flag = False

            return flag

        r = []
        if key is not None:
            include += (key, )
        cur(module)
        return r


class WeightsFormats:
    formats = pd.DataFrame([
        ['PyTorch', '-', '.pt/.pth/.ckpt/.bin', True],
        ['TorchScript', 'torchscript', '.torchscript', True],
        ['Safetensors', 'safetensors', '.safetensors', True],
        ['ONNX', 'onnx', '.onnx', True],
        ['OpenVINO', 'openvino', '_openvino_model', False],
        ['TensorRT', 'engine', '.engine', True],
        ['CoreML', 'coreml', '.mlmodel', False],
        ['TensorFlow', '-', '.ckpt/.h5', True],
        ['TensorFlow SavedModel', 'saved_model', '_saved_model', True],
        ['TensorFlow GraphDef', 'pb', '.pb', True],
        ['TensorFlow Lite', 'tflite', '.tflite', False],
        ['TensorFlow Edge TPU', 'edgetpu', '_edgetpu.tflite', False],
        ['TensorFlow.js', 'tfjs', '_web_model', False],
    ], columns=['format', 'argument', 'suffix', 'GPU'])

    @classmethod
    def get_format_from_suffix(cls, save_path):
        suffix = Path(save_path).suffix
        k = None
        for i, row in cls.formats.iterrows():
            if suffix in row['suffix']:
                k = row['format']
                break

        return k


class Export:
    @staticmethod
    def to_torchscript(model, *trace_input, **export_kwargs):
        """note that, dynamic python script change to static c++ script, according to trace the code
        so, such as `if...else...`, 'for...in...`, etc., if trace in a dynamic variable,
        will cause some unexpectedly bugs"""
        model.eval()
        with torch.no_grad():
            # warmup, make sure that the model is initialized right
            model(*trace_input)
            jit_model = torch.jit.trace(model, trace_input, **export_kwargs)

        return jit_model

    @staticmethod
    def to_onnx(model, f, *trace_input, **export_kwargs):
        torch.onnx.export(model=model, f=f, args=trace_input, **export_kwargs)


class Load:
    @classmethod
    def from_file(cls, save_path, **kwargs):
        load_dict = {
            'PyTorch': cls.from_ckpt,
            'TorchScript': cls.from_jit,
            'Safetensors': cls.from_save_tensor
        }
        k = WeightsFormats.get_format_from_suffix(save_path)
        return load_dict.get(k)(save_path, **kwargs)

    @staticmethod
    def from_ckpt(save_path, **kwargs):
        return torch.load(save_path, **kwargs)

    @staticmethod
    def from_tf_ckpt(save_path, var_names=None, key_types=None, value_types=None, **kwargs):
        import tensorflow as tf

        loader = lambda name: torch.from_numpy(tf.train.load_variable(save_path, name))
        assert var_names
        tensors = OrderedDict()
        for name in var_names:
            tensors[name] = loader(name)

        tensors = Converter.tensors_from_tf_to_torch(tensors, key_types, value_types)
        return tensors

    @staticmethod
    def from_save_tensor(save_path, **kwargs):
        from safetensors import safe_open

        tensors = OrderedDict()
        with safe_open(save_path, framework="pt", **kwargs) as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)

        return tensors

    @staticmethod
    def from_jit(save_path, **kwargs):
        return torch.jit.load(save_path, **kwargs)


class EarlyStopping:
    def __init__(self, thres=0.005, patience=None, min_period=0, ignore_min_score=-1,
                 verbose=True, stdout_method=print):
        self.thres = thres
        self.best_score = -1
        self.best_period = 0
        self.acc_period = 0
        self.last_period = 0
        self.min_period = min_period
        self.ignore_min_score = ignore_min_score
        self.patience = patience or float('inf')
        self.verbose = verbose
        self.stdout_method = stdout_method

    def step(self, period, score):
        if period < self.min_period or score < self.ignore_min_score:
            self.last_period = period
            return False

        if score - self.best_score > self.thres:
            self.acc_period = 0
        elif abs(self.best_score - score) <= self.thres:
            self.acc_period += period - self.last_period

        if score > self.best_score:
            self.best_period = period
            self.best_score = score

        self.last_period = period
        stop = self.acc_period >= self.patience
        if stop and self.verbose:
            self.stdout_method(f'Early Stopping training. Best results observed at period {self.best_period}, and best score is {self.best_score}')
        return stop

    def __call__(self, period, score):
        return self.step(period, score)

    def state_dict(self):
        return dict(
            last_epoch=self.last_period,
            acc_epoch=self.acc_period,
            best_epoch=self.best_period,
            best_score=self.best_score
        )

    def load_state_dict(self, items: dict):
        self.__dict__.update(items)


class EMA:
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, cur_step=0, step_start_ema=0, decay=0.9999, tau=2000):
        """
        as functions:
            - beta = decay * (1 - math.exp(-step / tau))
            - w = beta * w_o + (1- beta) * w_n
        there are some rows following:
            - step -> 0, beta -> 0, w -> w_n
            - step -> +inf, beta -> 1, w -> w_o
            - step -> 3*tau, beta -> exp(-3)*decay = 0.95*decay
              it gives that, w is w_o almost after 3*tau steps
            - tau -> +inf, beta -> 0
            - tau -> 0, beta -> decay
              it gives that, the ema model is update unavailable forever when tau=0 or tau=inf
              if tau=0, the ema model is the initializing model
              if tau=inf, the ema model is the training model
        """
        self.model = model
        self.ema_model = self.copy(model)
        self.cur_step = cur_step
        self.step_start_ema = step_start_ema
        self.decay_fn = lambda x: decay * (1 - math.exp(-x / tau))

    @staticmethod
    def copy(model):
        ema_model = copy.deepcopy(model)
        ema_model.requires_grad_(False)
        return ema_model

    def restore(self):
        for c_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            param.data.copy_(c_param.data)

    @torch.no_grad()
    def update_attr(self):
        # larger the beta, more closed the weights to the old model
        beta = self.decay_fn(self.cur_step)

        msd = ModuleManager.de_parallel(self.model).state_dict()  # model state_dict
        for k, v in self.ema_model.state_dict().items():
            if v.dtype.is_floating_point:
                v *= beta
                v += (1 - beta) * msd[k].detach()

    @torch.no_grad()
    def copy_attr(self, include=(), exclude=()):
        for k, v in self.model.__dict__.items():
            if (len(include) and k not in include) or k.startswith('_') or k in exclude:
                continue
            else:
                setattr(self.ema_model, k, v)
        self.ema_model.load_state_dict(self.model.state_dict())

    def step(self):
        if self.cur_step < self.step_start_ema:
            self.copy_attr()
        else:
            self.update_attr()
        self.cur_step += 1
        return self.cur_step

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.ema_model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.ema_model.load_state_dict(*args, **kwargs)


class Converter:
    @staticmethod
    def arrays_to_tensors(objs: dict, device=None):
        for k, v in objs.items():
            if isinstance(v, np.ndarray):
                objs[k] = torch.from_numpy(v).to(device)
            elif isinstance(v, (list, tuple)):
                objs[k] = torch.tensor(v, device=device)

        return objs

    @staticmethod
    def tensors_to_array(objs: dict):
        for k, v in objs.items():
            if isinstance(k, torch.Tensor):
                objs[k] = v.cpu().numpy()

        return objs

    @staticmethod
    def convert_keys(state_dict: OrderedDict, convert_dict: dict):
        """
        Usages:
            .. code-block:: python

                model1 = ...
                model2 = ...
                convert_dict = {'before': 'after', 'before.a': 'after.aa', 'before.a.{0}.a': 'after.b.{0}.b}

                state_dict = model1.state_dict()
                # OrderedDict([('before.a.weight', ...), ('before.b.weight', ...), ('before.a.1.a.weight', ...), ('same.weight', ...)])

                state_dict = Converter.convert_keys(state_dict, convert_dict)
                # OrderedDict([('after.aa.weight', ...), ('after.b.weight', ...), ('before.b.1.b.weight', ...), ('same.weight', ...)])

                model2.load_state_dict(state_dict)
        """
        from .nlp_utils import PrefixTree

        def parse(s):
            """split the string with wildcards, and retrun their indexes
            >>> parse('a.{0}.a')
            (('a', '.', '{0}', '.', 'a'), {'idx1': [0], 'idx2': [2]})
            # idx1 is wildcard's values, idx2 is index of wildcard in string
            """
            match = re.findall('\{\d+?\}', s)
            end, tmp, spans, idx1 = 0, s, [], []
            for m in match:
                start = tmp.index(m) + end
                end = start + len(m)
                spans.append((start, end))
                idx1.append(int(m[1:-1]))
                tmp = s[end:]

            r = []
            end = 0
            idx2 = []
            for i, span in enumerate(spans):
                start, end1 = span
                tmp = list(s[end:start])
                r += tmp + [match[i]]
                idx2.append(len(r) - 1)
                end = end1

            r += list(s[end: len(s)])
            return tuple(r), {'idx1': idx1, 'idx2': idx2}

        split_convert_dict = {}
        a_values, b_values = {}, {}
        for a, b in convert_dict.items():
            a_key, a_value = parse(a)
            b_key, b_value = parse(b)
            split_convert_dict[a_key] = b_key
            a_values[a_key] = a_value
            b_values[b_key] = b_value

        tree = PrefixTree(list(split_convert_dict.keys()), list(split_convert_dict.keys()), unique_value=True)
        d = OrderedDict()

        for k, v in state_dict.items():
            # find string before convert
            a = tree.get(k, return_last=True)
            if a in split_convert_dict:
                # make string after convert
                b = split_convert_dict[a]
                a_value = a_values[a]
                b_value = b_values[b]

                p, pa = '', ''
                for i, s in enumerate(a):
                    if i in a_value['idx2']:
                        p += '(.+?)'
                        pa += '%s'
                    else:
                        p += '\\' + s if s == '.' else s
                        pa += s

                if a_value['idx2']:
                    # fill the wildcards with the values for before string
                    ra = re.findall(p, k)[0]
                    if isinstance(ra, str):
                        ra = (ra,)
                    pa = pa % ra

                    sort_ra = [None] * (max(a_value['idx1']) + 1)
                    for i, idx in enumerate(a_value['idx1']):
                        sort_ra[idx] = ra[i]

                    # fill the wildcards with the values for after string
                    for idx1, idx2 in zip(b_value['idx1'], b_value['idx2']):
                        b = list(b)
                        b[idx2] = sort_ra[idx1]

                # replace before string to after string
                pb = ''.join(b)
                k = k.replace(pa, pb)
            d[k] = v

        return d

    @staticmethod
    def conv_weight_from_tf_to_torch(weight):
        """(h, w, c, n) -> (n, c, h, w)"""
        return weight.transpose(3, 2, 0, 1)

    @staticmethod
    def dw_conv_weight_from_tf_to_torch(weight):
        """(h, w, n, c) -> (n, c, h, w)"""
        return weight.transpose(2, 3, 0, 1)

    @staticmethod
    def linear_weight_from_tf_to_torch(weight):
        """(n, h, w) -> (w, h)"""
        if len(weight.size()) == 3:
            weight = weight.squeeze(0)
        if len(weight.size()) == 2:
            weight = weight.transpose(1, 0)
        return weight

    @classmethod
    def make_convert_tf_funcs(cls):
        return {
            'c': cls.conv_weight_from_tf_to_torch,
            'gc': cls.dw_conv_weight_from_tf_to_torch,
            'l': cls.linear_weight_from_tf_to_torch,
        }

    convert_tf_types = {
        'w': 'weight',
        'b': 'bias',
        'g': 'gamma',
        'bt': 'beta',
    }

    @classmethod
    def tensors_from_tf_to_torch(cls, state_dict: OrderedDict, key_types=None, value_types=None):
        """

        Args:
            state_dict:
            key_types (list): see `convert_tf_types`
            value_types (list): see `convert_tf_funcs`

        """

        key_types = key_types or [''] * len(state_dict)
        value_types = value_types or [''] * len(state_dict)
        d = OrderedDict()
        convert_tf_funcs = cls.make_convert_tf_funcs()

        for i, (k, v) in enumerate(state_dict.items()):
            tmp = k.split('/')
            suffix = tmp[-1]
            suffix = cls.convert_tf_types.get(key_types[i], suffix)
            k = '.'.join(tmp[:-1]) + '.' + suffix

            if key_types[i] == 'w' and value_types[i] in convert_tf_funcs:
                v = convert_tf_funcs[value_types[i]](v)
            d[k] = v

        return d
