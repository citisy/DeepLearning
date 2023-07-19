import copy
import math
import torch
from torch import nn


class ModuleInfo:
    @classmethod
    def profile_per_layer(cls, module: nn.Module, depth=None):
        profiles = []

        def cur(current_m, dep):
            for name, m in current_m._modules.items():
                if dep == 1 or len(m._modules) == 0:
                    profiles.append((name, str(type(m))[8:-2], cls.profile(m)))
                else:
                    cur(m, dep - 1)

        depth = depth or float('inf')
        cur(module, depth)
        return profiles

    @classmethod
    def profile(cls, module):
        return dict(
            params=cls.profile_params(module),
            grads=cls.profile_grads(module),
        )

    @staticmethod
    def profile_params(module):
        return sum(x.numel() for x in module.parameters())

    @staticmethod
    def profile_grads(module):
        return sum(x.numel() for x in module.parameters() if x.requires_grad)

    @staticmethod
    def profile_flops(module, input_size, *test_args):
        import thop

        p = next(module.parameters())
        test_data = torch.empty((1, *input_size), device=p.device)
        flops = thop.profile(copy.deepcopy(module), inputs=(test_data, *test_args), verbose=False)[0] / 1E9 * 2  # stride GFLOPs

        return flops


class EarlyStopping:
    def __init__(self, thres=0.005, patience=30, verbose=True, stdout_method=print):
        self.thres = thres
        self.best_fitness = 0.0
        self.best_epoch = 0
        self.acc_epoch = 0
        self.last_epoch = 0
        self.patience = patience or float('inf')
        self.verbose = verbose
        self.stdout_method = stdout_method

    def __call__(self, epoch, fitness, ):
        if fitness >= self.best_fitness:
            self.best_epoch = epoch
            self.best_fitness = fitness
            self.acc_epoch = 0

        elif self.best_fitness - fitness < self.thres:
            self.acc_epoch += epoch - self.last_epoch

        self.last_epoch = epoch
        stop = self.acc_epoch >= self.patience
        if stop and self.verbose:
            self.stdout_method(
                f'Stopping training early as no improvement observed in last {self.patience} epochs. '
                f'Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n'
                f'To update EarlyStopping(patience={self.patience}) pass a new patience value'
            )
        return stop


def initialize_layers(module, init_gain=0.02, init_type='normal'):
    for m in module.modules():
        t = type(m)

        if t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
            m.weight.data.normal_(1.0, init_gain)
            m.bias.data.fill_(0.)

        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True

        elif t in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]:
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)


class Export:
    @staticmethod
    def to_pickle(model):
        pass

    @staticmethod
    def to_jit(model, trace_input):
        with torch.no_grad():
            model.eval()
            # warmup, make sure that the model is initialized right
            model(trace_input)
            jit_model = torch.jit.trace(model, trace_input)

        return jit_model


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = copy.deepcopy(de_parallel(model)).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)
