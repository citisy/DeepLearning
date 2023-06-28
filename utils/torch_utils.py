import copy
import torch
from torch import nn


class ModuleInfo:
    @classmethod
    def profile_per_layer(cls, module: nn.Module, depth=1):
        profiles = []

        def cur(current_m, dep):
            for name, m in current_m._modules.items():
                if dep == 1 or len(m._modules) == 0:
                    profiles.append((name, str(type(m))[8:-2], cls.profile(m)))
                else:
                    cur(m, dep - 1)

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
    def __init__(self, patience=30, verbose=True, stdout_method=print):
        self.best_fitness = 0.0
        self.best_epoch = 0
        self.acc_epoch = 0
        self.last_epoch = 0
        self.patience = patience or float('inf')
        self.verbose = verbose
        self.stdout_method = stdout_method

    def __call__(self, epoch, fitness, thres=10e-3):
        if fitness >= self.best_fitness:
            self.best_epoch = epoch
            self.best_fitness = fitness
            self.acc_epoch = 0

        elif self.best_fitness - fitness < thres:
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


def initialize_layers(module):
    for m in module.modules():
        t = type(m)

        if t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03

        if t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True


class Export:
    @staticmethod
    def to_jit(model, trace_input):
        with torch.no_grad():
            model.eval()
            # warmup, make sure that the model is initialized right
            model(trace_input)
            jit_model = torch.jit.trace(model, trace_input)

        return jit_model
