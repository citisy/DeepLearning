import logging
import os
import time
import numpy as np
import math
import torch
from torch import nn, optim
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Callable, Dict, Tuple, Union
from tqdm import tqdm
from functools import partial
from . import bundled, data_process
from utils import os_lib, configs, visualize, log_utils, torch_utils

MODEL = 'model'
WEIGHT = 'weight'
SAFETENSORS = 'safetensors'
ONNX = 'onnx'
JIT = 'jit'
TRITON = 'triton'

STEP = 'step'
EPOCH = 'epoch'


class CheckpointHooks:
    def __init__(self):
        super().__init__()
        self.state_dict_cache = {}

        self.save_funcs = {
            MODEL: self.save_model,
            WEIGHT: self.save_weight,
            JIT: self.save_torchscript,
            TRITON: self.save_triton,
            SAFETENSORS: self.save_safetensors
        }

        self.load_funcs = {
            MODEL: self.load_model,
            WEIGHT: self.load_weight,
            SAFETENSORS: self.load_safetensors,
            JIT: self.load_jit
        }

    log: 'bundled.LogHooks.log'

    def save(self, save_path, save_type=WEIGHT, verbose=True, **kwargs):
        os_lib.mk_parent_dir(save_path)
        func = self.save_funcs.get(save_type)
        assert func, ValueError(f'dont support {save_type = }')

        func(save_path, **kwargs)
        if verbose:
            self.log(f'Successfully saved to {save_path} !')

    def save_model(self, save_path, additional_items: dict = {}, **kwargs):
        """

        Args:
            save_path:
            additional_items: {path: item}
                path, where the item saved
                item, which obj wanted to save

        """
        torch.save(self.model, save_path, **kwargs)
        for path, item in additional_items.items():
            torch.save(item, path, **kwargs)

    def save_weight(self, save_path, additional_items: dict = {}, additional_path=None, raw_tensors=True, **kwargs):
        """

        Args:
            save_path:
            additional_items: {name: item}
                name, which key of the item;
                item, which obj wanted to save
            additional_path:
                if provided, additional_items are saved with the path
            raw_tensors:
        """
        if additional_path:
            state_dict = self.model.state_dict()
            if not raw_tensors:
                state_dict = {self.model_name: state_dict}
            torch.save(state_dict, save_path, **kwargs)
            torch.save(additional_items, additional_path, **kwargs)

        else:
            if raw_tensors:
                ckpt = self.model.state_dict()

            else:
                ckpt = {
                    self.model_name: self.model.state_dict(),
                    **additional_items
                }

            torch.save(ckpt, save_path, **kwargs)

    def save_safetensors(self, save_path):
        torch_utils.Export.to_safetensors(self.model.state_dict(), save_path)

    device: Union[str, int, torch.device] = None

    def save_torchscript(self, save_path, trace_input=None, model_warp=None, **kwargs):
        assert trace_input is not None

        model = self.model
        if model_warp is not None:
            model = model_warp(model)
        model.to(self.device)
        model = torch_utils.Export.to_torchscript(model, *trace_input, **kwargs)
        model.save(save_path)

    def save_onnx(self, save_path, trace_input=None, model_warp=None, **kwargs):
        assert trace_input is not None

        model = self.model
        if model_warp is not None:
            model = model_warp(model)
        model.to(self.device)
        torch_utils.Export.to_onnx(model, save_path, *trace_input, **kwargs)

    def save_triton(self, save_path, **kwargs):
        raise NotImplementedError

    def load(self, save_path, save_type=WEIGHT, verbose=True, **kwargs):
        func = self.load_funcs.get(save_type)
        assert func, ValueError(f'dont support {save_type = }')

        func(save_path, **kwargs)
        if verbose:
            self.log(f'Successfully load {save_path} !')

    model: nn.Module
    model_name: str

    def load_model(self, save_path, additional_items: dict = {}, **kwargs):
        """

        Args:
            save_path:
            additional_items: {path: name}
                path, where the item saved
                name, which key of the item

        """
        self.model = torch.load(save_path, map_location=self.device, **kwargs)
        for path, name in additional_items.items():
            self.__dict__.update({name: torch.load(path, map_location=self.device)})

    def load_weight(self, save_path, include=None, exclude=None, cache=False, additional_path=None, raw_tensors=True, **kwargs):
        """

        Args:
            save_path:
            include: None or [name]
                name, which key of the item
                if None, load all the items
            exclude: None or [name]
            cache: cache the obj which not load for var, else create a new var to load
            additional_path:
            raw_tensors:
        Returns:

        """
        if additional_path:
            state_dict = torch_utils.Load.from_ckpt(save_path, map_location=self.device, **kwargs)
            if raw_tensors:
                state_dict = {self.model_name: state_dict}
            state_dict = {
                **state_dict,
                **torch_utils.Load.from_ckpt(additional_path, map_location=self.device, **kwargs)
            }

        else:
            state_dict = torch_utils.Load.from_ckpt(save_path, map_location=self.device, **kwargs)
            if raw_tensors:
                state_dict = {self.model_name: state_dict}

        self.load_state_dict(state_dict, include, exclude, cache)

    def load_safetensors(self, save_path, **kwargs):
        state_dict = torch_utils.Load.from_safetensors(save_path, **kwargs)
        self.model.load_state_dict(state_dict, strict=False)

    def load_state_dict(self, state_dict: dict, include=None, exclude=None, cache=False, **kwargs):
        if self.model_name in state_dict:
            self.model.load_state_dict(state_dict.pop(self.model_name), strict=False)
        self._load_state_dict(state_dict, include, exclude, cache)

    def _load_state_dict(self, state_dict: dict, include=None, exclude=None, cache=False):
        for name, item in state_dict.items():
            if (include is None or name in include) and (exclude is None or name not in exclude):
                if name in self.__dict__ and hasattr(self.__dict__[name], 'load_state_dict'):
                    self.__dict__[name].load_state_dict(item)
                elif cache:
                    self.state_dict_cache[name] = item
                else:
                    self.log(
                        f'Found that `{name}` in weight file but not in processor, '
                        f'creating the new var by the name `{name}`, and making sure it is not harmful, '
                        f'or using `exclude` kwargs instead',
                        level=logging.WARNING
                    )
                    self.__dict__[name] = item

    def load_jit(self, save_path, **kwargs):
        self.model = torch.jit.load(save_path, map_location=self.device, **kwargs)

    pretrain_model: str

    def load_pretrain(self):
        if hasattr(self, 'pretrain_model'):
            self.load(self.pretrain_model, save_type=WEIGHT, include=())

    pretrain_checkpoint: str

    def load_pretrain_checkpoint(self):
        if hasattr(self, 'pretrain_checkpoint'):
            self.load(self.pretrain_checkpoint, save_type=WEIGHT, raw_tensors=False)


class ModelHooks:
    trace: 'bundled.LogHooks.trace'
    log: 'bundled.LogHooks.log'
    log_trace: 'bundled.LogHooks.log_trace'
    device: Optional[str]

    def __init__(self):
        super().__init__()

        # all models will be run while on predict
        self.models: Dict[str, nn.Module] = dict()

        # todo: perhaps, for convenient management, move all the modules like optimizer, ema, stopper, ect. to aux_modules?
        # self.aux_modules = {}

    model: nn.Module

    def set_model(self):
        raise NotImplementedError

    use_ema = False
    ema: Optional

    def set_ema(self):
        if self.use_ema:
            self.ema = torch_utils.EMA(self.model)
            # self.aux_modules['ema'] = self.ema
            self.models['ema'] = self.ema.ema_model

    def set_mode(self, train=True):
        for v in self.models.values():
            v.train(train)

    optimizer: Optional

    def set_optimizer(self, lr=1e-3, betas=(0.9, 0.999), **kwargs):
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=betas)

    use_early_stop = True
    stopper: Optional

    def set_stopper(self, **kwargs):
        if self.use_early_stop:
            from utils.torch_utils import EarlyStopping

            if self.check_strategy == EPOCH:
                self.stopper = EarlyStopping(patience=10, min_period=10, ignore_min_score=0.1, stdout_method=self.log)
            elif self.check_strategy == STEP:
                self.stopper = EarlyStopping(patience=10 * 5000, min_period=10 * 5000, ignore_min_score=0.1, stdout_method=self.log)

    scheduler: Optional
    use_scheduler = False
    scheduler_strategy = EPOCH

    def set_scheduler(self, max_epoch, lr_lambda=None, lrf=0.01, **kwargs):
        if self.use_scheduler:
            if self.scheduler_strategy == EPOCH:
                if not lr_lambda:
                    # lr_lambda = lambda x: (1 - x / max_epoch) * (1.0 - lrf) + lrf
                    lr_lambda = lambda x: ((1 - math.cos(x * math.pi / max_epoch)) / 2) * (lrf - 1) + 1  # cos_lr
                self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
                self.scheduler.last_epoch = -1

            elif self.scheduler_strategy == STEP:
                from transformers import get_scheduler

                self.scheduler = get_scheduler(
                    "linear",
                    optimizer=self.optimizer,
                    num_warmup_steps=0,
                    num_training_steps=max_epoch * len(self.train_container['train_dataloader']),
                )

    use_scaler = False
    scaler: Optional

    def set_scaler(self, **kwargs):
        if self.use_scaler:
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)

    def model_info(self, **kwargs):
        from utils.torch_utils import ModuleInfo

        s, infos = ModuleInfo.std_profile(self.model, **kwargs)
        self.log(s)
        return infos

    def model_visual(self):
        """https://github.com/spfrommer/torchexplorer
        sudo apt-get install libgraphviz-dev graphviz
        pip install torchexplorer
        todo: there are some bugs, for example, do not support dict type output. Find another better visual tool.
        """
        import torchexplorer
        torchexplorer.watch(self.model, log=['io', 'params'], disable_inplace=True, backend='standalone')

    train_container: dict

    def fit(self, **kwargs):
        """
        the fit procedure will be run the following pipelines:
            on_train_start()
            Loop epochs:
                on_train_epoch_start()
                Loop batches:
                    on_train_step_start()
                    on_train_step()
                    on_backward()
                    on_train_step_end()
                on_train_epoch_end()
            on_train_end()


        kwargs:
            as the input parameters, transmitting to all the above-mentioned pipelines,
            e.g.
                there is a pipeline defined like that:
                    def on_train_epoch_start(..., batch_size=16, ...):
                        ...
                and then, you can set the special value of `batch_size` like that:
                    fit(..., batch_size=32, ...)
            please make sure what parameters of pipeline is needed before transmission
            suggest to include mainly the following parameters:
                max_epoch:
                batch_size: for fit() and predict()
                check_period:
                metric_kwargs: for metric() and predict()
                return_val_dataloader:
                dataloader_kwargs: for fit() and predict()

        train_container:
            be loaded what parameters generated or changed in all the train pipelines
            suggest to include mainly the following parameters:
                train_dataloader:
                val_dataloader:

        """
        self.on_train_start(**kwargs)
        self.on_train(**kwargs)
        self.on_train_end(**kwargs)

    init_wandb: 'bundled.LogHooks.init_wandb'
    work_dir: str
    model_name: str
    get_train_dataloader: 'data_process.DataHooks.get_train_dataloader'
    get_val_dataloader: 'data_process.DataHooks.get_val_dataloader'
    save: 'CheckpointHooks.save'
    load: 'CheckpointHooks.load'
    load_pretrain_checkpoint: 'CheckpointHooks.load_pretrain_checkpoint'
    counters: dict
    train_start_container: dict = {}
    train_end_container: dict = {}

    def register_train_start(self, func, **kwargs):
        self.train_start_container.update({func: kwargs})

    def register_train_end(self, func, **kwargs):
        self.train_end_container.update({func: kwargs})

    def on_train_start(
            self, batch_size=None, max_epoch=None,
            train_dataloader=None, val_dataloader=None, check_period=None,
            metric_kwargs=dict(), data_get_kwargs=dict(), dataloader_kwargs=dict(), **kwargs):
        assert batch_size, 'please set batch_size'
        assert max_epoch, 'please set max_epoch'
        self.log(f'{batch_size = }')

        self.train_container = dict()
        metric_kwargs = metric_kwargs.copy()
        metric_kwargs.setdefault('batch_size', batch_size)
        metric_kwargs.setdefault('dataloader_kwargs', {})
        metric_kwargs['dataloader_kwargs'] = configs.merge_dict(dataloader_kwargs, metric_kwargs['dataloader_kwargs'])

        _counters = ['epoch', 'total_nums', 'total_steps', 'check_nums']
        for c in _counters:
            self.counters.setdefault(c, 0)

        dataloader_kwargs.setdefault('batch_size', batch_size)
        if train_dataloader is None:
            train_dataloader = self.get_train_dataloader(data_get_kwargs=data_get_kwargs, dataloader_kwargs=dataloader_kwargs)
        self.train_container['train_dataloader'] = train_dataloader

        if check_period:
            if val_dataloader is None:
                val_dataloader = self.get_val_dataloader(data_get_kwargs=data_get_kwargs, dataloader_kwargs=dataloader_kwargs)
            metric_kwargs.setdefault('val_dataloader', val_dataloader)
            s = 'epochs' if self.check_strategy == EPOCH else 'nums'
            self.log(f'check_strategy = `{self.check_strategy}`, it will be check the training result in every {check_period} {s}')

        self.train_container['metric_kwargs'] = metric_kwargs
        self.train_container['end_flag'] = False
        self.train_container['last_check_time'] = time.time()

        for item in ('optimizer', 'stopper', 'scaler', 'scheduler'):
            if not hasattr(self, item) or getattr(self, item) is None:
                getattr(self, f'set_{item}')(max_epoch=max_epoch, **kwargs)

        for func, params in self.train_start_container.items():
            func(**params)

        self.load_pretrain_checkpoint()
        self.set_mode(train=True)

    register_logger: 'bundled.LogHooks.register_logger'
    log_methods: dict
    check_strategy: str = EPOCH

    def on_train(self, max_epoch=100, **kwargs):
        for i in range(self.counters['epoch'], max_epoch):
            self.on_train_epoch_start(**kwargs)
            pbar = tqdm(self.train_container['train_dataloader'], desc=visualize.TextVisualize.highlight_str(f'Train {i}/{max_epoch}'))
            self.register_logger('pbar', pbar.set_postfix)

            for rets in pbar:
                self.on_train_step_start(rets, **kwargs)
                outputs = self.on_train_step(rets, **kwargs)
                self.on_backward(outputs, **kwargs)
                if self.on_train_step_end(rets, outputs, **kwargs):
                    break

            if self.on_train_epoch_end(**kwargs):
                break

    def on_train_epoch_start(self, _counters=('per_epoch_nums',), **kwargs):
        for c in _counters:
            self.counters[c] = 0

    def on_train_step_start(self, rets, **kwargs):
        pass

    def on_train_step(self, rets, **kwargs) -> dict:
        """logic of model training step, and expected to return a dict of model output
        must return a dict included:
            loss: loss to backward
        """
        raise NotImplementedError

    def on_backward(self, outputs, accumulate=None, batch_size=None, **kwargs):
        loss = outputs['loss']

        if self.use_scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if accumulate:
            if self.counters['total_nums'] % accumulate < batch_size:
                self._backward()
        else:
            self._backward()

        if self.use_ema:
            self.ema.step()

    def _backward(self):
        if self.use_scaler:
            self.scaler.unscale_(self.optimizer)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
            self.scaler.step(self.optimizer)  # optimizer.step
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad()

    def on_train_step_end(self, rets, outputs, more_log=False, ignore_non_loss=False, **kwargs) -> bool:
        self.counters['total_nums'] += len(rets)
        self.counters['total_steps'] += 1
        self.counters['per_epoch_nums'] += len(rets)
        self.counters['check_nums'] += len(rets)

        losses = {}
        for k, v in outputs.items():
            if k.startswith('loss'):
                v = v.item()
                n = f'check_{k}'
                if ignore_non_loss and np.isnan(v):
                    pass
                else:
                    self.counters[n] = self.counters.get(n, 0) + v
                    losses[k] = v
                    losses[f'mean_{k}'] = self.counters[n] / self.counters['check_nums']

        self.train_container['losses'] = losses

        mem_info = {
            'cpu_info': log_utils.MemoryInfo.get_process_mem_info(),
            'gpu_info': log_utils.MemoryInfo.get_gpu_mem_info()
        } if more_log else {}

        self.log({
            **losses,
            **mem_info
        }, 'pbar')

        if self.use_scheduler and self.scheduler_strategy == STEP:
            self.scheduler.step()

        if self.check_strategy == STEP:
            self._check_on_train_step_end(**kwargs)

        return self.train_container.get('end_flag', False)  # cancel the training when end_flag is True

    def on_train_epoch_end(self, **kwargs) -> bool:
        self.counters['epoch'] += 1
        if self.use_scheduler and self.scheduler_strategy == EPOCH:
            self.scheduler.step()
        if self.check_strategy == EPOCH:
            self._check_on_train_epoch_end(**kwargs)
        return self.train_container.get('end_flag', False)  # cancel the training when end_flag is True

    def _check_on_train_step_end(self, check_period=None, batch_size=None, max_save_weight_num=None, **kwargs):
        total_nums = self.counters['total_nums']
        if check_period and total_nums % check_period < batch_size:
            self.trace({'total_nums': total_nums}, (bundled.LOGGING, bundled.WANDB))

            ckpt = self._check_train(max_save_weight_num, total_nums)
            self.log_trace(bundled.LOGGING)

            self._check_metric(ckpt, total_nums, max_save_weight_num)
            self.log_trace(bundled.WANDB)

    def _check_on_train_epoch_end(self, check_period=None, max_save_weight_num=None, **kwargs):
        epoch = self.counters['epoch'] - 1  # epoch in counters is the next epoch, not the last
        self.trace({'epoch': epoch}, (bundled.LOGGING, bundled.WANDB))

        state_dict = self._check_train(max_save_weight_num, epoch)
        self.log_trace(bundled.LOGGING)

        if check_period and epoch % check_period == check_period - 1:
            self._check_metric(state_dict, epoch, max_save_weight_num)
        self.log_trace(bundled.WANDB)

    def _check_train(self, max_save_weight_num, check_num):
        losses = self.train_container.get('losses')
        if losses is not None:
            for k, v in losses.items():
                self.trace({f'loss/{k}': v}, (bundled.LOGGING, bundled.WANDB))
                if np.isnan(v) or np.isinf(v):
                    self.train_container['end_flag'] = True
                    self.log(f'Train will be stop soon, got {v} value from {k}')

            for k in self.counters:
                if k.startswith('check_'):
                    self.counters[k] = 0

        last_check_time = self.train_container.get('last_check_time')
        if last_check_time is not None:
            now = time.time()
            self.trace({'time_consume': (now - last_check_time) / 60}, (bundled.LOGGING, bundled.WANDB))
            self.train_container['last_check_time'] = now

        state_dict = self.state_dict()

        if not isinstance(max_save_weight_num, int):  # None
            self.save(
                f'{self.work_dir}/last.pth',
                additional_path=f'{self.work_dir}/last.additional.pth',
                additional_items=state_dict,
                save_type=WEIGHT,
            )

        elif max_save_weight_num > 0:
            self.save(
                f'{self.work_dir}/{check_num}.pth',
                additional_path=f'{self.work_dir}/{check_num}.additional.pth',
                additional_items=state_dict,
                save_type=WEIGHT,
            )
            os_lib.FileCacher(self.work_dir, max_size=max_save_weight_num, stdout_method=self.log).delete_over_range(suffix='pth')

        return state_dict

    def _check_metric(self, state_dict, check_num, max_save_weight_num):
        results = self.metric(**self.train_container.get('metric_kwargs', {}))
        scores = {}
        for name, result in results.items():
            for k, v in result.items():
                if k.startswith('score'):
                    scores[f'val_score/{name}.{k}'] = v

        self.trace(scores, bundled.WANDB)
        self.log(f'val log: score: {scores}')
        self.set_mode(train=True)

        if hasattr(self, 'stopper'):
            score = results[self.model_name]['score']
            if score > self.stopper.best_score:
                if not isinstance(max_save_weight_num, int) or max_save_weight_num > 0:
                    self.save(
                        f'{self.work_dir}/best.pth',
                        additional_path=f'{self.work_dir}/best.additional.pth',
                        additional_items=state_dict,
                        save_type=WEIGHT,
                    )

            self.train_container['end_flag'] = self.train_container['end_flag'] or self.stopper(check_num, score)

    def state_dict(self):
        state_dict = {
            'date': datetime.now().isoformat()
        }

        for name in ('optimizer', 'stopper', 'ema', 'counters', 'wandb_id'):
            if hasattr(self, name):
                var = getattr(self, name)
                if hasattr(var, 'state_dict'):
                    state_dict[name] = var.state_dict()
                elif var is not None:
                    state_dict[name] = var

        return state_dict

    def on_train_end(self, **kwargs):
        for func, params in self.train_end_container.items():
            func(**params)

        for item in ('optimizer', 'stopper', 'scaler'):
            if hasattr(self, item):
                delattr(self, item)

    def metric(self, *args, **kwargs) -> dict:
        """call the `predict()` function to get model output, then count the score, expected to return a dict of model score"""
        raise NotImplementedError

    val_container: dict

    @torch.no_grad()
    def predict(self, **kwargs) -> dict:
        """
        do not distinguish val and test strategy
        the prediction procedure will be run the following pipelines:
            on_val_start()
            Loop batches:
                on_val_step_start()
                on_val_step()
                on_val_reprocess()
                on_val_step_end()
            on_val_end()

        kwargs
            be the input parameters, for all the above-mentioned pipelines,
            please make sure what parameters of pipeline is needed before transmission
            suggest to include mainly the following parameters:
                val_dataloader:
                batch_size:
                is_visualize:
                max_vis_num:
                save_ret_func:
                dataloader_kwargs:

        val_container:
            be loaded what parameters generated by all the prediction pipelines
            suggest to include mainly the following parameters:

        """
        self.on_val_start(**kwargs)

        for rets in tqdm(self.val_container['val_dataloader'], desc=visualize.TextVisualize.highlight_str('Val')):
            self.on_val_step_start(rets, **kwargs)
            model_results = self.on_val_step(rets, **kwargs)
            self.on_val_reprocess(rets, model_results, **kwargs)
            self.on_val_step_end(rets, model_results, **kwargs)

        self.on_val_end(**kwargs)
        return self.val_container

    val_start_container: dict = {}
    val_end_container: dict = {}

    def register_val_start(self, func, **kwargs):
        self.val_start_container.update({func: kwargs})

    def register_end_start(self, func, **kwargs):
        self.val_end_container.update({func: kwargs})

    def on_val_start(self, val_dataloader=None, batch_size=None, data_get_kwargs=dict(), dataloader_kwargs=dict(), **kwargs):
        assert batch_size, 'please set batch_size'
        self.val_container = {}
        dataloader_kwargs.setdefault('batch_size', batch_size)
        if val_dataloader is None:
            val_dataloader = self.get_val_dataloader(data_get_kwargs=data_get_kwargs, dataloader_kwargs=dataloader_kwargs)
        self.val_container['val_dataloader'] = val_dataloader

        self.set_mode(train=False)
        self.counters['vis_num'] = 0
        self.counters.setdefault('epoch', -1)
        self.val_container['model_results'] = dict()

        for func, params in self.val_start_container.items():
            func(**params)

    def on_val_step_start(self, rets, **kwargs):
        pass

    def on_val_step(self, rets, **kwargs) -> dict:
        """logic of validating step, expected to return a dict of model output included preds
        must return a dict included:
            outputs: original model output
            preds: normalized model output

        """
        raise NotImplementedError

    def on_val_reprocess(self, rets, model_results, **kwargs):
        """prepare true and pred label for `visualize()` or `metric()`
        reprocess data will be cached in val_container"""

    def on_val_step_end(self, rets, model_results, is_visualize=False, batch_size=16, max_vis_num=None, **kwargs):
        """visualize the model outputs usually"""
        if is_visualize:
            max_vis_num = max_vis_num or float('inf')
            n = min(batch_size, max_vis_num - self.counters['vis_num'])
            if n > 0:
                self.visualize(rets, model_results, n, **kwargs)
                self.counters['vis_num'] += n

    def visualize(self, rets, model_results, n, **kwargs):
        """logic of predict results visualizing"""
        pass

    def on_val_end(self, **kwargs):
        """save the results usually"""
        for func, params in self.val_end_container.items():
            func(**params)

    model_input_template: 'namedtuple'
    predict_container: dict

    @torch.no_grad()
    def single_predict(self, *obj, **kwargs):
        return self.batch_predict(*[[o] for o in obj], **kwargs)[0]

    @torch.no_grad()
    def batch_predict(self, *objs, batch_size=16, **kwargs):
        self.on_predict_start(**kwargs)

        for i in tqdm(range(0, len(objs[0]), batch_size), desc=visualize.TextVisualize.highlight_str('Predict')):
            rets = self.gen_predict_inputs(*objs, start_idx=i, end_idx=i + batch_size, **kwargs)
            rets = self.on_predict_step_start(rets, **kwargs)
            model_results = self.on_predict_step(rets, **kwargs)
            self.on_predict_reprocess(rets, model_results, **kwargs)
            self.on_predict_step_end(rets, model_results, **kwargs)

        return self.on_predict_end(**kwargs)

    @torch.no_grad()
    def fragment_predict(self, image: np.ndarray, *args, **kwargs):
        """Tear large picture to pieces for prediction, and then, merge the results and restore them"""
        raise NotImplementedError

    def on_predict_start(self, **kwargs):
        self.predict_container = dict()
        self.set_mode(train=False)
        self.counters['vis_num'] = 0
        self.counters["total_nums"] = -1
        self.predict_container['model_results'] = dict()

    def gen_predict_inputs(self, *objs, start_idx=None, end_idx=None, **kwargs) -> List[dict]:
        raise NotImplementedError

    def on_predict_step_start(self, rets, **kwargs):
        """preprocess the model inputs"""
        if hasattr(self, 'predict_data_augment'):
            rets = [self.predict_data_augment(ret) for ret in rets]
        return rets

    def on_predict_step(self, rets, **kwargs):
        return self.on_val_step(rets, **kwargs)

    def on_predict_reprocess(self, rets, model_results, **kwargs):
        """prepare true and pred label for `visualize()`
        reprocess data will be cached in predict_container"""
        self.predict_container['model_results'].setdefault(self.model_name, []).extend(model_results[self.model_name]['preds'])

    def on_predict_step_end(self, rets, model_results, **kwargs):
        """visualize the model outputs usually"""

    def on_predict_end(self, **kwargs) -> List:
        """visualize results and the return the results"""
        return self.predict_container['model_results'][self.model_name]
