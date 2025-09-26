import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Dict, Union, Annotated, overload

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

from utils import os_lib, configs, visualize, log_utils, torch_utils
from . import bundled, data_process

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

        self.checkpoint_container: dict = {}

    log: bundled.LogHooks.log

    def save(self, save_path, save_type=WEIGHT, verbose=True, **kwargs):
        os_lib.mk_parent_dir(save_path)
        func = self.save_funcs.get(save_type)
        assert func, ValueError(f'dont support {save_type = }')

        func(save_path, **kwargs)

    def save_model(self, save_path, additional_items: dict = {}, verbose=True, **kwargs):
        """

        Args:
            save_path:
            additional_items: {path: item}
                path, where the item saved
                item, which obj wanted to save

        """
        torch.save(self.model, save_path, **kwargs)

        if verbose:
            self.log(f'Successfully saved to {save_path} !')

        for path, item in additional_items.items():
            torch.save(item, path, **kwargs)

            if verbose:
                self.log(f'Successfully saved to {path} !')

    def save_weight(self, save_path, additional_items: dict = {}, additional_path=None, raw_tensors=True, verbose=True, **kwargs):
        """

        Args:
            save_path:
            additional_items: {name: item}
                name, which key of the item;
                item, which obj wanted to save
                if `raw_tensors=True`, additional_items would not be saved
            additional_path:
                if provided, additional_items are saved in additional_path
                if not, additional_items are saved in save_path, together with the model weights
            raw_tensors:
                if True, the fmt of weight file is liked, `{weight_names: weights}`
                if False, the fmt of weight file is liked, `{model_name: {weight_names: weights}}`
        """
        if additional_path:
            state_dict = self.model.state_dict()
            if not raw_tensors:
                state_dict = {self.model_name: state_dict}
            torch.save(state_dict, save_path, **kwargs)
            torch.save(additional_items, additional_path, **kwargs)
            if verbose:
                self.log(f'Successfully saved to {save_path} !')
                self.log(f'Successfully saved to {additional_path} !')

        else:
            if raw_tensors:
                ckpt = self.model.state_dict()

            else:
                ckpt = {
                    self.model_name: self.model.state_dict(),
                    **additional_items
                }

            torch.save(ckpt, save_path, **kwargs)
            if verbose:
                self.log(f'Successfully saved to {save_path} !')

    def save_safetensors(self, save_path, verbose=True, **kwargs):
        torch_utils.Export.to_safetensors(self.model.state_dict(), save_path)
        if verbose:
            self.log(f'Successfully saved to {save_path} !')

    device: Annotated[
        Union[str, int, torch.device],
        "'cpu', or id of gpu device if gpu is available"
    ] = None

    def save_torchscript(self, save_path, trace_input=None, model_wrap=None, verbose=True, **kwargs):
        assert trace_input is not None

        model = self.model
        if model_wrap is not None:
            model = model_wrap(model)
        model.to(self.device)
        model = torch_utils.Export.to_torchscript(model, *trace_input, **kwargs)
        model.save(save_path)
        if verbose:
            self.log(f'Successfully saved to {save_path} !')

    def save_onnx(self, save_path, trace_input=None, model_wrap=None, verbose=True, **kwargs):
        assert trace_input is not None

        model = self.model
        if model_wrap is not None:
            model = model_wrap(model)
        model.to(self.device)
        torch_utils.Export.to_onnx(model, save_path, *trace_input, **kwargs)
        if verbose:
            self.log(f'Successfully saved to {save_path} !')

    def save_triton(self, save_path, **kwargs):
        raise NotImplementedError

    def register_save_checkpoint(self, func, **kwargs):
        self.checkpoint_container.update({func: kwargs})

    work_dir: str

    def save_pretrained_checkpoint(self, suffix, max_save_weight_num, state_dict):
        self.save(
            f'{self.work_dir}/{suffix}.pth',
            additional_path=f'{self.work_dir}/{suffix}.additional.pth',
            additional_items=state_dict,
            save_type=WEIGHT,
        )

        for func, kwargs in self.checkpoint_container.items():
            func(suffix, max_save_weight_num, **kwargs)

        if max_save_weight_num:
            os_lib.FileCacher(self.work_dir, max_size=max_save_weight_num, stdout_method=self.log).delete_over_range(suffix=r'\d+\.pth')
            os_lib.FileCacher(self.work_dir, max_size=max_save_weight_num, stdout_method=self.log).delete_over_range(suffix=r'\d+\.additional\.pth')

    def state_dict(self):
        """get additional info except model weights"""
        state_dict = {
            'date': datetime.now().isoformat()
        }

        for name in ('optimizer', 'stopper', 'counters', 'wandb_id'):
            if hasattr(self, name):
                var = getattr(self, name)
                if hasattr(var, 'state_dict'):
                    state_dict[name] = var.state_dict()
                elif var is not None:
                    state_dict[name] = var

        return state_dict

    def load(self, save_path, save_type=WEIGHT, **kwargs):
        func = self.load_funcs.get(save_type)
        assert func, ValueError(f'dont support {save_type = }')

        return func(save_path, **kwargs)

    model: nn.Module
    model_name: str

    def load_model(self, save_path, additional_items: dict = {}, verbose=True, **kwargs):
        """

        Args:
            save_path:
            additional_items: {path: name}
                path, where the item saved
                name, which key of the item

        """
        self.model = torch.load(save_path, map_location=self.device, **kwargs)
        if verbose:
            self.log(f'Successfully load {save_path} !')
        for path, name in additional_items.items():
            self.__dict__.update({name: torch.load(path, map_location=self.device)})
            if verbose:
                self.log(f'Successfully load {name} from {path} !')

    def load_weight(self, save_path, include=None, exclude=None, cache=False, additional_path=None, raw_tensors=True, verbose=True, **kwargs):
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
            state_dict = torch_utils.Load.from_ckpt(save_path, map_location=self.device, weights_only=False, **kwargs)
            if raw_tensors:
                state_dict = {self.model_name: state_dict}
            state_dict = {
                **state_dict,
                **torch_utils.Load.from_ckpt(additional_path, map_location=self.device, weights_only=False, **kwargs)
            }
            if verbose:
                self.log(f'Successfully load {save_path} !')
                self.log(f'Successfully load {additional_path} !')

        else:
            state_dict = torch_utils.Load.from_ckpt(save_path, map_location=self.device, weights_only=False, **kwargs)
            if raw_tensors:
                state_dict = {self.model_name: state_dict}
            if verbose:
                self.log(f'Successfully load {save_path} !')

        self.load_state_dict(state_dict, include, exclude, cache)
        return state_dict

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

    def load_jit(self, save_path, verbose=True, **kwargs):
        self.model = torch.jit.load(save_path, map_location=self.device, **kwargs)
        if verbose:
            self.log(f'Successfully load {save_path} !')

    pretrain_model: str

    def load_pretrained(self):
        if hasattr(self, 'pretrain_model'):
            self.load(self.pretrain_model, save_type=WEIGHT, include=())

    pretrained_checkpoint: str

    def load_pretrained_checkpoint(self, raw_tensors=True) -> dict:
        if hasattr(self, 'pretrained_checkpoint'):
            pretrained_checkpoint = Path(self.pretrained_checkpoint)
            return self.load(
                self.pretrained_checkpoint,
                additional_path=f'{pretrained_checkpoint.parent}/{pretrained_checkpoint.stem}.additional.pth',
                save_type=WEIGHT,
                raw_tensors=raw_tensors
            )
        else:
            return {}


class ModelHooks:
    trace: bundled.LogHooks.trace
    log: bundled.LogHooks.log
    log_trace: bundled.LogHooks.log_trace
    device: Optional[str]
    register_save_checkpoint: CheckpointHooks.register_save_checkpoint
    state_dict: CheckpointHooks.state_dict
    save_pretrained_checkpoint: CheckpointHooks.save_pretrained_checkpoint

    def __init__(self):
        super().__init__()

        # all models will be run while on predict
        self.models: Dict[str, nn.Module] = dict()

        # todo: perhaps, for convenient management, move all the modules like optimizer, ema, stopper, ect. to aux_modules?
        # self.aux_modules = {}

        self.train_start_container: dict = {}
        self.train_end_container: dict = {}
        self.val_start_container: dict = {}
        self.val_end_container: dict = {}

    model: nn.Module

    def set_model(self):
        raise NotImplementedError

    use_ema = False
    ema: Optional

    def set_ema(self):
        if self.use_ema:
            self.ema = torch_utils.EMA(self.model)
            self.models['ema'] = self.ema.ema_model

            def save(suffix, max_save_weight_num, **kwargs):
                torch.save(self.ema.ema_model.state_dict(), f'{self.work_dir}/{suffix}.ema.pth')
                os_lib.FileCacher(self.work_dir, max_size=max_save_weight_num, stdout_method=self.log).delete_over_range(suffix=r'\d+\.ema\.pth')
                self.log(f'Successfully save ema model to {self.work_dir}/{suffix}.ema.pth')

            self.register_save_checkpoint(save)
            self.log('Successfully init ema model!')

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
    scheduler_strategy: Annotated[
        str,
        'every `epoch` or every `step` to run scheduler'
    ] = EPOCH

    def set_scheduler(self, max_epoch, lr_lambda=None, lrf=0.01, train_dataloader=None, **kwargs):
        if self.use_scheduler:
            if self.scheduler_strategy == EPOCH:
                if not lr_lambda:
                    lr_lambda = torch_utils.SchedulerMaker.cosine_lr_lambda(max_epoch, lrf, **kwargs)
                self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda, last_epoch=-1)

            elif self.scheduler_strategy == STEP:
                from transformers import get_scheduler

                self.scheduler = get_scheduler(
                    "linear",
                    optimizer=self.optimizer,
                    num_warmup_steps=0,
                    num_training_steps=max_epoch * len(train_dataloader),
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

    @overload
    def fit(
            self,
            batch_size: int = -1,
            max_epoch: int = -1,

            train_dataloader: 'torch.utils.data.DataLoader' = None,
            val_dataloader: 'torch.utils.data.DataLoader' = None,

            # for `get_data()`
            data_get_kwargs: dict = dict(),

            # for `torch.utils.data.DataLoader`
            dataloader_kwargs: dict = dict(),

            # num of epoch/step to run training check
            check_period: int = None,

            # for `metric()`
            is_metric: bool = True,
            metric_kwargs: dict = dict(),

            # for model run
            model_kwargs: dict = dict(),

            # which vars will be recorded when training
            _counters: Optional[tuple] = ('per_epoch_nums',),

            # num for accumulate if not None
            accumulate: int = None,

            # if True, log more info, like gpu info
            more_log: bool = False,

            # if True, while occur nan output, training will be stopped
            ignore_non_loss: bool = False,

            # the value is
            # None,
            #     if is_metric=False, only save weight with name of `last.pth`,
            #     else, save weight with name of `[last/best].pth`
            # 0, don't save any weight file
            # >0,
            #     if is_metric=False, only save weight with name of `{check_period}.pth`,
            max_save_weight_num: int = None,
    ):
        pass

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
                    None or 0, do not metric or save weights.
                    >0, epoch period to metric or save weights, `self.check_strategy` also affect it
                is_metric: whether to metric when training end or not
                max_save_weight_num:
                    None,
                        if is_metric=False, only save weight with name of `last.pth`,
                        else, save weight with name of `[last/best].pth`
                    0, don't save any weight file
                    >0,
                        if is_metric=False, only save weight with name of `{check_period}.pth`,
                        else, also save weight with name of 'best.pth' additional
                metric_kwargs: for metric() and predict()
                dataloader_kwargs: for fit() and predict()

        train_container:
            be loaded what parameters generated or changed in all the train pipelines
            suggest to include mainly the following parameters:
                train_dataloader:
                val_dataloader:

        """
        loop_objs, process_kwargs = self.on_train_start(**kwargs)
        kwargs.update(process_kwargs)
        self.on_train(loop_objs, **kwargs)
        return self.on_train_end(**kwargs)

    init_wandb: bundled.LogHooks.init_wandb
    work_dir: str
    model_name: str
    get_train_dataloader: data_process.DataHooks.get_train_dataloader
    get_val_dataloader: data_process.DataHooks.get_val_dataloader
    save: CheckpointHooks.save
    load: CheckpointHooks.load
    load_pretrained_checkpoint: CheckpointHooks.load_pretrained_checkpoint

    def register_train_start(self, func, **kwargs):
        self.train_start_container.update({func: kwargs})

    def register_train_end(self, func, **kwargs):
        self.train_end_container.update({func: kwargs})

    counters_keys = ['epoch', 'total_nums', 'total_steps', 'check_nums']

    def on_train_start(
            self, batch_size=None, max_epoch=None,
            train_dataloader=None, val_dataloader=None, check_period=None,
            metric_kwargs=dict(), data_get_kwargs=dict(), dataloader_kwargs=dict(),
            **kwargs
    ):
        assert self.models, 'model list is empty, it seems that you have not init the processor first, perhaps run `processor.init()` first?'
        assert batch_size, 'please set batch_size'
        assert max_epoch, 'please set max_epoch'
        self.log(f'{batch_size = }')

        metric_kwargs = metric_kwargs.copy()
        metric_kwargs.setdefault('batch_size', batch_size)
        metric_kwargs.setdefault('dataloader_kwargs', {})
        metric_kwargs['dataloader_kwargs'] = configs.ConfigObjParse.merge_dict(dataloader_kwargs, metric_kwargs['dataloader_kwargs'])

        dataloader_kwargs.setdefault('batch_size', batch_size)
        if train_dataloader is None:
            train_dataloader = self.get_train_dataloader(data_get_kwargs=data_get_kwargs, dataloader_kwargs=dataloader_kwargs)

        if check_period:
            if val_dataloader is None:
                val_dataloader = self.get_val_dataloader(data_get_kwargs=data_get_kwargs, dataloader_kwargs=dataloader_kwargs)
            metric_kwargs.setdefault('val_dataloader', val_dataloader)
            s = 'epochs' if self.check_strategy == EPOCH else 'nums'
            self.log(f'check_strategy = `{self.check_strategy}`, it will be check the training result in every {check_period} {s}')

        for item in ('optimizer', 'stopper', 'scaler', 'scheduler'):
            if not hasattr(self, item) or getattr(self, item) is None:
                getattr(self, f'set_{item}')(max_epoch=max_epoch, train_dataloader=train_dataloader, **kwargs)

        for func, params in self.train_start_container.items():
            func(**params)

        state_dict = self.load_pretrained_checkpoint()
        self.set_mode(train=True)

        loop_objs = dict(
            end_flag=False,
            last_check_time=time.time(),
        )

        for c in self.counters_keys:
            loop_objs.setdefault(c, 0)

        loop_objs.update(state_dict.get('loop_objs', {}))

        process_kwargs = dict(
            train_dataloader=train_dataloader,
            metric_kwargs=metric_kwargs,
        )

        return loop_objs, process_kwargs

    register_logger: bundled.LogHooks.register_logger
    log_methods: dict
    check_strategy: Annotated[
        str,
        'every epoch or every step to run training check, like saving checkpoint, run the metric step, etc'
    ] = EPOCH

    def on_train(self, loop_objs, **kwargs):
        max_epoch = kwargs['max_epoch']
        train_dataloader = kwargs['train_dataloader']

        for i in range(loop_objs['epoch'], max_epoch):
            self.on_train_epoch_start(loop_objs, **kwargs)
            pbar = tqdm(train_dataloader, desc=visualize.TextVisualize.highlight_str(f'Train {i}/{max_epoch}'))
            self.register_logger('pbar', pbar.set_postfix)

            for loop_inputs in pbar:
                loop_objs.update(
                    loop_inputs=loop_inputs,
                )

                self.on_train_step_start(loop_objs, **kwargs)
                model_results = self.on_train_step(loop_objs, **kwargs)

                loop_objs.update(
                    model_results=model_results,
                )

                self.on_backward(loop_objs, **kwargs)
                if self.on_train_step_end(loop_objs, **kwargs):
                    break

            if self.on_train_epoch_end(loop_objs, **kwargs):
                break

    def on_train_epoch_start(self, loop_objs, _counters=('per_epoch_nums',), **kwargs):
        for c in _counters:
            loop_objs[c] = 0

    def on_train_step_start(self, loop_objs, **kwargs):
        pass

    def on_train_step(self, loop_objs, model_kwargs=dict(), **kwargs) -> dict:
        """logic of model training step, and expected to return a dict of model output
        must return a dict included:
            loss: loss to backward
        """
        raise NotImplementedError

    def on_backward(self, loop_objs, accumulate=None, batch_size=None, **kwargs):
        model_results = loop_objs['model_results']
        loss = model_results['loss']

        if self.use_scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        if accumulate:
            if loop_objs['total_nums'] % accumulate < batch_size:
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

    def on_train_step_end(self, loop_objs, more_log=False, ignore_non_loss=False, **kwargs) -> bool:
        loop_inputs = loop_objs['loop_inputs']
        model_results = loop_objs['model_results']

        loop_objs['total_nums'] += len(loop_inputs)
        loop_objs['total_steps'] += 1
        loop_objs['per_epoch_nums'] += len(loop_inputs)
        loop_objs['check_nums'] += len(loop_inputs)

        losses = {}
        for k, v in model_results.items():
            if k.startswith('loss'):
                v = v.item()
                n = f'check_{k}'
                if ignore_non_loss and np.isnan(v):
                    pass
                else:
                    loop_objs[n] = loop_objs.get(n, 0) + v
                    losses[k] = v
                    losses[f'mean_{k}'] = loop_objs[n] / loop_objs['check_nums']

        loop_objs['losses'] = losses

        mem_info = {
            'cpu_info': log_utils.MemoryInfo.get_process_mem_info(),
            'gpu_info': log_utils.MemoryInfo.get_gpu_mem_info()
        } if more_log else {}

        self.log({
            **losses,
            'lr': self.optimizer.param_groups[0]['lr'],
            **mem_info
        }, 'pbar')

        if self.use_scheduler and self.scheduler_strategy == STEP:
            self.scheduler.step()

        if self.check_strategy == STEP:
            self._check_on_train_step_end(loop_objs, **kwargs)

        return loop_objs.get('end_flag', False)  # cancel the training when end_flag is True

    def on_train_epoch_end(self, loop_objs, **kwargs) -> bool:
        loop_objs['epoch'] += 1
        if self.use_scheduler and self.scheduler_strategy == EPOCH:
            self.scheduler.step()
        if self.check_strategy == EPOCH:
            self._check_on_train_epoch_end(loop_objs, **kwargs)
        return loop_objs.get('end_flag', False)  # cancel the training when end_flag is True

    def _check_on_train_step_end(self, loop_objs, check_period=None, batch_size=None, max_save_weight_num=None, is_metric=True, **kwargs):
        total_nums = loop_objs['total_nums']
        if check_period and total_nums % check_period < batch_size:
            self.trace({'total_nums': total_nums}, (bundled.LOGGING, bundled.WANDB))

            state_dict = self._check_train(loop_objs, max_save_weight_num, total_nums, **kwargs)
            if is_metric:
                self._check_metric(loop_objs, state_dict, total_nums, max_save_weight_num, **kwargs)

            self.log_trace(bundled.LOGGING)
            self.log_trace(bundled.WANDB)

    def _check_on_train_epoch_end(self, loop_objs, check_period=None, max_save_weight_num=None, is_metric=True, **kwargs):
        epoch = loop_objs['epoch'] - 1  # epoch in counters is the next epoch, not the last
        self.trace({'epoch': epoch}, (bundled.LOGGING, bundled.WANDB))

        if check_period and epoch % check_period == check_period - 1:
            state_dict = self._check_train(loop_objs, max_save_weight_num, epoch, **kwargs)
            if is_metric:
                self._check_metric(loop_objs, state_dict, epoch, max_save_weight_num, **kwargs)

        self.log_trace(bundled.LOGGING)
        self.log_trace(bundled.WANDB)

    def _check_train(self, loop_objs, max_save_weight_num, check_num, **kwargs):
        """

        Args:
            max_save_weight_num (int|None):
                None, save two weight file with name of `[last/best].pth`
                0, don't save any weight file
                >0, save num file with name of `{check_num}.pth`
            check_num:

        Returns:

        """
        losses = loop_objs.get('losses')
        if losses is not None:
            for k, v in losses.items():
                self.trace({f'loss/{k}': v}, (bundled.LOGGING, bundled.WANDB))
                if np.isnan(v) or np.isinf(v):
                    loop_objs['end_flag'] = True
                    self.log(f'Train will be stop soon, got {v} value from {k}')

            for k in loop_objs:
                if k.startswith('check_'):
                    loop_objs[k] = 0

        last_check_time = loop_objs.get('last_check_time')
        if last_check_time is not None:
            now = time.time()
            self.trace({'time_consume': (now - last_check_time) / 60}, (bundled.LOGGING, bundled.WANDB))
            loop_objs['last_check_time'] = now

        state_dict = self.state_dict()
        state_dict['loop_objs'] = {c: loop_objs[c] for c in self.counters_keys}

        if not isinstance(max_save_weight_num, int):  # None
            self.save_pretrained_checkpoint('last', max_save_weight_num, state_dict)

        elif max_save_weight_num > 0:
            self.save_pretrained_checkpoint(str(check_num), max_save_weight_num, state_dict)

        return state_dict

    def _check_metric(self, loop_objs, state_dict, check_num, max_save_weight_num, metric_kwargs=dict(), **kwargs):
        results = self.metric(epoch=loop_objs['epoch'], **metric_kwargs)
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
                    self.save_pretrained_checkpoint('best', max_save_weight_num, state_dict)

            loop_objs['end_flag'] = loop_objs['end_flag'] or self.stopper(check_num, score)

    def on_train_end(self, **kwargs):
        for func, params in self.train_end_container.items():
            func(**params)

        for item in ('optimizer', 'stopper', 'scaler'):
            if hasattr(self, item):
                delattr(self, item)

    def metric(self, *args, **kwargs) -> dict:
        """call the `predict()` function to get model output, then count the score, expected to return a dict of model score"""
        raise NotImplementedError

    @overload
    def predict(
            self,
            batch_size: int,
            val_dataloader: 'torch.utils.data.DataLoader' = None,
            val_data: Optional[Iterable] = None,

            # for `get_data()`
            data_get_kwargs: dict = dict(),

            # for `torch.utils.data.DataLoader`
            dataloader_kwargs: dict = dict(),

            # for model run
            model_kwargs: dict = dict(),

            # whether to visualize or not
            is_visualize: bool = False,

            # num for visualize
            max_vis_num: int = None,

            vis_pbar=True,

            **kwargs
    ) -> dict:
        pass

    @torch.no_grad()
    def predict(self, vis_pbar=True, **kwargs) -> dict:
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
                data_get_kwargs:
                dataloader_kwargs:

        val_container:
            be loaded what parameters generated by all the prediction pipelines
            suggest to include mainly the following parameters:

        """
        loop_objs, process_kwargs = self.on_val_start(**kwargs)
        kwargs.update(process_kwargs)

        val_dataloader = kwargs['val_dataloader']
        if vis_pbar:
            val_dataloader = tqdm(val_dataloader, desc=visualize.TextVisualize.highlight_str('Val'))

        for loop_inputs in val_dataloader:
            loop_objs.update(
                loop_inputs=loop_inputs,
            )

            self.on_val_step_start(loop_objs, **kwargs)

            model_results = self.on_val_step(loop_objs, **kwargs)
            loop_objs.update(
                model_results=model_results,
            )

            self.on_val_reprocess(loop_objs, **kwargs)
            self.on_val_step_end(loop_objs, **kwargs)

        return self.on_val_end(**kwargs)

    def register_val_start(self, func, **kwargs):
        self.val_start_container.update({func: kwargs})

    def register_end_start(self, func, **kwargs):
        self.val_end_container.update({func: kwargs})

    def on_val_start(self, val_data=None, val_dataloader=None, batch_size=None, data_get_kwargs=dict(), dataloader_kwargs=dict(), epoch=-1, **kwargs):
        assert self.models, 'model list is empty, it seems that you have not init the processor first, perhaps run `processor.init()` first?'
        assert batch_size, 'please set batch_size'
        dataloader_kwargs.setdefault('batch_size', batch_size)
        if val_dataloader is None:
            val_dataloader = self.get_val_dataloader(val_data=val_data, data_get_kwargs=data_get_kwargs, dataloader_kwargs=dataloader_kwargs)

        self.set_mode(train=False)

        loop_objs = dict(
            vis_num=0,
            epoch=epoch
        )

        process_kwargs = dict(
            val_dataloader=val_dataloader,
            process_results=dict(),
        )

        for func, params in self.val_start_container.items():
            func(**params)

        return loop_objs, process_kwargs

    def on_val_step_start(self, loop_objs, **kwargs):
        pass

    def on_val_step(self, loop_objs, model_kwargs=dict(), **kwargs) -> dict:
        """logic of validating step, expected to return a dict of model output included preds
        must return a dict included:
            outputs: original model output
            preds: normalized model output

        """
        raise NotImplementedError

    def on_val_reprocess(self, loop_objs, **kwargs):
        """prepare true and pred label for `visualize()` or `metric()`
        reprocess data will be cached in val_container"""

    def on_val_step_end(self, loop_objs, is_visualize=False, batch_size=16, max_vis_num=None, **kwargs):
        """visualize the model outputs usually"""
        if is_visualize:
            max_vis_num = max_vis_num or float('inf')
            n = min(batch_size, max_vis_num - loop_objs['vis_num'])
            if n > 0:
                self.visualize(loop_objs, n, **kwargs)
                loop_objs['vis_num'] += n

    def visualize(self, loop_objs, n, **kwargs):
        """logic of predict results visualizing"""
        pass

    def on_val_end(self, process_results=dict(), **kwargs):
        """save the results usually"""
        for func, params in self.val_end_container.items():
            func(process_results=process_results, **params, **kwargs)

        return process_results

    model_input_template: 'namedtuple'

    @torch.no_grad()
    def single_predict(self, *obj, **kwargs):
        if not len(obj):
            obj = [None]
        ret = self.batch_predict(*[[o] for o in obj], **kwargs)
        if isinstance(ret, list):
            return ret[0]
        else:
            return ret

    @torch.no_grad()
    def batch_predict(self, *objs, total=None, vis_pbar=True, **kwargs):
        batch_size = kwargs.setdefault('batch_size', 16)
        loop_objs, process_kwargs = self.on_predict_start(**kwargs)
        kwargs.update(process_kwargs)

        total = total or (isinstance(objs[0], (list, tuple)) and len(objs[0])) or 1
        if vis_pbar:
            pbar = tqdm(total=total, desc=visualize.TextVisualize.highlight_str('Predict'))
        for i in range(0, total, batch_size):
            start_idx = i
            end_idx = min(i + batch_size, total)
            loop_inputs = self.gen_predict_inputs(*objs, start_idx=start_idx, end_idx=end_idx, **kwargs)
            loop_inputs = self.on_predict_step_start(loop_inputs, **kwargs)
            loop_objs.update(
                loop_inputs=loop_inputs,
            )
            model_results = self.on_predict_step(loop_objs, **kwargs)
            loop_objs.update(
                model_results=model_results,
            )
            self.on_predict_reprocess(loop_objs, **kwargs)
            self.on_predict_step_end(loop_objs, **kwargs)
            if vis_pbar:
                pbar.update(end_idx - start_idx)

        return self.on_predict_end(**kwargs)

    @torch.no_grad()
    def fragment_predict(self, *obj, **kwargs):
        """Tear large inputs to pieces for prediction, and then, merge the results and restore them"""
        raise NotImplementedError

    def on_predict_start(self, **kwargs):
        assert self.models, 'model list is empty, it seems that you have not init the processor first, perhaps run `processor.init()` first?'

        self.set_mode(train=False)
        loop_objs = dict(
            vis_num=0
        )
        process_kwargs = dict(
            process_results=dict(),
        )
        return loop_objs, process_kwargs

    def gen_predict_inputs(self, *objs, start_idx=None, end_idx=None, **kwargs) -> List[dict]:
        """gen inputs for on_predict_step"""
        raise NotImplementedError

    def on_predict_step_start(self, loop_inputs, **kwargs):
        """preprocess the model inputs"""
        if hasattr(self, 'predict_data_augment'):
            loop_inputs = [self.predict_data_augment(ret) for ret in loop_inputs]
        return loop_inputs

    def on_predict_step(self, loop_inputs, **kwargs):
        return self.on_val_step(loop_inputs, **kwargs)

    def on_predict_reprocess(self, loop_objs, process_results=dict(), return_keys=(), **kwargs):
        """prepare true and pred label for `visualize()`
        reprocess data will be cached in predict_container"""
        model_results = loop_objs['model_results']
        ret = process_results.setdefault(self.model_name, {})
        for k in return_keys:
            if k in model_results[self.model_name]:
                data = ret.setdefault(k, [])
                data.extend(model_results[self.model_name][k])

    def on_predict_step_end(self, loop_objs, **kwargs):
        """visualize the model outputs usually"""
        return self.on_val_step_end(loop_objs, **kwargs)

    def on_predict_end(self, process_results=dict(), **kwargs):
        """visualize results and the return the results"""
        return process_results[self.model_name]
