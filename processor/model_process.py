import time
import numpy as np
import math
import torch
from torch import nn, optim
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Callable, Dict, Tuple
from tqdm import tqdm
from . import bundled, data_process
from utils import os_lib, configs, visualize, log_utils
from utils.torch_utils import Export

MODEL = 1
WEIGHT = 2
ONNX = 3
JIT = 4
TRITON = 5


class CheckpointHooks:
    logger: Optional

    def save(self, save_path, save_type=WEIGHT, verbose=True, **kwargs):
        os_lib.mk_dir(Path(save_path).parent)
        func = self.save_funcs.get(save_type)
        assert func, ValueError(f'dont support {save_type = }')

        func(self, save_path, **kwargs)
        if verbose:
            self.logger.info(f'Successfully saved to {save_path} !')

    def save_model(self, save_path, save_items: dict = None, **kwargs):
        """

        Args:
            save_path:
            save_items: {path: item}
                path, where the item saved
                item, which obj wanted to save

        """
        torch.save(self.model, save_path, **kwargs)
        if save_items:
            for path, item in save_items.items():
                torch.save(item, path, **kwargs)

    def save_weight(self, save_path, save_items: dict = None, **kwargs):
        """

        Args:
            save_path:
            save_items: {name: item}
                name, which key of the item;
                item, which obj wanted to save
        """
        ckpt = dict(
            model=self.model.state_dict(),
        )
        if save_items:
            ckpt.update(save_items)
        torch.save(ckpt, save_path, **kwargs)

    device = '0'

    def save_torchscript(self, save_path, trace_input=None, model_warp=None, **kwargs):
        if trace_input is None:
            trace_input = (self.gen_example_data(),)

        model = self.model
        if model_warp is not None:
            model = model_warp(model)
        model.to(self.device)

        # note, check model in eval mode first
        model = Export.to_torchscript(model, *trace_input, **kwargs)
        model.save(save_path)

    def save_onnx(self, save_path, trace_input=None, model_warp=None, **kwargs):
        if trace_input is None:
            trace_input = (self.gen_example_data(),)

        model = self.model
        if model_warp is not None:
            model = model_warp(model)
        model.to(self.device)
        Export.to_onnx(model, save_path, *trace_input, **kwargs)

    def save_triton(self, save_path, **kwargs):
        raise NotImplementedError

    save_funcs = {
        MODEL: save_model,
        WEIGHT: save_weight,
        JIT: save_torchscript,
        TRITON: save_triton,
    }

    def load(self, save_path, save_type=WEIGHT, verbose=True, **kwargs):
        func = self.load_funcs.get(save_type)
        assert func, ValueError(f'dont support {save_type = }')

        func(self, save_path, **kwargs)
        if verbose:
            self.logger.info(f'Successfully load {save_path} !')

    model: nn.Module
    aux_model: Dict[str, nn.Module]

    def load_model(self, save_path, save_items: dict = None, **kwargs):
        """

        Args:
            save_path:
            save_items: {path: name}
                path, where the item saved
                name, which key of the item

        """
        self.model = torch.load(save_path, map_location=self.device, **kwargs)
        if save_items:
            for path, name in save_items.items():
                self.__dict__.update({name: torch.load(path, map_location=self.device)})

    def load_weight(self, save_path, save_items: list = None, **kwargs):
        """

        Args:
            save_path:
            save_items: [name]
                name, which key of the item
                if None, load all the items
        Returns:

        """
        ckpt = torch.load(save_path, map_location=self.device, **kwargs)
        self.model.load_state_dict(ckpt.pop('model'), strict=False)

        if 'aux_model' in ckpt:
            aux_model = ckpt.pop('aux_model')
            for name, w in aux_model.items():
                self.aux_model[name].load_state_dict(w, strict=False)

        def _load(name, item):
            if name in self.__dict__ and hasattr(self.__dict__[name], 'load_state_dict'):
                self.__dict__[name].load_state_dict(item)
            else:
                self.__dict__[name] = item

        if save_items is None:
            for name, item in ckpt.items():
                _load(name, item)

        else:
            for name in save_items:
                if name not in ckpt:
                    continue

                item = ckpt[name]
                _load(name, item)

    def load_jit(self, save_path, **kwargs):
        self.model = torch.jit.load(save_path, map_location=self.device, **kwargs)

    load_funcs = {
        MODEL: load_model,
        WEIGHT: load_weight,
        JIT: load_jit
    }


class ModelHooks:
    model: nn.Module
    aux_model: Dict[str, nn.Module]
    optimizer: Optional
    stopper: Optional
    scheduler: Optional
    scaler: Optional
    device: Optional[str]
    trace: 'bundled.LogHooks.trace'
    log: 'bundled.LogHooks.log'
    log_trace: 'bundled.LogHooks.log_trace'

    def set_model(self):
        raise NotImplementedError

    def set_aux_model(self):
        raise NotImplementedError

    def set_mode(self, train=True):
        if train:
            self.model.train()
            if hasattr(self, 'aux_model'):
                for v in self.aux_model.values():
                    v.train()
        else:
            self.model.eval()
            if hasattr(self, 'aux_model'):
                for v in self.aux_model.values():
                    v.eval()

    def set_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters())

    def set_stopper(self):
        from utils.torch_utils import EarlyStopping
        self.stopper = EarlyStopping(patience=10, min_epoch=10, ignore_min_score=0.1, stdout_method=self.log)

    lrf = 0.01

    def lf(self, x, max_epoch):
        # return (1 - x / max_epoch) * (1.0 - self.lrf) + self.lrf
        return ((1 - math.cos(x * math.pi / max_epoch)) / 2) * (self.lrf - 1) + 1  # cos_lr

    def set_scheduler(self, max_epoch):
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda x: self.lf(x, max_epoch))
        self.scheduler.last_epoch = -1

    def set_scaler(self):
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)

    def model_info(self, **kwargs):
        return self._model_info(self.model, **kwargs)

    def _model_info(self, model, depth=None, human_readable=True):
        from utils.torch_utils import ModuleInfo
        profile = ModuleInfo.profile_per_layer(model, depth=depth)
        cols = ('name', 'module', 'params', 'grads', 'args')
        lens = [-1] * len(cols)
        infos = []
        for p in profile:
            info = (
                p[0],
                p[1],
                visualize.TextVisualize.num_to_human_readable_str(p[2]["params"]) if human_readable else p[2]["params"],
                visualize.TextVisualize.num_to_human_readable_str(p[2]["grads"]) if human_readable else p[2]["grads"],
                visualize.TextVisualize.dict_to_str(p[2]["args"])
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
            params = visualize.TextVisualize.num_to_human_readable_str(params)
            grads = visualize.TextVisualize.num_to_human_readable_str(grads)

        s += template % tuple('-' * l for l in lens) + '\n'
        s += template % ('sum', '', params, grads, '')
        self.log(s)
        return infos

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

        container:
            be loaded what parameters generated or changed in all the pipelines
            suggest to include mainly the following parameters:
                train_dataloader:
                val_dataloader:

        """
        container = dict()
        self.on_train_start(container, **kwargs)
        self.on_train(container, **kwargs)
        self.on_train_end(container, **kwargs)

    init_wandb: 'bundled.LogHooks.init_wandb'
    wandb: Optional
    wandb_id: str
    model_version: str
    dataset_version: str
    work_dir: str
    model_name: str
    get_train_dataloader: 'data_process.DataHooks.get_train_dataloader'
    get_val_dataloader: 'data_process.DataHooks.get_val_dataloader'

    def on_train_start(self, container, batch_size=16, metric_kwargs=dict(), return_val_dataloader=True, dataloader_kwargs=dict(), **kwargs):
        metric_kwargs.setdefault('batch_size', batch_size)
        metric_kwargs.setdefault('dataloader_kwargs', {})
        metric_kwargs['dataloader_kwargs'] = configs.merge_dict(dataloader_kwargs, metric_kwargs['dataloader_kwargs'])

        # only init wandb runner before training
        wandb_run = self.wandb.init(
            project=self.model_version,
            name=self.dataset_version,
            dir=f'{self.work_dir}',
            id=self.__dict__.get('wandb_id'),
            reinit=True
        )
        self.wandb_id = wandb_run.id

        self.set_mode(train=True)

        _counters = ['start_epoch', 'total_nums', 'total_steps']
        for c in _counters:
            self.counters.setdefault(c, 0)

        container['train_dataloader'] = self.get_train_dataloader(batch_size=batch_size, **dataloader_kwargs)

        if return_val_dataloader:
            container['val_dataloader'] = self.get_val_dataloader(batch_size=batch_size, **dataloader_kwargs)

        container['metric_kwargs'] = metric_kwargs

    register_logger: 'bundled.LogHooks.register_logger'
    log_methods: dict

    def on_train(self, container, max_epoch=100, **kwargs):

        for i in range(self.counters['start_epoch'], max_epoch):
            self.on_train_epoch_start(container, **kwargs)
            pbar = tqdm(container['train_dataloader'], desc=visualize.TextVisualize.highlight_str(f'Train {i}/{max_epoch}'))
            self.register_logger('pbar', pbar.set_postfix)

            for rets in pbar:
                self.on_train_step_start(rets, container, **kwargs)
                outputs = self.on_train_step(rets, container, **kwargs)
                self.on_backward(outputs, container, **kwargs)
                self.on_train_step_end(rets, outputs, container, **kwargs)

            if self.on_train_epoch_end(container, **kwargs):
                break

    def on_train_epoch_start(self, container, _counters=('per_epoch_loss', 'per_epoch_nums', 'epoch'), **kwargs):
        container['epoch_start_time'] = time.time()
        for c in _counters:
            self.counters.setdefault(c, 0)

    def on_train_step_start(self, rets, container, **kwargs):
        pass

    def on_train_step(self, rets, container, **kwargs) -> dict:
        """logic of model training step, and expected to return a dict of model output
        must return a dict included:
            loss: loss to backward
        """
        raise NotImplementedError

    def on_backward(self, outputs, container, **kwargs):
        loss = outputs['loss']
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def on_train_step_end(self, rets, outputs, container, more_log=False, **kwargs):
        loss = outputs['loss']
        self.counters['total_nums'] += len(rets)
        self.counters['total_steps'] += 1
        self.counters['per_epoch_loss'] += loss.item()
        self.counters['per_epoch_nums'] += len(rets)

        mean_loss = self.counters['per_epoch_loss'] / self.counters['per_epoch_nums']
        losses = {'mean_loss': mean_loss}
        for k, v in outputs.items():
            if k.startswith('loss'):
                losses[k] = v.item()
        container['losses'] = losses

        mem_info = {
            'cpu_info': log_utils.MemoryInfo.get_process_mem_info(),
            'gpu_info': log_utils.MemoryInfo.get_gpu_mem_info()
        } if more_log else {}

        self.log({
            **losses,
            **mem_info
        }, 'pbar')

    save: 'CheckpointHooks.save'
    counters: dict

    def on_train_epoch_end(self, container, check_period=None, batch_size=None, **kwargs) -> bool:
        end_flag = False
        epoch = self.counters['epoch']
        self.counters['epoch'] += 1
        self.trace({'epoch': epoch}, (bundled.LOGGING, bundled.WANDB))

        losses = container.get('losses')
        if losses is not None:
            for k, v in losses.items():
                self.trace({f'loss/{k}': v}, (bundled.LOGGING, bundled.WANDB))
                if np.isnan(v) or np.isinf(v):
                    end_flag = True
                    self.log(f'Train will be stop soon, got {v} value from {k}')

        epoch_start_time = container.get('epoch_start_time')
        if epoch_start_time is not None:
            self.trace({'time_consume': (time.time() - epoch_start_time) / 60}, (bundled.LOGGING, bundled.WANDB))

        self.log_trace(bundled.LOGGING)
        ckpt = {
            'optimizer': self.optimizer.state_dict(),
            'stopper': self.stopper.state_dict(),
            'counters': self.counters,
            'wandb_id': self.wandb_id,
            'date': datetime.now().isoformat()
        }

        if hasattr(self, 'aux_model'):
            ckpt['aux_model'] = {k: v.state_dict() for k, v in self.aux_model.items()}

        self.save(f'{self.work_dir}/last.pth', save_type=WEIGHT, save_items=ckpt)

        if check_period and epoch % check_period == check_period - 1:
            results = self.metric(
                val_dataloader=container.get('val_dataloader'),
                **container.get('metric_kwargs', {})
            )

            scores = {}
            for name, result in results.items():
                scores[f'val_score/{name}'] = result['score']

            self.trace(scores, bundled.WANDB)
            self.log(f'val log: epoch: {epoch}, score: {scores}')

            score = results[self.model_name]['score']

            if score > self.stopper.best_score:
                self.save(f'{self.work_dir}/best.pth', save_type=WEIGHT, save_items=ckpt)

            end_flag = end_flag or self.stopper(epoch=epoch, score=score)
            self.set_mode(train=True)

        self.log_trace(bundled.WANDB)
        return end_flag

    def on_train_end(self, container, **kwargs):
        self.wandb.finish()

    def metric(self, *args, **kwargs) -> dict:
        """call the `predict()` function to get model output, then count the score, expected to return a dict of model score"""
        raise NotImplementedError

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
                model:
                is_visualize:
                max_vis_num:
                save_ret_func:
                dataloader_kwargs:

        container:
            be loaded what parameters generated by all the pipelines
            suggest to include mainly the following parameters:

        """
        container = {}
        self.on_val_start(container, **kwargs)

        for rets in tqdm(container['val_dataloader'], desc=visualize.TextVisualize.highlight_str('Val')):
            self.on_val_step_start(rets, container, **kwargs)
            model_results = self.on_val_step(rets, container, **kwargs)
            self.on_val_reprocess(rets, model_results, container, **kwargs)
            self.on_val_step_end(rets, model_results, container, **kwargs)

        self.on_val_end(container, **kwargs)

        return container

    def on_val_start(self, container, val_dataloader=None, batch_size=16, dataloader_kwargs=dict(), model=None, **kwargs):
        container['val_dataloader'] = val_dataloader if val_dataloader is not None else self.get_val_dataloader(batch_size=batch_size, **dataloader_kwargs)

        self.set_mode(train=False)
        self.counters['vis_num'] = 0
        self.counters.setdefault('epoch', -1)
        container['model_results'] = dict()

    def on_val_step_start(self, rets, container, **kwargs):
        pass

    def on_val_step(self, rets, container, **kwargs) -> dict:
        """logic of validating step, expected to return a dict of model output included preds
        must return a dict included:
            outputs: original model output
            preds: normalized model output

        """
        raise NotImplementedError

    def on_val_reprocess(self, rets, model_results, container, **kwargs):
        """prepare true and pred label for `visualize()` or `metric()`
        reprocess data will be cached in container"""

    def on_val_step_end(self, rets, model_results, container, is_visualize=False, batch_size=16, max_vis_num=None, **kwargs):
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

    def on_val_end(self, container, **kwargs):
        """save the results usually"""

    @torch.no_grad()
    def single_predict(self, image: np.ndarray, **kwargs):
        self.set_mode(train=False)
        ret = self.val_data_augment({'image': image})
        model_results = self.on_val_step([ret], {}, **kwargs)
        return model_results[self.model_name]['preds'][0]

    @torch.no_grad()
    def batch_predict(self, images: List[np.ndarray], batch_size=16, **kwargs):
        self.set_mode(train=False)
        results = []

        for i in range(0, len(images), batch_size):
            rets = [self.val_data_augment({'image': image}) for image in images[i:i + batch_size]]
            model_results = self.on_val_step(rets, {}, **kwargs)
            results.extend(model_results[self.model_name]['preds'])

        return results

    @torch.no_grad()
    def fragment_predict(self, image: np.ndarray, **kwargs):
        """Tear large picture to pieces for prediction, and then, merge the results and restore them"""
        raise NotImplementedError
