import os

from utils import converter, log_utils
from .bundled import *
from .data_process import *
from .model_process import *


class Process(
    LogHooks,
    DataHooks,
    CheckpointHooks,
    ModelHooks,
    ApiHooks
):
    """
    for implementation, expected to override the following methods usually:
        on_train_step(): logic of training step, expected to return a dict of model output included loss
        metric(): call the `predict()` function to get model output, then count the score, expected to return a dict of model score
        on_val_step(): logic of validating step, expected to return a dict of model output included preds
        on_val_reprocess(): prepare true and pred label for `visualize()` or `metric()`
        visualize(): logic of predict results visualizing

    for arguments, all the instance attributes can be set specially when called by accepting the arguments,
    e.g.
        there is a class defined with instance attributes like that:
            class MyProcess(Process):
                input_size = 640
                ...

        and then, you can set special value of `input_size` when called like that:
            MyProcess(input_size=512, ...)

        run `Process.help()` to get more kwargs info,
        also can see `ProcessConfig` to get more kwargs info,
    """

    @overload
    def __init__(
            self,
            # for work_dir and cache_dir
            model_version: str = '',
            dataset_version: str = '',

            # for saving some model data, like weight, config, vocab, etc
            # self.work_dir = os.path.abspath(f'{self._model_cache_dir}/{self.model_version}/{self.dataset_version}')
            _model_cache_dir: str = 'model_data',

            # for saving some cache data, like visual image, result text, etc.
            # self.cache_dir = os.path.abspath(f'{self._result_cache_dir}/{self.model_version}/{self.dataset_version}')
            _result_cache_dir: str = 'cache_data',

            # for saving log info, if None, do not save the log info
            log_dir: str = None,

            # for marking the base model
            # self.default_model_path = f'{self.work_dir}/{self.model_name}.pth'
            model_name: str = 'model',

            # for loading the train or test data
            data_dir: str = None,
            train_dataset_ins: 'torch.utils.data.Dataset' = BaseImgDataset,
            val_dataset_ins: 'torch.utils.data.Dataset' = BaseImgDataset,

            # 'cpu', or id of gpu device if gpu is available
            device: Union[int, str, torch.device] = None,

            use_pretrained: bool = True,

            # for wandb logging
            wandb_id: str = None,

            **kwargs
    ):
        pass

    def __init__(self, **kwargs):
        self.date = datetime.now().isoformat()
        self.default_model_path: str = ''
        self.cache_dir: str = ''

        super().__init__()
        self.__dict__.update(kwargs)

    log_dir: Annotated[
        str,
        'for saving log info, if None, do not save the log info'
    ] = None
    _model_cache_dir: Annotated[
        str,
        'for saving some model data, like weight, config, vocab, etc.',
        "self.work_dir = os.path.abspath(f'{self._model_cache_dir}/{self.model_version}/{self.dataset_version}')"
    ] = 'model_data'

    _result_cache_dir: Annotated[
        str,
        'for saving some cache data, like visual image, result text, etc.',
        "self.cache_dir = os.path.abspath(f'{self._result_cache_dir}/{self.model_version}/{self.dataset_version}')"
    ] = 'cache_data'

    model_name: Annotated[
        str,
        'for marking the base model',
        "self.default_model_path = f'{self.work_dir}/{self.model_name}.pth'"
    ] = 'model'

    @classmethod
    def help(cls):
        info = log_utils.get_class_info(cls)
        anno_dict = log_utils.get_class_annotations(cls)
        keys = sorted(anno_dict.keys())
        help_str = ''
        for k in keys:
            s = f'    ' + k

            if 'type' in anno_dict[k]:
                s += ': ' + str(anno_dict[k]['type'])

            if 'default' in anno_dict[k]:
                default = anno_dict[k]['default']
                if isinstance(default, str):
                    default = f"'{default}'"
                s += ' = ' + str(default)

            if 'comments' in anno_dict[k]:
                c = ''
                for comment in anno_dict[k]['comments']:
                    c += '    # ' + comment + '\n'
                s = c + s

            help_str += s + '\n\n'
        help_str = f'{info["path"]}(\n{help_str})'
        return help_str

    def init(self):
        self.init_logs()
        self.init_paths()
        self.init_components()

    def init_logs(self):
        self.init_log_base(self.log_dir)
        self.init_wandb()

    def init_paths(self):
        self.work_dir = os.path.abspath(f'{self._model_cache_dir}/{self.model_version}/{self.dataset_version}')
        self.cache_dir = os.path.abspath(f'{self._result_cache_dir}/{self.model_version}/{self.dataset_version}')
        self.default_model_path = f'{self.work_dir}/{self.model_name}.pth'
        os_lib.mk_dir(self.work_dir)

        self.log(f'{self.model_version = }')
        self.log(f'{self.dataset_version = }')
        self.log(f'{self.work_dir = }')
        self.log(f'{self.cache_dir = }')
        self.log(f'{self.model_name = }')

    def init_components(self):
        torch_utils.setup_seed()

        self.set_device()
        self.set_tokenizer()

        if not hasattr(self, 'model') or self.model is None:
            self.set_model()

        self.models[self.model_name] = self.model

        try_init_components = [self.set_model_status]
        for components in try_init_components:
            try:
                components()
            except NotImplementedError:
                self.log(f'{components} not init', level=logging.DEBUG)

        self.log(f'{torch.__version__ = }')
        self.log(f'{self.device = }')
        self.log(f'{self.models.keys() = }')

    def set_device(self):
        if torch.cuda.is_available():
            if isinstance(self.device, (str, int)) and self.device != 'cpu':
                self.device = torch.device(f"cuda:{self.device}")
            elif self.device is None:  # default None, use cuda:0 possible
                self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    use_pretrained: bool = True

    def set_model_status(self):
        if self.use_pretrained:
            self.load_pretrained()
        if not isinstance(self.device, list):
            self.model.to(self.device)

    def run(self, max_epoch=100, train_batch_size=16, predict_batch_size=None, fit_kwargs=dict(), metric_kwargs=dict()):
        self.init()
        self.model_info()

        # also add metric_kwargs to fit_kwargs, 'cause there will be metric strategy while fitting
        fit_kwargs.setdefault('metric_kwargs', metric_kwargs)
        fit_kwargs.setdefault('dataloader_kwargs', dict(num_workers=min(train_batch_size, 16)))
        self.fit(
            max_epoch=max_epoch,
            batch_size=train_batch_size,
            **fit_kwargs
        )

        self.save(self.default_model_path, save_type=WEIGHT)

        # self.load(self.model_path, save_type=WEIGHT)
        # self.load(f'{self.work_dir}/last.pth', save_type=WEIGHT)

        metric_kwargs.setdefault('dataloader_kwargs', dict(num_workers=min(predict_batch_size or train_batch_size, 16)))
        r = self.metric(
            batch_size=predict_batch_size or train_batch_size,
            **metric_kwargs
        )
        for k, v in r.items():
            self.log({k: v})


def ddp_process_wrap(process: Process):
    """
    Usage:
        xxx.py:
            @ddp_process_wrap
            class Process():
                ...

        # !/bin/bash
        torchrun --nproc_per_node=2 --nnodes=1 xxx.py

    """
    from torch.utils.data import DataLoader

    local_rank = torch_utils.init_distributed_mode()

    class DDPDataLoader(DataLoader):
        def __init__(self, dataset, *args, shuffle=True, **kwargs):
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
            kwargs.update(
                sampler=sampler,
                shuffle=False
            )
            super().__init__(dataset, *args, **kwargs)

    class DDPProcess(process):
        train_dataloader_ins = DDPDataLoader
        val_dataloader_ins = DDPDataLoader

        def set_device(self):
            self.device = torch.device(f"cuda:{local_rank}")

        def init(self):
            super().init()

            def set_ddp(**kwargs):
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    # static_graph=True
                )

            self.register_train_start(set_ddp)

        def _check_on_train_step_end(self, *arg, **kwargs):
            if local_rank == 0:
                super()._check_on_train_step_end(*arg, **kwargs)

        def _check_on_train_epoch_end(self, *arg, **kwargs):
            if local_rank == 0:
                super()._check_on_train_epoch_end(*arg, **kwargs)

        def model_state_dict(self):
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()
            return state_dict

    return DDPProcess


def ds_process_wrap(process: Process):
    """
    Usage:
        xxx.py:
            @ds_process_wrap
            class Process():
                ...

        # !/bin/bash
        deepspeed --num_gpus=N xxx.py

    """
    from torch.utils.data import DataLoader
    import argparse
    import deepspeed  # pip install deepspeed

    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()

    local_rank = args.local_rank

    deepspeed.init_distributed()

    class DDPDataLoader(DataLoader):
        def __init__(self, dataset, *args, shuffle=True, **kwargs):
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
            kwargs.update(
                sampler=sampler,
                shuffle=False
            )
            super().__init__(dataset, *args, **kwargs)

    class DSProcess(process):
        train_dataloader_ins = DDPDataLoader
        val_dataloader_ins = DDPDataLoader

        # see https://www.deepspeed.ai/docs/config-json/
        ds_config: dict = {
            # "zero_optimization": {
            #     "stage": 0,
            #     "contiguous_gradients": True,
            #     "overlap_comm": True
            # },
        }

        def set_device(self):
            self.device = torch.device(f"cuda:{local_rank}")

        def init(self):
            super().init()

            def set_ddp(process_kwargs={}, **kwargs):
                train_dataloader = process_kwargs['train_dataloader']
                ds_config = self.ds_config
                ds_config.setdefault('train_micro_batch_size_per_gpu', train_dataloader.batch_size)
                self.model, self.optimizer, _, self.scheduler = deepspeed.initialize(
                    args=args,
                    model=self.model,
                    optimizer=self.optimizer,
                    lr_scheduler=self.scheduler,
                    config=ds_config,
                )

            self.register_train_start(set_ddp)

        def on_backward(self, loop_objs, accumulate=None, batch_size=None, use_ema=False, use_scaler=False, **kwargs):
            model_results = loop_objs['model_results']
            loss = model_results['loss']

            # runs backpropagation
            self.model.backward(loss)

            # weight update
            self.model.step()

        def _check_on_train_step_end(self, *arg, **kwargs):
            if local_rank == 0:
                super()._check_on_train_step_end(*arg, **kwargs)

        def _check_on_train_epoch_end(self, *arg, **kwargs):
            if local_rank == 0:
                super()._check_on_train_epoch_end(*arg, **kwargs)

        def model_state_dict(self):
            if isinstance(self.model, deepspeed.runtime.engine.DeepSpeedEngine):
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()
            return state_dict

    return DSProcess


class ParamsSearch:
    """
    Usage:
        .. code-block:: python

            ######## example 1 ########
            from examples.image_classifier import ClsProcess, ImageNet
            from models.image_classifier.ResNet import Model
            from torch import optim

            class ResNet_ImageNet(ClsProcess, ImageNet):
                '''define your own process'''

            params_search = ParamsSearch(
                process=ResNet_ImageNet,
                params=dict(
                    model=dict(
                        instance=Model,
                        const=dict(in_ch=3, out_features=2),
                        var=dict(input_size=(224, 256, 320))
                    ),
                    callable_optimizer=dict(
                        instance=lambda **kwargs: lambda params: optim.Adam(params, **kwargs),
                        var=dict(lr=(0.001, 0.01))
                    ),
                    sys=dict(
                        var=dict(lf=[
                            lambda x, max_epoch, lrf=0.01: (1 - x / max_epoch) * (1.0 - lrf) + lrf,
                            lambda x, max_epoch, lrf=0.01: ((1 - math.cos(x * math.pi / max_epoch)) / 2) * (lrf - 1) + 1,
                        ])
                    )
                ),
                run_kwargs=dict(max_epoch=100, check_period=4),
                process_kwargs=dict(use_wandb=True),
                model_version='ResNet',
                dataset_version='ImageNet2012.ps',
            )
            # there is 12(3*2*2) test group
            params_search.run()

            ######## example 2 ########
            from models.object_detection.YoloV5 import Model, head_config, make_config, default_model_multiple
            from examples.object_detection import xxx as Process
            params_search = ParamsSearch(
                process=Process,
                params=dict(
                    model=dict(
                        instance=Model,
                        const=dict(
                            n_classes=20,
                            in_module_config=dict(in_ch=3, input_size=640),
                            head_config=head_config
                        ),
                        var=[
                            {k: [v] for k, v in make_config(**default_model_multiple['yolov5n']).items()},
                            {k: [v] for k, v in make_config(**default_model_multiple['yolov5s']).items()},
                            {'head_config.anchor_t': [3, 4, 5]}
                        ]
                    ),
                    sys=dict(
                        const=dict(
                            input_size=640,
                            device=0,
                            cls_alias=classes
                        ),
                    )
                ),
                run_kwargs=dict(max_epoch=100, check_period=4, metric_kwargs=dict(is_visualize=True, max_vis_num=8)),
                process_kwargs=dict(use_wandb=True),
                model_version='yolov5-test',
                dataset_version='Voc.ps',
            )
            # there are 5(1+1+3) test groups
            params_search.run()

    """

    def __init__(
            self,
            process, params=dict(),
            process_kwargs=dict(), run_kwargs=dict(),
            model_version='', dataset_version='',
    ):
        self.process = process
        self.process_kwargs = process_kwargs
        self.run_kwargs = run_kwargs
        self.model_version = model_version
        self.dataset_version = dataset_version

        var_params = {k: v['var'] for k, v in params.items() if 'var' in v}
        self.var_params = {k: configs.permute_obj(var_p) for k, var_p in var_params.items()}
        self.const_params = {k: v['const'] for k, v in params.items() if 'const' in v}
        self.var_instance = {k: v['instance'] for k, v in params.items() if 'instance' in v}
        self.total_params = [[]]
        keys = set(self.var_params.keys()) | set(self.const_params.keys())
        for k in keys:
            var_ps = self.var_params.get(k, [{}])
            const_p = self.const_params.get(k, {})
            self.total_params = [_ + [(var_p, const_p)] for _ in self.total_params for var_p in var_ps]

        self.keys = keys

    def run(self):
        kwargs = copy.deepcopy(self.process_kwargs)
        for _ in self.total_params:
            sub_version = ''
            info_msg = ''
            for key, (var_p, const_p) in zip(self.keys, _):
                tmp_var_p = configs.ConfigObjParse.collapse_dict(var_p)
                for k, v in tmp_var_p.items():
                    if len(str(v)) > 8:
                        s = converter.DataConvert.str_to_md5(str(v))
                        sub_version += f'{k}={s[:6]};'
                    else:
                        sub_version += f'{k}={v};'
                    info_msg += f'{k}={v};'

                var_p = configs.ConfigObjParse.expand_dict(var_p)
                params = configs.ConfigObjParse.merge_dict(var_p, const_p)
                if key in self.var_instance:
                    ins = self.var_instance[key]
                    kwargs[key] = ins(**params)
                else:
                    kwargs.update(params)

            sub_version = sub_version[:-1]
            info_msg = info_msg[:-1]

            kwargs['model_version'] = self.model_version
            kwargs['dataset_version'] = f'{self.dataset_version}/{sub_version}'
            kwargs['log_dir'] = f'model_data/{self.model_version}/{self.dataset_version}/{sub_version}/logs'
            process = self.process(**kwargs)
            process.logger.info(info_msg)
            process.run(**self.run_kwargs)
            process.model.cpu()
            torch.cuda.empty_cache()
