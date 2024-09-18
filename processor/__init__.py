from .bundled import *
from .data_process import *
from .model_process import *
from utils import converter
import os


class ProcessConfig:
    # for work_dir and cache_dir
    model_version: str
    dataset_version: str

    # for saving some model data, like weight, config, vocab, etc
    # self.work_dir = os.path.abspath(f'{self._model_cache_dir}/{self.model_version}/{self.dataset_version}')
    _model_cache_dir: str = 'model_data'

    # for saving some cache data, like visual image, result text, etc.
    # self.cache_dir = os.path.abspath(f'{self._result_cache_dir}/{self.model_version}/{self.dataset_version}')
    _result_cache_dir: str = 'cache_data'

    # for saving log info, if None, do not save the log info
    log_dir: str = None

    # for marking the base model
    # self.default_model_path = f'{self.work_dir}/{self.model_name}.pth'
    model_name: str = 'model'

    # for loading the train or test data
    data_dir: str = None
    train_dataset_ins: 'torch.utils.data.Dataset' = BaseImgDataset
    val_dataset_ins: 'torch.utils.data.Dataset' = BaseImgDataset

    # 'cpu', or id of gpu device if gpu is available
    device: Union[int, str, torch.device] = None

    use_ema: bool = False
    use_early_stop: bool = True
    use_scaler: bool = False
    use_scheduler: bool = False

    # every epoch or every step to run scheduler
    scheduler_strategy: str = EPOCH
    lrf: float = 0.01

    # for wandb logging
    wandb_id: str = None

    # every epoch or every step to run training check, like saving checkpoint, run the metric step, etc
    check_strategy: str = EPOCH


class FitConfig:
    batch_size: int
    max_epoch: int

    train_dataloader: 'torch.utils.data.DataLoader' = None
    val_dataloader: 'torch.utils.data.DataLoader' = None

    # for `get_data()`
    data_get_kwargs: dict = dict()

    # for `torch.utils.data.DataLoader`
    dataloader_kwargs: dict = dict()

    # num of epoch/step to run training check
    check_period: int = None

    # for `metric()`
    metric_kwargs: dict = dict()

    # which vars will be recorded when training
    _counters: Optional[tuple] = ('per_epoch_nums',)

    # num for accumulate if not None
    accumulate: int = None

    # if True, log more info, like gpu info
    more_log: bool = False

    # if True, while occur nan output, training will be stopped
    ignore_non_loss: bool = False

    # nums of checkpoint files saving when training
    # if None, save 2 checkpoints: 'best.pth', 'last.pth'
    # if int, save {num} checkpoints, use '{check_num}.pth'
    # specially, if 0, do not save weight
    max_save_weight_num: int = None


class PredictConfig:
    batch_size: int
    val_dataloader: 'torch.utils.data.DataLoader' = None

    # for `get_data()`
    data_get_kwargs: dict = dict()

    # for `torch.utils.data.DataLoader`
    dataloader_kwargs: dict = dict()

    # whether to visualize or not
    is_visualize: bool = False

    # num for visualize
    max_vis_num: int = None


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

        see `ProcessConfig` to get more kwargs info,
        also can use the following code to get all possible input kwargs
            from utils.log_utils import get_class_annotations
            print(get_class_annotations(MyProcess))
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

    log_dir = None
    date = datetime.now().isoformat()
    _model_cache_dir = 'model_data'
    _result_cache_dir = 'cache_data'
    default_model_path: str
    cache_dir: str
    model_name = 'model'

    @staticmethod
    def setup_seed(seed=42):
        """42 is a lucky number"""
        import random
        import torch.backends.cudnn as cudnn

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ["PL_GLOBAL_SEED"] = str(seed)
        cudnn.benchmark = False
        cudnn.deterministic = True

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
        self.setup_seed()
        self.counters = dict()
        if torch.cuda.is_available():
            if isinstance(self.device, (str, int)) and self.device != 'cpu':
                self.device = torch.device(f"cuda:{self.device}")
            elif self.device is None:  # default None, use cuda:0 possible
                self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        if not hasattr(self, 'model') or self.model is None:
            self.set_model()

        if not isinstance(self.device, list):
            # note that, it must be set device before load_state_dict()
            self.model.to(self.device)

        self.load_pretrain()
        self.models[self.model_name] = self.model

        # todo: multi device
        # if isinstance(self.device, list):
        #     assert torch.cuda.device_count() >= len(self.device)
        #     device_ids = self.device
        #     self.device = torch.device(f"cuda:{self.device[0]}")
        #     self.model.to(self.device)
        #     self.model = nn.DataParallel(self.model, device_ids=device_ids)
        #     self.optimizer = nn.DataParallel(self.optimizer, device_ids=device_ids)

        try_init_components = [self.set_ema]
        for components in try_init_components:
            try:
                components()
            except NotImplementedError:
                self.log(f'{components} not init', level=logging.DEBUG)

        self.log(f'{torch.__version__ = }')
        self.log(f'{self.device = }')
        self.log(f'{self.models.keys() = }')

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
