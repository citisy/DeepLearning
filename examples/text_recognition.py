from .base import Process, BaseDataset, WEIGHT
import random
import itertools
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.torch_utils import EarlyStopping, ModuleInfo, Export
from utils import os_lib, configs
from data_parse.cv_data_parse.base import DataRegister, DataVisualizer
from data_parse.cv_data_parse.data_augmentation import crop, scale, geometry, pixel_perturbation, RandomApply, Apply, channel, RandomChoice
from pathlib import Path
import numpy as np
from datetime import datetime
import time
from metrics import text_generation


class MJSynth(Process):
    data_dir = 'data/MJSynth'

    train_data_num = int(5e5)
    val_data_num = int(5e4)

    def get_train_data(self, *args, **kwargs):
        from data_parse.cv_data_parse.MJSynth import Loader

        loader = Loader(self.data_dir)
        iter_data = loader.load(
            set_type=DataRegister.TRAIN, image_type=DataRegister.ARRAY, generator=False,
            return_lower=True,
            max_size=self.train_data_num,
        )[0]

        self.char_dict = {c: i + 1 for i, c in enumerate(loader.lower_char_list)}

        return iter_data

    def get_val_data(self, *args, **kwargs):
        from data_parse.cv_data_parse.MJSynth import Loader

        loader = Loader(self.data_dir)
        iter_data = loader.load(
            set_type=DataRegister.VAL, image_type=DataRegister.ARRAY, generator=False,
            return_lower=True,
            max_size=self.val_data_num,
        )[0]

        return iter_data

    aug = Apply([
        scale.UnrestrictedRectangle(),
        channel.BGR2Gray(),
        pixel_perturbation.MinMax(),
        channel.HWC2CHW()
    ])

    def data_augment(self, ret) -> dict:
        ret.update(dst=self.input_size)
        ret.update(self.aug(**ret))

        return ret

    def val_data_augment(self, ret) -> dict:
        ret.update(dst=self.input_size)
        ret.update(self.aug(**ret))

        return ret


class CRNN(Process):
    def __init__(self,
                 model_version='CRNN',
                 input_size=(100, 32),
                 in_ch=1,
                 max_len=40,
                 out_features=36,  # 26 for a-z + 10 for 0-9
                 **kwargs
                 ):
        from models.text_recognition.crnn import Model

        model = Model(
            in_ch=in_ch,
            input_size=input_size,
            max_len=max_len,
            out_features=out_features
        )

        super().__init__(
            model=model,
            optimizer=optim.Adam(model.parameters(), lr=0.001),
            model_version=model_version,
            **kwargs
        )

    def fit(self, max_epoch=100, batch_size=16, save_period=None, metric_kwargs=dict(), **dataloader_kwargs):
        train_dataloader, val_dataloader, metric_kwargs = self.on_train_start(batch_size, metric_kwargs, **dataloader_kwargs)

        self.model.char2id = self.char_dict
        self.model.id2char = {v: k for k, v in self.char_dict.items()}

        # scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda x: self.lf(x, max_epoch, self.lrf))
        # scheduler.last_epoch = -1

        # scaler = torch.cuda.amp.GradScaler(enabled=True)
        #
        # accumulate = 64 // batch_size
        # j = 0

        for i in range(self.start_epoch, max_epoch):
            self.model.train()
            pbar = tqdm(train_dataloader, desc=f'train {i}/{max_epoch}')
            total_loss = 0
            total_batch = 0
            losses = None
            epoch_start_time = time.time()

            for rets in pbar:
                images = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
                transcription = [ret['transcription'] for ret in rets]

                images = torch.stack(images)

                self.optimizer.zero_grad()
                output = self.model(images, transcription)
                loss = output['loss']
                loss.backward()
                self.optimizer.step()
                # ema.step(self.model)

                total_loss += loss.item()
                total_batch += len(rets)

                mean_loss = total_loss / total_batch

                losses = {
                    'loss': loss.item(),
                    'mean_loss': mean_loss,
                }
                # mem_info = {
                #     'cpu_info': log_utils.MemoryInfo.get_process_mem_info(),
                #     'gpu_info': log_utils.MemoryInfo.get_gpu_mem_info()
                # }

                pbar.set_postfix({
                    **losses,
                    # **mem_info
                })

            if self.on_train_epoch_end(i, save_period, val_dataloader,
                                       losses=losses,
                                       epoch_start_time=epoch_start_time,
                                       # model=ema.ema_model,
                                       **metric_kwargs):
                break

        self.wandb.finish()

    def metric(self, *args, **kwargs):
        true, pred = self.predict(*args, **kwargs)

        result = text_generation.top_metric.f_measure(true, pred)

        result.update(
            score=result['f']
        )

        return result

    def predict(self, val_dataloader=None, batch_size=16, cur_epoch=-1, model=None, visualize=False, max_vis_num=float('inf'), save_ret_func=None, **dataloader_kwargs):
        if val_dataloader is None:
            val_dataloader = self.on_val_start(batch_size, **dataloader_kwargs)

        model = model or self.model
        model.to(self.device)

        pred = []
        true = []
        vis_num = 0

        with torch.no_grad():
            self.model.eval()
            for rets in tqdm(val_dataloader):
                images = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
                images = torch.stack(images)
                transcription = [ret['transcription'] for ret in rets]

                outputs = self.model(images)
                pred.extend(outputs['pred'])
                true.extend(transcription)

                vis_num = self.on_val_step_end(rets, outputs, cur_epoch, visualize, batch_size, max_vis_num, vis_num)

        return true, pred

    def on_val_step_end(self, rets, outputs, cur_epoch, visualize, batch_size, max_vis_num, vis_num):
        if visualize:
            n = min(batch_size, max_vis_num - vis_num)
            if n > 0:
                for ret, _p in zip(rets, outputs['pred']):
                    _id = Path(ret['_id'])
                    ret['_id'] = f'{_id.stem}({_p}){_id.suffix}'
                    ret['image'] = ret['ori_image']
                DataVisualizer(f'{self.save_result_dir}/{cur_epoch}', verbose=False, pbar=False)(rets[:n])
                self.log_info.setdefault('val_image', []).extend([self.wandb.Image(ret['image'], caption=ret['_id']) for ret in rets[:n]])
                vis_num += n

        return vis_num


class CRNN_MJSynth(CRNN, MJSynth):
    """
    Usage:
        .. code-block:: python

            from examples.text_recognition import CRNN_MJSynth as Process

            Process().run(max_epoch=100, train_batch_size=256, predict_batch_size=256)
            {'score': }
    """

    def __init__(self, dataset_version='MJSynth', **kwargs):
        super().__init__(dataset_version=dataset_version, **kwargs)
