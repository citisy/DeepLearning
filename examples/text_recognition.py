from .base import Process, MixDataset, BaseDataset, IterDataset
import torch
from torch import nn, optim
from tqdm import tqdm
from data_parse.cv_data_parse.base import DataRegister, DataVisualizer
from data_parse.cv_data_parse.data_augmentation import crop, scale, geometry, pixel_perturbation, RandomApply, Apply, channel, RandomChoice
from pathlib import Path
import time
from metrics import text_generation
from utils import os_lib


class TrProcess(Process):
    def fit(self, max_epoch=100, batch_size=16, save_period=None, metric_kwargs=dict(), **dataloader_kwargs):
        train_dataloader, val_dataloader, metric_kwargs = self.on_train_start(batch_size, metric_kwargs, **dataloader_kwargs)

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
                self.log_info.setdefault('val_image', []).extend([self.wandb.Image(ret['image'], mode='BGR', caption=ret['_id']) for ret in rets[:n]])
                vis_num += n

        return vis_num


class DataProcess(Process):
    train_data_num = int(5e5)
    val_data_num = int(5e4)

    aug = Apply([
        scale.LetterBox(pad_type=(crop.RIGHT, crop.CENTER)),
        channel.Keep3Dims(),
        # pixel_perturbation.MinMax(),
        # pixel_perturbation.Normalize(0.5, 0.5),
        pixel_perturbation.Normalize(127.5, 127.5),
        channel.HWC2CHW()
    ])

    def data_augment(self, ret) -> dict:
        ret.update(
            RandomApply([
                pixel_perturbation.GaussNoise(),
            ], probs=[0.2])(**ret)
        )
        ret.update(dst=self.input_size)
        ret.update(self.aug(**ret))

        return ret

    def val_data_augment(self, ret) -> dict:
        ret.update(dst=self.input_size)
        ret.update(self.aug(**ret))

        return ret

    def save_char_dict(self, char_dict):
        saver = os_lib.Saver(stdout_method=self.logger.info)
        saver.save_json(char_dict, f'{self.model_dir}/{self.dataset_version}/char_dict.json')

    def load_char_dict(self):
        loader = os_lib.Loader(stdout_method=self.logger.info)
        return loader.load_json(f'{self.model_dir}/{self.dataset_version}/char_dict.json')


class MJSynth(DataProcess):
    data_dir = 'data/MJSynth'

    def get_train_data(self, *args, **kwargs):
        from data_parse.cv_data_parse.MJSynth import Loader

        loader = Loader(self.data_dir)
        iter_data = loader.load(
            set_type=DataRegister.TRAIN, image_type=DataRegister.GRAY_ARRAY, generator=False,
            return_lower=True,
            max_size=self.train_data_num,
        )[0]

        try:
            char_dict = self.load_char_dict()
        except:
            char_dict = {c: i + 1 for i, c in enumerate(loader.lower_char_list)}
        self.model.char2id = char_dict
        self.model.id2char = {v: k for k, v in char_dict.items()}
        self.save_char_dict(char_dict)

        return iter_data

    def get_val_data(self, *args, **kwargs):
        from data_parse.cv_data_parse.MJSynth import Loader

        loader = Loader(self.data_dir)
        iter_data = loader.load(
            set_type=DataRegister.VAL, image_type=DataRegister.GRAY_ARRAY, generator=False,
            return_lower=True,
            max_size=self.val_data_num,
        )[0]

        char_dict = self.load_char_dict()
        self.model.char2id = char_dict
        self.model.id2char = {v: k for k, v in char_dict.items()}

        return iter_data


class SynthText(DataProcess):
    train_dataset_ins = IterDataset
    train_dataset_ins.length = DataProcess.train_data_num
    data_dir = 'data/SynthText'

    def get_train_data(self, *args, **kwargs):
        from data_parse.cv_data_parse.SynthOcrText import Loader

        loader = Loader(self.data_dir, verbose=False)
        iter_data = loader.load(
            image_type=DataRegister.GRAY_ARRAY, generator=True,
            max_size=self.train_data_num,
        )[0]

        try:
            char_dict = self.load_char_dict()
        except:
            char_list = loader.get_char_list()
            char_list.remove(' ')
            char_dict = {c: i + 1 for i, c in enumerate(char_list)}

        self.model.char2id = char_dict
        self.model.id2char = {v: k for k, v in char_dict.items()}
        self.save_char_dict(char_dict)

        return iter_data

    def get_val_data(self, *args, **kwargs):
        from data_parse.cv_data_parse.SynthOcrText import Loader

        loader = Loader(self.data_dir)
        iter_data = loader.load(
            image_type=DataRegister.GRAY_ARRAY, generator=False,
            max_size=self.val_data_num,
        )[0]

        char_dict = self.load_char_dict()
        self.model.char2id = char_dict
        self.model.id2char = {v: k for k, v in char_dict.items()}

        return iter_data


class MixMJSynthSynthText(DataProcess):
    train_dataset_ins = MixDataset
    data_dir1 = 'data/MJSynth'
    data_dir2 = 'data/SynthText'
    dataset_ratio = [0.5, 0.5]

    def get_train_data(self, *args, **kwargs):
        from data_parse.cv_data_parse.MJSynth import Loader

        loader1 = Loader(self.data_dir1)
        num = int(self.train_data_num * self.dataset_ratio[0])
        iter_data1 = loader1.load(
            set_type=DataRegister.TRAIN, image_type=DataRegister.GRAY_ARRAY, generator=False,
            max_size=num,
        )[0]

        from data_parse.cv_data_parse.SynthOcrText import Loader

        loader2 = Loader(self.data_dir2, verbose=False)
        num = int(self.train_data_num * self.dataset_ratio[1])
        iter_data2 = loader2.load(
            image_type=DataRegister.GRAY_ARRAY, generator=True,
            max_size=num,
        )[0]
        IterDataset.length = num

        try:
            char_dict = self.load_char_dict()
        except:
            char_set = set(loader1.char_list)
            char_list = loader2.get_char_list()
            char_list.remove(' ')
            char_set |= set(char_list)
            char_dict = {c: i + 1 for i, c in enumerate(char_set)}

        self.model.char2id = char_dict
        self.model.id2char = {v: k for k, v in char_dict.items()}
        self.save_char_dict(char_dict)

        return (iter_data1, BaseDataset), (iter_data2, IterDataset)

    def get_val_data(self, *args, **kwargs):
        from data_parse.cv_data_parse.MJSynth import Loader

        loader = Loader(self.data_dir1)
        iter_data = loader.load(
            set_type=DataRegister.VAL, image_type=DataRegister.GRAY_ARRAY, generator=False,
            return_lower=True,
            max_size=self.val_data_num,
        )[0]

        char_dict = self.load_char_dict()
        self.model.char2id = char_dict
        self.model.id2char = {v: k for k, v in char_dict.items()}

        return iter_data


class CRNN(TrProcess):
    def __init__(self,
                 model_version='CRNN',
                 input_size=(100, 32),  # make sure that image_w / 4 - 1 > max_len
                 in_ch=1,
                 out_features=36,  # 26 for a-z + 10 for 0-9
                 max_seq_len=25,
                 **kwargs
                 ):
        from models.text_recognition.crnn import Model

        model = Model(
            in_ch=in_ch,
            input_size=input_size,
            out_features=out_features,
            max_seq_len=max_seq_len
        )

        super().__init__(
            model=model,
            optimizer=optim.Adam(model.parameters(), lr=0.0005),
            model_version=model_version,
            **kwargs
        )


class CRNN_MJSynth(CRNN, MJSynth):
    """
    Usage:
        .. code-block:: python

            from examples.text_recognition import CRNN_MJSynth as Process

            Process().run(max_epoch=500, train_batch_size=256, predict_batch_size=256)
            {'score': 0.7878}
    """

    def __init__(self, dataset_version='MJSynth', out_features=36, **kwargs):
        super().__init__(dataset_version=dataset_version, out_features=out_features, **kwargs)


class CRNN_SynthText(CRNN, SynthText):
    def __init__(self, dataset_version='SynthText', out_features=62, **kwargs):
        super().__init__(dataset_version=dataset_version, out_features=out_features, **kwargs)


class CRNN_MixMJSynthSynthText(CRNN, MixMJSynthSynthText):
    def __init__(self, dataset_version='MixMJSynthSynthText', out_features=62, **kwargs):
        super().__init__(dataset_version=dataset_version, out_features=out_features, **kwargs)


class Svtr(TrProcess):
    def __init__(self,
                 model_version='Svtr',
                 input_size=(100, 32),
                 in_ch=1,
                 out_features=36,  # 26 for a-z + 10 for 0-9
                 max_seq_len=25,
                 **kwargs
                 ):
        from models.text_recognition.svtr import Model

        model = Model(
            in_ch=in_ch,
            input_size=input_size,
            out_features=out_features,
            max_seq_len=max_seq_len
        )

        super().__init__(
            model=model,
            optimizer=optim.Adam(model.parameters(), lr=0.0005),
            model_version=model_version,
            **kwargs
        )


class Svtr_MJSynth(Svtr, MJSynth):
    """
    Usage:
        .. code-block:: python

            from examples.text_recognition import Svtr_MJSynth as Process

            Process().run(max_epoch=500, train_batch_size=256, predict_batch_size=256)
            {'score': 0.7962}
    """

    def __init__(self, dataset_version='SynthText', out_features=36, **kwargs):
        super().__init__(dataset_version=dataset_version, out_features=out_features, **kwargs)
