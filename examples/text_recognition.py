import cv2
import torch
from torch import optim, nn
from metrics import text_generation
from data_parse.cv_data_parse.data_augmentation import crop, scale, geometry, channel, RandomApply, Apply, complex, pixel_perturbation
from data_parse import DataRegister
from pathlib import Path
from data_parse.cv_data_parse.base import DataVisualizer
from processor import Process, DataHooks, bundled, BaseImgDataset, MixDataset, IterImgDataset
from utils import configs, cv_utils, os_lib, torch_utils


class TrProcess(Process):
    def on_train_step(self, rets, container, **kwargs) -> dict:
        images = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
        text = [ret['text'] for ret in rets]
        images = torch.stack(images)
        output = self.model(images, text)

        return output

    def metric(self, **kwargs):
        container = self.predict(**kwargs)

        metric_results = {}
        for name, results in container['model_results'].items():
            result = text_generation.top_metric.f_measure(results['trues'], results['preds'])

            result.update(
                score=result['f']
            )

            metric_results[name] = result

        return metric_results

    def on_val_step(self, rets, container, **kwargs) -> dict:
        images = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
        images = torch.stack(images)

        models = container['models']
        model_results = {}
        for name, model in models.items():
            outputs = model(images)

            model_results[name] = dict(
                outputs=outputs['pred'],
                preds=outputs['pred'],
            )

        return model_results

    def on_val_reprocess(self, rets, model_results, container, **kwargs):
        for name, results in model_results.items():
            r = container['model_results'].setdefault(name, dict())
            r.setdefault('trues', []).extend([ret['text'] for ret in rets])
            r.setdefault('preds', []).extend(results['preds'])

    def visualize(self, rets, model_results, n, **kwargs):
        for name, results in model_results.items():
            vis_rets = []
            for i in range(n):
                ret = rets[i]
                _p = results['preds'][i]
                _id = Path(ret['_id'])
                vis_rets.append(dict(
                    _id=f'{_id.stem}({_p}){_id.suffix}',  # true(pred).jpg
                    image=ret['ori_image']
                ))

            DataVisualizer(f'{self.cache_dir}/{self.counters["epoch"]}/{name}', verbose=False, pbar=False)(vis_rets)
            self.get_log_trace(bundled.WANDB).setdefault(f'val_image/{name}', []).extend(
                [self.wandb.Image(cv2.cvtColor(ret['image'], cv2.COLOR_BGR2RGB), caption=ret['_id']) for ret in vis_rets]
            )


class DataProcess(DataHooks):
    train_data_num = int(5e5)
    val_data_num = int(5e4)

    aug = RandomApply([
        pixel_perturbation.GaussNoise(),
    ], probs=[0.2])

    post_aug = Apply([
        scale.LetterBox(pad_type=(crop.RIGHT, crop.CENTER)),
        channel.Keep3Dims(),
        # pixel_perturbation.MinMax(),
        # pixel_perturbation.Normalize(0.5, 0.5),
        pixel_perturbation.Normalize(127.5, 127.5),
        channel.HWC2CHW()
    ])

    def train_data_augment(self, ret) -> dict:
        ret.update(self.aug(**ret))
        ret.update(dst=self.input_size)
        ret.update(self.post_aug(**ret))

        return ret

    def val_data_augment(self, ret) -> dict:
        ret.update(dst=self.input_size)
        ret.update(self.post_aug(**ret))

        return ret

    word_dict: dict
    vocab_fn = 'vocab.txt'

    def load_vocab(self):
        loader = os_lib.Loader(stdout_method=self.log)
        return loader.auto_load(f'{self.work_dir}/{self.vocab_fn}')

    def save_vocab(self, vocab):
        saver = os_lib.Saver(stdout_method=self.log)
        saver.auto_save(vocab, f'{self.work_dir}/{self.vocab_fn}')

    def make_vocab(self):
        raise NotImplemented

    def get_vocab(self):
        try:
            vocab = self.load_vocab()
        except OSError:
            vocab = self.make_vocab()

        self.word_dict = {c: i for i, c in enumerate(vocab)}


class MJSynth(DataProcess):
    dataset_version = 'MJSynth'
    data_dir = 'data/MJSynth'

    input_size = (100, 32)  # make sure that image_w / 4 - 1 > max_len
    in_ch = 1
    out_features = 36  # 26 for a-z + 10 for 0-9
    max_seq_len = 25

    def make_vocab(self):
        from data_parse.cv_data_parse.MJSynth import Loader
        vocab = [' '] + Loader.lower_char_list
        self.save_vocab(vocab)
        return vocab

    def get_train_data(self, *args, **kwargs):
        from data_parse.cv_data_parse.MJSynth import Loader

        loader = Loader(self.data_dir)
        iter_data = loader.load(
            set_type=DataRegister.TRAIN, image_type=DataRegister.GRAY_ARRAY, generator=False,
            return_lower=True,
            max_size=self.train_data_num,
        )[0]

        return iter_data

    def get_val_data(self, *args, **kwargs):
        from data_parse.cv_data_parse.MJSynth import Loader

        loader = Loader(self.data_dir)
        iter_data = loader.load(
            set_type=DataRegister.VAL, image_type=DataRegister.GRAY_ARRAY, generator=False,
            return_lower=True,
            max_size=self.val_data_num,
        )[0]

        return iter_data


class SynthText(DataProcess):
    # so slow...
    dataset_version = 'SynthText'
    data_dir = 'data/SynthText'
    train_dataset_ins = IterImgDataset
    train_dataset_ins.length = DataProcess.train_data_num

    input_size = (100, 32)  # make sure that image_w / 4 - 1 > max_len
    in_ch = 1
    out_features = 62  # 26 * 2 for a-z + 10 for 0-9
    max_seq_len = 25

    def make_vocab(self):
        from data_parse.cv_data_parse.SynthOcrText import Loader
        loader = Loader(self.data_dir, verbose=False)
        return loader.get_char_list()

    def get_train_data(self, *args, **kwargs):
        from data_parse.cv_data_parse.SynthOcrText import Loader

        loader = Loader(self.data_dir, verbose=False)
        iter_data = loader.load(
            image_type=DataRegister.GRAY_ARRAY, generator=True,
            max_size=self.train_data_num,
        )[0]

        return iter_data

    def get_val_data(self, *args, **kwargs):
        from data_parse.cv_data_parse.SynthOcrText import Loader

        loader = Loader(self.data_dir)
        iter_data = loader.load(
            image_type=DataRegister.GRAY_ARRAY, generator=False,
            max_size=self.val_data_num,
        )[0]

        return iter_data


class MixMJSynthSynthText(DataProcess):
    dataset_version = 'MixMJSynthSynthText'
    data_dir1 = 'data/MJSynth'
    data_dir2 = 'data/SynthText'
    train_dataset_ins = MixDataset
    dataset_ratio = [0.5, 0.5]

    input_size = (100, 32)  # make sure that image_w / 4 - 1 > max_len
    in_ch = 1
    out_features = 62  # 26 * 2 for a-z + 10 for 0-9
    max_seq_len = 25

    def make_vocab(self):
        from data_parse.cv_data_parse.MJSynth import Loader
        loader1 = Loader(self.data_dir1)

        from data_parse.cv_data_parse.SynthOcrText import Loader
        loader2 = Loader(self.data_dir2, verbose=False)

        char_set = set(loader1.char_list)
        char_list = loader2.get_char_list()
        char_set |= set(char_list)
        return list(char_set)

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
        IterImgDataset.length = num

        return (iter_data1, BaseImgDataset), (iter_data2, IterImgDataset)

    def get_val_data(self, *args, **kwargs):
        from data_parse.cv_data_parse.MJSynth import Loader

        loader = Loader(self.data_dir1)
        iter_data = loader.load(
            set_type=DataRegister.VAL, image_type=DataRegister.GRAY_ARRAY, generator=False,
            return_lower=True,
            max_size=self.val_data_num,
        )[0]

        return iter_data


class CRNN(TrProcess):
    model_version = 'CRNN'

    def set_model(self):
        from models.text_recognition.crnn import Model

        self.get_vacab()
        self.model = Model(
            in_ch=self.in_ch,
            input_size=self.input_size,
            out_features=self.out_features,
            max_seq_len=self.max_seq_len,
            char2id=self.word_dict
        )

    def set_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)


class CRNN_MJSynth(CRNN, MJSynth):
    """
    Usage:
        .. code-block:: python

            from examples.text_recognition import CRNN_MJSynth as Process

            Process().run(max_epoch=500, train_batch_size=256, predict_batch_size=256)
            {'score': 0.7878}
    """


class CRNN_SynthText(CRNN, SynthText):
    """
    Usage:
        .. code-block:: python

            from examples.text_recognition import CRNN_SynthText as Process

            Process().run(max_epoch=500, train_batch_size=256, predict_batch_size=256)
    """


class CRNN_MixMJSynthSynthText(CRNN, MixMJSynthSynthText):
    """
    Usage:
        .. code-block:: python

            from examples.text_recognition import CRNN_MixMJSynthSynthText as Process

            Process().run(max_epoch=500, train_batch_size=256, predict_batch_size=256)
    """


class Svtr(TrProcess):
    model_version = 'Svtr'

    def set_model(self):
        from models.text_recognition.svtr import Model
        self.get_vacab()
        self.model = Model(
            in_ch=self.in_ch,
            input_size=self.input_size,
            out_features=self.out_features,
            max_seq_len=self.max_seq_len,
            char2id=self.word_dict
        )

    def set_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)


class Svtr_MJSynth(Svtr, MJSynth):
    """
    Usage:
        .. code-block:: python

            from examples.text_recognition import Svtr_MJSynth as Process

            Process().run(max_epoch=500, train_batch_size=256, predict_batch_size=256)
            {'score': 0.7962}
    """
