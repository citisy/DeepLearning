from typing import List

import cv2
import torch
from torch import optim
from metrics import text_generation
from data_parse.cv_data_parse.data_augmentation import crop, scale, channel, RandomApply, Apply, pixel_perturbation
from data_parse import DataRegister
from pathlib import Path
from data_parse.cv_data_parse.datasets.base import DataVisualizer
from processor import Process, DataHooks, bundled, BaseImgDataset, MixDataset, IterImgDataset
from utils import log_utils, os_lib, torch_utils


class TrProcess(Process):
    word_dict: dict

    def set_tokenizer(self):
        vocab = os_lib.loader.auto_load(self.vocab_fn)
        self.word_dict = {c: i for i, c in enumerate(vocab)}

    def save_vocab(self, vocab):
        saver = os_lib.Saver(stdout_method=self.log)
        saver.auto_save(vocab, f'{self.work_dir}/{self.vocab_fn}')

    def set_optimizer(self, lr=0.0005, **kwargs):
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def get_model_inputs(self, loop_inputs, train=True):
        images = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in loop_inputs]
        images = torch.stack(images)
        inputs = dict(x=images)
        if train:
            inputs.update(
                true_label=[ret['text'] for ret in loop_inputs]
            )

        return inputs

    def on_train_step(self, loop_objs, **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        inputs = self.get_model_inputs(loop_inputs)
        output = self.model(**inputs)
        return output

    def metric(self, **kwargs):
        process_results = self.predict(**kwargs)

        metric_results = {}
        for name, results in process_results.items():
            result = text_generation.top_metric.f_measure(results['trues'], results['preds'])

            result.update(
                score=result['f']
            )

            metric_results[name] = result

        return metric_results

    def on_val_step(self, loop_objs, **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        inputs = self.get_model_inputs(loop_inputs, train=False)

        model_results = {}
        for name, model in self.models.items():
            outputs = model(**inputs)

            model_results[name] = dict(
                outputs=outputs['pred'],
                preds=outputs['pred'],
            )

        return model_results

    def on_val_reprocess(self, loop_objs, process_results=dict(), val=True, **kwargs):
        model_results = loop_objs['model_results']
        loop_inputs = loop_objs['loop_inputs']

        for name, results in model_results.items():
            r = process_results.setdefault(name, dict())
            if val:
                r.setdefault('trues', []).extend([ret['text'] for ret in loop_inputs])
            r.setdefault('preds', []).extend(results['preds'])

    def visualize(self, loop_objs, n, **kwargs):
        model_results = loop_objs['model_results']
        loop_inputs = loop_objs['loop_inputs']

        for name, results in model_results.items():
            vis_rets = []
            for i in range(n):
                ret = loop_inputs[i]
                _p = results['preds'][i]
                _id = Path(ret['_id'])
                vis_rets.append(dict(
                    _id=f'{_id.stem}({_p}){_id.suffix}',  # true(pred).jpg
                    image=ret['ori_image']
                ))

            DataVisualizer(f'{self.cache_dir}/{loop_objs["epoch"]}/{name}', verbose=False, pbar=False)(vis_rets)
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
        # scale.LetterBox(
        #     pad_type=(crop.RIGHT, crop.DOWN),
        #     fill=(0, 0, 0),
        #     interpolation=4
        # ),
        scale.Proportion(interpolation=4, choice_type=scale.SHORTEST),
        crop.Corner(fill=(0, 0, 0), pad_type=1),
        channel.Keep3Dims(),
        # pixel_perturbation.MinMax(),
        # pixel_perturbation.Normalize(0.5, 0.5),
        pixel_perturbation.Normalize(127.5, 127.5),
        channel.HWC2CHW()
    ])

    def data_augment(self, ret, train=True) -> dict:
        if train:
            ret.update(self.aug(**ret))
        ret.update(dst=self.input_size)
        ret.update(self.post_aug(**ret))

        return ret


class MJSynth(DataProcess):
    dataset_version = 'MJSynth'
    data_dir = 'data/MJSynth'

    input_size = (100, 32)  # make sure that image_w / 4 - 1 > max_len
    in_ch = 1
    out_features = 36  # 26 for a-z + 10 for 0-9
    max_seq_len = 25

    def make_vocab(self):
        from data_parse.cv_data_parse.datasets.MJSynth import Loader
        vocab = [' '] + Loader.lower_char_list
        self.save_vocab(vocab)
        return vocab

    def get_data(self, *args, train=True, **kwargs):
        from data_parse.cv_data_parse.datasets.MJSynth import Loader

        loader = Loader(self.data_dir)
        if train:
            return loader.load(
                set_type=DataRegister.TRAIN, image_type=DataRegister.GRAY_ARRAY, generator=False,
                return_lower=True,
                max_size=self.train_data_num,
            )[0]
        else:
            return loader.load(
                set_type=DataRegister.VAL, image_type=DataRegister.GRAY_ARRAY, generator=False,
                return_lower=True,
                max_size=self.val_data_num,
            )[0]


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
        from data_parse.cv_data_parse.datasets.SynthOcrText import Loader
        loader = Loader(self.data_dir, verbose=False)
        return loader.get_char_list()

    def get_data(self, *args, train=True, **kwargs):
        from data_parse.cv_data_parse.datasets.SynthOcrText import Loader

        loader = Loader(self.data_dir, verbose=False)
        if train:
            return loader.load(
                image_type=DataRegister.GRAY_ARRAY, generator=True,
                max_size=self.train_data_num,
            )[0]
        else:
            return loader.load(
                image_type=DataRegister.GRAY_ARRAY, generator=False,
                max_size=self.val_data_num,
            )[0]


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
        from data_parse.cv_data_parse.datasets.MJSynth import Loader
        loader1 = Loader(self.data_dir1)

        from data_parse.cv_data_parse.datasets.SynthOcrText import Loader
        loader2 = Loader(self.data_dir2, verbose=False)

        char_set = set(loader1.char_list)
        char_list = loader2.get_char_list()
        char_set |= set(char_list)
        return list(char_set)

    def get_data(self, *args, train=True, **kwargs):
        from data_parse.cv_data_parse.datasets.MJSynth import Loader

        loader1 = Loader(self.data_dir1)
        if train:
            num = int(self.train_data_num * self.dataset_ratio[0])
            iter_data1 = loader1.load(
                set_type=DataRegister.TRAIN, image_type=DataRegister.GRAY_ARRAY, generator=False,
                max_size=num,
            )[0]

            from data_parse.cv_data_parse.datasets.SynthOcrText import Loader

            loader2 = Loader(self.data_dir2, verbose=False)
            num = int(self.train_data_num * self.dataset_ratio[1])
            iter_data2 = loader2.load(
                image_type=DataRegister.GRAY_ARRAY, generator=True,
                max_size=num,
            )[0]

            return (
                BaseImgDataset(iter_data1, augment_func=self.train_data_augment),
                IterImgDataset(iter_data2, augment_func=self.train_data_augment, length=num)
            )

        else:
            return loader1.load(
                set_type=DataRegister.VAL, image_type=DataRegister.GRAY_ARRAY, generator=False,
                return_lower=True,
                max_size=self.val_data_num,
            )[0]


class CRNN(TrProcess):
    model_version = 'CRNN'

    def set_model(self):
        from models.text_recognition.crnn import Model

        self.model = Model(
            in_ch=self.in_ch,
            input_size=self.input_size,
            out_features=self.out_features,
            max_seq_len=self.max_seq_len,
            char2id=self.word_dict
        )


class CRNN_MJSynth(CRNN, MJSynth):
    """
    Usage:
        .. code-block:: python

            from bundles.text_recognition import CRNN_MJSynth as Process

            Process().run(max_epoch=500, train_batch_size=256, predict_batch_size=256)
            {'score': 0.7878}
    """


class CRNN_SynthText(CRNN, SynthText):
    """
    Usage:
        .. code-block:: python

            from bundles.text_recognition import CRNN_SynthText as Process

            Process().run(max_epoch=500, train_batch_size=256, predict_batch_size=256)
    """


class CRNN_MixMJSynthSynthText(CRNN, MixMJSynthSynthText):
    """
    Usage:
        .. code-block:: python

            from bundles.text_recognition import CRNN_MixMJSynthSynthText as Process

            Process().run(max_epoch=500, train_batch_size=256, predict_batch_size=256)
    """


class Svtr(TrProcess):
    model_version = 'Svtr'

    def set_model(self):
        from models.text_recognition.svtr import Model
        self.model = Model(
            in_ch=self.in_ch,
            input_size=self.input_size,
            out_features=self.out_features,
            max_seq_len=self.max_seq_len,
            char2id=self.word_dict
        )


class Svtr_MJSynth(Svtr, MJSynth):
    """
    Usage:
        .. code-block:: python

            from bundles.text_recognition import Svtr_MJSynth as Process

            Process().run(max_epoch=500, train_batch_size=256, predict_batch_size=256)
            {'score': 0.7962}
    """


class PPOCRv4Rec(TrProcess):
    model_version = 'PPOCRv4_rec'
    config_version = 'teacher'

    def set_model(self):
        from models.text_recognition.PPOCRv4_rec import Model, Config

        self.model = Model(
            id2char=self.word_dict,
            **Config.get(self.config_version),
        )

    def load_pretrained(self):
        from models.text_recognition.PPOCRv4_rec import WeightConverter

        state_dict = torch_utils.Load.from_file(self.pretrained_model)
        if self.config_version == 'teacher':
            state_dict = WeightConverter.from_teacher(state_dict)
        else:
            state_dict = WeightConverter.from_student(state_dict)
        self.model.load_state_dict(state_dict, strict=False)
        # so silly that, import paddle will clear the logger settings, so reinit the logger
        log_utils.logger_init()

    def set_tokenizer(self):
        words = os_lib.loader.load_txt(self.vocab_fn)
        words = [''] + words + [' ']
        self.word_dict = dict(enumerate(words))

    def gen_predict_inputs(self, *objs, start_idx=None, end_idx=None, **kwargs) -> List[dict]:
        images = objs[0][start_idx: end_idx]
        ids = [Path(image).name if isinstance(image, str) else f'{i}.png' for i, image in zip(range(start_idx, end_idx), images)]
        images = [os_lib.loader.load_img(image, channel_fixed_3=True) if isinstance(image, str) else image for image in images]
        rets = []
        for _id, image in zip(ids, images):
            rets.append(dict(
                _id=_id,
                image=image
            ))
        return rets

    def on_predict_reprocess(self, *args, **kwargs):
        return self.on_val_reprocess(*args, val=False, **kwargs)


class PPOCRv4Rec_MJSynth(PPOCRv4Rec, MJSynth):
    """
    from bundles.text_recognition import PPOCRv4Rec_MJSynth as Process

    model_dir = 'xxx'
    process = Process(
        config_version='student',
        pretrained_model=f'{model_dir}/ch_PP-OCRv4_rec_train/student.pdparams',
        # config_version='teacher',
        # pretrained_model=f'{model_dir}/ch_PP-OCRv4_rec_server_train/best_accuracy.pdparams',

        vocab_fn=f'{model_dir}/ppocr_keys_v1.txt'
    )
    process.init()
    process.single_predict('xxx.png')
    """
    input_size = (1000, 48)
