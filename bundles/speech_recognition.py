import copy
import re
from typing import Iterable, List, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from data_parse.au_data_parse.data_augmentation import Apply, feature, perturbation
from data_parse.nl_data_parse.pre_process import cleaner, spliter
from metrics import text_generation
from processor import BaseDataset, DataHooks, Process
from utils import os_lib


def load_cmvn(cmvn_file):
    with open(cmvn_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    means_list = []
    vars_list = []
    for i in range(len(lines)):
        line_item = lines[i].split()
        if line_item[0] == "<AddShift>":
            line_item = lines[i + 1].split()
            if line_item[0] == "<LearnRateCoef>":
                add_shift_line = line_item[3: (len(line_item) - 1)]
                means_list = list(add_shift_line)
                continue
        elif line_item[0] == "<Rescale>":
            line_item = lines[i + 1].split()
            if line_item[0] == "<LearnRateCoef>":
                rescale_line = line_item[3: (len(line_item) - 1)]
                vars_list = list(rescale_line)
                continue
    means = np.array(means_list).astype(np.float32)
    vars = np.array(vars_list).astype(np.float32)
    cmvn = np.array([means, vars])
    return cmvn


class BaseAuDataset(BaseDataset):
    complex_augment_func: Optional

    def __init__(self, iter_data, augment_func=None, complex_augment_func=None, **kwargs):
        super().__init__(iter_data, augment_func, complex_augment_func=complex_augment_func, **kwargs)
        self.loader = os_lib.Loader(verbose=False)

    def __getitem__(self, idx):
        if self.complex_augment_func:
            return self.complex_augment_func(idx, self.iter_data, self.process_one)
        else:
            return self.process_one(idx)

    def process_one(self, idx):
        ret = copy.deepcopy(self.iter_data[idx])
        if isinstance(ret['audio'], str):
            ret['audio_path'] = ret['audio']
            ret['audio'] = self.loader.load_audio(ret['audio'])

        ret['ori_audio'] = ret['audio']
        ret['idx'] = idx

        if self.augment_func:
            ret = self.augment_func(ret)

        return ret


class Funasr(DataHooks):
    dataset_version = 'funasr'
    data_dir = 'data/funasr'

    train_dataset_ins = BaseAuDataset
    val_dataset_ins = BaseAuDataset

    train_data_num: int = None
    val_data_num: int = None

    cmvn_path: str

    def get_data(self, *args, train=True, set_task='', **kwargs) -> Optional[Iterable | Dataset | List[Dataset]]:
        from data_parse.au_data_parse.datasets.funasr import Loader, DataRegister

        loader = Loader(self.data_dir)
        return loader.load(
            set_type=DataRegister.TRAIN if train else DataRegister.TEST,
            audio_type=DataRegister.ARRAY,
            set_task=set_task,
            generator=False,
            max_size=self.train_data_num if train else self.val_data_num
        )[0]

    aug = Apply([
        perturbation.UpSamples(),
        feature.FBank(
            window_type="hamming",
            num_mel_bins=80,
            dither=1.0,
        ),
        feature.LFR(
            lfr_m=7,
            lfr_n=6
        ),
    ])

    def data_augment(self, ret, train=True) -> dict:
        ret.update(self.aug(**ret))
        audio = ret['audio']
        dim = audio.shape[-1]
        ret.update(perturbation.Normalize(
            mean=self.cmvn[0:1, :dim],
            std=self.cmvn[1:2, :dim]
        )(**ret))
        return ret


class BiCifParaformer(Process):
    model_version = 'BiCifParaformer'
    cmvn_path: str
    seg_dict_path: str

    def set_model(self):
        from models.speech_recognition.BiCifParaformer import Model

        self.model = Model()
        self.cmvn = load_cmvn(self.cmvn_path)

    def load_pretrained(self):
        if hasattr(self, 'pretrain_model'):
            from models.speech_recognition.BiCifParaformer import WeightConverter, WeightLoader

            state_dict = WeightLoader.auto_load(self.pretrain_model, map_location=self.device)
            state_dict = WeightConverter.from_official(state_dict)
            self.model.load_state_dict(state_dict, strict=True)

    def set_tokenizer(self):
        from data_parse.nl_data_parse.pre_process.bundled import SimpleTokenizer

        seg_dict = {}
        for line in os_lib.loader.load_txt(self.seg_dict_path):
            s = line.strip().split()
            seg_dict[s[0]] = s[1:]

        def from_paragraph_with_zh_en_mix(paragraph):
            """
            '你好' -> ['你', '好']
            'uncased' -> ['un@@', 'cased']
            """
            pattern = re.compile(r"([\u4E00-\u9FA5A-Za-z0-9])")
            segment = []
            if paragraph in seg_dict:
                segment.extend(seg_dict[paragraph])
            else:
                if pattern.match(paragraph):
                    for char in paragraph:
                        if char in seg_dict:
                            segment.extend(seg_dict[char])
                        else:
                            segment.append("<unk>")
                else:
                    segment.append("<unk>")
            return segment

        sp_token_dict = dict(
            blank='<blank>',
            sos='<s>',
            eos='</s>',
            unk='<unk>',
        )

        _spliter = spliter.ToSegments(
            sep=' ',
            sp_tokens=sp_token_dict.values(),
            cleaner=cleaner.Lower().from_paragraph
        )
        _spliter.to_segment.deep_split_funcs.append(from_paragraph_with_zh_en_mix)

        self.tokenizer = SimpleTokenizer.from_pretrained(
            self.vocab_fn,
            sp_token_dict=sp_token_dict,
            pad_id=-1,
            spliter=_spliter
        )

    def get_model_inputs(self, loop_inputs, train=True):
        speech = []
        texts = []
        speech_lens = []
        for ret in loop_inputs:
            audio = ret['audio']
            audio = torch.from_numpy(audio)
            audio = audio.to(self.device)
            speech.append(audio)
            speech_lens.append(audio.shape[0])
            text = ret['text']
            if train:
                text += self.tokenizer.eos_token
            texts.append(text)
        speech = pad_sequence(speech, batch_first=True, padding_value=0.0)
        model_inputs = dict(
            speech=speech,
            speech_lens=speech_lens,
            texts=texts,
        )
        if train:
            r = self.tokenizer.encode_paragraphs(texts)
            model_inputs.update(
                text_ids=torch.tensor(r['segments_ids'], device=self.device),
                text_lens=torch.tensor(r['seq_lens'], device=self.device)
            )
        return model_inputs

    def on_train_step(self, loop_objs, model_kwargs=dict(), **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        model_inputs = self.get_model_inputs(loop_inputs, train=True)
        model_inputs.update(model_kwargs)
        output = self.model(**model_inputs)
        return output

    def on_val_step(self, loop_objs, model_kwargs=dict(), **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        model_inputs = self.get_model_inputs(loop_inputs, train=False)
        model_inputs.update(model_kwargs)

        model_results = {}
        for name, model in self.models.items():
            model_output = model(**model_inputs)
            model_results[name] = model_output
            model_results[name]['trues'] = model_inputs['texts']

        return model_results

    def on_val_reprocess(self, loop_objs, process_results=dict(), **kwargs):
        model_results = loop_objs['model_results']
        for name in model_results:
            preds = model_results[name]['preds']
            trues = model_results[name]['trues']
            timestamps = model_results[name]['timestamps']
            segments = self.tokenizer.decode_to_segments(preds)

            results = []
            for segment, timestamp, true in zip(segments, timestamps, trues):
                segment, timestamp = spliter.ToSegment().from_segment_by_word_piece_v2(segment, timestamp)
                segment, timestamp = spliter.ToSegment().from_segment_by_merge_single_char(segment, timestamp)
                pred = spliter.ToParagraphs().from_segments_with_zh_en_mix([segment])[0]
                results.append(dict(
                    pred=pred,
                    true=true,
                    segment=segment,
                    timestamp=timestamp,
                ))
            process_results.setdefault(name, []).extend(results)

    def metric(self, **kwargs):
        process_results = self.predict(**kwargs)

        metric_results = {}
        for name, results in process_results.items():
            trues = [result["true"] for result in results]
            preds = [result["pred"] for result in results]
            result = text_generation.TopMetric(
                confusion_method=text_generation.WordLCSConfusionMatrix
            ).f_measure(trues, preds)

            result.update(
                score=result['f']
            )

            metric_results[name] = result

        return metric_results

    sample_rate = 16000

    def gen_predict_inputs(
            self, *objs, start_idx=None, end_idx=None,
            speech=None, speech_path=None,
            source_speech=None, source_speech_path=None,
            **kwargs
    ) -> List[dict]:
        if speech is None and speech_path:
            if isinstance(speech_path, str):
                speech_path = [speech_path]
            speech = [os_lib.loader.load_audio(path, self.sample_rate) for path in speech_path]

        if not isinstance(speech, list):
            speech = [None] * start_idx + [speech] * (end_idx - start_idx)

        inputs = []
        for i in range(start_idx, end_idx):
            per_input = dict(
                audio=speech[i],
                text=''
            )
            inputs.append(per_input)

        return inputs

    def on_predict_reprocess(self, loop_objs, **kwargs):
        self.on_val_reprocess(loop_objs, **kwargs)


class BiCifParaformer_Funasr(BiCifParaformer, Funasr):
    """

    Usage:
        from bundles.speech_recognition import BiCifParaformer_Funasr as Processor

        model_dir = 'xxx'
        processor = Processor(
            vocab_fn=f'{model_dir}/tokens.json',
            seg_dict_path=f'{model_dir}/seg_dict',
            cmvn_path=f'{model_dir}/am.mvn',
            pretrain_model=f'{model_dir}/model.pt',
        )

        processor.init()

        processor.single_predict(
            speech_path='xxx'
        )

        processor.metric(
            batch_size=batch_size,
            data_get_kwargs=dict(
                set_task=set_task
            )
        )

        processor.fit(
            max_epoch=max_epoch,
            batch_size=train_batch_size,
            check_period=check_period,
            data_get_kwargs=dict(
                set_task=set_task,
            ),
            dataloader_kwargs=dict(
                num_workers=num_workers,
                shuffle=True
            ),
            metric_kwargs=dict(
                batch_size=test_batch_size
            ),
            lr=lr,
        )
    """
