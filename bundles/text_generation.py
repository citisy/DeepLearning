import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from data_parse.nl_data_parse.pre_process import bundled, dict_maker, cleaner, snack
from processor import Process, DataHooks, BaseDataset, CheckpointHooks, IterIterDataset
from utils import math_utils, os_lib, torch_utils


class TextProcessForGpt(DataHooks):
    data_dir: str
    max_seq_len: int = 512

    val_dataset_ins = BaseDataset
    train_dataset_ins = BaseDataset

    train_data_num = None
    val_data_num = None

    def data_preprocess(self, iter_data, train=True):
        paragraphs = [ret['text'] for ret in iter_data]
        # ['hello world!'] -> [['hello', ' world', '!']]
        segments = self.tokenizer.spliter.from_paragraphs(paragraphs)

        for ret, segment in zip(iter_data, segments):
            ret.update(
                segment=segment,
                segment_pair_tags=[0] * len(segment),
                text=''.join(segment)
            )

        return iter_data


class SimpleTextForGpt(TextProcessForGpt):
    dataset_version = 'simple_text'
    data_dir: str

    def get_data(self, *args, train=True, **kwargs):
        from data_parse.nl_data_parse.datasets.SimpleText import Loader, DataRegister
        loader = Loader(self.data_dir)

        if train:
            return loader.load(set_type=DataRegister.TRAIN, max_size=self.train_data_num, return_label=True, generator=False)[0]

        else:
            return loader.load(set_type=DataRegister.TEST, max_size=self.val_data_num, return_label=True, generator=False)[0]


class GPT2(Process):
    model_version = 'GPT2'
    use_scaler = True
    scheduler_strategy = 'step'  # step
    max_seq_len: int
    max_gen_len = 20

    def set_model(self):
        from models.text_generation.gpt2 import Model, Config
        self.model = Model(
            self.tokenizer.vocab_size,
            pad_id=self.tokenizer.pad_id,
            **Config.get('117M')
        )

    def set_tokenizer(self):
        self.tokenizer = bundled.GPT2Tokenizer.from_pretrained(self.vocab_fn, self.encoder_fn)

    def set_optimizer(self, lr=1e-4, betas=(0.5, 0.999), **kwargs):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=betas)

    def get_model_inputs(self, loop_inputs, train=True):
        segments = [ret['segment'] for ret in loop_inputs]
        r = self.tokenizer.encode_segments(segments)
        return dict(
            x=torch.tensor(r['segments_ids'], device=self.device, dtype=torch.long),
            seq_lens=r['seq_lens'],
            max_gen_len=self.max_gen_len
        )

    def on_train_step(self, rets, **kwargs) -> dict:
        inputs = self.get_model_inputs(rets)
        with torch.cuda.amp.autocast(True):
            output = self.model(**inputs)

        return output

    def on_val_step(self, loop_objs, **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        inputs = self.get_model_inputs(loop_inputs, train=False)
        seq_lens = inputs['seq_lens']
        max_gen_len = inputs['max_gen_len']

        model_results = {}
        for name, model in self.models.items():
            outputs = model(**inputs)

            ret = dict()

            preds = outputs['preds'].cpu().numpy().tolist()
            preds = [pred[:seq_lens[i] + max_gen_len] for i, pred in enumerate(preds)]
            ret.update(
                preds=preds,
                pred_segment=self.tokenizer.numerizer.decode(preds)
            )

            model_results[name] = ret

        return model_results

    def on_val_step_end(self, *args, **kwargs):
        """do not visualize"""

    def gen_predict_inputs(self, *objs, start_idx=None, end_idx=None, **kwargs):
        rets = []
        for text in objs[0][start_idx: end_idx]:
            ret = dict(text=text)
            rets.append(ret)
        rets = self.val_data_preprocess(rets)
        return rets

    def on_predict_reprocess(self, loop_objs, process_results=dict(), **kwargs):
        model_results = loop_objs['model_results']
        for name, results in model_results.items():
            process_results.setdefault(name, []).extend(results['pred_segment'])


class LoadGPT2FromOpenaiPretrain(CheckpointHooks):
    """load pretrain model from openai"""

    def load_pretrained(self):
        if hasattr(self, 'pretrain_model'):
            from models.text_generation.gpt2 import WeightConverter, WeightLoader

            state_dict = WeightLoader.from_openai_tf(self.pretrain_model, n_layer=self.model.n_layer)
            state_dict = WeightConverter.from_openai(state_dict)
            self.model.load_state_dict(state_dict, strict=False)


class LoadGPT2FromHFPretrain(CheckpointHooks):
    """load pretrain model from huggingface"""

    def load_pretrained(self):
        if hasattr(self, 'pretrain_model'):
            from models.text_generation.gpt2 import WeightLoader, WeightConverter
            state_dict = WeightLoader.from_hf(self.pretrain_model)
            self.model.load_state_dict(WeightConverter.from_huggingface(state_dict), strict=False)


class GPT2FromOpenaiPretrain(GPT2, LoadGPT2FromOpenaiPretrain, TextProcessForGpt):
    """
    Usage:
        .. code-block:: python

            from bundles.text_generation import GPT2FromOpenaiPretrain as Process

            process = Process(
                pretrain_model='...',
                vocab_fn='xxx/vocab.json',
                encoder_fn='xxx/merges.txt'
            )
            process.init()

            # if using `117M` pretrain model
            process.single_predict('My name is Julien and I like to')
            # My name is Julien and I like to play with my friends. I'm a big fan of the game and I'm looking forward to playing

            process.batch_predict([
                'My name is Julien and I like to',
                'My name is Thomas and my main'
            ])
            # My name is Julien and I like to play with my friends. I'm a big fan of the game and I'm looking forward to playing
            # My name is Thomas and my main goal is to make sure that I'm not just a guy who's going to be a part of
    """
    dataset_version = 'openai_pretrain'


class SimpleTextForT5(DataHooks):
    dataset_version = 'simple_text'
    data_dir: str

    max_seq_len: int = 512

    val_dataset_ins = BaseDataset
    train_dataset_ins = BaseDataset

    train_data_num = None
    val_data_num = None

    def get_data(self, *args, train=True, **kwargs):
        from data_parse.nl_data_parse.datasets.SimpleText import Loader, DataRegister
        loader = Loader(self.data_dir)

        if train:
            return loader.load(set_type=DataRegister.TRAIN, max_size=self.train_data_num, return_label=True, generator=False)[0]

        else:
            return loader.load(set_type=DataRegister.TEST, max_size=self.val_data_num, return_label=True, generator=False)[0]


class BaseT5(Process):
    model_version = 'T5'
    config_version = 'small'
    use_scaler = True
    scheduler_strategy = 'step'  # step
    max_seq_len: int
    max_gen_len = 20

    def set_model(self):
        from models.text_generation.T5 import Model, Config
        self.model = Model(
            self.tokenizer.vocab_size,
            eos_id=self.tokenizer.eos_id,
            **Config.get(self.config_version)
        )

    def set_tokenizer(self):
        self.tokenizer = bundled.T5Tokenizer.from_pretrained(self.vocab_fn, self.encoder_fn)

    def set_optimizer(self, lr=1e-4, betas=(0.5, 0.999), **kwargs):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=betas)

    def get_model_inputs(self, loop_inputs, train=True):
        paragraphs = [ret['text'] for ret in loop_inputs]
        inputs = self.tokenizer.encode_paragraphs(paragraphs)
        inputs = torch_utils.Converter.force_to_tensors(inputs, self.device)
        return dict(
            x=inputs['segments_ids'],
            seq_lens=inputs['seq_lens'],
            attention_mask=inputs['valid_segment_tags'],
            max_gen_len=self.max_gen_len
        )

    def on_train_step(self, loop_objs, **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        inputs = self.get_model_inputs(loop_inputs)
        with torch.cuda.amp.autocast(True):
            output = self.model(**inputs)

        return output

    def on_val_step(self, loop_objs, **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        inputs = self.get_model_inputs(loop_inputs, train=False)
        inputs.pop('seq_lens')
        inputs.pop('max_gen_len')

        model_results = {}
        for name, model in self.models.items():
            outputs = model(**inputs)

            ret = dict()

            preds = outputs['preds'].cpu().numpy().tolist()
            ret.update(
                preds=preds,
                pred_segment=self.tokenizer.decode(preds)
            )

            model_results[name] = ret

        return model_results

    def on_val_step_end(self, *args, **kwargs):
        """do not visualize"""

    def gen_predict_inputs(self, *objs, start_idx=None, end_idx=None, **kwargs):
        rets = []
        for text in objs[0][start_idx: end_idx]:
            ret = dict(text=text)
            rets.append(ret)
        rets = self.val_data_preprocess(rets)
        return rets

    def on_predict_reprocess(self, loop_objs, process_results=dict(), **kwargs):
        model_results = loop_objs['model_results']
        for name, results in model_results.items():
            process_results.setdefault(name, []).extend(results['pred_segment'])


class FromT5HFPretrained(CheckpointHooks):
    """load pretrain model from huggingface"""

    def load_pretrained(self):
        if hasattr(self, 'pretrain_model'):
            from models.text_generation.T5 import WeightLoader, WeightConverter
            state_dict = WeightLoader.from_hf(self.pretrain_model)
            state_dict = WeightConverter.from_hf(state_dict)
            self.model.load_state_dict(state_dict, strict=False)


class T5(BaseT5, FromT5HFPretrained, SimpleTextForT5):
    """
    Usage:
        .. code-block:: python

            from bundles.text_generation import T5 as Process

            model_dir = 'xxx'
            process = Process(
                pretrain_model=f'{model_dir}/pytorch_model.bin',
                vocab_fn=f'{model_dir}/tokenizer.json',
                encoder_fn=f'{model_dir}/spiece.model',
                config_version='small'  # ['small', 'base', 'large', '3B', '11B']
            )
            process.init()

            # if using `117M` pretrain model
            process.single_predict('translate English to German: The house is wonderful.')
            # Das Haus ist wunderbar.

            process.batch_predict([
                'translate English to German: The house is wonderful.',
                'summarize: studies have shown that owning a dog is good for you',
            ])
            # Das Haus ist wunderbar.
            # studies have shown that owning a dog is good for you .
    """
    dataset_version = 'huggingface_pretrain'


class FromQwen2Pretrained(CheckpointHooks):
    pretrain_model: str | List[str]

    def load_pretrained(self):
        if hasattr(self, 'pretrain_model'):
            from models.text_generation.qwen2 import WeightLoader, WeightConverter

            if Path(self.pretrain_model).is_dir():
                self.pretrain_model = [str(fp) for fp in os_lib.find_all_suffixes_files(self.pretrain_model, ['.safetensors'])]

            state_dict = WeightLoader.auto_load(self.pretrain_model)
            state_dict = WeightConverter.from_official(state_dict)
            self.model.load_state_dict(state_dict, strict=False, assign=True)


class BaseQwen2(Process):
    config_version: str = '0.5b'

    def set_model(self):
        from models.text_generation.qwen2 import Model, Config

        with torch.device('meta'):
            self.model = Model(**Config.get(self.config_version))

    def set_tokenizer(self):
        from data_parse.nl_data_parse.pre_process.bundled import Qwen2Tokenizer

        self.tokenizer = Qwen2Tokenizer.from_pretrained(
            vocab_fn=self.vocab_fn,
            encoder_fn=self.encoder_fn
        )

    def get_model_inputs(self, loop_inputs, train=True):
        if train:
            raise NotImplementedError
        else:
            return self.get_model_val_inputs(loop_inputs)


class Qwen2Predictor(BaseQwen2):
    def get_model_val_inputs(self, loop_inputs):
        model_inputs = []
        for ret in loop_inputs:
            messages = ret['messages']
            model_input = self.tokenizer.encode_dialog(messages)
            model_input['input_ids'] = model_input.pop('segments_ids')
            model_input = torch_utils.Converter.force_to_tensors(model_input, self.device)
            model_inputs.append(model_input)

        return model_inputs

    def on_val_step(self, loop_objs, model_kwargs=dict(), **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        model_inputs = self.get_model_inputs(loop_inputs, train=False)

        model_results = {}
        for name, model in self.models.items():
            generated_ids = []
            per_seq_lens = []
            for model_input in model_inputs:
                per_seq_lens.extend(model_input.pop('per_seq_lens'))
                model_output = model(**model_input, **model_kwargs)

                preds = model_output['preds']
                preds = preds.cpu().numpy()

                seq_lens = model_input['seq_lens']
                generated_ids += [pred[seq_len:-1] for seq_len, pred in zip(seq_lens, preds)]

            model_results[name] = dict(
                generated_ids=generated_ids,
                per_seq_lens=per_seq_lens
            )

        return model_results

    def gen_predict_inputs(self, *objs, start_idx=None, end_idx=None, messages=None, content=None, **kwargs) -> List[dict]:
        if messages is None:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": content}
            ]

        return [dict(
            messages=messages
        )] * (end_idx - start_idx)

    def on_predict_reprocess(self, loop_objs, process_results=dict(), **kwargs):
        model_results = loop_objs['model_results']
        generated_ids = model_results[self.model_name]['generated_ids']
        per_seq_lens = model_results[self.model_name]['per_seq_lens']
        texts = self.tokenizer.decode_to_segments(generated_ids)
        results = [dict(
            text=text,
            per_seq_len=per_seq_len
        ) for text, per_seq_len in zip(texts, per_seq_lens)]
        process_results.setdefault(self.model_name, []).extend(results)


class Qwen2(FromQwen2Pretrained, Qwen2Predictor):
    """
    Usage:
        .. code-block:: python

            from bundles.text_generation import Qwen2 as Process

            model_dir = 'xxx'
            process = Process(
                pretrain_model=model_dir,
                vocab_fn=f'{model_dir}/vocab.json',
                encoder_fn=f'{model_dir}/merges.txt',
                config_version='0.5b'  # ['0.5b', '1.5b', '7b', '72b']
            )

            process.init()

            # todo: only support single predict
            content = '简单介绍一下大模型。'
            process.single_predict(content=content)['text']
            # '大模型是一种计算机视觉模型，它能够模拟人类视觉感知，能够识别和理解图像中的物体、形状、颜色等特征，并能够进行分类、识别、预测等任务。大模型可以用于图像识别、自然语言处理、计算机视觉等领域。'

            past_kvs = process.model.make_caches()

            # Custom management kv caches
            # first usage
            content = '简单介绍一下大模型。'
            response = process.single_predict(
                content=content,
                model_kwargs=dict(
                    start_pos=0,
                    past_kvs=past_kvs,
                )
            )
            print(response['text'])

            # second usage, using the same system prompt as the first
            start_pos = response['per_seq_len'][0]
            past_kvs = [
                dict(
                    k=d['k'][:, :, :start_pos, :],
                    v=d['v'][:, :, :start_pos, :],
                )
                for d in past_kvs
            ]  # only use system caches

            content = '怎么做炒鸡蛋'
            response = process.single_predict(
                content=content,
                model_kwargs=dict(
                    start_pos=start_pos,
                    past_kvs=past_kvs,
                )
            )
            print(response['text'])
    """
