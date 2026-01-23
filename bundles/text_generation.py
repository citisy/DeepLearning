import copy
import json
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence

from data_parse.nl_data_parse.pre_process import bundled, snack
from processor import BaseDataset, CheckpointHooks, DataHooks, Process, data_process
from utils import torch_utils
from . import text_pretrain


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
            text_ids=torch.tensor(r['segments_ids'], device=self.device, dtype=torch.long),
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
        from models.text_generation.gpt2 import WeightConverter, WeightLoader

        state_dict = WeightLoader.from_openai_tf(self.pretrained_model, n_layer=self.model.n_layer)
        state_dict = WeightConverter.from_openai(state_dict)
        self.model.load_state_dict(state_dict, strict=False)


class LoadGPT2FromHFPretrain(CheckpointHooks):
    """load pretrain model from huggingface"""

    def load_pretrained(self):
        from models.text_generation.gpt2 import WeightLoader, WeightConverter
        state_dict = WeightLoader.from_hf(self.pretrained_model)
        self.model.load_state_dict(WeightConverter.from_huggingface(state_dict), strict=False)


class GPT2FromOpenaiPretrain(GPT2, LoadGPT2FromOpenaiPretrain, TextProcessForGpt):
    """
    Usage:
        .. code-block:: python

            from bundles.text_generation import GPT2FromOpenaiPretrain as Process

            process = Process(
                pretrained_model='...',
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
        self.tokenizer = bundled.T5Tokenizer.from_pretrained(vocab_fn=self.vocab_fn, encoder_fn=self.encoder_fn)

    def set_optimizer(self, lr=1e-4, betas=(0.5, 0.999), **kwargs):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=betas)

    def get_model_inputs(self, loop_inputs, train=True):
        paragraphs = [ret['text'] for ret in loop_inputs]
        inputs = self.tokenizer.encode_paragraphs(paragraphs)
        inputs = torch_utils.Converter.force_to_tensors(inputs, self.device)
        return dict(
            text_ids=inputs['segments_ids'],
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
        from models.text_generation.T5 import WeightLoader, WeightConverter
        state_dict = WeightLoader.from_hf(self.pretrained_model)
        state_dict = WeightConverter.from_hf(state_dict)
        self.model.load_state_dict(state_dict, strict=False)


class T5(BaseT5, FromT5HFPretrained, SimpleTextForT5):
    """
    Usage:
        .. code-block:: python

            from bundles.text_generation import T5 as Process

            model_dir = 'xxx'
            process = Process(
                pretrained_model=f'{model_dir}/pytorch_model.bin',
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


class ChatText(DataHooks):
    dataset_version = 'simple_chat_text'
    train_dataset_ins = data_process.IterRedisDataset
    val_dataset_ins = data_process.IterRedisDataset

    def get_train_data(self, *args, fn=None, cacher_kwargs=dict(), train_data_num=None, data_loader_kwargs=dict(), **kwargs):
        from data_parse.nl_data_parse.datasets.SimpleChatText import Loader
        loader = Loader(self.data_dir, **data_loader_kwargs)
        iter_data = loader.load(
            generator=True,
            fn=fn,
            max_size=train_data_num
        )[0]
        dataset_ins = self.train_dataset_ins
        return dataset_ins(
            iter_data,
            augment_func=self.train_data_augment,
            cacher_kwargs=cacher_kwargs
        )

    def data_augment(self, ret, train=True) -> dict:
        messages = ret['messages']
        if isinstance(messages, str):
            messages = json.loads(messages)
        ret = self.tokenizer.encode_dialog(messages)
        ret['messages'] = messages
        return ret


class Qwen2Predictor(Process):
    def get_model_val_inputs(self, loop_inputs):
        text_ids = []
        seq_lens = []
        per_seq_lens = []

        for ret in loop_inputs:
            text_ids.append(ret['segment_ids'])
            seq_lens.append(ret['seq_lens'])
            if 'per_seq_lens' in ret:
                per_seq_lens.append(ret['per_seq_lens'])
            else:
                per_seq_lens.append([len(ret['segment_ids'])])

        model_inputs = dict(
            text_ids=text_ids,
            seq_lens=seq_lens,
            per_seq_lens=per_seq_lens,
        )
        model_inputs = torch_utils.Converter.force_to_tensors(model_inputs, self.device)
        model_inputs['text_ids'] = pad_sequence(model_inputs['text_ids'], batch_first=True, padding_value=self.tokenizer.pad_id)

        return model_inputs

    def on_val_step(self, loop_objs, model_kwargs=dict(), **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        model_inputs = self.get_model_inputs(loop_inputs, train=False)
        per_seq_lens = model_inputs.pop('per_seq_lens')

        model_results = {}
        for name, model in self.models.items():
            model_output = model(**model_inputs, **model_kwargs)
            preds = model_output['preds']
            preds = preds.cpu().numpy()
            seq_lens = model_inputs['seq_lens']
            end_pos = model_output['end_pos']
            generated_ids = [pred[seq_len:end_p] for seq_len, pred, end_p in zip(seq_lens, preds, end_pos)]

            model_results[name] = dict(
                generated_ids=generated_ids,
                per_seq_lens=per_seq_lens
            )

        return model_results

    def gen_predict_inputs(self, *objs, start_idx=None, end_idx=None, messages=None, content=None, text=None, **kwargs) -> List[dict]:
        if not isinstance(text, list):
            text = [None] * start_idx + [text] * (end_idx - start_idx)

        if not isinstance(content, list):
            content = [None] * start_idx + [content] * (end_idx - start_idx)

        if messages is None:
            messages = [None] * start_idx
            for i in range(start_idx, end_idx):
                messages.append([
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": content[i]}
                ])

        ret = [dict(messages=messages[i], text=text[i]) for i in range(start_idx, end_idx)]
        return ret

    def on_predict_reprocess(self, loop_objs, process_results=dict(), **kwargs):
        model_results = loop_objs['model_results']
        for name, ret in model_results.items():
            generated_ids = ret['generated_ids']
            per_seq_lens = ret['per_seq_lens']
            texts = self.tokenizer.decode_to_segments(generated_ids)
            results = [dict(
                text=text,
                per_seq_len=per_seq_len
            ) for text, per_seq_len in zip(texts, per_seq_lens)]
            process_results.setdefault(name, []).extend(results)


class Qwen2ForPretrainText(text_pretrain.Qwen2ForPretrainText, Qwen2Predictor):
    """see `Qwen2ForChatText`"""


class Qwen2ForChatText(text_pretrain.BaseQwen2, text_pretrain.Qwen2Trainer, Qwen2Predictor, ChatText, text_pretrain.FromQwen2Pretrained):
    """
    Usage:
        .. code-block:: python

            from bundles.text_generation import Qwen2ForChatText as Process

            model_dir = 'xxx'
            process = Process(
                pretrained_model=model_dir,
                vocab_fn=f'{model_dir}/vocab.json',
                encoder_fn=f'{model_dir}/merges.txt',
                config_version='0.5b'  # ['0.5b', '1.5b', '7b', '72b']
            )

            process.init()

            #### predict
            content = '简单介绍一下大模型。'
            process.single_predict(
                content=content,
                model_kwargs=dict(
                    max_gen_len=512,
                    top_k=10
                ),
            )['text']
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

            #### train
            process.fit(
                batch_size=16,
                max_epoch=2,
                data_get_kwargs=dict(
                    fn='xxx.jsonl',
                    cacher_kwargs=dict(
                        host='xxx',
                        port=xxx,
                        password='xxx',
                        db=0,
                        verbose=False
                    ),
                ),
                data_preprocess_kwargs=dict(
                    is_preprocess=True    # Only run on the first time
                ),
                dataloader_kwargs=dict(
                    num_workers=8,
                    shuffle=True
                ),
                # use_scaler=True,
                use_scheduler=True,
                scheduler_strategy='step',
                check_strategy='step',
                check_period=100000,
                is_metric=False,
                init_weight=True,

                max_save_weight_num=5,
                accumulate=8*16,
            )
    """


class ChatTextTrainDataWithDpo(ChatText):
    dataset_version = 'simple_dpo_chat_text'

    def get_train_data(self, *args, fn=None, cacher_kwargs=dict(), train_data_num=None, data_loader_kwargs=dict(), **kwargs):
        from data_parse.nl_data_parse.datasets.SimpleDpoChatText import Loader
        loader = Loader(self.data_dir, **data_loader_kwargs)
        iter_data = loader.load(
            generator=True,
            fn=fn,
            max_size=train_data_num
        )[0]
        dataset_ins = self.train_dataset_ins
        return dataset_ins(
            iter_data,
            augment_func=self.train_data_augment,
            cacher_kwargs=cacher_kwargs
        )

    def train_data_augment(self, ret, train=True) -> dict:
        ret_ = {}
        for k in ['chosen', 'rejected']:
            messages = ret[k]
            if isinstance(messages, str):
                messages = json.loads(messages)
            tmp = self.tokenizer.encode_dialog(messages)
            tmp['messages'] = messages
            ret_[k] = tmp
        return ret_


class Qwen2TrainerWithDpo(text_pretrain.Qwen2Trainer):
    def init(self):
        super().init()

        def add_dpo(**kwargs):
            from models.tuning.dpo import ModelWrap

            ref_model = copy.deepcopy(self.model)
            dpo_wrap = ModelWrap(ref_model, self.model.decode, ref_model.decode)
            self.model = dpo_wrap.wrap(self.model)
            self.dpo_wrap = dpo_wrap
            self.models['dpo'] = ref_model

        self.register_train_start(add_dpo)

    def get_model_train_inputs(self, loop_inputs):
        segment_ids = []
        attention_mask = []
        chosen_idx = []

        for ret in loop_inputs:
            for k in ['chosen', 'rejected']:
                segment_ids.append(ret[k]['segment_ids'])
                seq_lens = ret[k]['seq_lens']
                per_seq_lens = ret[k]['per_seq_lens']
                # ignore the first two content
                valid_segment_tags = [i >= per_seq_lens[0] + per_seq_lens[1] for i in range(seq_lens)]
                attention_mask.append(valid_segment_tags)
                chosen_idx.append(k == 'chosen')

        segment_ids = snack.align(
            segment_ids,
            max_seq_len=self.tokenizer.max_seq_len,
            pad_obj=self.tokenizer.pad_id,
            pad_type=snack.MAX_LEN
        )

        attention_mask = snack.align(
            attention_mask,
            max_seq_len=self.tokenizer.max_seq_len,
            pad_obj=False,
            pad_type=snack.MAX_LEN
        )

        segment_ids = torch.tensor(segment_ids)
        attention_mask = torch.tensor(attention_mask)
        text_ids = segment_ids[:, :-1]
        label_ids = segment_ids.clone()[:, 1:]
        attention_mask = attention_mask[:, 1:]

        model_inputs = dict(
            text_ids=text_ids,
            label_ids=label_ids,
            attention_mask=attention_mask,
            chosen_idx=chosen_idx,
        )
        model_inputs = torch_utils.Converter.force_to_tensors(model_inputs, self.device)
        return model_inputs


class Qwen2ForChatTextWithDpo(Qwen2ForChatText, Qwen2TrainerWithDpo, ChatTextTrainDataWithDpo):
    pass
