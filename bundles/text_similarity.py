from typing import List

from data_parse.nl_data_parse.pre_process import bundled
from processor import Process
from utils import math_utils, torch_utils


class BgeReranker(Process):
    """
    Usage:
        model_dir = 'xxx'
        processor = BgeReranker(
            pretrained_model=f'{model_dir}/model.safetensors',
            vocab_fn=f'{model_dir}/tokenizer.json',
            encoder_fn=f'{model_dir}/sentencepiece.bpe.model'
        )

        processor.init()

        text = [
            'what is panda?',
            'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.'
        ]
        outs = processor.single_predict(*text)
        # {'score': [[5.265036106109619]]}
    """
    model_version = 'bge_reranker'

    def set_model(self):
        from models.text_similarity.bge_reranker import Model
        self.model = Model()

    def set_tokenizer(self):
        self.tokenizer = bundled.XLMRobertaTokenizer.from_pretrained(
            vocab_fn=self.vocab_fn,
            encoder_fn=self.encoder_fn
        )

    def load_pretrained(self):
        from models.text_similarity.bge_reranker import WeightConverter
        from models.bundles import WeightLoader
        tensors = WeightLoader.auto_load(self.pretrained_model)
        tensors = WeightConverter.from_hf(tensors)
        self.model.load_state_dict(tensors, strict=True)

    def get_model_inputs(self, loop_inputs, train=True):
        pair_paragraphs = [ret['text_pair'] for ret in loop_inputs]
        r = self.tokenizer.encode_pair_paragraphs(pair_paragraphs)
        inputs = dict(
            text_ids=r['segments_ids'],
            attention_mask=r['valid_segment_tags'],
        )
        inputs = torch_utils.Converter.force_to_tensors(inputs, self.device)

        return inputs

    def on_val_step(self, loop_objs, model_kwargs=dict(), **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        model_inputs = self.get_model_inputs(loop_inputs, train=False)
        model_inputs.update(model_kwargs)

        model_results = {}
        for name, model in self.models.items():
            model_output = model(**model_inputs)
            model_results[name] = {'score': model_output.cpu().numpy().tolist()}

        return model_results

    def on_predict_reprocess(self, loop_objs, return_keys=(), **kwargs):
        super().on_predict_reprocess(
            loop_objs,
            return_keys=('score',),
            **kwargs
        )

    def gen_predict_inputs(self, *objs, start_idx=None, end_idx=None, **kwargs) -> List[dict]:
        texts1, texts2 = objs[:2]
        if isinstance(texts1, str):
            texts1 = [texts1] * (end_idx - start_idx)
        if isinstance(texts2, str):
            texts2 = [texts2] * (end_idx - start_idx)
        assert len(texts1) == len(texts2)

        inputs = [dict(text_pair=[text1, text2]) for text1, text2 in zip(texts1, texts2)]

        return inputs


class BgeM3(Process):
    """
    Usage:
        model_dir = 'xxx'
        processor = BgeM3(
            pretrained_model=model_dir,
            vocab_fn=f'{model_dir}/tokenizer.json',
            encoder_fn=f'{model_dir}/sentencepiece.bpe.model'
        )

        processor.init()

        text = [
            'what is panda?',
            'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.'
        ]
        outs = processor.single_predict(*text)
        # {'dense_scores': [[0.6277193427085876]]}

        outs = processor.batch_predict(
            *text,
            model_kwargs=dict(
                return_sparse=True,
                return_colbert=True,
            )
        )
        # {'dense_scores': [[0.6277193427085876]], 'sparse_scores': [[0.124709352850914]], 'colbert_scores': [[0.7298586964607239]]}
    """
    model_version = 'bge_m3'

    def set_model(self):
        from models.text_similarity.bge_m3 import Model
        self.model = Model()

    def set_tokenizer(self):
        self.tokenizer = bundled.XLMRobertaTokenizer.from_pretrained(
            vocab_fn=self.vocab_fn,
            encoder_fn=self.encoder_fn
        )

    def load_pretrained(self):
        from models.text_pretrain.bge_m3 import WeightConverter
        from models.bundles import WeightLoader
        tensors = {
            'backbone': WeightLoader.auto_load(f'{self.pretrained_model}/pytorch_model.bin'),
            'sparse': WeightLoader.auto_load(f'{self.pretrained_model}/sparse_linear.pt'),
            'colbert': WeightLoader.auto_load(f'{self.pretrained_model}/colbert_linear.pt'),
        }
        tensors = WeightConverter.from_hf(tensors)
        self.model.load_state_dict(tensors, strict=False)

    def get_model_inputs(self, loop_inputs, train=True):
        pair_paragraphs = [ret['text_pair'] for ret in loop_inputs]
        paragraphs1, paragraphs2 = math_utils.transpose(pair_paragraphs)
        r1 = self.tokenizer.encode_paragraphs(paragraphs1)
        r2 = self.tokenizer.encode_paragraphs(paragraphs2)
        inputs = dict(
            text_ids1=r1['segments_ids'],
            attention_mask1=r1['valid_segment_tags'],
            text_ids2=r2['segments_ids'],
            attention_mask2=r2['valid_segment_tags'],
        )
        inputs = torch_utils.Converter.force_to_tensors(inputs, self.device)

        return inputs

    def on_val_step(self, loop_objs, model_kwargs=dict(), **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        model_inputs = self.get_model_inputs(loop_inputs, train=False)
        model_inputs.update(model_kwargs)

        model_results = {}
        for name, model in self.models.items():
            model_output = model(**model_inputs)
            model_results[name] = {k: v.cpu().numpy().tolist() for k, v in model_output.items()}

        return model_results

    def on_predict_reprocess(self, loop_objs, return_keys=(), **kwargs):
        super().on_predict_reprocess(
            loop_objs,
            return_keys=('dense_scores', 'sparse_scores', 'colbert_scores'),
            **kwargs
        )

    def gen_predict_inputs(self, *objs, start_idx=None, end_idx=None, **kwargs) -> List[dict]:
        texts1, texts2 = objs[:2]
        if isinstance(texts1, str):
            texts1 = [texts1] * (end_idx - start_idx)
        if isinstance(texts2, str):
            texts2 = [texts2] * (end_idx - start_idx)
        assert len(texts1) == len(texts2)

        inputs = [dict(text_pair=[text1, text2]) for text1, text2 in zip(texts1, texts2)]

        return inputs
