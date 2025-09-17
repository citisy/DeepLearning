from typing import List

from data_parse.nl_data_parse.pre_process import bundled
from processor import Process
from utils import torch_utils


class BgeReranker(Process):
    """
    Usage:
        model_dir = 'xxx'
        processor = BgeReranker(
            pretrain_model=f'{model_dir}/model.safetensors',
            vocab_fn=f'{model_dir}/tokenizer.json',
            encoder_fn=f'{model_dir}/sentencepiece.bpe.model'
        )

        processor.init()

        text = [
            'what is panda?',
            'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.'
        ]
        outs = processor.single_predict(*text)
        {'score': [[5.265036106109619]]}
    """
    model_version = 'bge_reranker'

    def set_model(self):
        from models.text_similarity.bge_reranker import Model
        self.model = Model()

    def set_tokenizer(self):
        self.tokenizer = bundled.XLMRobertaTokenizer.from_pretrained(self.vocab_fn, self.encoder_fn)

    def load_pretrained(self):
        if self.pretrain_model:
            from models.text_similarity.bge_reranker import WeightConverter
            from models.bundles import WeightLoader
            tensors = WeightLoader.auto_load(self.pretrain_model)
            tensors = WeightConverter.from_hf(tensors)
            self.model.load_state_dict(tensors, strict=True)

    def get_model_inputs(self, loop_inputs, train=True):
        pair_paragraphs = [ret['text_pair'] for ret in loop_inputs]
        r = self.tokenizer.encode_pair_paragraphs(pair_paragraphs)
        r = torch_utils.Converter.force_to_tensors(r, self.device)
        inputs = dict(
            text_ids=r['segments_ids'],
            attention_mask=r['valid_segment_tags'],
        )

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
            return_keys=('score', ),
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
