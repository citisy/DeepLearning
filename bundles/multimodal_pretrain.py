from pathlib import Path
from typing import List

import torch

from data_parse.cv_data_parse.data_augmentation import scale, channel, pixel_perturbation, Apply
from processor import Process, DataHooks, CheckpointHooks
from utils import os_lib, torch_utils


class DataProcessForQwen2Vl(DataHooks):
    input_size = 756

    post_aug = Apply([
        channel.BGR2RGB(),
        scale.Rectangle(),
        pixel_perturbation.MinMax(),
        pixel_perturbation.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        ),
        channel.HWC2CHW()
    ])

    def data_augment(self, ret, train=True) -> dict:
        if train:
            raise NotImplementedError

        ret.update(self.post_aug(dst=self.input_size, **ret))
        return ret


class FromQwen2VlPretrained(CheckpointHooks):
    pretrain_model: str | List[str]

    def load_pretrained(self):
        if hasattr(self, 'pretrain_model'):
            from models.multimodal_pretrain.Qwen2_VL import WeightLoader, WeightConverter

            if Path(self.pretrain_model).is_dir():
                self.pretrain_model = [str(fp) for fp in os_lib.find_all_suffixes_files(self.pretrain_model, ['.safetensors'])]

            state_dict = WeightLoader.auto_load(self.pretrain_model)
            state_dict = WeightConverter.from_official(state_dict)
            self.model.load_state_dict(state_dict, strict=True, assign=True)

            self.log(f'Loaded pretrained model!')


class BaseQwen2Vl(Process):
    config_version: str = '2b'
    use_half = True

    def set_model(self):
        from models.multimodal_pretrain.Qwen2_VL import Model, Config

        with torch.device('meta'):  # fast to init model
            self.model = Model(**Config.get(self.config_version))

    def set_model_status(self):
        self.load_pretrained()
        if not isinstance(self.device, list):
            self.model.to(self.device)

        if self.use_half:
            self.model.set_half()

    def set_tokenizer(self):
        from data_parse.nl_data_parse.pre_process.bundled import Qwen2VLTokenizer

        self.tokenizer = Qwen2VLTokenizer.from_pretrained(
            self.vocab_fn,
            self.encoder_fn,
        )

    def get_model_inputs(self, loop_inputs, train=True):
        if train:
            raise NotImplementedError
        else:
            return self.get_model_val_inputs(loop_inputs)


class Qwen2VlPredictor(BaseQwen2Vl):
    def get_model_val_inputs(self, loop_inputs):
        model_inputs = []
        for ret in loop_inputs:
            image = ret['image']
            text = ret['text']

            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": text
                    },
                ],
            },
            ]
            prompt_inputs = self.tokenizer.encode_dialog(messages)
            prompt_inputs['text_ids'] = prompt_inputs.pop('segments_ids')
            prompt_inputs = torch_utils.Converter.force_to_tensors(prompt_inputs, self.device)
            prompt_inputs['image_pixel_values'] = prompt_inputs['image_pixel_values'].to(self.model.dtype)
            model_inputs.append(prompt_inputs)

        return model_inputs

    def on_val_step(self, loop_objs, model_kwargs=dict(), **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        model_inputs = self.get_model_inputs(loop_inputs, train=False)

        model_results = {}
        for name, model in self.models.items():
            generated_ids = []
            for model_input in model_inputs:
                model_output = model(**model_input, **model_kwargs)

                preds = model_output['preds']
                preds = preds.cpu().numpy()

                seq_lens = model_input['seq_lens']
                generated_ids += [pred[seq_len:-1] for seq_len, pred in zip(seq_lens, preds)]

            model_results[name] = dict(
                generated_ids=generated_ids,
            )

        return model_results

    def gen_predict_inputs(self, *objs, start_idx=None, end_idx=None, text=None, image=None, **kwargs) -> List[dict]:
        if isinstance(image, str):
            image = os_lib.loader.load_img(image)

        return [dict(
            image=image,
            text=text
        )] * (end_idx - start_idx)

    def on_predict_reprocess(self, loop_objs, process_results=dict(), **kwargs):
        model_results = loop_objs['model_results']
        generated_ids = model_results[self.model_name]['generated_ids']
        response = self.tokenizer.decode_to_segments(generated_ids)
        process_results.setdefault(self.model_name, []).extend(response)


class Qwen2Vl(FromQwen2VlPretrained, DataProcessForQwen2Vl, Qwen2VlPredictor):
    """
    Usage:
        .. code-block:: python

            from bundles.multimodal_pretrain import Qwen2Vl as Process

            model_dir = 'xxx'
            process = Process(
                pretrain_model=model_dir,
                vocab_fn=f'{model_dir}/vocab.json',
                encoder_fn=f'{model_dir}/merges.txt',
                config_version='2b'  # ['2b', '7b', '72b']
            )

            process.init()

            # todo: only support single predict
            text = '描述一下这张图片。'
            image = 'xxx'
            process.single_predict(text=text, image=image)
    """
